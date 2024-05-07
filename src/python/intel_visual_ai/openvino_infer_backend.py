from collections import namedtuple
import logging as log

import openvino as ov
from openvino import layout_helpers, VAContext
import openvino.properties.hint as hints
from intel_visual_ai.frame import Frame

ModelInputInfo = namedtuple("ModelInputInfo", ["width", "height"])


def ov_default_preproc(model, ppp):
    """Default PrePostProcessor steps"""
    ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(
        ov.Layout("NHWC")
    ).set_color_format(ov.preprocess.ColorFormat.NV12_TWO_PLANES, ["y", "uv"]).set_memory_type(
        ov.runtime.properties.intel_gpu.MemoryType.surface
    )
    # 2) Adding explicit preprocessing steps:
    # - apply linear resize from tensor spatial dims to model spatial dims
    ppp.input().preprocess().convert_color(ov.preprocess.ColorFormat.BGR)
    ppp.input().model().set_layout(ov.Layout("NCHW"))


class OpenVinoInferBackend:
    class BatchedRequest:
        def __init__(self, batch_size, remote_context, input_names) -> None:
            self.__batch_size = batch_size
            self.__remote_context = remote_context
            self.__frames = []
            self.__inputs = input_names
            self.__tensors = {}
            for input_name in self.__inputs:
                self.__tensors[input_name] = []

        def add_frame(self, frame: Frame):
            if self.ready:
                raise RuntimeError("Trying to add frame to already full batch request.")
            self.__frames.append(frame)
            tensors = self.__remote_context.create_tensor_nv12(
                frame.height, frame.width, frame.va_surface_id
            )
            for input_name, tensor in zip(self.__inputs, tensors):
                self.__tensors[input_name].append(tensor)

        @property
        def ready(self):
            return self.__batch_size == len(self.__frames)

        @property
        def frames(self):
            return self.__frames

        def get_batched_tensor(self, input_name):
            return [tensor for tensor in self.__tensors[input_name]]

        @property
        def empty(self):
            return len(self.frames) == 0

    def __init__(
        self,
        model_name,
        va_display,
        nireq=1,
        batch_size=1,
        interval=1,
        preproc=None,
        model_shape=None,
        logger=None,
    ):
        self._logger = logger if logger is not None else log.getLogger("visual_ai")
        self._logger.info(f"Creating OvBackend parameters:")
        self._logger.info(f"    model_name: {model_name}")
        self._logger.info(f"    va_display: {va_display}")
        self._logger.info(f"    nireq: {nireq}")
        self._logger.info(f"    batch_size: {batch_size}")
        self._logger.info(f"    inference_interval: {interval}")
        self.__batch_size = batch_size
        self.__interval = interval
        self.__interval_counter = interval - 1
        self.__core = ov.Core()
        self._logger.debug(f"Reading model: {model_name}")
        model = self.__core.read_model(model_name)
        if model_shape:
            model.reshape(model_shape)
        ppp = ov.preprocess.PrePostProcessor(model)
        if preproc is not None:
            preproc(model, ppp)
        else:
            ov_default_preproc(model, ppp)

        self.__model = ppp.build()

        ov.set_batch(self.__model, batch_size)

        self.__inputs = self.__model.get_parameters()

        self.__remote_context = VAContext(self.__core, va_display.native())
        config = {
            hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
            hints.allow_auto_batching: False,
        }
        self.__compiled_model = self.__core.compile_model(model, self.__remote_context, config)
        self._batched_request = self.__get_batched_request()
        self._infer_request_queue = ov.AsyncInferQueue(self.__compiled_model, nireq)

    def __get_batched_request(self):
        return OpenVinoInferBackend.BatchedRequest(
            batch_size=self.__batch_size,
            remote_context=self.__remote_context,
            input_names=[input.friendly_name for input in self.__inputs],
        )

    def _batched_infer(self):
        free_request = self._infer_request_queue[self._infer_request_queue.get_idle_request_id()]
        free_request.set_tensors(
            self.__inputs[0].friendly_name,
            self._batched_request.get_batched_tensor(self.__inputs[0].friendly_name),
        )
        free_request.set_tensors(
            self.__inputs[1].friendly_name,
            self._batched_request.get_batched_tensor(self.__inputs[1].friendly_name),
        )
        self._infer_request_queue.start_async(userdata=(self._batched_request.frames))

    def infer(self, frame: Frame):
        self.__interval_counter += 1
        if self.__interval_counter != self.__interval:
            return
        self.__interval_counter = 0
        self._batched_request.add_frame(frame)
        if self._batched_request.ready:
            self._batched_infer()
            self._batched_request = self.__get_batched_request()

    def set_completion_callback(self, completion_callback):
        self._infer_request_queue.set_callback(completion_callback)

    def flush(self):
        no_frames_to_fill_batch = 0
        if not self._batched_request.empty:
            # WA: Fill non-complete batch with last element. Can be removed once supported in OV
            while not self._batched_request.ready:
                no_frames_to_fill_batch += 1
                self._batched_request.add_frame(self._batched_request.frames[-1])
            self._batched_infer()
        self._infer_request_queue.wait_all()
        self._batched_request = self.__get_batched_request()
        # Reset interval counter to initial value
        self.__interval_counter = self.__interval - 1
        return no_frames_to_fill_batch

    def get_model_input_info(self) -> ModelInputInfo:
        layout = layout_helpers.get_layout(self.__model.input(self.__inputs[0].friendly_name))
        width = self.__inputs[0].shape[layout_helpers.width_idx(layout)]
        height = self.__inputs[0].shape[layout_helpers.height_idx(layout)]
        return ModelInputInfo(height=height, width=width)
