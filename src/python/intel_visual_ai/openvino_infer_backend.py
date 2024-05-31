from abc import abstractmethod, ABC
from collections import namedtuple
import logging as log

import numpy as np
import openvino as ov
from openvino import layout_helpers, VAContext
import openvino.properties.hint as hints
from intel_visual_ai.frame import Frame
from intel_visual_ai.itt import IttTask, itt_task


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


class BatchedRequestBase(ABC):
    def __init__(self, batch_size) -> None:
        self._batch_size = batch_size
        self._frames = []
        self._tensors = []

    def add_frame(self, frame: Frame):
        if self.ready:
            raise RuntimeError("Trying to add frame to already full batch request.")
        self._frames.append(frame)

    @property
    def ready(self):
        return self._batch_size == len(self._frames)

    @property
    def frames(self):
        return self._frames

    @abstractmethod
    def get_batched_tensor(self): ...

    @property
    def empty(self):
        return not self.frames


class BatchedRequestShared(BatchedRequestBase):
    def __init__(self, batch_size, remote_context, input_names) -> None:
        super().__init__(batch_size)
        self.__remote_context = remote_context
        self.__inputs = input_names
        self._tensors = {input_name: [] for input_name in input_names}
        for input_name in self.__inputs:
            self._tensors[input_name] = []

    @itt_task
    def add_frame(self, frame: Frame):
        super().add_frame(frame)
        tensors = self.__remote_context.create_tensor_nv12(
            frame.height, frame.width, frame.va_surface_id
        )
        for input_name, tensor in zip(self.__inputs, tensors):
            self._tensors[input_name].append(tensor)

    def get_batched_tensor(self, input_name):
        return [tensor for tensor in self._tensors[input_name]]


class BatchedRequestCPU(BatchedRequestBase):
    @itt_task
    def add_frame(self, frame: Frame):
        super().add_frame(frame)
        numpy_tensor = np.array(frame.raw, copy=False)
        if len(numpy_tensor.shape) == 3:
            numpy_tensor = np.expand_dims(numpy_tensor, axis=0)
        self._tensors.append(numpy_tensor)

    def get_batched_tensor(self):
        return np.vstack(self._tensors)


class _ImplBase(ABC):
    def __init__(
        self,
        device,
        model_name,
        nireq=1,
        batch_size=1,
        interval=1,
        preproc=None,
        model_shape=None,
        logger=None,
        **kwargs,
    ):
        self._logger = logger if logger is not None else log.getLogger("visual_ai")
        self._logger.info(f"Creating OvBackend parameters:")
        self._logger.info(f"    model_name: {model_name}")
        self._logger.info(f"    device: {device}")
        self._logger.info(f"    nireq: {nireq}")
        self._logger.info(f"    batch_size: {batch_size}")
        self._logger.info(f"    inference_interval: {interval}")
        self._device = device
        self._batch_size = batch_size
        self._nireq = nireq
        self._interval = interval
        self._interval_counter = interval - 1
        self._core = ov.Core()
        self._logger.debug(f"Reading model: {model_name}")
        model = self._core.read_model(model_name)
        if model_shape:
            model.reshape(model_shape)
        ppp = ov.preprocess.PrePostProcessor(model)
        if preproc is not None:
            preproc(model, ppp)
        else:
            ov_default_preproc(model, ppp)

        self._model = ppp.build()

        ov.set_batch(self._model, batch_size)
        self._inputs = self._model.get_parameters()
        self._batched_request = None
        self._compiled_model = None
        self._infer_request_queue = None

    @abstractmethod
    def _get_compiled_model(self, device, **kwargs): ...

    @abstractmethod
    def _get_batched_request(self) -> BatchedRequestBase: ...

    def compile_model(self, **kwargs):
        self._logger.info(f"Compile model for device {self._device}")
        self._compiled_model = self._get_compiled_model(self._device, **kwargs)
        self._infer_request_queue = ov.AsyncInferQueue(self._compiled_model, self._nireq)
        self._batched_request = self._get_batched_request()

    @abstractmethod
    def _batched_infer(self): ...

    def set_completion_callback(self, completion_callback):
        self._infer_request_queue.set_callback(completion_callback)

    def get_model_input_info(self) -> ModelInputInfo:
        layout = layout_helpers.get_layout(self._model.input(self._inputs[0].friendly_name))
        width = self._inputs[0].shape[layout_helpers.width_idx(layout)]
        height = self._inputs[0].shape[layout_helpers.height_idx(layout)]
        return ModelInputInfo(height=height, width=width)

    @itt_task
    def infer(self, frame: Frame):
        self._interval_counter += 1
        if self._interval_counter != self._interval:
            return
        self._interval_counter = 0
        self._batched_request.add_frame(frame)
        if self._batched_request.ready:
            self._batched_infer()
            self._batched_request = self._get_batched_request()

    def flush(self):
        no_frames_to_fill_batch = 0
        if not self._batched_request.empty:
            # WA: Fill non-complete batch with last element. Can be removed once supported in OV
            while not self._batched_request.ready:
                no_frames_to_fill_batch += 1
                self._batched_request.add_frame(self._batched_request.frames[-1])
            self._batched_infer()
        self._infer_request_queue.wait_all()
        self._batched_request = self._get_batched_request()
        # Reset interval counter to initial value
        self._interval_counter = self._interval - 1
        return no_frames_to_fill_batch


class OpenVinoInferBackend(_ImplBase):
    def __new__(cls, device, *args, **kwargs):
        if device == "cpu":
            return _ImplCpu(*args, device=device, **kwargs)
        elif "xpu" in device:
            return _ImplGpuVaShare(*args, device=device, **kwargs)
        else:
            raise NotImplementedError(f"Unsupported device: {device}")


class _ImplGpuVaShare(_ImplBase):
    def __init__(
        self,
        device,
        model_name,
        nireq=1,
        batch_size=1,
        interval=1,
        preproc=None,
        model_shape=None,
        logger=None,
        **kwargs,
    ):
        self.__remote_context = None
        super().__init__(
            device,
            model_name,
            nireq,
            batch_size,
            interval,
            preproc,
            model_shape,
            logger,
            **kwargs,
        )

    def _get_batched_request(self):
        return BatchedRequestShared(
            batch_size=self._batch_size,
            remote_context=self.__remote_context,
            input_names=[input.friendly_name for input in self._inputs],
        )

    def _get_compiled_model(self, device, va_display, **kwargs):
        if device == "cpu":
            raise RuntimeError("Cannot create GPU backend from CPU device")
        if va_display is None:
            raise ValueError("VA display is required for GPU backend")
        self.__remote_context = VAContext(self._core, va_display.native())
        config = {
            hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
            hints.allow_auto_batching: False,
        }
        compiled_model = self._core.compile_model(self._model, self.__remote_context, config)
        return compiled_model

    def _batched_infer(self):
        free_request = self._infer_request_queue[self._infer_request_queue.get_idle_request_id()]
        free_request.set_tensors(
            self._inputs[0].friendly_name,
            self._batched_request.get_batched_tensor(self._inputs[0].friendly_name),
        )
        free_request.set_tensors(
            self._inputs[1].friendly_name,
            self._batched_request.get_batched_tensor(self._inputs[1].friendly_name),
        )
        self._infer_request_queue.start_async(userdata=(self._batched_request.frames))


class _ImplCpu(_ImplBase):
    def _get_batched_request(self):
        return BatchedRequestCPU(batch_size=self._batch_size)

    def _get_compiled_model(self, device, **kwargs):
        if device != "cpu":
            raise RuntimeError("Cannot create CPU backend from non-CPU device")
        config = {
            hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
            "PERFORMANCE_HINT_NUM_REQUESTS": self._nireq,
        }
        compiled_model = self._core.compile_model(self._model, device.upper(), config)
        return compiled_model

    def _batched_infer(self):
        self._infer_request_queue.start_async(
            self._batched_request.get_batched_tensor(), userdata=(self._batched_request.frames)
        )
