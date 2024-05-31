import os
from typing import List
import logging
from pathlib import Path
from intel_visual_ai.multi_stream_videoreader import MultiStreamVideoReader
from intel_visual_ai.frame import Frame
from intel_visual_ai.openvino_infer_backend import OpenVinoInferBackend, ov
from intel_visual_ai.metrics import MetricsTimer
from intel_visual_ai import XpuMemoryFormat
from intel_visual_ai.itt import IttTask, itt_task


from samples.common.stop_condition import (
    EosStopCondition,
    FramesProcessedStopCondition,
    StopCondition,
)
from samples.openvino.utils.ov_backend_inference_only import InferenceOnlyWrapper
from samples.common.utils import text_file_to_list
from samples.common.logger import init_and_get_metrics_logger
from samples.models import get_model_by_name


class OvPipeline:
    def __init__(
        self,
        model,
        input,
        nireq,
        batch_size,
        log_predictions,
        labels_path,
        device,
        inference_interval,
        output_dir,
        stop_condition: StopCondition,
        threshold,
        sample_name,
        num_streams,
        inference_only,
        async_decoding_depth,
        warmup_iterations,
        precision,
        logger=None,
        preproc=None,
        model_shape=None,
        decode_device=None,
        media_only=None,
        **kwargs,
    ) -> None:
        self.logger = logger if logger is not None else logging.getLogger(sample_name)
        self.sample_name = sample_name
        self._print_predictions = log_predictions
        self._inference_only = inference_only
        self._media_only = media_only
        self._inference_interval = inference_interval
        self.__stop_condition = stop_condition
        self._threshold = threshold
        self._warmup_iter = warmup_iterations

        self._device = device.lower()
        self._decode_device = decode_device.lower() if decode_device is not None else device
        if "cpu" in self._decode_device:
            raise NotImplementedError("CPU decode is not supported in OV Backend.")
        self._media_path = input
        self._nireq = nireq
        self._async_depth = async_decoding_depth
        self.__batch_size = batch_size

        if Path(model).is_file():
            model_path = model
        else:
            model_path = get_model_by_name(model)(logger=self.logger).get_model(
                precision=precision, inference_backend="openvino", quantization_backend="nncf"
            )
        self.ov_backend = OpenVinoInferBackend(
            device=self._device,
            model_name=model_path,
            nireq=nireq,
            batch_size=batch_size,
            interval=inference_interval,
            preproc=preproc,
            model_shape=model_shape,
            logger=self.logger,
        )
        self._model_info = self.ov_backend.get_model_input_info()

        if self._inference_only:
            self.ov_backend = InferenceOnlyWrapper(self.ov_backend, batch_size=self.__batch_size)

        self.video = MultiStreamVideoReader()
        self._configure_video_streams(self.video, num_streams)

        if isinstance(self.__stop_condition, (EosStopCondition, FramesProcessedStopCondition)):
            self.__stop_condition.set_stream(self.video)

        self.ov_backend.compile_model(
            va_display=self.video.va_display if not self.is_cpu_device else None
        )

        self._output_dir = output_dir
        os.makedirs(self._output_dir, exist_ok=True)
        self.metrics_logger = None
        self._warmup()
        if log_predictions:
            if labels_path is not None:
                self.labels = text_file_to_list(labels_path)
            self.configure_completion_callback(**kwargs)
            self.ov_backend.set_completion_callback(self.completion_callback)

    @property
    def is_cpu_device(self) -> bool:
        return "cpu" in self._device

    def _configure_video_streams(self, video: MultiStreamVideoReader, num_streams):
        memory_format = XpuMemoryFormat.openvino_planar
        if self.is_cpu_device:
            memory_format = XpuMemoryFormat.system_rgbp

        play_in_loop = False if isinstance(self.__stop_condition, EosStopCondition) else True
        video.set_common_stream_params(
            out_img_size=(self._model_info.width, self._model_info.height),
            pool_size=self.__batch_size * self._nireq * 2,
            memory_format=memory_format,
            async_depth=self._async_depth * self.__batch_size,
            play_in_loop=play_in_loop,
        )
        for _ in range(num_streams):
            video.add_stream(self._media_path, device=self._decode_device)

    def _warmup(self):
        self.logger.info(
            f"Opening Warmup stream and running warmup for iterations {self._warmup_iter}"
        )
        warmup_video = MultiStreamVideoReader()
        self._configure_video_streams(warmup_video, num_streams=1)
        warmup_frame = next(warmup_video)

        for _ in range(self._warmup_iter):
            self.ov_backend.infer(warmup_frame)
            self.ov_backend.flush()
        self.logger.info(f"Completed warmup")

    def run(self):
        if self.metrics_logger is None:
            self.metrics_logger = init_and_get_metrics_logger(
                sample_name=self.sample_name,
                log_dir=self._output_dir,
                device=self._device,
            )
        if self._inference_only:
            self._run_inference_only()
        elif self._media_only:
            self._run_media_only()
        else:
            self._run_full()

    def _run_full(self):
        with MetricsTimer(
            stream=self.video,
            batch_size=self.__batch_size,
            print_fps_to_stdout=True,
            output_dir=self._output_dir,
            logger=self.metrics_logger,
        ):
            while not self.__stop_condition.stopped:
                try:
                    frame = next(self.video)
                except StopIteration as e:
                    break
                self.ov_backend.infer(frame)
        self.ov_backend.flush()

    def _run_media_only(self):
        with MetricsTimer(
            stream=self.video,
            batch_size=self.__batch_size,
            print_fps_to_stdout=True,
            output_dir=self._output_dir,
            logger=self.metrics_logger,
        ):
            while not self.__stop_condition.stopped:
                try:
                    frame = next(self.video)
                except StopIteration as e:
                    break

    def _run_inference_only(self):
        if isinstance(self.__stop_condition, (FramesProcessedStopCondition)):
            self.__stop_condition.set_stream(self.ov_backend)
        frame = next(self.video)
        with MetricsTimer(
            stream=self.ov_backend,
            batch_size=self.__batch_size,
            print_fps_to_stdout=True,
            output_dir=self._output_dir,
            logger=self.metrics_logger,
        ):
            while not self.__stop_condition.stopped:
                self.ov_backend.infer(frame)
        self.ov_backend.flush()

    def configure_completion_callback(self, **kwargs):
        pass

    def completion_callback(self, infer_request: ov.InferRequest, frames: List[Frame]) -> None:
        """
        Post process output
        """
        pass

    @classmethod
    def create_from_args(cls, args, stop_condition, logger, preproc=None, model_shape=None):
        return cls(
            stop_condition=stop_condition,
            logger=logger,
            preproc=preproc,
            model_shape=model_shape,
            **vars(args),
        )
