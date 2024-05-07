from typing import List
from intel_visual_ai.openvino_infer_backend import OpenVinoInferBackend
from intel_visual_ai.stream import StreamMeta
from intel_visual_ai.frame import Frame


class OvBackendInferenceOnly(OpenVinoInferBackend, StreamMeta):
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
        super().__init__(
            model_name, va_display, nireq, batch_size, interval, preproc, model_shape, logger
        )
        self._frames_processed = 0
        self.__batch_size = batch_size

    def infer(self, frame):
        while not self._batched_request.ready:
            self._batched_request.add_frame(frame)
        self._batched_infer()
        self._frames_processed += self.__batch_size

    @property
    def frames_processed(self):
        return self._frames_processed

    def __next__(self):
        pass

    @property
    def finished(self):
        pass
