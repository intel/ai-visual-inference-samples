from typing import List
from intel_visual_ai.openvino_infer_backend import _ImplGpuVaShare
from intel_visual_ai.stream import StreamMeta
from intel_visual_ai.frame import Frame


class InferenceOnlyWrapper(StreamMeta):
    def __init__(
        self,
        inference_backend: _ImplGpuVaShare,
        batch_size: int,
    ):
        self._inference_backend = inference_backend
        self.__batch_size = batch_size
        self._frames_processed = 0

    def infer(self, frame):
        for _ in range(self.__batch_size):
            self._inference_backend.infer(frame)
        self._frames_processed += self.__batch_size

    def flush(self):
        self._inference_backend.flush()

    @property
    def frames_processed(self):
        return self._frames_processed

    def compile_model(self, **kwargs):
        self._inference_backend.compile_model(**kwargs)

    def __next__(self):
        pass

    @property
    def finished(self):
        pass
