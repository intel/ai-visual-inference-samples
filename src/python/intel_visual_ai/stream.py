from typing import Literal
from collections import namedtuple
from intel_visual_ai import VideoReader, set_video_backend, set_video_backend_params
from intel_visual_ai.frame import Frame

try:
    from intel_visual_ai import XpuMemoryFormat
    from intel_visual_ai.libvisual_ai.videoreader import XpuDecoder
except ImportError:
    pass

from abc import ABC, abstractmethod

Resolution = namedtuple("Resolution", ("width", "height"))


class StreamMeta(ABC):
    @abstractmethod
    def __next__(self): ...

    @property
    @abstractmethod
    def finished(self): ...

    @property
    @abstractmethod
    def frames_processed(self): ...


class Stream(StreamMeta):
    def __init__(
        self,
        videoreader: VideoReader,
        stream_id=0,
        backend_type: Literal["openvino", "pytorch"] = "pytorch",
    ):
        self._stream = videoreader
        self._finished = False
        self._stream_id = stream_id
        self._frames_processed = 0
        self._backend_type = backend_type

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self._frames_processed += 1
            if self._backend_type == "openvino":
                return Frame(next(self._stream), self._stream_id, self._frames_processed)
            else:
                return next(self._stream)
        except StopIteration:
            self._finished = True
            raise StopIteration

    @property
    def finished(self):
        return self._finished

    @property
    def va_display(self):
        return self._stream.get_va_device()

    @property
    def original_size(self) -> Resolution:
        if isinstance(self._stream, VideoReader):
            original_size = self._stream._c.get_original_size()
        else:
            original_size = self._stream.get_original_size()
        return Resolution(original_size[0], original_size[1])

    @property
    def frames_processed(self):
        return self._frames_processed


class StreamFromSource(Stream):
    def __init__(
        self,
        stream_path,
        device="xpu",
        stream_id=0,
        backend_type: Literal["openvino", "pytorch"] = "openvino",
        **kwargs,
    ):
        try:
            if isinstance(device, str):
                set_video_backend(device)
                set_video_backend_params(**kwargs)
                stream = VideoReader(str(stream_path))._c
            else:
                stream = XpuDecoder(str(stream_path), device)
        except Exception as e:
            raise RuntimeError(f"Cannot open stream: {stream_path}.") from e
        super().__init__(videoreader=stream, stream_id=stream_id, backend_type=backend_type)
