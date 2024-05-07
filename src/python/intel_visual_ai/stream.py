from typing import Literal
from collections import namedtuple
from intel_visual_ai import VideoReader, set_video_backend
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
        original_size = self._stream.get_original_size()
        return Resolution(original_size[0], original_size[1])

    @property
    def frames_processed(self):
        return self._frames_processed

    def set_output_resolution(self, resolution: Resolution):
        self._stream.set_output_resolution(resolution.width, resolution.height)

    def set_memory_format(self, memory_format):
        self._stream.set_memory_format(memory_format)

    def set_frame_pool_params(self, pool_size):
        self._stream.set_frame_pool_params(pool_size)

    def set_async_depth(self, async_depth):
        self._stream.set_async_depth(async_depth)

    def set_output_original_frame(self, enable: bool):
        self._stream.set_output_original_nv12(enable)


class StreamFromSource(Stream):
    def __init__(
        self,
        stream_path,
        device="xpu",
        stream_id=0,
        play_in_loop=False,
        backend_type: Literal["openvino", "pytorch"] = "openvino",
    ):
        try:
            if isinstance(device, str):
                set_video_backend(device)
                stream = VideoReader(str(stream_path))._c
            else:
                stream = XpuDecoder(str(stream_path), device)
            stream.set_loop_mode(play_in_loop)
        except Exception as e:
            raise RuntimeError(f"Cannot open stream: {stream_path}.") from e
        super().__init__(videoreader=stream, stream_id=stream_id, backend_type=backend_type)
