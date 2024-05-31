from typing import Union, Literal
from intel_visual_ai import VideoReader
from intel_visual_ai.stream import Stream, StreamMeta, Resolution

try:
    from intel_visual_ai.stream import StreamFromSource
except (ImportError, ModuleNotFoundError):
    pass

DEFAULT_DEVICE = "xpu"


class MultiStreamVideoReader(StreamMeta):
    def __init__(self, *args, backend_type: Literal["openvino", "pytorch"] = "openvino", **kwrargs):
        self.__streams = []
        for stream in args:
            self.add_stream(stream, backend_type=backend_type, **kwrargs)
        self.__iter = None
        self.__frames_processed = 0
        self._backend_type = backend_type
        self._common_params = {}

    def add_stream(self, stream: Union[Stream, VideoReader, str], device=None, **kwargs):
        if isinstance(stream, Stream):
            self.__streams.append(stream)
        elif isinstance(stream, VideoReader):
            self.__streams.append(Stream(stream, stream_id=len(self.__streams)))
        else:
            params = self._common_params | kwargs
            self.__streams.append(
                StreamFromSource(
                    stream,
                    device or DEFAULT_DEVICE,
                    len(self.__streams),
                    self._backend_type,
                    **params,
                )
            )

    def __iter__(self):
        while not all([stream.finished for stream in self.__streams]):
            for stream in self.__streams:
                if stream.finished:
                    continue
                try:
                    self.__frames_processed += 1
                    yield next(stream)
                except StopIteration:
                    self.__frames_processed -= 1
                    continue

    def __next__(self):
        if not self.__iter:
            self.__iter = iter(self)
        try:
            return next(self.__iter)
        except Exception:
            raise StopIteration

    @property
    def num_streams(self):
        return len(self.__streams)

    @property
    def frames_processed(self):
        return self.__frames_processed

    @property
    def va_display(self):
        return self.__streams[0].va_display

    @property
    def original_size(self):
        return self.__streams[0].original_size

    def stop(self):
        for stream in self.__streams:
            stream.stop()

    @property
    def finished(self):
        return all([stream.finished for stream in self.__streams])

    def set_common_stream_params(
        self,
        out_img_size: Resolution,
        pool_size,
        memory_format,
        async_depth=0,
        play_in_loop=False,
        out_orig_nv12=False,
        ff_preproc=False,
    ):
        self._common_params = {
            "out_img_size": out_img_size,
            "pool_size": pool_size,
            "memory_format": memory_format,
            "async_depth": async_depth,
            "loop_mode": play_in_loop,
            "output_original_nv12": out_orig_nv12,
            "ff_preproc": ff_preproc,
        }
