import logging as log
from collections import namedtuple
from intel_visual_ai import VideoReader, XpuMemoryFormat, set_video_backend
from intel_visual_ai.libvisual_ai.videoreader import XpuDecoder

DEFAULT_DEVICE = "xpu"

Resolution = namedtuple("Resolution", ("width", "height"))


class Frame:
    def __init__(self, c_frame, stream_id) -> None:
        self.__tensor = c_frame[0]
        self.__stream_id = stream_id
        self.__original_frame = None  # TODO for getting original frame from XpuDecoder

    @property
    def va_display(self):
        return self.__tensor.va_display

    @property
    def va_surface_id(self):
        return self.__tensor.va_surface_id

    @property
    def stream_id(self):
        return self.__stream_id

    @property
    def width(self):
        return self.__tensor.width

    @property
    def height(self):
        return self.__tensor.height

    @property
    def raw(self):
        return self.__tensor


class Stream:
    def __init__(self, stream_path, device, stream_id=0, play_in_loop=False):
        try:
            if isinstance(device, str):
                set_video_backend(device)
                self.__stream = VideoReader(str(stream_path))._c
            else:
                self.__stream = XpuDecoder(str(stream_path), device)
            self.__stream.set_loop_mode(play_in_loop)
            self.__finished = False
            self.__stream_id = stream_id
        except Exception as e:
            raise RuntimeError(f"Cannot open stream: {stream_path}.") from e

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return Frame(next(self.__stream), self.__stream_id)
        except Exception as e:
            self.__finished = True
            raise StopIteration

    @property
    def finished(self):
        return self.__finished

    @property
    def va_display(self):
        return self.__stream.get_va_device()

    def stop(self):
        self.__stream.stop()

    def set_output_resolution(self, resolution: Resolution):
        log.debug(f"Set stream: {self.__stream_id} outpur resolution to: {resolution}")
        self.__stream.set_output_resolution(resolution.width, resolution.height)

    def set_memory_format(self, memory_format: XpuMemoryFormat):
        self.__stream.set_memory_format(memory_format)

    def set_frame_pool_params(self, pool_size):
        self.__stream.set_frame_pool_params(pool_size)


class MultiStreamVideoReader:
    def __init__(self, *args, device=DEFAULT_DEVICE, play_in_loop=False, **kwrargs):
        self.__device = device
        self.__streams = []
        self.play_in_loop = play_in_loop
        for stream in args:
            self.add_stream(stream, self.__va_config)
        self.__iter = None
        self.__frames_processed = 0

    def add_stream(self, stream_path=None):
        self.__streams.append(
            self.stream_from_device_name(
                stream_path,
                stream_id=len(self.__streams),
                device_name=self.__device,
                play_in_loop=self.play_in_loop,
            )
        )

    def __iter__(self):
        while not all([stream.finished for stream in self.__streams]):
            for stream in self.__streams:
                if stream.finished:
                    continue
                try:
                    yield next(stream)
                    self.__frames_processed += 1
                except StopIteration:
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

    def stop(self):
        for stream in self.__streams:
            stream.stop()

    @property
    def stream_ended(self):
        return all([stream.finished for stream in self.__streams])

    @staticmethod
    def stream_from_device_name(stream_path, device_name: str, stream_id=0, play_in_loop=False):
        return Stream(stream_path, device_name, stream_id=stream_id, play_in_loop=play_in_loop)

    def configure_preproc(self, output_width, output_height, pool_size):
        for stream in self.__streams:
            stream.set_output_resolution(
                resolution=Resolution(width=output_width, height=output_height)
            )
            stream.set_frame_pool_params(pool_size=pool_size)
            # TODO MultiStreamVideoReader supports only OV backend right now
            stream.set_memory_format(memory_format=XpuMemoryFormat.openvino_planar)
