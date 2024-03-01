# This file is made to be as close as possible to
# https://pytorch.org/vision/main/_modules/torchvision/io/video_reader.html#VideoReader
# For future upstream possibilities
import logging as log
import importlib

libvideoreader = importlib.import_module(".libvisual_ai.videoreader", package=__package__)
from typing import Any, Dict, Iterator, Optional


class VideoReader:
    def __init__(
        self,
        src: str = "",
        stream: str = "video",
        num_threads: int = 0,
        path: Optional[str] = None,
        output_original_nv12: bool = False,
    ) -> None:
        from . import get_video_backend

        self.backend = get_video_backend()
        self.output_original_nv12 = output_original_nv12
        if isinstance(src, str):
            if src == "":
                raise TypeError("src cannot be empty")
        else:
            raise TypeError("`src` must be string object.")

        if "xpu" in self.backend:
            # FIXME: XpuEncoder and XpuDecoder if we are using the same device the string MUST BE the same!
            # Tentatively it is fixed using the self.backend which equates to "xpu" but if there are more
            # Devices this mechanism will create two different context! This affects ContextManager!
            self._c = libvideoreader.XpuDecoder(src, self.backend)
            self._c.set_output_original_nv12(self.output_original_nv12)
        else:
            raise RuntimeError("Unknown video backend: {}".format(self.backend))

    def __next__(self) -> Dict[str, Any]:
        """Decodes and returns the next frame of the current stream.
        Frames are encoded as a dict with mandatory
        data and pts fields, where data is a tensor, and pts is a
        presentation timestamp of the frame expressed in seconds
        as a float.

        Returns:
            (dict): a dictionary and containing decoded frame (``data``)
            and corresponding timestamp (``pts``) in seconds

        """
        if "xpu" in self.backend:
            if self.output_original_nv12:
                tensor, va_surface = next(self._c)
                return tensor, va_surface
            else:
                # Throw away the output_original_nv12 frame
                # FIXME: Instead of throwing away we should pass to the decoder
                #       the flag to disable output_original_nv12 copy if we are not using it
                tensor, _ = next(self._c)
                return tensor

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return self

    def seek(self, time_s: float, keyframes_only: bool = False) -> "VideoReader":
        """Seek within current stream.

        Args:
            time_s (float): seek time in seconds
            keyframes_only (bool): allow to seek only to keyframes

        .. note::
            Current implementation is the so-called precise seek. This
            means following seek, call to :mod:`next()` will return the
            frame with the exact timestamp if it exists or
            the first frame with timestamp larger than ``time_s``.
        """
        # TODO This method defined in original pytorch VideoReader but not defined in our backed. Should be considered to add
        raise NotImplementedError()

    def get_metadata(self) -> Dict[str, Any]:
        """Returns video metadata

        Returns:
            (dict): dictionary containing duration and frame rate for every stream
        """
        # TODO This method defined in original pytorch VideoReader but not defined in our backed. Should be considered to add
        raise NotImplementedError()

    def set_current_stream(self, stream: str) -> bool:
        """Set current stream.
        Explicitly define the stream we are operating on.

        Args:
            stream (string): descriptor of the required stream. Defaults to ``"video:0"``
                Currently available stream types include ``['video', 'audio']``.
                Each descriptor consists of two parts: stream type (e.g. 'video') and
                a unique stream id (which are determined by video encoding).
                In this way, if the video container contains multiple
                streams of the same type, users can access the one they want.
                If only stream type is passed, the decoder auto-detects first stream
                of that type and returns it.

        Returns:
            (bool): True on success, False otherwise
        """
        # TODO This method defined in original pytorch VideoReader but not defined in our backed. Should be considered to add
        raise NotImplementedError()
