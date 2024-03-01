from .videoreader import VideoReader
from .videowriter import VideoWriter, CpuVideoWriter, VaapiVideoWriter
from .imageoverlay import ImageOverlay
from .data_loader import MediaDataLoader
from intel_visual_ai.libvisual_ai.videoreader import XpuMemoryFormat

try:
    from .model_loader import ModelLoader
    from .torchvision_videoreader import (
        VideoReaderWithLoopMode as TorchvisionVideoReaderWithLoopMode,
    )
except ImportError as err:
    print(
        f"ImportError: Most likely torchvision or intel_extension_for_pytorch is not available, CPU-based video reader won't be available. Details: {err}"
    )

# This property to be aligned with https://pytorch.org/vision/main/_modules/torchvision.html
_video_backend = "xpu"


def set_video_backend(backend):
    """
    Specifies the package used to decode videos.
    """
    global _video_backend
    if "xpu" in backend:
        # TODO torch has check that there is dGPU available
        _video_backend = backend
    else:
        raise ValueError("Invalid video backend '%s'. Options are xpu, xpu:<id>" % backend)


def get_video_backend():
    """
    Returns the currently active video backend used to decode videos.

    Returns:
        str: Name of the video backend.
    """
    return _video_backend
