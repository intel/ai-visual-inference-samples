from .videoreader import VideoReader
from .videowriter import VideoWriter, CpuVideoWriter, VaapiVideoWriter
from .data_loader import MediaDataLoader

try:
    from intel_visual_ai.libvisual_ai import XpuMemoryFormat
    from intel_visual_ai.libvisual_ai.transform import FrameTransform
except ImportError as err:
    pass

try:
    from .imageoverlay import ImageOverlay
except ImportError as err:
    pass

try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

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
