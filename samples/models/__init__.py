
try:
    import openvino as ov
    import nncf
    ov_backend_available = True
except ImportError:
    ov_backend_available = False

from .models_base import _PRECISION, ModelBase

MODELS_LIST = {}
__import_exceptions = {}

try:
    from .resnet50 import ResNet50
    MODELS_LIST["resnet50"] = ResNet50
except ImportError as e:
    __import_exceptions["resnet50"] = e

try:
    from .fbnet import FBNet
    MODELS_LIST["fbnet"] = FBNet
except ImportError as e:
    __import_exceptions["fbnet"] = e

try:
    from .yolov5m import YOLOv5m
    MODELS_LIST["yolov5m"] = YOLOv5m
except ImportError as e:
    __import_exceptions["yolov5m"] = e

try:
    from .ssd_mobilenet_v1_coco import SSDMobilenetv1
    MODELS_LIST["ssd_mobilenet_v1_coco"] = SSDMobilenetv1
except ImportError as e:
    __import_exceptions["ssd_mobilenet_v1_coco"] = e

try:
    from .unetpp import UnetPP
    MODELS_LIST["unet++"] = UnetPP
except ImportError as e:
    __import_exceptions["unet++"] = e

try:
    from .swin_transformer import SwinTransformer
    MODELS_LIST["swin_transformer"] = SwinTransformer
except ImportError as e:
    __import_exceptions["swin_transformer"] = e

def get_model_by_name(model_name) -> ModelBase:
    if not model_name.lower() in MODELS_LIST:
        if model_name in __import_exceptions:
            raise RuntimeError(f"Failed to import model with name: {model_name}. Exception: {__import_exceptions[model_name]}")
        raise RuntimeError(f"Unsupported model with name: {model_name}")
    return MODELS_LIST[model_name.lower()]
