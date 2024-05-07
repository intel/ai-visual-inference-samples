import timm
import os
import torch
from pathlib import Path
from .models_base import ModelBase, DEFAULT_MODELS_STORAGE_PATH, _PRECISION, _INFERENCE_BACKEND, _QUANTIZATION_BACKEND
from .dataset.imagenette import ImageNette
from .resnet50 import imagenet_quantize_with_nncf
try:
    import nncf
except ImportError:
    pass

class FBNet(ModelBase):
    def __init__(self, models_storage_path: Path = DEFAULT_MODELS_STORAGE_PATH, logger=None) -> None:
        super().__init__(models_storage_path=models_storage_path, model_name="fbnet", model_shape=(-1, 3, 224, 224), logger=logger)

    def _download_default_pytorch_model(self) -> Path:
        weights="fbnetc_100.rmsp_in1k"
        self._logger.info(f"Downloading FBNet model with weights: {weights} from timm.")
        model = timm.create_model(weights, pretrained=True)
        return self._save_pytorch_model(model, precision="fp32")

    def _save_pytorch_model(self, model, precision) -> Path:
        model = model.eval()
        model_storage_path = self._get_model_location_in_storage(precision=precision, inference_backend="pytorch")
        os.makedirs(model_storage_path, exist_ok=True)
        target_file = model_storage_path / f'{self._model_name}.pth'
        torch.save(model.state_dict(), target_file)
        return target_file

    def _load_pt_model_from_local_storage(self, storage_path: Path):
        if not storage_path.exists():
            raise FileNotFoundError(f"Model {self} path: '{storage_path}' does not exist")
        model = timm.create_model("fbnetc_100.rmsp_in1k")
        model.load_state_dict(torch.load(storage_path))
        return model


    def _quantize_model_nncf(self, model):
        return imagenet_quantize_with_nncf(model)
