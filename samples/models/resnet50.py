import os
import torch
import torchvision.models as models
from pathlib import Path
from .models_base import ModelBase, DEFAULT_MODELS_STORAGE_PATH
from .dataset.imagenette import ImageNette
try:
    import nncf
except ImportError:
    pass

def imagenet_quantize_with_nncf(model):
    def transform_fn(data_item):
        images, _ = data_item
        return images
    train_loader = ImageNette().prepare_calibration_dataloader()
    calibration_dataset = nncf.Dataset(train_loader, transform_fn)
    quantized_model = nncf.quantize(model, calibration_dataset, subset_size=10)
    return quantized_model

class ResNet50(ModelBase):
    def __init__(self, models_storage_path: Path = DEFAULT_MODELS_STORAGE_PATH, logger=None) -> None:
        super().__init__(models_storage_path=models_storage_path, model_name="resnet50", model_shape=(-1, 3, 224, 224), logger=logger)

    def _download_default_pytorch_model(self) -> Path:
        weights="ResNet50_Weights.DEFAULT"
        self._logger.info(f"Downloading ResNet50 model with weights: {weights} from torchvision.models.")
        model = models.resnet50(weights=weights)
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
            raise FileNotFoundError(f"Model {self} path: {storage_path} is not exist")
        model = models.resnet50()
        model.load_state_dict(torch.load(storage_path))
        return model

    def _quantize_model_nncf(self, model):
        return imagenet_quantize_with_nncf(model)
