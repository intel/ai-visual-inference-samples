import torchvision.models as models
from pathlib import Path
from .models_base import ModelBase, DEFAULT_MODELS_STORAGE_PATH


class SwinTransformer(ModelBase):
    def __init__(self, models_storage_path: Path = DEFAULT_MODELS_STORAGE_PATH, logger=None) -> None:
        super().__init__(models_storage_path=models_storage_path, model_name="swin_transformer", model_shape=(-1, 3, 224, 224), logger=logger)

    def _download_default_pytorch_model(self) -> Path:
        weights="DEFAULT"
        self._logger.info(f"Downloading SwinTransformer model with weights: {weights} from torchvision.models.")
        model = models.swin_b(weights=weights)
        return self._save_pytorch_model(model, precision="fp32")
