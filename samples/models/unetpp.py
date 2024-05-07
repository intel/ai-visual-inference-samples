import segmentation_models_pytorch as smp
from pathlib import Path
from .models_base import ModelBase, DEFAULT_MODELS_STORAGE_PATH


class UnetPP(ModelBase):
    def __init__(self, models_storage_path: Path = DEFAULT_MODELS_STORAGE_PATH, logger=None) -> None:
        super().__init__(models_storage_path=models_storage_path, model_name="unet++", model_shape=(-1, 3, 1216, 1216), logger=logger)

    def _download_default_pytorch_model(self) -> Path:
        weights="imagenet"
        self._logger.info(f"Downloading Unet++ model with weights: {weights} from torchvision.models.")
        model = smp.UnetPlusPlus(encoder_name="resnet50", encoder_weights=weights, in_channels=3, classes=2)
        return self._save_pytorch_model(model, precision="fp32")
