import shutil
from pathlib import Path
from .models_base import ModelBase, DEFAULT_MODELS_STORAGE_PATH, _PRECISION, _INFERENCE_BACKEND, _QUANTIZATION_BACKEND
from ultralytics.models import YOLO
from ultralytics import settings


class YOLOv5m(ModelBase):
    def __init__(self, models_storage_path: Path = DEFAULT_MODELS_STORAGE_PATH, logger=None, weights="yolov5m.pt") -> None:
        super().__init__(models_storage_path=models_storage_path, model_name="yolov5m", model_shape=(-1, 3, 320, 320), logger=logger)
        self.weights = weights

    def __export_model(self, target_dir: Path, weights: str, params: dict):
        model = YOLO(model=target_dir / weights).export(**params)
        output_files = [list(target_dir.glob(f"**/*{ext}")) for ext in ("bin", "xml", "yaml", "pt", "torchscript")]
        output_files = [x for xs in output_files for x in xs]
        if not output_files:
            raise RuntimeError(f"Yolo model export failed. No output files in target directory: {target_dir}")
        for file_ in output_files:
            shutil.move(file_, target_dir / file_.name)

    def __cleanup(self):
        datasets_dir = settings.get('datasets_dir', None)
        if datasets_dir is not None and Path(datasets_dir).exists():
            shutil.rmtree(datasets_dir)

    def _download_default_pytorch_model(self):
        params = {"format":"TorchScript"}
        self.__export_model(target_dir=self._get_model_location_in_storage(precision="fp32", inference_backend="pytorch"), weights=self.weights, params=params)
        return self._get_model_path_from_storage(precision="fp32", inference_backend="pytorch")

    def _get_ov_model(self, precision: _PRECISION, quantization_backend: _QUANTIZATION_BACKEND):
        model_path_in_storage = self._get_model_path_from_storage(precision=precision, inference_backend="openvino")
        if model_path_in_storage is not None:
            self._logger.info(f"Model path found in local storage: {model_path_in_storage}")
            return model_path_in_storage
        params = {"format":"openvino", "dynamic": True}
        if precision == "fp16":
            params["half"] = True
        if precision == "int8":
            params["int8"] = True
        self.__export_model(weights=self.weights, target_dir=self._get_model_location_in_storage(precision=precision, inference_backend="openvino"), params=params)
        self.__cleanup()
        return self._get_model_path_from_storage(precision=precision, inference_backend="openvino")
