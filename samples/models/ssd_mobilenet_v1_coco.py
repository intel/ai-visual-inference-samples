import os
from pathlib import Path
from .models_base import ModelBase, DEFAULT_MODELS_STORAGE_PATH, _PRECISION, _INFERENCE_BACKEND, _QUANTIZATION_BACKEND
import requests


class SSDMobilenetv1(ModelBase):
    def __init__(self, models_storage_path: Path = DEFAULT_MODELS_STORAGE_PATH, logger=None) -> None:
        super().__init__(models_storage_path=models_storage_path, model_name="ssd_mobilenet_v1_coco", model_shape=(-1, 3, 224, 224), logger=logger)

    def _download_ir_model(self, target_dir: Path):
        os.makedirs(target_dir, exist_ok=True)
        files_to_download = [f"ssd_mobilenet_v1_coco.{ext}" for ext in ("bin", "xml", "json")]
        source_url = "https://github.com/dlstreamer/pipeline-zoo-models/blob/main/storage/ssd_mobilenet_v1_coco_INT8/{}?raw=true"
        for file_to_download in files_to_download:
            with requests.get(source_url.format(file_to_download), stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(target_dir / file_to_download, 'wb') as fd:
                    fd.write(r.content)


    def get_model(self, precision: _PRECISION, inference_backend: _INFERENCE_BACKEND, quantization_backend: _QUANTIZATION_BACKEND):
        self._logger.info(f"Getting model: {self} with precision: {precision}, inference backend: {inference_backend}, quantization backend: {quantization_backend}")
        if inference_backend == "pytorch":
            raise NotImplementedError()
        if inference_backend == "openvino":
            model_path_in_storage = self._get_model_path_from_storage(precision=precision, inference_backend=inference_backend)
            if model_path_in_storage is not None:
                self._logger.info(f"Model path found in local storage: {model_path_in_storage}")
                return model_path_in_storage
            if precision == "int8":
                self._download_ir_model(target_dir=self._get_model_location_in_storage(precision="int8", inference_backend="openvino"))
                return self._get_model_path_from_storage(precision=precision, inference_backend=inference_backend)
            else:
                raise NotImplementedError(f"precision: {precision} is not supported for: {self._model_name}")
        else:
            raise RuntimeError(f"Unknown inference backend: {inference_backend}")
