import shutil
import os
import torch
from pathlib import Path
from typing import Tuple, List
from torchvision import tv_tensors
from torchvision.transforms import v2
from .models_base import ModelBase, DEFAULT_MODELS_STORAGE_PATH, _PRECISION, _INFERENCE_BACKEND, _QUANTIZATION_BACKEND
from ultralytics.models import YOLO
from ultralytics import settings



def scale_output_boxes(
    batch_output: List[torch.Tensor],
    scaled_tensor_size: torch.Size,
    original_tensor_size: torch.Size,
    format="XYXY",
) -> List[torch.Tensor]:
    """Scale ROI boxes received from yolov5 model to original frame size.
    As yolov5 detection was executed on rescaled frame.

    Args:
        batch_output (List[torch.Tensor]): list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        scaled_tensor_size (torch.Size): The size of scaled tensor that was passed to yolov5 model as input
        original_tensor_size (torch.Size): The size of original video frame tensor

    Returns:
        List[torch.Tensor]: _description_
    """
    transforms = v2.Resize(size=(original_tensor_size))
    scaled_output = []
    for output in batch_output:
        output[:, :4] = transforms(
            tv_tensors.BoundingBoxes(output[:, :4], format=format, canvas_size=scaled_tensor_size)
        )
        scaled_output.append(output)
    return scaled_output



class YoloModelBase(ModelBase):
    def __init__(self, device="cpu", models_storage_path: Path=DEFAULT_MODELS_STORAGE_PATH, logger=None, model_shape: Tuple | None = None, weights: str | None = None) -> None:
        if model_shape is None:
            model_shape = self.MODEL_SHAPE
        super().__init__(models_storage_path=models_storage_path, model_name=self.MODEL_NAME, model_shape=model_shape, logger=logger, device=device)
        self.weights = weights if weights else self.MODEL_WEIGHTS

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

    def _quantize_model_nncf(self, model):
        ## nncf quantization for this model requires torch version >=2.1.2, ipex currently uses 2.1.0 version
        raise NotImplementedError(f"Quantization using nncf backend is not supported for {self} model")

    def _download_default_pytorch_model(self):
        params = {"format":"TorchScript",  "imgsz":self._model_shape[2]}
        target_dir=self._get_model_location_in_storage(precision="fp32", inference_backend="pytorch")
        target_dir.mkdir(parents=True, exist_ok=True)
        YOLO(model=self.weights).export(**params)
        output_files = [list(Path(os.getcwd()).glob(f"**/*{ext}")) for ext in ("bin", "xml", "yaml", "pt", "torchscript")]
        output_files = list(set([x for xs in output_files for x in xs]))
        if not output_files:
            raise RuntimeError(f"Yolo model export failed. No output files in target directory: {target_dir}")
        for file_ in output_files:
            shutil.move(file_, target_dir / file_.name)
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

class YOLOv5m(YoloModelBase):
    MODEL_NAME="yolov5m"
    MODEL_WEIGHTS="yolov5m.pt"
    MODEL_SHAPE=(-1, 3, 640, 640)
