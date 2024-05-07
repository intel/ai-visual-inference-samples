import logging
import os
from typing import Literal
from pathlib import Path
from abc import ABC, abstractmethod
import torch
from . import ov_backend_available
try:
    import openvino as ov
except ImportError:
    pass


_INFERENCE_BACKEND = Literal["openvino", "pytorch"]
_QUANTIZATION_BACKEND = Literal["nncf", "ipex"]
_PRECISION = Literal["fp32", "fp16", "int8"]
DEFAULT_MODELS_STORAGE_PATH = Path(__file__).parent.parent.parent / "data" / "models"

class ModelBase(ABC):
    """
    An abstract base class for machine learning models, providing functionalities for downloading, quantizing,
    converting, and managing models.

    Attributes:
        _models_storage_path (Path): The path to the directory where models are stored.
        _model_name (str): The name of the model.
        _model_shape (tuple): The shape of the model input.
        _logger (logging.Logger): Logger instance for logging information, warnings, and errors.
    """
    def __init__(self, model_name, model_shape, models_storage_path: Path=DEFAULT_MODELS_STORAGE_PATH, logger=None) -> None:
        """
        Initializes the ModelBase instance with the specified model name, shape, storage path, and logger.

        Parameters:
            model_name (str): The name of the model.
            model_shape (tuple): The shape of the model input.
            models_storage_path (Path, optional): The path to the directory where models are stored. Defaults to
                DEFAULT_MODELS_STORAGE_PATH.
            logger (logging.Logger, optional): Logger instance for logging. If None, a default logger is created.
        """
        self._models_storage_path: Path = models_storage_path
        self._model_name = model_name
        self._model_shape = model_shape
        self._logger = logger if logger is not None else logging.getLogger("visual_ai")

    def __repr__(self) -> str:
        return self._model_name

    def _download_default_pytorch_model(self) -> Path:
        raise NotImplementedError(f"Downloading pytorch model is not implemented for {self} model")

    def _get_and_load_default_pytorch_model(self):
        if (model_pth := self._get_model_path_from_storage(precision="fp32", inference_backend="pytorch")) is None:
            model_pth = self._download_default_pytorch_model()
        return self._load_pt_model_from_local_storage(model_pth)

    def _quantize_model(self, model, backend: _QUANTIZATION_BACKEND):
        self._logger.info(f"Quantizing model: {self} using {backend} backend")
        if backend == "nncf":
            return self._quantize_model_nncf(model)
        if backend == "ipex":
            return self._quantize_model_ipex(model)

    def _quantize_model_nncf(self, model):
        raise NotImplementedError(f"Quantization using nncf backend is not supported for {self} model")

    def _quantize_model_ipex(self, model):
        raise NotImplementedError(f"Quantization using ipex backend is not supported for {self} model")

    def _get_model_location_in_storage(self, precision: _PRECISION, inference_backend: _INFERENCE_BACKEND) -> Path:
        location = self._models_storage_path / inference_backend / self._model_name / precision
        self._logger.debug(f"Model location in storage: {location}")
        return location

    def _save_ov_model(self, ov_model, precision):
        model_storage_path = self._get_model_location_in_storage(precision=precision, inference_backend="openvino")
        target_file = model_storage_path / f'{self._model_name}.xml'
        self._logger.info(f"Saving model: {self} to file: {target_file}")
        compress_to_fp16 = False if precision == "fp32" else True
        ov.save_model(ov_model, target_file, compress_to_fp16=compress_to_fp16)
        model_files = self._get_model_path_from_storage(precision=precision, inference_backend="openvino")
        if model_files is None:
            raise RuntimeError(f"Unable to save OV model. No files in: {model_storage_path}")
        return model_files

    def _save_pytorch_model(self, model, precision) -> Path:
        model = model.eval()
        model_storage_path = self._get_model_location_in_storage(precision=precision, inference_backend="pytorch")
        os.makedirs(model_storage_path, exist_ok=True)
        target_file = model_storage_path / f'{self._model_name}.pth'
        data_shape = self._model_shape
        if data_shape[0] == -1:
            data_shape = (1,) + data_shape[1:]
        dummy_input = torch.randn(data_shape)
        traced_graph = torch.jit.trace(model, dummy_input)
        traced_graph.save(target_file)
        return target_file

    def _load_pt_model_from_local_storage(self, storage_path: Path):
        if not storage_path.exists():
            raise FileNotFoundError(f"Model {self} path: {storage_path} is not exist")
        return torch.jit.load(storage_path).eval()

    def _convert_to_ov_model(self, model, model_shape):
        self._logger.debug("Converting model to openvino format.")
        data_shape = model_shape
        # to support dynamic batch
        if data_shape[0] == -1:
            data_shape = (1,) + model_shape[1:]
        dummy_input = torch.randn(data_shape)
        return ov.convert_model(model, example_input=dummy_input, input=model_shape)

    def _prepare_ov_model(self, precision: _PRECISION):
        self._logger.info(f"Preparing openvino model with precision: {precision}")
        model = self._get_and_load_default_pytorch_model()
        if precision == "int8":
            model = self._quantize_model(model=model, backend="nncf")
        model = self._convert_to_ov_model(model=model, model_shape=self._model_shape)
        return self._save_ov_model(model, precision=precision)

    def _get_model_path_from_storage(self, precision: _PRECISION, inference_backend: _INFERENCE_BACKEND):
        model_location = self._get_model_location_in_storage(precision=precision, inference_backend=inference_backend)
        self._logger.debug(f"Getting model path from storage for precision: {precision}, backend: {inference_backend}")
        if not model_location.exists():
            self._logger.warning(f"Model location does not exist: {model_location}")
            return None
        if inference_backend == "openvino":
            bin_files, xml_files = list(model_location.glob("*.bin")), list(model_location.glob("*.xml"))
            if all((bin_files, xml_files)):
                self._logger.info(f"Model files: {xml_files[0]} found for openvino backend.")
                return xml_files[0]
        if inference_backend == "pytorch":
            torch_script_files = list(model_location.glob("*.pth")) + list(model_location.glob("*.torchscript"))
            if torch_script_files:
                self._logger.info(f"Model files: {torch_script_files[0]} found for PyTorch backend.")
                return torch_script_files[0]
        self._logger.warning("No model files found for the specified backend and precision.")
        return None

    def _get_pytorch_model(self, precision: _PRECISION, quantization_backend: _QUANTIZATION_BACKEND):
        if (model_pth := self._get_model_path_from_storage(precision=precision, inference_backend="pytorch")) is not None:
            return self._load_pt_model_from_local_storage(model_pth)
        model = self._get_and_load_default_pytorch_model()
        if precision == "fp16":
            model = model.half()
        elif precision == "int8" and quantization_backend == "nncf":
            model = self._quantize_model_nncf(model)
        elif precision == "int8" and quantization_backend == "ipex":
            raise NotImplementedError(f"Precision: {precision} for pytorch with quantization backend: {quantization_backend} is not supported yet")
        self._save_pytorch_model(model=model, precision=precision)
        return model

    def _get_ov_model(self, precision: _PRECISION, quantization_backend: _QUANTIZATION_BACKEND):
        model_path_in_storage = self._get_model_path_from_storage(precision=precision, inference_backend="openvino")
        if model_path_in_storage is not None:
            self._logger.info(f"Model path found in local storage: {model_path_in_storage}")
            return model_path_in_storage
        if not ov_backend_available:
            raise RuntimeError("Requested inference backend is openvino but it is not available. Please check its installation")
        if precision == "int8" and quantization_backend != "nncf":
            raise RuntimeError(f"Quantization backend: {quantization_backend} is not supported for openvino")
        # model_path_in_storage = None # TODO use line above before merge
        self._logger.info("Model not found in local storage, preparing openvino model.")
        return self._prepare_ov_model(precision=precision)

    def get_model(self, precision: _PRECISION, inference_backend: _INFERENCE_BACKEND, quantization_backend: _QUANTIZATION_BACKEND="nncf"):
        """
        Retrieves a model with the specified precision, inference backend, and quantization backend.

        Parameters:
            precision (_PRECISION): The desired precision of the model ('fp32', 'fp16', 'int8').
            inference_backend (_INFERENCE_BACKEND): The inference backend to use ('openvino', 'pytorch').
            quantization (_QUANTIZATION_BACKEND): The backend to use for model quantization ('nncf', 'ipex').

        Returns:
            The path to the model file if the inference backend is 'openvino', or a callable that downloads the model
            if the inference backend is 'pytorch'. For 'openvino', the method ensures that the model is available in
            the specified precision and has been converted to the openvino format.
        """
        self._logger.info(f"Getting model: {self} with precision: {precision}, inference backend: {inference_backend}, quantization backend: {quantization_backend}")
        if inference_backend == "pytorch":
            return self._get_pytorch_model(precision=precision, quantization_backend=quantization_backend)
        if inference_backend == "openvino":
            return self._get_ov_model(precision=precision, quantization_backend=quantization_backend)
        raise RuntimeError(f"Unsupported inference backend: {inference_backend}")
