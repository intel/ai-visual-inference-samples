import torch
import logging
from samples.models.dataset.imagenette import ImageNette


class Quantization:
    """
    Handles the quantization process for models, including accuracy calculation and INT8 quantization.

    Attributes:
        calibration_dataset (DataLoader): DataLoader for the calibration dataset.
        validation_dataset (DataLoader): DataLoader for the validation dataset.
        device (str): The device to use for calculations (e.g., 'cpu', 'cuda').
        calib_iters (int): Number of iterations for calibration.
    """

    def __init__(
        self,
        device: str = "cpu",
        calib_iters: int = 10,
        dataset_path=None,
        logger: logging.Logger | None = None,
    ):
        """
        Initializes the Quantization class.

        Args:
            imagenet_url (str): URL to download the ImageNette dataset.
            device (str): The device to use for calculations (e.g., 'cpu', 'cuda'). Defaults to 'cpu'.
            calib_iters (int): Number of iterations to use for calibration. Defaults to 10.
        """
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.imagenette = ImageNette(dataset_path=dataset_path)
        self.calibration_dataset = self.imagenette.prepare_calibration_dataloader(device=device)
        self.validation_dataset = self.calibration_dataset
        self.device = device
        self.calib_iters = calib_iters

    def calculate_accuracy(self, model: torch.nn.Module) -> None:
        """
        Calculates and prints the top-1 and top-5 accuracy of the provided model on the validation dataset.

        Args:
            model (torch.nn.Module): The model to evaluate.
        """
        model.eval()
        correct_top1, correct_top5, total = 0, 0, 0

        with torch.no_grad():
            for images, labels in self.validation_dataset:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct_top1 += (predicted == labels).sum().item()

                _, top5_pred = outputs.topk(5, 1, True, True)
                top5_pred = top5_pred.t()
                correct_top5 += top5_pred.eq(labels.view(1, -1).expand_as(top5_pred)).sum().item()

        self.logger.info(f"Top 1 Accuracy: {100 * correct_top1 / total:.2f}%")
        self.logger.info(f"Top 5 Accuracy: {100 * correct_top5 / total:.2f}%")

    def quantize_int8(self, model_fp32_gpu: torch.nn.Module) -> torch.jit.ScriptModule:
        """
        Quantizes the given model to INT8 using JIT calibration.

        Args:
            model_fp32_gpu (torch.nn.Module): The FP32 model to quantize.

        Returns:
            torch.jit.ScriptModule: The quantized model.
        """
        self.logger.info("INT8 JIT Calibration")
        from torch.jit._recursive import wrap_cpp_module
        from torch.quantization.quantize_jit import convert_jit, prepare_jit

        modelJit = torch.jit.script(model_fp32_gpu)
        modelJit = wrap_cpp_module(torch._C._jit_pass_fold_convbn(modelJit._c))
        dtype = torch.qint8

        with torch.inference_mode():
            qconfig = torch.quantization.QConfig(
                activation=torch.quantization.observer.MinMaxObserver.with_args(
                    qscheme=torch.per_tensor_symmetric, reduce_range=False, dtype=dtype
                ),
                weight=torch.quantization.default_weight_observer,
            )
            modelJit = prepare_jit(modelJit, {"": qconfig}, True)
            for i, (input, _) in enumerate(self.calibration_dataset):
                modelJit(input.to(self.device))
                if i == self.calib_iters - 1:
                    break
            modelJit = convert_jit(modelJit, True)
            self.logger.info("Successfully quantized to INT8")

        return modelJit
