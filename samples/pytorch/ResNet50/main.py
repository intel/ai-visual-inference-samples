"""
=============================================
Object Classification with ResNet-50
=============================================

This example illustrates object classification inferencing with ResNet-50 model with pretrained weights.
The input is the path to a media file and the inference output (per frame) is a list of identified labels and a corresponding image superimposed with the label (watermark).
The watermark is drawn with PIL python package. To best follow the code, one can start with the `main()` function below and subsequently read about individual functions invoked.
"""

import intel_extension_for_pytorch as ipex
import torch
import torchvision.models as models
import argparse
from pathlib import Path

import sys

root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(root_dir)
from samples.pytorch.utils.imagenet2012_util import add_arguments, ImageNet2012Util
from samples.pytorch.utils.resnet50_util.quantization import Quantization
from intel_visual_ai.dataset.imagenette import ImageNette

import warnings

warnings.filterwarnings("ignore")


def main():
    #########################
    # 1. Inputs
    # ----------------------------------
    # The following code takes in a optional input file
    parser = argparse.ArgumentParser(
        prog="ResNet-50 Classification Sample", description="PyTorch sample for ResNet-50"
    )

    parser = add_arguments(parser, default_batch_size=64, default_num_frames=400_000)

    parser.add_argument(
        "--run_accuracy_test",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Runs the Quantization accuracy test",
    )

    args = parser.parse_args()

    media_path = args.input
    if not (Path(media_path).is_file()):
        raise ValueError(f"Cannot find input media {args.input}")

    #########################
    # 2. Loading pre-trained model
    # ----------------------------------
    # ResNet-50 pre-trained model is loaded with :func `torchvision.models.resnet50`
    # and the default weights correspond to `ResNet50_Weights.DEFAULT`
    model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
    if args.only_download_models:
        imagenette = ImageNette(dataset_path=args.dataset_dir)
        imagenette.download_and_extract_files()
        return

    #########################
    # 3. Model Evaluation
    # ----------------------------------
    # Model evaluation with :func `torch.nn.Module.eval`
    model.eval()

    #########################
    # 4. Device Availability
    # ----------------------------------
    # Using `torch.xpu.is_available()` supported by Intel® Extension for PyTorch*,
    # the presence of an Intel GPU device can be checked and fallback is CPU.
    if "xpu" in args.device:
        device = args.device if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    elif "cpu" == args.device:
        device = "cpu"
    else:
        raise ValueError(f"Unknown acceleration type - {args.device}")

    #########################
    # 5. Model Load to device
    # ----------------------------------
    # Using :func `Quantization().quantize_int8`, this will quantize the model from FP32 to INT8 version of the model to device CPU/XPU
    quantization = Quantization(device, dataset_path=args.dataset_dir)
    model = model.to(device)
    model = quantization.quantize_int8(model)

    #########################
    # 5a. Verify quantized model
    # ----------------------------------
    # Using :func `Quantization().calculate_accuracy` and `FileHandler().prepare_calibration_dataset`, this will prepare the Dataset and run accuracy test
    # Expected quantized results should show:
    # Quantization results to INT8:
    # Using ImageNette val dataset:
    #     Top 1 Accuracy: 84 %
    #     Top 5 Accuracy: 98 %
    if args.run_accuracy_test:
        print("[INFO] Running accuracy test")
        quantization.calculate_accuracy(model)

    #########################
    # 6. Model Optimization
    # ----------------------------------
    # Using `optimize` provided by Intel® Extension for PyTorch* for optimization on Intel GPU
    if "xpu" in device:
        model = ipex.optimize(model)

    imagenet_util = ImageNet2012Util(model, media_path, args, "resnet50", model_precision="FP32")
    #########################
    # Warm Up with random data
    imagenet_util.warmup()

    #########################
    # 7. Processing Frames
    # ----------------------------------
    # Fine grained video reader API is used for frame by frame processing. A total of 50 frames are processed here but it's possible to process all of them.
    # Every video frame is sequentially subject to the following steps:
    # - Decode
    # - Preprocessing
    # - Inference
    # - Watermark
    # Each of these are explained above
    imagenet_util.process_frames()


if __name__ == "__main__":
    main()
