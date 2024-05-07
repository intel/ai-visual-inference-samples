"""
=============================================
Object Classification with ResNet-50
=============================================

This example illustrates object classification inferencing with ResNet-50 model with pretrained weights.
The input is the path to a media file and the inference output (per frame) is a list of identified labels and a corresponding image superimposed with the label (watermark).
The watermark is drawn with PIL python package. To best follow the code, one can start with the `main()` function below and subsequently read about individual functions invoked.
"""

import torchvision.models as models
import argparse
from pathlib import Path

import sys

root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(root_dir)
from samples.pytorch.utils.imagenet_util.imagenet_pipeline import (
    ImageNetPipeline,
    ImageNetArguments,
)
from samples.pytorch.utils.pt_pipeline import optimize_model

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
from samples.pytorch.utils.resnet50_util.quantization import Quantization

# TODO: ImageNette and Dataset utils needs to be removed after TorchVision 0.17.0 release by IPEX
from samples.models.dataset.imagenette import ImageNette

NUM_FRAMES = 400_000

import warnings

warnings.filterwarnings("ignore")


def main():
    #########################
    # 1. Inputs
    # ----------------------------------
    # The following code takes in a optional input file
    pt_args = ImageNetArguments(
        "resnet50", batch_size=64, output_dir=OUTPUT_DIR, num_frames=NUM_FRAMES
    )

    pt_args.parser.add_argument(
        "--run_accuracy_test",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Runs the Quantization accuracy test",
    )
    args = pt_args.parse_args()

    logger = pt_args.create_logger()

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
    # 5. Model Load to device
    # ----------------------------------
    # Using :func `Quantization().quantize_int8`, this will quantize the model from FP32 to INT8 version of the model to device CPU/XPU
    quantization = Quantization(args.device, dataset_path=args.dataset_dir, logger=logger)

    model = model.to(args.device)
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
    # Using ImageNet val dataset:
    #     Top 1 Accuracy: 78 %
    #     Top 5 Accuracy: 94 %
    if args.run_accuracy_test:
        print("[INFO] Running accuracy test")
        quantization.calculate_accuracy(model)

    #########################
    # 6. Model Optimization
    # ----------------------------------
    # Using `optimize` provided by Intel® Extension for PyTorch* for optimization on Intel GPU
    model = optimize_model(model, args, convert_to_fp16=False)

    pipeline = ImageNetPipeline(model, args, logger=logger)

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
    pipeline.run()


if __name__ == "__main__":
    main()
