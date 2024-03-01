"""
=============================================
Object Classification with Swin B Transformer
=============================================

This example illustrates object classification inferencing with Swin B Transformer model with pretrained weights.
The input is the path to a media file and the inference output (per frame) is a list of identified labels and a corresponding image superimposed with the label (watermark).
The watermark is drawn with PIL python package. To best follow the code, one can start with the `main()` function below and subsequently read about individual functions invoked.
"""


import intel_extension_for_pytorch as ipex
import torch
import argparse
import torchvision.models as models
from pathlib import Path
import sys

root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(root_dir)
from samples.pytorch.utils.imagenet2012_util import add_arguments, ImageNet2012Util


def main():
    #########################
    # 1. Inputs
    # ----------------------------------
    # The following code takes in a optional input file
    parser = argparse.ArgumentParser(
        prog="Swin T Classification Sample", description="PyTorch sample for Swin Transformer"
    )
    parser = add_arguments(parser, default_num_frames=40_000)

    args = parser.parse_args()

    media_path = args.input
    if not (Path(media_path).is_file()):
        raise ValueError(f"Cannot find input media {args.input}")

    #########################
    # 2. Loading pre-trained model
    # ----------------------------------
    # Swin Transformer pre-trained model is loaded with :func `torchvision.models.swin_b`
    # and the default weights correspond to `Swin_B_Weights.IMAGENET1K_V1`
    model = models.swin_b(weights="DEFAULT")
    if args.only_download_models:
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
    # Using :func `torch.Tensor.half`, one can load the Float16 version of the model to device CPU/XPU as determined previously

    model = model.half().to(device)

    #########################
    # 6. Model Optimization
    # ----------------------------------
    # Using `optimize` provided by Intel® Extension for PyTorch* for optimization on Intel GPU
    if "xpu" in device:
        model = ipex.optimize(model)

    # Jit optimizations
    if not args.disable_jit:
        print("Running Jit Optimization on model")
        random_tensor = torch.randn((1, 3, 224, 224), device=device).half()
        model = torch.jit.trace(model, random_tensor)

    imagenet_util = ImageNet2012Util(model, media_path, args, "swin_b")

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
