"""
=============================================
Object Classification with Swin B Transformer
=============================================

This example illustrates object classification inferencing with Swin B Transformer model with pretrained weights.
The input is the path to a media file and the inference output (per frame) is a list of identified labels and a corresponding image superimposed with the label (watermark).
The watermark is drawn with PIL python package. To best follow the code, one can start with the `main()` function below and subsequently read about individual functions invoked.
"""

import intel_extension_for_pytorch as ipex
import torchvision.models as models
from pathlib import Path
import sys

root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(root_dir)
from samples.pytorch.utils.imagenet_util.imagenet_pipeline import (
    ImageNetPipeline,
    ImageNetArguments,
)
from samples.pytorch.utils.pt_pipeline import optimize_model, jit_optimize_model

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
NUM_FRAMES = 40000


def main():
    #########################
    # 1. Inputs
    # ----------------------------------
    # The following code takes in a optional input file
    pt_args = ImageNetArguments(
        "swin_b", batch_size=4, output_dir=OUTPUT_DIR, num_frames=NUM_FRAMES
    )
    args = pt_args.parse_args()

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
    model = optimize_model(model, args, convert_to_fp16=True)
    model = jit_optimize_model(model, args, width=224, height=224, trace=True)

    pipeline = ImageNetPipeline(model, args)

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
