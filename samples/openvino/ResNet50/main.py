from pathlib import Path
import cProfile
import sys
from typing import List
import numpy as np
from intel_visual_ai.frame import Frame
from intel_visual_ai.openvino_infer_backend import ov

root_dir = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(root_dir)
from samples.common.get_dirs import media_dir, models_dir, labels_dir
from samples.openvino.utils.ov_pipeline import OvPipeline
from samples.openvino.utils.ov_arguments import OvArguments
from samples.common.utils import get_stop_condition
from samples.openvino.utils.imagenet_utils import ImageNetPipeline

MEDIA_PATH = media_dir / "20230104_dog_bark_1920x1080_3mbps_30fps_ld_h265.mp4"
MODEL_PRECISION = "int8"
LABELS_PATH = labels_dir / "imagenet_2012_bkgr.txt"
NIREQ = 4
BATCH_SIZE = 64
DEVICE = "xpu"
INFERENCE_INTERVAL = 1
NUM_STREAMS = 2
THRESHOLD = 0.5
MODEL_NAME = "resnet50"
SAMPLE_NAME = MODEL_NAME
NUM_FRAMES = 400_000


def resnet50_preproc(model, ppp):
    ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(
        ov.Layout("NHWC")
    ).set_color_format(ov.preprocess.ColorFormat.NV12_TWO_PLANES, ["y", "uv"]).set_memory_type(
        ov.runtime.properties.intel_gpu.MemoryType.surface
    )
    ppp.input().preprocess().convert_color(ov.preprocess.ColorFormat.RGB).convert_element_type(
        ov.Type.f32
    ).scale(255.0).mean([0.485, 0.456, 0.406]).scale([0.229, 0.224, 0.225])
    ppp.input().model().set_layout(ov.Layout("NCHW"))


class Resnet50Pipeline(ImageNetPipeline): ...


def main():

    ov_args = OvArguments(
        sample_name=SAMPLE_NAME,
        media=MEDIA_PATH,
        model=MODEL_NAME,
        device=DEVICE,
        nireq=NIREQ,
        batch_size=BATCH_SIZE,
        inference_interval=INFERENCE_INTERVAL,
        streams=NUM_STREAMS,
        threshold=THRESHOLD,
        output_dir=Path(__file__).resolve().parent / "output",
        labels_path=LABELS_PATH,
        num_frames=NUM_FRAMES,
        precision=MODEL_PRECISION,
    )

    args = ov_args.parse_args()

    stop_condition = get_stop_condition(args)
    logger = ov_args.create_logger()

    pipeline = Resnet50Pipeline.create_from_args(
        args, stop_condition, logger, preproc=resnet50_preproc
    )
    if args.only_download_models:
        logger.info("Model downloaded successfully, exiting")
        return

    if args.profile:
        cProfile.run("pipeline.run()", sort="cumtime")
    else:
        pipeline.run()


if __name__ == "__main__":
    main()
