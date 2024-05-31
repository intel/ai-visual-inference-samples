import argparse
import torch
from samples.common.get_dirs import media_dir
from pathlib import Path
from samples.common.arguments import Arguments

DEFAULT_MEDIA_FILE = media_dir / "20230104_dog_bark_1920x1080_3mbps_30fps_ld_h265.mp4"
DEFAULT_DEVICE = "xpu"


class PtArguments(Arguments):
    def __init__(
        self,
        sample_name,
        batch_size,
        output_dir,
        decode_device=None,
        labels_path=None,
        device=DEFAULT_DEVICE,
        threshold=0.5,
        media=DEFAULT_MEDIA_FILE,
        num_frames=None,
        inference_interval=1,
    ):

        super().__init__(
            sample_name=sample_name,
            media=media,
            device=device,
            batch_size=batch_size,
            threshold=threshold,
            output_dir=output_dir,
            labels_path=labels_path,
            num_frames=num_frames,
            inference_interval=inference_interval,
            decode_device=decode_device,
        )
        self._parser.add_argument(
            "--watermark",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Performs watermark and includes it for metrics calculation",
        )
        self._parser.add_argument(
            "--output-csv",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Stores the inference result to a csv file",
        )
        self._parser.add_argument(
            "--disable-jit-optimization",
            dest="disable_jit",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Disables model JIT optimization",
        )
        self._parser.add_argument(
            "--resize-method",
            help="select input resizing method",
            type=str,
            choices=["torch", "va"],
            default="va",
        )
        self._parser.add_argument(
            "--normalize-inputs",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Normalize tensors using mean and std",
        )
        self._parser.add_argument(
            "--dataset-dir",
            default=None,
            help="Path to dataset folder. If folder is empty will download default dataset",
            type=Path,
        )

    def validate(self):
        if "xpu" in self._args.device:
            if not hasattr(torch, "xpu") or not torch.xpu.is_available():
                raise ValueError(f"{self._args.device} not availble")
        elif "cpu" not in self._args.device:
            raise ValueError(f"Unsupported device type: {self._args.device}")

        return self._args
