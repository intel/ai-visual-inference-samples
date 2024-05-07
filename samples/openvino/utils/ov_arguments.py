import argparse
from pathlib import Path
from samples.common.arguments import Arguments
from samples.models import _PRECISION


class OvArguments(Arguments):
    def __init__(
        self,
        sample_name,
        media,
        model,
        device,
        nireq,
        batch_size,
        inference_interval,
        streams,
        threshold,
        output_dir,
        precision: _PRECISION,
        labels_path=None,
        num_frames=None,
    ):

        super().__init__(
            sample_name=sample_name,
            media=media,
            device=device,
            batch_size=batch_size,
            threshold=threshold,
            output_dir=output_dir,
            labels_path=labels_path,
            streams=streams,
            num_frames=num_frames,
        )

        self.parser.add_argument(
            "--model", type=str, default=model, help="Model name or path to model xml file"
        )
        self.parser.add_argument(
            "--precision",
            type=str,
            choices=["fp32", "fp16", "int8"],
            default=precision,
            help="Model precision",
        )
        self.parser.add_argument(
            "--nireq", help="Number of inference requests", type=int, default=nireq
        )
        self.parser.add_argument(
            "--inference-interval",
            default=inference_interval,
            type=int,
            help="Interval between inference requests. "
            "An interval of 1 performs inference on every frame. "
            "An interval of 2 performs inference on every other frame. "
            "An interval of N performs inference on every Nth frame.",
        )
        self.parser.add_argument("--profile", default=False, action=argparse.BooleanOptionalAction)
