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
        decode_device=None,
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
            inference_interval=inference_interval,
            decode_device=decode_device,
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

        self.parser.add_argument("--profile", default=False, action=argparse.BooleanOptionalAction)
        self.parser.add_argument(
            "--media-only",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Runs media only pipeline and measures performance",
        )
