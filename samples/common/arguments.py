import argparse
from pathlib import Path
import logging
from samples.common.utils import restricted_float
from samples.common.logger import init_and_get_logger

DEFAULT_WARMUP_ITERATIONS = 4
DEFAULT_NUM_STREAMS = 2
DEFAULT_ASYNC_DECODE_DEPTH = 2


class Arguments:
    def __init__(
        self,
        sample_name,
        media,
        device,
        batch_size,
        threshold,
        output_dir,
        labels_path,
        num_frames=None,
        streams=DEFAULT_NUM_STREAMS,
        warmup_iterations=DEFAULT_WARMUP_ITERATIONS,
    ):
        self._parser = argparse.ArgumentParser(description=sample_name)

        self.parser.add_argument("--input", type=Path, default=media, help="Path to input media")
        self.parser.add_argument(
            "--labels-path", type=Path, default=labels_path, help="Path to labels text file"
        )
        self.parser.add_argument(
            "--device",
            help="Device to run pipeline on. Example xpu:0",
            type=str,
            default=device,
        )
        self.parser.add_argument(
            "--sample-name",
            type=str,
            default=sample_name,
            help="Sample Name to be use in creating log file names",
        )
        self.parser.add_argument("--batch-size", help="Batch size", type=int, default=batch_size)
        self.parser.add_argument(
            "--output-dir",
            help="Folder to store workload results",
            type=Path,
            default=output_dir,
        )
        self.parser.add_argument(
            "--log-predictions",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Log models predictions",
        )
        self.parser.add_argument(
            "--num-streams", help="Number of media streams", type=int, default=streams
        )
        self.parser.add_argument(
            "--inference-only",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Runs inference only pipeline and measures performance",
        )
        self.parser.add_argument(
            "--threshold",
            type=restricted_float,
            default=threshold,
            help="Threshold for inference results. "
            "Only objects with confidence values above the threshold will be considered",
        )
        self.parser.add_argument(
            "--async-decoding-depth",
            default=DEFAULT_ASYNC_DECODE_DEPTH,
            type=int,
            help="number of batches of frames to decode asynchronously",
        )
        self.parser.add_argument(
            "--warmup-iterations",
            default=warmup_iterations,
            type=int,
            help="Number of iterations to run warmup",
        )
        self.parser.add_argument(
            "--live-fps",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Live FPS output",
        )
        self.parser.add_argument(
            "--log-level", type=str, default=logging.INFO, help="Set log level"
        )
        self.parser.add_argument(
            "--only-download-models",
            default=False,
            action=argparse.BooleanOptionalAction,
            help="Download models and exit",
        )
        stop_rule_group = self.parser.add_mutually_exclusive_group(required=False)
        stop_rule_group.add_argument("--duration", help="Desired stream duration in sec", type=int)
        stop_rule_group.add_argument(
            "--num-frames",
            default=num_frames,
            help="Total number of frames processed during pipeline",
            type=int,
        )

        self._args = None

    @property
    def parser(self):
        return self._parser

    def parse_args(self):
        self._args = self.parser.parse_args()
        if not (Path(self._args.input).is_file()):
            raise ValueError(f"Cannot find input media {self._args.input}")
        self.validate()
        return self._args

    def validate(self):
        pass

    @property
    def args(self):
        return self._args

    def create_logger(self):
        logger = init_and_get_logger(
            self._args.sample_name,
            log_dir=self._args.output_dir,
            device=self._args.device,
            log_level=self._args.log_level,
        )
        logger.info(" ".join(f"{k}={v}\n" for k, v in vars(self.args).items()))
        return logger
