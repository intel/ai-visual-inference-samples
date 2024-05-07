import argparse

from samples.common.stop_condition import (
    DurationStopCondition,
    FramesProcessedStopCondition,
    EosStopCondition,
    StopCondition,
)


def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} not a floating-point literal")

    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x} not in range [0.0, 1.0]")
    return x


def text_file_to_list(file_path):
    lines = []
    with open(file_path) as fh:
        for line in fh:
            lines.append(line.strip())
    return lines


def get_stop_condition(args) -> StopCondition:
    if args.duration is not None:
        return DurationStopCondition(args.duration)
    if args.num_frames is not None:
        return FramesProcessedStopCondition(args.num_frames)
    if args.inference_only:
        return FramesProcessedStopCondition(args.batch_size)

    return EosStopCondition()
