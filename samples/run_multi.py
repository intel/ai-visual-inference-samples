#!/usr/bin/env python3

import argparse
import os
import subprocess  # nosec B404
import sys
from pathlib import Path
import itertools
import shlex
import logging

# Constants
DEFAULT_N_PROCS = 1
DEFAULT_N_DEVICES = 1
DEFAULT_INSTALL_REQUIREMENTS = True

logger = logging.getLogger("run_multi")


def configure_logger(log_level=logging.INFO):
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)8s] %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run sample in multi-process/multi-device mode")
    parser.add_argument("--sample-name", required=True, help="Name of sample directory")
    parser.add_argument(
        "--backend",
        required=True,
        choices=["pytorch", "openvino"],
        help="Backend to run the sample on",
    )
    parser.add_argument(
        "--n-procs",
        "--num-processes",
        type=int,
        default=DEFAULT_N_PROCS,
        help="Number of processes per device",
        dest="n_procs",
    )
    parser.add_argument(
        "--output-dir",
        help="Path to sample outputs dir [default: SAMPLE_NAME/output]",
        type=Path,
        default=None,
    )
    parser.add_argument("--sample-args", default="", help="Sample arguments")
    parser.add_argument(
        "--num-devices",
        type=int,
        default=DEFAULT_N_DEVICES,
        help="Number of devices to run the sample on",
    )
    parser.add_argument(
        "--use-taskset", action="store_true", help="Use taskset to bind processes to CPUs"
    )
    parser.add_argument(
        "--install-requirements",
        action="store_true",
        help="Runs pip install -r requirements.txt in the sample directory if it exists",
    )
    return parser.parse_args()


def show_options(args):
    logger.info(f"Running Sample: '{args.sample_name}'")
    logger.info(f"   Number of processes : {args.n_procs}")
    logger.info(f"   Number of devices   : {args.num_devices}")
    logger.info(f"   Sample arguments    : '{args.sample_args}'")


def run_command(command):
    logger.info(f"Running command: {command}")
    command = shlex.split(command)
    return subprocess.Popen(
        command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )  # nosec B603


def install_requirements(sample_dir):
    requirement_file = sample_dir / "requirements.txt"
    if requirement_file.exists():
        p = run_command(f"pip install -r {requirement_file}")
        if p.wait() != 0:
            raise RuntimeError(f"Failed to install requirements for sample {sample_dir}")


def generate_taskset_command(process_id, n_procs):
    if process_id >= n_procs:
        raise ValueError(f"Process ID {process_id} is greater than number of processes {n_procs}")
    if process_id >= os.cpu_count():
        raise ValueError(f"Process ID {process_id} is greater than number of CPUs {os.cpu_count()}")
    total_cores = os.cpu_count()
    cores_per_proc = total_cores // n_procs
    remainder = total_cores % n_procs
    cpu_list = f"{process_id*cores_per_proc + min(process_id, remainder)}-{(process_id+1)*cores_per_proc + min(process_id+1, remainder) - 1}"
    return f"taskset -c {cpu_list}"


def run_sample(args, sample_dir):
    results_dir = args.output_dir or (sample_dir / "output")
    results_dir.mkdir(parents=True, exist_ok=True)
    sample_args = f"{args.sample_args} --output-dir {results_dir}"
    total_number_of_processes = args.n_procs * args.num_devices
    if "GPU_CPU" in args.sample_name and not "--nireq" in sample_args:
        nireq = os.cpu_count() // total_number_of_processes or 1
        sample_args = f"{sample_args} --nireq {nireq}"
    show_options(args)

    if args.install_requirements:
        install_requirements(sample_dir)

    command = f"python3 {sample_dir / 'main.py'} {sample_args}"
    download_models_comand = f"{command} --only-download-models"
    logger.info(f"Running download models command: {download_models_comand}")
    subprocess.run(
        shlex.split(download_models_comand),
        shell=False,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )  # nosec B603

    # Clear previous latency logs
    for latency_file in results_dir.glob("*latency*.log"):
        latency_file.unlink()

    pids = []
    for device_id, process_id in itertools.product(
        range(0, args.num_devices), range(0, args.n_procs)
    ):
        devices_args = f"--decode-device xpu:{device_id}"
        if not "GPU_CPU" in args.sample_name:
            devices_args = f"--device xpu:{device_id} {devices_args}"
        launch_command = f"{command} {devices_args}"
        if args.use_taskset:
            launch_command = f"{generate_taskset_command(device_id * args.n_procs + process_id, total_number_of_processes)} {launch_command}"
        logger.info(f"Launching process: {process_id} on device: {device_id}")
        pids.append(run_command(launch_command))

    logger.info("Waiting for processes to complete")
    for p in pids:
        p.wait()
        if p.returncode != 0:
            stdout, stderr = p.communicate()
            raise RuntimeError(
                f"One or more processes failed with non-zero exit code. Return code: {p.returncode}. stdout: {stdout}. stderr: {stderr}"
            )

    # Process results
    process_results(results_dir, args.n_procs, args.num_devices)


def process_results(results_dir, n_procs, num_devices):
    total_fps, total_latency, total_frames = 0, 0, 0
    for file in results_dir.glob("*latency*.log"):
        with open(file) as f:
            content = f.read()
            fps = float(search("Throughput\s*:\s*(\d+\.\d+)", content))
            total_fps += fps

            batch_size = search("Batch_size\s*:\s*(\d+)", content)
            latency = float(search("Total latency\s*:\s*(\d+\.\d+)", content))
            total_latency += latency

            frame_count = int(search("Number of frames\s*:\s*(\d+)", content))
            total_frames += frame_count

    frame_per_process = total_frames // n_procs
    avg_latency = total_latency / n_procs
    latency_per_frame = avg_latency / total_frames

    logger.info("SUMMARY")
    logger.info(f"   Number of Processes   : {n_procs}")
    logger.info(f"   Number of Devices     : {num_devices}")
    logger.info(f"   Batch  Size           : {batch_size}")
    logger.info(f"   Total Throughput      : {total_fps:.2f} fps")
    logger.info(f"   Average Total Latency : {avg_latency:.2f} ms")
    logger.info(f"   Total Frames          : {total_frames}")
    logger.info(f"   Frames Per Process    : {frame_per_process}")
    logger.info(f"   Latency Per Frame     : {latency_per_frame:.2f}\n")


def search(pattern, text):
    import re

    match = re.search(pattern, text)
    if match:
        return match.group(1)
    raise RuntimeError(f"Pattern {pattern} not found in text")


def main():
    configure_logger(log_level=logging.INFO)
    args = parse_arguments()
    script_dir = Path(__file__).resolve().parent
    sample_dir = script_dir / args.backend / args.sample_name

    if not sample_dir.is_dir():
        raise RuntimeError(
            f"Invalid sample directory {sample_dir}, please specify correct sample name"
        )

    run_sample(args, sample_dir)


if __name__ == "__main__":
    main()
