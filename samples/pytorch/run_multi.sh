#!/bin/bash
#
# Copyright (C) 2023 Intel Corporation.
#
# SPDX-License-Identifier: MIT
#

SAMPLE_NAME=
RESULTS_DIR=
N_PROCS=1
SAMPLE_ARGS=
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
MULTI_DEVICE=false
INSTALL_REQUIREMENTS=true

show_options() {
    echo ""
    echo "Running Sample: '${SAMPLE_NAME}'"
    echo "   Number of processes : '${N_PROCS}'"
    echo "   Multi-device: '${MULTI_DEVICE}'"
    echo "   Sample arguments: '${SAMPLE_ARGS}'"
    echo ""
}

show_help() {
    echo 'Usage: run_multi.sh --sample-name <sample> [--n-procs <value>] [--sample-args "<args>"] [--multi-device] [--install-requirements] [--help]'
    echo ""
    echo "Run sample in multi-process/multi-device mode"
    echo ""
    echo 'Example: ./run_multi.sh --sample-name SwinTransformer --n-procs 2 --sample-args "--device xpu:0"'
    echo ""
    echo "Options:"
    echo "  --sample-name <sampledir> Name of sample directory"
    echo "  --n-procs <num>           Number of processes to run [default: $N_PROCS]"
    echo "  --output-dir <dir>        Path to sample outputs dir [default: SAMPLE_NAME/output]"
    echo "  --sample-args <args>      Sample arguments"
    echo "  --multi-device            Distribute processes proportionally on available GPU devices"
    echo "  --install-requirements    Runs pip install -r requirements.txt in the sample directory if it exists"
    echo "  -?, -h, --help            Show help and usage information"
    exit 0
}

error() {
    printf '%s\n' "$1" >&2
    exit
}


while [[ "$#" -gt 0 ]]; do
    case $1 in
    -h | -\? | --help)
        show_help # Display a usage synopsis.
        exit
        ;;
    --sample-name) # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            SAMPLE_NAME="$2"
            shift
        else
            error 'ERROR: "--sample-name" requires a non-empty option argument.'
        fi
        ;;
    --n-procs)
        if [ "$2" ]; then
            N_PROCS=$2
            shift
        else
            error 'ERROR: "--n-procs" requires a non-empty option argument.'
        fi
        ;;
    --output-dir)
        if [ "$2" ]; then
            RESULTS_DIR=$2
            shift
        else
            error 'ERROR: "--output-dir" requires a non-empty option argument.'
        fi
        ;;
    --sample-args)
        if [ "$2" ]; then
            SAMPLE_ARGS+="$2 "
            shift
        else
            error 'ERROR: "--sample-args" requires a non-empty option argument.'
        fi
        ;;
    --multi-device)
        MULTI_DEVICE=true
        ;;
    --install-requirements)
        INSTALL_REQUIREMENTS=true
        ;;
    --) # End of all options.
        shift
        break
        ;;
    -?*)
        printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
        ;;
    *) # Default case: No more options, so break out of the loop.
        break ;;
    esac

    shift
done
echo $N_PROCS
echo $SAMPLE_ARGS
if [ -z "$SAMPLE_NAME" ]; then
    error '--sample-name must be set '
fi

SAMPLE_DIR="$SCRIPT_DIR/$SAMPLE_NAME"

if [ ! -d $SAMPLE_DIR ]; then
  error "Invalid sample directory ${SAMPLE_DIR}, please specify correct sample name"
fi

if [ -z "$RESULTS_DIR" ]; then
    RESULTS_DIR="$SAMPLE_DIR/output"
fi

SAMPLE_ARGS+="--output-dir $RESULTS_DIR "

show_options

if [ $INSTALL_REQUIREMENTS == true ]; then
    REQUIREMENT_FILE=$SAMPLE_DIR/requirements.txt
    if [[ -f "$REQUIREMENT_FILE" ]]; then
        echo "Running pip install -r $REQUIREMENT_FILE"
        pip install -r $REQUIREMENT_FILE
    fi
fi
rm -rf $RESULTS_DIR/*latency*.log

command="python $SAMPLE_DIR/main.py $SAMPLE_ARGS"

# Downloading models in first run without running full sample
eval "$command --only-download-model"

if [ $MULTI_DEVICE == true ]; then
    #distribute processes equally on all available devices
    device_count=$(ls -1 /dev/dri/render* | wc -l)
    procs_per_device=$(expr $N_PROCS / $device_count)
    device_number=0
    for ((n=1;n<=$N_PROCS;n++))
    do
        if [ $device_number -ge $device_count ]; then
            device_number=0
        fi
        multi_command="$command --device xpu:$device_number"
        echo "launching process $n"
        echo $multi_command
        eval $multi_command &
        pids[${n}]=$!
        device_number=$(expr $device_number + 1)
    done
else
    for ((n=1;n<=$N_PROCS;n++))
    do
        echo "launching process $n"
        echo $command
        eval $command &
        pids[${n}]=$!
    done

fi

echo "waiting for processes to complete"

failed=false
# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
    pid_status=$?
    if [[ ${pid_status} -ne 0 ]]; then
        failed=true
    fi
done

# Sleeping for all standard out to flush
sleep 2

if [ $failed == true ]; then
    error "One or more processes failed with non zero exit code, exiting"
fi

total_fps=0
total_latency=0
total_frames=0
for file in $RESULTS_DIR/*latency*.log
do
    fps=$(grep -Po 'Throughput :\K[^fps]*' $file | tail -1)
    total_fps=$(awk "BEGIN {printf \"%.4f\",${total_fps}+${fps}}")

    batch_size=$(grep -Po 'Batch_size: \K[^*]*' $file | tail -1)

    latency=$(grep -Po 'Total latency : \K[^ms]*' $file | tail -1)
    total_latency=$(awk "BEGIN {printf \"%.4f\",${total_latency}+${latency}}")

    frame_count=$(grep -Po 'Number of frames : \K[^*]*' $file | tail -1)
    frame_count=${frame_count%.*}
    total_frames=`expr $total_frames + $frame_count`
done

frame_per_process=`expr $total_frames / $N_PROCS`
avg_latency=$(awk "BEGIN {printf \"%.4f\",${total_latency}/${N_PROCS}}")
latency_per_frame=$(awk "BEGIN {printf \"%.4f\",${avg_latency}/${total_frames}}")

echo ""
echo "SUMMARY"
echo "   Number of Processes   : ${N_PROCS}"
echo "   Batch  Size           : ${batch_size}"
echo "   Total Throughput      : ${total_fps} fps"
echo "   Average Total Latency : ${avg_latency} ms"
echo "   Total Frames          : ${total_frames}"
echo "   Frames Per Process    : ${frame_per_process}"
echo "   Latency Per Frame     : ${latency_per_frame}"
echo ""
