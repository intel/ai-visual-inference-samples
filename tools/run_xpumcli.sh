# List of supported metrics
METRICS="0,1,2,3,4,5,14,17,22,24,25,26,27,33,35"

# Discover number of devices
DEVICES_OUTPUT=$(xpumcli discovery --dump 1)
DEVICES_OUTPUT=$(echo "$DEVICES_OUTPUT" | sed 's/\\n/ /g')
DEVICES_OUTPUT=$(echo "$DEVICES_OUTPUT" | sed 's/Device ID//g')
DEVICES=($DEVICES_OUTPUT)

# Run either start/stop per input
COMMAND=$1
if [[ $COMMAND == "start" ]]; then
        for device in ${DEVICES[@]};
    do
        XPU_CMD="xpumcli dump --rawdata --start -d $device -m $METRICS"
        echo "Running command: $XPU_CMD"
        $XPU_CMD
    done
elif [[ $COMMAND == "stop" ]]; then
    # Expected output: Task <digit> is running
    TASK_LIST=$(xpumcli dump --rawdata --list)
    TASK_LIST=$(echo "$TASK_LIST" | sed 's/ //g')
    for task in ${TASK_LIST[@]};
    do
        task=$(echo "$task" | tr -cd [:digit:])
        XPU_CMD="xpumcli dump --rawdata --stop $task"
        echo "Running command: $XPU_CMD"
        $XPU_CMD
    done
else
        echo "Please specify either start/stop"
        exit 1;
fi
exit 0;

