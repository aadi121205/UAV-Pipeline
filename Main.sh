#!/bin/bash

# Main.sh - Script to run MAVProxy and pass the device to a Python script with error handling

MAVPROXY_CMD="mavproxy.py --master=/dev/ttyUSB0 --baudrate 57600"
PYTHON_SCRIPT="your_python_script.py"
DEVICE="/dev/ttyUSB0"

# Start MAVProxy
echo "Starting MAVProxy..."
$MAVPROXY_CMD &
MAVPROXY_PID=$!

# Wait a bit to ensure MAVProxy starts
sleep 5

# Check if MAVProxy started successfully
if ps -p $MAVPROXY_PID > /dev/null; then
    echo "MAVProxy started successfully (PID: $MAVPROXY_PID)."
else
    echo "Error: MAVProxy failed to start."
    exit 1
fi

# Run the Python script with the device as argument
echo "Running Python script..."
python3 "$PYTHON_SCRIPT" "$DEVICE"
PYTHON_EXIT_CODE=$?

if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    echo "Error: Python script exited with code $PYTHON_EXIT_CODE."
    # Optionally kill MAVProxy if needed
    kill $MAVPROXY_PID
    exit 2
fi

# Clean up: kill MAVProxy
kill $MAVPROXY_PID
wait $MAVPROXY_PID 2>/dev/null

echo "Done."