#!/bin/bash
# Stop background training

set -e

cd "$(dirname "$0")"

if [ ! -f training.pid ]; then
    echo "No training.pid file found. Is training running?"
    exit 1
fi

PID=$(cat training.pid)

if ps -p $PID > /dev/null 2>&1; then
    echo "Stopping training (PID: $PID)..."
    kill $PID
    rm training.pid
    echo "✓ Training stopped"
    echo ""
    echo "Note: The model may have saved checkpoints in:"
    echo "  ./models/charliegpt-8b-lora/"
    echo ""
    echo "To resume training from a checkpoint, you'll need to modify"
    echo "the training script to load from the last checkpoint."
else
    echo "Training process (PID: $PID) is not running"
    rm training.pid
fi
