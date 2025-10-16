#!/bin/bash
# Train CharlieGPT 8B model in the background on Raspberry Pi
# This script runs training with low priority so it doesn't interfere with the bot

set -e

cd "$(dirname "$0")"

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Error: Do not run this script with sudo!"
    echo "Run it as your regular user: ./train_background.sh"
    exit 1
fi

# Check if training is already running
if [ -f training.pid ]; then
    PID=$(cat training.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "Training is already running (PID: $PID)"
        echo "To stop it, run: ./stop_training.sh"
        echo "To view logs: tail -f training.log"
        exit 1
    else
        # Stale PID file
        rm training.pid
    fi
fi

# Check if required files exist
if [ ! -f "config_training_8b.yaml" ]; then
    echo "Error: config_training_8b.yaml not found!"
    echo "Please ensure the training config file exists."
    exit 1
fi

if [ ! -d "processed_data" ]; then
    echo "Error: processed_data directory not found!"
    echo "Please transfer your training data from Mac first:"
    echo "  rsync -avz processed_data/ raspi@charliegpt.local:~/CharlieGPT/processed_data/"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: Virtual environment not found. Using system Python."
fi

# Remove old log file if it exists and has wrong permissions
if [ -f training.log ] && [ ! -w training.log ]; then
    echo "Removing old log file with wrong permissions..."
    rm -f training.log
fi

# Start training in background with low priority
echo "Starting 8B model training in background..."
echo "This will take approximately 3-7 days on Raspberry Pi."
echo ""
echo "Training settings:"
echo "  Model: Qwen/Qwen2.5-7B-Instruct (closest to 8B)"
echo "  Config: config_training_8b.yaml"
echo "  Output: ./models/charliegpt-8b-lora"
echo ""

# Use nice -n 19 for lowest priority (won't interfere with bot)
# Use nohup to survive SSH disconnection
nohup nice -n 19 python training/train.py --config config_training_8b.yaml > training.log 2>&1 &
TRAINING_PID=$!
echo $TRAINING_PID > training.pid

echo "✓ Training started (PID: $TRAINING_PID)"
echo "  Priority: Low (nice -n 19) - won't slow down bot"
echo "  Log file: training.log"
echo "  Config: config_training_8b.yaml"
echo ""
echo "Commands:"
echo "  View logs: tail -f training.log"
echo "  Check progress: grep -i 'step\|epoch\|loss' training.log | tail -20"
echo "  Stop training: ./stop_training.sh"
echo ""
echo "Note: Training will continue even if you close SSH."
echo "The bot will keep running normally during training."
