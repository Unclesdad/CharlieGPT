#!/bin/bash
# Start CharlieGPT bot in the background

set -e

cd "$(dirname "$0")"

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Error: Do not run this script with sudo!"
    echo "Run it as your regular user: ./start_bot.sh"
    exit 1
fi

# Check if bot is already running
if [ -f bot.pid ]; then
    PID=$(cat bot.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "Bot is already running (PID: $PID)"
        echo "To stop it, run: ./stop_bot.sh"
        exit 1
    else
        # Stale PID file
        rm bot.pid
    fi
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Remove old log file if it exists and has wrong permissions
if [ -f bot.log ] && [ ! -w bot.log ]; then
    echo "Removing old log file with wrong permissions..."
    rm -f bot.log
fi

# Start bot in background
echo "Starting CharlieGPT bot..."
nohup python bot/bot.py > bot.log 2>&1 &
BOT_PID=$!
echo $BOT_PID > bot.pid

echo "✓ Bot started (PID: $BOT_PID)"
echo "  Log file: bot.log"
echo "  To view logs: tail -f bot.log"
echo "  To stop bot: ./stop_bot.sh"
