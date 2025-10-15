#!/bin/bash
# Stop CharlieGPT bot

set -e

cd "$(dirname "$0")"

if [ ! -f bot.pid ]; then
    echo "No bot.pid file found. Is the bot running?"
    exit 1
fi

PID=$(cat bot.pid)

if ps -p $PID > /dev/null 2>&1; then
    echo "Stopping CharlieGPT bot (PID: $PID)..."
    kill $PID
    rm bot.pid
    echo "✓ Bot stopped"
else
    echo "Bot process (PID: $PID) is not running"
    rm bot.pid
fi
