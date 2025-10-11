#!/bin/bash

# CharlieGPT Setup Script for Raspberry Pi 5
# This script sets up the bot for inference on RPi

set -e

echo "======================================="
echo "CharlieGPT Setup (Raspberry Pi - Bot)"
echo "======================================="

# Check Python version
echo -e "\nChecking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo -e "\nCreating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo -e "\nInstalling inference dependencies..."
pip install --upgrade pip
pip install -r requirements-inference.txt

# Install llama.cpp
echo -e "\nInstalling llama.cpp..."
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    make
    cd ..
    echo "✓ llama.cpp installed"
else
    echo "llama.cpp already installed"
fi

# Check for model file
echo -e "\nChecking for model file..."
if [ ! -f "models/charliegpt-Q4_K_M.gguf" ]; then
    echo "⚠️  Model file not found!"
    echo "Please copy the model from your Mac:"
    echo "  rsync -avz models/charliegpt-Q4_K_M.gguf pi@raspberrypi:/home/pi/CharlieGPT/models/"
fi

# Check for vector database
echo -e "\nChecking for vector database..."
if [ ! -d "vectordb" ] || [ -z "$(ls -A vectordb)" ]; then
    echo "⚠️  Vector database not found!"
    echo "Please copy the vectordb from your Mac:"
    echo "  rsync -avz vectordb/ pi@raspberrypi:/home/pi/CharlieGPT/vectordb/"
fi

# Check for .env file
if [ ! -f .env ]; then
    echo -e "\nCreating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your Discord bot token!"
fi

echo -e "\n======================================="
echo "Setup complete!"
echo "======================================="
echo -e "\nNext steps:"
echo "1. Edit .env with your Discord bot token"
echo "2. Ensure model and vectordb are copied from Mac"
echo "3. Run: python bot/bot.py"
echo -e "\nTo run the bot in the background:"
echo "  nohup python bot/bot.py > bot.log 2>&1 &"
