#!/bin/bash

# CharlieGPT Setup Script
# This script helps set up the project on Mac for training

set -e

echo "=================================="
echo "CharlieGPT Setup (Mac - Training)"
echo "=================================="

# Check Python version
echo -e "\nChecking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo -e "\nCreating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo -e "\nInstalling dependencies..."
pip install --upgrade pip
pip install -r requirements-training.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "\nCreating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your tokens!"
fi

# Check if config.yaml exists
if [ ! -f config.yaml ]; then
    echo "⚠️  config.yaml not found!"
    exit 1
fi

echo -e "\n=================================="
echo "Setup complete!"
echo "=================================="
echo -e "\nNext steps:"
echo "1. Edit .env with your Discord bot token and user ID"
echo "2. Edit config.yaml with your preferences"
echo "3. Place Discord HTML exports in the data/ directory"
echo "4. Run: python scripts/parse_exports.py"
echo "5. Run: python scripts/prepare_dataset.py"
echo "6. Run: python scripts/build_vectordb.py"
echo "7. Run: python scripts/add_wpilib_docs.py"
echo "8. Run: python training/train.py"
echo "9. Run: python training/export_model.py"
echo -e "\nSee README.md for detailed instructions"
