# CharlieGPT Quick Start Guide

This guide will get you up and running with CharlieGPT in ~30 minutes (excluding training time).

## Prerequisites

- Mac M3 Pro (for training)
- Raspberry Pi 5 16GB (for running the bot)
- Discord Bot Token ([create one here](https://discord.com/developers/applications))
- Hugging Face Token ([get one here](https://huggingface.co/settings/tokens)) - needed for Llama models
- Discord Channel Exports from DiscordChatExporter

## Part 1: Training on Mac (1-3 hours)

### Step 1: Setup
```bash
cd CharlieGPT
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### Step 2: Configure
Edit `.env`:
```bash
DISCORD_BOT_TOKEN=your_bot_token_here
DISCORD_USER_ID=your_user_id_here
DISCORD_USERNAME=your_username_here
HF_TOKEN=your_huggingface_token_here
```

Edit `config.yaml` with your Discord user ID and username.

### Step 3: Export Discord Data
1. Download [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter)
2. Export your Discord channels as HTML
3. Place all HTML files in `data/` directory

### Step 4: Process Data
```bash
# Parse Discord exports
python scripts/parse_exports.py

# Prepare training dataset
python scripts/prepare_dataset.py

# Build vector database
python scripts/build_vectordb.py

# Add WPILib documentation (optional but recommended for FRC teams)
python scripts/add_wpilib_docs.py
```

### Step 5: Train Model
```bash
# This will take 1-3 hours depending on dataset size
python training/train.py
```

### Step 6: Export for Raspberry Pi
```bash
# Convert to GGUF format
python training/export_model.py
```

## Part 2: Deploy to Raspberry Pi (15 minutes)

### Step 1: Copy Files to RPi
```bash
# From your Mac, in the CharlieGPT directory
rsync -avz --exclude='venv' --exclude='data' --exclude='.git' \
  . pi@raspberrypi:/home/pi/CharlieGPT/
```

### Step 2: Setup on RPi
```bash
# SSH into your Raspberry Pi
ssh pi@raspberrypi

# Navigate to project
cd /home/pi/CharlieGPT

# Run setup
chmod +x setup_rpi.sh
./setup_rpi.sh
source venv/bin/activate
```

### Step 3: Configure
```bash
# Edit .env on RPi
nano .env
# Add your Discord bot token
```

### Step 4: Run the Bot
```bash
# Test run
python bot/bot.py

# Or run in background
nohup python bot/bot.py > bot.log 2>&1 &
```

## Usage

Once the bot is running:

1. **Invite bot to your server**
   - Go to Discord Developer Portal
   - Select your bot application
   - Go to OAuth2 → URL Generator
   - Select `bot` scope and necessary permissions
   - Use generated URL to invite bot

2. **Interact with the bot**
   - Mention: `@CharlieGPT tell me about the robot`
   - Command: `!charlie what's up?`
   - Debug: `!context motor controllers` (shows retrieved context)
   - Stats: `!stats` (shows bot statistics)

## Troubleshooting

### Training Issues

**Out of Memory**
- Reduce `batch_size` in `config.yaml` (try 2 or 1)
- Reduce `max_seq_length` (try 1024)

**No messages found for user**
- Check `discord_id` and `discord_username` in `config.yaml`
- Run `python scripts/parse_exports.py` again and check output

**Unsloth installation fails**
- Make sure you're on Mac Silicon (M1/M2/M3)
- Try: `pip install --upgrade pip setuptools wheel`

### RPi Inference Issues

**Model not found**
- Ensure model file was copied: `ls -lh models/`
- Check path in `config.yaml`

**Too slow (>30 seconds)**
- Reduce `max_tokens` in `config.yaml` (try 256)
- Try Q3_K_M quantization instead of Q4_K_M

**Out of memory on RPi**
- Check you're using 16GB RPi5
- Try Q3_K_M or Q2_K quantization
- Reduce `context_length` in `config.yaml`

### Bot Issues

**Bot not responding**
- Check bot is online in Discord
- Verify token in `.env`
- Check bot has read/send permissions in channel
- View logs: `tail -f bot.log`

**Vector DB errors**
- Ensure vectordb was copied from Mac
- Try rebuilding: `python scripts/build_vectordb.py`

## Performance Expectations

**Training (Mac M3 Pro)**
- Time: 1-3 hours for ~3,500 messages
- Memory: ~8-12GB RAM
- Storage: ~20GB

**Inference (RPi5 16GB)**
- Response time: 10-20 seconds
- RAM usage: ~6-7GB
- Tokens/sec: 15-20

## Next Steps

- Fine-tune the personality by adjusting training epochs
- Add more documentation sources (e.g., Python docs, team wiki)
- Customize the system prompt in `bot/inference.py`
- Set up automatic startup on RPi boot
- Add more Discord commands

## Support

For issues, check:
1. README.md for detailed documentation
2. Bot logs on RPi: `tail -f bot.log`
3. Training logs in `models/charliegpt-lora/`

Enjoy your personalized Discord bot!
