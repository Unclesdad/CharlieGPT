# CharlieGPT

A personalized Discord bot that mimics your writing style using fine-tuned language models with RAG (Retrieval Augmented Generation) for contextual awareness.

## Overview

CharlieGPT is a three-layer AI system that:
1. **Fine-tunes** a language model on your Discord messages to learn your writing style
2. **Retrieves** relevant context from past conversations using RAG
3. **Generates** responses that sound like you, with awareness of conversation history

The bot runs on a Raspberry Pi 5 for 24/7 availability, while training happens on a Mac for speed.

## Architecture

```
┌─────────────────────────────────────────────┐
│  Mac M3 Pro (Training)                      │
│  • Process Discord exports                  │
│  • Fine-tune 3B/7B model with LoRA          │
│  • Export to GGUF format                    │
└─────────────────────────────────────────────┘
                    ↓ Transfer model
┌─────────────────────────────────────────────┐
│  Raspberry Pi 5 (Inference)                 │
│  • Discord bot (bot/bot.py)                 │
│  • llama.cpp inference engine               │
│  • ChromaDB vector database (RAG)           │
│  • 24/7 uptime                              │
└─────────────────────────────────────────────┘
```

### Three-Layer Context System

1. **Fine-tuning Layer**: Model learns your writing style, tone, and vocabulary
2. **RAG Layer**: Retrieves relevant past messages and documentation for context
3. **Immediate Context**: Uses recent channel messages for conversation flow

## Data Collection

You need your own Discord messages for training. Here are methods to collect data:

### Method 1: DiscordChatExporter (Recommended)

**Step 1: Install DiscordChatExporter**
- Download from: https://github.com/Tyrrrz/DiscordChatExporter
- Or install with: `dotnet tool install -g DiscordChatExporter.Cli`

**Step 2: Get Your Discord Token**
1. Open Discord in a web browser
2. Press `Ctrl+Shift+I` (Windows) or `Cmd+Option+I` (Mac) to open DevTools
3. Go to the Console tab
4. Type: `(webpackChunkdiscord_app.push([[''],{},e=>{m=[];for(let c in e.c)m.push(e.c[c])}]),m).find(m=>m?.exports?.default?.getToken!==void 0).exports.default.getToken()`
5. Copy the token (keep it secret!)

**Step 3: Export Channels**
```bash
# Export a single channel (replace CHANNEL_ID and YOUR_TOKEN)
DiscordChatExporter.Cli export -c CHANNEL_ID -t YOUR_TOKEN -f HtmlDark

# Export entire server (replace GUILD_ID)
DiscordChatExporter.Cli exportguild -g GUILD_ID -t YOUR_TOKEN -f HtmlDark

# Export all accessible channels
DiscordChatExporter.Cli exportall -t YOUR_TOKEN -f HtmlDark
```

**Step 4: Save HTML Files**
- Place all exported `.html` files in `data/` directory
- The parser will extract your messages automatically

### Method 2: Mac iMessage (Brief)
- iMessage database: `~/Library/Messages/chat.db`
- Use SQL queries or third-party tools to export
- Convert to required JSON format (see below)

### Method 3: Instagram Data Download (Brief)
1. Go to Instagram → Settings → Privacy & Security → Data Download
2. Request your data (takes 24-48 hours)
3. Extract messages from the JSON files
4. Convert to required format

## Data Format

The training system expects JSON files with this structure:

```json
[
  {
    "conversations": [
      {"role": "user", "content": "Hey, what's up?"},
      {"role": "assistant", "content": "not much, just working on robotics stuff"}
    ]
  },
  {
    "conversations": [
      {"role": "user", "content": "Did you see the match?"},
      {"role": "assistant", "content": "yeah that autonomous was insane"}
    ]
  }
]
```

- **`conversations`**: List of messages forming a dialogue
- **`role`**: Either `"user"` (messages to you) or `"assistant"` (your responses)
- **`content`**: The actual message text

## Setup

### Mac Setup (Training)

**1. Clone Repository**
```bash
git clone https://github.com/yourusername/CharlieGPT.git
cd CharlieGPT
```

**2. Run Setup Script**
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create Python virtual environment
- Install PyTorch, Transformers, and training dependencies
- Set up the project structure

**3. Configure Environment**

Create a `.env` file:
```bash
cp .env.example .env
```

Edit `.env` and add:
```env
# Hugging Face token (for downloading base models)
HF_TOKEN=your_huggingface_token_here

# Discord Bot Configuration (for deployment)
DISCORD_BOT_TOKEN=your_discord_bot_token
DISCORD_USER_ID=your_discord_user_id
DISCORD_USERNAME=your_discord_username
```

Get HuggingFace token:
- Sign up at https://huggingface.co
- Go to Settings → Access Tokens → Create new token
- Select "Read" permissions

**4. Configure Training Settings**

Edit `config.yaml`:
```yaml
# Training Settings
training:
  base_model: "Qwen/Qwen2.5-3B-Instruct"  # or Qwen2.5-7B-Instruct for better quality
  output_dir: "./models/charliegpt-lora"
  epochs: 1
  batch_size: 4  # Reduce if running out of memory
  max_seq_length: 256
  lora_rank: 8
```

Adjust settings based on your Mac's memory:
- **16GB RAM**: Use 3B model, batch_size=4
- **32GB+ RAM**: Use 7B model, batch_size=4-8

### Raspberry Pi Setup (Deployment)

**1. Install Dependencies**
```bash
ssh raspi@your-pi-hostname
cd ~
git clone https://github.com/yourusername/CharlieGPT.git
cd CharlieGPT
chmod +x setup_rpi.sh
./setup_rpi.sh
```

**2. Configure llama.cpp Path**

Edit `config.yaml` on Pi:
```yaml
paths:
  llama_cpp_path: "/home/raspi/llama.cpp/build/bin"  # Adjust to your path
```

**3. Set Up Environment**

Create `.env` file with Discord bot credentials (same as Mac setup).

## Training Workflow

### Step 1: Process Discord Exports

Place your `.html` exports in `data/` directory, then:

```bash
# Parse HTML exports to JSON
python scripts/parse_exports.py

# Prepare training datasets (train/validation split)
python scripts/prepare_dataset.py
```

This creates:
- `processed_data/train_dataset.json` - Training data
- `processed_data/val_dataset.json` - Validation data
- `processed_data/all_messages.json` - All messages for RAG

### Step 2: Build Vector Database (RAG)

```bash
python scripts/build_vectordb.py
```

This creates `vectordb/` directory with ChromaDB index for context retrieval.

Optional: Add external documentation (e.g., WPILib for FRC robotics):
```bash
python scripts/add_wpilib_docs.py
```

### Step 3: Train the Model

On your Mac:

```bash
python training/train.py
```

Training progress:
```
Loading datasets...
Training examples: 3500
Validation examples: 400

Setting up model and tokenizer...
Loading model: Qwen/Qwen2.5-3B-Instruct

Training...
Epoch 1/1: 100%|███████| 875/875 [1:23:45<00:00,  5.8s/it]

Training complete!
LoRA adapters saved to: ./models/charliegpt-lora
Merged model saved to: ./models/charliegpt-lora_merged
```

**Expected Training Times** (Mac M3 Pro):
- 3B model: 1-2 hours
- 7B model: 4-8 hours

### Step 4: Export to GGUF Format

```bash
python training/export_model.py
```

This creates:
- `models/charliegpt-f16.gguf` - Full precision (~6GB for 3B)
- `models/charliegpt-Q4_K_M.gguf` - Quantized (~2GB for 3B)

### Step 5: Transfer to Raspberry Pi

```bash
# Transfer quantized model
rsync -avz --progress models/charliegpt-Q4_K_M.gguf raspi@your-pi:~/CharlieGPT/models/

# Transfer vector database
rsync -avz --progress vectordb/ raspi@your-pi:~/CharlieGPT/vectordb/

# Transfer processed data (for rebuilding vectordb)
rsync -avz --progress processed_data/ raspi@your-pi:~/CharlieGPT/processed_data/
```

## Discord Bot Setup

### 1. Create Discord Bot

1. Go to https://discord.com/developers/applications
2. Click "New Application"
3. Go to "Bot" tab → Click "Add Bot"
4. Under "Privileged Gateway Intents", enable:
   - ✅ MESSAGE CONTENT INTENT
   - ✅ SERVER MEMBERS INTENT
5. Copy bot token → Add to `.env` as `DISCORD_BOT_TOKEN`

### 2. Invite Bot to Server

1. Go to "OAuth2" → "URL Generator"
2. Select scopes:
   - ✅ `bot`
3. Select permissions:
   - ✅ Read Messages/View Channels
   - ✅ Send Messages
   - ✅ Read Message History
4. Copy generated URL and open in browser to invite bot

### 3. Get Your Discord User ID

1. Enable Developer Mode: Discord Settings → Advanced → Developer Mode
2. Right-click your username → Copy User ID
3. Add to `.env` as `DISCORD_USER_ID`

## Running the Bot

### On Raspberry Pi

**Start the bot:**
```bash
./start_bot.sh
```

**View logs:**
```bash
tail -f bot.log
```

**Stop the bot:**
```bash
./stop_bot.sh
```

The bot runs in the background and survives SSH disconnection.

### Bot Usage

**Mention the bot:**
```
@CharlieGPT hey, what do you think about this strategy?
```

**Use command:**
```
!charlie how's it going?
```

**Debug commands:**
```
!stats          # Show bot statistics
!context query  # Show what context would be retrieved
!help_charlie   # Show help message
```

## Configuration Reference

### config.yaml

```yaml
# Discord Bot Settings
discord:
  command_prefix: "!"
  max_response_length: 2000

# Model Settings (for inference)
model:
  quantization: "Q4_K_M"  # Model file to use
  context_length: 1024    # Max context window
  max_tokens: 200         # Max response length
  temperature: 0.8        # Creativity (0.0-2.0)

# RAG Settings
rag:
  enabled: true
  num_contexts: 2               # Number of past messages to retrieve
  channel_history_limit: 5      # Recent messages from current channel
  embedding_model: "all-MiniLM-L6-v2"

# Training Settings (for Mac)
training:
  base_model: "Qwen/Qwen2.5-3B-Instruct"
  output_dir: "./models/charliegpt-lora"
  epochs: 1
  batch_size: 4
  learning_rate: 0.0002
  lora_rank: 8
  max_seq_length: 256

# Paths
paths:
  data_dir: "./data"
  processed_data_dir: "./processed_data"
  models_dir: "./models"
  vectordb_dir: "./vectordb"
  llama_cpp_path: "~/llama.cpp/build/bin"
```

### Tuning Response Quality

**Make responses longer/shorter:**
```yaml
model:
  max_tokens: 200  # Increase for longer, decrease for shorter
```

**Make responses more/less creative:**
```yaml
model:
  temperature: 0.8  # Higher = more creative, lower = more consistent
```

**Improve context awareness:**
```yaml
rag:
  num_contexts: 5           # More past context
  channel_history_limit: 10 # More recent messages
```

**Speed vs quality trade-off:**
- Use `Q4_K_M` quantization for fast inference (~2-5 seconds)
- Use `Q8_0` or `f16` for better quality but slower (~5-10 seconds)

## Project Structure

```
CharlieGPT/
├── bot/
│   ├── bot.py              # Discord bot implementation
│   ├── inference.py        # llama.cpp wrapper for generation
│   └── rag.py              # RAG retriever with ChromaDB
├── training/
│   ├── train.py            # Fine-tuning script
│   └── export_model.py     # GGUF export script
├── scripts/
│   ├── parse_exports.py    # Parse DiscordChatExporter HTML
│   ├── prepare_dataset.py  # Create train/val splits
│   ├── build_vectordb.py   # Build vector database
│   └── add_wpilib_docs.py  # Add FRC documentation to RAG
├── data/                   # Raw Discord exports (.html)
├── processed_data/         # Processed JSON datasets
├── models/                 # Trained models and GGUF files
├── vectordb/               # ChromaDB vector database
├── config.yaml             # Main configuration
├── .env                    # Environment variables (secrets)
├── setup.sh               # Mac setup script
├── setup_rpi.sh           # Raspberry Pi setup script
├── start_bot.sh           # Start bot in background
└── stop_bot.sh            # Stop bot
```

## Troubleshooting

### Training Issues

**Out of memory:**
- Reduce `batch_size` in config.yaml
- Use smaller model (3B instead of 7B)
- Reduce `max_seq_length`

**Model download fails:**
- Check HF_TOKEN in .env
- Try downloading manually: `huggingface-cli download Qwen/Qwen2.5-3B-Instruct`

### Bot Issues

**Bot not responding:**
- Check privileged intents enabled in Discord Developer Portal
- Verify bot token in .env
- Check bot.log for errors

**"Response took too long":**
- Reduce max_tokens in config.yaml
- Check CPU usage: `top` or `htop`
- Ensure only one bot instance running

**RAG not working:**
- Rebuild vector database: `python scripts/build_vectordb.py`
- Check vectordb/ directory exists and has content

### Performance

**Inference too slow on Pi:**
- Use Q4_K_M quantization (not f16)
- Reduce max_tokens and context_length
- Ensure no other heavy processes running

**Training too slow on Mac:**
- Close other applications
- Use 3B model instead of 7B
- Reduce max_seq_length

## Tips for Best Results

1. **Training Data Quality**:
   - Use 3,000-10,000 of YOUR messages
   - Include diverse conversations (casual, technical, etc.)
   - More data = better style mimicry

2. **Context Tuning**:
   - Adjust RAG retrieval count based on response relevance
   - Too much context = slower, too little = less accurate

3. **Response Length**:
   - Discord messages are usually short (1-3 sentences)
   - Set max_tokens to 100-200 for Discord-like responses

4. **Temperature Settings**:
   - 0.7-0.8: Balanced (recommended)
   - 0.5-0.6: More conservative, safer
   - 0.9-1.0: More creative, riskier

## License

MIT License

## Acknowledgments

- Built on [Qwen 2.5](https://huggingface.co/Qwen) language models
- Powered by [llama.cpp](https://github.com/ggerganov/llama.cpp) for efficient inference
- Uses [ChromaDB](https://www.trychroma.com/) for vector storage
- Training with [🤗 Transformers](https://huggingface.co/transformers) and [PEFT](https://github.com/huggingface/peft)
