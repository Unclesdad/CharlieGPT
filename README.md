# CharlieGPT

A personalized Discord bot trained on your messages to mimic your writing style, powered by fine-tuned Llama 3.1 8B with RAG (Retrieval Augmented Generation) for contextual awareness.

## Overview

CharlieGPT combines:
- **Fine-tuned LLM**: Learns your writing style from your Discord messages
- **RAG System**: Retrieves relevant context from team discussions
- **Discord Bot**: Responds in Discord channels mimicking your voice

## System Requirements

### For Training (Mac M3 Pro)
- macOS with Apple Silicon (M3)
- Python 3.10+
- 16GB+ RAM recommended
- ~20GB free disk space

### For Inference (Raspberry Pi 5)
- Raspberry Pi 5 with 16GB RAM
- 64-bit OS (Ubuntu or Raspberry Pi OS)
- Python 3.10+
- ~15GB free disk space

## Setup Instructions

### Phase 1: Data Preparation (Mac)

1. **Export Discord Messages**
   - Use [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) to export channels as HTML
   - Place HTML files in the `data/` directory
   - Export all channels from both servers you want to train on

2. **Install Dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Mac/Linux
   pip install -r requirements-training.txt
   ```

3. **Configure Settings**
   ```bash
   cp .env.example .env
   # Edit .env with your Discord token, user ID, and Hugging Face token
   # Edit config.yaml with your preferences
   ```

4. **Process Discord Exports**
   ```bash
   # Parse HTML exports
   python scripts/parse_exports.py

   # Prepare fine-tuning dataset (your messages only)
   python scripts/prepare_dataset.py

   # Build vector database (all messages for RAG)
   python scripts/build_vectordb.py
   ```

### Phase 2: Model Training (Mac)

5. **Fine-tune the Model**
   ```bash
   python training/train.py
   ```
   This will:
   - Download Llama 3.1 8B Instruct base model
   - Fine-tune with LoRA on your messages
   - Save adapter weights to `models/charliegpt-lora/`
   - Takes ~1-3 hours depending on dataset size

6. **Export for llama.cpp**
   ```bash
   python training/export_model.py
   ```
   This converts the model to GGUF format with 4-bit quantization for efficient inference.

### Phase 3: Deploy to Raspberry Pi 5

7. **Transfer Files to RPi**
   ```bash
   # On Mac, from project directory
   rsync -avz --exclude='venv' --exclude='data' --exclude='.git' \
     . pi@raspberrypi:/home/pi/CharlieGPT/
   ```

8. **Install llama.cpp on RPi**
   ```bash
   # On RPi
   cd /home/pi/CharlieGPT
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   make
   cd ..
   ```

9. **Install Python Dependencies on RPi**
   ```bash
   # On RPi
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements-inference.txt
   ```

10. **Run the Discord Bot**
    ```bash
    # On RPi
    python bot/bot.py
    ```

## Project Structure

```
CharlieGPT/
├── data/                          # Discord HTML exports (gitignored)
├── scripts/
│   ├── parse_exports.py          # Parse HTML to structured JSON
│   ├── prepare_dataset.py        # Create fine-tuning dataset
│   └── build_vectordb.py         # Build RAG vector database
├── training/
│   ├── train.py                  # Fine-tuning script (LoRA)
│   ├── export_model.py           # Convert to GGUF format
│   └── configs/                  # Training configurations
├── bot/
│   ├── bot.py                    # Main Discord bot
│   ├── inference.py              # Model inference with llama.cpp
│   └── rag.py                    # RAG retrieval system
├── processed_data/               # Parsed messages (gitignored)
├── models/                       # Trained models (gitignored)
├── vectordb/                     # Vector database (gitignored)
├── config.yaml                   # Main configuration
├── .env                          # Environment variables (gitignored)
├── requirements-training.txt     # Mac dependencies
└── requirements-inference.txt    # RPi dependencies
```

## Usage

Once the bot is running on your RPi:

1. Invite the bot to your Discord server
2. The bot will listen to messages in channels it has access to
3. Mention the bot or use commands to interact:
   - `@CharlieGPT <message>` - Get a response in your style
   - `!charlie <message>` - Alternative command
   - `!context` - Show retrieved context for debugging

## How It Works

1. **User sends message** → Discord bot receives it
2. **Channel history** → Bot fetches last 10 messages from current channel for immediate context
3. **RAG retrieval** → Bot searches vector database for relevant past conversations and documentation
4. **Context injection** → Both immediate context and RAG context are added to the prompt
5. **Model inference** → Fine-tuned model generates response in your style
6. **Response sent** → Bot sends message back to Discord

## Performance

On Raspberry Pi 5 (16GB):
- Model size: ~4.5GB RAM (4-bit quantized)
- Response time: 10-20 seconds
- Throughput: ~15-20 tokens/second

## Troubleshooting

### Training issues
- If CUDA/Metal errors occur, make sure you have the latest transformers/torch
- For memory issues, reduce `batch_size` in `config.yaml`

### RPi inference issues
- If responses are too slow, reduce `max_tokens` in `config.yaml`
- If out of memory, try Q3_K_M quantization instead of Q4_K_M

### Discord bot issues
- Make sure bot token is correct in `.env`
- Ensure bot has proper permissions in Discord server
- Check bot can read/send messages in channels

## License

MIT

## Credits

Built with:
- [Llama 3.1](https://ai.meta.com/llama/) by Meta
- [Unsloth](https://github.com/unslothai/unsloth) for optimized training
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for efficient inference
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [discord.py](https://discordpy.readthedocs.io/) for Discord integration
