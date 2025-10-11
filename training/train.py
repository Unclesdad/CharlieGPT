"""
Fine-tune Llama 3.1 8B using Unsloth and LoRA on user's Discord messages.

This script trains the model to mimic the user's writing style.
"""

import json
import os
import warnings
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Remove MPS memory cap before importing torch
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable memory limit

import torch
from datasets import Dataset
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Load environment variables
load_dotenv()

# Suppress bitsandbytes warnings on Mac
warnings.filterwarnings('ignore', message='.*bitsandbytes.*')

# Try to import unsloth, fall back to standard transformers if not available
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
    print("✓ Unsloth available - using optimized training")
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("⚠ Unsloth not available - using standard transformers (slower but works)")


def load_config() -> dict:
    """Load configuration from config.yaml."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_dataset(train_file: str, val_file: str) -> tuple:
    """Load training and validation datasets."""
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    with open(val_file, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    return train_data, val_data


def format_chat_template(example):
    """
    Format conversation into Llama 3.1 chat template.

    Llama 3.1 format:
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    {user message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    {assistant response}<|eot_id|>
    """
    conversation = example['conversations']

    formatted_text = "<|begin_of_text|>"

    for message in conversation:
        role = message['role']
        content = message['content']

        formatted_text += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
        formatted_text += f"{content}<|eot_id|>"

    return {'text': formatted_text}


def prepare_datasets(train_data: list, val_data: list):
    """Convert to HuggingFace datasets and apply formatting."""
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    # Apply chat template formatting
    train_dataset = train_dataset.map(format_chat_template)
    val_dataset = val_dataset.map(format_chat_template)

    return train_dataset, val_dataset


def setup_model_and_tokenizer(config: dict):
    """Load model and tokenizer with optimizations."""
    model_name = config['training']['base_model']
    max_seq_length = config['training']['max_seq_length']

    print(f"Loading model: {model_name}")
    print(f"Max sequence length: {max_seq_length}")

    if UNSLOTH_AVAILABLE:
        # Use Unsloth for faster training
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=config['training']['lora_rank'],
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=config['training']['lora_alpha'],
            lora_dropout=config['training']['lora_dropout'],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
    else:
        # Use standard transformers without quantization (for Mac)
        # Check if we're on Mac (MPS available) - no quantization needed
        use_quantization = torch.cuda.is_available()  # Only quantize on CUDA

        if use_quantization:
            # Use 4-bit quantization on CUDA
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

            # Prepare model for training
            model = prepare_model_for_kbit_training(model)
        else:
            # No quantization on Mac - use MPS (Apple GPU)
            print("Training on Mac with MPS (Apple Silicon GPU)")

            # Load to CPU first to avoid MPS buffer allocation issues
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 to match training config
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                use_cache=False,  # Disable KV cache to save memory
            )

            # Enable gradient checkpointing for memory efficiency
            model.gradient_checkpointing_enable()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Configure LoRA
        lora_config = LoraConfig(
            r=config['training']['lora_rank'],
            lora_alpha=config['training']['lora_alpha'],
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=config['training']['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)

        # Move model to MPS after LoRA is applied (much smaller memory footprint)
        if torch.backends.mps.is_available() and not torch.cuda.is_available():
            print("Moving model to MPS device...")
            model = model.to("mps")

    return model, tokenizer


def train(config: dict):
    """Main training function."""

    # Load datasets
    processed_data_dir = config['paths']['processed_data_dir']
    train_file = Path(processed_data_dir) / 'train_dataset.json'
    val_file = Path(processed_data_dir) / 'val_dataset.json'

    print("Loading datasets...")
    train_data, val_data = load_dataset(str(train_file), str(val_file))
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")

    # Prepare datasets
    print("\nPreparing datasets with chat template...")
    train_dataset, val_dataset = prepare_datasets(train_data, val_data)

    # Setup model and tokenizer
    print("\nSetting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(config)

    # Print model info
    print("\nModel configuration:")
    print(f"  LoRA rank: {config['training']['lora_rank']}")
    print(f"  LoRA alpha: {config['training']['lora_alpha']}")
    print(f"  LoRA dropout: {config['training']['lora_dropout']}")

    # Training arguments
    output_dir = config['training']['output_dir']
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Determine if we should use mixed precision
    # MPS doesn't support fp16 in Accelerate, only CUDA does
    use_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=2,  # Moderate accumulation for batch_size=4
        warmup_steps=10,
        num_train_epochs=config['training']['epochs'],
        learning_rate=config['training']['learning_rate'],
        fp16=use_fp16,  # Only use fp16 on CUDA, not MPS
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,  # More frequent evaluation with faster model
        save_strategy="steps",
        save_steps=100,  # More frequent saves
        save_total_limit=3,  # Keep more checkpoints
        optim="adamw_torch",  # Use standard PyTorch optimizer
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="none",  # Disable wandb/tensorboard
        dataloader_pin_memory=False,  # Disable pin memory for MPS
        gradient_checkpointing=True,  # Enable gradient checkpointing
        max_grad_norm=0.3,  # Gradient clipping for stability
    )

    # Create trainer with updated API
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # Show GPU/CPU memory stats
    gpu_stats = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    if gpu_stats:
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU: {gpu_stats.name}")
        print(f"GPU memory: {start_gpu_memory} GB / {max_memory} GB")

    # Train!
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    trainer.train()

    # Save final model
    print("\nSaving final model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save merged model
    merged_output_dir = f"{output_dir}_merged"
    if UNSLOTH_AVAILABLE:
        print("Saving merged model with Unsloth...")
        model.save_pretrained_merged(
            merged_output_dir,
            tokenizer,
            save_method="merged_16bit"
        )
    else:
        print("Saving merged model...")
        # Merge LoRA weights into base model
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)

    print("\n" + "="*60)
    print("Training complete!")
    print(f"LoRA adapters saved to: {output_dir}")
    print(f"Merged model saved to: {merged_output_dir}")
    print("="*60)


if __name__ == '__main__':
    # Load config
    config = load_config()

    # Check if HF token is set
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        print(f"Using Hugging Face token from environment")
    else:
        print("⚠️  Warning: HF_TOKEN not set. You may need it to download Llama models.")
        print("   Set it in your .env file or export HF_TOKEN=your_token")

    # Run training
    train(config)
