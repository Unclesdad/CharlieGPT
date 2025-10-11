"""
Export trained model to GGUF format for llama.cpp inference.

This script converts the fine-tuned model to a quantized GGUF format
that can run efficiently on Raspberry Pi 5.
"""

import os
import subprocess
from pathlib import Path
import yaml

# Try to import unsloth for GGUF export
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("⚠ Unsloth not available - will need to use llama.cpp conversion manually")


def load_config() -> dict:
    """Load configuration from config.yaml."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def export_to_gguf(config: dict):
    """Export model to GGUF format with quantization."""

    model_dir = f"{config['training']['output_dir']}_merged"
    output_dir = config['paths']['models_dir']
    quantization = config['model']['quantization']

    print("="*60)
    print("Exporting model to GGUF format")
    print("="*60)

    if not UNSLOTH_AVAILABLE:
        print("\n⚠️  Unsloth not available for automatic GGUF export")
        print("\nManual export instructions:")
        print("1. Install llama.cpp:")
        print("   git clone https://github.com/ggerganov/llama.cpp")
        print("   cd llama.cpp && make")
        print("\n2. Convert model to GGUF:")
        print(f"   python llama.cpp/convert.py {model_dir} --outtype f16 --outfile {output_dir}/charliegpt-f16.gguf")
        print("\n3. Quantize to 4-bit:")
        print(f"   ./llama.cpp/quantize {output_dir}/charliegpt-f16.gguf {output_dir}/charliegpt-{quantization}.gguf {quantization}")
        return

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load the merged model
    print(f"\nLoading merged model from: {model_dir}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=config['training']['max_seq_length'],
        dtype=None,
        load_in_4bit=False,  # Load full precision for conversion
    )

    # Export to GGUF
    output_path = Path(output_dir) / f"charliegpt-{quantization}.gguf"

    print(f"\nExporting to GGUF with {quantization} quantization...")
    print(f"Output: {output_path}")

    model.save_pretrained_gguf(
        str(output_dir),
        tokenizer,
        quantization_method=quantization,  # Q4_K_M for 4-bit quantization
    )

    # The file will be saved with Unsloth's naming convention
    # Find the generated file
    gguf_files = list(Path(output_dir).glob("*.gguf"))

    if gguf_files:
        generated_file = gguf_files[0]
        final_path = Path(output_dir) / f"charliegpt-{quantization}.gguf"

        # Rename to our preferred name
        if generated_file != final_path:
            generated_file.rename(final_path)

        print(f"\n✓ Model exported successfully!")
        print(f"  Location: {final_path}")
        print(f"  Size: {final_path.stat().st_size / (1024**3):.2f} GB")
        print(f"  Quantization: {quantization}")

        # Print instructions for copying to RPi
        print("\n" + "="*60)
        print("Next steps:")
        print("="*60)
        print(f"\n1. Copy the model to your Raspberry Pi:")
        print(f"   rsync -avz {final_path} pi@raspberrypi:/home/pi/CharlieGPT/models/")
        print(f"\n2. Copy the vector database:")
        print(f"   rsync -avz {config['paths']['vectordb_dir']} pi@raspberrypi:/home/pi/CharlieGPT/")
        print(f"\n3. Copy the config file:")
        print(f"   rsync -avz config.yaml .env pi@raspberrypi:/home/pi/CharlieGPT/")
        print(f"\n4. On the RPi, run the Discord bot:")
        print(f"   python bot/bot.py")

    else:
        print("⚠️  Warning: No GGUF files found after export")


def main():
    config = load_config()

    # Check if merged model exists
    merged_model_dir = f"{config['training']['output_dir']}_merged"
    if not Path(merged_model_dir).exists():
        print(f"Error: Merged model not found at {merged_model_dir}")
        print("Please run training/train.py first to create the fine-tuned model.")
        return

    export_to_gguf(config)


if __name__ == '__main__':
    main()
