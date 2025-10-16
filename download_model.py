#!/usr/bin/env python3
"""
Download Qwen model to cache before training.
This helps avoid timeouts during training startup.
"""

import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

load_dotenv()

model_name = "Qwen/Qwen2.5-7B-Instruct"

print(f"Downloading {model_name}...")
print("This is a ~15GB download and may take 10-30 minutes depending on your connection.")
print("")

# Get HF token if available
token = os.getenv('HF_TOKEN')

try:
    # Download entire model to cache
    cache_dir = snapshot_download(
        repo_id=model_name,
        token=token,
        resume_download=True,  # Resume if interrupted
        local_files_only=False,
    )

    print(f"\n✓ Model downloaded successfully!")
    print(f"  Cache location: {cache_dir}")
    print("\nYou can now run: ./train_background.sh")

except Exception as e:
    print(f"\n✗ Download failed: {e}")
    print("\nTroubleshooting:")
    print("  1. Check internet connection")
    print("  2. Check disk space: df -h")
    print("  3. Try again - download will resume where it left off")
    if "401" in str(e) or "403" in str(e):
        print("  4. This might be a gated model - set HF_TOKEN in .env")
