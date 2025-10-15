#!/usr/bin/env python3
"""
Test script to debug llama.cpp subprocess hanging issue.
Run this on the Raspberry Pi to test if subprocess works.
"""

import subprocess
import time
from pathlib import Path

# Simple test prompt (like the working command line test)
simple_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHey what's up?<|im_end|>\n<|im_start|>assistant\n"

# More complex prompt (like what the bot generates)
complex_prompt = """<|im_start|>system
You are a helpful assistant responding in a casual, conversational tone.

Recent conversation in this channel:
unclesdad: test message 1
unclesdad: test message 2

Relevant context from past conversations and documentation:
- [general] user1: some context
- [general] user2: more context
<|im_end|>
<|im_start|>user
hey<|im_end|>
<|im_start|>assistant
"""

llama_binary = Path.home() / "llama.cpp/build/bin/llama-cli"
model_path = Path.home() / "CharlieGPT/models/charliegpt-Q4_K_M.gguf"

def test_prompt(prompt, description):
    """Test a prompt with llama.cpp."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Prompt length: {len(prompt)} chars")
    print(f"{'='*60}")

    cmd = [
        str(llama_binary),
        '-m', str(model_path),
        '-p', prompt,
        '-n', '50',
        '-c', '512',
        '--temp', '0.8',
        '--top-p', '0.9',
        '--top-k', '40',
        '-ngl', '0',
        '--no-display-prompt',
        '-t', '4',
        '--mlock',
    ]

    try:
        start_time = time.time()
        print("Running subprocess...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL  # Close stdin - don't wait for input
        )

        elapsed = time.time() - start_time
        print(f"✓ Completed in {elapsed:.1f}s")
        print(f"Return code: {result.returncode}")
        print(f"Output length: {len(result.stdout)} chars")
        print(f"First 200 chars: {result.stdout[:200]}")

        if result.stderr:
            print(f"Stderr: {result.stderr[:200]}")

    except subprocess.TimeoutExpired:
        print("✗ TIMEOUT after 30 seconds")
    except Exception as e:
        print(f"✗ ERROR: {e}")

if __name__ == '__main__':
    print("Testing llama.cpp subprocess behavior")
    print(f"Binary: {llama_binary}")
    print(f"Model: {model_path}")

    # Test 1: Simple prompt (should work)
    test_prompt(simple_prompt, "Simple prompt (known to work)")

    # Test 2: Complex prompt (like the bot)
    test_prompt(complex_prompt, "Complex prompt (like bot generates)")
