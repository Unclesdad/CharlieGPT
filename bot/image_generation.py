"""
Image generation using Latent Consistency Models (LCM) for fast inference on Raspberry Pi.
"""

import os
import re
from pathlib import Path
from typing import Optional
import torch
from diffusers import DiffusionPipeline
from PIL import Image


class ImageGenerator:
    """Generate images using LCM for fast CPU inference."""

    def __init__(self):
        self.model = None
        self.device = "cpu"  # Pi 5 doesn't have GPU
        self.output_dir = Path("./generated_images")
        self.output_dir.mkdir(exist_ok=True)

    def load_model(self):
        """Load the LCM model (lazy loading to save memory)."""
        if self.model is not None:
            return

        print("Loading LCM image generation model...")

        # Use LCM-LoRA with Dreamshaper for fast generation
        self.model = DiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            torch_dtype=torch.float32,  # CPU needs float32
        )

        self.model = self.model.to(self.device)

        # Optimize for CPU
        self.model.safety_checker = None  # Remove safety checker for speed

        print("LCM model loaded successfully")

    def generate_image(self, prompt: str, filename: Optional[str] = None) -> Path:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the image
            filename: Optional custom filename (will be auto-generated if not provided)

        Returns:
            Path to the generated image file
        """
        self.load_model()

        # Clean up the prompt
        prompt = self._clean_prompt(prompt)

        print(f"Generating image for prompt: {prompt}")

        # Generate with LCM (only needs 4-8 steps)
        image = self.model(
            prompt=prompt,
            num_inference_steps=6,  # LCM is fast with few steps
            guidance_scale=1.0,  # LCM works best with low guidance
            width=512,
            height=512,
        ).images[0]

        # Save the image
        if filename is None:
            # Generate filename from prompt
            safe_prompt = re.sub(r'[^\w\s-]', '', prompt)[:50]
            safe_prompt = re.sub(r'[-\s]+', '-', safe_prompt)
            filename = f"{safe_prompt}.png"

        output_path = self.output_dir / filename
        image.save(output_path)

        print(f"Image saved to {output_path}")
        return output_path

    def _clean_prompt(self, prompt: str) -> str:
        """Clean up the prompt text."""
        # Remove @Meta AI and /imagine commands
        prompt = re.sub(r'@Meta AI', '', prompt, flags=re.IGNORECASE)
        prompt = re.sub(r'/imagine', '', prompt, flags=re.IGNORECASE)
        prompt = prompt.strip()

        # Limit length
        if len(prompt) > 200:
            prompt = prompt[:200]

        return prompt


def extract_imagine_prompt(text: str) -> Optional[str]:
    """
    Extract the prompt from a /imagine command.

    Args:
        text: The message text to search

    Returns:
        The prompt if found, None otherwise
    """
    # Match patterns like:
    # - "@Meta AI /imagine a cat"
    # - "/imagine a dog"
    patterns = [
        r'@Meta\s+AI\s+/imagine\s+(.+)',
        r'/imagine\s+(.+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


# Singleton instance
_generator = None


def get_generator() -> ImageGenerator:
    """Get the global image generator instance."""
    global _generator
    if _generator is None:
        _generator = ImageGenerator()
    return _generator
