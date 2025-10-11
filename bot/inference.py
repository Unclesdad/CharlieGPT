"""
Model inference using llama.cpp for efficient generation on Raspberry Pi.

This module handles text generation using the fine-tuned model via llama.cpp.
"""

import subprocess
import json
from pathlib import Path
from typing import List, Dict, Optional
import yaml


def load_config() -> dict:
    """Load configuration from config.yaml."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


class LlamaCppInference:
    """Wrapper for llama.cpp inference."""

    def __init__(self, config: dict):
        self.config = config
        self.model_path = self._get_model_path()
        self.llama_cpp_path = Path(config['paths']['llama_cpp_path'])
        self.main_binary = self.llama_cpp_path / 'llama-cli'

        # Model parameters
        self.context_length = config['model']['context_length']
        self.max_tokens = config['model']['max_tokens']
        self.temperature = config['model']['temperature']
        self.top_p = config['model']['top_p']
        self.top_k = config['model']['top_k']

        # Verify setup
        self._verify_setup()

    def _get_model_path(self) -> Path:
        """Get the path to the GGUF model file."""
        models_dir = Path(self.config['paths']['models_dir'])
        quantization = self.config['model']['quantization']
        model_path = models_dir / f"charliegpt-{quantization}.gguf"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Please ensure the model has been exported and copied to the models directory."
            )

        return model_path

    def _verify_setup(self):
        """Verify llama.cpp is installed and model exists."""
        if not self.main_binary.exists():
            # Try alternative name
            self.main_binary = self.llama_cpp_path / 'main'

        if not self.main_binary.exists():
            raise FileNotFoundError(
                f"llama.cpp binary not found at {self.llama_cpp_path}\n"
                "Please install llama.cpp and update the path in config.yaml"
            )

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        print(f"✓ Inference engine initialized")
        print(f"  Model: {self.model_path.name}")
        print(f"  Binary: {self.main_binary}")

    def format_prompt(
        self,
        user_message: str,
        immediate_context: Optional[List[str]] = None,
        rag_context: Optional[List[str]] = None
    ) -> str:
        """
        Format the prompt with Llama 3.1 chat template.

        Args:
            user_message: The user's message
            immediate_context: Recent messages from the current Discord channel
            rag_context: Retrieved context from vector database

        Returns:
            Formatted prompt string
        """
        # Start with system message
        prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        prompt += "You are a helpful assistant responding in a casual, conversational tone."

        # Add immediate channel context (recent messages)
        if immediate_context:
            prompt += "\n\nRecent conversation in this channel:\n"
            for msg in immediate_context:
                prompt += f"{msg}\n"

        # Add RAG context (similar past conversations and documentation)
        if rag_context:
            prompt += "\n\nRelevant context from past conversations and documentation:\n"
            for ctx in rag_context:
                prompt += f"- {ctx}\n"

        prompt += "<|eot_id|>"

        # Add user message
        prompt += "<|start_header_id|>user<|end_header_id|>\n\n"
        prompt += f"{user_message}<|eot_id|>"

        # Start assistant response
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"

        return prompt

    def generate(self, prompt: str) -> str:
        """
        Generate text using llama.cpp.

        Args:
            prompt: Formatted prompt string

        Returns:
            Generated text
        """
        # Build command
        cmd = [
            str(self.main_binary),
            '-m', str(self.model_path),
            '-p', prompt,
            '-n', str(self.max_tokens),
            '-c', str(self.context_length),
            '--temp', str(self.temperature),
            '--top-p', str(self.top_p),
            '--top-k', str(self.top_k),
            '-ngl', '0',  # Number of layers to offload to GPU (0 for CPU only)
            '--no-display-prompt',  # Don't echo the prompt
        ]

        try:
            # Run inference
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )

            if result.returncode != 0:
                print(f"Error running llama.cpp: {result.stderr}")
                return "Sorry, I encountered an error generating a response."

            # Extract generated text
            generated_text = result.stdout.strip()

            # Clean up the output
            # Remove any remaining template tokens
            generated_text = generated_text.replace('<|eot_id|>', '')
            generated_text = generated_text.replace('<|end_header_id|>', '')

            return generated_text.strip()

        except subprocess.TimeoutExpired:
            return "Sorry, the response took too long to generate."
        except Exception as e:
            print(f"Error during inference: {e}")
            return "Sorry, I encountered an error."

    def generate_with_context(
        self,
        user_message: str,
        immediate_context: Optional[List[str]] = None,
        rag_context: Optional[List[str]] = None
    ) -> str:
        """
        Generate response with optional context.

        Args:
            user_message: The user's message
            immediate_context: Recent channel messages
            rag_context: Retrieved context from vector database

        Returns:
            Generated response
        """
        prompt = self.format_prompt(user_message, immediate_context, rag_context)
        return self.generate(prompt)


def test_inference():
    """Test the inference engine."""
    print("Testing inference engine...\n")

    config = load_config()
    inference = LlamaCppInference(config)

    # Test prompts
    test_prompts = [
        "Hey, what's up?",
        "Tell me about robotics",
        "How's your day going?",
    ]

    for prompt in test_prompts:
        print(f"\nUser: {prompt}")
        response = inference.generate_with_context(prompt)
        print(f"Bot: {response}")
        print("-" * 60)


if __name__ == '__main__':
    test_inference()
