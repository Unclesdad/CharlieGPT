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
        self.llama_cpp_path = Path(config['paths']['llama_cpp_path']).expanduser()
        self.main_binary = self.llama_cpp_path / 'llama-cli'
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
        Format the prompt with Qwen 2.5 chat template.

        Args:
            user_message: The user's message
            immediate_context: Recent messages from the current Discord channel
            rag_context: Retrieved context from vector database

        Returns:
            Formatted prompt string
        """
        prompt = "<|im_start|>system\n"
        prompt += "You are CharlieGPT, a helpful AI assistant. Respond in a casual, conversational tone. You can insult people when it's funny."

        if immediate_context:
            prompt += "\n\nRecent conversation:\n"
            for msg in immediate_context:
                prompt += f"{msg}\n"

        if rag_context:
            prompt += "\n\nRelevant context:\n"
            for ctx in rag_context:
                prompt += f"- {ctx}\n"

        prompt += "<|im_end|>\n"
        prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"

        return prompt

    def _is_prompt_repetition(self, generated: str) -> bool:
        """Check if the generated text is just repeating the prompt."""
        generated_lower = generated.lower()

        # Check if it's repeating system prompt
        system_indicators = [
            'you are charliegpt',
            'helpful ai assistant',
            'recent conversation',
            'relevant context',
            'i am a helpful assistant',
            'i am charliegpt'
        ]

        for indicator in system_indicators:
            if generated_lower.startswith(indicator):
                return True

        return False

    def _fix_encoding(self, text: str) -> str:
        """Fix common UTF-8 encoding issues in generated text."""
        # Fix double-encoded UTF-8 (common with Instagram/Discord data)
        # These are UTF-8 smart quotes/apostrophes that got mangled
        replacements = {
            # Smart quotes and apostrophes
            '\u00e2\u0080\u0099': "'",  # right single quote
            '\u00e2\u0080\u0098': "'",  # left single quote
            '\u00e2\u0080\u009c': '"',  # left double quote
            '\u00e2\u0080\u009d': '"',  # right double quote
            '\u00e2\u0080\u0094': '—',  # em dash
            '\u00e2\u0080\u0093': '–',  # en dash
            '\u00e2\u0080\u00a6': '...',  # ellipsis
            # Also try the visible form
            'â\u0080\u0099': "'",
            'â\u0080\u0098': "'",
            'â\u0080\u009c': '"',
            'â\u0080\u009d': '"',
            'â\u0080\u0094': '—',
            'â\u0080\u0093': '–',
            'donât': "don't",
            'theyâre': "they're",
            'itâs': "it's",
            'Iâm': "I'm",
            'youâre': "you're",
            'weâre': "we're",
            'theyâve': "they've",
            'wouldnât': "wouldn't",
            'couldnât': "couldn't",
            'shouldnât': "shouldn't",
        }

        for bad, good in replacements.items():
            text = text.replace(bad, good)

        return text

    def generate(self, prompt: str) -> str:
        """
        Generate text using llama.cpp.

        Args:
            prompt: Formatted prompt string

        Returns:
            Generated text
        """
        cmd = [
            str(self.main_binary),
            '-m', str(self.model_path),
            '-p', prompt,
            '-n', str(self.max_tokens),
            '-c', str(self.context_length),
            '--temp', '0.9',
            '--top-p', str(self.top_p),
            '--top-k', '60',
            '--repeat-penalty', '1.1',
            '-ngl', '0',
            '--no-display-prompt',
            '-t', '4',
            '--mlock',
        ]

        try:
            import time
            start_time = time.time()

            print(f"  Running llama.cpp with prompt length: {len(prompt)} chars")
            print(f"  Prompt preview: {prompt[:200]}...")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                stdin=subprocess.DEVNULL
            )

            elapsed = time.time() - start_time
            print(f"  Inference took {elapsed:.1f}s")
            print(f"  Return code: {result.returncode}")

            if result.returncode != 0:
                print(f"Error running llama.cpp:")
                print(f"  stderr: {result.stderr[:500]}")
                print(f"  stdout: {result.stdout[:500]}")
                return "Sorry, I encountered an error generating a response."

            generated_text = result.stdout.strip()

            print(f"  Raw output length: {len(generated_text)} chars")
            print(f"  First 200 chars: {generated_text[:200]}")

            # Extract only the assistant's response, stopping at end token or next turn
            if '<|im_end|>' in generated_text:
                generated_text = generated_text.split('<|im_end|>')[0].strip()

            if '<|im_start|>' in generated_text:
                generated_text = generated_text.split('<|im_start|>')[0].strip()

            if '> EOF by user' in generated_text:
                generated_text = generated_text.split('> EOF by user')[0].strip()

            generated_text = generated_text.replace('<|im_end|>', '').replace('<|im_start|>', '').strip()

            # Detect prompt repetition
            if self._is_prompt_repetition(generated_text):
                print("  Warning: Detected prompt repetition, returning fallback")
                return "what"

            # Fix UTF-8 encoding issues
            generated_text = self._fix_encoding(generated_text)

            return generated_text

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
