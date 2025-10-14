"""
Local chat testing script for CharlieGPT.

Test your trained model locally on Mac before deploying to Raspberry Pi.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional, List

import torch
import yaml
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load environment variables
load_dotenv()


def load_config() -> dict:
    """Load configuration from config.yaml."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


class LocalChatTester:
    """Chat tester for locally testing the trained model."""

    def __init__(self, model_path: str, use_rag: bool = False, max_tokens: int = 512, temperature: float = 0.8):
        self.model_path = model_path
        self.use_rag = use_rag
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.rag = None

        print("Loading model and tokenizer...")
        self._load_model()

        if use_rag:
            print("Initializing RAG retriever...")
            self._load_rag()

    def _load_model(self):
        """Load the trained model (with quantization on CUDA, without on Mac)."""
        # Check if model exists
        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}\n"
                "Please ensure the model has been trained and the path is correct.\n"
                "Expected path: ./models/charliegpt-lora_merged/"
            )

        # Load model
        print(f"Loading from: {self.model_path}")

        if torch.cuda.is_available():
            # Use 4-bit quantization on CUDA for faster inference
            print("Loading with 4-bit quantization (CUDA)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # Load on CPU for Mac (MPS has temporary array size limitations)
            print("Loading on CPU (Mac)")
            print("Note: Inference will be slower on CPU but more stable")
            print("Loading model into RAM (this may take 30-60 seconds)...")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                device_map={"": "cpu"},  # Force load everything to CPU RAM
            )
            print("Model fully loaded into RAM")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        print("✓ Model loaded successfully")

    def _load_rag(self):
        """Load RAG retriever for context."""
        try:
            # Import RAG module
            import sys
            sys.path.append('bot')
            from rag import RAGRetriever

            config = load_config()
            self.rag = RAGRetriever(config)
            print("✓ RAG retriever loaded")
        except Exception as e:
            print(f"⚠️  Could not load RAG: {e}")
            print("Continuing without RAG context")
            self.rag = None
            self.use_rag = False

    def format_prompt(
        self,
        user_message: str,
        immediate_context: Optional[List[str]] = None,
        rag_context: Optional[List[str]] = None
    ) -> str:
        """
        Format prompt with Qwen 2.5 chat template.

        Args:
            user_message: The user's message
            immediate_context: Recent conversation history
            rag_context: Retrieved context from RAG

        Returns:
            Formatted prompt string
        """
        # Start with system message
        prompt = "<|im_start|>system\n"
        prompt += "You are a helpful assistant responding in a casual, conversational tone."

        # Add immediate context (conversation history)
        if immediate_context:
            prompt += "\n\nRecent conversation:\n"
            for msg in immediate_context:
                prompt += f"{msg}\n"

        # Add RAG context
        if rag_context:
            prompt += "\n\nRelevant context:\n"
            for ctx in rag_context:
                prompt += f"- {ctx}\n"

        prompt += "<|im_end|>\n"

        # Add user message
        prompt += "<|im_start|>user\n"
        prompt += f"{user_message}<|im_end|>\n"

        # Start assistant response
        prompt += "<|im_start|>assistant\n"

        return prompt

    def generate_response(
        self,
        user_message: str,
        conversation_history: Optional[List[str]] = None
    ) -> str:
        """
        Generate a response to the user's message.

        Args:
            user_message: The user's message
            conversation_history: Recent conversation history

        Returns:
            Generated response
        """
        # Get RAG context if enabled
        rag_context = None
        if self.use_rag and self.rag:
            try:
                rag_context = self.rag.get_mixed_context(user_message)
                if rag_context:
                    print(f"  [Retrieved {len(rag_context)} context(s)]")
            except Exception as e:
                print(f"  [RAG error: {e}]")

        # Format prompt
        prompt = self.format_prompt(user_message, conversation_history, rag_context)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.9,
                top_k=40,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode (keep special tokens to see what's happening)
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Debug: print what was generated
        print(f"\n[DEBUG] Full response:\n{full_response}\n")

        # Extract just the assistant's response (after the prompt)
        # Find the last assistant header
        assistant_marker = "<|im_start|>assistant"
        if assistant_marker in full_response:
            response = full_response.split(assistant_marker)[-1].strip()
        else:
            # Fallback: just remove the prompt
            response = full_response[len(prompt):].strip()

        # Clean up any remaining template tokens
        response = response.replace('<|im_end|>', '').strip()
        response = response.replace('<|im_start|>', '').strip()

        # Calculate stats
        elapsed_time = time.time() - start_time
        tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
        tokens_per_sec = tokens_generated / elapsed_time if elapsed_time > 0 else 0

        print(f"  [Generated {tokens_generated} tokens in {elapsed_time:.2f}s ({tokens_per_sec:.1f} tok/s)]")

        return response

    def chat_loop(self):
        """Run interactive chat loop."""
        print("\n" + "="*60)
        print("CharlieGPT Local Chat Tester")
        print("="*60)
        print("\nCommands:")
        print("  /help     - Show this help")
        print("  /rag on   - Enable RAG context retrieval")
        print("  /rag off  - Disable RAG context retrieval")
        print("  /clear    - Clear conversation history")
        print("  /exit     - Exit chat")
        print("\nStart chatting! (Type your message and press Enter)")
        print("="*60 + "\n")

        conversation_history = []

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    if user_input == '/exit':
                        print("\nGoodbye!")
                        break
                    elif user_input == '/help':
                        print("\nCommands:")
                        print("  /help     - Show this help")
                        print("  /rag on   - Enable RAG context retrieval")
                        print("  /rag off  - Disable RAG context retrieval")
                        print("  /clear    - Clear conversation history")
                        print("  /exit     - Exit chat")
                        continue
                    elif user_input == '/rag on':
                        if self.rag:
                            self.use_rag = True
                            print("✓ RAG enabled")
                        else:
                            print("⚠️  RAG not available (vector database not found)")
                        continue
                    elif user_input == '/rag off':
                        self.use_rag = False
                        print("✓ RAG disabled")
                        continue
                    elif user_input == '/clear':
                        conversation_history = []
                        print("✓ Conversation history cleared")
                        continue
                    else:
                        print(f"Unknown command: {user_input}")
                        continue

                # Generate response
                response = self.generate_response(user_input, conversation_history)

                # Print response
                print(f"\nBot: {response}\n")

                # Update conversation history (keep last 5 exchanges)
                conversation_history.append(f"You: {user_input}")
                conversation_history.append(f"Bot: {response}")
                if len(conversation_history) > 10:  # 5 exchanges = 10 messages
                    conversation_history = conversation_history[-10:]

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n⚠️  Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description='Test CharlieGPT locally')
    parser.add_argument(
        '--model-path',
        type=str,
        default='./models/charliegpt-lora_merged',
        help='Path to trained model directory'
    )
    parser.add_argument(
        '--rag',
        action='store_true',
        help='Enable RAG context retrieval'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Maximum tokens to generate (default: 512)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (default: 0.8)'
    )

    args = parser.parse_args()

    # Create chat tester
    tester = LocalChatTester(
        model_path=args.model_path,
        use_rag=args.rag,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )

    # Run chat loop
    tester.chat_loop()


if __name__ == '__main__':
    main()
