"""
Prepare fine-tuning dataset from parsed Discord messages.

This script filters messages to only include the user's messages and formats
them as instruction-response pairs for fine-tuning.
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def load_config() -> Dict:
    """Load configuration from config.yaml."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_messages(processed_data_dir: str) -> List[Dict]:
    """Load all processed messages."""
    messages_file = Path(processed_data_dir) / 'all_messages.json'

    if not messages_file.exists():
        raise FileNotFoundError(
            f"Messages file not found: {messages_file}\n"
            "Please run parse_exports.py first to process the HTML exports."
        )

    with open(messages_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def filter_user_messages(messages: List[Dict], user_id: str, username: str) -> List[Dict]:
    """Filter messages to only include those from the specified user."""
    user_messages = []

    for msg in messages:
        # Match by ID (preferred) or username (fallback)
        is_user_message = (
            (msg.get('author_id') == user_id) or
            (msg.get('author_name') == username)
        ) and not msg.get('is_bot', False)

        if is_user_message:
            user_messages.append(msg)

    return user_messages


def create_conversation_pairs(messages: List[Dict]) -> List[Tuple[str, str]]:
    """
    Create instruction-response pairs from message sequences.

    This looks at the context (previous messages) before each user message
    to create realistic conversation pairs.
    """
    # Sort messages by timestamp to ensure chronological order
    sorted_messages = sorted(messages, key=lambda x: x.get('timestamp', ''))

    pairs = []
    context_window = 5  # Number of previous messages to consider

    for i, msg in enumerate(sorted_messages):
        if msg.get('is_bot'):
            continue

        # Get context from previous messages
        start_idx = max(0, i - context_window)
        context_messages = sorted_messages[start_idx:i]

        # Build context string
        context_parts = []
        for ctx_msg in context_messages:
            author = ctx_msg.get('author_name', 'Unknown')
            content = ctx_msg.get('content', '').strip()
            if content:
                context_parts.append(f"{author}: {content}")

        # Create instruction based on context
        if context_parts:
            # If there's context, make it a conversation continuation
            context_str = "\n".join(context_parts[-3:])  # Last 3 messages
            instruction = f"Continue this conversation:\n{context_str}\n\nYour response:"
        else:
            # If no context, use the message as a standalone response
            # We'll create a generic prompt
            instruction = "Write a message in a casual, conversational tone:"

        response = msg.get('content', '').strip()

        # Only add if response is substantial (not just "lol" or emojis)
        if len(response) > 10:
            pairs.append((instruction, response))

    return pairs


def create_instruct_dataset(user_messages: List[Dict]) -> List[Dict]:
    """Create instruction-following dataset for fine-tuning."""

    # Group messages by channel for better context
    channels_messages = {}
    for msg in user_messages:
        channel_key = f"{msg.get('guild_name', 'unknown')}_{msg.get('channel_name', 'unknown')}"
        if channel_key not in channels_messages:
            channels_messages[channel_key] = []
        channels_messages[channel_key].append(msg)

    # Create conversation pairs from each channel
    all_pairs = []
    for channel, messages in channels_messages.items():
        if len(messages) > 0:
            pairs = create_conversation_pairs(messages)
            all_pairs.extend(pairs)

    # Convert to instruction format for training
    dataset = []
    for instruction, response in all_pairs:
        dataset.append({
            'instruction': instruction,
            'response': response,
        })

    # Also add some standalone messages with varied prompts
    prompts = [
        "Say something about robotics:",
        "Share your thoughts:",
        "What's on your mind?",
        "Respond to this conversation:",
        "Write a casual message:",
    ]

    for msg in user_messages:
        content = msg.get('content', '').strip()
        if len(content) > 15:  # Only substantial messages
            dataset.append({
                'instruction': random.choice(prompts),
                'response': content,
            })

    return dataset


def create_chat_format(dataset: List[Dict]) -> List[Dict]:
    """
    Convert dataset to chat format for Llama/instruction models.

    Format:
    {
        "conversations": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """
    chat_dataset = []

    for item in dataset:
        chat_dataset.append({
            "conversations": [
                {"role": "user", "content": item['instruction']},
                {"role": "assistant", "content": item['response']}
            ]
        })

    return chat_dataset


def split_dataset(dataset: List[Dict], train_ratio: float = 0.9) -> Tuple[List[Dict], List[Dict]]:
    """Split dataset into training and validation sets."""
    random.shuffle(dataset)
    split_idx = int(len(dataset) * train_ratio)

    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    return train_data, val_data


def main():
    # Load config
    config = load_config()

    # Get user info from environment variables
    user_id = os.getenv('DISCORD_USER_ID')
    username = os.getenv('DISCORD_USERNAME')
    processed_data_dir = config['paths']['processed_data_dir']

    if not user_id or not username:
        print("Error: DISCORD_USER_ID and DISCORD_USERNAME must be set in .env file")
        return

    print("Loading processed messages...")
    all_messages = load_messages(processed_data_dir)
    print(f"Total messages loaded: {len(all_messages)}")

    print(f"\nFiltering messages for user: {username} (ID: {user_id})")
    user_messages = filter_user_messages(all_messages, user_id, username)
    print(f"User messages found: {len(user_messages)}")

    if len(user_messages) == 0:
        print("\n⚠️  Warning: No messages found for the specified user!")
        print("Please check your DISCORD_USER_ID and DISCORD_USERNAME in .env file")
        return

    print("\nCreating instruction dataset...")
    dataset = create_instruct_dataset(user_messages)
    print(f"Created {len(dataset)} training examples")

    print("\nConverting to chat format...")
    chat_dataset = create_chat_format(dataset)

    print("\nSplitting into train/validation sets...")
    train_data, val_data = split_dataset(chat_dataset, train_ratio=0.9)
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")

    # Save datasets
    output_dir = Path(processed_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / 'train_dataset.json'
    val_file = output_dir / 'val_dataset.json'
    full_file = output_dir / 'full_dataset.json'

    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    with open(full_file, 'w', encoding='utf-8') as f:
        json.dump(chat_dataset, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("Dataset preparation complete!")
    print(f"Training set: {train_file}")
    print(f"Validation set: {val_file}")
    print(f"Full dataset: {full_file}")
    print(f"{'='*60}")

    # Show a sample
    print("\nSample training example:")
    print(json.dumps(train_data[0], indent=2))


if __name__ == '__main__':
    main()
