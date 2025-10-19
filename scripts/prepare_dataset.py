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
from datetime import datetime
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


def get_user_identifiers(user_id: str, username: str, instagram_username: str) -> List[str]:
    """Get all possible identifiers for the user across platforms."""
    identifiers = []
    if user_id:
        identifiers.append(user_id)
    if username:
        identifiers.append(username)
    if instagram_username:
        identifiers.append(instagram_username)
    return identifiers


def is_user_message(msg: Dict, user_identifiers: List[str]) -> bool:
    """Check if a message is from the user."""
    if msg.get('is_bot', False):
        return False

    # Check author_id or author_name
    author_id = msg.get('author_id', '')
    author_name = msg.get('author_name', '')

    return author_id in user_identifiers or author_name in user_identifiers


def is_from_2025(msg: Dict) -> bool:
    """Check if a message is from 2025."""
    timestamp = msg.get('timestamp')
    if not timestamp:
        return False

    try:
        # Parse ISO format timestamp
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.year == 2025
    except (ValueError, AttributeError):
        return False


def create_conversation_pairs(messages: List[Dict], user_identifiers: List[str]) -> List[Tuple[str, str]]:
    """
    Create instruction-response pairs from real conversation sequences.

    This looks at the previous messages before each user message to create
    realistic conversation pairs with actual context.
    Only includes messages from 2025.
    """
    # Filter to only 2025 messages
    messages_2025 = [msg for msg in messages if is_from_2025(msg)]

    # Sort messages by timestamp to ensure chronological order
    sorted_messages = sorted(messages_2025, key=lambda x: x.get('timestamp', ''))

    pairs = []
    context_window = 2  # Number of previous messages (reduced for faster training)

    for i, msg in enumerate(sorted_messages):
        # Only process messages from the user
        if not is_user_message(msg, user_identifiers):
            continue

        # Get context from previous messages (3-5 messages)
        start_idx = max(0, i - context_window)
        context_messages = sorted_messages[start_idx:i]

        # Skip if no context (can't learn from isolated messages)
        if not context_messages:
            continue

        # Build context string from previous messages
        context_parts = []
        for ctx_msg in context_messages:
            author = ctx_msg.get('author_name', 'Unknown')
            content = ctx_msg.get('content', '').strip()
            if content:
                context_parts.append(f"{author}: {content}")

        # Only add if we have actual context
        if not context_parts:
            continue

        # Create the context prompt (last 3-5 messages)
        context_str = "\n".join(context_parts)
        instruction = context_str

        # The user's actual response
        response = msg.get('content', '').strip()

        # Add the pair (we keep all responses, including short ones like "lol" per user request)
        if response:
            pairs.append((instruction, response))

    return pairs


def create_instruct_dataset(all_messages: List[Dict], user_identifiers: List[str]) -> List[Dict]:
    """Create instruction-following dataset from real conversation context."""

    # Group messages by conversation for better context
    # For Discord: guild_name + channel_name
    # For Instagram: conversation_name
    conversations = {}
    for msg in all_messages:
        if msg.get('source') == 'instagram':
            conv_key = f"instagram_{msg.get('conversation_name', 'unknown')}"
        else:
            # Discord message
            conv_key = f"discord_{msg.get('guild_name', 'unknown')}_{msg.get('channel_name', 'unknown')}"

        if conv_key not in conversations:
            conversations[conv_key] = []
        conversations[conv_key].append(msg)

    # Create conversation pairs from each conversation
    all_pairs = []
    for conv_name, messages in conversations.items():
        if len(messages) > 1:  # Need at least 2 messages for context
            pairs = create_conversation_pairs(messages, user_identifiers)
            all_pairs.extend(pairs)

    # Convert to instruction format for training
    dataset = []
    for instruction, response in all_pairs:
        dataset.append({
            'instruction': instruction,
            'response': response,
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
    instagram_username = os.getenv('INSTAGRAM_USERNAME', 'charlekerr')  # Default to charlekerr
    processed_data_dir = config['paths']['processed_data_dir']

    if not user_id and not username and not instagram_username:
        print("Error: At least one of DISCORD_USER_ID, DISCORD_USERNAME, or INSTAGRAM_USERNAME must be set in .env file")
        return

    # Get all user identifiers
    user_identifiers = get_user_identifiers(user_id, username, instagram_username)
    print(f"User identifiers: {user_identifiers}")

    print("\nLoading processed messages...")
    all_messages = load_messages(processed_data_dir)
    print(f"Total messages loaded: {len(all_messages)}")

    # Count messages by source
    discord_count = sum(1 for msg in all_messages if msg.get('source') != 'instagram')
    instagram_count = sum(1 for msg in all_messages if msg.get('source') == 'instagram')
    print(f"  Discord messages: {discord_count}")
    print(f"  Instagram messages: {instagram_count}")

    # Count 2025 messages
    messages_2025 = [msg for msg in all_messages if is_from_2025(msg)]
    print(f"\nMessages from 2025: {len(messages_2025)} ({len(messages_2025)*100//len(all_messages)}% of total)")
    discord_2025 = sum(1 for msg in messages_2025 if msg.get('source') != 'instagram')
    instagram_2025 = sum(1 for msg in messages_2025 if msg.get('source') == 'instagram')
    print(f"  Discord 2025: {discord_2025}")
    print(f"  Instagram 2025: {instagram_2025}")

    # Count user messages
    user_message_count = sum(1 for msg in all_messages if is_user_message(msg, user_identifiers))
    print(f"\nUser messages found: {user_message_count}")

    if user_message_count == 0:
        print("\n⚠️  Warning: No messages found for the specified user!")
        print("Please check your DISCORD_USER_ID, DISCORD_USERNAME, and INSTAGRAM_USERNAME in .env file")
        return

    print("\nCreating instruction dataset from real conversation context...")
    dataset = create_instruct_dataset(all_messages, user_identifiers)
    print(f"Created {len(dataset)} training examples")

    if len(dataset) == 0:
        print("\n⚠️  Warning: No training examples created!")
        print("This usually means there are no messages with sufficient context.")
        return

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
