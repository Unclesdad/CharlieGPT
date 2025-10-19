"""
Parse Instagram message exports and extract structured message data.

This script reads Instagram JSON message files from data/inbox/ and outputs
structured JSON files to processed_data/ in a unified format compatible with Discord messages.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm


class InstagramMessageParser:
    """Parser for Instagram message JSON exports."""

    def __init__(self, conversation_dir: Path):
        self.conversation_dir = conversation_dir
        self.messages = []
        self.participants = []
        self.conversation_name = None

    def parse(self) -> List[Dict]:
        """Parse all message files in the conversation directory."""
        # Find all message_*.json files
        message_files = sorted(self.conversation_dir.glob('message_*.json'))

        if not message_files:
            return []

        # Parse each message file
        for message_file in message_files:
            self._parse_message_file(message_file)

        return self.messages

    def _parse_message_file(self, message_file: Path):
        """Parse a single message JSON file."""
        try:
            with open(message_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract participants (only needed once)
            if not self.participants:
                self.participants = data.get('participants', [])
                # Set conversation name to the other participant(s)
                # Filter out yourself from the name
                participant_names = [p.get('name', 'Unknown') for p in self.participants]
                self.conversation_name = ', '.join(participant_names)

            # Extract messages
            messages = data.get('messages', [])

            for msg in messages:
                parsed_msg = self._parse_message(msg)
                if parsed_msg:
                    self.messages.append(parsed_msg)

        except Exception as e:
            print(f"Error parsing {message_file}: {e}")

    def _parse_message(self, msg: Dict) -> Optional[Dict]:
        """Parse a single message into unified format."""
        try:
            # Get sender name
            sender_name = msg.get('sender_name', 'Unknown')

            # Get timestamp (milliseconds since epoch)
            timestamp_ms = msg.get('timestamp_ms')
            if timestamp_ms:
                timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0).isoformat()
            else:
                timestamp = None

            # Get content
            content = msg.get('content', '').strip()

            # Filter out attachment-only messages (reels, videos, photos)
            # Instagram formats these as "sender sent an attachment" or "You sent an attachment"
            if not content or 'sent an attachment' in content.lower():
                return None

            # Filter out pure reaction messages (e.g., "Reacted 😂 to your message")
            if content.startswith('Reacted ') and 'to your message' in content:
                return None

            # Check if there's a share link (Instagram posts, reels)
            if 'share' in msg:
                # Skip messages that are just shared reels/posts
                return None

            # Generate a message ID (Instagram doesn't provide one)
            message_id = f"ig_{timestamp_ms}" if timestamp_ms else f"ig_{hash(content)}"

            # Create unified message format
            unified_message = {
                'message_id': message_id,
                'author_id': sender_name,  # Instagram doesn't provide numeric IDs in exports
                'author_name': sender_name,
                'content': content,
                'timestamp': timestamp,
                'is_bot': False,
                'source': 'instagram',
                'conversation_name': self.conversation_name,
            }

            return unified_message

        except Exception as e:
            print(f"Error parsing message: {e}")
            return None


def process_instagram_inbox(inbox_dir: str, output_dir: str):
    """Process all Instagram conversations in the inbox directory."""
    inbox_path = Path(inbox_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all conversation directories
    conversation_dirs = [d for d in inbox_path.iterdir() if d.is_dir() and not d.name.startswith('.')]

    if not conversation_dirs:
        print(f"No conversation directories found in {inbox_dir}")
        return

    print(f"Found {len(conversation_dirs)} Instagram conversations to process")

    all_messages = []
    metadata = {
        'total_messages': 0,
        'conversations': [],
        'processed_conversations': [],
    }

    for conv_dir in tqdm(conversation_dirs, desc="Processing Instagram conversations"):
        try:
            # Parse messages
            parser = InstagramMessageParser(conv_dir)
            messages = parser.parse()

            if not messages:
                continue

            all_messages.extend(messages)

            # Update metadata
            conv_info = {
                'conversation_name': parser.conversation_name,
                'participants': [p.get('name') for p in parser.participants],
                'message_count': len(messages),
            }
            metadata['conversations'].append(conv_info)
            metadata['processed_conversations'].append(conv_dir.name)
            metadata['total_messages'] += len(messages)

            # Save individual conversation data
            safe_name = conv_dir.name.replace('/', '_').replace('\\', '_')
            conv_output_file = output_path / f"instagram_{safe_name}.json"
            with open(conv_output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'conversation_info': conv_info,
                    'messages': messages,
                }, f, indent=2, ensure_ascii=False)

            print(f"  Processed {conv_dir.name}: {len(messages)} messages")

        except Exception as e:
            print(f"  Error processing {conv_dir.name}: {e}")

    # Load existing all_messages.json if it exists (from Discord)
    all_messages_file = output_path / 'all_messages.json'
    existing_messages = []
    if all_messages_file.exists():
        print("\nMerging with existing messages from Discord...")
        with open(all_messages_file, 'r', encoding='utf-8') as f:
            existing_messages = json.load(f)

    # Combine and save all messages
    combined_messages = existing_messages + all_messages
    with open(all_messages_file, 'w', encoding='utf-8') as f:
        json.dump(combined_messages, f, indent=2, ensure_ascii=False)

    # Save Instagram-specific metadata
    instagram_metadata_file = output_path / 'instagram_metadata.json'
    with open(instagram_metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Instagram processing complete!")
    print(f"Instagram messages extracted: {metadata['total_messages']}")
    print(f"Conversations processed: {len(metadata['conversations'])}")
    print(f"Total messages (Discord + Instagram): {len(combined_messages)}")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Parse Instagram message exports')
    parser.add_argument('--inbox-dir', type=str, default='./data/inbox',
                        help='Directory containing Instagram inbox folders')
    parser.add_argument('--output-dir', type=str, default='./processed_data',
                        help='Directory to save processed JSON files')

    args = parser.parse_args()

    process_instagram_inbox(args.inbox_dir, args.output_dir)
