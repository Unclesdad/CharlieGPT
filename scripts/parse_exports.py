"""
Parse DiscordChatExporter HTML files and extract structured message data.

This script reads HTML exports from the data/ directory and outputs
structured JSON files to processed_data/.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from bs4 import BeautifulSoup
from tqdm import tqdm


class DiscordHTMLParser:
    """Parser for DiscordChatExporter HTML files."""

    def __init__(self, html_path: str):
        self.html_path = html_path
        self.soup = None
        self.messages = []
        self.guild_name = None
        self.channel_name = None

    def parse(self) -> List[Dict]:
        """Parse the HTML file and extract all messages."""
        with open(self.html_path, 'r', encoding='utf-8') as f:
            self.soup = BeautifulSoup(f.read(), 'html.parser')

        # Extract guild and channel info from preamble
        self._extract_channel_info()

        # Find all message containers
        message_containers = self.soup.find_all('div', class_='chatlog__message-container')

        print(f"Found {len(message_containers)} messages in {Path(self.html_path).name}")

        for container in message_containers:
            message_data = self._extract_message(container)
            if message_data:
                self.messages.append(message_data)

        return self.messages

    def _extract_channel_info(self):
        """Extract guild and channel information from the preamble."""
        preamble_entries = self.soup.find_all('div', class_='preamble__entry')

        if len(preamble_entries) >= 2:
            # First entry is guild name
            self.guild_name = preamble_entries[0].get_text(strip=True)

            # Second entry is "Category / Channel" or just "Channel"
            channel_path = preamble_entries[1].get_text(strip=True)
            # Extract just the channel name (after last /)
            if ' / ' in channel_path:
                self.channel_name = channel_path.split(' / ')[-1]
            else:
                self.channel_name = channel_path
        else:
            # Fallback: try to get from filename
            filename = Path(self.html_path).stem
            # Format: "Guild - Category - Channel [ID]"
            match = re.match(r'(.+?) - .+? - (.+?) \[(\d+)\]', filename)
            if match:
                self.guild_name = match.group(1)
                self.channel_name = match.group(2)
            else:
                self.guild_name = "Unknown Guild"
                self.channel_name = "Unknown Channel"

    def _extract_message(self, container) -> Optional[Dict]:
        """Extract structured data from a message container."""
        try:
            # Extract message ID from container
            message_id = container.get('data-message-id')

            # Find the message element
            message_elem = container.find('div', class_='chatlog__message')
            if not message_elem:
                return None

            # Find primary content area
            primary = message_elem.find('div', class_='chatlog__message-primary')
            if not primary:
                return None

            # Extract author information from header
            header = primary.find('div', class_='chatlog__header')
            if header:
                # New message with full header
                author_elem = header.find('span', class_='chatlog__author')
                if author_elem:
                    author_name = author_elem.get_text(strip=True)
                    author_id = author_elem.get('data-user-id')
                    title_attr = author_elem.get('title')
                    if title_attr:
                        author_name = title_attr  # Title has the original username
                else:
                    author_name = "Unknown"
                    author_id = None

                # Extract timestamp
                timestamp_elem = header.find('span', class_='chatlog__timestamp')
                if timestamp_elem:
                    timestamp_title = timestamp_elem.get('title')
                    timestamp = self._parse_timestamp(timestamp_title) if timestamp_title else None
                else:
                    timestamp = None
            else:
                # Continuation message (same author as previous)
                # These don't have headers, skip for now or use previous author
                return None

            # Extract message content
            content_elem = primary.find('div', class_='chatlog__content')
            if content_elem:
                # Get text content from markdown preserve span
                markdown_elem = content_elem.find('span', class_='chatlog__markdown-preserve')
                if markdown_elem:
                    content = markdown_elem.get_text(separator='\n', strip=False).strip()
                else:
                    content = content_elem.get_text(separator='\n', strip=False).strip()
            else:
                content = ""

            # Check if it's a bot message (usually has a tag)
            is_bot = primary.find('span', class_='chatlog__author-tag') is not None

            # Build message dictionary
            message_data = {
                'message_id': message_id,
                'author_id': author_id,
                'author_name': author_name,
                'content': content,
                'timestamp': timestamp,
                'is_bot': is_bot,
                'guild_name': self.guild_name,
                'channel_name': self.channel_name,
            }

            # Only return if we have actual content
            if content.strip():
                return message_data

        except Exception as e:
            print(f"Error parsing message: {e}")
            return None

    def _parse_timestamp(self, timestamp_str: str) -> Optional[str]:
        """Parse timestamp string to ISO format."""
        if not timestamp_str:
            return None

        try:
            # DiscordChatExporter format: "Thursday, October 13, 2022 5:07 PM"
            # Try to parse it
            dt = datetime.strptime(timestamp_str, "%A, %B %d, %Y %I:%M %p")
            return dt.isoformat()
        except ValueError:
            # Try alternative formats
            formats = [
                "%A, %B %d, %Y %I:%M:%S %p",
                "%m/%d/%Y %I:%M %p",
                "%Y-%m-%d %H:%M:%S",
            ]

            for fmt in formats:
                try:
                    dt = datetime.strptime(timestamp_str, fmt)
                    return dt.isoformat()
                except ValueError:
                    continue

            # If all else fails, return original string
            return timestamp_str
        except Exception as e:
            print(f"Error parsing timestamp '{timestamp_str}': {e}")
            return timestamp_str


def process_all_exports(data_dir: str, output_dir: str):
    """Process all HTML files in the data directory."""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all HTML files
    html_files = list(data_path.glob('*.html'))

    if not html_files:
        print(f"No HTML files found in {data_dir}")
        print("Please export your Discord channels using DiscordChatExporter and place them in the data/ directory")
        return

    print(f"Found {len(html_files)} HTML files to process")

    all_messages = []
    metadata = {
        'total_messages': 0,
        'channels': [],
        'processed_files': [],
    }

    for html_file in tqdm(html_files, desc="Processing exports"):
        try:
            # Parse messages
            parser = DiscordHTMLParser(str(html_file))
            messages = parser.parse()

            all_messages.extend(messages)

            # Update metadata
            channel_info = {
                'guild_name': parser.guild_name,
                'channel_name': parser.channel_name,
                'message_count': len(messages),
            }
            metadata['channels'].append(channel_info)
            metadata['processed_files'].append(html_file.name)
            metadata['total_messages'] += len(messages)

            # Save individual channel data
            channel_output_file = output_path / f"{html_file.stem}.json"
            with open(channel_output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'channel_info': channel_info,
                    'messages': messages,
                }, f, indent=2, ensure_ascii=False)

            print(f"✓ Processed {html_file.name}: {len(messages)} messages")

        except Exception as e:
            print(f"✗ Error processing {html_file.name}: {e}")

    # Save all messages to a single file
    all_messages_file = output_path / 'all_messages.json'
    with open(all_messages_file, 'w', encoding='utf-8') as f:
        json.dump(all_messages, f, indent=2, ensure_ascii=False)

    # Save metadata
    metadata_file = output_path / 'metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total messages extracted: {metadata['total_messages']}")
    print(f"Channels processed: {len(metadata['channels'])}")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Parse DiscordChatExporter HTML files')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory containing HTML exports')
    parser.add_argument('--output-dir', type=str, default='./processed_data',
                        help='Directory to save processed JSON files')

    args = parser.parse_args()

    process_all_exports(args.data_dir, args.output_dir)
