"""
CharlieGPT Discord Bot

A personalized Discord bot that responds in your writing style using a
fine-tuned LLM with RAG for contextual awareness.
"""

import os
import asyncio
from typing import Optional
import discord
from discord.ext import commands
import yaml
from dotenv import load_dotenv

from inference import LlamaCppInference
from rag import RAGRetriever


# Load environment variables
load_dotenv()


def load_config() -> dict:
    """Load configuration from config.yaml."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


class CharlieGPT(commands.Bot):
    """Custom Discord bot with LLM capabilities."""

    def __init__(self, config: dict):
        self.config = config

        # Set up intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True

        # Initialize bot
        super().__init__(
            command_prefix=config['discord']['command_prefix'],
            intents=intents
        )

        # Initialize inference engine
        print("Initializing inference engine...")
        self.inference = LlamaCppInference(config)

        # Initialize RAG retriever if enabled
        self.rag_enabled = config['rag']['enabled']
        if self.rag_enabled:
            print("Initializing RAG retriever...")
            self.rag = RAGRetriever(config)
        else:
            self.rag = None

        print("✓ Bot initialized successfully")

    async def on_ready(self):
        """Called when the bot is ready."""
        print(f'\n{"="*60}')
        print(f'CharlieGPT is online!')
        print(f'Logged in as: {self.user.name} (ID: {self.user.id})')
        print(f'Servers: {len(self.guilds)}')
        print(f'RAG enabled: {self.rag_enabled}')
        print(f'{"="*60}\n')

        # Set status
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="for messages | !charlie <msg>"
            )
        )

    async def on_message(self, message: discord.Message):
        """Handle incoming messages."""
        # Ignore own messages
        if message.author == self.user:
            return

        # Process commands first
        await self.process_commands(message)

        # Check if bot was mentioned
        if self.user.mentioned_in(message):
            # Remove the mention from the message
            content = message.content.replace(f'<@{self.user.id}>', '').strip()

            if content:
                await self.generate_response(message, content)

    async def get_channel_history(self, message: discord.Message, limit: int = 10) -> list[str]:
        """
        Get recent messages from the channel for immediate context.

        Args:
            message: The Discord message object
            limit: Number of recent messages to fetch

        Returns:
            List of formatted recent messages
        """
        history = []
        try:
            # Fetch recent messages (before the current message)
            async for msg in message.channel.history(limit=limit, before=message):
                # Skip bot messages and format the message
                if not msg.author.bot:
                    # Format: "username: message"
                    history.append(f"{msg.author.name}: {msg.content}")

            # Reverse to get chronological order (oldest first)
            history.reverse()

        except Exception as e:
            print(f"Error fetching channel history: {e}")

        return history

    async def generate_response(self, message: discord.Message, user_message: str):
        """
        Generate and send a response to a message.

        Args:
            message: The Discord message object
            user_message: The user's message content
        """
        # Show typing indicator
        async with message.channel.typing():
            try:
                # Get recent channel messages for immediate context
                channel_history_limit = self.config['rag'].get('channel_history_limit', 10)
                immediate_context = await self.get_channel_history(message, limit=channel_history_limit)

                if immediate_context:
                    print(f"\nChannel history ({len(immediate_context)} messages):")
                    for msg in immediate_context[-3:]:  # Show last 3
                        print(f"  {msg[:80]}...")

                # Retrieve relevant context if RAG is enabled
                rag_context = []
                if self.rag_enabled and self.rag:
                    rag_context = self.rag.get_mixed_context(user_message)

                    # Debug: print contexts
                    if rag_context:
                        print(f"\nRetrieved {len(rag_context)} RAG context(s) for: '{user_message}'")
                        for i, ctx in enumerate(rag_context, 1):
                            print(f"  {i}. {ctx[:100]}...")

                # Generate response
                response = await asyncio.to_thread(
                    self.inference.generate_with_context,
                    user_message,
                    immediate_context,
                    rag_context
                )

                # Ensure response isn't too long for Discord
                max_length = self.config['discord']['max_response_length']
                if len(response) > max_length:
                    response = response[:max_length-3] + "..."

                # Send response
                await message.reply(response)

            except Exception as e:
                print(f"Error generating response: {e}")
                await message.reply("Sorry, I encountered an error generating a response.")

    @commands.command(name='charlie')
    async def charlie_command(self, ctx: commands.Context, *, message: str):
        """
        Generate a response using the bot.

        Usage: !charlie <your message>
        """
        await self.generate_response(ctx.message, message)

    @commands.command(name='context')
    async def show_context(self, ctx: commands.Context, *, query: str):
        """
        Show what context would be retrieved for a query (debug command).

        Usage: !context <query>
        """
        if not self.rag_enabled or not self.rag:
            await ctx.reply("RAG is not enabled.")
            return

        contexts = self.rag.get_mixed_context(query)

        if not contexts:
            await ctx.reply("No relevant context found.")
            return

        # Format context for display
        response = f"Retrieved {len(contexts)} context(s) for: '{query}'\n\n"
        for i, ctx in enumerate(contexts[:3], 1):  # Show first 3
            # Truncate long contexts
            display_ctx = ctx[:200] + "..." if len(ctx) > 200 else ctx
            response += f"{i}. {display_ctx}\n\n"

        await ctx.reply(response[:2000])  # Discord message limit

    @commands.command(name='stats')
    async def show_stats(self, ctx: commands.Context):
        """Show bot statistics."""
        stats = f"""
**CharlieGPT Stats**

Model: {self.config['model']['name']}
Quantization: {self.config['model']['quantization']}
RAG enabled: {self.rag_enabled}
Servers: {len(self.guilds)}
"""

        if self.rag_enabled and self.rag:
            stats += f"Vector DB documents: {self.rag.collection.count()}\n"

        await ctx.reply(stats)

    @commands.command(name='help_charlie')
    async def help_command(self, ctx: commands.Context):
        """Show help message."""
        help_text = """
**CharlieGPT Commands**

`@CharlieGPT <message>` - Mention the bot to get a response
`!charlie <message>` - Generate a response
`!context <query>` - Show what context would be retrieved (debug)
`!stats` - Show bot statistics
`!help_charlie` - Show this help message

The bot uses a fine-tuned LLM trained on Discord messages to respond in a natural, conversational tone.
"""
        await ctx.reply(help_text)


def main():
    """Main entry point for the bot."""
    # Load config
    config = load_config()

    # Get bot token from environment
    token = os.getenv('DISCORD_BOT_TOKEN')
    if not token:
        print("Error: DISCORD_BOT_TOKEN not found in environment variables")
        print("Please set it in your .env file")
        return

    # Create and run bot
    bot = CharlieGPT(config)

    try:
        bot.run(token)
    except KeyboardInterrupt:
        print("\nShutting down bot...")
    except Exception as e:
        print(f"Error running bot: {e}")


if __name__ == '__main__':
    main()
