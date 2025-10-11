"""
RAG (Retrieval Augmented Generation) system for context retrieval.

This module retrieves relevant context from the vector database to enhance
the bot's responses with knowledge from past conversations and documentation.
"""

from typing import List, Dict, Optional
from pathlib import Path
import yaml

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


def load_config() -> dict:
    """Load configuration from config.yaml."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


class RAGRetriever:
    """Retrieve relevant context from vector database."""

    def __init__(self, config: dict):
        self.config = config
        self.vectordb_dir = config['paths']['vectordb_dir']
        self.num_contexts = config['rag']['num_contexts']
        self.similarity_threshold = config['rag']['similarity_threshold']
        self.embedding_model_name = config['rag']['embedding_model']

        # Initialize components
        self._load_embedding_model()
        self._connect_to_db()

    def _load_embedding_model(self):
        """Load the sentence transformer model for creating query embeddings."""
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        print("✓ Embedding model loaded")

    def _connect_to_db(self):
        """Connect to ChromaDB."""
        if not Path(self.vectordb_dir).exists():
            raise FileNotFoundError(
                f"Vector database not found at {self.vectordb_dir}\n"
                "Please run scripts/build_vectordb.py first."
            )

        print(f"Connecting to vector database: {self.vectordb_dir}")

        self.client = chromadb.PersistentClient(
            path=self.vectordb_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_collection(name="discord_messages")

        print(f"✓ Connected to vector database")
        print(f"  Documents in collection: {self.collection.count()}")

    def retrieve(self, query: str, n_results: Optional[int] = None) -> List[str]:
        """
        Retrieve relevant context for a query.

        Args:
            query: The user's message or query
            n_results: Number of results to retrieve (uses config default if not specified)

        Returns:
            List of relevant context strings
        """
        if n_results is None:
            n_results = self.num_contexts

        # Query the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        # Extract and format results
        contexts = []

        if results['documents'] and len(results['documents']) > 0:
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                # Format based on source type
                if metadata.get('type') == 'discord_message':
                    # Discord message format
                    context = f"[{metadata.get('channel_name', 'Unknown')}] {doc}"
                elif metadata.get('type') == 'wpilib_documentation':
                    # WPILib doc format
                    context = f"[WPILib Docs: {metadata.get('title', 'Unknown')}] {doc[:200]}..."
                else:
                    context = doc

                contexts.append(context)

        return contexts

    def retrieve_with_filters(
        self,
        query: str,
        doc_type: Optional[str] = None,
        channel: Optional[str] = None,
        n_results: Optional[int] = None
    ) -> List[str]:
        """
        Retrieve context with optional filters.

        Args:
            query: The user's message
            doc_type: Filter by document type ('discord_message' or 'wpilib_documentation')
            channel: Filter by Discord channel name
            n_results: Number of results

        Returns:
            List of relevant context strings
        """
        if n_results is None:
            n_results = self.num_contexts

        # Build where clause for filtering
        where_clause = {}
        if doc_type:
            where_clause['type'] = doc_type
        if channel:
            where_clause['channel_name'] = channel

        # Query with filters
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause if where_clause else None,
        )

        # Format results
        contexts = []
        if results['documents'] and len(results['documents']) > 0:
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                if metadata.get('type') == 'discord_message':
                    context = f"[{metadata.get('channel_name')}] {doc}"
                elif metadata.get('type') == 'wpilib_documentation':
                    context = f"[WPILib] {metadata.get('title')}: {doc[:150]}..."
                else:
                    context = doc

                contexts.append(context)

        return contexts

    def get_mixed_context(self, query: str) -> List[str]:
        """
        Get a mix of Discord messages and documentation for context.

        This retrieves both conversation history and relevant docs.
        """
        contexts = []

        # Get Discord messages
        discord_contexts = self.retrieve_with_filters(
            query,
            doc_type='discord_message',
            n_results=3
        )
        contexts.extend(discord_contexts)

        # Get WPILib docs if query seems robotics-related
        robotics_keywords = [
            'robot', 'motor', 'sensor', 'autonomous', 'teleop',
            'pid', 'control', 'drive', 'wpiliib', 'frc',
            'command', 'subsystem', 'encoder', 'gyro'
        ]

        if any(keyword in query.lower() for keyword in robotics_keywords):
            doc_contexts = self.retrieve_with_filters(
                query,
                doc_type='wpilib_documentation',
                n_results=2
            )
            contexts.extend(doc_contexts)

        return contexts


def test_rag():
    """Test the RAG retrieval system."""
    print("Testing RAG retrieval system...\n")

    config = load_config()
    retriever = RAGRetriever(config)

    test_queries = [
        "What are we working on for robotics?",
        "How do I control a motor in FRC?",
        "What's the latest with the team?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)

        # Get mixed context
        contexts = retriever.get_mixed_context(query)

        print(f"Retrieved {len(contexts)} context(s):")
        for i, ctx in enumerate(contexts, 1):
            print(f"\n{i}. {ctx[:200]}..." if len(ctx) > 200 else f"\n{i}. {ctx}")

        print("\n" + "=" * 60)


if __name__ == '__main__':
    test_rag()
