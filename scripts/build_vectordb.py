"""
Build vector database for RAG (Retrieval Augmented Generation).

This script creates embeddings for all Discord messages and external documentation
(like WPILib) and stores them in ChromaDB for efficient retrieval.
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import yaml

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_config() -> Dict:
    """Load configuration from config.yaml."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def load_messages(processed_data_dir: str) -> List[Dict]:
    """Load all processed Discord messages."""
    messages_file = Path(processed_data_dir) / 'all_messages.json'

    if not messages_file.exists():
        raise FileNotFoundError(
            f"Messages file not found: {messages_file}\n"
            "Please run parse_exports.py first."
        )

    with open(messages_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_documents(messages: List[Dict]) -> tuple[List[str], List[Dict], List[str]]:
    """
    Prepare documents for embedding.

    Returns:
        documents: List of text content to embed
        metadatas: List of metadata dicts for each document
        ids: List of unique IDs for each document
    """
    documents = []
    metadatas = []
    ids = []

    for i, msg in enumerate(messages):
        # Skip empty messages or bot messages
        content = msg.get('content', '').strip()
        if not content or msg.get('is_bot'):
            continue

        # Create document text with context
        # Include author name and channel for better retrieval
        doc_text = f"{msg.get('author_name', 'Unknown')}: {content}"

        # Prepare metadata
        metadata = {
            'message_id': msg.get('message_id', f'msg_{i}'),
            'author_id': msg.get('author_id', ''),
            'author_name': msg.get('author_name', 'Unknown'),
            'channel_name': msg.get('channel_name', 'Unknown'),
            'guild_name': msg.get('guild_name', 'Unknown'),
            'timestamp': msg.get('timestamp', ''),
            'type': 'discord_message',
        }

        documents.append(doc_text)
        metadatas.append(metadata)
        ids.append(f"discord_msg_{i}")

    return documents, metadatas, ids


def build_vector_database(
    documents: List[str],
    metadatas: List[Dict],
    ids: List[str],
    db_path: str,
    embedding_model_name: str,
    collection_name: str = "discord_messages"
):
    """Build ChromaDB vector database with embeddings."""

    print(f"Loading embedding model: {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_name)

    print("Initializing ChromaDB...")
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )

    # Create or get collection
    # Delete existing collection if it exists
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )

    print(f"Creating embeddings for {len(documents)} documents...")

    # Process in batches to avoid memory issues
    batch_size = 100
    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding batches"):
        batch_docs = documents[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]

        # Create embeddings
        embeddings = embedding_model.encode(
            batch_docs,
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()

        # Add to collection
        collection.add(
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=batch_meta,
            ids=batch_ids
        )

    print(f"✓ Vector database built successfully!")
    print(f"  Collection: {collection_name}")
    print(f"  Documents: {collection.count()}")
    print(f"  Location: {db_path}")

    return collection


def test_retrieval(collection, query: str, n_results: int = 5):
    """Test the retrieval system with a sample query."""
    print(f"\nTesting retrieval with query: '{query}'")

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    print(f"\nTop {n_results} results:")
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"\n{i+1}. [{metadata['author_name']}] in #{metadata['channel_name']}")
        print(f"   {doc[:150]}..." if len(doc) > 150 else f"   {doc}")


def main():
    # Load config
    config = load_config()
    processed_data_dir = config['paths']['processed_data_dir']
    vectordb_dir = config['paths']['vectordb_dir']
    embedding_model = config['rag']['embedding_model']

    # Create vectordb directory
    Path(vectordb_dir).mkdir(parents=True, exist_ok=True)

    # Load messages
    print("Loading Discord messages...")
    messages = load_messages(processed_data_dir)
    print(f"Loaded {len(messages)} messages")

    # Prepare documents
    print("\nPreparing documents for embedding...")
    documents, metadatas, ids = prepare_documents(messages)
    print(f"Prepared {len(documents)} documents")

    if len(documents) == 0:
        print("⚠️  No documents to embed! Please check your processed data.")
        return

    # Build vector database
    print("\nBuilding vector database...")
    collection = build_vector_database(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        db_path=vectordb_dir,
        embedding_model_name=embedding_model,
        collection_name="discord_messages"
    )

    # Test retrieval
    test_queries = [
        "robotics",
        "programming",
        "what are you working on",
    ]

    for query in test_queries:
        test_retrieval(collection, query, n_results=3)

    print(f"\n{'='*60}")
    print("Vector database setup complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
