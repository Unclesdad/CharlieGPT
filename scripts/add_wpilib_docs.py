"""
Fetch and add WPILib documentation to the vector database.

This script clones the WPILib documentation repository, processes the markdown
files, and adds them to the ChromaDB for RAG context retrieval.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import yaml

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_config() -> Dict:
    """Load configuration from config.yaml."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def clone_wpilib_docs(docs_dir: str) -> Path:
    """Clone WPILib documentation repository."""
    docs_path = Path(docs_dir)
    docs_path.mkdir(parents=True, exist_ok=True)

    wpilib_docs_path = docs_path / 'wpilib-docs'

    if wpilib_docs_path.exists():
        print("WPILib docs already cloned, pulling latest changes...")
        subprocess.run(
            ['git', 'pull'],
            cwd=str(wpilib_docs_path),
            check=True
        )
    else:
        print("Cloning WPILib documentation repository...")
        subprocess.run(
            ['git', 'clone', 'https://github.com/wpilibsuite/frc-docs.git', str(wpilib_docs_path)],
            check=True
        )

    return wpilib_docs_path


def parse_markdown_file(file_path: Path) -> List[Dict]:
    """
    Parse a markdown file into chunks suitable for RAG.

    Splits on headers to create logical sections.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Get relative path for metadata
    rel_path = str(file_path.relative_to(file_path.parent.parent.parent))

    # Split into sections based on headers
    sections = []
    current_section = {
        'title': file_path.stem,
        'content': '',
        'level': 0,
        'path': rel_path,
    }

    lines = content.split('\n')
    for line in lines:
        # Check if line is a header
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

        if header_match:
            # Save previous section if it has content
            if current_section['content'].strip():
                sections.append(current_section.copy())

            # Start new section
            level = len(header_match.group(1))
            title = header_match.group(2).strip()

            current_section = {
                'title': title,
                'content': '',
                'level': level,
                'path': rel_path,
            }
        else:
            current_section['content'] += line + '\n'

    # Add the last section
    if current_section['content'].strip():
        sections.append(current_section)

    return sections


def process_wpilib_docs(wpilib_docs_path: Path) -> Tuple[List[str], List[Dict], List[str]]:
    """
    Process WPILib documentation files.

    Returns:
        documents: List of text content to embed
        metadatas: List of metadata dicts
        ids: List of unique IDs
    """
    # Find all markdown files in the source directory
    source_dir = wpilib_docs_path / 'source'

    if not source_dir.exists():
        # Try alternate structure
        source_dir = wpilib_docs_path / 'docs'

    if not source_dir.exists():
        print(f"Warning: Could not find docs source directory in {wpilib_docs_path}")
        return [], [], []

    md_files = list(source_dir.rglob('*.md')) + list(source_dir.rglob('*.rst'))

    print(f"Found {len(md_files)} documentation files")

    documents = []
    metadatas = []
    ids = []

    doc_id = 0
    for md_file in tqdm(md_files, desc="Processing WPILib docs"):
        try:
            sections = parse_markdown_file(md_file)

            for section in sections:
                # Create document text
                doc_text = f"# {section['title']}\n\n{section['content'].strip()}"

                # Skip very short sections
                if len(doc_text) < 50:
                    continue

                # Create metadata
                metadata = {
                    'title': section['title'],
                    'file_path': section['path'],
                    'type': 'wpilib_documentation',
                    'source': 'WPILib FRC Documentation',
                }

                documents.append(doc_text)
                metadatas.append(metadata)
                ids.append(f"wpilib_doc_{doc_id}")
                doc_id += 1

        except Exception as e:
            print(f"Error processing {md_file}: {e}")

    return documents, metadatas, ids


def add_to_vectordb(
    documents: List[str],
    metadatas: List[Dict],
    ids: List[str],
    db_path: str,
    embedding_model_name: str,
    collection_name: str = "discord_messages"
):
    """Add WPILib docs to existing vector database."""

    print(f"Loading embedding model: {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_name)

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )

    # Get or create collection
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Connected to existing collection: {collection_name}")
    except:
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Created new collection: {collection_name}")

    print(f"Adding {len(documents)} WPILib documents to vector database...")

    # Process in batches
    batch_size = 50
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

    print(f"✓ WPILib documentation added successfully!")
    print(f"  Total documents in collection: {collection.count()}")

    return collection


def test_wpilib_retrieval(collection, query: str, n_results: int = 3):
    """Test retrieval of WPILib docs."""
    print(f"\nTesting WPILib docs retrieval: '{query}'")

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where={"type": "wpilib_documentation"}  # Filter for WPILib docs only
    )

    print(f"\nTop {n_results} WPILib results:")
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"\n{i+1}. {metadata['title']}")
        print(f"   File: {metadata['file_path']}")
        print(f"   {doc[:200]}..." if len(doc) > 200 else f"   {doc}")


def main():
    # Load config
    config = load_config()
    vectordb_dir = config['paths']['vectordb_dir']
    embedding_model = config['rag']['embedding_model']

    # Clone/update WPILib docs
    docs_dir = './external_docs'
    wpilib_docs_path = clone_wpilib_docs(docs_dir)

    # Process documentation
    print("\nProcessing WPILib documentation...")
    documents, metadatas, ids = process_wpilib_docs(wpilib_docs_path)

    if len(documents) == 0:
        print("⚠️  No WPILib documents found to add!")
        return

    print(f"Processed {len(documents)} WPILib documentation sections")

    # Add to vector database
    collection = add_to_vectordb(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        db_path=vectordb_dir,
        embedding_model_name=embedding_model,
        collection_name="discord_messages"
    )

    # Test retrieval
    test_queries = [
        "How do I program a motor controller?",
        "What is command-based programming?",
        "How to use PID control?",
    ]

    for query in test_queries:
        test_wpilib_retrieval(collection, query, n_results=3)

    print(f"\n{'='*60}")
    print("WPILib documentation added to vector database!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
