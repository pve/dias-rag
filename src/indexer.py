"""Indexer module for parsing markdown files and generating embeddings."""

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from src.utils import chunk_content, parse_frontmatter


@dataclass
class Document:
    """Represents a markdown document with metadata and chunks."""

    id: str  # Unique ID (MD5 hash of file path)
    file_path: str  # Relative path to file
    title: str  # From frontmatter or first H1, fallback to filename
    content: str  # Full markdown content (without frontmatter)
    chunks: List[str]  # Content split into semantic chunks
    frontmatter: Dict  # YAML frontmatter as dict (empty if missing)
    word_count: int  # Content statistics


def scan_markdown_files(directory: str) -> List[Document]:
    """
    Recursively scan directory for .md files and parse them into Documents.

    Args:
        directory: Root directory to scan

    Returns:
        List of Document objects

    Raises:
        ValueError: If directory doesn't exist or no markdown files found
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory}")

    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")

    # Find all .md files recursively
    md_files = list(dir_path.rglob("*.md"))

    if not md_files:
        raise ValueError(f"No markdown files found in: {directory}")

    documents = []

    for md_file in md_files:
        try:
            # Read file content
            with open(md_file, "r", encoding="utf-8") as f:
                raw_content = f.read()

            # Parse frontmatter
            frontmatter_dict, markdown_content = parse_frontmatter(raw_content)

            # Generate document ID (MD5 hash of relative path)
            relative_path = str(md_file.relative_to(dir_path))
            doc_id = hashlib.md5(relative_path.encode()).hexdigest()

            # Extract title
            title = _extract_title(frontmatter_dict, markdown_content, md_file.name)

            # Chunk content
            chunks = chunk_content(markdown_content, max_chunk_size=512)

            # Calculate word count
            word_count = len(markdown_content.split())

            # Create Document
            doc = Document(
                id=doc_id,
                file_path=relative_path,
                title=title,
                content=markdown_content,
                chunks=chunks,
                frontmatter=frontmatter_dict,
                word_count=word_count,
            )

            documents.append(doc)

        except Exception as e:
            # Log warning and continue with other files
            print(f"Warning: Failed to process {md_file}: {e}")
            continue

    if not documents:
        raise ValueError(f"Failed to process any markdown files in: {directory}")

    return documents


def _extract_title(frontmatter: Dict, content: str, filename: str) -> str:
    """
    Extract title from frontmatter, first H1, or filename.

    Args:
        frontmatter: Parsed frontmatter dictionary
        content: Markdown content
        filename: Filename as fallback

    Returns:
        Document title
    """
    # Try frontmatter first
    if "title" in frontmatter:
        return str(frontmatter["title"])

    # Try to find first H1
    h1_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()

    # Fallback to filename (without extension)
    return Path(filename).stem


def generate_embeddings(chunks: List[str], model_name: str = "all-MiniLM-L6-v2") -> List:
    """
    Generate vector embeddings using sentence-transformers.

    Args:
        chunks: List of text chunks to embed
        model_name: Model name for sentence-transformers (default: all-MiniLM-L6-v2)

    Returns:
        List of embeddings (numpy arrays converted to lists for ChromaDB)
    """
    if not chunks:
        return []

    # Load model (cached after first run)
    model = SentenceTransformer(model_name)

    # Generate embeddings
    embeddings = model.encode(chunks, convert_to_numpy=True)

    # Convert to list for ChromaDB compatibility
    return [embedding.tolist() for embedding in embeddings]


def index_documents(
    docs: List[Document],
    data_dir: str = "./data",
    collection_name: str = "dias_content",
) -> None:
    """
    Store documents and embeddings in ChromaDB.

    Args:
        docs: List of Document objects to index
        data_dir: Directory for ChromaDB storage
        collection_name: Name of the ChromaDB collection
    """
    if not docs:
        raise ValueError("No documents to index")

    # Initialize ChromaDB
    chroma_path = os.path.join(data_dir, "chroma")
    os.makedirs(chroma_path, exist_ok=True)

    client = chromadb.PersistentClient(
        path=chroma_path,
        settings=Settings(anonymized_telemetry=False),
    )

    # Delete existing collection if it exists (full rebuild in MVP)
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass  # Collection doesn't exist yet

    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity
    )

    # Prepare all chunks for batch processing
    all_ids = []
    all_documents = []
    all_embeddings = []
    all_metadatas = []

    print("Generating embeddings...")
    for doc in docs:
        if not doc.chunks:
            continue

        # Generate embeddings for all chunks in this document
        embeddings = generate_embeddings(doc.chunks)

        for i, (chunk, embedding) in enumerate(zip(doc.chunks, embeddings)):
            # Create unique ID for each chunk
            chunk_id = f"{doc.id}_{i}"

            # Metadata for this chunk
            metadata = {
                "document_id": doc.id,
                "file_path": doc.file_path,
                "title": doc.title,
                "chunk_index": i,
                "word_count": doc.word_count,
                # Add frontmatter fields
                **{
                    f"fm_{k}": str(v)
                    for k, v in doc.frontmatter.items()
                    if isinstance(v, (str, int, float, bool))
                },
            }

            all_ids.append(chunk_id)
            all_documents.append(chunk)
            all_embeddings.append(embedding)
            all_metadatas.append(metadata)

    # Batch insert into ChromaDB
    print(f"Indexing {len(all_ids)} chunks from {len(docs)} documents...")
    collection.add(
        ids=all_ids,
        documents=all_documents,
        embeddings=all_embeddings,
        metadatas=all_metadatas,
    )

    print(f"✓ Successfully indexed {len(docs)} documents ({len(all_ids)} chunks)")


def rebuild_index(
    directory: str,
    data_dir: str = "./data",
    collection_name: str = "dias_content",
) -> None:
    """
    Full reindex of all documents (MVP: always full rebuild).

    Args:
        directory: Content directory to index
        data_dir: Directory for ChromaDB storage
        collection_name: Name of the ChromaDB collection
    """
    print(f"Scanning markdown files in: {directory}")
    documents = scan_markdown_files(directory)

    print(f"Found {len(documents)} markdown files")

    # Index all documents
    index_documents(documents, data_dir=data_dir, collection_name=collection_name)

    print("✓ Indexing complete!")
