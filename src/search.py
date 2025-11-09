"""Search module for semantic search and result formatting."""

import os
import time
from dataclasses import dataclass
from typing import List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


@dataclass
class SearchResult:
    """Represents a search result with metadata."""

    document_id: str  # MD5 hash of file path
    file_path: str  # Relative path to file
    title: str  # Document title
    matched_chunk: str  # The chunk that matched
    score: float  # Similarity score (0-1)
    chunk_index: int  # Which chunk matched (0-indexed)
    draft: bool = False  # Draft status from frontmatter


def semantic_search(
    query: str,
    limit: int = 10,
    data_dir: str = "./data",
    collection_name: str = "dias_content",
) -> List[SearchResult]:
    """
    Vector similarity search using ChromaDB.

    Args:
        query: Search query
        limit: Maximum number of results to return
        data_dir: Directory containing ChromaDB data
        collection_name: Name of the ChromaDB collection

    Returns:
        List of SearchResult objects sorted by relevance

    Raises:
        ValueError: If query is empty or ChromaDB not initialized
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    # Check if ChromaDB exists
    chroma_path = os.path.join(data_dir, "chroma")
    if not os.path.exists(chroma_path):
        raise ValueError(
            f"ChromaDB not found at {chroma_path}. Please run 'dias-rag index' first."
        )

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(
        path=chroma_path,
        settings=Settings(anonymized_telemetry=False),
    )

    # Get collection
    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        raise ValueError(
            f"Collection '{collection_name}' not found. Please run 'dias-rag index' first."
        )

    # Generate query embedding
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query], convert_to_numpy=True)[0].tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=limit,
        include=["documents", "metadatas", "distances"],
    )

    # Convert to SearchResult objects
    search_results = []

    if results["ids"] and len(results["ids"][0]) > 0:
        for i in range(len(results["ids"][0])):
            # ChromaDB returns distances, convert to similarity scores
            # For cosine distance: similarity = 1 - distance
            distance = results["distances"][0][i]
            score = max(0.0, 1.0 - distance)  # Ensure non-negative

            metadata = results["metadatas"][0][i]
            matched_chunk = results["documents"][0][i]

            # Extract draft status from frontmatter metadata
            draft = False
            if "fm_draft" in metadata:
                draft_value = metadata["fm_draft"]
                # Handle various representations of boolean values
                if isinstance(draft_value, bool):
                    draft = draft_value
                elif isinstance(draft_value, str):
                    draft = draft_value.lower() in ("true", "1", "yes")

            result = SearchResult(
                document_id=metadata["document_id"],
                file_path=metadata["file_path"],
                title=metadata["title"],
                matched_chunk=matched_chunk,
                score=score,
                chunk_index=int(metadata["chunk_index"]),
                draft=draft,
            )

            search_results.append(result)

    return search_results


def format_results(results: List[SearchResult], query: str = "", search_time: float = 0.0) -> str:
    """
    Format search results for CLI output.

    Args:
        results: List of SearchResult objects
        query: Original search query (for display)
        search_time: Time taken for search in seconds

    Returns:
        Formatted string ready for printing
    """
    if not results:
        return f"No results found for '{query}'\n"

    output = []
    output.append(f"\nFound {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        # Truncate long chunks for display
        snippet = result.matched_chunk
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."

        # Format result
        output.append(f"[{i}] {result.file_path} (score: {result.score:.2f})")

        # Display title and draft status
        title_line = f"    Title: {result.title}"
        if result.draft:
            title_line += " [DRAFT]"
        output.append(title_line)

        output.append(f"    {snippet}")
        output.append("")  # Blank line between results

    if search_time > 0:
        output.append(f"Search completed in {search_time:.2f}s")

    return "\n".join(output)
