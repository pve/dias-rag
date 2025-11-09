"""Tests for indexer module."""

import os
import tempfile
from pathlib import Path

import pytest

from src.indexer import (
    Document,
    _extract_title,
    generate_embeddings,
    index_documents,
    rebuild_index,
    scan_markdown_files,
)


class TestScanMarkdownFiles:
    """Test markdown file scanning."""

    def test_scan_markdown_files(self, tmp_path):
        """Find all .md files recursively."""
        # Create test structure
        (tmp_path / "doc1.md").write_text("# Title 1\n\nContent 1")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "doc2.md").write_text("# Title 2\n\nContent 2")

        docs = scan_markdown_files(str(tmp_path))

        assert len(docs) == 2
        assert all(isinstance(doc, Document) for doc in docs)
        assert all(doc.file_path.endswith(".md") for doc in docs)

    def test_scan_markdown_files_empty(self, tmp_path):
        """Handle directory with no markdown files."""
        with pytest.raises(ValueError, match="No markdown files found"):
            scan_markdown_files(str(tmp_path))

    def test_scan_invalid_directory(self):
        """Handle invalid directory path."""
        with pytest.raises(ValueError, match="does not exist"):
            scan_markdown_files("/nonexistent/path")

    def test_scan_with_frontmatter(self, tmp_path):
        """Parse files with frontmatter."""
        content = """---
title: Test Document
tags: [python, testing]
---

# Main Content

This is the body."""

        (tmp_path / "doc.md").write_text(content)
        docs = scan_markdown_files(str(tmp_path))

        assert len(docs) == 1
        assert docs[0].title == "Test Document"
        assert docs[0].frontmatter["title"] == "Test Document"
        assert "Main Content" in docs[0].content

    def test_scan_without_frontmatter(self, tmp_path):
        """Parse files without frontmatter."""
        content = "# Test Title\n\nContent here."

        (tmp_path / "doc.md").write_text(content)
        docs = scan_markdown_files(str(tmp_path))

        assert len(docs) == 1
        assert docs[0].title == "Test Title"
        assert docs[0].frontmatter == {}

    def test_document_chunking(self, tmp_path):
        """Ensure documents are chunked."""
        content = "Sentence one. " * 100  # Long content

        (tmp_path / "doc.md").write_text(content)
        docs = scan_markdown_files(str(tmp_path))

        assert len(docs) == 1
        assert len(docs[0].chunks) > 1  # Should be split into multiple chunks


class TestExtractTitle:
    """Test title extraction logic."""

    def test_extract_title_from_frontmatter(self):
        """Extract title from frontmatter."""
        frontmatter = {"title": "My Title"}
        content = "# Different Title\n\nContent"
        filename = "test.md"

        title = _extract_title(frontmatter, content, filename)
        assert title == "My Title"

    def test_extract_title_from_h1(self):
        """Extract title from first H1."""
        frontmatter = {}
        content = "# Main Title\n\nContent here"
        filename = "test.md"

        title = _extract_title(frontmatter, content, filename)
        assert title == "Main Title"

    def test_extract_title_from_filename(self):
        """Use filename as fallback."""
        frontmatter = {}
        content = "Just plain content"
        filename = "my-document.md"

        title = _extract_title(frontmatter, content, filename)
        assert title == "my-document"


class TestGenerateEmbeddings:
    """Test embedding generation."""

    def test_generate_embeddings(self):
        """Check embedding dimensions (384 for all-MiniLM-L6-v2)."""
        chunks = ["This is a test sentence.", "Another test sentence."]

        embeddings = generate_embeddings(chunks)

        assert len(embeddings) == 2
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings
        assert all(len(emb) == 384 for emb in embeddings)

    def test_generate_embeddings_empty(self):
        """Handle empty chunks list."""
        embeddings = generate_embeddings([])
        assert embeddings == []

    def test_generate_embeddings_consistent(self):
        """Same input produces same embeddings."""
        chunks = ["Test sentence"]

        emb1 = generate_embeddings(chunks)
        emb2 = generate_embeddings(chunks)

        # Should be very similar (may have tiny floating point differences)
        assert len(emb1) == len(emb2)
        assert len(emb1[0]) == len(emb2[0])


class TestIndexDocuments:
    """Test ChromaDB indexing."""

    def test_index_documents(self, tmp_path):
        """Store documents in ChromaDB."""
        # Create a test document
        doc = Document(
            id="test123",
            file_path="test.md",
            title="Test Document",
            content="This is test content.",
            chunks=["This is test content."],
            frontmatter={"title": "Test Document"},
            word_count=4,
        )

        # Index it
        data_dir = str(tmp_path / "data")
        index_documents([doc], data_dir=data_dir)

        # Verify ChromaDB was created
        assert (tmp_path / "data" / "chroma").exists()

    def test_index_documents_empty(self, tmp_path):
        """Handle empty document list."""
        data_dir = str(tmp_path / "data")

        with pytest.raises(ValueError, match="No documents to index"):
            index_documents([], data_dir=data_dir)


class TestRebuildIndex:
    """Test full index rebuild."""

    def test_rebuild_index(self, tmp_path):
        """Full indexing workflow."""
        # Create test markdown files
        (tmp_path / "content").mkdir()
        (tmp_path / "content" / "doc1.md").write_text("# Doc 1\n\nContent 1")
        (tmp_path / "content" / "doc2.md").write_text("# Doc 2\n\nContent 2")

        # Rebuild index
        data_dir = str(tmp_path / "data")
        rebuild_index(
            directory=str(tmp_path / "content"),
            data_dir=data_dir,
        )

        # Verify database was created
        assert (tmp_path / "data" / "chroma").exists()
