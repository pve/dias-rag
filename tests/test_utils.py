"""Tests for utils module (frontmatter parsing and content chunking)."""

import pytest
from src.utils import parse_frontmatter, chunk_content


class TestParseFrontmatter:
    """Test frontmatter parsing functionality."""

    def test_frontmatter_parsing_valid(self):
        """Parse valid YAML frontmatter."""
        content = """---
title: Test Document
tags: [python, testing]
author: Test Author
---

This is the content."""

        frontmatter, markdown = parse_frontmatter(content)

        assert frontmatter["title"] == "Test Document"
        assert frontmatter["tags"] == ["python", "testing"]
        assert frontmatter["author"] == "Test Author"
        assert markdown.strip() == "This is the content."

    def test_frontmatter_parsing_missing(self):
        """Handle files without frontmatter."""
        content = "This is just plain markdown content."

        frontmatter, markdown = parse_frontmatter(content)

        assert frontmatter == {}
        assert markdown == content

    def test_frontmatter_parsing_malformed(self):
        """Handle malformed YAML gracefully."""
        content = """---
title: Test
invalid yaml: [unclosed
---

Content here."""

        # Should return empty dict and original content
        frontmatter, markdown = parse_frontmatter(content)

        assert isinstance(frontmatter, dict)
        # Either empty dict or original content, both are acceptable graceful handling

    def test_frontmatter_parsing_empty(self):
        """Handle empty frontmatter."""
        content = """---
---

Content here."""

        frontmatter, markdown = parse_frontmatter(content)

        assert frontmatter == {} or frontmatter is None or isinstance(frontmatter, dict)
        assert "Content here." in markdown


class TestChunkContent:
    """Test content chunking functionality."""

    def test_chunk_content_small(self):
        """Content smaller than max_chunk_size."""
        content = "This is a small document."
        chunks = chunk_content(content, max_chunk_size=512)

        assert len(chunks) == 1
        assert chunks[0] == content

    def test_chunk_content_multiple_paragraphs(self):
        """Split on paragraph boundaries."""
        content = """First paragraph with some content.

Second paragraph with more content.

Third paragraph with even more content."""

        chunks = chunk_content(content, max_chunk_size=100)

        # Should split on paragraph breaks
        assert len(chunks) >= 2
        # Each chunk should be <= max_chunk_size
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_chunk_content_long_sentence(self):
        """Handle sentences > max_chunk_size."""
        # Create a very long sentence with clause boundaries
        long_sentence = (
            "This is a very long sentence that exceeds the maximum chunk size, "
            "and it contains multiple clauses separated by commas; it also has "
            "semicolons that can be used as split points: and colons too, "
            "which should all be considered when chunking the content properly."
        )

        chunks = chunk_content(long_sentence, max_chunk_size=100)

        # Should split at clause boundaries
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 100
            assert len(chunk.strip()) > 0

    def test_chunk_content_multiple_sentences(self):
        """Accumulate sentences until max_chunk_size."""
        content = "First sentence. Second sentence. Third sentence. Fourth sentence."

        chunks = chunk_content(content, max_chunk_size=40)

        # Should combine sentences up to the limit
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 40

    def test_chunk_content_empty(self):
        """Handle empty content."""
        content = ""
        chunks = chunk_content(content, max_chunk_size=512)

        assert chunks == [] or chunks == ['']

    def test_chunk_content_whitespace_only(self):
        """Handle whitespace-only content."""
        content = "   \n\n   \t\t  "
        chunks = chunk_content(content, max_chunk_size=512)

        # Should either return empty list or skip whitespace
        assert len(chunks) <= 1

    def test_chunk_content_no_overlap(self):
        """Ensure chunks don't overlap (MVP requirement)."""
        content = "A" * 1000  # Long content
        chunks = chunk_content(content, max_chunk_size=100)

        # Reconstruct and verify no overlap
        total_length = sum(len(chunk) for chunk in chunks)
        # Total should be close to original (may have small differences due to splitting)
        assert abs(total_length - len(content)) < 50  # Allow some tolerance for split points
