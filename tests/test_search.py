"""Tests for search module."""

import pytest

from src.indexer import Document, index_documents
from src.search import SearchResult, format_results, semantic_search


class TestSemanticSearch:
    """Test semantic search functionality."""

    @pytest.fixture
    def indexed_data(self, tmp_path):
        """Create indexed test data."""
        # Create test documents
        docs = [
            Document(
                id="doc1",
                file_path="auth.md",
                title="Authentication Guide",
                content="Authentication patterns for microservices using JWT tokens.",
                chunks=["Authentication patterns for microservices using JWT tokens."],
                frontmatter={"title": "Authentication Guide"},
                word_count=7,
            ),
            Document(
                id="doc2",
                file_path="api.md",
                title="API Design",
                content="API design principles and best practices for REST endpoints.",
                chunks=["API design principles and best practices for REST endpoints."],
                frontmatter={"title": "API Design"},
                word_count=9,
            ),
        ]

        # Index documents
        data_dir = str(tmp_path / "data")
        index_documents(docs, data_dir=data_dir)

        return data_dir

    def test_semantic_search_basic(self, indexed_data):
        """Basic search returns results."""
        results = semantic_search("authentication", data_dir=indexed_data)

        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_semantic_search_relevance(self, indexed_data):
        """Query "auth" should match "authentication" content."""
        results = semantic_search("auth", data_dir=indexed_data)

        assert len(results) > 0
        # First result should be about authentication
        assert "auth" in results[0].file_path.lower() or "auth" in results[0].matched_chunk.lower()

    def test_semantic_search_empty_query(self, indexed_data):
        """Handle empty query gracefully."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            semantic_search("", data_dir=indexed_data)

        with pytest.raises(ValueError, match="Query cannot be empty"):
            semantic_search("   ", data_dir=indexed_data)

    def test_semantic_search_no_index(self, tmp_path):
        """Handle missing ChromaDB."""
        data_dir = str(tmp_path / "nonexistent")

        with pytest.raises(ValueError, match="ChromaDB not found"):
            semantic_search("test", data_dir=data_dir)

    def test_semantic_search_limit(self, indexed_data):
        """Respect result limit."""
        results = semantic_search("design", limit=1, data_dir=indexed_data)

        assert len(results) <= 1

    def test_semantic_search_scores(self, indexed_data):
        """Scores should be between 0 and 1."""
        results = semantic_search("authentication", data_dir=indexed_data)

        for result in results:
            assert 0.0 <= result.score <= 1.0

    def test_semantic_search_sorted(self, indexed_data):
        """Results should be sorted by relevance."""
        results = semantic_search("API design", data_dir=indexed_data)

        if len(results) > 1:
            # Scores should be in descending order
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_semantic_search_draft_field(self, tmp_path):
        """Extract draft field from frontmatter."""
        # Create documents with draft field
        docs = [
            Document(
                id="doc1",
                file_path="draft.md",
                title="Draft Document",
                content="This is a draft document.",
                chunks=["This is a draft document."],
                frontmatter={"title": "Draft Document", "draft": True},
                word_count=5,
            ),
            Document(
                id="doc2",
                file_path="published.md",
                title="Published Document",
                content="This is a published document.",
                chunks=["This is a published document."],
                frontmatter={"title": "Published Document", "draft": False},
                word_count=5,
            ),
            Document(
                id="doc3",
                file_path="no-draft.md",
                title="No Draft Field",
                content="This document has no draft field.",
                chunks=["This document has no draft field."],
                frontmatter={"title": "No Draft Field"},
                word_count=6,
            ),
        ]

        # Index documents
        data_dir = str(tmp_path / "data")
        index_documents(docs, data_dir=data_dir)

        # Search
        results = semantic_search("document", data_dir=data_dir)

        # Find each document in results
        draft_result = next((r for r in results if "draft.md" in r.file_path), None)
        published_result = next((r for r in results if "published.md" in r.file_path), None)
        no_draft_result = next((r for r in results if "no-draft.md" in r.file_path), None)

        # Verify draft field is correctly extracted
        assert draft_result is not None
        assert draft_result.draft is True

        assert published_result is not None
        assert published_result.draft is False

        assert no_draft_result is not None
        assert no_draft_result.draft is False  # Should default to False


class TestFormatResults:
    """Test result formatting."""

    def test_format_results_basic(self):
        """Format basic results."""
        results = [
            SearchResult(
                document_id="doc1",
                file_path="test.md",
                title="Test Document",
                matched_chunk="This is a test chunk.",
                score=0.89,
                chunk_index=0,
            )
        ]

        output = format_results(results, query="test")

        assert "Found 1 results" in output
        assert "test.md" in output
        assert "score: 0.89" in output
        assert "Test Document" in output

    def test_format_results_empty(self):
        """Handle no results."""
        output = format_results([], query="nonexistent")

        assert "No results found" in output
        assert "nonexistent" in output

    def test_format_results_long_snippet(self):
        """Truncate long snippets."""
        long_chunk = "A" * 300
        results = [
            SearchResult(
                document_id="doc1",
                file_path="test.md",
                title="Test",
                matched_chunk=long_chunk,
                score=0.85,
                chunk_index=0,
            )
        ]

        output = format_results(results)

        # Should be truncated with ellipsis
        assert "..." in output
        assert len(output) < len(long_chunk)

    def test_format_results_multiple(self):
        """Format multiple results."""
        results = [
            SearchResult(
                document_id=f"doc{i}",
                file_path=f"test{i}.md",
                title=f"Test {i}",
                matched_chunk=f"Chunk {i}",
                score=0.9 - i * 0.1,
                chunk_index=0,
            )
            for i in range(3)
        ]

        output = format_results(results, query="test")

        assert "Found 3 results" in output
        assert "[1]" in output
        assert "[2]" in output
        assert "[3]" in output

    def test_format_results_with_time(self):
        """Include search time."""
        results = [
            SearchResult(
                document_id="doc1",
                file_path="test.md",
                title="Test",
                matched_chunk="Content",
                score=0.85,
                chunk_index=0,
            )
        ]

        output = format_results(results, search_time=0.15)

        assert "0.15s" in output or "0.15" in output

    def test_format_results_with_draft(self):
        """Display draft status in results."""
        results = [
            SearchResult(
                document_id="doc1",
                file_path="draft.md",
                title="Draft Document",
                matched_chunk="This is a draft.",
                score=0.85,
                chunk_index=0,
                draft=True,
            )
        ]

        output = format_results(results, query="test")

        assert "[DRAFT]" in output
        assert "Draft Document" in output

    def test_format_results_without_draft(self):
        """Don't show draft status for published documents."""
        results = [
            SearchResult(
                document_id="doc1",
                file_path="published.md",
                title="Published Document",
                matched_chunk="This is published.",
                score=0.85,
                chunk_index=0,
                draft=False,
            )
        ]

        output = format_results(results, query="test")

        assert "[DRAFT]" not in output
        assert "Published Document" in output

    def test_format_results_mixed_draft_status(self):
        """Display both draft and published documents."""
        results = [
            SearchResult(
                document_id="doc1",
                file_path="draft.md",
                title="Draft Document",
                matched_chunk="This is a draft.",
                score=0.90,
                chunk_index=0,
                draft=True,
            ),
            SearchResult(
                document_id="doc2",
                file_path="published.md",
                title="Published Document",
                matched_chunk="This is published.",
                score=0.85,
                chunk_index=0,
                draft=False,
            ),
        ]

        output = format_results(results, query="test")

        # Check both documents are shown
        assert "Draft Document" in output
        assert "Published Document" in output
        # Check draft marker appears only once
        assert output.count("[DRAFT]") == 1
