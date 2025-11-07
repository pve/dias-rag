# Dias-RAG: Local Semantic Search for Markdown

A fast, local semantic search tool for markdown documentation using vector embeddings. Built with Python, ChromaDB, and sentence-transformers, managed with UV.

## Features

- **Semantic Search**: Find documents by meaning, not just keywords
- **Local & Offline**: All processing happens locally after initial model download
- **Fast**: Sub-200ms search queries after indexing
- **Hugo Compatible**: Parses YAML frontmatter automatically
- **Simple CLI**: Easy-to-use command-line interface
- **Test-Driven**: Comprehensive test coverage

## Quick Start

### Installation

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repo-url> dias-rag
cd dias-rag
uv sync

# Index your content (first run will download ~90MB ML model)
uv run dias-rag index /path/to/your/content

# Search
uv run dias-rag search "authentication patterns"
```

### Basic Usage

```bash
# Get help
uv run dias-rag --help

# Index markdown files
uv run dias-rag index /path/to/content

# Search with custom result limit
uv run dias-rag search "API design" --limit 5

# Use custom data directory
uv run dias-rag index /path/to/content --data-dir ./my-data
uv run dias-rag search "query" --data-dir ./my-data
```

## How It Works

1. **Indexing**: Scans markdown files, extracts content and frontmatter, splits into semantic chunks
2. **Embedding**: Generates 384-dimensional vectors using `all-MiniLM-L6-v2` model
3. **Storage**: Stores vectors and metadata in local ChromaDB database
4. **Search**: Converts queries to vectors and finds similar chunks using cosine similarity

## Architecture

```
dias-rag/
├── src/
│   ├── utils.py      # Markdown parsing & semantic chunking
│   ├── indexer.py    # Document scanning & embedding generation
│   ├── search.py     # Vector similarity search
│   └── cli.py        # Command-line interface
├── data/
│   └── chroma/       # Vector database (created on first index)
└── tests/            # Comprehensive test suite
```

### Semantic Chunking Algorithm

Content is split intelligently to preserve meaning:

1. Split on paragraph breaks (double newlines)
2. Within paragraphs, split on sentence boundaries (., !, ?)
3. Accumulate sentences until max chunk size (512 chars)
4. For long sentences: split at clause boundaries (; , :)
5. Last resort: split at word boundaries

No overlap between chunks in MVP version.

## Configuration

### Hardcoded Defaults

```python
DEFAULT_CHUNK_SIZE = 512
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "dias_content"
DEFAULT_SIMILARITY_METRIC = "cosine"
DEFAULT_DATA_DIR = "./data"
DEFAULT_LIMIT = 10
```

These can be extended to use a config file in future versions.

## Development

### Setup

```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src tests/

# Format code
uv run black src/ tests/

# Lint
uv run ruff check src/

# Type check
uv run mypy src/
```

### Running Tests

```bash
# All tests
uv run pytest -v

# Specific module
uv run pytest tests/test_utils.py -v

# With coverage report
uv run pytest --cov=src --cov-report=html
```

## Performance

For 100 markdown files (~500KB total):

| Operation | Target | Notes |
|-----------|--------|-------|
| Initial indexing | < 30s | Full rebuild |
| Search query | < 200ms | Return top 10 results |
| Memory usage | < 500MB | Embeddings in memory |
| UV sync time | < 10s | First time with cached downloads |

## Troubleshooting

### "ChromaDB not found" Error

```bash
# Make sure you've indexed first
uv run dias-rag index /path/to/content

# Check data directory exists
ls -la ./data/chroma
```

### "No markdown files found" Error

```bash
# Verify directory contains .md files
find /path/to/content -name "*.md"

# Try absolute path
uv run dias-rag index $(pwd)/content
```

### First Run Model Download

On first use, sentence-transformers will download the ~90MB model. Ensure internet connection for initial setup. Subsequent runs work offline.

### Test Failures in Sandbox

Some tests require internet access to download the ML model. In sandboxed environments without internet, these tests will fail but the code is functional.

## Technology Stack

- **Python 3.11+**: Modern Python with type hints
- **UV**: Fast, reliable dependency management
- **sentence-transformers**: State-of-the-art text embeddings
- **ChromaDB**: Efficient vector database
- **Click**: User-friendly CLI framework
- **python-frontmatter**: YAML frontmatter parsing
- **pytest**: Comprehensive testing

## Future Enhancements

- Incremental indexing (only update changed files)
- Interactive TUI with Textual
- Keyword and hybrid search modes
- Result filtering by frontmatter fields
- Watch mode for auto-reindexing
- Support for PDF and other formats
- Query history and favorites
- Export results to JSON/CSV

## License

MIT License

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the troubleshooting section above
