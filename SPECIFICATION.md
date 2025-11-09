# **Specification: Dias-RAG - Local Semantic Search MVP**

## **Project Overview**

A simple CLI-based semantic search tool for ~100 markdown files (Hugo format with frontmatter). Provides fast, local vector search using Python and UV for dependency management.

**MVP Scope:** Command-line interface with semantic search only. No TUI, no incremental updates, minimal configuration.

---

## **1. System Architecture**

```
dias-rag/
├── src/
│   ├── __init__.py
│   ├── indexer.py           # Vector embedding & indexing
│   ├── search.py            # Search logic
│   ├── cli.py               # CLI interface
│   └── utils.py             # Helper functions (markdown parsing, chunking)
├── data/
│   └── chroma/              # Vector database storage (created on first run)
├── tests/
│   ├── test_indexer.py
│   ├── test_search.py
│   ├── test_utils.py
│   └── fixtures/            # Sample markdown files
├── pyproject.toml           # UV project configuration
├── uv.lock                  # UV lockfile
├── .python-version          # Python version for UV
└── README.md
```

---

## **2. Core Components**

### **2.1 Indexer Module** (`indexer.py`)

**Purpose:** Parse markdown files, generate embeddings, store in vector database

**Key Functions:**
```python
def scan_markdown_files(directory: str) -> List[Document]:
    """
    Recursively scan directory for .md files
    Returns: List of Document objects
    """

def parse_frontmatter(content: str) -> Tuple[Dict, str]:
    """
    Extract YAML frontmatter and content
    Returns: (frontmatter_dict, markdown_content)
    """

def chunk_content(content: str, max_chunk_size: int = 512) -> List[str]:
    """
    Split content into semantic chunks using sentence boundaries.

    Algorithm:
    1. Split on paragraph breaks (double newlines)
    2. Within paragraphs, split on sentence boundaries (., !, ?)
    3. Accumulate sentences until max_chunk_size would be exceeded
    4. If single sentence > max_chunk_size, split at clause boundaries (; , :)
    5. If still too large, split at word boundaries

    Returns: List of text chunks, each ≤ max_chunk_size characters
    Note: No overlap between chunks in MVP
    """

def generate_embeddings(chunks: List[str]) -> List[np.ndarray]:
    """
    Generate vector embeddings using sentence-transformers
    Model: all-MiniLM-L6-v2 (fast, good quality, 384 dimensions)
    """

def index_documents(docs: List[Document], collection_name: str = "dias_content"):
    """
    Store documents and embeddings in ChromaDB
    Includes: content, metadata, embeddings
    """

def rebuild_index(directory: str):
    """
    Full reindex of all documents (MVP: always full rebuild)
    Shows progress during indexing
    """
```

**Document Schema:**
```python
@dataclass
class Document:
    id: str                    # Unique ID (MD5 hash of file path)
    file_path: str             # Relative path to file
    title: str                 # From frontmatter or first H1, fallback to filename
    content: str               # Full markdown content (without frontmatter)
    chunks: List[str]          # Content split into semantic chunks
    frontmatter: Dict          # YAML frontmatter as dict (empty if missing)
    word_count: int            # Content statistics
```

---

### **2.2 Search Module** (`search.py`)

**Purpose:** Execute semantic searches, format output

**Key Functions:**
```python
def semantic_search(query: str, limit: int = 10,
                   collection_name: str = "dias_content") -> List[SearchResult]:
    """
    Vector similarity search using ChromaDB
    Returns: Top N results with similarity scores (cosine similarity)
    """

def format_results(results: List[SearchResult]) -> str:
    """
    Format search results for CLI output
    Returns: Formatted string with file paths, titles, scores, and snippets
    """
```

**SearchResult Schema:**
```python
@dataclass
class SearchResult:
    document_id: str           # MD5 hash of file path
    file_path: str             # Relative path to file
    title: str                 # Document title
    matched_chunk: str         # The chunk that matched
    score: float               # Similarity score (0-1)
    chunk_index: int           # Which chunk matched (0-indexed)
```

---

### **2.3 CLI Module** (`cli.py`)

**Purpose:** Command-line interface using Click

**Commands:**
```python
@click.group()
def cli():
    """Dias-RAG: Local semantic search for markdown files"""
    pass

@cli.command()
@click.argument('content_directory', type=click.Path(exists=True))
@click.option('--data-dir', default='./data', help='Directory for vector database')
def index(content_directory: str, data_dir: str):
    """
    Index all markdown files in CONTENT_DIRECTORY

    Example: uv run dias-rag index /path/to/content
    """
    # Call indexer.rebuild_index()
    # Show progress bar during indexing

@cli.command()
@click.argument('query')
@click.option('--limit', '-n', default=10, help='Number of results to return')
@click.option('--data-dir', default='./data', help='Directory for vector database')
def search(query: str, limit: int, data_dir: str):
    """
    Search indexed documents

    Example: uv run dias-rag search "authentication patterns"
    """
    # Call search.semantic_search()
    # Format and print results
```

**CLI Output Format:**
```
$ uv run dias-rag search "authentication patterns"

Found 8 results:

[1] chapter-03-security.md (score: 0.89)
    Title: Security Best Practices
    ...discusses JWT authentication patterns for microservices.
    The key consideration is...

[2] chapter-07-api-design.md (score: 0.84)
    Title: API Design Principles
    ...authentication middleware should handle token validation...

[3] appendix-b-patterns.md (score: 0.78)
    Title: Common Patterns
    ...common authentication antipatterns include...

Search completed in 0.15s
```

### **2.4 Configuration** (MVP: Minimal)

**Approach:** Command-line arguments only, no config file required

**Key Parameters (hardcoded defaults):**
```python
DEFAULT_CHUNK_SIZE = 512
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "dias_content"
DEFAULT_SIMILARITY_METRIC = "cosine"
DEFAULT_DATA_DIR = "./data"
DEFAULT_LIMIT = 10
```

**Future:** Can add optional YAML config file support later

---

## **3. Technology Stack & UV Configuration**

### **pyproject.toml:**

```toml
[project]
name = "dias-rag"
version = "0.1.0"
description = "Local semantic search for markdown documentation"
readme = "README.md"
requires-python = ">=3.11"
authors = [
    { name = "Your Name", email = "your.email@example.com" }
]
keywords = ["search", "vector", "markdown", "embeddings", "rag", "semantic-search"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    # Vector Search & Embeddings
    "sentence-transformers>=2.2.2",
    "chromadb>=0.4.15",
    "torch>=2.1.0",

    # CLI Framework
    "click>=8.1.7",

    # Markdown Processing
    "python-frontmatter>=1.0.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.3",
    "black>=23.11.0",
    "mypy>=1.7.0",
    "ruff>=0.1.6",
]

[project.scripts]
dias-rag = "src.cli:cli"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.black]
line-length = 100
target-version = ['py311']

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
```

### **.python-version:**
```
3.11
```

---

## **4. UV Setup & Commands**

### **Initial Project Setup:**

```bash
# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create new project
uv init book-search
cd book-search

# Set Python version
echo "3.11" > .python-version

# Install dependencies
uv sync

# Install with dev dependencies
uv sync --extra dev
```

### **UV Commands Reference:**

```bash
# Add a new dependency
uv add sentence-transformers

# Add a dev dependency
uv add --dev pytest

# Remove a dependency
uv remove package-name

# Update all dependencies
uv lock --upgrade

# Update specific dependency
uv lock --upgrade-package chromadb

# Run the CLI
uv run dias-rag --help

# Index documents
uv run dias-rag index /path/to/content

# Search
uv run dias-rag search "your query here"

# Run tests
uv run pytest

# Format code
uv run black src/

# Type checking
uv run mypy src/

# Lint
uv run ruff check src/
```

### **Development Workflow:**

ALWAYS use test driven development.

```bash
# Install project in editable mode with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src tests/

# Format and lint
uv run black src/ tests/
uv run ruff check src/ tests/

# Type check
uv run mypy src/
```

---

## **5. Usage Examples**

### **Initial Setup:**
```bash
# Clone repository
git clone <your-repo> dias-rag
cd dias-rag

# Install with UV
uv sync

# Index your content (one-time, ~30 seconds for 100 files)
uv run dias-rag index /path/to/your/hugo/content

# Search
uv run dias-rag search "authentication patterns"
```

### **Daily Workflow:**
```bash
# Search indexed content
uv run dias-rag search "your query here"

# Limit number of results
uv run dias-rag search "security best practices" --limit 5

# Rebuild index after adding/updating content
uv run dias-rag index /path/to/your/hugo/content

# Use custom data directory
uv run dias-rag index /path/to/content --data-dir ./my-data
uv run dias-rag search "query" --data-dir ./my-data
```

### **Development Workflow:**
```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/test_search.py -v

# Format code
uv run black src/ tests/

# Lint
uv run ruff check src/

# Type check
uv run mypy src/

# Update dependencies
uv lock --upgrade
```

---

## **6. Performance Targets**

For 100 markdown files (~500KB total):

| Operation | Target | Notes |
|-----------|--------|-------|
| Initial indexing | < 30s | Full rebuild |
| Search query | < 200ms | Return top 10 results |
| Memory usage | < 500MB | Embeddings in memory |
| UV sync time | < 10s | First time with cached downloads |

---

## **7. Implementation Plan (MVP)**

### **Step 1: Project Setup**
- [ ] Initialize UV project structure
- [ ] Create pyproject.toml with dependencies
- [ ] Set up .python-version file
- [ ] Create src/ directory structure
- [ ] Run `uv sync` to install dependencies
- **Deliverable:** Project structure in place, dependencies installed

### **Step 2: Utils Module (Markdown & Chunking)**
- [ ] Implement `parse_frontmatter()` for YAML extraction
- [ ] Implement `chunk_content()` with semantic splitting algorithm
- [ ] Write tests for frontmatter parsing
- [ ] Write tests for chunking (various scenarios)
- **Deliverable:** `src/utils.py` with tested parsing/chunking functions

### **Step 3: Indexer Module**
- [ ] Implement `scan_markdown_files()` to recursively find .md files
- [ ] Implement Document dataclass
- [ ] Implement `generate_embeddings()` using sentence-transformers
- [ ] Implement `index_documents()` with ChromaDB
- [ ] Implement `rebuild_index()` orchestration function
- [ ] Write tests for indexing workflow
- **Deliverable:** `src/indexer.py` - can index markdown files into ChromaDB

### **Step 4: Search Module**
- [ ] Implement SearchResult dataclass
- [ ] Implement `semantic_search()` using ChromaDB query
- [ ] Implement `format_results()` for CLI output
- [ ] Write tests for search functionality
- **Deliverable:** `src/search.py` - can search and return results

### **Step 5: CLI Module**
- [ ] Implement Click CLI group
- [ ] Implement `index` command with progress indication
- [ ] Implement `search` command with formatted output
- [ ] Add --help documentation
- [ ] Test CLI commands end-to-end
- **Deliverable:** `src/cli.py` - working CLI interface

### **Step 6: Integration & Testing**
- [ ] End-to-end test: index sample files and search
- [ ] Test error handling (missing directories, empty results, etc.)
- [ ] Performance testing (ensure < 30s indexing, < 200ms search)
- [ ] Fix any bugs found during integration
- **Deliverable:** Fully functional MVP

### **Step 7: Documentation**
- [ ] Write README.md with installation and usage
- [ ] Document chunking algorithm
- [ ] Add inline code documentation
- **Deliverable:** Complete MVP with documentation

---

## **8. Testing Strategy**

**Test-Driven Development:** Write tests before implementation

```python
# tests/test_utils.py
def test_frontmatter_parsing_valid():
    """Parse valid YAML frontmatter"""

def test_frontmatter_parsing_missing():
    """Handle files without frontmatter"""

def test_frontmatter_parsing_malformed():
    """Handle malformed YAML gracefully"""

def test_chunk_content_small():
    """Content smaller than max_chunk_size"""

def test_chunk_content_multiple_paragraphs():
    """Split on paragraph boundaries"""

def test_chunk_content_long_sentence():
    """Handle sentences > max_chunk_size"""

# tests/test_indexer.py
def test_scan_markdown_files():
    """Find all .md files recursively"""

def test_scan_markdown_files_empty():
    """Handle directory with no markdown files"""

def test_generate_embeddings():
    """Check embedding dimensions (384 for all-MiniLM-L6-v2)"""

def test_index_documents():
    """Store documents in ChromaDB"""

# tests/test_search.py
def test_semantic_search_basic():
    """Basic search returns results"""

def test_semantic_search_relevance():
    """Query "auth" should match "authentication" content"""

def test_semantic_search_empty_query():
    """Handle empty query gracefully"""

def test_semantic_search_no_results():
    """Handle no matching results"""

def test_format_results():
    """Format results for CLI output"""

# tests/test_cli.py
def test_cli_index_command():
    """Test index command execution"""

def test_cli_search_command():
    """Test search command execution"""
```

**Run tests with UV:**
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_utils.py -v

# Run with verbose output
uv run pytest -v
```

---

## **9. Documentation Requirements**

**README.md should include:**
1. Project overview and features
2. Quick start guide with UV commands
3. CLI usage examples
4. Troubleshooting common issues
5. Architecture overview (brief)
6. Development setup

**Example README snippet:**
```markdown
## Installation

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repo-url> dias-rag
cd dias-rag
uv sync

# Index your content
uv run dias-rag index /path/to/your/content

# Search
uv run dias-rag search "your query"
```

## Usage

```bash
# Get help
uv run dias-rag --help

# Index markdown files
uv run dias-rag index /path/to/content

# Search with custom limit
uv run dias-rag search "query" --limit 5

# Use custom data directory
uv run dias-rag index /path/to/content --data-dir ./my-data
uv run dias-rag search "query" --data-dir ./my-data
```
```

---

## **10. Edge Cases & Error Handling**

**Handle gracefully:**
- **Empty search results** → Display "No results found for '{query}'"
- **Malformed markdown** → Skip file, print warning, continue indexing
- **Missing frontmatter** → Use empty dict, extract title from first H1 or filename
- **Embedding model download** → First run downloads model (~90MB), show progress
- **Invalid content directory** → Exit with clear error message
- **No markdown files found** → Exit with helpful message
- **No write permissions for data dir** → Exit with error suggesting different --data-dir
- **Empty query** → Display error "Query cannot be empty"
- **ChromaDB not initialized** → Inform user to run `index` command first

---

## **11. Success Metrics (MVP)**

**The MVP is successful if:**
1. ✅ Indexes 100 files in under 30 seconds
2. ✅ Returns relevant results in <200ms
3. ✅ Can find conceptually related content (not just keywords)
4. ✅ Works offline after initial model download (no external API calls)
5. ✅ Minimal resource usage (<500MB RAM)
6. ✅ Setup takes < 2 minutes from clone to first search
7. ✅ All tests pass with >80% code coverage

---

## **12. UV-Specific Advantages**

**Why UV?**
1. **Speed:** 10-100x faster than pip for dependency resolution
2. **Reliability:** Deterministic builds with uv.lock
3. **Simplicity:** Single tool for project management, no virtualenv confusion
4. **Caching:** Aggressive caching of downloads and builds
5. **Modern:** Built for modern Python workflows (>=3.11)
6. **Cross-platform:** Works identically on macOS, Linux, Windows

**UV vs Traditional Tools:**
```bash
# Traditional
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python search.py

# With UV (simpler, faster)
uv sync
uv run search.py
```

---

## **Estimated Timeline (MVP)**

- **Step 1-2:** Project setup + Utils (2-3 hours)
- **Step 3:** Indexer module (3-4 hours)
- **Step 4:** Search module (2-3 hours)
- **Step 5:** CLI interface (2 hours)
- **Step 6:** Integration & testing (2-3 hours)
- **Step 7:** Documentation (1-2 hours)

**Total MVP: 12-17 hours of focused work**

---

## **Future Enhancements (Post-MVP)**

Ideas for future versions:
- Incremental indexing (only update changed files)
- Interactive TUI with Textual
- Keyword search and hybrid search modes
- Result filtering by frontmatter fields
- Watch mode for auto-reindexing
- Support for other document formats (PDF, etc.)
- Query history and favorites
- Export results to JSON/CSV

---

## **Distribution (MVP)**

### **GitHub Repository:**
```bash
# Users clone and install
git clone <repo-url> dias-rag
cd dias-rag
uv sync

# Use directly
uv run dias-rag index /path/to/content
uv run dias-rag search "query"
```

### **Future: PyPI Distribution:**
```bash
# Build and publish with UV
uv build
uv publish

# Users install globally
uv tool install dias-rag

# Then run from anywhere
dias-rag search "query"
```
