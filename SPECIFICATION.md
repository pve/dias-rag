# **Specification: Local Vector Search with Embeddings (UV Edition)**

## **Project Overview**

A terminal-based semantic search tool for ~100 markdown files (Hugo format with frontmatter). Provides fast, local vector search with an elegant TUI interface using Python and UV for dependency management.

---

## **1. System Architecture**

```
book-search/
├── src/
│   ├── __init__.py
│   ├── indexer.py           # Vector embedding & indexing
│   ├── search.py            # Search logic & ranking
│   ├── tui.py               # Terminal UI (Textual)
│   ├── config.py            # Configuration management
│   └── utils.py             # Helper functions (markdown parsing, etc.)
├── data/
│   ├── chroma/              # Vector database storage
│   └── cache/               # Cached embeddings
├── tests/
│   ├── test_indexer.py
│   ├── test_search.py
│   └── fixtures/            # Sample markdown files
├── config.yaml              # User configuration
├── pyproject.toml           # UV project configuration
├── uv.lock                  # UV lockfile
├── .python-version          # Python version for UV
├── README.md
└── search.py                # Main entry point
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
    Split content into semantic chunks
    Uses: sentence boundaries, paragraph breaks
    """

def generate_embeddings(chunks: List[str]) -> List[np.ndarray]:
    """
    Generate vector embeddings using sentence-transformers
    Model: all-MiniLM-L6-v2 (fast, good quality, 384 dimensions)
    """

def index_documents(docs: List[Document], collection_name: str = "book_content"):
    """
    Store documents and embeddings in ChromaDB
    Includes: content, metadata, embeddings
    """

def rebuild_index(directory: str, force: bool = False):
    """
    Full reindex of all documents
    Shows progress bar during indexing
    """

def incremental_update(directory: str):
    """
    Update only changed/new files (based on mtime)
    """
```

**Document Schema:**
```python
@dataclass
class Document:
    id: str                    # Unique ID (file path hash)
    file_path: str             # Relative path to file
    title: str                 # From frontmatter or H1
    content: str               # Full markdown content
    chunks: List[str]          # Content split into chunks
    frontmatter: Dict          # YAML frontmatter as dict
    last_modified: float       # File mtime for incremental updates
    word_count: int            # Content statistics
```

---

### **2.2 Search Module** (`search.py`)

**Purpose:** Execute searches, rank results, format output

**Key Functions:**
```python
def semantic_search(query: str, limit: int = 10) -> List[SearchResult]:
    """
    Vector similarity search using ChromaDB
    Returns: Top N results with similarity scores
    """

def keyword_search(query: str, limit: int = 10) -> List[SearchResult]:
    """
    Traditional keyword/phrase search (fallback)
    Uses: simple text matching on content
    """

def hybrid_search(query: str, limit: int = 10, 
                 semantic_weight: float = 0.7) -> List[SearchResult]:
    """
    Combine semantic + keyword search
    Weighted fusion of results
    """

def get_context_snippet(content: str, query: str, 
                       window: int = 150) -> str:
    """
    Extract relevant snippet around query terms
    Returns: "...context before **match** context after..."
    """

def rank_results(results: List[SearchResult], 
                query: str) -> List[SearchResult]:
    """
    Post-processing ranking:
    - Boost by frontmatter relevance (tags, category)
    - Penalize very short chunks
    - Boost recent files (optional)
    """
```

**SearchResult Schema:**
```python
@dataclass
class SearchResult:
    document_id: str
    file_path: str
    title: str
    snippet: str               # Context around match
    score: float               # Similarity score (0-1)
    frontmatter: Dict          # For display filtering
    chunk_index: int           # Which chunk matched
    matched_chunk: str         # Full chunk text
```

---

### **2.3 TUI Module** (`tui.py`)

**Purpose:** Beautiful terminal interface using Textual framework

**Main Screen Layout:**
```
┌─ Book Search ────────────────────────────────────────────┐
│ 🔍 Search: authentication patterns            [Ctrl+R]   │
│ Mode: [●Semantic] [○Keyword] [○Hybrid]    Results: 8    │
├──────────────────────────────────────────────────────────┤
│                                                           │
│ ► 📄 chapter-03-security.md                    (0.89)    │
│   Tags: security, auth                                    │
│   ...discusses JWT authentication patterns for            │
│   microservices. The key consideration is...              │
│   ──────────────────────────────────────────────         │
│                                                           │
│   📄 chapter-07-api-design.md                  (0.84)    │
│   Tags: api, design                                       │
│   ...authentication middleware should handle...           │
│   ──────────────────────────────────────────────         │
│                                                           │
│   📄 appendix-b-patterns.md                    (0.78)    │
│   Tags: patterns, reference                               │
│   ...common authentication antipatterns...                │
│                                                           │
├──────────────────────────────────────────────────────────┤
│ [↑↓] Navigate [Enter] View [Tab] Mode [/] Search [Q]uit  │
└──────────────────────────────────────────────────────────┘
```

**Detail View (on Enter):**
```
┌─ chapter-03-security.md ─────────────────────────────────┐
│ ← Back [Esc]                                              │
├──────────────────────────────────────────────────────────┤
│ Title: Security Best Practices                           │
│ Tags: security, auth, jwt                                 │
│ Last Modified: 2024-11-05                                 │
│ Word Count: 2,450                                         │
├──────────────────────────────────────────────────────────┤
│                                                           │
│ [Scrollable markdown preview with syntax highlighting]   │
│                                                           │
│ ## Authentication                                         │
│                                                           │
│ JWT authentication patterns for microservices require...  │
│                                                           │
├──────────────────────────────────────────────────────────┤
│ [O] Open in VS Code  [C] Copy Path  [Esc] Back           │
└──────────────────────────────────────────────────────────┘
```

**Key TUI Components:**

```python
class SearchApp(App):
    """Main Textual application"""
    
    CSS_PATH = "styles.css"  # Textual CSS for styling
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("/", "focus_search", "Search"),
        ("tab", "cycle_mode", "Toggle Mode"),
        ("ctrl+r", "rebuild_index", "Rebuild Index"),
        ("ctrl+h", "show_help", "Help"),
    ]
    
    def compose(self) -> ComposeResult:
        """Build UI layout"""
        yield Header()
        yield SearchInput(placeholder="Enter search query...")
        yield ModeSelector()
        yield ResultsList()
        yield Footer()
    
    def on_search_input_submitted(self, event):
        """Handle search execution"""
        
    def on_result_selected(self, event):
        """Show detail view"""

class ResultsList(ListView):
    """Scrollable list of search results"""
    
class ResultItem(ListItem):
    """Single search result display"""
    
class DetailView(Screen):
    """Full document preview modal"""
```

**Keyboard Shortcuts:**
- `/` or `Ctrl+K`: Focus search input
- `↑↓`: Navigate results
- `Enter`: View full document
- `Tab`: Cycle search mode (Semantic → Keyword → Hybrid)
- `O`: Open in VS Code (from detail view)
- `C`: Copy file path to clipboard
- `Ctrl+R`: Rebuild index
- `Esc`: Go back / close detail view
- `Q`: Quit application

**Visual Features:**
- **Color coding:** Score-based highlighting (green >0.8, yellow >0.6, white <0.6)
- **Icons:** File type indicators (📄 .md, 📁 folder)
- **Progress bars:** During indexing operations
- **Status messages:** Bottom bar shows current operation
- **Markdown preview:** Syntax highlighted in detail view

---

### **2.4 Config Module** (`config.py`)

**Purpose:** Manage user configuration

**config.yaml:**
```yaml
# Paths
content_directory: "/path/to/your/hugo/content"
excluded_patterns:
  - "drafts/*"
  - "archive/*"
  - "*.draft.md"

# Search Settings
default_search_mode: "semantic"  # semantic | keyword | hybrid
max_results: 10
snippet_length: 150

# Embedding Model
model_name: "all-MiniLM-L6-v2"
chunk_size: 512
chunk_overlap: 50

# Vector Database
collection_name: "book_content"
similarity_metric: "cosine"  # cosine | euclidean | dot

# TUI Settings
theme: "monokai"  # monokai | dracula | nord
show_frontmatter: true
auto_open_vscode: false

# Performance
cache_embeddings: true
incremental_updates: true
index_on_startup: false
```

---

## **3. Technology Stack & UV Configuration**

### **pyproject.toml:**

```toml
[project]
name = "book-search"
version = "0.1.0"
description = "Local vector search for markdown documentation with TUI"
readme = "README.md"
requires-python = ">=3.11"
authors = [
    { name = "Your Name", email = "your.email@example.com" }
]
keywords = ["search", "vector", "markdown", "tui", "embeddings"]
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
    
    # TUI Framework
    "textual>=0.41.0",
    "rich>=13.6.0",
    
    # Markdown & Content Processing
    "python-frontmatter>=1.0.1",
    "markdown>=3.5.1",
    
    # Utilities
    "pyyaml>=6.0.1",
    "click>=8.1.7",
    "watchdog>=3.0.0",
    "pyperclip>=1.8.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.3",
    "pytest-asyncio>=0.21.0",
    "black>=23.11.0",
    "mypy>=1.7.0",
    "ruff>=0.1.6",
]

[project.scripts]
book-search = "src.tui:main"
book-index = "src.indexer:cli"

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

# Run the application
uv run book-search

# Run with arguments
uv run book-search query "authentication patterns"

# Run tests
uv run pytest

# Run with specific Python version
uv run --python 3.12 book-search

# Format code
uv run black src/

# Type checking
uv run mypy src/

# Create a virtual environment (if needed for IDE)
uv venv
source .venv/bin/activate  # On Unix
# or
.venv\Scripts\activate     # On Windows
```

### **Development Workflow:**

ALWAYS use test driven development.

```bash
# Install project in editable mode with dev dependencies
uv sync --extra dev

# Run the indexer
uv run book-index --rebuild

# Launch TUI
uv run book-search

# Run tests with coverage
uv run pytest --cov=src tests/

# Format and lint
uv run black src/ tests/
uv run ruff check src/ tests/

# Type check
uv run mypy src/
```

### **UV Tool Installation (Alternative):**

```bash
# Install as a UV tool (global installation)
uv tool install book-search

# Then run from anywhere
book-search

# Update tool
uv tool upgrade book-search

# Uninstall
uv tool uninstall book-search
```

---

## **5. Usage Examples**

### **Initial Setup:**
```bash
# Clone or create project
git clone <your-repo> book-search
cd book-search

# Install with UV
uv sync

# Configure
cp config.example.yaml config.yaml
# Edit config.yaml with your Hugo content path

# Build index (one-time, ~30 seconds for 100 files)
uv run book-index --rebuild

# Launch TUI
uv run book-search
```

### **Daily Workflow:**
```bash
# Start search TUI
uv run book-search

# Or direct search from CLI
uv run book-search query "authentication patterns" --mode semantic

# Rebuild index after writing new content
uv run book-index --update

# Watch mode (auto-reindex on file changes)
uv run book-search watch
```

### **Development Workflow:**
```bash
# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Format code
uv run black src/ tests/

# Lint
uv run ruff check src/

# Type check
uv run mypy src/

# Run specific test
uv run pytest tests/test_search.py -v

# Update dependencies
uv lock --upgrade
```

---

## **6. Performance Targets**

For 100 markdown files (~500KB total):

| Operation | Target | Notes |
|-----------|--------|-------|
| Initial indexing | < 30s | One-time setup |
| Incremental update | < 5s | Only changed files |
| Search query | < 200ms | Return top 10 results |
| TUI startup | < 2s | With cached embeddings |
| Memory usage | < 500MB | Embeddings in memory |
| UV sync time | < 10s | First time with cached downloads |

---

## **7. Implementation Phases**

### **Phase 1: Project Setup & Core Indexing (Week 1)**
- [ ] Initialize UV project with pyproject.toml
- [ ] Set up project structure
- [ ] File scanning and markdown parsing
- [ ] Frontmatter extraction
- [ ] Content chunking strategy
- [ ] Embedding generation
- [ ] ChromaDB integration
- [ ] Basic CLI for indexing
- **Deliverable:** Can index 100 files, generate embeddings

### **Phase 2: Search Engine (Week 1)**
- [ ] Vector similarity search
- [ ] Keyword search fallback
- [ ] Hybrid search with ranking
- [ ] Context snippet extraction
- [ ] Result filtering by frontmatter
- **Deliverable:** Search works via Python API

### **Phase 3: Basic TUI (Week 2)**
- [ ] Textual app skeleton
- [ ] Search input component
- [ ] Results list display
- [ ] Basic keyboard navigation
- [ ] Mode switching (semantic/keyword/hybrid)
- **Deliverable:** Minimal working TUI

### **Phase 4: Enhanced TUI (Week 2)**
- [ ] Detail view modal
- [ ] Markdown preview with syntax highlighting
- [ ] VS Code integration (open file)
- [ ] Clipboard operations
- [ ] Progress indicators
- [ ] Help screen
- **Deliverable:** Polished user experience

### **Phase 5: Polish & Performance (Week 3)**
- [ ] Incremental indexing
- [ ] Caching strategy
- [ ] Startup optimization
- [ ] Error handling
- [ ] Unit tests
- [ ] Documentation
- **Deliverable:** Production-ready tool

---

## **8. Testing Strategy**

```python
# tests/test_indexer.py
def test_frontmatter_parsing():
    """Ensure Hugo frontmatter is correctly extracted"""

def test_content_chunking():
    """Verify chunk sizes and overlaps"""

def test_embedding_generation():
    """Check embedding dimensions and consistency"""

# tests/test_search.py
def test_semantic_search_relevance():
    """Query "auth" should match "authentication" content"""

def test_keyword_search():
    """Exact phrase matching works"""

def test_hybrid_search_fusion():
    """Combined results are properly ranked"""

# tests/test_tui.py (Textual has built-in testing)
async def test_search_input():
    """Search input captures and submits queries"""
```

**Run tests with UV:**
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_search.py -v

# Run tests in parallel
uv run pytest -n auto
```

---

## **9. Documentation Requirements**

**README.md should include:**
1. Quick start guide with UV commands
2. Configuration options
3. Keyboard shortcuts reference
4. Troubleshooting common issues
5. Architecture overview
6. How to contribute
7. UV installation instructions

**Example README snippet:**
```markdown
## Installation

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repo-url> book-search
cd book-search
uv sync

# Configure
cp config.example.yaml config.yaml
# Edit config.yaml with your content path

# Build index
uv run book-index --rebuild

# Launch
uv run book-search
```
```

**In-app help (`Ctrl+H`):**
- Keyboard shortcuts
- Search tips (semantic vs keyword)
- How to interpret scores

---

## **10. Edge Cases & Error Handling**

**Handle gracefully:**
- Empty search results → Show helpful message
- Malformed markdown → Skip file, log warning
- Missing frontmatter → Use filename as title
- Embedding model download → Show progress (UV handles caching)
- Corrupted ChromaDB → Offer rebuild
- Large files (>10KB) → Warn about chunking
- No write permissions → Suggest alternative data dir
- Python version mismatch → UV will warn automatically

---

## **11. Success Metrics**

**The tool is successful if:**
1. ✅ Indexes 100 files in under 30 seconds
2. ✅ Returns relevant results in <200ms
3. ✅ Can find conceptually related content (not just keywords)
4. ✅ TUI is intuitive for daily use
5. ✅ Works offline (no external API calls)
6. ✅ Minimal resource usage (<500MB RAM)
7. ✅ UV setup takes < 2 minutes from clone to running
8. ✅ Dependency resolution is fast and reliable

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

## **Estimated Timeline**

- **Week 1:** Core indexing + search engine (8-10 hours)
- **Week 2:** TUI implementation (8-10 hours)
- **Week 3:** Polish, testing, docs (4-6 hours)

**Total: 2-3 weeks of part-time work**

---

## **Distribution Options**

### **1. GitHub Release:**
```bash
# Users install via UV
uv tool install git+https://github.com/yourusername/book-search

# Or clone and install
git clone <repo>
cd book-search
uv sync
```

### **2. PyPI (Future):**
```bash
# Build and publish with UV
uv build
uv publish

# Users install
uv tool install book-search
```
