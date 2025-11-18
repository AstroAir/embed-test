# VectorFlow

[![PyPI version](https://badge.fury.io/py/vectorflow.svg)](https://badge.fury.io/py/vectorflow)
[![Python versions](https://img.shields.io/pypi/pyversions/vectorflow.svg)](https://pypi.org/project/vectorflow/)
[![License](https://img.shields.io/github/license/your-username/vectorflow.svg)](https://github.com/your-username/vectorflow/blob/main/LICENSE)
[![CI](https://github.com/your-username/vectorflow/workflows/CI/badge.svg)](https://github.com/your-username/vectorflow/actions)
[![Coverage](https://codecov.io/gh/your-username/vectorflow/branch/main/graph/badge.svg)](https://codecov.io/gh/your-username/vectorflow)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://your-username.github.io/vectorflow/)

**Streamlined PDF-to-vector processing pipeline with multi-provider embeddings and multi-backend vector storage**

VectorFlow is a comprehensive, production-ready system for extracting text from PDF documents, generating embeddings through multiple AI providers, and storing vectors in scalable databases. Designed for flexibility, performance, and ease of use.

## Key Features

### Multi-Provider Embedding Support
Choose from 7 embedding providers optimized for different use cases:
- **OpenAI**: text-embedding-3-small, text-embedding-3-large (highest quality)
- **Azure OpenAI**: Enterprise-grade embeddings with Azure integration
- **Sentence Transformers**: Fast, local embeddings (all-MiniLM-L6-v2, all-mpnet-base-v2, etc.)
- **Cohere**: Advanced semantic understanding with Cohere models
- **HuggingFace**: Access to 1000+ open-source models
- **Google Gemini**: Cutting-edge embeddings from Google
- **Google Universal Sentence Encoder**: Lightweight, multilingual embeddings

### Multi-Backend Vector Storage
Store vectors in any of 5 production-ready database backends:
- **ChromaDB**: Lightweight, in-memory or persistent storage
- **Milvus**: High-performance, distributed vector database
- **Pinecone**: Serverless vector database with semantic search
- **Qdrant**: Fully-featured vector search engine
- **Weaviate**: GraphQL-powered vector database

### Dual Interfaces
- **CLI**: Full-featured command-line interface for batch processing and search
- **GUI**: Modern, user-friendly desktop application built with PySide6 and qfluentwidgets

### Developer-Focused Design
- **Type-Safe Configuration**: Pydantic-based configuration with IDE autocomplete
- **Modular Architecture**: Core library, CLI, and GUI modules for flexible integration
- **Batch Processing**: Efficient handling of large document sets
- **Progress Tracking**: Real-time progress bars and performance metrics
- **Comprehensive Error Handling**: Graceful failure recovery with detailed logging
- **Full Documentation**: API docs, examples, and architecture guides

## Installation

### From PyPI (Recommended)

```bash
pip install vectorflow
```

### From Source

```bash
# Clone the repository
git clone https://github.com/your-username/vectorflow.git
cd vectorflow

# Install dependencies using uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

### Using UV (Fast)

```bash
uv add vectorflow
```

### Quick Install Script

For a guided installation experience:

```bash
# Linux/macOS
curl -sSL https://raw.githubusercontent.com/your-username/vectorflow/main/scripts/install.sh | bash

# Windows
powershell -Command "iwr https://raw.githubusercontent.com/your-username/vectorflow/main/scripts/install.bat -OutFile install.bat; .\install.bat"
```

### Prerequisites

- Python 3.9 or higher
- 2GB+ RAM (recommended for embedding models)
- 1GB+ disk space for models and data

### Optional Dependencies

```bash
# Development dependencies
pip install vectorflow[dev]

# Documentation dependencies
pip install vectorflow[docs]

# All dependencies
pip install vectorflow[dev,docs]
```

### Configuration

For API-based embeddings, set your API keys:

```bash
# OpenAI
export OPENAI_API_KEY="your-api-key-here"

# Cohere
export COHERE_API_KEY="your-api-key-here"

# Google
export GOOGLE_API_KEY="your-api-key-here"
```

### Verification

Verify your installation:

```bash
# Check CLI installation
vectorflow --help

# Check Python import
python -c "import vectorflow; print(vectorflow.__version__)"
```

## Quick Start

### Basic Usage with Sentence Transformers

```python
from vectorflow import Config, PDFVectorPipeline
from vectorflow.core.config.settings import EmbeddingModelType

# Create configuration for local embeddings
config = Config()
config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
config.embedding.model_name = "all-MiniLM-L6-v2"

# Initialize pipeline
pipeline = PDFVectorPipeline(config)

# Process a PDF
result = pipeline.process_pdf("document.pdf", show_progress=True)

if result.success:
    print(f"Processed {result.chunks_processed} chunks in {result.processing_time:.2f}s")

    # Search the document
    search_results = pipeline.search("machine learning", n_results=5)
    for result in search_results:
        print(f"Score: {result.score:.3f} - {result.content[:100]}...")
```

### Advanced Configuration with Multi-Backend Support

```python
from pathlib import Path

from vectorflow import Config, PDFVectorPipeline
from vectorflow.core.config.settings import EmbeddingModelType
from vectorflow.core.vector_db.config import PineconeConfig

# Advanced configuration
config = Config()

# OpenAI embeddings
config.embedding.model_type = EmbeddingModelType.OPENAI
config.embedding.model_name = "text-embedding-3-small"
config.embedding.batch_size = 50

# Text processing
config.text_processing.chunk_size = 1200
config.text_processing.chunk_overlap = 200

# Use Pinecone for vector storage
config.vector_db = PineconeConfig(
    api_key="your-pinecone-key",
    environment="your-pinecone-environment",
    index_name="pdf-documents",
    dimension=1536,  # must match your embedding model
    collection_name="pdf_documents",
)

# Performance tuning
config.max_workers = 6

# Initialize and use
pipeline = PDFVectorPipeline(config)
```

### Using Different Embedding Providers

```python
from vectorflow import Config, PDFVectorPipeline
from vectorflow.core.config.settings import EmbeddingModelType

config = Config()

# Option 1: Google Gemini
config.embedding.model_type = EmbeddingModelType.GOOGLE_GEMINI
config.embedding.model_name = "gemini-embedding-001"

# Option 2: Cohere
config.embedding.model_type = EmbeddingModelType.COHERE
config.embedding.model_name = "embed-english-v3.0"

# Option 3: HuggingFace
config.embedding.model_type = EmbeddingModelType.HUGGINGFACE
config.embedding.model_name = "sentence-transformers/all-mpnet-base-v2"

pipeline = PDFVectorPipeline(config)
```

## Configuration

VectorFlow uses Pydantic models for type-safe configuration. You can configure via:

1. **Environment variables** (recommended for production)
2. **Configuration objects** (recommended for development)
3. **`.env` files**

### Environment Variables

```bash
# Embedding Configuration
EMBEDDING__MODEL_TYPE=openai
EMBEDDING__MODEL_NAME=text-embedding-3-small
EMBEDDING__BATCH_SIZE=32

# OpenAI Configuration
OPENAI_API_KEY=your_api_key_here

# ChromaDB Configuration (default local vector database)
CHROMA_DB__PERSIST_DIRECTORY=./chroma_db
CHROMA_DB__COLLECTION_NAME=pdf_documents
CHROMA_DB__DISTANCE_METRIC=cosine
CHROMA_DB__MAX_RESULTS=100

# Text Processing
TEXT_PROCESSING__CHUNK_SIZE=1000
TEXT_PROCESSING__CHUNK_OVERLAP=200

# Logging
LOGGING__LEVEL=INFO
LOGGING__FILE_PATH=./logs/vectorflow.log
```

### Configuration File

Copy `.env.example` to `.env` and modify as needed:

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

## Architecture

VectorFlow is built on a modular architecture with three main components:

### Core Module (`vectorflow.core`)

The heart of the system providing:

1. **PDF Processor** (`pdf.processor`)
   - Robust text extraction from various PDF formats using PyMuPDF
   - Metadata extraction and validation
   - Support for large documents with streaming

2. **Text Processor** (`pdf.text_processor`)
   - Intelligent text cleaning and normalization
   - Configurable chunking with LangChain's RecursiveCharacterTextSplitter
   - Preserves context with configurable overlap

3. **Embedding Services** (`embeddings`)
   - Factory pattern for 7+ embedding providers
   - Batch processing with automatic retry logic
   - Performance tracking and error handling
   - Automatic model downloading and caching

4. **Vector Database** (`vector_db`)
   - Multi-backend support (ChromaDB, Milvus, Pinecone, Qdrant, Weaviate)
   - Unified interface across all backends
   - Advanced search and filtering capabilities
   - Collection management utilities

5. **Pipeline Orchestrator** (`pipeline`)
   - Coordinates all components
   - Progress tracking and performance monitoring
   - High-level API for PDF processing
   - Document lifecycle management

### CLI Module (`vectorflow.cli`)

Command-line interface with commands for:
- Document processing and batch operations
- Vector database search
- Collection statistics and management
- Configuration inspection and validation

### GUI Module (`vectorflow.gui`)

Desktop application featuring:
- Modern PySide6 interface with qfluentwidgets styling
- MVC architecture for maintainability
- Real-time progress tracking
- Multi-document batch processing
- Semantic search with result preview
- Configuration management UI

## Usage Examples

### Command Line Interface

```bash
# Process a single PDF
vectorflow process document.pdf

# Process multiple PDFs
vectorflow process *.pdf

# Search the database
vectorflow search "machine learning"

# Show collection statistics
vectorflow stats

# List all documents
vectorflow list-docs

# Clear a collection
vectorflow clear-collection

# Export search results (using shell redirection)
vectorflow search "query" --results 20 > results.txt
```

### GUI Application

```bash
# Launch the desktop application
vectorflow-gui
```

### Python API Examples

#### Basic Search

```python
from vectorflow import Config, PDFVectorPipeline

config = Config()
pipeline = PDFVectorPipeline(config)

# Process and search
pipeline.process_pdf("document.pdf")
results = pipeline.search("your query", n_results=10)

for result in results:
    print(f"{result.content}\n(Score: {result.score:.3f})")
```

#### Batch Processing

```python
from pathlib import Path
from vectorflow import Config, PDFVectorPipeline

config = Config()
pipeline = PDFVectorPipeline(config)

# Process all PDFs in a directory
pdf_files = Path("documents").glob("*.pdf")
for pdf_file in pdf_files:
    result = pipeline.process_pdf(str(pdf_file), show_progress=True)
    if result.success:
        print(f"Processed: {pdf_file}")
```

#### Custom Chunking Strategy

```python
from vectorflow import Config, PDFVectorPipeline

config = Config()
config.text_processing.chunk_size = 800  # Smaller chunks for precision
config.text_processing.chunk_overlap = 100

pipeline = PDFVectorPipeline(config)
result = pipeline.process_pdf("document.pdf")
```

## API Reference

### PDFVectorPipeline

Main class for processing PDFs and managing the vector database.

#### Methods

- `process_pdf(pdf_path, document_id=None, clean_text=True, show_progress=True)`: Process a PDF file
- `search(query_text, n_results=10, document_id=None, page_number=None)`: Search the vector database
- `get_document_info(document_id)`: Get information about a processed document
- `delete_document(document_id)`: Delete a document from the database
- `get_collection_stats()`: Get collection statistics
- `health_check()`: Perform system health check
- `get_documents()`: List all processed documents

### Configuration Classes

- `Config`: Main configuration class
- `EmbeddingConfig`: Embedding provider configuration (supports 7 providers)
- `VectorDBConfig`: Vector database configuration (supports 5 backends)
- `TextProcessingConfig`: Text processing and chunking settings
- `LoggingConfig`: Logging configuration
- `PDFConfig`: PDF processing settings

## Performance

### Benchmarks

Typical performance on a modern laptop (M1 MacBook Pro):

- **PDF Text Extraction**: ~50-100 pages/second
- **Text Chunking**: ~10,000 chunks/second
- **Sentence Transformers Embeddings**: ~100-500 texts/second
- **OpenAI Embeddings**: ~1,000-2,000 texts/second (API dependent)
- **Vector Database Storage**: ~1,000-5,000 chunks/second (backend dependent)

### Optimization Tips

1. **Use appropriate batch sizes**:
   - Sentence Transformers: 16-32
   - OpenAI: 50-100
   - Google/Cohere: 32-64

2. **Adjust chunk size**:
   - Larger chunks (1000-1500 chars): Better context
   - Smaller chunks (500-800 chars): Better precision

3. **Use multiple workers**: Set `max_workers` to 4-8 for CPU-bound tasks

4. **Choose the right embedding model**:
   - `all-MiniLM-L6-v2`: Fast, good quality
   - `all-mpnet-base-v2`: Slower, higher quality
   - OpenAI models: Highest quality, API costs
   - Google Gemini: Balanced performance and quality

5. **Select appropriate vector database**:
   - ChromaDB: Development and small-scale deployments
   - Pinecone: Cloud-hosted, managed infrastructure
   - Milvus: Self-hosted, high-performance
   - Weaviate: GraphQL interface, rich metadata support

## Troubleshooting

### Common Issues

1. **"Model not found" error**: Ensure the model name is correct and available for the provider
2. **API errors**: Check your API keys and rate limits for cloud-based providers
3. **Vector database connection issues**: Verify connection strings and credentials
4. **Memory issues with large PDFs**: Reduce batch size or chunk size
5. **Slow processing**: Increase batch size, use more workers, or optimize chunk size

### Debug Mode

Enable debug mode for detailed logging:

```python
config = Config()
config.debug = True
```

Or set environment variable:

```bash
DEBUG=true
LOGGING__LEVEL=DEBUG
```

### Getting Help

- Check the [troubleshooting guide](docs/troubleshooting.md)
- Review [architecture documentation](llmdoc/feature/core-module-architecture.md)
- Open an [issue on GitHub](https://github.com/your-username/vectorflow/issues)

## Development

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-username/vectorflow.git
cd vectorflow

# Install development dependencies
uv sync --extra dev

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run quality checks
ruff check vectorflow tests
mypy vectorflow
```

### Build and Release

```bash
# Build the package
./scripts/build.sh

# Build with all checks
./scripts/build.sh --docs --clean

# Install locally
pip install dist/*.whl
```

### Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=vectorflow --cov-report=term-missing

# Run specific test file
pytest tests/test_pipeline.py

# Run with parallel execution
pytest -n auto

# Run integration tests only
pytest -m integration

# Run tests excluding external dependencies
pytest -m "not external"
```

### Code Quality

```bash
# Lint with Ruff
ruff check vectorflow tests

# Format with Ruff
ruff format vectorflow tests

# Type checking
mypy vectorflow

# Security scanning
bandit -r vectorflow

# Dependency safety
safety check
```

### CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment:

- **Automated Testing**: Tests run on Python 3.9-3.12 across Linux, macOS, and Windows
- **Code Quality**: Automated linting, formatting, and type checking with ruff, black, isort, and mypy
- **Security Scanning**: Automated security checks with bandit and safety
- **Documentation**: Automatic documentation building and deployment with MkDocs
- **Package Publishing**: Automated PyPI publishing on release tags
- **Dependency Updates**: Weekly automated dependency updates

### Quality Assurance Standards

- **Test Coverage**: >80% code coverage requirement
- **Type Safety**: Full type hints with mypy checking
- **Code Style**: Enforced with ruff, black, and isort
- **Security**: Regular security scanning with bandit and safety
- **Pre-commit Hooks**: Automated quality checks before commits

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and quality checks: `pytest tests/ && ruff check . && mypy .`
5. Commit your changes following [conventional commits](https://www.conventionalcommits.org/):
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `test:` for tests
   - `refactor:` for code refactoring
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

For detailed contribution guidelines, see our [Contributing Guide](docs/development/contributing.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [https://your-username.github.io/vectorflow/](https://your-username.github.io/vectorflow/)
- **Issues**: [GitHub Issues](https://github.com/your-username/vectorflow/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/vectorflow/discussions)
- **PyPI**: [https://pypi.org/project/vectorflow/](https://pypi.org/project/vectorflow/)

## Acknowledgments

- [PyMuPDF](https://pymupdf.readthedocs.io/) for robust PDF processing
- [sentence-transformers](https://www.sbert.net/) for local embedding models
- [ChromaDB](https://www.trychroma.com/) for vector database
- [LangChain](https://langchain.com/) for text processing utilities
- [OpenAI](https://openai.com/) for embeddings API
- [Cohere](https://cohere.com/) for semantic embeddings
- [Google](https://cloud.google.com/ai-platform) for embedding services
- [PySide6](https://wiki.qt.io/Qt_for_Python) for desktop application framework
- [qfluentwidgets](https://github.com/zhiyiYo/PyQt-Fluent-Widgets) for modern UI components

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes.

## Roadmap

### Planned Features
- Support for additional document formats (DOCX, TXT, MD)
- Streaming API for real-time processing
- Advanced caching and incremental updates
- Multi-language support
- Custom embedding fine-tuning
- Real-time collaboration features

Stay tuned for exciting updates!
