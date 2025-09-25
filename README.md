# PDF Vector System

[![PyPI version](https://badge.fury.io/py/pdf-vector-system.svg)](https://badge.fury.io/py/pdf-vector-system)
[![Python versions](https://img.shields.io/pypi/pyversions/pdf-vector-system.svg)](https://pypi.org/project/pdf-vector-system/)
[![License](https://img.shields.io/github/license/your-username/pdf-vector-system.svg)](https://github.com/your-username/pdf-vector-system/blob/main/LICENSE)
[![CI](https://github.com/your-username/pdf-vector-system/workflows/CI/badge.svg)](https://github.com/your-username/pdf-vector-system/actions)
[![Coverage](https://codecov.io/gh/your-username/pdf-vector-system/branch/main/graph/badge.svg)](https://codecov.io/gh/your-username/pdf-vector-system)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://your-username.github.io/pdf-vector-system/)

A comprehensive PDF content processing and vector storage system with ChromaDB integration. This system extracts text from PDF files, processes it for embedding generation, integrates with language models to generate embeddings, and stores them in ChromaDB for similarity search and retrieval.

## Features

- **PDF Content Extraction**: Robust text extraction from various PDF formats using PyMuPDF
- **Text Processing**: Intelligent text cleaning, normalization, and chunking with configurable parameters
- **Dual Embedding Support**:
  - Local embeddings with sentence-transformers (all-MiniLM-L6-v2, all-mpnet-base-v2, etc.)
  - API-based embeddings with OpenAI (text-embedding-3-small, text-embedding-3-large)
- **Vector Database**: ChromaDB integration with comprehensive metadata and similarity search
- **Batch Processing**: Efficient processing of multiple documents with progress tracking
- **Configuration Management**: Type-safe configuration with Pydantic models
- **Comprehensive Logging**: Detailed logging with loguru and progress tracking with rich
- **Error Handling**: Robust error handling and graceful failure recovery

## Installation

### From PyPI (Recommended)

```bash
pip install pdf-vector-system
```

### From Source

```bash
# Clone the repository
git clone https://github.com/your-username/pdf-vector-system.git
cd pdf-vector-system

# Install dependencies using uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

### Using UV (Fast)

```bash
uv add pdf-vector-system
```

### Quick Install Script

For a guided installation experience:

```bash
# Linux/macOS
curl -sSL https://raw.githubusercontent.com/your-username/pdf-vector-system/main/scripts/install.sh | bash

# Windows
powershell -Command "iwr https://raw.githubusercontent.com/your-username/pdf-vector-system/main/scripts/install.bat -OutFile install.bat; .\install.bat"
```

### Prerequisites

- Python 3.9 or higher
- 2GB+ RAM (recommended for embedding models)
- 1GB+ disk space for models and data

### Optional Dependencies

```bash
# Development dependencies
pip install pdf-vector-system[dev]

# Documentation dependencies
pip install pdf-vector-system[docs]

# All dependencies
pip install pdf-vector-system[dev,docs]
```

### Configuration

For OpenAI embeddings, set your API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Verification

Verify your installation:

```bash
# Check CLI installation
pdf-vector --help

# Check Python import
python -c "import pdf_vector_system; print(pdf_vector_system.__version__)"
```

## Quick Start

### Basic Usage

```python
from pdf_vector_system import Config, PDFVectorPipeline
from pdf_vector_system.config.settings import EmbeddingModelType

# Create configuration
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

### Advanced Configuration

```python
from pathlib import Path
from pdf_vector_system import Config, PDFVectorPipeline
from pdf_vector_system.config.settings import EmbeddingModelType

# Advanced configuration
config = Config()

# OpenAI embeddings
config.embedding.model_type = EmbeddingModelType.OPENAI
config.embedding.model_name = "text-embedding-3-small"
config.embedding.batch_size = 50

# Text processing
config.text_processing.chunk_size = 1200
config.text_processing.chunk_overlap = 200

# ChromaDB
config.chroma_db.collection_name = "my_documents"
config.chroma_db.persist_directory = Path("./my_vector_db")

# Performance
config.max_workers = 6

# Initialize and use
pipeline = PDFVectorPipeline(config)
```

## Configuration

The system uses Pydantic models for type-safe configuration. You can configure via:

1. **Environment variables** (recommended for production)
2. **Configuration objects** (recommended for development)
3. **`.env` files**

### Environment Variables

```bash
# Embedding Configuration
EMBEDDING__MODEL_TYPE=sentence-transformers
EMBEDDING__MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING__BATCH_SIZE=32

# OpenAI Configuration (if using OpenAI embeddings)
OPENAI_API_KEY=your_api_key_here

# ChromaDB Configuration
CHROMA_DB__PERSIST_DIRECTORY=./chroma_db
CHROMA_DB__COLLECTION_NAME=pdf_documents

# Text Processing
TEXT_PROCESSING__CHUNK_SIZE=1000
TEXT_PROCESSING__CHUNK_OVERLAP=200

# Logging
LOGGING__LEVEL=INFO
LOGGING__FILE_PATH=./logs/pdf_vector_system.log
```

### Configuration File

Copy `.env.example` to `.env` and modify as needed:

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

## Architecture

The system consists of several key components:

### 1. PDF Processor (`pdf_vector_system.pdf.processor`)
- Extracts text from PDF files using PyMuPDF
- Handles various PDF formats and structures
- Provides metadata extraction and validation

### 2. Text Processor (`pdf_vector_system.pdf.text_processor`)
- Cleans and normalizes extracted text
- Implements intelligent chunking with LangChain's RecursiveCharacterTextSplitter
- Configurable chunk size and overlap

### 3. Embedding Services (`pdf_vector_system.embeddings`)
- **Sentence Transformers**: Local embedding generation
- **OpenAI**: API-based embedding generation
- Batch processing with progress tracking
- Automatic retry logic and error handling

### 4. Vector Database (`pdf_vector_system.vector_db`)
- ChromaDB integration with persistent storage
- Comprehensive metadata support
- Advanced search and filtering capabilities
- Collection management utilities

### 5. Main Pipeline (`pdf_vector_system.pipeline`)
- Orchestrates all components
- Provides high-level API for PDF processing
- Progress tracking and performance monitoring

## Examples

### Running Examples

```bash
# Basic usage with sentence-transformers
python examples/basic_usage.py

# Advanced usage with OpenAI embeddings (requires API key)
python examples/advanced_usage.py
```

### Command Line Interface

```bash
# Process a single PDF
pdf-vector process document.pdf

# Process multiple PDFs
pdf-vector process *.pdf

# Search the database
pdf-vector search "machine learning"

# Show collection statistics
pdf-vector stats

# List all documents
pdf-vector list
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

### Configuration Classes

- `Config`: Main configuration class
- `PDFConfig`: PDF processing configuration
- `TextProcessingConfig`: Text processing and chunking configuration
- `EmbeddingConfig`: Embedding generation configuration
- `ChromaDBConfig`: Vector database configuration
- `LoggingConfig`: Logging configuration

## Performance

### Benchmarks

Typical performance on a modern laptop (M1 MacBook Pro):

- **PDF Text Extraction**: ~50-100 pages/second
- **Text Chunking**: ~10,000 chunks/second
- **Sentence Transformers Embeddings**: ~100-500 texts/second
- **OpenAI Embeddings**: ~1,000-2,000 texts/second (API dependent)
- **ChromaDB Storage**: ~1,000-5,000 chunks/second

### Optimization Tips

1. **Use appropriate batch sizes**: 16-32 for sentence-transformers, 50-100 for OpenAI
2. **Adjust chunk size**: Larger chunks (1000-1500 chars) for better context, smaller (500-800) for precision
3. **Use multiple workers**: Set `max_workers` to 4-8 for CPU-bound tasks
4. **Choose the right embedding model**:
   - `all-MiniLM-L6-v2`: Fast, good quality
   - `all-mpnet-base-v2`: Slower, higher quality
   - OpenAI models: Highest quality, API costs

## Troubleshooting

### Common Issues

1. **"Model not found" error**: Ensure the model name is correct and the model is available
2. **OpenAI API errors**: Check your API key and rate limits
3. **ChromaDB persistence issues**: Ensure the persist directory is writable
4. **Memory issues with large PDFs**: Reduce batch size or chunk size
5. **Slow processing**: Increase batch size or use more workers

### Debug Mode

Enable debug mode for detailed logging:

```python
config = Config()
config.debug = True
```

Or set environment variable:

```bash
DEBUG=true
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing
- [sentence-transformers](https://www.sbert.net/) for local embeddings
- [ChromaDB](https://www.trychroma.com/) for vector database
- [LangChain](https://langchain.com/) for text splitting
- [OpenAI](https://openai.com/) for embedding APIs

## Development and CI/CD

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-username/pdf-vector-system.git
cd pdf-vector-system

# Install development dependencies
uv sync --extra dev

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run quality checks
ruff check pdf_vector_system tests
mypy pdf_vector_system
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

### CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment:

- **Automated Testing**: Tests run on Python 3.9-3.12 across Linux, macOS, and Windows
- **Code Quality**: Automated linting, formatting, and type checking with ruff, black, isort, and mypy
- **Security Scanning**: Automated security checks with bandit and safety
- **Documentation**: Automatic documentation building and deployment with MkDocs
- **Package Publishing**: Automated PyPI publishing on release tags
- **Dependency Updates**: Weekly automated dependency updates

### Quality Assurance

- **Test Coverage**: >80% code coverage requirement
- **Type Safety**: Full type hints with mypy checking
- **Code Style**: Enforced with ruff, black, and isort
- **Security**: Regular security scanning with bandit and safety
- **Pre-commit Hooks**: Automated quality checks before commits

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/development/contributing.md) for details.

### Quick Contribution Steps

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and quality checks
5. Commit your changes (`git commit -m 'feat: add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [https://your-username.github.io/pdf-vector-system/](https://your-username.github.io/pdf-vector-system/)
- **Issues**: [GitHub Issues](https://github.com/your-username/pdf-vector-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/pdf-vector-system/discussions)
- **PyPI**: [https://pypi.org/project/pdf-vector-system/](https://pypi.org/project/pdf-vector-system/)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes.