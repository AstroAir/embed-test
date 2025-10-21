# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Package Management

- **Install dependencies**: `uv sync` (recommended) or `pip install -e .`
- **Install with dev dependencies**: `uv sync --extra dev`
- **Build package**: `./scripts/build.sh` or `hatch build`
- **Install locally**: `pip install dist/*.whl`

### Testing

- **Run all tests**: `pytest tests/`
- **Run with coverage**: `pytest --cov=pdf_vector_system --cov-report=term-missing`
- **Run specific test file**: `pytest tests/test_pipeline.py`
- **Run with parallel execution**: `pytest -n auto`
- **Run integration tests**: `pytest -m integration`
- **Run tests excluding external dependencies**: `pytest -m "not external"`

### Code Quality

- **Lint and format**: `ruff check pdf_vector_system tests` and `ruff format pdf_vector_system tests`
- **Type checking**: `mypy pdf_vector_system`
- **Security checks**: `bandit -r pdf_vector_system`
- **Dependency safety**: `safety check`
- **Pre-commit hooks**: `pre-commit run --all-files`

### Documentation

- **Build docs**: `mkdocs build`
- **Serve docs locally**: `mkdocs serve`
- **Generate API docs**: `mkdocs-gen-files`

### CLI Tools

- **Main CLI**: `pdf-vector --help`
- **GUI application**: `pdf-vector-gui`
- **Process PDF**: `pdf-vector process document.pdf`
- **Search**: `pdf-vector search "query text"`
- **Stats**: `pdf-vector stats`

## Architecture Overview

This is a comprehensive PDF vector processing system that extracts text from PDFs, generates embeddings, and stores them in vector databases for similarity search.

### Core Components

1. **PDF Processing Pipeline** (`pdf_vector_system.pipeline`)
   - Main orchestrator class `PDFVectorPipeline`
   - Coordinates PDF extraction → text processing → embedding → storage
   - Handles progress tracking and error recovery

2. **PDF Processing** (`pdf_vector_system.pdf`)
   - `processor.py`: Text extraction using PyMuPDF
   - `text_processor.py`: Text cleaning, normalization, and chunking with LangChain

3. **Embedding Services** (`pdf_vector_system.embeddings`)
   - Factory pattern for multiple providers (OpenAI, Sentence Transformers, Cohere, etc.)
   - `base.py`: Abstract base interface
   - `factory.py`: Service creation and batch processing
   - Individual service files for each provider

4. **Vector Database** (`pdf_vector_system.vector_db`)
   - Multi-backend support (ChromaDB, Milvus, Pinecone, Qdrant, Weaviate)
   - `factory.py`: Client creation using factory pattern
   - `models.py`: Data models for chunks, searches, results
   - Each backend has its own client implementation

5. **Configuration** (`pdf_vector_system.config`)
   - Pydantic-based type-safe configuration
   - Environment variable support
   - Settings for all components (embedding, processing, vector DB)

6. **GUI Application** (`pdf_vector_system.gui`)
   - PySide6-based desktop application
   - MVC architecture with controllers, widgets, and dialogs
   - Uses qfluentwidgets for modern styling

### Key Design Patterns

- **Factory Pattern**: Used extensively for embedding services and vector database clients
- **Configuration Objects**: Pydantic models for type-safe settings
- **Progress Tracking**: Rich progress bars and performance timing
- **Error Handling**: Comprehensive exception handling with logging
- **Batch Processing**: Efficient handling of multiple documents/texts

### Testing Strategy

- **Unit tests**: Individual component testing
- **Integration tests**: Cross-component interactions
- **External tests**: Tests requiring APIs/databases (marked with `@pytest.mark.external`)
- **GUI tests**: PySide6 application testing
- **Mocking**: Extensive use of mocks for external dependencies

## Development Notes

### Configuration

- Default configuration in `config/settings.py`
- Environment variables override defaults (use double underscores: `EMBEDDING__MODEL_NAME`)
- Supports both ChromaDB config (legacy) and new vector_db config structure

### Adding New Components

- Follow factory pattern for extensible services
- Use Pydantic models for configuration
- Implement proper error handling and logging
- Add comprehensive tests with mocks

### Vector Database Backends

- Each backend implements the same interface from `interface.py`
- Use `VectorDBFactory.create_client()` to instantiate
- Configuration handled through provider-specific config classes

### Performance Considerations

- Batch processing is preferred for embeddings
- Use `max_workers` configuration for parallel processing
- Chunk size affects both processing time and search quality
- Consider memory usage with large documents

### GUI Development

- Uses PySide6 with qfluentwidgets for modern UI
- Controllers handle business logic, widgets handle display
- Threading utilities for background processing
- Progress dialogs for long-running operations
