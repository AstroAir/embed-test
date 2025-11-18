<!-- markdownlint-disable MD046 -->

# VectorFlow

Streamlined PDF-to-vector processing pipeline with multi-provider embeddings and multi-backend vector storage.

!!! note "New to VectorFlow?"
    Start with the **Getting Started** section to install VectorFlow, configure your environment and run your first search.

## Overview

VectorFlow is a production-ready system for extracting text from PDF documents, generating embeddings using multiple AI providers, and storing vectors in scalable databases for semantic search and retrieval.

## Key Features

- **PDF Processing**: Robust text and metadata extraction from PDFs
- **Multiple Embedding Providers**: Sentence Transformers, OpenAI, Cohere, HuggingFace, Google Gemini, Google USE, Azure OpenAI
- **Vector Database Backends**: ChromaDB, Pinecone, Weaviate, Qdrant, Milvus (via a unified interface)
- **Flexible Configuration**: Pydantic-based configuration with environment variables and `.env` support
- **CLI Interface**: `vectorflow` command for batch processing, search, stats, health checks, etc.
- **GUI Application**: `vectorflow-gui` desktop app for interactive processing and search

## Quick Start

=== "Python"

    ```python
    from vectorflow import Config, PDFVectorPipeline
    from vectorflow.core.config.settings import EmbeddingModelType

    # Configure local Sentence Transformers embeddings
    config = Config()
    config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
    config.embedding.model_name = "all-MiniLM-L6-v2"

    pipeline = PDFVectorPipeline(config)

    # Process a PDF document
    result = pipeline.process_pdf("document.pdf")

    # Search for similar content
    results = pipeline.search("query text", n_results=5)
    ```

=== "CLI"

    ```bash
    vectorflow process docs/*.pdf --model all-MiniLM-L6-v2
    vectorflow search "query text" --results 5
    ```

## Installation

```bash
pip install vectorflow
```

## Documentation Structure

- **Getting Started**: Installation, configuration, and basic usage
- **User Guide**: CLI, GUI and Python API usage patterns
- **API Reference**: Core pipeline, configuration, embeddings, and vector DB APIs
- **Development**: Contributing, testing and release process

## License

This project is licensed under the MIT License.
