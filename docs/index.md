# PDF Vector System

A comprehensive PDF content processing and vector storage system with ChromaDB integration.

## Overview

The PDF Vector System is a powerful Python library designed to process PDF documents, extract meaningful content, generate embeddings, and store them in vector databases for efficient similarity search and retrieval.

## Key Features

- **PDF Processing**: Extract text and metadata from PDF documents
- **Multiple Embedding Providers**: Support for OpenAI, Sentence Transformers, HuggingFace, and more
- **Vector Database Integration**: ChromaDB, Pinecone, Weaviate, Qdrant, and Milvus support
- **Flexible Configuration**: Comprehensive configuration system with validation
- **CLI Interface**: Command-line tools for batch processing
- **GUI Application**: User-friendly graphical interface
- **High Performance**: Optimized for large-scale document processing

## Quick Start

```python
from pdf_vector_system import PDFVectorPipeline

# Initialize the pipeline
pipeline = PDFVectorPipeline()

# Process a PDF document
result = pipeline.process_pdf("document.pdf")

# Search for similar content
results = pipeline.search("query text", top_k=5)
```

## Installation

```bash
pip install pdf-vector-system
```

## Documentation Structure

- **Getting Started**: Installation, configuration, and basic usage
- **User Guide**: Detailed guides for CLI and Python API usage
- **API Reference**: Complete API documentation
- **Development**: Contributing guidelines and development setup

## License

This project is licensed under the MIT License.
