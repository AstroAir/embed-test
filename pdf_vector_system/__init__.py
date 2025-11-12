"""
PDF Vector System - A comprehensive PDF content processing and vector storage system.

This package provides functionality to:
- Extract text content from PDF files
- Process and chunk text for embedding generation
- Generate embeddings using various models (local and API-based)
- Store embeddings in ChromaDB vector database
- Perform similarity search and retrieval
"""

try:
    from pdf_vector_system._version import __version__
except ImportError:
    # fallback for development installations
    __version__ = "0.1.0-dev"

__author__ = "The Augster"

from pdf_vector_system.core.config import Config
from pdf_vector_system.core.pipeline import PDFVectorPipeline

__all__ = ["Config", "PDFVectorPipeline"]
