"""Pytest fixtures for CLI tests."""

import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from vectorflow.core.config.settings import Config
from vectorflow.core.pipeline import ProcessingResult
from vectorflow.core.vector_db.models import CollectionInfo, DocumentInfo, SearchResult


@pytest.fixture
def cli_runner() -> CliRunner:
    """Create a Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_pdf_file(temp_dir: Path) -> Path:
    """Create a temporary PDF file for testing."""
    pdf_path = temp_dir / "test_document.pdf"
    pdf_path.write_text("Mock PDF content for CLI testing")
    return pdf_path


@pytest.fixture
def multiple_pdf_files(temp_dir: Path) -> list[Path]:
    """Create multiple temporary PDF files for testing."""
    pdf_files = []
    for i in range(3):
        pdf_path = temp_dir / f"test_document_{i}.pdf"
        pdf_path.write_text(f"Mock PDF content {i}")
        pdf_files.append(pdf_path)
    return pdf_files


@pytest.fixture
def non_pdf_file(temp_dir: Path) -> Path:
    """Create a non-PDF file for testing."""
    txt_path = temp_dir / "test_document.txt"
    txt_path.write_text("This is not a PDF file")
    return txt_path


@pytest.fixture
def large_pdf_file(temp_dir: Path) -> Path:
    """Create a large PDF file for testing size warnings."""
    pdf_path = temp_dir / "large_document.pdf"
    # Create a file larger than 100MB (simulated)
    pdf_path.write_text("Large PDF content" * 1000)
    return pdf_path


@pytest.fixture
def mock_processing_result() -> ProcessingResult:
    """Create a mock successful processing result."""
    return ProcessingResult(
        document_id="test_doc",
        file_path="test_document.pdf",
        success=True,
        chunks_processed=10,
        embeddings_generated=10,
        chunks_stored=10,
        processing_time=2.5,
        error_message=None,
        metadata={
            "pdf_metadata": {"title": "Test Document"},
            "page_count": 5,
            "total_characters": 1000,
            "embedding_model": "test-model",
            "embedding_dimension": 384,
        },
    )


@pytest.fixture
def mock_failed_processing_result() -> ProcessingResult:
    """Create a mock failed processing result."""
    return ProcessingResult(
        document_id="test_doc",
        file_path="test_document.pdf",
        success=False,
        chunks_processed=0,
        embeddings_generated=0,
        chunks_stored=0,
        processing_time=0.5,
        error_message="Failed to process PDF",
        metadata=None,
    )


@pytest.fixture
def mock_search_results() -> list[SearchResult]:
    """Create mock search results."""
    return [
        SearchResult(
            id="chunk_1",
            content="This is the first search result with relevant content about machine learning.",
            score=0.95,
            metadata={
                "document_id": "test_doc",
                "page_number": 1,
                "chunk_index": 0,
            },
        ),
        SearchResult(
            id="chunk_2",
            content="This is the second search result with information about neural networks.",
            score=0.87,
            metadata={
                "document_id": "test_doc",
                "page_number": 2,
                "chunk_index": 1,
            },
        ),
        SearchResult(
            id="chunk_3",
            content="This is the third search result about deep learning algorithms.",
            score=0.72,
            metadata={
                "document_id": "another_doc",
                "page_number": 1,
                "chunk_index": 0,
            },
        ),
    ]


@pytest.fixture
def mock_collection_stats() -> dict:
    """Create mock collection statistics."""
    return {
        "total_chunks": 150,
        "unique_documents": 5,
        "total_characters": 50000,
        "average_chunk_size": 333.33,
        "sampled": False,
    }


@pytest.fixture
def mock_document_info() -> dict:
    """Create mock document information."""
    return {
        "document_id": "test_doc.pdf",
        "chunk_count": 10,
        "total_characters": 5000,
        "average_chunk_size": 500.0,
    }


@pytest.fixture
def mock_health_status() -> dict[str, bool]:
    """Create mock health check status."""
    return {
        "embedding_service": True,
        "vector_database": True,
        "pipeline": True,
    }


@pytest.fixture
def mock_collection_list() -> list[str]:
    """Create mock list of collection names."""
    return ["collection_1", "collection_2", "test_collection"]
