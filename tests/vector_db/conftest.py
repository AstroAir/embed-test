"""Pytest configuration and fixtures for vector database tests."""

import tempfile
from pathlib import Path
from typing import Generator, List, Dict, Any
from unittest.mock import Mock, MagicMock
import pytest

from pdf_vector_system.config.settings import ChromaDBConfig
from pdf_vector_system.vector_db.models import (
    DocumentChunk, SearchResult, SearchQuery, CollectionInfo, DocumentInfo
)
from tests.mocks.chromadb_mocks import MockChromaDBClient, MockCollection


@pytest.fixture
def vector_db_temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory specifically for vector DB tests."""
    temp_path = Path(tempfile.mkdtemp(prefix="vector_db_test_"))
    try:
        yield temp_path
    finally:
        import shutil
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def chroma_config_test(vector_db_temp_dir: Path) -> ChromaDBConfig:
    """Create ChromaDB configuration for testing."""
    return ChromaDBConfig(
        persist_directory=vector_db_temp_dir / "test_chroma",
        collection_name="test_collection",
        max_results=10
    )


@pytest.fixture
def mock_chromadb_client() -> MockChromaDBClient:
    """Create a mock ChromaDB client for testing."""
    return MockChromaDBClient()


@pytest.fixture
def mock_collection() -> MockCollection:
    """Create a mock ChromaDB collection for testing."""
    return MockCollection(name="test_collection")


@pytest.fixture
def sample_document_chunks() -> List[DocumentChunk]:
    """Create sample DocumentChunk objects for testing."""
    return [
        DocumentChunk(
            id="doc1_chunk_0",
            content="This is the first chunk of the first document.",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            metadata={
                "document_id": "doc1",
                "chunk_index": 0,
                "page_number": 1,
                "start_char": 0,
                "end_char": 47
            }
        ),
        DocumentChunk(
            id="doc1_chunk_1",
            content="This is the second chunk of the first document.",
            embedding=[0.2, 0.3, 0.4, 0.5, 0.6],
            metadata={
                "document_id": "doc1",
                "chunk_index": 1,
                "page_number": 1,
                "start_char": 48,
                "end_char": 96
            }
        ),
        DocumentChunk(
            id="doc2_chunk_0",
            content="This is the first chunk of the second document.",
            embedding=[0.3, 0.4, 0.5, 0.6, 0.7],
            metadata={
                "document_id": "doc2",
                "chunk_index": 0,
                "page_number": 1,
                "start_char": 0,
                "end_char": 48
            }
        )
    ]


@pytest.fixture
def sample_search_results() -> List[SearchResult]:
    """Create sample SearchResult objects for testing."""
    return [
        SearchResult(
            id="doc1_chunk_0",
            content="This is the first search result.",
            score=0.95,
            metadata={
                "document_id": "doc1",
                "chunk_index": 0,
                "page_number": 1
            }
        ),
        SearchResult(
            id="doc1_chunk_1",
            content="This is the second search result.",
            score=0.87,
            metadata={
                "document_id": "doc1",
                "chunk_index": 1,
                "page_number": 1
            }
        ),
        SearchResult(
            id="doc2_chunk_0",
            content="This is the third search result.",
            score=0.82,
            metadata={
                "document_id": "doc2",
                "chunk_index": 0,
                "page_number": 1
            }
        )
    ]


@pytest.fixture
def sample_search_query() -> SearchQuery:
    """Create a sample SearchQuery for testing."""
    return SearchQuery(
        query_text="machine learning algorithms",
        max_results=5,
        min_score=0.7,
        filter_metadata={"document_id": "doc1"}
    )


@pytest.fixture
def sample_collection_info() -> CollectionInfo:
    """Create sample CollectionInfo for testing."""
    return CollectionInfo(
        name="test_collection",
        document_count=10,
        chunk_count=50,
        total_size_bytes=102400,
        created_at="2024-01-01T00:00:00",
        last_modified="2024-01-01T12:00:00",
        metadata={"created_by": "pdf_vector_system"}
    )


@pytest.fixture
def sample_document_info() -> DocumentInfo:
    """Create sample DocumentInfo for testing."""
    return DocumentInfo(
        document_id="test_doc_123",
        filename="test_document.pdf",
        chunk_count=5,
        total_characters=2500,
        page_count=3,
        file_size_bytes=51200,
        created_at="2024-01-01T00:00:00",
        metadata={
            "title": "Test Document",
            "author": "Test Author"
        }
    )


@pytest.fixture
def sample_embeddings_5d() -> List[List[float]]:
    """Provide sample 5D embeddings for testing."""
    return [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.3, 0.4, 0.5, 0.6, 0.7],
        [0.4, 0.5, 0.6, 0.7, 0.8],
        [0.5, 0.6, 0.7, 0.8, 0.9]
    ]


@pytest.fixture
def invalid_document_chunk() -> Dict[str, Any]:
    """Create invalid document chunk data for error testing."""
    return {
        "id": "",  # Invalid empty ID
        "content": "",  # Invalid empty content
        "embedding": [],  # Invalid empty embedding
        "metadata": {}
    }
