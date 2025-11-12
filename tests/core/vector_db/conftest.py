"""Pytest configuration and fixtures for vector database tests."""

import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from pdf_vector_system.core.vector_db.config import (
    ChromaDBConfig,
    MilvusConfig,
    PineconeConfig,
    QdrantConfig,
    WeaviateConfig,
)
from pdf_vector_system.core.vector_db.models import (
    CollectionInfo,
    DocumentChunk,
    DocumentInfo,
    SearchQuery,
    SearchResult,
)
from tests.mocks.chromadb_mocks import MockChromaDBClient, MockCollection


@pytest.fixture
def vector_db_temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory specifically for vector DB tests."""
    temp_path = Path(tempfile.mkdtemp(prefix="vector_db_test_"))
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def chroma_config_test(vector_db_temp_dir: Path) -> ChromaDBConfig:
    """Create ChromaDB configuration for testing."""
    return ChromaDBConfig(
        persist_directory=vector_db_temp_dir / "test_chroma",
        collection_name="test_collection",
        max_results=10,
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
def sample_document_chunks() -> list[DocumentChunk]:
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
                "end_char": 47,
            },
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
                "end_char": 96,
            },
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
                "end_char": 48,
            },
        ),
    ]


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    """Create sample SearchResult objects for testing."""
    return [
        SearchResult(
            id="doc1_chunk_0",
            content="This is the first search result.",
            score=0.95,
            metadata={"document_id": "doc1", "chunk_index": 0, "page_number": 1},
        ),
        SearchResult(
            id="doc1_chunk_1",
            content="This is the second search result.",
            score=0.87,
            metadata={"document_id": "doc1", "chunk_index": 1, "page_number": 1},
        ),
        SearchResult(
            id="doc2_chunk_0",
            content="This is the third search result.",
            score=0.82,
            metadata={"document_id": "doc2", "chunk_index": 0, "page_number": 1},
        ),
    ]


@pytest.fixture
def sample_search_query() -> SearchQuery:
    """Create a sample SearchQuery for testing."""
    return SearchQuery(
        query_text="machine learning algorithms",
        max_results=5,
        min_score=0.7,
        filter_metadata={"document_id": "doc1"},
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
        metadata={"created_by": "pdf_vector_system"},
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
        metadata={"title": "Test Document", "author": "Test Author"},
    )


@pytest.fixture
def sample_embeddings_5d() -> list[list[float]]:
    """Provide sample 5D embeddings for testing."""
    return [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.3, 0.4, 0.5, 0.6, 0.7],
        [0.4, 0.5, 0.6, 0.7, 0.8],
        [0.5, 0.6, 0.7, 0.8, 0.9],
    ]


@pytest.fixture
def invalid_document_chunk() -> dict[str, Any]:
    """Create invalid document chunk data for error testing."""
    return {
        "id": "",  # Invalid empty ID
        "content": "",  # Invalid empty content
        "embedding": [],  # Invalid empty embedding
        "metadata": {},
    }


# Additional Backend Configuration Fixtures


@pytest.fixture
def pinecone_config_test() -> PineconeConfig:
    """Create PineconeConfig for testing."""
    return PineconeConfig(
        api_key="test-api-key",
        environment="test-env",
        index_name="test-index",
        max_results=10,
    )


@pytest.fixture
def weaviate_config_test() -> WeaviateConfig:
    """Create WeaviateConfig for testing."""
    return WeaviateConfig(
        url="http://localhost:8080", class_name="TestDocument", max_results=10
    )


@pytest.fixture
def qdrant_config_test() -> QdrantConfig:
    """Create QdrantConfig for testing."""
    return QdrantConfig(
        url="http://localhost:6333", collection_name="test_collection", max_results=10
    )


@pytest.fixture
def milvus_config_test() -> MilvusConfig:
    """Create MilvusConfig for testing."""
    return MilvusConfig(
        host="localhost", port=19530, collection_name="test_collection", max_results=10
    )


# Mock Backend Clients


@pytest.fixture
def mock_pinecone_client():
    """Create a mock Pinecone client for testing."""
    mock_client = Mock()
    mock_index = Mock()

    # Configure mock index
    mock_index.describe_index_stats.return_value = {
        "total_vector_count": 100,
        "dimension": 384,
    }
    mock_index.upsert.return_value = {"upserted_count": 3}
    mock_index.query.return_value = {
        "matches": [
            {
                "id": "doc1_chunk_0",
                "score": 0.95,
                "metadata": {"content": "Test content 1", "document_id": "doc1"},
            },
            {
                "id": "doc1_chunk_1",
                "score": 0.87,
                "metadata": {"content": "Test content 2", "document_id": "doc1"},
            },
        ]
    }
    mock_index.delete.return_value = {}

    # Configure mock client
    mock_client.Index.return_value = mock_index
    mock_client.list_indexes.return_value = ["test-index"]
    mock_client.create_index.return_value = None
    mock_client.delete_index.return_value = None

    return mock_client


@pytest.fixture
def mock_weaviate_client():
    """Create a mock Weaviate client for testing."""
    mock_client = Mock()

    # Configure batch operations
    mock_batch = Mock()
    mock_batch.add_data_object.return_value = None
    mock_batch.create_objects.return_value = [{"result": {"status": "SUCCESS"}}]
    mock_client.batch = mock_batch

    # Configure query operations
    mock_query = Mock()
    mock_query.get.return_value.with_near_vector.return_value.with_limit.return_value.do.return_value = {
        "data": {
            "Get": {
                "TestDocument": [
                    {
                        "content": "Test content 1",
                        "_additional": {"id": "doc1_chunk_0", "distance": 0.05},
                    },
                    {
                        "content": "Test content 2",
                        "_additional": {"id": "doc1_chunk_1", "distance": 0.13},
                    },
                ]
            }
        }
    }
    mock_client.query = mock_query

    # Configure schema operations
    mock_schema = Mock()
    mock_schema.get.return_value = {"classes": [{"class": "TestDocument"}]}
    mock_schema.create_class.return_value = None
    mock_schema.delete_class.return_value = None
    mock_client.schema = mock_schema

    return mock_client


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client for testing."""
    mock_client = Mock()

    # Configure collection operations
    mock_client.get_collections.return_value.collections = [
        Mock(name="test_collection")
    ]
    mock_client.create_collection.return_value = None
    mock_client.delete_collection.return_value = None

    # Configure point operations
    mock_client.upsert.return_value = Mock(status="completed")
    mock_client.search.return_value = [
        Mock(
            id="doc1_chunk_0",
            score=0.95,
            payload={"content": "Test content 1", "document_id": "doc1"},
        ),
        Mock(
            id="doc1_chunk_1",
            score=0.87,
            payload={"content": "Test content 2", "document_id": "doc1"},
        ),
    ]
    mock_client.delete.return_value = None

    # Configure info operations
    mock_client.get_collection_info.return_value = Mock(
        vectors_count=100, indexed_vectors_count=100, points_count=100
    )

    return mock_client


@pytest.fixture
def mock_milvus_client():
    """Create a mock Milvus client for testing."""
    mock_client = Mock()

    # Configure collection operations
    mock_client.list_collections.return_value = ["test_collection"]
    mock_client.create_collection.return_value = None
    mock_client.drop_collection.return_value = None
    mock_client.has_collection.return_value = True

    # Configure data operations
    mock_client.insert.return_value = Mock(insert_count=3)
    mock_client.search.return_value = [
        [
            Mock(
                id="doc1_chunk_0",
                distance=0.05,
                entity={"content": "Test content 1", "document_id": "doc1"},
            ),
            Mock(
                id="doc1_chunk_1",
                distance=0.13,
                entity={"content": "Test content 2", "document_id": "doc1"},
            ),
        ]
    ]
    mock_client.delete.return_value = None

    # Configure collection info
    mock_client.get_collection_stats.return_value = {"row_count": 100}

    return mock_client


# Error Testing Fixtures


@pytest.fixture
def mock_connection_error():
    """Create a mock connection error for testing."""
    from pdf_vector_system.core.vector_db.models import ConnectionError

    return ConnectionError("Failed to connect to vector database")


@pytest.fixture
def mock_authentication_error():
    """Create a mock authentication error for testing."""
    from pdf_vector_system.core.vector_db.models import AuthenticationError

    return AuthenticationError("Invalid API key or credentials")


@pytest.fixture
def mock_configuration_error():
    """Create a mock configuration error for testing."""
    from pdf_vector_system.core.vector_db.models import ConfigurationError

    return ConfigurationError("Invalid configuration parameters")


@pytest.fixture
def mock_quota_exceeded_error():
    """Create a mock quota exceeded error for testing."""
    from pdf_vector_system.core.vector_db.models import QuotaExceededError

    return QuotaExceededError("API quota exceeded")


# Edge Case Testing Data


@pytest.fixture
def large_document_chunks(sample_embeddings_5d) -> list[DocumentChunk]:
    """Create a large number of document chunks for stress testing."""
    chunks = []
    for i in range(100):
        chunk = DocumentChunk(
            id=f"large_doc_chunk_{i}",
            content=f"This is chunk {i} with substantial content " * 10,
            embedding=sample_embeddings_5d[i % len(sample_embeddings_5d)],
            metadata={
                "document_id": f"large_doc_{i // 10}",
                "chunk_index": i % 10,
                "page_number": (i // 10) + 1,
                "start_char": i * 100,
                "end_char": (i + 1) * 100,
            },
        )
        chunks.append(chunk)
    return chunks


@pytest.fixture
def empty_search_results() -> list[SearchResult]:
    """Create empty search results for testing."""
    return []


@pytest.fixture
def malformed_embeddings() -> list[list[float]]:
    """Create malformed embeddings for error testing."""
    return [
        [],  # Empty embedding
        [0.1],  # Too short
        [float("inf"), 0.2, 0.3],  # Contains infinity
        [float("nan"), 0.2, 0.3],  # Contains NaN
        [0.1, 0.2, None, 0.4],  # Contains None
    ]


@pytest.fixture
def invalid_search_queries() -> list[dict[str, Any]]:
    """Create invalid search query data for error testing."""
    return [
        {"query_text": "", "n_results": 10},  # Empty query
        {"query_text": "test", "n_results": 0},  # Zero results
        {"query_text": "test", "n_results": -1},  # Negative results
        {"query_text": "test", "n_results": 10001},  # Too many results
        {"query_text": None, "n_results": 10},  # None query
    ]


# Backend Availability Testing


@pytest.fixture
def mock_backend_unavailable():
    """Mock backend unavailable scenario."""

    def side_effect(*_args, **_kwargs):
        from pdf_vector_system.core.vector_db.models import BackendNotAvailableError

        raise BackendNotAvailableError("Backend service is unavailable")

    return side_effect


@pytest.fixture
def mock_import_error():
    """Mock import error for missing dependencies."""

    def side_effect(*_args, **_kwargs):
        raise ImportError("Required package not installed")

    return side_effect


# Performance Testing Fixtures


@pytest.fixture
def performance_timer():
    """Create a performance timer for testing."""
    from pdf_vector_system.core.utils.progress import PerformanceTimer

    return PerformanceTimer()


@pytest.fixture
def timeout_config():
    """Create configuration with timeout settings for testing."""
    return {
        "connection_timeout": 5.0,
        "read_timeout": 10.0,
        "health_check_timeout": 3.0,
    }
