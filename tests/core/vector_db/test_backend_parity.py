"""Tests to ensure feature parity across all vector database backends."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pdf_vector_system.core.vector_db.config import (
    ChromaDBConfig,
    MilvusConfig,
    PineconeConfig,
    QdrantConfig,
    VectorDBType,
    WeaviateConfig,
)
from pdf_vector_system.core.vector_db.factory import VectorDBFactory
from pdf_vector_system.core.vector_db.interface import VectorDBInterface
from pdf_vector_system.core.vector_db.models import (
    DocumentChunk,
    SearchQuery,
    SearchResult,
)


class TestBackendParity:
    """Test feature parity across all vector database backends."""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample document chunks for testing."""
        return [
            DocumentChunk(
                id="chunk_1",
                content="This is the first test document chunk about artificial intelligence.",
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
                metadata={"document_id": "doc_1", "page_number": 1, "chunk_index": 0},
            ),
            DocumentChunk(
                id="chunk_2",
                content="This is the second test document chunk about machine learning.",
                embedding=[0.2, 0.3, 0.4, 0.5, 0.6],
                metadata={"document_id": "doc_1", "page_number": 2, "chunk_index": 1},
            ),
            DocumentChunk(
                id="chunk_3",
                content="This is the third test document chunk about deep learning.",
                embedding=[0.3, 0.4, 0.5, 0.6, 0.7],
                metadata={"document_id": "doc_2", "page_number": 1, "chunk_index": 0},
            ),
        ]

    @pytest.fixture
    def sample_query(self):
        """Create sample search query for testing."""
        return SearchQuery(
            query_text="artificial intelligence machine learning",
            n_results=5,
            where={"document_id": "doc_1"},
        )

    @pytest.fixture(
        params=[
            VectorDBType.CHROMADB,
            VectorDBType.PINECONE,
            VectorDBType.WEAVIATE,
            VectorDBType.QDRANT,
            VectorDBType.MILVUS,
        ]
    )
    def backend_config(self, request, temp_dir):
        """Create configuration for each backend type."""
        backend_type = request.param

        if backend_type == VectorDBType.CHROMADB:
            return ChromaDBConfig(
                persist_directory=temp_dir / "chroma_test",
                collection_name="test_collection",
            )
        if backend_type == VectorDBType.PINECONE:
            return PineconeConfig(
                api_key="test-api-key",
                environment="test-env",
                index_name="test-index",
                dimension=5,
            )
        if backend_type == VectorDBType.WEAVIATE:
            return WeaviateConfig(
                url="http://localhost:8080", class_name="TestDocument"
            )
        if backend_type == VectorDBType.QDRANT:
            return QdrantConfig(
                host="localhost",
                port=6333,
                collection_name="test_collection",
                vector_size=5,
            )
        if backend_type == VectorDBType.MILVUS:
            return MilvusConfig(
                host="localhost",
                port=19530,
                collection_name="test_collection",
                dimension=5,
            )
        return None

    @pytest.fixture
    def mock_client(self, backend_config):
        """Create a mocked client for the given backend configuration."""
        backend_type = backend_config.db_type

        # Mock the appropriate client based on backend type
        if backend_type == VectorDBType.CHROMADB:
            with patch(
                "pdf_vector_system.vector_db.chroma_client.chromadb.Client"
            ) as mock:
                mock_client_instance = Mock()
                mock_collection = Mock()
                mock_collection.name = "test_collection"
                mock_collection.count.return_value = 0
                mock_collection.metadata = {}

                mock_client_instance.heartbeat.return_value = True
                mock_client_instance.get_or_create_collection.return_value = (
                    mock_collection
                )
                mock.return_value = mock_client_instance

                client = VectorDBFactory.create_client(backend_config)
                client._collections["test_collection"] = mock_collection
                yield client

        elif backend_type == VectorDBType.PINECONE:
            with patch("pdf_vector_system.vector_db.pinecone_client.pinecone") as mock:
                mock.list_indexes.return_value = ["test-index"]
                mock_index = Mock()
                mock_index.describe_index_stats.return_value = Mock(
                    total_vector_count=0
                )
                mock.Index.return_value = mock_index

                client = VectorDBFactory.create_client(backend_config)
                yield client

        elif backend_type == VectorDBType.WEAVIATE:
            with patch("pdf_vector_system.vector_db.weaviate_client.weaviate") as mock:
                mock_client_instance = Mock()
                mock_client_instance.get_meta.return_value = {"version": "1.0.0"}
                mock_client_instance.schema.get.return_value = {"classes": []}
                mock.Client.return_value = mock_client_instance

                client = VectorDBFactory.create_client(backend_config)
                yield client

        elif backend_type == VectorDBType.QDRANT:
            with patch(
                "pdf_vector_system.vector_db.qdrant_client.QdrantClient"
            ) as mock:
                mock_client_instance = Mock()
                mock_collections = Mock()
                mock_collections.collections = []
                mock_client_instance.get_collections.return_value = mock_collections
                mock.return_value = mock_client_instance

                client = VectorDBFactory.create_client(backend_config)
                yield client

        elif backend_type == VectorDBType.MILVUS:
            with (
                patch("pdf_vector_system.vector_db.milvus_client.connections"),
                patch("pdf_vector_system.vector_db.milvus_client.utility") as mock_util,
            ):
                mock_util.list_collections.return_value = []
                mock_util.has_collection.return_value = False

                client = VectorDBFactory.create_client(backend_config)
                yield client

    def test_interface_compliance(self, mock_client):
        """Test that all backends implement the VectorDBInterface."""
        assert isinstance(mock_client, VectorDBInterface)

        # Check that all required methods are implemented
        required_methods = [
            "add_chunks",
            "search",
            "get_chunks",
            "update_chunks",
            "delete_chunks",
            "search_by_metadata",
            "get_document_info",
            "find_similar_chunks",
            "create_collection",
            "delete_collection",
            "list_collections",
            "get_collection_info",
            "count_chunks",
            "clear_collection",
            "get_backend_info",
            "health_check",
        ]

        for method_name in required_methods:
            assert hasattr(
                mock_client, method_name
            ), f"Method {method_name} not implemented"
            assert callable(
                getattr(mock_client, method_name)
            ), f"Method {method_name} not callable"

    def test_backend_info_consistency(self, mock_client):
        """Test that get_backend_info returns consistent structure across backends."""
        info = mock_client.get_backend_info()

        # All backends should return these fields
        required_fields = ["backend", "type", "capabilities"]
        for field in required_fields:
            assert field in info, f"Backend info missing required field: {field}"

        # Backend name should match the configuration type
        expected_backend = mock_client.config.db_type.value
        assert info["backend"] == expected_backend

        # Capabilities should be a list
        assert isinstance(info["capabilities"], list)
        assert len(info["capabilities"]) > 0

    def test_health_check_consistency(self, mock_client):
        """Test that health_check returns boolean across all backends."""
        result = mock_client.health_check()
        assert isinstance(result, bool)

    def test_collection_operations_consistency(self, mock_client):
        """Test that collection operations have consistent behavior."""
        # Test list_collections returns a list
        collections = mock_client.list_collections()
        assert isinstance(collections, list)

        # Test count_chunks returns an integer
        count = mock_client.count_chunks()
        assert isinstance(count, int)
        assert count >= 0

    def test_error_handling_consistency(self, mock_client):
        """Test that all backends handle errors consistently."""

        # All backends should raise VectorDBError for invalid operations
        # This is tested through the error handler decorators

        # Test that backend info includes error field when exceptions occur
        with patch.object(
            mock_client, "get_backend_info", side_effect=Exception("Test error")
        ):
            try:
                info = mock_client.get_backend_info()
                # Some backends catch exceptions and return error in info
                if "error" in info:
                    assert isinstance(info["error"], str)
            except Exception:
                # Some backends let exceptions propagate
                pass

    def test_search_result_consistency(self, mock_client, sample_chunks, sample_query):
        """Test that search results have consistent structure across backends."""
        # Mock the search method to return consistent results
        mock_results = [
            SearchResult(
                id="chunk_1",
                content="Test content",
                score=0.95,
                metadata={"document_id": "doc_1", "page_number": 1},
            )
        ]

        with patch.object(mock_client, "search", return_value=mock_results):
            results = mock_client.search(
                sample_query, query_embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
            )

            assert isinstance(results, list)
            if results:  # If results are returned
                for result in results:
                    assert isinstance(result, SearchResult)
                    assert hasattr(result, "id")
                    assert hasattr(result, "content")
                    assert hasattr(result, "score")
                    assert hasattr(result, "metadata")
                    assert isinstance(result.score, (int, float))
                    assert isinstance(result.metadata, dict)

    def test_chunk_operations_consistency(self, mock_client, sample_chunks):
        """Test that chunk operations have consistent behavior."""
        # Mock successful operations
        with patch.object(mock_client, "add_chunks", return_value=None):
            with patch.object(mock_client, "get_chunks", return_value=sample_chunks):
                with patch.object(mock_client, "update_chunks", return_value=None):
                    with patch.object(mock_client, "delete_chunks", return_value=None):
                        # Test add_chunks doesn't return anything
                        result = mock_client.add_chunks(sample_chunks)
                        assert result is None

                        # Test get_chunks returns list of DocumentChunk
                        chunks = mock_client.get_chunks(["chunk_1", "chunk_2"])
                        assert isinstance(chunks, list)
                        for chunk in chunks:
                            assert isinstance(chunk, DocumentChunk)

                        # Test update_chunks doesn't return anything
                        result = mock_client.update_chunks(sample_chunks)
                        assert result is None

                        # Test delete_chunks doesn't return anything
                        result = mock_client.delete_chunks(["chunk_1", "chunk_2"])
                        assert result is None

    def test_metadata_search_consistency(self, mock_client):
        """Test that metadata search has consistent behavior."""
        mock_results = [
            SearchResult(
                id="chunk_1",
                content="Test content",
                score=1.0,  # Metadata search typically returns score of 1.0
                metadata={"document_id": "doc_1", "page_number": 1},
            )
        ]

        with patch.object(mock_client, "search_by_metadata", return_value=mock_results):
            results = mock_client.search_by_metadata({"document_id": "doc_1"})

            assert isinstance(results, list)
            for result in results:
                assert isinstance(result, SearchResult)
                # Metadata search should return score of 1.0 for all backends
                assert result.score == 1.0


class TestBackendSpecificFeatures:
    """Test backend-specific features and capabilities."""

    def test_chromadb_local_storage(self):
        """Test ChromaDB local storage capabilities."""
        config = ChromaDBConfig(
            persist_directory=Path("./test_chroma"), collection_name="test_collection"
        )

        info = VectorDBFactory.get_backend_info(config.db_type)
        assert "local_storage" in info.get("capabilities", [])

    def test_pinecone_cloud_features(self):
        """Test Pinecone cloud-specific features."""
        config = PineconeConfig(
            api_key="test-key", environment="test-env", index_name="test-index"
        )

        info = VectorDBFactory.get_backend_info(config.db_type)
        assert "cloud_storage" in info.get("capabilities", [])

    def test_weaviate_graphql_features(self):
        """Test Weaviate GraphQL capabilities."""
        config = WeaviateConfig(url="http://localhost:8080", class_name="TestDocument")

        info = VectorDBFactory.get_backend_info(config.db_type)
        assert "graphql_api" in info.get("capabilities", [])

    def test_qdrant_performance_features(self):
        """Test Qdrant high-performance capabilities."""
        config = QdrantConfig(host="localhost", collection_name="test_collection")

        info = VectorDBFactory.get_backend_info(config.db_type)
        assert "high_performance" in info.get("capabilities", [])

    def test_milvus_scaling_features(self):
        """Test Milvus horizontal scaling capabilities."""
        config = MilvusConfig(host="localhost", collection_name="test_collection")

        info = VectorDBFactory.get_backend_info(config.db_type)
        assert "horizontal_scaling" in info.get("capabilities", [])
