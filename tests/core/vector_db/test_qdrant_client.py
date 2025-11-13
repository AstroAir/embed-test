"""Tests for QdrantClient implementation."""

from unittest.mock import Mock, patch

import pytest

from vectorflow.core.vector_db.models import (
    CollectionInfo,
    DocumentChunk,
    DocumentNotFoundError,
    SearchQuery,
    SearchResult,
    VectorDBError,
)
from vectorflow.core.vector_db.qdrant_client import QdrantClient


class TestQdrantClient:
    """Test QdrantClient class."""

    def test_initialization(self, qdrant_config_test):
        """Test QdrantClient initialization."""
        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            client = QdrantClient(qdrant_config_test)

            assert client.config == qdrant_config_test
            assert client._client is None  # Lazy initialization

    def test_lazy_client_initialization(self, qdrant_config_test):
        """Test lazy client initialization."""
        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            client = QdrantClient(qdrant_config_test)

            # Access client property to trigger initialization
            actual_client = client.client

            assert actual_client == mock_client
            mock_client_class.assert_called_once()

    def test_create_collection_success(self, qdrant_config_test):
        """Test successful collection creation."""
        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.create_collection.return_value = None
            mock_client_class.return_value = mock_client

            client = QdrantClient(qdrant_config_test)
            result = client.create_collection("test_collection", vector_size=384)

            assert result is True
            mock_client.create_collection.assert_called_once()
            call_args = mock_client.create_collection.call_args[1]
            assert call_args["collection_name"] == "test_collection"
            assert call_args["vectors_config"]["size"] == 384

    def test_create_collection_with_distance_metric(self, qdrant_config_test):
        """Test collection creation with custom distance metric."""
        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.create_collection.return_value = None
            mock_client_class.return_value = mock_client

            client = QdrantClient(qdrant_config_test)
            result = client.create_collection(
                "test_collection", vector_size=384, distance="Dot"
            )

            assert result is True
            call_args = mock_client.create_collection.call_args[1]
            assert call_args["vectors_config"]["distance"] == "Dot"

    def test_delete_collection_success(self, qdrant_config_test):
        """Test successful collection deletion."""
        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.delete_collection.return_value = None
            mock_client_class.return_value = mock_client

            client = QdrantClient(qdrant_config_test)
            result = client.delete_collection("test_collection")

            assert result is True
            mock_client.delete_collection.assert_called_once_with("test_collection")

    def test_list_collections(self, qdrant_config_test):
        """Test listing collections."""
        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client = Mock()
            mock_collections = [
                Mock(name="collection1"),
                Mock(name="collection2"),
                Mock(name="collection3"),
            ]
            mock_client.get_collections.return_value.collections = mock_collections
            mock_client_class.return_value = mock_client

            client = QdrantClient(qdrant_config_test)
            collections = client.list_collections()

            assert collections == ["collection1", "collection2", "collection3"]

    def test_collection_exists_true(self, qdrant_config_test):
        """Test collection_exists returns True for existing collection."""
        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client = Mock()
            mock_collections = [
                Mock(name="test_collection"),
                Mock(name="other_collection"),
            ]
            mock_client.get_collections.return_value.collections = mock_collections
            mock_client_class.return_value = mock_client

            client = QdrantClient(qdrant_config_test)
            exists = client.collection_exists("test_collection")

            assert exists is True

    def test_collection_exists_false(self, qdrant_config_test):
        """Test collection_exists returns False for non-existing collection."""
        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client = Mock()
            mock_collections = [Mock(name="other_collection")]
            mock_client.get_collections.return_value.collections = mock_collections
            mock_client_class.return_value = mock_client

            client = QdrantClient(qdrant_config_test)
            exists = client.collection_exists("test_collection")

            assert exists is False

    def test_add_chunks_success(self, qdrant_config_test, sample_document_chunks):
        """Test successful chunk addition."""
        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.upsert.return_value = Mock(status="completed")
            mock_client_class.return_value = mock_client

            client = QdrantClient(qdrant_config_test)
            client.add_chunks(sample_document_chunks)

            mock_client.upsert.assert_called_once()
            call_args = mock_client.upsert.call_args[1]
            assert call_args["collection_name"] == qdrant_config_test.collection_name
            assert "points" in call_args
            assert len(call_args["points"]) == len(sample_document_chunks)

    def test_add_chunks_with_custom_collection(
        self, qdrant_config_test, sample_document_chunks
    ):
        """Test chunk addition with custom collection name."""
        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.upsert.return_value = Mock(status="completed")
            mock_client_class.return_value = mock_client

            client = QdrantClient(qdrant_config_test)
            client.add_chunks(
                sample_document_chunks, collection_name="custom_collection"
            )

            call_args = mock_client.upsert.call_args[1]
            assert call_args["collection_name"] == "custom_collection"

    def test_search_success(self, qdrant_config_test, sample_search_query):
        """Test successful search operation."""
        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client = Mock()
            mock_search_results = [
                Mock(
                    id="chunk_1",
                    score=0.95,
                    payload={"content": "Test content 1", "document_id": "doc1"},
                ),
                Mock(
                    id="chunk_2",
                    score=0.87,
                    payload={"content": "Test content 2", "document_id": "doc1"},
                ),
            ]
            mock_client.search.return_value = mock_search_results
            mock_client_class.return_value = mock_client

            client = QdrantClient(qdrant_config_test)
            results = client.search(
                sample_search_query, query_embedding=[0.1, 0.2, 0.3]
            )

            assert len(results) == 2
            assert all(isinstance(result, SearchResult) for result in results)
            assert results[0].id == "chunk_1"
            assert results[0].score == 0.95
            assert results[0].content == "Test content 1"

    def test_search_with_filter(self, qdrant_config_test):
        """Test search with metadata filter."""
        query = SearchQuery(
            query_text="test query", n_results=5, where={"document_id": "doc1"}
        )

        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.search.return_value = []
            mock_client_class.return_value = mock_client

            client = QdrantClient(qdrant_config_test)
            client.search(query, query_embedding=[0.1, 0.2, 0.3])

            mock_client.search.assert_called_once()
            call_args = mock_client.search.call_args[1]
            assert "query_filter" in call_args

    def test_search_no_embedding_provided(
        self, qdrant_config_test, sample_search_query
    ):
        """Test search without providing query embedding."""
        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            client = QdrantClient(qdrant_config_test)

            with pytest.raises(VectorDBError, match="query_embedding is required"):
                client.search(sample_search_query)

    def test_delete_chunks_success(self, qdrant_config_test):
        """Test successful chunk deletion."""
        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.delete.return_value = None
            mock_client_class.return_value = mock_client

            client = QdrantClient(qdrant_config_test)
            chunk_ids = ["chunk_1", "chunk_2", "chunk_3"]
            client.delete_chunks(chunk_ids)

            mock_client.delete.assert_called_once()
            call_args = mock_client.delete.call_args[1]
            assert call_args["collection_name"] == qdrant_config_test.collection_name
            assert call_args["points_selector"]["points"] == chunk_ids

    def test_get_chunk_success(self, qdrant_config_test):
        """Test successful chunk retrieval."""
        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.retrieve.return_value = [
                Mock(
                    id="chunk_1",
                    vector=[0.1, 0.2, 0.3],
                    payload={"content": "Test content", "document_id": "doc1"},
                )
            ]
            mock_client_class.return_value = mock_client

            client = QdrantClient(qdrant_config_test)
            chunk = client.get_chunk("chunk_1")

            assert isinstance(chunk, DocumentChunk)
            assert chunk.id == "chunk_1"
            assert chunk.content == "Test content"
            assert chunk.embedding == [0.1, 0.2, 0.3]

    def test_get_chunk_not_found(self, qdrant_config_test):
        """Test chunk retrieval when chunk not found."""
        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.retrieve.return_value = []
            mock_client_class.return_value = mock_client

            client = QdrantClient(qdrant_config_test)

            with pytest.raises(DocumentNotFoundError):
                client.get_chunk("nonexistent_chunk")

    def test_update_chunk_success(self, qdrant_config_test, sample_document_chunks):
        """Test successful chunk update."""
        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.upsert.return_value = Mock(status="completed")
            mock_client_class.return_value = mock_client

            client = QdrantClient(qdrant_config_test)
            chunk = sample_document_chunks[0]
            client.update_chunk(chunk)

            mock_client.upsert.assert_called_once()
            call_args = mock_client.upsert.call_args[1]
            assert len(call_args["points"]) == 1
            assert call_args["points"][0]["id"] == chunk.id

    def test_get_collection_info(self, qdrant_config_test):
        """Test getting collection information."""
        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.get_collection_info.return_value = Mock(
                vectors_count=1000, indexed_vectors_count=1000, points_count=1000
            )
            mock_client_class.return_value = mock_client

            client = QdrantClient(qdrant_config_test)
            info = client.get_collection_info()

            assert isinstance(info, CollectionInfo)
            assert info.name == qdrant_config_test.collection_name
            assert info.chunk_count == 1000

    def test_get_document_info_not_supported(self, qdrant_config_test):
        """Test that get_document_info raises NotImplementedError."""
        with patch("vectorflow.vector_db.qdrant_client.QdrantClientLib"):
            client = QdrantClient(qdrant_config_test)

            with pytest.raises(NotImplementedError):
                client.get_document_info("doc_1")

    def test_health_check_success(self, qdrant_config_test):
        """Test successful health check."""
        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.get_collections.return_value.collections = []
            mock_client_class.return_value = mock_client

            client = QdrantClient(qdrant_config_test)
            health = client.health_check()

            assert health is True

    def test_health_check_failure(self, qdrant_config_test):
        """Test health check failure."""
        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.get_collections.side_effect = Exception("Connection error")
            mock_client_class.return_value = mock_client

            client = QdrantClient(qdrant_config_test)
            health = client.health_check()

            assert health is False


class TestQdrantClientErrorHandling:
    """Test error handling in QdrantClient."""

    def test_connection_error(self, qdrant_config_test):
        """Test connection error handling."""
        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client_class.side_effect = Exception("Connection failed")

            client = QdrantClient(qdrant_config_test)

            with pytest.raises(VectorDBError):
                _ = client.client  # Trigger client initialization

    def test_collection_not_found_error(self, qdrant_config_test):
        """Test collection not found error handling."""
        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.get_collection_info.side_effect = Exception(
                "Collection not found"
            )
            mock_client_class.return_value = mock_client

            client = QdrantClient(qdrant_config_test)

            with pytest.raises(VectorDBError):
                client.get_collection_info("nonexistent_collection")

    def test_upsert_error(self, qdrant_config_test, sample_document_chunks):
        """Test upsert operation error handling."""
        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.upsert.side_effect = Exception("Upsert failed")
            mock_client_class.return_value = mock_client

            client = QdrantClient(qdrant_config_test)

            with pytest.raises(VectorDBError):
                client.add_chunks(sample_document_chunks)


class TestQdrantClientIntegration:
    """Test QdrantClient integration scenarios."""

    def test_end_to_end_workflow(
        self, qdrant_config_test, sample_document_chunks, sample_search_query
    ):
        """Test end-to-end workflow with mocked Qdrant."""
        with patch(
            "vectorflow.vector_db.qdrant_client.QdrantClientLib"
        ) as mock_client_class:
            mock_client = Mock()

            # Setup mock responses
            mock_client.get_collections.return_value.collections = []
            mock_client.create_collection.return_value = None
            mock_client.upsert.return_value = Mock(status="completed")
            mock_client.search.return_value = [
                Mock(
                    id=sample_document_chunks[0].id,
                    score=0.95,
                    payload={
                        "content": sample_document_chunks[0].content,
                        **sample_document_chunks[0].metadata,
                    },
                )
            ]
            mock_client.delete.return_value = None
            mock_client.delete_collection.return_value = None

            mock_client_class.return_value = mock_client

            client = QdrantClient(qdrant_config_test)

            # Test workflow
            # 1. Create collection
            assert client.create_collection("test_collection", vector_size=384) is True

            # 2. Add chunks
            client.add_chunks(sample_document_chunks)

            # 3. Search
            results = client.search(
                sample_search_query, query_embedding=[0.1, 0.2, 0.3]
            )
            assert len(results) == 1
            assert isinstance(results[0], SearchResult)

            # 4. Delete chunks
            chunk_ids = [chunk.id for chunk in sample_document_chunks]
            client.delete_chunks(chunk_ids)

            # 5. Delete collection
            assert client.delete_collection("test_collection") is True
