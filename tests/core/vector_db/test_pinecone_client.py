"""Tests for PineconeClient implementation."""

from unittest.mock import Mock, patch

import pytest

from vectorflow.core.vector_db.models import (
    CollectionInfo,
    SearchQuery,
    SearchResult,
    VectorDBError,
)
from vectorflow.core.vector_db.pinecone_client import PineconeClient


class TestPineconeClient:
    """Test PineconeClient class."""

    def test_initialization(self, pinecone_config_test):
        """Test PineconeClient initialization."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)

            assert client.config == pinecone_config_test
            assert client._client is None  # Lazy initialization
            assert client._index is None

    def test_lazy_client_initialization(self, pinecone_config_test):
        """Test lazy client initialization."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_index = Mock()
            mock_client.Index.return_value = mock_index
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)

            # Access client property to trigger initialization
            actual_client = client.client

            assert actual_client == mock_client
            mock_pinecone.assert_called_once_with(api_key=pinecone_config_test.api_key)

    def test_lazy_index_initialization(self, pinecone_config_test):
        """Test lazy index initialization."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_index = Mock()
            mock_client.Index.return_value = mock_index
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)

            # Access index property to trigger initialization
            actual_index = client.index

            assert actual_index == mock_index
            mock_client.Index.assert_called_once_with(pinecone_config_test.index_name)

    def test_create_collection_success(self, pinecone_config_test):
        """Test successful collection (index) creation."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_client.create_index.return_value = None
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)
            result = client.create_collection("test-index", dimension=384)

            assert result is True
            mock_client.create_index.assert_called_once()
            call_args = mock_client.create_index.call_args
            assert call_args[1]["name"] == "test-index"
            assert call_args[1]["dimension"] == 384

    def test_create_collection_already_exists(self, pinecone_config_test):
        """Test collection creation when index already exists."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_client.create_index.side_effect = Exception("Index already exists")
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)

            with pytest.raises(VectorDBError):
                client.create_collection("existing-index", dimension=384)

    def test_delete_collection_success(self, pinecone_config_test):
        """Test successful collection (index) deletion."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_client.delete_index.return_value = None
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)
            result = client.delete_collection("test-index")

            assert result is True
            mock_client.delete_index.assert_called_once_with("test-index")

    def test_list_collections(self, pinecone_config_test):
        """Test listing collections (indexes)."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_client.list_indexes.return_value = ["index1", "index2", "index3"]
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)
            collections = client.list_collections()

            assert collections == ["index1", "index2", "index3"]
            mock_client.list_indexes.assert_called_once()

    def test_collection_exists_true(self, pinecone_config_test):
        """Test collection_exists returns True for existing index."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_client.list_indexes.return_value = ["test-index", "other-index"]
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)
            exists = client.collection_exists("test-index")

            assert exists is True

    def test_collection_exists_false(self, pinecone_config_test):
        """Test collection_exists returns False for non-existing index."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_client.list_indexes.return_value = ["other-index"]
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)
            exists = client.collection_exists("test-index")

            assert exists is False

    def test_add_chunks_success(self, pinecone_config_test, sample_document_chunks):
        """Test successful chunk addition."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_index = Mock()
            mock_index.upsert.return_value = {
                "upserted_count": len(sample_document_chunks)
            }
            mock_client.Index.return_value = mock_index
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)
            client.add_chunks(sample_document_chunks)

            mock_index.upsert.assert_called_once()
            call_args = mock_index.upsert.call_args[1]
            assert "vectors" in call_args
            assert len(call_args["vectors"]) == len(sample_document_chunks)

    def test_add_chunks_with_namespace(
        self, pinecone_config_test, sample_document_chunks
    ):
        """Test chunk addition with namespace."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_index = Mock()
            mock_index.upsert.return_value = {
                "upserted_count": len(sample_document_chunks)
            }
            mock_client.Index.return_value = mock_index
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)
            client.add_chunks(sample_document_chunks, collection_name="test_namespace")

            mock_index.upsert.assert_called_once()
            call_args = mock_index.upsert.call_args[1]
            assert call_args["namespace"] == "test_namespace"

    def test_search_success(self, pinecone_config_test, sample_search_query):
        """Test successful search operation."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_index = Mock()
            mock_index.query.return_value = {
                "matches": [
                    {
                        "id": "chunk_1",
                        "score": 0.95,
                        "metadata": {
                            "content": "Test content 1",
                            "document_id": "doc1",
                        },
                    },
                    {
                        "id": "chunk_2",
                        "score": 0.87,
                        "metadata": {
                            "content": "Test content 2",
                            "document_id": "doc1",
                        },
                    },
                ]
            }
            mock_client.Index.return_value = mock_index
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)
            results = client.search(
                sample_search_query, query_embedding=[0.1, 0.2, 0.3]
            )

            assert len(results) == 2
            assert all(isinstance(result, SearchResult) for result in results)
            assert results[0].id == "chunk_1"
            assert results[0].score == 0.95
            assert results[0].content == "Test content 1"

    def test_search_with_filter(self, pinecone_config_test):
        """Test search with metadata filter."""
        query = SearchQuery(
            query_text="test query", n_results=5, where={"document_id": "doc1"}
        )

        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_index = Mock()
            mock_index.query.return_value = {"matches": []}
            mock_client.Index.return_value = mock_index
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)
            client.search(query, query_embedding=[0.1, 0.2, 0.3])

            mock_index.query.assert_called_once()
            call_args = mock_index.query.call_args[1]
            assert call_args["filter"] == {"document_id": "doc1"}

    def test_search_no_embedding_provided(
        self, pinecone_config_test, sample_search_query
    ):
        """Test search without providing query embedding."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_index = Mock()
            mock_pinecone.return_value = mock_client
            mock_client.Index.return_value = mock_index

            client = PineconeClient(pinecone_config_test)

            with pytest.raises(VectorDBError, match="query_embedding is required"):
                client.search(sample_search_query)

    def test_delete_chunks_success(self, pinecone_config_test):
        """Test successful chunk deletion."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_index = Mock()
            mock_index.delete.return_value = None
            mock_client.Index.return_value = mock_index
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)
            chunk_ids = ["chunk_1", "chunk_2", "chunk_3"]
            client.delete_chunks(chunk_ids)

            mock_index.delete.assert_called_once()
            call_args = mock_index.delete.call_args[1]
            assert call_args["ids"] == chunk_ids

    def test_delete_chunks_with_namespace(self, pinecone_config_test):
        """Test chunk deletion with namespace."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_index = Mock()
            mock_index.delete.return_value = None
            mock_client.Index.return_value = mock_index
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)
            chunk_ids = ["chunk_1", "chunk_2"]
            client.delete_chunks(chunk_ids, collection_name="test_namespace")

            mock_index.delete.assert_called_once()
            call_args = mock_index.delete.call_args[1]
            assert call_args["namespace"] == "test_namespace"

    def test_get_chunk_not_supported(self, pinecone_config_test):
        """Test that get_chunk raises NotImplementedError."""
        with patch("vectorflow.vector_db.pinecone_client.pinecone.Pinecone"):
            client = PineconeClient(pinecone_config_test)

            with pytest.raises(NotImplementedError):
                client.get_chunk("chunk_1")

    def test_update_chunk_success(self, pinecone_config_test, sample_document_chunks):
        """Test successful chunk update."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_index = Mock()
            mock_index.upsert.return_value = {"upserted_count": 1}
            mock_client.Index.return_value = mock_index
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)
            chunk = sample_document_chunks[0]
            client.update_chunk(chunk)

            mock_index.upsert.assert_called_once()
            call_args = mock_index.upsert.call_args[1]
            assert len(call_args["vectors"]) == 1
            assert call_args["vectors"][0]["id"] == chunk.id

    def test_get_collection_info(self, pinecone_config_test):
        """Test getting collection information."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_index = Mock()
            mock_index.describe_index_stats.return_value = {
                "total_vector_count": 1000,
                "dimension": 384,
                "index_fullness": 0.1,
            }
            mock_client.Index.return_value = mock_index
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)
            info = client.get_collection_info()

            assert isinstance(info, CollectionInfo)
            assert info.name == pinecone_config_test.index_name
            assert info.chunk_count == 1000
            assert "dimension" in info.metadata
            assert info.metadata["dimension"] == 384

    def test_get_document_info_not_supported(self, pinecone_config_test):
        """Test that get_document_info raises NotImplementedError."""
        with patch("vectorflow.vector_db.pinecone_client.pinecone.Pinecone"):
            client = PineconeClient(pinecone_config_test)

            with pytest.raises(NotImplementedError):
                client.get_document_info("doc_1")

    def test_health_check_success(self, pinecone_config_test):
        """Test successful health check."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_client.list_indexes.return_value = ["test-index"]
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)
            health = client.health_check()

            assert health is True

    def test_health_check_failure(self, pinecone_config_test):
        """Test health check failure."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_client.list_indexes.side_effect = Exception("API error")
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)
            health = client.health_check()

            assert health is False


class TestPineconeClientErrorHandling:
    """Test error handling in PineconeClient."""

    def test_authentication_error(self, pinecone_config_test):
        """Test authentication error handling."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_pinecone.side_effect = Exception("Invalid API key")

            client = PineconeClient(pinecone_config_test)

            with pytest.raises(VectorDBError):
                _ = client.client  # Trigger client initialization

    def test_connection_error(self, pinecone_config_test):
        """Test connection error handling."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_client.list_indexes.side_effect = Exception("Connection timeout")
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)

            with pytest.raises(VectorDBError):
                client.list_collections()

    def test_index_not_found_error(self, pinecone_config_test):
        """Test index not found error handling."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_client.Index.side_effect = Exception("Index not found")
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)

            with pytest.raises(VectorDBError):
                _ = client.index  # Trigger index initialization

    def test_quota_exceeded_error(self, pinecone_config_test, sample_document_chunks):
        """Test quota exceeded error handling."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_index = Mock()
            mock_index.upsert.side_effect = Exception("Quota exceeded")
            mock_client.Index.return_value = mock_index
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)

            with pytest.raises(VectorDBError):
                client.add_chunks(sample_document_chunks)


class TestPineconeClientIntegration:
    """Test PineconeClient integration scenarios."""

    def test_end_to_end_workflow(
        self, pinecone_config_test, sample_document_chunks, sample_search_query
    ):
        """Test end-to-end workflow with mocked Pinecone."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_index = Mock()

            # Setup mock responses
            mock_client.list_indexes.return_value = []
            mock_client.create_index.return_value = None
            mock_index.upsert.return_value = {
                "upserted_count": len(sample_document_chunks)
            }
            mock_index.query.return_value = {
                "matches": [
                    {
                        "id": sample_document_chunks[0].id,
                        "score": 0.95,
                        "metadata": {
                            "content": sample_document_chunks[0].content,
                            **sample_document_chunks[0].metadata,
                        },
                    }
                ]
            }
            mock_index.delete.return_value = None
            mock_client.delete_index.return_value = None

            mock_client.Index.return_value = mock_index
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)

            # Test workflow
            # 1. Create index
            assert client.create_collection("test-index", dimension=384) is True

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

            # 5. Delete index
            assert client.delete_collection("test-index") is True

    def test_batch_operations(self, pinecone_config_test, large_document_chunks):
        """Test batch operations with large datasets."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_index = Mock()
            mock_index.upsert.return_value = {
                "upserted_count": len(large_document_chunks)
            }
            mock_client.Index.return_value = mock_index
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)

            # Test adding large batch
            client.add_chunks(large_document_chunks)

            # Verify upsert was called with all chunks
            mock_index.upsert.assert_called_once()
            call_args = mock_index.upsert.call_args[1]
            assert len(call_args["vectors"]) == len(large_document_chunks)

    def test_namespace_isolation(self, pinecone_config_test, sample_document_chunks):
        """Test namespace isolation in operations."""
        with patch(
            "vectorflow.vector_db.pinecone_client.pinecone.Pinecone"
        ) as mock_pinecone:
            mock_client = Mock()
            mock_index = Mock()
            mock_index.upsert.return_value = {
                "upserted_count": len(sample_document_chunks)
            }
            mock_index.query.return_value = {"matches": []}
            mock_index.delete.return_value = None
            mock_client.Index.return_value = mock_index
            mock_pinecone.return_value = mock_client

            client = PineconeClient(pinecone_config_test)

            # Test operations with different namespaces
            client.add_chunks(sample_document_chunks, collection_name="namespace1")
            client.add_chunks(sample_document_chunks, collection_name="namespace2")

            # Verify both calls used different namespaces
            assert mock_index.upsert.call_count == 2
            call_args_1 = mock_index.upsert.call_args_list[0][1]
            call_args_2 = mock_index.upsert.call_args_list[1][1]
            assert call_args_1["namespace"] == "namespace1"
            assert call_args_2["namespace"] == "namespace2"
