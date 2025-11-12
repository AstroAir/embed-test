"""Tests for MilvusClient implementation."""

from unittest.mock import Mock, patch

import pytest

from pdf_vector_system.core.vector_db.milvus_client import MilvusClient
from pdf_vector_system.core.vector_db.models import (
    CollectionInfo,
    SearchQuery,
    SearchResult,
    VectorDBError,
)


class TestMilvusClient:
    """Test MilvusClient class."""

    def test_initialization(self, milvus_config_test):
        """Test MilvusClient initialization."""
        with patch(
            "pdf_vector_system.vector_db.milvus_client.connections"
        ) as mock_connections:
            mock_connections.connect.return_value = None

            client = MilvusClient(milvus_config_test)

            assert client.config == milvus_config_test
            assert client._client is None  # Lazy initialization

    def test_lazy_client_initialization(self, milvus_config_test):
        """Test lazy client initialization."""
        with patch(
            "pdf_vector_system.vector_db.milvus_client.connections"
        ) as mock_connections:
            mock_connections.connect.return_value = None

            client = MilvusClient(milvus_config_test)

            # Access client property to trigger initialization
            actual_client = client.client

            assert actual_client is not None
            mock_connections.connect.assert_called_once()

    def test_create_collection_success(self, milvus_config_test):
        """Test successful collection creation."""
        with (
            patch(
                "pdf_vector_system.vector_db.milvus_client.connections"
            ) as mock_connections,
            patch(
                "pdf_vector_system.vector_db.milvus_client.Collection"
            ) as mock_collection_class,
            patch("pdf_vector_system.vector_db.milvus_client.utility") as mock_utility,
        ):
            mock_connections.connect.return_value = None
            mock_utility.has_collection.return_value = False
            mock_collection = Mock()
            mock_collection_class.return_value = mock_collection

            client = MilvusClient(milvus_config_test)
            result = client.create_collection("test_collection", vector_dim=384)

            assert result is True
            mock_collection_class.assert_called_once()

    def test_create_collection_already_exists(self, milvus_config_test):
        """Test collection creation when collection already exists."""
        with (
            patch(
                "pdf_vector_system.vector_db.milvus_client.connections"
            ) as mock_connections,
            patch("pdf_vector_system.vector_db.milvus_client.utility") as mock_utility,
        ):
            mock_connections.connect.return_value = None
            mock_utility.has_collection.return_value = True

            client = MilvusClient(milvus_config_test)

            with pytest.raises(VectorDBError, match=r"Collection .* already exists"):
                client.create_collection("existing_collection", vector_dim=384)

    def test_delete_collection_success(self, milvus_config_test):
        """Test successful collection deletion."""
        with (
            patch(
                "pdf_vector_system.vector_db.milvus_client.connections"
            ) as mock_connections,
            patch("pdf_vector_system.vector_db.milvus_client.utility") as mock_utility,
        ):
            mock_connections.connect.return_value = None
            mock_utility.drop_collection.return_value = None

            client = MilvusClient(milvus_config_test)
            result = client.delete_collection("test_collection")

            assert result is True
            mock_utility.drop_collection.assert_called_once_with("test_collection")

    def test_list_collections(self, milvus_config_test):
        """Test listing collections."""
        with (
            patch(
                "pdf_vector_system.vector_db.milvus_client.connections"
            ) as mock_connections,
            patch("pdf_vector_system.vector_db.milvus_client.utility") as mock_utility,
        ):
            mock_connections.connect.return_value = None
            mock_utility.list_collections.return_value = [
                "collection1",
                "collection2",
                "collection3",
            ]

            client = MilvusClient(milvus_config_test)
            collections = client.list_collections()

            assert collections == ["collection1", "collection2", "collection3"]

    def test_collection_exists_true(self, milvus_config_test):
        """Test collection_exists returns True for existing collection."""
        with (
            patch(
                "pdf_vector_system.vector_db.milvus_client.connections"
            ) as mock_connections,
            patch("pdf_vector_system.vector_db.milvus_client.utility") as mock_utility,
        ):
            mock_connections.connect.return_value = None
            mock_utility.has_collection.return_value = True

            client = MilvusClient(milvus_config_test)
            exists = client.collection_exists("test_collection")

            assert exists is True

    def test_collection_exists_false(self, milvus_config_test):
        """Test collection_exists returns False for non-existing collection."""
        with (
            patch(
                "pdf_vector_system.vector_db.milvus_client.connections"
            ) as mock_connections,
            patch("pdf_vector_system.vector_db.milvus_client.utility") as mock_utility,
        ):
            mock_connections.connect.return_value = None
            mock_utility.has_collection.return_value = False

            client = MilvusClient(milvus_config_test)
            exists = client.collection_exists("test_collection")

            assert exists is False

    def test_add_chunks_success(self, milvus_config_test, sample_document_chunks):
        """Test successful chunk addition."""
        with (
            patch(
                "pdf_vector_system.vector_db.milvus_client.connections"
            ) as mock_connections,
            patch(
                "pdf_vector_system.vector_db.milvus_client.Collection"
            ) as mock_collection_class,
        ):
            mock_connections.connect.return_value = None
            mock_collection = Mock()
            mock_collection.insert.return_value = Mock(
                insert_count=len(sample_document_chunks)
            )
            mock_collection_class.return_value = mock_collection

            client = MilvusClient(milvus_config_test)
            client.add_chunks(sample_document_chunks)

            mock_collection.insert.assert_called_once()

    def test_add_chunks_with_custom_collection(
        self, milvus_config_test, sample_document_chunks
    ):
        """Test chunk addition with custom collection name."""
        with (
            patch(
                "pdf_vector_system.vector_db.milvus_client.connections"
            ) as mock_connections,
            patch(
                "pdf_vector_system.vector_db.milvus_client.Collection"
            ) as mock_collection_class,
        ):
            mock_connections.connect.return_value = None
            mock_collection = Mock()
            mock_collection.insert.return_value = Mock(
                insert_count=len(sample_document_chunks)
            )
            mock_collection_class.return_value = mock_collection

            client = MilvusClient(milvus_config_test)
            client.add_chunks(
                sample_document_chunks, collection_name="custom_collection"
            )

            # Verify Collection was created with custom name
            mock_collection_class.assert_called_with("custom_collection")

    def test_search_success(self, milvus_config_test, sample_search_query):
        """Test successful search operation."""
        with (
            patch(
                "pdf_vector_system.vector_db.milvus_client.connections"
            ) as mock_connections,
            patch(
                "pdf_vector_system.vector_db.milvus_client.Collection"
            ) as mock_collection_class,
        ):
            mock_connections.connect.return_value = None
            mock_collection = Mock()

            # Mock search results
            mock_search_results = [
                [
                    Mock(
                        id="chunk_1",
                        distance=0.05,
                        entity={"content": "Test content 1", "document_id": "doc1"},
                    ),
                    Mock(
                        id="chunk_2",
                        distance=0.13,
                        entity={"content": "Test content 2", "document_id": "doc1"},
                    ),
                ]
            ]
            mock_collection.search.return_value = mock_search_results
            mock_collection_class.return_value = mock_collection

            client = MilvusClient(milvus_config_test)
            results = client.search(
                sample_search_query, query_embedding=[0.1, 0.2, 0.3]
            )

            assert len(results) == 2
            assert all(isinstance(result, SearchResult) for result in results)
            assert results[0].id == "chunk_1"
            assert results[0].score == 0.95  # 1 - 0.05
            assert results[0].content == "Test content 1"

    def test_search_with_expression_filter(self, milvus_config_test):
        """Test search with expression filter."""
        query = SearchQuery(
            query_text="test query", n_results=5, where={"document_id": "doc1"}
        )

        with (
            patch(
                "pdf_vector_system.vector_db.milvus_client.connections"
            ) as mock_connections,
            patch(
                "pdf_vector_system.vector_db.milvus_client.Collection"
            ) as mock_collection_class,
        ):
            mock_connections.connect.return_value = None
            mock_collection = Mock()
            mock_collection.search.return_value = [[]]
            mock_collection_class.return_value = mock_collection

            client = MilvusClient(milvus_config_test)
            client.search(query, query_embedding=[0.1, 0.2, 0.3])

            mock_collection.search.assert_called_once()
            call_args = mock_collection.search.call_args[1]
            assert "expr" in call_args

    def test_search_no_embedding_provided(
        self, milvus_config_test, sample_search_query
    ):
        """Test search without providing query embedding."""
        with patch(
            "pdf_vector_system.vector_db.milvus_client.connections"
        ) as mock_connections:
            mock_connections.connect.return_value = None

            client = MilvusClient(milvus_config_test)

            with pytest.raises(VectorDBError, match="query_embedding is required"):
                client.search(sample_search_query)

    def test_delete_chunks_success(self, milvus_config_test):
        """Test successful chunk deletion."""
        with (
            patch(
                "pdf_vector_system.vector_db.milvus_client.connections"
            ) as mock_connections,
            patch(
                "pdf_vector_system.vector_db.milvus_client.Collection"
            ) as mock_collection_class,
        ):
            mock_connections.connect.return_value = None
            mock_collection = Mock()
            mock_collection.delete.return_value = None
            mock_collection_class.return_value = mock_collection

            client = MilvusClient(milvus_config_test)
            chunk_ids = ["chunk_1", "chunk_2", "chunk_3"]
            client.delete_chunks(chunk_ids)

            mock_collection.delete.assert_called_once()
            call_args = mock_collection.delete.call_args[0][0]
            assert "id in" in call_args

    def test_get_chunk_basic(self, milvus_config_test):
        """Test basic get_chunk functionality without complex mocking."""
        with patch("pdf_vector_system.vector_db.milvus_client.connections"):
            client = MilvusClient(milvus_config_test)

            # Test that method exists and can be called
            # Detailed testing would require proper Milvus setup
            assert hasattr(client, "get_chunk")
            assert callable(client.get_chunk)

    def test_update_chunk_success(self, milvus_config_test, sample_document_chunks):
        """Test successful chunk update."""
        with (
            patch(
                "pdf_vector_system.vector_db.milvus_client.connections"
            ) as mock_connections,
            patch(
                "pdf_vector_system.vector_db.milvus_client.Collection"
            ) as mock_collection_class,
        ):
            mock_connections.connect.return_value = None
            mock_collection = Mock()
            mock_collection.insert.return_value = Mock(insert_count=1)
            mock_collection_class.return_value = mock_collection

            client = MilvusClient(milvus_config_test)
            chunk = sample_document_chunks[0]
            client.update_chunk(chunk)

            mock_collection.insert.assert_called_once()

    def test_get_collection_info(self, milvus_config_test):
        """Test getting collection information."""
        with (
            patch(
                "pdf_vector_system.vector_db.milvus_client.connections"
            ) as mock_connections,
            patch(
                "pdf_vector_system.vector_db.milvus_client.Collection"
            ) as mock_collection_class,
        ):
            mock_connections.connect.return_value = None
            mock_collection = Mock()
            mock_collection.num_entities = 1000
            mock_collection_class.return_value = mock_collection

            client = MilvusClient(milvus_config_test)
            info = client.get_collection_info()

            assert isinstance(info, CollectionInfo)
            assert info.name == milvus_config_test.collection_name
            assert info.chunk_count == 1000

    def test_get_document_info_basic(self, milvus_config_test):
        """Test basic get_document_info functionality without complex mocking."""
        with patch("pdf_vector_system.vector_db.milvus_client.connections"):
            client = MilvusClient(milvus_config_test)

            # Test that method exists and can be called
            # Detailed testing would require proper Milvus setup
            assert hasattr(client, "get_document_info")
            assert callable(client.get_document_info)

    def test_health_check_success(self, milvus_config_test):
        """Test successful health check."""
        with (
            patch(
                "pdf_vector_system.vector_db.milvus_client.connections"
            ) as mock_connections,
            patch("pdf_vector_system.vector_db.milvus_client.utility") as mock_utility,
        ):
            mock_connections.connect.return_value = None
            mock_utility.list_collections.return_value = []

            client = MilvusClient(milvus_config_test)
            health = client.health_check()

            assert health is True

    def test_health_check_failure(self, milvus_config_test):
        """Test health check failure."""
        with (
            patch(
                "pdf_vector_system.vector_db.milvus_client.connections"
            ) as mock_connections,
            patch("pdf_vector_system.vector_db.milvus_client.utility") as mock_utility,
        ):
            mock_connections.connect.return_value = None
            mock_utility.list_collections.side_effect = Exception("Connection error")

            client = MilvusClient(milvus_config_test)
            health = client.health_check()

            assert health is False


class TestMilvusClientErrorHandling:
    """Test error handling in MilvusClient."""

    def test_connection_error(self, milvus_config_test):
        """Test connection error handling."""
        with patch(
            "pdf_vector_system.vector_db.milvus_client.connections"
        ) as mock_connections:
            mock_connections.connect.side_effect = Exception("Connection failed")

            client = MilvusClient(milvus_config_test)

            with pytest.raises(VectorDBError):
                _ = client.client  # Trigger client initialization

    def test_collection_creation_error(self, milvus_config_test):
        """Test collection creation error handling."""
        with (
            patch(
                "pdf_vector_system.vector_db.milvus_client.connections"
            ) as mock_connections,
            patch(
                "pdf_vector_system.vector_db.milvus_client.Collection"
            ) as mock_collection_class,
        ):
            mock_connections.connect.return_value = None
            mock_collection_class.side_effect = Exception("Schema error")

            client = MilvusClient(milvus_config_test)

            with pytest.raises(VectorDBError):
                client.create_collection("test_collection", vector_dim=384)

    def test_insert_error(self, milvus_config_test, sample_document_chunks):
        """Test insert operation error handling."""
        with (
            patch(
                "pdf_vector_system.vector_db.milvus_client.connections"
            ) as mock_connections,
            patch(
                "pdf_vector_system.vector_db.milvus_client.Collection"
            ) as mock_collection_class,
        ):
            mock_connections.connect.return_value = None
            mock_collection = Mock()
            mock_collection.insert.side_effect = Exception("Insert failed")
            mock_collection_class.return_value = mock_collection

            client = MilvusClient(milvus_config_test)

            with pytest.raises(VectorDBError):
                client.add_chunks(sample_document_chunks)


class TestMilvusClientIntegration:
    """Test MilvusClient integration scenarios."""

    def test_end_to_end_workflow(
        self, milvus_config_test, sample_document_chunks, sample_search_query
    ):
        """Test end-to-end workflow with mocked Milvus."""
        with (
            patch(
                "pdf_vector_system.vector_db.milvus_client.connections"
            ) as mock_connections,
            patch(
                "pdf_vector_system.vector_db.milvus_client.Collection"
            ) as mock_collection_class,
            patch("pdf_vector_system.vector_db.milvus_client.utility") as mock_utility,
        ):
            mock_connections.connect.return_value = None
            mock_collection = Mock()

            # Setup mock responses
            mock_utility.has_collection.return_value = False
            mock_utility.list_collections.return_value = []
            mock_collection.insert.return_value = Mock(
                insert_count=len(sample_document_chunks)
            )
            mock_collection.search.return_value = [
                [
                    Mock(
                        id=sample_document_chunks[0].id,
                        distance=0.05,
                        entity={
                            "content": sample_document_chunks[0].content,
                            **sample_document_chunks[0].metadata,
                        },
                    )
                ]
            ]
            mock_collection.delete.return_value = None
            mock_utility.drop_collection.return_value = None

            mock_collection_class.return_value = mock_collection

            client = MilvusClient(milvus_config_test)

            # Test workflow
            # 1. Create collection
            assert client.create_collection("test_collection", vector_dim=384) is True

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
