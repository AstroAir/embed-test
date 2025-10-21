"""Tests for WeaviateClient implementation."""

from unittest.mock import Mock, patch

import pytest

from pdf_vector_system.vector_db.models import (
    CollectionInfo,
    DocumentChunk,
    DocumentNotFoundError,
    SearchQuery,
    SearchResult,
    VectorDBError,
)
from pdf_vector_system.vector_db.weaviate_client import WeaviateClient


class TestWeaviateClient:
    """Test WeaviateClient class."""

    def test_initialization(self, weaviate_config_test):
        """Test WeaviateClient initialization."""
        client = WeaviateClient(weaviate_config_test)

        assert client.config == weaviate_config_test
        assert client._client is None  # Lazy initialization

    def test_lazy_client_initialization(self, weaviate_config_test):
        """Test lazy client initialization."""
        with patch(
            "pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"
        ) as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client

            client = WeaviateClient(weaviate_config_test)

            # Access client property to trigger initialization
            actual_client = client.client

            assert actual_client == mock_client
            mock_connect.assert_called_once()

    def test_create_collection_success(self, weaviate_config_test):
        """Test successful collection (class) creation."""
        with patch(
            "pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"
        ) as mock_connect:
            mock_client = Mock()
            Mock()
            mock_client.schema.get.return_value = {"classes": []}
            mock_client.schema.create_class.return_value = None
            mock_connect.return_value = mock_client

            client = WeaviateClient(weaviate_config_test)
            result = client.create_collection("TestClass")

            # Should return the collection object (mock client in this case)
            assert result is not None
            mock_client.schema.create_class.assert_called_once()

    def test_create_collection_with_properties(self, weaviate_config_test):
        """Test collection creation with custom properties."""
        with patch(
            "pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"
        ) as mock_connect:
            mock_client = Mock()
            mock_client.schema.get.return_value = {"classes": []}
            mock_client.schema.create_class.return_value = None
            mock_connect.return_value = mock_client

            client = WeaviateClient(weaviate_config_test)
            properties = [
                {"name": "content", "dataType": ["text"]},
                {"name": "document_id", "dataType": ["string"]},
            ]
            result = client.create_collection(
                "TestClass", metadata={"properties": properties}
            )

            assert result is not None
            mock_client.schema.create_class.assert_called_once()

    def test_delete_collection_success(self, weaviate_config_test):
        """Test successful collection (class) deletion."""
        with patch(
            "pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"
        ) as mock_connect:
            mock_client = Mock()
            mock_client.schema.get.return_value = {"classes": [{"class": "TestClass"}]}
            mock_client.schema.delete_class.return_value = None
            mock_connect.return_value = mock_client

            client = WeaviateClient(weaviate_config_test)
            result = client.delete_collection("TestClass")

            # delete_collection returns None now
            assert result is None
            mock_client.schema.delete_class.assert_called_once_with("TestClass")

    def test_list_collections(self, weaviate_config_test):
        """Test listing collections (classes)."""
        with patch(
            "pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"
        ) as mock_connect:
            mock_client = Mock()
            mock_client.schema.get.return_value = {
                "classes": [
                    {"class": "Document", "vectorIndexConfig": {}},
                    {"class": "Article", "vectorIndexConfig": {}},
                    {"class": "Book", "vectorIndexConfig": {}},
                ]
            }
            mock_connect.return_value = mock_client

            client = WeaviateClient(weaviate_config_test)
            collections = client.list_collections()

            # Now returns List[CollectionInfo]
            assert len(collections) == 3
            assert collections[0].name == "Document"
            assert collections[1].name == "Article"
            assert collections[2].name == "Book"

    def test_collection_exists_true(self, weaviate_config_test):
        """Test collection_exists returns True for existing class."""
        with patch(
            "pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"
        ) as mock_connect:
            mock_client = Mock()
            mock_client.schema.get.return_value = {
                "classes": [{"class": "TestClass"}, {"class": "OtherClass"}]
            }
            mock_connect.return_value = mock_client

            client = WeaviateClient(weaviate_config_test)
            exists = client.collection_exists("TestClass")

            assert exists is True

    def test_collection_exists_false(self, weaviate_config_test):
        """Test collection_exists returns False for non-existing class."""
        with patch(
            "pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"
        ) as mock_connect:
            mock_client = Mock()
            mock_client.schema.get.return_value = {"classes": [{"class": "OtherClass"}]}
            mock_connect.return_value = mock_client

            client = WeaviateClient(weaviate_config_test)
            exists = client.collection_exists("TestClass")

            assert exists is False

    def test_add_chunks_success(self, weaviate_config_test, sample_document_chunks):
        """Test successful chunk addition."""
        with patch(
            "pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"
        ) as mock_connect:
            mock_client = Mock()
            # Set up v4 API mock
            mock_collection = Mock()
            mock_collection.data.insert_many.return_value = None
            mock_collections = Mock()
            mock_collections.get.return_value = mock_collection
            mock_client.collections = mock_collections
            mock_client.schema.get.return_value = {
                "classes": [{"class": "Testdocument"}]
            }
            mock_connect.return_value = mock_client

            client = WeaviateClient(weaviate_config_test)
            client.add_chunks(sample_document_chunks)

            mock_collection.data.insert_many.assert_called_once()

    def test_add_chunks_with_custom_class(
        self, weaviate_config_test, sample_document_chunks
    ):
        """Test chunk addition with custom class name."""
        with patch(
            "pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"
        ) as mock_connect:
            mock_client = Mock()
            # Set up v4 API mock
            mock_collection = Mock()
            mock_collection.data.insert_many.return_value = None
            mock_collections = Mock()
            mock_collections.get.return_value = mock_collection
            mock_client.collections = mock_collections
            mock_client.schema.get.return_value = {
                "classes": [{"class": "Customclass"}]
            }
            mock_connect.return_value = mock_client

            client = WeaviateClient(weaviate_config_test)
            client.add_chunks(sample_document_chunks, collection_name="CustomClass")

            mock_collection.data.insert_many.assert_called_once()
            # Verify that the correct collection was retrieved (CustomClass -> CustomClass)
            mock_collections.get.assert_called_once_with("CustomClass")

    def test_search_success(self, weaviate_config_test, sample_search_query):
        """Test successful search operation."""
        with patch(
            "pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"
        ) as mock_connect:
            mock_client = Mock()
            # Set up v4 API mock
            mock_collection = Mock()

            # Create mock objects that mimic v4 API response
            mock_obj1 = Mock()
            mock_obj1.uuid = "chunk_1"
            mock_obj1.properties = {
                "content": "Test content 1",
                "document_id": "doc1",
                "page_number": 1,
                "chunk_index": 0,
                "created_at": 1234567890,
            }
            mock_obj1.metadata = Mock()
            mock_obj1.metadata.certainty = 0.95

            mock_obj2 = Mock()
            mock_obj2.uuid = "chunk_2"
            mock_obj2.properties = {
                "content": "Test content 2",
                "document_id": "doc1",
                "page_number": 1,
                "chunk_index": 1,
                "created_at": 1234567891,
            }
            mock_obj2.metadata = Mock()
            mock_obj2.metadata.certainty = 0.87

            mock_result = Mock()
            mock_result.objects = [mock_obj1, mock_obj2]

            mock_collection.query.near_vector.return_value = mock_result
            mock_collections = Mock()
            mock_collections.get.return_value = mock_collection
            mock_client.collections = mock_collections

            mock_client.schema.get.return_value = {
                "classes": [{"class": "Testdocument"}]
            }
            mock_connect.return_value = mock_client

            client = WeaviateClient(weaviate_config_test)
            results = client.search(
                sample_search_query, query_embedding=[0.1, 0.2, 0.3]
            )

            assert len(results) == 2
            assert all(isinstance(result, SearchResult) for result in results)
            assert results[0].id == "chunk_1"
            assert results[0].content == "Test content 1"
            assert results[0].score == 0.95  # 1 - 0.05

    def test_search_with_where_filter(self, weaviate_config_test):
        """Test search with where filter."""
        query = SearchQuery(
            query_text="test query",
            n_results=5,
            where={"path": ["document_id"], "operator": "Equal", "valueString": "doc1"},
        )

        with patch(
            "pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"
        ) as mock_connect:
            mock_client = Mock()
            mock_query = Mock()
            mock_query.get.return_value.with_near_vector.return_value.with_where.return_value.with_limit.return_value.do.return_value = {
                "data": {"Get": {"TestDocument": []}}
            }
            mock_client.query = mock_query
            mock_connect.return_value = mock_client

            client = WeaviateClient(weaviate_config_test)
            client.search(query, query_embedding=[0.1, 0.2, 0.3])

            # Verify where filter was applied
            mock_query.get.return_value.with_near_vector.return_value.with_where.assert_called_once()

    def test_search_no_embedding_provided(
        self, weaviate_config_test, sample_search_query
    ):
        """Test search without providing query embedding."""
        with patch(
            "pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"
        ) as mock_connect:
            mock_client = Mock()
            mock_connect.return_value = mock_client

            client = WeaviateClient(weaviate_config_test)

            with pytest.raises(VectorDBError, match="query_embedding is required"):
                client.search(sample_search_query)

    def test_delete_chunks_success(self, weaviate_config_test):
        """Test successful chunk deletion."""
        with patch(
            "pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"
        ) as mock_connect:
            mock_client = Mock()
            mock_client.data_object.delete.return_value = None
            mock_connect.return_value = mock_client

            client = WeaviateClient(weaviate_config_test)
            chunk_ids = ["chunk_1", "chunk_2", "chunk_3"]
            client.delete_chunks(chunk_ids)

            # Should call delete for each chunk
            assert mock_client.data_object.delete.call_count == len(chunk_ids)

    def test_get_chunk_success(self, weaviate_config_test):
        """Test successful chunk retrieval."""
        with patch(
            "pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"
        ) as mock_connect:
            mock_client = Mock()
            mock_client.data_object.get_by_id.return_value = {
                "id": "chunk_1",
                "properties": {
                    "content": "Test content",
                    "document_id": "doc1",
                    "chunk_index": 0,
                },
                "vector": [0.1, 0.2, 0.3],
            }
            mock_connect.return_value = mock_client

            client = WeaviateClient(weaviate_config_test)
            chunk = client.get_chunk("chunk_1")

            assert isinstance(chunk, DocumentChunk)
            assert chunk.id == "chunk_1"
            assert chunk.content == "Test content"
            assert chunk.embedding == [0.1, 0.2, 0.3]

    def test_get_chunk_not_found(self, weaviate_config_test):
        """Test chunk retrieval when chunk not found."""
        with patch(
            "pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"
        ) as mock_connect:
            mock_client = Mock()
            mock_client.data_object.get_by_id.return_value = None
            mock_connect.return_value = mock_client

            client = WeaviateClient(weaviate_config_test)

            with pytest.raises(DocumentNotFoundError):
                client.get_chunk("nonexistent_chunk")

    def test_update_chunk_success(self, weaviate_config_test, sample_document_chunks):
        """Test successful chunk update."""
        with patch(
            "pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"
        ) as mock_connect:
            mock_client = Mock()
            mock_client.data_object.replace.return_value = None
            mock_connect.return_value = mock_client

            client = WeaviateClient(weaviate_config_test)
            chunk = sample_document_chunks[0]
            client.update_chunk(chunk)

            mock_client.data_object.replace.assert_called_once()
            call_args = mock_client.data_object.replace.call_args[1]
            assert call_args["uuid"] == chunk.id
            assert call_args["data_object"]["properties"]["content"] == chunk.content

    def test_get_collection_info(self, weaviate_config_test):
        """Test getting collection information."""
        with patch(
            "pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"
        ) as mock_connect:
            mock_client = Mock()
            mock_client.query.aggregate.return_value.with_meta_count.return_value.do.return_value = {
                "data": {"Aggregate": {"TestDocument": [{"meta": {"count": 500}}]}}
            }
            mock_connect.return_value = mock_client

            client = WeaviateClient(weaviate_config_test)
            info = client.get_collection_info()

            assert isinstance(info, CollectionInfo)
            assert info.name == weaviate_config_test.class_name
            assert info.chunk_count == 500

    def test_get_document_info_not_supported(self, weaviate_config_test):
        """Test that get_document_info raises NotImplementedError."""
        with patch(
            "pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"
        ):
            client = WeaviateClient(weaviate_config_test)

            with pytest.raises(NotImplementedError):
                client.get_document_info("doc_1")

    def test_health_check_success(self, weaviate_config_test):
        """Test successful health check."""
        with patch(
            "pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"
        ) as mock_connect:
            mock_client = Mock()
            mock_client.is_ready.return_value = True
            mock_connect.return_value = mock_client

            client = WeaviateClient(weaviate_config_test)
            health = client.health_check()

            assert health is True

    def test_health_check_failure(self, weaviate_config_test):
        """Test health check failure."""
        with patch(
            "pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"
        ) as mock_connect:
            mock_client = Mock()
            mock_client.is_ready.return_value = False
            mock_connect.return_value = mock_client

            client = WeaviateClient(weaviate_config_test)
            health = client.health_check()

            assert health is False


class TestWeaviateClientErrorHandling:
    """Test error handling in WeaviateClient."""

    def test_connection_error(self, weaviate_config_test):
        """Test connection error handling."""
        with patch(
            "pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"
        ) as mock_client_class:
            mock_client_class.side_effect = Exception("Connection failed")

            client = WeaviateClient(weaviate_config_test)

            with pytest.raises(VectorDBError):
                _ = client.client  # Trigger client initialization

    def test_schema_error(self, weaviate_config_test):
        """Test schema operation error handling."""
        with patch(
            "pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"
        ) as mock_connect:
            mock_client = Mock()
            mock_client.schema.create_class.side_effect = Exception("Schema error")
            mock_connect.return_value = mock_client

            client = WeaviateClient(weaviate_config_test)

            with pytest.raises(VectorDBError):
                client.create_collection("TestClass")

    def test_batch_operation_error(self, weaviate_config_test, sample_document_chunks):
        """Test batch operation error handling."""
        with patch(
            "pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"
        ) as mock_connect:
            mock_client = Mock()
            mock_batch = Mock()
            mock_batch.create_objects.side_effect = Exception("Batch error")
            mock_client.batch = mock_batch
            mock_connect.return_value = mock_client

            client = WeaviateClient(weaviate_config_test)

            with pytest.raises(VectorDBError):
                client.add_chunks(sample_document_chunks)
