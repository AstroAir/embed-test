"""Tests for VectorDBInterface compliance across all backends."""

from unittest.mock import Mock, patch

import pytest

from vectorflow.core.vector_db.chroma_client import ChromaDBClient
from vectorflow.core.vector_db.config import ChromaDBConfig
from vectorflow.core.vector_db.factory import VectorDBFactory
from vectorflow.core.vector_db.interface import VectorDBInterface
from vectorflow.core.vector_db.models import (
    CollectionInfo,
    DocumentChunk,
    SearchQuery,
    SearchResult,
)


class TestVectorDBInterfaceCompliance:
    """Test that all vector database backends comply with the interface."""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample document chunks for testing."""
        return [
            DocumentChunk(
                id="chunk_1",
                content="This is the first test document chunk.",
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
                metadata={"document_id": "doc_1", "page_number": 1},
            ),
            DocumentChunk(
                id="chunk_2",
                content="This is the second test document chunk.",
                embedding=[0.2, 0.3, 0.4, 0.5, 0.6],
                metadata={"document_id": "doc_1", "page_number": 2},
            ),
        ]

    @pytest.fixture
    def sample_query(self):
        """Create sample search query for testing."""
        return SearchQuery(
            query_text="test document", n_results=5, where={"document_id": "doc_1"}
        )

    @pytest.fixture
    def chroma_client(self, temp_dir):
        """Create ChromaDB client for testing."""
        config = ChromaDBConfig(
            persist_directory=temp_dir / "test_chroma",
            collection_name="test_collection",
        )

        with patch(
            "vectorflow.vector_db.chroma_client.chromadb.Client"
        ) as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.name = "test_collection"
            mock_collection.count.return_value = 0
            mock_collection.metadata = {}

            mock_client.heartbeat.return_value = True
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            client = ChromaDBClient(config)
            client._collections["test_collection"] = mock_collection
            yield client

    def test_interface_implementation(self, chroma_client):
        """Test that ChromaDBClient implements VectorDBInterface."""
        assert isinstance(chroma_client, VectorDBInterface)

        # Check that all abstract methods are implemented
        interface_methods = [
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

        for method_name in interface_methods:
            assert hasattr(chroma_client, method_name)
            assert callable(getattr(chroma_client, method_name))

    def test_add_chunks(self, chroma_client, sample_chunks):
        """Test add_chunks method."""
        with patch.object(chroma_client, "get_collection") as mock_get_collection:
            mock_collection = Mock()
            mock_get_collection.return_value = mock_collection

            chroma_client.add_chunks(sample_chunks)

            mock_collection.add.assert_called_once()
            call_args = mock_collection.add.call_args
            assert "ids" in call_args.kwargs
            assert "documents" in call_args.kwargs
            assert "embeddings" in call_args.kwargs
            assert "metadatas" in call_args.kwargs

    def test_search(self, chroma_client, sample_query):
        """Test search method."""
        with patch.object(chroma_client, "get_collection") as mock_get_collection:
            mock_collection = Mock()
            mock_collection.name = "test_collection"
            mock_collection.query.return_value = {
                "ids": [["chunk_1", "chunk_2"]],
                "documents": [["doc1", "doc2"]],
                "distances": [[0.1, 0.2]],
                "metadatas": [[{"doc_id": "1"}, {"doc_id": "2"}]],
            }
            mock_get_collection.return_value = mock_collection

            results = chroma_client.search(sample_query)

            assert isinstance(results, list)
            assert len(results) == 2
            assert all(isinstance(r, SearchResult) for r in results)
            mock_collection.query.assert_called_once()

    def test_get_chunks(self, chroma_client):
        """Test get_chunks method."""
        chunk_ids = ["chunk_1", "chunk_2"]

        with patch.object(chroma_client, "get_collection") as mock_get_collection:
            mock_collection = Mock()
            mock_collection.get.return_value = {
                "ids": chunk_ids,
                "documents": ["doc1", "doc2"],
                "metadatas": [{"doc_id": "1"}, {"doc_id": "2"}],
                "embeddings": [[0.1, 0.2], [0.3, 0.4]],
            }
            mock_get_collection.return_value = mock_collection

            chunks = chroma_client.get_chunks(chunk_ids, include_embeddings=True)

            assert isinstance(chunks, list)
            assert len(chunks) == 2
            assert all(isinstance(c, DocumentChunk) for c in chunks)
            mock_collection.get.assert_called_once()

    def test_update_chunks(self, chroma_client, sample_chunks):
        """Test update_chunks method."""
        with patch.object(chroma_client, "get_collection") as mock_get_collection:
            mock_collection = Mock()
            mock_get_collection.return_value = mock_collection

            with patch.object(chroma_client, "add_chunks") as mock_add_chunks:
                chroma_client.update_chunks(sample_chunks)

                mock_collection.delete.assert_called_once()
                mock_add_chunks.assert_called_once_with(sample_chunks, None)

    def test_delete_chunks(self, chroma_client):
        """Test delete_chunks method."""
        chunk_ids = ["chunk_1", "chunk_2"]

        with patch.object(chroma_client, "get_collection") as mock_get_collection:
            mock_collection = Mock()
            mock_get_collection.return_value = mock_collection

            chroma_client.delete_chunks(chunk_ids)

            mock_collection.delete.assert_called_once_with(ids=chunk_ids)

    def test_search_by_metadata(self, chroma_client):
        """Test search_by_metadata method."""
        metadata_filter = {"document_id": "doc_1"}

        with patch.object(chroma_client, "get_collection") as mock_get_collection:
            mock_collection = Mock()
            mock_collection.get.return_value = {
                "ids": ["chunk_1"],
                "documents": ["doc1"],
                "metadatas": [{"document_id": "doc_1"}],
            }
            mock_get_collection.return_value = mock_collection

            results = chroma_client.search_by_metadata(metadata_filter)

            assert isinstance(results, list)
            assert len(results) == 1
            assert isinstance(results[0], SearchResult)
            mock_collection.get.assert_called_once()

    def test_get_backend_info(self, chroma_client):
        """Test get_backend_info method."""
        info = chroma_client.get_backend_info()

        assert isinstance(info, dict)
        assert info["backend"] == "chromadb"
        assert "type" in info
        assert "capabilities" in info

    def test_count_chunks(self, chroma_client):
        """Test count_chunks method."""
        with patch.object(chroma_client, "get_collection") as mock_get_collection:
            mock_collection = Mock()
            mock_collection.count.return_value = 42
            mock_get_collection.return_value = mock_collection

            count = chroma_client.count_chunks()

            assert count == 42
            mock_collection.count.assert_called_once()

    def test_clear_collection(self, chroma_client):
        """Test clear_collection method."""
        with patch.object(chroma_client, "get_collection") as mock_get_collection:
            mock_collection = Mock()
            mock_collection.get.return_value = {"ids": ["chunk_1", "chunk_2"]}
            mock_get_collection.return_value = mock_collection

            chroma_client.clear_collection()

            mock_collection.delete.assert_called_once_with(ids=["chunk_1", "chunk_2"])

    def test_health_check(self, chroma_client):
        """Test health_check method."""
        with patch.object(chroma_client, "client") as mock_client:
            mock_client.heartbeat.return_value = True

            result = chroma_client.health_check()

            assert result is True
            mock_client.heartbeat.assert_called_once()

    def test_get_collection_info(self, chroma_client):
        """Test get_collection_info method."""
        with patch.object(chroma_client, "get_collection") as mock_get_collection:
            mock_collection = Mock()
            mock_collection.name = "test_collection"
            mock_collection.count.return_value = 10
            mock_collection.metadata = {"created": "2024-01-01"}
            mock_get_collection.return_value = mock_collection

            info = chroma_client.get_collection_info()

            assert isinstance(info, CollectionInfo)
            assert info.name == "test_collection"
            assert info.count == 10
            assert info.metadata == {"created": "2024-01-01"}


class TestFactoryIntegration:
    """Test factory integration with interface compliance."""

    def test_factory_creates_interface_compliant_client(self, temp_dir):
        """Test that factory creates clients that implement the interface."""
        config = ChromaDBConfig(
            persist_directory=temp_dir / "factory_test",
            collection_name="test_collection",
        )

        with patch("vectorflow.vector_db.chroma_client.chromadb.Client"):
            client = VectorDBFactory.create_client(config)

            assert isinstance(client, VectorDBInterface)
            assert isinstance(client, ChromaDBClient)

    def test_factory_with_string_config(self):
        """Test factory with string-based configuration."""
        config_dict = {
            "db_type": "chromadb",
            "persist_directory": "./test_chroma",
            "collection_name": "test_collection",
        }

        with patch("vectorflow.vector_db.chroma_client.chromadb.Client"):
            client = VectorDBFactory.create_client(config_dict)

            assert isinstance(client, VectorDBInterface)
            assert isinstance(client, ChromaDBClient)
