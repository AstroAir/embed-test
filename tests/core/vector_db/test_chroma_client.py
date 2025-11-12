"""Tests for ChromaDBClient class."""

from unittest.mock import Mock, patch

import pytest

from pdf_vector_system.core.vector_db.chroma_client import ChromaDBClient
from pdf_vector_system.core.vector_db.config import ChromaDBConfig
from pdf_vector_system.core.vector_db.models import (
    CollectionInfo,
    CollectionNotFoundError,
    DocumentChunk,
    DocumentInfo,
    DocumentNotFoundError,
    SearchQuery,
    SearchResult,
    VectorDBError,
)


class TestChromaDBClient:
    """Test ChromaDBClient class."""

    def test_initialization(self, temp_dir):
        """Test ChromaDBClient initialization."""
        config = ChromaDBConfig(
            persist_directory=temp_dir / "chroma_test",
            collection_name="test_collection",
            max_results=50,
        )

        with patch(
            "pdf_vector_system.vector_db.chroma_client.chromadb.Client"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            client = ChromaDBClient(config)

            assert client.config == config
            # Client is created lazily, so check it's None initially
            assert client._client is None

    def test_initialization_creates_directory(self, temp_dir):
        """Test that initialization creates persist directory."""
        chroma_dir = temp_dir / "new_chroma_dir"
        config = ChromaDBConfig(persist_directory=chroma_dir)

        with patch(
            "pdf_vector_system.vector_db.chroma_client.chromadb.PersistentClient"
        ):
            ChromaDBClient(config)

            assert chroma_dir.exists()
            assert chroma_dir.is_dir()

    def test_create_collection_success(self):
        """Test successful collection creation."""
        config = ChromaDBConfig()

        with patch(
            "pdf_vector_system.vector_db.chroma_client.chromadb.PersistentClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_client.create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            client = ChromaDBClient(config)
            result = client.create_collection("test_collection")

            assert result is True
            mock_client.create_collection.assert_called_once_with(
                name="test_collection", metadata={"created_by": "pdf_vector_system"}
            )

    def test_create_collection_already_exists(self):
        """Test collection creation when collection already exists."""
        config = ChromaDBConfig()

        with patch(
            "pdf_vector_system.vector_db.chroma_client.chromadb.PersistentClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.create_collection.side_effect = Exception(
                "Collection already exists"
            )
            mock_client_class.return_value = mock_client

            client = ChromaDBClient(config)

            with pytest.raises(VectorDBError, match="Error creating collection"):
                client.create_collection("existing_collection")

    def test_get_collection_success(self):
        """Test successful collection retrieval."""
        config = ChromaDBConfig()

        with patch(
            "pdf_vector_system.vector_db.chroma_client.chromadb.PersistentClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_client.get_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            client = ChromaDBClient(config)
            collection = client._get_collection("test_collection")

            assert collection == mock_collection
            mock_client.get_collection.assert_called_once_with("test_collection")

    def test_get_collection_not_found(self):
        """Test collection retrieval when collection doesn't exist."""
        config = ChromaDBConfig()

        with patch(
            "pdf_vector_system.vector_db.chroma_client.chromadb.PersistentClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.get_collection.side_effect = Exception("Collection not found")
            mock_client_class.return_value = mock_client

            client = ChromaDBClient(config)

            with pytest.raises(CollectionNotFoundError):
                client._get_collection("nonexistent_collection")

    def test_add_documents_success(self):
        """Test successful document addition."""
        config = ChromaDBConfig(collection_name="test_collection")

        chunks = [
            DocumentChunk(
                id="chunk_1",
                content="Test content 1",
                embedding=[0.1, 0.2, 0.3],
                metadata={"document_id": "doc_1"},
            ),
            DocumentChunk(
                id="chunk_2",
                content="Test content 2",
                embedding=[0.4, 0.5, 0.6],
                metadata={"document_id": "doc_1"},
            ),
        ]

        with patch(
            "pdf_vector_system.vector_db.chroma_client.chromadb.PersistentClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_client.get_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            client = ChromaDBClient(config)
            result = client.add_documents(chunks)

            assert result is True
            mock_collection.add.assert_called_once()

            # Verify the add call arguments
            call_args = mock_collection.add.call_args[1]
            assert call_args["ids"] == ["chunk_1", "chunk_2"]
            assert call_args["documents"] == ["Test content 1", "Test content 2"]
            assert len(call_args["embeddings"]) == 2
            assert len(call_args["metadatas"]) == 2

    def test_add_documents_empty_list(self):
        """Test adding empty document list."""
        config = ChromaDBConfig()
        client = ChromaDBClient(config)

        result = client.add_documents([])

        assert result is True  # Should succeed with no operation

    def test_add_documents_error(self):
        """Test document addition with error."""
        config = ChromaDBConfig(collection_name="test_collection")
        chunks = [DocumentChunk("chunk_1", "content", [0.1, 0.2])]

        with patch(
            "pdf_vector_system.vector_db.chroma_client.chromadb.PersistentClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.add.side_effect = Exception("Database error")
            mock_client.get_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            client = ChromaDBClient(config)

            with pytest.raises(VectorDBError, match="Error adding documents"):
                client.add_documents(chunks)

    def test_search_documents_success(self):
        """Test successful document search."""
        config = ChromaDBConfig(collection_name="test_collection")
        query = SearchQuery(query_text="test query", n_results=5)

        # Mock search results
        mock_results = {
            "ids": [["chunk_1", "chunk_2"]],
            "documents": [["Content 1", "Content 2"]],
            "distances": [[0.1, 0.3]],
            "metadatas": [[{"document_id": "doc_1"}, {"document_id": "doc_2"}]],
        }

        with patch(
            "pdf_vector_system.vector_db.chroma_client.chromadb.PersistentClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.query.return_value = mock_results
            mock_client.get_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            # Mock embedding service
            with patch.object(ChromaDBClient, "_get_query_embedding") as mock_embed:
                mock_embed.return_value = [0.1, 0.2, 0.3]

                client = ChromaDBClient(config)
                results = client.search_documents(query)

                assert len(results) == 2
                assert all(isinstance(r, SearchResult) for r in results)
                assert results[0].id == "chunk_1"
                assert results[0].content == "Content 1"
                assert results[0].score == 0.9  # 1 - 0.1 distance
                assert results[1].id == "chunk_2"
                assert results[1].content == "Content 2"
                assert results[1].score == 0.7  # 1 - 0.3 distance

    def test_search_documents_empty_results(self):
        """Test search with no results."""
        config = ChromaDBConfig(collection_name="test_collection")
        query = SearchQuery(query_text="no matches", n_results=5)

        mock_results = {
            "ids": [[]],
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }

        with patch(
            "pdf_vector_system.vector_db.chroma_client.chromadb.PersistentClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.query.return_value = mock_results
            mock_client.get_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            with patch.object(ChromaDBClient, "_get_query_embedding") as mock_embed:
                mock_embed.return_value = [0.1, 0.2, 0.3]

                client = ChromaDBClient(config)
                results = client.search_documents(query)

                assert len(results) == 0

    def test_search_documents_error(self):
        """Test search with error."""
        config = ChromaDBConfig(collection_name="test_collection")
        query = SearchQuery(query_text="test query")

        with patch(
            "pdf_vector_system.vector_db.chroma_client.chromadb.PersistentClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.query.side_effect = Exception("Search error")
            mock_client.get_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            with patch.object(ChromaDBClient, "_get_query_embedding") as mock_embed:
                mock_embed.return_value = [0.1, 0.2, 0.3]

                client = ChromaDBClient(config)

                with pytest.raises(VectorDBError, match="Error searching documents"):
                    client.search_documents(query)

    def test_delete_documents_success(self):
        """Test successful document deletion."""
        config = ChromaDBConfig(collection_name="test_collection")

        with patch(
            "pdf_vector_system.vector_db.chroma_client.chromadb.PersistentClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_client.get_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            client = ChromaDBClient(config)
            result = client.delete_documents(["chunk_1", "chunk_2"])

            assert result is True
            mock_collection.delete.assert_called_once_with(ids=["chunk_1", "chunk_2"])

    def test_delete_documents_by_filter(self):
        """Test document deletion by filter."""
        config = ChromaDBConfig(collection_name="test_collection")

        with patch(
            "pdf_vector_system.vector_db.chroma_client.chromadb.PersistentClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_client.get_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            client = ChromaDBClient(config)
            result = client.delete_documents_by_filter({"document_id": "doc_1"})

            assert result is True
            mock_collection.delete.assert_called_once_with(
                where={"document_id": "doc_1"}
            )

    def test_get_collection_info_success(self):
        """Test successful collection info retrieval."""
        config = ChromaDBConfig(collection_name="test_collection")

        with patch(
            "pdf_vector_system.vector_db.chroma_client.chromadb.PersistentClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.name = "test_collection"
            mock_collection.count.return_value = 100
            mock_collection.metadata = {"created_by": "test"}
            mock_client.get_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            client = ChromaDBClient(config)
            info = client.get_collection_info()

            assert isinstance(info, CollectionInfo)
            assert info.name == "test_collection"
            assert info.count == 100
            assert info.metadata["created_by"] == "test"

    def test_get_document_info_success(self):
        """Test successful document info retrieval."""
        config = ChromaDBConfig(collection_name="test_collection")

        # Mock collection query results
        mock_results = {
            "ids": [["chunk_1", "chunk_2", "chunk_3"]],
            "metadatas": [
                [
                    {"document_id": "doc_1", "page_number": 1, "content_length": 100},
                    {"document_id": "doc_1", "page_number": 2, "content_length": 150},
                    {"document_id": "doc_1", "page_number": 3, "content_length": 200},
                ]
            ],
        }

        with patch(
            "pdf_vector_system.vector_db.chroma_client.chromadb.PersistentClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.get.return_value = mock_results
            mock_client.get_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            client = ChromaDBClient(config)
            info = client.get_document_info("doc_1")

            assert isinstance(info, DocumentInfo)
            assert info.document_id == "doc_1"
            assert info.chunk_count == 3
            assert info.total_characters == 450  # 100 + 150 + 200
            assert info.page_count == 3

    def test_get_document_info_not_found(self):
        """Test document info retrieval for non-existent document."""
        config = ChromaDBConfig(collection_name="test_collection")

        mock_results = {"ids": [[]], "metadatas": [[]]}

        with patch(
            "pdf_vector_system.vector_db.chroma_client.chromadb.PersistentClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.get.return_value = mock_results
            mock_client.get_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            client = ChromaDBClient(config)

            with pytest.raises(DocumentNotFoundError):
                client.get_document_info("nonexistent_doc")

    def test_list_collections(self):
        """Test listing collections."""
        config = ChromaDBConfig()

        mock_collections = [
            Mock(name="collection_1", metadata={"type": "test"}),
            Mock(name="collection_2", metadata={"type": "prod"}),
        ]

        with patch(
            "pdf_vector_system.vector_db.chroma_client.chromadb.PersistentClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.list_collections.return_value = mock_collections
            mock_client_class.return_value = mock_client

            client = ChromaDBClient(config)
            collections = client.list_collections()

            assert len(collections) == 2
            assert collections[0]["name"] == "collection_1"
            assert collections[0]["metadata"]["type"] == "test"
            assert collections[1]["name"] == "collection_2"
            assert collections[1]["metadata"]["type"] == "prod"

    def test_delete_collection_success(self):
        """Test successful collection deletion."""
        config = ChromaDBConfig()

        with patch(
            "pdf_vector_system.vector_db.chroma_client.chromadb.PersistentClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            client = ChromaDBClient(config)
            result = client.delete_collection("test_collection")

            assert result is True
            mock_client.delete_collection.assert_called_once_with("test_collection")

    def test_delete_collection_error(self):
        """Test collection deletion with error."""
        config = ChromaDBConfig()

        with patch(
            "pdf_vector_system.vector_db.chroma_client.chromadb.PersistentClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.delete_collection.side_effect = Exception("Delete error")
            mock_client_class.return_value = mock_client

            client = ChromaDBClient(config)

            with pytest.raises(VectorDBError, match="Error deleting collection"):
                client.delete_collection("test_collection")

    def test_health_check_success(self):
        """Test successful health check."""
        config = ChromaDBConfig()

        with patch(
            "pdf_vector_system.vector_db.chroma_client.chromadb.PersistentClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.heartbeat.return_value = 12345  # Timestamp
            mock_client_class.return_value = mock_client

            client = ChromaDBClient(config)
            result = client.health_check()

            assert result is True
            mock_client.heartbeat.assert_called_once()

    def test_health_check_failure(self):
        """Test health check failure."""
        config = ChromaDBConfig()

        with patch(
            "pdf_vector_system.vector_db.chroma_client.chromadb.PersistentClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.heartbeat.side_effect = Exception("Connection error")
            mock_client_class.return_value = mock_client

            client = ChromaDBClient(config)
            result = client.health_check()

            assert result is False
