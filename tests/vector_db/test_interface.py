"""Tests for VectorDBInterface abstract interface."""

from abc import ABC
from typing import Optional
from unittest.mock import Mock

import pytest

from pdf_vector_system.vector_db.interface import VectorDBInterface
from pdf_vector_system.vector_db.models import (
    CollectionInfo,
    DocumentChunk,
    DocumentInfo,
    SearchQuery,
    SearchResult,
)


class TestVectorDBInterface:
    """Test VectorDBInterface abstract base class."""

    def test_is_abstract_base_class(self):
        """Test that VectorDBInterface is an abstract base class."""
        assert issubclass(VectorDBInterface, ABC)

        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            VectorDBInterface()

    def test_abstract_methods_exist(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = VectorDBInterface.__abstractmethods__

        expected_methods = {
            "__init__",
            "create_collection",
            "get_collection",
            "delete_collection",
            "list_collections",
            "collection_exists",
            "add_chunks",
            "get_chunks",
            "get_chunk",
            "update_chunks",
            "update_chunk",
            "delete_chunks",
            "delete_document",
            "search",
            "search_by_metadata",
            "find_similar_chunks",
            "get_collection_info",
            "get_document_info",
            "health_check",
            "get_backend_info",
            "count_chunks",
            "clear_collection",
        }

        assert abstract_methods == expected_methods

    def test_method_signatures(self):
        """Test that abstract methods have correct signatures."""
        # Test __init__ signature
        init_method = VectorDBInterface.__init__
        assert init_method is not None

        # Test other method signatures by checking they exist
        methods_to_check = [
            "create_collection",
            "delete_collection",
            "list_collections",
            "collection_exists",
            "add_chunks",
            "search",
            "delete_chunks",
            "get_chunk",
            "update_chunk",
            "get_collection_info",
            "get_document_info",
            "health_check",
        ]

        for method_name in methods_to_check:
            assert hasattr(VectorDBInterface, method_name)
            method = getattr(VectorDBInterface, method_name)
            assert callable(method)


class ConcreteVectorDBImplementation(VectorDBInterface):
    """Concrete implementation for testing interface contract."""

    def __init__(self, config):
        self.config = config
        self._collections = {}
        self._chunks = {}

    def create_collection(
        self,
        name: Optional[str] = None,
        metadata: Optional[dict] = None,
        get_or_create: bool = True,
    ):
        collection_name = name or "default"
        self._collections[collection_name] = {"created": True, "metadata": metadata}
        # Return a mock collection object (interface returns Any, but typically a collection object)
        return Mock(name=collection_name, metadata=metadata)

    def delete_collection(self, name: Optional[str] = None) -> None:
        collection_name = name or "default"
        if collection_name in self._collections:
            del self._collections[collection_name]

    def list_collections(self) -> list[CollectionInfo]:
        return [
            CollectionInfo(
                name=name,
                document_count=0,
                chunk_count=0,
                total_size_bytes=0,
                created_at="2024-01-01T00:00:00",
                last_modified="2024-01-01T00:00:00",
                metadata=data.get("metadata", {}),
            )
            for name, data in self._collections.items()
        ]

    def collection_exists(self, name: Optional[str] = None) -> bool:
        collection_name = name or "default"
        return collection_name in self._collections

    def add_chunks(
        self, chunks: list[DocumentChunk], collection_name: Optional[str] = None
    ) -> None:
        collection = collection_name or "default"
        if collection not in self._chunks:
            self._chunks[collection] = {}

        for chunk in chunks:
            self._chunks[collection][chunk.id] = chunk

    def search(
        self,
        query: SearchQuery,
        query_embedding: Optional[list[float]] = None,
        collection_name: Optional[str] = None,
    ) -> list[SearchResult]:
        # Simple mock search implementation
        return [
            SearchResult(
                id="test_chunk_1",
                content="Test content 1",
                score=0.95,
                metadata={"document_id": "doc1"},
            )
        ]

    def delete_chunks(
        self, chunk_ids: list[str], collection_name: Optional[str] = None
    ) -> None:
        collection = collection_name or "default"
        if collection in self._chunks:
            for chunk_id in chunk_ids:
                self._chunks[collection].pop(chunk_id, None)

    def get_chunk(
        self,
        chunk_id: str,
        collection_name: Optional[str] = None,
        include_embeddings: bool = False,
    ) -> DocumentChunk:
        collection = collection_name or "default"
        if collection in self._chunks and chunk_id in self._chunks[collection]:
            return self._chunks[collection][chunk_id]
        raise ValueError(f"Chunk {chunk_id} not found")

    def update_chunk(
        self, chunk: DocumentChunk, collection_name: Optional[str] = None
    ) -> None:
        collection = collection_name or "default"
        if collection not in self._chunks:
            self._chunks[collection] = {}
        self._chunks[collection][chunk.id] = chunk

    def get_collection_info(
        self, collection_name: Optional[str] = None
    ) -> CollectionInfo:
        return CollectionInfo(
            name=collection_name or "default",
            document_count=10,
            chunk_count=50,
            total_size_bytes=1024,
            created_at="2024-01-01T00:00:00",
            last_modified="2024-01-01T12:00:00",
            metadata={"test": True},
        )

    def get_document_info(
        self, document_id: str, collection_name: Optional[str] = None
    ) -> DocumentInfo:
        return DocumentInfo(
            document_id=document_id,
            filename="test.pdf",
            chunk_count=5,
            total_characters=1000,
            page_count=2,
            file_size_bytes=2048,
            created_at="2024-01-01T00:00:00",
            metadata={"test": True},
        )

    def health_check(self) -> bool:
        return True

    def get_collection(self, name: Optional[str] = None):
        """Get an existing collection."""
        collection_name = name or "default"
        if collection_name in self._collections:
            return self._collections[collection_name]
        raise ValueError(f"Collection {collection_name} not found")

    def get_chunks(
        self,
        chunk_ids: list[str],
        collection_name: Optional[str] = None,
        include_embeddings: bool = False,
    ) -> list[DocumentChunk]:
        """Retrieve multiple chunks by IDs."""
        collection = collection_name or "default"
        chunks = []
        if collection in self._chunks:
            for chunk_id in chunk_ids:
                if chunk_id in self._chunks[collection]:
                    chunks.append(self._chunks[collection][chunk_id])
        return chunks

    def update_chunks(
        self, chunks: list[DocumentChunk], collection_name: Optional[str] = None
    ) -> None:
        """Update multiple chunks."""
        for chunk in chunks:
            self.update_chunk(chunk, collection_name)

    def delete_document(
        self, document_id: str, collection_name: Optional[str] = None
    ) -> int:
        """Delete all chunks for a document."""
        collection = collection_name or "default"
        deleted_count = 0
        if collection in self._chunks:
            chunk_ids_to_delete = [
                chunk_id
                for chunk_id, chunk in self._chunks[collection].items()
                if chunk.metadata.get("document_id") == document_id
            ]
            for chunk_id in chunk_ids_to_delete:
                del self._chunks[collection][chunk_id]
                deleted_count += 1
        return deleted_count

    def search_by_metadata(
        self,
        metadata_filter: dict,
        collection_name: Optional[str] = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search by metadata only."""
        return [
            SearchResult(
                id="test_chunk_1",
                content="Test content 1",
                score=1.0,
                metadata=metadata_filter,
            )
        ]

    def find_similar_chunks(
        self,
        chunk_id: str,
        collection_name: Optional[str] = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Find similar chunks."""
        return [
            SearchResult(
                id="similar_chunk_1",
                content="Similar content",
                score=0.9,
                metadata={"document_id": "doc1"},
            )
        ]

    def get_backend_info(self) -> dict:
        """Get backend information."""
        return {
            "name": "TestBackend",
            "version": "1.0.0",
            "type": "test",
            "capabilities": ["search", "metadata_filter"],
        }

    def count_chunks(self, collection_name: Optional[str] = None) -> int:
        """Count chunks in collection."""
        collection = collection_name or "default"
        if collection in self._chunks:
            return len(self._chunks[collection])
        return 0

    def clear_collection(self, collection_name: Optional[str] = None) -> None:
        """Clear all chunks from collection."""
        collection = collection_name or "default"
        if collection in self._chunks:
            self._chunks[collection] = {}


@pytest.fixture()
def concrete_client():
    """Create a concrete implementation for testing."""
    config = Mock()
    return ConcreteVectorDBImplementation(config)


class TestConcreteImplementation:
    """Test concrete implementation of VectorDBInterface."""

    def test_can_instantiate_concrete_implementation(self, concrete_client):
        """Test that concrete implementation can be instantiated."""
        assert isinstance(concrete_client, VectorDBInterface)
        assert isinstance(concrete_client, ConcreteVectorDBImplementation)

    def test_collection_operations(self, concrete_client):
        """Test collection management operations."""
        # Test collection creation
        result = concrete_client.create_collection("test_collection")
        assert result is not None  # Should return a collection object

        # Test collection exists
        assert concrete_client.collection_exists("test_collection") is True
        assert concrete_client.collection_exists("nonexistent") is False

        # Test list collections
        collections = concrete_client.list_collections()
        collection_names = [col.name for col in collections]
        assert "test_collection" in collection_names

        # Test collection deletion
        concrete_client.delete_collection("test_collection")
        assert concrete_client.collection_exists("test_collection") is False

    def test_chunk_operations(self, concrete_client, sample_document_chunks):
        """Test chunk management operations."""
        # Create collection first
        concrete_client.create_collection("test_collection")

        # Test adding chunks
        concrete_client.add_chunks(sample_document_chunks, "test_collection")

        # Test getting chunk
        chunk = concrete_client.get_chunk(
            sample_document_chunks[0].id, "test_collection"
        )
        assert chunk.id == sample_document_chunks[0].id
        assert chunk.content == sample_document_chunks[0].content

        # Test updating chunk
        updated_chunk = DocumentChunk(
            id=sample_document_chunks[0].id,
            content="Updated content",
            embedding=sample_document_chunks[0].embedding,
            metadata=sample_document_chunks[0].metadata,
        )
        concrete_client.update_chunk(updated_chunk, "test_collection")

        retrieved_chunk = concrete_client.get_chunk(updated_chunk.id, "test_collection")
        assert retrieved_chunk.content == "Updated content"

        # Test deleting chunks
        chunk_ids = [chunk.id for chunk in sample_document_chunks]
        concrete_client.delete_chunks(chunk_ids, "test_collection")

        # Should raise error when trying to get deleted chunk
        with pytest.raises(ValueError):
            concrete_client.get_chunk(sample_document_chunks[0].id, "test_collection")

    def test_search_operations(self, concrete_client, sample_search_query):
        """Test search operations."""
        results = concrete_client.search(sample_search_query)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(result, SearchResult) for result in results)

    def test_info_operations(self, concrete_client):
        """Test information retrieval operations."""
        # Test collection info
        collection_info = concrete_client.get_collection_info("test_collection")
        assert isinstance(collection_info, CollectionInfo)
        assert collection_info.name == "test_collection"

        # Test document info
        document_info = concrete_client.get_document_info("test_doc")
        assert isinstance(document_info, DocumentInfo)
        assert document_info.document_id == "test_doc"

    def test_health_check(self, concrete_client):
        """Test health check operation."""
        health = concrete_client.health_check()
        assert isinstance(health, bool)
        assert health is True


class TestInterfaceContract:
    """Test that interface contract is properly enforced."""

    def test_incomplete_implementation_fails(self):
        """Test that incomplete implementations cannot be instantiated."""

        class IncompleteImplementation(VectorDBInterface):
            def __init__(self, config):
                pass

            # Missing most required methods
            def health_check(self) -> bool:
                return True

        # Should fail to instantiate due to missing abstract methods
        with pytest.raises(TypeError):
            IncompleteImplementation(Mock())

    def test_method_return_types(
        self, concrete_client, sample_document_chunks, sample_search_query
    ):
        """Test that methods return expected types."""
        # Collection operations
        # create_collection returns Any (typically a collection object)
        result = concrete_client.create_collection("test")
        assert result is not None
        # delete_collection returns None
        result = concrete_client.delete_collection("test")
        assert result is None
        assert isinstance(concrete_client.list_collections(), list)
        assert isinstance(concrete_client.collection_exists("test"), bool)

        # Chunk operations return None or specific types
        result = concrete_client.add_chunks(sample_document_chunks)
        assert result is None

        result = concrete_client.delete_chunks(["test_id"])
        assert result is None

        result = concrete_client.update_chunk(sample_document_chunks[0])
        assert result is None

        # Search returns list of SearchResult
        results = concrete_client.search(sample_search_query)
        assert isinstance(results, list)

        # Info operations return specific model types
        collection_info = concrete_client.get_collection_info()
        assert isinstance(collection_info, CollectionInfo)

        document_info = concrete_client.get_document_info("test_doc")
        assert isinstance(document_info, DocumentInfo)

        # Health check returns bool
        health = concrete_client.health_check()
        assert isinstance(health, bool)

    def test_optional_parameters(
        self, concrete_client, sample_document_chunks, sample_search_query
    ):
        """Test that optional parameters work correctly."""
        # Methods with optional collection_name parameter
        concrete_client.add_chunks(sample_document_chunks)  # No collection_name
        concrete_client.add_chunks(
            sample_document_chunks, "specific_collection"
        )  # With collection_name

        concrete_client.delete_chunks(["test_id"])  # No collection_name
        concrete_client.delete_chunks(
            ["test_id"], "specific_collection"
        )  # With collection_name

        # Search with optional parameters
        results1 = concrete_client.search(sample_search_query)  # Basic search
        results2 = concrete_client.search(
            sample_search_query, [0.1, 0.2, 0.3]
        )  # With embedding
        results3 = concrete_client.search(
            sample_search_query, None, "specific_collection"
        )  # With collection

        assert isinstance(results1, list)
        assert isinstance(results2, list)
        assert isinstance(results3, list)
