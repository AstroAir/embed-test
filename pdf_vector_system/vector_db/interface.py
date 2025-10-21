"""Abstract interface for vector database operations."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from pdf_vector_system.vector_db.models import (
    CollectionInfo,
    DocumentChunk,
    DocumentInfo,
    SearchQuery,
    SearchResult,
)


class VectorDBInterface(ABC):
    """Abstract interface for vector database operations.

    This interface defines the contract that all vector database backends
    must implement to ensure consistent behavior across different providers.
    """

    @abstractmethod
    def __init__(self, config: Any) -> None:
        """
        Initialize the vector database client.

        Args:
            config: Backend-specific configuration object
        """

    # Collection Management
    @abstractmethod
    def create_collection(
        self,
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        get_or_create: bool = True,
    ) -> Any:
        """
        Create or get a collection.

        Args:
            name: Collection name (uses default if None)
            metadata: Optional metadata for the collection
            get_or_create: If True, get existing collection or create new one

        Returns:
            Collection object (backend-specific type)

        Raises:
            VectorDBError: If collection creation fails
        """

    @abstractmethod
    def get_collection(self, name: Optional[str] = None) -> Any:
        """
        Get an existing collection.

        Args:
            name: Collection name (uses default if None)

        Returns:
            Collection object (backend-specific type)

        Raises:
            CollectionNotFoundError: If collection doesn't exist
        """

    @abstractmethod
    def collection_exists(self, name: Optional[str] = None) -> bool:
        """
        Check if a collection exists.

        Args:
            name: Collection name (uses default if None)

        Returns:
            True if collection exists, False otherwise

        Raises:
            VectorDBError: If check operation fails
        """

    @abstractmethod
    def delete_collection(self, name: Optional[str] = None) -> None:
        """
        Delete a collection.

        Args:
            name: Collection name (uses default if None)

        Raises:
            VectorDBError: If collection deletion fails
        """

    @abstractmethod
    def list_collections(self) -> list[CollectionInfo]:
        """
        List all collections.

        Returns:
            List of CollectionInfo objects

        Raises:
            VectorDBError: If listing collections fails
        """

    # Document Operations (CRUD)
    @abstractmethod
    def add_chunks(
        self, chunks: list[DocumentChunk], collection_name: Optional[str] = None
    ) -> None:
        """
        Add document chunks to a collection.

        Args:
            chunks: List of DocumentChunk objects to add
            collection_name: Target collection name (uses default if None)

        Raises:
            VectorDBError: If adding chunks fails
            ValueError: If chunks list is empty
        """

    @abstractmethod
    def get_chunks(
        self,
        chunk_ids: list[str],
        collection_name: Optional[str] = None,
        include_embeddings: bool = False,
    ) -> list[DocumentChunk]:
        """
        Retrieve specific chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to retrieve
            collection_name: Source collection name (uses default if None)
            include_embeddings: Whether to include embedding vectors

        Returns:
            List of DocumentChunk objects

        Raises:
            VectorDBError: If retrieval fails
            DocumentNotFoundError: If some chunks are not found
        """

    @abstractmethod
    def get_chunk(
        self,
        chunk_id: str,
        collection_name: Optional[str] = None,
        include_embeddings: bool = False,
    ) -> DocumentChunk:
        """
        Retrieve a specific chunk by its ID.

        Args:
            chunk_id: ID of the chunk to retrieve
            collection_name: Source collection name (uses default if None)
            include_embeddings: Whether to include embedding vectors

        Returns:
            DocumentChunk object

        Raises:
            VectorDBError: If retrieval fails
            DocumentNotFoundError: If chunk is not found
        """

    @abstractmethod
    def update_chunks(
        self, chunks: list[DocumentChunk], collection_name: Optional[str] = None
    ) -> None:
        """
        Update existing document chunks.

        Args:
            chunks: List of DocumentChunk objects with updated data
            collection_name: Target collection name (uses default if None)

        Raises:
            VectorDBError: If update fails
            DocumentNotFoundError: If some chunks don't exist
        """

    @abstractmethod
    def update_chunk(
        self, chunk: DocumentChunk, collection_name: Optional[str] = None
    ) -> None:
        """
        Update an existing document chunk.

        Args:
            chunk: DocumentChunk object with updated data
            collection_name: Target collection name (uses default if None)

        Raises:
            VectorDBError: If update fails
            DocumentNotFoundError: If chunk doesn't exist
        """

    @abstractmethod
    def delete_chunks(
        self, chunk_ids: list[str], collection_name: Optional[str] = None
    ) -> None:
        """
        Delete specific chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to delete
            collection_name: Target collection name (uses default if None)

        Raises:
            VectorDBError: If deletion fails
        """

    @abstractmethod
    def delete_document(
        self, document_id: str, collection_name: Optional[str] = None
    ) -> int:
        """
        Delete all chunks belonging to a document.

        Args:
            document_id: ID of the document to delete
            collection_name: Target collection name (uses default if None)

        Returns:
            Number of chunks deleted

        Raises:
            VectorDBError: If deletion fails
        """

    # Search Operations
    @abstractmethod
    def search(
        self,
        query: SearchQuery,
        query_embedding: Optional[list[float]] = None,
        collection_name: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents using vector similarity.

        Args:
            query: SearchQuery object with search parameters
            query_embedding: Pre-computed query embedding (optional)
            collection_name: Target collection name (uses default if None)

        Returns:
            List of SearchResult objects ordered by similarity

        Raises:
            VectorDBError: If search fails
            InvalidQueryError: If query parameters are invalid
        """

    @abstractmethod
    def search_by_metadata(
        self,
        metadata_filter: dict[str, Any],
        collection_name: Optional[str] = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Search for documents by metadata filters only.

        Args:
            metadata_filter: Dictionary of metadata key-value pairs to filter by
            collection_name: Target collection name (uses default if None)
            limit: Maximum number of results to return

        Returns:
            List of SearchResult objects

        Raises:
            VectorDBError: If search fails
        """

    @abstractmethod
    def find_similar_chunks(
        self,
        chunk_id: str,
        collection_name: Optional[str] = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Find chunks similar to a given chunk.

        Args:
            chunk_id: ID of the reference chunk
            collection_name: Target collection name (uses default if None)
            limit: Maximum number of results to return

        Returns:
            List of SearchResult objects ordered by similarity

        Raises:
            VectorDBError: If search fails
            DocumentNotFoundError: If reference chunk doesn't exist
        """

    # Information and Statistics
    @abstractmethod
    def get_collection_info(self, name: Optional[str] = None) -> CollectionInfo:
        """
        Get information about a collection.

        Args:
            name: Collection name (uses default if None)

        Returns:
            CollectionInfo object with collection statistics

        Raises:
            CollectionNotFoundError: If collection doesn't exist
            VectorDBError: If retrieval fails
        """

    @abstractmethod
    def get_document_info(
        self, document_id: str, collection_name: Optional[str] = None
    ) -> DocumentInfo:
        """
        Get information about a specific document.

        Args:
            document_id: ID of the document
            collection_name: Target collection name (uses default if None)

        Returns:
            DocumentInfo object with document statistics

        Raises:
            DocumentNotFoundError: If document doesn't exist
            VectorDBError: If retrieval fails
        """

    # Health and Connectivity
    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the vector database is healthy and accessible.

        Returns:
            True if healthy, False otherwise
        """

    @abstractmethod
    def get_backend_info(self) -> dict[str, Any]:
        """
        Get information about the backend implementation.

        Returns:
            Dictionary with backend name, version, and capabilities
        """

    # Utility Methods
    @abstractmethod
    def count_chunks(self, collection_name: Optional[str] = None) -> int:
        """
        Count total number of chunks in a collection.

        Args:
            collection_name: Target collection name (uses default if None)

        Returns:
            Number of chunks in the collection

        Raises:
            CollectionNotFoundError: If collection doesn't exist
            VectorDBError: If count operation fails
        """

    @abstractmethod
    def clear_collection(self, collection_name: Optional[str] = None) -> None:
        """
        Remove all chunks from a collection without deleting the collection.

        Args:
            collection_name: Target collection name (uses default if None)

        Raises:
            CollectionNotFoundError: If collection doesn't exist
            VectorDBError: If clear operation fails
        """
