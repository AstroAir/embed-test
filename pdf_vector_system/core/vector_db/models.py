"""Data models for vector database operations."""

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DocumentChunk:
    """Represents a document chunk with embedding and metadata."""

    id: str
    content: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate chunk after initialization."""
        if not self.id:
            raise ValueError("Chunk ID cannot be empty")
        if not self.content:
            raise ValueError("Chunk content cannot be empty")
        if not self.embedding:
            raise ValueError("Chunk embedding cannot be empty")

    @classmethod
    def create(
        cls,
        content: str,
        embedding: list[float],
        document_id: str,
        chunk_index: int,
        page_number: Optional[int] = None,
        start_char: Optional[int] = None,
        end_char: Optional[int] = None,
        additional_metadata: Optional[dict[str, Any]] = None,
    ) -> "DocumentChunk":
        """
        Create a DocumentChunk with standard metadata.

        Args:
            content: Text content of the chunk
            embedding: Embedding vector
            document_id: ID of the source document
            chunk_index: Index of this chunk within the document
            page_number: Page number if applicable
            start_char: Starting character position
            end_char: Ending character position
            additional_metadata: Additional metadata

        Returns:
            DocumentChunk instance
        """
        chunk_id = f"{document_id}_chunk_{chunk_index}"

        metadata = {
            "document_id": document_id,
            "chunk_index": chunk_index,
            "content_length": len(content),
            "created_at": time.time(),
            **(additional_metadata or {}),
        }

        if page_number is not None:
            metadata["page_number"] = page_number
        if start_char is not None:
            metadata["start_char"] = start_char
        if end_char is not None:
            metadata["end_char"] = end_char

        return cls(id=chunk_id, content=content, embedding=embedding, metadata=metadata)

    @classmethod
    def from_text_chunk(
        cls,
        content: str,
        embedding: list[float],
        document_id: str,
        chunk_index: int,
        page_number: Optional[int] = None,
        start_char: Optional[int] = None,
        end_char: Optional[int] = None,
        additional_metadata: Optional[dict[str, Any]] = None,
    ) -> "DocumentChunk":
        """
        Create a DocumentChunk from text content and metadata.

        Args:
            content: Text content of the chunk
            embedding: Embedding vector
            document_id: ID of the source document
            chunk_index: Index of this chunk within the document
            page_number: Page number if applicable
            start_char: Starting character position
            end_char: Ending character position
            additional_metadata: Additional metadata

        Returns:
            DocumentChunk instance
        """
        return cls.create(
            content=content,
            embedding=embedding,
            document_id=document_id,
            chunk_index=chunk_index,
            page_number=page_number,
            start_char=start_char,
            end_char=end_char,
            additional_metadata=additional_metadata,
        )

    @classmethod
    def create_chunk(
        cls,
        document_id: str,
        chunk_index: int,
        content: str,
        embedding: list[float],
        page_number: Optional[int] = None,
        start_char: Optional[int] = None,
        end_char: Optional[int] = None,
        additional_metadata: Optional[dict[str, Any]] = None,
    ) -> "DocumentChunk":
        """
        Create a DocumentChunk with standard metadata (alias for create method).

        Args:
            document_id: ID of the source document
            chunk_index: Index of this chunk within the document
            content: Text content of the chunk
            embedding: Embedding vector
            page_number: Page number if applicable
            start_char: Starting character position
            end_char: Ending character position
            additional_metadata: Additional metadata

        Returns:
            DocumentChunk instance
        """
        return cls.create(
            content=content,
            embedding=embedding,
            document_id=document_id,
            chunk_index=chunk_index,
            page_number=page_number,
            start_char=start_char,
            end_char=end_char,
            additional_metadata=additional_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dictionary representation of the chunk
        """
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
        }

    def to_chroma_format(self) -> dict[str, Any]:
        """
        Convert to ChromaDB format for batch operations.

        Returns:
            Dictionary in ChromaDB batch format
        """
        return {
            "ids": [self.id],
            "documents": [self.content],
            "embeddings": [self.embedding],
            "metadatas": [self.metadata],
        }


@dataclass
class SearchResult:
    """Represents a search result from vector database."""

    id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def similarity_score(self) -> float:
        """Get the similarity score (alias for score for backward compatibility)."""
        return self.score

    @property
    def document_id(self) -> Optional[str]:
        """Get the document ID from metadata."""
        return self.metadata.get("document_id")

    @property
    def chunk_index(self) -> Optional[int]:
        """Get the chunk index from metadata."""
        return self.metadata.get("chunk_index")

    @property
    def page_number(self) -> Optional[int]:
        """Get the page number from metadata."""
        return self.metadata.get("page_number")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
        }


@dataclass
class SearchQuery:
    """Represents a search query with parameters."""

    query_text: str
    n_results: int = 10
    where: Optional[dict[str, Any]] = None
    where_document: Optional[dict[str, Any]] = None
    include_distances: bool = True
    include_metadata: bool = True
    include_documents: bool = True
    # Alias for n_results for backward compatibility
    max_results: Optional[int] = None
    min_score: Optional[float] = None  # Minimum similarity score threshold
    filter_metadata: Optional[dict[str, Any]] = (
        None  # Alias for where for backward compatibility
    )

    def __post_init__(self) -> None:
        """Validate query after initialization."""
        # Handle max_results alias
        if self.max_results is not None:
            self.n_results = self.max_results

        # Handle filter_metadata alias
        if self.filter_metadata is not None and self.where is None:
            self.where = self.filter_metadata

        if not self.query_text or not self.query_text.strip():
            raise ValueError("Query text cannot be empty")
        if self.n_results <= 0:
            raise ValueError("Number of results must be positive")

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dictionary representation of the query
        """
        return {
            "query_text": self.query_text,
            "n_results": self.n_results,
            "where": self.where,
            "where_document": self.where_document,
            "include_distances": self.include_distances,
            "include_metadata": self.include_metadata,
            "include_documents": self.include_documents,
        }

    def to_chroma_params(self) -> dict[str, Any]:
        """
        Convert to ChromaDB query parameters.

        Returns:
            Dictionary of ChromaDB query parameters
        """
        from pdf_vector_system.core.vector_db.converters import VectorDBConverter

        return VectorDBConverter.query_to_chromadb_params(self)


@dataclass
class CollectionInfo:
    """Information about a vector database collection."""

    name: str
    count: int = 0  # Total count (backward compatibility)
    document_count: int = 0
    chunk_count: int = 0
    total_size_bytes: int = 0
    created_at: Optional[str] = None
    last_modified: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Handle backward compatibility."""
        # If count is set but chunk_count is not, use count as chunk_count
        if self.count > 0 and self.chunk_count == 0:
            self.chunk_count = self.count

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access for backward compatibility."""
        if key == "name":
            return self.name
        if key == "count":
            return self.chunk_count
        if key == "document_count":
            return self.document_count
        if key == "chunk_count":
            return self.chunk_count
        if key == "total_size_bytes":
            return self.total_size_bytes
        if key == "created_at":
            return self.created_at
        if key == "last_modified":
            return self.last_modified
        if key == "metadata":
            return self.metadata
        raise KeyError(f"Invalid key: {key}")

    @property
    def is_empty(self) -> bool:
        """Check if collection is empty."""
        return self.chunk_count == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "count": self.chunk_count,  # For backward compatibility
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "total_size_bytes": self.total_size_bytes,
            "created_at": self.created_at,
            "last_modified": self.last_modified,
            "metadata": self.metadata,
            "is_empty": self.is_empty,
        }


@dataclass
class DocumentInfo:
    """Information about a document in the vector database."""

    document_id: str
    chunk_count: int
    total_characters: int
    filename: Optional[str] = None
    page_count: Optional[int] = None
    file_size_bytes: Optional[int] = None
    created_at: Optional[str] = (
        None  # Changed to str for consistency with CollectionInfo
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def average_chunk_size(self) -> float:
        """Calculate average chunk size."""
        return self.total_characters / self.chunk_count if self.chunk_count > 0 else 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "chunk_count": self.chunk_count,
            "total_characters": self.total_characters,
            "page_count": self.page_count,
            "file_size_bytes": self.file_size_bytes,
            "created_at": self.created_at,
            "average_chunk_size": self.average_chunk_size,
            "metadata": self.metadata,
        }


class VectorDBError(Exception):
    """Base exception for vector database errors."""

    def __init__(
        self,
        message: str,
        backend: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize VectorDBError.

        Args:
            message: Error message
            backend: Name of the vector database backend
            original_error: Original exception that caused this error
        """
        self.backend = backend
        self.original_error = original_error

        if backend:
            message = f"[{backend}] {message}"

        super().__init__(message)


class CollectionNotFoundError(VectorDBError):
    """Raised when a collection is not found."""


class DocumentNotFoundError(VectorDBError):
    """Raised when a document is not found."""


class InvalidQueryError(VectorDBError):
    """Raised when a query is invalid."""


class ConnectionError(VectorDBError):
    """Raised when connection to vector database fails."""


class AuthenticationError(VectorDBError):
    """Raised when authentication with vector database fails."""


class ConfigurationError(VectorDBError):
    """Raised when vector database configuration is invalid."""


class BackendNotAvailableError(VectorDBError):
    """Raised when a vector database backend is not available."""


class IndexNotFoundError(VectorDBError):
    """Raised when an index is not found."""


class QuotaExceededError(VectorDBError):
    """Raised when API quota or limits are exceeded."""
