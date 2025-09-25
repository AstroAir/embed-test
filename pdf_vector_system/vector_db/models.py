"""Data models for vector database operations."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time
import uuid


@dataclass
class DocumentChunk:
    """Represents a document chunk with embedding and metadata."""
    
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
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
        embedding: List[float],
        document_id: str,
        chunk_index: int,
        page_number: Optional[int] = None,
        start_char: Optional[int] = None,
        end_char: Optional[int] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
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
            **(additional_metadata or {})
        }
        
        if page_number is not None:
            metadata["page_number"] = page_number
        if start_char is not None:
            metadata["start_char"] = start_char
        if end_char is not None:
            metadata["end_char"] = end_char
        
        return cls(
            id=chunk_id,
            content=content,
            embedding=embedding,
            metadata=metadata
        )
    
    def to_chroma_format(self) -> Dict[str, Any]:
        """
        Convert to ChromaDB format.
        
        Returns:
            Dictionary in ChromaDB format
        """
        return {
            "ids": [self.id],
            "documents": [self.content],
            "embeddings": [self.embedding],
            "metadatas": [self.metadata]
        }


@dataclass
class SearchResult:
    """Represents a search result from vector database."""
    
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number
        }


@dataclass
class SearchQuery:
    """Represents a search query with parameters."""
    
    query_text: str
    n_results: int = 10
    where: Optional[Dict[str, Any]] = None
    where_document: Optional[Dict[str, Any]] = None
    include_distances: bool = True
    include_metadata: bool = True
    include_documents: bool = True
    
    def __post_init__(self) -> None:
        """Validate query after initialization."""
        if not self.query_text or not self.query_text.strip():
            raise ValueError("Query text cannot be empty")
        if self.n_results <= 0:
            raise ValueError("Number of results must be positive")
    
    def to_chroma_params(self) -> Dict[str, Any]:
        """
        Convert to ChromaDB query parameters.
        
        Returns:
            Dictionary of ChromaDB query parameters
        """
        include = []
        if self.include_distances:
            include.append("distances")
        if self.include_metadata:
            include.append("metadatas")
        if self.include_documents:
            include.append("documents")
        
        params = {
            "n_results": self.n_results,
            "include": include
        }
        
        if self.where:
            params["where"] = self.where
        if self.where_document:
            params["where_document"] = self.where_document
        
        return params


@dataclass
class CollectionInfo:
    """Information about a ChromaDB collection."""
    
    name: str
    count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_empty(self) -> bool:
        """Check if collection is empty."""
        return self.count == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "count": self.count,
            "metadata": self.metadata,
            "is_empty": self.is_empty
        }


@dataclass
class DocumentInfo:
    """Information about a document in the vector database."""
    
    document_id: str
    chunk_count: int
    total_characters: int
    page_count: Optional[int] = None
    created_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def average_chunk_size(self) -> float:
        """Calculate average chunk size."""
        return self.total_characters / self.chunk_count if self.chunk_count > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "document_id": self.document_id,
            "chunk_count": self.chunk_count,
            "total_characters": self.total_characters,
            "page_count": self.page_count,
            "created_at": self.created_at,
            "average_chunk_size": self.average_chunk_size,
            "metadata": self.metadata
        }


class VectorDBError(Exception):
    """Base exception for vector database errors."""
    pass


class CollectionNotFoundError(VectorDBError):
    """Raised when a collection is not found."""
    pass


class DocumentNotFoundError(VectorDBError):
    """Raised when a document is not found."""
    pass


class InvalidQueryError(VectorDBError):
    """Raised when a query is invalid."""
    pass
