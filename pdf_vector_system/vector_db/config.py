"""Configuration classes for vector database backends."""

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, field_validator


class VectorDBType(str, Enum):
    """Supported vector database types."""

    CHROMADB = "chromadb"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    MILVUS = "milvus"


class VectorDBConfig(BaseModel, ABC):
    """Abstract base class for vector database configurations."""

    db_type: VectorDBType = Field(description="Type of vector database")
    collection_name: str = Field(
        default="pdf_documents",
        description="Default collection name for storing embeddings",
    )
    max_results: int = Field(default=10, description="Maximum number of search results")

    @field_validator("collection_name")
    @classmethod
    def validate_collection_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("collection_name cannot be empty")
        return v.strip()

    @field_validator("max_results")
    @classmethod
    def validate_max_results(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_results must be positive")
        return v

    @abstractmethod
    def get_client_kwargs(self) -> dict[str, Any]:
        """Get keyword arguments for client initialization."""


class ChromaDBConfig(VectorDBConfig):
    """Configuration for ChromaDB vector database."""

    db_type: VectorDBType = Field(default=VectorDBType.CHROMADB, frozen=True)
    persist_directory: Path = Field(
        default=Path("./chroma_db"), description="Directory to persist ChromaDB data"
    )
    distance_metric: str = Field(
        default="cosine", description="Distance metric for similarity search"
    )

    @field_validator("collection_name")
    @classmethod
    def validate_collection_name(cls, v: str) -> str:
        v = super().validate_collection_name(v)
        # ChromaDB collection names must be alphanumeric with underscores/hyphens
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "ChromaDB collection_name must be alphanumeric with underscores or hyphens"
            )
        return v

    @field_validator("distance_metric")
    @classmethod
    def validate_distance_metric(cls, v: str) -> str:
        valid_metrics = ["cosine", "l2", "ip"]
        if v not in valid_metrics:
            raise ValueError(f"distance_metric must be one of {valid_metrics}")
        return v

    def get_client_kwargs(self) -> dict[str, Any]:
        """Get ChromaDB client initialization arguments."""
        return {
            "persist_directory": self.persist_directory,
            "collection_name": self.collection_name,
            "distance_metric": self.distance_metric,
            "max_results": self.max_results,
        }


class PineconeConfig(VectorDBConfig):
    """Configuration for Pinecone vector database."""

    db_type: VectorDBType = Field(default=VectorDBType.PINECONE, frozen=True)
    api_key: str = Field(description="Pinecone API key")
    environment: str = Field(description="Pinecone environment")
    index_name: str = Field(description="Pinecone index name")
    dimension: int = Field(
        default=384, description="Vector dimension (must match embedding model)"
    )
    metric: str = Field(
        default="cosine", description="Distance metric for similarity search"
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Pinecone API key cannot be empty")
        return v.strip()

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Pinecone environment cannot be empty")
        return v.strip()

    @field_validator("index_name")
    @classmethod
    def validate_index_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Pinecone index name cannot be empty")
        return v.strip()

    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Vector dimension must be positive")
        return v

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        valid_metrics = ["cosine", "euclidean", "dotproduct"]
        if v not in valid_metrics:
            raise ValueError(f"Pinecone metric must be one of {valid_metrics}")
        return v

    def get_client_kwargs(self) -> dict[str, Any]:
        """Get Pinecone client initialization arguments."""
        return {
            "api_key": self.api_key,
            "environment": self.environment,
            "index_name": self.index_name,
            "dimension": self.dimension,
            "metric": self.metric,
            "collection_name": self.collection_name,
            "max_results": self.max_results,
        }


class WeaviateConfig(VectorDBConfig):
    """Configuration for Weaviate vector database."""

    db_type: VectorDBType = Field(default=VectorDBType.WEAVIATE, frozen=True)
    url: str = Field(
        default="http://localhost:8080", description="Weaviate instance URL"
    )
    api_key: Optional[str] = Field(
        default=None, description="Weaviate API key (for cloud instances)"
    )
    class_name: str = Field(
        default="Document", description="Weaviate class name for documents"
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Weaviate URL cannot be empty")
        v = v.strip()
        if not (v.startswith(("http://", "https://"))):
            raise ValueError("Weaviate URL must start with http:// or https://")
        return v

    @field_validator("class_name")
    @classmethod
    def validate_class_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Weaviate class name cannot be empty")
        v = v.strip()
        # Weaviate class names must start with uppercase letter
        if not v[0].isupper():
            raise ValueError("Weaviate class name must start with uppercase letter")
        return v

    def get_client_kwargs(self) -> dict[str, Any]:
        """Get Weaviate client initialization arguments."""
        kwargs = {
            "url": self.url,
            "class_name": self.class_name,
            "collection_name": self.collection_name,
            "max_results": self.max_results,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return kwargs


class QdrantConfig(VectorDBConfig):
    """Configuration for Qdrant vector database."""

    db_type: VectorDBType = Field(default=VectorDBType.QDRANT, frozen=True)
    url: Optional[str] = Field(
        default=None, description="Qdrant server URL (for remote instances)"
    )
    host: str = Field(default="localhost", description="Qdrant server host")
    port: int = Field(default=6333, description="Qdrant server port")
    api_key: Optional[str] = Field(
        default=None, description="Qdrant API key (for cloud instances)"
    )
    distance_metric: str = Field(
        default="Cosine", description="Distance metric for similarity search"
    )

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError("Port must be between 1 and 65535")
        return v

    @field_validator("distance_metric")
    @classmethod
    def validate_distance_metric(cls, v: str) -> str:
        valid_metrics = ["Cosine", "Euclid", "Dot"]
        if v not in valid_metrics:
            raise ValueError(f"Qdrant distance_metric must be one of {valid_metrics}")
        return v

    def get_client_kwargs(self) -> dict[str, Any]:
        """Get Qdrant client initialization arguments."""
        kwargs = {
            "host": self.host,
            "port": self.port,
            "distance_metric": self.distance_metric,
            "collection_name": self.collection_name,
            "max_results": self.max_results,
        }
        if self.url:
            kwargs["url"] = self.url
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return kwargs


class MilvusConfig(VectorDBConfig):
    """Configuration for Milvus vector database."""

    db_type: VectorDBType = Field(default=VectorDBType.MILVUS, frozen=True)
    host: str = Field(default="localhost", description="Milvus server host")
    port: int = Field(default=19530, description="Milvus server port")
    user: Optional[str] = Field(default=None, description="Milvus username")
    password: Optional[str] = Field(default=None, description="Milvus password")
    dimension: int = Field(default=384, description="Vector dimension for embeddings")
    metric_type: str = Field(
        default="COSINE", description="Distance metric for similarity search"
    )

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError("Port must be between 1 and 65535")
        return v

    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("dimension must be positive")
        return v

    @field_validator("metric_type")
    @classmethod
    def validate_metric_type(cls, v: str) -> str:
        valid_metrics = ["COSINE", "L2", "IP"]
        if v not in valid_metrics:
            raise ValueError(f"Milvus metric_type must be one of {valid_metrics}")
        return v

    def get_client_kwargs(self) -> dict[str, Any]:
        """Get Milvus client initialization arguments."""
        kwargs = {
            "host": self.host,
            "port": self.port,
            "metric_type": self.metric_type,
            "collection_name": self.collection_name,
            "max_results": self.max_results,
        }
        if self.user:
            kwargs["user"] = self.user
        if self.password:
            kwargs["password"] = self.password
        return kwargs


# Type alias for any vector database configuration
VectorDBConfigType = Union[
    ChromaDBConfig, PineconeConfig, WeaviateConfig, QdrantConfig, MilvusConfig
]
