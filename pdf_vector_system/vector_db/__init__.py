"""Vector database module with multi-backend support."""

from pdf_vector_system.vector_db.chroma_client import ChromaDBClient
from pdf_vector_system.vector_db.config import (
    ChromaDBConfig,
    MilvusConfig,
    PineconeConfig,
    QdrantConfig,
    VectorDBConfig,
    VectorDBType,
    WeaviateConfig,
)
from pdf_vector_system.vector_db.converters import VectorDBConverter
from pdf_vector_system.vector_db.error_handler import (
    VectorDBErrorHandler,
    VectorDBLogger,
)
from pdf_vector_system.vector_db.factory import (
    VectorDBFactory,
    create_vector_db,
    get_available_backends,
)
from pdf_vector_system.vector_db.health_check import (
    HealthCheckResult,
    HealthStatus,
    VectorDBHealthManager,
)
from pdf_vector_system.vector_db.interface import VectorDBInterface
from pdf_vector_system.vector_db.milvus_client import MilvusClient
from pdf_vector_system.vector_db.models import (
    CollectionInfo,
    DocumentChunk,
    DocumentInfo,
    SearchQuery,
    SearchResult,
)
from pdf_vector_system.vector_db.pinecone_client import PineconeClient
from pdf_vector_system.vector_db.qdrant_client import QdrantClient
from pdf_vector_system.vector_db.weaviate_client import WeaviateClient

__all__ = [
    "ChromaDBClient",
    "ChromaDBConfig",
    "CollectionInfo",
    # Data models
    "DocumentChunk",
    "DocumentInfo",
    "HealthCheckResult",
    "HealthStatus",
    "MilvusClient",
    "MilvusConfig",
    "PineconeClient",
    "PineconeConfig",
    "QdrantClient",
    "QdrantConfig",
    "SearchQuery",
    "SearchResult",
    "VectorDBConfig",
    "VectorDBConverter",
    # Error handling and logging
    "VectorDBErrorHandler",
    # Factory and utilities
    "VectorDBFactory",
    # Health checking
    "VectorDBHealthManager",
    # Core interface and client
    "VectorDBInterface",
    "VectorDBLogger",
    # Configuration
    "VectorDBType",
    "WeaviateClient",
    "WeaviateConfig",
    "create_vector_db",
    "get_available_backends",
]
