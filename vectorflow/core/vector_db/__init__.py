"""Vector database module with multi-backend support."""

from vectorflow.core.vector_db.chroma_client import ChromaDBClient
from vectorflow.core.vector_db.config import (
    ChromaDBConfig,
    MilvusConfig,
    PineconeConfig,
    QdrantConfig,
    VectorDBConfig,
    VectorDBType,
    WeaviateConfig,
)
from vectorflow.core.vector_db.converters import VectorDBConverter
from vectorflow.core.vector_db.error_handler import VectorDBErrorHandler, VectorDBLogger
from vectorflow.core.vector_db.factory import (
    VectorDBFactory,
    create_vector_db,
    get_available_backends,
)
from vectorflow.core.vector_db.health_check import (
    HealthCheckResult,
    HealthStatus,
    VectorDBHealthManager,
)
from vectorflow.core.vector_db.interface import VectorDBInterface
from vectorflow.core.vector_db.models import (
    CollectionInfo,
    DocumentChunk,
    DocumentInfo,
    SearchQuery,
    SearchResult,
)

# Optional imports for vector DB backends
try:
    from vectorflow.core.vector_db.milvus_client import MilvusClient
except (ImportError, Exception):
    MilvusClient = None  # type: ignore

try:
    from vectorflow.core.vector_db.pinecone_client import PineconeClient
except (ImportError, Exception):
    PineconeClient = None  # type: ignore

try:
    from vectorflow.core.vector_db.qdrant_client import QdrantClient
except (ImportError, Exception):
    QdrantClient = None  # type: ignore

try:
    from vectorflow.core.vector_db.weaviate_client import WeaviateClient
except (ImportError, Exception):
    WeaviateClient = None  # type: ignore

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
