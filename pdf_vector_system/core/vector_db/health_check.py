"""Health check and validation mechanisms for vector database backends."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from pdf_vector_system.core.vector_db.config import (
    ChromaDBConfig,
    MilvusConfig,
    PineconeConfig,
    QdrantConfig,
    VectorDBConfigType,
    VectorDBType,
    WeaviateConfig,
)
from pdf_vector_system.core.vector_db.error_handler import VectorDBLogger


class HealthStatus(str, Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""

    status: HealthStatus
    backend: str
    response_time: Optional[float] = None
    timestamp: Optional[float] = None
    details: Optional[dict[str, Any]] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        """Set default values after initialization."""
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.details is None:
            self.details = {}

    @property
    def response_time_ms(self) -> Optional[float]:
        """Get response time in milliseconds (backward compatibility)."""
        return self.response_time * 1000 if self.response_time is not None else None

    @property
    def error_message(self) -> Optional[str]:
        """Get error message (backward compatibility)."""
        return self.error

    @property
    def is_healthy(self) -> bool:
        """Check if the backend is healthy."""
        return self.status == HealthStatus.HEALTHY

    @property
    def is_available(self) -> bool:
        """Check if the backend is available (healthy or degraded)."""
        return self.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "backend": self.backend,
            "response_time": self.response_time,
            "response_time_ms": self.response_time_ms,
            "timestamp": self.timestamp,
            "is_healthy": self.is_healthy,
            "is_available": self.is_available,
            "details": self.details,
            "error": self.error,  # Include both for backward compatibility
            "error_message": self.error_message,
        }


class HealthChecker(ABC):
    """Abstract base class for backend health checkers."""

    def __init__(self, config: VectorDBConfigType):
        """Initialize health checker with configuration."""
        self.config = config
        self.backend_name = config.db_type.value

    @abstractmethod
    def check_health(self, timeout: float = 5.0) -> HealthCheckResult:
        """
        Perform health check on the backend.

        Args:
            timeout: Timeout in seconds for the health check

        Returns:
            HealthCheckResult object
        """

    @abstractmethod
    def validate_configuration(self) -> dict[str, Any]:
        """
        Validate the backend configuration.

        Returns:
            Dictionary with validation results
        """

    def _create_result(
        self,
        status: HealthStatus,
        response_time: float,
        details: dict[str, Any],
        error_message: Optional[str] = None,
    ) -> HealthCheckResult:
        """Create a health check result."""
        return HealthCheckResult(
            status=status,
            backend=self.backend_name,
            response_time=response_time,
            timestamp=time.time(),
            details=details,
            error=error_message,
        )


class ChromaDBHealthChecker(HealthChecker):
    """Health checker for ChromaDB."""

    def __init__(self, config):
        """Initialize with ChromaDBConfig."""
        if not isinstance(config, ChromaDBConfig):
            raise ValueError("ChromaDBHealthChecker requires ChromaDBConfig")
        super().__init__(config)
        # Store typed config for proper attribute access
        self.chroma_config: ChromaDBConfig = config

    def check_health(self, timeout: float = 5.0) -> HealthCheckResult:
        """Check ChromaDB health."""
        start_time = time.time()

        try:
            # Import ChromaDB client
            import chromadb
            from chromadb.config import Settings

            # Create temporary client (v0.4+ compatible)
            settings = Settings(
                persist_directory=str(self.chroma_config.persist_directory)
                # Note: is_persistent and anonymized_telemetry are deprecated in v0.4+
            )

            client = chromadb.Client(settings)

            # Perform heartbeat
            heartbeat = client.heartbeat()

            # Check if persist directory is accessible
            persist_dir_exists = self.chroma_config.persist_directory.exists()
            persist_dir_writable = True

            # Try to create the directory if it doesn't exist
            if not persist_dir_exists:
                try:
                    self.chroma_config.persist_directory.mkdir(
                        parents=True, exist_ok=True
                    )
                    persist_dir_exists = True
                except Exception:
                    persist_dir_writable = False

            # Check if it's actually a directory and writable
            if persist_dir_exists:
                persist_dir_writable = self.chroma_config.persist_directory.is_dir()
                if persist_dir_writable:
                    # Test write access
                    try:
                        test_file = self.chroma_config.persist_directory / ".test_write"
                        test_file.touch()
                        test_file.unlink()
                    except Exception:
                        persist_dir_writable = False

            response_time = time.time() - start_time

            details = {
                "heartbeat": heartbeat,
                "persist_directory": str(self.chroma_config.persist_directory),
                "persist_dir_exists": persist_dir_exists,
                "persist_dir_writable": persist_dir_writable,
                "collection_name": self.chroma_config.collection_name,
            }

            if persist_dir_exists and persist_dir_writable:
                status = HealthStatus.HEALTHY
            else:
                status = HealthStatus.DEGRADED
                details["warning"] = "Persist directory issues detected"

            return self._create_result(status, response_time, details)

        except ImportError as e:
            response_time = time.time() - start_time
            return self._create_result(
                HealthStatus.UNHEALTHY,
                response_time,
                {"error": "ChromaDB not installed"},
                str(e),
            )
        except Exception as e:
            response_time = time.time() - start_time
            return self._create_result(
                HealthStatus.UNHEALTHY,
                response_time,
                {"error": "Connection failed"},
                str(e),
            )

    def validate_configuration(self) -> dict[str, Any]:
        """Validate ChromaDB configuration."""
        validation: dict[str, Any] = {"valid": True, "issues": [], "warnings": []}

        # Check persist directory
        if not self.chroma_config.persist_directory.parent.exists():
            validation["issues"].append(
                f"Parent directory does not exist: {self.chroma_config.persist_directory.parent}"
            )
            validation["valid"] = False

        # Check collection name
        if (
            not self.chroma_config.collection_name.replace("_", "")
            .replace("-", "")
            .isalnum()
        ):
            validation["issues"].append(
                "Collection name must be alphanumeric with underscores or hyphens"
            )
            validation["valid"] = False

        # Check distance metric
        valid_metrics = ["cosine", "l2", "ip"]
        if self.chroma_config.distance_metric not in valid_metrics:
            validation["issues"].append(
                f"Invalid distance metric: {self.chroma_config.distance_metric}. Must be one of {valid_metrics}"
            )
            validation["valid"] = False

        return validation


class PineconeHealthChecker(HealthChecker):
    """Health checker for Pinecone."""

    def __init__(self, config):
        """Initialize with PineconeConfig."""
        if not isinstance(config, PineconeConfig):
            raise ValueError("PineconeHealthChecker requires PineconeConfig")
        super().__init__(config)
        # Store typed config for proper attribute access
        self.pinecone_config: PineconeConfig = config

    def check_health(self, timeout: float = 5.0) -> HealthCheckResult:
        """Check Pinecone health."""
        start_time = time.time()

        try:
            # Import Pinecone client
            from pinecone import Pinecone

            # Initialize Pinecone client (v7+ API)
            init_kwargs = {"api_key": self.pinecone_config.api_key}
            if (
                hasattr(self.pinecone_config, "environment")
                and self.pinecone_config.environment
            ):
                # Environment parameter is optional in v7+ for serverless indexes
                init_kwargs["environment"] = self.pinecone_config.environment

            pc = Pinecone(**init_kwargs)

            # List indexes to test connection
            indexes_response = pc.list_indexes()
            indexes = (
                [idx.name for idx in indexes_response.indexes]
                if hasattr(indexes_response, "indexes")
                else []
            )

            # Check if our index exists
            index_exists = self.pinecone_config.index_name in indexes

            response_time = time.time() - start_time

            details = {
                "environment": getattr(
                    self.pinecone_config, "environment", "serverless"
                ),
                "index_name": self.pinecone_config.index_name,
                "index_exists": index_exists,
                "available_indexes": indexes,
            }

            if index_exists:
                status = HealthStatus.HEALTHY
            else:
                status = HealthStatus.DEGRADED
                details["warning"] = (
                    f"Index '{self.pinecone_config.index_name}' does not exist"
                )

            return self._create_result(status, response_time, details)

        except ImportError as e:
            response_time = time.time() - start_time
            return self._create_result(
                HealthStatus.UNHEALTHY,
                response_time,
                {
                    "error": "Pinecone client not installed. Install with: pip install pinecone"
                },
                str(e),
            )
        except Exception as e:
            response_time = time.time() - start_time
            return self._create_result(
                HealthStatus.UNHEALTHY,
                response_time,
                {"error": "Connection failed"},
                str(e),
            )

    def validate_configuration(self) -> dict[str, Any]:
        """Validate Pinecone configuration."""
        validation: dict[str, Any] = {"valid": True, "issues": [], "warnings": []}

        # Check API key
        if not self.pinecone_config.api_key or len(self.pinecone_config.api_key) < 10:
            validation["issues"].append("Invalid or missing API key")
            validation["valid"] = False

        # Check environment (optional in v7+ for serverless indexes)
        if (
            hasattr(self.pinecone_config, "environment")
            and self.pinecone_config.environment == ""
        ):
            validation["warnings"].append(
                "Environment is empty - this is acceptable for serverless indexes in Pinecone v7+"
            )
        elif (
            hasattr(self.pinecone_config, "environment")
            and self.pinecone_config.environment
            and not isinstance(self.pinecone_config.environment, str)
        ):
            validation["issues"].append("Environment must be a string")
            validation["valid"] = False

        # Check index name
        if not self.pinecone_config.index_name:
            validation["issues"].append("Index name cannot be empty")
            validation["valid"] = False

        # Check dimension
        if self.pinecone_config.dimension <= 0:
            validation["issues"].append("Vector dimension must be positive")
            validation["valid"] = False

        # Check metric
        valid_metrics = ["cosine", "euclidean", "dotproduct"]
        if self.pinecone_config.metric not in valid_metrics:
            validation["issues"].append(
                f"Invalid metric: {self.pinecone_config.metric}. Must be one of {valid_metrics}"
            )
            validation["valid"] = False

        return validation


class WeaviateHealthChecker(HealthChecker):
    """Health checker for Weaviate."""

    def __init__(self, config):
        """Initialize with WeaviateConfig."""
        if not isinstance(config, WeaviateConfig):
            raise ValueError("WeaviateHealthChecker requires WeaviateConfig")
        super().__init__(config)
        # Store typed config for proper attribute access
        self.weaviate_config: WeaviateConfig = config

    def check_health(self, timeout: float = 5.0) -> HealthCheckResult:
        """Check Weaviate health using v4 API."""
        start_time = time.time()

        try:
            # Import Weaviate client
            import weaviate
            from weaviate.auth import AuthApiKey

            # Create temporary client using v4 API
            auth_credentials = None
            if self.weaviate_config.api_key:
                auth_credentials = AuthApiKey(self.weaviate_config.api_key)

            # Connect using appropriate method for v4 API
            if (
                "weaviate.io" in self.weaviate_config.url
                or "wcs" in self.weaviate_config.url
            ):
                client = weaviate.connect_to_wcs(
                    cluster_url=self.weaviate_config.url,
                    auth_credentials=auth_credentials,
                    timeout=(timeout, timeout),
                )
            else:
                client = weaviate.connect_to_custom(
                    http_host=self.weaviate_config.url.replace("http://", "")
                    .replace("https://", "")
                    .split(":")[0],
                    http_port=(
                        int(self.weaviate_config.url.split(":")[-1])
                        if ":" in self.weaviate_config.url.split("//")[-1]
                        else 8080
                    ),
                    http_secure="https" in self.weaviate_config.url,
                    auth_credentials=auth_credentials,
                    timeout=(timeout, timeout),
                )

            try:
                # Get meta information to test connection
                meta = client.get_meta()

                response_time = time.time() - start_time

                details = {
                    "url": self.weaviate_config.url,
                    "class_name": self.weaviate_config.class_name,
                    "version": meta.get("version", "unknown"),
                    "hostname": meta.get("hostname", "unknown"),
                }

                status = HealthStatus.HEALTHY

                return self._create_result(status, response_time, details)

            finally:
                # Always close the client connection (v4 requirement)
                client.close()

        except ImportError as e:
            response_time = time.time() - start_time
            return self._create_result(
                HealthStatus.UNHEALTHY,
                response_time,
                {"error": "Weaviate client not installed"},
                str(e),
            )
        except Exception as e:
            response_time = time.time() - start_time
            return self._create_result(
                HealthStatus.UNHEALTHY,
                response_time,
                {"error": "Connection failed"},
                str(e),
            )

    def validate_configuration(self) -> dict[str, Any]:
        """Validate Weaviate configuration."""
        validation: dict[str, Any] = {"valid": True, "issues": [], "warnings": []}

        # Check URL
        if not self.weaviate_config.url:
            validation["issues"].append("URL cannot be empty")
            validation["valid"] = False
        elif not self.weaviate_config.url.startswith(("http://", "https://")):
            validation["issues"].append("URL must start with http:// or https://")
            validation["valid"] = False

        # Check class name
        if not self.weaviate_config.class_name:
            validation["issues"].append("Class name cannot be empty")
            validation["valid"] = False
        elif not self.weaviate_config.class_name[0].isupper():
            validation["warnings"].append(
                "Class name should start with uppercase letter"
            )

        return validation


class QdrantHealthChecker(HealthChecker):
    """Health checker for Qdrant."""

    def __init__(self, config):
        """Initialize with QdrantConfig."""
        if not isinstance(config, QdrantConfig):
            raise ValueError("QdrantHealthChecker requires QdrantConfig")
        super().__init__(config)
        # Store typed config for proper attribute access
        self.qdrant_config: QdrantConfig = config

    def check_health(self, timeout: float = 5.0) -> HealthCheckResult:
        """Check Qdrant health."""
        start_time = time.time()

        try:
            # Import Qdrant client
            from qdrant_client import QdrantClient as QdrantClientLib

            # Create temporary client
            if self.qdrant_config.url:
                client = QdrantClientLib(
                    url=self.qdrant_config.url,
                    api_key=self.qdrant_config.api_key,
                    timeout=timeout,
                )
            else:
                client = QdrantClientLib(
                    host=self.qdrant_config.host,
                    port=self.qdrant_config.port,
                    timeout=timeout,
                )

            # Get collections to test connection
            collections = client.get_collections()

            response_time = time.time() - start_time

            details = {
                "host": self.qdrant_config.host,
                "port": self.qdrant_config.port,
                "url": self.qdrant_config.url,
                "collection_name": self.qdrant_config.collection_name,
                "vector_size": self.qdrant_config.vector_size,
                "collections_count": len(collections.collections),
            }

            status = HealthStatus.HEALTHY

            return self._create_result(status, response_time, details)

        except ImportError as e:
            response_time = time.time() - start_time
            return self._create_result(
                HealthStatus.UNHEALTHY,
                response_time,
                {"error": "Qdrant client not installed"},
                str(e),
            )
        except Exception as e:
            response_time = time.time() - start_time
            return self._create_result(
                HealthStatus.UNHEALTHY,
                response_time,
                {"error": "Connection failed"},
                str(e),
            )

    def validate_configuration(self) -> dict[str, Any]:
        """Validate Qdrant configuration."""
        validation: dict[str, Any] = {"valid": True, "issues": [], "warnings": []}

        # Check connection settings
        if not self.qdrant_config.url and not self.qdrant_config.host:
            validation["issues"].append("Either URL or host must be provided")
            validation["valid"] = False

        if self.qdrant_config.url and (
            self.qdrant_config.host or self.qdrant_config.port != 6333
        ):
            validation["warnings"].append(
                "URL takes precedence over host/port settings"
            )

        # Check collection name
        if not self.qdrant_config.collection_name:
            validation["issues"].append("Collection name cannot be empty")
            validation["valid"] = False

        # Check vector size
        if self.qdrant_config.vector_size <= 0:
            validation["issues"].append("Vector size must be positive")
            validation["valid"] = False

        return validation


class MilvusHealthChecker(HealthChecker):
    """Health checker for Milvus."""

    def __init__(self, config):
        """Initialize with MilvusConfig."""
        if not isinstance(config, MilvusConfig):
            raise ValueError("MilvusHealthChecker requires MilvusConfig")
        super().__init__(config)
        # Store typed config for proper attribute access
        self.milvus_config: MilvusConfig = config

    def check_health(self, timeout: float = 5.0) -> HealthCheckResult:
        """Check Milvus health."""
        start_time = time.time()

        try:
            # Import Milvus client
            from pymilvus import connections, utility

            # Create temporary connection
            connections.connect(
                alias="health_check",
                host=self.milvus_config.host,
                port=self.milvus_config.port,
                user=self.milvus_config.user,
                password=self.milvus_config.password,
            )

            # List collections to test connection
            collections = utility.list_collections()

            response_time = time.time() - start_time

            details = {
                "host": self.milvus_config.host,
                "port": self.milvus_config.port,
                "collection_name": self.milvus_config.collection_name,
                "dimension": self.milvus_config.dimension,
                "metric_type": self.milvus_config.metric_type,
                "collections_count": len(collections),
            }

            status = HealthStatus.HEALTHY

            # Disconnect health check connection
            connections.disconnect("health_check")

            return self._create_result(status, response_time, details)

        except ImportError as e:
            response_time = time.time() - start_time
            return self._create_result(
                HealthStatus.UNHEALTHY,
                response_time,
                {"error": "Milvus client not installed"},
                str(e),
            )
        except Exception as e:
            response_time = time.time() - start_time
            return self._create_result(
                HealthStatus.UNHEALTHY,
                response_time,
                {"error": "Connection failed"},
                str(e),
            )

    def validate_configuration(self) -> dict[str, Any]:
        """Validate Milvus configuration."""
        validation: dict[str, Any] = {"valid": True, "issues": [], "warnings": []}

        # Check host
        if not self.milvus_config.host:
            validation["issues"].append("Host cannot be empty")
            validation["valid"] = False

        # Check port
        if self.milvus_config.port <= 0 or self.milvus_config.port > 65535:
            validation["issues"].append("Port must be between 1 and 65535")
            validation["valid"] = False

        # Check collection name
        if not self.milvus_config.collection_name:
            validation["issues"].append("Collection name cannot be empty")
            validation["valid"] = False

        # Check dimension
        if self.milvus_config.dimension <= 0:
            validation["issues"].append("Vector dimension must be positive")
            validation["valid"] = False

        # Check metric type
        valid_metrics = ["L2", "IP", "COSINE", "HAMMING", "JACCARD"]
        if self.milvus_config.metric_type not in valid_metrics:
            validation["issues"].append(
                f"Invalid metric type: {self.milvus_config.metric_type}. Must be one of {valid_metrics}"
            )
            validation["valid"] = False

        return validation


class VectorDBHealthManager:
    """Manager for health checks across different vector database backends."""

    _HEALTH_CHECKERS = {
        VectorDBType.CHROMADB: ChromaDBHealthChecker,
        VectorDBType.PINECONE: PineconeHealthChecker,
        VectorDBType.WEAVIATE: WeaviateHealthChecker,
        VectorDBType.QDRANT: QdrantHealthChecker,
        VectorDBType.MILVUS: MilvusHealthChecker,
    }

    @classmethod
    def create_health_checker(cls, config: VectorDBConfigType) -> HealthChecker:
        """
        Create appropriate health checker for the given configuration.

        Args:
            config: Vector database configuration

        Returns:
            Health checker instance

        Raises:
            ValueError: If backend type is not supported
        """
        checker_class = cls._HEALTH_CHECKERS.get(config.db_type)
        if not checker_class:
            raise ValueError(f"No health checker available for {config.db_type}")

        return checker_class(config)  # type: ignore[abstract]

    @classmethod
    def get_health_checker(cls, config: VectorDBConfigType) -> HealthChecker:
        """
        Get appropriate health checker for the given configuration (alias for create_health_checker).

        Args:
            config: Vector database configuration

        Returns:
            Health checker instance

        Raises:
            ValueError: If backend type is not supported
        """
        return cls.create_health_checker(config)

    @classmethod
    def check_backend_health(
        cls, config: VectorDBConfigType, timeout: float = 5.0
    ) -> HealthCheckResult:
        """
        Perform health check on a vector database backend.

        Args:
            config: Vector database configuration
            timeout: Timeout in seconds

        Returns:
            Health check result
        """
        try:
            checker = cls.create_health_checker(config)
            result = checker.check_health(timeout)

            VectorDBLogger.log_health_check(
                backend=config.db_type.value,
                healthy=result.is_healthy,
                details=result.details,
            )

            return result

        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                backend=config.db_type.value,
                response_time=0.0,
                timestamp=time.time(),
                details={"error": "Health check failed"},
                error=str(e),
            )

    @classmethod
    def validate_backend_config(cls, config: VectorDBConfigType) -> dict[str, Any]:
        """
        Validate vector database backend configuration.

        Args:
            config: Vector database configuration

        Returns:
            Validation results
        """
        try:
            checker = cls.create_health_checker(config)
            return checker.validate_configuration()
        except Exception as e:
            return {
                "valid": False,
                "issues": [f"Configuration validation failed: {e!s}"],
                "warnings": [],
            }

    @classmethod
    def get_supported_backends(cls) -> list[VectorDBType]:
        """Get list of backends with health check support."""
        return list(cls._HEALTH_CHECKERS.keys())

    @classmethod
    def register_health_checker(
        cls, backend_type: VectorDBType, checker_class: type[HealthChecker]
    ) -> None:
        """
        Register a health checker for a backend type.

        Args:
            backend_type: Vector database type
            checker_class: Health checker class
        """
        cls._HEALTH_CHECKERS[backend_type] = checker_class
