"""Factory for creating vector database clients."""

import importlib
from typing import Any

from loguru import logger

from pdf_vector_system.core.vector_db.config import (
    ChromaDBConfig,
    MilvusConfig,
    PineconeConfig,
    QdrantConfig,
    VectorDBConfig,
    VectorDBConfigType,
    VectorDBType,
    WeaviateConfig,
)
from pdf_vector_system.core.vector_db.interface import VectorDBInterface
from pdf_vector_system.core.vector_db.models import VectorDBError


class VectorDBFactory:
    """Factory for creating vector database clients based on configuration."""

    # Registry mapping database types to their client classes and required packages
    _REGISTRY: dict[VectorDBType, dict[str, Any]] = {
        VectorDBType.CHROMADB: {
            "client_class": "pdf_vector_system.core.vector_db.chroma_client.ChromaDBClient",
            "config_class": ChromaDBConfig,
            "required_packages": ["chromadb"],
            "description": "ChromaDB - Local vector database with persistence",
        },
        VectorDBType.PINECONE: {
            "client_class": "pdf_vector_system.core.vector_db.pinecone_client.PineconeClient",
            "config_class": PineconeConfig,
            "required_packages": ["pinecone"],
            "description": "Pinecone - Managed cloud vector database",
        },
        VectorDBType.WEAVIATE: {
            "client_class": "pdf_vector_system.core.vector_db.weaviate_client.WeaviateClient",
            "config_class": WeaviateConfig,
            "required_packages": ["weaviate"],
            "description": "Weaviate - Open-source vector database with GraphQL API",
        },
        VectorDBType.QDRANT: {
            "client_class": "pdf_vector_system.core.vector_db.qdrant_client.QdrantClient",
            "config_class": QdrantConfig,
            "required_packages": ["qdrant_client"],
            "description": "Qdrant - High-performance vector similarity search engine",
        },
        VectorDBType.MILVUS: {
            "client_class": "pdf_vector_system.core.vector_db.milvus_client.MilvusClient",
            "config_class": MilvusConfig,
            "required_packages": ["pymilvus"],
            "description": "Milvus - Scalable vector database for AI applications",
        },
    }

    @classmethod
    def create_client(cls, config: VectorDBConfigType) -> VectorDBInterface:
        """
        Create a vector database client based on the provided configuration.

        Args:
            config: Vector database configuration object

        Returns:
            Vector database client implementing VectorDBInterface

        Raises:
            VectorDBError: If client creation fails
            ValueError: If unsupported database type
        """
        db_type = config.db_type

        if db_type not in cls._REGISTRY:
            raise ValueError(f"Unsupported vector database type: {db_type}")

        registry_entry = cls._REGISTRY[db_type]

        # Check if required packages are available
        cls._check_dependencies(db_type, registry_entry["required_packages"])

        # Import and instantiate the client class
        try:
            client_class = cls._import_client_class(registry_entry["client_class"])
            client = client_class(config)

            # Handle both VectorDBType enum and string
            db_type_str = db_type.value if hasattr(db_type, "value") else str(db_type)
            logger.info(
                f"Created {db_type_str} client: {registry_entry['description']}"
            )
            return client

        except Exception as e:
            # Handle both VectorDBType enum and string
            db_type_str = db_type.value if hasattr(db_type, "value") else str(db_type)
            error_msg = f"Failed to create {db_type_str} client: {e!s}"
            logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    @classmethod
    def create_config(cls, db_type: VectorDBType, **kwargs: Any) -> VectorDBConfigType:
        """
        Create a configuration object for the specified database type.

        Args:
            db_type: Type of vector database
            **kwargs: Configuration parameters

        Returns:
            Configuration object for the specified database type

        Raises:
            ValueError: If unsupported database type
        """
        if db_type not in cls._REGISTRY:
            raise ValueError(f"Unsupported vector database type: {db_type}")

        config_class = cls._REGISTRY[db_type]["config_class"]
        return config_class(**kwargs)

    @classmethod
    def get_supported_types(cls) -> dict[VectorDBType, str]:
        """
        Get all supported vector database types and their descriptions.

        Returns:
            Dictionary mapping database types to descriptions
        """
        return {
            db_type: entry["description"] for db_type, entry in cls._REGISTRY.items()
        }

    @classmethod
    def get_backend_info(cls, db_type: VectorDBType) -> dict[str, Any]:
        """
        Get detailed information about a vector database backend.

        Args:
            db_type: Type of vector database

        Returns:
            Dictionary containing backend information including capabilities

        Raises:
            ValueError: If unsupported database type
        """
        if db_type not in cls._REGISTRY:
            raise ValueError(f"Unsupported vector database type: {db_type}")

        entry = cls._REGISTRY[db_type]

        # Define capabilities for each backend
        capabilities_map = {
            VectorDBType.CHROMADB: [
                "local_storage",
                "persistence",
                "metadata_filtering",
            ],
            VectorDBType.PINECONE: ["cloud_storage", "managed_service", "auto_scaling"],
            VectorDBType.WEAVIATE: [
                "graphql_api",
                "schema_management",
                "hybrid_search",
            ],
            VectorDBType.QDRANT: ["high_performance", "filtering", "payload_indexing"],
            VectorDBType.MILVUS: [
                "horizontal_scaling",
                "distributed",
                "gpu_acceleration",
            ],
        }

        return {
            "description": entry["description"],
            "required_packages": entry["required_packages"],
            "capabilities": capabilities_map.get(db_type, []),
            "config_class": entry["config_class"].__name__,
        }

    @classmethod
    def is_supported(cls, db_type: VectorDBType) -> bool:
        """
        Check if a vector database type is supported.

        Args:
            db_type: Type of vector database to check

        Returns:
            True if supported, False otherwise
        """
        return db_type in cls._REGISTRY

    @classmethod
    def check_availability(cls, db_type: VectorDBType) -> bool:
        """
        Check if a vector database type is available (dependencies installed).

        Args:
            db_type: Type of vector database to check

        Returns:
            True if available, False otherwise
        """
        if not cls.is_supported(db_type):
            return False

        try:
            registry_entry = cls._REGISTRY[db_type]
            cls._check_dependencies(db_type, registry_entry["required_packages"])
            return True
        except VectorDBError:
            return False

    @classmethod
    def get_available_types(cls) -> dict[VectorDBType, str]:
        """
        Get all available vector database types (with dependencies installed).

        Returns:
            Dictionary mapping available database types to descriptions
        """
        available = {}
        for db_type, entry in cls._REGISTRY.items():
            if cls.check_availability(db_type):
                available[db_type] = entry["description"]
        return available

    @classmethod
    def get_available_backends(cls) -> list[VectorDBType]:
        """
        Get list of available vector database backends.

        Returns:
            List of available VectorDBType enums
        """
        return list(cls.get_available_types().keys())

    @classmethod
    def check_dependencies(cls) -> dict[VectorDBType, list[str]]:
        """
        Check dependencies for all backends and return missing packages.

        Returns:
            Dictionary mapping backend types to lists of missing packages
        """
        missing = {}
        for db_type, entry in cls._REGISTRY.items():
            try:
                cls._check_dependencies(db_type, entry["required_packages"])
            except VectorDBError as e:
                # Extract missing packages from error message
                error_msg = str(e)
                if "not installed" in error_msg:
                    missing[db_type] = entry["required_packages"]
        return missing

    @classmethod
    def _check_dependencies(cls, db_type: VectorDBType, packages: list[str]) -> None:
        """
        Check if required packages are installed.

        Args:
            db_type: Database type being checked
            packages: List of required package names

        Raises:
            VectorDBError: If required packages are missing
        """
        missing_packages = []

        for package in packages:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            # Handle both VectorDBType enum and string
            db_type_str = db_type.value if hasattr(db_type, "value") else str(db_type)
            error_msg = (
                f"Missing required packages for {db_type_str}: {missing_packages}. "
                f"Install with: pip install {' '.join(missing_packages)}"
            )
            raise VectorDBError(error_msg)

    @classmethod
    def _import_client_class(cls, class_path: str) -> type[VectorDBInterface]:
        """
        Import a client class from its module path.

        Args:
            class_path: Full module path to the client class

        Returns:
            Client class

        Raises:
            ImportError: If class cannot be imported
        """
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    @classmethod
    def register_backend(
        cls,
        db_type: VectorDBType,
        client_class_path: str,
        config_class: type[VectorDBConfig],
        required_packages: list[str],
        description: str,
    ) -> None:
        """
        Register a new vector database backend.

        Args:
            db_type: Database type identifier
            client_class_path: Full module path to client class
            config_class: Configuration class for the backend
            required_packages: List of required package names
            description: Human-readable description
        """
        cls._REGISTRY[db_type] = {
            "client_class": client_class_path,
            "config_class": config_class,
            "required_packages": required_packages,
            "description": description,
        }
        logger.info(f"Registered vector database backend: {db_type.value}")


# Convenience functions for common operations
def create_vector_db(config: VectorDBConfigType) -> VectorDBInterface:
    """
    Convenience function to create a vector database client.

    Args:
        config: Vector database configuration

    Returns:
        Vector database client
    """
    return VectorDBFactory.create_client(config)


def get_available_backends() -> dict[VectorDBType, str]:
    """
    Convenience function to get available vector database backends.

    Returns:
        Dictionary of available backends and their descriptions
    """
    return VectorDBFactory.get_available_types()


def create_chromadb_config(**kwargs: Any) -> ChromaDBConfig:
    """Create ChromaDB configuration with defaults."""
    return ChromaDBConfig(**kwargs)


def create_pinecone_config(**kwargs: Any) -> PineconeConfig:
    """Create Pinecone configuration with defaults."""
    return PineconeConfig(**kwargs)


def create_weaviate_config(**kwargs: Any) -> WeaviateConfig:
    """Create Weaviate configuration with defaults."""
    return WeaviateConfig(**kwargs)


def create_qdrant_config(**kwargs: Any) -> QdrantConfig:
    """Create Qdrant configuration with defaults."""
    return QdrantConfig(**kwargs)


def create_milvus_config(**kwargs: Any) -> MilvusConfig:
    """Create Milvus configuration with defaults."""
    return MilvusConfig(**kwargs)
