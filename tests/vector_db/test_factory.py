"""Tests for VectorDBFactory and related factory functions."""

from unittest.mock import Mock, patch

import pytest

from pdf_vector_system.vector_db.config import (
    ChromaDBConfig,
    MilvusConfig,
    PineconeConfig,
    QdrantConfig,
    VectorDBConfig,
    VectorDBType,
    WeaviateConfig,
)
from pdf_vector_system.vector_db.factory import (
    VectorDBFactory,
    create_vector_db,
    get_available_backends,
)
from pdf_vector_system.vector_db.models import BackendNotAvailableError, VectorDBError


class TestVectorDBFactory:
    """Test VectorDBFactory class."""

    def test_registry_contains_all_backends(self):
        """Test that registry contains all expected backends."""
        registry = VectorDBFactory._REGISTRY

        expected_backends = [
            VectorDBType.CHROMADB,
            VectorDBType.PINECONE,
            VectorDBType.WEAVIATE,
            VectorDBType.QDRANT,
            VectorDBType.MILVUS,
        ]

        for backend in expected_backends:
            assert backend in registry
            assert "client_class" in registry[backend]
            assert "config_class" in registry[backend]
            assert "required_packages" in registry[backend]
            assert "description" in registry[backend]

    def test_registry_structure(self):
        """Test that registry entries have correct structure."""
        registry = VectorDBFactory._REGISTRY

        for db_type, entry in registry.items():
            assert isinstance(db_type, VectorDBType)
            assert isinstance(entry["client_class"], str)
            assert issubclass(entry["config_class"], VectorDBConfig)
            assert isinstance(entry["required_packages"], list)
            assert isinstance(entry["description"], str)

            # Check that client_class is a valid module path
            assert "." in entry["client_class"]
            assert entry["client_class"].startswith("pdf_vector_system.vector_db.")

    @patch("pdf_vector_system.vector_db.factory.importlib.import_module")
    def test_import_client_class_success(self, mock_import):
        """Test successful client class import."""
        mock_module = Mock()
        mock_client_class = Mock()
        mock_module.ChromaDBClient = mock_client_class
        mock_import.return_value = mock_module

        result = VectorDBFactory._import_client_class(
            "pdf_vector_system.vector_db.chroma_client.ChromaDBClient"
        )

        assert result == mock_client_class
        mock_import.assert_called_once_with("pdf_vector_system.vector_db.chroma_client")

    @patch("pdf_vector_system.vector_db.factory.importlib.import_module")
    def test_import_client_class_failure(self, mock_import):
        """Test client class import failure."""
        mock_import.side_effect = ImportError("Module not found")

        with pytest.raises(ImportError):
            VectorDBFactory._import_client_class(
                "pdf_vector_system.vector_db.nonexistent.NonexistentClient"
            )

    @patch("pdf_vector_system.vector_db.factory.importlib.util.find_spec")
    def test_check_dependencies_success(self, mock_find_spec):
        """Test successful dependency check."""
        mock_find_spec.return_value = Mock()  # Package found

        # Should not raise any exception
        VectorDBFactory._check_dependencies(VectorDBType.CHROMADB, ["chromadb"])

    @patch("pdf_vector_system.vector_db.factory.importlib.util.find_spec")
    def test_check_dependencies_failure(self, mock_find_spec):
        """Test dependency check failure."""
        mock_find_spec.return_value = None  # Package not found

        with pytest.raises(
            BackendNotAvailableError, match="Required packages not installed"
        ):
            VectorDBFactory._check_dependencies(VectorDBType.PINECONE, ["pinecone"])

    @patch("pdf_vector_system.vector_db.factory.VectorDBFactory._check_dependencies")
    @patch("pdf_vector_system.vector_db.factory.VectorDBFactory._import_client_class")
    def test_create_client_success(self, mock_import_class, mock_check_deps):
        """Test successful client creation."""
        # Setup mocks
        mock_client_class = Mock()
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_import_class.return_value = mock_client_class

        config = ChromaDBConfig()

        result = VectorDBFactory.create_client(config)

        assert result == mock_client_instance
        mock_check_deps.assert_called_once()
        mock_import_class.assert_called_once()
        mock_client_class.assert_called_once_with(config)

    def test_create_client_unsupported_type(self):
        """Test client creation with unsupported database type."""
        # Create a mock config with unsupported type
        mock_config = Mock()
        mock_config.db_type = "unsupported_db"

        with pytest.raises(ValueError, match="Unsupported vector database type"):
            VectorDBFactory.create_client(mock_config)

    @patch("pdf_vector_system.vector_db.factory.VectorDBFactory._check_dependencies")
    @patch("pdf_vector_system.vector_db.factory.VectorDBFactory._import_client_class")
    def test_create_client_import_failure(self, mock_import_class, mock_check_deps):
        """Test client creation with import failure."""
        mock_import_class.side_effect = ImportError("Failed to import")

        config = ChromaDBConfig()

        with pytest.raises(VectorDBError, match="Failed to create chromadb client"):
            VectorDBFactory.create_client(config)

    def test_create_config_chromadb(self):
        """Test creating ChromaDB configuration."""
        config = VectorDBFactory.create_config(
            VectorDBType.CHROMADB,
            persist_directory="./test_chroma",
            collection_name="test_collection",
        )

        assert isinstance(config, ChromaDBConfig)
        assert config.db_type == VectorDBType.CHROMADB
        assert str(config.persist_directory) == "./test_chroma"
        assert config.collection_name == "test_collection"

    def test_create_config_pinecone(self):
        """Test creating Pinecone configuration."""
        config = VectorDBFactory.create_config(
            VectorDBType.PINECONE,
            api_key="test-key",
            environment="test-env",
            index_name="test-index",
        )

        assert isinstance(config, PineconeConfig)
        assert config.db_type == VectorDBType.PINECONE
        assert config.api_key == "test-key"
        assert config.environment == "test-env"
        assert config.index_name == "test-index"

    def test_create_config_unsupported_type(self):
        """Test creating configuration with unsupported type."""
        with pytest.raises(ValueError, match="Unsupported vector database type"):
            VectorDBFactory.create_config("unsupported_db")

    @patch("pdf_vector_system.vector_db.factory.importlib.util.find_spec")
    def test_get_available_backends(self, mock_find_spec):
        """Test getting available backends."""

        # Mock that only chromadb is available
        def mock_find_spec_side_effect(package_name):
            if package_name == "chromadb":
                return Mock()  # Package found
            return None  # Package not found

        mock_find_spec.side_effect = mock_find_spec_side_effect

        available = VectorDBFactory.get_available_backends()

        assert VectorDBType.CHROMADB in available
        assert VectorDBType.PINECONE not in available
        assert VectorDBType.WEAVIATE not in available
        assert VectorDBType.QDRANT not in available
        assert VectorDBType.MILVUS not in available

    def test_get_backend_info(self):
        """Test getting backend information."""
        info = VectorDBFactory.get_backend_info(VectorDBType.CHROMADB)

        assert "description" in info
        assert "required_packages" in info
        assert "config_class" in info
        assert isinstance(info["required_packages"], list)
        assert isinstance(info["description"], str)

    def test_get_backend_info_unsupported(self):
        """Test getting info for unsupported backend."""
        with pytest.raises(ValueError, match="Unsupported vector database type"):
            VectorDBFactory.get_backend_info("unsupported_db")

    def test_register_backend(self):
        """Test registering a new backend."""
        # Create a custom backend type for testing
        custom_type = "custom_db"

        class CustomConfig(VectorDBConfig):
            db_type: str = "custom_db"

        VectorDBFactory.register_backend(
            custom_type,
            "custom.module.CustomClient",
            CustomConfig,
            ["custom_package"],
            "Custom database backend",
        )

        # Verify registration
        assert custom_type in VectorDBFactory._REGISTRY
        registry_entry = VectorDBFactory._REGISTRY[custom_type]
        assert registry_entry["client_class"] == "custom.module.CustomClient"
        assert registry_entry["config_class"] == CustomConfig
        assert registry_entry["required_packages"] == ["custom_package"]
        assert registry_entry["description"] == "Custom database backend"

        # Clean up
        del VectorDBFactory._REGISTRY[custom_type]


class TestFactoryFunctions:
    """Test standalone factory functions."""

    @patch("pdf_vector_system.vector_db.factory.VectorDBFactory.create_client")
    def test_create_vector_db_function(self, mock_create_client):
        """Test create_vector_db convenience function."""
        mock_client = Mock()
        mock_create_client.return_value = mock_client

        config = ChromaDBConfig()
        result = create_vector_db(config)

        assert result == mock_client
        mock_create_client.assert_called_once_with(config)

    @patch("pdf_vector_system.vector_db.factory.VectorDBFactory.get_available_backends")
    def test_get_available_backends_function(self, mock_get_available):
        """Test get_available_backends convenience function."""
        mock_backends = [VectorDBType.CHROMADB, VectorDBType.PINECONE]
        mock_get_available.return_value = mock_backends

        result = get_available_backends()

        assert result == mock_backends
        mock_get_available.assert_called_once()


class TestFactoryIntegration:
    """Test factory integration with actual configurations."""

    def test_all_config_types_supported(self):
        """Test that factory supports all configuration types."""
        config_types = [
            (VectorDBType.CHROMADB, ChromaDBConfig, {}),
            (
                VectorDBType.PINECONE,
                PineconeConfig,
                {"api_key": "test", "environment": "test", "index_name": "test"},
            ),
            (
                VectorDBType.WEAVIATE,
                WeaviateConfig,
                {"url": "http://localhost:8080", "class_name": "Test"},
            ),
            (
                VectorDBType.QDRANT,
                QdrantConfig,
                {"url": "http://localhost:6333", "collection_name": "test"},
            ),
            (
                VectorDBType.MILVUS,
                MilvusConfig,
                {"host": "localhost", "port": 19530, "collection_name": "test"},
            ),
        ]

        for db_type, config_class, params in config_types:
            # Test config creation
            config = VectorDBFactory.create_config(db_type, **params)
            assert isinstance(config, config_class)
            assert config.db_type == db_type

    def test_factory_error_handling(self):
        """Test factory error handling scenarios."""
        # Test with None config
        with pytest.raises((TypeError, AttributeError)):
            VectorDBFactory.create_client(None)

        # Test with invalid config object
        invalid_config = Mock()
        invalid_config.db_type = "invalid"

        with pytest.raises(ValueError):
            VectorDBFactory.create_client(invalid_config)

    @patch("pdf_vector_system.vector_db.factory.logger")
    def test_factory_logging(self, mock_logger):
        """Test that factory operations are properly logged."""
        with (
            patch(
                "pdf_vector_system.vector_db.factory.VectorDBFactory._check_dependencies"
            ),
            patch(
                "pdf_vector_system.vector_db.factory.VectorDBFactory._import_client_class"
            ) as mock_import,
        ):
            mock_client_class = Mock()
            mock_import.return_value = mock_client_class

            config = ChromaDBConfig()
            VectorDBFactory.create_client(config)

            # Verify logging calls
            mock_logger.info.assert_called()
            log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
            assert any("Created chromadb client" in call for call in log_calls)
