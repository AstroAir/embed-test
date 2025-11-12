"""Tests for vector database configuration classes."""

from pathlib import Path

import pytest

from pdf_vector_system.core.vector_db.config import (
    ChromaDBConfig,
    MilvusConfig,
    PineconeConfig,
    QdrantConfig,
    VectorDBConfig,
    VectorDBType,
    WeaviateConfig,
)


class TestVectorDBType:
    """Test VectorDBType enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        expected_values = ["chromadb", "pinecone", "weaviate", "qdrant", "milvus"]
        actual_values = [db_type.value for db_type in VectorDBType]

        for expected in expected_values:
            assert expected in actual_values

    def test_enum_from_string(self):
        """Test creating enum from string values."""
        assert VectorDBType("chromadb") == VectorDBType.CHROMADB
        assert VectorDBType("pinecone") == VectorDBType.PINECONE
        assert VectorDBType("weaviate") == VectorDBType.WEAVIATE
        assert VectorDBType("qdrant") == VectorDBType.QDRANT
        assert VectorDBType("milvus") == VectorDBType.MILVUS

    def test_invalid_enum_value(self):
        """Test that invalid enum values raise ValueError."""
        with pytest.raises(
            ValueError, match="'invalid_db' is not a valid VectorDBType"
        ):
            VectorDBType("invalid_db")


class TestVectorDBConfig:
    """Test base VectorDBConfig class."""

    def test_abstract_base_class(self):
        """Test that VectorDBConfig cannot be instantiated directly."""
        with pytest.raises(TypeError):
            VectorDBConfig()


class TestChromaDBConfig:
    """Test ChromaDBConfig class."""

    def test_default_initialization(self):
        """Test ChromaDBConfig with default values."""
        config = ChromaDBConfig()

        assert config.db_type == VectorDBType.CHROMADB
        assert config.persist_directory == Path("./chroma_db")
        assert config.collection_name == "pdf_documents"
        assert config.distance_metric == "cosine"
        assert config.max_results == 10

    def test_custom_initialization(self, vector_db_temp_dir):
        """Test ChromaDBConfig with custom values."""
        config = ChromaDBConfig(
            persist_directory=vector_db_temp_dir / "custom_chroma",
            collection_name="custom_collection",
            distance_metric="l2",
            max_results=50,
        )

        assert config.persist_directory == vector_db_temp_dir / "custom_chroma"
        assert config.collection_name == "custom_collection"
        assert config.distance_metric == "l2"
        assert config.max_results == 50

    def test_invalid_distance_metric(self):
        """Test that invalid distance metric raises ValueError."""
        with pytest.raises(ValueError, match="Distance metric must be one of"):
            ChromaDBConfig(distance_metric="invalid_metric")

    def test_invalid_max_results(self):
        """Test that invalid max_results raises ValueError."""
        with pytest.raises(ValueError, match="max_results must be positive"):
            ChromaDBConfig(max_results=0)

        with pytest.raises(ValueError, match="max_results must be positive"):
            ChromaDBConfig(max_results=-1)

    def test_get_client_kwargs(self, vector_db_temp_dir):
        """Test get_client_kwargs method."""
        config = ChromaDBConfig(
            persist_directory=vector_db_temp_dir / "test_chroma",
            collection_name="test_collection",
            max_results=25,
        )

        kwargs = config.get_client_kwargs()

        assert kwargs["persist_directory"] == vector_db_temp_dir / "test_chroma"
        assert kwargs["collection_name"] == "test_collection"
        assert kwargs["distance_metric"] == "cosine"
        assert kwargs["max_results"] == 25


class TestPineconeConfig:
    """Test PineconeConfig class."""

    def test_initialization(self):
        """Test PineconeConfig initialization."""
        config = PineconeConfig(
            api_key="test-api-key", environment="test-env", index_name="test-index"
        )

        assert config.db_type == VectorDBType.PINECONE
        assert config.api_key == "test-api-key"
        assert config.environment == "test-env"
        assert config.index_name == "test-index"
        assert config.max_results == 10  # default

    def test_missing_required_fields(self):
        """Test that missing required fields raise ValueError."""
        with pytest.raises(ValueError, match="api_key|index_name"):
            PineconeConfig()  # Missing required fields

    def test_invalid_max_results(self):
        """Test that invalid max_results raises ValueError."""
        with pytest.raises(ValueError, match="max_results must be positive"):
            PineconeConfig(
                api_key="test-key",
                environment="test-env",
                index_name="test-index",
                max_results=0,
            )

    def test_get_client_kwargs(self):
        """Test get_client_kwargs method."""
        config = PineconeConfig(
            api_key="test-api-key",
            environment="test-env",
            index_name="test-index",
            max_results=20,
        )

        kwargs = config.get_client_kwargs()

        assert kwargs["api_key"] == "test-api-key"
        assert kwargs["environment"] == "test-env"
        assert kwargs["index_name"] == "test-index"
        assert kwargs["max_results"] == 20


class TestWeaviateConfig:
    """Test WeaviateConfig class."""

    def test_initialization(self):
        """Test WeaviateConfig initialization."""
        config = WeaviateConfig(url="http://localhost:8080", class_name="TestDocument")

        assert config.db_type == VectorDBType.WEAVIATE
        assert config.url == "http://localhost:8080"
        assert config.class_name == "TestDocument"
        assert config.max_results == 10  # default

    def test_with_auth(self):
        """Test WeaviateConfig with authentication."""
        config = WeaviateConfig(
            url="http://localhost:8080",
            class_name="TestDocument",
            api_key="test-api-key",
        )

        assert config.api_key == "test-api-key"

    def test_invalid_url(self):
        """Test that invalid URL raises ValueError."""
        with pytest.raises(ValueError, match="URL must be a valid HTTP/HTTPS URL"):
            WeaviateConfig(url="invalid-url", class_name="TestDocument")

    def test_get_client_kwargs(self):
        """Test get_client_kwargs method."""
        config = WeaviateConfig(
            url="http://localhost:8080",
            class_name="TestDocument",
            api_key="test-key",
            max_results=15,
        )

        kwargs = config.get_client_kwargs()

        assert kwargs["url"] == "http://localhost:8080"
        assert kwargs["class_name"] == "TestDocument"
        assert kwargs["api_key"] == "test-key"
        assert kwargs["max_results"] == 15


class TestQdrantConfig:
    """Test QdrantConfig class."""

    def test_initialization(self):
        """Test QdrantConfig initialization."""
        config = QdrantConfig(
            url="http://localhost:6333", collection_name="test_collection"
        )

        assert config.db_type == VectorDBType.QDRANT
        assert config.url == "http://localhost:6333"
        assert config.collection_name == "test_collection"
        assert config.max_results == 10  # default

    def test_with_api_key(self):
        """Test QdrantConfig with API key."""
        config = QdrantConfig(
            url="http://localhost:6333",
            collection_name="test_collection",
            api_key="test-api-key",
        )

        assert config.api_key == "test-api-key"

    def test_invalid_url(self):
        """Test that invalid URL raises ValueError."""
        with pytest.raises(ValueError, match="URL must be a valid HTTP/HTTPS URL"):
            QdrantConfig(url="invalid-url", collection_name="test_collection")

    def test_get_client_kwargs(self):
        """Test get_client_kwargs method."""
        config = QdrantConfig(
            url="http://localhost:6333",
            collection_name="test_collection",
            api_key="test-key",
            max_results=25,
        )

        kwargs = config.get_client_kwargs()

        assert kwargs["url"] == "http://localhost:6333"
        assert kwargs["collection_name"] == "test_collection"
        assert kwargs["api_key"] == "test-key"
        assert kwargs["max_results"] == 25


class TestMilvusConfig:
    """Test MilvusConfig class."""

    def test_initialization(self):
        """Test MilvusConfig initialization."""
        config = MilvusConfig(
            host="localhost", port=19530, collection_name="test_collection"
        )

        assert config.db_type == VectorDBType.MILVUS
        assert config.host == "localhost"
        assert config.port == 19530
        assert config.collection_name == "test_collection"
        assert config.metric_type == "COSINE"  # default
        assert config.max_results == 10  # default

    def test_with_auth(self):
        """Test MilvusConfig with authentication."""
        config = MilvusConfig(
            host="localhost",
            port=19530,
            collection_name="test_collection",
            user="test_user",
            password="test_password",
        )

        assert config.user == "test_user"
        assert config.password == "test_password"

    def test_invalid_metric_type(self):
        """Test that invalid metric type raises ValueError."""
        with pytest.raises(ValueError, match="Milvus metric_type must be one of"):
            MilvusConfig(
                host="localhost",
                port=19530,
                collection_name="test_collection",
                metric_type="INVALID",
            )

    def test_get_client_kwargs(self):
        """Test get_client_kwargs method."""
        config = MilvusConfig(
            host="localhost",
            port=19530,
            collection_name="test_collection",
            user="test_user",
            password="test_password",
            max_results=30,
        )

        kwargs = config.get_client_kwargs()

        assert kwargs["host"] == "localhost"
        assert kwargs["port"] == 19530
        assert kwargs["collection_name"] == "test_collection"
        assert kwargs["user"] == "test_user"
        assert kwargs["password"] == "test_password"
        assert kwargs["max_results"] == 30


class TestConfigValidation:
    """Test configuration validation across all config types."""

    @pytest.mark.parametrize(
        ("config_class", "valid_params"),
        [
            (ChromaDBConfig, {}),
            (
                PineconeConfig,
                {"api_key": "test", "environment": "test", "index_name": "test"},
            ),
            (WeaviateConfig, {"url": "http://localhost:8080", "class_name": "Test"}),
            (QdrantConfig, {"url": "http://localhost:6333", "collection_name": "test"}),
            (
                MilvusConfig,
                {"host": "localhost", "port": 19530, "collection_name": "test"},
            ),
        ],
    )
    def test_valid_configurations(self, config_class, valid_params):
        """Test that valid configurations can be created."""
        config = config_class(**valid_params)
        assert isinstance(config, VectorDBConfig)
        assert hasattr(config, "db_type")
        assert hasattr(config, "max_results")

    def test_config_type_union(self):
        """Test VectorDBConfigType union type."""
        configs = [
            ChromaDBConfig(),
            PineconeConfig(api_key="test", environment="test", index_name="test"),
            WeaviateConfig(url="http://localhost:8080", class_name="Test"),
            QdrantConfig(url="http://localhost:6333", collection_name="test"),
            MilvusConfig(host="localhost", port=19530, collection_name="test"),
        ]

        for config in configs:
            # Test that all configs are valid VectorDBConfigType
            assert isinstance(config, VectorDBConfig)
            # Test that they have required methods
            assert hasattr(config, "get_client_kwargs")
            assert callable(config.get_client_kwargs)
