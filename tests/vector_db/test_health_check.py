"""Tests for vector database health checking system."""

from unittest.mock import Mock, patch

import pytest

from pdf_vector_system.vector_db.config import (
    ChromaDBConfig,
    MilvusConfig,
    PineconeConfig,
    QdrantConfig,
    VectorDBType,
    WeaviateConfig,
)
from pdf_vector_system.vector_db.health_check import (
    ChromaDBHealthChecker,
    HealthChecker,
    HealthCheckResult,
    HealthStatus,
    MilvusHealthChecker,
    PineconeHealthChecker,
    QdrantHealthChecker,
    VectorDBHealthManager,
    WeaviateHealthChecker,
)


class TestHealthStatus:
    """Test HealthStatus enum."""

    def test_health_status_values(self):
        """Test that all expected health status values exist."""
        expected_values = ["healthy", "unhealthy", "degraded", "unknown"]
        actual_values = [status.value for status in HealthStatus]

        for expected in expected_values:
            assert expected in actual_values

    def test_health_status_from_string(self):
        """Test creating HealthStatus from string values."""
        assert HealthStatus("healthy") == HealthStatus.HEALTHY
        assert HealthStatus("unhealthy") == HealthStatus.UNHEALTHY
        assert HealthStatus("degraded") == HealthStatus.DEGRADED
        assert HealthStatus("unknown") == HealthStatus.UNKNOWN


class TestHealthCheckResult:
    """Test HealthCheckResult model."""

    def test_healthy_result_creation(self):
        """Test creating a healthy result."""
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            backend="chromadb",
            response_time=0.05,
            details={"connection": "ok", "collections": 5},
        )

        assert result.status == HealthStatus.HEALTHY
        assert result.backend == "chromadb"
        assert result.response_time == 0.05
        assert result.details["connection"] == "ok"
        assert result.details["collections"] == 5
        assert result.error is None

    def test_unhealthy_result_creation(self):
        """Test creating an unhealthy result."""
        result = HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            backend="pinecone",
            response_time=None,
            error="Connection timeout",
            details={"last_attempt": "2024-01-01T12:00:00"},
        )

        assert result.status == HealthStatus.UNHEALTHY
        assert result.backend == "pinecone"
        assert result.response_time is None
        assert result.error == "Connection timeout"
        assert result.details["last_attempt"] == "2024-01-01T12:00:00"

    def test_result_is_healthy(self):
        """Test is_healthy property."""
        healthy_result = HealthCheckResult(
            status=HealthStatus.HEALTHY, backend="test", response_time=0.1
        )

        unhealthy_result = HealthCheckResult(
            status=HealthStatus.UNHEALTHY,
            backend="test",
            response_time=None,
            error="Failed",
        )

        assert healthy_result.is_healthy is True
        assert unhealthy_result.is_healthy is False

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            backend="chromadb",
            response_time=0.05,
            details={"test": "value"},
        )

        result_dict = result.to_dict()

        assert result_dict["status"] == "healthy"
        assert result_dict["backend"] == "chromadb"
        assert result_dict["response_time"] == 0.05
        assert result_dict["details"]["test"] == "value"
        assert result_dict["error"] is None


class TestHealthChecker:
    """Test abstract HealthChecker base class."""

    def test_is_abstract(self):
        """Test that HealthChecker is abstract."""
        with pytest.raises(TypeError):
            HealthChecker(ChromaDBConfig())

    def test_concrete_implementation_required_methods(self):
        """Test that concrete implementations must implement required methods."""

        class IncompleteHealthChecker(HealthChecker):
            def __init__(self, config):
                super().__init__(config)

            # Missing check_health and validate_configuration methods

        with pytest.raises(TypeError):
            IncompleteHealthChecker(ChromaDBConfig())


class TestChromaDBHealthChecker:
    """Test ChromaDBHealthChecker implementation."""

    def test_initialization(self, chroma_config_test):
        """Test ChromaDBHealthChecker initialization."""
        checker = ChromaDBHealthChecker(chroma_config_test)

        assert checker.config == chroma_config_test
        assert checker.backend_name == "chromadb"
        assert checker.chroma_config == chroma_config_test

    @patch("chromadb.Client")
    def test_check_health_success(self, mock_client_class, chroma_config_test):
        """Test successful health check."""
        mock_client = Mock()
        mock_client.heartbeat.return_value = 1
        mock_client.list_collections.return_value = [Mock(name="test_collection")]
        mock_client_class.return_value = mock_client

        checker = ChromaDBHealthChecker(chroma_config_test)
        result = checker.check_health(timeout=5.0)

        assert isinstance(result, HealthCheckResult)
        assert result.status == HealthStatus.HEALTHY
        assert result.backend == "chromadb"
        assert result.response_time is not None
        assert result.response_time > 0
        assert result.error is None

    @patch("chromadb.Client")
    def test_check_health_failure(self, mock_client_class, chroma_config_test):
        """Test health check failure."""
        mock_client_class.side_effect = Exception("Connection failed")

        checker = ChromaDBHealthChecker(chroma_config_test)
        result = checker.check_health(timeout=5.0)

        assert isinstance(result, HealthCheckResult)
        assert result.status == HealthStatus.UNHEALTHY
        assert result.backend == "chromadb"
        assert result.error is not None
        assert "Connection failed" in result.error

    def test_validate_configuration_valid(self, chroma_config_test):
        """Test configuration validation with valid config."""
        checker = ChromaDBHealthChecker(chroma_config_test)
        validation = checker.validate_configuration()

        assert isinstance(validation, dict)
        assert "valid" in validation
        assert "issues" in validation
        assert validation["valid"] is True
        assert len(validation["issues"]) == 0

    def test_validate_configuration_invalid(self, vector_db_temp_dir):
        """Test configuration validation with invalid config."""
        # Create a valid config first
        valid_config = ChromaDBConfig(
            persist_directory=vector_db_temp_dir / "nonexistent" / "deeply" / "nested",
            collection_name="valid_name",
        )

        # Bypass Pydantic validation by directly setting an invalid collection name
        object.__setattr__(valid_config, "collection_name", "invalid@name!")

        checker = ChromaDBHealthChecker(valid_config)
        validation = checker.validate_configuration()

        assert validation["valid"] is False
        assert len(validation["issues"]) > 0


class TestPineconeHealthChecker:
    """Test PineconeHealthChecker implementation."""

    def test_initialization(self, pinecone_config_test):
        """Test PineconeHealthChecker initialization."""
        checker = PineconeHealthChecker(pinecone_config_test)

        assert checker.config == pinecone_config_test
        assert checker.backend_name == "pinecone"
        assert checker.pinecone_config == pinecone_config_test

    @patch("pinecone.Pinecone")
    def test_check_health_success(self, mock_pinecone_class, pinecone_config_test):
        """Test successful Pinecone health check."""
        mock_client = Mock()
        mock_indexes_response = Mock()
        mock_index = Mock()
        mock_index.name = "test-index"
        mock_indexes_response.indexes = [mock_index]
        mock_client.list_indexes.return_value = mock_indexes_response
        mock_pinecone_class.return_value = mock_client

        checker = PineconeHealthChecker(pinecone_config_test)
        result = checker.check_health(timeout=5.0)

        assert isinstance(result, HealthCheckResult)
        assert result.status == HealthStatus.HEALTHY
        assert result.backend == "pinecone"
        assert result.response_time is not None

    def test_validate_configuration_valid(self, pinecone_config_test):
        """Test Pinecone configuration validation."""
        checker = PineconeHealthChecker(pinecone_config_test)
        validation = checker.validate_configuration()

        assert isinstance(validation, dict)
        assert "valid" in validation
        assert "issues" in validation

    def test_validate_configuration_invalid(self):
        """Test Pinecone configuration validation with invalid config."""
        # Create a valid config first
        valid_config = PineconeConfig(
            api_key="test-key",
            environment="test-env",
            index_name="test-index",
        )

        # Bypass Pydantic validation by directly setting invalid values
        object.__setattr__(valid_config, "api_key", "")
        object.__setattr__(valid_config, "index_name", "")

        checker = PineconeHealthChecker(valid_config)
        validation = checker.validate_configuration()

        assert validation["valid"] is False
        assert len(validation["issues"]) > 0


class TestWeaviateHealthChecker:
    """Test WeaviateHealthChecker implementation."""

    def test_initialization(self, weaviate_config_test):
        """Test WeaviateHealthChecker initialization."""
        checker = WeaviateHealthChecker(weaviate_config_test)

        assert checker.config == weaviate_config_test
        assert checker.backend_name == "weaviate"
        assert checker.weaviate_config == weaviate_config_test

    @patch("weaviate.connect_to_custom")
    def test_check_health_success(self, mock_connect, weaviate_config_test):
        """Test successful Weaviate health check."""
        mock_client = Mock()
        mock_client.get_meta.return_value = {
            "version": "1.0.0",
            "hostname": "localhost",
        }
        mock_client.is_ready.return_value = True
        mock_client.close = Mock()
        mock_connect.return_value = mock_client

        checker = WeaviateHealthChecker(weaviate_config_test)
        result = checker.check_health(timeout=5.0)

        assert isinstance(result, HealthCheckResult)
        assert result.status == HealthStatus.HEALTHY
        assert result.backend == "weaviate"
        mock_client.close.assert_called_once()

    def test_validate_configuration_valid(self, weaviate_config_test):
        """Test Weaviate configuration validation."""
        checker = WeaviateHealthChecker(weaviate_config_test)
        validation = checker.validate_configuration()

        assert isinstance(validation, dict)
        assert validation["valid"] is True


class TestQdrantHealthChecker:
    """Test QdrantHealthChecker implementation."""

    def test_initialization(self, qdrant_config_test):
        """Test QdrantHealthChecker initialization."""
        checker = QdrantHealthChecker(qdrant_config_test)

        assert checker.config == qdrant_config_test
        assert checker.backend_name == "qdrant"
        assert checker.qdrant_config == qdrant_config_test

    @patch("qdrant_client.QdrantClient")
    def test_check_health_success(self, mock_client_class, qdrant_config_test):
        """Test successful Qdrant health check."""
        mock_client = Mock()
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        mock_client_class.return_value = mock_client

        checker = QdrantHealthChecker(qdrant_config_test)
        result = checker.check_health(timeout=5.0)

        assert isinstance(result, HealthCheckResult)
        assert result.status == HealthStatus.HEALTHY
        assert result.backend == "qdrant"


class TestMilvusHealthChecker:
    """Test MilvusHealthChecker implementation."""

    def test_initialization(self, milvus_config_test):
        """Test MilvusHealthChecker initialization."""
        checker = MilvusHealthChecker(milvus_config_test)

        assert checker.config == milvus_config_test
        assert checker.backend_name == "milvus"
        assert checker.milvus_config == milvus_config_test

    def test_validate_configuration_valid(self, milvus_config_test):
        """Test Milvus configuration validation."""
        checker = MilvusHealthChecker(milvus_config_test)
        validation = checker.validate_configuration()

        assert isinstance(validation, dict)
        assert validation["valid"] is True

    def test_validate_configuration_invalid_metric(self):
        """Test Milvus configuration validation with invalid metric."""
        # Create a valid config first
        valid_config = MilvusConfig(
            host="localhost",
            port=19530,
            collection_name="test",
            metric_type="COSINE",
        )

        # Bypass Pydantic validation by directly setting an invalid metric
        object.__setattr__(valid_config, "metric_type", "INVALID_METRIC")

        checker = MilvusHealthChecker(valid_config)
        validation = checker.validate_configuration()

        assert validation["valid"] is False
        assert any("Invalid metric type" in issue for issue in validation["issues"])


class TestVectorDBHealthManager:
    """Test VectorDBHealthManager class."""

    def test_initialization(self):
        """Test VectorDBHealthManager initialization."""
        manager = VectorDBHealthManager()

        assert hasattr(manager, "_HEALTH_CHECKERS")
        assert isinstance(manager._HEALTH_CHECKERS, dict)

        # Check that all expected backends have health checkers
        expected_backends = [
            VectorDBType.CHROMADB,
            VectorDBType.PINECONE,
            VectorDBType.WEAVIATE,
            VectorDBType.QDRANT,
            VectorDBType.MILVUS,
        ]

        for backend in expected_backends:
            assert backend in manager._HEALTH_CHECKERS

    def test_get_health_checker(self, chroma_config_test):
        """Test getting health checker for specific backend."""
        manager = VectorDBHealthManager()
        checker = manager.get_health_checker(chroma_config_test)

        assert isinstance(checker, ChromaDBHealthChecker)
        assert checker.config == chroma_config_test

    def test_get_health_checker_unsupported(self):
        """Test getting health checker for unsupported backend."""
        manager = VectorDBHealthManager()

        mock_config = Mock()
        mock_config.db_type = "unsupported_backend"

        with pytest.raises(ValueError, match="No health checker available"):
            manager.get_health_checker(mock_config)

    def test_check_all_backends(self):
        """Test checking health of all available backends."""
        manager = VectorDBHealthManager()

        configs = [
            ChromaDBConfig(),
            PineconeConfig(api_key="test", environment="test", index_name="test"),
            WeaviateConfig(url="http://localhost:8080", class_name="Test"),
            QdrantConfig(url="http://localhost:6333", collection_name="test"),
            MilvusConfig(host="localhost", port=19530, collection_name="test"),
        ]

        with patch.object(manager, "check_health") as mock_check:
            mock_check.return_value = HealthCheckResult(
                status=HealthStatus.HEALTHY, backend="test", response_time=0.1
            )

            results = manager.check_all_backends(configs, timeout=5.0)

            assert len(results) == len(configs)
            assert all(isinstance(result, HealthCheckResult) for result in results)

    def test_check_health_with_timeout(self, chroma_config_test):
        """Test health check with timeout."""
        manager = VectorDBHealthManager()

        with patch.object(ChromaDBHealthChecker, "check_health") as mock_check:
            mock_check.return_value = HealthCheckResult(
                status=HealthStatus.HEALTHY, backend="chromadb", response_time=0.05
            )

            result = manager.check_health(chroma_config_test, timeout=10.0)

            assert isinstance(result, HealthCheckResult)
            mock_check.assert_called_once_with(timeout=10.0)

    def test_validate_all_configurations(self):
        """Test validating all backend configurations."""
        manager = VectorDBHealthManager()

        configs = [
            ChromaDBConfig(),
            PineconeConfig(api_key="test", environment="test", index_name="test"),
        ]

        validations = manager.validate_all_configurations(configs)

        assert len(validations) == len(configs)
        assert all("valid" in validation for validation in validations)
        assert all("backend" in validation for validation in validations)
