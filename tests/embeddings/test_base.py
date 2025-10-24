# pyright: reportGeneralTypeIssues=false
"""Tests for base embedding classes and data structures."""

import pytest

from pdf_vector_system.embeddings.base import (
    EmbeddingBatch,
    EmbeddingResult,
    EmbeddingService,
    EmbeddingServiceError,
)
from tests.mocks.embedding_mocks import MockEmbeddingService


class TestEmbeddingResult:
    """Test EmbeddingResult dataclass."""

    def test_creation(self):
        """Test EmbeddingResult creation."""
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        result = EmbeddingResult(
            embeddings=embeddings,
            model_name="test-model",
            embedding_dimension=3,
            processing_time=1.5,
            token_count=10,
            metadata={"test": True},
        )

        assert result.embeddings == embeddings
        assert result.model_name == "test-model"
        assert result.embedding_dimension == 3
        assert result.processing_time == 1.5
        assert result.token_count == 10
        assert result.metadata["test"] is True

    def test_text_count_property(self):
        """Test text_count property."""
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        result = EmbeddingResult(
            embeddings=embeddings, model_name="test-model", embedding_dimension=2
        )

        assert result.text_count == 3

    def test_texts_per_second_property(self):
        """Test texts_per_second property."""
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2]] * 10,
            model_name="test-model",
            embedding_dimension=2,
            processing_time=2.0,
        )

        assert result.texts_per_second == 5.0

    def test_texts_per_second_zero_time(self):
        """Test texts_per_second with zero processing time."""
        result = EmbeddingResult(
            embeddings=[[0.1, 0.2]] * 5,
            model_name="test-model",
            embedding_dimension=2,
            processing_time=0.0,
        )

        assert result.texts_per_second == 0.0


class TestEmbeddingBatch:
    """Test EmbeddingBatch dataclass."""

    def test_creation(self):
        """Test EmbeddingBatch creation."""
        texts = ["text 1", "text 2", "text 3"]
        batch = EmbeddingBatch(
            texts=texts, batch_id="batch_1", metadata={"source": "test"}
        )

        assert batch.texts == texts
        assert batch.batch_id == "batch_1"
        assert batch.metadata["source"] == "test"

    def test_size_property(self):
        """Test size property."""
        texts = ["text 1", "text 2", "text 3", "text 4"]
        batch = EmbeddingBatch(texts=texts)

        assert batch.size == 4

    def test_default_values(self):
        """Test default values."""
        texts = ["text 1", "text 2"]
        batch = EmbeddingBatch(texts=texts)

        assert batch.batch_id is not None  # Should generate UUID
        assert isinstance(batch.metadata, dict)
        assert len(batch.metadata) == 0


class TestMockEmbeddingService:
    """Test MockEmbeddingService for testing infrastructure."""

    def test_initialization(self):
        """Test mock service initialization."""
        service = MockEmbeddingService("test-model", embedding_dim=5)

        assert service.model_name == "test-model"
        assert service.embedding_dimension == 5
        assert service.call_count == 0

    def test_embed_single(self):
        """Test single text embedding."""
        service = MockEmbeddingService("test-model", embedding_dim=3)

        embedding = service.embed_single("test text")

        assert len(embedding) == 3
        assert all(isinstance(x, float) for x in embedding)
        assert service.call_count == 1

    def test_embed_texts(self):
        """Test multiple text embedding."""
        service = MockEmbeddingService("test-model", embedding_dim=4)

        texts = ["text 1", "text 2", "text 3"]
        result = service.embed_texts(texts)

        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 3
        assert all(len(emb) == 4 for emb in result.embeddings)
        assert result.model_name == "test-model"
        assert result.embedding_dimension == 4
        assert service.call_count == 1
        assert service.last_texts == texts

    def test_health_check(self):
        """Test health check."""
        service = MockEmbeddingService()

        assert service.health_check() is True

    def test_get_model_info(self):
        """Test model info retrieval."""
        service = MockEmbeddingService("test-model", embedding_dim=5)

        info = service.get_model_info()

        assert info["model_name"] == "test-model"
        assert info["embedding_dimension"] == 5
        assert info["is_mock"] is True


class TestEmbeddingServiceError:
    """Test EmbeddingServiceError exception."""

    def test_basic_exception(self):
        """Test basic exception creation."""
        error = EmbeddingServiceError("Test error message")

        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_exception_with_cause(self):
        """Test exception with underlying cause."""
        original_error = ValueError("Original error")

        try:
            raise EmbeddingServiceError("Wrapper error") from original_error
        except EmbeddingServiceError as exc:
            error = exc
        assert str(error) == "Wrapper error"
        assert error.__cause__ == original_error

    def test_exception_inheritance(self):
        """Test that EmbeddingServiceError inherits from Exception."""
        error = EmbeddingServiceError("Test")

        assert isinstance(error, Exception)
        assert isinstance(error, EmbeddingServiceError)


class TestEmbeddingServiceBase:
    """Test EmbeddingService base class behavior."""

    def test_abstract_methods(self):
        """Test that EmbeddingService cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EmbeddingService()

    def test_mock_service_implements_interface(self):
        """Test that MockEmbeddingService properly implements the interface."""
        service = MockEmbeddingService("test-model")

        # Should have all required methods
        assert hasattr(service, "embed_single")
        assert hasattr(service, "embed_texts")
        assert hasattr(service, "health_check")
        assert hasattr(service, "get_model_info")

        # Should have required properties
        assert hasattr(service, "model_name")
        assert hasattr(service, "embedding_dimension")

    def test_mock_service_method_signatures(self):
        """Test that MockEmbeddingService methods have correct signatures."""
        service = MockEmbeddingService("test-model")

        # Test embed_single signature
        embedding = service.embed_single("test")
        assert isinstance(embedding, list)

        # Test embed_texts signature
        result = service.embed_texts(["test1", "test2"])
        assert isinstance(result, EmbeddingResult)

        # Test health_check signature
        health = service.health_check()
        assert isinstance(health, bool)

        # Test get_model_info signature
        info = service.get_model_info()
        assert isinstance(info, dict)


class TestEmbeddingErrorScenarios:
    """Test error scenarios for embedding classes."""

    def test_embedding_result_invalid_embeddings(self):
        """Test EmbeddingResult with invalid embeddings."""
        with pytest.raises((ValueError, TypeError)):
            EmbeddingResult(
                embeddings="not_a_list",  # Should be List[List[float]]
                texts=["text1"],
                processing_time=1.0,
            )

        with pytest.raises((ValueError, TypeError)):
            EmbeddingResult(
                embeddings=[["not", "numbers"]],  # Should be floats
                texts=["text1"],
                processing_time=1.0,
            )

    def test_embedding_result_mismatched_lengths(self):
        """Test EmbeddingResult with mismatched embeddings and texts lengths."""
        with pytest.raises(
            ValueError, match="embeddings and texts must have the same length"
        ):
            EmbeddingResult(
                embeddings=[[1.0, 2.0], [3.0, 4.0]],  # 2 embeddings
                texts=["text1"],  # 1 text
                model_name="test-model",
                embedding_dimension=2,
                processing_time=1.0,
            )

    def test_embedding_result_negative_processing_time(self):
        """Test EmbeddingResult with negative processing time."""
        with pytest.raises(ValueError, match="processing_time must be non-negative"):
            EmbeddingResult(
                embeddings=[[1.0, 2.0]],
                texts=["text1"],
                model_name="test-model",
                embedding_dimension=2,
                processing_time=-1.0,
            )

    def test_embedding_batch_invalid_batch_size(self):
        """Test EmbeddingBatch with invalid batch size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            EmbeddingBatch(texts=["text1", "text2"], batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            EmbeddingBatch(texts=["text1", "text2"], batch_size=-5)

    def test_embedding_batch_empty_texts(self):
        """Test EmbeddingBatch with empty texts."""
        with pytest.raises(ValueError, match="texts cannot be empty"):
            EmbeddingBatch(texts=[], batch_size=10)

    def test_embedding_batch_invalid_texts_type(self):
        """Test EmbeddingBatch with invalid texts type."""
        with pytest.raises((ValueError, TypeError)):
            EmbeddingBatch(texts="not_a_list", batch_size=10)

        with pytest.raises((ValueError, TypeError)):
            # Should be strings
            EmbeddingBatch(texts=[123, 456], batch_size=10)

    def test_mock_service_embed_single_empty_text(self):
        """Test MockEmbeddingService with empty text."""
        service = MockEmbeddingService("test-model")

        with pytest.raises(ValueError, match="text cannot be empty"):
            service.embed_single("")

        with pytest.raises(ValueError, match="text cannot be empty"):
            service.embed_single("   ")  # Only whitespace

    def test_mock_service_embed_batch_empty_texts(self):
        """Test MockEmbeddingService with empty texts in batch."""
        service = MockEmbeddingService("test-model")

        with pytest.raises(ValueError, match="texts cannot be empty"):
            service.embed_texts([])

        with pytest.raises(ValueError, match="text cannot be empty"):
            service.embed_texts(["valid text", "", "another valid text"])

    def test_mock_service_embed_batch_invalid_types(self):
        """Test MockEmbeddingService with invalid text types."""
        service = MockEmbeddingService("test-model")

        with pytest.raises((ValueError, TypeError)):
            service.embed_texts([123, "valid text"])

        with pytest.raises((ValueError, TypeError)):
            service.embed_texts(["valid text", None])

    def test_embedding_service_error_inheritance(self):
        """Test EmbeddingServiceError inheritance and attributes."""
        error = EmbeddingServiceError("Test error message")

        assert isinstance(error, Exception)
        assert str(error) == "Test error message"

        # Test with additional context
        error_with_context = EmbeddingServiceError(
            "Error", {"context": "additional info"}
        )
        assert "Error" in str(error_with_context)

    def test_mock_service_invalid_configuration(self):
        """Test MockEmbeddingService with invalid configuration."""
        with pytest.raises((ValueError, TypeError)):
            MockEmbeddingService(
                model_name="test",
                dimension="not_an_int",  # Should be int
            )

        with pytest.raises(ValueError, match="dimension must be positive"):
            MockEmbeddingService("test", dimension=0)

        with pytest.raises(ValueError, match="dimension must be positive"):
            MockEmbeddingService("test", dimension=-10)

    def test_embedding_result_metadata_validation(self):
        """Test EmbeddingResult metadata validation."""
        # Valid metadata should work
        result = EmbeddingResult(
            embeddings=[[1.0, 2.0]],
            texts=["text1"],
            model_name="test-model",
            embedding_dimension=2,
            processing_time=1.0,
            metadata={"key": "value"},
        )
        assert result.metadata["key"] == "value"

        # Invalid metadata types should be handled gracefully
        with pytest.raises((ValueError, TypeError)):
            EmbeddingResult(
                embeddings=[[1.0, 2.0]],
                texts=["text1"],
                model_name="test-model",
                embedding_dimension=2,
                processing_time=1.0,
                metadata="not_a_dict",
            )
