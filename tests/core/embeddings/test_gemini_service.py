"""Tests for Google Gemini embedding service."""

from unittest.mock import Mock, patch

import pytest

from pdf_vector_system.core.embeddings.base import EmbeddingServiceError
from pdf_vector_system.core.embeddings.gemini_service import GeminiEmbeddingService


class TestGeminiEmbeddingService:
    """Test cases for GeminiEmbeddingService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.api_key = "test-api-key"
        self.model_name = "gemini-embedding-001"

    def test_init_google_ai_api(self):
        """Test initialization with Google AI API."""
        service = GeminiEmbeddingService(
            model_name=self.model_name, api_key=self.api_key, use_vertex_ai=False
        )

        assert service.model_name == self.model_name
        assert service.api_key == self.api_key
        assert not service.use_vertex_ai
        assert service.base_url == "https://generativelanguage.googleapis.com/v1beta"

    def test_init_vertex_ai(self):
        """Test initialization with Vertex AI."""
        project_id = "test-project"
        location = "us-central1"

        service = GeminiEmbeddingService(
            model_name=self.model_name,
            api_key=self.api_key,
            project_id=project_id,
            location=location,
            use_vertex_ai=True,
        )

        assert service.use_vertex_ai
        assert service.project_id == project_id
        assert service.location == location
        expected_url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models"
        assert service.base_url == expected_url

    def test_init_missing_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="Google Gemini API key is required"):
                GeminiEmbeddingService(model_name=self.model_name)

    def test_init_vertex_ai_missing_project_id(self):
        """Test initialization fails for Vertex AI without project ID."""
        with pytest.raises(
            ValueError, match="project_id is required when using Vertex AI"
        ):
            GeminiEmbeddingService(
                model_name=self.model_name, api_key=self.api_key, use_vertex_ai=True
            )

    @patch("pdf_vector_system.embeddings.gemini_service.requests.post")
    def test_embed_single_google_ai(self, mock_post):
        """Test single text embedding with Google AI API."""
        # Mock response
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {"embedding": {"values": [0.1, 0.2, 0.3]}}
        mock_post.return_value = mock_response

        service = GeminiEmbeddingService(
            model_name=self.model_name, api_key=self.api_key, use_vertex_ai=False
        )

        result = service.embed_single("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_post.assert_called_once()

        # Verify request format
        call_args = mock_post.call_args
        assert "x-goog-api-key" in call_args[1]["headers"]
        assert call_args[1]["headers"]["x-goog-api-key"] == self.api_key

    @patch("pdf_vector_system.embeddings.gemini_service.requests.post")
    def test_embed_texts_batch_google_ai(self, mock_post):
        """Test batch text embedding with Google AI API."""
        # Mock response for batch request
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "embeddings": [{"values": [0.1, 0.2, 0.3]}, {"values": [0.4, 0.5, 0.6]}]
        }
        mock_post.return_value = mock_response

        service = GeminiEmbeddingService(
            model_name=self.model_name, api_key=self.api_key, use_vertex_ai=False
        )

        texts = ["text 1", "text 2"]
        result = service.embed_texts(texts)

        assert len(result.embeddings) == 2
        assert result.embeddings[0] == [0.1, 0.2, 0.3]
        assert result.embeddings[1] == [0.4, 0.5, 0.6]
        assert result.model_name == self.model_name
        assert result.embedding_dimension == 3

    @patch("pdf_vector_system.embeddings.gemini_service.requests.post")
    def test_embed_texts_vertex_ai(self, mock_post):
        """Test text embedding with Vertex AI."""
        # Mock response for Vertex AI
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "predictions": [
                {"embeddings": {"values": [0.1, 0.2, 0.3]}},
                {"values": [0.4, 0.5, 0.6]},
            ]
        }
        mock_post.return_value = mock_response

        service = GeminiEmbeddingService(
            model_name=self.model_name,
            api_key=self.api_key,
            project_id="test-project",
            location="us-central1",
            use_vertex_ai=True,
        )

        texts = ["text 1", "text 2"]
        result = service.embed_texts(texts)

        assert len(result.embeddings) == 2
        assert result.embeddings[0] == [0.1, 0.2, 0.3]
        assert result.embeddings[1] == [0.4, 0.5, 0.6]

        # Verify Vertex AI request format
        call_args = mock_post.call_args
        assert "Authorization" in call_args[1]["headers"]
        assert call_args[1]["headers"]["Authorization"] == f"Bearer {self.api_key}"

    @patch("pdf_vector_system.embeddings.gemini_service.requests.post")
    def test_api_error_handling(self, mock_post):
        """Test API error handling."""
        # Mock error response
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 429
        mock_response.text = "Rate limit exceeded"
        mock_post.return_value = mock_response

        service = GeminiEmbeddingService(
            model_name=self.model_name, api_key=self.api_key, use_vertex_ai=False
        )

        with pytest.raises(EmbeddingServiceError):
            service.embed_single("test text")

    def test_get_embedding_dimension(self):
        """Test getting embedding dimension."""
        service = GeminiEmbeddingService(
            model_name=self.model_name, api_key=self.api_key, use_vertex_ai=False
        )

        # Should return dimension from model config
        assert service.get_embedding_dimension() == 768

    def test_get_model_info(self):
        """Test getting model information."""
        service = GeminiEmbeddingService(
            model_name=self.model_name,
            api_key=self.api_key,
            project_id="test-project",
            location="us-central1",
            use_vertex_ai=True,
        )

        info = service.get_model_info()

        assert info["model_name"] == self.model_name
        assert info["api_provider"] == "google_gemini"
        assert info["endpoint_type"] == "vertex_ai"
        assert info["project_id"] == "test-project"
        assert info["location"] == "us-central1"
        assert info["has_api_key"] is True

    @patch("pdf_vector_system.embeddings.gemini_service.requests.post")
    def test_health_check_success(self, mock_post):
        """Test successful health check."""
        # Mock successful response
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "embedding": {"values": [0.1] * 768}  # Correct dimension
        }
        mock_post.return_value = mock_response

        service = GeminiEmbeddingService(
            model_name=self.model_name, api_key=self.api_key, use_vertex_ai=False
        )

        assert service.health_check() is True

    @patch("pdf_vector_system.embeddings.gemini_service.requests.post")
    def test_health_check_failure(self, mock_post):
        """Test failed health check."""
        # Mock error response
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_post.return_value = mock_response

        service = GeminiEmbeddingService(
            model_name=self.model_name, api_key=self.api_key, use_vertex_ai=False
        )

        assert service.health_check() is False

    def test_empty_texts_validation(self):
        """Test validation of empty texts."""
        service = GeminiEmbeddingService(
            model_name=self.model_name, api_key=self.api_key, use_vertex_ai=False
        )

        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            service.embed_texts([])

        with pytest.raises(ValueError, match="Text cannot be empty"):
            service.embed_single("")

    def test_extract_embeddings_no_data(self):
        """Test embedding extraction with no data."""
        service = GeminiEmbeddingService(
            model_name=self.model_name, api_key=self.api_key, use_vertex_ai=False
        )

        with pytest.raises(
            EmbeddingServiceError, match="No embeddings found in API response"
        ):
            service._extract_embeddings({})


if __name__ == "__main__":
    pytest.main([__file__])
