"""Tests for configuration classes."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from pdf_vector_system.core.config.settings import (
    ChromaDBConfig,
    Config,
    EmbeddingConfig,
    EmbeddingModelType,
    LoggingConfig,
    LogLevel,
    PDFConfig,
    TextProcessingConfig,
)


class TestPDFConfig:
    """Test PDFConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PDFConfig()

        assert config.max_file_size_mb == 100
        assert config.timeout_seconds == 300
        assert config.extract_images is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PDFConfig(
            max_file_size_mb=50, timeout_seconds=120, extract_images=True
        )

        assert config.max_file_size_mb == 50
        assert config.timeout_seconds == 120
        assert config.extract_images is True

    def test_validation_max_file_size(self):
        """Test validation of max_file_size_mb."""
        with pytest.raises(ValueError, match="max_file_size_mb must be positive"):
            PDFConfig(max_file_size_mb=0)

        with pytest.raises(ValueError, match="max_file_size_mb must be positive"):
            PDFConfig(max_file_size_mb=-10)

    def test_validation_timeout(self):
        """Test validation of timeout_seconds."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            PDFConfig(timeout_seconds=0)

        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            PDFConfig(timeout_seconds=-5)

    def test_valid_edge_cases(self):
        """Test valid edge case values."""
        config = PDFConfig(max_file_size_mb=1, timeout_seconds=1)
        assert config.max_file_size_mb == 1
        assert config.timeout_seconds == 1


class TestTextProcessingConfig:
    """Test TextProcessingConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TextProcessingConfig()

        assert config.chunk_size == 1000
        assert config.chunk_overlap == 100
        assert config.min_chunk_size == 50
        assert config.separators == ["\n\n", "\n", " ", ""]

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TextProcessingConfig(
            chunk_size=500, chunk_overlap=50, min_chunk_size=25, separators=["\n", " "]
        )

        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.min_chunk_size == 25
        assert config.separators == ["\n", " "]

    def test_validation_chunk_size(self):
        """Test validation of chunk_size."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextProcessingConfig(chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextProcessingConfig(chunk_size=-100)

    def test_validation_chunk_overlap(self):
        """Test validation of chunk_overlap."""
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            TextProcessingConfig(chunk_overlap=-10)

    def test_validation_min_chunk_size(self):
        """Test validation of min_chunk_size."""
        with pytest.raises(ValueError, match="min_chunk_size must be positive"):
            TextProcessingConfig(min_chunk_size=0)

        with pytest.raises(ValueError, match="min_chunk_size must be positive"):
            TextProcessingConfig(min_chunk_size=-5)

    def test_validation_overlap_vs_chunk_size(self):
        """Test validation that overlap is less than chunk size."""
        with pytest.raises(
            ValueError, match="chunk_overlap cannot be larger than chunk_size"
        ):
            TextProcessingConfig(chunk_size=100, chunk_overlap=100)

        with pytest.raises(
            ValueError, match="chunk_overlap cannot be larger than chunk_size"
        ):
            TextProcessingConfig(chunk_size=100, chunk_overlap=150)

    def test_valid_edge_cases(self):
        """Test valid edge case values."""
        config = TextProcessingConfig(
            chunk_size=100, chunk_overlap=99, min_chunk_size=1
        )
        assert config.chunk_size == 100
        assert config.chunk_overlap == 99
        assert config.min_chunk_size == 1


class TestEmbeddingConfig:
    """Test EmbeddingConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EmbeddingConfig()

        assert config.model_type == EmbeddingModelType.SENTENCE_TRANSFORMERS
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.batch_size == 32
        assert config.openai_api_key is None

    def test_sentence_transformers_config(self):
        """Test sentence transformers configuration."""
        config = EmbeddingConfig(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
            model_name="all-mpnet-base-v2",
            batch_size=16,
        )

        assert config.model_type == EmbeddingModelType.SENTENCE_TRANSFORMERS
        assert config.model_name == "all-mpnet-base-v2"
        assert config.batch_size == 16

    def test_openai_config(self):
        """Test OpenAI configuration."""
        config = EmbeddingConfig(
            model_type=EmbeddingModelType.OPENAI,
            model_name="text-embedding-3-small",
            batch_size=100,
            openai_api_key="test-key",
        )

        assert config.model_type == EmbeddingModelType.OPENAI
        assert config.model_name == "text-embedding-3-small"
        assert config.batch_size == 100
        assert config.openai_api_key == "test-key"

    def test_validation_batch_size(self):
        """Test validation of batch_size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            EmbeddingConfig(batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            EmbeddingConfig(batch_size=-10)

    def test_validation_model_name(self):
        """Test validation of model_name."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            EmbeddingConfig(model_name="")

        with pytest.raises(ValueError, match="model_name cannot be empty"):
            EmbeddingConfig(model_name="   ")


class TestChromaDBConfig:
    """Test ChromaDBConfig class."""

    def test_default_values(self, temp_dir):
        """Test default configuration values."""
        config = ChromaDBConfig()

        assert config.persist_directory == Path("./chroma_db")
        assert config.collection_name == "pdf_documents"
        assert config.max_results == 100

    def test_custom_values(self, temp_dir):
        """Test custom configuration values."""
        config = ChromaDBConfig(
            persist_directory=temp_dir / "custom_db",
            collection_name="custom_collection",
            max_results=50,
        )

        assert config.persist_directory == temp_dir / "custom_db"
        assert config.collection_name == "custom_collection"
        assert config.max_results == 50

    def test_validation_max_results(self):
        """Test validation of max_results."""
        with pytest.raises(ValueError, match="max_results must be positive"):
            ChromaDBConfig(max_results=0)

        with pytest.raises(ValueError, match="max_results must be positive"):
            ChromaDBConfig(max_results=-10)

    def test_validation_collection_name(self):
        """Test validation of collection_name."""
        with pytest.raises(ValueError, match="collection_name cannot be empty"):
            ChromaDBConfig(collection_name="")

        with pytest.raises(ValueError, match="collection_name cannot be empty"):
            ChromaDBConfig(collection_name="   ")


class TestLoggingConfig:
    """Test LoggingConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LoggingConfig()

        assert config.level == LogLevel.INFO
        assert config.file_path is None
        assert config.format == "{time} | {level} | {name} | {message}"
        assert config.rotation == "10 MB"
        assert config.retention == "30 days"

    def test_custom_values(self, temp_dir):
        """Test custom configuration values."""
        log_file = temp_dir / "test.log"
        config = LoggingConfig(
            level=LogLevel.DEBUG,
            file_path=log_file,
            format="{message}",
            rotation="1 MB",
            retention="7 days",
        )

        assert config.level == LogLevel.DEBUG
        assert config.file_path == log_file
        assert config.format == "{message}"
        assert config.rotation == "1 MB"
        assert config.retention == "7 days"

    def test_all_log_levels(self):
        """Test all log level values."""
        for level in LogLevel:
            config = LoggingConfig(level=level)
            assert config.level == level


class TestConfig:
    """Test main Config class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = Config()

        assert isinstance(config.pdf, PDFConfig)
        assert isinstance(config.text_processing, TextProcessingConfig)
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.chroma_db, ChromaDBConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert config.debug is False
        assert config.max_workers == 4

    def test_custom_subconfigs(self, temp_dir):
        """Test custom sub-configurations."""
        pdf_config = PDFConfig(max_file_size_mb=50)
        text_config = TextProcessingConfig(chunk_size=500)
        embedding_config = EmbeddingConfig(batch_size=16)
        chroma_config = ChromaDBConfig(persist_directory=temp_dir)
        logging_config = LoggingConfig(level=LogLevel.DEBUG)

        config = Config(
            pdf=pdf_config,
            text_processing=text_config,
            embedding=embedding_config,
            chroma_db=chroma_config,
            logging=logging_config,
            debug=True,
            max_workers=8,
        )

        assert config.pdf.max_file_size_mb == 50
        assert config.text_processing.chunk_size == 500
        assert config.embedding.batch_size == 16
        assert config.chroma_db.persist_directory == temp_dir
        assert config.logging.level == LogLevel.DEBUG
        assert config.debug is True
        assert config.max_workers == 8

    def test_validation_max_workers(self):
        """Test validation of max_workers."""
        with pytest.raises(ValueError, match="max_workers must be positive"):
            Config(max_workers=0)

        with pytest.raises(ValueError, match="max_workers must be positive"):
            Config(max_workers=-2)

    def test_model_post_init_openai_key(self, test_env_vars):
        """Test post-initialization setup for OpenAI API key."""
        # Test with OpenAI embedding type
        embedding_config = EmbeddingConfig(
            model_type=EmbeddingModelType.OPENAI, openai_api_key=None
        )

        config = Config(embedding=embedding_config)

        # Should pick up API key from environment
        assert config.embedding.openai_api_key == "test-api-key"

    def test_model_post_init_debug_mode(self, temp_dir):
        """Test post-initialization setup for debug mode."""
        config = Config(debug=True)

        # Debug mode should set logging level to DEBUG
        assert config.logging.level == LogLevel.DEBUG

    def test_model_post_init_directory_creation(self, temp_dir):
        """Test post-initialization directory creation."""
        chroma_dir = temp_dir / "test_chroma"
        chroma_config = ChromaDBConfig(persist_directory=chroma_dir)

        Config(chroma_db=chroma_config)

        # Directory should be created
        assert chroma_dir.exists()
        assert chroma_dir.is_dir()

    @patch.dict(os.environ, {}, clear=True)
    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        # Set environment variables
        env_vars = {
            "PDF__MAX_FILE_SIZE_MB": "25",
            "TEXT_PROCESSING__CHUNK_SIZE": "800",
            "EMBEDDING__BATCH_SIZE": "64",
            "CHROMA_DB__MAX_RESULTS": "200",
            "DEBUG": "true",
            "MAX_WORKERS": "6",
        }

        with patch.dict(os.environ, env_vars):
            config = Config()

            assert config.pdf.max_file_size_mb == 25
            assert config.text_processing.chunk_size == 800
            assert config.embedding.batch_size == 64
            assert config.chroma_db.max_results == 200
            assert config.debug is True
            assert config.max_workers == 6

    def test_config_serialization(self):
        """Test configuration serialization."""
        config = Config()

        # Should be able to convert to dict
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert "pdf" in config_dict
        assert "text_processing" in config_dict
        assert "embedding" in config_dict
        assert "chroma_db" in config_dict
        assert "logging" in config_dict

    def test_config_from_dict(self):
        """Test creating configuration from dictionary."""
        config_dict = {
            "pdf": {"max_file_size_mb": 75},
            "text_processing": {"chunk_size": 600},
            "embedding": {"batch_size": 24},
            "debug": True,
            "max_workers": 3,
        }

        config = Config(**config_dict)

        assert config.pdf.max_file_size_mb == 75
        assert config.text_processing.chunk_size == 600
        assert config.embedding.batch_size == 24
        assert config.debug is True
        assert config.max_workers == 3


class TestConfigErrorScenarios:
    """Test error scenarios for configuration classes."""

    def test_pdf_config_invalid_file_size(self):
        """Test PDFConfig with invalid file size."""
        with pytest.raises(ValueError, match="max_file_size_mb must be positive"):
            PDFConfig(max_file_size_mb=-1)

        with pytest.raises(ValueError, match="max_file_size_mb must be positive"):
            PDFConfig(max_file_size_mb=0)

    def test_pdf_config_invalid_timeout(self):
        """Test PDFConfig with invalid timeout."""
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            PDFConfig(timeout_seconds=-10)

        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            PDFConfig(timeout_seconds=0)

    def test_text_processing_config_invalid_chunk_size(self):
        """Test TextProcessingConfig with invalid chunk size."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextProcessingConfig(chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextProcessingConfig(chunk_size=-100)

    def test_text_processing_config_invalid_overlap(self):
        """Test TextProcessingConfig with invalid overlap."""
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            TextProcessingConfig(chunk_overlap=-50)

        # Overlap larger than chunk size should also be invalid
        with pytest.raises(
            ValueError, match="chunk_overlap cannot be larger than chunk_size"
        ):
            TextProcessingConfig(chunk_size=100, chunk_overlap=150)

    def test_embedding_config_invalid_batch_size(self):
        """Test EmbeddingConfig with invalid batch size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            EmbeddingConfig(batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            EmbeddingConfig(batch_size=-5)

    def test_embedding_config_invalid_model_type(self):
        """Test EmbeddingConfig with invalid model type."""
        with pytest.raises(ValueError):
            EmbeddingConfig(model_type="invalid_model_type")

    def test_chromadb_config_empty_collection_name(self):
        """Test ChromaDBConfig with empty collection name."""
        with pytest.raises(ValueError, match="collection_name cannot be empty"):
            ChromaDBConfig(collection_name="")

        with pytest.raises(ValueError, match="collection_name cannot be empty"):
            ChromaDBConfig(collection_name="   ")  # Only whitespace

    def test_chromadb_config_invalid_max_results(self):
        """Test ChromaDBConfig with invalid max_results."""
        with pytest.raises(ValueError, match="max_results must be positive"):
            ChromaDBConfig(max_results=0)

        with pytest.raises(ValueError, match="max_results must be positive"):
            ChromaDBConfig(max_results=-10)

    def test_logging_config_invalid_level(self):
        """Test LoggingConfig with invalid level."""
        with pytest.raises(ValueError):
            LoggingConfig(level="INVALID_LEVEL")

    def test_config_with_invalid_nested_values(self):
        """Test Config creation with invalid nested values."""
        with pytest.raises(ValueError):
            Config(
                pdf=PDFConfig(max_file_size_mb=-1),  # Invalid
                text_processing=TextProcessingConfig(chunk_size=1000),  # Valid
            )

    def test_config_type_validation_errors(self):
        """Test Config with wrong types."""
        with pytest.raises((TypeError, ValueError)):
            Config(pdf="not_a_config")

        with pytest.raises((TypeError, ValueError)):
            Config(text_processing=123)

        with pytest.raises((TypeError, ValueError)):
            Config(embedding=["invalid", "type"])

    def test_config_from_invalid_dict(self):
        """Test Config creation from invalid dictionary."""
        invalid_config_dict = {
            "pdf": {"max_file_size_mb": -50},  # Invalid
            "text_processing": {"chunk_size": 1000},  # Valid
        }

        with pytest.raises(ValueError):
            Config(**invalid_config_dict)

    def test_config_partial_invalid_environment_vars(self, temp_dir):
        """Test Config with partially invalid environment variables."""
        env_vars = {
            "PDF__MAX_FILE_SIZE_MB": "-10",  # Invalid
            "TEXT_PROCESSING__CHUNK_SIZE": "1000",  # Valid
        }

        with patch.dict(os.environ, env_vars), pytest.raises(ValueError):
            Config()

    def test_config_invalid_file_path_types(self, temp_dir):
        """Test Config with invalid file path types."""
        with pytest.raises((TypeError, ValueError)):
            LoggingConfig(file_path=123)  # Should be Path or str

        with pytest.raises((TypeError, ValueError)):
            ChromaDBConfig(persist_directory=["not", "a", "path"])

    def test_config_missing_required_environment_vars(self):
        """Test Config behavior when required environment variables are missing."""
        # Clear all environment variables
        with patch.dict(os.environ, {}, clear=True):
            # Should still work with defaults
            config = Config()
            assert config is not None

    def test_config_malformed_environment_vars(self):
        """Test Config with malformed environment variable values."""
        env_vars = {
            "PDF__MAX_FILE_SIZE_MB": "not_a_number",
            "TEXT_PROCESSING__CHUNK_SIZE": "also_not_a_number",
            "DEBUG": "not_a_boolean",
        }

        with patch.dict(os.environ, env_vars):
            with pytest.raises((ValueError, TypeError)):
                Config()
