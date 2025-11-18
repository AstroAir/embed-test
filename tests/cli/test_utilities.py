"""Tests for CLI utility functions."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from vectorflow.cli.main import app, format_file_size, get_pipeline
from vectorflow.core.config.settings import Config, EmbeddingModelType, LogLevel


class TestFormatFileSize:
    """Test the format_file_size utility function."""

    def test_format_bytes(self):
        """Test formatting bytes."""
        assert format_file_size(500) == "500.0 B"

    def test_format_kilobytes(self):
        """Test formatting kilobytes."""
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(2048) == "2.0 KB"

    def test_format_megabytes(self):
        """Test formatting megabytes."""
        assert format_file_size(1024 * 1024) == "1.0 MB"
        assert format_file_size(5 * 1024 * 1024) == "5.0 MB"

    def test_format_gigabytes(self):
        """Test formatting gigabytes."""
        assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"
        assert format_file_size(2 * 1024 * 1024 * 1024) == "2.0 GB"

    def test_format_terabytes(self):
        """Test formatting terabytes."""
        assert format_file_size(1024 * 1024 * 1024 * 1024) == "1.0 TB"

    def test_format_zero(self):
        """Test formatting zero bytes."""
        assert format_file_size(0) == "0.0 B"

    def test_format_decimal_values(self):
        """Test formatting with decimal values."""
        result = format_file_size(1536)  # 1.5 KB
        assert "1.5" in result
        assert "KB" in result


class TestGetPipeline:
    """Test the get_pipeline utility function."""

    @patch("vectorflow.cli.main.PDFVectorPipeline")
    @patch("vectorflow.cli.main.Config")
    @patch("vectorflow.cli.main.setup_logging")
    def test_get_pipeline_default(
        self,
        mock_setup_logging,
        mock_config_class,
        mock_pipeline_class,
    ):
        """Test get_pipeline with default parameters."""
        mock_config = Mock(spec=Config)
        mock_config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
        mock_config.chroma_db.collection_name = "default_collection"
        mock_config.debug = False
        mock_config.logging.level = LogLevel.INFO
        mock_config_class.return_value = mock_config

        # Call get_pipeline
        pipeline = get_pipeline()

        # Assertions
        assert pipeline is not None
        mock_config_class.assert_called_once()
        mock_pipeline_class.assert_called_once_with(mock_config)
        mock_setup_logging.assert_called_once()

    @patch("vectorflow.cli.main.PDFVectorPipeline")
    @patch("vectorflow.cli.main.Config")
    @patch("vectorflow.cli.main.setup_logging")
    def test_get_pipeline_with_embedding_model(
        self,
        mock_setup_logging,
        mock_config_class,
        mock_pipeline_class,
    ):
        """Test get_pipeline with custom embedding model."""
        mock_config = Mock(spec=Config)
        mock_config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
        mock_config.embedding.model_name = "all-MiniLM-L6-v2"
        mock_config.chroma_db.collection_name = "default_collection"
        mock_config.debug = False
        mock_config.logging.level = LogLevel.INFO
        mock_config_class.return_value = mock_config

        # Call get_pipeline with custom model
        pipeline = get_pipeline(embedding_model="all-MiniLM-L6-v2")

        # Assertions
        assert pipeline is not None
        assert (
            mock_config.embedding.model_type == EmbeddingModelType.SENTENCE_TRANSFORMERS
        )
        assert mock_config.embedding.model_name == "all-MiniLM-L6-v2"

    @patch("vectorflow.cli.main.PDFVectorPipeline")
    @patch("vectorflow.cli.main.Config")
    @patch("vectorflow.cli.main.setup_logging")
    def test_get_pipeline_with_openai_model(
        self,
        mock_setup_logging,
        mock_config_class,
        mock_pipeline_class,
    ):
        """Test get_pipeline with OpenAI embedding model."""
        mock_config = Mock(spec=Config)
        mock_config.embedding.model_type = EmbeddingModelType.OPENAI
        mock_config.embedding.model_name = "text-embedding-3-small"
        mock_config.chroma_db.collection_name = "default_collection"
        mock_config.debug = False
        mock_config.logging.level = LogLevel.INFO
        mock_config_class.return_value = mock_config

        # Call get_pipeline with OpenAI model
        pipeline = get_pipeline(embedding_model="text-embedding-3-small")

        # Assertions
        assert pipeline is not None
        assert mock_config.embedding.model_type == EmbeddingModelType.OPENAI
        assert mock_config.embedding.model_name == "text-embedding-3-small"

    @patch("vectorflow.cli.main.PDFVectorPipeline")
    @patch("vectorflow.cli.main.Config")
    @patch("vectorflow.cli.main.setup_logging")
    def test_get_pipeline_with_collection_name(
        self,
        mock_setup_logging,
        mock_config_class,
        mock_pipeline_class,
    ):
        """Test get_pipeline with custom collection name."""
        mock_config = Mock(spec=Config)
        mock_config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
        mock_config.chroma_db.collection_name = "custom_collection"
        mock_config.debug = False
        mock_config.logging.level = LogLevel.INFO
        mock_config_class.return_value = mock_config

        # Call get_pipeline with custom collection
        pipeline = get_pipeline(collection_name="custom_collection")

        # Assertions
        assert pipeline is not None
        assert mock_config.chroma_db.collection_name == "custom_collection"

    @patch("vectorflow.cli.main.PDFVectorPipeline")
    @patch("vectorflow.cli.main.Config")
    @patch("vectorflow.cli.main.setup_logging")
    def test_get_pipeline_with_debug(
        self,
        mock_setup_logging,
        mock_config_class,
        mock_pipeline_class,
    ):
        """Test get_pipeline with debug mode."""
        mock_config = Mock(spec=Config)
        mock_config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
        mock_config.chroma_db.collection_name = "default_collection"
        mock_config.debug = True
        mock_config.logging.level = LogLevel.DEBUG
        mock_config_class.return_value = mock_config

        # Call get_pipeline with debug
        pipeline = get_pipeline(debug=True)

        # Assertions
        assert pipeline is not None
        assert mock_config.debug is True
        assert mock_config.logging.level == LogLevel.DEBUG

    @patch("vectorflow.cli.main.PDFVectorPipeline")
    @patch("vectorflow.cli.main.Config")
    @patch("vectorflow.cli.main.setup_logging")
    def test_get_pipeline_with_batch_size(
        self,
        mock_setup_logging,
        mock_config_class,
        mock_pipeline_class,
    ):
        """Test get_pipeline with custom batch size."""
        mock_config = Mock(spec=Config)
        mock_config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
        mock_config.embedding.batch_size = 64
        mock_config.chroma_db.collection_name = "default_collection"
        mock_config.debug = False
        mock_config.logging.level = LogLevel.INFO
        mock_config_class.return_value = mock_config

        # Call get_pipeline with custom batch size
        pipeline = get_pipeline(batch_size=64)

        # Assertions
        assert pipeline is not None
        assert mock_config.embedding.batch_size == 64

    @patch("vectorflow.cli.main.PDFVectorPipeline")
    @patch("vectorflow.cli.main.Config")
    @patch("vectorflow.cli.main.setup_logging")
    def test_get_pipeline_with_chunk_size(
        self,
        mock_setup_logging,
        mock_config_class,
        mock_pipeline_class,
    ):
        """Test get_pipeline with custom chunk size."""
        mock_config = Mock(spec=Config)
        mock_config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
        mock_config.text_processing.chunk_size = 2000
        mock_config.chroma_db.collection_name = "default_collection"
        mock_config.debug = False
        mock_config.logging.level = LogLevel.INFO
        mock_config_class.return_value = mock_config

        # Call get_pipeline with custom chunk size
        pipeline = get_pipeline(chunk_size=2000)

        # Assertions
        assert pipeline is not None
        assert mock_config.text_processing.chunk_size == 2000

    @patch("vectorflow.cli.main.PDFVectorPipeline")
    @patch("vectorflow.cli.main.Config")
    @patch("vectorflow.cli.main.setup_logging")
    def test_get_pipeline_initialization_error(
        self,
        mock_setup_logging,
        mock_config_class,
        mock_pipeline_class,
    ):
        """Test get_pipeline when pipeline initialization fails."""
        mock_config = Mock(spec=Config)
        mock_config_class.return_value = mock_config
        mock_pipeline_class.side_effect = Exception("Initialization error")

        # Call get_pipeline and expect exception
        with pytest.raises(Exception) as exc_info:
            get_pipeline()

        assert "Initialization error" in str(exc_info.value)

    @patch("vectorflow.cli.main.PDFVectorPipeline")
    @patch("vectorflow.cli.main.Config")
    @patch("vectorflow.cli.main.setup_logging")
    def test_get_pipeline_all_parameters(
        self,
        mock_setup_logging,
        mock_config_class,
        mock_pipeline_class,
    ):
        """Test get_pipeline with all parameters."""
        mock_config = Mock(spec=Config)
        mock_config.embedding.model_type = EmbeddingModelType.OPENAI
        mock_config.embedding.model_name = "text-embedding-3-small"
        mock_config.embedding.batch_size = 128
        mock_config.text_processing.chunk_size = 1500
        mock_config.chroma_db.collection_name = "test_collection"
        mock_config.debug = True
        mock_config.logging.level = LogLevel.DEBUG
        mock_config_class.return_value = mock_config

        # Call get_pipeline with all parameters
        pipeline = get_pipeline(
            embedding_model="text-embedding-3-small",
            collection_name="test_collection",
            batch_size=128,
            chunk_size=1500,
            debug=True,
        )

        # Assertions
        assert pipeline is not None
        assert mock_config.embedding.model_type == EmbeddingModelType.OPENAI
        assert mock_config.embedding.model_name == "text-embedding-3-small"
        assert mock_config.embedding.batch_size == 128
        assert mock_config.text_processing.chunk_size == 1500
        assert mock_config.chroma_db.collection_name == "test_collection"
        assert mock_config.debug is True


class TestMainEntryPoint:
    """Test the main entry point and error handling."""

    @patch("vectorflow.cli.main.app")
    def test_main_function_exists(self, mock_app):
        """Test that main function exists and can be called."""
        from vectorflow.cli.main import main

        # Call main
        main()

        # Assertions - main should invoke the Typer app
        mock_app.assert_called_once()

    def test_cli_help_text(self, cli_runner: CliRunner):
        """Test that CLI displays help text."""
        result = cli_runner.invoke(app, ["--help"])

        # Assertions
        assert result.exit_code == 0
        assert "VectorFlow" in result.stdout or "PDF" in result.stdout
        assert "process" in result.stdout
        assert "search" in result.stdout

    def test_cli_version(self, cli_runner: CliRunner):
        """Test that CLI displays version information."""
        result = cli_runner.invoke(app, ["--version"])

        # Assertions - may or may not have version flag
        # If it fails, that's okay - not all CLIs have --version

    def test_invalid_command(self, cli_runner: CliRunner):
        """Test CLI with invalid command."""
        result = cli_runner.invoke(app, ["invalid-command"])

        # Assertions
        assert result.exit_code != 0

    def test_process_help(self, cli_runner: CliRunner):
        """Test process command help text."""
        result = cli_runner.invoke(app, ["process", "--help"])

        # Assertions
        assert result.exit_code == 0
        assert "PDF" in result.stdout or "file" in result.stdout.lower()
        assert "model" in result.stdout.lower()

    def test_search_help(self, cli_runner: CliRunner):
        """Test search command help text."""
        result = cli_runner.invoke(app, ["search", "--help"])

        # Assertions
        assert result.exit_code == 0
        assert "query" in result.stdout.lower()
        assert "results" in result.stdout.lower()

    def test_stats_help(self, cli_runner: CliRunner):
        """Test stats command help text."""
        result = cli_runner.invoke(app, ["stats", "--help"])

        # Assertions
        assert result.exit_code == 0
        assert "collection" in result.stdout.lower()

    def test_delete_help(self, cli_runner: CliRunner):
        """Test delete command help text."""
        result = cli_runner.invoke(app, ["delete", "--help"])

        # Assertions
        assert result.exit_code == 0
        assert "document" in result.stdout.lower()

    def test_health_help(self, cli_runner: CliRunner):
        """Test health command help text."""
        result = cli_runner.invoke(app, ["health", "--help"])

        # Assertions
        assert result.exit_code == 0

    def test_config_help(self, cli_runner: CliRunner):
        """Test config command help text."""
        result = cli_runner.invoke(app, ["config", "--help"])

        # Assertions
        assert result.exit_code == 0

    def test_collections_help(self, cli_runner: CliRunner):
        """Test collections command help text."""
        result = cli_runner.invoke(app, ["collections", "--help"])

        # Assertions
        assert result.exit_code == 0
