"""Tests for the CLI process command."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from vectorflow.cli.main import app


class TestProcessCommand:
    """Test the process command."""

    @patch("vectorflow.cli.main.get_pipeline")
    def test_process_single_pdf_success(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        temp_pdf_file: Path,
        mock_processing_result,
    ):
        """Test processing a single PDF file successfully."""
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.process_pdf.return_value = mock_processing_result
        mock_pipeline.embedding_service.model_name = "test-model"
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_pipeline.config.text_processing.chunk_size = 1000
        mock_pipeline.config.embedding.batch_size = 32
        mock_get_pipeline.return_value = mock_pipeline

        # Run command
        result = cli_runner.invoke(app, ["process", str(temp_pdf_file)])

        # Assertions
        assert result.exit_code == 0
        assert "Processing complete" in result.stdout or "✓" in result.stdout
        mock_pipeline.process_pdf.assert_called_once()

    @patch("vectorflow.cli.main.get_pipeline")
    def test_process_multiple_pdfs(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        multiple_pdf_files: list[Path],
        mock_processing_result,
    ):
        """Test processing multiple PDF files."""
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.process_pdf.return_value = mock_processing_result
        mock_pipeline.embedding_service.model_name = "test-model"
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_pipeline.config.text_processing.chunk_size = 1000
        mock_pipeline.config.embedding.batch_size = 32
        mock_get_pipeline.return_value = mock_pipeline

        # Run command with multiple files
        file_args = [str(f) for f in multiple_pdf_files]
        result = cli_runner.invoke(app, ["process"] + file_args)

        # Assertions
        assert result.exit_code == 0
        assert mock_pipeline.process_pdf.call_count == len(multiple_pdf_files)

    @patch("vectorflow.cli.main.get_pipeline")
    def test_process_with_custom_model(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        temp_pdf_file: Path,
        mock_processing_result,
    ):
        """Test processing with custom embedding model."""
        mock_pipeline = Mock()
        mock_pipeline.process_pdf.return_value = mock_processing_result
        mock_pipeline.embedding_service.model_name = "all-MiniLM-L6-v2"
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_pipeline.config.text_processing.chunk_size = 1000
        mock_pipeline.config.embedding.batch_size = 32
        mock_get_pipeline.return_value = mock_pipeline

        # Run command with custom model
        result = cli_runner.invoke(
            app, ["process", str(temp_pdf_file), "--model", "all-MiniLM-L6-v2"]
        )

        # Assertions
        assert result.exit_code == 0
        mock_get_pipeline.assert_called_once()
        call_kwargs = mock_get_pipeline.call_args[1]
        assert call_kwargs["embedding_model"] == "all-MiniLM-L6-v2"

    @patch("vectorflow.cli.main.get_pipeline")
    def test_process_with_custom_collection(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        temp_pdf_file: Path,
        mock_processing_result,
    ):
        """Test processing with custom collection name."""
        mock_pipeline = Mock()
        mock_pipeline.process_pdf.return_value = mock_processing_result
        mock_pipeline.embedding_service.model_name = "test-model"
        mock_pipeline.config.chroma_db.collection_name = "my_custom_collection"
        mock_pipeline.config.text_processing.chunk_size = 1000
        mock_pipeline.config.embedding.batch_size = 32
        mock_get_pipeline.return_value = mock_pipeline

        # Run command with custom collection
        result = cli_runner.invoke(
            app, ["process", str(temp_pdf_file), "--collection", "my_custom_collection"]
        )

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_get_pipeline.call_args[1]
        assert call_kwargs["collection_name"] == "my_custom_collection"

    @patch("vectorflow.cli.main.get_pipeline")
    def test_process_with_batch_size(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        temp_pdf_file: Path,
        mock_processing_result,
    ):
        """Test processing with custom batch size."""
        mock_pipeline = Mock()
        mock_pipeline.process_pdf.return_value = mock_processing_result
        mock_pipeline.embedding_service.model_name = "test-model"
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_pipeline.config.text_processing.chunk_size = 1000
        mock_pipeline.config.embedding.batch_size = 64
        mock_get_pipeline.return_value = mock_pipeline

        # Run command with custom batch size
        result = cli_runner.invoke(
            app, ["process", str(temp_pdf_file), "--batch-size", "64"]
        )

        # Assertions
        assert result.exit_code == 0
        assert mock_pipeline.config.embedding.batch_size == 64

    @patch("vectorflow.cli.main.get_pipeline")
    def test_process_with_chunk_size(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        temp_pdf_file: Path,
        mock_processing_result,
    ):
        """Test processing with custom chunk size."""
        mock_pipeline = Mock()
        mock_pipeline.process_pdf.return_value = mock_processing_result
        mock_pipeline.embedding_service.model_name = "test-model"
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_pipeline.config.text_processing.chunk_size = 1500
        mock_pipeline.config.embedding.batch_size = 32
        mock_get_pipeline.return_value = mock_pipeline

        # Run command with custom chunk size
        result = cli_runner.invoke(
            app, ["process", str(temp_pdf_file), "--chunk-size", "1500"]
        )

        # Assertions
        assert result.exit_code == 0
        assert mock_pipeline.config.text_processing.chunk_size == 1500

    @patch("vectorflow.cli.main.get_pipeline")
    def test_process_with_no_clean(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        temp_pdf_file: Path,
        mock_processing_result,
    ):
        """Test processing with text cleaning disabled."""
        mock_pipeline = Mock()
        mock_pipeline.process_pdf.return_value = mock_processing_result
        mock_pipeline.embedding_service.model_name = "test-model"
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_pipeline.config.text_processing.chunk_size = 1000
        mock_pipeline.config.embedding.batch_size = 32
        mock_get_pipeline.return_value = mock_pipeline

        # Run command with --no-clean
        result = cli_runner.invoke(app, ["process", str(temp_pdf_file), "--no-clean"])

        # Assertions
        assert result.exit_code == 0
        call_args = mock_pipeline.process_pdf.call_args
        assert call_args[1]["clean_text"] is False

    @patch("vectorflow.cli.main.get_pipeline")
    def test_process_with_verbose(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        temp_pdf_file: Path,
        mock_processing_result,
    ):
        """Test processing with verbose output."""
        mock_pipeline = Mock()
        mock_pipeline.process_pdf.return_value = mock_processing_result
        mock_pipeline.embedding_service.model_name = "test-model"
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_pipeline.config.text_processing.chunk_size = 1000
        mock_pipeline.config.embedding.batch_size = 32
        mock_get_pipeline.return_value = mock_pipeline

        # Run command with verbose flag
        result = cli_runner.invoke(app, ["process", str(temp_pdf_file), "--verbose"])

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_get_pipeline.call_args[1]
        assert call_kwargs["verbose"] is True

    @patch("vectorflow.cli.main.get_pipeline")
    def test_process_with_debug(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        temp_pdf_file: Path,
        mock_processing_result,
    ):
        """Test processing with debug mode."""
        mock_pipeline = Mock()
        mock_pipeline.process_pdf.return_value = mock_processing_result
        mock_pipeline.embedding_service.model_name = "test-model"
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_pipeline.config.text_processing.chunk_size = 1000
        mock_pipeline.config.embedding.batch_size = 32
        mock_get_pipeline.return_value = mock_pipeline

        # Run command with debug flag
        result = cli_runner.invoke(app, ["process", str(temp_pdf_file), "--debug"])

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_get_pipeline.call_args[1]
        assert call_kwargs["debug"] is True

    def test_process_nonexistent_file(self, cli_runner: CliRunner):
        """Test processing a file that doesn't exist."""
        result = cli_runner.invoke(app, ["process", "nonexistent.pdf"])

        # Assertions
        assert result.exit_code == 1
        assert (
            "not found" in result.stdout.lower() or "no valid" in result.stdout.lower()
        )

    @patch("vectorflow.cli.main.get_pipeline")
    def test_process_non_pdf_file(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        non_pdf_file: Path,
    ):
        """Test processing a non-PDF file."""
        mock_pipeline = Mock()
        mock_get_pipeline.return_value = mock_pipeline

        # Run command with non-PDF file
        result = cli_runner.invoke(app, ["process", str(non_pdf_file)])

        # Assertions
        assert result.exit_code == 1
        assert "non-PDF" in result.stdout or "No valid" in result.stdout

    @patch("vectorflow.cli.main.get_pipeline")
    def test_process_failed_pdf(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        temp_pdf_file: Path,
        mock_failed_processing_result,
    ):
        """Test processing a PDF that fails."""
        mock_pipeline = Mock()
        mock_pipeline.process_pdf.return_value = mock_failed_processing_result
        mock_pipeline.embedding_service.model_name = "test-model"
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_pipeline.config.text_processing.chunk_size = 1000
        mock_pipeline.config.embedding.batch_size = 32
        mock_get_pipeline.return_value = mock_pipeline

        # Run command
        result = cli_runner.invoke(app, ["process", str(temp_pdf_file)])

        # Assertions
        assert result.exit_code == 1
        assert "failed" in result.stdout.lower() or "✗" in result.stdout

    @patch("vectorflow.cli.main.get_pipeline")
    def test_process_exception_during_processing(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        temp_pdf_file: Path,
    ):
        """Test handling of exceptions during processing."""
        mock_pipeline = Mock()
        mock_pipeline.process_pdf.side_effect = Exception("Processing error")
        mock_pipeline.embedding_service.model_name = "test-model"
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_pipeline.config.text_processing.chunk_size = 1000
        mock_pipeline.config.embedding.batch_size = 32
        mock_get_pipeline.return_value = mock_pipeline

        # Run command
        result = cli_runner.invoke(app, ["process", str(temp_pdf_file)])

        # Assertions
        assert result.exit_code == 1
        assert "error" in result.stdout.lower() or "✗" in result.stdout

    @patch("vectorflow.cli.main.get_pipeline")
    def test_process_mixed_files(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        temp_pdf_file: Path,
        non_pdf_file: Path,
        mock_processing_result,
    ):
        """Test processing a mix of valid and invalid files."""
        mock_pipeline = Mock()
        mock_pipeline.process_pdf.return_value = mock_processing_result
        mock_pipeline.embedding_service.model_name = "test-model"
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_pipeline.config.text_processing.chunk_size = 1000
        mock_pipeline.config.embedding.batch_size = 32
        mock_get_pipeline.return_value = mock_pipeline

        # Run command with mixed files
        result = cli_runner.invoke(
            app, ["process", str(temp_pdf_file), str(non_pdf_file)]
        )

        # Assertions - should process valid PDF and skip invalid file
        assert result.exit_code == 0
        assert mock_pipeline.process_pdf.call_count == 1
        assert "Skipping" in result.stdout or "non-PDF" in result.stdout

    @patch("vectorflow.cli.main.get_pipeline")
    def test_process_with_openai_model(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        temp_pdf_file: Path,
        mock_processing_result,
    ):
        """Test processing with OpenAI embedding model."""
        mock_pipeline = Mock()
        mock_pipeline.process_pdf.return_value = mock_processing_result
        mock_pipeline.embedding_service.model_name = "text-embedding-3-small"
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_pipeline.config.text_processing.chunk_size = 1000
        mock_pipeline.config.embedding.batch_size = 32
        mock_get_pipeline.return_value = mock_pipeline

        # Run command with OpenAI model
        result = cli_runner.invoke(
            app, ["process", str(temp_pdf_file), "--model", "text-embedding-3-small"]
        )

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_get_pipeline.call_args[1]
        assert call_kwargs["embedding_model"] == "text-embedding-3-small"

    @patch("vectorflow.cli.main.get_pipeline")
    def test_process_all_options_combined(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        temp_pdf_file: Path,
        mock_processing_result,
    ):
        """Test processing with all options combined."""
        mock_pipeline = Mock()
        mock_pipeline.process_pdf.return_value = mock_processing_result
        mock_pipeline.embedding_service.model_name = "all-MiniLM-L6-v2"
        mock_pipeline.config.chroma_db.collection_name = "custom_collection"
        mock_pipeline.config.text_processing.chunk_size = 2000
        mock_pipeline.config.embedding.batch_size = 64
        mock_get_pipeline.return_value = mock_pipeline

        # Run command with all options
        result = cli_runner.invoke(
            app,
            [
                "process",
                str(temp_pdf_file),
                "--model",
                "all-MiniLM-L6-v2",
                "--collection",
                "custom_collection",
                "--batch-size",
                "64",
                "--chunk-size",
                "2000",
                "--no-clean",
                "--verbose",
            ],
        )

        # Assertions
        assert result.exit_code == 0
        assert mock_pipeline.config.embedding.batch_size == 64
        assert mock_pipeline.config.text_processing.chunk_size == 2000

    @patch("vectorflow.cli.main.get_pipeline")
    def test_process_keyboard_interrupt(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        temp_pdf_file: Path,
    ):
        """Test handling of keyboard interrupt during processing."""
        mock_pipeline = Mock()
        mock_pipeline.process_pdf.side_effect = KeyboardInterrupt()
        mock_pipeline.embedding_service.model_name = "test-model"
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_pipeline.config.text_processing.chunk_size = 1000
        mock_pipeline.config.embedding.batch_size = 32
        mock_get_pipeline.return_value = mock_pipeline

        # Run command
        result = cli_runner.invoke(app, ["process", str(temp_pdf_file)])

        # Assertions
        assert result.exit_code == 130
        assert (
            "cancelled" in result.stdout.lower()
            or "interrupted" in result.stdout.lower()
        )

    @patch("vectorflow.cli.main.get_pipeline")
    def test_process_pipeline_initialization_error(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        temp_pdf_file: Path,
    ):
        """Test handling of pipeline initialization errors."""
        mock_get_pipeline.side_effect = Exception("Failed to initialize pipeline")

        # Run command
        result = cli_runner.invoke(app, ["process", str(temp_pdf_file)])

        # Assertions
        assert result.exit_code == 1
        assert "error" in result.stdout.lower()

    @patch("vectorflow.cli.main.get_pipeline")
    def test_process_short_options(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        temp_pdf_file: Path,
        mock_processing_result,
    ):
        """Test processing with short option flags."""
        mock_pipeline = Mock()
        mock_pipeline.process_pdf.return_value = mock_processing_result
        mock_pipeline.embedding_service.model_name = "test-model"
        mock_pipeline.config.chroma_db.collection_name = "test_coll"
        mock_pipeline.config.text_processing.chunk_size = 1000
        mock_pipeline.config.embedding.batch_size = 16
        mock_get_pipeline.return_value = mock_pipeline

        # Run command with short options
        result = cli_runner.invoke(
            app,
            [
                "process",
                str(temp_pdf_file),
                "-m",
                "test-model",
                "-c",
                "test_coll",
                "-b",
                "16",
                "-v",
            ],
        )

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_get_pipeline.call_args[1]
        assert call_kwargs["embedding_model"] == "test-model"
        assert call_kwargs["collection_name"] == "test_coll"
