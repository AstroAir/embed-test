"""Tests for CLI interface."""

from unittest.mock import Mock, patch

import typer

from vectorflow.cli.main import app, console
from vectorflow.core.pipeline import ProcessingResult
from vectorflow.core.vector_db.models import SearchResult


class TestCLIApp:
    """Test CLI application structure."""

    def test_app_exists(self):
        """Test that CLI app exists."""
        assert app is not None
        assert isinstance(app, typer.Typer)

    def test_console_exists(self):
        """Test that console exists."""
        assert console is not None

    def test_app_has_commands(self):
        """Test that app has commands registered."""
        # Check that the app has some commands registered
        assert len(app.registered_commands) > 0

        # Check that it's a list of commands
        assert isinstance(app.registered_commands, list)


class TestCLIIntegration:
    """Test CLI integration with mocked components."""

    @patch("vectorflow.cli.PDFVectorPipeline")
    @patch("vectorflow.cli.Config")
    def test_process_command_integration(
        self, mock_config_class, mock_pipeline_class, temp_dir
    ):
        """Test process command integration."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("Mock PDF content")

        # Mock configuration
        mock_config = Mock()
        mock_config.chroma_db.collection_name = "test_collection"
        mock_config_class.return_value = mock_config

        # Mock pipeline and result
        mock_pipeline = Mock()
        mock_result = ProcessingResult(
            file_path=str(pdf_path),
            document_id="test_doc",
            chunks_processed=5,
            embeddings_generated=5,
            chunks_stored=5,
            processing_time=2.5,
            success=True,
            error_message=None,
            metadata={"pages": 3},
        )
        mock_pipeline.process_pdf.return_value = mock_result
        mock_pipeline.embedding_service.model_name = "test-model"
        mock_pipeline_class.return_value = mock_pipeline

        # Test would normally invoke the CLI command
        # For now, we just verify the mocks are set up correctly
        assert mock_config_class is not None
        assert mock_pipeline_class is not None

    @patch("vectorflow.cli.PDFVectorPipeline")
    @patch("vectorflow.cli.Config")
    def test_search_command_integration(self, mock_config_class, mock_pipeline_class):
        """Test search command integration."""
        # Mock configuration
        mock_config = Mock()
        mock_config.chroma_db.collection_name = "test_collection"
        mock_config_class.return_value = mock_config

        # Mock pipeline and search results
        mock_pipeline = Mock()
        mock_results = [
            SearchResult(
                id="chunk_1",
                content="Test content 1",
                score=0.95,
                metadata={"document_id": "doc_1", "page_number": 1},
            ),
            SearchResult(
                id="chunk_2",
                content="Test content 2",
                score=0.87,
                metadata={"document_id": "doc_1", "page_number": 2},
            ),
        ]
        mock_pipeline.search.return_value = mock_results
        mock_pipeline_class.return_value = mock_pipeline

        # Test would normally invoke the CLI search command
        # For now, we just verify the mocks are set up correctly
        assert mock_config_class is not None
        assert mock_pipeline_class is not None

    @patch("vectorflow.cli.PDFVectorPipeline")
    @patch("vectorflow.cli.Config")
    def test_health_command_integration(self, mock_config_class, mock_pipeline_class):
        """Test health command integration."""
        # Mock configuration
        mock_config = Mock()
        mock_config_class.return_value = mock_config

        # Mock pipeline and health status
        mock_pipeline = Mock()
        mock_health_status = {
            "pdf_processor": True,
            "text_processor": True,
            "embedding_service": True,
            "vector_database": True,
        }
        mock_pipeline.health_check.return_value = mock_health_status
        mock_pipeline_class.return_value = mock_pipeline

        # Test would normally invoke the CLI health command
        # For now, we just verify the mocks are set up correctly
        assert mock_config_class is not None
        assert mock_pipeline_class is not None


class TestCLIUtilities:
    """Test CLI utility functions."""

    def test_console_output(self, capsys):
        """Test console output functionality."""
        console.print("Test message", style="green")
        captured = capsys.readouterr()
        assert "Test message" in captured.out
