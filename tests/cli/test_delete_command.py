"""Tests for the CLI delete command."""

from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from vectorflow.cli.main import app


class TestDeleteCommand:
    """Test the delete command."""

    @patch("vectorflow.cli.main.get_pipeline")
    def test_delete_with_confirmation_yes(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_document_info,
    ):
        """Test delete with --yes flag (skip confirmation)."""
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.get_document_info.return_value = mock_document_info
        mock_pipeline.delete_document.return_value = 10
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run delete command with --yes flag
        result = cli_runner.invoke(app, ["delete", "test_doc.pdf", "--yes"])

        # Assertions
        assert result.exit_code == 0
        assert "Successfully deleted" in result.stdout or "✓" in result.stdout
        mock_pipeline.delete_document.assert_called_once_with("test_doc.pdf")

    @patch("vectorflow.cli.main.get_pipeline")
    def test_delete_with_confirmation_prompt_yes(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_document_info,
    ):
        """Test delete with confirmation prompt (user confirms)."""
        mock_pipeline = Mock()
        mock_pipeline.get_document_info.return_value = mock_document_info
        mock_pipeline.delete_document.return_value = 10
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run delete command with confirmation (simulate user typing 'y')
        result = cli_runner.invoke(app, ["delete", "test_doc.pdf"], input="y\n")

        # Assertions
        assert result.exit_code == 0
        assert "Successfully deleted" in result.stdout or "✓" in result.stdout

    @patch("vectorflow.cli.main.get_pipeline")
    def test_delete_with_confirmation_prompt_no(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_document_info,
    ):
        """Test delete with confirmation prompt (user cancels)."""
        mock_pipeline = Mock()
        mock_pipeline.get_document_info.return_value = mock_document_info
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run delete command with confirmation (simulate user typing 'n')
        result = cli_runner.invoke(app, ["delete", "test_doc.pdf"], input="n\n")

        # Assertions
        assert result.exit_code == 0
        assert "cancelled" in result.stdout.lower()
        mock_pipeline.delete_document.assert_not_called()

    @patch("vectorflow.cli.main.get_pipeline")
    def test_delete_nonexistent_document(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
    ):
        """Test delete with nonexistent document."""
        mock_pipeline = Mock()
        mock_pipeline.get_document_info.side_effect = Exception("Document not found")
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run delete command
        result = cli_runner.invoke(app, ["delete", "nonexistent.pdf", "--yes"])

        # Assertions
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    @patch("vectorflow.cli.main.get_pipeline")
    def test_delete_with_collection(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_document_info,
    ):
        """Test delete with custom collection."""
        mock_pipeline = Mock()
        mock_pipeline.get_document_info.return_value = mock_document_info
        mock_pipeline.delete_document.return_value = 10
        mock_pipeline.config.chroma_db.collection_name = "custom_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run delete with custom collection
        result = cli_runner.invoke(
            app,
            ["delete", "test_doc.pdf", "--collection", "custom_collection", "--yes"],
        )

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_get_pipeline.call_args[1]
        assert call_kwargs["collection_name"] == "custom_collection"

    @patch("vectorflow.cli.main.get_pipeline")
    def test_delete_with_debug(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_document_info,
    ):
        """Test delete with debug mode."""
        mock_pipeline = Mock()
        mock_pipeline.get_document_info.return_value = mock_document_info
        mock_pipeline.delete_document.return_value = 10
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run delete with debug flag
        result = cli_runner.invoke(app, ["delete", "test_doc.pdf", "--yes", "--debug"])

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_get_pipeline.call_args[1]
        assert call_kwargs["debug"] is True

    @patch("vectorflow.cli.main.get_pipeline")
    def test_delete_exception(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_document_info,
    ):
        """Test delete with exception during deletion."""
        mock_pipeline = Mock()
        mock_pipeline.get_document_info.return_value = mock_document_info
        mock_pipeline.delete_document.side_effect = Exception("Delete error")
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run delete
        result = cli_runner.invoke(app, ["delete", "test_doc.pdf", "--yes"])

        # Assertions
        assert result.exit_code == 1
        assert "error" in result.stdout.lower()

    @patch("vectorflow.cli.main.get_pipeline")
    def test_delete_short_options(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_document_info,
    ):
        """Test delete with short option flags."""
        mock_pipeline = Mock()
        mock_pipeline.get_document_info.return_value = mock_document_info
        mock_pipeline.delete_document.return_value = 10
        mock_pipeline.config.chroma_db.collection_name = "test_coll"
        mock_get_pipeline.return_value = mock_pipeline

        # Run delete with short options
        result = cli_runner.invoke(
            app, ["delete", "test_doc.pdf", "-c", "test_coll", "-y"]
        )

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_get_pipeline.call_args[1]
        assert call_kwargs["collection_name"] == "test_coll"
