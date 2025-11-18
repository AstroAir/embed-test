"""Tests for the CLI list-docs command."""

from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from vectorflow.cli.main import app


class TestListDocsCommand:
    """Test the list-docs command."""

    @patch("vectorflow.cli.main.get_pipeline")
    def test_list_docs_basic(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_document_info,
    ):
        """Test basic list-docs functionality."""
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "ids": ["chunk_1", "chunk_2"],
            "metadatas": [
                {"document_id": "doc1.pdf"},
                {"document_id": "doc2.pdf"},
            ],
        }
        mock_pipeline.vector_db.get_collection.return_value = mock_collection
        mock_pipeline.get_document_info.return_value = mock_document_info
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run list-docs command
        result = cli_runner.invoke(app, ["list-docs"])

        # Assertions
        assert result.exit_code == 0
        assert "Documents" in result.stdout

    @patch("vectorflow.cli.main.get_pipeline")
    def test_list_docs_with_limit(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_document_info,
    ):
        """Test list-docs with custom limit."""
        mock_pipeline = Mock()
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "ids": ["chunk_1"],
            "metadatas": [{"document_id": "doc1.pdf"}],
        }
        mock_pipeline.vector_db.get_collection.return_value = mock_collection
        mock_pipeline.get_document_info.return_value = mock_document_info
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run list-docs with limit
        result = cli_runner.invoke(app, ["list-docs", "--limit", "10"])

        # Assertions
        assert result.exit_code == 0

    @patch("vectorflow.cli.main.get_pipeline")
    def test_list_docs_with_collection(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_document_info,
    ):
        """Test list-docs with custom collection."""
        mock_pipeline = Mock()
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "ids": ["chunk_1"],
            "metadatas": [{"document_id": "doc1.pdf"}],
        }
        mock_pipeline.vector_db.get_collection.return_value = mock_collection
        mock_pipeline.get_document_info.return_value = mock_document_info
        mock_pipeline.config.chroma_db.collection_name = "custom_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run list-docs with custom collection
        result = cli_runner.invoke(
            app, ["list-docs", "--collection", "custom_collection"]
        )

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_get_pipeline.call_args[1]
        assert call_kwargs["collection_name"] == "custom_collection"

    @patch("vectorflow.cli.main.get_pipeline")
    def test_list_docs_empty_collection(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
    ):
        """Test list-docs with empty collection."""
        mock_pipeline = Mock()
        mock_collection = Mock()
        mock_collection.get.return_value = {"ids": [], "metadatas": []}
        mock_pipeline.vector_db.get_collection.return_value = mock_collection
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run list-docs
        result = cli_runner.invoke(app, ["list-docs"])

        # Assertions
        assert result.exit_code == 0
        assert "No documents" in result.stdout or "not found" in result.stdout.lower()

    @patch("vectorflow.cli.main.get_pipeline")
    def test_list_docs_with_debug(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_document_info,
    ):
        """Test list-docs with debug mode."""
        mock_pipeline = Mock()
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "ids": ["chunk_1"],
            "metadatas": [{"document_id": "doc1.pdf"}],
        }
        mock_pipeline.vector_db.get_collection.return_value = mock_collection
        mock_pipeline.get_document_info.return_value = mock_document_info
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run list-docs with debug flag
        result = cli_runner.invoke(app, ["list-docs", "--debug"])

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_get_pipeline.call_args[1]
        assert call_kwargs["debug"] is True

    @patch("vectorflow.cli.main.get_pipeline")
    def test_list_docs_exception(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
    ):
        """Test list-docs with exception."""
        mock_pipeline = Mock()
        mock_pipeline.vector_db.get_collection.side_effect = Exception("List error")
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run list-docs
        result = cli_runner.invoke(app, ["list-docs"])

        # Assertions
        assert result.exit_code == 1
        assert "error" in result.stdout.lower()

    @patch("vectorflow.cli.main.get_pipeline")
    def test_list_docs_short_options(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_document_info,
    ):
        """Test list-docs with short option flags."""
        mock_pipeline = Mock()
        mock_collection = Mock()
        mock_collection.get.return_value = {
            "ids": ["chunk_1"],
            "metadatas": [{"document_id": "doc1.pdf"}],
        }
        mock_pipeline.vector_db.get_collection.return_value = mock_collection
        mock_pipeline.get_document_info.return_value = mock_document_info
        mock_pipeline.config.chroma_db.collection_name = "test_coll"
        mock_get_pipeline.return_value = mock_pipeline

        # Run list-docs with short options
        result = cli_runner.invoke(app, ["list-docs", "-c", "test_coll", "-l", "25"])

        # Assertions
        assert result.exit_code == 0
