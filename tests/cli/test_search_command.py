"""Tests for the CLI search command."""

from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from vectorflow.cli.main import app


class TestSearchCommand:
    """Test the search command."""

    @patch("vectorflow.cli.main.get_pipeline")
    def test_search_basic(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_search_results,
    ):
        """Test basic search functionality."""
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.search.return_value = mock_search_results
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run search command
        result = cli_runner.invoke(app, ["search", "machine learning"])

        # Assertions
        assert result.exit_code == 0
        assert "Found" in result.stdout or "result" in result.stdout.lower()
        mock_pipeline.search.assert_called_once()

    @patch("vectorflow.cli.main.get_pipeline")
    def test_search_with_n_results(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_search_results,
    ):
        """Test search with custom number of results."""
        mock_pipeline = Mock()
        mock_pipeline.search.return_value = mock_search_results[:2]
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run search with custom n_results
        result = cli_runner.invoke(app, ["search", "test query", "--results", "2"])

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_pipeline.search.call_args[1]
        assert call_kwargs["n_results"] == 2

    @patch("vectorflow.cli.main.get_pipeline")
    def test_search_with_document_filter(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_search_results,
    ):
        """Test search with document ID filter."""
        mock_pipeline = Mock()
        mock_pipeline.search.return_value = [mock_search_results[0]]
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run search with document filter
        result = cli_runner.invoke(
            app, ["search", "test query", "--document", "test_doc"]
        )

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_pipeline.search.call_args[1]
        assert call_kwargs["document_id"] == "test_doc"

    @patch("vectorflow.cli.main.get_pipeline")
    def test_search_with_page_filter(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_search_results,
    ):
        """Test search with page number filter."""
        mock_pipeline = Mock()
        mock_pipeline.search.return_value = [mock_search_results[0]]
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run search with page filter
        result = cli_runner.invoke(app, ["search", "test query", "--page", "1"])

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_pipeline.search.call_args[1]
        assert call_kwargs["page_number"] == 1

    @patch("vectorflow.cli.main.get_pipeline")
    def test_search_with_collection(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_search_results,
    ):
        """Test search with custom collection."""
        mock_pipeline = Mock()
        mock_pipeline.search.return_value = mock_search_results
        mock_pipeline.config.chroma_db.collection_name = "custom_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run search with custom collection
        result = cli_runner.invoke(
            app, ["search", "test query", "--collection", "custom_collection"]
        )

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_get_pipeline.call_args[1]
        assert call_kwargs["collection_name"] == "custom_collection"

    @patch("vectorflow.cli.main.get_pipeline")
    def test_search_with_score_threshold(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_search_results,
    ):
        """Test search with score threshold filtering."""
        mock_pipeline = Mock()
        mock_pipeline.search.return_value = mock_search_results
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run search with score threshold
        result = cli_runner.invoke(app, ["search", "test query", "--threshold", "0.8"])

        # Assertions
        assert result.exit_code == 0
        # Should filter out results with score < 0.8
        assert "Filtered" in result.stdout or "result" in result.stdout.lower()

    @patch("vectorflow.cli.main.get_pipeline")
    def test_search_with_verbose(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_search_results,
    ):
        """Test search with verbose output."""
        mock_pipeline = Mock()
        mock_pipeline.search.return_value = mock_search_results
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run search with verbose flag
        result = cli_runner.invoke(app, ["search", "test query", "--verbose"])

        # Assertions
        assert result.exit_code == 0
        # Verbose should show full content (not truncated)

    @patch("vectorflow.cli.main.get_pipeline")
    def test_search_no_results(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
    ):
        """Test search with no results found."""
        mock_pipeline = Mock()
        mock_pipeline.search.return_value = []
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run search
        result = cli_runner.invoke(app, ["search", "nonexistent query"])

        # Assertions
        assert result.exit_code == 0
        assert "No results" in result.stdout or "not found" in result.stdout.lower()

    @patch("vectorflow.cli.main.get_pipeline")
    def test_search_invalid_n_results(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
    ):
        """Test search with invalid n_results."""
        mock_pipeline = Mock()
        mock_get_pipeline.return_value = mock_pipeline

        # Run search with invalid n_results
        result = cli_runner.invoke(app, ["search", "test query", "--results", "0"])

        # Assertions
        assert result.exit_code == 1
        assert "positive" in result.stdout.lower() or "must be" in result.stdout.lower()

    @patch("vectorflow.cli.main.get_pipeline")
    def test_search_invalid_threshold(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
    ):
        """Test search with invalid score threshold."""
        mock_pipeline = Mock()
        mock_get_pipeline.return_value = mock_pipeline

        # Run search with invalid threshold
        result = cli_runner.invoke(app, ["search", "test query", "--threshold", "1.5"])

        # Assertions
        assert result.exit_code == 1
        assert "between" in result.stdout.lower() or "0.0" in result.stdout

    @patch("vectorflow.cli.main.get_pipeline")
    def test_search_exception(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
    ):
        """Test search with exception during search."""
        mock_pipeline = Mock()
        mock_pipeline.search.side_effect = Exception("Search error")
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run search
        result = cli_runner.invoke(app, ["search", "test query"])

        # Assertions
        assert result.exit_code == 1
        assert "error" in result.stdout.lower()

    @patch("vectorflow.cli.main.get_pipeline")
    def test_search_keyboard_interrupt(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
    ):
        """Test search with keyboard interrupt."""
        mock_pipeline = Mock()
        mock_pipeline.search.side_effect = KeyboardInterrupt()
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run search
        result = cli_runner.invoke(app, ["search", "test query"])

        # Assertions
        assert result.exit_code == 130
        assert (
            "cancelled" in result.stdout.lower()
            or "interrupted" in result.stdout.lower()
        )

    @patch("vectorflow.cli.main.get_pipeline")
    def test_search_with_debug(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_search_results,
    ):
        """Test search with debug mode."""
        mock_pipeline = Mock()
        mock_pipeline.search.return_value = mock_search_results
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run search with debug flag
        result = cli_runner.invoke(app, ["search", "test query", "--debug"])

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_get_pipeline.call_args[1]
        assert call_kwargs["debug"] is True

    @patch("vectorflow.cli.main.get_pipeline")
    def test_search_short_options(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_search_results,
    ):
        """Test search with short option flags."""
        mock_pipeline = Mock()
        mock_pipeline.search.return_value = mock_search_results
        mock_pipeline.config.chroma_db.collection_name = "test_coll"
        mock_get_pipeline.return_value = mock_pipeline

        # Run search with short options
        result = cli_runner.invoke(
            app,
            [
                "search",
                "test query",
                "-n",
                "5",
                "-d",
                "doc_id",
                "-p",
                "2",
                "-c",
                "test_coll",
                "-t",
                "0.7",
                "-v",
            ],
        )

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_pipeline.search.call_args[1]
        assert call_kwargs["n_results"] == 5
        assert call_kwargs["document_id"] == "doc_id"
        assert call_kwargs["page_number"] == 2

    @patch("vectorflow.cli.main.get_pipeline")
    def test_search_all_filters_combined(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_search_results,
    ):
        """Test search with all filters combined."""
        mock_pipeline = Mock()
        mock_pipeline.search.return_value = mock_search_results
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run search with all filters
        result = cli_runner.invoke(
            app,
            [
                "search",
                "test query",
                "--results",
                "3",
                "--document",
                "test_doc",
                "--page",
                "1",
                "--collection",
                "test_collection",
                "--threshold",
                "0.85",
                "--verbose",
            ],
        )

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_pipeline.search.call_args[1]
        assert call_kwargs["n_results"] == 3
        assert call_kwargs["document_id"] == "test_doc"
        assert call_kwargs["page_number"] == 1

    @patch("vectorflow.cli.main.get_pipeline")
    def test_search_high_threshold_filters_all(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_search_results,
    ):
        """Test search where threshold filters out all results."""
        mock_pipeline = Mock()
        mock_pipeline.search.return_value = mock_search_results
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run search with very high threshold
        result = cli_runner.invoke(app, ["search", "test query", "--threshold", "0.99"])

        # Assertions
        assert result.exit_code == 0
        assert "No results" in result.stdout or "Filtered" in result.stdout

    @patch("vectorflow.cli.main.get_pipeline")
    def test_search_score_display_colors(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_search_results,
    ):
        """Test that search displays scores with appropriate formatting."""
        mock_pipeline = Mock()
        mock_pipeline.search.return_value = mock_search_results
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run search
        result = cli_runner.invoke(app, ["search", "test query"])

        # Assertions
        assert result.exit_code == 0
        # Check that scores are displayed
        assert "Score" in result.stdout or "0." in result.stdout
