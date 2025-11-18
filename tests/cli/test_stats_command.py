"""Tests for the CLI stats command."""

from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from vectorflow.cli.main import app


class TestStatsCommand:
    """Test the stats command."""

    @patch("vectorflow.cli.main.get_pipeline")
    def test_stats_basic(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_collection_stats,
    ):
        """Test basic stats functionality."""
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.get_collection_stats.return_value = mock_collection_stats
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run stats command
        result = cli_runner.invoke(app, ["stats"])

        # Assertions
        assert result.exit_code == 0
        assert "Statistics" in result.stdout or "Total Chunks" in result.stdout
        mock_pipeline.get_collection_stats.assert_called_once()

    @patch("vectorflow.cli.main.get_pipeline")
    def test_stats_with_collection(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_collection_stats,
    ):
        """Test stats with custom collection."""
        mock_pipeline = Mock()
        mock_pipeline.get_collection_stats.return_value = mock_collection_stats
        mock_pipeline.config.chroma_db.collection_name = "custom_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run stats with custom collection
        result = cli_runner.invoke(app, ["stats", "--collection", "custom_collection"])

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_get_pipeline.call_args[1]
        assert call_kwargs["collection_name"] == "custom_collection"

    @patch("vectorflow.cli.main.get_pipeline")
    def test_stats_with_debug(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_collection_stats,
    ):
        """Test stats with debug mode."""
        mock_pipeline = Mock()
        mock_pipeline.get_collection_stats.return_value = mock_collection_stats
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run stats with debug flag
        result = cli_runner.invoke(app, ["stats", "--debug"])

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_get_pipeline.call_args[1]
        assert call_kwargs["debug"] is True

    @patch("vectorflow.cli.main.get_pipeline")
    def test_stats_exception(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
    ):
        """Test stats with exception."""
        mock_pipeline = Mock()
        mock_pipeline.get_collection_stats.side_effect = Exception("Stats error")
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run stats
        result = cli_runner.invoke(app, ["stats"])

        # Assertions
        assert result.exit_code == 1
        assert "error" in result.stdout.lower()

    @patch("vectorflow.cli.main.get_pipeline")
    def test_stats_short_option(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_collection_stats,
    ):
        """Test stats with short option flag."""
        mock_pipeline = Mock()
        mock_pipeline.get_collection_stats.return_value = mock_collection_stats
        mock_pipeline.config.chroma_db.collection_name = "test_coll"
        mock_get_pipeline.return_value = mock_pipeline

        # Run stats with short option
        result = cli_runner.invoke(app, ["stats", "-c", "test_coll"])

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_get_pipeline.call_args[1]
        assert call_kwargs["collection_name"] == "test_coll"

    @patch("vectorflow.cli.main.get_pipeline")
    def test_stats_sampled_data(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
    ):
        """Test stats with sampled data."""
        mock_pipeline = Mock()
        sampled_stats = {
            "total_chunks": 10000,
            "unique_documents": 100,
            "total_characters": 5000000,
            "average_chunk_size": 500.0,
            "sampled": True,
            "sample_size": 1000,
        }
        mock_pipeline.get_collection_stats.return_value = sampled_stats
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run stats
        result = cli_runner.invoke(app, ["stats"])

        # Assertions
        assert result.exit_code == 0
        assert "Sample" in result.stdout or "approximate" in result.stdout.lower()

    @patch("vectorflow.cli.main.get_pipeline")
    def test_stats_displays_metrics(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_collection_stats,
    ):
        """Test that stats displays all expected metrics."""
        mock_pipeline = Mock()
        mock_pipeline.get_collection_stats.return_value = mock_collection_stats
        mock_pipeline.config.chroma_db.collection_name = "test_collection"
        mock_get_pipeline.return_value = mock_pipeline

        # Run stats
        result = cli_runner.invoke(app, ["stats"])

        # Assertions
        assert result.exit_code == 0
        assert "Total Chunks" in result.stdout
        assert "Unique Documents" in result.stdout
        assert "Total Characters" in result.stdout
        assert "Average Chunk Size" in result.stdout
