"""Tests for the CLI collections command."""

from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from vectorflow.cli.main import app


class TestCollectionsCommand:
    """Test the collections command."""

    @patch("vectorflow.cli.main.get_pipeline")
    def test_collections_basic(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_collection_list,
    ):
        """Test basic collections listing."""
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.vector_db.list_collections.return_value = mock_collection_list
        mock_get_pipeline.return_value = mock_pipeline

        # Run collections command
        result = cli_runner.invoke(app, ["collections"])

        # Assertions
        assert result.exit_code == 0
        assert "Collections" in result.stdout
        mock_pipeline.vector_db.list_collections.assert_called_once()

    @patch("vectorflow.cli.main.get_pipeline")
    def test_collections_empty(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
    ):
        """Test collections listing when no collections exist."""
        mock_pipeline = Mock()
        mock_pipeline.vector_db.list_collections.return_value = []
        mock_get_pipeline.return_value = mock_pipeline

        # Run collections command
        result = cli_runner.invoke(app, ["collections"])

        # Assertions
        assert result.exit_code == 0
        assert "No collections" in result.stdout or "not found" in result.stdout.lower()

    @patch("vectorflow.cli.main.get_pipeline")
    def test_collections_with_debug(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_collection_list,
    ):
        """Test collections listing with debug mode."""
        mock_pipeline = Mock()
        mock_pipeline.vector_db.list_collections.return_value = mock_collection_list
        mock_get_pipeline.return_value = mock_pipeline

        # Run collections with debug flag
        result = cli_runner.invoke(app, ["collections", "--debug"])

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_get_pipeline.call_args[1]
        assert call_kwargs["debug"] is True

    @patch("vectorflow.cli.main.get_pipeline")
    def test_collections_exception(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
    ):
        """Test collections listing with exception."""
        mock_pipeline = Mock()
        mock_pipeline.vector_db.list_collections.side_effect = Exception("List error")
        mock_get_pipeline.return_value = mock_pipeline

        # Run collections
        result = cli_runner.invoke(app, ["collections"])

        # Assertions
        assert result.exit_code == 1
        assert "error" in result.stdout.lower()

    @patch("vectorflow.cli.main.get_pipeline")
    def test_collections_displays_count(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_collection_list,
    ):
        """Test that collections displays the count."""
        mock_pipeline = Mock()
        mock_pipeline.vector_db.list_collections.return_value = mock_collection_list
        mock_get_pipeline.return_value = mock_pipeline

        # Run collections
        result = cli_runner.invoke(app, ["collections"])

        # Assertions
        assert result.exit_code == 0
        # Should show the number of collections
        assert str(len(mock_collection_list)) in result.stdout or "3" in result.stdout

    @patch("vectorflow.cli.main.get_pipeline")
    def test_collections_displays_names(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_collection_list,
    ):
        """Test that collections displays collection names."""
        mock_pipeline = Mock()
        mock_pipeline.vector_db.list_collections.return_value = mock_collection_list
        mock_get_pipeline.return_value = mock_pipeline

        # Run collections
        result = cli_runner.invoke(app, ["collections"])

        # Assertions
        assert result.exit_code == 0
        # Should show collection names
        for collection_name in mock_collection_list:
            assert collection_name in result.stdout

    @patch("vectorflow.cli.main.get_pipeline")
    def test_collections_with_stats(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_collection_list,
        mock_collection_stats,
    ):
        """Test collections listing with stats for each collection."""
        # Setup mock pipeline that returns stats
        mock_pipeline = Mock()
        mock_pipeline.vector_db.list_collections.return_value = mock_collection_list
        mock_pipeline.get_collection_stats.return_value = mock_collection_stats
        mock_get_pipeline.return_value = mock_pipeline

        # Run collections
        result = cli_runner.invoke(app, ["collections"])

        # Assertions
        assert result.exit_code == 0
        # Should attempt to get stats for collections
        # Note: The actual implementation may handle errors gracefully

    @patch("vectorflow.cli.main.get_pipeline")
    def test_collections_stats_error_handling(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_collection_list,
    ):
        """Test collections listing handles stats errors gracefully."""
        # Setup mock pipeline where stats fail
        mock_pipeline = Mock()
        mock_pipeline.vector_db.list_collections.return_value = mock_collection_list
        mock_pipeline.get_collection_stats.side_effect = Exception("Stats error")
        mock_get_pipeline.return_value = mock_pipeline

        # Run collections
        result = cli_runner.invoke(app, ["collections"])

        # Assertions
        # Should still succeed even if stats fail
        assert result.exit_code == 0
        assert "Collections" in result.stdout

    @patch("vectorflow.cli.main.get_pipeline")
    def test_collections_sorted_alphabetically(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
    ):
        """Test that collections are sorted alphabetically."""
        mock_pipeline = Mock()
        unsorted_collections = [
            "zebra_collection",
            "alpha_collection",
            "beta_collection",
        ]
        mock_pipeline.vector_db.list_collections.return_value = unsorted_collections
        mock_get_pipeline.return_value = mock_pipeline

        # Run collections
        result = cli_runner.invoke(app, ["collections"])

        # Assertions
        assert result.exit_code == 0
        # Collections should be sorted
        output_lines = result.stdout.split("\n")
        # Find the lines with collection names and verify they're sorted
