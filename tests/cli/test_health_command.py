"""Tests for the CLI health command."""

from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from vectorflow.cli.main import app


class TestHealthCommand:
    """Test the health command."""

    @patch("vectorflow.cli.main.get_pipeline")
    def test_health_all_healthy(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_health_status,
    ):
        """Test health check when all components are healthy."""
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.health_check.return_value = mock_health_status
        mock_get_pipeline.return_value = mock_pipeline

        # Run health command
        result = cli_runner.invoke(app, ["health"])

        # Assertions
        assert result.exit_code == 0
        assert (
            "operational" in result.stdout.lower() or "healthy" in result.stdout.lower()
        )
        mock_pipeline.health_check.assert_called_once()

    @patch("vectorflow.cli.main.get_pipeline")
    def test_health_some_unhealthy(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
    ):
        """Test health check when some components are unhealthy."""
        mock_pipeline = Mock()
        unhealthy_status = {
            "embedding_service": True,
            "vector_database": False,
            "pipeline": True,
        }
        mock_pipeline.health_check.return_value = unhealthy_status
        mock_get_pipeline.return_value = mock_pipeline

        # Run health command
        result = cli_runner.invoke(app, ["health"])

        # Assertions
        assert result.exit_code == 1
        assert (
            "attention" in result.stdout.lower() or "unhealthy" in result.stdout.lower()
        )

    @patch("vectorflow.cli.main.get_pipeline")
    def test_health_with_collection(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_health_status,
    ):
        """Test health check with custom collection."""
        mock_pipeline = Mock()
        mock_pipeline.health_check.return_value = mock_health_status
        mock_get_pipeline.return_value = mock_pipeline

        # Run health with custom collection
        result = cli_runner.invoke(app, ["health", "--collection", "custom_collection"])

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_get_pipeline.call_args[1]
        assert call_kwargs["collection_name"] == "custom_collection"

    @patch("vectorflow.cli.main.get_pipeline")
    def test_health_with_debug(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_health_status,
    ):
        """Test health check with debug mode."""
        mock_pipeline = Mock()
        mock_pipeline.health_check.return_value = mock_health_status
        mock_get_pipeline.return_value = mock_pipeline

        # Run health with debug flag
        result = cli_runner.invoke(app, ["health", "--debug"])

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_get_pipeline.call_args[1]
        assert call_kwargs["debug"] is True

    @patch("vectorflow.cli.main.get_pipeline")
    def test_health_exception(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
    ):
        """Test health check with exception."""
        mock_pipeline = Mock()
        mock_pipeline.health_check.side_effect = Exception("Health check error")
        mock_get_pipeline.return_value = mock_pipeline

        # Run health
        result = cli_runner.invoke(app, ["health"])

        # Assertions
        assert result.exit_code == 1
        assert "failed" in result.stdout.lower() or "error" in result.stdout.lower()

    @patch("vectorflow.cli.main.get_pipeline")
    def test_health_short_option(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_health_status,
    ):
        """Test health check with short option flag."""
        mock_pipeline = Mock()
        mock_pipeline.health_check.return_value = mock_health_status
        mock_get_pipeline.return_value = mock_pipeline

        # Run health with short option
        result = cli_runner.invoke(app, ["health", "-c", "test_coll"])

        # Assertions
        assert result.exit_code == 0
        call_kwargs = mock_get_pipeline.call_args[1]
        assert call_kwargs["collection_name"] == "test_coll"

    @patch("vectorflow.cli.main.get_pipeline")
    def test_health_displays_components(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
        mock_health_status,
    ):
        """Test that health check displays all component statuses."""
        mock_pipeline = Mock()
        mock_pipeline.health_check.return_value = mock_health_status
        mock_get_pipeline.return_value = mock_pipeline

        # Run health
        result = cli_runner.invoke(app, ["health"])

        # Assertions
        assert result.exit_code == 0
        # Check that component names are displayed
        assert "Embedding" in result.stdout or "embedding" in result.stdout.lower()
        assert "Vector" in result.stdout or "database" in result.stdout.lower()

    @patch("vectorflow.cli.main.get_pipeline")
    def test_health_all_unhealthy(
        self,
        mock_get_pipeline,
        cli_runner: CliRunner,
    ):
        """Test health check when all components are unhealthy."""
        mock_pipeline = Mock()
        all_unhealthy = {
            "embedding_service": False,
            "vector_database": False,
            "pipeline": False,
        }
        mock_pipeline.health_check.return_value = all_unhealthy
        mock_get_pipeline.return_value = mock_pipeline

        # Run health
        result = cli_runner.invoke(app, ["health"])

        # Assertions
        assert result.exit_code == 1
        assert (
            "attention" in result.stdout.lower() or "unhealthy" in result.stdout.lower()
        )
