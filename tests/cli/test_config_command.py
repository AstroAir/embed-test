"""Tests for the CLI config command."""

from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from vectorflow.cli.main import app
from vectorflow.core.config.settings import Config


class TestConfigCommand:
    """Test the config command."""

    @patch("vectorflow.cli.main.Config")
    def test_config_basic(
        self,
        mock_config_class,
        cli_runner: CliRunner,
        test_config: Config,
    ):
        """Test basic config display."""
        # Setup mock config
        mock_config_class.return_value = test_config

        # Run config command
        result = cli_runner.invoke(app, ["config"])

        # Assertions
        assert result.exit_code == 0
        assert "Configuration" in result.stdout
        assert "Embedding" in result.stdout or "Model" in result.stdout

    @patch("vectorflow.cli.main.Config")
    def test_config_show_all(
        self,
        mock_config_class,
        cli_runner: CliRunner,
        test_config: Config,
    ):
        """Test config display with --show-all flag."""
        mock_config_class.return_value = test_config

        # Run config with --show-all
        result = cli_runner.invoke(app, ["config", "--show-all"])

        # Assertions
        assert result.exit_code == 0
        # Should show additional configuration sections
        assert "Text Processing" in result.stdout or "Performance" in result.stdout

    @patch("vectorflow.cli.main.Config")
    def test_config_with_debug(
        self,
        mock_config_class,
        cli_runner: CliRunner,
        test_config: Config,
    ):
        """Test config display with debug mode."""
        mock_config_class.return_value = test_config

        # Run config with debug flag
        result = cli_runner.invoke(app, ["config", "--debug"])

        # Assertions
        assert result.exit_code == 0

    @patch("vectorflow.cli.main.Config")
    def test_config_exception(
        self,
        mock_config_class,
        cli_runner: CliRunner,
    ):
        """Test config display with exception."""
        mock_config_class.side_effect = Exception("Config error")

        # Run config
        result = cli_runner.invoke(app, ["config"])

        # Assertions
        assert result.exit_code == 1
        assert "error" in result.stdout.lower()

    @patch("vectorflow.cli.main.Config")
    def test_config_short_option(
        self,
        mock_config_class,
        cli_runner: CliRunner,
        test_config: Config,
    ):
        """Test config display with short option flag."""
        mock_config_class.return_value = test_config

        # Run config with short option
        result = cli_runner.invoke(app, ["config", "-a"])

        # Assertions
        assert result.exit_code == 0

    @patch("vectorflow.cli.main.Config")
    def test_config_displays_embedding_settings(
        self,
        mock_config_class,
        cli_runner: CliRunner,
        test_config: Config,
    ):
        """Test that config displays embedding settings."""
        mock_config_class.return_value = test_config

        # Run config
        result = cli_runner.invoke(app, ["config"])

        # Assertions
        assert result.exit_code == 0
        assert "Model Type" in result.stdout or "model" in result.stdout.lower()
        assert "Batch Size" in result.stdout or "batch" in result.stdout.lower()

    @patch("vectorflow.cli.main.Config")
    def test_config_displays_vector_db_settings(
        self,
        mock_config_class,
        cli_runner: CliRunner,
        test_config: Config,
    ):
        """Test that config displays vector database settings."""
        mock_config_class.return_value = test_config

        # Run config
        result = cli_runner.invoke(app, ["config"])

        # Assertions
        assert result.exit_code == 0
        assert "Collection" in result.stdout or "collection" in result.stdout.lower()
        assert "Directory" in result.stdout or "directory" in result.stdout.lower()

    @patch("vectorflow.cli.main.Config")
    def test_config_show_all_displays_text_processing(
        self,
        mock_config_class,
        cli_runner: CliRunner,
        test_config: Config,
    ):
        """Test that config --show-all displays text processing settings."""
        mock_config_class.return_value = test_config

        # Run config with --show-all
        result = cli_runner.invoke(app, ["config", "--show-all"])

        # Assertions
        assert result.exit_code == 0
        assert "Chunk Size" in result.stdout or "chunk" in result.stdout.lower()

    @patch("vectorflow.cli.main.Config")
    def test_config_show_all_displays_performance(
        self,
        mock_config_class,
        cli_runner: CliRunner,
        test_config: Config,
    ):
        """Test that config --show-all displays performance settings."""
        mock_config_class.return_value = test_config

        # Run config with --show-all
        result = cli_runner.invoke(app, ["config", "--show-all"])

        # Assertions
        assert result.exit_code == 0
        assert "Workers" in result.stdout or "Debug" in result.stdout

    @patch("vectorflow.cli.main.Config")
    def test_config_masks_api_keys(
        self,
        mock_config_class,
        cli_runner: CliRunner,
        test_config: Config,
    ):
        """Test that config masks sensitive API keys."""
        # Set an API key
        test_config.embedding.openai_api_key = "sk-test-key-12345"
        mock_config_class.return_value = test_config

        # Run config with --show-all
        result = cli_runner.invoke(app, ["config", "--show-all"])

        # Assertions
        assert result.exit_code == 0
        # API key should be masked
        assert "sk-test-key-12345" not in result.stdout
        if "OpenAI" in result.stdout or "API Key" in result.stdout:
            assert "***" in result.stdout or "configured" in result.stdout.lower()
