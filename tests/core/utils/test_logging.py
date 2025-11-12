"""Tests for logging utility classes and functions."""

from unittest.mock import patch

from pdf_vector_system.core.config.settings import LoggingConfig, LogLevel
from pdf_vector_system.core.utils.logging import (
    LoggerMixin,
    get_logger,
    log_error,
    log_function_call,
    log_performance,
    log_progress,
    log_warning,
    setup_logging,
)


class TestLoggerMixin:
    """Test LoggerMixin class."""

    def test_logger_property(self):
        """Test logger property provides correct logger."""

        class TestClass(LoggerMixin):
            pass

        instance = TestClass()
        logger_instance = instance.logger

        # Should have a logger with the class name
        assert logger_instance is not None
        # The logger should be a loguru logger (has _core attribute)
        assert hasattr(logger_instance, "_core")

    def test_different_classes_get_different_loggers(self):
        """Test that different classes get loggers with their names."""

        class ClassA(LoggerMixin):
            pass

        class ClassB(LoggerMixin):
            pass

        instance_a = ClassA()
        instance_b = ClassB()

        # Both should have loggers, but they should be different contexts
        assert instance_a.logger is not None
        assert instance_b.logger is not None
        # They should have different contexts based on class names
        assert instance_a.logger != instance_b.logger

    def test_logger_inheritance(self):
        """Test logger behavior with inheritance."""

        class BaseClass(LoggerMixin):
            pass

        class DerivedClass(BaseClass):
            pass

        base_instance = BaseClass()
        derived_instance = DerivedClass()

        # Both should have loggers
        assert base_instance.logger is not None
        assert derived_instance.logger is not None


class TestLoggingFunctions:
    """Test logging utility functions."""

    def test_get_logger(self):
        """Test get_logger function."""
        logger = get_logger("test_logger")
        assert logger is not None

    @patch("pdf_vector_system.utils.logging.logger")
    def test_log_function_call(self, mock_logger):
        """Test log_function_call function."""
        log_function_call("test_function", param1="value1", param2=42)

        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0][0]
        assert "test_function" in call_args
        assert "param1=value1" in call_args
        assert "param2=42" in call_args

    @patch("pdf_vector_system.utils.logging.logger")
    def test_log_performance(self, mock_logger):
        """Test log_performance function."""
        log_performance("test_operation", 1.5, items=100, rate=66.7)

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "test_operation" in call_args
        assert "1.50s" in call_args
        assert "items=100" in call_args
        assert "rate=66.7" in call_args

    @patch("pdf_vector_system.utils.logging.logger")
    def test_log_error(self, mock_logger):
        """Test log_error function."""
        test_error = ValueError("Test error message")
        log_error(test_error, "Test context")

        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        assert "ValueError" in call_args
        assert "Test error message" in call_args
        assert "Test context" in call_args

    @patch("pdf_vector_system.utils.logging.logger")
    def test_log_error_without_context(self, mock_logger):
        """Test log_error function without context."""
        test_error = RuntimeError("Runtime error")
        log_error(test_error)

        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        assert "RuntimeError" in call_args
        assert "Runtime error" in call_args

    @patch("pdf_vector_system.utils.logging.logger")
    def test_log_warning(self, mock_logger):
        """Test log_warning function."""
        log_warning("Test warning", severity="high", component="test")

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        assert "Test warning" in call_args
        assert "severity=high" in call_args
        assert "component=test" in call_args

    @patch("pdf_vector_system.utils.logging.logger")
    def test_log_progress(self, mock_logger):
        """Test log_progress function."""
        log_progress("processing", 75, 100, stage="final")

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "processing" in call_args
        assert "75/100" in call_args
        assert "75.0%" in call_args
        assert "stage=final" in call_args


class TestSetupLogging:
    """Test setup_logging function."""

    @patch("pdf_vector_system.utils.logging.logger")
    def test_setup_logging_console_only(self, mock_logger):
        """Test setup_logging with console output only."""
        config = LoggingConfig(level=LogLevel.INFO, file_path=None, format="{message}")

        setup_logging(config)

        # Should remove existing handlers and add console handler
        mock_logger.remove.assert_called_once()
        assert mock_logger.add.call_count >= 1

    @patch("pdf_vector_system.utils.logging.logger")
    def test_setup_logging_with_file(self, mock_logger, temp_dir):
        """Test setup_logging with file output."""
        log_file = temp_dir / "test.log"
        config = LoggingConfig(
            level=LogLevel.DEBUG,
            file_path=log_file,
            format="{time} | {message}",
            rotation="1 MB",
            retention="7 days",
        )

        setup_logging(config)

        # Should add both console and file handlers
        mock_logger.remove.assert_called_once()
        assert mock_logger.add.call_count >= 2

    @patch("pdf_vector_system.utils.logging.logger")
    def test_setup_logging_creates_directory(self, mock_logger, temp_dir):
        """Test that setup_logging creates log directory if it doesn't exist."""
        log_dir = temp_dir / "logs"
        log_file = log_dir / "test.log"

        # Directory doesn't exist initially
        assert not log_dir.exists()

        config = LoggingConfig(level=LogLevel.INFO, file_path=log_file)

        setup_logging(config)

        # Directory should be created
        assert log_dir.exists()
        assert log_dir.is_dir()

    @patch("pdf_vector_system.utils.logging.logger")
    def test_setup_logging_different_levels(self, mock_logger):
        """Test setup_logging with different log levels."""
        for level in LogLevel:
            config = LoggingConfig(level=level)
            setup_logging(config)

            # Should call logger.add with the correct level
            assert mock_logger.add.called
            # Reset for next iteration
            mock_logger.reset_mock()

    @patch("pdf_vector_system.utils.logging.logger")
    def test_setup_logging_custom_format(self, mock_logger):
        """Test setup_logging with custom format."""
        custom_format = "{time:YYYY-MM-DD} | {level} | {message}"
        config = LoggingConfig(level=LogLevel.INFO, format=custom_format)

        setup_logging(config)

        # Should use the custom format
        mock_logger.add.assert_called()
        # Check that format is passed to add method
        call_args = mock_logger.add.call_args
        assert any(
            custom_format in str(arg)
            for arg in call_args[0] + tuple(call_args[1].values())
        )

    @patch("pdf_vector_system.utils.logging.logger")
    def test_setup_logging_rotation_and_retention(self, mock_logger, temp_dir):
        """Test setup_logging with rotation and retention settings."""
        log_file = temp_dir / "rotating.log"
        config = LoggingConfig(
            level=LogLevel.INFO,
            file_path=log_file,
            rotation="5 MB",
            retention="14 days",
        )

        setup_logging(config)

        # Should configure rotation and retention
        mock_logger.add.assert_called()
        # Verify that rotation and retention parameters are used
        call_kwargs = mock_logger.add.call_args[1]
        assert "rotation" in call_kwargs or any(
            "5 MB" in str(arg) for arg in mock_logger.add.call_args[0]
        )
        assert "retention" in call_kwargs or any(
            "14 days" in str(arg) for arg in mock_logger.add.call_args[0]
        )
