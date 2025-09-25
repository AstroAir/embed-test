"""Logging utilities for PDF Vector System using loguru."""

import sys
from pathlib import Path
from typing import Optional, Any, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger

from ..config.settings import LoggingConfig, LogLevel


def is_logging_configured() -> bool:
    """
    Check if logging has been configured (i.e., has custom handlers).

    Returns:
        True if logging is configured, False otherwise
    """
    # Check if there are any handlers configured
    # Default loguru has one handler, configured logging typically removes it and adds custom ones
    return len(logger._core.handlers) > 0


def ensure_logging_configured(config: Optional[LoggingConfig] = None) -> None:
    """
    Ensure logging is configured, setting up basic logging if not already done.

    Args:
        config: Optional logging configuration (uses default if None)
    """
    if not is_logging_configured():
        from ..config.settings import LoggingConfig
        setup_logging(config or LoggingConfig())


def setup_logging(config: LoggingConfig) -> None:
    """
    Set up logging configuration using loguru.

    Args:
        config: Logging configuration object
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler with color and formatting
    logger.add(
        sys.stderr,
        level=config.level.value,
        format=config.format,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    
    # Add file handler if file path is specified
    if config.file_path:
        # Ensure log directory exists
        log_dir = Path(config.file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            config.file_path,
            level=config.level.value,
            format=config.format,
            rotation=config.rotation,
            retention=config.retention,
            compression="gz",
            backtrace=True,
            diagnose=True,
        )
    
    # Log the configuration
    logger.info(f"Logging configured with level: {config.level.value}")
    if config.file_path:
        logger.info(f"Log file: {config.file_path}")


def get_logger(name: str) -> "Logger":
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logger.bind(name=name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> "Logger":
        """Get logger instance for this class."""
        return get_logger(self.__class__.__name__)


def log_function_call(func_name: str, **kwargs: Any) -> None:
    """
    Log a function call with its parameters.
    
    Args:
        func_name: Name of the function being called
        **kwargs: Function parameters to log
    """
    params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.debug(f"Calling {func_name}({params})")


def log_performance(operation: str, duration: float, **metadata: Any) -> None:
    """
    Log performance metrics for an operation.
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        **metadata: Additional metadata to log
    """
    metadata_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
    logger.info(f"Performance: {operation} completed in {duration:.2f}s | {metadata_str}")


def log_error(error: Exception, context: Optional[str] = None) -> None:
    """
    Log an error with context information.
    
    Args:
        error: Exception that occurred
        context: Additional context information
    """
    context_str = f" | Context: {context}" if context else ""
    logger.error(f"Error: {type(error).__name__}: {str(error)}{context_str}")


def log_warning(message: str, **metadata: Any) -> None:
    """
    Log a warning message with metadata.
    
    Args:
        message: Warning message
        **metadata: Additional metadata
    """
    metadata_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
    warning_msg = f"{message}"
    if metadata_str:
        warning_msg += f" | {metadata_str}"
    logger.warning(warning_msg)


def log_progress(operation: str, current: int, total: int, **metadata: Any) -> None:
    """
    Log progress information.
    
    Args:
        operation: Name of the operation
        current: Current progress count
        total: Total count
        **metadata: Additional metadata
    """
    percentage = (current / total) * 100 if total > 0 else 0
    metadata_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
    progress_msg = f"Progress: {operation} {current}/{total} ({percentage:.1f}%)"
    if metadata_str:
        progress_msg += f" | {metadata_str}"
    logger.info(progress_msg)
