"""Pytest configuration and fixtures for utility tests."""

import shutil
import tempfile
import time
from collections.abc import Generator
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from vectorflow.core.config.settings import LoggingConfig, LogLevel
from vectorflow.core.utils.logging import LoggerMixin


@pytest.fixture
def utils_temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory specifically for utils tests."""
    temp_path = Path(tempfile.mkdtemp(prefix="utils_test_"))
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_log_file(utils_temp_dir: Path) -> Path:
    """Create a test log file path."""
    return utils_temp_dir / "test.log"


@pytest.fixture
def logging_config_console_only() -> LoggingConfig:
    """Create logging config for console output only."""
    return LoggingConfig(
        level=LogLevel.DEBUG,
        file_path=None,
        format="{time} | {level} | {message}",
        rotation="1 MB",
        retention="1 day",
    )


@pytest.fixture
def logging_config_with_file(test_log_file: Path) -> LoggingConfig:
    """Create logging config with file output."""
    return LoggingConfig(
        level=LogLevel.INFO,
        file_path=test_log_file,
        format="{time} | {level} | {name} | {message}",
        rotation="10 MB",
        retention="30 days",
    )


@pytest.fixture
def mock_console():
    """Create a mock console for progress testing."""
    mock = Mock()
    mock.print = Mock()
    mock.status = Mock()
    return mock


@pytest.fixture
def mock_progress():
    """Create a mock progress instance."""
    mock = Mock()
    mock.add_task = Mock(return_value="task_id")
    mock.update = Mock()
    mock.remove_task = Mock()
    mock.stop = Mock()
    return mock


@pytest.fixture
def sample_performance_data() -> dict[str, Any]:
    """Provide sample performance data for testing."""
    return {
        "operation": "test_operation",
        "start_time": time.time() - 1.5,
        "end_time": time.time(),
        "duration": 1.5,
        "memory_usage": 1024 * 1024,  # 1MB
        "cpu_usage": 25.5,
    }


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    mock = Mock()
    mock.debug = Mock()
    mock.info = Mock()
    mock.warning = Mock()
    mock.error = Mock()
    mock.critical = Mock()
    mock.bind = Mock(return_value=mock)
    return mock


@pytest.fixture
def test_class_with_logger():
    """Create a test class that uses LoggerMixin."""

    class TestClass(LoggerMixin):
        def __init__(self):
            self.test_data = "test"

        def test_method(self):
            self.logger.info("Test method called")
            return "success"

    return TestClass


@pytest.fixture
def mock_rich_progress():
    """Create a mock Rich Progress instance."""
    mock = Mock()
    mock.__enter__ = Mock(return_value=mock)
    mock.__exit__ = Mock(return_value=None)
    mock.add_task = Mock(return_value="task_id")
    mock.update = Mock()
    mock.remove_task = Mock()
    mock.stop = Mock()
    return mock


@pytest.fixture
def mock_time_operation():
    """Create a mock for time operation testing."""

    def _mock_operation(duration: float = 0.1):
        """Mock operation that takes specified time."""
        time.sleep(duration)
        return "operation_result"

    return _mock_operation


@pytest.fixture
def captured_logs():
    """Capture log output for testing."""
    log_capture = StringIO()

    with patch("sys.stderr", log_capture):
        yield log_capture


@pytest.fixture
def mock_system_resources():
    """Mock system resource monitoring."""
    with (
        patch("psutil.cpu_percent") as mock_cpu,
        patch("psutil.virtual_memory") as mock_memory,
        patch("psutil.disk_usage") as mock_disk,
    ):
        mock_cpu.return_value = 25.5
        mock_memory.return_value = Mock(percent=45.2, available=1024 * 1024 * 1024)
        mock_disk.return_value = Mock(percent=60.0, free=1024 * 1024 * 1024 * 10)

        yield {"cpu": mock_cpu, "memory": mock_memory, "disk": mock_disk}
