"""Pytest configuration and fixtures for GUI tests."""

import sys
from unittest.mock import Mock, patch

import pytest
from PySide6.QtCore import Qt

# Import Qt modules
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

# Import GUI components
from pdf_vector_system.config.settings import Config
from pdf_vector_system.gui.app import PDFVectorGUIApp
from pdf_vector_system.gui.main_window import MainWindow
from pdf_vector_system.pipeline import PDFVectorPipeline


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for GUI tests."""
    # Check if QApplication already exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Set test-specific properties
    app.setApplicationName("PDF Vector System Test")
    app.setOrganizationName("Test")

    yield app

    # Clean up
    if app:
        app.quit()


@pytest.fixture()
def qtbot(qapp, request):
    """Create QtBot instance for widget testing."""
    return QtBot(request)


@pytest.fixture()
def mock_pipeline():
    """Create a mock PDFVectorPipeline for testing."""
    mock = Mock(spec=PDFVectorPipeline)

    # Configure common return values
    mock.process_pdf.return_value = {
        "document_id": "test_doc_123",
        "chunks_created": 5,
        "processing_time": 1.23,
        "success": True,
    }

    mock.search.return_value = [
        Mock(
            id="chunk_1",
            content="Test search result 1",
            score=0.95,
            metadata={"document_id": "test_doc", "page_number": 1},
        ),
        Mock(
            id="chunk_2",
            content="Test search result 2",
            score=0.87,
            metadata={"document_id": "test_doc", "page_number": 2},
        ),
    ]

    mock.get_documents.return_value = [
        {
            "document_id": "test_doc_1",
            "filename": "test1.pdf",
            "chunks_count": 5,
            "total_characters": 1000,
            "created_at": "2024-01-01T00:00:00",
        },
        {
            "document_id": "test_doc_2",
            "filename": "test2.pdf",
            "chunks_count": 3,
            "total_characters": 600,
            "created_at": "2024-01-02T00:00:00",
        },
    ]

    mock.delete_document.return_value = 5  # chunks deleted
    mock.get_document_info.return_value = {
        "document_id": "test_doc",
        "filename": "test.pdf",
        "chunks_count": 5,
        "total_characters": 1000,
        "pages": 2,
        "created_at": "2024-01-01T00:00:00",
    }

    mock.get_collection_stats.return_value = {
        "total_documents": 10,
        "total_chunks": 50,
        "total_characters": 10000,
        "collection_name": "test_collection",
    }

    mock.health_check.return_value = {
        "pipeline": True,
        "chromadb": True,
        "embedding_service": True,
        "pdf_processor": True,
        "configuration": True,
    }

    return mock


@pytest.fixture()
def mock_config(temp_dir):
    """Create a mock configuration for testing."""
    config = Mock(spec=Config)
    config.debug = True
    config.max_workers = 2

    # Create nested mock objects
    config.chroma_db = Mock()
    config.chroma_db.persist_directory = temp_dir / "test_chroma"
    config.chroma_db.collection_name = "test_collection"

    config.embedding = Mock()
    config.embedding.model_type = Mock()
    config.embedding.model_type.value = "sentence_transformers"
    config.embedding.model_name = "test-model"
    config.embedding.batch_size = 16
    config.embedding.openai_api_key = ""
    config.embedding.azure_openai_endpoint = ""
    config.embedding.azure_openai_api_key = ""
    config.embedding.azure_openai_api_version = "2023-05-15"
    config.embedding.google_api_key = ""
    config.embedding.google_project_id = ""
    config.embedding.google_use_version = "v1"
    config.embedding.google_gemini_location = "us-central1"

    config.text_processing = Mock()
    config.text_processing.chunk_size = 1000
    config.text_processing.chunk_overlap = 200
    config.text_processing.min_chunk_size = 100
    config.text_processing.separators = ["\n\n", "\n", " ", ""]

    config.pdf = Mock()
    config.pdf.max_file_size_mb = 100
    config.pdf.timeout_seconds = 300
    config.pdf.extract_images = False

    config.logging = Mock()
    config.logging.level = "DEBUG"

    return config


@pytest.fixture()
def mock_file_dialog():
    """Mock file dialog for testing file selection."""
    with patch("PySide6.QtWidgets.QFileDialog") as mock:
        # Configure getOpenFileNames to return test files
        mock.getOpenFileNames.return_value = (
            ["/test/file1.pdf", "/test/file2.pdf"],
            "PDF Files (*.pdf)",
        )

        # Configure getExistingDirectory to return test directory
        mock.getExistingDirectory.return_value = "/test/directory"

        # Configure getSaveFileName to return test save path
        mock.getSaveFileName.return_value = ("/test/config.json", "JSON Files (*.json)")

        yield mock


@pytest.fixture()
def mock_message_box():
    """Mock message box for testing dialogs."""
    with patch("PySide6.QtWidgets.QMessageBox") as mock:
        # Configure standard responses
        mock.question.return_value = mock.StandardButton.Yes
        mock.information.return_value = mock.StandardButton.Ok
        mock.warning.return_value = mock.StandardButton.Ok
        mock.critical.return_value = mock.StandardButton.Ok

        yield mock


@pytest.fixture()
def gui_app(qapp, mock_config):
    """Create GUI application instance for testing."""
    return PDFVectorGUIApp(config=mock_config)


@pytest.fixture()
def main_window(qtbot, mock_config):
    """Create main window instance for testing."""
    window = MainWindow(config=mock_config)
    qtbot.addWidget(window)
    return window


@pytest.fixture()
def widget_test_timeout():
    """Timeout for widget operations in milliseconds."""
    return 1000


# Mock system resources for status monitoring
@pytest.fixture()
def mock_system_resources():
    """Mock system resource monitoring."""
    with (
        patch("psutil.cpu_percent") as mock_cpu,
        patch("psutil.virtual_memory") as mock_memory,
    ):
        mock_cpu.return_value = 25.5
        mock_memory.return_value = Mock(percent=45.2)

        yield {"cpu": mock_cpu, "memory": mock_memory}


# Helper functions for GUI testing
def wait_for_signal(qtbot, signal, timeout=1000):
    """Wait for a Qt signal to be emitted."""
    with qtbot.waitSignal(signal, timeout=timeout):
        pass


def click_button(qtbot, button, delay=100):
    """Click a button with optional delay."""
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    if delay:
        qtbot.wait(delay)


def enter_text(qtbot, widget, text, delay=50):
    """Enter text into a widget."""
    widget.clear()
    qtbot.keyClicks(widget, text)
    if delay:
        qtbot.wait(delay)


def select_table_row(qtbot, table, row):
    """Select a row in a table widget."""
    table.selectRow(row)
    qtbot.wait(50)


# Pytest markers for GUI tests
pytest.mark.gui = pytest.mark.gui
pytest.mark.widget = pytest.mark.widget
pytest.mark.controller = pytest.mark.controller
pytest.mark.dialog = pytest.mark.dialog
pytest.mark.integration = pytest.mark.integration
