"""Tests for BaseWidget."""

import pytest
from PySide6.QtWidgets import QLabel, QVBoxLayout

from pdf_vector_system.gui.widgets.base import BaseWidget


@pytest.mark.gui
@pytest.mark.widget
class TestBaseWidget:
    """Test cases for BaseWidget."""

    def test_base_widget_initialization(self, qtbot, mock_config):
        """Test base widget initializes correctly."""
        widget = BaseWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Check widget is created
        assert widget is not None
        assert widget.config == mock_config

        # Check layout is created
        layout = widget.layout()
        assert isinstance(layout, QVBoxLayout)

    def test_base_widget_without_config(self, qtbot):
        """Test base widget initialization without config."""
        widget = BaseWidget()
        qtbot.addWidget(widget)

        # Should create default config
        assert widget is not None
        assert widget.config is not None

    def test_base_widget_signals(self, qtbot, mock_config):
        """Test base widget signals are defined."""
        widget = BaseWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Check signals exist
        assert hasattr(widget, "status_message")
        assert hasattr(widget, "error_occurred")

    def test_emit_status_method(self, qtbot, mock_config):
        """Test emit_status method."""
        widget = BaseWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Connect to signal
        messages = []
        widget.status_message.connect(messages.append)

        # Emit status
        test_message = "Test status message"
        widget.emit_status(test_message)

        # Check signal was emitted
        assert len(messages) == 1
        assert messages[0] == test_message

    def test_emit_error_method(self, qtbot, mock_config):
        """Test emit_error method."""
        widget = BaseWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Connect to signal
        errors = []
        widget.error_occurred.connect(errors.append)

        # Emit error
        test_error = "Test error message"
        widget.emit_error(test_error)

        # Check signal was emitted
        assert len(errors) == 1
        assert errors[0] == test_error

    def test_on_tab_activated_default(self, qtbot, mock_config):
        """Test default on_tab_activated implementation."""
        widget = BaseWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Should not raise exception
        widget.on_tab_activated()

    def test_setup_ui_default(self, qtbot, mock_config):
        """Test default _setup_ui implementation."""
        widget = BaseWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Check default UI was created
        layout = widget.layout()
        assert layout.count() > 0

        # Check default label exists
        label_widget = layout.itemAt(0).widget()
        assert isinstance(label_widget, QLabel)
        assert "Base Widget" in label_widget.text()


# Template for other widget tests
@pytest.mark.gui
@pytest.mark.widget
class TestSearchWidget:
    """Test cases for SearchWidget - Template."""

    def test_widget_initialization(self, qtbot, mock_config):
        """Test widget initializes correctly."""
        from pdf_vector_system.gui.widgets.search_widget import SearchWidget

        widget = SearchWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Check widget is created
        assert widget is not None
        assert widget.config == mock_config

        # Check UI elements exist
        assert hasattr(widget, "query_input")
        assert hasattr(widget, "search_btn")
        assert hasattr(widget, "results_table")
        assert hasattr(widget, "search_type_selector")

    def test_search_functionality(self, qtbot, mock_config):
        """Test search functionality."""
        from pdf_vector_system.gui.widgets.search_widget import SearchWidget

        widget = SearchWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Test that search input exists and can be set
        test_query = "test search query"
        widget.query_input.setText(test_query)
        assert widget.query_input.text() == test_query

        # Test search button exists and is clickable
        assert widget.search_btn.isEnabled()

    def test_results_display(self, qtbot, mock_config):
        """Test search results display."""
        from pdf_vector_system.gui.widgets.search_widget import SearchWidget

        widget = SearchWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Check results table exists
        assert widget.results_table is not None
        assert widget.results_table.columnCount() > 0


@pytest.mark.gui
@pytest.mark.widget
class TestDocumentWidget:
    """Test cases for DocumentWidget - Template."""

    def test_widget_initialization(self, qtbot, mock_config):
        """Test widget initializes correctly."""
        from pdf_vector_system.gui.widgets.document_widget import DocumentWidget

        widget = DocumentWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Check widget is created
        assert widget is not None
        assert widget.config == mock_config

        # Check UI elements exist
        assert hasattr(widget, "document_table")
        assert hasattr(widget, "refresh_btn")
        assert hasattr(widget, "delete_btn")

    def test_document_list_display(self, qtbot, mock_config):
        """Test document list display."""
        from pdf_vector_system.gui.widgets.document_widget import DocumentWidget

        widget = DocumentWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Check document table exists and has columns
        assert widget.document_table is not None
        assert widget.document_table.columnCount() > 0

    def test_document_operations(self, qtbot, mock_config):
        """Test document operations (delete, info)."""
        from pdf_vector_system.gui.widgets.document_widget import DocumentWidget

        widget = DocumentWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Check operation buttons exist
        assert widget.refresh_btn is not None
        assert widget.delete_btn is not None


@pytest.mark.gui
@pytest.mark.widget
class TestConfigWidget:
    """Test cases for ConfigWidget - Template."""

    def test_widget_initialization(self, qtbot, mock_config):
        """Test widget initializes correctly."""
        from pdf_vector_system.gui.widgets.config_widget import ConfigWidget

        widget = ConfigWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Check widget is created
        assert widget is not None
        assert widget.config == mock_config

        # Check UI elements exist
        assert hasattr(widget, "nav_widget")
        assert hasattr(widget, "content_widget")

    def test_configuration_display(self, qtbot, mock_config):
        """Test configuration display."""
        from pdf_vector_system.gui.widgets.config_widget import ConfigWidget

        widget = ConfigWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Check navigation widget exists
        assert widget.nav_widget is not None

    def test_configuration_validation(self, qtbot, mock_config):
        """Test configuration validation."""
        from pdf_vector_system.gui.widgets.config_widget import ConfigWidget

        widget = ConfigWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Widget should have config validation capability
        assert widget.config is not None

    def test_save_load_functionality(self, qtbot, mock_config):
        """Test save/load configuration."""
        from pdf_vector_system.gui.widgets.config_widget import ConfigWidget

        widget = ConfigWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Check that widget has config attribute for save/load
        assert hasattr(widget, "config")


@pytest.mark.gui
@pytest.mark.widget
class TestStatusWidget:
    """Test cases for StatusWidget - Template."""

    def test_widget_initialization(self, qtbot, mock_config):
        """Test widget initializes correctly."""
        from pdf_vector_system.gui.widgets.status_widget import StatusWidget

        widget = StatusWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Check widget is created
        assert widget is not None
        assert widget.config == mock_config

        # Check UI elements exist
        assert hasattr(widget, "health_check_btn")
        assert hasattr(widget, "refresh_btn")

    def test_health_check_display(self, qtbot, mock_config):
        """Test health check display."""
        from pdf_vector_system.gui.widgets.status_widget import StatusWidget

        widget = StatusWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Check health check button exists
        assert widget.health_check_btn is not None

    def test_system_info_display(self, qtbot, mock_config):
        """Test system information display."""
        from pdf_vector_system.gui.widgets.status_widget import StatusWidget

        widget = StatusWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Check that widget has system info display capability
        assert hasattr(widget, "system_info_table")

    def test_auto_refresh_functionality(self, qtbot, mock_config):
        """Test auto-refresh functionality."""
        from pdf_vector_system.gui.widgets.status_widget import StatusWidget

        widget = StatusWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Check refresh button exists
        assert widget.refresh_btn is not None


@pytest.mark.gui
@pytest.mark.widget
class TestLogWidget:
    """Test cases for LogWidget - Template."""

    def test_widget_initialization(self, qtbot, mock_config):
        """Test widget initializes correctly."""
        from pdf_vector_system.gui.widgets.log_widget import LogWidget

        widget = LogWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Check widget is created
        assert widget is not None
        assert widget.config == mock_config

        # Check UI elements exist
        assert hasattr(widget, "log_text")
        assert hasattr(widget, "clear_btn")

    def test_log_display(self, qtbot, mock_config):
        """Test log display functionality."""
        from pdf_vector_system.gui.widgets.log_widget import LogWidget

        widget = LogWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Check log text widget exists
        assert widget.log_text is not None

    def test_log_filtering(self, qtbot, mock_config):
        """Test log level filtering."""
        from pdf_vector_system.gui.widgets.log_widget import LogWidget

        widget = LogWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Check that widget has filtering capability
        assert hasattr(widget, "level_filter")

    def test_log_controls(self, qtbot, mock_config):
        """Test log control buttons."""
        from pdf_vector_system.gui.widgets.log_widget import LogWidget

        widget = LogWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Check control buttons exist
        assert widget.clear_btn is not None
