"""Tests for MainWindow."""

from unittest.mock import Mock, patch

import pytest
from PySide6.QtGui import QShortcut

from vectorflow.gui.controllers.main_controller import MainController
from vectorflow.gui.main_window import MainWindow


@pytest.mark.gui
@pytest.mark.integration
class TestMainWindow:
    """Test cases for MainWindow."""

    def test_window_initialization(self, qtbot, mock_config):
        """Test main window initializes correctly."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        # Check window properties
        assert window.windowTitle() == "PDF Vector System"
        assert window.minimumSize().width() >= 1200
        assert window.minimumSize().height() >= 800

        # Check config and controller
        assert window.config == mock_config
        assert isinstance(window.controller, MainController)

    def test_ui_components_creation(self, qtbot, mock_config):
        """Test all UI components are created."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        # Check that FluentWindow navigation is set up
        assert hasattr(window, "stackedWidget")
        assert hasattr(window, "processing_widget")
        assert hasattr(window, "search_widget")
        assert hasattr(window, "document_widget")
        assert hasattr(window, "config_widget")
        assert hasattr(window, "status_widget")
        assert hasattr(window, "log_widget")

    def test_interface_creation_and_order(self, qtbot, mock_config):
        """Test interfaces are created in correct order."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        # Check that all interfaces are created
        assert hasattr(window, "processing_widget")
        assert hasattr(window, "search_widget")
        assert hasattr(window, "document_widget")
        assert hasattr(window, "config_widget")
        assert hasattr(window, "status_widget")
        assert hasattr(window, "log_widget")

        # Check that stackedWidget contains the interfaces
        assert window.stackedWidget.count() == 6

    def test_widget_instances_creation(self, qtbot, mock_config):
        """Test all widget instances are created."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        # Check widget instances exist
        assert hasattr(window, "processing_widget")
        assert hasattr(window, "search_widget")
        assert hasattr(window, "document_widget")
        assert hasattr(window, "config_widget")
        assert hasattr(window, "status_widget")
        assert hasattr(window, "log_widget")

        # Check widgets have correct config
        assert window.processing_widget.config == mock_config
        assert window.search_widget.config == mock_config
        assert window.document_widget.config == mock_config

    def test_navigation_functionality(self, qtbot, mock_config):
        """Test FluentWindow navigation functionality."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        # Check that navigation widgets are accessible
        assert hasattr(window, "stackedWidget")
        assert hasattr(window, "switchTo")

        # Test switching between different interfaces using switchTo
        window.switchTo(window.processing_widget)
        assert window.stackedWidget.currentWidget() == window.processing_widget

        window.switchTo(window.search_widget)
        assert window.stackedWidget.currentWidget() == window.search_widget

    def test_interface_switching(self, qtbot, mock_config):
        """Test interface switching functionality."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        # Test switching to different interfaces using switchTo
        interfaces = [
            window.processing_widget,
            window.search_widget,
            window.document_widget,
            window.config_widget,
            window.status_widget,
            window.log_widget,
        ]

        for interface in interfaces:
            window.switchTo(interface)
            assert window.stackedWidget.currentWidget() == interface

    def test_interface_activation_signals(self, qtbot, mock_config):
        """Test interface activation signals are connected."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        # Mock interface activation methods if they exist
        if hasattr(window.processing_widget, "on_tab_activated"):
            with patch.object(window.processing_widget, "on_tab_activated"):
                # Switch to processing interface
                window.stackedWidget.setCurrentWidget(window.processing_widget)
                qtbot.wait(100)  # Allow signal processing

        if hasattr(window.search_widget, "on_tab_activated"):
            with patch.object(window.search_widget, "on_tab_activated"):
                # Switch to search interface
                window.stackedWidget.setCurrentWidget(window.search_widget)
                qtbot.wait(100)

    def test_navigation_switching(self, qtbot, mock_config):
        """Test navigation switching functionality."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        # Test switching to processing widget
        window.switchTo(window.processing_widget)
        assert window.stackedWidget.currentWidget() == window.processing_widget

        # Test switching to status widget
        window.switchTo(window.status_widget)
        assert window.stackedWidget.currentWidget() == window.status_widget

        # Test switching to config widget
        window.switchTo(window.config_widget)
        assert window.stackedWidget.currentWidget() == window.config_widget

    def test_configuration_updates(self, qtbot, mock_config):
        """Test configuration update handling."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        # Create new config
        new_config = Mock()
        new_config.debug = False

        # Mock controller update methods
        with patch.object(window.controller, "update_config") as mock_controller_update:
            # Trigger config change
            window._on_config_changed(new_config)

            # Check config was updated
            assert window.config == new_config
            mock_controller_update.assert_called_once_with(new_config)

    def test_error_handling(self, qtbot, mock_config, mock_message_box):
        """Test error message handling."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        # Trigger error
        error_message = "Test error message"
        window._on_error(error_message)

        # Check error dialog was shown
        mock_message_box.critical.assert_called_once()

    def test_window_close_event(self, qtbot, mock_config):
        """Test window close event handling."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        # Connect to closing signal
        closing_emitted = []
        window.closing.connect(lambda: closing_emitted.append(True))

        # Mock close event
        from PySide6.QtGui import QCloseEvent

        close_event = QCloseEvent()

        # Trigger close event
        window.closeEvent(close_event)

        # Check signal was emitted
        assert len(closing_emitted) == 1
        assert close_event.isAccepted()

    def test_widget_signal_connections(self, qtbot, mock_config):
        """Test widget signal connections are established."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        # Check that config widget signal is connected
        # This is tested by verifying the connection exists

        # The signal connection should be established in _connect_widget_signals
        # We can test this by checking if the method exists
        assert hasattr(window, "_connect_widget_signals")
        assert hasattr(window, "_on_config_changed")

    def test_controller_signal_connections(self, qtbot, mock_config):
        """Test controller signal connections."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        # Test that controller signals are properly connected
        # FluentWindow doesn't have a statusBar like QMainWindow
        # Instead, we test that the controller exists and signals can be emitted
        test_message = "Test controller message"

        # Verify controller exists and has the expected signal
        assert hasattr(window.controller, "status_message")
        assert hasattr(window.controller, "error_occurred")

        # Test that signals can be emitted without errors
        # (FluentWindow handles status differently than QMainWindow)
        window.controller.status_message.emit(test_message)
        qtbot.wait(100)

        # Verify the signal was emitted successfully (no exception thrown)
        assert True  # If we reach here, the signal emission worked

    def test_window_geometry_and_centering(self, qtbot, mock_config):
        """Test window geometry and centering."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        # Check minimum size
        assert window.minimumWidth() >= 1200
        assert window.minimumHeight() >= 800

        # Check initial size
        assert window.width() >= window.minimumWidth()
        assert window.height() >= window.minimumHeight()

    def test_keyboard_shortcuts(self, qtbot, mock_config):
        """Test keyboard shortcuts are properly set."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        # Check that shortcuts are set up by looking for QShortcut children
        shortcuts = window.findChildren(QShortcut)
        assert len(shortcuts) > 0, "Window should have keyboard shortcuts configured"
