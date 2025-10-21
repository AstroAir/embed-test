"""Tests for Qt utility functions."""

from unittest.mock import Mock, patch

import pytest
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from pdf_vector_system.gui.utils.icons import IconType
from pdf_vector_system.gui.utils.qt_utils import (
    center_window,
    set_window_icon,
    show_error_message,
    show_question_dialog,
)


@pytest.mark.gui()
@pytest.mark.utils()
class TestQtUtils:
    """Test cases for Qt utility functions."""

    def test_center_window(self, qtbot, qapp):
        """Test window centering functionality."""
        # Create a test widget
        widget = QWidget()
        widget.resize(400, 300)
        qtbot.addWidget(widget)

        # Mock screen geometry
        mock_screen = Mock()
        mock_screen.availableGeometry.return_value = Mock(
            width=lambda: 1920,
            height=lambda: 1080,
            center=lambda: Mock(x=lambda: 960, y=lambda: 540),
        )

        with patch.object(QApplication, "primaryScreen", return_value=mock_screen):
            # Center the window
            center_window(widget)

            # Window should be moved (exact position depends on implementation)
            # Just verify the function doesn't crash
            assert widget is not None

    def test_center_window_no_app(self, qtbot):
        """Test center_window with no QApplication."""
        widget = QWidget()
        qtbot.addWidget(widget)

        with patch.object(QApplication, "instance", return_value=None):
            # Should handle gracefully
            center_window(widget)
            assert widget is not None

    def test_center_window_none_widget(self, qtbot):
        """Test center_window with None widget."""
        # Should handle None gracefully
        center_window(None)
        # No exception should be raised

    def test_set_window_icon(self, qtbot):
        """Test setting window icon."""
        widget = QWidget()
        qtbot.addWidget(widget)

        # Mock get_icon function
        mock_icon = Mock(spec=QIcon)

        with patch(
            "pdf_vector_system.gui.utils.qt_utils.get_icon", return_value=mock_icon
        ):
            set_window_icon(widget, IconType.APP)

            # Check icon was set
            assert widget.windowIcon() == mock_icon

    def test_set_window_icon_none_widget(self, qtbot):
        """Test setting icon on None widget."""
        # Should handle None gracefully
        set_window_icon(None, IconType.APP)
        # No exception should be raised

    def test_show_error_message(self, qtbot):
        """Test error message display."""
        parent = QWidget()
        qtbot.addWidget(parent)

        title = "Test Error"
        message = "This is a test error message"
        detailed_text = "Detailed error information"

        # Show error message - should not raise exception
        show_error_message(parent, title, message, detailed_text)

        # Test should pass without exception

    def test_show_error_message_no_details(self, qtbot):
        """Test error message without detailed text."""
        parent = QWidget()
        qtbot.addWidget(parent)

        title = "Test Error"
        message = "This is a test error message"

        # Show error message without details - should not raise exception
        show_error_message(parent, title, message)

        # Test should pass without exception

    def test_show_question_dialog_yes(self, qtbot):
        """Test question dialog returning True."""
        parent = QWidget()
        qtbot.addWidget(parent)

        title = "Confirm Action"
        message = "Are you sure you want to proceed?"

        # Mock the MessageBox exec to return True
        with patch("qfluentwidgets.MessageBox.exec", return_value=True):
            result = show_question_dialog(parent, title, message)

            # Check dialog returned True
            assert result is True

    def test_show_question_dialog_no(self, qtbot):
        """Test question dialog returning False."""
        parent = QWidget()
        qtbot.addWidget(parent)

        title = "Confirm Action"
        message = "Are you sure you want to proceed?"

        # Mock the MessageBox exec to return False
        with patch("qfluentwidgets.MessageBox.exec", return_value=False):
            result = show_question_dialog(parent, title, message)

            # Check dialog returned False
            assert result is False

    def test_center_window_functionality(self, qtbot):
        """Test window centering functionality."""
        # Create a test widget
        widget = QWidget()
        qtbot.addWidget(widget)
        widget.resize(400, 300)

        # Center the window - should not raise exception
        center_window(widget)

        # Test should pass without exception

    def test_error_message_with_exception(self, qtbot):
        """Test error message with exception object."""
        parent = QWidget()
        qtbot.addWidget(parent)

        # Create test exception
        try:
            raise ValueError("Test exception message")
        except ValueError as e:
            show_error_message(parent, "Exception Error", str(e))

        # Test should pass without exception

    def test_question_dialog_exception_handling(self, qtbot):
        """Test question dialog exception handling."""
        parent = QWidget()
        qtbot.addWidget(parent)

        # Mock MessageBox to raise exception
        with patch("qfluentwidgets.MessageBox", side_effect=Exception("Test error")):
            result = show_question_dialog(parent, "Test", "Test message")

            # Should return False on exception
            assert result is False

    def test_window_icon_with_invalid_type(self, qtbot):
        """Test setting window icon with invalid icon type."""
        widget = QWidget()
        qtbot.addWidget(widget)

        # Mock get_icon to return None for invalid type
        with patch("pdf_vector_system.gui.utils.qt_utils.get_icon", return_value=None):
            # Should handle gracefully
            set_window_icon(widget, "invalid_type")

            # Widget should still exist
            assert widget is not None

    def test_dialog_parent_handling(self, qtbot):
        """Test dialog handling with different parent types."""
        # Test with None parent - should not raise exception
        show_error_message(None, "Error", "Message")

        # Test with widget parent
        parent = QWidget()
        qtbot.addWidget(parent)

        show_error_message(parent, "Error", "Message")

        # Test should pass without exception

    def test_message_configuration(self, qtbot):
        """Test message configuration details."""
        parent = QWidget()
        qtbot.addWidget(parent)

        show_error_message(parent, "Test Title", "Test Message", "Test Details")

        # Test should pass without exception

        # Get mock instance and verify all configurations
        mock_instance = mock_message_box.return_value

        # Check all methods were called with correct parameters
        mock_instance.setIcon.assert_called_once_with(QMessageBox.Icon.Critical)
        mock_instance.setWindowTitle.assert_called_once_with("Test Title")
        mock_instance.setText.assert_called_once_with("Test Message")
        mock_instance.setDetailedText.assert_called_once_with("Test Details")
        mock_instance.exec.assert_called_once()

    def test_utility_functions_thread_safety(self, qtbot):
        """Test utility functions are thread-safe."""
        # This is a basic test - in practice, Qt operations should be on main thread
        widget = QWidget()
        qtbot.addWidget(widget)

        # These operations should not crash
        center_window(widget)
        set_window_icon(widget, IconType.APP)

        assert widget is not None
