"""
Qt utility functions for PDF Vector System GUI.

This module provides common Qt utility functions for window management,
dialogs, and other GUI operations.
"""

from typing import Optional, Union
from PySide6.QtWidgets import (
    QWidget, QApplication
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QScreen
from qfluentwidgets import MessageBox

from .icons import get_icon, IconType


def center_window(window: QWidget) -> None:
    """
    Center a window on the screen.

    Args:
        window: Window to center
    """
    if not window:
        return

    # Get screen geometry
    screen: Optional[QScreen] = QApplication.primaryScreen()
    if screen:
        screen_geometry = screen.availableGeometry()
        window_geometry = window.frameGeometry()

        # Calculate center position
        center_point = screen_geometry.center()
        window_geometry.moveCenter(center_point)
        window.move(window_geometry.topLeft())


def set_window_icon(window: QWidget, icon_type: IconType) -> None:
    """
    Set window icon.

    Args:
        window: Window to set icon for
        icon_type: Type of icon to use
    """
    icon: Optional[QIcon] = get_icon(icon_type)
    if icon and window:
        window.setWindowIcon(icon)


def show_error_message(
    parent: Optional[QWidget],
    title: str,
    message: str,
    detailed_text: Optional[str] = None
) -> None:
    """
    Show an error message dialog using QFluentWidgets.

    Args:
        parent: Parent widget
        title: Dialog title
        message: Error message
        detailed_text: Optional detailed error text (currently not supported by QFluentWidgets MessageBox)
    """
    # Use QFluentWidgets MessageBox for consistent fluent design
    full_message = message
    if detailed_text:
        full_message += f"\n\nDetails:\n{detailed_text}"

    w = MessageBox(title, full_message, parent)
    w.cancelButton.setText("OK")
    w.yesButton.hide()  # Hide yes button for simple acknowledgment
    w.exec()


def show_info_message(
    parent: Optional[QWidget],
    title: str,
    message: str
) -> None:
    """
    Show an information message dialog using QFluentWidgets.

    Args:
        parent: Parent widget
        title: Dialog title
        message: Information message
    """
    w = MessageBox(title, message, parent)
    w.cancelButton.setText("OK")
    w.yesButton.hide()  # Hide yes button for simple acknowledgment
    w.exec()


def show_warning_message(
    parent: Optional[QWidget],
    title: str,
    message: str
) -> None:
    """
    Show a warning message dialog using QFluentWidgets.

    Args:
        parent: Parent widget
        title: Dialog title
        message: Warning message
    """
    w = MessageBox(title, message, parent)
    w.cancelButton.setText("OK")
    w.yesButton.hide()  # Hide yes button for simple acknowledgment
    w.exec()


def show_question_dialog(
    parent: Optional[QWidget],
    title: str,
    message: str
) -> bool:
    """
    Show a yes/no question dialog using QFluentWidgets.

    Args:
        parent: Parent widget
        title: Dialog title
        message: Question message

    Returns:
        True if user clicked Yes, False otherwise
    """
    try:
        w = MessageBox(title, message, parent)
        w.yesButton.setText("Yes")
        w.cancelButton.setText("No")
        result = w.exec()
        return bool(result)  # Ensure boolean return type
    except Exception as e:
        # Fallback to simple print if MessageBox fails
        print(f"Error showing question dialog: {e}")
        return False
