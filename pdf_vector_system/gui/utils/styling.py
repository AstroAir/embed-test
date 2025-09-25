"""
Styling and theming for PDF Vector System GUI.

This module provides application styling and theme management.
"""

from typing import Optional


def get_app_style() -> Optional[str]:
    """
    Get the application stylesheet.

    Note: FluentWindow and QFluentWidgets have their own theming system.
    This stylesheet provides minimal additional styling that doesn't conflict
    with QFluentWidgets' built-in themes.

    Returns:
        CSS stylesheet string or None
    """
    return """
    /* FluentWindow styling - minimal to avoid conflicts with QFluentWidgets theming */
    QWidget {
        background-color: transparent;
    }

    /* Note: QTabWidget styles removed as FluentWindow uses its own navigation system */
    /* FluentWindow uses addSubInterface() instead of traditional tabs */

    /* Note: QPushButton styles removed - QFluentWidgets PushButton has its own styling */
    /* QFluentWidgets components (PushButton, LineEdit, etc.) use their own theming */

    /* Keep only native Qt widgets that aren't replaced by QFluentWidgets */

    /* Note: QGroupBox styles removed - QFluentWidgets uses CardWidget instead */
    /* Note: QProgressBar styles removed - QFluentWidgets ProgressBar has its own styling */
    /* Note: QStatusBar and QMenuBar styles removed - FluentWindow doesn't use these */

    /* Keep only styles for native Qt widgets that may still be used */
    QScrollBar:vertical {
        background-color: #f0f0f0;
        width: 12px;
        border-radius: 6px;
    }

    QScrollBar::handle:vertical {
        background-color: #c0c0c0;
        border-radius: 6px;
        min-height: 20px;
    }

    QScrollBar::handle:vertical:hover {
        background-color: #a0a0a0;
    }

    /* Note: QTableWidget styles removed - QFluentWidgets TableWidget has its own styling */
    """


def apply_dark_theme() -> Optional[str]:
    """
    Get dark theme stylesheet.
    
    Returns:
        Dark theme CSS stylesheet string
    """
    # This would contain a dark theme implementation
    # For now, return None to use default theme
    return None


def apply_light_theme() -> Optional[str]:
    """
    Get light theme stylesheet.
    
    Returns:
        Light theme CSS stylesheet string
    """
    return get_app_style()
