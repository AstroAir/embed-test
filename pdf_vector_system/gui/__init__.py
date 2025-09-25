"""
GUI package for PDF Vector System.

This package provides a comprehensive PySide6-based graphical user interface
for the PDF Vector System, offering all functionality available through the CLI
in an intuitive, user-friendly interface.

Main Components:
- MainWindow: Primary application window with tabbed interface
- Widgets: Reusable GUI components for specific functionality
- Dialogs: Modal dialogs for user interaction
- Controllers: Business logic integration and state management
- Utils: GUI utilities and helper functions
"""

from .main_window import MainWindow
from .app import PDFVectorGUIApp

__all__ = ["MainWindow", "PDFVectorGUIApp"]
