"""
Main GUI application class for PDF Vector System.

This module contains the main application class that initializes and manages
the PySide6 GUI application.
"""

import sys
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from qfluentwidgets import FluentThemeColor, Theme, setTheme, setThemeColor

from pdf_vector_system.config.settings import Config
from pdf_vector_system.gui.main_window import MainWindow
from pdf_vector_system.gui.utils.icons import IconType, get_icon
from pdf_vector_system.gui.utils.styling import get_app_style


class PDFVectorGUIApp:
    """Main GUI application class for PDF Vector System."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the GUI application.

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or Config()
        self.app: Optional[QApplication] = None
        self.main_window: Optional[MainWindow] = None

        # Initialize the application
        self.initialize()

    def initialize(self) -> None:
        """Initialize the Qt application and main window."""
        # Create QApplication if it doesn't exist
        if not QApplication.instance():
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()

        # Set application properties
        self.app.setApplicationName("PDF Vector System")
        self.app.setApplicationVersion("1.0.0")
        self.app.setOrganizationName("PDF Vector System")
        self.app.setOrganizationDomain("pdf-vector-system.com")

        # Set application icon
        app_icon = get_icon(IconType.APP)
        if app_icon:
            self.app.setWindowIcon(app_icon)

        # Apply QFluentWidgets theme
        setTheme(Theme.AUTO)  # Auto theme based on system
        setThemeColor(FluentThemeColor.DEFAULT_BLUE.value)

        # Apply additional application style if needed
        style = get_app_style()
        if style:
            self.app.setStyleSheet(style)

        # Enable high DPI scaling
        self.app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        self.app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

        # Create main window
        self.main_window = MainWindow(self.config)

    def run(self) -> int:
        """
        Run the GUI application.

        Returns:
            Application exit code
        """
        if not self.app or not self.main_window:
            self.initialize()

        # Show main window
        self.main_window.show()

        # Start event loop
        return self.app.exec()

    def quit(self) -> None:
        """Quit the application."""
        if self.app:
            self.app.quit()


def main() -> int:
    """
    Main entry point for the GUI application.

    Returns:
        Application exit code
    """
    try:
        # Create and run application
        app = PDFVectorGUIApp()
        return app.run()

    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
