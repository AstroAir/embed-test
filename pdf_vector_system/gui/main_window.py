"""
Main window for PDF Vector System GUI.

This module contains the main application window with modern navigation interface
using QFluentWidgets FluentWindow.
"""

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import QWidget
from qfluentwidgets import FluentIcon, FluentWindow, MessageBox

from pdf_vector_system.config.settings import Config
from pdf_vector_system.gui.controllers.main_controller import MainController
from pdf_vector_system.gui.utils.icons import IconType
from pdf_vector_system.gui.utils.qt_utils import center_window, set_window_icon
from pdf_vector_system.gui.widgets import (
    ConfigWidget,
    DocumentWidget,
    LogWidget,
    ProcessingWidget,
    SearchWidget,
    StatusWidget,
)


class MainWindow(FluentWindow):
    """Main application window with tabbed interface."""

    # Signals
    closing = Signal()

    def __init__(
        self, config: Optional[Config] = None, parent: Optional[QWidget] = None
    ):
        """
        Initialize the main window.

        Args:
            config: Configuration object
            parent: Parent widget
        """
        super().__init__(parent)

        self.config = config or Config()
        self.controller = MainController(self.config)

        # Initialize UI
        self._setup_ui()
        self._setup_shortcuts()
        self._setup_connections()

        # Center window on screen
        center_window(self)

    def _setup_ui(self) -> None:
        """Set up the main user interface."""
        # Set window properties
        self.setWindowTitle("PDF Vector System")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # Set window icon
        set_window_icon(self, IconType.APP)

        # Create and add tabs using FluentWindow's interface system
        self._create_tabs()

    def _create_tabs(self) -> None:
        """Create and add all tabs using FluentWindow's interface system."""
        # Processing tab
        self.processing_widget = ProcessingWidget(self.config)
        self.processing_widget.setObjectName("ProcessingWidget")
        self.addSubInterface(
            self.processing_widget, FluentIcon.DOCUMENT, "Process PDFs"
        )

        # Search tab
        self.search_widget = SearchWidget(self.config)
        self.search_widget.setObjectName("SearchWidget")
        self.addSubInterface(self.search_widget, FluentIcon.SEARCH, "Search")

        # Document management tab
        self.document_widget = DocumentWidget(self.config)
        self.document_widget.setObjectName("DocumentWidget")
        self.addSubInterface(self.document_widget, FluentIcon.FOLDER, "Documents")

        # Configuration tab
        self.config_widget = ConfigWidget(self.config)
        self.config_widget.setObjectName("ConfigWidget")
        self.addSubInterface(self.config_widget, FluentIcon.SETTING, "Settings")

        # System status tab
        self.status_widget = StatusWidget(self.config)
        self.status_widget.setObjectName("StatusWidget")
        self.addSubInterface(self.status_widget, FluentIcon.HEART, "Status")

        # Log viewer tab
        self.log_widget = LogWidget(self.config)
        self.log_widget.setObjectName("LogWidget")
        self.addSubInterface(self.log_widget, FluentIcon.HISTORY, "Logs")

        # Connect widget signals to main controller
        self._connect_widget_signals()

    def _setup_shortcuts(self) -> None:
        """Set up keyboard shortcuts."""
        # Ctrl+Q to quit application
        quit_shortcut = QShortcut(QKeySequence.StandardKey.Quit, self)
        quit_shortcut.activated.connect(self.close)

        # Ctrl+O to switch to processing tab
        process_shortcut = QShortcut(QKeySequence.StandardKey.Open, self)
        process_shortcut.activated.connect(
            lambda: self.switchTo(self.processing_widget)
        )

        # Ctrl+, to switch to settings tab (on systems that support it)
        settings_shortcut = QShortcut(QKeySequence.StandardKey.Preferences, self)
        settings_shortcut.activated.connect(lambda: self.switchTo(self.config_widget))

    def _connect_widget_signals(self) -> None:
        """Connect widget signals to main controller and each other."""
        # Connect config widget to update all other widgets when config changes
        if hasattr(self.config_widget, "config_changed"):
            self.config_widget.config_changed.connect(self._on_config_changed)

        # Connect processing widget to update document widget when files are processed
        if hasattr(self.processing_widget, "controller"):
            self.processing_widget.controller.processing_completed.connect(
                lambda successful, total: (
                    self.document_widget.refresh_documents() if successful > 0 else None
                )
            )

    def _setup_connections(self) -> None:
        """Set up signal/slot connections."""
        # Connect controller signals
        self.controller.error_occurred.connect(self._on_error)
        self.controller.status_message.connect(self._on_status_message)

        # Connect navigation change signal (FluentWindow uses stackedWidget)
        if hasattr(self.stackedWidget, "currentChanged"):
            self.stackedWidget.currentChanged.connect(self._on_tab_changed)

    def _on_config_changed(self, new_config: Config) -> None:
        """Handle configuration changes."""
        self.config = new_config

        # Update all widget controllers with new config
        for widget in [
            self.processing_widget,
            self.search_widget,
            self.document_widget,
            self.status_widget,
        ]:
            if hasattr(widget, "controller") and hasattr(
                widget.controller, "update_config"
            ):
                widget.controller.update_config(new_config)

        # Update main controller
        self.controller.update_config(new_config)

    def _on_error(self, error_message: str) -> None:
        """Handle error messages."""
        w = MessageBox("Error", error_message, self)
        w.cancelButton.setText("OK")
        w.yesButton.hide()
        w.exec()

    def _on_status_message(self, message: str) -> None:
        """
        Handle status messages from the main controller.

        Since FluentWindow doesn't have a built-in status bar like QMainWindow,
        we forward status messages to the status widget for display.

        Args:
            message: Status message to display
        """
        # Forward status message to the status widget if it exists
        if hasattr(self, "status_widget") and hasattr(
            self.status_widget, "emit_status"
        ):
            self.status_widget.emit_status(message)

    def _on_tab_changed(self, index: int) -> None:
        """Handle tab change event."""
        widget = self.stackedWidget.widget(index)
        if widget and hasattr(widget, "on_tab_activated"):
            widget.on_tab_activated()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Handle window close event."""
        self.closing.emit()
        event.accept()
