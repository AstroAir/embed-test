"""
Base widget classes for PDF Vector System GUI.

This module contains base widget classes that provide common functionality
for all GUI widgets.
"""

from typing import Optional
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Signal
from qfluentwidgets import VBoxLayout, BodyLabel

from ...config.settings import Config


class BaseWidget(QWidget):
    """Base widget class with common functionality."""
    
    # Signals
    status_changed: Signal = Signal(str)
    error_occurred: Signal = Signal(str)
    
    def __init__(self, config: Optional[Config] = None, parent: Optional[QWidget] = None):
        """
        Initialize the base widget.
        
        Args:
            config: Configuration object
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.config = config or Config()
        self._setup_ui()
        
    def _setup_ui(self) -> None:
        """Set up the user interface. Override in subclasses."""
        layout = VBoxLayout(self)
        label = BodyLabel("Base Widget - Override _setup_ui() in subclass")
        layout.addWidget(label)
        
    def on_tab_activated(self) -> None:
        """Called when this widget's tab is activated. Override in subclasses."""
        pass
        
    def emit_status(self, message: str) -> None:
        """
        Emit a status message.
        
        Args:
            message: Status message
        """
        self.status_changed.emit(message)
        
    def emit_error(self, message: str) -> None:
        """
        Emit an error message.
        
        Args:
            message: Error message
        """
        self.error_occurred.emit(message)
