"""
Error dialog for PDF Vector System GUI.

This module contains error display dialogs.
"""

from typing import Optional

from PySide6.QtWidgets import QWidget, QHBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from qfluentwidgets import (
    VBoxLayout, PushButton,
    BodyLabel, TextEdit, MaskDialogBase
)


class ErrorDialog(MaskDialogBase):
    """
    Error display dialog using QFluentWidgets components.

    Uses MaskDialogBase for consistent fluent design for error reporting
    with modern styling and theming support.
    """
    
    def __init__(self, title: str, message: str, details: Optional[str] = None, 
                 parent: Optional[QWidget] = None):
        """
        Initialize the error dialog.
        
        Args:
            title: Dialog title
            message: Error message
            details: Optional detailed error information
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self._setup_ui(message, details)
        
    def _setup_ui(self, message: str, details: Optional[str]) -> None:
        """Set up the user interface."""
        layout = VBoxLayout(self)

        # Error message
        label = BodyLabel(message)
        label.setWordWrap(True)
        layout.addWidget(label)

        # Details section (if provided)
        if details:
            details_text = TextEdit()
            details_text.setPlainText(details)
            details_text.setReadOnly(True)
            details_text.setMaximumHeight(200)

            # Use monospace font for details
            font = QFont("Consolas", 9)
            if not font.exactMatch():
                font = QFont("Courier New", 9)
            details_text.setFont(font)

            layout.addWidget(BodyLabel("Details:"))
            layout.addWidget(details_text)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        ok_btn = PushButton("OK")
        ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(ok_btn)
        
        layout.addLayout(button_layout)
        
        # Set default button
        ok_btn.setDefault(True)
