"""
Progress dialog for PDF Vector System GUI.

This module contains progress dialogs for long-running operations.
"""

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    MaskDialogBase,
    ProgressBar,
    PushButton,
    VBoxLayout,
)


class ProgressDialog(MaskDialogBase):
    """Progress dialog for long-running operations using QFluentWidgets MaskDialogBase."""

    # Signals
    cancelled: Signal = Signal()

    def __init__(self, title: str, message: str, parent: Optional[QWidget] = None):
        """
        Initialize the progress dialog.

        Args:
            title: Dialog title
            message: Progress message
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self._setup_ui(message)

    def _setup_ui(self, message: str) -> None:
        """Set up the user interface."""
        layout = VBoxLayout(self)

        # Message
        self.message_label = BodyLabel(message)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.message_label)

        # Progress bar
        self.progress_bar = ProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = BodyLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        # Cancel button
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_btn = PushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

    def set_progress(self, value: int) -> None:
        """
        Set progress value.

        Args:
            value: Progress value (0-100)
        """
        self.progress_bar.setValue(
            max(0, min(100, value))
        )  # Ensure value is within valid range

    def set_status(self, status: str) -> None:
        """
        Set status message.

        Args:
            status: Status message
        """
        self.status_label.setText(status)

    def set_message(self, message: str) -> None:
        """
        Set main message.

        Args:
            message: Main message
        """
        self.message_label.setText(message)

    def _on_cancel(self) -> None:
        """Handle cancel button click."""
        self.cancelled.emit()
        self.reject()
