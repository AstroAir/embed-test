"""
Confirmation dialog for PDF Vector System GUI.

This module contains confirmation dialogs for user actions.
"""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import BodyLabel, MaskDialogBase, PushButton, VBoxLayout


class ConfirmDialog(MaskDialogBase):
    """
    Confirmation dialog for user actions.

    Uses QFluentWidgets MaskDialogBase and PushButton components
    for consistent fluent design throughout the application.
    """

    def __init__(self, title: str, message: str, parent: Optional[QWidget] = None):
        """
        Initialize the confirmation dialog.

        Args:
            title: Dialog title
            message: Confirmation message
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
        label = BodyLabel(message)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        yes_btn = PushButton("Yes")
        no_btn = PushButton("No")

        yes_btn.clicked.connect(self.accept)
        no_btn.clicked.connect(self.reject)

        button_layout.addWidget(yes_btn)
        button_layout.addWidget(no_btn)

        layout.addLayout(button_layout)

        # Set default button
        yes_btn.setDefault(True)
