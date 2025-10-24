"""
About dialog for PDF Vector System GUI.

This module contains the About dialog showing application information.
"""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    MaskDialogBase,
    PushButton,
    SubtitleLabel,
    TextEdit,
    TitleLabel,
    VBoxLayout,
)


class AboutDialog(MaskDialogBase):
    """
    About dialog showing application information.

    Uses QFluentWidgets MaskDialogBase for modern fluent design.
    Replaces traditional QTabWidget with CardWidget sections for better visual hierarchy.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize the about dialog.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("About PDF Vector System")
        self.setFixedSize(500, 400)
        self.setModal(True)

        # Apply consistent card title styling
        self.setStyleSheet(
            """
            SubtitleLabel[objectName="cardTitle"] {
                font-weight: bold;
                font-size: 14px;
                margin-bottom: 10px;
                color: palette(text);
            }
        """
        )

        layout = VBoxLayout(self)

        # Header section
        header_layout = QHBoxLayout()

        # Application icon (placeholder) - using BodyLabel for emoji display
        icon_label = BodyLabel("📄")
        icon_label.setStyleSheet("font-size: 48px;")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setFixedSize(64, 64)
        header_layout.addWidget(icon_label)

        # Application info using QFluentWidgets label hierarchy
        info_layout = VBoxLayout()

        # TitleLabel provides the largest, most prominent text style
        title_label = TitleLabel("PDF Vector System")
        info_layout.addWidget(title_label)

        # SubtitleLabel for secondary information like version
        version_label = SubtitleLabel("Version 1.0.0")
        info_layout.addWidget(version_label)

        # BodyLabel for regular descriptive text
        description_label = BodyLabel(
            "A comprehensive PDF content processing and vector storage system"
        )
        description_label.setWordWrap(True)
        info_layout.addWidget(description_label)

        header_layout.addLayout(info_layout)
        header_layout.addStretch()

        layout.addLayout(header_layout)

        # About section using CardWidget for modern card-based layout
        about_card = CardWidget()
        about_layout = VBoxLayout(about_card)

        # Card title using SubtitleLabel for consistent typography
        about_title = SubtitleLabel("About")
        about_title.setObjectName("cardTitle")  # Use object name for consistent styling
        about_layout.addWidget(about_title)

        # TextEdit from QFluentWidgets provides better theming support
        about_text = TextEdit()
        about_text.setReadOnly(True)
        about_text.setHtml(
            """
        <h3>PDF Vector System</h3>
        <p>A powerful Python application for processing PDF documents and storing them in a vector database for semantic search and retrieval.</p>

        <h4>Key Features:</h4>
        <ul>
            <li>Extract text content from PDF files using PyMuPDF</li>
            <li>Process and chunk text for optimal embedding generation</li>
            <li>Generate embeddings using various models (local and API-based)</li>
            <li>Store embeddings in ChromaDB vector database</li>
            <li>Perform similarity search and retrieval operations</li>
            <li>Comprehensive GUI and CLI interfaces</li>
        </ul>

        <h4>Author:</h4>
        <p>The Augster</p>

        <h4>License:</h4>
        <p>MIT License</p>
        """
        )
        about_layout.addWidget(about_text)
        layout.addWidget(about_card)

        # Dependencies section using CardWidget
        deps_card = CardWidget()
        deps_layout = VBoxLayout(deps_card)

        # Add title for the card
        deps_title = SubtitleLabel("Dependencies")
        deps_title.setObjectName("cardTitle")  # Use object name for consistent styling
        deps_layout.addWidget(deps_title)

        deps_text = TextEdit()
        deps_text.setReadOnly(True)
        deps_text.setHtml(
            """
        <h3>Dependencies</h3>
        <p>This application is built using the following key libraries:</p>

        <h4>Core Dependencies:</h4>
        <ul>
            <li><b>PySide6</b> - Qt-based GUI framework</li>
            <li><b>QFluentWidgets</b> - Modern fluent design components</li>
            <li><b>PyMuPDF</b> - PDF processing and text extraction</li>
            <li><b>ChromaDB</b> - Vector database for embeddings</li>
            <li><b>sentence-transformers</b> - Local embedding models</li>
            <li><b>OpenAI</b> - API-based embedding models</li>
            <li><b>Pydantic</b> - Data validation and settings</li>
            <li><b>Typer</b> - CLI framework</li>
            <li><b>Rich</b> - Rich text and beautiful formatting</li>
            <li><b>Loguru</b> - Advanced logging</li>
        </ul>

        <h4>Development Dependencies:</h4>
        <ul>
            <li><b>pytest</b> - Testing framework</li>
            <li><b>black</b> - Code formatting</li>
            <li><b>mypy</b> - Type checking</li>
            <li><b>ruff</b> - Linting</li>
        </ul>
        """
        )
        deps_layout.addWidget(deps_text)
        layout.addWidget(deps_card)

        # System info section using CardWidget
        system_card = CardWidget()
        system_layout = VBoxLayout(system_card)

        # Add title for the card
        system_title = SubtitleLabel("System Information")
        system_title.setObjectName(
            "cardTitle"
        )  # Use object name for consistent styling
        system_layout.addWidget(system_title)

        system_text = TextEdit()
        system_text.setReadOnly(True)

        # Get system information
        import platform
        import sys

        try:
            import PySide6

            pyside_version = PySide6.__version__
        except Exception:
            pyside_version = "Unknown"

        try:
            import qfluentwidgets

            qfw_version = qfluentwidgets.__version__
        except Exception:
            qfw_version = "Unknown"

        system_info = f"""
        <h3>System Information</h3>

        <h4>Python Environment:</h4>
        <ul>
            <li><b>Python Version:</b> {sys.version}</li>
            <li><b>Platform:</b> {platform.platform()}</li>
            <li><b>Architecture:</b> {platform.architecture()[0]}</li>
            <li><b>PySide6 Version:</b> {pyside_version}</li>
            <li><b>QFluentWidgets Version:</b> {qfw_version}</li>
        </ul>

        <h4>Application:</h4>
        <ul>
            <li><b>Executable:</b> {sys.executable}</li>
            <li><b>Python Path:</b> {sys.path[0]}</li>
        </ul>
        """

        system_text.setHtml(system_info)
        system_layout.addWidget(system_text)
        layout.addWidget(system_card)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_btn = PushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)
