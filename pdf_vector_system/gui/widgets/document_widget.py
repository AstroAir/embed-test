"""
Document management widget for PDF Vector System GUI.

This module contains the widget for document management functionality.
"""

from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QSplitter,
    QTableWidgetItem,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    FluentIcon,
    MessageBox,
    PushButton,
    TableWidget,
    TextEdit,
    VBoxLayout,
)

from pdf_vector_system.config.settings import Config
from pdf_vector_system.gui.controllers.document_controller import DocumentController
from pdf_vector_system.gui.widgets.base import BaseWidget


class DocumentWidget(BaseWidget):
    """Widget for document management functionality."""

    def __init__(
        self, config: Optional[Config] = None, parent: Optional[QWidget] = None
    ):
        """
        Initialize the document widget.

        Args:
            config: Configuration object
            parent: Parent widget
        """
        super().__init__(config, parent)

        # Initialize controller
        self.controller = DocumentController(self.config, self)
        self._connect_controller_signals()

        # Document data
        self._documents: list[dict[str, Any]] = []

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = VBoxLayout(self)

        # Controls section
        controls_layout = QHBoxLayout()
        self.refresh_btn = PushButton("Refresh")
        self.refresh_btn.setIcon(FluentIcon.SYNC.icon())
        self.delete_btn = PushButton("Delete Selected")
        self.delete_btn.setIcon(FluentIcon.DELETE.icon())
        self.delete_btn.setEnabled(False)
        self.stats_btn = PushButton("Show Statistics")
        self.stats_btn.setIcon(FluentIcon.DOCUMENT.icon())

        controls_layout.addWidget(self.refresh_btn)
        controls_layout.addWidget(self.delete_btn)
        controls_layout.addWidget(self.stats_btn)
        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        # Main content splitter
        splitter = QSplitter(Qt.Horizontal)

        # Documents table
        docs_group = CardWidget()
        docs_layout = VBoxLayout(docs_group)

        # Add title for the card
        docs_title = BodyLabel("Documents")
        docs_title.setStyleSheet(
            "font-weight: bold; font-size: 14px; margin-bottom: 10px;"
        )
        docs_layout.addWidget(docs_title)

        self.documents_table = TableWidget()
        self.documents_table.setColumnCount(4)
        self.documents_table.setHorizontalHeaderLabels(
            ["Document ID", "Chunks", "Characters", "Avg Chunk Size"]
        )
        self.documents_table.setSelectionBehavior(TableWidget.SelectRows)

        # Make table columns resize to content
        header = self.documents_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)

        docs_layout.addWidget(self.documents_table)
        docs_group.setLayout(docs_layout)
        splitter.addWidget(docs_group)

        # Document details
        details_group = CardWidget()
        details_layout = VBoxLayout(details_group)

        # Add title for the card
        details_title = BodyLabel("Document Details")
        details_title.setStyleSheet(
            "font-weight: bold; font-size: 14px; margin-bottom: 10px;"
        )
        details_layout.addWidget(details_title)

        self.details_text = TextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setPlaceholderText("Select a document to view details...")
        details_layout.addWidget(self.details_text)

        splitter.addWidget(details_group)

        # Set splitter proportions
        splitter.setSizes([500, 300])

        layout.addWidget(splitter)

        # Collection statistics
        stats_group = CardWidget()
        stats_layout = VBoxLayout(stats_group)

        # Add title for the card
        stats_title = BodyLabel("Collection Statistics")
        stats_title.setStyleSheet(
            "font-weight: bold; font-size: 14px; margin-bottom: 10px;"
        )
        stats_layout.addWidget(stats_title)

        self.stats_label = BodyLabel("Click 'Refresh' to load statistics")
        stats_layout.addWidget(self.stats_label)

        layout.addWidget(stats_group)

        # Connect signals
        self._setup_connections()

    def _connect_controller_signals(self) -> None:
        """Connect controller signals to widget slots."""
        self.controller.documents_loaded.connect(self._on_documents_loaded)
        self.controller.document_deleted.connect(self._on_document_deleted)
        self.controller.statistics_updated.connect(self._on_statistics_updated)
        self.controller.operation_error.connect(self._on_operation_error)
        self.controller.status_message.connect(self._on_status_message)

    def _setup_connections(self) -> None:
        """Set up signal/slot connections."""
        self.refresh_btn.clicked.connect(self.refresh_documents)
        self.delete_btn.clicked.connect(self.delete_selected)
        self.stats_btn.clicked.connect(self.show_statistics)
        self.documents_table.itemSelectionChanged.connect(self.on_document_selected)

    def refresh_documents(self) -> None:
        """Refresh the documents list."""
        # Clear current data
        self.documents_table.setRowCount(0)
        self.details_text.clear()

        # Load documents via controller
        self.controller.load_documents()

        # Also get collection statistics
        self.controller.get_collection_stats()

    def delete_selected(self) -> None:
        """Delete selected document."""
        current_row = self.documents_table.currentRow()
        if current_row >= 0 and current_row < len(self._documents):
            document = self._documents[current_row]
            doc_id = document["id"]

            # Confirm deletion
            w = MessageBox(
                "Confirm Deletion",
                f"Are you sure you want to delete document '{doc_id}'?\n\n"
                f"This will remove {document.get('chunks', 0)} chunks from the database.",
                self,
            )
            w.yesButton.setText("Delete")
            w.cancelButton.setText("Cancel")

            if w.exec():
                self.controller.delete_document(doc_id)

    def show_statistics(self) -> None:
        """Show detailed collection statistics."""
        self.controller.get_collection_stats()

    def on_document_selected(self) -> None:
        """Handle document selection."""
        current_row = self.documents_table.currentRow()
        if current_row >= 0 and current_row < len(self._documents):
            self.delete_btn.setEnabled(True)

            document = self._documents[current_row]
            doc_id = document["id"]

            # Show document details
            details = f"Document ID: {doc_id}\n\n"
            details += f"Chunks: {document.get('chunks', 'N/A')}\n"
            details += f"Characters: {document.get('characters', 'N/A'):,}\n"
            details += f"Average Chunk Size: {document.get('avg_chunk_size', 'N/A')}\n"
            details += f"Created: {document.get('created', 'N/A')}\n"

            self.details_text.setText(details)
        else:
            self.delete_btn.setEnabled(False)
            self.details_text.clear()

    def _on_documents_loaded(self, documents: list[dict[str, Any]]) -> None:
        """Handle documents loaded signal."""
        self._documents = documents
        self._populate_documents_table(documents)
        self.emit_status(f"Loaded {len(documents)} documents")

    def _on_document_deleted(self, document_id: str, chunks_deleted: int) -> None:
        """Handle document deleted signal."""
        self.emit_status(f"Deleted document '{document_id}' ({chunks_deleted} chunks)")
        # Refresh the documents list
        self.refresh_documents()

    def _on_statistics_updated(self, stats: dict[str, Any]) -> None:
        """Handle statistics updated signal."""
        stats_text = f"Total Documents: {stats.get('total_documents', 0)}\n"
        stats_text += f"Total Chunks: {stats.get('total_chunks', 0)}\n"
        stats_text += f"Total Characters: {stats.get('total_characters', 0):,}\n"
        stats_text += (
            f"Average Chunk Size: {stats.get('average_chunk_size', 0):.0f} characters"
        )

        self.stats_label.setText(stats_text)

    def _on_operation_error(self, error_message: str) -> None:
        """Handle operation error signal."""
        self.emit_status(f"Error: {error_message}")
        w = MessageBox("Operation Error", error_message, self)
        w.exec()

    def _on_status_message(self, message: str) -> None:
        """Handle status message signal."""
        self.emit_status(message)

    def _populate_documents_table(self, documents: list[dict[str, Any]]) -> None:
        """Populate the documents table with document data."""
        self.documents_table.setRowCount(len(documents))

        for row, doc in enumerate(documents):
            # Document ID
            id_item = QTableWidgetItem(doc.get("id", "Unknown"))
            self.documents_table.setItem(row, 0, id_item)

            # Chunks
            chunks_item = QTableWidgetItem(str(doc.get("chunks", 0)))
            self.documents_table.setItem(row, 1, chunks_item)

            # Characters
            chars_item = QTableWidgetItem(f"{doc.get('characters', 0):,}")
            self.documents_table.setItem(row, 2, chars_item)

            # Average chunk size
            avg_size_item = QTableWidgetItem(str(doc.get("avg_chunk_size", 0)))
            self.documents_table.setItem(row, 3, avg_size_item)

    def on_tab_activated(self) -> None:
        """Called when this tab is activated."""
        self.emit_status("Documents tab activated")
        # Auto-refresh when tab is activated
        if self.documents_table.rowCount() == 0:
            self.refresh_documents()
