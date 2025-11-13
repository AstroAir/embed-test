"""
Enhanced document management widget for PDF Vector System GUI.

This module contains the enhanced widget for document management functionality
with modern QFluentWidgets components and improved user experience.
"""

from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QSplitter,
    QTableWidgetItem,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    InfoBadge,
    InfoLevel,
    MessageBox,
    PrimaryPushButton,
    PushButton,
    SearchLineEdit,
    SmoothScrollArea,
    StateToolTip,
    SubtitleLabel,
    TableWidget,
    TextEdit,
    VBoxLayout,
)

from vectorflow.core.config.settings import Config
from vectorflow.gui.controllers.document_controller import DocumentController
from vectorflow.gui.utils.styling import (
    apply_card_title_style,
    create_styled_card_widget,
    get_fluent_icon_for_action,
)
from vectorflow.gui.widgets.base import BaseWidget


class DocumentWidget(BaseWidget):
    """Enhanced widget for document management functionality with modern UI components."""

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

        # Enhanced state tracking
        self._documents: list[dict[str, Any]] = []
        self._filtered_documents: list[dict[str, Any]] = []
        self._document_count_badge: InfoBadge  # Initialized in _setup_ui
        self._loading_state_tooltip: Optional[StateToolTip] = None
        self._current_filter: str = ""

        # Initialize controller
        self.controller = DocumentController(self.config, self)
        self._connect_controller_signals()

    def _setup_ui(self) -> None:
        """Set up the enhanced user interface."""
        # Create main scroll area for better UX
        scroll_area = SmoothScrollArea(self)
        scroll_area.setWidgetResizable(True)

        main_layout = VBoxLayout(self)
        main_layout.addWidget(scroll_area)

        # Create scroll content widget
        scroll_widget = QWidget()
        scroll_area.setWidget(scroll_widget)
        layout = VBoxLayout(scroll_widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Controls section with enhanced styling
        self._create_controls_section(layout)

        # Main content splitter
        self._create_content_section(layout)

        # Statistics section with modern card
        self._create_statistics_section(layout)

        # Connect signals
        self._setup_connections()

    def _create_controls_section(self, layout: VBoxLayout) -> None:
        """Create the enhanced controls section."""
        controls_card = create_styled_card_widget("Document Management", self)
        controls_layout = VBoxLayout(controls_card)

        # Search bar with action buttons
        top_controls = QHBoxLayout()

        # Enhanced search with icon
        self.search_input = SearchLineEdit()
        self.search_input.setPlaceholderText("Search documents by ID or content...")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.setMinimumWidth(300)
        self.search_input.textChanged.connect(self._on_search_changed)
        top_controls.addWidget(self.search_input)

        # Document count badge
        self._document_count_badge = InfoBadge.info("0", self)
        self._document_count_badge.hide()
        top_controls.addWidget(self._document_count_badge)

        top_controls.addStretch()

        # Help button with teaching tip
        self.help_btn = PushButton("?")
        self.help_btn.setFixedSize(24, 24)
        self.help_btn.setIcon(get_fluent_icon_for_action("help").icon())
        self.help_btn.clicked.connect(self._show_help)
        self.help_btn.setToolTip("Show help for document management")
        top_controls.addWidget(self.help_btn)

        controls_layout.addLayout(top_controls)

        # Action buttons with enhanced styling
        buttons_layout = QHBoxLayout()

        self.refresh_btn = PrimaryPushButton("Refresh")
        self.refresh_btn.setIcon(get_fluent_icon_for_action("refresh").icon())
        self.refresh_btn.setMinimumWidth(100)
        self.refresh_btn.setToolTip("Refresh document list from database")

        self.delete_btn = PushButton("Delete Selected")
        self.delete_btn.setIcon(get_fluent_icon_for_action("delete").icon())
        self.delete_btn.setEnabled(False)
        self.delete_btn.setToolTip("Delete the selected document")

        self.stats_btn = PushButton("Show Statistics")
        self.stats_btn.setIcon(get_fluent_icon_for_action("info").icon())
        self.stats_btn.setToolTip("Display collection statistics")

        self.export_btn = PushButton("Export List")
        self.export_btn.setIcon(get_fluent_icon_for_action("save").icon())
        self.export_btn.setEnabled(False)
        self.export_btn.setToolTip("Export document list to file")
        self.export_btn.clicked.connect(self._export_document_list)

        buttons_layout.addWidget(self.refresh_btn)
        buttons_layout.addWidget(self.delete_btn)
        buttons_layout.addWidget(self.stats_btn)
        buttons_layout.addWidget(self.export_btn)
        buttons_layout.addStretch()

        controls_layout.addLayout(buttons_layout)
        layout.addWidget(controls_card)

    def _create_content_section(self, layout: VBoxLayout) -> None:
        """Create the main content section with splitter."""
        # Create splitter for documents table and details
        splitter = QSplitter(Qt.Horizontal)

        # Documents table with enhanced card
        docs_card = create_styled_card_widget("Documents", self)
        docs_layout = VBoxLayout(docs_card)

        self.documents_table = TableWidget()
        self.documents_table.setColumnCount(4)
        self.documents_table.setHorizontalHeaderLabels(
            ["Document ID", "Chunks", "Characters", "Avg Chunk Size"]
        )
        self.documents_table.setSelectionBehavior(TableWidget.SelectRows)
        self.documents_table.setAlternatingRowColors(True)

        # Make table columns resize appropriately
        header = self.documents_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)

        docs_layout.addWidget(self.documents_table)
        splitter.addWidget(docs_card)

        # Document details with enhanced card
        details_card = create_styled_card_widget("Document Details", self)
        details_layout = VBoxLayout(details_card)

        self.details_text = TextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setPlaceholderText(
            "Select a document to view detailed information...\n\n"
            "Details will include:\n"
            "• Document ID\n"
            "• Number of chunks\n"
            "• Total characters\n"
            "• Average chunk size\n"
            "• Creation date"
        )
        details_layout.addWidget(self.details_text)

        splitter.addWidget(details_card)

        # Set splitter proportions
        splitter.setSizes([500, 300])

        layout.addWidget(splitter)

    def _create_statistics_section(self, layout: VBoxLayout) -> None:
        """Create the enhanced statistics section."""
        stats_card = create_styled_card_widget("Collection Statistics", self)
        stats_layout = VBoxLayout(stats_card)

        # Statistics header with subtitle
        stats_header = QHBoxLayout()

        self.stats_title = SubtitleLabel("Statistics")
        apply_card_title_style(self.stats_title)
        stats_header.addWidget(self.stats_title)

        # Add refresh indicator badge
        self.stats_badge = InfoBadge.info("Ready", self)
        stats_header.addWidget(self.stats_badge)
        stats_header.addStretch()

        stats_layout.addLayout(stats_header)

        # Statistics display
        self.stats_label = BodyLabel("Click 'Refresh' to load statistics")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)

        layout.addWidget(stats_card)

    def _show_help(self) -> None:
        """Show help for document management."""
        help_content = """
        <b>Document Management:</b><br><br>
        <b>Search:</b> Filter documents by ID or content<br>
        <b>Refresh:</b> Reload document list from database<br>
        <b>Delete:</b> Remove selected document and its chunks<br>
        <b>Statistics:</b> View collection metrics and summaries<br>
        <b>Export:</b> Save document list to a file
        """

        self.show_teaching_tip(
            "Document Management Help", help_content, self.help_btn, "doc_help"
        )

    def _on_search_changed(self, text: str) -> None:
        """Handle search input changes with filtering."""
        self._current_filter = text.lower().strip()

        if not self._current_filter:
            # Show all documents
            self._filtered_documents = self._documents.copy()
        else:
            # Filter documents
            self._filtered_documents = [
                doc
                for doc in self._documents
                if self._current_filter in doc.get("id", "").lower()
            ]

        # Update display
        self._populate_documents_table(self._filtered_documents)

        # Show filter feedback
        if self._current_filter:
            self.show_info_bar(
                "Filtered",
                f"Showing {len(self._filtered_documents)} of {len(self._documents)} documents",
                InfoLevel.INFOAMTION,
                duration=2000,
            )

    def _export_document_list(self) -> None:
        """Export document list to file."""
        if not self._documents:
            self.show_warning_info("No Documents", "No documents to export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Document List",
            "document_list.txt",
            "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)",
        )

        if file_path:
            try:
                with Path(file_path).open("w", encoding="utf-8") as f:
                    # Write header
                    f.write("PDF Vector System - Document List\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(
                        f"Total Documents: {len(self._documents)}\n"
                        f"Filtered Documents: {len(self._filtered_documents)}\n\n"
                    )
                    f.write("-" * 50 + "\n\n")

                    # Write document information
                    for doc in self._filtered_documents:
                        f.write(f"Document ID: {doc.get('id', 'Unknown')}\n")
                        f.write(f"Chunks: {doc.get('chunks', 0)}\n")
                        f.write(f"Characters: {doc.get('characters', 0):,}\n")
                        f.write(f"Avg Chunk Size: {doc.get('avg_chunk_size', 0)}\n")
                        f.write(f"Created: {doc.get('created', 'Unknown')}\n")
                        f.write("-" * 30 + "\n\n")

                self.show_success_info(
                    "Export Complete", f"Document list exported to {file_path}"
                )
            except Exception as e:
                self.show_error_info("Export Failed", f"Failed to export: {e!s}")

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
        """Refresh the documents list with enhanced feedback."""
        # Show loading state
        self._loading_state_tooltip = self.show_state_tooltip(
            "Loading Documents",
            "Fetching document list from database...",
            self.refresh_btn,
        )

        # Update stats badge
        self.stats_badge.setText("Loading...")

        # Clear current data
        self.documents_table.setRowCount(0)
        self.details_text.clear()

        # Disable controls during loading
        self.refresh_btn.setEnabled(False)
        self.delete_btn.setEnabled(False)

        # Load documents via controller
        self.controller.load_documents()

        # Also get collection statistics
        self.controller.get_collection_stats()

        self.emit_status("Loading documents...")

    def delete_selected(self) -> None:
        """Delete selected document with enhanced confirmation."""
        current_row = self.documents_table.currentRow()
        if current_row >= 0 and current_row < len(self._filtered_documents):
            document = self._filtered_documents[current_row]
            doc_id = document["id"]

            # Enhanced confirmation dialog
            w = MessageBox(
                "Confirm Deletion",
                f"Are you sure you want to delete document '{doc_id}'?\n\n"
                f"This will permanently remove:\n"
                f"• {document.get('chunks', 0)} chunks\n"
                f"• {document.get('characters', 0):,} characters\n\n"
                f"This action cannot be undone.",
                self,
            )
            w.yesButton.setText("Delete")
            w.cancelButton.setText("Cancel")

            if w.exec():
                # Show deleting state
                delete_tooltip = self.show_state_tooltip(
                    "Deleting Document",
                    f"Removing document '{doc_id}' and its chunks...",
                    self.delete_btn,
                )

                self.controller.delete_document(doc_id)

    def show_statistics(self) -> None:
        """Show detailed collection statistics with enhanced feedback."""
        # Show loading state
        stats_tooltip = self.show_state_tooltip(
            "Loading Statistics",
            "Calculating collection statistics...",
            self.stats_btn,
        )

        self.stats_badge.setText("Calculating...")
        self.controller.get_collection_stats()

    def on_document_selected(self) -> None:
        """Handle document selection with enhanced details."""
        current_row = self.documents_table.currentRow()
        if current_row >= 0 and current_row < len(self._filtered_documents):
            self.delete_btn.setEnabled(True)

            document = self._filtered_documents[current_row]
            doc_id = document["id"]

            # Show enhanced document details with formatting
            details = f"<h3>Document Information</h3>\n\n"
            details += f"<p><b>Document ID:</b> {doc_id}</p>\n"
            details += f"<p><b>Chunks:</b> {document.get('chunks', 'N/A')}</p>\n"
            details += (
                f"<p><b>Characters:</b> {document.get('characters', 'N/A'):,}</p>\n"
            )
            details += f"<p><b>Average Chunk Size:</b> {document.get('avg_chunk_size', 'N/A')} characters</p>\n"
            details += f"<p><b>Created:</b> {document.get('created', 'N/A')}</p>\n"

            self.details_text.setHtml(details)

            # Show selection feedback
            self.emit_status(f"Selected document: {doc_id}")
        else:
            self.delete_btn.setEnabled(False)
            self.details_text.clear()

    def _on_documents_loaded(self, documents: list[dict[str, Any]]) -> None:
        """Handle documents loaded signal with enhanced feedback."""
        self._documents = documents
        self._filtered_documents = documents.copy()

        # Hide loading state
        if self._loading_state_tooltip:
            self._loading_state_tooltip.close()
            self._loading_state_tooltip = None

        # Re-enable controls
        self.refresh_btn.setEnabled(True)

        # Update document count badge
        doc_count = len(documents)
        if doc_count > 0:
            self._document_count_badge.setText(str(doc_count))
            self._document_count_badge.show()
            self.export_btn.setEnabled(True)

            # Show success feedback
            self.show_success_info(
                "Documents Loaded", f"Loaded {doc_count} document(s)", duration=2000
            )
        else:
            self._document_count_badge.hide()
            self.export_btn.setEnabled(False)

            # Show info feedback
            self.show_info_bar(
                "No Documents", "No documents found in the collection", InfoLevel.INFO
            )

        # Apply current filter if any
        if self._current_filter:
            self._on_search_changed(self._current_filter)
        else:
            self._populate_documents_table(documents)

        self.emit_status(f"Loaded {doc_count} documents")

    def _on_document_deleted(self, document_id: str, chunks_deleted: int) -> None:
        """Handle document deleted signal with enhanced feedback."""
        # Show success feedback
        self.show_success_info(
            "Document Deleted",
            f"Deleted document '{document_id}' ({chunks_deleted} chunks)",
        )

        self.emit_status(f"Deleted document '{document_id}' ({chunks_deleted} chunks)")

        # Refresh the documents list
        self.refresh_documents()

    def _on_statistics_updated(self, stats: dict[str, Any]) -> None:
        """Handle statistics updated signal with enhanced display."""
        # Hide loading state
        self.hide_state_tooltip()

        # Update stats badge
        self.stats_badge.setText("Updated")

        # Format statistics with HTML for better presentation
        stats_html = "<h3>Collection Statistics</h3>\n\n"
        stats_html += (
            f"<p><b>Total Documents:</b> {stats.get('total_documents', 0)}</p>\n"
        )
        stats_html += f"<p><b>Total Chunks:</b> {stats.get('total_chunks', 0):,}</p>\n"
        stats_html += (
            f"<p><b>Total Characters:</b> {stats.get('total_characters', 0):,}</p>\n"
        )
        stats_html += f"<p><b>Average Chunk Size:</b> {stats.get('average_chunk_size', 0):.0f} characters</p>\n"

        # Use both the label and show an info bar for immediate visibility
        self.stats_label.setText(
            f"Total Documents: {stats.get('total_documents', 0)} | "
            f"Total Chunks: {stats.get('total_chunks', 0):,} | "
            f"Avg Size: {stats.get('average_chunk_size', 0):.0f} chars"
        )

        # Show info bar with stats summary
        self.show_success_info(
            "Statistics Updated",
            f"Collection contains {stats.get('total_documents', 0)} documents with {stats.get('total_chunks', 0):,} chunks",
            duration=3000,
        )

    def _on_operation_error(self, error_message: str) -> None:
        """Handle operation error signal with enhanced feedback."""
        # Hide loading state
        if self._loading_state_tooltip:
            self._loading_state_tooltip.close()
            self._loading_state_tooltip = None

        # Re-enable controls
        self.refresh_btn.setEnabled(True)

        # Update stats badge
        self.stats_badge.setText("Error")

        # Show error feedback
        self.show_error_info("Operation Error", error_message)
        self.emit_status(f"Error: {error_message}")

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

        # Resize columns to fit content
        self.documents_table.resizeColumnsToContents()

    def on_tab_activated(self) -> None:
        """Called when this tab is activated."""
        self.emit_status("Documents tab activated")
        # Auto-refresh when tab is activated if no documents loaded
        if self.documents_table.rowCount() == 0:
            self.refresh_documents()
