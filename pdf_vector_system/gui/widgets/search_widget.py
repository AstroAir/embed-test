"""
Search widget for PDF Vector System GUI.

This module contains the widget for search functionality.
"""

from typing import Optional, List

from PySide6.QtWidgets import QHBoxLayout, QFormLayout, QSplitter, QWidget, QTableWidgetItem
from PySide6.QtCore import Qt
from qfluentwidgets import (
    VBoxLayout, PushButton, BodyLabel,
    LineEdit, TableWidget, CardWidget,
    SpinBox, TextEdit
)

from ...config.settings import Config
from ...vector_db.models import SearchResult
from .base import BaseWidget
from ..controllers.search_controller import SearchController


class SearchWidget(BaseWidget):
    """Widget for search functionality."""
    
    def __init__(self, config: Optional[Config] = None, parent: Optional[QWidget] = None):
        """
        Initialize the search widget.

        Args:
            config: Configuration object
            parent: Parent widget
        """
        super().__init__(config, parent)

        # Initialize controller
        self.controller = SearchController(self.config, self)
        self._connect_controller_signals()
        
    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = VBoxLayout(self)

        # Search input section
        search_group = CardWidget()
        search_layout = VBoxLayout(search_group)

        # Add title for the card
        search_title = BodyLabel("Search Query")
        search_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        search_layout.addWidget(search_title)

        # Query input
        query_layout = QHBoxLayout()
        self.query_input = LineEdit()
        self.query_input.setPlaceholderText("Enter your search query...")
        self.search_btn = PushButton("Search")
        self.clear_btn = PushButton("Clear")

        query_layout.addWidget(BodyLabel("Query:"))
        query_layout.addWidget(self.query_input)
        query_layout.addWidget(self.search_btn)
        query_layout.addWidget(self.clear_btn)

        search_layout.addLayout(query_layout)

        # Search options
        options_layout = QFormLayout()

        self.max_results_spin = SpinBox()
        self.max_results_spin.setRange(1, 100)
        self.max_results_spin.setValue(10)
        options_layout.addRow("Max Results:", self.max_results_spin)

        self.document_filter = LineEdit()
        self.document_filter.setPlaceholderText("Filter by document ID (optional)")
        options_layout.addRow("Document Filter:", self.document_filter)
        
        search_layout.addLayout(options_layout)
        layout.addWidget(search_group)
        
        # Results section
        results_splitter = QSplitter(Qt.Horizontal)
        
        # Results table
        results_group = CardWidget()
        results_layout = VBoxLayout(results_group)

        # Add title for the card
        results_title = BodyLabel("Search Results")
        results_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        results_layout.addWidget(results_title)

        self.results_table = TableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels([
            "Score", "Document", "Page", "Preview"
        ])
        self.results_table.setSelectionBehavior(TableWidget.SelectRows)
        results_layout.addWidget(self.results_table)

        results_splitter.addWidget(results_group)

        # Content preview
        preview_group = CardWidget()
        preview_layout = VBoxLayout(preview_group)

        # Add title for the card
        preview_title = BodyLabel("Content Preview")
        preview_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        preview_layout.addWidget(preview_title)

        self.content_preview = TextEdit()
        self.content_preview.setReadOnly(True)
        self.content_preview.setPlaceholderText("Select a search result to view content...")
        preview_layout.addWidget(self.content_preview)

        results_splitter.addWidget(preview_group)
        
        # Set splitter proportions
        results_splitter.setSizes([400, 300])
        
        layout.addWidget(results_splitter)
        
        # Connect signals
        self._setup_connections()

    def _connect_controller_signals(self) -> None:
        """Connect controller signals to widget slots."""
        self.controller.search_started.connect(self._on_search_started)
        self.controller.search_completed.connect(self._on_search_completed)
        self.controller.search_error.connect(self._on_search_error)
        self.controller.status_message.connect(self._on_status_message)

    def _setup_connections(self) -> None:
        """Set up signal/slot connections."""
        self.search_btn.clicked.connect(self.perform_search)
        self.clear_btn.clicked.connect(self.clear_search)
        self.query_input.returnPressed.connect(self.perform_search)
        self.results_table.itemSelectionChanged.connect(self.on_result_selected)
        
    def perform_search(self) -> None:
        """Perform search with current query."""
        query = self.query_input.text().strip()
        if not query:
            self.emit_status("Please enter a search query")
            return

        # Clear previous results
        self.results_table.setRowCount(0)
        self.content_preview.clear()

        # Get search options
        max_results = self.max_results_spin.value()

        # Start search
        self.controller.search(query, max_results)

        # Update UI state
        self.search_btn.setEnabled(False)
        self.search_btn.setText("Searching...")
        
    def clear_search(self) -> None:
        """Clear search query and results."""
        self.query_input.clear()
        self.results_table.setRowCount(0)
        self.content_preview.clear()
        self.emit_status("Search cleared")
        
    def on_result_selected(self) -> None:
        """Handle result selection."""
        current_row = self.results_table.currentRow()
        if current_row >= 0 and hasattr(self, '_search_results'):
            result = self._search_results[current_row]
            self.content_preview.setText(result.content)

    def _reset_search_ui(self) -> None:
        """Reset UI to non-searching state."""
        self.search_btn.setEnabled(True)
        self.search_btn.setText("Search")

    def _on_search_started(self, query: str) -> None:
        """Handle search started signal."""
        self.emit_status(f"Searching for: {query}")

    def _on_search_completed(self, results: List[SearchResult]) -> None:
        """Handle search completed signal."""
        self._search_results = results
        self._populate_results_table(results)
        self._reset_search_ui()
        self.emit_status(f"Search completed: {len(results)} results found")

    def _on_search_error(self, error_message: str) -> None:
        """Handle search error signal."""
        self.emit_status(f"Search error: {error_message}")
        self._reset_search_ui()

    def _on_status_message(self, message: str) -> None:
        """Handle status message signal."""
        self.emit_status(message)

    def _populate_results_table(self, results: List[SearchResult]) -> None:
        """Populate the results table with search results."""
        self.results_table.setRowCount(len(results))

        for row, result in enumerate(results):
            # Score
            score_item = QTableWidgetItem(f"{result.score:.3f}")
            self.results_table.setItem(row, 0, score_item)

            # Document
            doc_item = QTableWidgetItem(result.document_id or "Unknown")
            self.results_table.setItem(row, 1, doc_item)

            # Page
            page_item = QTableWidgetItem(str(result.page_number or "N/A"))
            self.results_table.setItem(row, 2, page_item)

            # Preview (first 100 characters)
            preview = result.content[:100] + "..." if len(result.content) > 100 else result.content
            preview_item = QTableWidgetItem(preview)
            self.results_table.setItem(row, 3, preview_item)

    def on_tab_activated(self) -> None:
        """Called when this tab is activated."""
        self.emit_status("Search tab activated")
        self.query_input.setFocus()
