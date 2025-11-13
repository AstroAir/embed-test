"""
Enhanced search widget for PDF Vector System GUI.

This module contains the enhanced widget for search functionality with modern
QFluentWidgets components and improved user experience.
"""

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QSplitter,
    QTableWidgetItem,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CheckBox,
    ComboBox,
    InfoBadge,
    InfoLevel,
    LineEdit,
    PrimaryPushButton,
    PushButton,
    SearchLineEdit,
    SegmentedWidget,
    SmoothScrollArea,
    SpinBox,
    StateToolTip,
    SubtitleLabel,
    TableWidget,
    TextEdit,
    VBoxLayout,
)

from vectorflow.core.config.settings import Config
from vectorflow.core.vector_db.models import SearchResult
from vectorflow.gui.controllers.search_controller import SearchController
from vectorflow.gui.utils.styling import (
    apply_card_title_style,
    create_styled_card_widget,
    get_fluent_icon_for_action,
)
from vectorflow.gui.widgets.base import BaseWidget


class SearchWidget(BaseWidget):
    """Enhanced widget for search functionality with modern UI components."""

    def __init__(
        self, config: Optional[Config] = None, parent: Optional[QWidget] = None
    ):
        """
        Initialize the search widget.

        Args:
            config: Configuration object
            parent: Parent widget
        """
        super().__init__(config, parent)

        # Search state tracking
        self._current_search_type = "semantic"
        self._search_results: list[SearchResult] = []
        self._result_count_badge: InfoBadge  # Initialized in _setup_ui
        self._search_state_tooltip: Optional[StateToolTip] = None

        # Search debounce timer
        self._search_timer = QTimer()
        self._search_timer.setSingleShot(True)
        self._search_timer.timeout.connect(self._perform_search)

        # Initialize controller
        self.controller = SearchController(self.config, self)
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

        # Search type selection
        self._create_search_type_section(layout)

        # Search input section
        self._create_search_input_section(layout)

        # Search options section
        self._create_search_options_section(layout)

        # Search results section
        self._create_search_results_section(layout)

        # Connect signals
        self._setup_connections()

    def _create_search_type_section(self, layout: VBoxLayout) -> None:
        """Create the search type selection section."""
        type_card = create_styled_card_widget("Search Type", self)
        type_layout = VBoxLayout(type_card)

        # Add description
        desc_label = BodyLabel("Choose the type of search to perform:")
        type_layout.addWidget(desc_label)

        # Create segmented widget for search types
        self.search_type_selector = SegmentedWidget(self)
        self.search_type_selector.addItem(
            "semantic", "Semantic", None, get_fluent_icon_for_action("search")
        )
        self.search_type_selector.addItem(
            "keyword", "Keyword", None, get_fluent_icon_for_action("tag")
        )
        self.search_type_selector.addItem(
            "hybrid", "Hybrid", None, get_fluent_icon_for_action("code")
        )

        # Set current type
        self.search_type_selector.setCurrentItem("semantic")
        self.search_type_selector.currentItemChanged.connect(
            self._on_search_type_changed
        )

        type_layout.addWidget(self.search_type_selector)

        # Add help button with teaching tip
        help_layout = QHBoxLayout()
        help_layout.addStretch()

        self.search_type_help_btn = PushButton("?")
        self.search_type_help_btn.setFixedSize(24, 24)
        self.search_type_help_btn.setIcon(get_fluent_icon_for_action("help").icon())
        self.search_type_help_btn.clicked.connect(self._show_search_type_help)

        help_layout.addWidget(self.search_type_help_btn)
        type_layout.addLayout(help_layout)

        layout.addWidget(type_card)

    def _create_search_input_section(self, layout: VBoxLayout) -> None:
        """Create the enhanced search input section."""
        search_card = create_styled_card_widget("Search Query", self)
        search_layout = VBoxLayout(search_card)

        # Enhanced query input with real-time search
        query_layout = QHBoxLayout()

        self.query_input = SearchLineEdit()
        self.query_input.setPlaceholderText("Enter your search query...")
        self.query_input.setClearButtonEnabled(True)
        self.query_input.textChanged.connect(self._on_query_changed)
        self.query_input.searchSignal.connect(self._perform_search)

        self.search_btn = PrimaryPushButton("Search")
        self.search_btn.setIcon(get_fluent_icon_for_action("search").icon())
        self.search_btn.setMinimumWidth(100)

        self.clear_btn = PushButton("Clear")
        self.clear_btn.setIcon(get_fluent_icon_for_action("delete").icon())

        query_layout.addWidget(self.query_input, 1)
        query_layout.addWidget(self.search_btn)
        query_layout.addWidget(self.clear_btn)

        search_layout.addLayout(query_layout)

        # Real-time search toggle
        realtime_layout = QHBoxLayout()
        self.realtime_search_cb = CheckBox("Real-time search")
        self.realtime_search_cb.setToolTip("Search as you type (with 500ms delay)")
        self.realtime_search_cb.setChecked(True)

        realtime_layout.addWidget(self.realtime_search_cb)
        realtime_layout.addStretch()
        search_layout.addLayout(realtime_layout)

        layout.addWidget(search_card)

    def _create_search_options_section(self, layout: VBoxLayout) -> None:
        """Create the enhanced search options section."""
        options_card = create_styled_card_widget("Search Options", self)
        options_layout = QFormLayout(options_card)

        # Add help button for search options
        options_title_layout = QHBoxLayout()
        options_title_layout.addStretch()

        self.options_help_btn = PushButton("?")
        self.options_help_btn.setFixedSize(24, 24)
        self.options_help_btn.setIcon(get_fluent_icon_for_action("help").icon())
        self.options_help_btn.clicked.connect(self._show_search_options_help)

        options_title_layout.addWidget(self.options_help_btn)
        options_layout.addRow(options_title_layout)

        # Enhanced search options
        self.max_results_spin = SpinBox()
        self.max_results_spin.setRange(1, 100)
        self.max_results_spin.setValue(10)
        self.max_results_spin.setToolTip("Maximum number of search results to return")
        options_layout.addRow("Max Results:", self.max_results_spin)

        self.similarity_threshold_spin = SpinBox()
        self.similarity_threshold_spin.setRange(0, 100)
        self.similarity_threshold_spin.setValue(70)
        self.similarity_threshold_spin.setSuffix("%")
        self.similarity_threshold_spin.setToolTip(
            "Minimum similarity threshold for results"
        )
        options_layout.addRow("Similarity Threshold:", self.similarity_threshold_spin)

        self.document_filter = LineEdit()
        self.document_filter.setPlaceholderText("Filter by document ID (optional)")
        self.document_filter.setToolTip("Filter results to specific document")
        options_layout.addRow("Document Filter:", self.document_filter)

        # Search scope selection
        self.search_scope = ComboBox()
        self.search_scope.addItems(
            ["All Documents", "Recent Documents", "Specific Collection"]
        )
        self.search_scope.setToolTip("Scope of the search")
        options_layout.addRow("Search Scope:", self.search_scope)

        layout.addWidget(options_card)

    def _create_search_results_section(self, layout: VBoxLayout) -> None:
        """Create the enhanced search results section."""
        results_card = create_styled_card_widget("Search Results", self)
        results_layout = VBoxLayout(results_card)

        # Results header with count badge
        results_header = QHBoxLayout()

        results_title = SubtitleLabel("Results")
        results_header.addWidget(results_title)

        # Add result count badge
        self._result_count_badge = InfoBadge.info("0", self)
        self._result_count_badge.hide()
        results_header.addWidget(self._result_count_badge)
        results_header.addStretch()

        # Export results button
        self.export_results_btn = PushButton("Export Results")
        self.export_results_btn.setIcon(get_fluent_icon_for_action("save").icon())
        self.export_results_btn.setEnabled(False)
        self.export_results_btn.clicked.connect(self._export_results)
        results_header.addWidget(self.export_results_btn)

        results_layout.addLayout(results_header)

        # Create splitter for results table and preview
        splitter = QSplitter(Qt.Horizontal)

        # Results table with smooth scrolling
        table_container = QWidget()
        table_layout = VBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)

        self.results_table = TableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(
            ["Score", "Document", "Content", "Metadata"]
        )
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(TableWidget.SelectRows)
        self.results_table.itemSelectionChanged.connect(self._on_result_selected)

        table_layout.addWidget(self.results_table)
        splitter.addWidget(table_container)

        # Result preview with smooth scrolling
        preview_container = QWidget()
        preview_layout = VBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        preview_title = BodyLabel("Result Preview")
        apply_card_title_style(preview_title)
        preview_layout.addWidget(preview_title)

        self.result_preview = TextEdit()
        self.result_preview.setReadOnly(True)
        self.result_preview.setPlaceholderText(
            "Select a search result to preview its content..."
        )
        preview_layout.addWidget(self.result_preview)

        splitter.addWidget(preview_container)
        splitter.setSizes([400, 300])  # Set initial sizes

        results_layout.addWidget(splitter)
        layout.addWidget(results_card)

    def _setup_connections(self) -> None:
        """Set up signal/slot connections."""
        self.search_btn.clicked.connect(self._perform_search)
        self.clear_btn.clicked.connect(self._clear_search)

    def _on_search_type_changed(self, item_key: str) -> None:
        """Handle search type change."""
        self._current_search_type = item_key

        # Show info about the selected search type
        type_descriptions = {
            "semantic": "Find results based on meaning and context",
            "keyword": "Find exact keyword matches in documents",
            "hybrid": "Combine semantic and keyword search for best results",
        }

        description = type_descriptions.get(item_key, "")
        if description:
            self.show_info_bar(
                f"Search Type: {item_key.title()}",
                description,
                InfoLevel.INFO,
                duration=2000,
            )

    def _on_query_changed(self, text: str) -> None:
        """Handle query text change for real-time search."""
        if self.realtime_search_cb.isChecked() and text.strip():
            # Debounce the search
            self._search_timer.stop()
            self._search_timer.start(500)  # 500ms delay

    def _show_search_type_help(self) -> None:
        """Show help for search types."""
        help_content = """
        <b>Semantic:</b> Uses AI to understand meaning and context<br>
        <b>Keyword:</b> Searches for exact word matches<br>
        <b>Hybrid:</b> Combines both approaches for comprehensive results
        """

        self.show_teaching_tip(
            "Search Types", help_content, self.search_type_help_btn, "search_type_help"
        )

    def _show_search_options_help(self) -> None:
        """Show help for search options."""
        help_content = """
        <b>Max Results:</b> Maximum number of results to return<br>
        <b>Similarity Threshold:</b> Minimum relevance score for results<br>
        <b>Document Filter:</b> Limit search to specific document<br>
        <b>Search Scope:</b> Define the scope of your search
        """

        self.show_teaching_tip(
            "Search Options", help_content, self.options_help_btn, "search_options_help"
        )

    def _perform_search(self) -> None:
        """Perform search with enhanced feedback."""
        query = self.query_input.text().strip()
        if not query:
            self.show_warning_info("Empty Query", "Please enter a search query")
            return

        # Show state tooltip for searching
        self._search_state_tooltip = self.show_state_tooltip(
            "Searching",
            f"Searching for '{query}' using {self._current_search_type} search...",
            self.search_btn,
        )

        # Disable search button during search
        self.search_btn.setEnabled(False)
        self.search_btn.setText("Searching...")

        # Get search parameters
        max_results = self.max_results_spin.value()
        # similarity_threshold = self.similarity_threshold_spin.value() / 100.0  # Not supported by controller
        document_id = self.document_filter.text().strip() or None

        # Perform search through controller
        self.controller.search(
            query=query,
            max_results=max_results,
            document_id=document_id,
        )

    def _clear_search(self) -> None:
        """Clear search with enhanced feedback."""
        self.query_input.clear()
        self.document_filter.clear()
        self._clear_results()
        self.show_info_bar(
            "Search Cleared", "Search query and results cleared", InfoLevel.INFO
        )

    def _clear_results(self) -> None:
        """Clear search results."""
        self._search_results = []
        self.results_table.setRowCount(0)
        self.result_preview.clear()
        self._result_count_badge.hide()
        self.export_results_btn.setEnabled(False)

    def _on_result_selected(self) -> None:
        """Handle result selection for preview."""
        current_row = self.results_table.currentRow()
        if 0 <= current_row < len(self._search_results):
            result = self._search_results[current_row]

            # Format preview content
            preview_content = f"""
            <h3>Document: {result.document_id}</h3>
            <p><b>Similarity Score:</b> {result.similarity_score:.3f}</p>
            <p><b>Content:</b></p>
            <div style="background-color: #f5f5f5; padding: 10px; border-radius: 5px;">
            {result.content}
            </div>
            """

            if result.metadata:
                preview_content += f"""
                <p><b>Metadata:</b></p>
                <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
                {result.metadata!s}
                </div>
                """

            self.result_preview.setHtml(preview_content)

    def _export_results(self) -> None:
        """Export search results to file."""
        if not self._search_results:
            self.show_warning_info("No Results", "No search results to export")
            return

        from PySide6.QtWidgets import QFileDialog

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Search Results",
            f"search_results_{self.query_input.text()}.txt",
            "Text Files (*.txt);;CSV Files (*.csv);;All Files (*)",
        )

        if file_path:
            try:
                with Path(file_path).open("w", encoding="utf-8") as f:
                    f.write(f"Search Results for: {self.query_input.text()}\n")
                    f.write(f"Search Type: {self._current_search_type}\n")
                    f.write(f"Total Results: {len(self._search_results)}\n")
                    f.write("=" * 50 + "\n\n")

                    for i, result in enumerate(self._search_results, 1):
                        f.write(f"Result {i}:\n")
                        f.write(f"Document: {result.document_id}\n")
                        f.write(f"Score: {result.similarity_score:.3f}\n")
                        f.write(f"Content: {result.content}\n")
                        if result.metadata:
                            f.write(f"Metadata: {result.metadata}\n")
                        f.write("-" * 30 + "\n\n")

                self.show_success_info(
                    "Results Exported", f"Results exported to {file_path}"
                )
            except Exception as e:
                self.show_error_info(
                    "Export Failed", f"Failed to export results: {e!s}"
                )

    def update_search_results(self, results: list[SearchResult]) -> None:
        """Update search results display with enhanced feedback."""
        self._search_results = results

        # Hide state tooltip
        if self._search_state_tooltip:
            self._search_state_tooltip.close()
            self._search_state_tooltip = None

        # Re-enable search button
        self.search_btn.setEnabled(True)
        self.search_btn.setText("Search")

        # Update results table
        self.results_table.setRowCount(len(results))

        for row, result in enumerate(results):
            # Score
            score_item = QTableWidgetItem(f"{result.similarity_score:.3f}")
            self.results_table.setItem(row, 0, score_item)

            # Document
            doc_item = QTableWidgetItem(result.document_id)
            self.results_table.setItem(row, 1, doc_item)

            # Content (truncated)
            content_preview = (
                result.content[:100] + "..."
                if len(result.content) > 100
                else result.content
            )
            content_item = QTableWidgetItem(content_preview)
            self.results_table.setItem(row, 2, content_item)

            # Metadata
            metadata_str = str(result.metadata) if result.metadata else "N/A"
            metadata_item = QTableWidgetItem(
                metadata_str[:50] + "..." if len(metadata_str) > 50 else metadata_str
            )
            self.results_table.setItem(row, 3, metadata_item)

        # Resize columns to content
        self.results_table.resizeColumnsToContents()

        # Update result count badge
        result_count = len(results)
        if result_count > 0:
            self._result_count_badge.setText(str(result_count))
            self._result_count_badge.show()
            self.export_results_btn.setEnabled(True)

            # Show success feedback
            self.show_success_info(
                "Search Complete",
                f"Found {result_count} result(s) for '{self.query_input.text()}'",
                duration=3000,
            )
        else:
            self._result_count_badge.hide()
            self.export_results_btn.setEnabled(False)

            # Show info feedback
            self.show_info_bar(
                "No Results",
                f"No results found for '{self.query_input.text()}'",
                InfoLevel.WARNING,
            )

        # Clear preview
        self.result_preview.clear()

    def search_error(self, error_message: str) -> None:
        """Handle search error with enhanced feedback."""
        # Hide state tooltip
        if self._search_state_tooltip:
            self._search_state_tooltip.close()
            self._search_state_tooltip = None

        # Re-enable search button
        self.search_btn.setEnabled(True)
        self.search_btn.setText("Search")

        # Clear results
        self._clear_results()

        # Show error feedback
        self.show_error_info("Search Error", error_message)

    def _connect_controller_signals(self) -> None:
        """Connect controller signals to widget slots."""
        self.controller.search_started.connect(self._on_search_started)
        self.controller.search_completed.connect(self._on_search_completed)
        self.controller.search_error.connect(self._on_search_error)
        self.controller.status_message.connect(self._on_status_message)

    def _on_search_started(self, query: str) -> None:
        """Handle search started signal."""
        self.emit_status(f"Searching for: {query}")

    def _on_search_completed(self, results: list[SearchResult]) -> None:
        """Handle search completed signal."""
        self.update_search_results(results)
        self.emit_status(f"Search completed: {len(results)} results found")

    def _on_search_error(self, error_message: str) -> None:
        """Handle search error signal."""
        self.search_error(error_message)
        self.emit_status(f"Search error: {error_message}")

    def _on_status_message(self, message: str) -> None:
        """Handle status message signal."""
        self.emit_status(message)

    def on_tab_activated(self) -> None:
        """Called when this tab is activated."""
        self.emit_status("Search tab activated")
        self.query_input.setFocus()
