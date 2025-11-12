"""
PDF processing widget for PDF Vector System GUI.

This module contains the enhanced widget for PDF file processing functionality
with modern QFluentWidgets components.
"""

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QFileDialog, QFormLayout, QHBoxLayout, QWidget
from qfluentwidgets import (
    BodyLabel,
    CheckBox,
    InfoBadge,
    InfoLevel,
    PrimaryPushButton,
    ProgressBar,
    ProgressRing,
    PushButton,
    SegmentedWidget,
    SmoothScrollArea,
    SpinBox,
    StateToolTip,
    TextEdit,
    VBoxLayout,
)

from pdf_vector_system.core.config.settings import Config
from pdf_vector_system.gui.controllers.processing_controller import ProcessingController
from pdf_vector_system.gui.utils.styling import (
    create_styled_card_widget,
    get_fluent_icon_for_action,
)
from pdf_vector_system.gui.widgets.base import BaseWidget


class ProcessingWidget(BaseWidget):
    """Enhanced widget for PDF processing functionality with modern UI components."""

    def __init__(
        self, config: Optional[Config] = None, parent: Optional[QWidget] = None
    ):
        """
        Initialize the processing widget.

        Args:
            config: Configuration object
            parent: Parent widget
        """
        super().__init__(config, parent)
        self.selected_files: list[Path] = []

        # Processing state tracking
        self._current_processing_mode = "standard"
        self._processing_state_tooltip: Optional[StateToolTip] = None
        self._file_count_badge: InfoBadge  # Initialized in _setup_ui

        # Initialize controller
        self.controller = ProcessingController(self.config, self)
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

        # Processing mode selection
        self._create_processing_mode_section(layout)

        # File selection section
        self._create_file_selection_section(layout)

        # Processing options section
        self._create_processing_options_section(layout)

        # Processing controls section
        self._create_processing_controls_section(layout)

        # Progress section
        self._create_progress_section(layout)

        # Results section
        self._create_results_section(layout)

    def _create_processing_mode_section(self, layout: VBoxLayout) -> None:
        """Create the processing mode selection section."""
        mode_card = create_styled_card_widget("Processing Mode", self)
        mode_layout = VBoxLayout(mode_card)

        # Add description
        desc_label = BodyLabel("Choose how you want to process your PDF files:")
        mode_layout.addWidget(desc_label)

        # Create segmented widget for processing modes
        self.mode_selector = SegmentedWidget(self)
        self.mode_selector.addItem(
            "standard", "Standard", icon=get_fluent_icon_for_action("document")
        )
        self.mode_selector.addItem(
            "batch", "Batch", icon=get_fluent_icon_for_action("add")
        )
        self.mode_selector.addItem(
            "advanced", "Advanced", icon=get_fluent_icon_for_action("settings")
        )

        # Set current mode
        self.mode_selector.setCurrentItem("standard")
        self.mode_selector.currentItemChanged.connect(self._on_processing_mode_changed)

        mode_layout.addWidget(self.mode_selector)

        # Add help button with teaching tip
        help_layout = QHBoxLayout()
        help_layout.addStretch()

        self.mode_help_btn = PushButton("?")
        self.mode_help_btn.setFixedSize(24, 24)
        self.mode_help_btn.setIcon(get_fluent_icon_for_action("help").icon())
        self.mode_help_btn.clicked.connect(self._show_mode_help)

        help_layout.addWidget(self.mode_help_btn)
        mode_layout.addLayout(help_layout)

        layout.addWidget(mode_card)

    def _create_file_selection_section(self, layout: VBoxLayout) -> None:
        """Create the enhanced file selection section."""
        file_card = create_styled_card_widget("PDF File Selection", self)
        file_layout = VBoxLayout(file_card)

        # File selection buttons with enhanced styling
        button_layout = QHBoxLayout()

        self.select_files_btn = PrimaryPushButton("Select PDF Files...")
        self.select_files_btn.setIcon(get_fluent_icon_for_action("document").icon())

        self.select_folder_btn = PushButton("Select Folder...")
        self.select_folder_btn.setIcon(get_fluent_icon_for_action("open").icon())

        self.clear_files_btn = PushButton("Clear Selection")
        self.clear_files_btn.setIcon(get_fluent_icon_for_action("delete").icon())

        button_layout.addWidget(self.select_files_btn)
        button_layout.addWidget(self.select_folder_btn)
        button_layout.addWidget(self.clear_files_btn)
        button_layout.addStretch()

        file_layout.addLayout(button_layout)

        # Selected files display with info badge
        files_display_layout = QHBoxLayout()

        self.files_label = BodyLabel("No files selected")
        files_display_layout.addWidget(self.files_label)

        # Add file count badge
        self._file_count_badge = InfoBadge.info("0", self)
        self._file_count_badge.hide()
        files_display_layout.addWidget(self._file_count_badge)
        files_display_layout.addStretch()

        file_layout.addLayout(files_display_layout)
        layout.addWidget(file_card)

    def _create_processing_options_section(self, layout: VBoxLayout) -> None:
        """Create the enhanced processing options section."""
        options_card = create_styled_card_widget("Processing Options", self)
        options_layout = QFormLayout(options_card)

        # Add help button for processing options
        options_title_layout = QHBoxLayout()
        options_title_layout.addStretch()

        self.options_help_btn = PushButton("?")
        self.options_help_btn.setFixedSize(24, 24)
        self.options_help_btn.setIcon(get_fluent_icon_for_action("help").icon())
        self.options_help_btn.clicked.connect(self._show_processing_help)

        options_title_layout.addWidget(self.options_help_btn)
        options_layout.addRow(options_title_layout)

        # Processing options with enhanced controls
        self.clean_text_cb = CheckBox("Clean extracted text")
        self.clean_text_cb.setChecked(True)
        self.clean_text_cb.setToolTip(
            "Remove extra whitespace and formatting from extracted text"
        )
        options_layout.addRow("Text Cleaning:", self.clean_text_cb)

        self.chunk_size_spin = SpinBox()
        self.chunk_size_spin.setRange(100, 5000)
        self.chunk_size_spin.setValue(self.config.text_processing.chunk_size)
        self.chunk_size_spin.setToolTip(
            "Size of text chunks for processing (characters)"
        )
        options_layout.addRow("Chunk Size:", self.chunk_size_spin)

        self.batch_size_spin = SpinBox()
        self.batch_size_spin.setRange(1, 100)
        self.batch_size_spin.setValue(self.config.embedding.batch_size)
        self.batch_size_spin.setToolTip("Number of chunks to process simultaneously")
        options_layout.addRow("Batch Size:", self.batch_size_spin)

        layout.addWidget(options_card)

    def _create_processing_controls_section(self, layout: VBoxLayout) -> None:
        """Create the enhanced processing controls section."""
        controls_layout = QHBoxLayout()

        self.process_btn = PrimaryPushButton("Process Files")
        self.process_btn.setIcon(get_fluent_icon_for_action("play").icon())
        self.process_btn.setEnabled(False)
        self.process_btn.setMinimumWidth(120)

        self.stop_btn = PushButton("Stop Processing")
        self.stop_btn.setIcon(get_fluent_icon_for_action("stop").icon())
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumWidth(120)

        controls_layout.addWidget(self.process_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addStretch()

        layout.addLayout(controls_layout)

    def _create_progress_section(self, layout: VBoxLayout) -> None:
        """Create the enhanced progress section with StateToolTip."""
        progress_card = create_styled_card_widget("Processing Progress", self)
        progress_layout = VBoxLayout(progress_card)

        # Enhanced progress indicators container
        progress_container = QHBoxLayout()

        self.progress_bar = ProgressBar()
        self.progress_bar.setTextVisible(True)
        progress_container.addWidget(self.progress_bar)

        # Enhanced ProgressRing for indeterminate operations
        self.progress_ring = ProgressRing()
        self.progress_ring.setFixedSize(32, 32)
        self.progress_ring.hide()  # Initially hidden
        progress_container.addWidget(self.progress_ring)

        progress_container.addStretch()
        progress_layout.addLayout(progress_container)

        # Progress status label
        self.progress_status = BodyLabel("Ready to process files")
        progress_layout.addWidget(self.progress_status)

        layout.addWidget(progress_card)

    def _create_results_section(self, layout: VBoxLayout) -> None:
        """Create the enhanced results section."""
        results_card = create_styled_card_widget("Processing Results", self)
        results_layout = VBoxLayout(results_card)

        # Results display with smooth scrolling
        self.results_text = TextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(200)
        self.results_text.setPlaceholderText("Processing results will appear here...")

        results_layout.addWidget(self.results_text)

        # Results actions
        results_actions = QHBoxLayout()

        self.clear_results_btn = PushButton("Clear Results")
        self.clear_results_btn.setIcon(get_fluent_icon_for_action("delete").icon())
        self.clear_results_btn.clicked.connect(self.results_text.clear)

        self.save_results_btn = PushButton("Save Results")
        self.save_results_btn.setIcon(get_fluent_icon_for_action("save").icon())
        self.save_results_btn.clicked.connect(self._save_results)

        results_actions.addWidget(self.clear_results_btn)
        results_actions.addWidget(self.save_results_btn)
        results_actions.addStretch()

        results_layout.addLayout(results_actions)
        layout.addWidget(results_card)

    def _on_processing_mode_changed(self, item_key: str) -> None:
        """Handle processing mode change."""
        self._current_processing_mode = item_key

        # Show info about the selected mode
        mode_descriptions = {
            "standard": "Process files individually with standard settings",
            "batch": "Process multiple files efficiently in batches",
            "advanced": "Use advanced processing options and custom settings",
        }

        description = mode_descriptions.get(item_key, "")
        if description:
            self.show_info_bar(
                f"Mode: {item_key.title()}", description, InfoLevel.INFO, duration=2000
            )

    def _show_mode_help(self) -> None:
        """Show help for processing modes."""
        help_content = """
        <b>Standard:</b> Best for individual files or small batches<br>
        <b>Batch:</b> Optimized for processing many files at once<br>
        <b>Advanced:</b> Provides additional configuration options
        """

        self.show_teaching_tip(
            "Processing Modes", help_content, self.mode_help_btn, "mode_help"
        )

    def _show_processing_help(self) -> None:
        """Show help for processing options."""
        help_content = """
        <b>Text Cleaning:</b> Removes extra whitespace and formatting<br>
        <b>Chunk Size:</b> Size of text segments for processing<br>
        <b>Batch Size:</b> Number of chunks processed simultaneously
        """

        self.show_teaching_tip(
            "Processing Options", help_content, self.options_help_btn, "options_help"
        )

    def _save_results(self) -> None:
        """Save processing results to file."""
        if not self.results_text.toPlainText().strip():
            self.show_warning_info("No Results", "No results to save")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processing Results",
            "processing_results.txt",
            "Text Files (*.txt);;All Files (*)",
        )

        if file_path:
            try:
                with Path(file_path).open("w", encoding="utf-8") as f:
                    f.write(self.results_text.toPlainText())
                self.show_success_info("Results Saved", f"Results saved to {file_path}")
            except Exception as e:
                self.show_error_info("Save Failed", f"Failed to save results: {e!s}")

    def update_file_selection(self, files: list[Path]) -> None:
        """Update the file selection display with enhanced feedback."""
        self.selected_files = files
        file_count = len(files)

        if file_count == 0:
            self.files_label.setText("No files selected")
            self._file_count_badge.hide()
            self.process_btn.setEnabled(False)
        else:
            self.files_label.setText(f"{file_count} file(s) selected")
            self._file_count_badge.setText(str(file_count))
            self._file_count_badge.show()
            self.process_btn.setEnabled(True)

            # Show success feedback
            self.show_success_info(
                "Files Selected",
                f"Selected {file_count} PDF file(s) for processing",
                duration=2000,
            )

    def start_processing(self) -> None:
        """Start processing with enhanced feedback."""
        if not self.selected_files:
            self.show_warning_info("No Files", "Please select files to process first")
            return

        # Show state tooltip for processing
        self._processing_state_tooltip = self.show_state_tooltip(
            "Processing Files",
            f"Processing {len(self.selected_files)} files in {self._current_processing_mode} mode...",
            self.progress_bar,
        )

        # Update UI state
        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_ring.show()
        self.progress_status.setText("Processing started...")

        # Show info bar
        self.show_info_bar(
            "Processing Started",
            f"Processing {len(self.selected_files)} files",
            InfoLevel.INFO,
        )

    def stop_processing(self) -> None:
        """Stop processing with enhanced feedback."""
        # Hide state tooltip
        if self._processing_state_tooltip:
            self._processing_state_tooltip.close()
            self._processing_state_tooltip = None

        # Update UI state
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_ring.hide()
        self.progress_status.setText("Processing stopped")

        # Show warning feedback
        self.show_warning_info("Processing Stopped", "Processing was stopped by user")

    def update_progress(self, current: int, total: int, message: str = "") -> None:
        """Update progress with enhanced feedback."""
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_bar.setValue(percentage)

        status_text = f"Processing: {current}/{total}"
        if message:
            status_text += f" - {message}"

        self.progress_status.setText(status_text)

        # Update state tooltip if active
        if self._processing_state_tooltip:
            self._processing_state_tooltip.setContent(
                f"Progress: {current}/{total} files processed"
            )

    def processing_completed(self, success: bool, message: str = "") -> None:
        """Handle processing completion with enhanced feedback."""
        # Hide state tooltip
        if self._processing_state_tooltip:
            self._processing_state_tooltip.close()
            self._processing_state_tooltip = None

        # Update UI state
        self.process_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_ring.hide()

        if success:
            self.progress_status.setText("Processing completed successfully")
            self.show_success_info(
                "Processing Complete", message or "All files processed successfully"
            )
        else:
            self.progress_status.setText("Processing failed")
            self.show_error_info(
                "Processing Failed", message or "Processing encountered errors"
            )

    def add_result_message(self, message: str) -> None:
        """Add a message to the results display."""
        self.results_text.append(message)

        # Connect signals
        self._setup_connections()

    def _connect_controller_signals(self) -> None:
        """Connect controller signals to widget slots."""
        self.controller.processing_started.connect(self._on_processing_started)
        self.controller.processing_progress.connect(self._on_processing_progress)
        self.controller.file_processed.connect(self._on_file_processed)
        self.controller.processing_completed.connect(self._on_processing_completed)
        self.controller.processing_error.connect(self._on_processing_error)
        self.controller.status_message.connect(self._on_status_message)

    def _setup_connections(self) -> None:
        """Set up signal/slot connections."""
        self.select_files_btn.clicked.connect(self.select_files)
        self.select_folder_btn.clicked.connect(self.select_folder)
        self.clear_files_btn.clicked.connect(self.clear_files)
        self.process_btn.clicked.connect(self.process_files)
        self.stop_btn.clicked.connect(self.stop_processing)

    def select_files(self) -> None:
        """Open file dialog to select PDF files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select PDF Files", "", "PDF Files (*.pdf);;All Files (*)"
        )

        if files:
            file_paths = [Path(f) for f in files]
            self.update_file_selection(file_paths)

    def select_folder(self) -> None:
        """Open dialog to select folder containing PDFs."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing PDFs")

        if folder:
            folder_path = Path(folder)
            pdf_files = list(folder_path.glob("*.pdf"))
            if pdf_files:
                self.update_file_selection(pdf_files)
            else:
                self.show_warning_info(
                    "No PDFs Found", "No PDF files found in selected folder"
                )

    def clear_files(self) -> None:
        """Clear selected files."""
        self.update_file_selection([])

    def process_files(self) -> None:
        """Start processing selected files."""
        if not self.selected_files:
            return

        # Get processing options
        clean_text = self.clean_text_cb.isChecked()

        # Update config with current values
        self.config.text_processing.chunk_size = self.chunk_size_spin.value()
        self.config.embedding.batch_size = self.batch_size_spin.value()
        self.controller.update_config(self.config)

        # Start processing with enhanced feedback
        self.start_processing()

        # Start actual processing
        self.controller.process_files(self.selected_files, clean_text)

    def _on_processing_started(self, file_paths: list[str]) -> None:
        """Handle processing started signal."""
        self.add_result_message(f"Started processing {len(file_paths)} files")
        self.emit_status(f"Processing {len(file_paths)} files...")

    def _on_processing_progress(self, current_file: int, total_files: int) -> None:
        """Handle processing progress signal."""
        self.update_progress(current_file, total_files)
        self.emit_status(f"Processing file {current_file} of {total_files}")

    def _on_file_processed(self, file_path: str, success: bool, message: str) -> None:
        """Handle file processed signal."""
        file_name = Path(file_path).name
        status = "✓" if success else "✗"
        self.add_result_message(f"{status} {file_name}: {message}")

    def _on_processing_completed(self, successful_count: int, total_count: int) -> None:
        """Handle processing completed signal."""
        success = successful_count == total_count
        message = (
            f"Processing completed: {successful_count}/{total_count} files successful"
        )
        self.add_result_message(message)
        self.processing_completed(success, message)
        self.emit_status(
            f"Processing completed: {successful_count}/{total_count} successful"
        )

    def _on_processing_error(self, error_message: str) -> None:
        """Handle processing error signal."""
        self.add_result_message(f"Error: {error_message}")
        self.processing_completed(False, error_message)
        self.emit_status(f"Processing error: {error_message}")

    def _on_status_message(self, message: str) -> None:
        """Handle status message signal."""
        self.emit_status(message)

    def on_tab_activated(self) -> None:
        """Called when this tab is activated."""
        self.emit_status("PDF Processing tab activated")
