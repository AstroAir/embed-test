"""
PDF processing widget for PDF Vector System GUI.

This module contains the widget for PDF file processing functionality.
"""

from typing import Optional, List
from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QWidget, QHBoxLayout, QFormLayout
from PySide6.QtCore import Qt
from qfluentwidgets import (
    VBoxLayout, PushButton, BodyLabel,
    ProgressBar, TextEdit, CardWidget,
    CheckBox, SpinBox
)

from ...config.settings import Config
from .base import BaseWidget
from ..controllers.processing_controller import ProcessingController


class ProcessingWidget(BaseWidget):
    """Widget for PDF processing functionality."""
    
    def __init__(self, config: Optional[Config] = None, parent: Optional[QWidget] = None):
        """
        Initialize the processing widget.

        Args:
            config: Configuration object
            parent: Parent widget
        """
        super().__init__(config, parent)
        self.selected_files: List[Path] = []

        # Initialize controller
        self.controller = ProcessingController(self.config, self)
        self._connect_controller_signals()
        
    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = VBoxLayout(self)

        # File selection section
        file_group = CardWidget()
        file_layout = VBoxLayout(file_group)

        # Add title for the card
        file_title = BodyLabel("PDF File Selection")
        file_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        file_layout.addWidget(file_title)

        # File selection buttons
        button_layout = QHBoxLayout()
        self.select_files_btn = PushButton("Select PDF Files...")
        self.select_folder_btn = PushButton("Select Folder...")
        self.clear_files_btn = PushButton("Clear Selection")

        button_layout.addWidget(self.select_files_btn)
        button_layout.addWidget(self.select_folder_btn)
        button_layout.addWidget(self.clear_files_btn)
        button_layout.addStretch()

        file_layout.addLayout(button_layout)

        # Selected files display
        self.files_label = BodyLabel("No files selected")
        file_layout.addWidget(self.files_label)

        layout.addWidget(file_group)
        
        # Processing options section
        options_group = CardWidget()
        options_layout = QFormLayout(options_group)

        # Add title for the card
        options_title = BodyLabel("Processing Options")
        options_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        options_layout.addRow(options_title)

        self.clean_text_cb = CheckBox("Clean extracted text")
        self.clean_text_cb.setChecked(True)
        options_layout.addRow("Text Cleaning:", self.clean_text_cb)

        self.chunk_size_spin = SpinBox()
        self.chunk_size_spin.setRange(100, 5000)
        self.chunk_size_spin.setValue(self.config.text_processing.chunk_size)
        options_layout.addRow("Chunk Size:", self.chunk_size_spin)

        self.batch_size_spin = SpinBox()
        self.batch_size_spin.setRange(1, 100)
        self.batch_size_spin.setValue(self.config.embedding.batch_size)
        options_layout.addRow("Batch Size:", self.batch_size_spin)

        layout.addWidget(options_group)
        
        # Processing controls
        controls_layout = QHBoxLayout()
        self.process_btn = PushButton("Process Files")
        self.process_btn.setEnabled(False)
        self.stop_btn = PushButton("Stop Processing")
        self.stop_btn.setEnabled(False)

        controls_layout.addWidget(self.process_btn)
        controls_layout.addWidget(self.stop_btn)
        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        # Progress section
        progress_group = CardWidget()
        progress_layout = VBoxLayout(progress_group)

        # Add title for the card
        progress_title = BodyLabel("Processing Progress")
        progress_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        progress_layout.addWidget(progress_title)

        self.progress_bar = ProgressBar()
        progress_layout.addWidget(self.progress_bar)

        self.status_text = TextEdit()
        self.status_text.setMaximumHeight(150)
        self.status_text.setReadOnly(True)
        progress_layout.addWidget(self.status_text)

        layout.addWidget(progress_group)
        
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
            self,
            "Select PDF Files",
            "",
            "PDF Files (*.pdf);;All Files (*)"
        )
        
        if files:
            self.selected_files = [Path(f) for f in files]
            self._update_files_display()
            
    def select_folder(self) -> None:
        """Open dialog to select folder containing PDFs."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Folder Containing PDFs"
        )
        
        if folder:
            folder_path = Path(folder)
            pdf_files = list(folder_path.glob("*.pdf"))
            if pdf_files:
                self.selected_files = pdf_files
                self._update_files_display()
            else:
                self.emit_status("No PDF files found in selected folder")
                
    def clear_files(self) -> None:
        """Clear selected files."""
        self.selected_files = []
        self._update_files_display()
        
    def _update_files_display(self) -> None:
        """Update the display of selected files."""
        if self.selected_files:
            file_names = [f.name for f in self.selected_files]
            if len(file_names) <= 5:
                display_text = "\n".join(file_names)
            else:
                display_text = "\n".join(file_names[:5]) + f"\n... and {len(file_names) - 5} more files"
            
            self.files_label.setText(f"Selected {len(self.selected_files)} file(s):\n{display_text}")
            self.process_btn.setEnabled(True)
        else:
            self.files_label.setText("No files selected")
            self.process_btn.setEnabled(False)
            
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

        # Start processing
        self.controller.process_files(self.selected_files, clean_text)

        # Update UI state
        self.process_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)

    def stop_processing(self) -> None:
        """Stop file processing."""
        if self.controller.stop_processing():
            self.emit_status("Processing stopped by user")
            self.status_text.append("Processing stopped by user")
            self._reset_processing_ui()
        
    def _reset_processing_ui(self) -> None:
        """Reset UI to non-processing state."""
        self.process_btn.setEnabled(len(self.selected_files) > 0)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setValue(0)

    def _on_processing_started(self, file_paths: List[str]) -> None:
        """Handle processing started signal."""
        self.status_text.append(f"Started processing {len(file_paths)} files")
        self.emit_status(f"Processing {len(file_paths)} files...")

    def _on_processing_progress(self, current_file: int, total_files: int) -> None:
        """Handle processing progress signal."""
        progress = int((current_file / total_files) * 100)
        self.progress_bar.setValue(progress)
        self.emit_status(f"Processing file {current_file} of {total_files}")

    def _on_file_processed(self, file_path: str, success: bool, message: str) -> None:
        """Handle file processed signal."""
        file_name = Path(file_path).name
        status = "✓" if success else "✗"
        self.status_text.append(f"{status} {file_name}: {message}")

    def _on_processing_completed(self, successful_count: int, total_count: int) -> None:
        """Handle processing completed signal."""
        self.status_text.append(f"Processing completed: {successful_count}/{total_count} files successful")
        self.emit_status(f"Processing completed: {successful_count}/{total_count} successful")
        self._reset_processing_ui()

    def _on_processing_error(self, error_message: str) -> None:
        """Handle processing error signal."""
        self.status_text.append(f"Error: {error_message}")
        self.emit_status(f"Processing error: {error_message}")
        self._reset_processing_ui()

    def _on_status_message(self, message: str) -> None:
        """Handle status message signal."""
        self.emit_status(message)

    def on_tab_activated(self) -> None:
        """Called when this tab is activated."""
        self.emit_status("PDF Processing tab activated")
