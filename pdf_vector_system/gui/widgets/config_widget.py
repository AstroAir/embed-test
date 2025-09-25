"""
Configuration widget for PDF Vector System GUI.

This module contains the widget for configuration management.
"""

from typing import Optional, Any

from PySide6.QtWidgets import QFileDialog, QWidget, QHBoxLayout, QFormLayout
from PySide6.QtCore import Qt, Signal
from qfluentwidgets import (
    VBoxLayout, PushButton, BodyLabel,
    LineEdit, SpinBox, DoubleSpinBox, ComboBox,
    CardWidget, CheckBox, ScrollArea,
    TextEdit, MessageBox
)

from ...config.settings import Config, EmbeddingModelType, LogLevel
from .base import BaseWidget
from ..controllers.config_controller import ConfigController


class ConfigWidget(BaseWidget):
    """Widget for configuration management."""

    # Signals
    config_changed: Signal = Signal(object)  # new_config

    def __init__(self, config: Optional[Config] = None, parent: Optional[QWidget] = None):
        """
        Initialize the config widget.

        Args:
            config: Configuration object
            parent: Parent widget
        """
        super().__init__(config, parent)

        # Initialize controller
        self.controller = ConfigController(self.config, self)
        self._connect_controller_signals()

        # Track if changes have been made
        self._has_changes = False
        
    def _setup_ui(self) -> None:
        """Set up the user interface."""
        # Apply consistent card title styling
        self.setStyleSheet("""
            BodyLabel[objectName="cardTitle"] {
                font-weight: bold;
                font-size: 14px;
                margin-bottom: 10px;
                color: palette(text);
            }
        """)

        layout = VBoxLayout(self)

        # Controls
        controls_layout = QHBoxLayout()
        self.apply_btn = PushButton("Apply Changes")
        self.reset_btn = PushButton("Reset to Defaults")
        self.save_btn = PushButton("Save to File")
        self.load_btn = PushButton("Load from File")

        controls_layout.addWidget(self.apply_btn)
        controls_layout.addWidget(self.reset_btn)
        controls_layout.addWidget(self.save_btn)
        controls_layout.addWidget(self.load_btn)
        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        # Scrollable configuration area
        scroll_area = ScrollArea()
        scroll_widget = QWidget()
        scroll_layout = VBoxLayout(scroll_widget)
        
        # Embedding Configuration
        embedding_group = CardWidget()
        embedding_layout = QFormLayout(embedding_group)

        # Add title for the card
        embedding_title = BodyLabel("Embedding Configuration")
        embedding_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        embedding_layout.addRow(embedding_title)

        self.model_type_combo = ComboBox()
        self.model_type_combo.addItems([e.value for e in EmbeddingModelType])
        self.model_type_combo.setCurrentText(self.config.embedding.model_type.value)
        embedding_layout.addRow("Model Type:", self.model_type_combo)

        self.model_name_edit = LineEdit()
        self.model_name_edit.setText(self.config.embedding.model_name)
        embedding_layout.addRow("Model Name:", self.model_name_edit)

        self.batch_size_spin = SpinBox()
        self.batch_size_spin.setRange(1, 1000)
        self.batch_size_spin.setValue(self.config.embedding.batch_size)
        embedding_layout.addRow("Batch Size:", self.batch_size_spin)

        self.openai_key_edit = LineEdit()
        self.openai_key_edit.setText(self.config.embedding.openai_api_key or "")
        self.openai_key_edit.setEchoMode(LineEdit.Password)
        embedding_layout.addRow("OpenAI API Key:", self.openai_key_edit)

        scroll_layout.addWidget(embedding_group)
        
        # Text Processing Configuration
        text_group = CardWidget()
        text_layout = QFormLayout(text_group)

        # Add title for the card
        text_title = BodyLabel("Text Processing Configuration")
        text_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        text_layout.addRow(text_title)

        self.chunk_size_spin = SpinBox()
        self.chunk_size_spin.setRange(100, 10000)
        self.chunk_size_spin.setValue(self.config.text_processing.chunk_size)
        text_layout.addRow("Chunk Size:", self.chunk_size_spin)

        self.chunk_overlap_spin = SpinBox()
        self.chunk_overlap_spin.setRange(0, 1000)
        self.chunk_overlap_spin.setValue(self.config.text_processing.chunk_overlap)
        text_layout.addRow("Chunk Overlap:", self.chunk_overlap_spin)

        self.min_chunk_size_spin = SpinBox()
        self.min_chunk_size_spin.setRange(10, 1000)
        self.min_chunk_size_spin.setValue(self.config.text_processing.min_chunk_size)
        text_layout.addRow("Min Chunk Size:", self.min_chunk_size_spin)

        scroll_layout.addWidget(text_group)
        
        # ChromaDB Configuration
        chroma_group = CardWidget()
        chroma_layout = QFormLayout(chroma_group)

        # Add title for the card
        chroma_title = BodyLabel("ChromaDB Configuration")
        chroma_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        chroma_layout.addRow(chroma_title)

        self.persist_dir_edit = LineEdit()
        self.persist_dir_edit.setText(str(self.config.chroma_db.persist_directory))
        chroma_layout.addRow("Persist Directory:", self.persist_dir_edit)

        self.collection_name_edit = LineEdit()
        self.collection_name_edit.setText(self.config.chroma_db.collection_name)
        chroma_layout.addRow("Collection Name:", self.collection_name_edit)

        self.max_results_spin = SpinBox()
        self.max_results_spin.setRange(1, 1000)
        self.max_results_spin.setValue(self.config.chroma_db.max_results)
        chroma_layout.addRow("Max Results:", self.max_results_spin)

        scroll_layout.addWidget(chroma_group)

        # PDF Configuration
        pdf_group = CardWidget()
        pdf_layout = QFormLayout(pdf_group)

        # Add title for the card
        pdf_title = BodyLabel("PDF Processing Configuration")
        pdf_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        pdf_layout.addRow(pdf_title)

        self.max_file_size_spin = SpinBox()
        self.max_file_size_spin.setRange(1, 1000)
        self.max_file_size_spin.setValue(self.config.pdf.max_file_size_mb)
        self.max_file_size_spin.setSuffix(" MB")
        pdf_layout.addRow("Max File Size:", self.max_file_size_spin)
        
        self.timeout_spin = SpinBox()
        self.timeout_spin.setRange(10, 3600)
        self.timeout_spin.setValue(self.config.pdf.timeout_seconds)
        self.timeout_spin.setSuffix(" seconds")
        pdf_layout.addRow("Timeout:", self.timeout_spin)
        
        scroll_layout.addWidget(pdf_group)
        
        # Logging Configuration using CardWidget for modern design
        logging_group = CardWidget()
        logging_layout = QFormLayout(logging_group)

        # CardWidget title using BodyLabel for consistent typography
        logging_title = BodyLabel("Logging Configuration")
        logging_title.setObjectName("cardTitle")  # Use object name for consistent styling
        logging_layout.addRow(logging_title)
        
        self.log_level_combo = ComboBox()
        self.log_level_combo.addItems([e.value for e in LogLevel])
        self.log_level_combo.setCurrentText(self.config.logging.level.value)
        logging_layout.addRow("Log Level:", self.log_level_combo)

        self.log_file_edit = LineEdit()
        self.log_file_edit.setText(str(self.config.logging.file_path) if self.config.logging.file_path else "")
        logging_layout.addRow("Log File:", self.log_file_edit)
        
        scroll_layout.addWidget(logging_group)
        
        # Global Settings
        global_group = CardWidget()
        global_layout = QFormLayout(global_group)

        # Add title for the card
        global_title = BodyLabel("Global Settings")
        global_title.setObjectName("cardTitle")  # Use object name for consistent styling
        global_layout.addRow(global_title)

        self.debug_cb = CheckBox()
        self.debug_cb.setChecked(self.config.debug)
        global_layout.addRow("Debug Mode:", self.debug_cb)

        self.max_workers_spin = SpinBox()
        self.max_workers_spin.setRange(1, 32)
        self.max_workers_spin.setValue(self.config.max_workers)
        global_layout.addRow("Max Workers:", self.max_workers_spin)

        scroll_layout.addWidget(global_group)
        
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        
        layout.addWidget(scroll_area)
        
        # Status area using QFluentWidgets TextEdit for consistent theming
        self.status_text = TextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        self.status_text.setPlaceholderText("Configuration status messages will appear here...")
        layout.addWidget(self.status_text)
        
        # Connect signals
        self._setup_connections()

    def _connect_controller_signals(self) -> None:
        """Connect controller signals to widget slots."""
        self.controller.config_updated.connect(self._on_config_updated)
        self.controller.config_saved.connect(self._on_config_saved)
        self.controller.config_loaded.connect(self._on_config_loaded)
        self.controller.validation_error.connect(self._on_validation_error)
        self.controller.operation_error.connect(self._on_operation_error)
        self.controller.status_message.connect(self._on_status_message)

    def _setup_connections(self) -> None:
        """Set up signal/slot connections."""
        self.apply_btn.clicked.connect(self.apply_changes)
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        self.save_btn.clicked.connect(self.save_to_file)
        self.load_btn.clicked.connect(self.load_from_file)

        # Connect input change signals to track modifications
        self._connect_input_signals()
        
    def _connect_input_signals(self) -> None:
        """Connect input widget signals to track changes."""
        # This would connect all input widgets to a change handler
        # For brevity, implementing a simplified version
        pass

    def _mark_changed(self) -> None:
        """Mark configuration as changed."""
        self._has_changes = True
        self.apply_btn.setEnabled(True)

    def _collect_config_values(self) -> dict:
        """Collect current values from all input widgets."""
        # This would collect values from all input widgets
        # For now, return current config as dict
        return self.config.model_dump()

    def apply_changes(self) -> None:
        """Apply configuration changes."""
        try:
            config_dict = self._collect_config_values()
            if self.controller.apply_config(config_dict):
                self._has_changes = False
                self.apply_btn.setEnabled(False)
                self.config_changed.emit(self.controller.get_current_config())
        except Exception as e:
            self.status_text.append(f"Error applying changes: {str(e)}")

    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults using QFluentWidgets MessageBox."""
        # Use QFluentWidgets MessageBox for consistent fluent design
        w = MessageBox(
            "Reset Configuration",
            "Are you sure you want to reset all settings to defaults?\n\n"
            "This will lose any unsaved changes.",
            self
        )
        w.yesButton.setText("Reset")
        w.cancelButton.setText("Cancel")

        if w.exec():
            self.controller.reset_to_defaults()

    def save_to_file(self) -> None:
        """Save configuration to file."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            "config.json",
            "JSON Files (*.json);;All Files (*)"
        )

        if filename:
            self.controller.save_to_file(filename)

    def load_from_file(self) -> None:
        """Load configuration from file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )

        if filename:
            self.controller.load_from_file(filename)
            
    def _on_config_updated(self, new_config: Config) -> None:
        """Handle config updated signal."""
        self.config = new_config
        self._update_ui_from_config()
        self.status_text.append("Configuration updated successfully")

    def _on_config_saved(self, file_path: str) -> None:
        """Handle config saved signal."""
        self.status_text.append(f"Configuration saved to {file_path}")

    def _on_config_loaded(self, loaded_config: Config) -> None:
        """Handle config loaded signal."""
        self.config = loaded_config
        self._update_ui_from_config()
        self.status_text.append("Configuration loaded successfully")
        self.config_changed.emit(loaded_config)

    def _on_validation_error(self, error_message: str) -> None:
        """Handle validation error signal."""
        self.status_text.append(f"Validation error: {error_message}")
        # Use consistent MessageBox pattern with proper button configuration
        w = MessageBox("Validation Error", error_message, self)
        w.cancelButton.setText("OK")
        w.yesButton.hide()  # Hide yes button for simple acknowledgment
        w.exec()

    def _on_operation_error(self, error_message: str) -> None:
        """Handle operation error signal."""
        self.status_text.append(f"Error: {error_message}")
        # Use consistent MessageBox pattern with proper button configuration
        w = MessageBox("Operation Error", error_message, self)
        w.cancelButton.setText("OK")
        w.yesButton.hide()  # Hide yes button for simple acknowledgment
        w.exec()

    def _on_status_message(self, message: str) -> None:
        """Handle status message signal."""
        self.emit_status(message)

    def _update_ui_from_config(self) -> None:
        """Update UI widgets from current configuration."""
        # This would update all input widgets with current config values
        # For now, just mark as no changes
        self._has_changes = False
        self.apply_btn.setEnabled(False)

    def on_tab_activated(self) -> None:
        """Called when this tab is activated."""
        self.emit_status("Configuration tab activated")
