"""
Log viewer widget for PDF Vector System GUI.

This module contains the widget for log viewing and monitoring.
"""

from typing import Optional

from PySide6.QtWidgets import QWidget, QHBoxLayout, QFormLayout
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from qfluentwidgets import (
    VBoxLayout, PushButton, BodyLabel,
    TextEdit, CardWidget, ComboBox, CheckBox,
    SpinBox
)

from ...config.settings import Config, LogLevel
from .base import BaseWidget


class LogWidget(BaseWidget):
    """Widget for log viewing and monitoring."""
    
    def __init__(self, config: Optional[Config] = None, parent: Optional[QWidget] = None):
        """
        Initialize the log widget.
        
        Args:
            config: Configuration object
            parent: Parent widget
        """
        super().__init__(config, parent)
        
        # Setup auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.auto_refresh_logs)
        
    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = VBoxLayout(self)

        # Controls section
        controls_group = CardWidget()
        controls_layout = QFormLayout(controls_group)

        # Add title for the card
        controls_title = BodyLabel("Log Controls")
        controls_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        controls_layout.addRow(controls_title)

        # Log level filter
        self.level_filter = ComboBox()
        self.level_filter.addItem("All Levels")
        self.level_filter.addItems([level.value for level in LogLevel])
        controls_layout.addRow("Filter Level:", self.level_filter)

        # Auto-refresh
        self.auto_refresh_cb = CheckBox("Auto Refresh")
        controls_layout.addRow("Auto Refresh:", self.auto_refresh_cb)

        # Max lines
        self.max_lines_spin = SpinBox()
        self.max_lines_spin.setRange(100, 10000)
        self.max_lines_spin.setValue(1000)
        controls_layout.addRow("Max Lines:", self.max_lines_spin)

        layout.addWidget(controls_group)

        # Action buttons
        buttons_layout = QHBoxLayout()
        self.refresh_btn = PushButton("Refresh")
        self.clear_btn = PushButton("Clear Display")
        self.save_btn = PushButton("Save Logs...")
        
        buttons_layout.addWidget(self.refresh_btn)
        buttons_layout.addWidget(self.clear_btn)
        buttons_layout.addWidget(self.save_btn)
        buttons_layout.addStretch()
        
        layout.addLayout(buttons_layout)
        
        # Log display
        log_group = CardWidget()
        log_layout = VBoxLayout(log_group)

        # Add title for the card
        log_title = BodyLabel("Log Messages")
        log_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        log_layout.addWidget(log_title)

        self.log_display = TextEdit()
        self.log_display.setReadOnly(True)

        # Use monospace font for better log readability
        font = QFont("Consolas", 9)
        if not font.exactMatch():
            font = QFont("Courier New", 9)
        self.log_display.setFont(font)

        # Set placeholder text
        self.log_display.setPlaceholderText(
            "Log messages will appear here...\n"
            "Click 'Refresh' to load recent logs or enable 'Auto Refresh'"
        )

        log_layout.addWidget(self.log_display)
        layout.addWidget(log_group)

        # Status info
        self.status_label = BodyLabel("Ready - Click 'Refresh' to load logs")
        layout.addWidget(self.status_label)
        
        # Connect signals
        self._setup_connections()
        
    def _setup_connections(self) -> None:
        """Set up signal/slot connections."""
        self.refresh_btn.clicked.connect(self.refresh_logs)
        self.clear_btn.clicked.connect(self.clear_logs)
        self.save_btn.clicked.connect(self.save_logs)
        self.auto_refresh_cb.toggled.connect(self.toggle_auto_refresh)
        self.level_filter.currentTextChanged.connect(self.filter_logs)
        
    def refresh_logs(self) -> None:
        """Refresh log display."""
        self.emit_status("Refreshing logs...")
        
        # TODO: Implement actual log reading
        # For now, show placeholder log messages
        sample_logs = [
            "2024-01-15 10:30:15 | INFO | pdf_vector_system.pipeline | Pipeline initialized successfully",
            "2024-01-15 10:30:16 | DEBUG | pdf_vector_system.config | Configuration loaded from defaults",
            "2024-01-15 10:30:17 | INFO | pdf_vector_system.embeddings | Embedding service created: sentence-transformers",
            "2024-01-15 10:30:18 | INFO | pdf_vector_system.vector_db | ChromaDB client initialized",
            "2024-01-15 10:30:19 | WARNING | pdf_vector_system.pdf | Large PDF file detected: sample.pdf (25MB)",
            "2024-01-15 10:30:20 | INFO | pdf_vector_system.pipeline | Processing PDF: sample.pdf",
            "2024-01-15 10:30:25 | INFO | pdf_vector_system.pdf | Extracted 50 pages from sample.pdf",
            "2024-01-15 10:30:30 | INFO | pdf_vector_system.text | Generated 125 text chunks",
            "2024-01-15 10:30:35 | INFO | pdf_vector_system.embeddings | Generated embeddings for 125 chunks",
            "2024-01-15 10:30:40 | INFO | pdf_vector_system.vector_db | Stored 125 chunks in ChromaDB",
            "2024-01-15 10:30:41 | INFO | pdf_vector_system.pipeline | Processing completed successfully"
        ]
        
        # Apply level filter
        filtered_logs = self._apply_level_filter(sample_logs)
        
        # Apply max lines limit
        max_lines = self.max_lines_spin.value()
        if len(filtered_logs) > max_lines:
            filtered_logs = filtered_logs[-max_lines:]
            
        # Update display
        self.log_display.clear()
        self.log_display.append("\n".join(filtered_logs))
        
        # Scroll to bottom
        cursor = self.log_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_display.setTextCursor(cursor)
        
        self.status_label.setText(f"Showing {len(filtered_logs)} log entries (placeholder data)")
        self.emit_status("Log refresh completed")
        
    def _apply_level_filter(self, logs: list) -> list:
        """Apply log level filter to log entries."""
        selected_level = self.level_filter.currentText()
        
        if selected_level == "All Levels":
            return logs
            
        # Filter logs by level (simple string matching for demo)
        filtered = []
        for log in logs:
            if f"| {selected_level} |" in log:
                filtered.append(log)
                
        return filtered
        
    def clear_logs(self) -> None:
        """Clear the log display."""
        self.log_display.clear()
        self.status_label.setText("Log display cleared")
        self.emit_status("Log display cleared")
        
    def save_logs(self) -> None:
        """Save current log display to file."""
        from PySide6.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Logs",
            "logs.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.log_display.toPlainText())
                self.status_label.setText(f"Logs saved to {filename}")
                self.emit_status(f"Logs saved to {filename}")
            except Exception as e:
                self.emit_error(f"Failed to save logs: {str(e)}")
                
    def toggle_auto_refresh(self, enabled: bool) -> None:
        """Toggle auto-refresh functionality."""
        if enabled:
            self.refresh_timer.start(2000)  # Refresh every 2 seconds
            self.status_label.setText("Auto-refresh enabled (2 second interval)")
        else:
            self.refresh_timer.stop()
            self.status_label.setText("Auto-refresh disabled")
            
    def auto_refresh_logs(self) -> None:
        """Perform automatic log refresh."""
        # Only refresh if there's content to avoid spam
        if self.log_display.toPlainText():
            self.refresh_logs()
            
    def filter_logs(self) -> None:
        """Apply log level filter to current display."""
        if self.log_display.toPlainText():
            self.refresh_logs()
            
    def on_tab_activated(self) -> None:
        """Called when this tab is activated."""
        self.emit_status("Logs tab activated")
        # Auto-load logs when tab is first activated
        if not self.log_display.toPlainText():
            self.refresh_logs()
