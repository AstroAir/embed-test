"""
Log viewer widget for PDF Vector System GUI.

This module contains the widget for log viewing and monitoring.
"""

from pathlib import Path
from typing import Optional, cast

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QResizeEvent
from PySide6.QtWidgets import QFormLayout, QHBoxLayout, QSizePolicy, QSplitter, QWidget
from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    CheckBox,
    ComboBox,
    PushButton,
    SpinBox,
    TextEdit,
    VBoxLayout,
)

from vectorflow.core.config.settings import Config, LogLevel
from vectorflow.gui.widgets.base import BaseWidget


class LogWidget(BaseWidget):
    """Widget for log viewing and monitoring."""

    def __init__(
        self, config: Optional[Config] = None, parent: Optional[QWidget] = None
    ):
        """
        Initialize the log widget.

        Args:
            config: Configuration object
            parent: Parent widget
        """
        super().__init__(config, parent)

        self.controls_layout: Optional[QFormLayout] = None

        # Setup auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.auto_refresh_logs)

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = VBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Header
        title_label = BodyLabel("Log Monitor")
        title_label.setStyleSheet(
            "font-weight: 600; font-size: 18px; margin-bottom: 2px;"
        )
        layout.addWidget(title_label)

        subtitle_label = BodyLabel(
            "Monitor pipeline activity in real time and keep an eye on potential issues."
        )
        subtitle_label.setStyleSheet("color: palette(mid); margin-bottom: 8px;")
        layout.addWidget(subtitle_label)

        # Main splitter for adaptive resizing
        content_splitter = QSplitter(Qt.Orientation.Vertical)
        content_splitter.setChildrenCollapsible(False)
        layout.addWidget(content_splitter, 1)

        # Controls section card
        controls_group = CardWidget()
        controls_group.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        controls_group.setMinimumHeight(190)
        controls_group_layout = VBoxLayout(controls_group)
        controls_group_layout.setContentsMargins(20, 16, 20, 20)
        controls_group_layout.setSpacing(12)

        controls_title = BodyLabel("Log Controls")
        controls_title.setStyleSheet("font-weight: 600; font-size: 15px;")
        controls_group_layout.addWidget(controls_title)

        self.controls_layout = QFormLayout()
        self.controls_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        self.controls_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.controls_layout.setFormAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        self.controls_layout.setHorizontalSpacing(14)
        self.controls_layout.setVerticalSpacing(10)
        controls_group_layout.addLayout(self.controls_layout)

        self.level_filter = ComboBox()
        self.level_filter.addItem("All Levels")
        self.level_filter.addItems([level.value for level in LogLevel])
        self.level_filter.setMinimumWidth(200)
        self.level_filter.setToolTip("Filter log messages by severity level")
        self.controls_layout.addRow("Filter Level:", cast("QWidget", self.level_filter))

        self.auto_refresh_cb = CheckBox("Auto Refresh")
        self.auto_refresh_cb.setToolTip(
            "Automatically refresh logs on a short interval"
        )
        self.controls_layout.addRow(
            "Auto Refresh:", cast("QWidget", self.auto_refresh_cb)
        )

        self.max_lines_spin = SpinBox()
        self.max_lines_spin.setRange(100, 10000)
        self.max_lines_spin.setSingleStep(100)
        self.max_lines_spin.setValue(1000)
        self.max_lines_spin.setToolTip("Limit how many log lines are displayed at once")
        self.controls_layout.addRow("Max Lines:", cast("QWidget", self.max_lines_spin))

        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 8, 0, 0)
        buttons_layout.setSpacing(10)

        self.refresh_btn = PushButton("Refresh")
        self.refresh_btn.setMinimumWidth(120)
        self.refresh_btn.setToolTip("Fetch the latest log entries")
        self.clear_btn = PushButton("Clear Display")
        self.clear_btn.setToolTip("Remove all currently displayed log text")
        self.save_btn = PushButton("Save Logs...")
        self.save_btn.setToolTip("Export the current log output to a text file")

        buttons_layout.addWidget(cast("QWidget", self.refresh_btn))
        buttons_layout.addWidget(cast("QWidget", self.clear_btn))
        buttons_layout.addWidget(cast("QWidget", self.save_btn))
        buttons_layout.addStretch()
        controls_group_layout.addLayout(buttons_layout)

        # Log display card
        log_panel = QWidget()
        log_panel_layout = VBoxLayout(log_panel)
        log_panel_layout.setContentsMargins(0, 0, 0, 0)
        log_panel_layout.setSpacing(12)

        log_group = CardWidget()
        log_group.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        log_layout = VBoxLayout(log_group)
        log_layout.setContentsMargins(20, 16, 20, 20)
        log_layout.setSpacing(10)

        log_title = BodyLabel("Log Messages")
        log_title.setStyleSheet("font-weight: 600; font-size: 15px;")
        log_layout.addWidget(log_title)

        self.log_display = TextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMinimumHeight(220)

        font = QFont("Consolas", 9)
        if not font.exactMatch():
            font = QFont("Courier New", 9)
        self.log_display.setFont(font)

        self.log_display.setPlaceholderText(
            "Log messages will appear here...\n"
            "Click 'Refresh' to load recent logs or enable 'Auto Refresh'"
        )

        log_layout.addWidget(self.log_display, 1)

        self.status_label = BodyLabel("Ready — click 'Refresh' to load logs")
        self.status_label.setStyleSheet("color: palette(mid); margin-top: 4px;")

        log_panel_layout.addWidget(log_group, 1)
        log_panel_layout.addWidget(self.status_label)

        content_splitter.addWidget(cast("QWidget", controls_group))
        content_splitter.addWidget(cast("QWidget", log_panel))
        content_splitter.setStretchFactor(0, 0)
        content_splitter.setStretchFactor(1, 1)
        content_splitter.setSizes([220, 480])

        # Connect signals
        self._setup_connections()
        self._update_controls_layout_policy(self.width())

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

        # Try to read actual log files
        from pathlib import Path

        logs = []

        # Common log file locations to check
        log_locations = [
            Path("logs/vectorflow.log"),
            Path("vectorflow.log"),
            Path.home() / ".vectorflow" / "logs" / "app.log",
            Path("app.log"),
        ]

        for log_path in log_locations:
            if log_path.exists() and log_path.is_file():
                try:
                    with log_path.open(encoding="utf-8", errors="ignore") as f:
                        # Read last 50 lines
                        lines = f.readlines()
                        logs = [line.strip() for line in lines[-50:] if line.strip()]
                    break
                except Exception:
                    continue

        # If no log file found, use fallback messages
        if not logs:
            logs = [
                "2024-01-15 10:30:15 | INFO | vectorflow.pipeline | Pipeline initialized successfully",
                "2024-01-15 10:30:16 | DEBUG | vectorflow.config | Configuration loaded from defaults",
                "2024-01-15 10:30:17 | INFO | vectorflow.embeddings | Embedding service created: sentence-transformers",
                "2024-01-15 10:30:18 | INFO | vectorflow.vector_db | Vector database client initialized",
                "2024-01-15 10:30:19 | INFO | vectorflow.gui | Log widget initialized - no log file found",
            ]

        # Apply level filter
        filtered_logs = self._apply_level_filter(logs)

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

        self.status_label.setText(f"Showing {len(filtered_logs)} log entries")
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
            self, "Save Logs", "logs.txt", "Text Files (*.txt);;All Files (*)"
        )

        if filename:
            try:
                with Path(filename).open("w", encoding="utf-8") as f:
                    f.write(self.log_display.toPlainText())
                self.status_label.setText(f"Logs saved to {filename}")
                self.emit_status(f"Logs saved to {filename}")
            except Exception as e:
                self.emit_error(f"Failed to save logs: {e!s}")

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

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Handle widget resizing for adaptive layouts."""
        super().resizeEvent(event)
        if event is not None:
            self._update_controls_layout_policy(event.size().width())

    def _update_controls_layout_policy(self, width: int) -> None:
        """Adjust control layout wrapping based on available width."""
        if not self.controls_layout:
            return

        if width < 640:
            self.controls_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        else:
            self.controls_layout.setRowWrapPolicy(
                QFormLayout.RowWrapPolicy.DontWrapRows
            )
