"""
System status widget for PDF Vector System GUI.

This module contains the widget for system status and health monitoring.
"""

from typing import Optional, Dict, Any

from PySide6.QtWidgets import QWidget, QHBoxLayout, QTableWidgetItem
from PySide6.QtCore import Qt, QTimer
from qfluentwidgets import (
    VBoxLayout, PushButton, BodyLabel,
    TableWidget, CardWidget,
    TextEdit, ProgressBar
)

from ...config.settings import Config
from .base import BaseWidget
from ..controllers.status_controller import StatusController


class StatusWidget(BaseWidget):
    """Widget for system status and health monitoring."""
    
    def __init__(self, config: Optional[Config] = None, parent: Optional[QWidget] = None):
        """
        Initialize the status widget.

        Args:
            config: Configuration object
            parent: Parent widget
        """
        super().__init__(config, parent)

        # Setup auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.auto_refresh)

        # Initialize controller after UI is set up
        self.controller = StatusController(self.config, self)
        self._connect_controller_signals()
        
    def _setup_ui(self) -> None:
        """Set up the user interface."""
        layout = VBoxLayout(self)

        # Controls
        controls_layout = QHBoxLayout()
        self.health_check_btn = PushButton("Run Health Check")
        self.refresh_btn = PushButton("Refresh Status")
        self.auto_refresh_btn = PushButton("Auto Refresh: OFF")
        self.auto_refresh_btn.setCheckable(True)

        controls_layout.addWidget(self.health_check_btn)
        controls_layout.addWidget(self.refresh_btn)
        controls_layout.addWidget(self.auto_refresh_btn)
        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        # System Health Status
        health_group = CardWidget()
        health_layout = VBoxLayout(health_group)

        # Add title for the card
        health_title = BodyLabel("System Health")
        health_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        health_layout.addWidget(health_title)

        self.health_table = TableWidget()
        self.health_table.setColumnCount(2)
        self.health_table.setHorizontalHeaderLabels(["Component", "Status"])
        self.health_table.setAlternatingRowColors(True)
        health_layout.addWidget(self.health_table)
        
        layout.addWidget(health_group)
        
        # System Information
        info_group = CardWidget()
        info_layout = VBoxLayout(info_group)

        # Add title for the card
        info_title = BodyLabel("System Information")
        info_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        info_layout.addWidget(info_title)

        self.info_table = TableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.info_table.setAlternatingRowColors(True)
        info_layout.addWidget(self.info_table)

        layout.addWidget(info_group)

        # Performance Metrics
        perf_group = CardWidget()
        perf_layout = VBoxLayout(perf_group)

        # Add title for the card
        perf_title = BodyLabel("Performance Metrics")
        perf_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        perf_layout.addWidget(perf_title)

        # Memory usage
        memory_layout = QHBoxLayout()
        memory_layout.addWidget(BodyLabel("Memory Usage:"))
        self.memory_bar = ProgressBar()
        self.memory_bar.setRange(0, 100)
        self.memory_label = BodyLabel("0%")
        memory_layout.addWidget(self.memory_bar)
        memory_layout.addWidget(self.memory_label)
        perf_layout.addLayout(memory_layout)

        # CPU usage
        cpu_layout = QHBoxLayout()
        cpu_layout.addWidget(BodyLabel("CPU Usage:"))
        self.cpu_bar = ProgressBar()
        self.cpu_bar.setRange(0, 100)
        self.cpu_label = BodyLabel("0%")
        cpu_layout.addWidget(self.cpu_bar)
        cpu_layout.addWidget(self.cpu_label)
        perf_layout.addLayout(cpu_layout)
        
        layout.addWidget(perf_group)
        
        # Status Messages
        messages_group = CardWidget()
        messages_layout = VBoxLayout(messages_group)

        # Add title for the card
        messages_title = BodyLabel("Status Messages")
        messages_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px;")
        messages_layout.addWidget(messages_title)

        self.messages_text = TextEdit()
        self.messages_text.setMaximumHeight(150)
        self.messages_text.setReadOnly(True)
        messages_layout.addWidget(self.messages_text)

        layout.addWidget(messages_group)
        
        # Connect signals
        self._setup_connections()

        # Initial status update
        self.update_status()

    def _connect_controller_signals(self) -> None:
        """Connect controller signals to widget slots."""
        self.controller.health_check_completed.connect(self._on_health_check_completed)
        self.controller.system_info_updated.connect(self._on_system_info_updated)
        self.controller.performance_metrics_updated.connect(self._on_performance_metrics_updated)
        self.controller.status_error.connect(self._on_status_error)
        self.controller.status_message.connect(self._on_status_message)

    def _setup_connections(self) -> None:
        """Set up signal/slot connections."""
        self.health_check_btn.clicked.connect(self.run_health_check)
        self.refresh_btn.clicked.connect(self.update_status)
        self.auto_refresh_btn.toggled.connect(self.toggle_auto_refresh)
        
    def run_health_check(self) -> None:
        """Run comprehensive health check."""
        # Check if controller is initialized before using it
        if not hasattr(self, 'controller') or self.controller is None:
            return

        self.controller.run_health_check()
        
    def update_status(self) -> None:
        """Update system status information."""
        # Check if controller is initialized before using it
        if not hasattr(self, 'controller') or self.controller is None:
            return

        # Update system information
        self.controller.update_system_info()

        # Update performance metrics
        self.controller.update_performance_metrics()
        
    def _update_health_table(self, health_data: Dict[str, str]) -> None:
        """Update the health status table."""
        self.health_table.setRowCount(len(health_data))
        
        for row, (component, status) in enumerate(health_data.items()):
            self.health_table.setItem(row, 0, QTableWidgetItem(component))
            self.health_table.setItem(row, 1, QTableWidgetItem(status))
            
        self.health_table.resizeColumnsToContents()
        
    def _update_info_table(self, info_data: Dict[str, str]) -> None:
        """Update the system information table."""
        self.info_table.setRowCount(len(info_data))
        
        for row, (prop, value) in enumerate(info_data.items()):
            self.info_table.setItem(row, 0, QTableWidgetItem(prop))
            self.info_table.setItem(row, 1, QTableWidgetItem(value))
            
        self.info_table.resizeColumnsToContents()
        
    def toggle_auto_refresh(self, enabled: bool) -> None:
        """Toggle auto-refresh functionality."""
        # Check if controller is initialized before using it
        if not hasattr(self, 'controller') or self.controller is None:
            return

        if enabled:
            self.controller.start_auto_refresh(5000)  # 5 second interval
            self.auto_refresh_btn.setText("Auto Refresh: ON")
            self.messages_text.append("Auto-refresh enabled (5 second interval)")
        else:
            self.controller.stop_auto_refresh()
            self.auto_refresh_btn.setText("Auto Refresh: OFF")
            self.messages_text.append("Auto-refresh disabled")

    def auto_refresh(self) -> None:
        """Perform automatic status refresh."""
        self.update_status()

    def _on_health_check_completed(self, health_status: Dict[str, bool]) -> None:
        """Handle health check completed signal."""
        # Convert boolean status to display strings
        health_data = {}
        for component, is_healthy in health_status.items():
            status = "✅ Healthy" if is_healthy else "❌ Unhealthy"
            health_data[component] = status

        self._update_health_table(health_data)
        self.messages_text.append("Health check completed")

    def _on_system_info_updated(self, info: Dict[str, Any]) -> None:
        """Handle system info updated signal."""
        # Convert info to display format
        info_data = {}
        for key, value in info.items():
            info_data[key.replace('_', ' ').title()] = str(value)

        self._update_info_table(info_data)

    def _on_performance_metrics_updated(self, metrics: Dict[str, Any]) -> None:
        """Handle performance metrics updated signal."""
        # Update memory usage
        memory_percent = metrics.get('memory_percent', 0)
        self.memory_bar.setValue(int(memory_percent))
        self.memory_label.setText(f"{memory_percent:.1f}%")

        # Update CPU usage
        cpu_percent = metrics.get('cpu_percent', 0)
        self.cpu_bar.setValue(int(cpu_percent))
        self.cpu_label.setText(f"{cpu_percent:.1f}%")

    def _on_status_error(self, error_message: str) -> None:
        """Handle status error signal."""
        self.messages_text.append(f"Error: {error_message}")
        self.emit_status(f"Status error: {error_message}")

    def _on_status_message(self, message: str) -> None:
        """Handle status message signal."""
        self.emit_status(message)

    def on_tab_activated(self) -> None:
        """Called when this tab is activated."""
        self.emit_status("System Status tab activated")
        self.update_status()
