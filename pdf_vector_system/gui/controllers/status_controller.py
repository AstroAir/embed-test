"""
Status controller for PDF Vector System GUI.

This module contains the controller for system status monitoring.
"""

import platform
import sys
from typing import Any, Optional

import psutil
from PySide6.QtCore import QObject, QTimer, Signal

from pdf_vector_system.core.config.settings import Config
from pdf_vector_system.core.pipeline import PDFVectorPipeline
from pdf_vector_system.gui.utils.threading import TaskRunner


class StatusController(QObject):
    """Controller for system status monitoring."""

    # Signals
    health_check_completed = Signal(dict)  # health_status
    system_info_updated = Signal(dict)  # system_info
    performance_metrics_updated = Signal(dict)  # metrics
    status_error = Signal(str)  # error_message
    status_message = Signal(str)  # status_message

    def __init__(
        self, config: Optional[Config] = None, parent: Optional[QObject] = None
    ):
        """
        Initialize the status controller.

        Args:
            config: Configuration object
            parent: Parent QObject
        """
        super().__init__(parent)

        self.config = config or Config()
        self.pipeline: Optional[PDFVectorPipeline] = None
        self.task_runner = TaskRunner(self)

        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.update_performance_metrics)

        # Connect task runner signals
        self.task_runner.task_started.connect(self._on_task_started)
        self.task_runner.task_finished.connect(self._on_task_finished)
        self.task_runner.task_error.connect(self._on_task_error)

        # Initialize pipeline
        self._initialize_pipeline()

    def _initialize_pipeline(self) -> None:
        """Initialize the PDF vector pipeline."""
        try:
            self.pipeline = PDFVectorPipeline(self.config)
            self.status_message.emit("Status monitoring ready")
        except Exception as e:
            self.status_error.emit(f"Failed to initialize pipeline: {e!s}")

    def run_health_check(self) -> str:
        """
        Run comprehensive health check in the background.

        Returns:
            Task ID for the health check operation
        """
        if not self.pipeline:
            self.status_error.emit("Pipeline not initialized")
            return ""

        # Start health check task
        return self.task_runner.run_task(self._health_check_task, "health_check")

    def _health_check_task(self) -> dict[str, bool]:
        """
        Background task for health checking.

        Returns:
            Dictionary with health status for each component
        """
        try:
            if self.pipeline is None:
                raise Exception("Pipeline not initialized")
            return self.pipeline.health_check()

        except Exception as e:
            raise Exception(f"Health check failed: {e!s}") from e

    def update_system_info(self) -> None:
        """Update system information."""
        try:
            # Get system information
            info = {
                "python_version": sys.version,
                "platform": platform.platform(),
                "architecture": platform.architecture()[0],
                "processor": platform.processor(),
                "hostname": platform.node(),
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            }

            # Add application-specific info
            try:
                import PySide6

                info["pyside6_version"] = PySide6.__version__
            except Exception:
                info["pyside6_version"] = "Unknown"

            info["application_version"] = "1.0.0"
            info["config_file"] = str(self.config.chroma_db.persist_directory)
            info["debug_mode"] = str(getattr(self.config, "debug", False))
            info["max_workers"] = str(getattr(self.config, "max_workers", 4))
            info["embedding_model"] = self.config.embedding.model_name
            info["collection_name"] = self.config.chroma_db.collection_name

            self.system_info_updated.emit(info)

        except Exception as e:
            self.status_error.emit(f"Failed to update system info: {e!s}")

    def update_performance_metrics(self) -> None:
        """Update performance metrics."""
        try:
            # Get performance metrics
            metrics = {}

            # Memory usage
            memory = psutil.virtual_memory()
            metrics["memory_total"] = memory.total
            metrics["memory_available"] = memory.available
            metrics["memory_used"] = memory.used
            metrics["memory_percent"] = memory.percent

            # CPU usage
            metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            metrics["cpu_count"] = psutil.cpu_count()

            # Disk usage (for the current directory)
            disk = psutil.disk_usage(".")
            metrics["disk_total"] = disk.total
            metrics["disk_used"] = disk.used
            metrics["disk_free"] = disk.free
            metrics["disk_percent"] = (disk.used / disk.total) * 100

            # Process-specific metrics
            process = psutil.Process()
            metrics["process_memory"] = process.memory_info().rss
            metrics["process_cpu"] = process.cpu_percent()
            metrics["process_threads"] = process.num_threads()

            self.performance_metrics_updated.emit(metrics)

        except Exception as e:
            self.status_error.emit(f"Failed to update performance metrics: {e!s}")

    def start_auto_refresh(self, interval_ms: int = 5000) -> None:
        """
        Start automatic performance metrics refresh.

        Args:
            interval_ms: Refresh interval in milliseconds
        """
        self.refresh_timer.start(interval_ms)
        self.status_message.emit(
            f"Auto-refresh started ({interval_ms / 1000:.1f}s interval)"
        )

    def stop_auto_refresh(self) -> None:
        """Stop automatic performance metrics refresh."""
        self.refresh_timer.stop()
        self.status_message.emit("Auto-refresh stopped")

    def is_auto_refreshing(self) -> bool:
        """
        Check if auto-refresh is active.

        Returns:
            True if auto-refresh is running
        """
        return self.refresh_timer.isActive()

    def get_collection_info(self) -> str:
        """
        Get collection information in the background.

        Returns:
            Task ID for the collection info operation
        """
        if not self.pipeline:
            self.status_error.emit("Pipeline not initialized")
            return ""

        # Start collection info task
        return self.task_runner.run_task(self._collection_info_task, "collection_info")

    def _collection_info_task(self) -> dict[str, Any]:
        """
        Background task for getting collection information.

        Returns:
            Dictionary with collection information
        """
        try:
            if self.pipeline is None:
                raise Exception("Pipeline not initialized")
            # Get collection statistics
            stats = self.pipeline.get_collection_stats()

            # Add additional collection info
            collection = self.pipeline.vector_db.get_collection()

            return {
                "collection_name": collection.name,
                "total_chunks": stats.get("total_chunks", 0),
                "unique_documents": stats.get("unique_documents", 0),
                "total_characters": stats.get("total_characters", 0),
                "average_chunk_size": stats.get("average_chunk_size", 0),
                "embedding_model": self.config.embedding.model_name,
                "distance_metric": self.config.chroma_db.distance_metric,
            }

        except Exception as e:
            raise Exception(f"Failed to get collection info: {e!s}") from e

    def update_config(self, new_config: Config) -> None:
        """
        Update configuration and reinitialize pipeline.

        Args:
            new_config: New configuration object
        """
        self.config = new_config
        self._initialize_pipeline()

    def _on_task_started(self, task_id: str) -> None:
        """Handle task started signal."""
        if task_id == "health_check":
            self.status_message.emit("Running health check...")
        elif task_id == "collection_info":
            self.status_message.emit("Getting collection info...")

    def _on_task_finished(self, task_id: str, result: Any) -> None:
        """Handle task finished signal."""
        if task_id == "health_check":
            self.health_check_completed.emit(result)
            healthy_count = sum(1 for status in result.values() if status)
            total_count = len(result)
            self.status_message.emit(
                f"Health check completed: {healthy_count}/{total_count} components healthy"
            )
        elif task_id == "collection_info":
            # Collection info can be handled by other components
            self.status_message.emit("Collection info retrieved")

    def _on_task_error(self, task_id: str, error: str) -> None:
        """Handle task error signal."""
        self.status_error.emit(error)
