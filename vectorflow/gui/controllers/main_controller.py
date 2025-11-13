"""
Main controller for PDF Vector System GUI.

This module contains the main controller that coordinates between
GUI components and business logic.
"""

from typing import Optional

from PySide6.QtCore import QObject, Signal

from vectorflow.core.config.settings import Config
from vectorflow.core.pipeline import PDFVectorPipeline


class MainController(QObject):
    """Main controller for the GUI application."""

    # Signals
    status_message = Signal(str)
    error_occurred = Signal(str)
    pipeline_ready = Signal()

    def __init__(
        self, config: Optional[Config] = None, parent: Optional[QObject] = None
    ):
        """
        Initialize the main controller.

        Args:
            config: Configuration object
            parent: Parent QObject
        """
        super().__init__(parent)

        self.config = config or Config()
        self._pipeline: Optional[PDFVectorPipeline] = None

        # Initialize pipeline
        self._initialize_pipeline()

    def _initialize_pipeline(self) -> None:
        """Initialize the PDF vector pipeline."""
        try:
            self._pipeline = PDFVectorPipeline(self.config)
            self.status_message.emit("Pipeline initialized successfully")
            self.pipeline_ready.emit()

        except Exception as e:
            error_msg = f"Failed to initialize pipeline: {e!s}"
            self.error_occurred.emit(error_msg)

    @property
    def pipeline(self) -> Optional[PDFVectorPipeline]:
        """Get the PDF vector pipeline."""
        return self._pipeline

    def update_config(self, new_config: Config) -> None:
        """
        Update the configuration and reinitialize pipeline.

        Args:
            new_config: New configuration object
        """
        self.config = new_config
        self._initialize_pipeline()

    def get_status(self) -> str:
        """
        Get current system status.

        Returns:
            Status message string
        """
        if self._pipeline:
            return "Ready"
        return "Pipeline not initialized"
