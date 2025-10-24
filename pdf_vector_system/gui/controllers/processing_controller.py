"""
Processing controller for PDF Vector System GUI.

This module contains the controller for PDF processing operations.
"""

from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal

from pdf_vector_system.config.settings import Config
from pdf_vector_system.gui.utils.threading import TaskRunner
from pdf_vector_system.pipeline import PDFVectorPipeline


class ProcessingController(QObject):
    """Controller for PDF processing operations."""

    # Signals
    processing_started = Signal(list)  # file_paths
    processing_progress = Signal(int, int)  # current_file, total_files
    file_processed = Signal(str, bool, str)  # file_path, success, message
    processing_completed = Signal(int, int)  # successful_count, total_count
    processing_error = Signal(str)  # error_message
    status_message = Signal(str)  # status_message

    def __init__(
        self, config: Optional[Config] = None, parent: Optional[QObject] = None
    ):
        """
        Initialize the processing controller.

        Args:
            config: Configuration object
            parent: Parent QObject
        """
        super().__init__(parent)

        self.config = config or Config()
        self.pipeline: Optional[PDFVectorPipeline] = None
        self.task_runner = TaskRunner(self)

        # Connect task runner signals
        self.task_runner.task_started.connect(self._on_task_started)
        self.task_runner.task_finished.connect(self._on_task_finished)
        self.task_runner.task_error.connect(self._on_task_error)
        self.task_runner.task_progress.connect(self._on_task_progress)
        self.task_runner.task_status.connect(self._on_task_status)

        # Initialize pipeline
        self._initialize_pipeline()

    def _initialize_pipeline(self) -> None:
        """Initialize the PDF vector pipeline."""
        try:
            self.pipeline = PDFVectorPipeline(self.config)
            self.status_message.emit("Processing pipeline ready")
        except Exception as e:
            self.processing_error.emit(f"Failed to initialize pipeline: {e!s}")

    def process_files(self, file_paths: list[Path], clean_text: bool = True) -> str:
        """
        Process PDF files in the background.

        Args:
            file_paths: List of PDF file paths to process
            clean_text: Whether to clean extracted text

        Returns:
            Task ID for the processing operation
        """
        if not self.pipeline:
            self.processing_error.emit("Pipeline not initialized")
            return ""

        if not file_paths:
            self.processing_error.emit("No files to process")
            return ""

        # Start processing task
        return self.task_runner.run_task(
            self._process_files_task, "pdf_processing", file_paths, clean_text
        )

    def _process_files_task(
        self, file_paths: list[Path], clean_text: bool
    ) -> dict[str, Any]:
        """
        Background task for processing PDF files.

        Args:
            file_paths: List of PDF file paths to process
            clean_text: Whether to clean extracted text

        Returns:
            Processing results dictionary
        """
        results: dict[str, Any] = {
            "successful": 0,
            "failed": 0,
            "total": len(file_paths),
            "files": [],
        }

        self.processing_started.emit([str(f) for f in file_paths])

        for i, file_path in enumerate(file_paths):
            try:
                # Update progress
                self.processing_progress.emit(i + 1, len(file_paths))
                self.status_message.emit(f"Processing {file_path.name}...")

                # Process the file
                if self.pipeline is None:
                    raise Exception("Pipeline not initialized")
                result = self.pipeline.process_pdf(
                    pdf_path=file_path, clean_text=clean_text, show_progress=False
                )

                if result.success:
                    results["successful"] += 1
                    self.file_processed.emit(
                        str(file_path),
                        True,
                        f"Processed {result.chunks_processed} chunks",
                    )
                else:
                    results["failed"] += 1
                    self.file_processed.emit(
                        str(file_path),
                        False,
                        result.error_message or "Processing failed",
                    )

                results["files"].append(
                    {
                        "path": str(file_path),
                        "success": result.success,
                        "chunks": result.chunks_processed if result.success else 0,
                        "message": result.error_message or "Success",
                    }
                )

            except Exception as e:
                results["failed"] += 1
                error_msg = f"Error processing {file_path.name}: {e!s}"
                self.file_processed.emit(str(file_path), False, error_msg)
                results["files"].append(
                    {
                        "path": str(file_path),
                        "success": False,
                        "chunks": 0,
                        "message": str(e),
                    }
                )

        return results

    def stop_processing(self) -> bool:
        """
        Stop current processing operation.

        Returns:
            True if processing was stopped
        """
        return self.task_runner.stop_task("pdf_processing")

    def is_processing(self) -> bool:
        """
        Check if processing is currently running.

        Returns:
            True if processing is running
        """
        return self.task_runner.is_task_running("pdf_processing")

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
        if task_id == "pdf_processing":
            self.status_message.emit("PDF processing started")

    def _on_task_finished(self, task_id: str, result: dict) -> None:
        """Handle task finished signal."""
        if task_id == "pdf_processing":
            self.processing_completed.emit(result["successful"], result["total"])
            self.status_message.emit(
                f"Processing completed: {result['successful']}/{result['total']} files successful"
            )

    def _on_task_error(self, task_id: str, error: str) -> None:
        """Handle task error signal."""
        if task_id == "pdf_processing":
            self.processing_error.emit(f"Processing error: {error}")

    def _on_task_progress(self, task_id: str, progress: int) -> None:
        """Handle task progress signal."""
        if task_id == "pdf_processing":
            # Progress updates are handled in the task itself
            pass

    def _on_task_status(self, task_id: str, status: str) -> None:
        """Handle task status signal."""
        if task_id == "pdf_processing":
            self.status_message.emit(status)
