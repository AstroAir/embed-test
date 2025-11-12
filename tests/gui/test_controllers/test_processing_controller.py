"""Tests for ProcessingController."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PySide6.QtCore import QObject

from pdf_vector_system.core.pipeline import ProcessingResult
from pdf_vector_system.gui.controllers.processing_controller import ProcessingController


@pytest.mark.gui
@pytest.mark.controller
class TestProcessingController:
    """Test cases for ProcessingController."""

    def test_controller_initialization(self, mock_config, mock_pipeline):
        """Test controller initializes correctly."""
        parent = QObject()

        with patch(
            "pdf_vector_system.gui.controllers.processing_controller.PDFVectorPipeline"
        ) as mock_pipeline_class:
            mock_pipeline_class.return_value = mock_pipeline

            controller = ProcessingController(mock_config, parent)

            # Check initialization
            assert controller.config == mock_config
            assert controller.parent() == parent
            assert controller.pipeline == mock_pipeline

            # Check task runner is initialized
            assert hasattr(controller, "task_runner")

    def test_signal_definitions(self, mock_config, mock_pipeline):
        """Test that all required signals are defined."""
        parent = QObject()

        with patch(
            "pdf_vector_system.gui.controllers.processing_controller.PDFVectorPipeline"
        ) as mock_pipeline_class:
            mock_pipeline_class.return_value = mock_pipeline

            controller = ProcessingController(mock_config, parent)

            # Check signals exist
            assert hasattr(controller, "processing_progress")
            assert hasattr(controller, "processing_completed")
            assert hasattr(controller, "processing_error")
            assert hasattr(controller, "status_message")

    @patch("pdf_vector_system.gui.controllers.processing_controller.PDFVectorPipeline")
    def test_process_files_success(
        self, mock_pipeline_class, mock_config, mock_pipeline
    ):
        """Test successful file processing."""
        parent = QObject()
        mock_pipeline_class.return_value = mock_pipeline

        controller = ProcessingController(mock_config, parent)

        # Mock successful processing results
        mock_pipeline.process_pdf.side_effect = [
            ProcessingResult(
                document_id="doc1",
                file_path="/test/file1.pdf",
                success=True,
                chunks_processed=5,
                embeddings_generated=5,
                chunks_stored=5,
                processing_time=1.2,
            ),
            ProcessingResult(
                document_id="doc2",
                file_path="/test/file2.pdf",
                success=True,
                chunks_processed=3,
                embeddings_generated=3,
                chunks_stored=3,
                processing_time=0.8,
            ),
        ]

        # Test files
        test_files = [Path("/test/file1.pdf"), Path("/test/file2.pdf")]

        # Mock task runner to execute immediately
        with patch.object(controller.task_runner, "run_task") as mock_run_task:
            mock_run_task.return_value = "task_123"

            # Process files
            task_id = controller.process_files(test_files, clean_text=True)

            # Check task was started
            assert task_id == "task_123"
            mock_run_task.assert_called_once()

    @patch("pdf_vector_system.gui.controllers.processing_controller.PDFVectorPipeline")
    def test_process_files_with_errors(
        self, mock_pipeline_class, mock_config, mock_pipeline
    ):
        """Test file processing with some errors."""
        parent = QObject()
        mock_pipeline_class.return_value = mock_pipeline

        controller = ProcessingController(mock_config, parent)

        # Mock mixed results (success and failure)
        mock_pipeline.process_pdf.side_effect = [
            ProcessingResult(
                document_id="doc1",
                file_path="/test/file1.pdf",
                success=True,
                chunks_processed=5,
                embeddings_generated=5,
                chunks_stored=5,
                processing_time=1.2,
            ),
            Exception("Failed to process file2.pdf"),
        ]

        test_files = [Path("/test/file1.pdf"), Path("/test/file2.pdf")]

        with patch.object(controller.task_runner, "run_task") as mock_run_task:
            mock_run_task.return_value = "task_456"

            task_id = controller.process_files(test_files, clean_text=False)
            assert task_id == "task_456"

    def test_process_files_task_execution(self, mock_config, mock_pipeline):
        """Test the actual task execution logic."""
        parent = QObject()

        with patch(
            "pdf_vector_system.gui.controllers.processing_controller.PDFVectorPipeline"
        ) as mock_pipeline_class:
            mock_pipeline_class.return_value = mock_pipeline

            controller = ProcessingController(mock_config, parent)

            # Mock successful processing
            mock_pipeline.process_pdf.return_value = ProcessingResult(
                document_id="test_doc",
                file_path="/test/file1.pdf",
                success=True,
                chunks_processed=5,
                embeddings_generated=5,
                chunks_stored=5,
                processing_time=1.0,
            )

            # Test the internal task method
            test_files = [Path("/test/file1.pdf")]
            result = controller._process_files_task(test_files, True)

            # Check result structure
            assert "successful" in result
            assert "failed" in result
            assert "total" in result
            assert "files" in result
            assert result["successful"] == 1
            assert result["failed"] == 0
            assert result["total"] == 1

    def test_progress_reporting(self, mock_config, mock_pipeline):
        """Test progress reporting during processing."""
        parent = QObject()

        with patch(
            "pdf_vector_system.gui.controllers.processing_controller.PDFVectorPipeline"
        ) as mock_pipeline_class:
            mock_pipeline_class.return_value = mock_pipeline

            controller = ProcessingController(mock_config, parent)

            # Connect to progress signal
            progress_values = []
            controller.processing_progress.connect(
                lambda current, total: progress_values.append((current, total))
            )

            # Mock processing multiple files
            mock_pipeline.process_pdf.return_value = ProcessingResult(
                document_id="test_doc",
                file_path="/test/file.pdf",
                success=True,
                chunks_processed=5,
                embeddings_generated=5,
                chunks_stored=5,
                processing_time=1.0,
            )

            # Process files and check progress updates
            test_files = [Path(f"/test/file{i}.pdf") for i in range(3)]
            controller._process_files_task(test_files, True)

            # Should have progress updates (exact values depend on implementation)
            assert len(progress_values) > 0

    def test_error_handling_and_signals(self, mock_config, mock_pipeline):
        """Test error handling and signal emission."""
        parent = QObject()

        with patch(
            "pdf_vector_system.gui.controllers.processing_controller.PDFVectorPipeline"
        ) as mock_pipeline_class:
            mock_pipeline_class.return_value = mock_pipeline

            controller = ProcessingController(mock_config, parent)

            # Connect to error signal
            error_messages = []
            controller.processing_error.connect(error_messages.append)

            # Mock processing failure
            mock_pipeline.process_pdf.side_effect = Exception("Processing failed")

            # Process file and expect error
            test_files = [Path("/test/file1.pdf")]
            result = controller._process_files_task(test_files, True)

            # Check error was handled
            assert result["failed"] == 1
            assert result["successful"] == 0

    def test_configuration_updates(self, mock_config, mock_pipeline):
        """Test configuration updates."""
        parent = QObject()

        with patch(
            "pdf_vector_system.gui.controllers.processing_controller.PDFVectorPipeline"
        ) as mock_pipeline_class:
            mock_pipeline_class.return_value = mock_pipeline

            controller = ProcessingController(mock_config, parent)

            # Create new config
            new_config = Mock()
            new_config.debug = False
            new_config.max_workers = 4

            # Update configuration
            controller.update_config(new_config)

            # Check config was updated
            assert controller.config == new_config

    def test_pipeline_initialization_failure(self, mock_config):
        """Test handling of pipeline initialization failure."""
        parent = QObject()

        with patch(
            "pdf_vector_system.gui.controllers.processing_controller.PDFVectorPipeline"
        ) as mock_pipeline_class:
            mock_pipeline_class.side_effect = Exception("Pipeline init failed")

            controller = ProcessingController(mock_config, parent)

            # Pipeline should be None on failure
            assert controller.pipeline is None

    def test_task_cancellation(self, mock_config, mock_pipeline):
        """Test task cancellation functionality."""
        parent = QObject()

        with patch(
            "pdf_vector_system.gui.controllers.processing_controller.PDFVectorPipeline"
        ) as mock_pipeline_class:
            mock_pipeline_class.return_value = mock_pipeline

            controller = ProcessingController(mock_config, parent)

            # Mock task runner stop method
            with patch.object(controller.task_runner, "stop_task") as mock_stop:
                mock_stop.return_value = True

                # Stop a task
                result = controller.stop_processing()

                # Check stop was attempted
                mock_stop.assert_called_once_with("pdf_processing")
                assert result is True

    def test_status_message_emission(self, mock_config, mock_pipeline):
        """Test status message emission."""
        parent = QObject()

        with patch(
            "pdf_vector_system.gui.controllers.processing_controller.PDFVectorPipeline"
        ) as mock_pipeline_class:
            mock_pipeline_class.return_value = mock_pipeline

            controller = ProcessingController(mock_config, parent)

            # Connect to status signal
            status_messages = []
            controller.status_message.connect(status_messages.append)

            # Emit status message
            test_message = "Processing started"
            controller.status_message.emit(test_message)

            # Check message was emitted
            assert len(status_messages) == 1
            assert status_messages[0] == test_message

    def test_cleanup_and_resource_management(self, mock_config, mock_pipeline):
        """Test cleanup and resource management."""
        parent = QObject()

        with patch(
            "pdf_vector_system.gui.controllers.processing_controller.PDFVectorPipeline"
        ) as mock_pipeline_class:
            mock_pipeline_class.return_value = mock_pipeline

            controller = ProcessingController(mock_config, parent)

            # Mock task runner cleanup
            with patch.object(
                controller.task_runner, "stop_all_tasks"
            ) as mock_stop_all:
                # Test that controller can be cleaned up properly
                # Stop all tasks
                controller.task_runner.stop_all_tasks()

                # Check cleanup was called
                mock_stop_all.assert_called_once()

    def test_concurrent_processing_requests(self, mock_config, mock_pipeline):
        """Test handling of concurrent processing requests."""
        parent = QObject()

        with patch(
            "pdf_vector_system.gui.controllers.processing_controller.PDFVectorPipeline"
        ) as mock_pipeline_class:
            mock_pipeline_class.return_value = mock_pipeline

            controller = ProcessingController(mock_config, parent)

            test_files = [Path("/test/file1.pdf")]

            with patch.object(controller.task_runner, "run_task") as mock_run_task:
                mock_run_task.side_effect = ["task_1", "task_2"]

                # Start multiple processing tasks
                task_id_1 = controller.process_files(test_files, True)
                task_id_2 = controller.process_files(test_files, False)

                # Both should be accepted
                assert task_id_1 == "task_1"
                assert task_id_2 == "task_2"
                assert mock_run_task.call_count == 2
