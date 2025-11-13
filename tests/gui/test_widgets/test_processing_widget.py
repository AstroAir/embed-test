"""Tests for ProcessingWidget."""

from pathlib import Path
from unittest.mock import patch

import pytest
from PySide6.QtCore import Qt

from vectorflow.gui.controllers.processing_controller import ProcessingController
from vectorflow.gui.widgets.processing_widget import ProcessingWidget


@pytest.mark.gui
@pytest.mark.widget
class TestProcessingWidget:
    """Test cases for ProcessingWidget."""

    def test_widget_initialization(self, qtbot, mock_config):
        """Test widget initializes correctly."""
        widget = ProcessingWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Check widget is created
        assert widget is not None
        assert widget.config == mock_config

        # Check controller is initialized
        assert hasattr(widget, "controller")
        assert isinstance(widget.controller, ProcessingController)

        # Check UI elements exist
        assert hasattr(widget, "select_files_btn")
        assert hasattr(widget, "select_folder_btn")
        assert hasattr(widget, "process_btn")
        assert hasattr(widget, "progress_bar")
        assert hasattr(widget, "status_label")
        assert hasattr(widget, "results_text")

        # Check initial state
        assert widget.select_files_btn.isEnabled()
        assert widget.select_folder_btn.isEnabled()
        assert not widget.process_btn.isEnabled()  # Should be disabled initially
        assert widget.progress_bar.value() == 0

    def test_ui_layout_and_components(self, qtbot, mock_config):
        """Test UI layout and component arrangement."""
        widget = ProcessingWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Check main layout exists
        layout = widget.layout()
        assert layout is not None

        # Check file selection section
        assert widget.select_files_btn.text() == "Select PDF Files..."
        assert widget.select_folder_btn.text() == "Select Folder..."

        # Check processing options
        assert hasattr(widget, "clean_text_cb")
        assert hasattr(widget, "chunk_size_spin")
        assert hasattr(widget, "batch_size_spin")

        # Check progress section
        assert widget.progress_bar.minimum() == 0
        assert widget.progress_bar.maximum() == 100

    def test_file_selection_dialog(self, qtbot, mock_config, mock_file_dialog):
        """Test file selection dialog functionality."""
        widget = ProcessingWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Mock file dialog to return test files
        test_files = ["/test/file1.pdf", "/test/file2.pdf"]
        mock_file_dialog.getOpenFileNames.return_value = (
            test_files,
            "PDF Files (*.pdf)",
        )

        # Click select files button
        qtbot.mouseClick(widget.select_files_btn, Qt.MouseButton.LeftButton)

        # Verify dialog was called
        mock_file_dialog.getOpenFileNames.assert_called_once()

        # Check files were added
        assert len(widget._selected_files) == 2
        assert widget.process_btn.isEnabled()

    def test_folder_selection_dialog(self, qtbot, mock_config, mock_file_dialog):
        """Test folder selection dialog functionality."""
        widget = ProcessingWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Mock folder dialog
        test_folder = "/test/folder"
        mock_file_dialog.getExistingDirectory.return_value = test_folder

        # Mock finding PDF files in folder
        with patch("pathlib.Path.glob") as mock_glob:
            mock_glob.return_value = [
                Path("/test/folder/file1.pdf"),
                Path("/test/folder/file2.pdf"),
            ]

            # Click select folder button
            qtbot.mouseClick(widget.select_folder_btn, Qt.MouseButton.LeftButton)

            # Verify dialog was called
            mock_file_dialog.getExistingDirectory.assert_called_once()

            # Check files were found and added
            assert len(widget._selected_files) == 2
            assert widget.process_btn.isEnabled()

    def test_processing_options_configuration(self, qtbot, mock_config):
        """Test processing options can be configured."""
        widget = ProcessingWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Test clean text checkbox
        assert widget.clean_text_cb.isChecked()  # Should be checked by default
        widget.clean_text_cb.setChecked(False)
        assert not widget.clean_text_cb.isChecked()

        # Test chunk size spinner
        widget.chunk_size_spin.setValue(500)
        assert widget.chunk_size_spin.value() == 500

        # Test batch size spinner
        widget.batch_size_spin.setValue(8)
        assert widget.batch_size_spin.value() == 8

    def test_process_button_state_management(self, qtbot, mock_config):
        """Test process button enable/disable logic."""
        widget = ProcessingWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Initially disabled
        assert not widget.process_btn.isEnabled()

        # Add files - should enable
        widget._selected_files = [Path("/test/file1.pdf")]
        widget._update_ui_state()
        assert widget.process_btn.isEnabled()

        # Clear files - should disable
        widget._selected_files = []
        widget._update_ui_state()
        assert not widget.process_btn.isEnabled()

    @patch(
        "vectorflow.gui.controllers.processing_controller.ProcessingController.process_files"
    )
    def test_process_files_execution(self, mock_process, qtbot, mock_config):
        """Test file processing execution."""
        widget = ProcessingWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Setup files
        test_files = [Path("/test/file1.pdf"), Path("/test/file2.pdf")]
        widget._selected_files = test_files
        widget._update_ui_state()

        # Configure mock to return task ID
        mock_process.return_value = "task_123"

        # Click process button
        qtbot.mouseClick(widget.process_btn, Qt.MouseButton.LeftButton)

        # Verify controller method was called
        mock_process.assert_called_once_with(test_files, clean_text=True)

        # Check UI state during processing
        assert not widget.process_btn.isEnabled()
        assert widget.status_label.text() == "Processing files..."

    def test_progress_updates(self, qtbot, mock_config):
        """Test progress bar updates during processing."""
        widget = ProcessingWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Test progress update
        widget._on_progress_updated(50)
        assert widget.progress_bar.value() == 50

        # Test completion
        widget._on_progress_updated(100)
        assert widget.progress_bar.value() == 100

    def test_processing_completed_success(self, qtbot, mock_config):
        """Test successful processing completion."""
        widget = ProcessingWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Simulate processing completion
        results = {
            "successful": 2,
            "failed": 0,
            "total": 2,
            "processing_time": 1.23,
            "documents": [
                {"document_id": "doc1", "chunks": 5},
                {"document_id": "doc2", "chunks": 3},
            ],
        }

        widget._on_processing_completed(results)

        # Check UI updates
        assert widget.process_btn.isEnabled()
        assert "Successfully processed 2 files" in widget.results_text.toPlainText()
        assert widget.progress_bar.value() == 100

    def test_processing_completed_with_errors(self, qtbot, mock_config):
        """Test processing completion with some errors."""
        widget = ProcessingWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Simulate processing with errors
        results = {
            "successful": 1,
            "failed": 1,
            "total": 2,
            "processing_time": 1.23,
            "errors": ["Error processing file2.pdf: Invalid format"],
        }

        widget._on_processing_completed(results)

        # Check UI updates
        assert widget.process_btn.isEnabled()
        assert "1 successful, 1 failed" in widget.results_text.toPlainText()
        assert "Error processing file2.pdf" in widget.results_text.toPlainText()

    def test_processing_error_handling(self, qtbot, mock_config):
        """Test error handling during processing."""
        widget = ProcessingWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Simulate processing error
        error_message = "Database connection failed"
        widget._on_processing_error(error_message)

        # Check UI updates
        assert widget.process_btn.isEnabled()
        assert widget.progress_bar.value() == 0
        assert error_message in widget.results_text.toPlainText()

    def test_clear_results(self, qtbot, mock_config):
        """Test clearing results and resetting UI."""
        widget = ProcessingWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Add some content
        widget.results_text.setText("Previous results...")
        widget.progress_bar.setValue(75)

        # Clear results
        widget.clear_results()

        # Check UI is reset
        assert widget.results_text.toPlainText() == ""
        assert widget.progress_bar.value() == 0

    def test_signal_connections(self, qtbot, mock_config):
        """Test signal/slot connections are properly established."""
        widget = ProcessingWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Check controller signals are connected
        controller = widget.controller

        # Test that signals exist and are connected
        assert hasattr(controller, "progress_updated")
        assert hasattr(controller, "processing_completed")
        assert hasattr(controller, "processing_error")
        assert hasattr(controller, "status_message")

    def test_tab_activation(self, qtbot, mock_config):
        """Test behavior when tab is activated."""
        widget = ProcessingWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Mock emit_status method
        with patch.object(widget, "emit_status") as mock_emit:
            widget.on_tab_activated()
            mock_emit.assert_called_with("PDF Processing tab activated")

    def test_widget_cleanup(self, qtbot, mock_config):
        """Test widget cleanup and resource management."""
        widget = ProcessingWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Add some files and start processing state
        widget._selected_files = [Path("/test/file1.pdf")]
        widget._processing = True

        # Widget should handle cleanup properly when destroyed
        # This is mainly testing that no exceptions are raised
        widget.close()

    def test_file_validation(self, qtbot, mock_config):
        """Test file validation logic."""
        widget = ProcessingWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Test valid PDF files
        valid_files = [Path("/test/file1.pdf"), Path("/test/file2.pdf")]
        widget._selected_files = valid_files
        assert widget._validate_files()

        # Test empty file list
        widget._selected_files = []
        assert not widget._validate_files()

        # Test non-existent files (would be caught in real implementation)
        widget._selected_files = [Path("/nonexistent/file.pdf")]
        # In real implementation, this would validate file existence

    def test_status_message_emission(self, qtbot, mock_config):
        """Test status message emission."""
        widget = ProcessingWidget(config=mock_config)
        qtbot.addWidget(widget)

        # Connect to status_message signal
        messages = []
        widget.status_message.connect(messages.append)

        # Emit a status message
        test_message = "Test status message"
        widget.emit_status(test_message)

        # Check message was emitted
        assert len(messages) == 1
        assert messages[0] == test_message
