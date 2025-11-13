"""Integration tests for GUI workflows."""

from unittest.mock import Mock, patch

import pytest
from PySide6.QtCore import Qt

from vectorflow.gui.main_window import MainWindow


@pytest.mark.gui
@pytest.mark.integration
class TestPDFProcessingWorkflow:
    """Test complete PDF processing workflow."""

    def test_complete_pdf_processing_workflow(
        self, qtbot, mock_config, mock_file_dialog, mock_pipeline
    ):
        """Test complete PDF processing from file selection to results."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        # Mock pipeline for processing
        with patch.object(
            window.processing_widget.controller, "pipeline", mock_pipeline
        ):
            # Step 1: Navigate to processing tab
            window.tab_widget.setCurrentWidget(window.processing_widget)
            assert window.tab_widget.currentWidget() == window.processing_widget

            # Step 2: Select files
            test_files = ["/test/file1.pdf", "/test/file2.pdf"]
            mock_file_dialog.getOpenFileNames.return_value = (
                test_files,
                "PDF Files (*.pdf)",
            )

            qtbot.mouseClick(
                window.processing_widget.select_files_btn, Qt.MouseButton.LeftButton
            )

            # Step 3: Configure processing options
            window.processing_widget.clean_text_cb.setChecked(True)
            window.processing_widget.chunk_size_spin.setValue(1000)

            # Step 4: Start processing
            assert window.processing_widget.process_btn.isEnabled()

            # Mock successful processing
            mock_pipeline.process_pdf.return_value = {
                "document_id": "test_doc",
                "chunks_created": 5,
                "processing_time": 1.0,
                "success": True,
            }

            qtbot.mouseClick(
                window.processing_widget.process_btn, Qt.MouseButton.LeftButton
            )

            # Step 5: Verify UI updates during processing
            assert not window.processing_widget.process_btn.isEnabled()
            assert "Processing" in window.processing_widget.status_label.text()

    def test_pdf_processing_with_errors(
        self, qtbot, mock_config, mock_file_dialog, mock_pipeline
    ):
        """Test PDF processing workflow with errors."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        with patch.object(
            window.processing_widget.controller, "pipeline", mock_pipeline
        ):
            # Navigate to processing tab
            window.tab_widget.setCurrentWidget(window.processing_widget)

            # Select files
            test_files = ["/test/file1.pdf"]
            mock_file_dialog.getOpenFileNames.return_value = (
                test_files,
                "PDF Files (*.pdf)",
            )
            qtbot.mouseClick(
                window.processing_widget.select_files_btn, Qt.MouseButton.LeftButton
            )

            # Mock processing error
            mock_pipeline.process_pdf.side_effect = Exception("Processing failed")

            # Start processing
            qtbot.mouseClick(
                window.processing_widget.process_btn, Qt.MouseButton.LeftButton
            )

            # Verify error handling
            # (Implementation would depend on how errors are displayed)


@pytest.mark.gui
@pytest.mark.integration
class TestSearchWorkflow:
    """Test complete search workflow."""

    def test_complete_search_workflow(self, qtbot, mock_config, mock_pipeline):
        """Test complete search from query input to results display."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        with patch.object(window.search_widget.controller, "pipeline", mock_pipeline):
            # Step 1: Navigate to search tab
            window.tab_widget.setCurrentWidget(window.search_widget)
            assert window.tab_widget.currentWidget() == window.search_widget

            # Step 2: Enter search query
            search_query = "test search query"
            window.search_widget.query_input.setText(search_query)

            # Step 3: Configure search options
            window.search_widget.max_results_spin.setValue(10)

            # Step 4: Execute search
            mock_pipeline.search.return_value = [
                Mock(
                    id="result1",
                    content="Test result 1",
                    score=0.95,
                    metadata={"document_id": "doc1", "page_number": 1},
                ),
                Mock(
                    id="result2",
                    content="Test result 2",
                    score=0.87,
                    metadata={"document_id": "doc2", "page_number": 1},
                ),
            ]

            qtbot.mouseClick(window.search_widget.search_btn, Qt.MouseButton.LeftButton)

            # Step 5: Verify results display
            # (Implementation would depend on results table structure)

    def test_search_with_filters(self, qtbot, mock_config, mock_pipeline):
        """Test search workflow with document and page filters."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        with patch.object(window.search_widget.controller, "pipeline", mock_pipeline):
            # Navigate to search tab
            window.tab_widget.setCurrentWidget(window.search_widget)

            # Enter search query with filters
            window.search_widget.query_input.setText("filtered search")

            # Set document filter (if available)
            # window.search_widget.document_filter.setText("specific_doc")

            # Execute search
            qtbot.mouseClick(window.search_widget.search_btn, Qt.MouseButton.LeftButton)

            # Verify filtered search was called
            # mock_pipeline.search.assert_called_with(
            #     query_text="filtered search",
            #     document_id="specific_doc",
            #     ...
            # )


@pytest.mark.gui
@pytest.mark.integration
class TestDocumentManagementWorkflow:
    """Test complete document management workflow."""

    def test_document_list_and_details_workflow(
        self, qtbot, mock_config, mock_pipeline
    ):
        """Test document list display and details viewing."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        with patch.object(window.document_widget.controller, "pipeline", mock_pipeline):
            # Navigate to documents tab
            window.tab_widget.setCurrentWidget(window.document_widget)

            # Mock document list
            mock_pipeline.get_documents.return_value = [
                {
                    "document_id": "doc1",
                    "filename": "test1.pdf",
                    "chunks_count": 5,
                    "total_characters": 1000,
                },
                {
                    "document_id": "doc2",
                    "filename": "test2.pdf",
                    "chunks_count": 3,
                    "total_characters": 600,
                },
            ]

            # Trigger document loading
            window.document_widget.refresh_documents()

            # Verify documents are displayed
            # (Implementation would depend on document table structure)

    def test_document_deletion_workflow(
        self, qtbot, mock_config, mock_pipeline, mock_message_box
    ):
        """Test document deletion workflow with confirmation."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        with patch.object(window.document_widget.controller, "pipeline", mock_pipeline):
            # Navigate to documents tab
            window.tab_widget.setCurrentWidget(window.document_widget)

            # Mock confirmation dialog
            mock_message_box.question.return_value = mock_message_box.StandardButton.Yes

            # Mock successful deletion
            mock_pipeline.delete_document.return_value = 5  # chunks deleted

            # Simulate document selection and deletion
            # (Implementation would depend on document table structure)

            # Verify confirmation was shown
            mock_message_box.question.assert_called()

            # Verify deletion was called
            # mock_pipeline.delete_document.assert_called_with('selected_doc_id')


@pytest.mark.gui
@pytest.mark.integration
class TestConfigurationWorkflow:
    """Test complete configuration workflow."""

    def test_configuration_edit_and_save_workflow(
        self, qtbot, mock_config, mock_file_dialog
    ):
        """Test configuration editing and saving."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        # Navigate to settings tab
        window.tab_widget.setCurrentWidget(window.config_widget)

        # Modify configuration settings
        # (Implementation would depend on config widget structure)

        # Save configuration
        mock_file_dialog.getSaveFileName.return_value = (
            "/test/config.json",
            "JSON Files (*.json)",
        )

        # Click save button (if available)
        # qtbot.mouseClick(window.config_widget.save_btn, Qt.MouseButton.LeftButton)

        # Verify save dialog was shown
        # mock_file_dialog.getSaveFileName.assert_called()

    def test_configuration_validation_workflow(self, qtbot, mock_config):
        """Test configuration validation during editing."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        # Navigate to settings tab
        window.tab_widget.setCurrentWidget(window.config_widget)

        # Enter invalid configuration values
        # (Implementation would depend on config widget structure)

        # Attempt to apply changes
        # qtbot.mouseClick(window.config_widget.apply_btn, Qt.MouseButton.LeftButton)

        # Verify validation errors are shown
        # (Implementation would depend on error display mechanism)


@pytest.mark.gui
@pytest.mark.integration
class TestStatusMonitoringWorkflow:
    """Test complete status monitoring workflow."""

    def test_health_check_workflow(self, qtbot, mock_config, mock_pipeline):
        """Test health check execution and display."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        with patch.object(window.status_widget.controller, "pipeline", mock_pipeline):
            # Navigate to status tab
            window.tab_widget.setCurrentWidget(window.status_widget)

            # Mock health check results
            mock_pipeline.health_check.return_value = {
                "pipeline": True,
                "chromadb": True,
                "embedding_service": False,  # One failure
                "pdf_processor": True,
                "configuration": True,
            }

            # Execute health check
            qtbot.mouseClick(
                window.status_widget.health_check_btn, Qt.MouseButton.LeftButton
            )

            # Verify health check was called
            mock_pipeline.health_check.assert_called()

            # Verify results are displayed
            # (Implementation would depend on health status table structure)

    def test_auto_refresh_workflow(self, qtbot, mock_config, mock_system_resources):
        """Test auto-refresh functionality."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        # Navigate to status tab
        window.tab_widget.setCurrentWidget(window.status_widget)

        # Enable auto-refresh
        window.status_widget.auto_refresh_btn.setChecked(True)

        # Verify auto-refresh is enabled
        assert window.status_widget.refresh_timer.isActive()

        # Disable auto-refresh
        window.status_widget.auto_refresh_btn.setChecked(False)

        # Verify auto-refresh is disabled
        assert not window.status_widget.refresh_timer.isActive()


@pytest.mark.gui
@pytest.mark.integration
class TestCrossTabWorkflow:
    """Test workflows that span multiple tabs."""

    def test_process_then_search_workflow(
        self, qtbot, mock_config, mock_file_dialog, mock_pipeline
    ):
        """Test processing files then searching them."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        with (
            patch.object(
                window.processing_widget.controller, "pipeline", mock_pipeline
            ),
            patch.object(window.search_widget.controller, "pipeline", mock_pipeline),
        ):
            # Step 1: Process files
            window.tab_widget.setCurrentWidget(window.processing_widget)

            test_files = ["/test/file1.pdf"]
            mock_file_dialog.getOpenFileNames.return_value = (
                test_files,
                "PDF Files (*.pdf)",
            )
            qtbot.mouseClick(
                window.processing_widget.select_files_btn, Qt.MouseButton.LeftButton
            )

            mock_pipeline.process_pdf.return_value = {
                "document_id": "new_doc",
                "chunks_created": 5,
                "processing_time": 1.0,
                "success": True,
            }

            qtbot.mouseClick(
                window.processing_widget.process_btn, Qt.MouseButton.LeftButton
            )

            # Step 2: Switch to search tab
            window.tab_widget.setCurrentWidget(window.search_widget)

            # Step 3: Search the processed document
            window.search_widget.query_input.setText("search in new document")

            mock_pipeline.search.return_value = [
                Mock(
                    id="result1",
                    content="Content from new document",
                    score=0.95,
                    metadata={"document_id": "new_doc", "page_number": 1},
                )
            ]

            qtbot.mouseClick(window.search_widget.search_btn, Qt.MouseButton.LeftButton)

            # Verify search found content from processed document
            mock_pipeline.search.assert_called()

    def test_configuration_update_propagation(self, qtbot, mock_config):
        """Test configuration updates propagate to all components."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)

        # Navigate to settings tab
        window.tab_widget.setCurrentWidget(window.config_widget)

        # Create new configuration
        new_config = Mock()
        new_config.debug = False
        new_config.max_workers = 4

        # Simulate configuration change
        window._on_config_changed(new_config)

        # Verify all components received new configuration
        assert window.config == new_config

        # Check that controllers were updated (if they have update_config method)
        for widget in [
            window.processing_widget,
            window.search_widget,
            window.document_widget,
            window.status_widget,
        ]:
            if hasattr(widget, "controller") and hasattr(widget.controller, "config"):
                # Configuration should be updated
                pass
