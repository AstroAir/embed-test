"""
Document controller for PDF Vector System GUI.

This module contains the controller for document management operations.
"""

from typing import Any, Optional

from PySide6.QtCore import QObject, Signal

from pdf_vector_system.config.settings import Config
from pdf_vector_system.gui.utils.threading import TaskRunner
from pdf_vector_system.pipeline import PDFVectorPipeline


class DocumentController(QObject):
    """Controller for document management operations."""

    # Signals
    documents_loaded = Signal(list)  # document_list
    document_deleted = Signal(str, int)  # document_id, chunks_deleted
    statistics_updated = Signal(dict)  # statistics_dict
    operation_error = Signal(str)  # error_message
    status_message = Signal(str)  # status_message

    def __init__(
        self, config: Optional[Config] = None, parent: Optional[QObject] = None
    ):
        """
        Initialize the document controller.

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

        # Initialize pipeline
        self._initialize_pipeline()

    def _initialize_pipeline(self) -> None:
        """Initialize the PDF vector pipeline."""
        try:
            self.pipeline = PDFVectorPipeline(self.config)
            self.status_message.emit("Document management ready")
        except Exception as e:
            self.operation_error.emit(f"Failed to initialize pipeline: {e!s}")

    def load_documents(self) -> str:
        """
        Load documents list in the background.

        Returns:
            Task ID for the load operation
        """
        if not self.pipeline:
            self.operation_error.emit("Pipeline not initialized")
            return ""

        # Start load task
        return self.task_runner.run_task(self._load_documents_task, "load_documents")

    def _load_documents_task(self) -> list[dict[str, Any]]:
        """
        Background task for loading documents.

        Returns:
            List of document information dictionaries
        """
        try:
            # Use vector database to get collection info which contains document statistics
            vector_db = self.pipeline.vector_db
            if not vector_db:
                raise Exception("Vector database not initialized")

            # Get collection information to extract document statistics
            collection_info = vector_db.get_collection_info()

            # For now, return basic collection info as document data
            # In a full implementation, we'd extract individual document statistics
            return [
                {
                    "id": "collection_docs",
                    "chunks": collection_info.chunk_count,
                    "characters": collection_info.chunk_count
                    * 700,  # Estimate average chunk size
                    "avg_chunk_size": 700,
                    "created": str(
                        collection_info.metadata.get("created_at", "Unknown")
                    ),
                }
            ]

        except Exception as e:
            raise Exception(f"Failed to load documents: {e!s}") from e

    def delete_document(self, document_id: str) -> str:
        """
        Delete a document in the background.

        Args:
            document_id: ID of document to delete

        Returns:
            Task ID for the delete operation
        """
        if not self.pipeline:
            self.operation_error.emit("Pipeline not initialized")
            return ""

        if not document_id.strip():
            self.operation_error.emit("Document ID cannot be empty")
            return ""

        # Start delete task
        return self.task_runner.run_task(
            self._delete_document_task, "delete_document", document_id
        )

    def _delete_document_task(self, document_id: str) -> dict[str, Any]:
        """
        Background task for deleting a document.

        Args:
            document_id: ID of document to delete

        Returns:
            Dictionary with deletion results
        """
        try:
            chunks_deleted = self.pipeline.delete_document(document_id)

            return {"document_id": document_id, "chunks_deleted": chunks_deleted}

        except Exception as e:
            raise Exception(f"Failed to delete document {document_id}: {e!s}") from e

    def get_document_info(self, document_id: str) -> str:
        """
        Get detailed information about a document.

        Args:
            document_id: ID of document to get info for

        Returns:
            Task ID for the info operation
        """
        if not self.pipeline:
            self.operation_error.emit("Pipeline not initialized")
            return ""

        # Start info task
        return self.task_runner.run_task(
            self._get_document_info_task, "document_info", document_id
        )

    def _get_document_info_task(self, document_id: str) -> dict[str, Any]:
        """
        Background task for getting document information.

        Args:
            document_id: ID of document to get info for

        Returns:
            Dictionary with document information
        """
        try:
            return self.pipeline.get_document_info(document_id)

        except Exception as e:
            raise Exception(
                f"Failed to get document info for {document_id}: {e!s}"
            ) from e

    def get_collection_stats(self) -> str:
        """
        Get collection statistics in the background.

        Returns:
            Task ID for the stats operation
        """
        if not self.pipeline:
            self.operation_error.emit("Pipeline not initialized")
            return ""

        # Start stats task
        return self.task_runner.run_task(
            self._get_collection_stats_task, "collection_stats"
        )

    def _get_collection_stats_task(self) -> dict[str, Any]:
        """
        Background task for getting collection statistics.

        Returns:
            Dictionary with collection statistics
        """
        try:
            return self.pipeline.get_collection_stats()

        except Exception as e:
            raise Exception(f"Failed to get collection stats: {e!s}") from e

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
        if task_id == "load_documents":
            self.status_message.emit("Loading documents...")
        elif task_id == "delete_document":
            self.status_message.emit("Deleting document...")
        elif task_id == "document_info":
            self.status_message.emit("Getting document info...")
        elif task_id == "collection_stats":
            self.status_message.emit("Getting collection statistics...")

    def _on_task_finished(self, task_id: str, result: Any) -> None:
        """Handle task finished signal."""
        if task_id == "load_documents":
            self.documents_loaded.emit(result)
            self.status_message.emit(f"Loaded {len(result)} documents")
        elif task_id == "delete_document":
            self.document_deleted.emit(result["document_id"], result["chunks_deleted"])
            self.status_message.emit(f"Deleted document {result['document_id']}")
        elif task_id == "collection_stats":
            self.statistics_updated.emit(result)
            self.status_message.emit("Statistics updated")

    def _on_task_error(self, task_id: str, error: str) -> None:
        """Handle task error signal."""
        self.operation_error.emit(error)
