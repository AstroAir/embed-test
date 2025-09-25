"""
Search controller for PDF Vector System GUI.

This module contains the controller for search operations.
"""

from typing import Optional, List

from PySide6.QtCore import QObject, Signal

from ...config.settings import Config
from ...pipeline import PDFVectorPipeline
from ...vector_db.models import SearchResult
from ..utils.threading import TaskRunner


class SearchController(QObject):
    """Controller for search operations."""
    
    # Signals
    search_started = Signal(str)  # query
    search_completed = Signal(list)  # search_results
    search_error = Signal(str)  # error_message
    status_message = Signal(str)  # status_message
    
    def __init__(self, config: Optional[Config] = None, parent: Optional[QObject] = None):
        """
        Initialize the search controller.
        
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
            self.status_message.emit("Search pipeline ready")
        except Exception as e:
            self.search_error.emit(f"Failed to initialize pipeline: {str(e)}")
            
    def search(self, query: str, max_results: int = 10, 
               document_id: Optional[str] = None, 
               page_number: Optional[int] = None) -> str:
        """
        Perform search in the background.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            document_id: Optional document ID filter
            page_number: Optional page number filter
            
        Returns:
            Task ID for the search operation
        """
        if not self.pipeline:
            self.search_error.emit("Pipeline not initialized")
            return ""
            
        if not query.strip():
            self.search_error.emit("Search query cannot be empty")
            return ""
            
        # Start search task
        task_id = self.task_runner.run_task(
            self._search_task,
            "search",
            query,
            max_results,
            document_id,
            page_number
        )
        
        return task_id
        
    def _search_task(self, query: str, max_results: int, 
                     document_id: Optional[str], 
                     page_number: Optional[int]) -> List[SearchResult]:
        """
        Background task for performing search.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            document_id: Optional document ID filter
            page_number: Optional page number filter
            
        Returns:
            List of search results
        """
        try:
            results = self.pipeline.search(
                query_text=query,
                n_results=max_results,
                document_id=document_id,
                page_number=page_number
            )
            
            return results
            
        except Exception as e:
            raise Exception(f"Search failed: {str(e)}")
            
    def get_similar_chunks(self, chunk_id: str, max_results: int = 5) -> str:
        """
        Find similar chunks to a given chunk.
        
        Args:
            chunk_id: ID of the reference chunk
            max_results: Maximum number of results
            
        Returns:
            Task ID for the similarity search operation
        """
        if not self.pipeline:
            self.search_error.emit("Pipeline not initialized")
            return ""
            
        # Start similarity search task
        task_id = self.task_runner.run_task(
            self._similarity_task,
            "similarity",
            chunk_id,
            max_results
        )
        
        return task_id
        
    def _similarity_task(self, chunk_id: str, max_results: int) -> List[SearchResult]:
        """
        Background task for finding similar chunks.
        
        Args:
            chunk_id: ID of the reference chunk
            max_results: Maximum number of results
            
        Returns:
            List of similar chunks
        """
        try:
            # TODO: Implement similarity search in pipeline
            # For now, return empty list
            return []
            
        except Exception as e:
            raise Exception(f"Similarity search failed: {str(e)}")
            
    def stop_search(self) -> bool:
        """
        Stop current search operation.
        
        Returns:
            True if search was stopped
        """
        return self.task_runner.stop_task("search")
        
    def is_searching(self) -> bool:
        """
        Check if search is currently running.
        
        Returns:
            True if search is running
        """
        return self.task_runner.is_task_running("search")
        
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
        if task_id == "search":
            self.status_message.emit("Search started")
        elif task_id == "similarity":
            self.status_message.emit("Finding similar chunks")
            
    def _on_task_finished(self, task_id: str, result: List[SearchResult]) -> None:
        """Handle task finished signal."""
        if task_id in ["search", "similarity"]:
            self.search_completed.emit(result)
            self.status_message.emit(f"Search completed: {len(result)} results found")
            
    def _on_task_error(self, task_id: str, error: str) -> None:
        """Handle task error signal."""
        if task_id in ["search", "similarity"]:
            self.search_error.emit(error)
