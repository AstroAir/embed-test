"""Main processing pipeline that orchestrates all components."""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from loguru import logger

from .config.settings import Config
from .pdf.processor import PDFProcessor
from .pdf.text_processor import TextProcessor, TextChunk
from .embeddings.factory import EmbeddingServiceFactory, BatchEmbeddingProcessor
from .vector_db.chroma_client import ChromaDBClient
from .vector_db.models import DocumentChunk, SearchQuery, SearchResult
from .utils.logging import LoggerMixin, setup_logging
from .utils.progress import ProgressTracker, PerformanceTimer


class PipelineError(Exception):
    """Exception raised for pipeline processing errors."""
    pass


@dataclass
class ProcessingResult:
    """Result of PDF processing pipeline."""
    
    document_id: str
    file_path: str
    success: bool
    chunks_processed: int
    embeddings_generated: int
    chunks_stored: int
    processing_time: float
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def chunks_per_second(self) -> float:
        """Calculate processing speed."""
        return self.chunks_processed / self.processing_time if self.processing_time > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "document_id": self.document_id,
            "file_path": self.file_path,
            "success": self.success,
            "chunks_processed": self.chunks_processed,
            "embeddings_generated": self.embeddings_generated,
            "chunks_stored": self.chunks_stored,
            "processing_time": self.processing_time,
            "chunks_per_second": self.chunks_per_second,
            "error_message": self.error_message,
            "metadata": self.metadata or {}
        }


class PDFVectorPipeline(LoggerMixin):
    """Main pipeline for processing PDFs and storing in vector database."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the PDF vector processing pipeline.

        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or Config()

        # Ensure logging is configured (fallback for direct pipeline usage)
        from .utils.logging import ensure_logging_configured
        ensure_logging_configured(self.config.logging)

        # Initialize components
        self.pdf_processor = PDFProcessor(self.config.pdf)
        self.text_processor = TextProcessor(self.config.text_processing)
        self.embedding_service = EmbeddingServiceFactory.create_service(self.config.embedding)
        self.batch_processor = BatchEmbeddingProcessor(
            self.embedding_service,
            max_workers=self.config.max_workers
        )
        self.vector_db = ChromaDBClient(self.config.chroma_db)
        
        # Ensure collection exists
        self.collection = self.vector_db.create_collection()
        
        self.logger.info("Initialized PDFVectorPipeline with all components")
    
    def process_pdf(
        self,
        pdf_path: Union[str, Path],
        document_id: Optional[str] = None,
        clean_text: bool = True,
        show_progress: bool = True
    ) -> ProcessingResult:
        """
        Process a single PDF file through the complete pipeline.
        
        Args:
            pdf_path: Path to the PDF file
            document_id: Custom document ID (uses filename if None)
            clean_text: Whether to clean extracted text
            show_progress: Whether to show progress bars
            
        Returns:
            ProcessingResult object
        """
        pdf_path = Path(pdf_path)
        doc_id = document_id or pdf_path.stem
        
        start_time = time.time()
        
        try:
            with PerformanceTimer(f"Processing PDF {pdf_path.name}", log_result=False):
                if show_progress:
                    with ProgressTracker() as progress:
                        return self._process_pdf_with_progress(
                            pdf_path, doc_id, clean_text, progress
                        )
                else:
                    return self._process_pdf_internal(pdf_path, doc_id, clean_text)
                    
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Failed to process PDF {pdf_path}: {str(e)}"
            self.logger.error(error_msg)
            
            return ProcessingResult(
                document_id=doc_id,
                file_path=str(pdf_path),
                success=False,
                chunks_processed=0,
                embeddings_generated=0,
                chunks_stored=0,
                processing_time=processing_time,
                error_message=error_msg
            )
    
    def _process_pdf_with_progress(
        self,
        pdf_path: Path,
        document_id: str,
        clean_text: bool,
        progress: ProgressTracker
    ) -> ProcessingResult:
        """Process PDF with progress tracking."""
        
        # Step 1: Extract text
        extract_task = progress.add_task("extract", "Extracting text from PDF")
        pdf_data = self.pdf_processor.extract_text(pdf_path)
        progress.complete_task("extract")
        
        # Step 2: Process and chunk text
        chunk_task = progress.add_task("chunk", "Processing and chunking text")
        all_chunks = self._process_and_chunk_text(pdf_data, document_id, clean_text)
        progress.complete_task("chunk")
        
        if not all_chunks:
            raise ValueError("No text chunks were generated")
        
        # Step 3: Generate embeddings
        embed_task = progress.add_task(
            "embed", 
            f"Generating embeddings for {len(all_chunks)} chunks",
            total=len(all_chunks)
        )
        
        chunk_texts = [chunk.content for chunk in all_chunks]
        embedding_result = self.batch_processor.process_texts(
            chunk_texts, show_progress=False
        )
        progress.complete_task("embed")
        
        # Step 4: Create document chunks with embeddings
        store_task = progress.add_task("store", "Storing in vector database")
        document_chunks = self._create_document_chunks(all_chunks, embedding_result.embeddings)
        
        # Step 5: Store in vector database
        self.vector_db.add_chunks(document_chunks)
        progress.complete_task("store")
        
        processing_time = time.time() - time.time()  # This will be updated by caller
        
        return ProcessingResult(
            document_id=document_id,
            file_path=str(pdf_path),
            success=True,
            chunks_processed=len(all_chunks),
            embeddings_generated=len(embedding_result.embeddings),
            chunks_stored=len(document_chunks),
            processing_time=processing_time,
            metadata={
                "pdf_metadata": pdf_data.get("metadata", {}),
                "page_count": pdf_data.get("page_count", 0),
                "total_characters": pdf_data.get("total_characters", 0),
                "embedding_model": self.embedding_service.model_name,
                "embedding_dimension": embedding_result.embedding_dimension
            }
        )
    
    def _process_pdf_internal(
        self,
        pdf_path: Path,
        document_id: str,
        clean_text: bool
    ) -> ProcessingResult:
        """Internal PDF processing without progress tracking."""
        
        start_time = time.time()
        
        # Extract text
        pdf_data = self.pdf_processor.extract_text(pdf_path)
        
        # Process and chunk text
        all_chunks = self._process_and_chunk_text(pdf_data, document_id, clean_text)
        
        if not all_chunks:
            raise ValueError("No text chunks were generated")
        
        # Generate embeddings
        chunk_texts = [chunk.content for chunk in all_chunks]
        embedding_result = self.batch_processor.process_texts(
            chunk_texts, show_progress=False
        )
        
        # Create document chunks with embeddings
        document_chunks = self._create_document_chunks(all_chunks, embedding_result.embeddings)
        
        # Store in vector database
        self.vector_db.add_chunks(document_chunks)
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            document_id=document_id,
            file_path=str(pdf_path),
            success=True,
            chunks_processed=len(all_chunks),
            embeddings_generated=len(embedding_result.embeddings),
            chunks_stored=len(document_chunks),
            processing_time=processing_time,
            metadata={
                "pdf_metadata": pdf_data.get("metadata", {}),
                "page_count": pdf_data.get("page_count", 0),
                "total_characters": pdf_data.get("total_characters", 0),
                "embedding_model": self.embedding_service.model_name,
                "embedding_dimension": embedding_result.embedding_dimension
            }
        )
    
    def _process_and_chunk_text(
        self,
        pdf_data: Dict[str, Any],
        document_id: str,
        clean_text: bool
    ) -> List[TextChunk]:
        """Process and chunk extracted text."""
        
        all_chunks = []
        
        for page_num, page_text in pdf_data["text_content"].items():
            if not page_text.strip():
                continue
            
            # Clean text if requested
            if clean_text:
                cleaned_text, _ = self.text_processor.clean_text(page_text)
            else:
                cleaned_text = page_text
            
            # Chunk the text
            page_chunks = self.text_processor.chunk_text_with_metadata(
                cleaned_text,
                document_id=document_id,
                page_number=page_num,
                additional_metadata={
                    "file_name": pdf_data.get("file_name"),
                    "file_size_bytes": pdf_data.get("file_size_bytes"),
                    "extraction_timestamp": pdf_data.get("extraction_timestamp")
                }
            )
            
            all_chunks.extend(page_chunks)
        
        return all_chunks
    
    def _create_document_chunks(
        self,
        text_chunks: List[TextChunk],
        embeddings: List[List[float]]
    ) -> List[DocumentChunk]:
        """Create DocumentChunk objects from text chunks and embeddings."""
        
        if len(text_chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch between text chunks ({len(text_chunks)}) "
                f"and embeddings ({len(embeddings)})"
            )
        
        document_chunks = []
        
        for text_chunk, embedding in zip(text_chunks, embeddings):
            source_info = text_chunk.source_info or {}
            doc_chunk = DocumentChunk(
                id=f"{source_info.get('document_id', 'unknown')}_chunk_{text_chunk.chunk_index}",
                content=text_chunk.content,
                embedding=embedding,
                metadata={
                    **source_info,
                    "chunk_length": text_chunk.length,
                    "start_char": text_chunk.start_char,
                    "end_char": text_chunk.end_char,
                    "stored_at": time.time()
                }
            )
            document_chunks.append(doc_chunk)
        
        return document_chunks
    
    def search(
        self,
        query_text: str,
        n_results: int = 10,
        document_id: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Search the vector database.
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return
            document_id: Optional document ID filter
            page_number: Optional page number filter
            
        Returns:
            List of SearchResult objects
        """
        where_clause: Dict[str, Any] = {}
        if document_id:
            where_clause["document_id"] = document_id
        if page_number:
            where_clause["page_number"] = page_number
        
        query = SearchQuery(
            query_text=query_text,
            n_results=n_results,
            where=where_clause if where_clause else None
        )
        
        return self.vector_db.search(query)
    
    def get_document_info(self, document_id: str) -> Dict[str, Any]:
        """
        Get information about a processed document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Dictionary containing document information
        """
        doc_info = self.vector_db.get_document_info(document_id)
        return doc_info.to_dict()
    
    def delete_document(self, document_id: str) -> int:
        """
        Delete a document from the vector database.
        
        Args:
            document_id: Document ID
            
        Returns:
            Number of chunks deleted
        """
        return self.vector_db.delete_document(document_id)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database collection.
        
        Returns:
            Dictionary containing collection statistics
        """
        return self.vector_db.get_collection_stats()
    
    def health_check(self) -> Dict[str, bool]:
        """
        Perform health check on all components.
        
        Returns:
            Dictionary with health status of each component
        """
        return {
            "embedding_service": self.embedding_service.health_check(),
            "vector_database": True,  # ChromaDB doesn't have a specific health check
            "pipeline": True
        }
