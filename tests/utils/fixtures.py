"""Test fixture utilities for creating test objects."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile
import shutil

from pdf_vector_system.config.settings import (
    Config, PDFConfig, TextProcessingConfig, EmbeddingConfig,
    ChromaDBConfig, LoggingConfig, EmbeddingModelType, LogLevel
)
from pdf_vector_system.vector_db.models import DocumentChunk, SearchResult, SearchQuery
from pdf_vector_system.pdf.text_processor import TextChunk
from .test_data import TestDataGenerator


def create_test_config(
    temp_dir: Optional[Path] = None,
    embedding_type: EmbeddingModelType = EmbeddingModelType.SENTENCE_TRANSFORMERS,
    **overrides
) -> Config:
    """Create a test configuration with sensible defaults."""
    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp())
    
    # Default configurations
    pdf_config = PDFConfig(
        max_file_size_mb=10,
        timeout_seconds=60,
        extract_images=False
    )
    
    text_config = TextProcessingConfig(
        chunk_size=200,
        chunk_overlap=20,
        min_chunk_size=10,
        separators=["\n\n", "\n", " ", ""]
    )
    
    embedding_config = EmbeddingConfig(
        model_type=embedding_type,
        model_name="all-MiniLM-L6-v2" if embedding_type == EmbeddingModelType.SENTENCE_TRANSFORMERS else "text-embedding-3-small",
        batch_size=8,
        openai_api_key="test-key" if embedding_type == EmbeddingModelType.OPENAI else None
    )
    
    chroma_config = ChromaDBConfig(
        persist_directory=temp_dir / "test_chroma",
        collection_name="test_collection",
        max_results=10
    )
    
    logging_config = LoggingConfig(
        level=LogLevel.ERROR,  # Minimize logging during tests
        file_path=None,  # No file logging during tests
        format="{message}",
        rotation="1 MB",
        retention="1 day"
    )
    
    # Create main config
    config = Config(
        pdf=pdf_config,
        text_processing=text_config,
        embedding=embedding_config,
        chroma_db=chroma_config,
        logging=logging_config,
        debug=False,
        max_workers=1  # Single worker for deterministic tests
    )
    
    # Apply any overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif '.' in key:
            # Handle nested attributes like 'pdf.max_file_size_mb'
            parts = key.split('.')
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
    
    return config


def create_test_document_chunks(
    count: int = 5,
    document_id: str = "test_doc",
    embedding_dim: int = 5,
    page_number: int = 1
) -> List[DocumentChunk]:
    """Create test DocumentChunk objects."""
    generator = TestDataGenerator()
    chunks = []
    
    for i in range(count):
        content = generator.generate_text(10, 30)
        embedding = generator.generate_embeddings(1, embedding_dim)[0]
        
        chunk = DocumentChunk.create_chunk(
            document_id=document_id,
            chunk_index=i,
            content=content,
            embedding=embedding,
            page_number=page_number,
            start_char=i * 50,
            end_char=(i + 1) * 50,
            additional_metadata={"test": True, "chunk_type": "generated"}
        )
        chunks.append(chunk)
    
    return chunks


def create_test_search_results(
    count: int = 3,
    document_id: str = "test_doc",
    base_score: float = 0.9
) -> List[SearchResult]:
    """Create test SearchResult objects."""
    generator = TestDataGenerator()
    results = []
    
    for i in range(count):
        result = SearchResult(
            id=f"{document_id}_chunk_{i}",
            content=generator.generate_text(15, 40),
            score=base_score - (i * 0.1),  # Decreasing scores
            metadata={
                "document_id": document_id,
                "chunk_index": i,
                "page_number": 1,
                "test": True
            }
        )
        results.append(result)
    
    return results


def create_test_search_query(
    query_text: str = "test query",
    n_results: int = 5,
    filters: Optional[Dict[str, Any]] = None
) -> SearchQuery:
    """Create a test SearchQuery object."""
    return SearchQuery(
        query_text=query_text,
        n_results=n_results,
        where=filters,
        include_distances=True,
        include_metadata=True,
        include_documents=True
    )


def create_test_text_chunks(
    count: int = 3,
    base_text: Optional[str] = None
) -> List[TextChunk]:
    """Create test TextChunk objects."""
    generator = TestDataGenerator()
    
    if base_text is None:
        base_text = generator.generate_text(100, 200)
    
    # Split text into chunks
    words = base_text.split()
    chunk_size = max(1, len(words) // count)
    
    chunks = []
    for i in range(count):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(words))
        
        chunk_words = words[start_idx:end_idx]
        chunk_content = " ".join(chunk_words)
        
        # Estimate character positions
        start_char = sum(len(word) + 1 for word in words[:start_idx])
        end_char = start_char + len(chunk_content)
        
        chunk = TextChunk(
            content=chunk_content,
            chunk_index=i,
            start_char=start_char,
            end_char=end_char
        )
        chunks.append(chunk)
    
    return chunks


def create_test_pdf_content(
    page_count: int = 2,
    realistic: bool = False
) -> Dict[int, str]:
    """Create test PDF content."""
    if realistic:
        # Return realistic content for comprehensive testing
        return {
            1: """
            Machine Learning Fundamentals
            
            Machine learning is a method of data analysis that automates analytical
            model building. It is a branch of artificial intelligence based on the
            idea that systems can learn from data, identify patterns and make
            decisions with minimal human intervention.
            
            The process typically involves training a model on a dataset, then
            using that trained model to make predictions on new, unseen data.
            """.strip(),
            
            2: """
            Applications and Use Cases
            
            Machine learning has numerous applications across various industries.
            In healthcare, it's used for medical diagnosis and drug discovery.
            In finance, it powers fraud detection and algorithmic trading.
            
            Other common applications include recommendation systems, image
            recognition, natural language processing, and autonomous vehicles.
            The field continues to grow rapidly with new applications emerging
            regularly.
            """.strip()
        }
    else:
        # Generate simple test content
        generator = TestDataGenerator()
        return generator.generate_pdf_content(page_count)


def create_temporary_test_directory() -> Path:
    """Create a temporary directory for tests."""
    return Path(tempfile.mkdtemp(prefix="pdf_vector_test_"))


def cleanup_test_directory(test_dir: Path) -> None:
    """Clean up a test directory."""
    if test_dir.exists() and test_dir.is_dir():
        shutil.rmtree(test_dir, ignore_errors=True)


class TestResourceManager:
    """Manager for test resources and cleanup."""
    
    def __init__(self):
        self.temp_dirs: List[Path] = []
        self.temp_files: List[Path] = []
    
    def create_temp_dir(self, prefix: str = "test_") -> Path:
        """Create a temporary directory."""
        temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
        self.temp_dirs.append(temp_dir)
        return temp_dir
    
    def create_temp_file(
        self,
        content: str = "",
        suffix: str = ".txt",
        prefix: str = "test_"
    ) -> Path:
        """Create a temporary file."""
        fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, text=True)
        temp_file = Path(temp_path)
        
        try:
            with open(fd, 'w', encoding='utf-8') as f:
                f.write(content)
        finally:
            # Close the file descriptor
            import os
            os.close(fd)
        
        self.temp_files.append(temp_file)
        return temp_file
    
    def cleanup(self) -> None:
        """Clean up all created resources."""
        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception:
                pass  # Ignore cleanup errors
        
        # Clean up temporary directories
        for temp_dir in self.temp_dirs:
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass  # Ignore cleanup errors
        
        self.temp_dirs.clear()
        self.temp_files.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
