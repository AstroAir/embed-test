"""Pytest configuration and shared fixtures for PDF Vector System tests."""

import os
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock

import pytest
from loguru import logger

from vectorflow.core.config.settings import (
    ChromaDBConfig,
    Config,
    EmbeddingConfig,
    EmbeddingModelType,
    LoggingConfig,
    LogLevel,
    PDFConfig,
    TextProcessingConfig,
)
from vectorflow.core.vector_db.models import DocumentChunk, SearchResult


@pytest.fixture(scope="session", autouse=True)
def setup_test_logging():
    """Configure logging for tests."""
    # Remove all existing handlers
    logger.remove()

    # Add a test-specific handler with minimal output
    logger.add(
        lambda msg: None,  # Suppress all log output during tests
        level="CRITICAL",
    )


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_pdf_file(temp_dir: Path) -> Path:
    """Create a mock PDF file for testing."""
    pdf_path = temp_dir / "test_document.pdf"
    # Create a simple text file that can be used as a mock PDF
    pdf_path.write_text("Mock PDF content for testing")
    return pdf_path


@pytest.fixture
def sample_text() -> str:
    """Provide sample text for testing."""
    return """
    This is a sample document for testing the PDF Vector System.
    It contains multiple paragraphs with various types of content.

    The system should be able to process this text effectively,
    cleaning it, chunking it, and generating embeddings.

    This paragraph contains some special characters: @#$%^&*()
    And some unicode characters: café, naïve, résumé

    The final paragraph tests edge cases and boundary conditions
    for the text processing pipeline.
    """.strip()


@pytest.fixture
def sample_chunks() -> list[str]:
    """Provide sample text chunks for testing."""
    return [
        "This is the first chunk of text for testing.",
        "This is the second chunk with different content.",
        "The third chunk contains more complex information.",
        "Final chunk for comprehensive testing scenarios.",
    ]


@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    """Provide sample embeddings for testing."""
    return [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.3, 0.4, 0.5, 0.6, 0.7],
        [0.4, 0.5, 0.6, 0.7, 0.8],
    ]


@pytest.fixture
def pdf_config() -> PDFConfig:
    """Create a test PDF configuration."""
    return PDFConfig(max_file_size_mb=50, timeout_seconds=120, extract_images=False)


@pytest.fixture
def text_processing_config() -> TextProcessingConfig:
    """Create a test text processing configuration."""
    return TextProcessingConfig(
        chunk_size=500,
        chunk_overlap=50,
        min_chunk_size=10,
        separators=["\n\n", "\n", " ", ""],
    )


@pytest.fixture
def embedding_config() -> EmbeddingConfig:
    """Create a test embedding configuration."""
    return EmbeddingConfig(
        model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
        model_name="all-MiniLM-L6-v2",
        batch_size=16,
        openai_api_key=None,
    )


@pytest.fixture
def chroma_config(temp_dir: Path) -> ChromaDBConfig:
    """Create a test ChromaDB configuration."""
    return ChromaDBConfig(
        persist_directory=temp_dir / "test_chroma_db",
        collection_name="test_collection",
        max_results=10,
    )


@pytest.fixture
def logging_config(temp_dir: Path) -> LoggingConfig:
    """Create a test logging configuration."""
    return LoggingConfig(
        level=LogLevel.DEBUG,
        file_path=temp_dir / "test.log",
        format="{time} | {level} | {message}",
        rotation="1 MB",
        retention="1 day",
    )


@pytest.fixture
def test_config(
    pdf_config: PDFConfig,
    text_processing_config: TextProcessingConfig,
    embedding_config: EmbeddingConfig,
    chroma_config: ChromaDBConfig,
    logging_config: LoggingConfig,
) -> Config:
    """Create a complete test configuration."""
    return Config(
        pdf=pdf_config,
        text_processing=text_processing_config,
        embedding=embedding_config,
        chroma_db=chroma_config,
        logging=logging_config,
        debug=True,
        max_workers=2,
    )


@pytest.fixture
def sample_document_chunks(
    sample_chunks: list[str], sample_embeddings: list[list[float]]
) -> list[DocumentChunk]:
    """Create sample DocumentChunk objects for testing."""
    chunks = []
    for i, (content, embedding) in enumerate(zip(sample_chunks, sample_embeddings)):
        chunk = DocumentChunk.create_chunk(
            document_id="test_doc",
            chunk_index=i,
            content=content,
            embedding=embedding,
            page_number=1,
            start_char=i * 50,
            end_char=(i + 1) * 50,
        )
        chunks.append(chunk)
    return chunks


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    """Create sample SearchResult objects for testing."""
    return [
        SearchResult(
            id="test_doc_chunk_0",
            content="This is the first search result.",
            score=0.95,
            metadata={"document_id": "test_doc", "chunk_index": 0, "page_number": 1},
        ),
        SearchResult(
            id="test_doc_chunk_1",
            content="This is the second search result.",
            score=0.87,
            metadata={"document_id": "test_doc", "chunk_index": 1, "page_number": 1},
        ),
    ]


@pytest.fixture
def mock_chromadb_client():
    """Create a mock ChromaDB client for testing."""
    mock_client = Mock()
    mock_collection = Mock()

    # Configure mock collection
    mock_collection.name = "test_collection"
    mock_collection.count.return_value = 10
    mock_collection.metadata = {"created_by": "pdf_vector_system"}

    # Configure mock client
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_client.create_collection.return_value = mock_collection
    mock_client.get_collection.return_value = mock_collection
    mock_client.list_collections.return_value = [mock_collection]

    return mock_client


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service for testing."""
    mock_service = Mock()
    mock_service.model_name = "test-model"
    mock_service.embedding_dimension = 5
    mock_service.embed_single.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    mock_service.embed_texts.return_value = Mock(
        embeddings=[[0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6]],
        model_name="test-model",
        embedding_dimension=5,
        processing_time=0.1,
        token_count=None,
    )
    mock_service.health_check.return_value = True
    return mock_service


@pytest.fixture
def mock_pdf_document():
    """Create a mock PyMuPDF document for testing."""
    mock_doc = Mock()
    mock_doc.page_count = 2
    mock_doc.needs_pass = False
    mock_doc.is_pdf = True
    mock_doc.metadata = {
        "title": "Test Document",
        "author": "Test Author",
        "subject": "Test Subject",
        "creator": "Test Creator",
    }

    # Mock pages
    mock_page1 = Mock()
    mock_page1.get_text.return_value = "This is page 1 content."
    mock_page2 = Mock()
    mock_page2.get_text.return_value = "This is page 2 content."

    mock_doc.__getitem__.side_effect = [mock_page1, mock_page2]

    return mock_doc


@pytest.fixture
def env_vars() -> Generator[dict[str, str], None, None]:
    """Provide environment variables for testing."""
    test_env = {
        "OPENAI_API_KEY": "test-api-key",
        "CHROMA_DB_PATH": "/tmp/test_chroma",
        "LOG_LEVEL": "DEBUG",
    }

    # Store original values
    original_values = {}
    for key, value in test_env.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        yield test_env
    finally:
        # Restore original values
        for key, original_value in original_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line(
        "markers", "external: mark test as requiring external dependencies"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add integration marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Add slow marker to tests that might be slow
        if any(keyword in item.nodeid for keyword in ["embedding", "pdf", "pipeline"]):
            item.add_marker(pytest.mark.slow)

        # Add external marker to tests requiring external services
        if any(keyword in item.nodeid for keyword in ["openai", "chromadb", "api"]):
            item.add_marker(pytest.mark.external)
