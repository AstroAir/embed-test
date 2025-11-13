"""Pytest configuration and fixtures for PDF processing tests."""

import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from tests.mocks.pdf_mocks import MockPDFDocument, MockPDFProcessor
from vectorflow.core.config.settings import PDFConfig, TextProcessingConfig
from vectorflow.core.pdf.text_processor import TextChunk


@pytest.fixture
def pdf_temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory specifically for PDF tests."""
    temp_path = Path(tempfile.mkdtemp(prefix="pdf_test_"))
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_pdf_file(pdf_temp_dir: Path) -> Path:
    """Create a mock PDF file for testing."""
    pdf_path = pdf_temp_dir / "test_document.pdf"
    # Create a simple file that can be used as a mock PDF
    pdf_path.write_bytes(b"%PDF-1.4\nMock PDF content for testing")
    return pdf_path


@pytest.fixture
def large_mock_pdf_file(pdf_temp_dir: Path) -> Path:
    """Create a large mock PDF file for testing size limits."""
    pdf_path = pdf_temp_dir / "large_document.pdf"
    # Create a file larger than typical limits
    content = b"%PDF-1.4\n" + b"Large PDF content " * 10000
    pdf_path.write_bytes(content)
    return pdf_path


@pytest.fixture
def pdf_config_default() -> PDFConfig:
    """Create default PDF configuration for testing."""
    return PDFConfig()


@pytest.fixture
def pdf_config_strict() -> PDFConfig:
    """Create strict PDF configuration for testing."""
    return PDFConfig(max_file_size_mb=10, timeout_seconds=30, extract_images=False)


@pytest.fixture
def text_processing_config_default() -> TextProcessingConfig:
    """Create default text processing configuration."""
    return TextProcessingConfig()


@pytest.fixture
def text_processing_config_small_chunks() -> TextProcessingConfig:
    """Create text processing config with small chunks for testing."""
    return TextProcessingConfig(
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_size=10,
        separators=["\n\n", "\n", " ", ""],
    )


@pytest.fixture
def sample_pdf_text() -> str:
    """Provide sample PDF text content for testing."""
    return """
    Chapter 1: Introduction to Machine Learning

    Machine learning is a subset of artificial intelligence that focuses on the development
    of algorithms and statistical models that enable computer systems to improve their
    performance on a specific task through experience.

    The field of machine learning has grown exponentially in recent years, driven by
    advances in computing power, the availability of large datasets, and improvements
    in algorithmic techniques.

    Chapter 2: Types of Machine Learning

    There are three main types of machine learning:

    1. Supervised Learning: Uses labeled training data to learn a mapping function
    2. Unsupervised Learning: Finds hidden patterns in data without labeled examples
    3. Reinforcement Learning: Learns through interaction with an environment

    Each type has its own strengths and is suitable for different kinds of problems.
    """.strip()


@pytest.fixture
def sample_pdf_pages() -> dict[int, str]:
    """Provide sample PDF pages content."""
    return {
        1: """Page 1 Content

        This is the first page of the document. It contains introductory material
        and sets the context for the rest of the document.

        The content here is designed to test PDF text extraction and processing
        capabilities of the system.""",
        2: """Page 2 Content

        This is the second page with different content. It includes more detailed
        information and examples.

        Special characters: @#$%^&*()
        Unicode characters: café, naïve, résumé

        Numbers and dates: 123, 456.789, 2024-01-01""",
        3: """Page 3 Content

        The final page contains concluding remarks and references.

        This page tests edge cases and boundary conditions for text processing.
        It includes various formatting and special cases.""",
    }


@pytest.fixture
def sample_text_chunks() -> list[TextChunk]:
    """Provide sample text chunks for testing."""
    return [
        TextChunk(
            content="This is the first chunk of text content.",
            start_char=0,
            end_char=42,
            page_number=1,
            chunk_index=0,
        ),
        TextChunk(
            content="This is the second chunk with different content.",
            start_char=43,
            end_char=91,
            page_number=1,
            chunk_index=1,
        ),
        TextChunk(
            content="The third chunk spans to the next page.",
            start_char=92,
            end_char=131,
            page_number=2,
            chunk_index=2,
        ),
    ]


@pytest.fixture
def mock_pdf_document() -> MockPDFDocument:
    """Create a mock PDF document for testing."""
    return MockPDFDocument()


@pytest.fixture
def mock_pdf_processor() -> MockPDFProcessor:
    """Create a mock PDF processor for testing."""
    return MockPDFProcessor()


@pytest.fixture
def sample_pdf_metadata() -> dict[str, Any]:
    """Provide sample PDF metadata."""
    return {
        "title": "Test Document",
        "author": "Test Author",
        "subject": "Testing PDF Processing",
        "creator": "PDF Vector System Tests",
        "producer": "Test Producer",
        "creation_date": "2024-01-01T00:00:00",
        "modification_date": "2024-01-01T12:00:00",
    }


@pytest.fixture
def corrupted_pdf_file(pdf_temp_dir: Path) -> Path:
    """Create a corrupted PDF file for error testing."""
    pdf_path = pdf_temp_dir / "corrupted.pdf"
    # Create a file that looks like PDF but is corrupted
    pdf_path.write_bytes(b"%PDF-1.4\nCorrupted content that will cause errors")
    return pdf_path
