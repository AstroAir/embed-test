"""Mock implementations for PDF processing components."""

import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import Mock

from pdf_vector_system.core.pdf.text_processor import TextChunk, TextCleaningStats


class MockPDFDocument:
    """Mock PyMuPDF document for testing."""

    def __init__(
        self,
        page_count: int = 2,
        metadata: Optional[dict[str, Any]] = None,
        page_contents: Optional[list[str]] = None,
    ):
        self.page_count = page_count
        self.needs_pass = False
        self.is_pdf = True
        self.metadata = metadata or {
            "title": "Test Document",
            "author": "Test Author",
            "subject": "Test Subject",
            "creator": "Test Creator",
            "producer": "Test Producer",
            "creationDate": "D:20240101120000+00'00'",
            "modDate": "D:20240101120000+00'00'",
        }

        # Default page contents
        if page_contents is None:
            page_contents = [
                f"This is the content of page {i + 1}. It contains sample text for testing PDF processing."
                for i in range(page_count)
            ]

        self.page_contents = page_contents
        self._pages = []

        # Create mock pages
        for i, content in enumerate(page_contents):
            page = Mock()
            page.get_text.return_value = content
            page.number = i
            self._pages.append(page)

    def __getitem__(self, page_num: int):
        """Get a page by index."""
        if 0 <= page_num < len(self._pages):
            return self._pages[page_num]
        raise IndexError(f"Page {page_num} out of range")

    def close(self):
        """Mock close method."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MockPDFProcessor:
    """Mock PDF processor for testing."""

    def __init__(self, config=None):
        self.config = config
        self.processed_files = []
        self.extraction_results = {}

    def extract_text(self, pdf_path: Path) -> dict[str, Any]:
        """Mock text extraction from PDF."""
        pdf_path = Path(pdf_path)
        self.processed_files.append(pdf_path)

        # Create mock extraction result
        mock_content = {
            1: f"Page 1 content from {pdf_path.name}",
            2: f"Page 2 content from {pdf_path.name}",
        }

        result = {
            "file_path": str(pdf_path),
            "file_name": pdf_path.name,
            "file_size_bytes": 1024,  # Mock file size
            "text_content": mock_content,
            "page_count": len(mock_content),
            "total_characters": sum(len(content) for content in mock_content.values()),
            "metadata": {
                "title": f"Mock {pdf_path.stem}",
                "author": "Test Author",
                "subject": "Test Subject",
            },
            "extraction_timestamp": time.time(),
        }

        self.extraction_results[str(pdf_path)] = result
        return result

    def get_pdf_info(self, pdf_path: Path) -> dict[str, Any]:
        """Mock PDF info extraction."""
        pdf_path = Path(pdf_path)

        return {
            "file_path": str(pdf_path),
            "file_name": pdf_path.name,
            "file_size_bytes": 1024,
            "page_count": 2,
            "metadata": {"title": f"Mock {pdf_path.stem}", "author": "Test Author"},
            "is_encrypted": False,
            "is_pdf": True,
        }

    def validate_pdf_file(self, pdf_path: Path) -> bool:
        """Mock PDF validation."""
        return pdf_path.suffix.lower() == ".pdf" and pdf_path.exists()


class MockTextProcessor:
    """Mock text processor for testing."""

    def __init__(self, config=None):
        self.config = config
        self.cleaned_texts = []
        self.chunked_texts = []

    def clean_text(
        self,
        text: str,
        normalize_unicode: bool = True,
        remove_extra_whitespace: bool = True,
        remove_special_chars: bool = False,
        preserve_structure: bool = True,
    ) -> tuple[str, TextCleaningStats]:
        """Mock text cleaning."""
        self.cleaned_texts.append(text)

        # Simple mock cleaning - just strip whitespace
        cleaned = text.strip()

        stats = TextCleaningStats(
            original_length=len(text),
            cleaned_length=len(cleaned),
            lines_removed=text.count("\n") - cleaned.count("\n"),
            characters_removed=len(text) - len(cleaned),
            unicode_normalized=normalize_unicode,
        )

        return cleaned, stats

    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        min_chunk_size: Optional[int] = None,
    ) -> list[TextChunk]:
        """Mock text chunking."""
        self.chunked_texts.append(text)

        # Simple mock chunking - split by sentences
        sentences = text.split(". ")
        chunks = []

        current_chunk = ""
        chunk_index = 0
        start_char = 0

        for sentence in sentences:
            if current_chunk and len(current_chunk + sentence) > (chunk_size or 500):
                # Create chunk
                chunk = TextChunk(
                    content=current_chunk.strip(),
                    chunk_index=chunk_index,
                    start_char=start_char,
                    end_char=start_char + len(current_chunk),
                )
                chunks.append(chunk)

                # Start new chunk
                chunk_index += 1
                start_char += len(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += sentence + ". "

        # Add final chunk
        if current_chunk.strip():
            chunk = TextChunk(
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=start_char + len(current_chunk),
            )
            chunks.append(chunk)

        return chunks

    def process_pdf_content(
        self, pdf_content: dict[int, str], clean_text: bool = True
    ) -> list[TextChunk]:
        """Mock PDF content processing."""
        all_chunks = []
        chunk_index = 0

        for page_num, page_text in pdf_content.items():
            if clean_text:
                cleaned_text, _ = self.clean_text(page_text)
            else:
                cleaned_text = page_text

            page_chunks = self.chunk_text(cleaned_text)

            # Update chunk indices and add page metadata
            for chunk in page_chunks:
                chunk.chunk_index = chunk_index
                chunk.page_number = page_num
                all_chunks.append(chunk)
                chunk_index += 1

        return all_chunks


def create_mock_pdf_document(
    page_count: int = 2, content_per_page: Optional[list[str]] = None, **kwargs
) -> MockPDFDocument:
    """Factory function to create mock PDF document."""
    return MockPDFDocument(
        page_count=page_count, page_contents=content_per_page, **kwargs
    )


def create_mock_pdf_processor(config=None) -> MockPDFProcessor:
    """Factory function to create mock PDF processor."""
    return MockPDFProcessor(config)


def create_mock_text_processor(config=None) -> MockTextProcessor:
    """Factory function to create mock text processor."""
    return MockTextProcessor(config)
