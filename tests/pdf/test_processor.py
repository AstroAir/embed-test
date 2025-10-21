"""Tests for PDFProcessor class."""

import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pdf_vector_system.config.settings import PDFConfig
from pdf_vector_system.pdf.processor import PDFProcessingError, PDFProcessor
from tests.mocks.pdf_mocks import create_mock_pdf_document


class TestPDFProcessor:
    """Test PDFProcessor class."""

    def test_initialization(self):
        """Test PDFProcessor initialization."""
        config = PDFConfig(max_file_size_mb=50, timeout_seconds=120)
        processor = PDFProcessor(config)

        assert processor.config == config
        assert processor.config.max_file_size_mb == 50
        assert processor.config.timeout_seconds == 120

    def test_initialization_with_defaults(self):
        """Test PDFProcessor initialization with default config."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        assert processor.config.max_file_size_mb == 100
        assert processor.config.timeout_seconds == 300
        assert processor.config.extract_images is False

    @patch("pdf_vector_system.pdf.processor.fitz.open")
    def test_extract_text_success(self, mock_fitz_open, temp_dir):
        """Test successful text extraction."""
        # Create a test PDF file
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("Mock PDF content")

        # Create mock document
        mock_doc = create_mock_pdf_document(
            page_count=2,
            content_per_page=[
                "This is page 1 content with some text.",
                "This is page 2 content with more text.",
            ],
        )
        mock_fitz_open.return_value = mock_doc

        # Test extraction
        config = PDFConfig()
        processor = PDFProcessor(config)
        result = processor.extract_text(pdf_path)

        # Verify results
        assert result["file_path"] == str(pdf_path)
        assert result["file_name"] == "test.pdf"
        assert result["page_count"] == 2
        assert result["text_content"][1] == "This is page 1 content with some text."
        assert result["text_content"][2] == "This is page 2 content with more text."
        assert result["total_characters"] > 0
        assert "metadata" in result
        assert "extraction_timestamp" in result

        # Verify mock was called
        mock_fitz_open.assert_called_once_with(pdf_path)
        mock_doc.close.assert_called_once()

    def test_extract_text_file_not_found(self):
        """Test text extraction with non-existent file."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        non_existent_path = Path("non_existent.pdf")

        with pytest.raises(PDFProcessingError, match="File does not exist"):
            processor.extract_text(non_existent_path)

    def test_extract_text_invalid_extension(self, temp_dir):
        """Test text extraction with invalid file extension."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        # Create a file with wrong extension
        txt_path = temp_dir / "test.txt"
        txt_path.write_text("Not a PDF")

        with pytest.raises(PDFProcessingError, match="File must have .pdf extension"):
            processor.extract_text(txt_path)

    def test_extract_text_file_too_large(self, temp_dir):
        """Test text extraction with file too large."""
        config = PDFConfig(max_file_size_mb=1)  # 1 MB limit
        processor = PDFProcessor(config)

        # Create a large file (mock the size)
        pdf_path = temp_dir / "large.pdf"
        pdf_path.write_text("PDF content")

        with patch.object(Path, "stat") as mock_stat:
            # Mock file size to be larger than limit
            mock_stat.return_value.st_size = 2 * 1024 * 1024  # 2 MB

            with pytest.raises(
                PDFProcessingError, match="File size .* exceeds maximum"
            ):
                processor.extract_text(pdf_path)

    @patch("pdf_vector_system.pdf.processor.fitz.open")
    def test_extract_text_fitz_error(self, mock_fitz_open, temp_dir):
        """Test text extraction with PyMuPDF error."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("Mock PDF")

        # Mock fitz.open to raise an exception
        mock_fitz_open.side_effect = Exception("PyMuPDF error")

        config = PDFConfig()
        processor = PDFProcessor(config)

        with pytest.raises(PDFProcessingError, match="Error extracting text"):
            processor.extract_text(pdf_path)

    @patch("pdf_vector_system.pdf.processor.fitz.open")
    def test_get_pdf_info_success(self, mock_fitz_open, temp_dir):
        """Test successful PDF info extraction."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("Mock PDF")

        mock_doc = create_mock_pdf_document(page_count=3)
        mock_fitz_open.return_value = mock_doc

        config = PDFConfig()
        processor = PDFProcessor(config)
        info = processor.get_pdf_info(pdf_path)

        assert info["file_path"] == str(pdf_path)
        assert info["file_name"] == "test.pdf"
        assert info["page_count"] == 3
        assert info["is_encrypted"] is False
        assert info["is_pdf"] is True
        assert "metadata" in info

        mock_fitz_open.assert_called_once_with(pdf_path)
        mock_doc.close.assert_called_once()

    @patch("pdf_vector_system.pdf.processor.fitz.open")
    def test_get_pdf_info_error(self, mock_fitz_open, temp_dir):
        """Test PDF info extraction with error."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("Mock PDF")

        mock_fitz_open.side_effect = Exception("PyMuPDF error")

        config = PDFConfig()
        processor = PDFProcessor(config)

        with pytest.raises(PDFProcessingError, match="Error getting PDF info"):
            processor.get_pdf_info(pdf_path)

    def test_validate_pdf_file_success(self, temp_dir):
        """Test successful PDF file validation."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        pdf_path = temp_dir / "valid.pdf"
        pdf_path.write_text("Mock PDF content")

        # Should not raise an exception
        processor._validate_pdf_file(pdf_path)

    def test_validate_pdf_file_not_exists(self):
        """Test PDF file validation with non-existent file."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        with pytest.raises(PDFProcessingError, match="File does not exist"):
            processor._validate_pdf_file(Path("non_existent.pdf"))

    def test_validate_pdf_file_wrong_extension(self, temp_dir):
        """Test PDF file validation with wrong extension."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        txt_path = temp_dir / "test.txt"
        txt_path.write_text("Not a PDF")

        with pytest.raises(PDFProcessingError, match="File must have .pdf extension"):
            processor._validate_pdf_file(txt_path)

    def test_validate_pdf_file_too_large(self, temp_dir):
        """Test PDF file validation with file too large."""
        config = PDFConfig(max_file_size_mb=1)
        processor = PDFProcessor(config)

        pdf_path = temp_dir / "large.pdf"
        pdf_path.write_text("PDF content")

        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value.st_size = 2 * 1024 * 1024  # 2 MB

            with pytest.raises(
                PDFProcessingError, match="File size .* exceeds maximum"
            ):
                processor._validate_pdf_file(pdf_path)

    def test_extract_metadata(self):
        """Test metadata extraction from PDF document."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        mock_doc = create_mock_pdf_document()
        metadata = processor._extract_metadata(mock_doc)

        assert metadata["title"] == "Test Document"
        assert metadata["author"] == "Test Author"
        assert metadata["subject"] == "Test Subject"
        assert metadata["creator"] == "Test Creator"

    def test_extract_metadata_empty(self):
        """Test metadata extraction with empty metadata."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        mock_doc = Mock()
        mock_doc.metadata = {}

        metadata = processor._extract_metadata(mock_doc)

        # Should return empty dict for empty metadata
        assert isinstance(metadata, dict)

    def test_extract_text_from_pages(self):
        """Test text extraction from pages."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        mock_doc = create_mock_pdf_document(
            page_count=3,
            content_per_page=[
                "Page 1 text content",
                "Page 2 text content",
                "Page 3 text content",
            ],
        )

        text_content = processor._extract_text_from_pages(mock_doc)

        assert len(text_content) == 3
        assert text_content[1] == "Page 1 text content"
        assert text_content[2] == "Page 2 text content"
        assert text_content[3] == "Page 3 text content"

    def test_extract_text_from_pages_with_error(self):
        """Test text extraction from pages with page error."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        mock_doc = Mock()
        mock_doc.page_count = 2

        # First page works, second page raises error
        mock_page1 = Mock()
        mock_page1.get_text.return_value = "Page 1 content"

        mock_page2 = Mock()
        mock_page2.get_text.side_effect = Exception("Page error")

        mock_doc.__getitem__.side_effect = [mock_page1, mock_page2]

        text_content = processor._extract_text_from_pages(mock_doc)

        assert len(text_content) == 2
        assert text_content[1] == "Page 1 content"
        assert text_content[2] == ""  # Error page should return empty string

    def test_clean_extracted_text(self):
        """Test text cleaning functionality."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        # Test with text that needs cleaning
        dirty_text = "  This is   text with\n\nextra   whitespace\t\tand tabs  "
        cleaned = processor._clean_extracted_text(dirty_text)

        # Should clean up whitespace
        assert cleaned != dirty_text
        assert "This is text" in cleaned

    def test_clean_extracted_text_empty(self):
        """Test text cleaning with empty text."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        assert processor._clean_extracted_text("") == ""
        assert processor._clean_extracted_text("   ") == ""

    @patch("pdf_vector_system.pdf.processor.fitz.open")
    def test_extract_text_with_performance_timing(self, mock_fitz_open, temp_dir):
        """Test that text extraction includes performance timing."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("Mock PDF")

        mock_doc = create_mock_pdf_document()
        mock_fitz_open.return_value = mock_doc

        config = PDFConfig()
        processor = PDFProcessor(config)

        start_time = time.time()
        result = processor.extract_text(pdf_path)
        end_time = time.time()

        # Should have extraction timestamp
        assert "extraction_timestamp" in result
        assert start_time <= result["extraction_timestamp"] <= end_time

    def test_path_conversion(self, temp_dir):
        """Test that string paths are converted to Path objects."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("Mock PDF")

        # Test with string path
        with patch.object(processor, "_validate_pdf_file") as mock_validate:
            with patch("pdf_vector_system.pdf.processor.fitz.open") as mock_fitz:
                mock_doc = create_mock_pdf_document()
                mock_fitz.return_value = mock_doc

                # Pass string path instead of Path object
                processor.extract_text(str(pdf_path))

                # Should have been converted to Path and validated
                mock_validate.assert_called_once()
                called_path = mock_validate.call_args[0][0]
                assert isinstance(called_path, Path)
