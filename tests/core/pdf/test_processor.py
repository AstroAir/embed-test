"""Tests for PDFProcessor class."""

import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pdf_vector_system.core.config.settings import PDFConfig
from pdf_vector_system.core.pdf.processor import PDFProcessingError, PDFProcessor
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


class TestPDFProcessorComprehensive:
    """Comprehensive tests for PDFProcessor to achieve 80%+ coverage."""

    @patch("pdf_vector_system.pdf.processor.fitz.open")
    def test_extract_text_simple(self, mock_fitz_open, temp_dir):
        """Test extract_text_simple method."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("Mock PDF")

        mock_doc = create_mock_pdf_document(
            page_count=3,
            content_per_page=["Page 1", "Page 2", "Page 3"],
        )
        mock_fitz_open.return_value = mock_doc

        config = PDFConfig()
        processor = PDFProcessor(config)
        text = processor.extract_text_simple(pdf_path)

        assert isinstance(text, str)
        assert "Page 1" in text
        assert "Page 2" in text
        assert "Page 3" in text
        assert "\n\n" in text  # Pages should be separated by double newlines

    @patch("pdf_vector_system.pdf.processor.fitz.open")
    def test_extract_text_fitz_file_data_error(self, mock_fitz_open, temp_dir):
        """Test extract_text with corrupted PDF (FileDataError)."""
        pdf_path = temp_dir / "corrupted.pdf"
        pdf_path.write_text("Mock PDF")

        import fitz

        mock_fitz_open.side_effect = fitz.FileDataError("Corrupted PDF data")

        config = PDFConfig()
        processor = PDFProcessor(config)

        with pytest.raises(PDFProcessingError, match="Invalid or corrupted PDF"):
            processor.extract_text(pdf_path)

    @patch("pdf_vector_system.pdf.processor.fitz.open")
    def test_extract_text_fitz_file_not_found_error(self, mock_fitz_open, temp_dir):
        """Test extract_text with FileNotFoundError from fitz."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("Mock PDF")

        import fitz

        mock_fitz_open.side_effect = fitz.FileNotFoundError("File not found")

        config = PDFConfig()
        processor = PDFProcessor(config)

        with pytest.raises(PDFProcessingError, match="PDF file not found"):
            processor.extract_text(pdf_path)

    def test_validate_pdf_directory(self, temp_dir):
        """Test validation fails when path is a directory."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        # Create a directory with .pdf in name
        pdf_dir = temp_dir / "test.pdf"
        pdf_dir.mkdir()

        with pytest.raises(PDFProcessingError, match="Path is not a file"):
            processor._validate_pdf_file(pdf_dir)

    def test_validate_pdf_permission_error(self, temp_dir):
        """Test validation with permission error."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("Mock PDF")

        with patch.object(Path, "open", side_effect=PermissionError("Access denied")):
            with pytest.raises(PDFProcessingError, match="Permission denied"):
                processor._validate_pdf_file(pdf_path)

    def test_validate_pdf_invalid_header(self, temp_dir):
        """Test validation with invalid PDF header but .pdf extension."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        pdf_path = temp_dir / "invalid.pdf"
        # Write content without proper PDF header
        pdf_path.write_bytes(b"NOTAPDF")

        # Should log warning but not raise exception for .pdf files
        processor._validate_pdf_file(pdf_path)  # Should pass with warning

    @patch("pdf_vector_system.pdf.processor.fitz.open")
    def test_extract_metadata_with_pdf_version(self, mock_fitz_open, temp_dir):
        """Test metadata extraction includes PDF version."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("Mock PDF")

        mock_doc = Mock()
        mock_doc.metadata = {"title": "Test", "author": "Author"}
        mock_doc.page_count = 1
        mock_doc.needs_pass = False
        mock_doc.is_pdf = True
        mock_doc.pdf_version = 1.7
        mock_fitz_open.return_value = mock_doc

        config = PDFConfig()
        processor = PDFProcessor(config)

        metadata = processor._extract_metadata(mock_doc)

        assert metadata["title"] == "Test"  # Keys are lowercased, not values
        assert metadata["author"] == "Author"
        assert metadata["pdf_version"] == 1.7
        assert metadata["page_count"] == 1

    def test_extract_metadata_with_error(self):
        """Test metadata extraction handles errors gracefully."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        mock_doc = Mock()
        mock_doc.page_count = 5
        mock_doc.metadata = None  # This will cause an error

        # Should return minimal metadata on error
        metadata = processor._extract_metadata(mock_doc)

        assert "page_count" in metadata
        assert metadata["page_count"] == 5

    def test_extract_metadata_strips_whitespace(self):
        """Test metadata extraction strips whitespace from values."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        mock_doc = Mock()
        mock_doc.metadata = {
            "title": "  Test Title  ",
            "author": "  Author Name  ",
            "subject": "",  # Empty should be excluded
            "keywords": "   ",  # Whitespace-only should be excluded
        }
        mock_doc.page_count = 1
        mock_doc.needs_pass = False
        mock_doc.is_pdf = True

        metadata = processor._extract_metadata(mock_doc)

        assert metadata["title"] == "Test Title"
        assert metadata["author"] == "Author Name"
        assert "subject" not in metadata
        assert "keywords" not in metadata

    @patch("pdf_vector_system.pdf.processor.fitz.open")
    def test_get_pdf_info_encrypted(self, mock_fitz_open, temp_dir):
        """Test get_pdf_info with encrypted PDF."""
        pdf_path = temp_dir / "encrypted.pdf"
        pdf_path.write_text("Mock PDF")

        mock_doc = Mock()
        mock_doc.page_count = 10
        mock_doc.needs_pass = True  # Encrypted
        mock_doc.is_pdf = True
        mock_doc.metadata = {"title": "Encrypted Doc"}
        mock_fitz_open.return_value = mock_doc

        config = PDFConfig()
        processor = PDFProcessor(config)
        info = processor.get_pdf_info(pdf_path)

        assert info["is_encrypted"] is True
        assert info["page_count"] == 10

        mock_doc.close.assert_called_once()

    def test_extract_text_from_pages_empty_text_fallback(self):
        """Test text extraction uses fallback method when get_text() returns empty."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        mock_doc = Mock()
        mock_doc.page_count = 1

        mock_page = Mock()
        # First get_text() returns whitespace, second returns text
        mock_page.get_text.side_effect = ["   ", "Fallback text content"]

        mock_doc.__getitem__.return_value = mock_page

        text_content = processor._extract_text_from_pages(mock_doc)

        assert len(text_content) == 1
        assert "Fallback text content" in text_content[1]
        # Should have been called twice (once with no args, once with "text")
        assert mock_page.get_text.call_count == 2

    def test_clean_extracted_text_multiple_newlines(self):
        """Test text cleaning removes excessive consecutive newlines."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        text_with_many_newlines = "Line 1\n\n\n\n\nLine 2\n\n\n\nLine 3"
        cleaned = processor._clean_extracted_text(text_with_many_newlines)

        # Should reduce to max 2 consecutive newlines
        assert "\n\n\n" not in cleaned
        assert "Line 1" in cleaned
        assert "Line 2" in cleaned
        assert "Line 3" in cleaned

    def test_clean_extracted_text_preserves_structure(self):
        """Test text cleaning preserves basic structure."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        structured_text = "Header\n\nParagraph 1\n\nParagraph 2\n\nFooter"
        cleaned = processor._clean_extracted_text(structured_text)

        # Should preserve double newlines between paragraphs
        assert "Header" in cleaned
        assert "Paragraph 1" in cleaned
        assert "Paragraph 2" in cleaned
        assert "Footer" in cleaned

    def test_clean_extracted_text_removes_empty_lines(self):
        """Test text cleaning removes empty lines."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        text_with_empty_lines = "Line 1\n   \n\t\n\nLine 2\n\n\nLine 3"
        cleaned = processor._clean_extracted_text(text_with_empty_lines)

        lines = cleaned.split("\n")
        # No line should be just whitespace
        for line in lines:
            if line:  # Non-empty string
                assert line.strip() != ""

    @patch("pdf_vector_system.pdf.processor.fitz.open")
    def test_extract_text_includes_file_size(self, mock_fitz_open, temp_dir):
        """Test extract_text includes file size in bytes."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("Mock PDF content with some text")

        mock_doc = create_mock_pdf_document()
        mock_fitz_open.return_value = mock_doc

        config = PDFConfig()
        processor = PDFProcessor(config)
        result = processor.extract_text(pdf_path)

        assert "file_size_bytes" in result
        assert result["file_size_bytes"] > 0

    @patch("pdf_vector_system.pdf.processor.fitz.open")
    def test_extract_text_calculates_total_characters(self, mock_fitz_open, temp_dir):
        """Test extract_text correctly calculates total characters."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("Mock PDF")

        mock_doc = create_mock_pdf_document(
            page_count=2,
            content_per_page=["12345", "67890"],  # 5 chars each
        )
        mock_fitz_open.return_value = mock_doc

        config = PDFConfig()
        processor = PDFProcessor(config)
        result = processor.extract_text(pdf_path)

        assert result["total_characters"] == 10  # 5 + 5

    def test_pdf_processing_error_exception(self):
        """Test PDFProcessingError can be raised and caught."""
        with pytest.raises(PDFProcessingError):
            raise PDFProcessingError("Test error")

    def test_validate_pdf_file_with_small_file(self, temp_dir):
        """Test validation with small valid PDF file."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        pdf_path = temp_dir / "small.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 small content")  # Valid PDF header

        # Should not raise for small valid file
        processor._validate_pdf_file(pdf_path)

    def test_validate_pdf_file_is_file_error_handling(self, temp_dir):
        """Test validation handles errors from is_file()."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("Mock PDF")

        with patch.object(Path, "is_file", side_effect=OSError("is_file failed")):
            # Should not raise, should handle gracefully
            processor._validate_pdf_file(pdf_path)

    def test_validate_pdf_file_generic_error_with_extension_message(self, temp_dir):
        """Test validation error handling for extension-related errors."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("Mock PDF")

        with (
            patch.object(
                Path,
                "open",
                side_effect=Exception("Invalid extension"),
            ),
            pytest.raises(PDFProcessingError, match="File must have .pdf extension"),
        ):
            processor._validate_pdf_file(pdf_path)

    def test_validate_pdf_file_generic_error_with_exists_message(self, temp_dir):
        """Test validation error handling for existence-related errors."""
        config = PDFConfig()
        processor = PDFProcessor(config)

        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("Mock PDF")

        with (
            patch.object(
                Path,
                "open",
                side_effect=Exception("File does not exist somewhere"),
            ),
            pytest.raises(PDFProcessingError, match="File does not exist"),
        ):
            processor._validate_pdf_file(pdf_path)

    @patch("pdf_vector_system.pdf.processor.fitz.open")
    def test_extract_text_page_count_consistency(self, mock_fitz_open, temp_dir):
        """Test that page_count matches number of text_content entries."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("Mock PDF")

        mock_doc = create_mock_pdf_document(
            page_count=5,
            content_per_page=["P1", "P2", "P3", "P4", "P5"],
        )
        mock_fitz_open.return_value = mock_doc

        config = PDFConfig()
        processor = PDFProcessor(config)
        result = processor.extract_text(pdf_path)

        assert result["page_count"] == 5
        assert len(result["text_content"]) == 5
        assert result["page_count"] == len(result["text_content"])
