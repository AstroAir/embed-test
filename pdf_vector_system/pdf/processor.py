"""PDF processing module using PyMuPDF for text extraction."""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import time

import fitz  # PyMuPDF
from loguru import logger

from ..config.settings import PDFConfig
from ..utils.logging import LoggerMixin, log_error, log_performance
from ..utils.progress import PerformanceTimer


class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors."""
    pass


class PDFProcessor(LoggerMixin):
    """PDF processor using PyMuPDF for robust text extraction."""
    
    def __init__(self, config: PDFConfig):
        """
        Initialize PDF processor.
        
        Args:
            config: PDF processing configuration
        """
        self.config = config
        self.logger.info(f"Initialized PDFProcessor with config: {config}")
    
    def extract_text(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
            
        Raises:
            PDFProcessingError: If PDF processing fails
        """
        pdf_path = Path(pdf_path)
        
        # Validate file
        self._validate_pdf_file(pdf_path)
        
        with PerformanceTimer(f"PDF text extraction from {pdf_path.name}"):
            try:
                # Open PDF document
                doc = fitz.open(pdf_path)
                
                # Extract metadata
                metadata = self._extract_metadata(doc)
                
                # Extract text from all pages
                text_content = self._extract_text_from_pages(doc)
                
                # Close document
                doc.close()
                
                result = {
                    "file_path": str(pdf_path),
                    "file_name": pdf_path.name,
                    "file_size_bytes": pdf_path.stat().st_size,
                    "text_content": text_content,
                    "page_count": len(text_content),
                    "total_characters": sum(len(page_text) for page_text in text_content.values()),
                    "metadata": metadata,
                    "extraction_timestamp": time.time(),
                }
                
                self.logger.info(
                    f"Successfully extracted text from {pdf_path.name}: "
                    f"{result['page_count']} pages, {result['total_characters']} characters"
                )
                
                return result
                
            except fitz.FileDataError as e:
                error_msg = f"Invalid or corrupted PDF file: {pdf_path}"
                log_error(e, error_msg)
                raise PDFProcessingError(error_msg) from e
            
            except fitz.FileNotFoundError as e:
                error_msg = f"PDF file not found: {pdf_path}"
                log_error(e, error_msg)
                raise PDFProcessingError(error_msg) from e
            
            except Exception as e:
                error_msg = f"Unexpected error processing PDF {pdf_path}"
                log_error(e, error_msg)
                raise PDFProcessingError(error_msg) from e
    
    def extract_text_simple(self, pdf_path: Path) -> str:
        """
        Extract text content from a PDF file as a single string.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content as a single string
        """
        result = self.extract_text(pdf_path)
        return "\n\n".join(result["text_content"].values())
    
    def get_pdf_info(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Get basic information about a PDF file without extracting text.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing PDF information
        """
        pdf_path = Path(pdf_path)
        self._validate_pdf_file(pdf_path)
        
        try:
            doc = fitz.open(pdf_path)
            metadata = self._extract_metadata(doc)
            
            info = {
                "file_path": str(pdf_path),
                "file_name": pdf_path.name,
                "file_size_bytes": pdf_path.stat().st_size,
                "page_count": doc.page_count,
                "metadata": metadata,
                "is_encrypted": doc.needs_pass,
                "is_pdf": doc.is_pdf,
            }
            
            doc.close()
            return info
            
        except Exception as e:
            error_msg = f"Error getting PDF info for {pdf_path}"
            log_error(e, error_msg)
            raise PDFProcessingError(error_msg) from e
    
    def _validate_pdf_file(self, pdf_path: Path) -> None:
        """
        Validate PDF file before processing.
        
        Args:
            pdf_path: Path to the PDF file
            
        Raises:
            PDFProcessingError: If validation fails
        """
        # Check if file exists
        if not pdf_path.exists():
            raise PDFProcessingError(f"PDF file does not exist: {pdf_path}")
        
        # Check if it's a file (not directory)
        if not pdf_path.is_file():
            raise PDFProcessingError(f"Path is not a file: {pdf_path}")
        
        # Check file extension
        if pdf_path.suffix.lower() != '.pdf':
            self.logger.warning(f"File does not have .pdf extension: {pdf_path}")
        
        # Check file size
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            raise PDFProcessingError(
                f"PDF file too large: {file_size_mb:.1f}MB > {self.config.max_file_size_mb}MB"
            )
        
        # Check if file is readable
        try:
            with open(pdf_path, 'rb') as f:
                # Read first few bytes to check if it's a PDF
                header = f.read(4)
                if header != b'%PDF':
                    raise PDFProcessingError(f"File does not appear to be a valid PDF: {pdf_path}")
        except PermissionError:
            raise PDFProcessingError(f"Permission denied reading PDF file: {pdf_path}")
        except Exception as e:
            raise PDFProcessingError(f"Error reading PDF file {pdf_path}: {str(e)}")
    
    def _extract_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        """
        Extract metadata from PDF document.
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            Dictionary containing metadata
        """
        try:
            metadata = doc.metadata
            
            # Clean and structure metadata
            cleaned_metadata = {}
            for key, value in metadata.items():
                if value and value.strip():
                    cleaned_metadata[key.lower()] = value.strip()
            
            # Add additional information
            cleaned_metadata.update({
                "page_count": doc.page_count,
                "is_encrypted": doc.needs_pass,
                "is_pdf": doc.is_pdf,
                "pdf_version": getattr(doc, 'pdf_version', None),
            })
            
            return cleaned_metadata
            
        except Exception as e:
            self.logger.warning(f"Error extracting metadata: {str(e)}")
            return {"page_count": doc.page_count}
    
    def _extract_text_from_pages(self, doc: fitz.Document) -> Dict[int, str]:
        """
        Extract text from all pages of the PDF document.
        
        Args:
            doc: PyMuPDF document object
            
        Returns:
            Dictionary mapping page numbers to text content
        """
        text_content = {}
        
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                
                # Extract text using different methods for better coverage
                text = page.get_text()
                
                # If no text found, try alternative extraction
                if not text.strip():
                    text = page.get_text("text")
                
                # Clean up the text
                text = self._clean_extracted_text(text)
                
                text_content[page_num + 1] = text  # 1-based page numbering
                
                self.logger.debug(f"Extracted {len(text)} characters from page {page_num + 1}")
                
            except Exception as e:
                self.logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                text_content[page_num + 1] = ""
        
        return text_content
    
    def _clean_extracted_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace while preserving structure
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Strip whitespace from each line
            cleaned_line = line.strip()
            if cleaned_line:  # Only keep non-empty lines
                cleaned_lines.append(cleaned_line)
        
        # Join lines with single newlines
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove excessive consecutive newlines (more than 2)
        while '\n\n\n' in cleaned_text:
            cleaned_text = cleaned_text.replace('\n\n\n', '\n\n')
        
        return cleaned_text
