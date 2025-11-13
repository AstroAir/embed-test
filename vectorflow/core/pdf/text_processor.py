"""Text processing utilities for cleaning and normalizing extracted PDF content."""

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from vectorflow.core.config.settings import TextProcessingConfig
from vectorflow.core.utils.logging import LoggerMixin


@dataclass
class TextCleaningStats:
    """Statistics about text cleaning operations."""

    original_length: int
    cleaned_length: int
    lines_removed: int
    characters_removed: int
    unicode_normalized: bool

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio (0-1, where 1 means no compression)."""
        if self.original_length == 0:
            return 1.0
        return self.cleaned_length / self.original_length


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""

    content: str
    chunk_index: int
    start_char: int
    end_char: int
    source_info: Optional[dict[str, Any]] = None
    page_number: Optional[int] = None

    @property
    def length(self) -> int:
        """Get the length of the chunk content."""
        return len(self.content)

    def to_dict(self) -> dict[str, Any]:
        """Convert chunk to dictionary representation."""
        result = {
            "content": self.content,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "length": self.length,
            "source_info": self.source_info or {},
        }
        if self.page_number is not None:
            result["page_number"] = self.page_number
        return result


class TextProcessor(LoggerMixin):
    """Text processor for cleaning and normalizing extracted PDF content."""

    def __init__(self, config: TextProcessingConfig):
        """
        Initialize text processor.

        Args:
            config: Text processing configuration
        """
        self.config = config
        self._text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        self.logger.info(f"Initialized TextProcessor with config: {config}")

    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Get or create the text splitter instance."""
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=self.config.separators,
                length_function=len,
                is_separator_regex=False,
            )
        if self._text_splitter is None:
            raise RuntimeError("Failed to create text splitter")
        return self._text_splitter

    def clean_text(
        self,
        text: str,
        normalize_unicode: bool = True,
        remove_extra_whitespace: bool = True,
        remove_special_chars: bool = False,
        preserve_structure: bool = True,
    ) -> tuple[str, TextCleaningStats]:
        """
        Clean and normalize text content.

        Args:
            text: Raw text to clean
            normalize_unicode: Whether to normalize Unicode characters
            remove_extra_whitespace: Whether to remove excessive whitespace
            remove_special_chars: Whether to remove special characters
            preserve_structure: Whether to preserve paragraph structure

        Returns:
            Tuple of (cleaned_text, cleaning_stats)
        """
        if not text:
            return "", TextCleaningStats(0, 0, 0, 0, False)

        original_length = len(text)
        original_lines = text.count("\n")

        cleaned_text = text
        unicode_normalized = False

        # Unicode normalization
        if normalize_unicode:
            cleaned_text = self._normalize_unicode(cleaned_text)
            unicode_normalized = True

        # Remove or fix common PDF extraction artifacts
        cleaned_text = self._fix_pdf_artifacts(cleaned_text)

        # Handle whitespace
        if remove_extra_whitespace:
            cleaned_text = self._normalize_whitespace(cleaned_text, preserve_structure)

        # Remove special characters if requested
        if remove_special_chars:
            cleaned_text = self._remove_special_characters(cleaned_text)

        # Final cleanup (preserve structure if requested)
        cleaned_text = self._final_cleanup(cleaned_text, preserve_structure)

        # Calculate statistics
        cleaned_length = len(cleaned_text)
        cleaned_lines = cleaned_text.count("\n")
        lines_removed = max(0, original_lines - cleaned_lines)
        characters_removed = max(0, original_length - cleaned_length)

        stats = TextCleaningStats(
            original_length=original_length,
            cleaned_length=cleaned_length,
            lines_removed=lines_removed,
            characters_removed=characters_removed,
            unicode_normalized=unicode_normalized,
        )

        self.logger.debug(
            f"Text cleaning completed: {original_length} -> {cleaned_length} chars "
            f"({stats.compression_ratio:.2%} retained)"
        )

        return cleaned_text, stats

    def split_into_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using simple heuristics.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        if not text:
            return []

        # Simple sentence splitting pattern
        sentence_pattern = r"(?<=[.!?])\s+"
        sentences = re.split(sentence_pattern, text)

        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter very short sentences
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def extract_paragraphs(self, text: str) -> list[str]:
        """
        Extract paragraphs from text.

        Args:
            text: Text to process

        Returns:
            List of paragraphs
        """
        if not text:
            return []

        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r"\n\s*\n", text)

        # Clean and filter paragraphs
        cleaned_paragraphs = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if paragraph and len(paragraph) > 20:  # Filter very short paragraphs
                # Replace single newlines with spaces within paragraphs
                paragraph = re.sub(r"\n+", " ", paragraph)
                cleaned_paragraphs.append(paragraph)

        return cleaned_paragraphs

    def chunk_text(
        self, text: str, source_info: Optional[dict[str, Any]] = None
    ) -> list[TextChunk]:
        """
        Split text into chunks using the configured text splitter.

        Args:
            text: Text to chunk
            source_info: Optional source information to include in chunks

        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []

        # Split text using LangChain's RecursiveCharacterTextSplitter
        text_chunks = self.text_splitter.split_text(text)

        # Create TextChunk objects with metadata
        chunks = []
        current_char = 0

        for i, chunk_content in enumerate(text_chunks):
            # Find the actual position of this chunk in the original text
            chunk_start = text.find(chunk_content, current_char)
            if chunk_start == -1:
                # Fallback if exact match not found
                chunk_start = current_char

            chunk_end = chunk_start + len(chunk_content)

            chunk = TextChunk(
                content=chunk_content,
                chunk_index=i,
                start_char=chunk_start,
                end_char=chunk_end,
                source_info=source_info,
            )

            chunks.append(chunk)
            current_char = chunk_end

        self.logger.debug(
            f"Split text into {len(chunks)} chunks "
            f"(avg size: {sum(c.length for c in chunks) / len(chunks):.0f} chars)"
        )

        return chunks

    def chunk_text_with_metadata(
        self,
        text: str,
        document_id: str,
        page_number: Optional[int] = None,
        additional_metadata: Optional[dict[str, Any]] = None,
    ) -> list[TextChunk]:
        """
        Chunk text with comprehensive metadata.

        Args:
            text: Text to chunk
            document_id: Unique identifier for the source document
            page_number: Page number if applicable
            additional_metadata: Additional metadata to include

        Returns:
            List of TextChunk objects with metadata
        """
        source_info = {
            "document_id": document_id,
            "page_number": page_number,
            **(additional_metadata or {}),
        }

        return self.chunk_text(text, source_info)

    def process_pdf_content(
        self,
        pdf_content: dict[int, str],
        clean_text: bool = True,
    ) -> list[TextChunk]:
        """
        Process PDF content from multiple pages.

        Args:
            pdf_content: Dictionary mapping page numbers to text content
            clean_text: Whether to clean the text before chunking

        Returns:
            List of TextChunk objects from all pages
        """
        all_chunks = []
        global_chunk_index = 0

        for page_num, page_text in sorted(pdf_content.items()):
            if not page_text or not page_text.strip():
                continue

            # Clean text if requested
            if clean_text:
                page_text, _ = self.clean_text(page_text)

            # Chunk the page text
            page_chunks = self.chunk_text(page_text)

            # Add page number and update chunk indices
            for chunk in page_chunks:
                chunk.page_number = page_num
                chunk.chunk_index = global_chunk_index
                global_chunk_index += 1

            all_chunks.extend(page_chunks)

        self.logger.info(
            f"Processed {len(pdf_content)} pages into {len(all_chunks)} chunks"
        )

        return all_chunks

    def validate_chunks(self, chunks: list[TextChunk]) -> dict[str, Any]:
        """
        Validate chunks and return statistics.

        Args:
            chunks: List of chunks to validate

        Returns:
            Dictionary containing validation results and statistics
        """
        if not chunks:
            return {
                "valid": True,
                "chunk_count": 0,
                "total_length": 0,
                "average_length": 0,
                "min_length": 0,
                "max_length": 0,
                "warnings": [],
            }

        warnings = []
        lengths = [chunk.length for chunk in chunks]

        # Check for chunks that are too small
        min_size = self.config.min_chunk_size
        small_chunks = [i for i, length in enumerate(lengths) if length < min_size]
        if small_chunks:
            warnings.append(
                f"Found {len(small_chunks)} chunks smaller than {min_size} characters"
            )

        # Check for chunks that are too large
        max_size = self.config.chunk_size * 1.5  # Allow some flexibility
        large_chunks = [i for i, length in enumerate(lengths) if length > max_size]
        if large_chunks:
            warnings.append(
                f"Found {len(large_chunks)} chunks larger than expected ({max_size} chars)"
            )

        # Check for empty chunks
        empty_chunks = [
            i for i, chunk in enumerate(chunks) if not chunk.content.strip()
        ]
        if empty_chunks:
            warnings.append(f"Found {len(empty_chunks)} empty chunks")

        return {
            "valid": len(warnings) == 0,
            "chunk_count": len(chunks),
            "total_length": sum(lengths),
            "average_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "warnings": warnings,
        }

    def remove_headers_footers(self, text: str, threshold: int = 3) -> str:
        """
        Attempt to remove repeated headers and footers.

        Args:
            text: Text to process
            threshold: Minimum number of repetitions to consider as header/footer

        Returns:
            Text with headers/footers removed
        """
        lines = text.split("\n")
        if len(lines) < 10:  # Too short to have meaningful headers/footers
            return text

        # Find repeated lines that might be headers/footers
        line_counts: dict[str, int] = {}
        for line in lines:
            line = line.strip()
            if line and len(line) < 100:  # Headers/footers are usually short
                line_counts[line] = line_counts.get(line, 0) + 1

        # Identify lines to remove
        lines_to_remove = set()
        for line, count in line_counts.items():
            if count >= threshold:
                lines_to_remove.add(line)

        # Remove identified headers/footers
        filtered_lines = []
        for line in lines:
            if line.strip() not in lines_to_remove:
                filtered_lines.append(line)

        if lines_to_remove:
            self.logger.debug(
                f"Removed {len(lines_to_remove)} repeated header/footer patterns"
            )

        return "\n".join(filtered_lines)

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        # Normalize to NFC form (canonical decomposition followed by canonical composition)
        normalized = unicodedata.normalize("NFC", text)

        # Replace common problematic characters
        replacements = {
            "\u2018": "'",  # Left single quotation mark
            "\u2019": "'",  # Right single quotation mark
            "\u201c": '"',  # Left double quotation mark
            "\u201d": '"',  # Right double quotation mark
            "\u2013": "-",  # En dash
            "\u2014": "--",  # Em dash
            "\u2026": "...",  # Horizontal ellipsis
            "\u00a0": " ",  # Non-breaking space
            "\u200b": "",  # Zero-width space
            "\ufeff": "",  # Byte order mark
        }

        for old_char, new_char in replacements.items():
            normalized = normalized.replace(old_char, new_char)

        return normalized

    def _fix_pdf_artifacts(self, text: str) -> str:
        """Fix common PDF extraction artifacts."""
        # Fix broken words (words split across lines with hyphens)
        text = re.sub(r"-\s*\n\s*", "", text)

        # Fix spacing issues around punctuation
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        # Fix spacing after sentence endings, but preserve paragraph breaks
        text = re.sub(
            r"([.!?])[ \t]+([A-Z])", r"\1 \2", text
        )  # Only match spaces/tabs, not newlines

        # Fix multiple spaces
        text = re.sub(r" {2,}", " ", text)

        # Fix page numbers and other artifacts (simple heuristic)
        return re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    def _normalize_whitespace(self, text: str, preserve_structure: bool = True) -> str:
        """Normalize whitespace while optionally preserving structure."""
        if preserve_structure:
            # Preserve paragraph breaks but normalize other whitespace
            paragraphs = text.split("\n\n")
            normalized_paragraphs = []

            for paragraph in paragraphs:
                # Normalize whitespace within paragraph
                paragraph = re.sub(r"\s+", " ", paragraph.strip())
                if paragraph:
                    normalized_paragraphs.append(paragraph)

            return "\n\n".join(normalized_paragraphs)
        # Aggressive whitespace normalization
        return re.sub(r"\s+", " ", text.strip())

    def _remove_special_characters(self, text: str) -> str:
        """Remove special characters while preserving basic punctuation."""
        # Keep alphanumeric, basic punctuation, and whitespace
        pattern = r'[^a-zA-Z0-9\s.,!?;:()\-"\'\n]'
        return re.sub(pattern, "", text)

    def _final_cleanup(self, text: str, preserve_structure: bool = True) -> str:
        """Perform final cleanup operations."""
        if preserve_structure:
            # Preserve paragraph structure - only remove excessive newlines
            cleaned = re.sub(r"\n{3,}", "\n\n", text)
            return cleaned.strip()
        # Remove empty lines
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        # Join lines and ensure proper spacing
        cleaned = "\n".join(lines)
        # Remove excessive newlines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def get_text_statistics(self, text: str) -> dict[str, Any]:
        """
        Get comprehensive statistics about text content.

        Args:
            text: Text to analyze

        Returns:
            Dictionary containing text statistics
        """
        if not text:
            return {
                "character_count": 0,
                "word_count": 0,
                "line_count": 0,
                "paragraph_count": 0,
                "sentence_count": 0,
                "average_words_per_sentence": 0,
                "average_chars_per_word": 0,
            }

        # Basic counts
        character_count = len(text)
        words = text.split()
        word_count = len(words)
        line_count = text.count("\n") + 1

        # Paragraph count (double newlines)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        paragraph_count = len(paragraphs)

        # Sentence count (approximate)
        sentences = self.split_into_sentences(text)
        sentence_count = len(sentences)

        # Averages
        avg_words_per_sentence = (sentence_count and word_count / sentence_count) or 0
        avg_chars_per_word = (word_count and character_count / word_count) or 0

        return {
            "character_count": character_count,
            "word_count": word_count,
            "line_count": line_count,
            "paragraph_count": paragraph_count,
            "sentence_count": sentence_count,
            "average_words_per_sentence": round(avg_words_per_sentence, 2),
            "average_chars_per_word": round(avg_chars_per_word, 2),
        }
