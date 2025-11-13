"""Tests for TextProcessor class."""

from unittest.mock import ANY, Mock, patch

from vectorflow.core.config.settings import TextProcessingConfig
from vectorflow.core.pdf.text_processor import (
    TextChunk,
    TextCleaningStats,
    TextProcessor,
)


class TestTextCleaningStats:
    """Test TextCleaningStats dataclass."""

    def test_creation(self):
        """Test TextCleaningStats creation."""
        stats = TextCleaningStats(
            original_length=100,
            cleaned_length=80,
            lines_removed=2,
            characters_removed=20,
            unicode_normalized=True,
        )

        assert stats.original_length == 100
        assert stats.cleaned_length == 80
        assert stats.lines_removed == 2
        assert stats.characters_removed == 20
        assert stats.unicode_normalized is True

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        stats = TextCleaningStats(
            original_length=100,
            cleaned_length=80,
            lines_removed=0,
            characters_removed=20,
            unicode_normalized=False,
        )

        assert stats.compression_ratio == 0.8

    def test_compression_ratio_zero_original(self):
        """Test compression ratio with zero original length."""
        stats = TextCleaningStats(
            original_length=0,
            cleaned_length=0,
            lines_removed=0,
            characters_removed=0,
            unicode_normalized=False,
        )

        assert stats.compression_ratio == 1.0


class TestTextChunk:
    """Test TextChunk dataclass."""

    def test_creation(self):
        """Test TextChunk creation."""
        chunk = TextChunk(
            content="This is a test chunk.", chunk_index=0, start_char=0, end_char=21
        )

        assert chunk.content == "This is a test chunk."
        assert chunk.chunk_index == 0
        assert chunk.start_char == 0
        assert chunk.end_char == 21

    def test_length_property(self):
        """Test length property."""
        chunk = TextChunk(
            content="Hello world", chunk_index=0, start_char=0, end_char=11
        )

        assert chunk.length == 11

    def test_to_dict(self):
        """Test conversion to dictionary."""
        chunk = TextChunk(
            content="Test content", chunk_index=1, start_char=10, end_char=22
        )
        chunk.page_number = 2

        chunk_dict = chunk.to_dict()

        assert chunk_dict["content"] == "Test content"
        assert chunk_dict["chunk_index"] == 1
        assert chunk_dict["start_char"] == 10
        assert chunk_dict["end_char"] == 22
        assert chunk_dict["length"] == 12
        assert chunk_dict["page_number"] == 2


class TestTextProcessor:
    """Test TextProcessor class."""

    def test_initialization(self):
        """Test TextProcessor initialization."""
        config = TextProcessingConfig(
            chunk_size=500, chunk_overlap=50, min_chunk_size=25
        )
        processor = TextProcessor(config)

        assert processor.config == config
        assert processor.config.chunk_size == 500
        assert processor.config.chunk_overlap == 50
        assert processor.config.min_chunk_size == 25

    def test_initialization_with_defaults(self):
        """Test TextProcessor initialization with default config."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        assert processor.config.chunk_size == 1000
        assert processor.config.chunk_overlap == 100
        assert processor.config.min_chunk_size == 50

    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        dirty_text = "  This is   a test   with extra   spaces.  "
        cleaned_text, stats = processor.clean_text(dirty_text)

        assert cleaned_text != dirty_text
        assert "This is a test" in cleaned_text
        assert stats.original_length == len(dirty_text)
        assert stats.cleaned_length == len(cleaned_text)
        assert stats.characters_removed >= 0

    def test_clean_text_empty(self):
        """Test text cleaning with empty text."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        cleaned_text, stats = processor.clean_text("")

        assert cleaned_text == ""
        assert stats.original_length == 0
        assert stats.cleaned_length == 0
        assert stats.characters_removed == 0
        assert stats.lines_removed == 0

    def test_clean_text_unicode_normalization(self):
        """Test text cleaning with unicode normalization."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        unicode_text = "café naïve résumé"
        cleaned_text, stats = processor.clean_text(unicode_text, normalize_unicode=True)

        assert cleaned_text is not None
        assert stats.unicode_normalized is True

    def test_clean_text_no_unicode_normalization(self):
        """Test text cleaning without unicode normalization."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        unicode_text = "café naïve résumé"
        cleaned_text, stats = processor.clean_text(
            unicode_text, normalize_unicode=False
        )

        assert cleaned_text is not None
        assert stats.unicode_normalized is False

    def test_clean_text_remove_special_chars(self):
        """Test text cleaning with special character removal."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        text_with_special = "Hello @#$% world! How are you?"
        cleaned_text, _stats = processor.clean_text(
            text_with_special, remove_special_chars=True
        )

        assert cleaned_text != text_with_special
        # Should remove some special characters
        assert "@#$%" not in cleaned_text

    def test_clean_text_preserve_structure(self):
        """Test text cleaning with structure preservation."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        structured_text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        cleaned_text, _stats = processor.clean_text(
            structured_text, preserve_structure=True
        )

        # Should preserve paragraph breaks
        assert "\n\n" in cleaned_text or cleaned_text.count("\n") >= 2

    def test_clean_text_no_preserve_structure(self):
        """Test text cleaning without structure preservation."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        structured_text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        cleaned_text, _stats = processor.clean_text(
            structured_text, preserve_structure=False
        )

        # Structure may be modified
        assert cleaned_text is not None

    @patch("vectorflow.pdf.text_processor.RecursiveCharacterTextSplitter")
    def test_chunk_text_basic(self, mock_splitter_class):
        """Test basic text chunking."""
        # Mock the text splitter
        mock_splitter = Mock()
        mock_splitter.split_text.return_value = [
            "This is chunk 1.",
            "This is chunk 2.",
            "This is chunk 3.",
        ]
        mock_splitter_class.return_value = mock_splitter

        config = TextProcessingConfig(chunk_size=100, chunk_overlap=10)
        processor = TextProcessor(config)

        text = "This is a long text that should be split into multiple chunks."
        chunks = processor.chunk_text(text)

        assert len(chunks) == 3
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        assert chunks[0].content == "This is chunk 1."
        assert chunks[1].content == "This is chunk 2."
        assert chunks[2].content == "This is chunk 3."

        # Verify splitter was configured correctly
        mock_splitter_class.assert_called_once_with(
            chunk_size=100,
            chunk_overlap=10,
            separators=["\n\n", "\n", " ", ""],
            length_function=ANY,
            is_separator_regex=ANY,
        )

    @patch("vectorflow.pdf.text_processor.RecursiveCharacterTextSplitter")
    def test_chunk_text_custom_parameters(self, mock_splitter_class):
        """Test text chunking with custom parameters."""
        mock_splitter = Mock()
        mock_splitter.split_text.return_value = ["Chunk 1", "Chunk 2"]
        mock_splitter_class.return_value = mock_splitter

        config = TextProcessingConfig(
            chunk_size=200, chunk_overlap=20, min_chunk_size=15
        )
        processor = TextProcessor(config)

        text = "Test text for chunking."
        chunks = processor.chunk_text(text)

        assert len(chunks) == 2

        # Should use custom parameters
        mock_splitter_class.assert_called_once_with(
            chunk_size=200,
            chunk_overlap=20,
            separators=["\n\n", "\n", " ", ""],
            length_function=ANY,
            is_separator_regex=ANY,
        )

    def test_chunk_text_empty(self):
        """Test text chunking with empty text."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        chunks = processor.chunk_text("")

        assert len(chunks) == 0

    @patch("vectorflow.pdf.text_processor.RecursiveCharacterTextSplitter")
    def test_chunk_text_character_positions(self, mock_splitter_class):
        """Test that chunks have correct character positions."""
        mock_splitter = Mock()
        mock_splitter.split_text.return_value = ["First chunk", "Second chunk"]
        mock_splitter_class.return_value = mock_splitter

        config = TextProcessingConfig()
        processor = TextProcessor(config)

        text = "First chunk Second chunk"
        chunks = processor.chunk_text(text)

        assert len(chunks) == 2
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1

        # Character positions should be calculated
        assert chunks[0].start_char >= 0
        assert chunks[0].end_char > chunks[0].start_char
        assert chunks[1].start_char >= chunks[0].end_char
        assert chunks[1].end_char > chunks[1].start_char

    def test_process_pdf_content_basic(self):
        """Test processing PDF content."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        pdf_content = {
            1: "This is page 1 content.",
            2: "This is page 2 content.",
            3: "This is page 3 content.",
        }

        with patch.object(processor, "chunk_text") as mock_chunk:
            # Mock chunking to return simple chunks
            mock_chunk.side_effect = [
                [TextChunk("Page 1 chunk", 0, 0, 13)],
                [TextChunk("Page 2 chunk", 0, 0, 13)],
                [TextChunk("Page 3 chunk", 0, 0, 13)],
            ]

            chunks = processor.process_pdf_content(pdf_content)

            assert len(chunks) == 3
            assert all(hasattr(chunk, "page_number") for chunk in chunks)
            assert chunks[0].page_number == 1
            assert chunks[1].page_number == 2
            assert chunks[2].page_number == 3

    def test_process_pdf_content_with_cleaning(self):
        """Test processing PDF content with text cleaning."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        pdf_content = {1: "  Dirty   text   with   spaces  "}

        with patch.object(processor, "clean_text") as mock_clean:
            with patch.object(processor, "chunk_text") as mock_chunk:
                mock_clean.return_value = ("Clean text", Mock())
                mock_chunk.return_value = [TextChunk("Clean text", 0, 0, 10)]

                processor.process_pdf_content(pdf_content, clean_text=True)

                mock_clean.assert_called_once()
                mock_chunk.assert_called_once_with("Clean text")

    def test_process_pdf_content_without_cleaning(self):
        """Test processing PDF content without text cleaning."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        pdf_content = {1: "Raw text content"}

        with patch.object(processor, "clean_text") as mock_clean:
            with patch.object(processor, "chunk_text") as mock_chunk:
                mock_chunk.return_value = [TextChunk("Raw text content", 0, 0, 16)]

                processor.process_pdf_content(pdf_content, clean_text=False)

                mock_clean.assert_not_called()
                mock_chunk.assert_called_once_with("Raw text content")

    def test_process_pdf_content_empty(self):
        """Test processing empty PDF content."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        chunks = processor.process_pdf_content({})

        assert len(chunks) == 0

    def test_process_pdf_content_chunk_indexing(self):
        """Test that chunks are indexed correctly across pages."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        pdf_content = {1: "Page 1 content", 2: "Page 2 content"}

        with patch.object(processor, "chunk_text") as mock_chunk:
            # Mock each page to return 2 chunks
            mock_chunk.side_effect = [
                [
                    TextChunk("Page 1 chunk 1", 0, 0, 15),
                    TextChunk("Page 1 chunk 2", 1, 15, 30),
                ],
                [
                    TextChunk("Page 2 chunk 1", 0, 0, 15),
                    TextChunk("Page 2 chunk 2", 1, 15, 30),
                ],
            ]

            chunks = processor.process_pdf_content(pdf_content)

            # Should have 4 chunks total with sequential indices
            assert len(chunks) == 4
            assert chunks[0].chunk_index == 0
            assert chunks[1].chunk_index == 1
            assert chunks[2].chunk_index == 2
            assert chunks[3].chunk_index == 3

            # Page numbers should be preserved
            assert chunks[0].page_number == 1
            assert chunks[1].page_number == 1
            assert chunks[2].page_number == 2
            assert chunks[3].page_number == 2


class TestTextProcessorComprehensive:
    """Comprehensive tests for TextProcessor to achieve 80%+ coverage."""

    def test_text_splitter_property_lazy_initialization(self):
        """Test that text_splitter is lazily initialized."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        # Should be None initially
        assert processor._text_splitter is None

        # Accessing property should create it
        splitter = processor.text_splitter
        assert splitter is not None
        assert processor._text_splitter is not None

        # Second access should return same instance
        splitter2 = processor.text_splitter
        assert splitter2 is splitter

    def test_split_into_sentences_basic(self):
        """Test basic sentence splitting."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        text = "This is sentence one. This is sentence two! Is this sentence three?"
        sentences = processor.split_into_sentences(text)

        assert len(sentences) == 3
        assert "sentence one" in sentences[0]
        assert "sentence two" in sentences[1]
        assert "sentence three" in sentences[2]

    def test_split_into_sentences_empty(self):
        """Test sentence splitting with empty text."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        sentences = processor.split_into_sentences("")
        assert sentences == []

    def test_split_into_sentences_filters_short(self):
        """Test that very short sentences are filtered out."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        text = "Hi. This is a longer sentence with more content. Ok."
        sentences = processor.split_into_sentences(text)

        # Short sentences (< 10 chars) should be filtered
        assert len(sentences) == 1
        assert "longer sentence" in sentences[0]

    def test_extract_paragraphs_basic(self):
        """Test basic paragraph extraction."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        text = "This is paragraph one.\n\nThis is paragraph two with more content.\n\nParagraph three here."
        paragraphs = processor.extract_paragraphs(text)

        assert len(paragraphs) == 3
        assert "paragraph one" in paragraphs[0].lower()
        assert "paragraph two" in paragraphs[1].lower()
        assert "paragraph three" in paragraphs[2].lower()

    def test_extract_paragraphs_empty(self):
        """Test paragraph extraction with empty text."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        paragraphs = processor.extract_paragraphs("")
        assert paragraphs == []

    def test_extract_paragraphs_filters_short(self):
        """Test that very short paragraphs are filtered out."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        text = "Short.\n\nThis is a longer paragraph with sufficient content.\n\nTiny."
        paragraphs = processor.extract_paragraphs(text)

        # Short paragraphs (< 20 chars) should be filtered
        assert len(paragraphs) == 1
        assert "longer paragraph" in paragraphs[0]

    def test_extract_paragraphs_replaces_newlines(self):
        """Test that single newlines within paragraphs are replaced with spaces."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        text = "This is a paragraph\nwith line breaks\ninside it that should become spaces."
        paragraphs = processor.extract_paragraphs(text)

        assert len(paragraphs) == 1
        assert "\n" not in paragraphs[0]
        assert "paragraph with line breaks inside" in paragraphs[0]

    def test_chunk_text_with_metadata(self):
        """Test chunking with metadata."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        with patch.object(processor, "chunk_text") as mock_chunk:
            mock_chunk.return_value = [
                TextChunk("Chunk 1", 0, 0, 7),
                TextChunk("Chunk 2", 1, 7, 14),
            ]

            processor.chunk_text_with_metadata(
                "Test text",
                document_id="doc123",
                page_number=5,
                additional_metadata={"author": "Test Author"},
            )

            # Verify chunk_text was called with correct source_info
            call_args = mock_chunk.call_args
            assert call_args[0][0] == "Test text"
            source_info = call_args[0][1]
            assert source_info["document_id"] == "doc123"
            assert source_info["page_number"] == 5
            assert source_info["author"] == "Test Author"

    def test_validate_chunks_empty(self):
        """Test chunk validation with empty list."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        result = processor.validate_chunks([])

        assert result["valid"] is True
        assert result["chunk_count"] == 0
        assert result["total_length"] == 0
        assert result["average_length"] == 0
        assert result["warnings"] == []

    def test_validate_chunks_success(self):
        """Test chunk validation with valid chunks."""
        config = TextProcessingConfig(chunk_size=100, min_chunk_size=20)
        processor = TextProcessor(config)

        chunks = [
            TextChunk("A" * 50, 0, 0, 50),
            TextChunk("B" * 60, 1, 50, 110),
            TextChunk("C" * 70, 2, 110, 180),
        ]

        result = processor.validate_chunks(chunks)

        assert result["valid"] is True
        assert result["chunk_count"] == 3
        assert result["total_length"] == 180
        assert result["average_length"] == 60
        assert result["min_length"] == 50
        assert result["max_length"] == 70
        assert result["warnings"] == []

    def test_validate_chunks_too_small(self):
        """Test chunk validation detects chunks that are too small."""
        config = TextProcessingConfig(min_chunk_size=50)
        processor = TextProcessor(config)

        chunks = [
            TextChunk("Small", 0, 0, 5),  # Too small
            TextChunk("A" * 60, 1, 5, 65),  # OK
        ]

        result = processor.validate_chunks(chunks)

        assert result["valid"] is False
        assert len(result["warnings"]) == 1
        assert "smaller than 50" in result["warnings"][0]

    def test_validate_chunks_too_large(self):
        """Test chunk validation detects chunks that are too large."""
        config = TextProcessingConfig(chunk_size=100)
        processor = TextProcessor(config)

        chunks = [
            TextChunk("A" * 200, 0, 0, 200),  # Too large (>150)
            TextChunk("B" * 80, 1, 200, 280),  # OK
        ]

        result = processor.validate_chunks(chunks)

        assert result["valid"] is False
        assert len(result["warnings"]) == 1
        assert "larger than expected" in result["warnings"][0]

    def test_validate_chunks_empty_content(self):
        """Test chunk validation detects empty chunks."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        chunks = [
            TextChunk("Valid content", 0, 0, 13),
            TextChunk("   ", 1, 13, 16),  # Empty (just whitespace)
        ]

        result = processor.validate_chunks(chunks)

        assert result["valid"] is False
        # Should detect either empty chunks or chunks that are too small
        assert len(result["warnings"]) >= 1

    def test_remove_headers_footers_basic(self):
        """Test removal of repeated headers and footers."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        # Text with repeated header and footer
        text = """Header Text
Page 1 content here
Footer Text
Header Text
Page 2 content here
Footer Text
Header Text
Page 3 content here
Footer Text"""

        result = processor.remove_headers_footers(text, threshold=3)

        # Content should still be present after processing
        assert "Page 1 content" in result
        assert "Page 2 content" in result
        assert "Page 3 content" in result
        # Method should process and return result
        assert isinstance(result, str)
        assert len(result) > 0

    def test_remove_headers_footers_short_text(self):
        """Test header/footer removal with text too short to process."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        short_text = "Line 1\nLine 2\nLine 3"
        result = processor.remove_headers_footers(short_text)

        # Should return unchanged for short texts
        assert result == short_text

    def test_remove_headers_footers_no_repetition(self):
        """Test header/footer removal when there are no repeated lines."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        text = "\n".join([f"Unique line {i}" for i in range(15)])
        result = processor.remove_headers_footers(text)

        # Should return unchanged when no repetitions
        assert result == text

    def test_normalize_unicode(self):
        """Test Unicode normalization."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        # Text with special Unicode characters
        text = "\u2018quoted\u2019 \u201ccurly quotes\u201d \u2013 en-dash \u2014 em-dash \u2026 ellipsis"
        normalized = processor._normalize_unicode(text)

        # Should convert to ASCII equivalents
        assert "'" in normalized
        assert '"' in normalized
        assert "-" in normalized or "--" in normalized
        assert "..." in normalized

    def test_fix_pdf_artifacts(self):
        """Test fixing common PDF extraction artifacts."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        # Text with hyphenated line breaks
        text = "This is a bro-\nken word that should be fixed."
        fixed = processor._fix_pdf_artifacts(text)

        assert "broken" in fixed
        assert "bro-\n" not in fixed

    def test_fix_pdf_artifacts_spacing(self):
        """Test fixing spacing issues."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        text = "Word  ,with   bad  .spacing !"
        fixed = processor._fix_pdf_artifacts(text)

        # Should fix spacing around punctuation
        assert " ," not in fixed
        assert " ." not in fixed
        assert " !" not in fixed

    def test_fix_pdf_artifacts_page_numbers(self):
        """Test removal of standalone page numbers."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        text = "Content here\n   42   \nMore content"
        fixed = processor._fix_pdf_artifacts(text)

        # Page number should be removed
        lines = fixed.split("\n")
        assert not any(line.strip() == "42" for line in lines)

    def test_normalize_whitespace_preserve_structure(self):
        """Test whitespace normalization while preserving structure."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        text = "Paragraph 1\n  with   extra    spaces.\n\nParagraph 2\nalso  with   spaces."
        normalized = processor._normalize_whitespace(text, preserve_structure=True)

        # Should preserve paragraph breaks
        assert "\n\n" in normalized
        # Should normalize whitespace within paragraphs
        assert "   " not in normalized

    def test_normalize_whitespace_no_preserve(self):
        """Test aggressive whitespace normalization."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        text = "Text\n\nwith\n\nmultiple\n\nparagraphs   and    spaces."
        normalized = processor._normalize_whitespace(text, preserve_structure=False)

        # Should not preserve paragraph breaks
        assert "\n\n" not in normalized
        # Should be single-line with single spaces
        assert "   " not in normalized

    def test_remove_special_characters(self):
        """Test removal of special characters."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        text = 'Hello @#$% world! Keep this: punctuation, (parentheses) and "quotes"'
        cleaned = processor._remove_special_characters(text)

        # Special chars should be removed
        assert "@" not in cleaned
        assert "#" not in cleaned
        assert "$" not in cleaned
        assert "%" not in cleaned

        # Basic punctuation should be kept
        assert "!" in cleaned
        assert "," in cleaned
        assert ":" in cleaned
        assert "(" in cleaned
        assert '"' in cleaned

    def test_final_cleanup_preserve_structure(self):
        """Test final cleanup with structure preservation."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        text = "Paragraph 1.\n\n\n\nParagraph 2.\n\n\n\nParagraph 3."
        cleaned = processor._final_cleanup(text, preserve_structure=True)

        # Should reduce to max 2 consecutive newlines
        assert "\n\n\n" not in cleaned
        assert "\n\n" in cleaned

    def test_final_cleanup_no_preserve(self):
        """Test final cleanup without structure preservation."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        text = "Line 1\n\nLine 2\n\n\nLine 3"
        cleaned = processor._final_cleanup(text, preserve_structure=False)

        # Should remove empty lines and excessive newlines
        assert "\n\n\n" not in cleaned

    def test_get_text_statistics_empty(self):
        """Test text statistics with empty text."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        stats = processor.get_text_statistics("")

        assert stats["character_count"] == 0
        assert stats["word_count"] == 0
        assert stats["line_count"] == 0
        assert stats["paragraph_count"] == 0
        assert stats["sentence_count"] == 0
        assert stats["average_words_per_sentence"] == 0
        assert stats["average_chars_per_word"] == 0

    def test_get_text_statistics_full(self):
        """Test comprehensive text statistics."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        text = "This is sentence one. This is sentence two!\n\nThis is a new paragraph. It has multiple sentences."
        stats = processor.get_text_statistics(text)

        assert stats["character_count"] > 0
        assert stats["word_count"] > 10
        assert stats["line_count"] >= 2
        assert stats["paragraph_count"] == 2
        assert stats["sentence_count"] >= 3
        assert stats["average_words_per_sentence"] > 0
        assert stats["average_chars_per_word"] > 0

    def test_chunk_text_empty_whitespace(self):
        """Test chunking with only whitespace."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        chunks = processor.chunk_text("   \n  \t  ")

        assert len(chunks) == 0

    def test_chunk_text_source_info(self):
        """Test that source_info is preserved in chunks."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        source_info = {"doc_id": "test123", "page": 5}

        with patch.object(processor.text_splitter, "split_text") as mock_split:
            mock_split.return_value = ["Chunk 1", "Chunk 2"]

            chunks = processor.chunk_text("Test text", source_info=source_info)

            assert len(chunks) == 2
            assert chunks[0].source_info == source_info
            assert chunks[1].source_info == source_info

    def test_chunk_text_chunk_not_found_fallback(self):
        """Test chunk position fallback when exact match not found."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        # Create a scenario where find() might not find exact match
        with patch.object(processor.text_splitter, "split_text") as mock_split:
            # Return chunks that might not be exact substrings
            mock_split.return_value = ["modified chunk", "another chunk"]

            chunks = processor.chunk_text("original text here")

            # Should still create chunks with fallback positions
            assert len(chunks) == 2
            assert chunks[0].start_char >= 0
            assert chunks[1].start_char >= chunks[0].end_char

    def test_process_pdf_content_skips_empty_pages(self):
        """Test that empty pages are skipped during processing."""
        config = TextProcessingConfig()
        processor = TextProcessor(config)

        pdf_content = {
            1: "Page 1 content",
            2: "",  # Empty page
            3: "   ",  # Whitespace only
            4: "Page 4 content",
        }

        with patch.object(processor, "chunk_text") as mock_chunk:
            mock_chunk.return_value = [TextChunk("Content", 0, 0, 7)]

            processor.process_pdf_content(pdf_content)

            # Should only process pages 1 and 4
            assert mock_chunk.call_count == 2

    def test_text_chunk_to_dict_no_page_number(self):
        """Test TextChunk to_dict when page_number is None."""
        chunk = TextChunk("Test", 0, 0, 4)
        # Don't set page_number

        result = chunk.to_dict()

        assert "page_number" not in result
        assert result["content"] == "Test"

    def test_text_chunk_to_dict_with_source_info(self):
        """Test TextChunk to_dict with source_info."""
        chunk = TextChunk("Test", 0, 0, 4, source_info={"key": "value"})

        result = chunk.to_dict()

        assert result["source_info"] == {"key": "value"}

    def test_text_chunk_to_dict_no_source_info(self):
        """Test TextChunk to_dict when source_info is None."""
        chunk = TextChunk("Test", 0, 0, 4)

        result = chunk.to_dict()

        assert result["source_info"] == {}
