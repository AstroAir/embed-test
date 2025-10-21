"""Tests for TextProcessor class."""

from unittest.mock import ANY, Mock, patch

from pdf_vector_system.config.settings import TextProcessingConfig
from pdf_vector_system.pdf.text_processor import (
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

    @patch("pdf_vector_system.pdf.text_processor.RecursiveCharacterTextSplitter")
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

    @patch("pdf_vector_system.pdf.text_processor.RecursiveCharacterTextSplitter")
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

    @patch("pdf_vector_system.pdf.text_processor.RecursiveCharacterTextSplitter")
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
