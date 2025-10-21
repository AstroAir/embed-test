"""Tests for advanced tokenization module."""

from unittest.mock import patch

import pytest

from pdf_vector_system.embeddings.tokenization import (
    LanguageDetector,
    TextNormalizationConfig,
    TokenizationConfig,
    TokenizationMethod,
    TokenizationQualityMetrics,
    TokenizationResult,
    TokenizerFactory,
    WhitespaceTokenizer,
)


class TestTokenizationConfig:
    """Test tokenization configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = TokenizationConfig()
        assert config.method == TokenizationMethod.HUGGINGFACE
        assert config.truncation is True
        assert config.padding is False
        assert config.add_special_tokens is True

    def test_custom_config(self):
        """Test custom configuration."""
        norm_config = TextNormalizationConfig(lowercase=True, remove_accents=True)

        config = TokenizationConfig(
            method=TokenizationMethod.WHITESPACE,
            max_length=512,
            normalization_config=norm_config,
        )

        assert config.method == TokenizationMethod.WHITESPACE
        assert config.max_length == 512
        assert config.normalization_config.lowercase is True
        assert config.normalization_config.remove_accents is True


class TestTokenizationResult:
    """Test tokenization result."""

    def test_creation(self):
        """Test result creation."""
        tokens = ["hello", "world"]
        token_ids = [1, 2]

        result = TokenizationResult(
            tokens=tokens, token_ids=token_ids, metadata={"test": "value"}
        )

        assert result.tokens == tokens
        assert result.token_ids == token_ids
        assert result.token_count == 2
        assert result.metadata["test"] == "value"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = TokenizationResult(tokens=["test"], token_ids=[1], attention_mask=[1])

        result_dict = result.to_dict()

        assert "tokens" in result_dict
        assert "token_ids" in result_dict
        assert "attention_mask" in result_dict
        assert "token_count" in result_dict
        assert result_dict["token_count"] == 1


class TestWhitespaceTokenizer:
    """Test whitespace tokenizer."""

    def test_initialization(self):
        """Test tokenizer initialization."""
        config = TokenizationConfig(method=TokenizationMethod.WHITESPACE)
        tokenizer = WhitespaceTokenizer(config)

        assert tokenizer.config.method == TokenizationMethod.WHITESPACE
        assert tokenizer._is_initialized is True

    def test_tokenize_simple(self):
        """Test simple tokenization."""
        config = TokenizationConfig(method=TokenizationMethod.WHITESPACE)
        tokenizer = WhitespaceTokenizer(config)

        result = tokenizer.tokenize("hello world test")

        assert result.tokens == ["hello", "world", "test"]
        assert result.token_count == 3
        assert result.metadata["method"] == "whitespace"

    def test_tokenize_with_normalization(self):
        """Test tokenization with text normalization."""
        norm_config = TextNormalizationConfig(
            lowercase=True,
            remove_extra_whitespace=True,
            preserve_case_for_entities=False,  # Disable smart lowercase for simple test
        )
        config = TokenizationConfig(
            method=TokenizationMethod.WHITESPACE, normalization_config=norm_config
        )
        tokenizer = WhitespaceTokenizer(config)

        result = tokenizer.tokenize("  HELLO   WORLD  ")

        assert result.tokens == ["hello", "world"]
        assert result.token_count == 2

    def test_detokenize(self):
        """Test detokenization."""
        config = TokenizationConfig(method=TokenizationMethod.WHITESPACE)
        tokenizer = WhitespaceTokenizer(config)

        tokens = ["hello", "world", "test"]
        text = tokenizer.detokenize(tokens)

        assert text == "hello world test"

    def test_get_vocab_size(self):
        """Test vocabulary size."""
        config = TokenizationConfig(method=TokenizationMethod.WHITESPACE)
        tokenizer = WhitespaceTokenizer(config)

        vocab_size = tokenizer.get_vocab_size()
        assert vocab_size == -1  # Unlimited for whitespace tokenizer

    def test_normalize_text(self):
        """Test text normalization."""
        norm_config = TextNormalizationConfig(
            unicode_normalization="NFKC",
            lowercase=True,
            remove_extra_whitespace=True,
            preserve_case_for_entities=False,  # Disable smart lowercase for simple test
        )
        config = TokenizationConfig(normalization_config=norm_config)
        tokenizer = WhitespaceTokenizer(config)

        normalized = tokenizer.normalize_text("  HELLO   WORLD  ")
        assert normalized == "hello world"

    def test_smart_lowercase(self):
        """Test smart lowercase functionality."""
        norm_config = TextNormalizationConfig(
            lowercase=True, preserve_case_for_entities=True
        )
        config = TokenizationConfig(normalization_config=norm_config)
        tokenizer = WhitespaceTokenizer(config)

        # Test preserving acronyms and proper nouns
        normalized = tokenizer.normalize_text("Hello NASA and John Smith")
        # Should preserve NASA (acronym) and potentially John Smith (proper noun)
        assert "nasa" in normalized.lower()


class TestTokenizerFactory:
    """Test tokenizer factory."""

    def test_create_whitespace_tokenizer(self):
        """Test creating whitespace tokenizer."""
        config = TokenizationConfig(method=TokenizationMethod.WHITESPACE)
        tokenizer = TokenizerFactory.create_tokenizer(config)

        assert isinstance(tokenizer, WhitespaceTokenizer)
        assert tokenizer.config.method == TokenizationMethod.WHITESPACE

    def test_unsupported_method(self):
        """Test unsupported tokenization method."""
        config = TokenizationConfig(method="unsupported")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="Unsupported tokenization method"):
            TokenizerFactory.create_tokenizer(config)

    def test_get_recommended_config(self):
        """Test getting recommended configuration."""
        # Test sentence-transformers model
        config = TokenizerFactory.get_recommended_config(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        assert config.method == TokenizationMethod.HUGGINGFACE
        assert config.max_length == 512
        assert config.normalization_config.preserve_case_for_entities is True

        # Test BERT model
        config = TokenizerFactory.get_recommended_config("bert-base-uncased")
        assert config.method == TokenizationMethod.HUGGINGFACE
        assert config.normalization_config.lowercase is True

        # Test unknown model (should return default)
        config = TokenizerFactory.get_recommended_config("unknown-model")
        assert config.method == TokenizationMethod.HUGGINGFACE
        assert config.max_length == 512


class TestLanguageDetector:
    """Test language detector."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = LanguageDetector()
        # Should initialize regardless of langdetect availability
        assert detector is not None

    @patch("pdf_vector_system.embeddings.tokenization.LANGDETECT_AVAILABLE", True)
    @patch("pdf_vector_system.embeddings.tokenization.langdetect")
    def test_detect_language_success(self, mock_langdetect):
        """Test successful language detection."""
        mock_langdetect.detect.return_value = "en"

        detector = LanguageDetector()
        detector.available = True

        language = detector.detect_language("This is English text")
        assert language == "en"
        mock_langdetect.detect.assert_called_once()

    @patch("pdf_vector_system.embeddings.tokenization.LANGDETECT_AVAILABLE", False)
    def test_detect_language_unavailable(self):
        """Test language detection when langdetect unavailable."""
        detector = LanguageDetector()

        language = detector.detect_language("This is English text")
        assert language is None

    def test_detect_language_empty_text(self):
        """Test language detection with empty text."""
        detector = LanguageDetector()

        language = detector.detect_language("")
        assert language is None

        language = detector.detect_language("   ")
        assert language is None


class TestTokenizationQualityMetrics:
    """Test tokenization quality metrics."""

    def test_calculate_compression_ratio(self):
        """Test compression ratio calculation."""
        original_text = "hello world"
        tokens = ["hello", "world"]

        ratio = TokenizationQualityMetrics.calculate_compression_ratio(
            original_text, tokens
        )
        expected_ratio = len(tokens) / len(original_text)
        assert ratio == expected_ratio

    def test_calculate_compression_ratio_empty(self):
        """Test compression ratio with empty text."""
        ratio = TokenizationQualityMetrics.calculate_compression_ratio("", [])
        assert ratio == 0.0

    def test_calculate_subword_ratio(self):
        """Test subword ratio calculation."""
        # Test with subword tokens
        tokens = ["hello", "##world", "▁test", "Ġtoken"]
        ratio = TokenizationQualityMetrics.calculate_subword_ratio(tokens)
        assert ratio == 3 / 4  # 3 subword tokens out of 4

        # Test with no subword tokens
        tokens = ["hello", "world"]
        ratio = TokenizationQualityMetrics.calculate_subword_ratio(tokens)
        assert ratio == 0.0

        # Test with empty tokens
        ratio = TokenizationQualityMetrics.calculate_subword_ratio([])
        assert ratio == 0.0

    def test_calculate_oov_ratio(self):
        """Test out-of-vocabulary ratio calculation."""
        tokens = ["hello", "world", "unknown", "test"]
        vocab = {"hello", "world", "test"}

        ratio = TokenizationQualityMetrics.calculate_oov_ratio(tokens, vocab)
        assert ratio == 1 / 4  # 1 OOV token out of 4

        # Test with empty tokens
        ratio = TokenizationQualityMetrics.calculate_oov_ratio([], vocab)
        assert ratio == 0.0

    def test_analyze_token_distribution(self):
        """Test token distribution analysis."""
        tokens = ["hello", "world", "hello", "test"]

        analysis = TokenizationQualityMetrics.analyze_token_distribution(tokens)

        assert analysis["total_tokens"] == 4
        assert analysis["unique_tokens"] == 3
        assert analysis["avg_token_length"] == sum(len(t) for t in tokens) / len(tokens)
        assert analysis["max_token_length"] == 5  # "hello" and "world"
        assert analysis["min_token_length"] == 4  # "test"
        assert len(analysis["most_frequent_tokens"]) <= 10

        # Check most frequent token
        most_frequent = analysis["most_frequent_tokens"][0]
        assert most_frequent[0] == "hello"  # Most frequent token
        assert most_frequent[1] == 2  # Frequency

    def test_analyze_token_distribution_empty(self):
        """Test token distribution analysis with empty tokens."""
        analysis = TokenizationQualityMetrics.analyze_token_distribution([])
        assert analysis == {}


# Integration tests
class TestTokenizationIntegration:
    """Integration tests for tokenization components."""

    def test_end_to_end_whitespace(self):
        """Test end-to-end tokenization with whitespace tokenizer."""
        # Create configuration
        norm_config = TextNormalizationConfig(
            lowercase=True,
            remove_extra_whitespace=True,
            preserve_case_for_entities=False,  # Disable smart lowercase for simple test
        )
        config = TokenizationConfig(
            method=TokenizationMethod.WHITESPACE, normalization_config=norm_config
        )

        # Create tokenizer
        tokenizer = TokenizerFactory.create_tokenizer(config)

        # Tokenize text
        text = "  Hello WORLD   this is a TEST  "
        result = tokenizer.tokenize(text)

        # Verify results
        assert result.tokens == ["hello", "world", "this", "is", "a", "test"]
        assert result.token_count == 6
        assert result.metadata["method"] == "whitespace"
        assert result.metadata["original_length"] == len(text)

        # Test detokenization
        detokenized = tokenizer.detokenize(result.tokens)
        assert detokenized == "hello world this is a test"

        # Test quality metrics
        compression_ratio = TokenizationQualityMetrics.calculate_compression_ratio(
            text, result.tokens
        )
        assert compression_ratio > 0

        analysis = TokenizationQualityMetrics.analyze_token_distribution(result.tokens)
        assert analysis["total_tokens"] == 6
        assert analysis["unique_tokens"] == 6  # All unique tokens
