"""Advanced tokenization module for modern embedding systems.

This module provides state-of-the-art tokenization methods including:
- Subword tokenization (BPE, SentencePiece, WordPiece)
- Advanced text normalization and preprocessing
- Language detection and multilingual support
- Token-aware text chunking
- Quality metrics and validation
"""

import re
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional, cast

import numpy as np

try:
    import sentencepiece as spm

    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    spm = None
    SENTENCEPIECE_AVAILABLE = False

try:
    from transformers import AutoTokenizer, PreTrainedTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoTokenizer = None
    PreTrainedTokenizer = None
    TRANSFORMERS_AVAILABLE = False

try:
    import langdetect

    LANGDETECT_AVAILABLE = True
except ImportError:
    langdetect = None
    LANGDETECT_AVAILABLE = False

from pdf_vector_system.core.utils.logging import LoggerMixin


class TokenizationMethod(Enum):
    """Supported tokenization methods."""

    WHITESPACE = "whitespace"
    SUBWORD_BPE = "subword_bpe"
    SENTENCEPIECE = "sentencepiece"
    WORDPIECE = "wordpiece"
    HUGGINGFACE = "huggingface"


@dataclass
class TokenizationResult:
    """Result of tokenization process."""

    tokens: list[str]
    token_ids: Optional[list[int]] = None
    attention_mask: Optional[list[int]] = None
    special_tokens_mask: Optional[list[int]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        """Get the number of tokens."""
        return len(self.tokens)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "tokens": self.tokens,
            "token_ids": self.token_ids,
            "attention_mask": self.attention_mask,
            "special_tokens_mask": self.special_tokens_mask,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }


@dataclass
class TextNormalizationConfig:
    """Configuration for text normalization."""

    unicode_normalization: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFKC"
    lowercase: bool = False
    remove_accents: bool = False
    remove_punctuation: bool = False
    remove_numbers: bool = False
    remove_extra_whitespace: bool = True
    preserve_case_for_entities: bool = True
    language_specific_rules: bool = True


@dataclass
class TokenizationConfig:
    """Configuration for tokenization."""

    method: TokenizationMethod = TokenizationMethod.HUGGINGFACE
    model_name: Optional[str] = None
    vocab_size: Optional[int] = None
    max_length: Optional[int] = None
    truncation: bool = True
    padding: bool = False
    add_special_tokens: bool = True
    return_attention_mask: bool = True
    return_token_type_ids: bool = False
    normalization_config: TextNormalizationConfig = field(
        default_factory=TextNormalizationConfig
    )


class BaseTokenizer(ABC, LoggerMixin):
    """Abstract base class for tokenizers."""

    def __init__(self, config: TokenizationConfig) -> None:
        self.config = config
        self._is_initialized = False

    @abstractmethod
    def tokenize(self, text: str) -> TokenizationResult:
        pass

    @abstractmethod
    def detokenize(self, tokens: list[str]) -> str:
        pass

    @abstractmethod
    def get_vocab_size(self) -> int:
        pass

    def normalize_text(self, text: str) -> str:
        if not text:
            return text

        normalized = text
        config = self.config.normalization_config

        if config.unicode_normalization:
            normalized = unicodedata.normalize(config.unicode_normalization, normalized)
        if config.remove_accents:
            normalized = self._remove_accents(normalized)
        if config.lowercase and not config.preserve_case_for_entities:
            normalized = normalized.lower()
        elif config.lowercase and config.preserve_case_for_entities:
            normalized = self._smart_lowercase(normalized)
        if config.remove_punctuation:
            normalized = re.sub(r"[^\w\s]", " ", normalized)
        if config.remove_numbers:
            normalized = re.sub(r"\d+", "", normalized)
        if config.remove_extra_whitespace:
            normalized = re.sub(r"\s+", " ", normalized.strip())
        return normalized

    def _remove_accents(self, text: str) -> str:
        return "".join(
            char
            for char in unicodedata.normalize("NFD", text)
            if unicodedata.category(char) != "Mn"
        )

    def _smart_lowercase(self, text: str) -> str:
        words = text.split()
        processed_words: list[str] = []
        for word in words:
            if (len(word) >= 2 and word.isupper()) or (
                word[0].isupper() and len(processed_words) > 0
            ):
                processed_words.append(word)
            else:
                processed_words.append(word.lower())
        return " ".join(processed_words)


class HuggingFaceTokenizer(BaseTokenizer):
    """HuggingFace transformers-based tokenizer."""

    def __init__(self, config: TokenizationConfig) -> None:
        super().__init__(config)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for HuggingFace tokenizer. "
                "Install with: pip install transformers"
            )
        if not config.model_name:
            raise ValueError("model_name is required for HuggingFace tokenizer")
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self._load_tokenizer()

    def _load_tokenizer(self) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, trust_remote_code=True
            )
            self._is_initialized = True
            self.logger.info(f"Loaded HuggingFace tokenizer: {self.config.model_name}")
        except Exception as e:
            error_msg = f"Failed to load tokenizer {self.config.model_name}: {e!s}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def tokenize(self, text: str) -> TokenizationResult:
        if not self._is_initialized or self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        normalized_text = self.normalize_text(text)
        encoding = self.tokenizer(
            normalized_text,
            max_length=self.config.max_length,
            truncation=self.config.truncation,
            padding=self.config.padding,
            add_special_tokens=self.config.add_special_tokens,
            return_attention_mask=self.config.return_attention_mask,
            return_token_type_ids=self.config.return_token_type_ids,
            return_tensors=None,
        )
        raw_input_ids = encoding["input_ids"]
        # Handle potential nested list (batch) vs flat list
        if raw_input_ids and isinstance(raw_input_ids[0], list):
            input_ids = raw_input_ids[0]
        else:
            input_ids = raw_input_ids
        input_ids_list = cast("list[int]", input_ids)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids_list)

        return TokenizationResult(
            tokens=tokens,
            token_ids=input_ids_list,
            attention_mask=(
                encoding.get("attention_mask")[0]
                if encoding.get("attention_mask")
                and isinstance(encoding["attention_mask"][0], list)
                else encoding.get("attention_mask")
            ),
            metadata={
                "model_name": self.config.model_name,
                "method": self.config.method.value,
                "normalized_text": normalized_text,
                "original_length": len(text),
                "normalized_length": len(normalized_text),
            },
        )

    def detokenize(self, tokens: list[str]) -> str:
        if not self._is_initialized or self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        if isinstance(tokens[0], str):
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        else:
            token_ids = tokens
        int_token_ids = cast("list[int]", token_ids)
        text = self.tokenizer.decode(int_token_ids, skip_special_tokens=True)
        return str(text)

    def get_vocab_size(self) -> int:
        if not self._is_initialized or self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        return len(self.tokenizer)


class SentencePieceTokenizer(BaseTokenizer):
    """SentencePiece-based tokenizer."""

    def __init__(self, config: TokenizationConfig) -> None:
        super().__init__(config)
        if not SENTENCEPIECE_AVAILABLE:
            raise ImportError(
                "sentencepiece library is required. "
                "Install with: pip install sentencepiece"
            )
        self.sp_model: Optional[Any] = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            self.sp_model = spm.SentencePieceProcessor()
            if self.config.model_name and self.config.model_name.endswith(".model"):
                self.sp_model.load(self.config.model_name)
                self.logger.info(
                    f"Loaded SentencePiece model: {self.config.model_name}"
                )
            else:
                raise ValueError(
                    "SentencePiece model file (.model) required or training data needed"
                )
            self._is_initialized = True
        except Exception as e:
            error_msg = f"Failed to load SentencePiece model: {e!s}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def tokenize(self, text: str) -> TokenizationResult:
        if not self._is_initialized or self.sp_model is None:
            raise RuntimeError("SentencePiece model not initialized")
        normalized_text = self.normalize_text(text)
        tokens = self.sp_model.encode_as_pieces(normalized_text)
        token_ids = self.sp_model.encode_as_ids(normalized_text)
        return TokenizationResult(
            tokens=list(tokens),
            token_ids=list(token_ids),
            metadata={
                "method": self.config.method.value,
                "model_name": self.config.model_name,
                "normalized_text": normalized_text,
                "original_length": len(text),
                "normalized_length": len(normalized_text),
            },
        )

    def detokenize(self, tokens: list[str]) -> str:
        if not self._is_initialized or self.sp_model is None:
            raise RuntimeError("SentencePiece model not initialized")
        if tokens and isinstance(tokens[0], str):
            text = self.sp_model.decode_pieces(tokens)
        else:
            text = self.sp_model.decode_ids(tokens)
        return str(text)

    def get_vocab_size(self) -> int:
        if not self._is_initialized or self.sp_model is None:
            raise RuntimeError("SentencePiece model not initialized")
        return int(self.sp_model.get_piece_size())


class WhitespaceTokenizer(BaseTokenizer):
    """Simple whitespace-based tokenizer."""

    def __init__(self, config: TokenizationConfig) -> None:
        super().__init__(config)
        self._is_initialized = True

    def tokenize(self, text: str) -> TokenizationResult:
        normalized_text = self.normalize_text(text)
        tokens = normalized_text.split()
        return TokenizationResult(
            tokens=tokens,
            metadata={
                "method": self.config.method.value,
                "normalized_text": normalized_text,
                "original_length": len(text),
                "normalized_length": len(normalized_text),
            },
        )

    def detokenize(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    def get_vocab_size(self) -> int:
        return -1  # Unlimited vocabulary


class TokenizerFactory:
    """Factory for creating tokenizers."""

    @staticmethod
    def create_tokenizer(config: TokenizationConfig) -> BaseTokenizer:
        if config.method == TokenizationMethod.HUGGINGFACE:
            return HuggingFaceTokenizer(config)
        if config.method == TokenizationMethod.SENTENCEPIECE:
            return SentencePieceTokenizer(config)
        if config.method == TokenizationMethod.WHITESPACE:
            return WhitespaceTokenizer(config)
        raise ValueError(f"Unsupported tokenization method: {config.method}")

    @staticmethod
    def get_recommended_config(model_name: str) -> TokenizationConfig:
        model_configs = {
            "sentence-transformers": TokenizationConfig(
                method=TokenizationMethod.HUGGINGFACE,
                model_name=model_name,
                max_length=512,
                normalization_config=TextNormalizationConfig(
                    lowercase=False,
                    remove_accents=False,
                    preserve_case_for_entities=True,
                ),
            ),
            "bert": TokenizationConfig(
                method=TokenizationMethod.HUGGINGFACE,
                model_name=model_name,
                max_length=512,
                normalization_config=TextNormalizationConfig(
                    lowercase=True, remove_accents=False
                ),
            ),
            "roberta": TokenizationConfig(
                method=TokenizationMethod.HUGGINGFACE,
                model_name=model_name,
                max_length=512,
                normalization_config=TextNormalizationConfig(
                    lowercase=False, remove_accents=False
                ),
            ),
            "openai": TokenizationConfig(
                method=TokenizationMethod.HUGGINGFACE,
                model_name="gpt2",
                max_length=8192,
                normalization_config=TextNormalizationConfig(
                    lowercase=False,
                    remove_accents=False,
                    preserve_case_for_entities=True,
                ),
            ),
        }
        for pattern, cfg in model_configs.items():
            if pattern.lower() in model_name.lower():
                cfg.model_name = model_name
                return cfg
        return TokenizationConfig(
            method=TokenizationMethod.HUGGINGFACE,
            model_name=model_name,
            max_length=512,
            normalization_config=TextNormalizationConfig(),
        )


class LanguageDetector(LoggerMixin):
    """Language detection utility."""

    def __init__(self) -> None:
        self.available = LANGDETECT_AVAILABLE
        if not self.available:
            self.logger.warning(
                "langdetect not available. Install with: pip install langdetect"
            )

    def detect_language(self, text: str) -> Optional[str]:
        if not self.available or not text.strip():
            return None
        try:
            detected_lang: str = langdetect.detect(text)
            return detected_lang
        except Exception as e:
            self.logger.debug(f"Language detection failed: {e!s}")
            return None

    def detect_languages(self, text: str) -> list[tuple[str, float]]:
        if not self.available or not text.strip():
            return []
        try:
            return [(lang.lang, lang.prob) for lang in langdetect.detect_langs(text)]
        except Exception as e:
            self.logger.debug(f"Language detection failed: {e!s}")
            return []


class TokenizationQualityMetrics:
    """Quality metrics for tokenization."""

    @staticmethod
    def calculate_compression_ratio(original_text: str, tokens: list[str]) -> float:
        if not original_text:
            return 0.0
        return len(tokens) / len(original_text)

    @staticmethod
    def calculate_subword_ratio(tokens: list[str]) -> float:
        if not tokens:
            return 0.0
        subword_count = sum(1 for token in tokens if token.startswith(("##", "▁", "Ġ")))
        return subword_count / len(tokens)

    @staticmethod
    def calculate_oov_ratio(tokens: list[str], vocab: set) -> float:
        if not tokens:
            return 0.0
        oov_count = sum(1 for token in tokens if token not in vocab)
        return oov_count / len(tokens)

    @staticmethod
    def analyze_token_distribution(tokens: list[str]) -> dict[str, Any]:
        if not tokens:
            return {}
        token_lengths = [len(token) for token in tokens]
        token_freq: dict[str, int] = {}
        for token in tokens:
            token_freq[token] = token_freq.get(token, 0) + 1
        return {
            "total_tokens": len(tokens),
            "unique_tokens": len(token_freq),
            "avg_token_length": np.mean(token_lengths),
            "median_token_length": np.median(token_lengths),
            "max_token_length": max(token_lengths),
            "min_token_length": min(token_lengths),
            "token_length_std": np.std(token_lengths),
            "most_frequent_tokens": sorted(
                token_freq.items(), key=lambda x: x[1], reverse=True
            )[:10],
        }
