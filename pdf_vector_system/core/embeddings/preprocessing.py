"""Advanced text preprocessing for optimal embedding generation.

This module provides sophisticated text preprocessing including:
- Advanced normalization and cleaning
- Language detection and multilingual support
- Content-aware preprocessing
- Domain-specific preprocessing rules
- Quality-aware text filtering
"""

import html
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional

from pdf_vector_system.core.utils.logging import LoggerMixin

try:
    import langdetect
    from langdetect.lang_detect_exception import LangDetectException

    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    LangDetectException = Exception

try:
    import ftfy

    FTFY_AVAILABLE = True
except ImportError:
    FTFY_AVAILABLE = False


class TextType(Enum):
    """Types of text content."""

    GENERAL = "general"
    ACADEMIC = "academic"
    TECHNICAL = "technical"
    LEGAL = "legal"
    MEDICAL = "medical"
    CODE = "code"
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    CONVERSATIONAL = "conversational"


class PreprocessingLevel(Enum):
    """Levels of preprocessing intensity."""

    MINIMAL = "minimal"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""

    level: PreprocessingLevel = PreprocessingLevel.STANDARD
    text_type: TextType = TextType.GENERAL

    # Unicode and encoding
    fix_encoding: bool = True
    unicode_normalization: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFKC"

    # Case handling
    lowercase: bool = False
    preserve_case_entities: bool = True

    # Whitespace and formatting
    normalize_whitespace: bool = True
    remove_extra_newlines: bool = True
    preserve_paragraph_breaks: bool = True

    # Punctuation and symbols
    normalize_punctuation: bool = True
    remove_special_chars: bool = False
    preserve_sentence_boundaries: bool = True

    # Numbers and dates
    normalize_numbers: bool = False
    remove_numbers: bool = False
    normalize_dates: bool = False

    # Language-specific
    remove_accents: bool = False
    expand_contractions: bool = False

    # Content filtering
    min_text_length: int = 10
    max_text_length: int = 100000
    remove_urls: bool = True
    remove_emails: bool = True
    remove_phone_numbers: bool = True

    # Quality filtering
    min_word_count: int = 3
    max_repetition_ratio: float = 0.3
    min_alpha_ratio: float = 0.5

    # Domain-specific
    remove_citations: bool = False
    remove_references: bool = False
    remove_code_blocks: bool = False

    # Custom patterns
    custom_patterns: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class PreprocessingResult:
    """Result of text preprocessing."""

    original_text: str
    processed_text: str
    language: Optional[str] = None
    text_type: Optional[TextType] = None
    quality_score: Optional[float] = None
    transformations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if not self.original_text:
            return 1.0
        return len(self.processed_text) / len(self.original_text)

    @property
    def character_reduction(self) -> int:
        """Calculate character reduction."""
        return len(self.original_text) - len(self.processed_text)


class AdvancedTextPreprocessor(LoggerMixin):
    """Advanced text preprocessor with multiple strategies."""

    def __init__(self, config: PreprocessingConfig) -> None:
        """Initialize preprocessor with configuration."""
        self.config = config
        self._compile_patterns()

        if not LANGDETECT_AVAILABLE:
            self.logger.warning(
                "langdetect not available. Language detection disabled. "
                "Install with: pip install langdetect"
            )

        if not FTFY_AVAILABLE:
            self.logger.warning(
                "ftfy not available. Advanced encoding fixes disabled. "
                "Install with: pip install ftfy"
            )

    def _compile_patterns(self) -> None:
        """Compile regex patterns for preprocessing."""
        self.patterns = {
            "url": re.compile(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
            ),
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            "phone": re.compile(
                r"(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}"
            ),
            "citation": re.compile(
                r"\[[0-9,\s-]+\]|\([A-Za-z]+\s+et\s+al\.?,?\s+[0-9]{4}\)"
            ),
            "reference": re.compile(
                r"^References?:?$|^Bibliography:?$", re.MULTILINE | re.IGNORECASE
            ),
            "code_block": re.compile(r"```[\s\S]*?```|`[^`]+`"),
            "multiple_spaces": re.compile(r"\s{2,}"),
            "multiple_newlines": re.compile(r"\n{3,}"),
            "bullet_points": re.compile(r"^[\s]*[•·▪▫‣⁃]\s*", re.MULTILINE),
            "numbered_lists": re.compile(r"^[\s]*\d+[\.\)]\s*", re.MULTILINE),
            "html_tags": re.compile(r"<[^>]+>"),
            "markdown_headers": re.compile(r"^#{1,6}\s+", re.MULTILINE),
            "markdown_links": re.compile(r"\[([^\]]+)\]\([^\)]+\)"),
            "excessive_punctuation": re.compile(r"[!?]{2,}|[.]{3,}"),
            "non_printable": re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]"),
        }

        # Contractions mapping
        self.contractions = {
            "ain't": "is not",
            "aren't": "are not",
            "can't": "cannot",
            "couldn't": "could not",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'll": "he will",
            "he's": "he is",
            "i'd": "i would",
            "i'll": "i will",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'll": "it will",
            "it's": "it is",
            "let's": "let us",
            "shouldn't": "should not",
            "that's": "that is",
            "there's": "there is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "we'd": "we would",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what's": "what is",
            "where's": "where is",
            "who's": "who is",
            "won't": "will not",
            "wouldn't": "would not",
            "you'd": "you would",
            "you'll": "you will",
            "you're": "you are",
            "you've": "you have",
        }

    def preprocess(self, text: str) -> PreprocessingResult:
        """
        Preprocess text according to configuration.
        """
        if not text or not text.strip():
            return PreprocessingResult(
                original_text=text,
                processed_text="",
                quality_score=0.0,
                transformations=["empty_input"],
            )

        original_text = text
        processed_text = text
        transformations: list[str] = []

        # Detect language
        language = self._detect_language(processed_text)

        # Fix encoding issues
        if self.config.fix_encoding:
            processed_text, encoding_fixed = self._fix_encoding(processed_text)
            if encoding_fixed:
                transformations.append("encoding_fixed")

        # Unicode normalization
        if self.config.unicode_normalization:
            processed_text = self._normalize_unicode(processed_text)
            transformations.append("unicode_normalized")

        # Remove non-printable characters
        processed_text = self.patterns["non_printable"].sub("", processed_text)
        transformations.append("non_printable_removed")

        # HTML decoding and tag removal
        processed_text = html.unescape(processed_text)
        processed_text = self.patterns["html_tags"].sub(" ", processed_text)
        transformations.append("html_cleaned")

        # Content-specific cleaning
        if self.config.remove_urls:
            processed_text = self.patterns["url"].sub(" [URL] ", processed_text)
            transformations.append("urls_removed")

        if self.config.remove_emails:
            processed_text = self.patterns["email"].sub(" [EMAIL] ", processed_text)
            transformations.append("emails_removed")

        if self.config.remove_phone_numbers:
            processed_text = self.patterns["phone"].sub(" [PHONE] ", processed_text)
            transformations.append("phones_removed")

        # Domain-specific cleaning
        if self.config.remove_citations:
            processed_text = self.patterns["citation"].sub("", processed_text)
            transformations.append("citations_removed")

        if self.config.remove_code_blocks:
            processed_text = self.patterns["code_block"].sub(" [CODE] ", processed_text)
            transformations.append("code_blocks_removed")

        # Markdown cleaning
        processed_text = self.patterns["markdown_headers"].sub("", processed_text)
        processed_text = self.patterns["markdown_links"].sub(r"\1", processed_text)
        transformations.append("markdown_cleaned")

        # Punctuation normalization
        if self.config.normalize_punctuation:
            processed_text = self.patterns["excessive_punctuation"].sub(
                ".", processed_text
            )
            transformations.append("punctuation_normalized")

        # Expand contractions
        if self.config.expand_contractions:
            processed_text = self._expand_contractions(processed_text)
            transformations.append("contractions_expanded")

        # Case handling
        if self.config.lowercase:
            if self.config.preserve_case_entities:
                processed_text = self._smart_lowercase(processed_text)
                transformations.append("smart_lowercase")
            else:
                processed_text = processed_text.lower()
                transformations.append("lowercase")

        # Remove accents
        if self.config.remove_accents:
            processed_text = self._remove_accents(processed_text)
            transformations.append("accents_removed")

        # Whitespace normalization
        if self.config.normalize_whitespace:
            processed_text = self._normalize_whitespace(processed_text)
            transformations.append("whitespace_normalized")

        # Apply custom patterns
        for pattern, replacement in self.config.custom_patterns:
            processed_text = re.sub(pattern, replacement, processed_text)
            transformations.append("custom_pattern_applied")

        # Final cleanup
        processed_text = processed_text.strip()

        # Quality score
        quality_score = self._calculate_quality_score(processed_text)

        # Text type detection
        text_type = self._detect_text_type(processed_text)

        metadata = {
            "original_length": len(original_text),
            "processed_length": len(processed_text),
            "compression_ratio": (
                len(processed_text) / len(original_text) if original_text else 1.0
            ),
            "word_count": len(processed_text.split()),
            "sentence_count": processed_text.count(".")
            + processed_text.count("!")
            + processed_text.count("?"),
            "preprocessing_level": self.config.level.value,
            "transformations_count": len(transformations),
        }

        return PreprocessingResult(
            original_text=original_text,
            processed_text=processed_text,
            language=language,
            text_type=text_type,
            quality_score=quality_score,
            transformations=transformations,
            metadata=metadata,
        )

    def _detect_language(self, text: str) -> Optional[str]:
        """Detect text language."""
        if not LANGDETECT_AVAILABLE or len(text.strip()) < 20:
            return None
        try:
            detected_lang: str = langdetect.detect(text)
            return detected_lang
        except LangDetectException:
            return None
        except Exception:
            return None

    def _fix_encoding(self, text: str) -> tuple[str, bool]:
        """Fix encoding issues in text."""
        if FTFY_AVAILABLE:
            try:
                fixed_text = ftfy.fix_text(text)
                return fixed_text, fixed_text != text
            except Exception:
                pass
        # Fallback basic handling
        try:
            fixed_text = text.encode("utf-8", errors="ignore").decode("utf-8")
            return fixed_text, fixed_text != text
        except Exception:
            return text, False

    def _normalize_unicode(self, text: str) -> str:
        return unicodedata.normalize(self.config.unicode_normalization, text)

    def _expand_contractions(self, text: str) -> str:
        words = text.split()
        expanded_words: list[str] = []
        for word in words:
            clean_word = word.lower().strip('.,!?;:"')
            if clean_word in self.contractions:
                expanded = self.contractions[clean_word]
                if word and word[0].isupper():
                    expanded = expanded.capitalize()
                punctuation = "".join(c for c in word if not c.isalnum())
                expanded_words.append(expanded + punctuation)
            else:
                expanded_words.append(word)
        return " ".join(expanded_words)

    def _smart_lowercase(self, text: str) -> str:
        words = text.split()
        processed: list[str] = []
        for i, word in enumerate(words):
            if (len(word) >= 2 and word.isupper() and word.isalpha()) or (
                word[0].isupper()
                and i > 0
                and not words[i - 1].endswith((".", "!", "?"))
            ):
                processed.append(word)
            else:
                processed.append(word.lower())
        return " ".join(processed)

    def _remove_accents(self, text: str) -> str:
        return "".join(
            ch
            for ch in unicodedata.normalize("NFD", text)
            if unicodedata.category(ch) != "Mn"
        )

    def _normalize_whitespace(self, text: str) -> str:
        if self.config.preserve_paragraph_breaks:
            text = self.patterns["multiple_newlines"].sub("\n\n", text)
        else:
            text = re.sub(r"\n+", " ", text)
        text = self.patterns["multiple_spaces"].sub(" ", text)
        return text.strip()

    def _calculate_quality_score(self, text: str) -> float:
        if not text.strip():
            return 0.0
        score = 1.0
        if len(text) < self.config.min_text_length:
            score *= 0.5
        if len(text) > self.config.max_text_length:
            score *= 0.8
        words = text.split()
        if len(words) < self.config.min_word_count:
            score *= 0.6
        alpha_chars = sum(1 for c in text if c.isalpha())
        alpha_ratio = alpha_chars / len(text) if text else 0
        if (
            alpha_ratio < self.config.min_alpha_ratio
            and self.config.min_alpha_ratio > 0
        ):
            score *= alpha_ratio / self.config.min_alpha_ratio
        if len(words) > 1:
            unique_words = len(set(words))
            repetition_ratio = 1.0 - (unique_words / len(words))
            if repetition_ratio > self.config.max_repetition_ratio:
                score *= 1.0 - repetition_ratio
        sentence_endings = text.count(".") + text.count("!") + text.count("?")
        if sentence_endings > 0:
            score *= 1.1
        return min(1.0, score)

    def _detect_text_type(self, text: str) -> Optional[TextType]:
        text_lower = text.lower()
        if any(
            k in text_lower
            for k in ["abstract", "methodology", "conclusion", "references"]
        ):
            return TextType.ACADEMIC
        if any(k in text_lower for k in ["function", "class", "import", "return"]):
            return TextType.CODE
        if any(k in text_lower for k in ["plaintiff", "defendant", "court", "legal"]):
            return TextType.LEGAL
        if any(
            k in text_lower for k in ["patient", "diagnosis", "treatment", "medical"]
        ):
            return TextType.MEDICAL
        if any(k in text_lower for k in ["@", "#", "lol", "omg"]):
            return TextType.SOCIAL_MEDIA
        return TextType.GENERAL


class PreprocessorFactory:
    """Factory for creating preprocessors."""

    @staticmethod
    def create_preprocessor(
        level: PreprocessingLevel = PreprocessingLevel.STANDARD,
        text_type: TextType = TextType.GENERAL,
    ) -> AdvancedTextPreprocessor:
        configs = {
            PreprocessingLevel.MINIMAL: PreprocessingConfig(
                level=PreprocessingLevel.MINIMAL,
                text_type=text_type,
                fix_encoding=True,
                normalize_whitespace=True,
                remove_urls=False,
                remove_emails=False,
                lowercase=False,
            ),
            PreprocessingLevel.STANDARD: PreprocessingConfig(
                level=PreprocessingLevel.STANDARD,
                text_type=text_type,
                fix_encoding=True,
                normalize_whitespace=True,
                normalize_punctuation=True,
                remove_urls=True,
                remove_emails=True,
                preserve_case_entities=True,
            ),
            PreprocessingLevel.AGGRESSIVE: PreprocessingConfig(
                level=PreprocessingLevel.AGGRESSIVE,
                text_type=text_type,
                fix_encoding=True,
                normalize_whitespace=True,
                normalize_punctuation=True,
                remove_urls=True,
                remove_emails=True,
                remove_phone_numbers=True,
                expand_contractions=True,
                lowercase=True,
                preserve_case_entities=False,
                remove_citations=True,
                remove_code_blocks=True,
            ),
        }

        config = configs.get(level, configs[PreprocessingLevel.STANDARD])

        # Adjustments per text type
        if text_type == TextType.ACADEMIC:
            config.remove_citations = False
            config.preserve_case_entities = True
        elif text_type == TextType.CODE:
            config.lowercase = False
            config.normalize_punctuation = False
            config.remove_code_blocks = False
        elif text_type == TextType.SOCIAL_MEDIA:
            config.expand_contractions = True
            config.normalize_punctuation = True

        return AdvancedTextPreprocessor(config)
