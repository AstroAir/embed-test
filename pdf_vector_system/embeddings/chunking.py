"""Advanced chunking strategies for optimal embedding generation.

This module provides sophisticated text chunking methods including:
- Semantic chunking based on sentence similarity
- Hierarchical chunking with multiple levels
- Token-aware chunking that respects tokenization boundaries
- Overlapping window strategies with smart overlap detection
- Content-aware chunking that preserves document structure
"""

import importlib.util
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

SENTENCE_TRANSFORMERS_AVAILABLE = (
    importlib.util.find_spec("sentence_transformers") is not None
)
SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None

from pdf_vector_system.embeddings.tokenization import BaseTokenizer
from pdf_vector_system.utils.logging import LoggerMixin


class ChunkingStrategy(Enum):
    """Supported chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    TOKEN_AWARE = "token_aware"
    SENTENCE_BASED = "sentence_based"
    PARAGRAPH_BASED = "paragraph_based"
    SLIDING_WINDOW = "sliding_window"
    CONTENT_AWARE = "content_aware"


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""

    chunk_id: str
    start_char: int
    end_char: int
    start_token: Optional[int] = None
    end_token: Optional[int] = None
    semantic_score: Optional[float] = None
    hierarchy_level: Optional[int] = None
    overlap_with_previous: Optional[int] = None
    overlap_with_next: Optional[int] = None
    content_type: Optional[str] = None
    language: Optional[str] = None
    quality_score: Optional[float] = None
    additional_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextChunk:
    """Enhanced text chunk with comprehensive metadata."""

    content: str
    metadata: ChunkMetadata

    @property
    def length(self) -> int:
        """Get chunk length in characters."""
        return len(self.content)

    @property
    def word_count(self) -> int:
        """Get approximate word count."""
        return len(self.content.split())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "content": self.content,
            "length": self.length,
            "word_count": self.word_count,
            "metadata": {
                "chunk_id": self.metadata.chunk_id,
                "start_char": self.metadata.start_char,
                "end_char": self.metadata.end_char,
                "start_token": self.metadata.start_token,
                "end_token": self.metadata.end_token,
                "semantic_score": self.metadata.semantic_score,
                "hierarchy_level": self.metadata.hierarchy_level,
                "overlap_with_previous": self.metadata.overlap_with_previous,
                "overlap_with_next": self.metadata.overlap_with_next,
                "content_type": self.metadata.content_type,
                "language": self.metadata.language,
                "quality_score": self.metadata.quality_score,
                **self.metadata.additional_metadata,
            },
        }


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies."""

    strategy: ChunkingStrategy = ChunkingStrategy.FIXED_SIZE
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000

    # Semantic chunking parameters
    semantic_threshold: float = 0.7
    sentence_model: Optional[str] = None

    # Token-aware parameters
    tokenizer_config: Optional[dict[str, Any]] = None
    respect_token_boundaries: bool = True

    # Hierarchical parameters
    hierarchy_levels: list[int] = field(default_factory=lambda: [2000, 1000, 500])

    # Content-aware parameters
    preserve_paragraphs: bool = True
    preserve_sentences: bool = True
    detect_headers: bool = True
    detect_lists: bool = True

    # Quality parameters
    min_quality_score: float = 0.5
    calculate_quality_scores: bool = True


class BaseChunker(ABC, LoggerMixin):
    """Abstract base class for text chunkers."""

    def __init__(self, config: ChunkingConfig) -> None:
        """Initialize chunker with configuration."""
        self.config = config

    @abstractmethod
    def chunk_text(self, text: str, document_id: str = "default") -> list[TextChunk]:
        """Chunk text into segments."""

    def _create_chunk_id(self, document_id: str, chunk_index: int) -> str:
        """Create unique chunk ID."""
        return f"{document_id}_chunk_{chunk_index:04d}"

    def _calculate_quality_score(self, chunk_content: str) -> float:
        """Calculate quality score for a chunk."""
        if not chunk_content.strip():
            return 0.0

        score = 1.0

        # Penalize very short chunks
        if len(chunk_content) < self.config.min_chunk_size:
            score *= 0.5

        # Penalize chunks with too much whitespace
        whitespace_ratio = (len(chunk_content) - len(chunk_content.strip())) / len(
            chunk_content
        )
        if whitespace_ratio > 0.3:
            score *= 1.0 - whitespace_ratio

        # Reward chunks with complete sentences
        sentence_endings = (
            chunk_content.count(".")
            + chunk_content.count("!")
            + chunk_content.count("?")
        )
        if sentence_endings > 0:
            score *= 1.1

        # Penalize chunks that start or end mid-word
        words = chunk_content.split()
        if len(words) > 1:
            if not chunk_content[0].isupper() and chunk_content[0].isalpha():
                score *= 0.9
            if chunk_content[-1].isalpha() and not chunk_content.endswith(
                (".", "!", "?")
            ):
                score *= 0.9

        return min(1.0, score)


class FixedSizeChunker(BaseChunker):
    """Fixed-size chunking with overlap."""

    def chunk_text(self, text: str, document_id: str = "default") -> list[TextChunk]:
        """Chunk text into fixed-size segments."""
        if not text.strip():
            return []

        chunks = []
        chunk_index = 0
        start_pos = 0

        while start_pos < len(text):
            # Calculate end position
            end_pos = min(start_pos + self.config.chunk_size, len(text))

            # Adjust to avoid breaking words if possible
            if end_pos < len(text) and not text[end_pos].isspace():
                # Find the last space before end_pos
                last_space = text.rfind(" ", start_pos, end_pos)
                if last_space > start_pos:
                    end_pos = last_space

            chunk_content = text[start_pos:end_pos].strip()

            if len(chunk_content) >= self.config.min_chunk_size:
                # Calculate overlap with previous chunk
                overlap_prev = 0
                if chunk_index > 0 and start_pos > 0:
                    overlap_prev = min(self.config.chunk_overlap, start_pos)

                # Calculate quality score
                quality_score = None
                if self.config.calculate_quality_scores:
                    quality_score = self._calculate_quality_score(chunk_content)

                metadata = ChunkMetadata(
                    chunk_id=self._create_chunk_id(document_id, chunk_index),
                    start_char=start_pos,
                    end_char=end_pos,
                    overlap_with_previous=overlap_prev,
                    quality_score=quality_score,
                    content_type="text",
                )

                chunk = TextChunk(content=chunk_content, metadata=metadata)
                chunks.append(chunk)
                chunk_index += 1

            # Move to next position with overlap
            start_pos = max(start_pos + 1, end_pos - self.config.chunk_overlap)

        # Update overlap information
        for i in range(len(chunks) - 1):
            overlap_next = (
                chunks[i].metadata.end_char - chunks[i + 1].metadata.start_char
            )
            chunks[i].metadata.overlap_with_next = max(0, overlap_next)

        self.logger.debug(f"Created {len(chunks)} fixed-size chunks")
        return chunks


class SentenceBasedChunker(BaseChunker):
    """Sentence-based chunking that preserves sentence boundaries."""

    def chunk_text(self, text: str, document_id: str = "default") -> list[TextChunk]:
        """Chunk text by sentences."""
        if not text.strip():
            return []

        # Split into sentences
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks = []
        chunk_index = 0
        current_chunk_sentences: list[str] = []
        current_chunk_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # Check if adding this sentence would exceed chunk size
            if (
                current_chunk_length + sentence_length > self.config.chunk_size
                and current_chunk_sentences
            ):
                # Create chunk from current sentences
                chunk = self._create_sentence_chunk(
                    current_chunk_sentences, document_id, chunk_index, text
                )
                chunks.append(chunk)
                chunk_index += 1

                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk_sentences, self.config.chunk_overlap
                )
                current_chunk_sentences = [*overlap_sentences, sentence]
                current_chunk_length = sum(len(s) for s in current_chunk_sentences)
            else:
                current_chunk_sentences.append(sentence)
                current_chunk_length += sentence_length

        # Create final chunk
        if current_chunk_sentences:
            chunk = self._create_sentence_chunk(
                current_chunk_sentences, document_id, chunk_index, text
            )
            chunks.append(chunk)

        self.logger.debug(f"Created {len(chunks)} sentence-based chunks")
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Enhanced sentence splitting pattern
        sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(sentence_pattern, text)

        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) >= 10:  # Filter very short sentences
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def _get_overlap_sentences(
        self, sentences: list[str], overlap_chars: int
    ) -> list[str]:
        """Get sentences for overlap based on character count."""
        if not sentences or overlap_chars <= 0:
            return []

        overlap_sentences: list[str] = []
        char_count = 0

        # Take sentences from the end until we reach overlap limit
        for sentence in reversed(sentences):
            if char_count + len(sentence) <= overlap_chars:
                overlap_sentences.insert(0, sentence)
                char_count += len(sentence)
            else:
                break

        return overlap_sentences

    def _create_sentence_chunk(
        self,
        sentences: list[str],
        document_id: str,
        chunk_index: int,
        original_text: str,
    ) -> TextChunk:
        """Create a chunk from sentences."""
        chunk_content = " ".join(sentences)

        # Find positions in original text
        start_char = original_text.find(sentences[0])
        end_char = start_char + len(chunk_content)

        # Calculate quality score
        quality_score = None
        if self.config.calculate_quality_scores:
            quality_score = self._calculate_quality_score(chunk_content)

        metadata = ChunkMetadata(
            chunk_id=self._create_chunk_id(document_id, chunk_index),
            start_char=max(0, start_char),
            end_char=min(len(original_text), end_char),
            quality_score=quality_score,
            content_type="sentences",
            additional_metadata={"sentence_count": len(sentences)},
        )

        return TextChunk(content=chunk_content, metadata=metadata)


class TokenAwareChunker(BaseChunker):
    """Token-aware chunking that respects tokenization boundaries."""

    def __init__(self, config: ChunkingConfig, tokenizer: BaseTokenizer) -> None:
        """Initialize with tokenizer."""
        super().__init__(config)
        self.tokenizer = tokenizer

    def chunk_text(self, text: str, document_id: str = "default") -> list[TextChunk]:
        """Chunk text respecting token boundaries."""
        if not text.strip():
            return []

        # Tokenize the entire text
        tokenization_result = self.tokenizer.tokenize(text)
        tokens = tokenization_result.tokens

        if not tokens:
            return []

        # Calculate target tokens per chunk
        avg_chars_per_token = len(text) / len(tokens)
        target_tokens_per_chunk = int(self.config.chunk_size / avg_chars_per_token)
        overlap_tokens = int(self.config.chunk_overlap / avg_chars_per_token)

        chunks = []
        chunk_index = 0
        start_token = 0

        while start_token < len(tokens):
            # Calculate end token
            end_token = min(start_token + target_tokens_per_chunk, len(tokens))

            # Extract chunk tokens
            chunk_tokens = tokens[start_token:end_token]

            # Convert back to text
            chunk_content = self.tokenizer.detokenize(chunk_tokens)

            if len(chunk_content.strip()) >= self.config.min_chunk_size:
                # Find character positions
                start_char = self._find_char_position(text, chunk_tokens[0])
                end_char = start_char + len(chunk_content)

                # Calculate quality score
                quality_score = None
                if self.config.calculate_quality_scores:
                    quality_score = self._calculate_quality_score(chunk_content)

                metadata = ChunkMetadata(
                    chunk_id=self._create_chunk_id(document_id, chunk_index),
                    start_char=start_char,
                    end_char=end_char,
                    start_token=start_token,
                    end_token=end_token,
                    quality_score=quality_score,
                    content_type="token_aware",
                    additional_metadata={
                        "token_count": len(chunk_tokens),
                        "avg_chars_per_token": len(chunk_content) / len(chunk_tokens),
                    },
                )

                chunk = TextChunk(content=chunk_content, metadata=metadata)
                chunks.append(chunk)
                chunk_index += 1

            # Move to next position with overlap
            start_token = max(start_token + 1, end_token - overlap_tokens)

        self.logger.debug(f"Created {len(chunks)} token-aware chunks")
        return chunks

    def _find_char_position(self, text: str, target_token: str) -> int:
        """Find character position of a token in text."""
        # Simple heuristic - find first occurrence
        # In practice, this would need more sophisticated alignment
        return text.find(target_token.replace("##", "").replace("▁", " "))


class ChunkerFactory:
    """Factory for creating chunkers."""

    @staticmethod
    def create_chunker(
        config: ChunkingConfig, tokenizer: Optional[BaseTokenizer] = None
    ) -> BaseChunker:
        """Create a chunker based on configuration."""
        if config.strategy == ChunkingStrategy.FIXED_SIZE:
            return FixedSizeChunker(config)
        if config.strategy == ChunkingStrategy.SENTENCE_BASED:
            return SentenceBasedChunker(config)
        if config.strategy == ChunkingStrategy.TOKEN_AWARE:
            if tokenizer is None:
                raise ValueError("Tokenizer required for token-aware chunking")
            return TokenAwareChunker(config, tokenizer)
        raise ValueError(f"Unsupported chunking strategy: {config.strategy}")

    @staticmethod
    def get_recommended_config(
        text_type: str = "general",
        target_embedding_model: str = "sentence-transformers",
    ) -> ChunkingConfig:
        """Get recommended chunking config for text type and model."""
        configs = {
            "general": ChunkingConfig(
                strategy=ChunkingStrategy.SENTENCE_BASED,
                chunk_size=1000,
                chunk_overlap=200,
                preserve_sentences=True,
            ),
            "academic": ChunkingConfig(
                strategy=ChunkingStrategy.SENTENCE_BASED,
                chunk_size=1500,
                chunk_overlap=300,
                preserve_paragraphs=True,
                detect_headers=True,
            ),
            "code": ChunkingConfig(
                strategy=ChunkingStrategy.FIXED_SIZE,
                chunk_size=800,
                chunk_overlap=100,
                preserve_sentences=False,
            ),
            "legal": ChunkingConfig(
                strategy=ChunkingStrategy.SENTENCE_BASED,
                chunk_size=2000,
                chunk_overlap=400,
                preserve_paragraphs=True,
                preserve_sentences=True,
            ),
        }

        return configs.get(text_type, configs["general"])
