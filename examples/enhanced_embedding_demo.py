"""
Enhanced Embedding System Demonstration

This example showcases the advanced features of the enhanced embedding system including:
- Modern tokenization methods
- Advanced text preprocessing
- Sophisticated chunking strategies
- Embedding quality validation
- Performance optimization and caching
"""

import time
from pathlib import Path

from pdf_vector_system.embeddings.caching import (
    CacheConfig,
    CacheStrategy,
    EmbeddingCache,
)
from pdf_vector_system.embeddings.chunking import (
    ChunkerFactory,
    ChunkingConfig,
    ChunkingStrategy,
)
from pdf_vector_system.embeddings.preprocessing import (
    PreprocessingLevel,
    PreprocessorFactory,
    TextType,
)
from pdf_vector_system.embeddings.sentence_transformers_service import (
    SentenceTransformersService,
)
from pdf_vector_system.embeddings.tokenization import (
    TextNormalizationConfig,
    TokenizationConfig,
    TokenizationMethod,
    TokenizerFactory,
)


def demonstrate_enhanced_tokenization():
    """Demonstrate advanced tokenization features."""

    # Sample text with various challenges
    sample_text = """
    Hello WORLD! This is a test of the advanced tokenization system.
    It handles contractions like don't, won't, and can't.
    It also processes URLs like https://example.com and emails like test@example.com.
    Special characters: àáâãäå, ñ, ç, and unicode: 🚀 🌟
    """

    # Create different tokenization configurations
    configs = [
        (
            "Basic Whitespace",
            TokenizationConfig(
                method=TokenizationMethod.WHITESPACE,
                normalization_config=TextNormalizationConfig(
                    lowercase=False, remove_extra_whitespace=True
                ),
            ),
        ),
        (
            "Advanced Whitespace",
            TokenizationConfig(
                method=TokenizationMethod.WHITESPACE,
                normalization_config=TextNormalizationConfig(
                    lowercase=True,
                    remove_accents=True,
                    remove_extra_whitespace=True,
                    preserve_case_for_entities=True,
                ),
            ),
        ),
    ]

    for _name, config in configs:
        tokenizer = TokenizerFactory.create_tokenizer(config)
        tokenizer.tokenize(sample_text)


def demonstrate_advanced_preprocessing():
    """Demonstrate advanced text preprocessing."""

    # Sample text with various issues
    messy_text = """
    <html><body>
    This is a MESSY text with    excessive   whitespace!!!

    It contains URLs: https://example.com/path?param=value
    Email addresses: contact@company.com
    Phone numbers: +1-555-123-4567

    HTML tags: <div class="content">Important content</div>

    Contractions: don't, won't, can't, I'm, you're

    Citations: [1] Smith et al. (2023) found that...

    Code blocks: ```python
    def hello():
        print("Hello World")
    ```

    Multiple    spaces   and


    excessive newlines.
    </body></html>
    """

    # Test different preprocessing levels
    levels = [
        PreprocessingLevel.MINIMAL,
        PreprocessingLevel.STANDARD,
        PreprocessingLevel.AGGRESSIVE,
    ]

    for level in levels:
        preprocessor = PreprocessorFactory.create_preprocessor(level, TextType.GENERAL)
        preprocessor.preprocess(messy_text)


def demonstrate_advanced_chunking():
    """Demonstrate sophisticated chunking strategies."""

    # Sample long text
    long_text = """
    Artificial Intelligence (AI) has revolutionized numerous industries and aspects of human life.
    From healthcare to transportation, AI technologies are being integrated into systems that were
    once purely human-operated.

    Machine learning, a subset of AI, enables computers to learn and improve from experience without
    being explicitly programmed. This capability has led to breakthroughs in image recognition,
    natural language processing, and predictive analytics.

    Deep learning, which uses neural networks with multiple layers, has been particularly successful
    in tasks such as speech recognition and computer vision. These technologies power many of the
    AI applications we use daily, from virtual assistants to recommendation systems.

    However, the rapid advancement of AI also raises important ethical considerations. Issues such as
    bias in algorithms, privacy concerns, and the potential for job displacement need to be carefully
    addressed as AI continues to evolve.

    The future of AI holds immense promise, but it requires thoughtful development and implementation
    to ensure that its benefits are realized while minimizing potential risks.
    """

    # Test different chunking strategies
    strategies = [
        (
            "Fixed Size",
            ChunkingStrategy.FIXED_SIZE,
            {"chunk_size": 200, "chunk_overlap": 50},
        ),
        (
            "Sentence Based",
            ChunkingStrategy.SENTENCE_BASED,
            {"chunk_size": 300, "chunk_overlap": 100},
        ),
    ]

    for _name, strategy, params in strategies:
        config = ChunkingConfig(strategy=strategy, **params)

        chunker = ChunkerFactory.create_chunker(config)
        chunks = chunker.chunk_text(long_text, "demo_doc")

        for _i, _chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            pass


def demonstrate_quality_validation():
    """Demonstrate embedding quality validation."""

    # Create enhanced embedding service
    service = SentenceTransformersService(
        model_name="all-MiniLM-L6-v2",
        enable_advanced_preprocessing=True,
        enable_quality_validation=True,
        preprocessing_level=PreprocessingLevel.STANDARD,
    )

    # Sample texts with varying quality
    texts = [
        "This is a high-quality, well-formed sentence about artificial intelligence.",
        "AI machine learning deep neural networks algorithms data science.",
        "ai ai ai ai ai ai ai ai ai ai",  # Low quality - repetitive
        "The quick brown fox jumps over the lazy dog in the beautiful garden.",
        "Natural language processing enables computers to understand human language.",
    ]

    start_time = time.time()

    result = service.embed_texts(texts)

    time.time() - start_time

    # Display quality validation results
    if "quality_validation" in result.metadata:
        quality_data = result.metadata["quality_validation"]

        for _score in quality_data["individual_scores"]:
            pass

        for _rec in quality_data["recommendations"]:
            pass

    # Display preprocessing results
    if "preprocessing_enabled" in result.metadata:
        pass


def demonstrate_caching():
    """Demonstrate intelligent caching."""

    # Create cache configuration
    cache_config = CacheConfig(
        strategy=CacheStrategy.MEMORY_DISK,
        max_memory_size=100,
        max_disk_size=1000,
        cache_dir=Path("./cache_demo"),
        compression_enabled=True,
    )

    # Initialize cache
    cache = EmbeddingCache(cache_config)

    # Create embedding service
    service = SentenceTransformersService(
        model_name="all-MiniLM-L6-v2", enable_advanced_preprocessing=True
    )

    texts = [
        "Caching improves performance by storing frequently accessed data.",
        "Machine learning models can benefit significantly from intelligent caching.",
        "This is a test of the caching system functionality.",
    ]

    # First run - cache miss
    cache_key = cache.get_cache_key(texts, service.model_name)

    time.time()
    cached_result = cache.get(cache_key)
    if cached_result is None:
        result = service.embed_texts(texts)
        cache.put(cache_key, result)

    # Second run - cache hit
    time.time()
    cached_result = cache.get(cache_key)
    if cached_result is not None:
        pass

    # Display cache statistics
    cache.get_stats()

    # Cleanup
    cache.clear()


def demonstrate_chunking_with_embeddings():
    """Demonstrate chunking integration with embeddings."""

    # Create enhanced embedding service
    service = SentenceTransformersService(
        model_name="all-MiniLM-L6-v2",
        enable_advanced_preprocessing=True,
        enable_quality_validation=False,  # Disable for speed
    )

    # Long document text
    document_text = """
    The field of artificial intelligence has undergone remarkable transformations over the past decade.
    What began as theoretical concepts in computer science laboratories has now become integral to
    countless applications that touch our daily lives.

    Machine learning algorithms have evolved from simple linear models to sophisticated deep neural
    networks capable of processing vast amounts of data. These systems can now recognize patterns,
    make predictions, and even generate creative content with unprecedented accuracy.

    Natural language processing has seen particularly dramatic advances. Modern language models can
    understand context, generate human-like text, and even engage in meaningful conversations.
    This progress has enabled the development of virtual assistants, automated translation services,
    and content generation tools.

    Computer vision has similarly revolutionized how machines interpret visual information. From
    medical imaging to autonomous vehicles, AI systems can now analyze images and videos with
    precision that often exceeds human capabilities.

    However, these advances also bring challenges. Questions about bias, fairness, and transparency
    in AI systems have become increasingly important. The development of responsible AI practices
    is crucial for ensuring that these powerful technologies benefit society as a whole.
    """

    # Test chunking with embeddings
    result = service.embed_with_chunking(
        document_text,
        chunking_strategy=ChunkingStrategy.SENTENCE_BASED,
        chunk_size=400,
        chunk_overlap=100,
    )

    # Display chunking metadata
    if "chunking" in result.metadata:
        chunking_data = result.metadata["chunking"]

        for _i, _chunk_meta in enumerate(chunking_data["chunks_metadata"][:3]):
            pass


def main():
    """Run all demonstrations."""

    try:
        demonstrate_enhanced_tokenization()
        demonstrate_advanced_preprocessing()
        demonstrate_advanced_chunking()
        demonstrate_quality_validation()
        demonstrate_caching()
        demonstrate_chunking_with_embeddings()

    except Exception:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
