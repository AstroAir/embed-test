"""
Embedding System Features Demonstration

This example showcases the features of the embedding system including:
- Tokenization methods
- Text preprocessing
- Chunking strategies
- Embedding quality validation
- Performance optimization and caching

Prerequisites:
- VectorFlow installed and importable
- Local SentenceTransformers model "all-MiniLM-L6-v2" available (it will be
  downloaded automatically if internet access is available)
- Optional: GPU and optimized libraries for faster inference (not required)

Usage:
    uv run python -m examples.embedding_features_demo

Expected Output:
    - Printed demonstrations of different tokenization configurations
    - Preprocessing statistics and cleaned text previews for messy input
    - Chunking summaries including number of chunks, sizes, and previews
    - Embedding quality validation scores and human-readable recommendations
    - Caching performance comparison between cache miss and cache hit runs
    - Chunking-with-embeddings statistics and sample chunk metadata

Learning Objectives:
- Understand how tokenization, preprocessing, and chunking work together
- Learn how to configure and evaluate embedding quality
- See how caching can accelerate repeated embedding calls
- Understand how chunking integrates with embedding services in practice
"""

import time
import traceback
from pathlib import Path

from vectorflow.core.embeddings.caching import (
    CacheConfig,
    CacheStrategy,
    EmbeddingCache,
)
from vectorflow.core.embeddings.chunking import (
    ChunkerFactory,
    ChunkingConfig,
    ChunkingStrategy,
)
from vectorflow.core.embeddings.preprocessing import (
    PreprocessingLevel,
    PreprocessorFactory,
    TextType,
)
from vectorflow.core.embeddings.sentence_transformers_service import (
    SentenceTransformersService,
)
from vectorflow.core.embeddings.tokenization import (
    TextNormalizationConfig,
    TokenizationConfig,
    TokenizationMethod,
    TokenizerFactory,
)


def demonstrate_tokenization():
    """Demonstrate tokenization features."""

    # Sample text with various challenges
    sample_text = """
    Hello WORLD! This is a test of the tokenization system.
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
            "Normalized Whitespace",
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

    print("\n=== Tokenization Demonstrations ===")

    for name, config in configs:
        print(f"\n{name} Tokenization:")
        tokenizer = TokenizerFactory.create_tokenizer(config)
        result = tokenizer.tokenize(sample_text)
        tokens = result.tokens
        print(f"  Token count: {result.token_count}")
        print(f"  First 10 tokens: {tokens[:10]}")


def demonstrate_preprocessing():
    """Demonstrate text preprocessing."""

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

    print("\n=== Preprocessing Demonstrations ===")

    for level in levels:
        print(f"\n{level.value} Preprocessing:")
        preprocessor = PreprocessorFactory.create_preprocessor(level, TextType.GENERAL)
        result = preprocessor.preprocess(messy_text)
        cleaned = result.processed_text

        print(f"  Original length: {len(messy_text)}")
        print(f"  Cleaned length: {len(cleaned)}")
        print(f"  Compression ratio: {result.compression_ratio:.3f}")
        print(f"  Preview: {cleaned[:100]}...")


def demonstrate_chunking():
    """Demonstrate chunking strategies."""

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

    print("\n=== Chunking Demonstrations ===")

    for name, strategy, params in strategies:
        print(f"\n{name} Chunking:")
        config = ChunkingConfig(strategy=strategy, **params)

        chunker = ChunkerFactory.create_chunker(config)
        chunks = chunker.chunk_text(long_text, "demo_doc")

        print(f"  Total chunks: {len(chunks)}")
        print(f"  Chunk sizes: {[chunk.length for chunk in chunks]}")

        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"\n  Chunk {i+1}:")
            print(f"    Length: {chunk.length}")
            print(f"    Preview: {chunk.content[:80]}...")


def demonstrate_quality_validation():
    """Demonstrate embedding quality validation."""

    print("\n=== Quality Validation Demonstration ===")

    # Create embedding service with quality validation
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

    print(f"\nProcessing {len(texts)} text samples...")

    start_time = time.time()
    result = service.embed_texts(texts)
    elapsed_time = time.time() - start_time

    print(f"\nEmbedding completed in {elapsed_time:.3f} seconds")

    if result.embeddings is not None:
        print(f"Embedding shape: ({result.count}, {result.embedding_dimension})")
        print(f"Generated {result.count} embeddings")
    else:
        print("Warning: No embeddings generated")
        return

    # Display quality validation results
    if "quality_validation" in result.metadata:
        quality_data = result.metadata["quality_validation"]
        print("\nQuality Validation Results:")

        overall_score = quality_data.get("overall_score")
        overall_pct = quality_data.get("overall_percentage")
        if overall_score is not None and overall_pct is not None:
            print(f"  Overall score: {overall_score:.3f} ({overall_pct:.1f}%)")
        elif overall_score is not None:
            print(f"  Overall score: {overall_score:.3f}")
        else:
            print("  Overall score: N/A")

        # Individual metric scores
        if "individual_scores" in quality_data:
            print("\n  Individual metric scores:")
            for metric_info in quality_data["individual_scores"]:
                metric_name = metric_info.get("metric", "unknown")
                percentage = metric_info.get("percentage")
                if percentage is not None:
                    print(f"    {metric_name}: {percentage:.1f}%")
                else:
                    print(f"    {metric_name}: N/A")

        if quality_data.get("recommendations"):
            print("\n  Recommendations:")
            for rec in quality_data["recommendations"]:
                print(f"    - {rec}")

    # Display preprocessing results
    if "preprocessing_enabled" in result.metadata:
        print(f"\nPreprocessing enabled: {result.metadata['preprocessing_enabled']}")


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

    print("\n=== Caching Demonstration ===")

    try:
        # First run - cache miss
        cache_key = cache.get_cache_key(texts, service.model_name)

        print("\nFirst run (cache miss):")
        start_time = time.time()
        cached_result = cache.get(cache_key)
        if cached_result is None:
            print("  Cache miss - generating embeddings...")
            result = service.embed_texts(texts)
            cache.put(cache_key, result)
            first_run_time = time.time() - start_time
            print(f"  Time taken: {first_run_time:.3f} seconds")
        else:
            print("  Unexpected cache hit on first run")
            first_run_time = None

        # Second run - cache hit
        print("\nSecond run (cache hit):")
        start_time = time.time()
        cached_result = cache.get(cache_key)
        second_run_time = time.time() - start_time

        if cached_result is not None:
            print(f"  Cache hit! Time taken: {second_run_time:.4f} seconds")

            # Calculate speedup properly
            if first_run_time and first_run_time > 0 and second_run_time > 0:
                speedup = first_run_time / second_run_time
                print(f"  Speedup: {speedup:.1f}x faster")
                print(f"  Time saved: {(first_run_time - second_run_time):.3f} seconds")
        else:
            print("  Cache miss - this should not happen!")

        # Display cache statistics
        stats = cache.get_stats()
        print("\nCache Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    finally:
        # Cleanup - ensure cache is cleared even if error occurs
        try:
            cache.clear()
            print("\nCache cleared successfully")
        except Exception as e:
            print(f"\nWarning: Failed to clear cache: {e}")


def demonstrate_chunking_with_embeddings():
    """Demonstrate chunking integration with embeddings."""

    # Create embedding service with preprocessing
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

    print("\n=== Chunking with Embeddings Demonstration ===")

    print(f"\nDocument processed into chunks")
    print(f"Embedding shape: ({result.count}, {result.embedding_dimension})")

    # Display chunking metadata
    if "chunking" in result.metadata:
        chunking_data = result.metadata["chunking"]

        print(f"\nChunking Statistics:")
        print(f"  Total chunks: {chunking_data.get('total_chunks', 'N/A')}")
        print(f"  Strategy: {chunking_data.get('strategy', 'N/A')}")

        if "chunks_metadata" in chunking_data and chunking_data["chunks_metadata"]:
            print("\n  First 3 chunks metadata:")
            for i, chunk_meta in enumerate(chunking_data["chunks_metadata"][:3]):
                print(f"\n    Chunk {i+1}:")
                print(f"      Length: {chunk_meta.get('length', 'N/A')}")
                print(f"      Start position: {chunk_meta.get('start_pos', 'N/A')}")
                if "preview" in chunk_meta:
                    print(f"      Preview: {chunk_meta['preview'][:60]}...")
        else:
            print("\n  No chunk metadata available")


def main():
    """Run all demonstrations."""

    try:
        demonstrate_tokenization()
        demonstrate_preprocessing()
        demonstrate_chunking()
        demonstrate_quality_validation()
        demonstrate_caching()
        demonstrate_chunking_with_embeddings()

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
