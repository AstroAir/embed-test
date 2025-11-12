"""
Health Check Example

This example demonstrates system health monitoring and diagnostics.
It shows:
- System component health checking
- Performance monitoring
- Resource usage tracking
- Troubleshooting common issues

Prerequisites:
- PDF Vector System installed
- Basic understanding of system monitoring

Usage:
    python health_check.py

Expected Output:
    - Component health status
    - Performance metrics
    - Resource usage information
    - Troubleshooting guidance

Learning Objectives:
- Learn system health monitoring
- Understand performance metrics
- Identify common issues
- Learn troubleshooting techniques
"""

import sys
import time
from pathlib import Path
from typing import Any

import psutil
from utils.example_helpers import (
    example_context,
    get_available_providers,
    print_section,
    print_subsection,
)

from pdf_vector_system import Config, PDFVectorPipeline
from pdf_vector_system.config.settings import EmbeddingModelType

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def check_system_resources() -> dict[str, Any]:
    """Check system resource availability."""
    print_subsection("System Resources")

    try:
        # CPU information
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory information
        memory = psutil.virtual_memory()
        memory_total = memory.total
        memory_available = memory.available
        memory_percent = memory.percent

        # Disk information
        disk = psutil.disk_usage(".")
        disk_total = disk.total
        disk_free = disk.free
        disk_percent = (disk.used / disk.total) * 100

        resources = {
            "cpu_count": cpu_count,
            "cpu_percent": cpu_percent,
            "memory_total": memory_total,
            "memory_available": memory_available,
            "memory_percent": memory_percent,
            "disk_total": disk_total,
            "disk_free": disk_free,
            "disk_percent": disk_percent,
        }

        # Resource warnings
        if cpu_percent > 80:
            pass
        if memory_percent > 80:
            pass
        if disk_percent > 90:
            pass

        return resources

    except Exception:
        return {}


def check_dependencies() -> dict[str, bool]:
    """Check if required dependencies are available."""
    print_subsection("Dependencies")

    dependencies = {}

    # Core dependencies
    import importlib.util

    dependencies["chromadb"] = importlib.util.find_spec("chromadb") is not None
    dependencies["sentence_transformers"] = (
        importlib.util.find_spec("sentence_transformers") is not None
    )
    dependencies["pymupdf"] = importlib.util.find_spec("fitz") is not None

    # Optional dependencies
    dependencies["openai"] = importlib.util.find_spec("openai") is not None
    dependencies["cohere"] = importlib.util.find_spec("cohere") is not None

    return dependencies


def check_embedding_providers() -> dict[str, bool]:
    """Check embedding provider availability."""
    print_subsection("Embedding Providers")

    providers = get_available_providers()

    for _provider, _available in providers.items():
        pass

    return providers


def check_pipeline_health(pipeline: PDFVectorPipeline) -> dict[str, bool]:
    """Check pipeline component health."""
    print_subsection("Pipeline Components")

    try:
        health = pipeline.health_check()

        for _component, _status in health.items():
            pass

        return health

    except Exception:
        return {}


def performance_benchmark(pipeline: PDFVectorPipeline) -> dict[str, float]:
    """Run a simple performance benchmark."""
    print_subsection("Performance Benchmark")

    # Test embedding generation
    test_texts = [
        "This is a test sentence for performance benchmarking.",
        "Machine learning enables computers to learn from data.",
        "Natural language processing helps computers understand text.",
        "Vector databases store high-dimensional embeddings efficiently.",
        "Semantic search finds content by meaning rather than keywords.",
    ]

    try:
        start_time = time.time()

        # Generate embeddings for test texts
        embedding_result = pipeline.batch_processor.process_texts(
            test_texts, show_progress=False
        )

        embedding_time = time.time() - start_time

        if embedding_result.success:
            texts_per_second = len(test_texts) / embedding_time

            # Test search performance if we have data
            try:
                stats = pipeline.get_collection_stats()
                if stats.get("total_chunks", 0) > 0:
                    search_start = time.time()
                    pipeline.search("test query", n_results=5)
                    search_time = time.time() - search_start

                else:
                    search_time = 0
            except Exception:
                search_time = 0

            return {
                "embedding_time": embedding_time,
                "texts_per_second": texts_per_second,
                "search_time": search_time,
            }
        return {}

    except Exception:
        return {}


def check_configuration(config: Config) -> dict[str, Any]:
    """Check configuration validity."""
    print_subsection("Configuration Check")

    config_status = {}

    # Check embedding configuration

    if config.embedding.batch_size <= 0:
        config_status["embedding_batch_size"] = False
    else:
        config_status["embedding_batch_size"] = True

    # Check text processing configuration

    if config.text_processing.chunk_size <= 0:
        config_status["text_chunk_size"] = False
    else:
        config_status["text_chunk_size"] = True

    if config.text_processing.chunk_overlap >= config.text_processing.chunk_size:
        config_status["text_overlap"] = False
    else:
        config_status["text_overlap"] = True

    # Check ChromaDB configuration

    # Check if directory is writable
    try:
        config.chroma_db.persist_directory.mkdir(parents=True, exist_ok=True)
        test_file = config.chroma_db.persist_directory / "test_write"
        test_file.write_text("test")
        test_file.unlink()
        config_status["chroma_writable"] = True
    except Exception:
        config_status["chroma_writable"] = False

    return config_status


def provide_troubleshooting_tips() -> None:
    """Provide troubleshooting tips for common issues."""
    print_subsection("Troubleshooting Tips")


def main() -> None:
    """
    Perform comprehensive health check of the PDF Vector System.

    This function checks all system components and provides
    troubleshooting guidance for common issues.
    """
    with example_context("System Health Check"):
        print_section("System Health Diagnostics")

        # Check system resources
        check_system_resources()

        # Check dependencies
        dependencies = check_dependencies()

        # Check embedding providers
        providers = check_embedding_providers()

        print_section("Pipeline Health Check")

        # Initialize pipeline for testing
        print_subsection("Initializing Test Pipeline")

        try:
            config = Config()
            config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
            config.embedding.model_name = "all-MiniLM-L6-v2"
            config.chroma_db.collection_name = "health_check"
            config.chroma_db.persist_directory = Path("./health_check_db")

            pipeline = PDFVectorPipeline(config)

            # Check configuration
            config_status = check_configuration(config)

            # Check pipeline health
            pipeline_health = check_pipeline_health(pipeline)

            # Run performance benchmark
            performance = performance_benchmark(pipeline)

        except Exception:
            config_status = {}
            pipeline_health = {}
            performance = {}

        print_section("Health Summary")

        # Overall health assessment
        all_checks = {**dependencies, **providers, **config_status, **pipeline_health}

        healthy_count = sum(1 for status in all_checks.values() if status)
        total_count = len(all_checks)
        health_percentage = (
            (healthy_count / total_count * 100) if total_count > 0 else 0
        )

        if health_percentage >= 90 or health_percentage >= 70:
            pass
        else:
            pass

        # Performance summary
        if performance:
            if "texts_per_second" in performance:
                pass
            if "search_time" in performance and performance["search_time"] > 0:
                pass

        print_section("Troubleshooting")
        provide_troubleshooting_tips()

        print_section("Next Steps")

        if health_percentage >= 90:
            pass
        else:
            failed_checks = [name for name, status in all_checks.items() if not status]
            for _check in failed_checks[:5]:  # Show top 5 issues
                pass


if __name__ == "__main__":
    main()
