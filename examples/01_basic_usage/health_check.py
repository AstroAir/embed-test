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
import psutil
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pdf_vector_system import Config, PDFVectorPipeline
from pdf_vector_system.config.settings import EmbeddingModelType
from utils.example_helpers import (
    print_section, print_subsection, example_context,
    format_bytes, format_duration, get_available_providers
)


def check_system_resources() -> Dict[str, Any]:
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
        disk = psutil.disk_usage('.')
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
            "disk_percent": disk_percent
        }
        
        print(f"💻 CPU: {cpu_count} cores, {cpu_percent:.1f}% usage")
        print(f"🧠 Memory: {format_bytes(memory_available)} available of {format_bytes(memory_total)} ({memory_percent:.1f}% used)")
        print(f"💾 Disk: {format_bytes(disk_free)} free of {format_bytes(disk_total)} ({disk_percent:.1f}% used)")
        
        # Resource warnings
        if cpu_percent > 80:
            print("   ⚠️  High CPU usage detected")
        if memory_percent > 80:
            print("   ⚠️  High memory usage detected")
        if disk_percent > 90:
            print("   ⚠️  Low disk space detected")
        
        return resources
        
    except Exception as e:
        print(f"❌ Error checking system resources: {e}")
        return {}


def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are available."""
    print_subsection("Dependencies")
    
    dependencies = {}
    
    # Core dependencies
    try:
        import chromadb
        dependencies["chromadb"] = True
        print("✅ ChromaDB: Available")
    except ImportError:
        dependencies["chromadb"] = False
        print("❌ ChromaDB: Not available")
    
    try:
        import sentence_transformers
        dependencies["sentence_transformers"] = True
        print("✅ Sentence Transformers: Available")
    except ImportError:
        dependencies["sentence_transformers"] = False
        print("❌ Sentence Transformers: Not available")
    
    try:
        import fitz  # PyMuPDF
        dependencies["pymupdf"] = True
        print("✅ PyMuPDF: Available")
    except ImportError:
        dependencies["pymupdf"] = False
        print("❌ PyMuPDF: Not available")
    
    # Optional dependencies
    try:
        import openai
        dependencies["openai"] = True
        print("✅ OpenAI: Available")
    except ImportError:
        dependencies["openai"] = False
        print("⚠️  OpenAI: Not available (optional)")
    
    try:
        import cohere
        dependencies["cohere"] = True
        print("✅ Cohere: Available")
    except ImportError:
        dependencies["cohere"] = False
        print("⚠️  Cohere: Not available (optional)")
    
    return dependencies


def check_embedding_providers() -> Dict[str, bool]:
    """Check embedding provider availability."""
    print_subsection("Embedding Providers")
    
    providers = get_available_providers()
    
    for provider, available in providers.items():
        status = "✅" if available else "❌"
        print(f"{status} {provider.replace('_', ' ').title()}: {'Available' if available else 'Not available'}")
    
    return providers


def check_pipeline_health(pipeline: PDFVectorPipeline) -> Dict[str, bool]:
    """Check pipeline component health."""
    print_subsection("Pipeline Components")
    
    try:
        health = pipeline.health_check()
        
        for component, status in health.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {component}: {'Healthy' if status else 'Unhealthy'}")
        
        return health
        
    except Exception as e:
        print(f"❌ Error checking pipeline health: {e}")
        return {}


def performance_benchmark(pipeline: PDFVectorPipeline) -> Dict[str, float]:
    """Run a simple performance benchmark."""
    print_subsection("Performance Benchmark")
    
    # Test embedding generation
    test_texts = [
        "This is a test sentence for performance benchmarking.",
        "Machine learning enables computers to learn from data.",
        "Natural language processing helps computers understand text.",
        "Vector databases store high-dimensional embeddings efficiently.",
        "Semantic search finds content by meaning rather than keywords."
    ]
    
    try:
        print("🏃 Running embedding benchmark...")
        
        start_time = time.time()
        
        # Generate embeddings for test texts
        embedding_result = pipeline.batch_processor.process_texts(
            test_texts, 
            show_progress=False
        )
        
        embedding_time = time.time() - start_time
        
        if embedding_result.success:
            texts_per_second = len(test_texts) / embedding_time
            print(f"   ✅ Processed {len(test_texts)} texts in {embedding_time:.2f}s")
            print(f"   📊 Speed: {texts_per_second:.1f} texts/second")
            
            # Test search performance if we have data
            try:
                stats = pipeline.get_collection_stats()
                if stats.get('total_chunks', 0) > 0:
                    print("\n🔍 Running search benchmark...")
                    
                    search_start = time.time()
                    search_results = pipeline.search("test query", n_results=5)
                    search_time = time.time() - search_start
                    
                    print(f"   ✅ Search completed in {search_time:.3f}s")
                    print(f"   📊 Found {len(search_results)} results")
                else:
                    search_time = 0
                    print("   ⚠️  No data available for search benchmark")
            except Exception as e:
                search_time = 0
                print(f"   ❌ Search benchmark failed: {e}")
            
            return {
                "embedding_time": embedding_time,
                "texts_per_second": texts_per_second,
                "search_time": search_time
            }
        else:
            print(f"   ❌ Benchmark failed: {embedding_result.error_message}")
            return {}
            
    except Exception as e:
        print(f"❌ Performance benchmark error: {e}")
        return {}


def check_configuration(config: Config) -> Dict[str, Any]:
    """Check configuration validity."""
    print_subsection("Configuration Check")
    
    config_status = {}
    
    # Check embedding configuration
    print(f"🤖 Embedding Model: {config.embedding.model_type.value}")
    print(f"   - Model name: {config.embedding.model_name}")
    print(f"   - Batch size: {config.embedding.batch_size}")
    
    if config.embedding.batch_size <= 0:
        print("   ❌ Invalid batch size")
        config_status["embedding_batch_size"] = False
    else:
        config_status["embedding_batch_size"] = True
    
    # Check text processing configuration
    print(f"\n📝 Text Processing:")
    print(f"   - Chunk size: {config.text_processing.chunk_size}")
    print(f"   - Chunk overlap: {config.text_processing.chunk_overlap}")
    
    if config.text_processing.chunk_size <= 0:
        print("   ❌ Invalid chunk size")
        config_status["text_chunk_size"] = False
    else:
        config_status["text_chunk_size"] = True
    
    if config.text_processing.chunk_overlap >= config.text_processing.chunk_size:
        print("   ❌ Chunk overlap >= chunk size")
        config_status["text_overlap"] = False
    else:
        config_status["text_overlap"] = True
    
    # Check ChromaDB configuration
    print(f"\n🗄️  ChromaDB:")
    print(f"   - Collection: {config.chroma_db.collection_name}")
    print(f"   - Directory: {config.chroma_db.persist_directory}")
    
    # Check if directory is writable
    try:
        config.chroma_db.persist_directory.mkdir(parents=True, exist_ok=True)
        test_file = config.chroma_db.persist_directory / "test_write"
        test_file.write_text("test")
        test_file.unlink()
        config_status["chroma_writable"] = True
        print("   ✅ Directory is writable")
    except Exception as e:
        config_status["chroma_writable"] = False
        print(f"   ❌ Directory not writable: {e}")
    
    return config_status


def provide_troubleshooting_tips() -> None:
    """Provide troubleshooting tips for common issues."""
    print_subsection("Troubleshooting Tips")
    
    print("🔧 Common Issues and Solutions:")
    print()
    print("1. 🚫 Import Errors:")
    print("   - Install missing dependencies: pip install pdf-vector-system")
    print("   - Check Python version (requires 3.9+)")
    print("   - Verify virtual environment activation")
    print()
    print("2. 🔑 API Key Issues:")
    print("   - Set environment variables: export OPENAI_API_KEY=your_key")
    print("   - Check .env file configuration")
    print("   - Verify API key validity")
    print()
    print("3. 💾 Memory Issues:")
    print("   - Reduce batch size in configuration")
    print("   - Use smaller embedding models")
    print("   - Process fewer documents at once")
    print()
    print("4. 🐌 Performance Issues:")
    print("   - Increase batch size for better throughput")
    print("   - Use more worker threads")
    print("   - Consider cloud-based embedding providers")
    print()
    print("5. 🗄️  Database Issues:")
    print("   - Check disk space availability")
    print("   - Verify directory permissions")
    print("   - Clear corrupted database files")


def main() -> None:
    """
    Perform comprehensive health check of the PDF Vector System.
    
    This function checks all system components and provides
    troubleshooting guidance for common issues.
    """
    with example_context("System Health Check"):
        
        print_section("System Health Diagnostics")
        
        # Check system resources
        resources = check_system_resources()
        
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
            print("✅ Test pipeline initialized")
            
            # Check configuration
            config_status = check_configuration(config)
            
            # Check pipeline health
            pipeline_health = check_pipeline_health(pipeline)
            
            # Run performance benchmark
            performance = performance_benchmark(pipeline)
            
        except Exception as e:
            print(f"❌ Failed to initialize test pipeline: {e}")
            config_status = {}
            pipeline_health = {}
            performance = {}
        
        print_section("Health Summary")
        
        # Overall health assessment
        all_checks = {
            **dependencies,
            **providers,
            **config_status,
            **pipeline_health
        }
        
        healthy_count = sum(1 for status in all_checks.values() if status)
        total_count = len(all_checks)
        health_percentage = (healthy_count / total_count * 100) if total_count > 0 else 0
        
        print(f"📊 Overall Health: {healthy_count}/{total_count} checks passed ({health_percentage:.1f}%)")
        
        if health_percentage >= 90:
            print("✅ System is healthy and ready for use")
        elif health_percentage >= 70:
            print("⚠️  System is mostly healthy with minor issues")
        else:
            print("❌ System has significant issues that need attention")
        
        # Performance summary
        if performance:
            print(f"\n⚡ Performance Summary:")
            if "texts_per_second" in performance:
                print(f"   - Embedding speed: {performance['texts_per_second']:.1f} texts/second")
            if "search_time" in performance and performance["search_time"] > 0:
                print(f"   - Search latency: {performance['search_time']:.3f} seconds")
        
        print_section("Troubleshooting")
        provide_troubleshooting_tips()
        
        print_section("Next Steps")
        
        print("🎯 Recommendations:")
        print()
        
        if health_percentage >= 90:
            print("✅ Your system is ready! Try these examples:")
            print("   - simple_pdf_processing.py for basic usage")
            print("   - first_search.py for search functionality")
            print("   - configuration_basics.py for advanced configuration")
        else:
            print("🔧 Address these issues first:")
            
            failed_checks = [name for name, status in all_checks.items() if not status]
            for check in failed_checks[:5]:  # Show top 5 issues
                print(f"   - Fix {check.replace('_', ' ')}")
            
            print("\n   Then re-run this health check to verify fixes")
        
        print(f"\n💾 Health check data saved to: {config.chroma_db.persist_directory}")


if __name__ == "__main__":
    main()
