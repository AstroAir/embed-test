"""
Sentence Transformers Examples

This example demonstrates using Sentence Transformers models:
- Different model variants and their characteristics
- Local model management and optimization
- Performance tuning for different use cases
- Model selection guidance

Prerequisites:
- PDF Vector System installed
- sentence-transformers library

Usage:
    python sentence_transformers_examples.py

Expected Output:
    - Available model demonstrations
    - Performance comparisons
    - Model selection guidance
    - Optimization techniques

Learning Objectives:
- Understand Sentence Transformers ecosystem
- Learn model selection criteria
- See performance optimization techniques
- Master local embedding deployment
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pdf_vector_system import Config, PDFVectorPipeline
from pdf_vector_system.config.settings import EmbeddingModelType
from utils.example_helpers import (
    print_section, print_subsection, example_context,
    format_duration, format_bytes
)


@dataclass
class ModelInfo:
    """Information about a Sentence Transformers model."""
    name: str
    description: str
    dimensions: int
    max_seq_length: int
    size_mb: int
    use_case: str
    performance_tier: str


def get_sentence_transformer_models() -> Dict[str, ModelInfo]:
    """Get information about available Sentence Transformers models."""
    return {
        "all-MiniLM-L6-v2": ModelInfo(
            name="all-MiniLM-L6-v2",
            description="Fast and lightweight model, good for development",
            dimensions=384,
            max_seq_length=256,
            size_mb=90,
            use_case="Development, testing, resource-constrained environments",
            performance_tier="Fast"
        ),
        "all-MiniLM-L12-v2": ModelInfo(
            name="all-MiniLM-L12-v2", 
            description="Balanced speed and quality",
            dimensions=384,
            max_seq_length=256,
            size_mb=120,
            use_case="General purpose applications",
            performance_tier="Balanced"
        ),
        "all-mpnet-base-v2": ModelInfo(
            name="all-mpnet-base-v2",
            description="High quality model, slower but better results",
            dimensions=768,
            max_seq_length=384,
            size_mb=420,
            use_case="Production applications requiring high quality",
            performance_tier="High Quality"
        ),
        "all-distilroberta-v1": ModelInfo(
            name="all-distilroberta-v1",
            description="Distilled RoBERTa model, good balance",
            dimensions=768,
            max_seq_length=512,
            size_mb=290,
            use_case="Applications needing longer context",
            performance_tier="Balanced"
        ),
        "multi-qa-mpnet-base-dot-v1": ModelInfo(
            name="multi-qa-mpnet-base-dot-v1",
            description="Optimized for question-answering tasks",
            dimensions=768,
            max_seq_length=512,
            size_mb=420,
            use_case="Question answering, FAQ systems",
            performance_tier="Specialized"
        )
    }


def demonstrate_model_characteristics() -> None:
    """Demonstrate characteristics of different Sentence Transformers models."""
    print_subsection("Model Characteristics")
    
    models = get_sentence_transformer_models()
    
    print("🤖 Available Sentence Transformers Models:")
    
    for model_name, info in models.items():
        print(f"\n📋 {model_name}:")
        print(f"   Description: {info.description}")
        print(f"   Dimensions: {info.dimensions}")
        print(f"   Max sequence length: {info.max_seq_length}")
        print(f"   Model size: ~{info.size_mb} MB")
        print(f"   Performance tier: {info.performance_tier}")
        print(f"   Best for: {info.use_case}")


def benchmark_model(model_name: str, model_info: ModelInfo) -> Optional[Dict[str, Any]]:
    """Benchmark a specific Sentence Transformers model."""
    print(f"   🔍 Benchmarking {model_name}...")
    
    try:
        # Create configuration for this model
        config = Config()
        config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
        config.embedding.model_name = model_name
        config.embedding.batch_size = 16
        config.chroma_db.collection_name = f"test_{model_name.replace('-', '_')}"
        config.chroma_db.persist_directory = Path(f"./test_{model_name.replace('-', '_')}_db")
        config.debug = False
        
        # Test texts
        test_texts = [
            "Machine learning algorithms can automatically learn patterns from data.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning uses neural networks with multiple hidden layers.",
            "Artificial intelligence aims to create systems that can perform tasks requiring human intelligence.",
            "Vector databases efficiently store and search high-dimensional embeddings."
        ]
        
        # Create pipeline and measure initialization time
        init_start = time.time()
        pipeline = PDFVectorPipeline(config)
        init_time = time.time() - init_start
        
        # Measure embedding generation time
        embed_start = time.time()
        result = pipeline.batch_processor.process_texts(test_texts, show_progress=False)
        embed_time = time.time() - embed_start
        
        if result.success:
            embeddings_per_second = len(test_texts) / embed_time
            
            return {
                "model_name": model_name,
                "init_time": init_time,
                "embed_time": embed_time,
                "embeddings_per_second": embeddings_per_second,
                "dimensions": len(result.embeddings[0]) if result.embeddings else 0,
                "success": True
            }
        else:
            print(f"      ❌ Failed: {result.error_message}")
            return None
            
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return None


def run_model_benchmarks() -> List[Dict[str, Any]]:
    """Run benchmarks for available models."""
    print_subsection("Performance Benchmarks")
    
    models = get_sentence_transformer_models()
    benchmarks = []
    
    print("⚡ Running performance benchmarks...")
    print("   (First run may be slower due to model downloads)")
    
    # Test a subset of models to avoid long execution time
    test_models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
    
    for model_name in test_models:
        if model_name in models:
            model_info = models[model_name]
            benchmark = benchmark_model(model_name, model_info)
            
            if benchmark:
                benchmarks.append(benchmark)
                print(f"      ✅ {model_name}:")
                print(f"         Init time: {format_duration(benchmark['init_time'])}")
                print(f"         Embedding time: {format_duration(benchmark['embed_time'])}")
                print(f"         Speed: {benchmark['embeddings_per_second']:.1f} embeddings/sec")
                print(f"         Dimensions: {benchmark['dimensions']}")
    
    return benchmarks


def demonstrate_optimization_techniques() -> None:
    """Demonstrate optimization techniques for Sentence Transformers."""
    print_subsection("Optimization Techniques")
    
    print("🚀 Optimization Strategies for Sentence Transformers:")
    
    # Batch size optimization
    print("\n1. 📦 Batch Size Optimization:")
    batch_sizes = [1, 8, 16, 32, 64]
    
    print("   Recommended batch sizes by use case:")
    print("   - Interactive applications: 1-8")
    print("   - Batch processing: 16-32")
    print("   - High-throughput systems: 32-64+")
    print("   - Memory-constrained: 1-16")
    
    # Model selection optimization
    print("\n2. 🎯 Model Selection:")
    optimization_guide = {
        "Speed Priority": {
            "model": "all-MiniLM-L6-v2",
            "batch_size": 32,
            "use_case": "Real-time applications, development"
        },
        "Quality Priority": {
            "model": "all-mpnet-base-v2",
            "batch_size": 16,
            "use_case": "Production search, high-quality requirements"
        },
        "Balanced": {
            "model": "all-MiniLM-L12-v2",
            "batch_size": 24,
            "use_case": "General production applications"
        }
    }
    
    for priority, config in optimization_guide.items():
        print(f"   {priority}:")
        print(f"     - Model: {config['model']}")
        print(f"     - Batch size: {config['batch_size']}")
        print(f"     - Use case: {config['use_case']}")
    
    # Hardware optimization
    print("\n3. 🖥️  Hardware Optimization:")
    hardware_tips = [
        "Use GPU acceleration when available (CUDA/MPS)",
        "Increase batch size with more GPU memory",
        "Use CPU with multiple workers for CPU-only setups",
        "Consider model quantization for memory savings",
        "Use SSD storage for faster model loading"
    ]
    
    for tip in hardware_tips:
        print(f"   ✅ {tip}")


def demonstrate_model_management() -> None:
    """Demonstrate model management and caching."""
    print_subsection("Model Management")
    
    print("📁 Model Management Best Practices:")
    
    # Model caching
    print("\n1. 🗄️  Model Caching:")
    caching_info = [
        "Models are cached locally after first download",
        "Cache location: ~/.cache/huggingface/transformers/",
        "Shared across applications using the same model",
        "Can be pre-downloaded for offline deployment",
        "Consider cache size for multiple models"
    ]
    
    for info in caching_info:
        print(f"   📋 {info}")
    
    # Model versioning
    print("\n2. 🏷️  Model Versioning:")
    versioning_tips = [
        "Pin specific model versions in production",
        "Test new model versions before deployment",
        "Keep track of model performance changes",
        "Use model tags for reproducible deployments",
        "Document model selection rationale"
    ]
    
    for tip in versioning_tips:
        print(f"   📌 {tip}")
    
    # Deployment considerations
    print("\n3. 🚀 Deployment Considerations:")
    deployment_tips = [
        "Pre-download models in container images",
        "Use model warm-up for faster first requests",
        "Monitor model memory usage",
        "Implement model health checks",
        "Consider model serving frameworks for scale"
    ]
    
    for tip in deployment_tips:
        print(f"   🎯 {tip}")


def demonstrate_advanced_configuration() -> None:
    """Demonstrate advanced configuration options."""
    print_subsection("Advanced Configuration")
    
    print("⚙️  Advanced Configuration Examples:")
    
    # Custom configuration example
    print("\n1. 🔧 Custom Configuration:")
    
    config_example = """
    config = Config()
    config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
    config.embedding.model_name = "all-mpnet-base-v2"
    config.embedding.batch_size = 32
    config.embedding.max_retries = 3
    config.embedding.timeout_seconds = 60
    
    # Optimize for your hardware
    config.max_workers = 4  # Adjust based on CPU cores
    """
    
    print(f"   {config_example}")
    
    # Environment-specific configurations
    print("\n2. 🌍 Environment-Specific Configurations:")
    
    env_configs = {
        "Development": {
            "model": "all-MiniLM-L6-v2",
            "batch_size": 16,
            "workers": 2,
            "rationale": "Fast iteration, minimal resources"
        },
        "Testing": {
            "model": "all-MiniLM-L6-v2",
            "batch_size": 8,
            "workers": 1,
            "rationale": "Deterministic, minimal resources"
        },
        "Production": {
            "model": "all-mpnet-base-v2",
            "batch_size": 32,
            "workers": 8,
            "rationale": "High quality, optimized performance"
        }
    }
    
    for env, config in env_configs.items():
        print(f"   {env}:")
        print(f"     - Model: {config['model']}")
        print(f"     - Batch size: {config['batch_size']}")
        print(f"     - Workers: {config['workers']}")
        print(f"     - Rationale: {config['rationale']}")


def main() -> None:
    """
    Demonstrate Sentence Transformers usage and optimization.
    
    This function shows how to effectively use Sentence Transformers
    models for different use cases and optimize their performance.
    """
    with example_context("Sentence Transformers Examples"):
        
        print_section("Sentence Transformers Overview")
        
        # Show model characteristics
        demonstrate_model_characteristics()
        
        # Run performance benchmarks
        benchmarks = run_model_benchmarks()
        
        print_section("Optimization and Management")
        
        # Show optimization techniques
        demonstrate_optimization_techniques()
        
        # Show model management
        demonstrate_model_management()
        
        # Show advanced configuration
        demonstrate_advanced_configuration()
        
        print_section("Sentence Transformers Summary")
        
        print("🎯 Key Advantages of Sentence Transformers:")
        print()
        print("1. 🏠 Local Processing:")
        print("   - No API keys required")
        print("   - Complete data privacy")
        print("   - No network dependencies")
        print("   - Predictable costs")
        print()
        print("2. 🎛️  Model Variety:")
        print("   - Multiple models for different use cases")
        print("   - Speed vs quality trade-offs")
        print("   - Specialized models (QA, multilingual)")
        print("   - Regular community updates")
        print()
        print("3. ⚡ Performance:")
        print("   - GPU acceleration support")
        print("   - Efficient batch processing")
        print("   - Optimized for inference")
        print("   - Scalable deployment options")
        print()
        print("4. 🔧 Flexibility:")
        print("   - Easy model switching")
        print("   - Custom fine-tuning possible")
        print("   - Integration with popular frameworks")
        print("   - Extensive documentation")
        print()
        print("Best Practices:")
        print("- Start with all-MiniLM-L6-v2 for development")
        print("- Use all-mpnet-base-v2 for production quality")
        print("- Optimize batch size for your hardware")
        print("- Monitor memory usage with larger models")
        print("- Pre-download models for production deployment")
        print()
        print("Next steps:")
        print("- Try different models with your data")
        print("- Experiment with batch sizes and optimization")
        print("- Compare with cloud providers using provider_comparison.py")


if __name__ == "__main__":
    main()
