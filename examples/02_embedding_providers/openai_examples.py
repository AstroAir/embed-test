"""
OpenAI Embeddings Examples

This example demonstrates using OpenAI's embedding models:
- Different OpenAI embedding models
- API configuration and authentication
- Cost optimization strategies
- Error handling and rate limiting

Prerequisites:
- PDF Vector System installed
- OpenAI API key (set OPENAI_API_KEY environment variable)

Usage:
    python openai_examples.py

Expected Output:
    - OpenAI model demonstrations
    - Cost analysis and optimization
    - Rate limiting and error handling examples
    - Best practices for production use

Learning Objectives:
- Understand OpenAI embedding models
- Learn cost optimization techniques
- Master API error handling
- See production deployment patterns
"""

import sys
import time
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pdf_vector_system import Config, PDFVectorPipeline
from pdf_vector_system.config.settings import EmbeddingModelType
from utils.example_helpers import (
    print_section, print_subsection, example_context,
    check_api_key, format_duration
)


@dataclass
class OpenAIModelInfo:
    """Information about OpenAI embedding models."""
    name: str
    description: str
    dimensions: int
    max_tokens: int
    cost_per_1k_tokens: float
    use_case: str
    performance_tier: str


def get_openai_models() -> Dict[str, OpenAIModelInfo]:
    """Get information about available OpenAI embedding models."""
    return {
        "text-embedding-3-small": OpenAIModelInfo(
            name="text-embedding-3-small",
            description="Most efficient OpenAI embedding model",
            dimensions=1536,
            max_tokens=8191,
            cost_per_1k_tokens=0.00002,  # $0.00002 per 1K tokens
            use_case="General purpose, cost-sensitive applications",
            performance_tier="Efficient"
        ),
        "text-embedding-3-large": OpenAIModelInfo(
            name="text-embedding-3-large",
            description="Highest quality OpenAI embedding model",
            dimensions=3072,
            max_tokens=8191,
            cost_per_1k_tokens=0.00013,  # $0.00013 per 1K tokens
            use_case="High-quality applications, research",
            performance_tier="Premium"
        ),
        "text-embedding-ada-002": OpenAIModelInfo(
            name="text-embedding-ada-002",
            description="Legacy OpenAI embedding model (deprecated)",
            dimensions=1536,
            max_tokens=8191,
            cost_per_1k_tokens=0.0001,  # $0.0001 per 1K tokens
            use_case="Legacy applications (use text-embedding-3-small instead)",
            performance_tier="Legacy"
        )
    }


def check_openai_availability() -> bool:
    """Check if OpenAI API is available and configured."""
    print_subsection("OpenAI API Configuration")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OpenAI API key not found")
        print("   Set the OPENAI_API_KEY environment variable:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print()
        print("   Or create a .env file with:")
        print("   OPENAI_API_KEY=your-api-key-here")
        return False
    
    # Mask the API key for display
    masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
    print(f"✅ OpenAI API key found: {masked_key}")
    
    return True


def demonstrate_model_characteristics() -> None:
    """Demonstrate characteristics of OpenAI embedding models."""
    print_subsection("OpenAI Model Characteristics")
    
    models = get_openai_models()
    
    print("🤖 OpenAI Embedding Models:")
    
    for model_name, info in models.items():
        print(f"\n📋 {model_name}:")
        print(f"   Description: {info.description}")
        print(f"   Dimensions: {info.dimensions:,}")
        print(f"   Max tokens: {info.max_tokens:,}")
        print(f"   Cost per 1K tokens: ${info.cost_per_1k_tokens:.5f}")
        print(f"   Performance tier: {info.performance_tier}")
        print(f"   Best for: {info.use_case}")
        
        if info.performance_tier == "Legacy":
            print(f"   ⚠️  Consider upgrading to text-embedding-3-small")


def benchmark_openai_model(model_name: str, model_info: OpenAIModelInfo) -> Optional[Dict[str, Any]]:
    """Benchmark a specific OpenAI model."""
    print(f"   🔍 Testing {model_name}...")
    
    try:
        # Create configuration for this model
        config = Config()
        config.embedding.model_type = EmbeddingModelType.OPENAI
        config.embedding.model_name = model_name
        config.embedding.batch_size = 50  # OpenAI supports larger batches
        config.chroma_db.collection_name = f"test_openai_{model_name.replace('-', '_')}"
        config.chroma_db.persist_directory = Path(f"./test_openai_{model_name.replace('-', '_')}_db")
        config.debug = False
        
        # Test texts
        test_texts = [
            "OpenAI's embedding models provide high-quality vector representations of text.",
            "These models are trained on diverse internet text and can understand context.",
            "The text-embedding-3-small model offers the best balance of cost and performance.",
            "For applications requiring the highest quality, text-embedding-3-large is recommended.",
            "Proper batch processing can significantly reduce API costs and improve throughput."
        ]
        
        # Create pipeline and measure performance
        start_time = time.time()
        pipeline = PDFVectorPipeline(config)
        
        # Measure embedding generation
        embed_start = time.time()
        result = pipeline.batch_processor.process_texts(test_texts, show_progress=False)
        embed_time = time.time() - embed_start
        
        if result.success:
            total_time = time.time() - start_time
            embeddings_per_second = len(test_texts) / embed_time
            
            # Estimate cost (rough approximation)
            total_chars = sum(len(text) for text in test_texts)
            estimated_tokens = total_chars // 4  # Rough token estimation
            estimated_cost = (estimated_tokens / 1000) * model_info.cost_per_1k_tokens
            
            return {
                "model_name": model_name,
                "total_time": total_time,
                "embed_time": embed_time,
                "embeddings_per_second": embeddings_per_second,
                "dimensions": len(result.embeddings[0]) if result.embeddings else 0,
                "estimated_cost": estimated_cost,
                "estimated_tokens": estimated_tokens,
                "success": True
            }
        else:
            print(f"      ❌ Failed: {result.error_message}")
            return None
            
    except Exception as e:
        print(f"      ❌ Error: {e}")
        return None


def run_openai_benchmarks() -> List[Dict[str, Any]]:
    """Run benchmarks for available OpenAI models."""
    print_subsection("OpenAI Performance Benchmarks")
    
    if not check_api_key("OPENAI_API_KEY"):
        print("⚠️  Skipping benchmarks - API key not available")
        return []
    
    models = get_openai_models()
    benchmarks = []
    
    print("⚡ Running OpenAI model benchmarks...")
    
    # Test current models (skip legacy)
    test_models = ["text-embedding-3-small", "text-embedding-3-large"]
    
    for model_name in test_models:
        if model_name in models:
            model_info = models[model_name]
            benchmark = benchmark_openai_model(model_name, model_info)
            
            if benchmark:
                benchmarks.append(benchmark)
                print(f"      ✅ {model_name}:")
                print(f"         Total time: {format_duration(benchmark['total_time'])}")
                print(f"         Embedding time: {format_duration(benchmark['embed_time'])}")
                print(f"         Speed: {benchmark['embeddings_per_second']:.1f} embeddings/sec")
                print(f"         Dimensions: {benchmark['dimensions']:,}")
                print(f"         Estimated cost: ${benchmark['estimated_cost']:.6f}")
                print(f"         Estimated tokens: {benchmark['estimated_tokens']:,}")
    
    return benchmarks


def demonstrate_cost_optimization() -> None:
    """Demonstrate cost optimization strategies for OpenAI."""
    print_subsection("Cost Optimization Strategies")
    
    print("💰 OpenAI Cost Optimization Techniques:")
    
    # Batch processing optimization
    print("\n1. 📦 Batch Processing:")
    batch_strategies = [
        "Use larger batch sizes (up to 2048 inputs per request)",
        "Combine multiple small requests into batches",
        "Process documents in chunks to maximize batch efficiency",
        "Monitor rate limits to avoid throttling",
        "Use async processing for large volumes"
    ]
    
    for strategy in batch_strategies:
        print(f"   ✅ {strategy}")
    
    # Model selection optimization
    print("\n2. 🎯 Model Selection:")
    model_guidance = {
        "text-embedding-3-small": {
            "cost_factor": "1x",
            "quality": "High",
            "recommendation": "Default choice for most applications"
        },
        "text-embedding-3-large": {
            "cost_factor": "6.5x",
            "quality": "Highest",
            "recommendation": "Use only when highest quality is required"
        }
    }
    
    for model, info in model_guidance.items():
        print(f"   {model}:")
        print(f"     - Relative cost: {info['cost_factor']}")
        print(f"     - Quality: {info['quality']}")
        print(f"     - Recommendation: {info['recommendation']}")
    
    # Cost estimation example
    print("\n3. 📊 Cost Estimation Example:")
    
    example_scenarios = [
        {
            "scenario": "Small application (10K documents, 500 tokens each)",
            "tokens": 5_000_000,
            "small_cost": 5_000_000 / 1000 * 0.00002,
            "large_cost": 5_000_000 / 1000 * 0.00013
        },
        {
            "scenario": "Medium application (100K documents, 500 tokens each)",
            "tokens": 50_000_000,
            "small_cost": 50_000_000 / 1000 * 0.00002,
            "large_cost": 50_000_000 / 1000 * 0.00013
        }
    ]
    
    for scenario in example_scenarios:
        print(f"   {scenario['scenario']}:")
        print(f"     - Total tokens: {scenario['tokens']:,}")
        print(f"     - text-embedding-3-small: ${scenario['small_cost']:.2f}")
        print(f"     - text-embedding-3-large: ${scenario['large_cost']:.2f}")
        print(f"     - Cost difference: ${scenario['large_cost'] - scenario['small_cost']:.2f}")


def demonstrate_error_handling() -> None:
    """Demonstrate error handling and rate limiting for OpenAI API."""
    print_subsection("Error Handling and Rate Limiting")
    
    print("🛡️  OpenAI API Error Handling:")
    
    # Common error scenarios
    print("\n1. 🚨 Common Error Scenarios:")
    error_scenarios = [
        {
            "error": "Rate limit exceeded",
            "cause": "Too many requests per minute/day",
            "solution": "Implement exponential backoff, reduce batch size"
        },
        {
            "error": "Invalid API key",
            "cause": "Missing or incorrect API key",
            "solution": "Verify API key configuration"
        },
        {
            "error": "Token limit exceeded",
            "cause": "Input text too long",
            "solution": "Split text into smaller chunks"
        },
        {
            "error": "Network timeout",
            "cause": "Network connectivity issues",
            "solution": "Implement retry logic with timeouts"
        }
    ]
    
    for scenario in error_scenarios:
        print(f"   ❌ {scenario['error']}:")
        print(f"      Cause: {scenario['cause']}")
        print(f"      Solution: {scenario['solution']}")
    
    # Rate limiting best practices
    print("\n2. ⏱️  Rate Limiting Best Practices:")
    rate_limit_tips = [
        "Monitor rate limit headers in API responses",
        "Implement exponential backoff for rate limit errors",
        "Use smaller batch sizes if hitting rate limits",
        "Consider upgrading API tier for higher limits",
        "Implement request queuing for high-volume applications"
    ]
    
    for tip in rate_limit_tips:
        print(f"   ✅ {tip}")
    
    # Configuration example
    print("\n3. 🔧 Robust Configuration Example:")
    
    config_example = """
    config = Config()
    config.embedding.model_type = EmbeddingModelType.OPENAI
    config.embedding.model_name = "text-embedding-3-small"
    config.embedding.batch_size = 100  # Optimize for throughput
    config.embedding.max_retries = 5   # Handle transient errors
    config.embedding.timeout_seconds = 60  # Reasonable timeout
    
    # Add retry delay for rate limiting
    config.embedding.retry_delay = 1.0  # Start with 1 second
    """
    
    print(f"   {config_example}")


def demonstrate_production_patterns() -> None:
    """Demonstrate production deployment patterns for OpenAI."""
    print_subsection("Production Deployment Patterns")
    
    print("🚀 Production Best Practices:")
    
    # Security practices
    print("\n1. 🔐 Security:")
    security_practices = [
        "Store API keys in environment variables or secret managers",
        "Never commit API keys to version control",
        "Use different API keys for different environments",
        "Implement API key rotation procedures",
        "Monitor API usage for anomalies"
    ]
    
    for practice in security_practices:
        print(f"   🔒 {practice}")
    
    # Monitoring and observability
    print("\n2. 📊 Monitoring:")
    monitoring_practices = [
        "Track API usage and costs",
        "Monitor response times and error rates",
        "Set up alerts for rate limit violations",
        "Log API errors with sufficient context",
        "Implement health checks for API connectivity"
    ]
    
    for practice in monitoring_practices:
        print(f"   📈 {practice}")
    
    # Scalability patterns
    print("\n3. 📈 Scalability:")
    scalability_patterns = [
        "Use connection pooling for API requests",
        "Implement caching for frequently used embeddings",
        "Use async processing for large batches",
        "Consider embedding caching strategies",
        "Plan for API quota management"
    ]
    
    for pattern in scalability_patterns:
        print(f"   ⚡ {pattern}")


def main() -> None:
    """
    Demonstrate OpenAI embedding usage and best practices.
    
    This function shows how to effectively use OpenAI's embedding
    models with proper configuration, error handling, and optimization.
    """
    with example_context("OpenAI Embeddings Examples"):
        
        print_section("OpenAI Embeddings Overview")
        
        # Check API availability
        api_available = check_openai_availability()
        
        # Show model characteristics
        demonstrate_model_characteristics()
        
        if api_available:
            # Run benchmarks if API is available
            benchmarks = run_openai_benchmarks()
        else:
            print("⚠️  Skipping benchmarks - configure API key to test")
        
        print_section("Optimization and Best Practices")
        
        # Show cost optimization
        demonstrate_cost_optimization()
        
        # Show error handling
        demonstrate_error_handling()
        
        # Show production patterns
        demonstrate_production_patterns()
        
        print_section("OpenAI Embeddings Summary")
        
        print("🎯 Key Advantages of OpenAI Embeddings:")
        print()
        print("1. 🏆 Quality:")
        print("   - State-of-the-art embedding quality")
        print("   - Regular model improvements")
        print("   - Excellent semantic understanding")
        print("   - Strong performance across domains")
        print()
        print("2. 🔧 Ease of Use:")
        print("   - Simple API integration")
        print("   - No model management required")
        print("   - Automatic scaling")
        print("   - Comprehensive documentation")
        print()
        print("3. 🚀 Performance:")
        print("   - High throughput with batching")
        print("   - Low latency for single requests")
        print("   - Reliable service availability")
        print("   - Global infrastructure")
        print()
        print("4. 💰 Cost Efficiency:")
        print("   - Pay-per-use pricing")
        print("   - No infrastructure costs")
        print("   - Predictable pricing")
        print("   - Volume discounts available")
        print()
        print("Best Practices:")
        print("- Use text-embedding-3-small for most applications")
        print("- Implement proper error handling and retries")
        print("- Optimize batch sizes for your use case")
        print("- Monitor costs and usage patterns")
        print("- Secure API keys properly")
        print()
        print("Next steps:")
        print("- Set up your OpenAI API key to test the examples")
        print("- Compare with other providers using provider_comparison.py")
        print("- Implement cost monitoring for your application")


if __name__ == "__main__":
    main()
