"""
Pipeline Operations Example

This example demonstrates advanced pipeline operations and monitoring:
- Pipeline lifecycle management
- Batch processing operations
- Error handling and recovery
- Performance monitoring
- Resource management

Prerequisites:
- PDF Vector System installed
- Sample PDF files
- Understanding of pipeline concepts

Usage:
    python pipeline_operations.py

Expected Output:
    - Pipeline initialization and configuration
    - Batch processing demonstrations
    - Error handling examples
    - Performance monitoring results
    - Resource usage tracking

Learning Objectives:
- Master pipeline lifecycle management
- Learn batch processing patterns
- Understand error handling strategies
- See performance monitoring techniques
- Learn resource management best practices
"""

import sys
import time
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pdf_vector_system import Config, PDFVectorPipeline
from pdf_vector_system.config.settings import EmbeddingModelType
from utils.example_helpers import (
    print_section, print_subsection, example_context,
    format_bytes, format_duration
)
from utils.sample_data_generator import ensure_sample_data


@dataclass
class ProcessingResult:
    """Result of a processing operation."""
    success: bool
    document_id: str
    processing_time: float
    chunks_processed: int
    error_message: Optional[str] = None


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics."""
    total_documents: int
    successful_documents: int
    failed_documents: int
    total_chunks: int
    total_processing_time: float
    average_processing_time: float
    throughput_docs_per_second: float
    throughput_chunks_per_second: float


class PipelineMonitor:
    """Monitor pipeline operations and collect metrics."""
    
    def __init__(self):
        self.start_time = None
        self.results: List[ProcessingResult] = []
        self.lock = threading.Lock()
    
    def start_monitoring(self):
        """Start monitoring pipeline operations."""
        self.start_time = time.time()
        self.results.clear()
        print("📊 Pipeline monitoring started")
    
    def record_result(self, result: ProcessingResult):
        """Record a processing result."""
        with self.lock:
            self.results.append(result)
    
    def get_metrics(self) -> PipelineMetrics:
        """Calculate and return current metrics."""
        if not self.results:
            return PipelineMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        total_documents = len(self.results)
        successful_documents = len(successful_results)
        failed_documents = len(failed_results)
        
        total_chunks = sum(r.chunks_processed for r in successful_results)
        total_processing_time = sum(r.processing_time for r in self.results)
        
        elapsed_time = time.time() - self.start_time if self.start_time else 1
        average_processing_time = total_processing_time / total_documents if total_documents > 0 else 0
        
        throughput_docs_per_second = total_documents / elapsed_time
        throughput_chunks_per_second = total_chunks / elapsed_time
        
        return PipelineMetrics(
            total_documents=total_documents,
            successful_documents=successful_documents,
            failed_documents=failed_documents,
            total_chunks=total_chunks,
            total_processing_time=total_processing_time,
            average_processing_time=average_processing_time,
            throughput_docs_per_second=throughput_docs_per_second,
            throughput_chunks_per_second=throughput_chunks_per_second
        )
    
    def print_metrics(self):
        """Print current metrics."""
        metrics = self.get_metrics()
        
        print(f"\n📊 Pipeline Metrics:")
        print(f"   Documents: {metrics.successful_documents}/{metrics.total_documents} successful")
        print(f"   Chunks: {metrics.total_chunks:,} processed")
        print(f"   Processing time: {format_duration(metrics.total_processing_time)}")
        print(f"   Average time/doc: {format_duration(metrics.average_processing_time)}")
        print(f"   Throughput: {metrics.throughput_docs_per_second:.1f} docs/sec, {metrics.throughput_chunks_per_second:.1f} chunks/sec")
        
        if metrics.failed_documents > 0:
            print(f"   ⚠️  Failed documents: {metrics.failed_documents}")


def setup_pipeline() -> PDFVectorPipeline:
    """Set up and configure the pipeline."""
    print_subsection("Pipeline Setup")
    
    config = Config()
    config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
    config.embedding.model_name = "all-MiniLM-L6-v2"
    config.embedding.batch_size = 32
    config.chroma_db.collection_name = "pipeline_operations"
    config.chroma_db.persist_directory = Path("./pipeline_operations_db")
    config.debug = True
    config.max_workers = 4
    
    pipeline = PDFVectorPipeline(config)
    
    print(f"✅ Pipeline configured:")
    print(f"   - Embedding model: {config.embedding.model_name}")
    print(f"   - Batch size: {config.embedding.batch_size}")
    print(f"   - Max workers: {config.max_workers}")
    print(f"   - Collection: {config.chroma_db.collection_name}")
    
    return pipeline


def demonstrate_single_document_processing(pipeline: PDFVectorPipeline, monitor: PipelineMonitor) -> None:
    """Demonstrate processing a single document with monitoring."""
    print_subsection("Single Document Processing")
    
    # Ensure sample data exists
    sample_dir = Path("examples/sample_data")
    if not ensure_sample_data(sample_dir):
        print("⚠️  No sample data available")
        return
    
    pdf_files = list(sample_dir.glob("*.pdf"))
    if not pdf_files:
        print("📄 No PDF files found")
        return
    
    pdf_file = pdf_files[0]
    print(f"📖 Processing: {pdf_file.name}")
    
    start_time = time.time()
    
    try:
        result = pipeline.process_pdf(
            pdf_path=pdf_file,
            document_id=pdf_file.stem,
            show_progress=True
        )
        
        processing_time = time.time() - start_time
        
        if result.success:
            processing_result = ProcessingResult(
                success=True,
                document_id=pdf_file.stem,
                processing_time=processing_time,
                chunks_processed=result.chunks_processed
            )
            
            print(f"✅ Successfully processed {pdf_file.name}")
            print(f"   - Chunks: {result.chunks_processed}")
            print(f"   - Time: {format_duration(processing_time)}")
            print(f"   - Speed: {result.chunks_per_second:.1f} chunks/sec")
        else:
            processing_result = ProcessingResult(
                success=False,
                document_id=pdf_file.stem,
                processing_time=processing_time,
                chunks_processed=0,
                error_message=result.error_message
            )
            
            print(f"❌ Failed to process {pdf_file.name}: {result.error_message}")
        
        monitor.record_result(processing_result)
        
    except Exception as e:
        processing_time = time.time() - start_time
        processing_result = ProcessingResult(
            success=False,
            document_id=pdf_file.stem,
            processing_time=processing_time,
            chunks_processed=0,
            error_message=str(e)
        )
        
        monitor.record_result(processing_result)
        print(f"❌ Exception processing {pdf_file.name}: {e}")


def demonstrate_batch_processing(pipeline: PDFVectorPipeline, monitor: PipelineMonitor) -> None:
    """Demonstrate batch processing multiple documents."""
    print_subsection("Batch Processing")
    
    # Get available PDF files
    sample_dir = Path("examples/sample_data")
    pdf_files = list(sample_dir.glob("*.pdf"))
    
    if len(pdf_files) < 2:
        print("⚠️  Need at least 2 PDF files for batch processing demonstration")
        return
    
    # Limit to first 3 files for demonstration
    pdf_files = pdf_files[:3]
    
    print(f"📚 Batch processing {len(pdf_files)} documents:")
    for pdf_file in pdf_files:
        print(f"   - {pdf_file.name}")
    
    def process_document(pdf_file: Path) -> ProcessingResult:
        """Process a single document and return result."""
        start_time = time.time()
        
        try:
            result = pipeline.process_pdf(
                pdf_path=pdf_file,
                document_id=f"batch_{pdf_file.stem}",
                show_progress=False  # Quiet for batch processing
            )
            
            processing_time = time.time() - start_time
            
            if result.success:
                return ProcessingResult(
                    success=True,
                    document_id=f"batch_{pdf_file.stem}",
                    processing_time=processing_time,
                    chunks_processed=result.chunks_processed
                )
            else:
                return ProcessingResult(
                    success=False,
                    document_id=f"batch_{pdf_file.stem}",
                    processing_time=processing_time,
                    chunks_processed=0,
                    error_message=result.error_message
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                success=False,
                document_id=f"batch_{pdf_file.stem}",
                processing_time=processing_time,
                chunks_processed=0,
                error_message=str(e)
            )
    
    # Process documents in parallel
    batch_start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_document, pdf_file): pdf_file 
            for pdf_file in pdf_files
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            pdf_file = future_to_file[future]
            
            try:
                result = future.result()
                monitor.record_result(result)
                
                if result.success:
                    print(f"   ✅ {pdf_file.name}: {result.chunks_processed} chunks in {format_duration(result.processing_time)}")
                else:
                    print(f"   ❌ {pdf_file.name}: {result.error_message}")
                    
            except Exception as e:
                print(f"   ❌ {pdf_file.name}: Exception - {e}")
    
    batch_time = time.time() - batch_start_time
    print(f"\n📊 Batch processing completed in {format_duration(batch_time)}")


def demonstrate_error_handling(pipeline: PDFVectorPipeline, monitor: PipelineMonitor) -> None:
    """Demonstrate error handling and recovery."""
    print_subsection("Error Handling and Recovery")
    
    print("🔧 Testing error handling scenarios:")
    
    # Test 1: Non-existent file
    print("\n1. Non-existent file:")
    fake_file = Path("non_existent_file.pdf")
    
    start_time = time.time()
    try:
        result = pipeline.process_pdf(
            pdf_path=fake_file,
            document_id="fake_document",
            show_progress=False
        )
        
        processing_time = time.time() - start_time
        
        if not result.success:
            print(f"   ✅ Correctly handled missing file: {result.error_message}")
            monitor.record_result(ProcessingResult(
                success=False,
                document_id="fake_document",
                processing_time=processing_time,
                chunks_processed=0,
                error_message=result.error_message
            ))
        else:
            print(f"   ⚠️  Unexpectedly succeeded with missing file")
            
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"   ✅ Exception correctly caught: {e}")
        monitor.record_result(ProcessingResult(
            success=False,
            document_id="fake_document",
            processing_time=processing_time,
            chunks_processed=0,
            error_message=str(e)
        ))
    
    # Test 2: Invalid configuration
    print("\n2. Invalid configuration recovery:")
    
    # Temporarily modify configuration to cause an error
    original_batch_size = pipeline.config.embedding.batch_size
    
    try:
        # Set invalid batch size
        pipeline.config.embedding.batch_size = 0
        print(f"   Set invalid batch size: {pipeline.config.embedding.batch_size}")
        
        # Try to process (should handle gracefully)
        sample_dir = Path("examples/sample_data")
        pdf_files = list(sample_dir.glob("*.pdf"))
        
        if pdf_files:
            result = pipeline.process_pdf(
                pdf_path=pdf_files[0],
                document_id="error_test",
                show_progress=False
            )
            
            if not result.success:
                print(f"   ✅ Gracefully handled invalid configuration")
            else:
                print(f"   ⚠️  Processing succeeded despite invalid configuration")
        
    except Exception as e:
        print(f"   ✅ Configuration error caught: {e}")
    
    finally:
        # Restore original configuration
        pipeline.config.embedding.batch_size = original_batch_size
        print(f"   🔄 Restored batch size to {original_batch_size}")
    
    # Test 3: Resource exhaustion simulation
    print("\n3. Resource management:")
    
    # Check current resource usage
    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        print(f"   Current memory usage: {memory_percent:.1f}%")
        
        if memory_percent > 80:
            print(f"   ⚠️  High memory usage detected")
        else:
            print(f"   ✅ Memory usage within normal range")
            
    except ImportError:
        print(f"   ⚠️  psutil not available for resource monitoring")


def demonstrate_pipeline_monitoring(pipeline: PDFVectorPipeline, monitor: PipelineMonitor) -> None:
    """Demonstrate pipeline monitoring and health checks."""
    print_subsection("Pipeline Monitoring")
    
    print("📊 Pipeline health and status monitoring:")
    
    # Health check
    print("\n1. Health Check:")
    try:
        health = pipeline.health_check()
        
        for component, status in health.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {component}: {'Healthy' if status else 'Unhealthy'}")
            
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
    
    # Collection statistics
    print("\n2. Collection Statistics:")
    try:
        stats = pipeline.get_collection_stats()
        
        print(f"   📊 Collection: {stats.get('collection_name', 'Unknown')}")
        print(f"   📄 Total chunks: {stats.get('total_chunks', 0):,}")
        print(f"   📚 Unique documents: {stats.get('unique_documents', 0)}")
        print(f"   📝 Total characters: {stats.get('total_characters', 0):,}")
        print(f"   📏 Average chunk size: {stats.get('average_chunk_size', 0):.0f} chars")
        
    except Exception as e:
        print(f"   ❌ Statistics collection failed: {e}")
    
    # Performance metrics from monitor
    print("\n3. Performance Metrics:")
    monitor.print_metrics()
    
    # Resource usage
    print("\n4. Resource Usage:")
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        print(f"   💻 CPU usage: {cpu_percent:.1f}%")
        print(f"   🧠 Memory usage: {memory.percent:.1f}% ({format_bytes(memory.used)}/{format_bytes(memory.total)})")
        
        # Check for resource warnings
        if cpu_percent > 80:
            print(f"   ⚠️  High CPU usage detected")
        if memory.percent > 80:
            print(f"   ⚠️  High memory usage detected")
            
    except ImportError:
        print(f"   ⚠️  Resource monitoring requires psutil")


def main() -> None:
    """
    Demonstrate advanced pipeline operations and monitoring.
    
    This function shows how to manage pipeline operations,
    handle errors, and monitor performance effectively.
    """
    with example_context("Pipeline Operations"):
        
        print_section("Pipeline Operations and Monitoring")
        
        # Set up pipeline and monitoring
        pipeline = setup_pipeline()
        monitor = PipelineMonitor()
        monitor.start_monitoring()
        
        # Demonstrate different operation patterns
        demonstrate_single_document_processing(pipeline, monitor)
        demonstrate_batch_processing(pipeline, monitor)
        demonstrate_error_handling(pipeline, monitor)
        demonstrate_pipeline_monitoring(pipeline, monitor)
        
        print_section("Pipeline Operations Summary")
        
        # Final metrics
        print_subsection("Final Performance Summary")
        monitor.print_metrics()
        
        print("\n🎯 Pipeline Operations Best Practices:")
        print()
        print("1. 🔧 Pipeline Management:")
        print("   - Initialize pipeline with appropriate configuration")
        print("   - Implement health checks and monitoring")
        print("   - Use connection pooling for better performance")
        print()
        print("2. 📊 Monitoring:")
        print("   - Track processing metrics and performance")
        print("   - Monitor resource usage (CPU, memory)")
        print("   - Implement alerting for failures")
        print()
        print("3. 🔄 Error Handling:")
        print("   - Implement graceful error handling")
        print("   - Use retry mechanisms for transient failures")
        print("   - Log errors with sufficient context")
        print()
        print("4. ⚡ Performance:")
        print("   - Use batch processing for multiple documents")
        print("   - Implement parallel processing where appropriate")
        print("   - Monitor and optimize resource usage")
        print()
        print("5. 🛡️  Reliability:")
        print("   - Validate inputs before processing")
        print("   - Implement circuit breakers for external dependencies")
        print("   - Use timeouts to prevent hanging operations")
        print()
        print("Next steps:")
        print("- Explore performance optimization in 09_performance/")
        print("- Check production deployment patterns in 10_production/")
        print("- Try the monitoring and alerting examples")


if __name__ == "__main__":
    main()
