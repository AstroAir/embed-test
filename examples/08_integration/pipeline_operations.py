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

import contextlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from examples.utils.example_helpers import (
    example_context,
    print_section,
    print_subsection,
)
from examples.utils.sample_data_generator import ensure_sample_data

from vectorflow import Config, PDFVectorPipeline
from vectorflow.core.config.settings import EmbeddingModelType

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


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
        self.results: list[ProcessingResult] = []
        self.lock = threading.Lock()

    def start_monitoring(self):
        """Start monitoring pipeline operations."""
        self.start_time = time.time()
        self.results.clear()

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
        average_processing_time = (
            total_processing_time / total_documents if total_documents > 0 else 0
        )

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
            throughput_chunks_per_second=throughput_chunks_per_second,
        )

    def print_metrics(self):
        """Print current metrics."""
        metrics = self.get_metrics()
        if metrics.total_documents == 0:
            print("No documents have been processed yet.")
            return

        print("\nPipeline metrics:")
        print(
            f"  Documents: total={metrics.total_documents}, "
            f"success={metrics.successful_documents}, "
            f"failed={metrics.failed_documents}"
        )
        print(f"  Total chunks processed: {metrics.total_chunks}")
        print(f"  Total processing time: {metrics.total_processing_time:.3f}s")
        print(
            "  Throughput: "
            f"{metrics.throughput_docs_per_second:.2f} docs/s, "
            f"{metrics.throughput_chunks_per_second:.2f} chunks/s"
        )

        if metrics.failed_documents > 0:
            print("  Warning: Some documents failed during processing.")


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

    return PDFVectorPipeline(config)


def demonstrate_single_document_processing(
    pipeline: PDFVectorPipeline, monitor: PipelineMonitor
) -> None:
    """Demonstrate processing a single document with monitoring."""
    print_subsection("Single Document Processing")

    # Ensure sample data exists
    sample_dir = Path("examples/sample_data")
    if not ensure_sample_data(sample_dir):
        return

    pdf_files = list(sample_dir.glob("*.pdf"))
    if not pdf_files:
        return

    pdf_file = pdf_files[0]

    start_time = time.time()

    try:
        result = pipeline.process_pdf(
            pdf_path=pdf_file, document_id=pdf_file.stem, show_progress=True
        )

        processing_time = time.time() - start_time

        if result.success:
            processing_result = ProcessingResult(
                success=True,
                document_id=pdf_file.stem,
                processing_time=processing_time,
                chunks_processed=result.chunks_processed,
            )

        else:
            processing_result = ProcessingResult(
                success=False,
                document_id=pdf_file.stem,
                processing_time=processing_time,
                chunks_processed=0,
                error_message=result.error_message,
            )

        monitor.record_result(processing_result)

    except Exception as e:
        processing_time = time.time() - start_time
        processing_result = ProcessingResult(
            success=False,
            document_id=pdf_file.stem,
            processing_time=processing_time,
            chunks_processed=0,
            error_message=str(e),
        )

        monitor.record_result(processing_result)


def demonstrate_batch_processing(
    pipeline: PDFVectorPipeline, monitor: PipelineMonitor
) -> None:
    """Demonstrate batch processing multiple documents."""
    print_subsection("Batch Processing")

    # Get available PDF files
    sample_dir = Path("examples/sample_data")
    pdf_files = list(sample_dir.glob("*.pdf"))

    if len(pdf_files) < 2:
        return

    # Limit to first 3 files for demonstration
    pdf_files = pdf_files[:3]

    print(f"Found {len(pdf_files)} PDF file(s) for batch processing:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")

    def process_document(pdf_file: Path) -> ProcessingResult:
        """Process a single document and return result."""
        start_time = time.time()

        try:
            result = pipeline.process_pdf(
                pdf_path=pdf_file,
                document_id=f"batch_{pdf_file.stem}",
                show_progress=False,  # Quiet for batch processing
            )

            processing_time = time.time() - start_time

            if result.success:
                return ProcessingResult(
                    success=True,
                    document_id=f"batch_{pdf_file.stem}",
                    processing_time=processing_time,
                    chunks_processed=result.chunks_processed,
                )
            return ProcessingResult(
                success=False,
                document_id=f"batch_{pdf_file.stem}",
                processing_time=processing_time,
                chunks_processed=0,
                error_message=result.error_message,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                success=False,
                document_id=f"batch_{pdf_file.stem}",
                processing_time=processing_time,
                chunks_processed=0,
                error_message=str(e),
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
                    print(
                        f"  Processed {result.document_id} "
                        f"({pdf_file.name}) in {result.processing_time:.3f}s, "
                        f"chunks={result.chunks_processed}"
                    )
                else:
                    print(
                        f"  Failed to process {result.document_id} "
                        f"({pdf_file.name}): {result.error_message}"
                    )

            except Exception as exc:
                print(f"  Batch task for {pdf_file.name} raised an exception: {exc}")

    batch_duration = time.time() - batch_start_time
    print(f"\nBatch processing completed in {batch_duration:.3f}s.")


def demonstrate_error_handling(
    pipeline: PDFVectorPipeline, monitor: PipelineMonitor
) -> None:
    """Demonstrate error handling and recovery."""
    print_subsection("Error Handling and Recovery")

    # Test 1: Non-existent file
    fake_file = Path("non_existent_file.pdf")

    start_time = time.time()
    try:
        result = pipeline.process_pdf(
            pdf_path=fake_file, document_id="fake_document", show_progress=False
        )

        processing_time = time.time() - start_time

        if not result.success:
            monitor.record_result(
                ProcessingResult(
                    success=False,
                    document_id="fake_document",
                    processing_time=processing_time,
                    chunks_processed=0,
                    error_message=result.error_message,
                )
            )
            print(
                "  Non-existent file correctly produced an error: "
                f"{result.error_message}"
            )
        else:
            print(
                "  Warning: processing non-existent file succeeded unexpectedly. "
                "Check error handling."
            )

    except Exception as e:
        processing_time = time.time() - start_time
        monitor.record_result(
            ProcessingResult(
                success=False,
                document_id="fake_document",
                processing_time=processing_time,
                chunks_processed=0,
                error_message=str(e),
            )
        )

    # Test 2: Invalid configuration

    # Temporarily modify configuration to cause an error
    original_batch_size = pipeline.config.embedding.batch_size

    try:
        # Set invalid batch size
        pipeline.config.embedding.batch_size = 0

        # Try to process (should handle gracefully)
        sample_dir = Path("examples/sample_data")
        pdf_files = list(sample_dir.glob("*.pdf"))

        if pdf_files:
            result = pipeline.process_pdf(
                pdf_path=pdf_files[0], document_id="error_test", show_progress=False
            )

            if not result.success:
                print(
                    "  Invalid configuration produced an error as expected: "
                    f"{result.error_message}"
                )
            else:
                print(
                    "  Warning: invalid configuration unexpectedly succeeded. "
                    "Validation may be missing."
                )

    except Exception as exc:
        print(f"  Error while testing invalid configuration: {exc}")

    finally:
        # Restore original configuration
        pipeline.config.embedding.batch_size = original_batch_size

    # Test 3: Resource exhaustion simulation

    # Check current resource usage
    try:
        import psutil

        memory_percent = psutil.virtual_memory().percent

        print(f"  Current memory usage: {memory_percent:.1f}%")
        if memory_percent > 80:
            print(
                "  Warning: high memory usage detected (> 80%). "
                "Consider reducing batch size or workers."
            )
        else:
            print("  Memory usage is within normal range.")

    except ImportError:
        print("  psutil not installed; skipping resource exhaustion simulation.")


def demonstrate_pipeline_monitoring(
    pipeline: PDFVectorPipeline, monitor: PipelineMonitor
) -> None:
    """Demonstrate pipeline monitoring and health checks."""
    print_subsection("Pipeline Monitoring")

    # Health check
    try:
        health = pipeline.health_check()

        for component, status in health.items():
            label = "OK" if status else "FAILED"
            print(f"  {component}: {label}")

    except Exception as exc:
        print(f"  Pipeline health check failed: {exc}")

    # Collection statistics
    with contextlib.suppress(Exception):
        stats = pipeline.get_collection_stats()
        if isinstance(stats, dict):
            total_chunks = stats.get("total_chunks")
            unique_docs = stats.get("unique_documents")
            print("\nCollection statistics:")
            print(f"  Total chunks: {total_chunks}")
            print(f"  Unique documents: {unique_docs}")

    # Performance metrics from monitor
    monitor.print_metrics()

    # Resource usage
    try:
        import psutil

        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # Check for resource warnings
        print(
            f"\nResource usage: CPU={cpu_percent:.1f}%, "
            f"Memory={memory.percent:.1f}%"
        )
        if cpu_percent > 80:
            print("  Warning: high CPU usage (> 80%).")
        if memory.percent > 80:
            print("  Warning: high memory usage (> 80%).")

    except ImportError:
        print("  psutil not installed; skipping detailed resource usage.")


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


if __name__ == "__main__":
    main()
