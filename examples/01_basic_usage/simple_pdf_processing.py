"""
Simple PDF Processing Example

This example demonstrates the most basic usage of the PDF Vector System.
It shows how to:
- Initialize the system with minimal configuration
- Process a single PDF file
- Perform basic searches
- Display results

Prerequisites:
- PDF Vector System installed
- Sample PDF files (will be created if not present)

Usage:
    python simple_pdf_processing.py

Expected Output:
    - PDF processing progress and statistics
    - Search results with similarity scores
    - System health status

Learning Objectives:
- Understand basic PDF Vector System workflow
- Learn minimal configuration requirements
- See how to process PDFs and search content
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import contextlib

from pdf_vector_system import Config, PDFVectorPipeline
from pdf_vector_system.config.settings import EmbeddingModelType
from utils.example_helpers import example_context, print_section, print_subsection
from utils.sample_data_generator import ensure_sample_data


def main() -> None:
    """
    Demonstrate simple PDF processing with the PDF Vector System.

    This function shows the complete workflow from initialization
    to search, using the simplest possible configuration.
    """
    with example_context("Simple PDF Processing"):
        # Step 1: Setup and Configuration
        print_section("Step 1: System Setup")

        # Create basic configuration
        config = Config()

        # Use local embeddings (no API key required)
        config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
        config.embedding.model_name = "all-MiniLM-L6-v2"
        config.embedding.batch_size = 16

        # Configure for example use
        config.chroma_db.collection_name = "simple_example"
        config.chroma_db.persist_directory = Path("./simple_example_db")
        config.debug = True

        # Step 2: Initialize Pipeline
        print_subsection("Initializing Pipeline")

        try:
            pipeline = PDFVectorPipeline(config)
        except Exception:
            return

        # Step 3: Ensure Sample Data
        print_subsection("Preparing Sample Data")

        sample_dir = Path("examples/sample_data")
        if not ensure_sample_data(sample_dir):
            # Create a simple text file as fallback
            test_file = sample_dir / "simple_test.txt"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text(
                """
            This is a simple test document for the PDF Vector System.

            It contains information about machine learning and artificial intelligence.
            Machine learning is a subset of artificial intelligence that enables
            computers to learn from data without being explicitly programmed.

            Key concepts in machine learning include:
            - Supervised learning with labeled data
            - Unsupervised learning for pattern discovery
            - Reinforcement learning for decision making
            - Deep learning with neural networks

            Applications of machine learning include:
            - Natural language processing
            - Computer vision
            - Recommendation systems
            - Predictive analytics
            """
            )

        # Find available PDF files
        pdf_files = list(sample_dir.glob("*.pdf"))
        if not pdf_files:
            return

        # Step 4: Process PDF
        print_section("Step 2: PDF Processing")

        # Use the first available PDF
        pdf_file = pdf_files[0]

        try:
            # Process the PDF with progress tracking
            result = pipeline.process_pdf(
                pdf_path=pdf_file, document_id=pdf_file.stem, show_progress=True
            )

            if result.success:
                pass
            else:
                return

        except Exception:
            return

        # Step 5: Perform Searches
        print_section("Step 3: Search Demonstration")

        # Define some example search queries
        search_queries = [
            "machine learning",
            "artificial intelligence",
            "data analysis",
            "neural networks",
        ]

        for query in search_queries:
            print_subsection(f"Searching for: '{query}'")

            try:
                # Perform search
                search_results = pipeline.search(query_text=query, n_results=3)

                if search_results:
                    for _i, result in enumerate(search_results, 1):
                        if result.page_number:
                            pass

                        # Show content preview (first 100 characters)
                        content_preview = result.content[:100]
                        if len(result.content) > 100:
                            content_preview += "..."
                else:
                    pass

            except Exception:
                pass

        # Step 6: System Information
        print_section("Step 4: System Information")

        # Show collection statistics
        print_subsection("Collection Statistics")
        with contextlib.suppress(Exception):
            pipeline.get_collection_stats()

        # Health check
        print_subsection("System Health Check")
        try:
            health = pipeline.health_check()
            for _component, _status in health.items():
                pass
        except Exception:
            pass

        # Step 7: Cleanup Information
        print_section("Step 5: Next Steps")


if __name__ == "__main__":
    main()
