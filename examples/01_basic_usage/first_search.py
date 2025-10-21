"""
First Search Example

This example demonstrates how to perform your first search with the PDF Vector System.
It shows:
- Processing a PDF and creating a searchable database
- Different types of search queries
- Understanding search results and similarity scores
- Search filtering and refinement

Prerequisites:
- PDF Vector System installed
- Sample PDF files (will be created if not present)

Usage:
    python first_search.py

Expected Output:
    - PDF processing with progress
    - Various search examples with results
    - Explanation of similarity scores
    - Search filtering demonstrations

Learning Objectives:
- Understand how semantic search works
- Learn to interpret similarity scores
- See different search patterns
- Learn search filtering techniques
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pdf_vector_system import Config, PDFVectorPipeline
from pdf_vector_system.config.settings import EmbeddingModelType
from utils.example_helpers import example_context, print_section, print_subsection
from utils.sample_data_generator import ensure_sample_data


def setup_pipeline() -> PDFVectorPipeline:
    """Set up and return a configured pipeline."""
    config = Config()

    # Use local embeddings for this example
    config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
    config.embedding.model_name = "all-MiniLM-L6-v2"
    config.embedding.batch_size = 16

    # Configure for search example
    config.chroma_db.collection_name = "search_example"
    config.chroma_db.persist_directory = Path("./search_example_db")
    config.debug = True

    return PDFVectorPipeline(config)


def process_sample_documents(pipeline: PDFVectorPipeline) -> bool:
    """Process sample documents for searching."""
    print_subsection("Processing Documents")

    # Ensure sample data exists
    sample_dir = Path("examples/sample_data")
    if not ensure_sample_data(sample_dir):
        return False

    # Find PDF files
    pdf_files = list(sample_dir.glob("*.pdf"))
    if not pdf_files:
        return False

    # Process each PDF
    processed_count = 0
    for pdf_file in pdf_files[:3]:  # Process up to 3 files
        try:
            result = pipeline.process_pdf(
                pdf_path=pdf_file,
                document_id=pdf_file.stem,
                show_progress=False,  # Quiet processing for this example
            )

            if result.success:
                processed_count += 1
            else:
                pass

        except Exception:
            pass

    return processed_count > 0


def demonstrate_basic_search(pipeline: PDFVectorPipeline) -> None:
    """Demonstrate basic search functionality."""
    print_subsection("Basic Search")

    # Simple search query
    query = "machine learning"

    try:
        results = pipeline.search(query_text=query, n_results=5)

        if results:
            for _i, result in enumerate(results, 1):
                if result.page_number:
                    pass

                # Show content preview
                content_lines = result.content.strip().split("\n")
                preview = content_lines[0][:100]
                if len(content_lines[0]) > 100:
                    preview += "..."
        else:
            pass

    except Exception:
        pass


def demonstrate_similarity_scores(pipeline: PDFVectorPipeline) -> None:
    """Demonstrate how similarity scores work."""
    print_subsection("Understanding Similarity Scores")

    # Test queries with different expected relevance
    test_queries = [
        ("artificial intelligence", "High relevance expected"),
        ("machine learning algorithms", "High relevance expected"),
        ("cooking recipes", "Low relevance expected"),
        ("the quick brown fox", "Very low relevance expected"),
    ]

    for query, _expectation in test_queries:
        try:
            results = pipeline.search(query_text=query, n_results=3)

            if results:
                best_score = results[0].score

                # Interpret the score
                if (
                    best_score > 0.8
                    or best_score > 0.6
                    or best_score > 0.4
                    or best_score > 0.2
                ):
                    pass
                else:
                    pass

                # Show score distribution
                [r.score for r in results]
            else:
                pass

        except Exception:
            pass


def demonstrate_search_filtering(pipeline: PDFVectorPipeline) -> None:
    """Demonstrate search filtering capabilities."""
    print_subsection("Search Filtering")

    # Get available documents
    try:
        stats = pipeline.get_collection_stats()
        stats.get("total_chunks", 0)
        unique_docs = stats.get("unique_documents", 0)

        if unique_docs == 0:
            return

    except Exception:
        return

    # Basic search without filtering
    query = "data analysis"

    try:
        all_results = pipeline.search(query_text=query, n_results=10)

        if all_results:
            # Show document distribution
            doc_counts = {}
            for result in all_results:
                doc_id = result.document_id
                doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

            for doc_id, _count in doc_counts.items():
                pass

            # Document-specific search
            if len(doc_counts) > 1:
                first_doc = next(iter(doc_counts.keys()))

                doc_results = pipeline.search(
                    query_text=query, n_results=5, document_id=first_doc
                )

                for _i, result in enumerate(doc_results[:3], 1):
                    pass

            # Page-specific search (if page info available)
            page_results = [r for r in all_results if r.page_number is not None]
            if page_results:
                page1_results = pipeline.search(
                    query_text=query, n_results=3, page_number=1
                )

                for _i, result in enumerate(page1_results, 1):
                    pass

    except Exception:
        pass


def demonstrate_search_tips(pipeline: PDFVectorPipeline) -> None:
    """Demonstrate search tips and best practices."""
    print_subsection("Search Tips and Best Practices")

    search_examples = [
        {"query": "neural networks", "tip": "Specific technical terms often work well"},
        {
            "query": "how to implement machine learning",
            "tip": "Natural language questions work with semantic search",
        },
        {
            "query": "deep learning CNN architecture",
            "tip": "Multiple related terms can improve results",
        },
        {
            "query": "performance optimization techniques",
            "tip": "Conceptual queries find related content",
        },
    ]

    for example in search_examples:
        query = example["query"]
        example["tip"]

        try:
            results = pipeline.search(query_text=query, n_results=3)

            if results:
                results[0].score

                # Show variety in results
                {r.document_id for r in results}
            else:
                pass

        except Exception:
            pass


def main() -> None:
    """
    Demonstrate search functionality with the PDF Vector System.

    This function shows how to perform various types of searches
    and understand the results.
    """
    with example_context("First Search"):
        print_section("Setup and Processing")

        # Initialize pipeline
        print_subsection("Initializing Pipeline")
        try:
            pipeline = setup_pipeline()
        except Exception:
            return

        # Process documents
        if not process_sample_documents(pipeline):
            return

        print_section("Search Demonstrations")

        # Demonstrate different search aspects
        demonstrate_basic_search(pipeline)
        demonstrate_similarity_scores(pipeline)
        demonstrate_search_filtering(pipeline)
        demonstrate_search_tips(pipeline)

        print_section("Search Summary")


if __name__ == "__main__":
    main()
