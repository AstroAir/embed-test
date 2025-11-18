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

    print_subsection("Pipeline Configuration")
    print("Using embedding model:", config.embedding.model_type.value)
    print("Model name:", config.embedding.model_name)
    print("Chroma collection:", config.chroma_db.collection_name)
    print("Persist directory:", config.chroma_db.persist_directory)

    return PDFVectorPipeline(config)


def process_sample_documents(pipeline: PDFVectorPipeline) -> bool:
    """Process sample documents for searching."""
    print_subsection("Processing Documents")

    # Ensure sample data exists
    sample_dir = Path("examples/sample_data")
    if not ensure_sample_data(sample_dir):
        print(
            "Sample data generation failed or no sample data available. "
            "Please check 'examples/sample_data'."
        )
        return False

    # Find PDF files
    pdf_files = list(sample_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in 'examples/sample_data'.")
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
                chunks = getattr(result, "chunks_processed", None)
                chunks_info = f", chunks={chunks}" if chunks is not None else ""
                print(f"  Processed {pdf_file.name} successfully{chunks_info}.")
            else:
                error_message = getattr(result, "error_message", "Unknown error")
                print(f"  Failed to process {pdf_file.name}: {error_message}")

        except Exception as e:
            print(f"  Error while processing {pdf_file.name}: {e}")

    if processed_count == 0:
        print("No documents were successfully processed.")
        return False

    print(f"Total processed documents: {processed_count}")
    return True


def demonstrate_basic_search(pipeline: PDFVectorPipeline) -> None:
    """Demonstrate basic search functionality."""
    print_subsection("Basic Search")

    # Simple search query
    query = "machine learning"

    try:
        results = pipeline.search(query_text=query, n_results=5)

        if results:
            print(f"Found {len(results)} result(s) for query '{query}'.")

            for i, result in enumerate(results, 1):
                page_info = (
                    f" (page {result.page_number})"
                    if getattr(result, "page_number", None) is not None
                    else ""
                )
                score = getattr(result, "score", None)
                score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "N/A"

                # Show content preview
                content_lines = result.content.strip().split("\n")
                first_line = content_lines[0] if content_lines else ""
                preview = first_line[:100]
                if len(first_line) > 100:
                    preview += "..."

                print(
                    f"  {i}. score={score_str}{page_info} - "
                    f"document_id={getattr(result, 'document_id', 'N/A')}"
                )
                print(f"     {preview}")
        else:
            print(f"No results found for query '{query}'.")

    except Exception as e:
        print(f"Basic search failed for query '{query}': {e}")


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

    for query, expectation in test_queries:
        try:
            results = pipeline.search(query_text=query, n_results=3)

            if results:
                best_score = results[0].score

                print(f"\nQuery: '{query}' ({expectation})")

                # Interpret the score
                if best_score is None:
                    interpretation = "Score unavailable"
                elif best_score >= 0.8:
                    interpretation = "Very strong semantic match"
                elif best_score >= 0.6:
                    interpretation = "Strong match"
                elif best_score >= 0.4:
                    interpretation = "Moderate match"
                elif best_score >= 0.2:
                    interpretation = "Weak match"
                else:
                    interpretation = "Very weak or no semantic match"

                print(f"  Best score: {best_score:.3f} -> {interpretation}")

                # Show score distribution
                scores = [r.score for r in results]
                score_str = ", ".join(f"{s:.3f}" for s in scores)
                print(f"  Score distribution: {score_str}")
            else:
                print(f"\nQuery: '{query}' ({expectation})")
                print("  No results returned for this query.")

        except Exception as e:
            print(f"Similarity score demonstration failed for query '{query}': {e}")


def demonstrate_search_filtering(pipeline: PDFVectorPipeline) -> None:
    """Demonstrate search filtering capabilities."""
    print_subsection("Search Filtering")

    # Get available documents
    try:
        stats = pipeline.get_collection_stats()
        total_chunks = stats.get("total_chunks", 0)
        unique_docs = stats.get("unique_documents", 0)

        if unique_docs == 0:
            print("Collection is empty. Skipping filtering demonstration.")
            return

        print(
            f"Collection has {unique_docs} unique document(s) and {total_chunks} chunks."
        )

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

            print(
                f"Found {len(all_results)} result(s) across {len(doc_counts)} document(s)."
            )
            for doc_id, count in doc_counts.items():
                print(f"  Document '{doc_id}': {count} result(s)")

            # Document-specific search
            if len(doc_counts) > 1:
                first_doc = next(iter(doc_counts.keys()))

                doc_results = pipeline.search(
                    query_text=query, n_results=5, document_id=first_doc
                )

                print(f"\nResults restricted to document '{first_doc}':")
                for i, result in enumerate(doc_results[:3], 1):
                    score = getattr(result, "score", None)
                    score_str = (
                        f"{score:.3f}" if isinstance(score, (int, float)) else "N/A"
                    )
                    print(
                        f"  {i}. score={score_str}, page={getattr(result, 'page_number', 'N/A')}"
                    )

            # Page-specific search (if page info available)
            page_results = [r for r in all_results if r.page_number is not None]
            if page_results:
                page1_results = pipeline.search(
                    query_text=query, n_results=3, page_number=1
                )

                print("\nResults restricted to page 1:")
                for i, result in enumerate(page1_results, 1):
                    score = getattr(result, "score", None)
                    score_str = (
                        f"{score:.3f}" if isinstance(score, (int, float)) else "N/A"
                    )
                    print(
                        f"  {i}. score={score_str}, "
                        f"document_id={getattr(result, 'document_id', 'N/A')}"
                    )

    except Exception as e:
        print(f"Search filtering demonstration failed: {e}")


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
        tip = example["tip"]

        print(f"\nQuery: '{query}'")
        print(f"Tip: {tip}")

        try:
            results = pipeline.search(query_text=query, n_results=3)

            if results:
                best = results[0]
                best_score = getattr(best, "score", None)
                best_score_str = (
                    f"{best_score:.3f}"
                    if isinstance(best_score, (int, float))
                    else "N/A"
                )

                doc_ids = {r.document_id for r in results}
                print(f"  Best score: {best_score_str}")
                print(
                    f"  Results come from {len(doc_ids)} distinct document(s): {doc_ids}"
                )
            else:
                print("  No results returned for this tip example.")

        except Exception as e:
            print(f"Search tip demonstration failed for query '{query}': {e}")


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
            print("Pipeline initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize pipeline: {e}")
            return

        # Process documents
        if not process_sample_documents(pipeline):
            print("Document processing failed. Cannot run search demonstrations.")
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
