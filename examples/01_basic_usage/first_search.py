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
from utils.example_helpers import (
    print_section, print_subsection, example_context,
    ensure_sample_data_exists
)
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
        print("⚠️  No sample data available")
        return False
    
    # Find PDF files
    pdf_files = list(sample_dir.glob("*.pdf"))
    if not pdf_files:
        print("📄 No PDF files found")
        return False
    
    # Process each PDF
    processed_count = 0
    for pdf_file in pdf_files[:3]:  # Process up to 3 files
        print(f"📖 Processing: {pdf_file.name}")
        
        try:
            result = pipeline.process_pdf(
                pdf_path=pdf_file,
                document_id=pdf_file.stem,
                show_progress=False  # Quiet processing for this example
            )
            
            if result.success:
                print(f"   ✅ {result.chunks_processed} chunks, {result.processing_time:.1f}s")
                processed_count += 1
            else:
                print(f"   ❌ Failed: {result.error_message}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n✅ Processed {processed_count} document(s)")
    return processed_count > 0


def demonstrate_basic_search(pipeline: PDFVectorPipeline) -> None:
    """Demonstrate basic search functionality."""
    print_subsection("Basic Search")
    
    # Simple search query
    query = "machine learning"
    print(f"🔍 Searching for: '{query}'")
    
    try:
        results = pipeline.search(query_text=query, n_results=5)
        
        if results:
            print(f"   Found {len(results)} result(s):")
            
            for i, result in enumerate(results, 1):
                print(f"\n   {i}. Score: {result.score:.3f}")
                print(f"      Document: {result.document_id}")
                if result.page_number:
                    print(f"      Page: {result.page_number}")
                
                # Show content preview
                content_lines = result.content.strip().split('\n')
                preview = content_lines[0][:100]
                if len(content_lines[0]) > 100:
                    preview += "..."
                print(f"      Content: {preview}")
        else:
            print("   No results found")
            
    except Exception as e:
        print(f"   ❌ Search error: {e}")


def demonstrate_similarity_scores(pipeline: PDFVectorPipeline) -> None:
    """Demonstrate how similarity scores work."""
    print_subsection("Understanding Similarity Scores")
    
    # Test queries with different expected relevance
    test_queries = [
        ("artificial intelligence", "High relevance expected"),
        ("machine learning algorithms", "High relevance expected"),
        ("cooking recipes", "Low relevance expected"),
        ("the quick brown fox", "Very low relevance expected")
    ]
    
    for query, expectation in test_queries:
        print(f"\n🔍 Query: '{query}' ({expectation})")
        
        try:
            results = pipeline.search(query_text=query, n_results=3)
            
            if results:
                best_score = results[0].score
                print(f"   Best score: {best_score:.3f}")
                
                # Interpret the score
                if best_score > 0.8:
                    interpretation = "Excellent match"
                elif best_score > 0.6:
                    interpretation = "Good match"
                elif best_score > 0.4:
                    interpretation = "Moderate match"
                elif best_score > 0.2:
                    interpretation = "Weak match"
                else:
                    interpretation = "Poor match"
                
                print(f"   Interpretation: {interpretation}")
                
                # Show score distribution
                scores = [r.score for r in results]
                print(f"   Score range: {min(scores):.3f} - {max(scores):.3f}")
            else:
                print("   No results found")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")


def demonstrate_search_filtering(pipeline: PDFVectorPipeline) -> None:
    """Demonstrate search filtering capabilities."""
    print_subsection("Search Filtering")
    
    # Get available documents
    try:
        stats = pipeline.get_collection_stats()
        total_chunks = stats.get('total_chunks', 0)
        unique_docs = stats.get('unique_documents', 0)
        
        print(f"📊 Collection: {total_chunks} chunks across {unique_docs} documents")
        
        if unique_docs == 0:
            print("   No documents to filter")
            return
            
    except Exception as e:
        print(f"   ❌ Error getting stats: {e}")
        return
    
    # Basic search without filtering
    query = "data analysis"
    print(f"\n🔍 Unfiltered search: '{query}'")
    
    try:
        all_results = pipeline.search(query_text=query, n_results=10)
        print(f"   Found {len(all_results)} total results")
        
        if all_results:
            # Show document distribution
            doc_counts = {}
            for result in all_results:
                doc_id = result.document_id
                doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
            
            print("   Results by document:")
            for doc_id, count in doc_counts.items():
                print(f"     - {doc_id}: {count} chunks")
            
            # Document-specific search
            if len(doc_counts) > 1:
                first_doc = list(doc_counts.keys())[0]
                print(f"\n🎯 Document-specific search in '{first_doc}':")
                
                doc_results = pipeline.search(
                    query_text=query,
                    n_results=5,
                    document_id=first_doc
                )
                
                print(f"   Found {len(doc_results)} results in {first_doc}")
                for i, result in enumerate(doc_results[:3], 1):
                    print(f"   {i}. Score: {result.score:.3f}, Page: {result.page_number or 'N/A'}")
            
            # Page-specific search (if page info available)
            page_results = [r for r in all_results if r.page_number is not None]
            if page_results:
                print(f"\n📄 Page-specific search (page 1):")
                
                page1_results = pipeline.search(
                    query_text=query,
                    n_results=3,
                    page_number=1
                )
                
                print(f"   Found {len(page1_results)} results on page 1")
                for i, result in enumerate(page1_results, 1):
                    print(f"   {i}. Score: {result.score:.3f}, Doc: {result.document_id}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")


def demonstrate_search_tips(pipeline: PDFVectorPipeline) -> None:
    """Demonstrate search tips and best practices."""
    print_subsection("Search Tips and Best Practices")
    
    search_examples = [
        {
            "query": "neural networks",
            "tip": "Specific technical terms often work well"
        },
        {
            "query": "how to implement machine learning",
            "tip": "Natural language questions work with semantic search"
        },
        {
            "query": "deep learning CNN architecture",
            "tip": "Multiple related terms can improve results"
        },
        {
            "query": "performance optimization techniques",
            "tip": "Conceptual queries find related content"
        }
    ]
    
    for example in search_examples:
        query = example["query"]
        tip = example["tip"]
        
        print(f"\n💡 Tip: {tip}")
        print(f"   Query: '{query}'")
        
        try:
            results = pipeline.search(query_text=query, n_results=3)
            
            if results:
                best_score = results[0].score
                print(f"   Best match: {best_score:.3f}")
                
                # Show variety in results
                unique_docs = set(r.document_id for r in results)
                print(f"   Documents found: {len(unique_docs)}")
            else:
                print("   No results found")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")


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
            print("✅ Pipeline initialized")
        except Exception as e:
            print(f"❌ Failed to initialize pipeline: {e}")
            return
        
        # Process documents
        if not process_sample_documents(pipeline):
            print("❌ No documents processed. Cannot demonstrate search.")
            return
        
        print_section("Search Demonstrations")
        
        # Demonstrate different search aspects
        demonstrate_basic_search(pipeline)
        demonstrate_similarity_scores(pipeline)
        demonstrate_search_filtering(pipeline)
        demonstrate_search_tips(pipeline)
        
        print_section("Search Summary")
        
        print("🎯 Key Concepts:")
        print()
        print("1. 🔍 Semantic Search:")
        print("   - Finds content by meaning, not just keywords")
        print("   - Works with natural language queries")
        print("   - Understands context and relationships")
        print()
        print("2. 📊 Similarity Scores:")
        print("   - Range from 0.0 (no similarity) to 1.0 (identical)")
        print("   - Scores > 0.8: Excellent matches")
        print("   - Scores > 0.6: Good matches")
        print("   - Scores > 0.4: Moderate matches")
        print()
        print("3. 🎯 Search Filtering:")
        print("   - Filter by document ID")
        print("   - Filter by page number")
        print("   - Combine filters for precise results")
        print()
        print("4. 💡 Best Practices:")
        print("   - Use specific technical terms for precise results")
        print("   - Try natural language questions")
        print("   - Combine multiple related terms")
        print("   - Experiment with different query styles")
        print()
        print("Next steps:")
        print("- Try health_check.py to learn about system monitoring")
        print("- Explore advanced search patterns in 05_vector_database/")
        print("- Add your own documents and test different queries")


if __name__ == "__main__":
    main()
