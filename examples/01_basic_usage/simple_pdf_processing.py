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

from pdf_vector_system import Config, PDFVectorPipeline
from pdf_vector_system.config.settings import EmbeddingModelType
from utils.example_helpers import (
    setup_example_environment, print_section, print_subsection,
    ensure_sample_data_exists, example_context
)
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
        
        print(f"✅ Configuration created")
        print(f"   - Embedding model: {config.embedding.model_name}")
        print(f"   - Collection: {config.chroma_db.collection_name}")
        print(f"   - Database: {config.chroma_db.persist_directory}")
        
        # Step 2: Initialize Pipeline
        print_subsection("Initializing Pipeline")
        
        try:
            pipeline = PDFVectorPipeline(config)
            print("✅ Pipeline initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize pipeline: {e}")
            return
        
        # Step 3: Ensure Sample Data
        print_subsection("Preparing Sample Data")
        
        sample_dir = Path("examples/sample_data")
        if not ensure_sample_data(sample_dir):
            print("⚠️  No sample data available. Creating minimal test content...")
            # Create a simple text file as fallback
            test_file = sample_dir / "simple_test.txt"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text("""
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
            """)
            print(f"✅ Created test file: {test_file}")
        
        # Find available PDF files
        pdf_files = list(sample_dir.glob("*.pdf"))
        if not pdf_files:
            print("📄 No PDF files found. This example works best with PDF files.")
            print("   Add PDF files to examples/sample_data/ for better demonstration.")
            return
        
        # Step 4: Process PDF
        print_section("Step 2: PDF Processing")
        
        # Use the first available PDF
        pdf_file = pdf_files[0]
        print(f"📖 Processing: {pdf_file.name}")
        
        try:
            # Process the PDF with progress tracking
            result = pipeline.process_pdf(
                pdf_path=pdf_file,
                document_id=pdf_file.stem,
                show_progress=True
            )
            
            if result.success:
                print(f"✅ Successfully processed {pdf_file.name}")
                print(f"   - Chunks created: {result.chunks_processed}")
                print(f"   - Embeddings generated: {result.embeddings_generated}")
                print(f"   - Processing time: {result.processing_time:.2f} seconds")
                print(f"   - Speed: {result.chunks_per_second:.1f} chunks/sec")
            else:
                print(f"❌ Failed to process {pdf_file.name}: {result.error_message}")
                return
                
        except Exception as e:
            print(f"❌ Error processing PDF: {e}")
            return
        
        # Step 5: Perform Searches
        print_section("Step 3: Search Demonstration")
        
        # Define some example search queries
        search_queries = [
            "machine learning",
            "artificial intelligence", 
            "data analysis",
            "neural networks"
        ]
        
        for query in search_queries:
            print_subsection(f"Searching for: '{query}'")
            
            try:
                # Perform search
                search_results = pipeline.search(
                    query_text=query,
                    n_results=3
                )
                
                if search_results:
                    print(f"   Found {len(search_results)} result(s):")
                    
                    for i, result in enumerate(search_results, 1):
                        print(f"   {i}. Score: {result.score:.3f}")
                        print(f"      Document: {result.document_id}")
                        if result.page_number:
                            print(f"      Page: {result.page_number}")
                        
                        # Show content preview (first 100 characters)
                        content_preview = result.content[:100]
                        if len(result.content) > 100:
                            content_preview += "..."
                        print(f"      Preview: {content_preview}")
                        print()
                else:
                    print("   No results found")
                    
            except Exception as e:
                print(f"   ❌ Search error: {e}")
        
        # Step 6: System Information
        print_section("Step 4: System Information")
        
        # Show collection statistics
        print_subsection("Collection Statistics")
        try:
            stats = pipeline.get_collection_stats()
            print(f"   - Total chunks: {stats.get('total_chunks', 0):,}")
            print(f"   - Unique documents: {stats.get('unique_documents', 0)}")
            print(f"   - Total characters: {stats.get('total_characters', 0):,}")
            print(f"   - Average chunk size: {stats.get('average_chunk_size', 0):.0f} chars")
        except Exception as e:
            print(f"   ❌ Error getting statistics: {e}")
        
        # Health check
        print_subsection("System Health Check")
        try:
            health = pipeline.health_check()
            for component, status in health.items():
                status_icon = "✅" if status else "❌"
                print(f"   {status_icon} {component}: {'Healthy' if status else 'Unhealthy'}")
        except Exception as e:
            print(f"   ❌ Error checking health: {e}")
        
        # Step 7: Cleanup Information
        print_section("Step 5: Next Steps")
        
        print("🎉 Simple PDF processing example completed!")
        print()
        print("What happened:")
        print("1. ✅ Configured the system with local embeddings")
        print("2. ✅ Initialized the PDF processing pipeline")
        print("3. ✅ Processed a PDF file into searchable chunks")
        print("4. ✅ Generated embeddings for semantic search")
        print("5. ✅ Performed similarity searches")
        print("6. ✅ Checked system health and statistics")
        print()
        print("Next steps:")
        print("- Try the configuration_basics.py example to learn about configuration options")
        print("- Explore advanced_configuration.py for more sophisticated setups")
        print("- Add your own PDF files to examples/sample_data/ for testing")
        print()
        print(f"💾 Data persisted in: {config.chroma_db.persist_directory}")
        print("   You can run searches again without reprocessing the PDF")


if __name__ == "__main__":
    main()
