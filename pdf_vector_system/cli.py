"""Command-line interface for PDF Vector System."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pdf_vector_system.config.settings import Config, EmbeddingModelType, LogLevel
from pdf_vector_system.pipeline import PDFVectorPipeline
from pdf_vector_system.utils.logging import setup_logging

app = typer.Typer(
    name="pdf-vector",
    help="PDF Vector System - Process PDFs and store in vector database",
    add_completion=False,
)
console = Console()


def get_pipeline(
    embedding_model: Optional[str] = None,
    collection_name: Optional[str] = None,
    debug: bool = False,
) -> PDFVectorPipeline:
    """Get configured pipeline instance."""

    # Load configuration
    config = Config()
    # TODO: Load from file if config_path and config_path.exists()

    # Override with CLI parameters
    if embedding_model:
        if embedding_model.startswith("text-embedding"):
            config.embedding.model_type = EmbeddingModelType.OPENAI
            config.embedding.model_name = embedding_model
        else:
            config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
            config.embedding.model_name = embedding_model

    if collection_name:
        config.chroma_db.collection_name = collection_name

    if debug:
        config.debug = True
        config.logging.level = LogLevel.DEBUG

    # Setup logging
    setup_logging(config.logging)

    return PDFVectorPipeline(config)


@app.command()
def process(
    files: list[Path] = typer.Argument(..., help="PDF files to process"),
    embedding_model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Embedding model (e.g., 'all-MiniLM-L6-v2' or 'text-embedding-3-small')",
    ),
    collection_name: Optional[str] = typer.Option(
        None, "--collection", "-c", help="ChromaDB collection name"
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", "-b", help="Batch size for processing"
    ),
    chunk_size: Optional[int] = typer.Option(
        None, "--chunk-size", help="Text chunk size"
    ),
    clean_text: bool = typer.Option(
        True, "--clean/--no-clean", help="Clean extracted text"
    ),
    show_progress: bool = typer.Option(
        True, "--progress/--no-progress", help="Show progress bars"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Process PDF files and store in vector database."""

    # Validate files
    valid_files = []
    for file_path in files:
        if not file_path.exists():
            console.print(f"❌ File not found: {file_path}", style="red")
            continue
        if file_path.suffix.lower() != ".pdf":
            console.print(f"⚠️  Skipping non-PDF file: {file_path}", style="yellow")
            continue
        valid_files.append(file_path)

    if not valid_files:
        console.print("❌ No valid PDF files to process", style="red")
        raise typer.Exit(1) from None

    try:
        # Get pipeline
        pipeline = get_pipeline(
            embedding_model=embedding_model,
            collection_name=collection_name,
            debug=debug,
        )

        # Override configuration if specified
        if batch_size:
            pipeline.config.embedding.batch_size = batch_size
        if chunk_size:
            pipeline.config.text_processing.chunk_size = chunk_size

        console.print(f"🚀 Processing {len(valid_files)} PDF file(s)...")
        console.print(f"📊 Model: {pipeline.embedding_service.model_name}")
        console.print(f"📦 Collection: {pipeline.config.chroma_db.collection_name}")

        # Process files
        results = []
        for i, pdf_file in enumerate(valid_files, 1):
            console.print(f"\n📖 [{i}/{len(valid_files)}] Processing: {pdf_file.name}")

            try:
                result = pipeline.process_pdf(
                    pdf_path=pdf_file,
                    clean_text=clean_text,
                    show_progress=show_progress,
                )

                results.append(result)

                if result.success:
                    console.print(
                        f"✅ Success: {result.chunks_processed} chunks in {result.processing_time:.2f}s",
                        style="green",
                    )
                else:
                    console.print(f"❌ Failed: {result.error_message}", style="red")

            except Exception as e:
                console.print(f"❌ Error: {e!s}", style="red")

        # Summary
        successful = [r for r in results if r.success]
        total_chunks = sum(r.chunks_processed for r in successful)
        total_time = sum(r.processing_time for r in successful)

        console.print("\n📊 Summary:")
        console.print(f"   ✅ Successful: {len(successful)}/{len(results)} files")
        console.print(f"   📄 Total chunks: {total_chunks:,}")
        console.print(f"   ⏱️  Total time: {total_time:.2f}s")
        if total_time > 0:
            console.print(f"   🚀 Speed: {total_chunks / total_time:.1f} chunks/sec")

    except Exception as e:
        console.print(f"❌ Pipeline error: {e!s}", style="red")
        if debug:
            console.print_exception()
        raise typer.Exit(1) from None


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    n_results: int = typer.Option(10, "--results", "-n", help="Number of results"),
    document_id: Optional[str] = typer.Option(
        None, "--document", "-d", help="Search within specific document"
    ),
    page_number: Optional[int] = typer.Option(
        None, "--page", "-p", help="Search within specific page"
    ),
    collection_name: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection name"
    ),
    score_threshold: float = typer.Option(
        0.0, "--threshold", "-t", help="Minimum similarity score"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Search the vector database."""

    try:
        pipeline = get_pipeline(collection_name=collection_name, debug=debug)

        console.print(f"🔍 Searching for: '{query}'")
        console.print(f"📦 Collection: {pipeline.config.chroma_db.collection_name}")

        # Perform search
        results = pipeline.search(
            query_text=query,
            n_results=n_results,
            document_id=document_id,
            page_number=page_number,
        )

        # Filter by score threshold
        if score_threshold > 0:
            results = [r for r in results if r.score >= score_threshold]

        if not results:
            console.print("❌ No results found", style="yellow")
            return

        # Display results
        console.print(f"\n📋 Found {len(results)} result(s):")

        for i, result in enumerate(results, 1):
            # Create result panel
            content_preview = (
                result.content[:200] + "..."
                if len(result.content) > 200
                else result.content
            )

            metadata_info = []
            if hasattr(result, "document_id") and result.document_id:
                metadata_info.append(f"Document: {result.document_id}")
            if hasattr(result, "page_number") and result.page_number:
                metadata_info.append(f"Page: {result.page_number}")

            metadata_str = " | ".join(metadata_info) if metadata_info else "No metadata"

            panel_content = f"[bold]Score: {result.score:.3f}[/bold]\n"
            panel_content += f"[dim]{metadata_str}[/dim]\n\n"
            panel_content += content_preview

            console.print(
                Panel(panel_content, title=f"Result {i}", border_style="blue")
            )

    except Exception as e:
        console.print(f"❌ Search error: {e!s}", style="red")
        if debug:
            console.print_exception()
        raise typer.Exit(1) from None


@app.command()
def stats(
    collection_name: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection name"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Show collection statistics."""

    try:
        pipeline = get_pipeline(collection_name=collection_name, debug=debug)

        console.print("📊 Collection Statistics")
        console.print(f"📦 Collection: {pipeline.config.chroma_db.collection_name}")

        stats = pipeline.get_collection_stats()

        # Create statistics table
        table = Table(title="Statistics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Chunks", f"{stats['total_chunks']:,}")
        table.add_row("Unique Documents", f"{stats['unique_documents']:,}")
        table.add_row("Total Characters", f"{stats['total_characters']:,}")
        table.add_row("Average Chunk Size", f"{stats['average_chunk_size']:.0f} chars")

        if stats.get("sampled", False):
            table.add_row("Sample Size", f"{stats['sample_size']:,}")
            table.add_row("Sampled", "Yes")

        console.print(table)

    except Exception as e:
        console.print(f"❌ Stats error: {e!s}", style="red")
        if debug:
            console.print_exception()
        raise typer.Exit(1) from None


@app.command()
def list_docs(
    collection_name: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection name"
    ),
    limit: int = typer.Option(
        20, "--limit", "-l", help="Maximum number of documents to show"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """List documents in the collection."""

    try:
        pipeline = get_pipeline(collection_name=collection_name, debug=debug)

        console.print("📋 Documents in Collection")
        console.print(f"📦 Collection: {pipeline.config.chroma_db.collection_name}")

        # Get collection and sample documents
        collection = pipeline.vector_db.get_collection()
        results = collection.get(limit=limit, include=["metadatas"])

        if not results["ids"]:
            console.print("❌ No documents found", style="yellow")
            return

        # Extract unique document IDs
        document_ids = set()
        if results["metadatas"]:
            for metadata in results["metadatas"]:
                if metadata and "document_id" in metadata:
                    doc_id = metadata["document_id"]
                    if isinstance(doc_id, str):
                        document_ids.add(doc_id)

        # Create documents table
        table = Table(
            title=f"Documents (showing up to {limit})",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Document ID", style="cyan")
        table.add_column("Chunks", style="green")
        table.add_column("Characters", style="yellow")
        table.add_column("Avg Chunk Size", style="blue")

        for doc_id in sorted(document_ids):
            try:
                doc_info = pipeline.get_document_info(doc_id)
                table.add_row(
                    doc_id,
                    f"{doc_info['chunk_count']:,}",
                    f"{doc_info['total_characters']:,}",
                    f"{doc_info['average_chunk_size']:.0f}",
                )
            except Exception as e:
                table.add_row(doc_id, "Error", str(e), "")

        console.print(table)

    except Exception as e:
        console.print(f"❌ List error: {e!s}", style="red")
        if debug:
            console.print_exception()
        raise typer.Exit(1) from None


@app.command()
def delete(
    document_id: str = typer.Argument(..., help="Document ID to delete"),
    collection_name: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection name"
    ),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Delete a document from the collection."""

    try:
        pipeline = get_pipeline(collection_name=collection_name, debug=debug)

        # Confirm deletion
        if not confirm:
            confirmed = typer.confirm(
                f"Delete document '{document_id}' from collection '{pipeline.config.chroma_db.collection_name}'?"
            )
            if not confirmed:
                console.print("❌ Deletion cancelled", style="yellow")
                return

        # Delete document
        chunks_deleted = pipeline.delete_document(document_id)

        console.print(
            f"✅ Deleted document '{document_id}' ({chunks_deleted} chunks)",
            style="green",
        )

    except Exception as e:
        console.print(f"❌ Delete error: {e!s}", style="red")
        if debug:
            console.print_exception()
        raise typer.Exit(1) from None


@app.command()
def health(
    collection_name: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection name"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Check system health."""

    try:
        pipeline = get_pipeline(collection_name=collection_name, debug=debug)

        console.print("🏥 System Health Check")

        health_status = pipeline.health_check()

        # Create health table
        table = Table(
            title="Health Status", show_header=True, header_style="bold magenta"
        )
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")

        for component, status in health_status.items():
            status_text = "✅ Healthy" if status else "❌ Unhealthy"
            table.add_row(component, status_text)

        console.print(table)

        # Overall status
        all_healthy = all(health_status.values())
        if all_healthy:
            console.print("🎉 All systems healthy!", style="green")
        else:
            console.print("⚠️  Some systems unhealthy", style="red")
            raise typer.Exit(1) from None

    except Exception as e:
        console.print(f"❌ Health check error: {e!s}", style="red")
        if debug:
            console.print_exception()
        raise typer.Exit(1) from None


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
