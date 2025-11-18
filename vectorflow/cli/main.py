"""Command-line interface for PDF Vector System."""

import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.status import Status
from rich.table import Table
from rich.text import Text

from vectorflow.core.config.settings import Config, EmbeddingModelType, LogLevel
from vectorflow.core.pipeline import PDFVectorPipeline
from vectorflow.core.utils.logging import setup_logging

# Create Typer app
app = typer.Typer(
    name="pdf-vector",
    help="PDF Vector System - Process PDFs and store in vector database",
    add_completion=False,
    rich_markup_mode="rich",
)

# Rich console for pretty output
console = Console()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def _compute_collection_stats(pipeline: PDFVectorPipeline) -> dict[str, Any]:
    """Compute collection stats in a backend-agnostic way using core APIs."""
    # Try backend-specific fast path if available (e.g., Chroma)
    get_stats = getattr(pipeline.vector_db, "get_collection_stats", None)
    if callable(get_stats):
        try:
            return get_stats()  # type: ignore[misc]
        except Exception:
            pass

    # Generic fallback using core interface
    try:
        total_chunks = pipeline.vector_db.count_chunks()
    except Exception:
        total_chunks = 0

    try:
        documents = pipeline.get_documents()
    except Exception:
        documents = []

    unique_documents = len(documents)
    total_characters = sum(d.get("total_characters", 0) for d in documents)
    average_chunk_size = (total_characters / total_chunks) if total_chunks > 0 else 0.0

    return {
        "total_chunks": total_chunks,
        "unique_documents": unique_documents,
        "total_characters": total_characters,
        "average_chunk_size": average_chunk_size,
        "sampled": False,
        "sample_size": unique_documents,
    }


def get_pipeline(
    embedding_model: Optional[str] = None,
    collection_name: Optional[str] = None,
    debug: bool = False,
    verbose: bool = False,
) -> PDFVectorPipeline:
    """Get configured pipeline instance.

    Args:
        embedding_model: Override embedding model
        collection_name: Override collection name
        debug: Enable debug mode
        verbose: Enable verbose output

    Returns:
        Configured PDFVectorPipeline instance
    """
    # Load configuration
    config = Config()

    # Override with CLI parameters
    if embedding_model:
        if embedding_model.startswith("text-embedding"):
            config.embedding.model_type = EmbeddingModelType.OPENAI
            config.embedding.model_name = embedding_model
        elif "gpt" in embedding_model.lower():
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
    elif verbose:
        config.logging.level = LogLevel.INFO

    # Setup logging
    setup_logging(config.logging)

    return PDFVectorPipeline(config)


@app.command(
    help="Process PDF files and store in vector database.\n\n"
    "Examples:\n\n"
    "  $ pdf-vector process document.pdf\n"
    "  $ pdf-vector process *.pdf --model all-MiniLM-L6-v2\n"
    "  $ pdf-vector process doc.pdf --collection my_docs --chunk-size 1500"
)
def process(
    files: list[Path] = typer.Argument(..., help="PDF files to process"),
    embedding_model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Embedding model (e.g., 'all-MiniLM-L6-v2' or 'text-embedding-3-small')",
    ),
    collection_name: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Vector database collection name"
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", "-b", help="Batch size for embedding generation"
    ),
    chunk_size: Optional[int] = typer.Option(
        None, "--chunk-size", help="Text chunk size in characters"
    ),
    clean_text: bool = typer.Option(
        True, "--clean/--no-clean", help="Clean and normalize extracted text"
    ),
    show_progress: bool = typer.Option(
        True, "--progress/--no-progress", help="Show progress bars"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Process PDF files and store embeddings in vector database."""

    # Validate and filter files
    valid_files = []
    errors = []

    for file_path in files:
        if not file_path.exists():
            errors.append(f"[red]✗[/red] File not found: {file_path}")
            continue
        if file_path.suffix.lower() != ".pdf":
            errors.append(f"[yellow]⚠[/yellow] Skipping non-PDF file: {file_path}")
            continue

        # Check file size (warn if too large)
        file_size = file_path.stat().st_size
        if file_size > 100 * 1024 * 1024:  # 100MB
            console.print(
                f"[yellow]⚠ Warning:[/yellow] Large file ({format_file_size(file_size)}): {file_path.name}"
            )

        valid_files.append(file_path)

    # Show errors
    for error in errors:
        console.print(error)

    if not valid_files:
        console.print("\n[red]✗ No valid PDF files to process[/red]")
        raise typer.Exit(1)

    try:
        # Initialize pipeline with status indicator
        with Status("[bold green]Initializing pipeline...", console=console):
            pipeline = get_pipeline(
                embedding_model=embedding_model,
                collection_name=collection_name,
                debug=debug,
                verbose=verbose,
            )

            # Override configuration
            if batch_size:
                pipeline.config.embedding.batch_size = batch_size
            if chunk_size:
                pipeline.config.text_processing.chunk_size = chunk_size

        # Show configuration
        console.print("\n[bold cyan]Configuration:[/bold cyan]")
        config_table = Table(show_header=False, box=None, padding=(0, 2))
        config_table.add_column(style="cyan")
        config_table.add_column(style="white")

        config_table.add_row("📊 Model:", pipeline.embedding_service.model_name)
        config_table.add_row(
            "📦 Collection:", pipeline.config.chroma_db.collection_name
        )
        config_table.add_row("📄 Files:", str(len(valid_files)))
        config_table.add_row(
            "🔢 Chunk Size:", f"{pipeline.config.text_processing.chunk_size} chars"
        )
        config_table.add_row(
            "📦 Batch Size:", str(pipeline.config.embedding.batch_size)
        )

        console.print(config_table)
        console.print()

        # Process files with progress bar
        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:

            task = progress.add_task("[cyan]Processing PDFs...", total=len(valid_files))

            for pdf_file in valid_files:
                progress.update(task, description=f"[cyan]Processing: {pdf_file.name}")

                try:
                    result = pipeline.process_pdf(
                        pdf_path=pdf_file,
                        clean_text=clean_text,
                        show_progress=False,  # Disable internal progress bar
                    )

                    results.append(result)

                    if result.success:
                        progress.console.print(
                            f"  [green]✓[/green] {pdf_file.name}: "
                            f"{result.chunks_processed} chunks in {result.processing_time:.2f}s"
                        )
                    else:
                        progress.console.print(
                            f"  [red]✗[/red] {pdf_file.name}: {result.error_message}"
                        )

                except Exception as e:
                    progress.console.print(f"  [red]✗[/red] {pdf_file.name}: {str(e)}")
                    results.append(None)

                progress.advance(task)

        # Summary table
        successful = [r for r in results if r and r.success]
        failed = len(results) - len(successful)

        if successful:
            total_chunks = sum(r.chunks_processed for r in successful)
            total_chars = sum(
                r.metadata.get("total_characters", 0) for r in successful if r.metadata
            )
            total_time = sum(r.processing_time for r in successful)
            avg_speed = total_chunks / total_time if total_time > 0 else 0

            console.print("\n[bold cyan]📊 Processing Summary[/bold cyan]")

            summary_table = Table(
                show_header=True, header_style="bold magenta", box=None
            )
            summary_table.add_column("Metric", style="cyan", no_wrap=True)
            summary_table.add_column("Value", justify="right", style="green")

            summary_table.add_row(
                "Files Processed", f"[green]{len(successful)}[/green] / {len(results)}"
            )
            if failed > 0:
                summary_table.add_row("Failed", f"[red]{failed}[/red]")
            summary_table.add_row("Total Chunks", f"{total_chunks:,}")
            summary_table.add_row("Total Characters", f"{total_chars:,}")
            summary_table.add_row("Processing Time", f"{total_time:.2f}s")
            summary_table.add_row("Speed", f"{avg_speed:.1f} chunks/sec")

            console.print(summary_table)
            console.print(f"\n[bold green]✓ Processing complete![/bold green]")
        else:
            console.print("\n[red]✗ All files failed to process[/red]")
            raise typer.Exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Processing cancelled by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n[red]✗ Pipeline error: {e}[/red]")
        if debug:
            console.print_exception()
        console.print(
            "\n[yellow]💡 Tip:[/yellow] Run with --debug for detailed error information"
        )
        raise typer.Exit(1)


@app.command(
    help="Search the vector database for similar content.\n\n"
    "Examples:\n\n"
    "  $ pdf-vector search 'machine learning algorithms'\n"
    "  $ pdf-vector search 'neural networks' --results 5\n"
    "  $ pdf-vector search 'deep learning' --document my_doc --threshold 0.7"
)
def search(
    query: str = typer.Argument(..., help="Search query text"),
    n_results: int = typer.Option(
        10, "--results", "-n", help="Number of results to return"
    ),
    document_id: Optional[str] = typer.Option(
        None, "--document", "-d", help="Filter by specific document ID"
    ),
    page_number: Optional[int] = typer.Option(
        None, "--page", "-p", help="Filter by specific page number"
    ),
    collection_name: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection name to search"
    ),
    score_threshold: float = typer.Option(
        0.0, "--threshold", "-t", help="Minimum similarity score (0.0-1.0)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full content"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Search the vector database for semantically similar content."""

    try:
        # Validate inputs
        if n_results < 1:
            console.print("[red]✗ Number of results must be positive[/red]")
            raise typer.Exit(1)

        if not 0.0 <= score_threshold <= 1.0:
            console.print("[red]✗ Score threshold must be between 0.0 and 1.0[/red]")
            raise typer.Exit(1)

        # Initialize pipeline
        with Status("[bold green]Initializing search...", console=console):
            pipeline = get_pipeline(collection_name=collection_name, debug=debug)

        console.print(f"\n[bold cyan]🔍 Search Query:[/bold cyan] {query}")
        console.print(
            f"[dim]Collection: {pipeline.config.chroma_db.collection_name}[/dim]"
        )

        # Perform search
        with Status("[bold green]Searching...", console=console):
            results = pipeline.search(
                query_text=query,
                n_results=n_results,
                document_id=document_id,
                page_number=page_number,
            )

        # Filter by score threshold
        if score_threshold > 0:
            filtered_results = [r for r in results if r.score >= score_threshold]
            if len(filtered_results) < len(results):
                console.print(
                    f"\n[dim]Filtered {len(results) - len(filtered_results)} results below threshold {score_threshold:.2f}[/dim]"
                )
            results = filtered_results

        if not results:
            console.print("\n[yellow]❌ No results found[/yellow]")
            console.print("\n[dim]💡 Tips:[/dim]")
            console.print("  • Try a different query")
            console.print("  • Lower the score threshold")
            console.print(
                "  • Check if documents exist: [cyan]pdf-vector list-docs[/cyan]"
            )
            return

        # Display results
        console.print(f"\n[bold green]✓ Found {len(results)} result(s)[/bold green]\n")

        for i, result in enumerate(results, 1):
            # Determine score color based on value
            if result.score >= 0.9:
                score_color = "green"
            elif result.score >= 0.7:
                score_color = "yellow"
            else:
                score_color = "red"

            # Format content
            if verbose:
                content = result.content
            else:
                max_length = 300
                content = (
                    result.content[:max_length] + "..."
                    if len(result.content) > max_length
                    else result.content
                )

            # Build metadata string
            metadata_parts = []
            if hasattr(result, "document_id") and result.document_id:
                metadata_parts.append(f"📄 {result.document_id}")
            if hasattr(result, "page_number") and result.page_number is not None:
                metadata_parts.append(f"📖 Page {result.page_number}")
            if hasattr(result, "chunk_index") and result.chunk_index is not None:
                metadata_parts.append(f"🔢 Chunk {result.chunk_index}")

            metadata_str = " | ".join(metadata_parts) if metadata_parts else ""

            # Build panel content
            panel_content = Text()
            panel_content.append(f"Score: ", style="bold")
            panel_content.append(f"{result.score:.4f}", style=f"bold {score_color}")
            panel_content.append("\n")

            if metadata_str:
                panel_content.append(f"{metadata_str}\n", style="dim")

            panel_content.append(f"\n{content}", style="white")

            # Display result panel
            console.print(
                Panel(
                    panel_content,
                    title=f"[bold]Result {i}/{len(results)}[/bold]",
                    border_style="blue" if result.score >= 0.7 else "dim",
                    padding=(1, 2),
                )
            )

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Search cancelled[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n[red]✗ Search error: {e}[/red]")
        if debug:
            console.print_exception()
        else:
            console.print(
                "\n[dim]💡 Run with --debug for detailed error information[/dim]"
            )
        raise typer.Exit(1)


@app.command(
    name="stats",
    help="Display collection statistics and metrics.\n\n"
    "Examples:\n\n"
    "  $ pdf-vector stats\n"
    "  $ pdf-vector stats --collection my_docs",
)
def stats(
    collection_name: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection name"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Show detailed statistics about the vector database collection."""

    try:
        pipeline = get_pipeline(collection_name=collection_name, debug=debug)

        console.print("\n[bold cyan]📊 Collection Statistics[/bold cyan]")
        console.print(
            f"[dim]Collection: {pipeline.config.chroma_db.collection_name}[/dim]\n"
        )

        with Status("[bold green]Calculating statistics...", console=console):
            stats = _compute_collection_stats(pipeline)

        # Create statistics table
        table = Table(
            show_header=True,
            header_style="bold magenta",
            border_style="blue",
            title_style="bold cyan",
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="green")

        # Add rows
        table.add_row("Total Chunks", f"{stats.get('total_chunks', 0):,}")
        table.add_row("Unique Documents", f"{stats.get('unique_documents', 0):,}")
        table.add_row("Total Characters", f"{stats.get('total_characters', 0):,}")
        table.add_row(
            "Average Chunk Size",
            f"{stats.get('average_chunk_size', 0.0):.0f} chars",
        )

        if stats.get("sampled", False):
            table.add_row("", "")  # Separator
            table.add_row(
                "[dim]Sample Size[/dim]", f"[dim]{stats['sample_size']:,}[/dim]"
            )
            table.add_row(
                "[dim]Sampled[/dim]", "[dim]Yes (statistics approximate)[/dim]"
            )

        console.print(table)
        console.print()

    except Exception as e:
        console.print(f"\n[red]✗ Stats error: {e}[/red]")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command(
    name="list-docs",
    help="List all documents in the collection.\n\n"
    "Examples:\n\n"
    "  $ pdf-vector list-docs\n"
    "  $ pdf-vector list-docs --limit 50\n"
    "  $ pdf-vector list-docs --collection my_docs",
)
def list_docs(
    collection_name: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection name"
    ),
    limit: int = typer.Option(
        50, "--limit", "-l", help="Maximum number of documents to show"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """List all documents stored in the vector database collection."""

    try:
        pipeline = get_pipeline(collection_name=collection_name, debug=debug)

        console.print("\n[bold cyan]📋 Documents in Collection[/bold cyan]")
        console.print(
            f"[dim]Collection: {pipeline.config.chroma_db.collection_name}[/dim]\n"
        )

        # Get documents via backend-agnostic pipeline API
        with Status("[bold green]Fetching documents...", console=console):
            docs = pipeline.get_documents()

        if not docs:
            console.print("[yellow]❌ No documents found in collection[/yellow]")
            console.print("\n[dim]💡 Process some PDFs first:[/dim]")
            console.print("   [cyan]pdf-vector process document.pdf[/cyan]")
            return

        # Apply limit and sort by document_id
        docs = sorted(docs, key=lambda d: d.get("document_id", ""))[:limit]

        # Create documents table
        table = Table(
            show_header=True,
            header_style="bold magenta",
            border_style="blue",
            title=f"Documents (showing {len(docs)})",
            title_style="bold cyan",
        )
        table.add_column("Document ID", style="cyan", no_wrap=False)
        table.add_column("Chunks", justify="right", style="green")
        table.add_column("Characters", justify="right", style="yellow")
        table.add_column("Avg Size", justify="right", style="blue")

        # Render document rows
        for d in docs:
            doc_id = d.get("document_id", "—")
            chunks_count = int(d.get("chunks_count", 0))
            total_chars = int(d.get("total_characters", 0))
            avg_size = (total_chars / chunks_count) if chunks_count > 0 else 0
            table.add_row(
                doc_id,
                f"{chunks_count:,}",
                f"{total_chars:,}",
                f"{avg_size:.0f}",
            )

        console.print(table)
        console.print()

        if len(docs) >= limit:
            console.print(
                f"[dim]Showing first {limit} documents. Use --limit to see more.[/dim]"
            )

    except Exception as e:
        console.print(f"\n[red]✗ List error: {e}[/red]")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command(
    name="doc-info", help="Show detailed information about a document by its ID."
)
def doc_info(
    document_id: str = typer.Argument(..., help="Document ID to inspect"),
    collection_name: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection name"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Display document statistics from the vector database."""
    try:
        pipeline = get_pipeline(collection_name=collection_name, debug=debug)
        with Status("[bold green]Fetching document info...", console=console):
            info = pipeline.get_document_info(document_id)

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="cyan")
        table.add_column(style="white")
        table.add_row("Document ID", document_id)
        table.add_row("Chunks", f"{info.get('chunk_count', 0):,}")
        table.add_row("Total Characters", f"{info.get('total_characters', 0):,}")
        table.add_row(
            "Average Chunk Size", f"{info.get('average_chunk_size', 0.0):.0f}"
        )
        if info.get("page_count") is not None:
            table.add_row("Page Count", str(info["page_count"]))
        if info.get("filename"):
            table.add_row("Filename", str(info["filename"]))
        console.print(table)
    except Exception as e:
        console.print(f"\n[red]✗ Doc info error: {e}[/red]")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command(name="similar", help="Find chunks similar to a given chunk ID.")
def similar_chunks(
    chunk_id: str = typer.Argument(..., help="Reference chunk ID"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results"),
    collection_name: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection name"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full content"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    try:
        pipeline = get_pipeline(collection_name=collection_name, debug=debug)
        with Status("[bold green]Searching similar chunks...", console=console):
            results = pipeline.vector_db.find_similar_chunks(
                chunk_id=chunk_id, collection_name=collection_name, limit=limit
            )

        if not results:
            console.print("[yellow]❌ No similar chunks found[/yellow]")
            raise typer.Exit(0)

        console.print(f"\n[bold green]✓ Found {len(results)} result(s)[/bold green]\n")
        for i, r in enumerate(results, 1):
            content = (
                r.content
                if verbose
                else (r.content[:300] + "..." if len(r.content) > 300 else r.content)
            )
            meta_parts = []
            if r.metadata.get("document_id"):
                meta_parts.append(f"📄 {r.metadata['document_id']}")
            if r.metadata.get("page_number") is not None:
                meta_parts.append(f"📖 Page {r.metadata['page_number']}")
            meta = " | ".join(meta_parts)
            panel = Panel(
                Text.from_markup(
                    f"Score: [bold]{r.score:.4f}[/bold]\n"
                    + (f"{meta}\n" if meta else "")
                    + f"\n{content}"
                ),
                title=f"Result {i}/{len(results)}",
                border_style="blue" if r.score >= 0.7 else "dim",
                padding=(1, 2),
            )
            console.print(panel)
    except Exception as e:
        console.print(f"\n[red]✗ Similar error: {e}[/red]")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command(
    name="search-meta",
    help="Search by metadata filters (key=value pairs).",
)
def search_by_metadata(
    filters: list[str] = typer.Argument(..., help="Metadata filters as key=value"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
    collection_name: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection name"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    try:
        filt: dict[str, Any] = {}
        for item in filters:
            if "=" not in item:
                raise ValueError(f"Invalid filter '{item}', expected key=value")
            k, v = item.split("=", 1)
            k = k.strip()
            v = v.strip()
            # attempt int conversion
            try:
                v_cast: Any = int(v)
            except Exception:
                v_cast = v
            filt[k] = v_cast

        pipeline = get_pipeline(collection_name=collection_name, debug=debug)
        with Status("[bold green]Searching by metadata...", console=console):
            results = pipeline.vector_db.search_by_metadata(
                metadata_filter=filt, collection_name=collection_name, limit=limit
            )

        if not results:
            console.print("[yellow]❌ No results found[/yellow]")
            return

        table = Table(
            show_header=True, header_style="bold magenta", border_style="blue"
        )
        table.add_column("ID", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Content")
        table.add_column("Document", style="dim")
        for r in results:
            doc_id = r.metadata.get("document_id", "")
            content = r.content if len(r.content) <= 120 else r.content[:120] + "..."
            table.add_row(str(r.id), f"{r.score:.4f}", content, str(doc_id))
        console.print(table)
    except Exception as e:
        console.print(f"\n[red]✗ Search-meta error: {e}[/red]")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command(name="backend", help="Display current vector database backend info.")
def backend_info(
    collection_name: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection name"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    try:
        pipeline = get_pipeline(collection_name=collection_name, debug=debug)
        info = pipeline.get_vector_db_info()
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="cyan")
        table.add_column(style="white")
        for k in [
            "backend",
            "type",
            "index_name",
            "collection_name",
            "dimension",
            "metric",
        ]:
            if k in info:
                table.add_row(k.replace("_", " ").title(), str(info[k]))
        if info.get("capabilities"):
            table.add_row("Capabilities", ", ".join(info["capabilities"]))
        if info.get("error"):
            table.add_row("Error", str(info["error"]))
        console.print(table)
    except Exception as e:
        console.print(f"\n[red]✗ Backend info error: {e}[/red]")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command(
    name="clear-collection", help="Remove all chunks from the current collection."
)
def clear_collection(
    collection_name: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection name"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    try:
        pipeline = get_pipeline(collection_name=collection_name, debug=debug)
        name = pipeline.config.chroma_db.collection_name
        if not yes and not typer.confirm(
            f"This will remove all chunks from '{name}'. Continue?"
        ):
            console.print("[dim]Cancelled[/dim]")
            return
        with Status("[bold red]Clearing collection...", console=console):
            pipeline.vector_db.clear_collection(collection_name)
        console.print(f"[bold green]✓ Cleared collection '{name}'[/bold green]")
    except Exception as e:
        console.print(f"\n[red]✗ Clear-collection error: {e}[/red]")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command(
    name="create-collection", help="Create the collection if it doesn't exist."
)
def create_collection_cmd(
    collection_name: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection name to create"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    try:
        pipeline = get_pipeline(collection_name=collection_name, debug=debug)
        created = pipeline.vector_db.create_collection(
            name=collection_name, get_or_create=True
        )
        if created:
            console.print("[bold green]✓ Collection is ready[/bold green]")
        else:
            console.print("[yellow]Collection not created[/yellow]")
    except Exception as e:
        console.print(f"\n[red]✗ Create-collection error: {e}[/red]")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command(name="delete-collection", help="Delete a collection.")
def delete_collection_cmd(
    collection_name: str = typer.Argument(..., help="Collection name to delete"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    try:
        if not yes and not typer.confirm(
            f"This will delete collection '{collection_name}'. Continue?"
        ):
            console.print("[dim]Cancelled[/dim]")
            return
        pipeline = get_pipeline(collection_name=collection_name, debug=debug)
        with Status("[bold red]Deleting collection...", console=console):
            pipeline.vector_db.delete_collection(collection_name)
        console.print(
            f"[bold green]✓ Deleted collection '{collection_name}'[/bold green]"
        )
    except Exception as e:
        console.print(f"\n[red]✗ Delete-collection error: {e}[/red]")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command(name="count", help="Count total number of chunks in the collection.")
def count_chunks_cmd(
    collection_name: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection name"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    try:
        pipeline = get_pipeline(collection_name=collection_name, debug=debug)
        with Status("[bold green]Counting chunks...", console=console):
            count = pipeline.vector_db.count_chunks(collection_name)
        console.print(f"[bold cyan]Total Chunks:[/bold cyan] [green]{count:,}[/green]")
    except Exception as e:
        console.print(f"\n[red]✗ Count error: {e}[/red]")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command(
    help="Delete a document from the collection.\n\n"
    "Examples:\n\n"
    "  $ pdf-vector delete my_document.pdf\n"
    "  $ pdf-vector delete doc_id --yes  # Skip confirmation"
)
def delete(
    document_id: str = typer.Argument(..., help="Document ID to delete"),
    collection_name: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection name"
    ),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Delete a document and all its chunks from the collection."""

    try:
        pipeline = get_pipeline(collection_name=collection_name, debug=debug)
        collection = pipeline.config.chroma_db.collection_name

        # Check if document exists
        try:
            doc_info = pipeline.get_document_info(document_id)
            chunk_count = doc_info["chunk_count"]
        except Exception:
            console.print(
                f"\n[yellow]⚠ Document '{document_id}' not found in collection '{collection}'[/yellow]"
            )
            console.print(
                "\n[dim]💡 List documents with:[/dim] [cyan]pdf-vector list-docs[/cyan]"
            )
            raise typer.Exit(1)

        # Confirm deletion
        if not confirm:
            console.print(
                f"\n[yellow]⚠ Warning:[/yellow] This will delete [bold]{document_id}[/bold]"
            )
            console.print(f"  Collection: {collection}")
            console.print(f"  Chunks to delete: {chunk_count:,}")
            console.print()

            confirmed = typer.confirm("Are you sure you want to delete this document?")
            if not confirmed:
                console.print("\n[dim]Deletion cancelled[/dim]")
                return

        # Delete document
        with Status(f"[bold red]Deleting document...", console=console):
            chunks_deleted = pipeline.delete_document(document_id)

        console.print(
            f"\n[bold green]✓ Successfully deleted '{document_id}'[/bold green]"
        )
        console.print(f"[dim]Removed {chunks_deleted:,} chunks from collection[/dim]")

    except Exception as e:
        console.print(f"\n[red]✗ Delete error: {e}[/red]")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command(
    help="Check the health of all system components.\n\n"
    "Examples:\n\n"
    "  $ pdf-vector health\n"
    "  $ pdf-vector health --collection my_docs"
)
def health(
    collection_name: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Collection name"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Perform health checks on all system components."""

    try:
        console.print("\n[bold cyan]🏥 System Health Check[/bold cyan]\n")

        with Status("[bold green]Running health checks...", console=console):
            pipeline = get_pipeline(collection_name=collection_name, debug=debug)
            health_status = pipeline.health_check()

        # Create health table
        table = Table(
            show_header=True,
            header_style="bold magenta",
            border_style="blue",
            title="Component Health",
            title_style="bold cyan",
        )
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", justify="center")
        table.add_column("Details", style="dim")

        all_healthy = True
        for component, status in health_status.items():
            if status:
                status_icon = "[green]✓ Healthy[/green]"
                details = "[dim]Operational[/dim]"
            else:
                status_icon = "[red]✗ Unhealthy[/red]"
                details = "[red]Check configuration[/red]"
                all_healthy = False

            # Format component name
            component_name = component.replace("_", " ").title()
            table.add_row(component_name, status_icon, details)

        console.print(table)
        console.print()

        # Overall status
        if all_healthy:
            console.print("[bold green]🎉 All systems operational![/bold green]")
        else:
            console.print("[bold red]⚠ Some systems need attention[/bold red]")
            console.print("\n[dim]💡 Run with --debug for detailed diagnostics[/dim]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"\n[red]✗ Health check failed: {e}[/red]")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command(
    name="config",
    help="Display current configuration settings.\n\n"
    "Examples:\n\n"
    "  $ pdf-vector config\n"
    "  $ pdf-vector config --show-all",
)
def config_show(
    show_all: bool = typer.Option(
        False, "--show-all", "-a", help="Show all configuration including defaults"
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Display the current system configuration."""

    try:
        config = Config()

        console.print("\n[bold cyan]⚙️  System Configuration[/bold cyan]\n")

        # Embedding configuration
        emb_table = Table(
            title="Embedding Configuration",
            show_header=False,
            border_style="blue",
            title_style="bold magenta",
            box=None,
            padding=(0, 2),
        )
        emb_table.add_column(style="cyan")
        emb_table.add_column(style="white")

        emb_table.add_row("Model Type", config.embedding.model_type.value)
        emb_table.add_row("Model Name", config.embedding.model_name)
        emb_table.add_row("Batch Size", str(config.embedding.batch_size))

        if show_all and config.embedding.openai_api_key:
            emb_table.add_row("OpenAI API Key", "[dim]***configured***[/dim]")

        console.print(emb_table)
        console.print()

        # Vector DB configuration
        db_table = Table(
            title="Vector Database Configuration",
            show_header=False,
            border_style="blue",
            title_style="bold magenta",
            box=None,
            padding=(0, 2),
        )
        db_table.add_column(style="cyan")
        db_table.add_column(style="white")

        db_table.add_row("Collection Name", config.chroma_db.collection_name)
        db_table.add_row("Persist Directory", str(config.chroma_db.persist_directory))

        console.print(db_table)
        console.print()

        # Text processing configuration
        if show_all:
            text_table = Table(
                title="Text Processing Configuration",
                show_header=False,
                border_style="blue",
                title_style="bold magenta",
                box=None,
                padding=(0, 2),
            )
            text_table.add_column(style="cyan")
            text_table.add_column(style="white")

            text_table.add_row(
                "Chunk Size", f"{config.text_processing.chunk_size} chars"
            )
            text_table.add_row(
                "Chunk Overlap", f"{config.text_processing.chunk_overlap} chars"
            )
            text_table.add_row(
                "Min Chunk Size", f"{config.text_processing.min_chunk_size} chars"
            )

            console.print(text_table)
            console.print()

        # Performance configuration
        if show_all:
            perf_table = Table(
                title="Performance Configuration",
                show_header=False,
                border_style="blue",
                title_style="bold magenta",
                box=None,
                padding=(0, 2),
            )
            perf_table.add_column(style="cyan")
            perf_table.add_column(style="white")

            perf_table.add_row("Max Workers", str(config.max_workers))
            perf_table.add_row("Debug Mode", "Yes" if config.debug else "No")
            perf_table.add_row("Log Level", config.logging.level.value)

            console.print(perf_table)
            console.print()

        console.print("[dim]💡 Use --show-all to see all configuration options[/dim]")

    except Exception as e:
        console.print(f"\n[red]✗ Config error: {e}[/red]")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


@app.command(
    name="collections",
    help="List all available collections.\n\n"
    "Examples:\n\n"
    "  $ pdf-vector collections",
)
def collections(
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """List all available vector database collections."""

    try:
        console.print("\n[bold cyan]📚 Available Collections[/bold cyan]\n")

        with Status("[bold green]Fetching collections...", console=console):
            pipeline = get_pipeline(debug=debug)
            collection_list = pipeline.vector_db.list_collections()

        if not collection_list:
            console.print("[yellow]No collections found[/yellow]")
            console.print("\n[dim]💡 Create a collection by processing a PDF:[/dim]")
            console.print(
                "   [cyan]pdf-vector process document.pdf --collection my_collection[/cyan]"
            )
            return

        # Create collections table
        table = Table(
            show_header=True,
            header_style="bold magenta",
            border_style="blue",
            title=f"Collections ({len(collection_list)})",
            title_style="bold cyan",
        )
        table.add_column("#", justify="right", style="dim")
        table.add_column("Name", style="cyan")
        table.add_column("Status", justify="center")

        for i, info in enumerate(collection_list, 1):
            # Normalize name across backends
            name = getattr(info, "name", str(info))
            # Try to get count for each collection
            try:
                col_info = pipeline.vector_db.get_collection_info(name)
                status = f"[green]{col_info.chunk_count:,} chunks[/green]"
            except Exception:
                status = "[dim]—[/dim]"

            table.add_row(str(i), name, status)

        console.print(table)
        console.print()

    except Exception as e:
        console.print(f"\n[red]✗ Collections error: {e}[/red]")
        if debug:
            console.print_exception()
        raise typer.Exit(1)


def main() -> None:
    """Main CLI entry point."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception:
        # Errors are handled by individual commands
        sys.exit(1)


if __name__ == "__main__":
    main()
