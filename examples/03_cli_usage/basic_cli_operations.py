"""
Basic CLI Operations Example

This example demonstrates the command-line interface of the PDF Vector System:
- Document processing commands
- Search operations
- Collection management
- Configuration options
- Batch operations

Prerequisites:
- PDF Vector System installed with CLI
- Sample PDF files
- Understanding of command-line interfaces

Usage:
    python basic_cli_operations.py

Expected Output:
    - CLI command demonstrations
    - Example command outputs
    - Best practices for CLI usage
    - Automation patterns

Learning Objectives:
- Master CLI command usage
- Learn batch processing with CLI
- Understand configuration options
- See automation patterns
"""

import subprocess
import sys
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.example_helpers import example_context, print_section, print_subsection
from utils.sample_data_generator import ensure_sample_data


def run_cli_command(command: list[str], description: str) -> dict[str, Any]:
    """Run a CLI command and return the result."""

    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=60
        )

        if result.stdout:
            for _line in result.stdout.strip().split("\n")[:10]:  # Show first 10 lines
                pass
            if len(result.stdout.strip().split("\n")) > 10:
                pass

        if result.stderr and result.returncode != 0:
            for _line in result.stderr.strip().split("\n")[
                :5
            ]:  # Show first 5 error lines
                pass

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def demonstrate_help_commands() -> None:
    """Demonstrate help and information commands."""
    print_subsection("Help and Information Commands")

    help_commands = [
        {"command": ["pdf-vector-system", "--help"], "description": "Show main help"},
        {
            "command": ["pdf-vector-system", "process", "--help"],
            "description": "Show process command help",
        },
        {
            "command": ["pdf-vector-system", "search", "--help"],
            "description": "Show search command help",
        },
        {
            "command": ["pdf-vector-system", "collection", "--help"],
            "description": "Show collection command help",
        },
        {
            "command": ["pdf-vector-system", "version"],
            "description": "Show version information",
        },
    ]

    for cmd_info in help_commands:
        run_cli_command(cmd_info["command"], cmd_info["description"])


def demonstrate_document_processing() -> None:
    """Demonstrate document processing commands."""
    print_subsection("Document Processing Commands")

    # Ensure sample data exists
    sample_dir = Path("examples/sample_data")
    if not ensure_sample_data(sample_dir):
        return

    pdf_files = list(sample_dir.glob("*.pdf"))
    if not pdf_files:
        return

    # Use first PDF file for demonstration
    pdf_file = pdf_files[0]

    processing_commands = [
        {
            "command": [
                "pdf-vector-system",
                "process",
                str(pdf_file),
                "--collection",
                "cli_demo",
                "--model-type",
                "sentence-transformers",
                "--model-name",
                "all-MiniLM-L6-v2",
                "--verbose",
            ],
            "description": f"Process {pdf_file.name} with verbose output",
        },
        {
            "command": [
                "pdf-vector-system",
                "process",
                str(sample_dir),
                "--collection",
                "cli_batch_demo",
                "--batch-size",
                "16",
                "--max-workers",
                "2",
            ],
            "description": f"Batch process all PDFs in {sample_dir}",
        },
    ]

    for cmd_info in processing_commands:
        run_cli_command(cmd_info["command"], cmd_info["description"])


def demonstrate_search_commands() -> None:
    """Demonstrate search commands."""
    print_subsection("Search Commands")

    search_commands = [
        {
            "command": [
                "pdf-vector-system",
                "search",
                "machine learning",
                "--collection",
                "cli_demo",
                "--limit",
                "5",
            ],
            "description": "Search for 'machine learning' with 5 results",
        },
        {
            "command": [
                "pdf-vector-system",
                "search",
                "artificial intelligence",
                "--collection",
                "cli_demo",
                "--limit",
                "3",
                "--min-score",
                "0.5",
                "--format",
                "json",
            ],
            "description": "Search with minimum score and JSON output",
        },
        {
            "command": [
                "pdf-vector-system",
                "search",
                "neural networks",
                "--collection",
                "cli_demo",
                "--document-id",
                "sample_research_1",
                "--limit",
                "3",
            ],
            "description": "Search within specific document",
        },
    ]

    for cmd_info in search_commands:
        run_cli_command(cmd_info["command"], cmd_info["description"])


def demonstrate_collection_management() -> None:
    """Demonstrate collection management commands."""
    print_subsection("Collection Management Commands")

    collection_commands = [
        {
            "command": ["pdf-vector-system", "collection", "list"],
            "description": "List all collections",
        },
        {
            "command": [
                "pdf-vector-system",
                "collection",
                "info",
                "--collection",
                "cli_demo",
            ],
            "description": "Show collection information",
        },
        {
            "command": [
                "pdf-vector-system",
                "collection",
                "stats",
                "--collection",
                "cli_demo",
            ],
            "description": "Show collection statistics",
        },
        {
            "command": [
                "pdf-vector-system",
                "collection",
                "documents",
                "--collection",
                "cli_demo",
            ],
            "description": "List documents in collection",
        },
    ]

    for cmd_info in collection_commands:
        run_cli_command(cmd_info["command"], cmd_info["description"])


def demonstrate_configuration_commands() -> None:
    """Demonstrate configuration commands."""
    print_subsection("Configuration Commands")

    config_commands = [
        {
            "command": ["pdf-vector-system", "config", "show"],
            "description": "Show current configuration",
        },
        {
            "command": ["pdf-vector-system", "config", "validate"],
            "description": "Validate configuration",
        },
        {
            "command": ["pdf-vector-system", "config", "providers"],
            "description": "Show available embedding providers",
        },
    ]

    for cmd_info in config_commands:
        run_cli_command(cmd_info["command"], cmd_info["description"])


def demonstrate_advanced_options() -> None:
    """Demonstrate advanced CLI options and patterns."""
    print_subsection("Advanced CLI Options")

    # Environment variable usage
    env_examples = [
        "export EMBEDDING__MODEL_TYPE=sentence-transformers",
        "export EMBEDDING__MODEL_NAME=all-mpnet-base-v2",
        "export CHROMA_DB__COLLECTION_NAME=production_docs",
        "export DEBUG=true",
    ]

    for _example in env_examples:
        pass

    # Configuration file usage
    config_examples = [
        "pdf-vector-system --config config.yaml process document.pdf",
        "pdf-vector-system --config production.env search 'query'",
        "pdf-vector-system --config-dir ./configs process *.pdf",
    ]

    for _example in config_examples:
        pass

    # Batch processing patterns
    batch_examples = [
        "# Process all PDFs in directory",
        "pdf-vector-system process ./documents/ --recursive",
        "",
        "# Process with custom settings",
        "pdf-vector-system process ./docs/ --chunk-size 1000 --overlap 200",
        "",
        "# Process and search in pipeline",
        "pdf-vector-system process doc.pdf && pdf-vector-system search 'query'",
    ]

    for _example in batch_examples:
        pass

    # Output formatting
    format_examples = [
        "pdf-vector-system search 'query' --format json",
        "pdf-vector-system search 'query' --format csv",
        "pdf-vector-system search 'query' --format table",
        "pdf-vector-system collection stats --format yaml",
    ]

    for _example in format_examples:
        pass


def demonstrate_automation_patterns() -> None:
    """Demonstrate CLI automation patterns."""
    print_subsection("Automation Patterns")

    # Shell script examples

    shell_script = '''#!/bin/bash
# Automated document processing script

# Set configuration
export EMBEDDING__MODEL_TYPE=sentence-transformers
export EMBEDDING__MODEL_NAME=all-MiniLM-L6-v2
export CHROMA_DB__COLLECTION_NAME=automated_docs

# Process new documents
echo "Processing new documents..."
pdf-vector-system process ./new_docs/ --verbose

# Verify processing
echo "Checking collection stats..."
pdf-vector-system collection stats --collection automated_docs

# Run test searches
echo "Running test searches..."
pdf-vector-system search "machine learning" --limit 3
pdf-vector-system search "artificial intelligence" --limit 3

echo "Automation complete!"'''

    for _line in shell_script.split("\n"):
        pass

    # Python automation

    python_automation = '''import subprocess
import json

def process_and_search(pdf_path, queries):
    """Process PDF and run searches."""

    # Process document
    process_cmd = [
        "pdf-vector-system", "process", pdf_path,
        "--collection", "auto_collection"
    ]
    subprocess.run(process_cmd, check=True)

    # Run searches
    results = {}
    for query in queries:
        search_cmd = [
            "pdf-vector-system", "search", query,
            "--collection", "auto_collection",
            "--format", "json", "--limit", "5"
        ]
        result = subprocess.run(search_cmd, capture_output=True, text=True)
        results[query] = json.loads(result.stdout)

    return results

# Usage
queries = ["machine learning", "data analysis"]
results = process_and_search("document.pdf", queries)'''

    for _line in python_automation.split("\n"):
        pass

    # CI/CD integration

    cicd_examples = [
        "# GitHub Actions example",
        "- name: Process Documents",
        "  run: |",
        "    pdf-vector-system process ./docs/",
        "    pdf-vector-system collection stats",
        "",
        "# Docker example",
        "docker run -v ./docs:/data pdf-vector-system \\",
        "  process /data --collection production",
    ]

    for _example in cicd_examples:
        pass


def main() -> None:
    """
    Demonstrate CLI usage patterns for the PDF Vector System.

    This function shows how to effectively use the command-line
    interface for various operations and automation scenarios.
    """
    with example_context("Basic CLI Operations"):
        print_section("CLI Command Overview")

        # Show help commands
        demonstrate_help_commands()

        print_section("Core Operations")

        # Document processing
        demonstrate_document_processing()

        # Search operations
        demonstrate_search_commands()

        # Collection management
        demonstrate_collection_management()

        # Configuration
        demonstrate_configuration_commands()

        print_section("Advanced Usage")

        # Advanced options
        demonstrate_advanced_options()

        # Automation patterns
        demonstrate_automation_patterns()

        print_section("CLI Usage Summary")


if __name__ == "__main__":
    main()
