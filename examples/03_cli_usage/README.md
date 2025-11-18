# CLI Usage Examples

This directory contains comprehensive examples of using the VectorFlow command-line interface.

## Examples

### `basic_cli_commands.py`

Demonstrates all basic CLI commands with example usage.

### `batch_processing_cli.py`

Shows how to process multiple PDFs efficiently using CLI commands.

### `search_cli.py`

Search patterns and filtering using the CLI.

### `collection_management_cli.py`

Managing multiple collections and databases through CLI.

### `configuration_cli.py`

Using environment variables and configuration files with CLI.

### `automation_scripts/`

Directory containing shell scripts for common automation tasks.

## CLI Commands Covered

### Processing Commands

```bash
# Process single PDF
vectorflow process document.pdf

# Process multiple PDFs
vectorflow process *.pdf

# Process with custom settings
vectorflow process document.pdf --model all-mpnet-base-v2 --batch-size 32
```

### Search Commands

```bash
# Basic search
vectorflow search "machine learning"

# Search with filters
vectorflow search "AI" --document doc1 --page 5 --results 10
```

### Management Commands

```bash
# Show statistics
vectorflow stats

# List documents
vectorflow list-docs

# Health check
vectorflow health
```

## Prerequisites

- VectorFlow installed (CLI entrypoint `vectorflow`)
- Sample PDF files
- Appropriate API keys for cloud providers

## What You'll Learn

- Complete CLI command reference
- Batch processing workflows
- Search and filtering techniques
- Collection management
- Automation scripting patterns
