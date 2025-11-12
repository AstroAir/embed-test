# CLI Usage Examples

This directory contains comprehensive examples of using the PDF Vector System command-line interface.

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
pdf-vector process document.pdf

# Process multiple PDFs
pdf-vector process *.pdf

# Process with custom settings
pdf-vector process document.pdf --model all-mpnet-base-v2 --batch-size 32
```

### Search Commands

```bash
# Basic search
pdf-vector search "machine learning"

# Search with filters
pdf-vector search "AI" --document doc1 --page 5 --results 10
```

### Management Commands

```bash
# Show statistics
pdf-vector stats

# List documents
pdf-vector list

# Health check
pdf-vector health
```

## Prerequisites

- PDF Vector System installed with CLI
- Sample PDF files
- Appropriate API keys for cloud providers

## What You'll Learn

- Complete CLI command reference
- Batch processing workflows
- Search and filtering techniques
- Collection management
- Automation scripting patterns
