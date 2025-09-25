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

import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.example_helpers import (
    print_section, print_subsection, example_context
)
from utils.sample_data_generator import ensure_sample_data


def run_cli_command(command: List[str], description: str) -> Dict[str, Any]:
    """Run a CLI command and return the result."""
    print(f"   💻 {description}")
    print(f"      Command: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print(f"      Exit code: {result.returncode}")
        
        if result.stdout:
            print(f"      Output:")
            for line in result.stdout.strip().split('\n')[:10]:  # Show first 10 lines
                print(f"        {line}")
            if len(result.stdout.strip().split('\n')) > 10:
                print(f"        ... (output truncated)")
        
        if result.stderr and result.returncode != 0:
            print(f"      Error:")
            for line in result.stderr.strip().split('\n')[:5]:  # Show first 5 error lines
                print(f"        {line}")
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
        
    except subprocess.TimeoutExpired:
        print(f"      ❌ Command timed out")
        return {"success": False, "error": "timeout"}
    except Exception as e:
        print(f"      ❌ Error running command: {e}")
        return {"success": False, "error": str(e)}


def demonstrate_help_commands() -> None:
    """Demonstrate help and information commands."""
    print_subsection("Help and Information Commands")
    
    help_commands = [
        {
            "command": ["pdf-vector-system", "--help"],
            "description": "Show main help"
        },
        {
            "command": ["pdf-vector-system", "process", "--help"],
            "description": "Show process command help"
        },
        {
            "command": ["pdf-vector-system", "search", "--help"],
            "description": "Show search command help"
        },
        {
            "command": ["pdf-vector-system", "collection", "--help"],
            "description": "Show collection command help"
        },
        {
            "command": ["pdf-vector-system", "version"],
            "description": "Show version information"
        }
    ]
    
    print("📚 CLI Help Commands:")
    
    for cmd_info in help_commands:
        result = run_cli_command(cmd_info["command"], cmd_info["description"])
        print()


def demonstrate_document_processing() -> None:
    """Demonstrate document processing commands."""
    print_subsection("Document Processing Commands")
    
    # Ensure sample data exists
    sample_dir = Path("examples/sample_data")
    if not ensure_sample_data(sample_dir):
        print("⚠️  No sample data available for CLI demonstration")
        return
    
    pdf_files = list(sample_dir.glob("*.pdf"))
    if not pdf_files:
        print("📄 No PDF files found for processing")
        return
    
    # Use first PDF file for demonstration
    pdf_file = pdf_files[0]
    
    processing_commands = [
        {
            "command": [
                "pdf-vector-system", "process",
                str(pdf_file),
                "--collection", "cli_demo",
                "--model-type", "sentence-transformers",
                "--model-name", "all-MiniLM-L6-v2",
                "--verbose"
            ],
            "description": f"Process {pdf_file.name} with verbose output"
        },
        {
            "command": [
                "pdf-vector-system", "process",
                str(sample_dir),
                "--collection", "cli_batch_demo",
                "--batch-size", "16",
                "--max-workers", "2"
            ],
            "description": f"Batch process all PDFs in {sample_dir}"
        }
    ]
    
    print("📖 Document Processing Commands:")
    
    for cmd_info in processing_commands:
        result = run_cli_command(cmd_info["command"], cmd_info["description"])
        print()


def demonstrate_search_commands() -> None:
    """Demonstrate search commands."""
    print_subsection("Search Commands")
    
    search_commands = [
        {
            "command": [
                "pdf-vector-system", "search",
                "machine learning",
                "--collection", "cli_demo",
                "--limit", "5"
            ],
            "description": "Search for 'machine learning' with 5 results"
        },
        {
            "command": [
                "pdf-vector-system", "search",
                "artificial intelligence",
                "--collection", "cli_demo",
                "--limit", "3",
                "--min-score", "0.5",
                "--format", "json"
            ],
            "description": "Search with minimum score and JSON output"
        },
        {
            "command": [
                "pdf-vector-system", "search",
                "neural networks",
                "--collection", "cli_demo",
                "--document-id", "sample_research_1",
                "--limit", "3"
            ],
            "description": "Search within specific document"
        }
    ]
    
    print("🔍 Search Commands:")
    
    for cmd_info in search_commands:
        result = run_cli_command(cmd_info["command"], cmd_info["description"])
        print()


def demonstrate_collection_management() -> None:
    """Demonstrate collection management commands."""
    print_subsection("Collection Management Commands")
    
    collection_commands = [
        {
            "command": [
                "pdf-vector-system", "collection", "list"
            ],
            "description": "List all collections"
        },
        {
            "command": [
                "pdf-vector-system", "collection", "info",
                "--collection", "cli_demo"
            ],
            "description": "Show collection information"
        },
        {
            "command": [
                "pdf-vector-system", "collection", "stats",
                "--collection", "cli_demo"
            ],
            "description": "Show collection statistics"
        },
        {
            "command": [
                "pdf-vector-system", "collection", "documents",
                "--collection", "cli_demo"
            ],
            "description": "List documents in collection"
        }
    ]
    
    print("🗄️  Collection Management Commands:")
    
    for cmd_info in collection_commands:
        result = run_cli_command(cmd_info["command"], cmd_info["description"])
        print()


def demonstrate_configuration_commands() -> None:
    """Demonstrate configuration commands."""
    print_subsection("Configuration Commands")
    
    config_commands = [
        {
            "command": [
                "pdf-vector-system", "config", "show"
            ],
            "description": "Show current configuration"
        },
        {
            "command": [
                "pdf-vector-system", "config", "validate"
            ],
            "description": "Validate configuration"
        },
        {
            "command": [
                "pdf-vector-system", "config", "providers"
            ],
            "description": "Show available embedding providers"
        }
    ]
    
    print("⚙️  Configuration Commands:")
    
    for cmd_info in config_commands:
        result = run_cli_command(cmd_info["command"], cmd_info["description"])
        print()


def demonstrate_advanced_options() -> None:
    """Demonstrate advanced CLI options and patterns."""
    print_subsection("Advanced CLI Options")
    
    print("🚀 Advanced CLI Usage Patterns:")
    
    # Environment variable usage
    print("\n1. 🌍 Environment Variables:")
    env_examples = [
        "export EMBEDDING__MODEL_TYPE=sentence-transformers",
        "export EMBEDDING__MODEL_NAME=all-mpnet-base-v2",
        "export CHROMA_DB__COLLECTION_NAME=production_docs",
        "export DEBUG=true"
    ]
    
    for example in env_examples:
        print(f"   {example}")
    
    # Configuration file usage
    print("\n2. 📁 Configuration Files:")
    config_examples = [
        "pdf-vector-system --config config.yaml process document.pdf",
        "pdf-vector-system --config production.env search 'query'",
        "pdf-vector-system --config-dir ./configs process *.pdf"
    ]
    
    for example in config_examples:
        print(f"   {example}")
    
    # Batch processing patterns
    print("\n3. 📦 Batch Processing Patterns:")
    batch_examples = [
        "# Process all PDFs in directory",
        "pdf-vector-system process ./documents/ --recursive",
        "",
        "# Process with custom settings",
        "pdf-vector-system process ./docs/ --chunk-size 1000 --overlap 200",
        "",
        "# Process and search in pipeline",
        "pdf-vector-system process doc.pdf && pdf-vector-system search 'query'"
    ]
    
    for example in batch_examples:
        print(f"   {example}")
    
    # Output formatting
    print("\n4. 📄 Output Formatting:")
    format_examples = [
        "pdf-vector-system search 'query' --format json",
        "pdf-vector-system search 'query' --format csv",
        "pdf-vector-system search 'query' --format table",
        "pdf-vector-system collection stats --format yaml"
    ]
    
    for example in format_examples:
        print(f"   {example}")


def demonstrate_automation_patterns() -> None:
    """Demonstrate CLI automation patterns."""
    print_subsection("Automation Patterns")
    
    print("🤖 CLI Automation Examples:")
    
    # Shell script examples
    print("\n1. 📜 Shell Script Automation:")
    
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
    
    print("   Example shell script:")
    for line in shell_script.split('\n'):
        print(f"     {line}")
    
    # Python automation
    print("\n2. 🐍 Python Automation:")
    
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
    
    print("   Example Python automation:")
    for line in python_automation.split('\n'):
        print(f"     {line}")
    
    # CI/CD integration
    print("\n3. 🔄 CI/CD Integration:")
    
    cicd_examples = [
        "# GitHub Actions example",
        "- name: Process Documents",
        "  run: |",
        "    pdf-vector-system process ./docs/",
        "    pdf-vector-system collection stats",
        "",
        "# Docker example",
        "docker run -v ./docs:/data pdf-vector-system \\",
        "  process /data --collection production"
    ]
    
    for example in cicd_examples:
        print(f"   {example}")


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
        
        print("🎯 CLI Best Practices:")
        print()
        print("1. 📚 Getting Help:")
        print("   - Use --help for any command")
        print("   - Check command-specific help")
        print("   - Review configuration options")
        print()
        print("2. 🔧 Configuration:")
        print("   - Use environment variables for settings")
        print("   - Create configuration files for complex setups")
        print("   - Validate configuration before processing")
        print()
        print("3. 📦 Batch Processing:")
        print("   - Process directories recursively")
        print("   - Use appropriate batch sizes")
        print("   - Monitor progress with verbose output")
        print()
        print("4. 🔍 Search Operations:")
        print("   - Use appropriate result limits")
        print("   - Filter by document or collection")
        print("   - Choose output format for integration")
        print()
        print("5. 🤖 Automation:")
        print("   - Create shell scripts for repeated tasks")
        print("   - Use Python for complex automation")
        print("   - Integrate with CI/CD pipelines")
        print()
        print("Common CLI Patterns:")
        print("- pdf-vector-system process ./docs/ --verbose")
        print("- pdf-vector-system search 'query' --limit 10 --format json")
        print("- pdf-vector-system collection stats --collection mydata")
        print("- pdf-vector-system config show")
        print()
        print("Next steps:")
        print("- Try the CLI commands with your own data")
        print("- Create automation scripts for your workflow")
        print("- Explore advanced CLI examples in this directory")


if __name__ == "__main__":
    main()
