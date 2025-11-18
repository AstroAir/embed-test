# Contributing to Examples

Thank you for your interest in contributing to the VectorFlow examples! This guide will help you create high-quality examples that benefit the community.

## Example Guidelines

### Structure and Organization

1. **Follow the directory structure**: Place examples in the appropriate category directory
2. **Use descriptive names**: File names should clearly indicate what the example demonstrates
3. **Include documentation**: Every example should have clear docstrings and comments
4. **Add README updates**: Update the relevant README.md files when adding new examples

### Code Quality Standards

1. **Follow project coding standards**: Use the same style as the main codebase
2. **Include error handling**: Examples should handle common error scenarios gracefully
3. **Add logging**: Use appropriate logging levels for debugging and information
4. **Resource cleanup**: Ensure proper cleanup of resources (files, connections, etc.)

### Documentation Requirements

1. **Clear purpose**: Explain what the example demonstrates
2. **Prerequisites**: List any requirements, dependencies, or setup needed
3. **Usage instructions**: Provide clear instructions on how to run the example
4. **Expected output**: Describe what users should expect to see
5. **Learning objectives**: Explain what users will learn from the example

## Example Template

Use this template for new examples:

```python
"""
[Brief description of what this example demonstrates]

This example shows how to [specific functionality]. It demonstrates:
- [Key concept 1]
- [Key concept 2]
- [Key concept 3]

Prerequisites:
- [Requirement 1]
- [Requirement 2]
- [API keys/credentials needed]

Usage:
    python example_name.py

Expected Output:
    [Description of expected output]

Learning Objectives:
- [What users will learn]
- [Key takeaways]
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from vectorflow import Config, PDFVectorPipeline
from vectorflow.core.config.settings import EmbeddingModelType

# Import example utilities
from utils.example_helpers import setup_example_environment, print_section
from utils.performance_utils import measure_performance


def main() -> None:
    """
    Main example function.

    Demonstrates [specific functionality] with proper error handling
    and resource management.
    """
    print_section("Example: [Example Name]")

    try:
        # Setup
        config = setup_example_environment()

        # Configure for this example
        config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
        config.embedding.model_name = "all-MiniLM-L6-v2"

        # Main example logic
        with measure_performance("example_operation") as timer:
            result = demonstrate_functionality(config)

        # Display results
        if result:
            print(f"✅ Example completed successfully in {timer.elapsed:.2f}s")
            display_results(result)
        else:
            print("❌ Example failed")

    except Exception as e:
        print(f"❌ Error running example: {e}")
        sys.exit(1)


def demonstrate_functionality(config: Config) -> Optional[dict]:
    """
    Core example functionality.

    Args:
        config: Configuration object

    Returns:
        Results dictionary or None if failed
    """
    # Implementation here
    pass


def display_results(results: dict) -> None:
    """
    Display example results in a user-friendly format.

    Args:
        results: Results dictionary from the example
    """
    # Results display logic here
    pass


if __name__ == "__main__":
    main()
```

## Testing Examples

Before submitting, ensure your example:

1. **Runs successfully** with default configuration
2. **Handles missing dependencies** gracefully
3. **Provides helpful error messages** when things go wrong
4. **Cleans up resources** properly
5. **Works with sample data** provided in the repository

## Submission Process

1. **Create a feature branch**: `git checkout -b feature/new-example-name`
2. **Add your example**: Place it in the appropriate directory
3. **Update documentation**: Update relevant README files
4. **Test thoroughly**: Ensure the example works as expected
5. **Submit a pull request**: Include a clear description of what the example demonstrates

## Review Criteria

Examples will be reviewed for:

- **Clarity and usefulness**: Does it clearly demonstrate the intended functionality?
- **Code quality**: Does it follow project standards and best practices?
- **Documentation**: Is it well-documented and easy to understand?
- **Error handling**: Does it handle errors gracefully?
- **Performance**: Is it reasonably efficient?
- **Completeness**: Does it include all necessary components?

## Getting Help

If you need help creating an example:

1. **Check existing examples**: Look for similar patterns in the codebase
2. **Read the documentation**: Review the main project documentation
3. **Ask questions**: Use GitHub Discussions or Issues for questions
4. **Start simple**: Begin with a basic example and iterate

## Common Pitfalls

Avoid these common issues:

1. **Hardcoded paths**: Use relative paths and Path objects
2. **Missing error handling**: Always handle potential exceptions
3. **Unclear documentation**: Explain not just what, but why
4. **Resource leaks**: Ensure proper cleanup of files, connections, etc.
5. **Overly complex examples**: Keep examples focused and simple

Thank you for contributing to the VectorFlow examples!
