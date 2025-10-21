# Example Utilities

This directory contains utility functions and helpers used across the examples.

## Modules

### `sample_data_generator.py`

Generates sample PDF files and test data for examples.

### `example_helpers.py`

Common helper functions used across multiple examples.

### `performance_utils.py`

Performance measurement and benchmarking utilities.

### `config_helpers.py`

Configuration management helpers and templates.

### `logging_setup.py`

Logging configuration for examples.

## Usage

```python
from examples.utils.example_helpers import setup_example_environment
from examples.utils.performance_utils import measure_performance
from examples.utils.config_helpers import get_example_config

# Set up example environment
config = setup_example_environment("basic_usage")

# Measure performance
with measure_performance("pdf_processing") as timer:
    # Your code here
    pass

print(f"Processing took {timer.elapsed:.2f} seconds")
```

## Common Patterns

### Example Setup

```python
def setup_example():
    """Standard example setup pattern."""
    # Load configuration
    config = get_example_config()

    # Set up logging
    setup_example_logging(config)

    # Create sample data if needed
    ensure_sample_data_exists()

    return config
```

### Error Handling

```python
def run_example_safely(example_func):
    """Run example with proper error handling."""
    try:
        return example_func()
    except Exception as e:
        logger.error(f"Example failed: {e}")
        return None
```

### Performance Measurement

```python
@measure_performance
def example_function():
    """Function with automatic performance measurement."""
    # Your code here
    pass
```
