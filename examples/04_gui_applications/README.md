# GUI Application Examples

This directory demonstrates the graphical user interface capabilities of the PDF Vector System.

## Examples

### `basic_gui_usage.py`

Simple GUI application demonstrating core functionality.

### `custom_gui_integration.py`

How to integrate PDF Vector System into your own GUI applications.

### `gui_automation.py`

Programmatic control of GUI components for testing and automation.

### `gui_features.py`

GUI features including drag-and-drop, progress monitoring, and real-time updates.

### `gui_configuration.py`

GUI-based configuration management and settings.

### `gui_plugins/`

Directory containing example GUI plugins and extensions.

## GUI Features Covered

### Main Interface

- PDF processing with drag-and-drop
- Real-time progress monitoring
- Search interface with filters
- Document management
- Configuration settings

### Additional Features

- Batch processing with progress tracking
- Real-time search results
- Document preview and metadata
- System health monitoring
- Log viewing and debugging

## Prerequisites

- Python 3.9+
- PDF Vector System with GUI dependencies
- PySide6 and QFluentWidgets
- Display environment (for GUI applications)

## Installation

```bash
# Install GUI dependencies
pip install pdf-vector-system[gui]

# Or with uv
uv add pdf-vector-system[gui]
```

## Running Examples

```bash
# Navigate to this directory
cd examples/04_gui_applications

# Run GUI examples
python basic_gui_usage.py
python custom_gui_integration.py
```

## What You'll Learn

- GUI application architecture
- Integration patterns for custom applications
- Real-time monitoring and progress tracking
- User experience best practices
- GUI testing and automation
