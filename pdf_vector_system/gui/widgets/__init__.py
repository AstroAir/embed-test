"""
GUI widgets package for PDF Vector System.

This package contains reusable widgets and components for the GUI application.

Widgets:
- ProcessingWidget: PDF file processing interface
- SearchWidget: Search and results display interface  
- DocumentWidget: Document management interface
- ConfigWidget: Configuration settings interface
- StatusWidget: System status and health monitoring
- LogWidget: Log display and monitoring
"""

from .base import BaseWidget
from .processing_widget import ProcessingWidget
from .search_widget import SearchWidget
from .document_widget import DocumentWidget
from .config_widget import ConfigWidget
from .status_widget import StatusWidget
from .log_widget import LogWidget

__all__ = [
    "BaseWidget",
    "ProcessingWidget", 
    "SearchWidget",
    "DocumentWidget",
    "ConfigWidget",
    "StatusWidget",
    "LogWidget"
]
