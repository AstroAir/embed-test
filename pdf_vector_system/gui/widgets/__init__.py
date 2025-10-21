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

from pdf_vector_system.gui.widgets.base import BaseWidget
from pdf_vector_system.gui.widgets.config_widget import ConfigWidget
from pdf_vector_system.gui.widgets.document_widget import DocumentWidget
from pdf_vector_system.gui.widgets.log_widget import LogWidget
from pdf_vector_system.gui.widgets.processing_widget import ProcessingWidget
from pdf_vector_system.gui.widgets.search_widget import SearchWidget
from pdf_vector_system.gui.widgets.status_widget import StatusWidget

__all__ = [
    "BaseWidget",
    "ConfigWidget",
    "DocumentWidget",
    "LogWidget",
    "ProcessingWidget",
    "SearchWidget",
    "StatusWidget",
]
