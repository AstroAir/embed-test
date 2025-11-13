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

from vectorflow.gui.widgets.base import BaseWidget
from vectorflow.gui.widgets.config_widget import ConfigWidget
from vectorflow.gui.widgets.document_widget import DocumentWidget
from vectorflow.gui.widgets.log_widget import LogWidget
from vectorflow.gui.widgets.processing_widget import ProcessingWidget
from vectorflow.gui.widgets.search_widget import SearchWidget
from vectorflow.gui.widgets.status_widget import StatusWidget

__all__ = [
    "BaseWidget",
    "ConfigWidget",
    "DocumentWidget",
    "LogWidget",
    "ProcessingWidget",
    "SearchWidget",
    "StatusWidget",
]
