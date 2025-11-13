"""
GUI controllers package for PDF Vector System.

This package contains controller classes that manage the interaction between
GUI components and the business logic layer.

Controllers:
- MainController: Primary application controller
- ProcessingController: PDF processing operations
- SearchController: Search and retrieval operations
- DocumentController: Document management operations
- ConfigController: Configuration management
- StatusController: System status monitoring
"""

from vectorflow.gui.controllers.config_controller import ConfigController
from vectorflow.gui.controllers.document_controller import DocumentController
from vectorflow.gui.controllers.main_controller import MainController
from vectorflow.gui.controllers.processing_controller import ProcessingController
from vectorflow.gui.controllers.search_controller import SearchController
from vectorflow.gui.controllers.status_controller import StatusController

__all__ = [
    "ConfigController",
    "DocumentController",
    "MainController",
    "ProcessingController",
    "SearchController",
    "StatusController",
]
