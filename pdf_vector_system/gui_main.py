"""
GUI entry point for PDF Vector System.

This module provides the main entry point for launching the GUI application.
It can be used as a script or imported to start the GUI programmatically.
"""

import sys
import os
from pathlib import Path
from typing import Optional

# Add the project root to Python path if needed
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pdf_vector_system.config.settings import Config
from pdf_vector_system.gui.app import PDFVectorGUIApp
from pdf_vector_system.utils.logging import setup_logging


def main() -> int:
    """
    Main entry point for the GUI application.
    
    Returns:
        Application exit code
    """
    try:
        # Load configuration
        config = Config()
        
        # Setup logging
        setup_logging(config.logging)
        
        # Create and run GUI application
        app = PDFVectorGUIApp(config)
        return app.run()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure PySide6 is installed: uv add PySide6")
        return 1
        
    except Exception as e:
        print(f"Error starting GUI application: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
