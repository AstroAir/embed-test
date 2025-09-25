#!/usr/bin/env python3
"""
Development launcher script for PDF Vector System GUI.

This script can be used during development to launch the GUI application
without installing the package.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pdf_vector_system.gui_main import main

if __name__ == "__main__":
    sys.exit(main())
