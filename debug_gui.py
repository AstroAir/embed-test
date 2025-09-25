#!/usr/bin/env python3
"""Debug script to isolate GUI initialization issues."""

import sys
import traceback
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_status_widget(app):
    """Test StatusWidget initialization in isolation."""
    print("Testing StatusWidget initialization...")

    try:
        from pdf_vector_system.config.settings import Config
        from pdf_vector_system.gui.widgets.status_widget import StatusWidget

        # Create config
        config = Config()

        # Try to create StatusWidget
        print("Creating StatusWidget...")
        widget = StatusWidget(config)
        print("StatusWidget created successfully!")

        # Check if controller exists
        if hasattr(widget, 'controller'):
            print(f"Controller exists: {widget.controller}")
        else:
            print("Controller does not exist!")

        return True, app

    except Exception as e:
        print(f"Error creating StatusWidget: {e}")
        traceback.print_exc()
        return False, app

def test_main_window(app):
    """Test MainWindow initialization."""
    print("\nTesting MainWindow initialization...")

    try:
        from pdf_vector_system.config.settings import Config
        from pdf_vector_system.gui.main_window import MainWindow

        # Create config
        config = Config()

        # Try to create MainWindow
        print("Creating MainWindow...")
        window = MainWindow(config)
        print("MainWindow created successfully!")

        return True

    except Exception as e:
        print(f"Error creating MainWindow: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting GUI debug tests...")

    from PySide6.QtWidgets import QApplication

    # Create QApplication once
    app = QApplication(sys.argv)

    # Test StatusWidget first
    status_ok, app = test_status_widget(app)

    if status_ok:
        # Test MainWindow
        main_ok = test_main_window(app)

        if main_ok:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ MainWindow test failed!")
    else:
        print("\n❌ StatusWidget test failed!")
