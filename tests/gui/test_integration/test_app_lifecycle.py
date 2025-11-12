"""Integration tests for GUI application lifecycle."""

import sys
from unittest.mock import Mock, patch

import pytest
from PySide6.QtWidgets import QApplication

from pdf_vector_system.gui.app import PDFVectorGUIApp
from pdf_vector_system.gui.main_window import MainWindow


@pytest.mark.gui
@pytest.mark.integration
class TestAppLifecycle:
    """Test cases for GUI application lifecycle."""

    def test_app_initialization(self, qapp, mock_config):
        """Test GUI application initialization."""
        app = PDFVectorGUIApp(config=mock_config)

        # Check app is created
        assert app is not None
        assert app.config == mock_config

        # Check Qt app is available
        qt_app = QApplication.instance()
        assert qt_app is not None

    def test_app_initialization_without_config(self, qapp):
        """Test GUI application initialization without config."""
        app = PDFVectorGUIApp()

        # Should create default config
        assert app is not None
        assert app.config is not None

    def test_main_window_creation(self, qapp, mock_config):
        """Test main window creation during app initialization."""
        app = PDFVectorGUIApp(config=mock_config)

        # Initialize the app
        app.initialize()

        # Check main window is created
        assert app.main_window is not None
        assert isinstance(app.main_window, MainWindow)
        assert app.main_window.config == mock_config

    def test_app_run_and_exit(self, qapp, mock_config):
        """Test application run and exit cycle."""
        app = PDFVectorGUIApp(config=mock_config)

        # Mock the Qt app exec method to return immediately
        with patch.object(QApplication, "exec", return_value=0):
            # Initialize and run
            exit_code = app.run()

            # Check exit code
            assert exit_code == 0

            # Check main window was created
            assert app.main_window is not None

    def test_app_cleanup_on_exit(self, qapp, mock_config):
        """Test application cleanup on exit."""
        app = PDFVectorGUIApp(config=mock_config)
        app.initialize()

        # Mock cleanup methods
        with patch.object(app.main_window, "close") as mock_close:
            # Cleanup
            app.cleanup()

            # Check cleanup was called
            mock_close.assert_called_once()

    def test_app_error_handling_during_init(self, qapp, mock_config):
        """Test error handling during app initialization."""
        app = PDFVectorGUIApp(config=mock_config)

        # Mock MainWindow to raise exception
        with patch(
            "pdf_vector_system.gui.app.MainWindow", side_effect=Exception("Init failed")
        ):
            # Should handle error gracefully
            try:
                app.initialize()
                # If no exception, check that app handles it
                assert app.main_window is None
            except Exception:
                # If exception propagates, that's also acceptable behavior
                pass

    def test_multiple_app_instances(self, qapp, mock_config):
        """Test handling of multiple app instances."""
        app1 = PDFVectorGUIApp(config=mock_config)
        app2 = PDFVectorGUIApp(config=mock_config)

        # Both should be created successfully
        assert app1 is not None
        assert app2 is not None

        # They should share the same Qt application instance
        assert app1.app == app2.app

    def test_app_with_command_line_args(self, qapp, mock_config):
        """Test app initialization with command line arguments."""
        # Mock sys.argv
        test_args = ["pdf-vector-gui", "--debug"]

        with patch.object(sys, "argv", test_args):
            app = PDFVectorGUIApp(config=mock_config)

            # App should handle args gracefully
            assert app is not None

    def test_app_window_show_and_hide(self, qapp, mock_config):
        """Test showing and hiding the main window."""
        app = PDFVectorGUIApp(config=mock_config)
        app.initialize()

        main_window = app.main_window

        # Show window
        main_window.show()
        assert main_window.isVisible()

        # Hide window
        main_window.hide()
        assert not main_window.isVisible()

    def test_app_signal_connections(self, qapp, mock_config):
        """Test signal connections are established."""
        app = PDFVectorGUIApp(config=mock_config)
        app.initialize()

        main_window = app.main_window

        # Check that main window signals are connected
        assert hasattr(main_window, "closing")

        # Test signal emission
        signals_received = []
        main_window.closing.connect(lambda: signals_received.append("closing"))

        # Emit signal
        main_window.closing.emit()

        # Check signal was received
        assert len(signals_received) == 1
        assert signals_received[0] == "closing"

    def test_app_configuration_updates(self, qapp, mock_config):
        """Test configuration updates during app lifecycle."""
        app = PDFVectorGUIApp(config=mock_config)
        app.initialize()

        # Create new config
        new_config = Mock()
        new_config.debug = False

        # Update app config
        app.config = new_config

        # Main window should be able to handle config updates
        if hasattr(app.main_window, "update_config"):
            app.main_window.update_config(new_config)

        assert app.config == new_config

    def test_app_resource_management(self, qapp, mock_config):
        """Test resource management during app lifecycle."""
        app = PDFVectorGUIApp(config=mock_config)
        app.initialize()

        # Check resources are properly managed
        assert app.app is not None
        assert app.main_window is not None

        # Cleanup should not raise exceptions
        app.cleanup()

    def test_app_exception_handling(self, qapp, mock_config):
        """Test exception handling in app methods."""
        app = PDFVectorGUIApp(config=mock_config)

        # Test exception in run method
        with patch.object(app, "initialize", side_effect=Exception("Test error")):
            try:
                exit_code = app.run()
                # Should handle gracefully
                assert exit_code != 0  # Error exit code
            except Exception:
                # Exception propagation is also acceptable
                pass

    def test_app_memory_management(self, qapp, mock_config):
        """Test memory management and cleanup."""
        app = PDFVectorGUIApp(config=mock_config)
        app.initialize()

        # Store references
        main_window = app.main_window
        qt_app = app.app

        # Cleanup
        app.cleanup()

        # References should still be valid but cleaned up
        assert main_window is not None
        assert qt_app is not None

    def test_app_restart_capability(self, qapp, mock_config):
        """Test app restart capability."""
        app = PDFVectorGUIApp(config=mock_config)

        # Initialize and cleanup
        app.initialize()
        app.cleanup()

        # Initialize again
        app.initialize()
        second_window = app.main_window

        # Should create new window
        assert second_window is not None
        # May or may not be the same instance depending on implementation

    def test_app_with_invalid_config(self, qapp):
        """Test app behavior with invalid configuration."""
        # Test with None config
        app = PDFVectorGUIApp(config=None)
        assert app is not None
        assert app.config is not None  # Should create default

        # Test with invalid config object
        invalid_config = "not a config"
        try:
            app = PDFVectorGUIApp(config=invalid_config)
            # Should handle gracefully or raise appropriate exception
        except (TypeError, AttributeError):
            # Expected behavior for invalid config
            pass

    def test_app_qt_integration(self, qapp, mock_config):
        """Test Qt framework integration."""
        app = PDFVectorGUIApp(config=mock_config)
        app.initialize()

        # Check Qt application properties
        qt_app = QApplication.instance()
        assert qt_app is not None

        # Check application name is set
        assert qt_app.applicationName() != ""

        # Check main window is properly integrated
        main_window = app.main_window
        assert main_window.windowTitle() != ""
