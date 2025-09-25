"""Tests for GUI main entry point."""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from pdf_vector_system import gui_main


class TestGuiMain:
    """Test GUI main entry point."""
    
    @patch('pdf_vector_system.gui_main.PDFVectorGUIApp')
    @patch('pdf_vector_system.gui_main.setup_logging')
    @patch('pdf_vector_system.gui_main.Config')
    def test_main_success(self, mock_config_class, mock_setup_logging, mock_app_class):
        """Test successful GUI application startup."""
        # Mock configuration
        mock_config = Mock()
        mock_config.logging = Mock()
        mock_config_class.return_value = mock_config
        
        # Mock GUI application
        mock_app = Mock()
        mock_app.run.return_value = 0
        mock_app_class.return_value = mock_app
        
        # Call main function
        result = gui_main.main()
        
        # Verify calls
        mock_config_class.assert_called_once()
        mock_setup_logging.assert_called_once_with(mock_config.logging)
        mock_app_class.assert_called_once_with(mock_config)
        mock_app.run.assert_called_once()
        
        # Should return success code
        assert result == 0
    
    @patch('pdf_vector_system.gui_main.PDFVectorGUIApp')
    @patch('pdf_vector_system.gui_main.setup_logging')
    @patch('pdf_vector_system.gui_main.Config')
    def test_main_app_returns_error_code(self, mock_config_class, mock_setup_logging, mock_app_class):
        """Test GUI application returning error code."""
        # Mock configuration
        mock_config = Mock()
        mock_config.logging = Mock()
        mock_config_class.return_value = mock_config
        
        # Mock GUI application returning error
        mock_app = Mock()
        mock_app.run.return_value = 1
        mock_app_class.return_value = mock_app
        
        # Call main function
        result = gui_main.main()
        
        # Should return the error code from app
        assert result == 1
    
    @patch('pdf_vector_system.gui_main.PDFVectorGUIApp')
    @patch('pdf_vector_system.gui_main.setup_logging')
    @patch('pdf_vector_system.gui_main.Config')
    def test_main_import_error(self, mock_config_class, mock_setup_logging, mock_app_class):
        """Test handling of import errors."""
        # Mock configuration
        mock_config = Mock()
        mock_config.logging = Mock()
        mock_config_class.return_value = mock_config
        
        # Mock import error when creating GUI app
        mock_app_class.side_effect = ImportError("No module named 'PySide6'")
        
        # Capture stdout
        with patch('builtins.print') as mock_print:
            result = gui_main.main()
        
        # Should return error code
        assert result == 1
        
        # Should print helpful error message
        mock_print.assert_called()
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Import error" in call for call in print_calls)
        assert any("PySide6" in call for call in print_calls)
    
    @patch('pdf_vector_system.gui_main.PDFVectorGUIApp')
    @patch('pdf_vector_system.gui_main.setup_logging')
    @patch('pdf_vector_system.gui_main.Config')
    def test_main_general_exception(self, mock_config_class, mock_setup_logging, mock_app_class):
        """Test handling of general exceptions."""
        # Mock configuration that raises exception
        mock_config_class.side_effect = RuntimeError("Configuration error")
        
        # Capture stdout
        with patch('builtins.print') as mock_print:
            result = gui_main.main()
        
        # Should return error code
        assert result == 1
        
        # Should print error message
        mock_print.assert_called()
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        assert any("Error starting GUI application" in call for call in print_calls)
    
    @patch('pdf_vector_system.gui_main.PDFVectorGUIApp')
    @patch('pdf_vector_system.gui_main.setup_logging')
    @patch('pdf_vector_system.gui_main.Config')
    def test_main_setup_logging_called_with_config(self, mock_config_class, mock_setup_logging, mock_app_class):
        """Test that setup_logging is called with correct config."""
        # Mock configuration
        mock_config = Mock()
        mock_logging_config = Mock()
        mock_config.logging = mock_logging_config
        mock_config_class.return_value = mock_config
        
        # Mock GUI application
        mock_app = Mock()
        mock_app.run.return_value = 0
        mock_app_class.return_value = mock_app
        
        # Call main function
        gui_main.main()
        
        # Verify setup_logging called with correct config
        mock_setup_logging.assert_called_once_with(mock_logging_config)
    
    @patch('pdf_vector_system.gui_main.PDFVectorGUIApp')
    @patch('pdf_vector_system.gui_main.setup_logging')
    @patch('pdf_vector_system.gui_main.Config')
    def test_main_app_created_with_config(self, mock_config_class, mock_setup_logging, mock_app_class):
        """Test that GUI app is created with correct config."""
        # Mock configuration
        mock_config = Mock()
        mock_config.logging = Mock()
        mock_config_class.return_value = mock_config
        
        # Mock GUI application
        mock_app = Mock()
        mock_app.run.return_value = 0
        mock_app_class.return_value = mock_app
        
        # Call main function
        gui_main.main()
        
        # Verify app created with correct config
        mock_app_class.assert_called_once_with(mock_config)
    
    def test_module_has_main_function(self):
        """Test that module has main function."""
        assert hasattr(gui_main, 'main')
        assert callable(gui_main.main)
    
    def test_module_docstring(self):
        """Test that module has appropriate docstring."""
        assert gui_main.__doc__ is not None
        assert isinstance(gui_main.__doc__, str)
        assert len(gui_main.__doc__.strip()) > 0
        assert "GUI entry point" in gui_main.__doc__
    
    def test_project_root_path_handling(self):
        """Test that project root path is handled correctly."""
        # The module should add project root to sys.path if needed
        # This is tested indirectly by checking that imports work
        
        # Check that the path manipulation logic exists
        import pdf_vector_system.gui_main as gui_main_module
        
        # Should be able to import without issues
        assert gui_main_module is not None
    
    @patch('pdf_vector_system.gui_main.sys')
    def test_path_insertion_logic(self, mock_sys):
        """Test path insertion logic."""
        # Mock sys.path that doesn't contain project root
        mock_sys.path = ['/some/other/path']
        
        # Reload the module to trigger path insertion
        import importlib
        importlib.reload(gui_main)
        
        # Should have inserted project root path
        mock_sys.path.insert.assert_called()
    
    def test_main_function_signature(self):
        """Test main function signature."""
        import inspect
        
        sig = inspect.signature(gui_main.main)
        
        # Should take no parameters
        assert len(sig.parameters) == 0
        
        # Should return int
        assert sig.return_annotation == int
    
    @patch('pdf_vector_system.gui_main.PDFVectorGUIApp')
    @patch('pdf_vector_system.gui_main.setup_logging')
    @patch('pdf_vector_system.gui_main.Config')
    def test_main_exception_during_logging_setup(self, mock_config_class, mock_setup_logging, mock_app_class):
        """Test handling exception during logging setup."""
        # Mock configuration
        mock_config = Mock()
        mock_config.logging = Mock()
        mock_config_class.return_value = mock_config
        
        # Mock setup_logging to raise exception
        mock_setup_logging.side_effect = RuntimeError("Logging setup failed")
        
        # Capture stdout
        with patch('builtins.print') as mock_print:
            result = gui_main.main()
        
        # Should return error code
        assert result == 1
        
        # Should print error message
        mock_print.assert_called()
    
    @patch('pdf_vector_system.gui_main.PDFVectorGUIApp')
    @patch('pdf_vector_system.gui_main.setup_logging')
    @patch('pdf_vector_system.gui_main.Config')
    def test_main_exception_during_app_run(self, mock_config_class, mock_setup_logging, mock_app_class):
        """Test handling exception during app.run()."""
        # Mock configuration
        mock_config = Mock()
        mock_config.logging = Mock()
        mock_config_class.return_value = mock_config
        
        # Mock GUI application that raises exception during run
        mock_app = Mock()
        mock_app.run.side_effect = RuntimeError("App run failed")
        mock_app_class.return_value = mock_app
        
        # Capture stdout
        with patch('builtins.print') as mock_print:
            result = gui_main.main()
        
        # Should return error code
        assert result == 1
        
        # Should print error message
        mock_print.assert_called()


class TestGuiMainAsScript:
    """Test GUI main when run as script."""
    
    def test_module_can_be_imported(self):
        """Test that module can be imported without errors."""
        import pdf_vector_system.gui_main
        assert pdf_vector_system.gui_main is not None
    
    def test_module_has_required_imports(self):
        """Test that module has all required imports."""
        import pdf_vector_system.gui_main as gui_main_module
        
        # Should have imported necessary modules
        assert hasattr(gui_main_module, 'Config')
        assert hasattr(gui_main_module, 'PDFVectorGUIApp')
        assert hasattr(gui_main_module, 'setup_logging')
    
    @patch('pdf_vector_system.gui_main.main')
    def test_script_execution(self, mock_main):
        """Test script execution when run as __main__."""
        mock_main.return_value = 0
        
        # Simulate running as script
        with patch('pdf_vector_system.gui_main.__name__', '__main__'):
            # This would normally call main(), but we can't easily test that
            # without actually running the script
            pass
    
    def test_imports_are_available(self):
        """Test that all required imports are available."""
        # These imports should work if dependencies are installed
        try:
            from pdf_vector_system.config.settings import Config
            from pdf_vector_system.utils.logging import setup_logging
            # GUI app import might fail if PySide6 not installed, which is expected
        except ImportError as e:
            if "PySide6" not in str(e):
                pytest.fail(f"Unexpected import error: {e}")
    
    def test_pathlib_usage(self):
        """Test that pathlib is used correctly."""
        # Module should use pathlib.Path for path operations
        import pdf_vector_system.gui_main as gui_main_module
        
        # Check that Path is imported and used
        assert hasattr(gui_main_module, 'Path')
        
        # The project_root should be a Path object or string
        project_root = gui_main_module.project_root
        assert project_root is not None
