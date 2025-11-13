"""Tests for GUI application class."""

import pytest
from PySide6.QtWidgets import QApplication

from vectorflow.core.config.settings import Config
from vectorflow.gui.app import PDFVectorGUIApp


@pytest.fixture
def qapp(qtbot):
    """Provide QApplication instance."""
    return QApplication.instance() or QApplication([])


class TestPDFVectorGUIApp:
    """Test PDFVectorGUIApp class."""

    @pytest.mark.skip(reason="TODO: Implement app initialization tests")
    def test_app_initialization(self, qapp):
        """Test app initialization."""
        pass

    @pytest.mark.skip(reason="TODO: Implement theme tests")
    def test_app_theme_initialization(self, qapp):
        """Test theme initialization."""
        pass

    @pytest.mark.skip(reason="TODO: Implement config tests")
    def test_app_with_custom_config(self, qapp):
        """Test app with custom configuration."""
        pass
