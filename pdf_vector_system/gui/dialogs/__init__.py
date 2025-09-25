"""
GUI dialogs package for PDF Vector System.

This package contains modal dialogs and popup windows for user interaction.

Dialogs:
- AboutDialog: Application information and credits
- SettingsDialog: Advanced configuration settings
- ConfirmDialog: User confirmation dialogs
- ErrorDialog: Error display and reporting
- ProgressDialog: Long-running operation progress
"""

from .about_dialog import AboutDialog
from .settings_dialog import SettingsDialog
from .confirm_dialog import ConfirmDialog
from .error_dialog import ErrorDialog
from .progress_dialog import ProgressDialog

__all__ = [
    "AboutDialog",
    "SettingsDialog", 
    "ConfirmDialog",
    "ErrorDialog",
    "ProgressDialog"
]
