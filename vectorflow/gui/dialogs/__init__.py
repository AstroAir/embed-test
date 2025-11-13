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

from vectorflow.gui.dialogs.about_dialog import AboutDialog
from vectorflow.gui.dialogs.confirm_dialog import ConfirmDialog
from vectorflow.gui.dialogs.error_dialog import ErrorDialog
from vectorflow.gui.dialogs.progress_dialog import ProgressDialog
from vectorflow.gui.dialogs.settings_dialog import SettingsDialog

__all__ = [
    "AboutDialog",
    "ConfirmDialog",
    "ErrorDialog",
    "ProgressDialog",
    "SettingsDialog",
]
