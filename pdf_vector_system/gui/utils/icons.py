"""
Icon management for PDF Vector System GUI.

This module provides icon management and resource handling for the GUI application.
Enhanced to use QFluentWidgets FluentIcon for better visual consistency and theming support.
"""

from enum import Enum
from typing import Optional

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QStyle
from qfluentwidgets import FluentIcon


class IconType(Enum):
    """Enumeration of available icon types."""

    APP = "app"
    PROCESS = "process"
    SEARCH = "search"
    DOCUMENT = "document"
    CONFIG = "config"
    STATUS = "status"
    LOG = "log"
    FILE = "file"
    FOLDER = "folder"
    ADD = "add"
    REMOVE = "remove"
    REFRESH = "refresh"
    SAVE = "save"
    OPEN = "open"
    CLOSE = "close"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"


def get_icon(icon_type: IconType) -> Optional[QIcon]:
    """
    Get an icon for the specified type using QFluentWidgets FluentIcon.

    Args:
        icon_type: Type of icon to retrieve

    Returns:
        QIcon object or None if not found
    """
    # Map icon types to FluentIcon for better visual consistency and theming
    fluent_icon_map = {
        IconType.APP: FluentIcon.APPLICATION,
        IconType.PROCESS: FluentIcon.DOCUMENT,
        IconType.SEARCH: FluentIcon.SEARCH,
        IconType.DOCUMENT: FluentIcon.DOCUMENT,
        IconType.CONFIG: FluentIcon.SETTING,
        IconType.STATUS: FluentIcon.HEART,
        IconType.LOG: FluentIcon.HISTORY,
        IconType.FILE: FluentIcon.DOCUMENT,
        IconType.FOLDER: FluentIcon.FOLDER,
        IconType.ADD: FluentIcon.ADD,
        IconType.REMOVE: FluentIcon.REMOVE,
        IconType.REFRESH: FluentIcon.SYNC,
        IconType.SAVE: FluentIcon.SAVE,
        IconType.OPEN: FluentIcon.FOLDER,
        IconType.CLOSE: FluentIcon.CLOSE,
        IconType.ERROR: FluentIcon.CANCEL,
        IconType.WARNING: FluentIcon.INFO,
        IconType.INFO: FluentIcon.INFO,
        IconType.SUCCESS: FluentIcon.COMPLETED,
    }

    fluent_icon = fluent_icon_map.get(icon_type)
    if fluent_icon:
        return fluent_icon.icon()

    # Fallback to Qt standard icons if FluentIcon is not available
    return _get_fallback_icon(icon_type)


def _get_fallback_icon(icon_type: IconType) -> Optional[QIcon]:
    """
    Get fallback Qt standard icon if FluentIcon is not available.

    Args:
        icon_type: Type of icon to retrieve

    Returns:
        QIcon object or None if not found
    """
    app = QApplication.instance()
    if not app:
        return None

    style = app.style()

    # Map icon types to Qt standard icons as fallback
    qt_icon_map = {
        IconType.APP: QStyle.SP_ComputerIcon,
        IconType.PROCESS: QStyle.SP_FileDialogDetailedView,
        IconType.SEARCH: QStyle.SP_FileDialogListView,
        IconType.DOCUMENT: QStyle.SP_FileIcon,
        IconType.CONFIG: QStyle.SP_ComputerIcon,
        IconType.STATUS: QStyle.SP_ComputerIcon,
        IconType.LOG: QStyle.SP_FileDialogDetailedView,
        IconType.FILE: QStyle.SP_FileIcon,
        IconType.FOLDER: QStyle.SP_DirIcon,
        IconType.ADD: QStyle.SP_DialogOkButton,
        IconType.REMOVE: QStyle.SP_DialogCancelButton,
        IconType.REFRESH: QStyle.SP_BrowserReload,
        IconType.SAVE: QStyle.SP_DialogSaveButton,
        IconType.OPEN: QStyle.SP_DialogOpenButton,
        IconType.CLOSE: QStyle.SP_DialogCloseButton,
        IconType.ERROR: QStyle.SP_MessageBoxCritical,
        IconType.WARNING: QStyle.SP_MessageBoxWarning,
        IconType.INFO: QStyle.SP_MessageBoxInformation,
        IconType.SUCCESS: QStyle.SP_DialogApplyButton,
    }

    qt_icon = qt_icon_map.get(icon_type)
    if qt_icon:
        return style.standardIcon(qt_icon)

    return None


def get_file_icon(file_path: str) -> Optional[QIcon]:
    """
    Get an icon for a file based on its extension.

    Args:
        file_path: Path to the file

    Returns:
        QIcon object or None if not found
    """
    if file_path.lower().endswith(".pdf"):
        return get_icon(IconType.DOCUMENT)
    return get_icon(IconType.FILE)


def get_fluent_icon(fluent_icon: FluentIcon) -> QIcon:
    """
    Get a QIcon from a FluentIcon directly.

    Args:
        fluent_icon: FluentIcon to convert

    Returns:
        QIcon object
    """
    return fluent_icon.icon()
