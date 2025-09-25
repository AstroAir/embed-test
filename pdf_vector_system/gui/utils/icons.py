"""
Icon management for PDF Vector System GUI.

This module provides icon management and resource handling for the GUI application.
"""

from enum import Enum
from typing import Optional
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QStyle, QApplication


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
    Get an icon for the specified type.
    
    Args:
        icon_type: Type of icon to retrieve
        
    Returns:
        QIcon object or None if not found
    """
    # For now, use standard Qt icons
    # In a production app, you would load custom icons from resources
    
    app = QApplication.instance()
    if not app:
        return None
        
    style = app.style()
    
    # Map icon types to Qt standard icons
    icon_map = {
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
    
    qt_icon = icon_map.get(icon_type)
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
    if file_path.lower().endswith('.pdf'):
        return get_icon(IconType.DOCUMENT)
    else:
        return get_icon(IconType.FILE)
