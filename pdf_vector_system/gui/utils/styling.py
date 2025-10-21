"""
Styling and theming for PDF Vector System GUI.

This module provides application styling, theme management, and utility functions
for configuring QFluentWidgets components consistently.
"""

from typing import Any, Optional, Union

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon
from qfluentwidgets import (
    CardWidget,
    FluentIcon,
    InfoBar,
    InfoBarPosition,
    InfoLevel,
    SettingCard,
    TeachingTipTailPosition,
)


def get_app_style() -> Optional[str]:
    """
    Get the application stylesheet.

    Note: FluentWindow and QFluentWidgets have their own theming system.
    This stylesheet provides minimal additional styling that doesn't conflict
    with QFluentWidgets' built-in themes.

    Returns:
        CSS stylesheet string or None
    """
    return """
    /* FluentWindow styling - minimal to avoid conflicts with QFluentWidgets theming */
    QWidget {
        background-color: transparent;
    }

    /* Note: QTabWidget styles removed as FluentWindow uses its own navigation system */
    /* FluentWindow uses addSubInterface() instead of traditional tabs */

    /* Note: QPushButton styles removed - QFluentWidgets PushButton has its own styling */
    /* QFluentWidgets components (PushButton, LineEdit, etc.) use their own theming */

    /* Keep only native Qt widgets that aren't replaced by QFluentWidgets */

    /* Note: QGroupBox styles removed - QFluentWidgets uses CardWidget instead */
    /* Note: QProgressBar styles removed - QFluentWidgets ProgressBar has its own styling */
    /* Note: QStatusBar and QMenuBar styles removed - FluentWindow doesn't use these */

    /* Keep only styles for native Qt widgets that may still be used */
    QScrollBar:vertical {
        background-color: #f0f0f0;
        width: 12px;
        border-radius: 6px;
    }

    QScrollBar::handle:vertical {
        background-color: #c0c0c0;
        border-radius: 6px;
        min-height: 20px;
    }

    QScrollBar::handle:vertical:hover {
        background-color: #a0a0a0;
    }

    /* Note: QTableWidget styles removed - QFluentWidgets TableWidget has its own styling */
    """


def apply_dark_theme() -> Optional[str]:
    """
    Get dark theme stylesheet.

    Returns:
        Dark theme CSS stylesheet string
    """
    # This would contain a dark theme implementation
    # For now, return None to use default theme
    return None


def apply_light_theme() -> Optional[str]:
    """
    Get light theme stylesheet.

    Returns:
        Light theme CSS stylesheet string
    """
    return get_app_style()


# QFluentWidgets Component Configuration Utilities


def configure_info_bar(
    title: str,
    content: str = "",
    level: InfoLevel = InfoLevel.INFOAMTION,
    position: InfoBarPosition = InfoBarPosition.TOP,
    duration: int = 3000,
    closable: bool = True,
) -> dict[str, Any]:
    """
    Create a standardized configuration for InfoBar components.

    Args:
        title: Info bar title
        content: Info bar content
        level: Info level (INFO, SUCCESS, WARNING, ERROR)
        position: Position of the info bar
        duration: Auto-hide duration in milliseconds (0 = no auto-hide)
        closable: Whether the info bar can be closed

    Returns:
        Configuration dictionary for InfoBar
    """
    icon_map = {
        InfoLevel.INFOAMTION: FluentIcon.INFO,
        InfoLevel.SUCCESS: FluentIcon.ACCEPT,
        InfoLevel.WARNING: FluentIcon.INFO,
        InfoLevel.ERROR: FluentIcon.CANCEL,
    }

    return {
        "title": title,
        "content": content,
        "icon": icon_map.get(level, FluentIcon.INFO),
        "position": position,
        "duration": duration,
        "isClosable": closable,
        "orient": Qt.Horizontal,
    }


def configure_setting_card(
    icon: Union[FluentIcon, QIcon],
    title: str,
    content: str,
    widget: Optional[Any] = None,
) -> dict[str, Any]:
    """
    Create a standardized configuration for SettingCard components.

    Args:
        icon: Card icon
        title: Card title
        content: Card description
        widget: Optional widget to add to the card

    Returns:
        Configuration dictionary for SettingCard
    """
    return {
        "icon": icon.icon() if hasattr(icon, "icon") else icon,
        "title": title,
        "content": content,
        "widget": widget,
    }


def configure_teaching_tip(
    title: str,
    content: str,
    target_position: TeachingTipTailPosition = TeachingTipTailPosition.BOTTOM,
    icon: FluentIcon = FluentIcon.HELP,
    closable: bool = True,
) -> dict[str, Any]:
    """
    Create a standardized configuration for TeachingTip components.

    Args:
        title: Tip title
        content: Tip content
        target_position: Position of the tip tail
        icon: Tip icon
        closable: Whether the tip can be closed

    Returns:
        Configuration dictionary for TeachingTip
    """
    return {
        "title": title,
        "content": content,
        "icon": icon,
        "tailPosition": target_position,
        "isClosable": closable,
    }


def configure_info_badge(
    text: str = "", level: InfoLevel = InfoLevel.INFOAMTION, position: str = "top-right"
) -> dict[str, Any]:
    """
    Create a standardized configuration for InfoBadge components.

    Args:
        text: Badge text
        level: Info level
        position: Badge position relative to parent

    Returns:
        Configuration dictionary for InfoBadge
    """
    return {"text": text, "level": level, "position": position}


def configure_segmented_widget(
    items: list, current_index: int = 0, orientation: Qt.Orientation = Qt.Horizontal
) -> dict[str, Any]:
    """
    Create a standardized configuration for SegmentedWidget components.

    Args:
        items: List of items (strings or tuples of (text, icon))
        current_index: Initially selected index
        orientation: Widget orientation

    Returns:
        Configuration dictionary for SegmentedWidget
    """
    return {"items": items, "currentIndex": current_index, "orientation": orientation}


def configure_card_widget(
    title: str = "",
    content_margins: tuple = (16, 16, 16, 16),
    spacing: int = 12,
    elevated: bool = False,
) -> dict[str, Any]:
    """
    Create a standardized configuration for CardWidget components.

    Args:
        title: Card title
        content_margins: Content margins (left, top, right, bottom)
        spacing: Layout spacing
        elevated: Whether to use elevated card style

    Returns:
        Configuration dictionary for CardWidget
    """
    return {
        "title": title,
        "contentMargins": content_margins,
        "spacing": spacing,
        "elevated": elevated,
    }


def configure_navigation_bar(
    items: list, orientation: Qt.Orientation = Qt.Horizontal, current_index: int = 0
) -> dict[str, Any]:
    """
    Create a standardized configuration for NavigationBar components.

    Args:
        items: List of navigation items (text, icon, route)
        orientation: Bar orientation
        current_index: Initially selected index

    Returns:
        Configuration dictionary for NavigationBar
    """
    return {"items": items, "orientation": orientation, "currentIndex": current_index}


# Styling utility functions


def apply_card_title_style(title_widget: Any) -> None:
    """
    Apply consistent styling to card title widgets.

    Args:
        title_widget: Widget to style (usually BodyLabel)
    """
    title_widget.setStyleSheet(
        "font-weight: bold; font-size: 14px; margin-bottom: 10px; color: #333333;"
    )


def apply_section_header_style(header_widget: Any) -> None:
    """
    Apply consistent styling to section header widgets.

    Args:
        header_widget: Widget to style (usually SubtitleLabel)
    """
    header_widget.setStyleSheet(
        "font-weight: 600; font-size: 16px; margin: 16px 0 8px 0; color: #2c2c2c;"
    )


def apply_help_button_style(button_widget: Any) -> None:
    """
    Apply consistent styling to help button widgets.

    Args:
        button_widget: Button widget to style
    """
    button_widget.setFixedSize(24, 24)
    button_widget.setStyleSheet(
        "border-radius: 12px; background-color: #f0f0f0; border: 1px solid #d0d0d0;"
    )


def get_status_color(level: InfoLevel) -> QColor:
    """
    Get the appropriate color for a status level.

    Args:
        level: Info level

    Returns:
        QColor for the level
    """
    color_map = {
        InfoLevel.INFOAMTION: QColor("#0078d4"),  # Blue
        InfoLevel.SUCCESS: QColor("#107c10"),  # Green
        InfoLevel.WARNING: QColor("#ff8c00"),  # Orange
        InfoLevel.ERROR: QColor("#d13438"),  # Red
    }
    return color_map.get(level, QColor("#0078d4"))


def get_fluent_icon_for_action(action: str) -> FluentIcon:
    """
    Get the appropriate FluentIcon for common actions.

    Args:
        action: Action name

    Returns:
        Appropriate FluentIcon
    """
    icon_map = {
        "add": FluentIcon.ADD,
        "delete": FluentIcon.DELETE,
        "edit": FluentIcon.EDIT,
        "save": FluentIcon.SAVE,
        "open": FluentIcon.FOLDER,
        "close": FluentIcon.CLOSE,
        "search": FluentIcon.SEARCH,
        "settings": FluentIcon.SETTING,
        "help": FluentIcon.HELP,
        "info": FluentIcon.INFO,
        "warning": FluentIcon.INFO,
        "error": FluentIcon.CANCEL,
        "success": FluentIcon.ACCEPT,
        "play": FluentIcon.PLAY,
        "pause": FluentIcon.PAUSE,
        "stop": FluentIcon.CANCEL,  # Use CANCEL as STOP alternative
        "refresh": FluentIcon.SYNC,
        "home": FluentIcon.HOME,
        "document": FluentIcon.DOCUMENT,
        "mail": FluentIcon.MAIL,
        "link": FluentIcon.LINK,
        "tag": FluentIcon.TAG,
        "code": FluentIcon.CODE,
        "heart": FluentIcon.HEART,
    }
    return icon_map.get(action.lower(), FluentIcon.INFO)


# Component factory functions


def create_styled_info_bar(
    title: str,
    content: str = "",
    level: InfoLevel = InfoLevel.INFOAMTION,
    parent: Optional[Any] = None,
    **kwargs,
) -> InfoBar:
    """
    Create a consistently styled InfoBar.

    Args:
        title: Info bar title
        content: Info bar content
        level: Info level
        parent: Parent widget
        **kwargs: Additional InfoBar arguments

    Returns:
        Configured InfoBar instance
    """
    config = configure_info_bar(title, content, level, **kwargs)
    return InfoBar(parent=parent, **config)


def create_styled_setting_card(
    icon: Union[FluentIcon, QIcon],
    title: str,
    content: str,
    parent: Optional[Any] = None,
    **kwargs,
) -> SettingCard:
    """
    Create a consistently styled SettingCard.

    Args:
        icon: Card icon
        title: Card title
        content: Card description
        parent: Parent widget
        **kwargs: Additional SettingCard arguments

    Returns:
        Configured SettingCard instance
    """
    config = configure_setting_card(icon, title, content, **kwargs)
    return SettingCard(parent=parent, **config)


def create_styled_card_widget(
    title: str = "", parent: Optional[Any] = None, **kwargs
) -> CardWidget:
    """
    Create a consistently styled CardWidget.

    Args:
        title: Card title
        parent: Parent widget
        **kwargs: Additional CardWidget arguments

    Returns:
        Configured CardWidget instance
    """
    config = configure_card_widget(title, **kwargs)
    card = CardWidget(parent)

    if title:
        # Add title styling
        from qfluentwidgets import BodyLabel, VBoxLayout

        layout = VBoxLayout(card)
        title_label = BodyLabel(title)
        apply_card_title_style(title_label)
        layout.addWidget(title_label)
        layout.setContentsMargins(*config["contentMargins"])
        layout.setSpacing(config["spacing"])

    return card
