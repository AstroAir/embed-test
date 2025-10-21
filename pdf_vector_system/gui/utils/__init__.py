"""
GUI utilities package for PDF Vector System.

This package contains utility functions and helper classes for GUI operations.

Utilities:
- qt_utils: Qt-specific utility functions
- styling: Application styling and theming
- icons: Icon management and resources
- validators: Input validation for GUI components
- threading: Background task management
"""

from pdf_vector_system.gui.utils.icons import IconType, get_icon
from pdf_vector_system.gui.utils.qt_utils import (
    center_window,
    set_window_icon,
    show_error_message,
    show_info_message,
    show_question_dialog,
    show_warning_message,
)
from pdf_vector_system.gui.utils.styling import (
    apply_card_title_style,
    apply_dark_theme,
    apply_help_button_style,
    apply_light_theme,
    apply_section_header_style,
    configure_card_widget,
    configure_info_badge,
    configure_info_bar,
    configure_navigation_bar,
    configure_segmented_widget,
    configure_setting_card,
    configure_teaching_tip,
    create_styled_card_widget,
    create_styled_info_bar,
    create_styled_setting_card,
    get_app_style,
    get_fluent_icon_for_action,
    get_status_color,
)
from pdf_vector_system.gui.utils.threading import TaskRunner, WorkerThread
from pdf_vector_system.gui.utils.validators import (
    ConfigValidator,
    NumberValidator,
    PathValidator,
)

__all__ = [
    "ConfigValidator",
    "IconType",
    "NumberValidator",
    "PathValidator",
    "TaskRunner",
    "WorkerThread",
    "apply_card_title_style",
    "apply_dark_theme",
    "apply_help_button_style",
    "apply_light_theme",
    "apply_section_header_style",
    "center_window",
    "configure_card_widget",
    "configure_info_badge",
    "configure_info_bar",
    "configure_navigation_bar",
    "configure_segmented_widget",
    "configure_setting_card",
    "configure_teaching_tip",
    "create_styled_card_widget",
    "create_styled_info_bar",
    "create_styled_setting_card",
    "get_app_style",
    "get_fluent_icon_for_action",
    "get_icon",
    "get_status_color",
    "set_window_icon",
    "show_error_message",
    "show_info_message",
    "show_question_dialog",
    "show_warning_message",
]
