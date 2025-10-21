"""
Base widget classes for PDF Vector System GUI.

This module contains base widget classes that provide common functionality
for all GUI widgets with enhanced QFluentWidgets support.
"""

from typing import Optional

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    FluentIcon,
    InfoBadge,
    InfoBar,
    InfoBarPosition,
    InfoLevel,
    SettingCardGroup,
    SmoothScrollArea,
    StateToolTip,
    TeachingTip,
    VBoxLayout,
)

from pdf_vector_system.config.settings import Config


class EnhancedBaseWidget(QWidget):
    """Enhanced base widget class with modern QFluentWidgets support."""

    # Signals
    status_changed: Signal = Signal(str)
    error_occurred: Signal = Signal(str)
    info_requested: Signal = Signal(str)

    def __init__(
        self, config: Optional[Config] = None, parent: Optional[QWidget] = None
    ):
        """
        Initialize the enhanced base widget.

        Args:
            config: Configuration object
            parent: Parent widget
        """
        super().__init__(parent)

        self.config = config or Config()

        # Enhanced UI components
        self._info_bars: dict[str, InfoBar] = {}
        self._state_tooltip: Optional[StateToolTip] = None
        self._teaching_tips: dict[str, TeachingTip] = {}
        self._info_badges: dict[str, InfoBadge] = {}

        # Auto-hide timer for info bars
        self._info_bar_timer = QTimer()
        self._info_bar_timer.setSingleShot(True)
        self._info_bar_timer.timeout.connect(self._hide_auto_info_bars)

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface. Override in subclasses."""
        layout = VBoxLayout(self)
        label = BodyLabel("Enhanced Base Widget - Override _setup_ui() in subclass")
        layout.addWidget(label)

    def on_tab_activated(self) -> None:
        """Called when this widget's tab is activated. Override in subclasses."""

    # Enhanced status and feedback methods
    def emit_status(self, message: str) -> None:
        """
        Emit a status message.

        Args:
            message: Status message
        """
        self.status_changed.emit(message)

    def emit_error(self, message: str) -> None:
        """
        Emit an error message.

        Args:
            message: Error message
        """
        self.error_occurred.emit(message)

    def show_info_bar(
        self,
        title: str,
        content: str = "",
        level: InfoLevel = InfoLevel.INFOAMTION,
        position: InfoBarPosition = InfoBarPosition.TOP,
        duration: int = 3000,
        closable: bool = True,
        key: Optional[str] = None,
    ) -> InfoBar:
        """
        Show an info bar with the specified message.

        Args:
            title: Info bar title
            content: Info bar content
            level: Info level (INFO, SUCCESS, WARNING, ERROR)
            position: Position of the info bar
            duration: Auto-hide duration in milliseconds (0 = no auto-hide)
            closable: Whether the info bar can be closed
            key: Unique key for the info bar (for management)

        Returns:
            The created InfoBar instance
        """
        # Close existing info bar with same key
        if key and key in self._info_bars:
            self._info_bars[key].close()

        info_bar = InfoBar(
            icon=self._get_info_icon(level),
            title=title,
            content=content,
            orient=Qt.Horizontal,
            isClosable=closable,
            position=position,
            duration=duration,
            parent=self,
        )

        if key:
            self._info_bars[key] = info_bar

        info_bar.show()

        # Auto-hide if duration is specified
        if duration > 0:
            QTimer.singleShot(duration, info_bar.close)

        return info_bar

    def show_success_info(
        self, title: str, content: str = "", duration: int = 3000
    ) -> InfoBar:
        """Show a success info bar."""
        return self.show_info_bar(title, content, InfoLevel.SUCCESS, duration=duration)

    def show_warning_info(
        self, title: str, content: str = "", duration: int = 5000
    ) -> InfoBar:
        """Show a warning info bar."""
        return self.show_info_bar(title, content, InfoLevel.WARNING, duration=duration)

    def show_error_info(
        self, title: str, content: str = "", duration: int = 0
    ) -> InfoBar:
        """Show an error info bar (no auto-hide by default)."""
        return self.show_info_bar(title, content, InfoLevel.ERROR, duration=duration)

    def show_state_tooltip(
        self, title: str, content: str = "", target: Optional[QWidget] = None
    ) -> StateToolTip:
        """
        Show a state tooltip for loading or processing states.

        Args:
            title: Tooltip title
            content: Tooltip content
            target: Target widget (defaults to self)

        Returns:
            The StateToolTip instance
        """
        if self._state_tooltip:
            self._state_tooltip.close()

        self._state_tooltip = StateToolTip(
            title=title, content=content, parent=target or self
        )

        self._state_tooltip.show()
        return self._state_tooltip

    def hide_state_tooltip(self) -> None:
        """Hide the current state tooltip."""
        if self._state_tooltip:
            self._state_tooltip.close()
            self._state_tooltip = None

    def show_teaching_tip(
        self, title: str, content: str, target: QWidget, key: Optional[str] = None
    ) -> TeachingTip:
        """
        Show a teaching tip for contextual help.

        Args:
            title: Tip title
            content: Tip content
            target: Target widget to point to
            key: Unique key for the tip

        Returns:
            The TeachingTip instance
        """
        # Close existing tip with same key
        if key and key in self._teaching_tips:
            self._teaching_tips[key].close()

        tip = TeachingTip.create(
            target=target,
            icon=FluentIcon.HELP,
            title=title,
            content=content,
            isClosable=True,
            tailPosition=TeachingTip.make_tail_position(target),
            parent=self,
        )

        if key:
            self._teaching_tips[key] = tip

        tip.show()
        return tip

    def create_info_badge(
        self,
        text: str = "",
        level: InfoLevel = InfoLevel.INFOAMTION,
        key: Optional[str] = None,
    ) -> InfoBadge:
        """
        Create an info badge for status indicators.

        Args:
            text: Badge text
            level: Info level
            key: Unique key for the badge

        Returns:
            The InfoBadge instance
        """
        badge = InfoBadge.info(text, self)

        # Set badge style based on level
        if level == InfoLevel.SUCCESS:
            badge = InfoBadge.success(text, self)
        elif level == InfoLevel.WARNING:
            badge = InfoBadge.warning(text, self)
        elif level == InfoLevel.ERROR:
            badge = InfoBadge.error(text, self)

        if key:
            self._info_badges[key] = badge

        return badge

    def create_card_group(self, title: str) -> CardWidget:
        """
        Create a card widget for grouping related controls.

        Args:
            title: Card title

        Returns:
            The CardWidget instance
        """
        card = CardWidget(self)
        layout = VBoxLayout(card)

        if title:
            title_label = BodyLabel(title)
            title_label.setStyleSheet(
                "font-weight: bold; font-size: 14px; margin-bottom: 10px;"
            )
            layout.addWidget(title_label)

        return card

    def create_setting_group(self, title: str) -> SettingCardGroup:
        """
        Create a setting card group for configuration options.

        Args:
            title: Group title

        Returns:
            The SettingCardGroup instance
        """
        return SettingCardGroup(title, self)

    def create_smooth_scroll_area(self) -> SmoothScrollArea:
        """
        Create a smooth scroll area for better scrolling experience.

        Returns:
            The SmoothScrollArea instance
        """
        scroll_area = SmoothScrollArea(self)
        scroll_area.setWidgetResizable(True)
        return scroll_area

    def _get_info_icon(self, level: InfoLevel) -> QIcon:
        """Get the appropriate icon for the info level."""
        icon_map = {
            InfoLevel.INFOAMTION: FluentIcon.INFO,
            InfoLevel.SUCCESS: FluentIcon.ACCEPT,
            InfoLevel.WARNING: FluentIcon.INFO,
            InfoLevel.ERROR: FluentIcon.CANCEL,
        }
        return icon_map.get(level, FluentIcon.INFO).icon()

    def _hide_auto_info_bars(self) -> None:
        """Hide info bars that should auto-hide."""
        for info_bar in list(self._info_bars.values()):
            if hasattr(info_bar, "_auto_hide") and info_bar._auto_hide:
                info_bar.close()

    def cleanup(self) -> None:
        """Clean up resources when widget is destroyed."""
        # Close all info bars
        for info_bar in self._info_bars.values():
            info_bar.close()
        self._info_bars.clear()

        # Close state tooltip
        if self._state_tooltip:
            self._state_tooltip.close()

        # Close teaching tips
        for tip in self._teaching_tips.values():
            tip.close()
        self._teaching_tips.clear()

        # Clean up info badges
        self._info_badges.clear()


# Backward compatibility alias
class BaseWidget(EnhancedBaseWidget):
    """Backward compatibility alias for EnhancedBaseWidget."""
