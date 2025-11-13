"""
Adaptive UI utilities for responsive and flexible layouts.

This module provides utilities for creating adaptive, responsive UI components
that adjust to different window sizes and screen resolutions.
"""

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QSizePolicy, QSpacerItem, QWidget


class AdaptiveSpacing:
    """Responsive spacing values that scale with UI."""

    # Base spacing values (in pixels)
    EXTRA_SMALL = 4
    SMALL = 8
    MEDIUM = 12
    LARGE = 16
    EXTRA_LARGE = 24
    HUGE = 32

    @staticmethod
    def get_spacing_for_width(width: int) -> int:
        """
        Get appropriate spacing based on container width.

        Args:
            width: Container width in pixels

        Returns:
            Appropriate spacing value
        """
        if width < 600:
            return AdaptiveSpacing.SMALL
        if width < 900:
            return AdaptiveSpacing.MEDIUM
        if width < 1200:
            return AdaptiveSpacing.LARGE
        return AdaptiveSpacing.EXTRA_LARGE


class AdaptiveMargins:
    """Responsive margin values for containers."""

    # Base margin values
    COMPACT = (8, 8, 8, 8)
    NORMAL = (16, 16, 16, 16)
    COMFORTABLE = (24, 24, 24, 24)
    SPACIOUS = (32, 32, 32, 32)

    @staticmethod
    def get_margins_for_width(width: int) -> tuple[int, int, int, int]:
        """
        Get appropriate margins based on container width.

        Args:
            width: Container width in pixels

        Returns:
            Tuple of (left, top, right, bottom) margins
        """
        if width < 600:
            return AdaptiveMargins.COMPACT
        if width < 900:
            return AdaptiveMargins.NORMAL
        if width < 1200:
            return AdaptiveMargins.COMFORTABLE
        return AdaptiveMargins.SPACIOUS


class AdaptiveSizePolicy:
    """Pre-configured size policies for common use cases."""

    @staticmethod
    def expanding_both() -> tuple[QSizePolicy.Policy, QSizePolicy.Policy]:
        """Size policy that expands in both directions."""
        return (QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @staticmethod
    def expanding_horizontal() -> tuple[QSizePolicy.Policy, QSizePolicy.Policy]:
        """Size policy that expands horizontally, minimum vertically."""
        return (QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    @staticmethod
    def expanding_vertical() -> tuple[QSizePolicy.Policy, QSizePolicy.Policy]:
        """Size policy that expands vertically, minimum horizontally."""
        return (QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

    @staticmethod
    def preferred_both() -> tuple[QSizePolicy.Policy, QSizePolicy.Policy]:
        """Size policy with preferred size in both directions."""
        return (QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

    @staticmethod
    def apply_to_widget(
        widget: QWidget,
        horizontal: QSizePolicy.Policy,
        vertical: QSizePolicy.Policy,
    ) -> None:
        """
        Apply size policy to a widget.

        Args:
            widget: Widget to apply policy to
            horizontal: Horizontal size policy
            vertical: Vertical size policy
        """
        policy = QSizePolicy(horizontal, vertical)
        widget.setSizePolicy(policy)


class AdaptiveLayout:
    """Utilities for creating adaptive layouts."""

    @staticmethod
    def create_horizontal_spacer(expanding: bool = True) -> QSpacerItem:
        """
        Create a horizontal spacer.

        Args:
            expanding: Whether spacer should expand

        Returns:
            QSpacerItem for horizontal spacing
        """
        policy = (
            QSizePolicy.Policy.Expanding if expanding else QSizePolicy.Policy.Minimum
        )
        return QSpacerItem(0, 0, policy, QSizePolicy.Policy.Minimum)

    @staticmethod
    def create_vertical_spacer(expanding: bool = True) -> QSpacerItem:
        """
        Create a vertical spacer.

        Args:
            expanding: Whether spacer should expand

        Returns:
            QSpacerItem for vertical spacing
        """
        policy = (
            QSizePolicy.Policy.Expanding if expanding else QSizePolicy.Policy.Minimum
        )
        return QSpacerItem(0, 0, QSizePolicy.Policy.Minimum, policy)

    @staticmethod
    def create_fixed_spacer(width: int = 0, height: int = 0) -> QSpacerItem:
        """
        Create a fixed-size spacer.

        Args:
            width: Fixed width
            height: Fixed height

        Returns:
            QSpacerItem with fixed size
        """
        return QSpacerItem(
            width, height, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )


class AdaptiveWindow:
    """Utilities for adaptive window sizing."""

    # Minimum window sizes for different screen categories
    MIN_SMALL = QSize(800, 600)
    MIN_MEDIUM = QSize(1000, 700)
    MIN_LARGE = QSize(1200, 800)
    MIN_XLARGE = QSize(1400, 900)

    # Recommended window sizes
    RECOMMENDED_SMALL = QSize(900, 650)
    RECOMMENDED_MEDIUM = QSize(1100, 750)
    RECOMMENDED_LARGE = QSize(1300, 850)
    RECOMMENDED_XLARGE = QSize(1600, 1000)

    @staticmethod
    def get_adaptive_window_size(screen_width: int, screen_height: int) -> QSize:
        """
        Get recommended window size based on screen dimensions.

        Args:
            screen_width: Available screen width
            screen_height: Available screen height

        Returns:
            Recommended window size
        """
        # Use 80% of screen size, but respect minimums and maximums
        target_width = int(screen_width * 0.8)
        target_height = int(screen_height * 0.8)

        # Clamp to reasonable ranges
        target_width = max(800, min(target_width, 1920))
        target_height = max(600, min(target_height, 1200))

        return QSize(target_width, target_height)

    @staticmethod
    def get_minimum_size_for_resolution(screen_width: int, screen_height: int) -> QSize:
        """
        Get appropriate minimum window size based on screen resolution.

        Args:
            screen_width: Screen width
            screen_height: Screen height

        Returns:
            Minimum window size
        """
        if screen_width >= 1920 and screen_height >= 1080:
            return AdaptiveWindow.MIN_XLARGE
        if screen_width >= 1440:
            return AdaptiveWindow.MIN_LARGE
        if screen_width >= 1024:
            return AdaptiveWindow.MIN_MEDIUM
        return AdaptiveWindow.MIN_SMALL


class ResponsiveGrid:
    """Utilities for creating responsive grid layouts."""

    @staticmethod
    def get_column_count(width: int, min_column_width: int = 250) -> int:
        """
        Calculate number of columns based on container width.

        Args:
            width: Container width
            min_column_width: Minimum width per column

        Returns:
            Number of columns that fit
        """
        # Calculate how many columns fit, minimum 1, maximum 4
        columns = max(1, width // min_column_width)
        return min(columns, 4)

    @staticmethod
    def get_item_size(
        container_width: int, column_count: int, spacing: int = 16
    ) -> int:
        """
        Calculate item width for grid layout.

        Args:
            container_width: Total container width
            column_count: Number of columns
            spacing: Spacing between items

        Returns:
            Width per item
        """
        total_spacing = spacing * (column_count - 1)
        available_width = container_width - total_spacing
        return available_width // column_count


def make_widget_adaptive(
    widget: QWidget, expand_horizontal: bool = True, expand_vertical: bool = True
) -> None:
    """
    Configure a widget to be adaptive with appropriate size policies.

    Args:
        widget: Widget to make adaptive
        expand_horizontal: Whether widget should expand horizontally
        expand_vertical: Whether widget should expand vertically
    """
    h_policy = (
        QSizePolicy.Policy.Expanding
        if expand_horizontal
        else QSizePolicy.Policy.Preferred
    )
    v_policy = (
        QSizePolicy.Policy.Expanding
        if expand_vertical
        else QSizePolicy.Policy.Preferred
    )

    policy = QSizePolicy(h_policy, v_policy)
    widget.setSizePolicy(policy)


def apply_adaptive_margins(widget: QWidget, current_width: int) -> None:
    """
    Apply adaptive margins to a widget based on its current width.

    Args:
        widget: Widget to apply margins to
        current_width: Current width of widget's container
    """
    margins = AdaptiveMargins.get_margins_for_width(current_width)
    if hasattr(widget, "layout") and widget.layout():
        widget.layout().setContentsMargins(*margins)
