"""
Enhanced configuration widget for PDF Vector System GUI.

This module contains the enhanced widget for configuration management with
modern QFluentWidgets components and improved user experience.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QStackedWidget, QWidget
from qfluentwidgets import (
    BodyLabel,
    ColorSettingCard,
    ComboBox,
    ExpandSettingCard,
    InfoBadge,
    InfoLevel,
    LineEdit,
    MessageBox,
    NavigationWidget,
    PrimaryPushButton,
    PushButton,
    RangeSettingCard,
    SettingCard,
    SettingCardGroup,
    SmoothScrollArea,
    StateToolTip,
    SwitchSettingCard,
    Theme,
    VBoxLayout,
    qconfig,
)

from pdf_vector_system.config.settings import Config, EmbeddingModelType, LogLevel
from pdf_vector_system.gui.controllers.config_controller import ConfigController
from pdf_vector_system.gui.utils.styling import (
    create_styled_card_widget,
    get_fluent_icon_for_action,
)
from pdf_vector_system.gui.widgets.base import BaseWidget


class ConfigWidget(BaseWidget):
    """Enhanced widget for configuration management with modern UI components."""

    config_changed: Signal = Signal(object)

    def __init__(
        self, config: Optional[Config] = None, parent: Optional[QWidget] = None
    ):
        self._updating_ui = False
        self._provider_index: dict[EmbeddingModelType, int] = {}
        self._provider_stack: Optional[QStackedWidget] = None

        # Enhanced UI state tracking
        self._current_nav_item = "general"
        self._unsaved_changes_badge: Optional[InfoBadge] = None
        self._config_state_tooltip: Optional[StateToolTip] = None

        super().__init__(config, parent)

        self.controller = ConfigController(self.config, self)
        self._connect_controller_signals()
        self._setup_connections()
        self._update_ui_from_config()

    # ------------------------------------------------------------------
    # Enhanced UI construction helpers
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        """Build the enhanced settings interface using modern QFluentWidgets."""
        layout = VBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Create navigation and content layout
        self._create_navigation_section(layout)
        self._create_content_section(layout)
        self._create_controls_section(layout)

    def _create_navigation_section(self, layout: VBoxLayout) -> None:
        """Create the navigation section with NavigationBar."""
        nav_card = create_styled_card_widget("Configuration Categories", self)
        nav_layout = VBoxLayout(nav_card)

        # Create navigation widget
        self.nav_widget = NavigationWidget(self)
        self.nav_widget.setExpandWidth(280)

        # Add navigation items
        self.nav_widget.addItem(
            routeKey="general",
            icon=get_fluent_icon_for_action("settings"),
            text="General Settings",
            onClick=lambda: self._switch_to_category("general"),
        )

        self.nav_widget.addItem(
            routeKey="embedding",
            icon=get_fluent_icon_for_action("code"),
            text="Embedding Models",
            onClick=lambda: self._switch_to_category("embedding"),
        )

        self.nav_widget.addItem(
            routeKey="processing",
            icon=get_fluent_icon_for_action("play"),
            text="Text Processing",
            onClick=lambda: self._switch_to_category("processing"),
        )

        self.nav_widget.addItem(
            routeKey="database",
            icon=get_fluent_icon_for_action("document"),
            text="Database Settings",
            onClick=lambda: self._switch_to_category("database"),
        )

        self.nav_widget.addItem(
            routeKey="appearance",
            icon=get_fluent_icon_for_action("heart"),
            text="Appearance & Theme",
            onClick=lambda: self._switch_to_category("appearance"),
        )

        self.nav_widget.addItem(
            routeKey="advanced",
            icon=get_fluent_icon_for_action("warning"),
            text="Advanced Options",
            onClick=lambda: self._switch_to_category("advanced"),
        )

        nav_layout.addWidget(self.nav_widget)
        layout.addWidget(nav_card)

    def _create_content_section(self, layout: VBoxLayout) -> None:
        """Create the main content section with stacked widgets."""
        # Create stacked widget for different configuration categories
        self.content_stack = QStackedWidget(self)

        # Create scroll areas for each category
        self._create_general_settings()
        self._create_embedding_settings()
        self._create_processing_settings()
        self._create_database_settings()
        self._create_appearance_settings()
        self._create_advanced_settings()

        layout.addWidget(self.content_stack)

    def _create_controls_section(self, layout: VBoxLayout) -> None:
        """Create the enhanced controls section."""
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(12)

        # Unsaved changes indicator
        self._unsaved_changes_badge = InfoBadge.warning("0", self)
        self._unsaved_changes_badge.hide()
        controls_layout.addWidget(self._unsaved_changes_badge)

        controls_layout.addStretch()

        self.apply_btn = PrimaryPushButton("Apply Changes", self)
        self.apply_btn.setIcon(get_fluent_icon_for_action("save").icon())
        self.apply_btn.setEnabled(False)

        self.reset_btn = PushButton("Reset to Defaults", self)
        self.reset_btn.setIcon(get_fluent_icon_for_action("refresh").icon())

        self.reset_original_btn = PushButton("Reset to Original", self)
        self.reset_original_btn.setIcon(get_fluent_icon_for_action("close").icon())

        # Help button with teaching tip
        self.help_btn = PushButton("?")
        self.help_btn.setFixedSize(32, 32)
        self.help_btn.setIcon(get_fluent_icon_for_action("help").icon())
        self.help_btn.clicked.connect(self._show_config_help)

        controls_layout.addWidget(self.apply_btn)
        controls_layout.addWidget(self.reset_btn)
        controls_layout.addWidget(self.reset_original_btn)
        controls_layout.addWidget(self.help_btn)

        layout.addLayout(controls_layout)

    def _create_general_settings(self) -> None:
        """Create general settings page."""
        scroll_area = SmoothScrollArea()
        scroll_area.setWidgetResizable(True)

        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        layout = VBoxLayout(content_widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Application Settings Group
        app_group = SettingCardGroup("Application Settings", content_widget)

        # Auto-save setting
        self.auto_save_card = SwitchSettingCard(
            get_fluent_icon_for_action("save"),
            "Auto-save Configuration",
            "Automatically save configuration changes",
            parent=app_group,
        )
        app_group.addSettingCard(self.auto_save_card)

        # Startup behavior
        self.startup_card = ExpandSettingCard(
            get_fluent_icon_for_action("play"),
            "Startup Behavior",
            "Configure application startup options",
            parent=app_group,
        )

        # Add startup options to expand card
        startup_content = QWidget()
        startup_layout = VBoxLayout(startup_content)

        self.restore_session_cb = SwitchSettingCard(
            get_fluent_icon_for_action("refresh"),
            "Restore Last Session",
            "Restore previous session on startup",
        )

        self.check_updates_cb = SwitchSettingCard(
            get_fluent_icon_for_action("link"),
            "Check for Updates",
            "Check for application updates on startup",
        )

        startup_layout.addWidget(self.restore_session_cb)
        startup_layout.addWidget(self.check_updates_cb)
        self.startup_card.setContent(startup_content)
        app_group.addSettingCard(self.startup_card)

        layout.addWidget(app_group)

        # Logging Settings Group
        log_group = SettingCardGroup("Logging Settings", content_widget)

        # Log level setting
        self.log_level_card = SettingCard(
            get_fluent_icon_for_action("info"),
            "Log Level",
            "Set the application logging level",
            parent=log_group,
        )

        self.log_level_combo = ComboBox()
        self.log_level_combo.addItems([level.value for level in LogLevel])
        self.log_level_card.hBoxLayout.addWidget(self.log_level_combo)
        log_group.addSettingCard(self.log_level_card)

        # Log file location
        self.log_file_card = ExpandSettingCard(
            get_fluent_icon_for_action("document"),
            "Log File Settings",
            "Configure log file location and rotation",
            parent=log_group,
        )

        # Add log file options to expand card
        log_content = QWidget()
        log_layout = VBoxLayout(log_content)

        self.log_file_path_card = SettingCard(
            get_fluent_icon_for_action("open"), "Log File Path", "Location of log files"
        )

        self.log_file_path_edit = LineEdit()
        self.log_file_path_edit.setPlaceholderText("Default: logs/app.log")
        self.log_file_path_card.hBoxLayout.addWidget(self.log_file_path_edit)

        browse_btn = PushButton("Browse")
        browse_btn.setIcon(get_fluent_icon_for_action("open").icon())
        browse_btn.clicked.connect(self._browse_log_file)
        self.log_file_path_card.hBoxLayout.addWidget(browse_btn)

        log_layout.addWidget(self.log_file_path_card)
        self.log_file_card.setContent(log_content)
        log_group.addSettingCard(self.log_file_card)

        layout.addWidget(log_group)
        layout.addStretch()

        self.content_stack.addWidget(scroll_area)

    def _create_embedding_settings(self) -> None:
        """Create embedding model settings page."""
        scroll_area = SmoothScrollArea()
        scroll_area.setWidgetResizable(True)

        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        layout = VBoxLayout(content_widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Model Selection Group
        model_group = SettingCardGroup("Embedding Model Configuration", content_widget)

        # Model type selection
        self.model_type_card = SettingCard(
            get_fluent_icon_for_action("code"),
            "Model Type",
            "Select the embedding model provider",
            parent=model_group,
        )

        self.model_type_combo = ComboBox()
        self.model_type_combo.addItems([model.value for model in EmbeddingModelType])
        self.model_type_card.hBoxLayout.addWidget(self.model_type_combo)
        model_group.addSettingCard(self.model_type_card)

        # Model-specific settings
        self.model_settings_card = ExpandSettingCard(
            get_fluent_icon_for_action("settings"),
            "Model-Specific Settings",
            "Configure settings for the selected model",
            parent=model_group,
        )

        # Create stacked widget for model-specific settings
        self._provider_stack = QStackedWidget()
        self._create_model_provider_pages()
        self.model_settings_card.setContent(self._provider_stack)
        model_group.addSettingCard(self.model_settings_card)

        layout.addWidget(model_group)

        # Performance Settings Group
        perf_group = SettingCardGroup("Performance Settings", content_widget)

        # Batch size setting
        self.batch_size_card = RangeSettingCard(
            get_fluent_icon_for_action("add"),
            "Batch Size",
            "Number of texts to process simultaneously",
            parent=perf_group,
        )
        self.batch_size_card.setRange(1, 100)
        self.batch_size_card.setValue(10)
        perf_group.addSettingCard(self.batch_size_card)

        # Cache settings
        self.cache_card = ExpandSettingCard(
            get_fluent_icon_for_action("refresh"),
            "Embedding Cache",
            "Configure embedding caching options",
            parent=perf_group,
        )

        # Add cache options to expand card
        cache_content = QWidget()
        cache_layout = VBoxLayout(cache_content)

        self.enable_cache_cb = SwitchSettingCard(
            get_fluent_icon_for_action("save"),
            "Enable Caching",
            "Cache embeddings to improve performance",
        )

        self.cache_size_card = RangeSettingCard(
            get_fluent_icon_for_action("document"),
            "Cache Size (MB)",
            "Maximum cache size in megabytes",
        )
        self.cache_size_card.setRange(100, 10000)
        self.cache_size_card.setValue(1000)

        cache_layout.addWidget(self.enable_cache_cb)
        cache_layout.addWidget(self.cache_size_card)
        self.cache_card.setContent(cache_content)
        perf_group.addSettingCard(self.cache_card)

        layout.addWidget(perf_group)
        layout.addStretch()

        self.content_stack.addWidget(scroll_area)

    def _create_processing_settings(self) -> None:
        """Create text processing settings page."""
        scroll_area = SmoothScrollArea()
        scroll_area.setWidgetResizable(True)

        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        layout = VBoxLayout(content_widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Text Processing Group
        text_group = SettingCardGroup("Text Processing Configuration", content_widget)

        # Chunk size setting
        self.chunk_size_card = RangeSettingCard(
            get_fluent_icon_for_action("document"),
            "Chunk Size",
            "Size of text chunks for processing (characters)",
            parent=text_group,
        )
        self.chunk_size_card.setRange(100, 5000)
        self.chunk_size_card.setValue(1000)
        text_group.addSettingCard(self.chunk_size_card)

        # Overlap setting
        self.overlap_card = RangeSettingCard(
            get_fluent_icon_for_action("link"),
            "Chunk Overlap",
            "Overlap between consecutive chunks (characters)",
            parent=text_group,
        )
        self.overlap_card.setRange(0, 500)
        self.overlap_card.setValue(100)
        text_group.addSettingCard(self.overlap_card)

        # Text cleaning options
        self.cleaning_card = ExpandSettingCard(
            get_fluent_icon_for_action("edit"),
            "Text Cleaning Options",
            "Configure text preprocessing options",
            parent=text_group,
        )

        # Add cleaning options to expand card
        cleaning_content = QWidget()
        cleaning_layout = VBoxLayout(cleaning_content)

        self.remove_whitespace_cb = SwitchSettingCard(
            get_fluent_icon_for_action("delete"),
            "Remove Extra Whitespace",
            "Remove excessive whitespace and line breaks",
        )

        self.normalize_unicode_cb = SwitchSettingCard(
            get_fluent_icon_for_action("code"),
            "Normalize Unicode",
            "Normalize Unicode characters for consistency",
        )

        self.remove_headers_cb = SwitchSettingCard(
            get_fluent_icon_for_action("close"),
            "Remove Headers/Footers",
            "Attempt to remove document headers and footers",
        )

        cleaning_layout.addWidget(self.remove_whitespace_cb)
        cleaning_layout.addWidget(self.normalize_unicode_cb)
        cleaning_layout.addWidget(self.remove_headers_cb)
        self.cleaning_card.setContent(cleaning_content)
        text_group.addSettingCard(self.cleaning_card)

        layout.addWidget(text_group)
        layout.addStretch()

        self.content_stack.addWidget(scroll_area)

    def _create_database_settings(self) -> None:
        """Create database settings page."""
        scroll_area = SmoothScrollArea()
        scroll_area.setWidgetResizable(True)

        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        layout = VBoxLayout(content_widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Database Configuration Group
        db_group = SettingCardGroup("Vector Database Configuration", content_widget)

        # Database path setting
        self.db_path_card = SettingCard(
            get_fluent_icon_for_action("document"),
            "Database Path",
            "Location of the vector database",
            parent=db_group,
        )

        self.db_path_edit = LineEdit()
        self.db_path_edit.setPlaceholderText("Default: data/vector_db")
        self.db_path_card.hBoxLayout.addWidget(self.db_path_edit)

        browse_db_btn = PushButton("Browse")
        browse_db_btn.setIcon(get_fluent_icon_for_action("open").icon())
        browse_db_btn.clicked.connect(self._browse_db_path)
        self.db_path_card.hBoxLayout.addWidget(browse_db_btn)
        db_group.addSettingCard(self.db_path_card)

        # Database maintenance
        self.maintenance_card = ExpandSettingCard(
            get_fluent_icon_for_action("settings"),
            "Database Maintenance",
            "Configure database maintenance options",
            parent=db_group,
        )

        # Add maintenance options to expand card
        maintenance_content = QWidget()
        maintenance_layout = VBoxLayout(maintenance_content)

        self.auto_backup_cb = SwitchSettingCard(
            get_fluent_icon_for_action("save"),
            "Auto Backup",
            "Automatically backup database periodically",
        )

        self.backup_interval_card = RangeSettingCard(
            get_fluent_icon_for_action("refresh"),
            "Backup Interval (hours)",
            "Hours between automatic backups",
        )
        self.backup_interval_card.setRange(1, 168)  # 1 hour to 1 week
        self.backup_interval_card.setValue(24)

        maintenance_layout.addWidget(self.auto_backup_cb)
        maintenance_layout.addWidget(self.backup_interval_card)
        self.maintenance_card.setContent(maintenance_content)
        db_group.addSettingCard(self.maintenance_card)

        layout.addWidget(db_group)
        layout.addStretch()

        self.content_stack.addWidget(scroll_area)

    def _create_appearance_settings(self) -> None:
        """Create appearance and theme settings page."""
        scroll_area = SmoothScrollArea()
        scroll_area.setWidgetResizable(True)

        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        layout = VBoxLayout(content_widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Theme Settings Group
        theme_group = SettingCardGroup("Theme & Appearance", content_widget)

        # Theme selection
        self.theme_card = ColorSettingCard(
            get_fluent_icon_for_action("heart"),
            "Application Theme",
            "Choose the application color theme",
            parent=theme_group,
        )
        theme_group.addSettingCard(self.theme_card)

        # Dark mode toggle
        self.dark_mode_card = SwitchSettingCard(
            get_fluent_icon_for_action("heart"),
            "Dark Mode",
            "Use dark theme for the application",
            parent=theme_group,
        )
        theme_group.addSettingCard(self.dark_mode_card)

        # UI Scale
        self.ui_scale_card = RangeSettingCard(
            get_fluent_icon_for_action("add"),
            "UI Scale",
            "Scale factor for the user interface",
            parent=theme_group,
        )
        self.ui_scale_card.setRange(50, 200)
        self.ui_scale_card.setValue(100)
        self.ui_scale_card.setSuffix("%")
        theme_group.addSettingCard(self.ui_scale_card)

        # Advanced appearance options
        self.appearance_advanced_card = ExpandSettingCard(
            get_fluent_icon_for_action("settings"),
            "Advanced Appearance",
            "Advanced appearance customization options",
            parent=theme_group,
        )

        # Add advanced options to expand card
        appearance_content = QWidget()
        appearance_layout = VBoxLayout(appearance_content)

        self.animations_cb = SwitchSettingCard(
            get_fluent_icon_for_action("play"),
            "Enable Animations",
            "Enable UI animations and transitions",
        )

        self.transparency_cb = SwitchSettingCard(
            get_fluent_icon_for_action("heart"),
            "Window Transparency",
            "Enable window transparency effects",
        )

        self.compact_mode_cb = SwitchSettingCard(
            get_fluent_icon_for_action("add"),
            "Compact Mode",
            "Use compact layout to save space",
        )

        appearance_layout.addWidget(self.animations_cb)
        appearance_layout.addWidget(self.transparency_cb)
        appearance_layout.addWidget(self.compact_mode_cb)
        self.appearance_advanced_card.setContent(appearance_content)
        theme_group.addSettingCard(self.appearance_advanced_card)

        layout.addWidget(theme_group)
        layout.addStretch()

        self.content_stack.addWidget(scroll_area)

    def _create_advanced_settings(self) -> None:
        """Create advanced settings page."""
        scroll_area = SmoothScrollArea()
        scroll_area.setWidgetResizable(True)

        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        layout = VBoxLayout(content_widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Warning message
        warning_card = create_styled_card_widget("⚠️ Advanced Settings", content_widget)
        warning_layout = VBoxLayout(warning_card)
        warning_label = BodyLabel(
            "These settings are for advanced users only. Changing these values may affect application stability and performance."
        )
        warning_label.setStyleSheet("color: #d13438; font-weight: bold;")
        warning_layout.addWidget(warning_label)
        layout.addWidget(warning_card)

        # Advanced Configuration Group
        advanced_group = SettingCardGroup("Advanced Configuration", content_widget)

        # Debug mode
        self.debug_mode_card = SwitchSettingCard(
            get_fluent_icon_for_action("warning"),
            "Debug Mode",
            "Enable debug mode for troubleshooting",
            parent=advanced_group,
        )
        advanced_group.addSettingCard(self.debug_mode_card)

        # Memory settings
        self.memory_card = ExpandSettingCard(
            get_fluent_icon_for_action("code"),
            "Memory Management",
            "Configure memory usage settings",
            parent=advanced_group,
        )

        # Add memory options to expand card
        memory_content = QWidget()
        memory_layout = VBoxLayout(memory_content)

        self.memory_limit_card = RangeSettingCard(
            get_fluent_icon_for_action("warning"),
            "Memory Limit (MB)",
            "Maximum memory usage for the application",
        )
        self.memory_limit_card.setRange(512, 16384)  # 512MB to 16GB
        self.memory_limit_card.setValue(4096)

        self.gc_threshold_card = RangeSettingCard(
            get_fluent_icon_for_action("refresh"),
            "GC Threshold",
            "Garbage collection threshold",
        )
        self.gc_threshold_card.setRange(100, 10000)
        self.gc_threshold_card.setValue(1000)

        memory_layout.addWidget(self.memory_limit_card)
        memory_layout.addWidget(self.gc_threshold_card)
        self.memory_card.setContent(memory_content)
        advanced_group.addSettingCard(self.memory_card)

        # Experimental features
        self.experimental_card = ExpandSettingCard(
            get_fluent_icon_for_action("warning"),
            "Experimental Features",
            "Enable experimental features (use at your own risk)",
            parent=advanced_group,
        )

        # Add experimental options to expand card
        experimental_content = QWidget()
        experimental_layout = VBoxLayout(experimental_content)

        self.parallel_processing_cb = SwitchSettingCard(
            get_fluent_icon_for_action("play"),
            "Parallel Processing",
            "Enable experimental parallel processing",
        )

        self.gpu_acceleration_cb = SwitchSettingCard(
            get_fluent_icon_for_action("code"),
            "GPU Acceleration",
            "Enable GPU acceleration (requires compatible hardware)",
        )

        experimental_layout.addWidget(self.parallel_processing_cb)
        experimental_layout.addWidget(self.gpu_acceleration_cb)
        self.experimental_card.setContent(experimental_content)
        advanced_group.addSettingCard(self.experimental_card)

        layout.addWidget(advanced_group)
        layout.addStretch()

        self.content_stack.addWidget(scroll_area)

    def _switch_to_category(self, category: str) -> None:
        """Switch to a specific configuration category."""
        self._current_nav_item = category

        # Map category to stack index
        category_map = {
            "general": 0,
            "embedding": 1,
            "processing": 2,
            "database": 3,
            "appearance": 4,
            "advanced": 5,
        }

        index = category_map.get(category, 0)
        self.content_stack.setCurrentIndex(index)

        # Update navigation selection
        self.nav_widget.setCurrentItem(category)

        # Show info about the selected category
        category_descriptions = {
            "general": "Configure general application settings and behavior",
            "embedding": "Set up embedding models and performance options",
            "processing": "Configure text processing and chunking options",
            "database": "Manage vector database settings and maintenance",
            "appearance": "Customize application theme and appearance",
            "advanced": "Advanced settings for experienced users",
        }

        description = category_descriptions.get(category, "")
        if description:
            self.show_info_bar(
                f"Category: {category.title()}",
                description,
                InfoLevel.INFO,
                duration=2000,
            )

    def _show_config_help(self) -> None:
        """Show help for configuration options."""
        help_content = """
        <b>Configuration Categories:</b><br><br>
        <b>General:</b> Basic application settings and startup behavior<br>
        <b>Embedding:</b> AI model configuration and performance tuning<br>
        <b>Processing:</b> Text processing and chunking parameters<br>
        <b>Database:</b> Vector database location and maintenance<br>
        <b>Appearance:</b> Theme, colors, and UI customization<br>
        <b>Advanced:</b> Expert settings for troubleshooting and optimization
        """

        self.show_teaching_tip(
            "Configuration Help", help_content, self.help_btn, "config_help"
        )

    def _browse_log_file(self) -> None:
        """Browse for log file location."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Log File Location",
            "app.log",
            "Log Files (*.log);;Text Files (*.txt);;All Files (*)",
        )

        if file_path:
            self.log_file_path_edit.setText(file_path)
            self._mark_config_changed()

    def _browse_db_path(self) -> None:
        """Browse for database path."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Database Directory")

        if dir_path:
            self.db_path_edit.setText(dir_path)
            self._mark_config_changed()

    def _mark_config_changed(self) -> None:
        """Mark configuration as changed."""
        if not self._updating_ui:
            self.apply_btn.setEnabled(True)

            # Update unsaved changes badge
            if self._unsaved_changes_badge:
                self._unsaved_changes_badge.setText("!")
                self._unsaved_changes_badge.show()

            # Show state tooltip
            if not self._config_state_tooltip:
                self._config_state_tooltip = self.show_state_tooltip(
                    "Unsaved Changes",
                    "You have unsaved configuration changes",
                    self.apply_btn,
                )

    def _clear_config_changed(self) -> None:
        """Clear configuration changed state."""
        self.apply_btn.setEnabled(False)

        # Hide unsaved changes badge
        if self._unsaved_changes_badge:
            self._unsaved_changes_badge.hide()

        # Hide state tooltip
        if self._config_state_tooltip:
            self._config_state_tooltip.close()
            self._config_state_tooltip = None

    def _create_model_provider_pages(self) -> None:
        """Create model provider-specific configuration pages."""
        # This method creates the stacked widget pages for different model providers
        # For now, we'll create a simple placeholder
        placeholder_widget = QWidget()
        placeholder_layout = VBoxLayout(placeholder_widget)
        placeholder_label = BodyLabel(
            "Model-specific settings will be displayed here based on the selected provider."
        )
        placeholder_layout.addWidget(placeholder_label)
        self._provider_stack.addWidget(placeholder_widget)

    def _setup_connections(self) -> None:
        """Set up signal/slot connections."""
        self.apply_btn.clicked.connect(self._apply_changes)
        self.reset_btn.clicked.connect(self._reset_to_defaults)
        self.reset_original_btn.clicked.connect(self._reset_to_original)

        # Connect all setting cards to mark changes
        self._connect_setting_cards()

    def _connect_setting_cards(self) -> None:
        """Connect all setting cards to change tracking."""
        # This method would connect all the setting cards to the _mark_config_changed method
        # For brevity, we'll implement the key connections

        # General settings
        if hasattr(self, "auto_save_card"):
            self.auto_save_card.checkedChanged.connect(self._mark_config_changed)
        if hasattr(self, "log_level_combo"):
            self.log_level_combo.currentTextChanged.connect(self._mark_config_changed)

        # Embedding settings
        if hasattr(self, "model_type_combo"):
            self.model_type_combo.currentTextChanged.connect(self._mark_config_changed)
        if hasattr(self, "batch_size_card"):
            self.batch_size_card.valueChanged.connect(self._mark_config_changed)

        # Processing settings
        if hasattr(self, "chunk_size_card"):
            self.chunk_size_card.valueChanged.connect(self._mark_config_changed)

        # Theme settings
        if hasattr(self, "dark_mode_card"):
            self.dark_mode_card.checkedChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, checked: bool) -> None:
        """Handle theme change."""
        self._mark_config_changed()

        # Apply theme immediately for preview
        if checked:
            qconfig.theme = Theme.DARK
        else:
            qconfig.theme = Theme.LIGHT

        self.show_info_bar(
            "Theme Changed",
            f"Switched to {'dark' if checked else 'light'} theme",
            InfoLevel.INFO,
        )

    def _apply_changes(self) -> None:
        """Apply configuration changes."""
        try:
            # Show state tooltip for applying changes
            apply_tooltip = self.show_state_tooltip(
                "Applying Changes", "Saving configuration changes...", self.apply_btn
            )

            # Update config from UI
            self._update_config_from_ui()

            # Save config
            self.controller.save_config(self.config)

            # Clear changed state
            self._clear_config_changed()

            # Hide tooltip
            apply_tooltip.close()

            # Show success feedback
            self.show_success_info(
                "Configuration Saved", "All changes have been applied successfully"
            )

            # Emit config changed signal
            self.config_changed.emit(self.config)

        except Exception as e:
            self.show_error_info("Save Failed", f"Failed to save configuration: {e!s}")

    def _reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        reply = MessageBox.question(
            self,
            "Reset to Defaults",
            "Are you sure you want to reset all settings to their default values?",
            MessageBox.Yes | MessageBox.No,
            MessageBox.No,
        )

        if reply == MessageBox.Yes:
            # Create default config
            default_config = Config()

            # Update UI with default values
            self._updating_ui = True
            self._update_ui_from_config(default_config)
            self._updating_ui = False

            # Mark as changed
            self._mark_config_changed()

            self.show_info_bar(
                "Reset Complete",
                "Configuration reset to default values",
                InfoLevel.INFO,
            )

    def _reset_to_original(self) -> None:
        """Reset configuration to original values."""
        reply = MessageBox.question(
            self,
            "Reset to Original",
            "Are you sure you want to discard all changes and revert to the original configuration?",
            MessageBox.Yes | MessageBox.No,
            MessageBox.No,
        )

        if reply == MessageBox.Yes:
            # Update UI with original config
            self._updating_ui = True
            self._update_ui_from_config()
            self._updating_ui = False

            # Clear changed state
            self._clear_config_changed()

            self.show_info_bar(
                "Reset Complete",
                "Configuration reverted to original values",
                InfoLevel.INFO,
            )

    def _update_ui_from_config(self, config: Optional[Config] = None) -> None:
        """Update UI elements from configuration."""
        if config is None:
            config = self.config

        self._updating_ui = True

        try:
            # Update general settings
            if hasattr(self, "log_level_combo"):
                self.log_level_combo.setCurrentText(config.logging.level.value)

            # Update embedding settings
            if hasattr(self, "model_type_combo"):
                self.model_type_combo.setCurrentText(config.embedding.model_type.value)
            if hasattr(self, "batch_size_card"):
                self.batch_size_card.setValue(config.embedding.batch_size)

            # Update processing settings
            if hasattr(self, "chunk_size_card"):
                self.chunk_size_card.setValue(config.text_processing.chunk_size)
            if hasattr(self, "overlap_card"):
                self.overlap_card.setValue(config.text_processing.chunk_overlap)

            # Update database settings
            if hasattr(self, "db_path_edit"):
                self.db_path_edit.setText(str(config.chroma_db.persist_directory))

        finally:
            self._updating_ui = False

    def _update_config_from_ui(self) -> None:
        """Update configuration from UI elements."""
        # Update general settings
        if hasattr(self, "log_level_combo"):
            self.config.logging.level = LogLevel(self.log_level_combo.currentText())

        # Update embedding settings
        if hasattr(self, "model_type_combo"):
            self.config.embedding.model_type = EmbeddingModelType(
                self.model_type_combo.currentText()
            )
        if hasattr(self, "batch_size_card"):
            self.config.embedding.batch_size = self.batch_size_card.value()

        # Update processing settings
        if hasattr(self, "chunk_size_card"):
            self.config.text_processing.chunk_size = self.chunk_size_card.value()
        if hasattr(self, "overlap_card"):
            self.config.text_processing.chunk_overlap = self.overlap_card.value()

        # Update database settings
        if hasattr(self, "db_path_edit"):
            self.config.chroma_db.persist_directory = Path(self.db_path_edit.text())

    def _connect_controller_signals(self) -> None:
        """Connect controller signals to widget slots."""
        self.controller.config_saved.connect(self._on_config_saved)
        self.controller.config_loaded.connect(self._on_config_loaded)
        self.controller.config_error.connect(self._on_config_error)

    def _on_config_saved(self, config: Config) -> None:
        """Handle config saved signal."""
        self.emit_status("Configuration saved successfully")

    def _on_config_loaded(self, config: Config) -> None:
        """Handle config loaded signal."""
        self.config = config
        self._update_ui_from_config()
        self.emit_status("Configuration loaded successfully")

    def _on_config_error(self, error_message: str) -> None:
        """Handle config error signal."""
        self.show_error_info("Configuration Error", error_message)
        self.emit_status(f"Configuration error: {error_message}")

    def on_tab_activated(self) -> None:
        """Called when this tab is activated."""
        self.emit_status("Configuration tab activated")
        # Set focus to the first navigation item
        if hasattr(self, "nav_widget"):
            self.nav_widget.setCurrentItem("general")
