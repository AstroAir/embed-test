"""
Settings dialog for PDF Vector System GUI.

This module contains the advanced settings dialog.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontMetrics, QGuiApplication, QTextOption
from PySide6.QtWidgets import QDialog, QFrame, QHBoxLayout, QScrollArea, QTabWidget
from PySide6.QtWidgets import QTextEdit as QtTextEdit
from PySide6.QtWidgets import QWidget
from qfluentwidgets import (
    BodyLabel,
    InfoBar,
    InfoBarPosition,
    PrimaryPushButton,
    PushButton,
    TextEdit,
    VBoxLayout,
)

from pdf_vector_system.core.config.settings import Config


class SettingsDialog(QDialog):
    """Advanced settings dialog that surfaces read-only configuration insights."""

    def __init__(
        self, config: Optional[Config] = None, parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self.config = config or Config()

        self.summary_tab: Optional[QWidget] = None
        self.summary_container: Optional[QWidget] = None
        self.summary_layout: Optional[VBoxLayout] = None
        self.summary_intro: Optional[BodyLabel] = None
        self.summary_scroll: Optional[QScrollArea] = None
        self.raw_edit: Optional[TextEdit] = None
        self.last_refreshed_label: Optional[BodyLabel] = None

        self._setup_ui()
        self._populate_views()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Advanced Settings")
        self.setModal(True)
        self.resize(720, 520)

        layout = VBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.tab_widget = QTabWidget(self)
        layout.addWidget(self.tab_widget, 1)

        # Summary tab ----------------------------------------------------------
        self.summary_tab = QWidget(self.tab_widget)
        summary_tab_layout = VBoxLayout(self.summary_tab)
        summary_tab_layout.setContentsMargins(16, 16, 16, 16)
        summary_tab_layout.setSpacing(12)

        self.summary_intro = BodyLabel(
            "Key configuration values pulled from the active profile.",
            self.summary_tab,
        )
        self.summary_intro.setWordWrap(True)
        self.summary_intro.setProperty("class", "summaryIntro")
        self.summary_intro.setStyleSheet("color: rgba(0, 0, 0, 0.6);")
        summary_tab_layout.addWidget(self.summary_intro)

        self.summary_scroll = QScrollArea(self.summary_tab)
        self.summary_scroll.setWidgetResizable(True)
        self.summary_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.summary_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.summary_scroll.setFrameShadow(QFrame.Shadow.Plain)
        self.summary_scroll.verticalScrollBar().setSingleStep(28)
        self.summary_scroll.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.summary_scroll.setStyleSheet(
            "QScrollArea { background-color: transparent; }"
        )
        summary_tab_layout.addWidget(self.summary_scroll, 1)

        self.summary_container = QWidget(self.summary_scroll)
        self.summary_layout = VBoxLayout(self.summary_container)
        self.summary_layout.setContentsMargins(0, 0, 0, 0)
        self.summary_layout.setSpacing(10)
        self.summary_layout.addStretch(1)
        self.summary_scroll.setWidget(self.summary_container)
        self.tab_widget.addTab(self.summary_tab, "Overview")

        # Raw JSON tab ---------------------------------------------------------
        raw_tab = QWidget(self.tab_widget)
        raw_layout = VBoxLayout(raw_tab)
        raw_layout.setContentsMargins(16, 16, 16, 16)
        raw_layout.setSpacing(12)

        raw_intro = BodyLabel("Full configuration payload in JSON format.", raw_tab)
        raw_intro.setWordWrap(True)
        raw_intro.setProperty("class", "rawIntro")
        raw_intro.setStyleSheet("color: rgba(0, 0, 0, 0.6);")
        raw_layout.addWidget(raw_intro)

        self.raw_edit = TextEdit(raw_tab)
        self.raw_edit.setReadOnly(True)
        self.raw_edit.setPlaceholderText("Configuration JSON will appear here...")
        self.raw_edit.setMinimumHeight(260)
        self.raw_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.raw_edit.setLineWrapMode(QtTextEdit.LineWrapMode.NoWrap)
        self.raw_edit.setWordWrapMode(QTextOption.WrapMode.NoWrap)
        font = self.raw_edit.font()
        font.setFamily("JetBrains Mono")
        font.setStyleHint(QFont.StyleHint.Monospace)
        size = font.pointSize()
        font.setPointSize(11 if size <= 0 else max(size, 11))
        self.raw_edit.setFont(font)
        metrics = QFontMetrics(font)
        self.raw_edit.setTabStopDistance(metrics.horizontalAdvance(" ") * 2)
        self.raw_edit.document().setIndentWidth(2)
        self.raw_edit.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.raw_edit.setStyleSheet(
            "TextEdit {"
            "background-color: rgba(20, 20, 20, 0.04);"
            "border: 1px solid rgba(20, 20, 20, 0.15);"
            "border-radius: 8px;"
            "padding: 14px;"
            "font-size: 12.5px;"
            "line-height: 1.52;"
            "letter-spacing: 0.2px;"
            "}"
        )
        raw_layout.addWidget(self.raw_edit)

        self.tab_widget.addTab(raw_tab, "Raw JSON")

        # Footer actions -------------------------------------------------------
        footer_layout = QHBoxLayout()
        footer_layout.setSpacing(8)

        self.copy_btn = PushButton("Copy JSON", self)
        self.copy_btn.setToolTip("Copy the configuration JSON to your clipboard")
        self.refresh_btn = PushButton("Refresh", self)
        self.refresh_btn.setToolTip(
            "Reload configuration values from the active profile"
        )
        footer_layout.addWidget(self.copy_btn)
        footer_layout.addWidget(self.refresh_btn)
        footer_layout.addSpacing(10)

        self.last_refreshed_label = BodyLabel("Last updated • —", self)
        self.last_refreshed_label.setProperty("class", "refreshMeta")
        self.last_refreshed_label.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.last_refreshed_label.setStyleSheet("color: rgba(0, 0, 0, 0.55);")
        self.last_refreshed_label.setMinimumWidth(210)
        self.last_refreshed_label.setToolTip(
            "Timestamp updates whenever configuration data refreshes"
        )
        footer_layout.addWidget(self.last_refreshed_label)
        footer_layout.addStretch(1)

        close_btn = PrimaryPushButton("Close", self)
        close_btn.clicked.connect(self.accept)
        footer_layout.addWidget(close_btn)

        layout.addLayout(footer_layout)

        self.copy_btn.clicked.connect(self._copy_json_to_clipboard)
        self.refresh_btn.clicked.connect(self._populate_views)

    def _populate_views(self) -> None:
        """Refresh both the summary view and raw JSON source."""
        if self.summary_layout is not None:
            # Remove previous summary widgets except the trailing stretch
            while self.summary_layout.count() > 1:
                item = self.summary_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()

            separators = self.config.text_processing.separators
            separators_preview = ", ".join(separators[:4])
            if len(separators) > 4:
                separators_preview += "…"
            separators_full = ", ".join(separators)

            vector_directory = str(self.config.chroma_db.persist_directory)
            log_file = str(self.config.logging.file_path or "(stdout)")

            sections = {
                "Embedding": [
                    ("Provider", self.config.embedding.model_type.value, None),
                    ("Model", self.config.embedding.model_name, None),
                    ("Batch Size", str(self.config.embedding.batch_size), None),
                    ("Timeout", f"{self.config.embedding.timeout_seconds}s", None),
                ],
                "Text Processing": [
                    ("Chunk Size", str(self.config.text_processing.chunk_size), None),
                    ("Overlap", str(self.config.text_processing.chunk_overlap), None),
                    (
                        "Min Chunk",
                        str(self.config.text_processing.min_chunk_size),
                        None,
                    ),
                    ("Separators", separators_preview, separators_full or None),
                ],
                "PDF": [
                    ("Max Size", f"{self.config.pdf.max_file_size_mb} MB", None),
                    ("Timeout", f"{self.config.pdf.timeout_seconds}s", None),
                    (
                        "Extract Images",
                        "Yes" if self.config.pdf.extract_images else "No",
                        None,
                    ),
                ],
                "Vector DB": [
                    ("Directory", vector_directory, vector_directory),
                    ("Collection", self.config.chroma_db.collection_name, None),
                    ("Metric", self.config.chroma_db.distance_metric, None),
                    ("Max Results", str(self.config.chroma_db.max_results), None),
                ],
                "Logging": [
                    ("Level", self.config.logging.level.value, None),
                    ("Rotation", self.config.logging.rotation, None),
                    ("Retention", self.config.logging.retention, None),
                    ("File", log_file, None if log_file == "(stdout)" else log_file),
                ],
                "Runtime": [
                    ("Debug", "Enabled" if self.config.debug else "Disabled", None),
                    ("Max Workers", str(self.config.max_workers), None),
                ],
            }

            section_items = list(sections.items())
            total_sections = len(section_items)

            for index, (section_title, values) in enumerate(section_items):
                # Ensure values is a list of tuples as expected by _build_summary_card
                if not isinstance(values, list):
                    continue
                # Optionally, check the type of each item in the list
                filtered_values = [
                    v for v in values if isinstance(v, tuple) and len(v) == 3
                ]
                section_card = self._build_summary_card(section_title, filtered_values)
                self.summary_layout.insertWidget(
                    self.summary_layout.count() - 1, section_card
                )

                if index < total_sections - 1:
                    self.summary_layout.insertSpacing(
                        self.summary_layout.count() - 1, 12
                    )

        if self.raw_edit is not None:
            self.raw_edit.setPlainText(self.config.model_dump_json(indent=2))

        self._update_refresh_timestamp()

    def _build_summary_card(
        self, title: str, rows: list[tuple[str, Optional[str], Optional[str]]]
    ) -> QWidget:
        """Create a styled card widget for a summary section."""
        card = QWidget(self.summary_container)
        card.setObjectName("summarySectionCard")
        card.setStyleSheet(
            "#summarySectionCard {"
            "background-color: rgba(255, 255, 255, 0.78);"
            "border: 1px solid rgba(15, 23, 42, 0.08);"
            "border-radius: 12px;"
            "}"
        )

        card_layout = VBoxLayout(card)
        card_layout.setContentsMargins(18, 18, 18, 18)
        card_layout.setSpacing(10)

        header = BodyLabel(title, card)
        header.setStyleSheet("font-weight: 600; font-size: 15px;")
        card_layout.addWidget(header)

        for index, (key, value, tooltip) in enumerate(rows):
            if index > 0:
                card_layout.addSpacing(6)

            row_widget = QWidget(card)
            row_widget.setObjectName("summaryRowContainer")
            row_widget.setMinimumHeight(28)

            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(12)

            key_label = BodyLabel(key, row_widget)
            key_label.setStyleSheet("color: rgba(0, 0, 0, 0.55); font-weight: 500;")
            key_label.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            key_label.setMinimumWidth(132)
            key_label.setProperty("class", "summaryKey")
            row_layout.addWidget(key_label, 0)
            row_layout.setStretch(0, 0)

            raw_value = "" if value is None else str(value)
            value_text = self._format_summary_value(raw_value)
            value_label = BodyLabel(value_text, row_widget)
            value_label.setWordWrap(True)
            value_label.setStyleSheet("color: rgba(0, 0, 0, 0.85);")
            value_label.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            value_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
                | Qt.TextInteractionFlag.TextSelectableByKeyboard
            )
            value_label.setProperty("class", "summaryValue")
            tooltip_text = tooltip if tooltip else None
            if tooltip_text:
                value_label.setToolTip(str(tooltip_text))
            elif raw_value and raw_value.strip() and len(raw_value) > 60:
                value_label.setToolTip(raw_value)
            row_layout.addWidget(value_label, 1)
            row_layout.setStretch(1, 1)

            card_layout.addWidget(row_widget)

        card_layout.addStretch(1)
        return card

    @staticmethod
    def _format_summary_value(value: Optional[str]) -> str:
        """Return a display-friendly string for summary values."""
        if value is None:
            return "—"

        text = str(value).strip()
        return text or "—"

    def _update_refresh_timestamp(self) -> None:
        if self.last_refreshed_label is None:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_refreshed_label.setText(f"Last updated • {timestamp}")
        self.last_refreshed_label.setToolTip(timestamp)

    def _copy_json_to_clipboard(self) -> None:
        if self.raw_edit is None:
            return
        QGuiApplication.clipboard().setText(self.raw_edit.toPlainText())
        InfoBar.success(
            "Copied",
            "Configuration JSON copied to clipboard.",
            duration=1800,
            position=InfoBarPosition.BOTTOM_RIGHT,
            parent=self,
        )

    def set_config(self, config: Config) -> None:
        """Update dialog data with a new configuration."""
        self.config = config
        self._populate_views()
