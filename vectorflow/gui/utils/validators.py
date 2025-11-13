"""
Input validators for PDF Vector System GUI.

This module provides validation classes for GUI input components.
"""

from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QObject
from PySide6.QtGui import QValidator

from vectorflow.core.config.settings import Config


class PathValidator(QValidator):
    """Validator for file and directory paths."""

    def __init__(
        self,
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False,
        parent: Optional[QObject] = None,
    ):
        """
        Initialize path validator.

        Args:
            must_exist: Whether path must exist
            must_be_file: Whether path must be a file
            must_be_dir: Whether path must be a directory
            parent: Parent QObject
        """
        super().__init__(parent)
        self.must_exist = must_exist
        self.must_be_file = must_be_file
        self.must_be_dir = must_be_dir

    def validate(self, input_str: str, pos: int) -> tuple:
        """
        Validate path input.

        Args:
            input_str: Input string to validate
            pos: Cursor position

        Returns:
            Tuple of (state, input_str, pos)
        """
        if not input_str.strip():
            return (QValidator.Intermediate, input_str, pos)

        try:
            path = Path(input_str)

            if self.must_exist and not path.exists():
                return (QValidator.Intermediate, input_str, pos)

            if self.must_be_file and path.exists() and not path.is_file():
                return (QValidator.Invalid, input_str, pos)

            if self.must_be_dir and path.exists() and not path.is_dir():
                return (QValidator.Invalid, input_str, pos)

            return (QValidator.Acceptable, input_str, pos)

        except Exception:
            return (QValidator.Invalid, input_str, pos)


class NumberValidator(QValidator):
    """Validator for numeric input with range checking."""

    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_float: bool = True,
        parent: Optional[QObject] = None,
    ):
        """
        Initialize number validator.

        Args:
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_float: Whether to allow floating point numbers
            parent: Parent QObject
        """
        super().__init__(parent)
        self.min_value = min_value
        self.max_value = max_value
        self.allow_float = allow_float

    def validate(self, input_str: str, pos: int) -> tuple:
        """
        Validate numeric input.

        Args:
            input_str: Input string to validate
            pos: Cursor position

        Returns:
            Tuple of (state, input_str, pos)
        """
        if not input_str.strip():
            return (QValidator.Intermediate, input_str, pos)

        try:
            value = float(input_str) if self.allow_float else int(input_str)

            if self.min_value is not None and value < self.min_value:
                return (QValidator.Invalid, input_str, pos)

            if self.max_value is not None and value > self.max_value:
                return (QValidator.Invalid, input_str, pos)

            return (QValidator.Acceptable, input_str, pos)

        except ValueError:
            return (QValidator.Invalid, input_str, pos)


class ConfigValidator:
    """Validator for configuration values."""

    @staticmethod
    def validate_config(config: Config) -> tuple[bool, list[str]]:
        """
        Validate a configuration object.

        Args:
            config: Configuration to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Validate embedding configuration
        if config.embedding.batch_size <= 0:
            errors.append("Embedding batch size must be positive")

        if config.embedding.max_retries < 0:
            errors.append("Embedding max retries must be non-negative")

        # Validate text processing configuration
        if config.text_processing.chunk_size <= 0:
            errors.append("Text chunk size must be positive")

        if config.text_processing.chunk_overlap < 0:
            errors.append("Text chunk overlap must be non-negative")

        if config.text_processing.chunk_overlap >= config.text_processing.chunk_size:
            errors.append("Text chunk overlap must be less than chunk size")

        if config.text_processing.min_chunk_size <= 0:
            errors.append("Minimum chunk size must be positive")

        # Validate PDF configuration
        if config.pdf.max_file_size_mb <= 0:
            errors.append("PDF max file size must be positive")

        if config.pdf.timeout_seconds <= 0:
            errors.append("PDF timeout must be positive")

        # Validate ChromaDB configuration
        if not config.chroma_db.collection_name.strip():
            errors.append("ChromaDB collection name cannot be empty")

        if config.chroma_db.max_results <= 0:
            errors.append("ChromaDB max results must be positive")

        # Validate global settings
        if config.max_workers <= 0:
            errors.append("Max workers must be positive")

        return len(errors) == 0, errors

    @staticmethod
    def validate_field(field_name: str, value: Any, config: Config) -> tuple[bool, str]:
        """
        Validate a specific configuration field.

        Args:
            field_name: Name of the field to validate
            value: Value to validate
            config: Current configuration for context

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Create a copy of config and update the field
            config_dict = config.model_dump()

            # Navigate to nested field if needed
            if "." in field_name:
                parts = field_name.split(".")
                current = config_dict
                for part in parts[:-1]:
                    current = current[part]
                current[parts[-1]] = value
            else:
                config_dict[field_name] = value

            # Try to create new config with updated value
            new_config = Config(**config_dict)

            # Validate the new config
            is_valid, errors = ConfigValidator.validate_config(new_config)

            if is_valid:
                return True, ""
            # Return first relevant error
            for error in errors:
                if field_name.lower() in error.lower():
                    return False, error
            return False, errors[0] if errors else "Validation failed"

        except Exception as e:
            return False, f"Invalid value: {e!s}"
