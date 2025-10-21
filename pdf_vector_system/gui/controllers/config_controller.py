"""
Configuration controller for PDF Vector System GUI.

This module contains the controller for configuration management.
"""

import json
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal

from pdf_vector_system.config.settings import Config
from pdf_vector_system.gui.utils.validators import ConfigValidator


class ConfigController(QObject):
    """Controller for configuration management."""

    # Signals
    config_updated = Signal(object)  # new_config
    config_saved = Signal(str)  # file_path
    config_loaded = Signal(object)  # loaded_config
    validation_error = Signal(str)  # error_message
    operation_error = Signal(str)  # error_message
    status_message = Signal(str)  # status_message

    def __init__(
        self, config: Optional[Config] = None, parent: Optional[QObject] = None
    ):
        """
        Initialize the configuration controller.

        Args:
            config: Configuration object
            parent: Parent QObject
        """
        super().__init__(parent)

        self.config = config or Config()
        self._original_config = self.config.model_copy()

    def update_field(self, field_path: str, value: Any) -> bool:
        """
        Update a configuration field.

        Args:
            field_path: Dot-separated path to the field (e.g., 'embedding.batch_size')
            value: New value for the field

        Returns:
            True if update was successful
        """
        try:
            # Validate the field value
            is_valid, error_msg = ConfigValidator.validate_field(
                field_path, value, self.config
            )

            if not is_valid:
                self.validation_error.emit(error_msg)
                return False

            # Update the field
            config_dict = self.config.model_dump()

            # Navigate to the field
            if "." in field_path:
                parts = field_path.split(".")
                current = config_dict
                for part in parts[:-1]:
                    current = current[part]
                current[parts[-1]] = value
            else:
                config_dict[field_path] = value

            # Create new config
            new_config = Config(**config_dict)

            # Validate the entire config
            is_valid, errors = ConfigValidator.validate_config(new_config)

            if not is_valid:
                self.validation_error.emit(
                    f"Configuration validation failed: {'; '.join(errors)}"
                )
                return False

            # Update current config
            self.config = new_config
            self.config_updated.emit(self.config)
            self.status_message.emit(f"Updated {field_path}")

            return True

        except Exception as e:
            self.operation_error.emit(f"Failed to update {field_path}: {e!s}")
            return False

    def apply_config(self, config_dict: dict[str, Any]) -> bool:
        """
        Apply a complete configuration dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            True if application was successful
        """
        try:
            # Create new config from dictionary
            new_config = Config(**config_dict)

            # Validate the config
            is_valid, errors = ConfigValidator.validate_config(new_config)

            if not is_valid:
                self.validation_error.emit(
                    f"Configuration validation failed: {'; '.join(errors)}"
                )
                return False

            # Apply the config
            self.config = new_config
            self.config_updated.emit(self.config)
            self.status_message.emit("Configuration applied successfully")

            return True

        except Exception as e:
            self.operation_error.emit(f"Failed to apply configuration: {e!s}")
            return False

    def reset_to_defaults(self) -> bool:
        """
        Reset configuration to defaults.

        Returns:
            True if reset was successful
        """
        try:
            # Create new default config
            default_config = Config()

            self.config = default_config
            self.config_updated.emit(self.config)
            self.status_message.emit("Configuration reset to defaults")

            return True

        except Exception as e:
            self.operation_error.emit(f"Failed to reset configuration: {e!s}")
            return False

    def reset_to_original(self) -> bool:
        """
        Reset configuration to original values.

        Returns:
            True if reset was successful
        """
        try:
            self.config = self._original_config.model_copy()
            self.config_updated.emit(self.config)
            self.status_message.emit("Configuration reset to original values")

            return True

        except Exception as e:
            self.operation_error.emit(f"Failed to reset configuration: {e!s}")
            return False

    def save_to_file(self, file_path: str) -> bool:
        """
        Save configuration to a file.

        Args:
            file_path: Path to save the configuration file

        Returns:
            True if save was successful
        """
        try:
            config_dict = self.config.model_dump()

            # Convert Path objects to strings for JSON serialization
            def convert_paths(obj):
                if isinstance(obj, dict):
                    return {k: convert_paths(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert_paths(item) for item in obj]
                if isinstance(obj, Path):
                    return str(obj)
                return obj

            serializable_dict = convert_paths(config_dict)

            # Save to file
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(serializable_dict, f, indent=2)

            self.config_saved.emit(file_path)
            self.status_message.emit(f"Configuration saved to {file_path}")

            return True

        except Exception as e:
            self.operation_error.emit(f"Failed to save configuration: {e!s}")
            return False

    def load_from_file(self, file_path: str) -> bool:
        """
        Load configuration from a file.

        Args:
            file_path: Path to load the configuration file from

        Returns:
            True if load was successful
        """
        try:
            # Load from file
            with file_path.open(encoding="utf-8") as f:
                config_dict = json.load(f)

            # Convert string paths back to Path objects
            def convert_strings_to_paths(obj, key=None):
                if isinstance(obj, dict):
                    return {k: convert_strings_to_paths(v, k) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert_strings_to_paths(item) for item in obj]
                if (
                    isinstance(obj, str)
                    and key
                    and ("directory" in key.lower() or "path" in key.lower())
                ):
                    return Path(obj)
                return obj

            processed_dict = convert_strings_to_paths(config_dict)

            # Create and validate config
            new_config = Config(**processed_dict)

            is_valid, errors = ConfigValidator.validate_config(new_config)

            if not is_valid:
                self.validation_error.emit(
                    f"Loaded configuration is invalid: {'; '.join(errors)}"
                )
                return False

            # Apply the loaded config
            self.config = new_config
            self.config_loaded.emit(self.config)
            self.config_updated.emit(self.config)
            self.status_message.emit(f"Configuration loaded from {file_path}")

            return True

        except Exception as e:
            self.operation_error.emit(f"Failed to load configuration: {e!s}")
            return False

    def get_current_config(self) -> Config:
        """
        Get the current configuration.

        Returns:
            Current configuration object
        """
        return self.config

    def has_changes(self) -> bool:
        """
        Check if configuration has changes from original.

        Returns:
            True if configuration has been modified
        """
        return self.config.model_dump() != self._original_config.model_dump()
