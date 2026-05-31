"""Persistence and validation adapter for the Settings hub."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import re
import sys
from typing import Any

import toml

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib

from ...config import (
    load_cli_config_and_ensure_existence,
    save_setting_to_cli_config,
    save_settings_to_cli_config,
)
from .settings_config_models import SettingsValidationResult


_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?P<key>[A-Za-z0-9_]*(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD)[A-Za-z0-9_-]*)"
    r"(?P<sep>\s*=\s*)"
    r"(?P<value>[^\s,;]+)",
    re.IGNORECASE,
)


def redact_secret_text(text: str) -> str:
    """Redact secret-looking assignment values from visible Settings output."""

    def _replace(match: re.Match[str]) -> str:
        return f"{match.group('key')}{match.group('sep')}<redacted>"

    return _SECRET_ASSIGNMENT_PATTERN.sub(_replace, str(text))


def _is_toml_scalar_value(text: str) -> bool:
    """Return whether text parses as a TOML value rather than a table."""
    try:
        toml.loads(f"__value__ = {text}")
    except toml.TomlDecodeError:
        return False
    return True


class SettingsConfigAdapter:
    """Narrow adapter over existing config helpers for Settings screens."""

    def load(self, *, force_reload: bool = False) -> dict[str, Any]:
        """Load CLI config through the existing config helper."""
        return deepcopy(load_cli_config_and_ensure_existence(force_reload=force_reload))

    def save_values(self, section: str, values: Mapping[str, Any]) -> bool:
        """Persist a group of values to one config section."""
        all_saved = True
        for key, value in values.items():
            if not save_setting_to_cli_config(section, key, value):
                all_saved = False
        return all_saved

    def save_sections(self, section_values: Mapping[str, Mapping[str, Any]]) -> bool:
        """Persist values across one or more sections in a single config write."""
        return save_settings_to_cli_config(section_values)

    def validate_raw_toml(self, text: str) -> SettingsValidationResult:
        """Validate TOML text and require a top-level table/mapping."""
        stripped_text = text.strip()

        try:
            parsed = toml.loads(text)
        except toml.TomlDecodeError as exc:
            if _is_toml_scalar_value(stripped_text):
                return SettingsValidationResult(
                    False,
                    "top-level TOML value must be a table",
                )
            return SettingsValidationResult(False, redact_secret_text(str(exc)))

        if not isinstance(parsed, dict):
            return SettingsValidationResult(
                False,
                "top-level TOML value must be a table",
            )

        return SettingsValidationResult(True, "TOML is valid")

    def validate_config_file(self, path: Path) -> SettingsValidationResult:
        """Strictly validate an on-disk TOML config file.

        Args:
            path: Config file path to parse.

        Returns:
            Validation result that reports parse and filesystem failures without
            falling back to default config values.
        """
        if not path.exists():
            return SettingsValidationResult(True, "Config file does not exist yet")

        try:
            with path.open("rb") as config_file:
                parsed = tomllib.load(config_file)
        except tomllib.TOMLDecodeError as exc:
            return SettingsValidationResult(False, redact_secret_text(str(exc)))
        except OSError as exc:
            return SettingsValidationResult(False, redact_secret_text(str(exc)))

        if not isinstance(parsed, dict):
            return SettingsValidationResult(False, "top-level TOML value must be a table")

        return SettingsValidationResult(True, "Config file TOML is valid")
