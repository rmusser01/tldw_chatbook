"""Persistence and validation adapter for the Settings hub."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import re
from typing import Any

import toml

from ...config import load_cli_config_and_ensure_existence, save_setting_to_cli_config
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

    def load(self) -> dict[str, Any]:
        """Load CLI config through the existing config helper."""
        return deepcopy(load_cli_config_and_ensure_existence())

    def save_values(self, section: str, values: Mapping[str, Any]) -> bool:
        """Persist a group of values to one config section."""
        all_saved = True
        for key, value in values.items():
            if not save_setting_to_cli_config(section, key, value):
                all_saved = False
        return all_saved

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
