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
_SCALAR_LIKE_TOML_PATTERN = re.compile(
    r"""(?x)
    (?:
        true|false|
        [+-]?\d+(?:\.\d+)?|
        \[.*\]|
        \{.*\}
    )
    """
)


def redact_secret_text(text: str) -> str:
    """Redact secret-looking assignment values from visible Settings output."""

    def _replace(match: re.Match[str]) -> str:
        return f"{match.group('key')}{match.group('sep')}<redacted>"

    return _SECRET_ASSIGNMENT_PATTERN.sub(_replace, str(text))


class SettingsConfigAdapter:
    """Narrow adapter over existing config helpers for Settings screens."""

    def load(self) -> dict[str, Any]:
        """Load CLI config through the existing config helper."""
        return deepcopy(load_cli_config_and_ensure_existence())

    def save_values(self, section: str, values: Mapping[str, Any]) -> bool:
        """Persist a group of values to one config section."""
        return all(
            save_setting_to_cli_config(section, key, value)
            for key, value in values.items()
        )

    def validate_raw_toml(self, text: str) -> SettingsValidationResult:
        """Validate TOML text and require a top-level table/mapping."""
        stripped_text = text.strip()

        try:
            parsed = toml.loads(text)
        except toml.TomlDecodeError as exc:
            if (
                len(stripped_text) >= 2
                and stripped_text[0] == stripped_text[-1]
                and stripped_text[0] in {"'", '"'}
            ) or _SCALAR_LIKE_TOML_PATTERN.fullmatch(stripped_text):
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
