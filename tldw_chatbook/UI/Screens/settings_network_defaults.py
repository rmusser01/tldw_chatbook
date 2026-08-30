"""Load/validate/save model for the Settings "Network" category ([network]).

Mirrors settings_appearance_defaults.py: pure functions over a config
mapping so they unit-test without an app.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .settings_config_models import SettingsValidationResult

_TRUE_STRINGS = frozenset({"true", "1", "on"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "off"})


@dataclass(frozen=True)
class SettingsNetworkTLS:
    mode: str  # "verify" | "off" | "custom-ca" | "invalid"
    ca_bundle_path: str = ""
    raw: object = None


def load_network_tls(app_config: Mapping[str, Any]) -> SettingsNetworkTLS:
    network = app_config.get("network") if isinstance(app_config, Mapping) else None
    value = network.get("ssl_verify", True) if isinstance(network, Mapping) else True
    if value is True:
        return SettingsNetworkTLS("verify")
    if value is False:
        return SettingsNetworkTLS("off")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if not lowered or lowered in _TRUE_STRINGS:
            return SettingsNetworkTLS("verify", raw=value)
        if lowered in _FALSE_STRINGS:
            return SettingsNetworkTLS("off", raw=value)
        path = Path(value.strip()).expanduser()
        if path.is_file():
            return SettingsNetworkTLS("custom-ca", ca_bundle_path=str(path), raw=value)
        return SettingsNetworkTLS("invalid", ca_bundle_path=value.strip(), raw=value)
    return SettingsNetworkTLS("invalid", raw=value)


def validate_network_tls(values: SettingsNetworkTLS) -> SettingsValidationResult:
    if values.mode == "invalid":
        return SettingsValidationResult(
            False, "ssl_verify value is invalid — choose a mode and save."
        )
    if values.mode == "custom-ca":
        raw = values.ca_bundle_path.strip()
        if not raw:
            return SettingsValidationResult(
                False, "Custom CA bundle requires a file path."
            )
        path = Path(raw).expanduser()
        if not path.is_file():
            return SettingsValidationResult(
                False, f"CA bundle file not found: {path}"
            )
        if not os.access(path, os.R_OK):
            return SettingsValidationResult(
                False, f"CA bundle file is not readable: {path}"
            )
    return SettingsValidationResult(True, "")


def network_ssl_toml_value(values: SettingsNetworkTLS) -> bool | str:
    if values.mode == "off":
        return False
    if values.mode == "custom-ca":
        return str(Path(values.ca_bundle_path.strip()).expanduser())
    return True


def build_network_save_sections(
    values: SettingsNetworkTLS,
) -> dict[str, dict[str, Any]]:
    return {"network": {"ssl_verify": network_ssl_toml_value(values)}}
