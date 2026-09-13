"""Load/validate/save model for the Settings "Network" category ([network]).

Mirrors settings_appearance_defaults.py: pure functions over a config
mapping so they unit-test without an app.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ...Utils.path_validation import validate_path_simple
from .settings_config_models import SettingsValidationResult

_TRUE_STRINGS = frozenset({"true", "1", "on"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "off"})


@dataclass(frozen=True)
class SettingsNetworkTLS:
    """The Network category's editable state.

    Attributes:
        mode: One of ``"verify"``, ``"off"``, ``"custom-ca"``, or
            ``"invalid"`` (a hand-edited config value the loader could not
            interpret; rendered as an error row, never saved).
        ca_bundle_path: Path of the custom CA bundle (used by
            ``"custom-ca"`` mode only).
        raw: The original config value, retained verbatim for the
            invalid-config error row.
    """

    mode: str
    ca_bundle_path: str = ""
    raw: object = None


def load_network_tls(app_config: Mapping[str, Any]) -> SettingsNetworkTLS:
    """Normalize ``[network] ssl_verify`` from a config mapping to UI state.

    Args:
        app_config: The app's config mapping (e.g. ``app_instance.app_config``).

    Returns:
        A :class:`SettingsNetworkTLS` whose ``mode`` reflects the effective
        TLS trust mode; uninterpretable values map to ``"invalid"``.
    """
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
    """Validate the Network category's editable state before saving.

    Args:
        values: The effective (pending-aware) UI state to validate.

    Returns:
        A :class:`SettingsValidationResult`; invalid modes and custom-CA
        paths that fail the shared path validator (or are missing or
        unreadable) are rejected with a user-actionable message.
    """
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
        try:
            # Shared path validator (traversal/security checks) rather than
            # raw filesystem probes (qodo PR #2223, findings 3 and 6).
            path = validate_path_simple(path, require_exists=True)
        except (ValueError, OSError) as exc:
            return SettingsValidationResult(False, f"CA bundle path invalid: {exc}")
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
    """Serialize UI state to the TOML value for ``[network] ssl_verify``.

    Args:
        values: The effective UI state.

    Returns:
        ``True``/``False`` for the boolean modes, or the expanded bundle
        path string for custom-CA mode.
    """
    if values.mode == "off":
        return False
    if values.mode == "custom-ca":
        return str(Path(values.ca_bundle_path.strip()).expanduser())
    return True


def build_network_save_sections(
    values: SettingsNetworkTLS,
) -> dict[str, dict[str, Any]]:
    """Build the config sections the Network category persists on save.

    Args:
        values: The validated UI state.

    Returns:
        A single-section mapping (``{"network": {"ssl_verify": ...}}`) in
        the shape ``SettingsConfigAdapter.save_sections`` consumes.
    """
    return {"network": {"ssl_verify": network_ssl_toml_value(values)}}
