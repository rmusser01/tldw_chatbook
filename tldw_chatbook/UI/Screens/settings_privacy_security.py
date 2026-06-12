"""Redacted Settings Privacy & Security posture helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


SENSITIVE_CONFIG_EXACT_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "api-key",
        "api_token",
        "auth_token",
        "access_token",
        "refresh_token",
        "client_secret",
        "secret_key",
        "secret",
        "token",
        "password",
    }
)
SENSITIVE_CONFIG_KEY_PATTERNS = (
    "api_key",
    "apikey",
    "api-key",
    "_key",
    "_token",
    "_secret",
    "_password",
)


@dataclass(frozen=True)
class SettingsPrivacyPosture:
    """Redacted privacy posture derived from app configuration.

    Attributes:
        encryption_enabled: Whether config encryption is currently enabled.
        sensitive_config_fields: Count of configured sensitive config fields.
        provider_env_present: Count of configured provider env vars that are set.
        provider_env_missing: Count of configured provider env vars that are missing.
        provider_env_configured: Total configured provider credential env-var references.
        provider_config_secrets: Count of configured provider secrets stored in config.
        redaction_active: Whether visible privacy output redacts raw secret values.
        data_boundary: User-facing local data boundary summary.
        server_boundary: User-facing server token boundary summary.
    """

    encryption_enabled: bool
    sensitive_config_fields: int
    provider_env_present: int
    provider_env_missing: int
    provider_env_configured: int
    provider_config_secrets: int
    redaction_active: bool = True
    data_boundary: str = (
        "local data stays local unless explicit server handoff or sync is enabled"
    )
    server_boundary: str = "server tokens are reported as configured/missing only"


def build_settings_privacy_posture(
    app_config: object,
    *,
    environ: Mapping[str, str] | None = None,
) -> SettingsPrivacyPosture:
    """Build a redacted Privacy & Security posture from config and environment.

    Args:
        app_config: The application configuration mapping to inspect.
        environ: Optional environment mapping. Defaults to ``os.environ``.

    Returns:
        A posture object containing only counts and status booleans.
    """

    env = os.environ if environ is None else environ
    encryption_config = (
        app_config.get("encryption", {}) if isinstance(app_config, Mapping) else {}
    )
    encryption_enabled = (
        bool(encryption_config.get("enabled"))
        if isinstance(encryption_config, Mapping)
        else False
    )
    env_present, env_missing, env_total = _provider_env_var_status_counts(
        app_config,
        env,
    )
    return SettingsPrivacyPosture(
        encryption_enabled=encryption_enabled,
        sensitive_config_fields=_sensitive_config_field_count(app_config),
        provider_env_present=env_present,
        provider_env_missing=env_missing,
        provider_env_configured=env_total,
        provider_config_secrets=_provider_config_secret_count(app_config),
    )


def build_privacy_posture_rows(posture: SettingsPrivacyPosture) -> tuple[str, ...]:
    """Return stable redacted rows for visible Privacy & Security status.

    Args:
        posture: Redacted posture values to render as user-facing status rows.

    Returns:
        Tuple of stable, redacted status strings safe to display in Settings.
    """

    return (
        f"Config encryption: {'enabled' if posture.encryption_enabled else 'disabled'}",
        "Redaction: active; raw secret values hidden"
        if posture.redaction_active
        else "Redaction: unavailable",
        f"Sensitive config fields: {posture.sensitive_config_fields} present",
        (
            "Provider env vars: "
            f"{posture.provider_env_present} present / "
            f"{posture.provider_env_missing} missing / "
            f"{posture.provider_env_configured} configured"
        ),
        f"Provider config secrets: {posture.provider_config_secrets} present",
        f"Data boundary: {posture.data_boundary}",
        f"Server boundary: {posture.server_boundary}",
        "Privacy safety: no secret values were printed or written.",
    )


def _is_sensitive_config_key(key: object) -> bool:
    key_text = str(key).strip().lower()
    if not key_text or key_text.endswith("_env_var"):
        return False
    return key_text in SENSITIVE_CONFIG_EXACT_KEYS or any(
        key_text.endswith(pattern) for pattern in SENSITIVE_CONFIG_KEY_PATTERNS
    )


def _is_configured_secret_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        value_text = value.strip()
        if not value_text or value_text in {"None", "null"}:
            return False
        if value_text.startswith("<") and value_text.endswith(">"):
            return False
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return True
    return False


def _iter_config_leaf_values(value: object):
    if not isinstance(value, Mapping):
        return
    for key, child_value in value.items():
        if isinstance(child_value, Mapping):
            yield from _iter_config_leaf_values(child_value)
        else:
            yield key, child_value


def _sensitive_config_field_count(app_config: object) -> int:
    return sum(
        1
        for key, value in _iter_config_leaf_values(app_config)
        if _is_sensitive_config_key(key) and _is_configured_secret_value(value)
    )


def _provider_env_var_status_counts(
    app_config: object,
    environ: Mapping[str, str],
) -> tuple[int, int, int]:
    if not isinstance(app_config, Mapping):
        return 0, 0, 0
    api_settings = app_config.get("api_settings", {})
    if not isinstance(api_settings, Mapping):
        return 0, 0, 0
    present = 0
    missing = 0
    for provider_config in api_settings.values():
        if not isinstance(provider_config, Mapping):
            continue
        for key, value in provider_config.items():
            key_text = str(key).strip().lower()
            env_var = str(value or "").strip()
            if not key_text.endswith("_env_var") or not env_var:
                continue
            if environ.get(env_var):
                present += 1
            else:
                missing += 1
    return present, missing, present + missing


def _provider_config_secret_count(app_config: object) -> int:
    if not isinstance(app_config, Mapping):
        return 0
    api_settings = app_config.get("api_settings", {})
    if not isinstance(api_settings, Mapping):
        return 0
    count = 0
    for provider_config in api_settings.values():
        if not isinstance(provider_config, Mapping):
            continue
        count += sum(
            1
            for key, value in _iter_config_leaf_values(provider_config)
            if _is_sensitive_config_key(key) and _is_configured_secret_value(value)
        )
    return count
