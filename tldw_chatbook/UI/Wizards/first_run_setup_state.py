"""Pure state contracts for the first-run setup wizard.

No Textual imports, no I/O — every function is a pure transform over the
in-memory app config, mirroring Chat/console_onboarding_state.py. The wizard
Screen owns rendering and persistence; this module owns every decision.
"""

from __future__ import annotations

from typing import Any, Mapping

WIZARD_STATE_SECTION = "first_run"
SETUP_STARTED_KEY = "setup_started"
SETUP_COMPLETED_KEY = "setup_completed"

# Endpoint keys a local provider may use (mirrors
# Chat/local_server_discovery._ENDPOINT_CONFIG_KEYS).
_ENDPOINT_KEYS = ("api_url", "api_base_url", "api_base", "base_url", "api_endpoint", "endpoint")

_PLACEHOLDER_MARKERS = ("<", ">")


def coerce_wizard_flag(raw: Any) -> bool:
    """Tolerantly parse a persisted wizard flag.

    Args:
        raw: Whatever the TOML loader produced for the key.

    Returns:
        True only for bool True, int 1, or the string "true" (case-insensitive).
    """
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return raw == 1
    if isinstance(raw, str):
        return raw.strip().lower() == "true"
    return False


def _is_real_secret(value: Any) -> bool:
    """A non-empty string that is not a <PLACEHOLDER> template value."""
    if not isinstance(value, str) or not value.strip():
        return False
    stripped = value.strip()
    return not (stripped.startswith(_PLACEHOLDER_MARKERS[0]) and stripped.endswith(_PLACEHOLDER_MARKERS[1]))


def any_provider_configured(
    app_config: Mapping[str, object], environ: Mapping[str, str]
) -> bool:
    """Return True when any provider has usable credentials or an endpoint.

    Walks the NESTED ``app_config["api_settings"]`` dict. Do not replace this
    with config.get_detected_api_providers(): that helper matches
    "api_settings.<p>" as a top-level key and always returns [].
    """
    api_settings = app_config.get("api_settings")
    if not isinstance(api_settings, Mapping):
        return False
    for settings in api_settings.values():
        if not isinstance(settings, Mapping):
            continue
        if _is_real_secret(settings.get("api_key")):
            return True
        env_var = settings.get("api_key_env_var")
        if isinstance(env_var, str) and env_var.strip() and environ.get(env_var.strip()):
            return True
        for endpoint_key in _ENDPOINT_KEYS:
            if _is_real_secret(settings.get(endpoint_key)):
                return True
    return False


def _wizard_flag(app_config: Mapping[str, object], key: str) -> bool:
    section = app_config.get(WIZARD_STATE_SECTION)
    if not isinstance(section, Mapping):
        return False
    return coerce_wizard_flag(section.get(key))


def should_offer_wizard(
    app_config: Mapping[str, object], environ: Mapping[str, str]
) -> bool:
    """Auto-offer once: no wizard state keys AND nothing configured."""
    if _wizard_flag(app_config, SETUP_STARTED_KEY):
        return False
    if _wizard_flag(app_config, SETUP_COMPLETED_KEY):
        return False
    return not any_provider_configured(app_config, environ)


def should_show_resume_toast(
    app_config: Mapping[str, object], environ: Mapping[str, str]
) -> bool:
    """Started but never finished: point at Settings, never re-push."""
    return _wizard_flag(app_config, SETUP_STARTED_KEY) and not _wizard_flag(
        app_config, SETUP_COMPLETED_KEY
    )
