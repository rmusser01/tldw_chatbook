"""The single "who am I" pointer for chats (task-442).

The user marks ONE user profile as their identity; its name feeds the
``{{user}}`` placeholder and message labels. Stored as a single config value
(single-active by construction, persists across sessions). "Persona" never
refers to the user in this app; the user-side concept is "user profile".
"""
from __future__ import annotations

from loguru import logger

from tldw_chatbook.config import get_cli_setting, save_setting_to_cli_config

_SECTION = "character_defaults"
_KEY = "active_user_profile"


def get_active_user_profile_pointer() -> str | None:
    """Return the configured active-profile name, or None when unset."""
    try:
        value = get_cli_setting(_SECTION, _KEY, None)
    except Exception:
        return None
    text = str(value).strip() if value is not None else ""
    return text or None


def set_active_user_profile(name: str) -> bool:
    """Point the active user profile at ``name``. Returns write success."""
    try:
        return bool(save_setting_to_cli_config(_SECTION, _KEY, str(name)))
    except Exception:
        logger.opt(exception=True).warning("Could not persist the active user profile.")
        return False


def clear_active_user_profile() -> bool:
    """Clear the pointer (no active user profile)."""
    try:
        return bool(save_setting_to_cli_config(_SECTION, _KEY, ""))
    except Exception:
        logger.opt(exception=True).warning("Could not clear the active user profile.")
        return False


def resolve_active_user_profile_name(service) -> str | None:
    """Resolve the pointer to a live profile name, or None.

    Unset pointer, dangling pointer (profile deleted/renamed), or ANY
    service failure -> None (treated as no-active; never raises). Cheap:
    one config read + one profile-list read.

    Args:
        service: the user-profile service (exposes ``list_user_profiles``).

    Returns:
        The active profile's name, or None.
    """
    pointer = get_active_user_profile_pointer()
    if pointer is None or service is None:
        return None
    try:
        lister = getattr(service, "list_user_profiles", None)
        if lister is None:
            return None
        for record in lister() or []:
            if isinstance(record, dict) and str(record.get("name") or "") == pointer:
                return pointer
    except Exception:
        logger.opt(exception=True).debug("Active user profile resolution failed.")
        return None
    return None
