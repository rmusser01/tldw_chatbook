"""The single "who am I" pointer for chats (task-442).

The user marks ONE user profile as their identity; its name feeds the
``{{user}}`` placeholder and message labels. Stored as a single config value
(single-active by construction, persists across sessions). "Persona" never
refers to the user in this app; the user-side concept is "user profile".
"""
from __future__ import annotations

from typing import Protocol

from loguru import logger

from tldw_chatbook.config import get_cli_setting, save_setting_to_cli_config

_SECTION = "character_defaults"
_KEY = "active_user_profile"


class _UserProfileLister(Protocol):
    """Minimal shape the resolver needs from a user-profile service (kept
    local so this module stays free of service-layer imports)."""

    def list_user_profiles(self, active_only: bool = False) -> list[dict]: ...


def get_active_user_profile_pointer() -> str | None:
    """Read the configured active-user-profile pointer.

    Returns:
        The configured profile name (whitespace-normalized), or ``None``
        when unset/blank or when the config read fails.
    """
    try:
        value = get_cli_setting(_SECTION, _KEY, None)
    except Exception:
        return None
    text = str(value).strip() if value is not None else ""
    return text or None


def set_active_user_profile(name: str) -> bool:
    """Point the active user profile at ``name``.

    Args:
        name: The profile name to mark as "who I am".

    Returns:
        True when the config write persisted, False otherwise.
    """
    try:
        return bool(save_setting_to_cli_config(_SECTION, _KEY, str(name)))
    except Exception:
        logger.opt(exception=True).warning("Could not persist the active user profile.")
        return False


def clear_active_user_profile() -> bool:
    """Clear the pointer (no active user profile).

    Returns:
        True when the config write persisted, False otherwise.
    """
    try:
        return bool(save_setting_to_cli_config(_SECTION, _KEY, ""))
    except Exception:
        logger.opt(exception=True).warning("Could not clear the active user profile.")
        return False


def resolve_active_user_profile_name(service: _UserProfileLister | None) -> str | None:
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
