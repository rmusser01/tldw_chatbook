"""Adapter between the Settings RAG category and the profile system.

The RAG category edits the ACTIVE PROFILE (SP2a storage, SP2b resolution) —
never the deprecated AppRAGSearchConfig.rag.* keys, which the engine ignores.
All functions here are headless (no Textual imports) and testable.
"""
from __future__ import annotations

from typing import Optional

from loguru import logger

from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig, get_profile_manager
# Both imported as module seams: tests monkeypatch `ad._active_profile_id` and
# `ad.set_active_profile` directly (unqualified name lookup at call time),
# the same pattern already used for `_active_profile_id` below.
from tldw_chatbook.RAG_Search.simplified.active_config import (
    _active_profile_id,
    set_active_profile,
)
from .settings_library_rag_defaults import SettingsLibraryRagDefaults


def _manager():
    return get_profile_manager()


def _active_profile() -> Optional[ProfileConfig]:
    return _manager().get_profile(_active_profile_id())


def load_rag_defaults_from_active_profile() -> SettingsLibraryRagDefaults:
    """Current active profile's search settings as the category dataclass."""
    profile = _active_profile()
    if profile is None:
        return SettingsLibraryRagDefaults()
    s = profile.rag_config.search
    return SettingsLibraryRagDefaults(
        default_search_mode=s.default_search_mode,
        default_top_k=int(s.default_top_k),
        fts_top_k=int(s.fts_top_k),
        vector_top_k=int(s.vector_top_k),
        hybrid_alpha=float(s.hybrid_alpha),
        score_threshold=float(s.score_threshold),
        include_citations=bool(s.include_citations),
        citation_style=s.citation_style,
        snippet_max_chars=int(s.snippet_max_chars),
        max_context_size=int(s.max_context_size),
    )


def apply_defaults_to_profile(
    profile: ProfileConfig, values: SettingsLibraryRagDefaults
) -> ProfileConfig:
    """Map the category dataclass onto the profile's SearchConfig (pure)."""
    s = profile.rag_config.search
    s.default_search_mode = values.default_search_mode
    s.default_top_k = int(values.default_top_k)
    s.fts_top_k = int(values.fts_top_k)
    s.vector_top_k = int(values.vector_top_k)
    s.hybrid_alpha = float(values.hybrid_alpha)
    s.score_threshold = float(values.score_threshold)
    s.include_citations = bool(values.include_citations)
    s.citation_style = values.citation_style
    s.snippet_max_chars = int(values.snippet_max_chars)
    s.max_context_size = int(values.max_context_size)
    return profile


def save_rag_defaults_to_active_profile(
    values: SettingsLibraryRagDefaults,
) -> tuple[bool, str]:
    """Persist values into the active profile's file. (False, "builtin") if read-only."""
    profile = _active_profile()
    if profile is None:
        return False, "no-active-profile"
    if profile.read_only:
        return False, "builtin"
    try:
        _manager().save_profile(apply_defaults_to_profile(profile, values))
        return True, ""
    except Exception as e:  # save must never crash the settings screen
        logger.error(f"Saving RAG defaults to active profile failed: {e}")
        return False, str(e)


def active_profile_info() -> dict:
    profile = _active_profile()
    if profile is None:
        return {"id": _active_profile_id(), "name": "(missing)", "read_only": True}
    return {"id": profile.id, "name": profile.name, "read_only": bool(profile.read_only)}


def list_profiles_grouped() -> dict:
    """All profiles grouped by builtin/user for the profile-manager region.

    Returns:
        ``{"builtin": [{"id","name"}...], "user": [...], "active_id": str}``,
        each group name-sorted.
    """
    manager = _manager()
    builtin: list[dict] = []
    user: list[dict] = []
    for profile_id in manager.list_profiles():
        profile = manager.get_profile(profile_id)
        if profile is None:
            continue
        entry = {"id": profile.id, "name": profile.name}
        (builtin if profile.read_only else user).append(entry)
    builtin.sort(key=lambda p: p["name"])
    user.sort(key=lambda p: p["name"])
    return {"builtin": builtin, "user": user, "active_id": _active_profile_id()}


def activate_profile(profile_id: str) -> tuple[bool, str]:
    """Point the active-profile pointer at `profile_id` (SP2b `set_active_profile`).

    Callers run this off-thread (it writes to the CLI config file and resets
    the shared RAG service). Any failure -- invalid/unsafe id, config write
    error -- is converted into ``(False, reason)`` rather than raised.
    """
    try:
        set_active_profile(profile_id)
        return True, ""
    except Exception as e:  # pointer flip must never crash the settings screen
        logger.error(f"activate_profile({profile_id!r}) failed: {e}")
        return False, str(e)


def clone_profile_as(source_id: str, new_name: str) -> tuple[bool, str]:
    """Clone `source_id` (builtin or user) into a new writable profile.

    Returns:
        ``(True, new_profile_id)`` on success, ``(False, reason)`` if
        `source_id` does not resolve to a registered profile.
    """
    try:
        clone = _manager().clone_profile(source_id, new_name)
        return True, clone.id
    except ValueError as e:
        return False, str(e)
    except Exception as e:  # manager does raw file I/O (OSError/PermissionError
        # possible); this runs inside a thread @work with exit_on_error=True,
        # so an uncaught exception here crashes the whole app -- must never raise.
        logger.error(f"clone_profile_as({source_id!r}, {new_name!r}) failed: {e}")
        return False, str(e)


def rename_user_profile(profile_id: str, new_name: str) -> tuple[bool, str]:
    """Rename a user profile's display name. (False, reason) for a builtin/missing id."""
    try:
        _manager().rename_profile(profile_id, new_name)
        return True, ""
    except ValueError as e:
        return False, str(e)
    except Exception as e:  # see clone_profile_as: raw file I/O, thread @work
        logger.error(f"rename_user_profile({profile_id!r}, {new_name!r}) failed: {e}")
        return False, str(e)


def delete_user_profile(profile_id: str) -> tuple[bool, str]:
    """Delete a user profile. (False, reason) for a builtin id or on failure."""
    try:
        deleted = _manager().delete_profile(profile_id)
        return (True, "") if deleted else (False, "not-found")
    except ValueError as e:
        return False, str(e)
    except Exception as e:  # see clone_profile_as: raw file I/O, thread @work
        logger.error(f"delete_user_profile({profile_id!r}) failed: {e}")
        return False, str(e)
