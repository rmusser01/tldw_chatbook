"""Adapter between the Settings RAG category and the profile system.

The RAG category edits the ACTIVE PROFILE (SP2a storage, SP2b resolution) —
never the deprecated AppRAGSearchConfig.rag.* keys, which the engine ignores.
All functions here are headless (no Textual imports) and testable.
"""
from __future__ import annotations

from typing import Optional

from loguru import logger

from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig, get_profile_manager
from tldw_chatbook.RAG_Search.simplified.active_config import _active_profile_id  # module seam
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
