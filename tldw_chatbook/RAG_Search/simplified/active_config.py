"""Active-profile config resolution — the single config source the RAG engine reads.

resolve_active_rag_config() = the active profile's rag_config (deep copy) with
the env-override layer applied. BOTH the search path (RAGConfig.from_settings)
and the ingestion path (get_shared_rag_service) route through it, so ingestion
and search never use divergent configs for the same active profile.
See Docs/superpowers/specs/2026-07-21-rag-profile-system-design.md §5.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Optional, Union

from loguru import logger

from tldw_chatbook.config import get_cli_setting, save_setting_to_cli_config
from .config import RAGConfig
from ..config_profiles import get_profile_manager, ProfileConfig
from ..ingestion_indexing import reset_shared_rag_service

DEFAULT_PROFILE = "hybrid_basic"
_IMPORTED_ID = "imported_settings"


def _manager():
    return get_profile_manager()


def _active_profile_id() -> str:
    """The active-profile pointer: [rag.service].profile (reused, single pointer)."""
    try:
        svc = get_cli_setting("rag", "service", {}) or {}
        if isinstance(svc, dict) and svc.get("profile"):
            return str(svc["profile"])
    except Exception as e:
        logger.debug(f"Could not read active profile pointer: {e}")
    return DEFAULT_PROFILE


def _apply_env_overrides(config: RAGConfig,
                         override_embedding_model: Optional[str] = None,
                         override_persist_dir: Optional[Union[str, Path]] = None) -> RAGConfig:
    """Apply the env / explicit-arg override layer onto `config` in place.

    This is the SAME layer RAGConfig.from_settings applied — moved here so both
    resolution paths apply env identically (parity). NOTE: this NO LONGER reads
    the deprecated AppRAGSearchConfig.rag.* value keys — the profile is the base.
    """
    e = config.embedding
    e.model = override_embedding_model or os.getenv("RAG_EMBEDDING_MODEL") or e.model
    dev = os.getenv("RAG_DEVICE") or e.device
    if dev == "auto":
        try:
            import torch
            e.device = ("cuda" if torch.cuda.is_available()
                        else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                        else "cpu")
        except ImportError:
            e.device = "cpu"
    else:
        e.device = dev
    cache = os.getenv("RAG_EMBEDDING_CACHE_SIZE")
    if cache:
        e.cache_size = int(cache)
    e.api_key = os.getenv("OPENAI_API_KEY") or get_cli_setting("API", "openai_api_key") or e.api_key
    e.base_url = os.getenv("RAG_EMBEDDING_BASE_URL") or e.base_url
    persist = override_persist_dir or os.getenv("RAG_PERSIST_DIR")
    if persist:
        config.vector_store.persist_directory = Path(persist)
    # vector_store.type: RAG_VECTOR_STORE env is honored by VectorStoreConfig.__post_init__

    # Chunking overrides
    chunk_size = os.getenv("RAG_CHUNK_SIZE")
    if chunk_size:
        config.chunking.chunk_size = int(chunk_size)
    chunk_overlap = os.getenv("RAG_CHUNK_OVERLAP")
    if chunk_overlap:
        config.chunking.chunk_overlap = int(chunk_overlap)

    # Search overrides
    top_k = os.getenv("RAG_TOP_K")
    if top_k:
        config.search.default_top_k = int(top_k)
    config.search.default_search_mode = os.getenv("RAG_SEARCH_MODE") or config.search.default_search_mode

    # Pipeline overrides
    config.pipeline.default_pipeline = os.getenv("RAG_DEFAULT_PIPELINE") or config.pipeline.default_pipeline

    return config


def resolve_active_rag_config(override_embedding_model: Optional[str] = None,
                              override_persist_dir: Optional[Union[str, Path]] = None) -> RAGConfig:
    """The active profile's rag_config (deep copy) + env overlay — the single source."""
    active = _active_profile_id()
    profile = _manager().get_profile(active) or _manager().get_profile(DEFAULT_PROFILE)
    base = copy.deepcopy(profile.rag_config) if profile else RAGConfig()
    return _apply_env_overrides(base, override_embedding_model, override_persist_dir)


def set_active_profile(profile_id: str) -> None:
    """Point [rag.service].profile at `profile_id` and drop the shared service.

    The next resolve_active_rag_config()/get_shared_rag_service() rebuilds on the
    new profile (and, via SP1, its fingerprinted collection). An in-flight worker
    keeps its own service reference — the reset never yanks a running op; it only
    clears the singleton so the NEXT caller rebuilds. The (potentially expensive)
    embedding-model reload is the caller's concern to run off-thread (SP3 UI).

    NOTE: save_setting_to_cli_config(section, key, value) nests via the
    `section` argument (it handles dotted sections like "api_settings.openai"),
    so the pointer write below is section="rag.service", key="profile" — this
    lands at TOML path [rag.service].profile, exactly what the read side
    (_active_profile_id() -> get_cli_setting("rag", "service", {}).get("profile"))
    resolves. section="rag", key="service.profile" would land at the WRONG path
    ([rag]["service.profile"], a literal dotted key) and silently break the
    active-profile pointer.
    """
    save_setting_to_cli_config("rag.service", "profile", profile_id)
    reset_shared_rag_service()


def ensure_imported_profile() -> Optional[str]:
    """On first run, capture the currently-resolved RAG config into a writable
    'Imported settings' profile and set it active. Idempotent (returns None if it
    already exists). The captured config's SP1 fingerprint matches what SP1 adopts
    the legacy collection under, so the user keeps their index on upgrade.

    Self-healing: existence of the profile is not enough to consider first-run
    import "done" -- if a previous run persisted the profile but failed before
    (or otherwise never got to) activating it, that leaves it created-but-never-
    active forever with no retry. So every call also checks the active pointer
    and (re)activates the imported profile if it isn't already active.

    Exception-safe: any failure here must never block RAG service creation, so
    every error is caught and logged, returning None (as if already imported /
    nothing to do) rather than propagating.
    """
    try:
        mgr = _manager()
        existing = mgr.get_profile(_IMPORTED_ID)
        if existing is not None:
            if _active_profile_id() != _IMPORTED_ID:
                set_active_profile(_IMPORTED_ID)  # heal a half-done first run
            return None
        # Snapshot the resolved config (active pointer may name a builtin default today).
        snapshot = resolve_active_rag_config()
        profile = ProfileConfig(id=_IMPORTED_ID, name="Imported settings",
                                description="Captured from your existing RAG configuration on first run.",
                                profile_type="custom", rag_config=snapshot)
        mgr.save_profile(profile)
        set_active_profile(_IMPORTED_ID)
        return _IMPORTED_ID
    except Exception as e:
        logger.warning(f"ensure_imported_profile: first-run import failed, continuing without it: {e}")
        return None
