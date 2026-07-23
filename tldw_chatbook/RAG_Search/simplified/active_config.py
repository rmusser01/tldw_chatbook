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
from .config import RAGConfig, _normalized_type_setting
from ..config_profiles import get_profile_manager, ProfileConfig, _slugify
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
        # Optional_deps only exposes a cheap find_spec-based installed-probe
        # (embeddings_rag_deps_installed / _embeddings_rag_available), not an
        # accessor for the imported torch module itself -- we still need the
        # real module to call torch.cuda.is_available()/torch.backends.mps,
        # so there's no helper to route through here. The try/except mirrors
        # the same pattern used in embeddings_wrapper.py's device auto-detect
        # and is already exception-safe (falls back to "cpu").
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
    # vector_store.type: resolve_active_rag_config() deep-copies an already-
    # constructed profile's RAGConfig (copy.deepcopy does NOT re-run
    # VectorStoreConfig.__post_init__), so RAG_VECTOR_STORE must be applied
    # explicitly here -- __post_init__ only ever ran once, at profile-save
    # time, and cannot see env vars set afterward. Normalized the same way
    # default_vector_store_type() does (stripped/lowercased; "auto" means
    # "no override").
    env_vector_store = _normalized_type_setting(os.getenv("RAG_VECTOR_STORE"))
    if env_vector_store:
        config.vector_store.type = env_vector_store

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
    """Resolve the single source-of-truth RAG config: active profile + env overlay.

    Reads the active-profile pointer, deep-copies that profile's stored
    ``rag_config`` (falling back to the ``hybrid_basic`` builtin, then a bare
    ``RAGConfig()``, if the pointer names a profile that no longer exists),
    and applies the env/explicit-arg override layer on top. Both the search
    path (``RAGConfig.from_settings``) and the ingestion path
    (``get_shared_rag_service``) route through this function so they never
    resolve divergent configs for the same active profile.

    Args:
        override_embedding_model: Explicit embedding model, taking priority
            over both the profile's stored value and ``RAG_EMBEDDING_MODEL``.
        override_persist_dir: Explicit vector-store persist directory, taking
            priority over both the profile's stored value and
            ``RAG_PERSIST_DIR``.

    Returns:
        A fresh ``RAGConfig`` (safe to mutate -- never the profile's own
        stored object) with all applicable env overrides applied.
    """
    active = _active_profile_id()
    mgr = _manager()
    profile = mgr.get_profile(active) or mgr.get_profile(DEFAULT_PROFILE)
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

    Args:
        profile_id: The profile id to activate. Must be a non-empty string
            that is already a safe filesystem slug (i.e. equal to
            ``_slugify(profile_id)`` -- the same constraint
            ``ConfigProfileManager`` enforces on stored profile ids).

    Returns:
        None.

    Raises:
        ValueError: If ``profile_id`` is not a non-empty string matching a
            safe slug (e.g. empty, ``None``, or containing path-traversal /
            non-slug characters like ``"../x"``).
    """
    if not isinstance(profile_id, str) or not profile_id or profile_id != _slugify(profile_id):
        raise ValueError(
            f"set_active_profile: invalid profile_id {profile_id!r}; must be a "
            "non-empty, already-slugified string (see config_profiles._slugify)"
        )
    wrote = save_setting_to_cli_config("rag.service", "profile", profile_id)
    if not wrote:
        logger.warning(
            f"set_active_profile: failed to write active-profile pointer for "
            f"{profile_id!r}; leaving the current pointer and shared service "
            "untouched (nothing to reset since the pointer didn't change)"
        )
        return
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

    Returns:
        The new profile's id (``"imported_settings"``) on the run that
        creates and activates it; ``None`` when it already existed (no-op,
        after healing the active pointer if needed) or when import failed
        (logged, swallowed).
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
                                description="Snapshot of your active RAG profile (plus any RAG_* env "
                                            "overrides) captured on first run; edit freely.",
                                profile_type="custom", rag_config=snapshot)
        mgr.save_profile(profile)
        set_active_profile(_IMPORTED_ID)
        return _IMPORTED_ID
    except Exception as e:
        logger.warning(f"ensure_imported_profile: first-run import failed, continuing without it: {e}")
        return None
