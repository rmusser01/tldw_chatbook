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
from .config import RAGConfig, _normalized_type_setting, validate_chroma_persist_directory
from ..config_profiles import get_profile_manager, ProfileConfig, _slugify
from ..ingestion_indexing import reset_shared_rag_service
from ..reranker import RerankingConfig

DEFAULT_PROFILE = "hybrid_basic"
_IMPORTED_ID = "imported_settings"

# Legacy TOML keys (task-495): non-index-determining query-time settings a
# user may have hand-set under the deprecated [AppRAGSearchConfig.rag.search]
# / [AppRAGSearchConfig.rag.processor] sections. These are the ONLY legacy
# keys ever merged into the first-run "Imported settings" snapshot -- NONE of
# embedding.model/max_length, any chunking.* field, or vector_store.
# distance_metric (collection_fingerprint._index_fields' exhaustive input
# set) are read here, so merging can never move
# collection_fingerprint.fingerprint_collection()'s output away from what
# SP1 adopts the legacy collection under.
_LEGACY_SEARCH_KEYS = ("default_top_k", "score_threshold", "include_citations")
_LEGACY_PROCESSOR_KEYS = ("enable_reranking", "reranker_model", "reranker_top_k")


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


def _first_run_import_done() -> bool:
    """The durable first-run-import marker: [rag.service].first_run_import_done.

    True once ensure_imported_profile() has deliberately activated
    ``imported_settings`` — either by completing a fresh import or by healing
    a half-done one (see ``_mark_first_run_import_done``, called at both
    sites). This is the proof `ensure_imported_profile`'s healing branch uses
    to tell "the pointer was never successfully settled" apart from "the
    pointer is settled, whatever it currently names" -- unlike comparing
    config *contents*, it can never be fooled by a customization it doesn't
    happen to check for (task-639 review: the prior fingerprint + allow-list
    heuristic missed several Settings-screen-editable fields, e.g.
    ``search.hybrid_alpha``, and could delete a genuinely customized
    profile).
    """
    try:
        svc = get_cli_setting("rag", "service", {}) or {}
        if isinstance(svc, dict):
            return bool(svc.get("first_run_import_done"))
    except Exception as e:
        logger.debug(f"Could not read first_run_import_done marker: {e}")
    return False


def _mark_first_run_import_done() -> None:
    """Persist [rag.service].first_run_import_done = True.

    Called at both places ``ensure_imported_profile`` activates
    ``imported_settings`` (a fresh import, and healing a half-done run) so
    that, from then on, the pointer is never second-guessed again -- even if
    the user later switches to a different profile, or back to the default
    builtin itself. Best-effort: a failure here is logged and swallowed, same
    as the rest of ``ensure_imported_profile`` -- it must never block RAG
    service creation. A failed write just means the next call may re-evaluate
    the healing branch once more, which is safe (idempotent), not corrupting.
    """
    try:
        save_setting_to_cli_config("rag.service", "first_run_import_done", True)
    except Exception as e:
        logger.debug(f"Could not persist first_run_import_done marker: {e}")


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
        # Routed through the SAME validate_chroma_persist_directory() the two
        # Chroma client-construction sites (vector_store.py, collection_
        # indexes.py) use, not a bare Path(persist) -- a persist_directory
        # PRODUCER that skips this normalization (e.g. leaving a literal "~"
        # unexpanded) would hand the consumers a path string that diverges
        # from what they'd compute themselves, defeating chromadb's
        # SharedSystemClient per-path client cache one hop earlier than the
        # task-482 fix closed it at. See validate_chroma_persist_directory's
        # docstring.
        config.vector_store.persist_directory = validate_chroma_persist_directory(persist)
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


def _hand_set_legacy_query_time_keys() -> dict:
    """Allow-listed query-time keys the user literally set in the deprecated
    [AppRAGSearchConfig.rag.search] / [AppRAGSearchConfig.rag.processor]
    TOML sections.

    "Hand-set" means present as a key in the raw loaded config dict -- NOT
    "differs from the dataclass default". An explicit value that happens to
    equal the default is still honored; an absent key is never synthesized
    from a default. Only keys in `_LEGACY_SEARCH_KEYS` / `_LEGACY_PROCESSOR_
    KEYS` are ever considered (see their docstring for why: they're exactly
    the non-index-determining fields).

    Returns:
        Dict of `{SearchConfig field name: value}` for every allow-listed
        key present in the user's config; empty when none are set or the
        legacy section can't be read (never raises).
    """
    try:
        rag_section = get_cli_setting("AppRAGSearchConfig", "rag", {}) or {}
        if not isinstance(rag_section, dict):
            return {}
    except Exception as e:
        logger.debug(f"Could not read legacy AppRAGSearchConfig.rag section: {e}")
        return {}

    found: dict = {}
    search = rag_section.get("search")
    if isinstance(search, dict):
        for key in _LEGACY_SEARCH_KEYS:
            if key in search:
                found[key] = search[key]
    processor = rag_section.get("processor")
    if isinstance(processor, dict):
        for key in _LEGACY_PROCESSOR_KEYS:
            if key in processor:
                found[key] = processor[key]
    return found


def _has_legacy_rag_config_material() -> bool:
    """True when the user has ANY hand-set ``[AppRAGSearchConfig.rag.*]``
    content at all -- the signal (task-635) that they're upgrading from
    before the profile system existed, not installing fresh.

    Deliberately broader than ``_hand_set_legacy_query_time_keys`` (which
    only surfaces the allow-listed query-time keys that get MERGED into the
    first-run snapshot, per task-495): a legacy user who hand-set
    embedding/chunking/vector_store keys under the deprecated section --
    never merged, see that function's module-level docstring -- still has a
    real pre-profile collection worth preserving SP1 fingerprint continuity
    for, even though none of those specific keys end up in the imported
    snapshot. Any non-empty ``[AppRAGSearchConfig.rag]`` subsection is
    therefore treated as "genuine legacy material", regardless of which
    keys it contains.

    A genuinely fresh install has no ``[AppRAGSearchConfig.rag]`` section at
    all (a config.toml that never had RAG settings hand-edited before the
    profile system shipped), so this returns False and
    ``ensure_imported_profile()`` leaves that user on the default builtin
    profile instead of auto-creating and silently activating "Imported
    settings" underneath them.

    Returns:
        True when ``[AppRAGSearchConfig.rag]`` resolves to a non-empty
        dict; False when it's absent/empty or unreadable (never raises).
    """
    try:
        rag_section = get_cli_setting("AppRAGSearchConfig", "rag", {}) or {}
        return isinstance(rag_section, dict) and bool(rag_section)
    except Exception as e:
        logger.debug(f"Could not probe for legacy AppRAGSearchConfig.rag section: {e}")
        return False


def _merge_legacy_query_time_keys(config: RAGConfig) -> dict:
    """Merge hand-set legacy query-time keys onto `config.search`, in place.

    Only ever called from ensure_imported_profile()'s one-time first-run
    snapshot -- this is a migration convenience, not part of the ongoing
    resolve_active_rag_config() resolution path (legacy AppRAGSearchConfig.
    rag.* value keys are otherwise dead per the module docstring above).

    Args:
        config: The RAGConfig to mutate (the first-run snapshot).

    Returns:
        The dict of hand-set legacy keys that were found (possibly empty),
        so callers that need to act on a key `SearchConfig` alone can't
        represent (e.g. reranking presence -- see `ensure_imported_profile`)
        don't have to re-read the legacy TOML section a second time.
    """
    legacy = _hand_set_legacy_query_time_keys()
    for key, value in legacy.items():
        try:
            setattr(config.search, key, value)
        except Exception as e:
            logger.debug(f"Could not apply legacy query-time key {key}={value!r}: {e}")
    return legacy


def ensure_imported_profile() -> Optional[str]:
    """On first run, capture the currently-resolved RAG config into a writable
    'Imported settings' profile and set it active -- but ONLY for a user
    upgrading from before the profile system existed (task-635: has any
    hand-set ``[AppRAGSearchConfig.rag.*]`` material, see
    ``_has_legacy_rag_config_material``). A genuinely fresh install has no
    legacy collection to preserve continuity for, so this is a no-op for
    them and they stay on the default builtin profile (``hybrid_basic``)
    with no profile created and no active-pointer write -- previously this
    ran unconditionally on first touch, which silently created and
    activated "Imported settings" underneath a brand-new user the first
    time anything called ``get_shared_rag_service()`` (e.g. Backfill),
    flipping their active profile to one they never chose.

    Idempotent (returns None if the profile already exists). For a genuine
    upgrader, the captured config's SP1 fingerprint matches what SP1 adopts
    the legacy collection under, so the user keeps their index on upgrade.

    Also merges (task-495) any hand-set, non-index-determining legacy
    query-time keys from the deprecated [AppRAGSearchConfig.rag.search] /
    [AppRAGSearchConfig.rag.processor] TOML sections (top_k, score_threshold,
    citations, reranking) onto the snapshot, so a user's query-time tuning
    survives the import instead of being silently discarded. See
    `_merge_legacy_query_time_keys` / `_LEGACY_SEARCH_KEYS`.

    Legacy reranking enablement additionally populates the new profile's
    `reranking_config` (not just `rag_config.search.enable_reranking`):
    `rag_factory.create_rag_service()` decides reranking enablement from
    `profile.reranking_config is not None` -- `search.enable_reranking` is
    UI-display-only and ignored by the live service (see
    `settings_rag_profile_adapter.py`'s `apply_defaults_to_profile`, the
    established convention this mirrors). Without this, a legacy user with
    reranking enabled would silently end up with it OFF after import.

    Self-healing (task-639, marker-based): existence of the profile is not
    enough to consider first-run import "done" -- if a previous run persisted
    the profile but failed before (or otherwise never got to) activating it,
    that leaves it created-but-never-active forever with no retry. The guard
    used to be "the active pointer differs from imported_settings", which
    cannot distinguish that half-done state from a user who deliberately
    activated a different profile afterward -- it silently flipped a
    deliberate switch back to Imported settings on the user's next launch. A
    second attempt compared the profile's *contents* against the default
    builtin (fingerprint + an allow-listed field set) to prove "nothing to
    lose", but that heuristic could itself be fooled: the Settings screen
    lets a user hand-tune fields the allow-list never checked (e.g.
    ``search.hybrid_alpha``, `default_search_mode`, `citation_style`,
    `embedding.batch_size`, ...), so a real customization outside that list
    passed the "provably safe" check and got permanently deleted.

    The fix is a durable marker instead of a content guess:
    ``[rag.service].first_run_import_done`` (``_first_run_import_done`` /
    ``_mark_first_run_import_done``), written every time this function
    deliberately activates ``imported_settings`` (both the fresh-import path
    below and the healing branch here). Once set, the pointer is NEVER
    second-guessed again, no matter what it currently names -- including the
    user switching back to the default builtin itself, which the pointer-only
    approach could not tell apart from "never written". While the marker is
    still absent (a config from before this function ever set it -- either a
    genuine half-done first run, or a pre-existing "Imported settings"
    activation from before the marker existed, task-639 AC #2), exactly one
    thing happens:

    - Pointer still names the default builtin (``DEFAULT_PROFILE`` ==
      ``hybrid_basic``, i.e. it was never successfully written by anyone) --
      this is the one condition a genuine half-done first run and "the
      marker predates this profile" share, so the healing here always
      completes the activation and sets the marker.
    - Pointer already names anything else (``imported_settings`` from a
      pre-635/pre-marker activation, or a different profile entirely) -- the
      pointer is left completely untouched (never deleted, never repointed)
      and the marker is simply set, adopting whatever is already active as
      confirmed going forward. Deliberately does NOT delete the profile even
      when it happens to look identical to the default: unlike a marker, no
      finite content comparison can prove a Settings-screen customization is
      absent, so guessing "safe to delete" risks real, irreversible data
      loss -- doing nothing here has no such risk and reaches the same
      settled state (the marker stops any further automatic handling).

    One inherent, bounded gap: for a config that predates the marker (an
    existing pointer that happens to already read as the default builtin,
    with no marker yet) there is no way to tell "never completed" apart from
    "the user deliberately switched back to the default sometime before this
    code shipped" -- this can only occur once, on the first run under
    marker-aware code, since the marker is written at every activation site
    from then on and this ambiguity can never arise again afterward.

    This healing path runs regardless of ``_has_legacy_rag_config_material()``
    -- once the profile exists, healing its activation is not a fresh-install
    concern.

    Exception-safe: any failure here must never block RAG service creation, so
    every error is caught and logged, returning None (as if already imported /
    nothing to do) rather than propagating.

    Returns:
        The new profile's id (``"imported_settings"``) on the run that
        creates and activates it; ``None`` when it already existed (no-op,
        after healing the active pointer / marker if needed), when there is
        no genuine legacy material to import (fresh install), or when import
        failed (logged, swallowed).
    """
    try:
        mgr = _manager()
        existing = mgr.get_profile(_IMPORTED_ID)
        if existing is not None:
            if not _first_run_import_done():
                if _active_profile_id() == DEFAULT_PROFILE:
                    # Never successfully written by anyone -- heal a
                    # half-done first run.
                    set_active_profile(_IMPORTED_ID)
                # else: pointer already names imported_settings (a
                # pre-marker activation, possibly pre-635 damage) or some
                # other profile entirely -- leave it exactly as-is (never
                # delete, never repoint; see docstring for why) and just
                # adopt it as settled.
                _mark_first_run_import_done()
            return None
        if not _has_legacy_rag_config_material():
            # task-635: nothing legacy to preserve continuity for -- leave a
            # fresh install on the default builtin profile untouched.
            return None
        # Snapshot the resolved config (active pointer may name a builtin default today).
        snapshot = resolve_active_rag_config()
        # task-495: preserve a user's hand-tuned, non-index-determining
        # query-time legacy keys instead of silently discarding them --
        # never affects the fingerprint (see _LEGACY_SEARCH_KEYS docstring).
        legacy = _merge_legacy_query_time_keys(snapshot)
        profile = ProfileConfig(id=_IMPORTED_ID, name="Imported settings",
                                description="Snapshot of your active RAG profile (plus any RAG_* env "
                                            "overrides) captured on first run; edit freely.",
                                profile_type="custom", rag_config=snapshot)
        if legacy.get("enable_reranking"):
            # Presence, not the mirrored search.enable_reranking flag, is
            # what actually turns reranking on for the live service (see the
            # docstring above) -- fabricate reranking_config here so a
            # legacy user's setting actually takes effect post-import.
            profile.reranking_config = RerankingConfig()
            reranker_model = legacy.get("reranker_model")
            if reranker_model:
                profile.reranking_config.model_name = reranker_model
            reranker_top_k = legacy.get("reranker_top_k")
            if reranker_top_k is not None:
                profile.reranking_config.top_k_to_rerank = int(reranker_top_k)
        mgr.save_profile(profile)
        set_active_profile(_IMPORTED_ID)
        _mark_first_run_import_done()
        return _IMPORTED_ID
    except Exception as e:
        logger.warning(f"ensure_imported_profile: first-run import failed, continuing without it: {e}")
        return None
