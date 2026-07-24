"""Adapter between the Settings RAG category and the profile system.

The RAG category edits the ACTIVE PROFILE (SP2a storage, SP2b resolution) —
never the deprecated AppRAGSearchConfig.rag.* keys, which the engine ignores.
All functions here are headless (no Textual imports) and testable.
"""
from __future__ import annotations

import copy
import dataclasses
from typing import Optional

from loguru import logger

from tldw_chatbook.RAG_Search.config_profiles import ProfileConfig, get_profile_manager
# Both imported as module seams: tests monkeypatch `ad._active_profile_id` and
# `ad.set_active_profile` directly (unqualified name lookup at call time),
# the same pattern already used for `_active_profile_id` below.
from tldw_chatbook.RAG_Search.simplified.active_config import (
    _active_profile_id,
    resolve_active_rag_config,
    set_active_profile,
)
from tldw_chatbook.RAG_Search.simplified.collection_fingerprint import (
    fingerprint_collection,
)
# Imported as a module seam (like `set_active_profile` above): tests
# monkeypatch `ad.index_status` directly to simulate a broken/unavailable
# vector store without touching a real Chroma instance.
from tldw_chatbook.RAG_Search.simplified.collection_indexes import index_status
from .settings_library_rag_defaults import (
    SettingsLibraryRagDefaults,
    _strict_float,
    _strict_int,
)


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
    e = profile.rag_config.embedding
    c = profile.rag_config.chunking
    v = profile.rag_config.vector_store
    rr = profile.reranking_config
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
        embedding_model=str(e.model),
        embedding_device=str(e.device) if e.device else "auto",
        embedding_batch_size=int(e.batch_size),
        embedding_max_length=int(e.max_length),
        chunk_size=int(c.chunk_size),
        chunk_overlap=int(c.chunk_overlap),
        chunking_method=str(c.chunking_method),
        distance_metric=str(v.distance_metric),
        # Rerank presence semantics: the service reads
        # `profile.reranking_config is not None` (rag_factory.py), so THAT is
        # the source of truth for whether reranking is on -- not
        # `rag_config.search.enable_reranking`, which apply_defaults_to_profile
        # only mirrors for display/consistency.
        enable_reranking=rr is not None,
        reranker_model=str(rr.model_name) if rr is not None else "",
        # When the profile has no reranking_config yet, fall back to the
        # SAME default SettingsLibraryRagDefaults itself uses (sourced from
        # RerankingConfig().top_k_to_rerank == 20) -- NOT
        # `s.reranker_top_k` (SearchConfig's functionally-dead field,
        # default 5; see rerank presence-semantics note above).
        reranker_top_k=(
            int(rr.top_k_to_rerank)
            if rr is not None
            else SettingsLibraryRagDefaults().reranker_top_k
        ),
    )


def apply_defaults_to_profile(
    profile: ProfileConfig, values: SettingsLibraryRagDefaults
) -> ProfileConfig:
    """Map the category dataclass onto the profile's RAGConfig (pure).

    Also applies rerank PRESENCE semantics: ``create_rag_service``
    (rag_factory.py) decides whether reranking is on from
    ``profile.reranking_config is not None`` -- so the ``enable_reranking``
    toggle here controls whether that attribute exists at all, not just a
    flag somewhere nobody reads.
    """
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

    e = profile.rag_config.embedding
    e.model = values.embedding_model
    # A blank device (like a blank reranker_model below) means "leave the
    # profile's existing device alone" -- never stomp it with an empty string.
    if values.embedding_device:
        e.device = values.embedding_device
    e.batch_size = int(values.embedding_batch_size)
    e.max_length = int(values.embedding_max_length)

    c = profile.rag_config.chunking
    c.chunk_size = int(values.chunk_size)
    c.chunk_overlap = int(values.chunk_overlap)
    c.chunking_method = values.chunking_method

    v = profile.rag_config.vector_store
    v.distance_metric = values.distance_metric

    if values.enable_reranking:
        if profile.reranking_config is None:
            from tldw_chatbook.RAG_Search.reranker import RerankingConfig
            profile.reranking_config = RerankingConfig()
        if values.reranker_model:
            profile.reranking_config.model_name = values.reranker_model
        profile.reranking_config.top_k_to_rerank = int(values.reranker_top_k)
        profile.rag_config.search.enable_reranking = True
    else:
        profile.reranking_config = None
        profile.rag_config.search.enable_reranking = False

    return profile


# Keyword fragments used to classify a `RAGConfig.validate()` message as
# concerning a field the Library/RAG Settings category actually exposes for
# editing. Covers exactly the four fields `RAGConfig.validate()` can emit a
# message about that this category also lets the user edit: chunk_size,
# chunk_overlap, distance_metric, and embedding_batch_size (the "batch_size"
# substring matches "embedding batch_size must be positive" without matching
# "embedding cache_size must be positive"). `RAGConfig.validate()` also
# checks vector_store.type, persist_directory, and embedding.cache_size --
# none of which are editable here -- plus default_top_k/score_threshold/
# hybrid_alpha, which validate_library_rag_defaults already validates with
# its own (narrower) rules; a violation on any of THOSE must never gate Save
# through this path. Kept as substrings (not a field-name dict) because
# RAGConfig.validate() hands back prose, not structured (field, reason)
# pairs.
_HARD_ERROR_UI_FIELD_KEYWORDS: tuple[str, ...] = (
    "chunk_size",
    "chunk_overlap",
    "distance metric",
    "batch_size",
)


def _is_ui_exposed_ragconfig_error(message: str) -> bool:
    """Whether a ``RAGConfig.validate()`` message concerns a field the
    Library/RAG Settings category actually exposes for editing.

    Args:
        message: One message from ``RAGConfig.validate()``.

    Returns:
        True when the message concerns a UI-exposed field.
    """
    lowered = message.lower()
    return any(keyword in lowered for keyword in _HARD_ERROR_UI_FIELD_KEYWORDS)


# Every numeric field `apply_defaults_to_profile` casts with a plain
# `int()`/`float()` call -- both reject float-like strings ("12.0") that
# validate_library_rag_defaults's own _strict_int/_strict_float helpers (and
# load_library_rag_defaults's callers) treat as valid input. See
# _tolerant_numeric_values below.
_HARD_ERROR_INT_FIELDS: tuple[str, ...] = (
    "default_top_k",
    "fts_top_k",
    "vector_top_k",
    "snippet_max_chars",
    "max_context_size",
    "embedding_batch_size",
    "embedding_max_length",
    "chunk_size",
    "chunk_overlap",
    "reranker_top_k",
)
_HARD_ERROR_FLOAT_FIELDS: tuple[str, ...] = ("hybrid_alpha", "score_threshold")


def _tolerant_numeric_values(
    values: SettingsLibraryRagDefaults,
) -> SettingsLibraryRagDefaults:
    """Pre-coerce ``apply_defaults_to_profile``'s numeric fields tolerantly.

    ``apply_defaults_to_profile`` casts these with plain ``int()``/``float()``
    for the SAVE path, where the UI already stages real numeric types. But a
    validator may legitimately be handed a still-stringly-typed, float-like
    value (e.g. ``"12.0"``) that plain ``int()`` raises on even though it is
    perfectly well-formed. Swap in the strictly-coerced value wherever it
    parses; a field that ISN'T even loosely parseable is left untouched, so
    ``apply_defaults_to_profile``'s own cast still raises on genuine garbage
    -- ``hard_config_errors`` converts that into a hard error instead of
    letting it crash the caller.

    Args:
        values: Candidate Library/RAG defaults to normalise.

    Returns:
        A copy of ``values`` with every loosely-numeric field coerced to a
        real ``int``/``float``.
    """
    updates: dict[str, object] = {}
    for field_name in _HARD_ERROR_INT_FIELDS:
        coerced = _strict_int(getattr(values, field_name))
        if coerced is not None:
            updates[field_name] = coerced
    for field_name in _HARD_ERROR_FLOAT_FIELDS:
        coerced = _strict_float(getattr(values, field_name))
        if coerced is not None:
            updates[field_name] = coerced
    return dataclasses.replace(values, **updates) if updates else values


def hard_config_errors(values: SettingsLibraryRagDefaults) -> list[str]:
    """Hard validation errors that must block Save.

    Applies ``values`` onto a scratch (deep-copied) clone of the active
    profile -- never the cached object other callers share -- and runs
    ``RAGConfig.validate()`` (the first caller of that method anywhere in the
    codebase), keeping only the messages that concern a field this category
    actually exposes for editing (see ``_HARD_ERROR_UI_FIELD_KEYWORDS``).
    Also flags ``reranker_top_k < 1`` when reranking is enabled --
    ``RAGConfig.validate()`` has no concept of reranking at all.

    ``validate_library_rag_defaults`` (settings_library_rag_defaults.py)
    calls this for the rules it would otherwise have to duplicate (and drift
    from); it keeps its own narrower rules for fields ``RAGConfig.validate()``
    doesn't cover.

    Args:
        values: Candidate Library/RAG defaults to validate.

    Returns:
        Human-readable hard-error messages; empty when nothing blocks Save.
        When no profile is active, or the active profile cannot be loaded,
        returns a single explanatory message rather than raising -- this
        function must fail CLOSED (block Save) rather than let a profile-
        manager error propagate out of validation.
    """
    try:
        profile = _active_profile()
    except Exception as e:  # profile fetch must never crash Save validation
        logger.error(f"Could not load the active profile for validation: {e}")
        return ["Could not load the active profile for validation."]
    if profile is None:
        return ["No active profile is selected."]

    scratch = copy.deepcopy(profile)
    try:
        apply_defaults_to_profile(scratch, _tolerant_numeric_values(values))
    except (TypeError, ValueError) as e:
        return [f"Library/RAG defaults contain an invalid value: {e}"]

    errors = [
        message
        for message in scratch.rag_config.validate()
        if _is_ui_exposed_ragconfig_error(message)
    ]

    if values.enable_reranking:
        reranker_top_k = _strict_int(values.reranker_top_k)
        if reranker_top_k is None or reranker_top_k < 1:
            errors.append("Reranker top-k must be at least 1.")

    return errors


def soft_config_warnings(values: SettingsLibraryRagDefaults) -> list[str]:
    """Advisory warnings that inform but never gate Save.

    Currently just one check: when reranking is enabled and
    ``reranker_top_k`` exceeds ``default_top_k``, reranking will never see
    all the requested results. Pure (no active-profile dependency) --
    everything it needs is already on ``values``.

    Args:
        values: Candidate Library/RAG defaults to check.

    Returns:
        Human-readable advisory messages; empty when nothing to report.
    """
    if not values.enable_reranking:
        return []
    reranker_top_k = _strict_int(values.reranker_top_k)
    default_top_k = _strict_int(values.default_top_k)
    if reranker_top_k is None or default_top_k is None:
        return []
    if reranker_top_k > default_top_k:
        return [
            f"Reranker top-k ({reranker_top_k}) exceeds default results "
            f"({default_top_k}); reranking will not see all requested results."
        ]
    return []


def index_change_pending(values: SettingsLibraryRagDefaults) -> bool:
    """Whether saving ``values`` right now would re-point the active profile
    at a different (not-yet-built) vector collection.

    Pure: applies ``values`` onto a deep-copied SCRATCH clone of the active
    profile via ``apply_defaults_to_profile`` and compares
    ``fingerprint_collection`` (SP1) before/after -- the live cached profile
    the manager hands to every other caller is never mutated. Never raises:
    a missing active profile, an unparseable numeric field, or any other
    failure is treated as "no pending change" (logged) rather than raising
    -- a broken fingerprint must never block the editor.

    Args:
        values: Candidate Library/RAG defaults to check.

    Returns:
        True when saving ``values`` would change the fingerprinted
        collection name; False otherwise (including on any internal error).
    """
    try:
        profile = _active_profile()
        if profile is None:
            return False
        before_fp = fingerprint_collection(profile.rag_config)
        scratch = copy.deepcopy(profile)
        apply_defaults_to_profile(scratch, _tolerant_numeric_values(values))
        after_fp = fingerprint_collection(scratch.rag_config)
        return before_fp != after_fp
    except Exception as e:  # a broken fingerprint must never block the editor
        logger.error(f"index_change_pending failed: {e}")
        return False


def fetch_index_status() -> dict:
    """Resolved vector-index state for the active profile: absent | empty | built.

    Wraps ``index_status(resolve_active_rag_config())`` (SP1) -- reads the
    on-disk Chroma collection list, so callers must run this off the UI
    thread. Any failure (a broken/unavailable store, a missing active
    profile) returns an "unknown" state rather than raising.

    Returns:
        ``{"state": "absent"|"empty"|"built"|"unknown", "count": int,
        "provenance": dict}``.
    """
    try:
        return index_status(resolve_active_rag_config())
    except Exception as e:
        logger.error(f"fetch_index_status failed: {e}")
        return {"state": "unknown", "count": 0, "provenance": {}}


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
