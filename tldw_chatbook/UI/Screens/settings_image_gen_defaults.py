"""Image Gen guided defaults for the Settings hub -- data layer.

Task 2 of the Settings > Image Gen plan (see
``Docs/superpowers/specs/2026-07-25-image-gen-settings-page-design.md``).
This module owns the pure/testable pieces the screen composes over: the
per-backend field schema, backend status rows, and the draft -> config-write
diffing/validation logic. No Textual widgets live here (mirrors the
``settings_library_rag_defaults.py`` pattern).

Task 3 adds the backend probe section (``probe_backend`` et al.): a short
live/filesystem "Test" check per backend, mirroring
``settings_endpoint_probe.py``'s philosophy for the Settings provider probe.
Probe outcomes are reduced to a small closed set of badge strings and never
include endpoint URLs, exception text, headers, or secrets -- callers may
render ``ImageGenProbeResult.badge`` directly in the UI.
"""

from __future__ import annotations

import os
import shutil
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import httpx
from loguru import logger

from tldw_chatbook.config import _get_effective_config_path
from tldw_chatbook.Image_Generation.config import (
    ImageGenerationConfig,
    _NON_SECRET,
    _resolve_secret,
    _SECRETS,
)
from tldw_chatbook.Image_Generation.listing import (
    _IMAGE_LISTING_NONCRITICAL_EXCEPTIONS,
    _is_fal_configured,
    _is_gemini_configured,
    _is_modelstudio_configured,
    _is_novita_configured,
    _is_openrouter_configured,
    _is_sd_cpp_configured,
    _is_swarmui_configured,
    _is_together_configured,
)
from tldw_chatbook.Utils.egress import EgressBlockedError, check_url_or_raise, origin_set
from tldw_chatbook.Utils.input_validation import validate_url


BACKEND_IDS: tuple[str, ...] = (
    "stable_diffusion_cpp",
    "swarmui",
    "openrouter",
    "novita",
    "together",
    "modelstudio",
    "fal",
    "gemini",
)

BACKEND_LABELS: dict[str, str] = {
    "stable_diffusion_cpp": "SD.cpp (local)",
    "swarmui": "SwarmUI (local)",
    "openrouter": "OpenRouter",
    "novita": "Novita",
    "together": "Together",
    "modelstudio": "ModelStudio",
    "fal": "fal.ai",
    "gemini": "Gemini (AI Studio)",
}

# backend_id -> the listing.py is_configured check to reuse (never
# reimplemented here -- see the module docstring).
_CONFIGURED_CHECKS = {
    "stable_diffusion_cpp": _is_sd_cpp_configured,
    "swarmui": _is_swarmui_configured,
    "openrouter": _is_openrouter_configured,
    "novita": _is_novita_configured,
    "together": _is_together_configured,
    "modelstudio": _is_modelstudio_configured,
    "fal": _is_fal_configured,
    "gemini": _is_gemini_configured,
}


@dataclass(frozen=True)
class FieldSpec:
    """One curated per-backend field, driving both the editor form and validation."""

    toml_key: str
    label: str
    kind: str  # "text" | "url" | "path" | "int" | "secret"
    min_value: float | None = None


# v1 curated field table (spec's "Per-backend field schema" table). Advanced
# keys (allowed_extra_params, sd.cpp vae/llm/lora/steps/cfg/sampler/
# diffusion_model_path, novita/modelstudio poll_interval_seconds, modelstudio
# mode) are intentionally excluded -- config-file-only in v1, per spec scope.
FIELD_SCHEMA: dict[str, tuple[FieldSpec, ...]] = {
    "stable_diffusion_cpp": (
        FieldSpec("binary_path", "Binary path", "path"),
        FieldSpec("model_path", "Model path", "path"),
        FieldSpec("timeout_seconds", "Timeout (seconds)", "int", min_value=1),
    ),
    "swarmui": (
        FieldSpec("base_url", "Base URL", "url"),
        FieldSpec("default_model", "Default model", "text"),
        FieldSpec("timeout_seconds", "Timeout (seconds)", "int", min_value=1),
        FieldSpec("swarm_token", "Swarm token", "secret"),
    ),
    "openrouter": (
        FieldSpec("base_url", "Base URL", "url"),
        FieldSpec("default_model", "Default model", "text"),
        FieldSpec("timeout_seconds", "Timeout (seconds)", "int", min_value=1),
        FieldSpec("api_key", "API key", "secret"),
    ),
    "novita": (
        FieldSpec("base_url", "Base URL", "url"),
        FieldSpec("default_model", "Default model", "text"),
        FieldSpec("timeout_seconds", "Timeout (seconds)", "int", min_value=1),
        FieldSpec("api_key", "API key", "secret"),
    ),
    "together": (
        FieldSpec("base_url", "Base URL", "url"),
        FieldSpec("default_model", "Default model", "text"),
        FieldSpec("timeout_seconds", "Timeout (seconds)", "int", min_value=1),
        FieldSpec("api_key", "API key", "secret"),
    ),
    "modelstudio": (
        FieldSpec("base_url", "Base URL", "url"),
        FieldSpec("default_model", "Default model", "text"),
        FieldSpec("region", "Region", "text"),
        FieldSpec("timeout_seconds", "Timeout (seconds)", "int", min_value=1),
        FieldSpec("api_key", "API key", "secret"),
    ),
    # fal's `poll_interval_seconds` stays config-only in v1 (matches
    # novita/modelstudio's own poll_interval_seconds precedent -- neither is
    # in the curated editor either).
    "fal": (
        FieldSpec("base_url", "Base URL", "url"),
        FieldSpec("default_model", "Default model", "text"),
        FieldSpec("timeout_seconds", "Timeout (seconds)", "int", min_value=1),
        FieldSpec("api_key", "API key", "secret"),
    ),
    "gemini": (
        FieldSpec("base_url", "Base URL", "url"),
        FieldSpec("default_model", "Default model", "text"),
        FieldSpec("timeout_seconds", "Timeout (seconds)", "int", min_value=1),
        FieldSpec("api_key", "API key", "secret"),
    ),
}


def _spec_for(backend_id: str, toml_key: str) -> FieldSpec | None:
    for spec in FIELD_SCHEMA.get(backend_id, ()):
        if spec.toml_key == toml_key:
            return spec
    return None


@dataclass(frozen=True)
class ImageGenBackendRow:
    """One row of the Backends table."""

    backend_id: str
    label: str
    configured: bool
    enabled: bool
    is_default: bool
    key_source: str
    secret_optional: bool


def build_backend_rows(cfg: ImageGenerationConfig) -> list[ImageGenBackendRow]:
    """Build one status row per backend from the effective config.

    Args:
        cfg: The effective ``ImageGenerationConfig`` (``get_image_generation_config()``).

    Returns:
        One ``ImageGenBackendRow`` per entry in ``BACKEND_IDS``, in that order.
    """
    enabled_backends = set(cfg.enabled_backends or [])
    key_sources = cfg.key_sources or {}
    rows: list[ImageGenBackendRow] = []
    for backend_id in BACKEND_IDS:
        enabled = backend_id in enabled_backends
        check = _CONFIGURED_CHECKS[backend_id]
        try:
            configured = bool(check(cfg, enabled))
        except _IMAGE_LISTING_NONCRITICAL_EXCEPTIONS:
            configured = False
        rows.append(
            ImageGenBackendRow(
                backend_id=backend_id,
                label=BACKEND_LABELS[backend_id],
                configured=configured,
                enabled=enabled,
                is_default=(cfg.default_backend == backend_id),
                key_source=key_sources.get(backend_id, "missing"),
                secret_optional=(backend_id == "swarmui"),
            )
        )
    return rows


def effective_placeholder(cfg: ImageGenerationConfig, backend_id: str, toml_key: str) -> str:
    """Return the resolved effective value for an unset non-secret field.

    Used as the editor's placeholder text so an empty field never hides what
    will actually be used at generation time (the task-620 lesson). Secrets
    are out of scope here -- they're never echoed as placeholders.

    Args:
        cfg: The effective ``ImageGenerationConfig``.
        backend_id: A key in ``BACKEND_IDS``.
        toml_key: A non-secret field's TOML key (must be a key of ``_NON_SECRET``).

    Returns:
        ``str(value)`` when the resolved flat field is set, else ``""``.
    """
    flat_field = _NON_SECRET[(backend_id, toml_key)]
    value = getattr(cfg, flat_field, None)
    return "" if value is None else str(value)


def effective_secret_value(cfg: ImageGenerationConfig, backend_id: str) -> str | None:
    """Return the resolved EFFECTIVE secret for ``backend_id``, or ``None``.

    ``cfg``'s secret fields already have env/config/keyring precedence
    applied (see ``_resolve_secret``) -- this just looks up the right flat
    attribute for ``backend_id`` via ``_SECRETS``. Used by the Settings >
    Image Gen Test action as the probe's ``secret`` fallback when the user
    hasn't pasted a fresh value this session (see ``probe_backend``'s
    ``secret`` parameter docstring).

    Args:
        cfg: The effective ``ImageGenerationConfig``.
        backend_id: One of ``BACKEND_IDS``.

    Returns:
        The resolved secret string, or ``None`` when the backend has no
        secret field (``stable_diffusion_cpp``) or none is currently
        resolved.
    """
    secret_entry = _SECRETS.get(backend_id)
    if secret_entry is None:
        return None
    value = getattr(cfg, secret_entry[0], None)
    return value if isinstance(value, str) and value else None


def load_user_image_generation_table() -> Mapping[str, Any]:
    """Read the user's OWN ``[image_generation]`` table, UNMERGED with baked defaults.

    DISPLAY-ONLY -- do not use this as a diff/save baseline (task 5's
    ``diff_to_sections`` must keep comparing against ``SettingsConfigAdapter
    .load()``'s merged raw config; that stays correct regardless of what this
    function returns, since comparing a draft against a MERGED config can
    only ever suppress writing an unedited key that already equals its baked
    default -- never emit an extra one).

    ``load_cli_config_and_ensure_existence`` (which both ``SettingsConfig
    Adapter.load()`` and ``Image_Generation.config.get_image_generation_
    config()`` read through) deep-merges ``config.py``'s bundled
    ``DEFAULT_CONFIG_FROM_TOML`` template into whatever's on disk. That
    template bakes a literal, non-empty value into nearly every
    ``[image_generation.<backend>]`` field (e.g. openrouter's
    ``default_model = "google/gemini-2.5-flash-image"``), so on a fresh
    install (or any scratch config that doesn't set a field), the merged
    config makes that field look "explicitly set" -- the Settings panel would
    render it as the input's VALUE instead of a placeholder, even though the
    user never typed it (a set-vs-default blur; the spec's own openrouter
    model example is the sharpest case).

    This function bypasses that merge: it parses ONLY the on-disk file at the
    effective config path (the same resolution ``config._get_effective_
    config_path()`` / ``SettingsScreen._config_path()`` already use) and
    returns its raw ``[image_generation]`` table exactly as the user wrote
    it -- the true "what did the user actually set" source for the Settings
    panel's input VALUES. Placeholders keep coming from
    ``effective_placeholder()`` (unchanged; it already reads the resolved/
    merged config, which is correct for "what will actually be used").

    Returns:
        The raw ``[image_generation]`` table, or ``{}`` if the config file
        doesn't exist, has no ``[image_generation]`` table, or fails to
        parse. Never raises.
    """
    try:
        config_path = _get_effective_config_path()
    except Exception as exc:
        logger.debug(f"image_generation: could not resolve config path: {exc}")
        return {}
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "rb") as f:
            parsed = tomllib.load(f)
    except Exception as exc:
        logger.debug(
            f"image_generation: could not parse config file {config_path}: {exc}"
        )
        return {}
    section = parsed.get("image_generation")
    return section if isinstance(section, dict) else {}


def key_source_after_clear(backend_id: str) -> str:
    """What the key-source line would show for ``backend_id`` if its locally
    saved config value were removed right now (env/keyring fallback, or
    ``"missing"``).

    Used by the Settings > Image Gen Clear action to optimistically
    re-render the source line before Save actually persists the deletion --
    reuses the loader's own precedence resolver (``_resolve_secret``),
    passing an EMPTY sub-config mapping so its ``"config"`` precedence
    branch can never win (there is nothing for it to read).

    Args:
        backend_id: One of ``BACKEND_IDS``.

    Returns:
        ``"env:<VAR>"``, ``"keyring"``, or ``"missing"``. Backends with no
        secret field (``stable_diffusion_cpp``) always return ``"missing"``.
    """
    if backend_id not in _SECRETS:
        return "missing"
    return _resolve_secret(backend_id, {})[2]


_GLOBAL_DRAFT_KEYS: tuple[str, ...] = (
    "default_backend",
    "enabled_backends",
    "default_batch",
    "max_variants_per_message",
    "context_llm_enabled",
    "context_llm_turns",
    "context_llm_timeout_seconds",
)


def canonical_backend_order(backend_ids: Any) -> list[str]:
    """Normalize an ``enabled_backends``-shaped list to ``BACKEND_IDS``'
    canonical order, dropping any unrecognized entries.

    Final review Minor 1: a config file's ``enabled_backends`` list order
    has no meaning (it's a set), but Python list equality IS order-
    sensitive -- comparing it against a draft's list built by iterating
    ``BACKEND_IDS`` (always canonical order) without normalizing BOTH
    sides first spuriously diffs whenever the file happens to list the
    same backends in a different order, causing an unnecessary rewrite on
    every save and a rail dirty marker that never clears on its own.

    Args:
        backend_ids: An iterable (or ``None``/falsy) of backend id
            strings, in whatever order the caller has them -- typically a
            raw config's ``enabled_backends`` list, in file order.
            Duplicates and unrecognized ids (not in ``BACKEND_IDS``) are
            tolerated and simply have no effect on the result.

    Returns:
        A new list containing only the recognized ids from
        ``backend_ids``, deduplicated, ordered to match ``BACKEND_IDS``.
    """
    ids = set(backend_ids or ())
    return [backend_id for backend_id in BACKEND_IDS if backend_id in ids]


@dataclass(frozen=True)
class ImageGenDraftValues:
    """Pending Settings > Image Gen edits.

    Scalar global fields default to ``None``, meaning "not touched this
    session" -- ``diff_to_sections`` skips them entirely rather than writing
    back a baked-in default. ``enabled_backends`` is the one exception (its
    declared type is ``list[str]``, never ``None``); an untouched ``[]``
    normalizes against an absent raw ``enabled_backends`` key (also treated
    as ``[]``), so it does not spuriously diff either.

    ``backend_fields`` holds the edited-this-session raw strings the user
    typed, keyed ``backend_id -> toml_key -> raw string`` (secrets included
    only when the user actually typed one -- masked inputs are never
    pre-filled with the existing saved value, so an untouched secret simply
    never appears here). ``cleared_fields`` holds ``backend_id -> [toml_key,
    ...]`` for fields the user explicitly cleared via the Clear action.
    """

    default_backend: str | None = None
    enabled_backends: list[str] = field(default_factory=list)
    default_batch: int | None = None
    max_variants_per_message: int | None = None
    context_llm_enabled: bool | None = None
    context_llm_turns: int | None = None
    context_llm_timeout_seconds: float | None = None
    backend_fields: dict[str, dict[str, str]] = field(default_factory=dict)
    cleared_fields: dict[str, list[str]] = field(default_factory=dict)


def _coerce_value(spec: FieldSpec | None, raw_value: str) -> Any:
    """Coerce an edited raw string per its FieldSpec kind (int fields only, v1)."""
    if spec is not None and spec.kind == "int":
        try:
            return int(str(raw_value).strip())
        except (TypeError, ValueError):
            return raw_value  # validate_draft() is responsible for catching this
    return raw_value


def diff_to_sections(
    draft: ImageGenDraftValues, raw_config: Mapping
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Diff a draft against the RAW config mapping -- never an ``ImageGenerationConfig``.

    ``ImageGenerationConfig``'s secret fields hold env/keyring-resolved
    values; reading it here would risk silently copying an env-provided
    secret into plaintext config.toml. This function only ever sees the
    draft and the raw ``[image_generation]`` TOML table (e.g. from
    ``SettingsConfigAdapter.load()``) -- a signature-level guarantee that
    holds regardless of what the diff logic below does.

    That logic is also, independently, field-level rather than a wholesale
    per-backend section rewrite: editing one field (e.g. ``default_model``)
    never carries a *different*, already-saved field on the same backend
    (e.g. a pre-existing ``api_key``) along with it -- only keys the draft
    actually names in ``backend_fields``/``cleared_fields`` can appear in
    the output at all. A secret is emitted only when the draft carries it
    verbatim (the user typed it this session).

    Args:
        draft: The pending edits.
        raw_config: The full raw config mapping (``adapter.load()``'s return).

    Returns:
        ``(sections, deletions)``. ``sections`` maps a dotted section name
        (``"image_generation"`` or ``"image_generation.<backend>"``) to the
        keys/values that differ from ``raw_config`` (int fields coerced
        before comparing, so a round-tripped-unchanged int is never
        re-emitted). ``deletions`` maps the same section names to keys the
        user explicitly cleared this session, OR emptied back to a blank
        string (final review Important 1: never write an empty-string
        sentinel over a previously-saved value -- that would both discard
        the saved value AND leave a falsy-but-present key shadowing its
        env/keyring/baked-default fallback). A key present in both
        ``backend_fields`` and ``cleared_fields`` for the same backend
        resolves to deletion, never a write of the stale edit. An emptied
        key that was never set in ``raw_config`` to begin with is a pure
        no-op (dropped from both ``sections`` and ``deletions``) rather
        than an empty diff or a delete-of-nothing.
    """
    raw_top: Mapping[str, Any] = (raw_config or {}).get("image_generation") or {}
    sections: dict[str, dict[str, Any]] = {}
    deletions: dict[str, list[str]] = {}

    global_diff: dict[str, Any] = {}
    global_deletions: list[str] = []
    for key in _GLOBAL_DRAFT_KEYS:
        value = getattr(draft, key)
        if value is None:
            continue  # not touched this session
        if key == "enabled_backends":
            # Both sides normalized to canonical order first (Minor 1) --
            # a config file's list order is meaningless (it's a set), but
            # Python list equality isn't.
            normalized_value = canonical_backend_order(value)
            raw_value = canonical_backend_order(raw_top.get(key))
            if normalized_value != raw_value:
                global_diff[key] = normalized_value
            continue
        if isinstance(value, str) and not value.strip():
            if key in raw_top:
                global_deletions.append(key)
            continue  # emptied, never set -- no-op, not a diff
        raw_value = raw_top.get(key)
        if value != raw_value:
            global_diff[key] = value
    if global_diff:
        sections["image_generation"] = global_diff
    if global_deletions:
        deletions["image_generation"] = global_deletions

    for backend_id, fields in draft.backend_fields.items():
        raw_backend: Mapping[str, Any] = raw_top.get(backend_id) or {}
        cleared = set(draft.cleared_fields.get(backend_id, ()))
        backend_diff: dict[str, Any] = {}
        empty_deletions: set[str] = set()
        for toml_key, raw_value in fields.items():
            if toml_key in cleared:
                continue
            if isinstance(raw_value, str) and not raw_value.strip():
                if toml_key in raw_backend:
                    empty_deletions.add(toml_key)
                continue  # emptied, never set -- no-op, not a diff
            spec = _spec_for(backend_id, toml_key)
            coerced = _coerce_value(spec, raw_value)
            if coerced != raw_backend.get(toml_key):
                backend_diff[toml_key] = coerced
        if backend_diff:
            sections[f"image_generation.{backend_id}"] = backend_diff
        if empty_deletions:
            section_key = f"image_generation.{backend_id}"
            deletions[section_key] = sorted(
                set(deletions.get(section_key, ())) | empty_deletions
            )

    for backend_id, keys in draft.cleared_fields.items():
        if keys:
            section_key = f"image_generation.{backend_id}"
            deletions[section_key] = sorted(
                set(deletions.get(section_key, ())) | set(keys)
            )

    return sections, deletions


# Global scalar fields that render as a plain-text Input (task 5): label
# used in an inline error, and whether the field is int- or float-typed,
# plus the design spec's own minimum ("Ints/floats validated to the
# loader's own clamps where they exist: default_batch >= 1,
# context_llm_turns >= 1, context_llm_timeout_seconds >= 0.1" --
# max_variants_per_message shares default_batch's >= 1 floor; the loader
# only type-coerces these, so validate_draft is the one place that
# actually enforces them). The screen stages an UNPARSEABLE OR EMPTIED
# edit as the raw string rather than silently dropping it (so it still
# marks dirty and surfaces feedback); validate_draft is what actually
# catches an unparseable one before it could ever reach diff_to_sections,
# matching the per-backend "int"-kind fields' treatment below exactly. An
# EMPTIED one is validated by neither -- diff_to_sections turns it into a
# deletion instead (final review Important 1).
_GLOBAL_INT_FIELD_SPECS: tuple[tuple[str, str, int], ...] = (
    ("default_batch", "Default batch", 1),
    ("max_variants_per_message", "Max variants / message", 1),
    ("context_llm_turns", "Context LLM turns", 1),
)
_GLOBAL_FLOAT_FIELD_SPECS: tuple[tuple[str, str, float], ...] = (
    ("context_llm_timeout_seconds", "Context LLM timeout (s)", 0.1),
)


def _is_valid_numeric(value: Any, *, kind: type) -> bool:
    """``True`` iff ``value`` is already a real (non-bool) ``kind`` or a
    string ``kind(...)`` can parse. ``bool`` is a ``int`` subclass in
    Python -- excluded so a stray ``True``/``False`` never passes as 0/1."""
    if isinstance(value, bool):
        return False
    if isinstance(value, kind):
        return True
    try:
        kind(str(value).strip())
    except (TypeError, ValueError):
        return False
    return True


def validate_draft(draft: ImageGenDraftValues) -> tuple[list[str], list[str]]:
    """Validate a draft before it can be saved.

    Args:
        draft: The pending Settings > Image Gen edits to validate --
            only the fields actually touched this session are checked
            (untouched scalars are ``None`` and skipped; see
            ``ImageGenDraftValues``'s docstring).

    Returns:
        ``(errors, warnings)``. ``errors`` block the save: the default
        backend must be enabled, backend "int"-kind fields (timeout_seconds)
        must parse and respect their ``FieldSpec.min_value``, "url"-kind
        fields (``base_url``) must parse as an http(s) URL with a host, and
        the global int/float fields (``default_batch``,
        ``max_variants_per_message``, ``context_llm_turns``,
        ``context_llm_timeout_seconds``) must parse as their expected type
        and respect the design spec's own minimum (the screen stages an
        unparseable edit as the raw string rather than silently dropping
        it -- this is what actually blocks it). An EMPTIED field (backend
        or global) is validated by neither check -- ``diff_to_sections``
        turns it into a deletion instead, never a value to reject.
        ``warnings`` are non-blocking hints: all backends disabled, and
        ``default_batch`` exceeding ``max_variants_per_message`` (the
        runtime already clamps this safely; skipped when either isn't
        currently a valid number, since the corresponding error above
        already covers that case).
    """
    errors: list[str] = []
    warnings: list[str] = []

    enabled = draft.enabled_backends or []
    if draft.default_backend is not None and draft.default_backend not in enabled:
        errors.append("Default backend must be enabled — pick another default first.")

    for backend_id, fields in draft.backend_fields.items():
        backend_label = BACKEND_LABELS.get(backend_id, backend_id)
        cleared = set(draft.cleared_fields.get(backend_id, ()))
        for toml_key, raw_value in fields.items():
            if toml_key in cleared:
                continue  # explicit Clear -- nothing to validate
            if isinstance(raw_value, str) and not raw_value.strip():
                continue  # emptied -- becomes a deletion, not a value to validate
            spec = _spec_for(backend_id, toml_key)
            if spec is None:
                continue
            if spec.kind == "int":
                try:
                    parsed = int(str(raw_value).strip())
                except (TypeError, ValueError):
                    errors.append(f"{backend_label} {spec.label} must be a whole number.")
                    continue
                if spec.min_value is not None and parsed < spec.min_value:
                    errors.append(
                        f"{backend_label} {spec.label} must be at least {int(spec.min_value)}."
                    )
            elif spec.kind == "url":
                # Qodo PR #901 fix 5: adopt the shared
                # `input_validation.validate_url` (already used by
                # settings_endpoint_probe.py for this same http(s)+host
                # contract) instead of the ad-hoc urlparse check that used
                # to live here. Its semantics are a strict superset of
                # what the ad-hoc check did -- same http/https scheme set
                # and required host, PLUS it rejects whitespace,
                # backslashes (a parser-discrepancy SSRF vector),
                # embedded credentials, and malformed hosts -- it's pure
                # string parsing with no network side effects, and it
                # returns a plain bool that slots straight into this
                # inline error collection (never raises).
                if not validate_url(str(raw_value).strip()):
                    errors.append(f"{backend_label} {spec.label} must be a valid http(s) URL.")

    for key, label, minimum in _GLOBAL_INT_FIELD_SPECS:
        value = getattr(draft, key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue  # emptied -- becomes a deletion, not invalid input
        if not _is_valid_numeric(value, kind=int):
            errors.append(f"{label} must be a whole number.")
            continue
        if int(value) < minimum:
            errors.append(f"{label} must be at least {minimum}.")
    for key, label, minimum in _GLOBAL_FLOAT_FIELD_SPECS:
        value = getattr(draft, key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if not _is_valid_numeric(value, kind=float):
            errors.append(f"{label} must be a number.")
            continue
        if float(value) < minimum:
            errors.append(f"{label} must be at least {minimum}.")

    if not enabled:
        warnings.append(
            "All backends are disabled — image generation will be unavailable "
            "until you enable at least one."
        )
    if (
        draft.default_batch is not None
        and draft.max_variants_per_message is not None
        and _is_valid_numeric(draft.default_batch, kind=int)
        and _is_valid_numeric(draft.max_variants_per_message, kind=int)
        and float(draft.default_batch) > float(draft.max_variants_per_message)
    ):
        warnings.append(
            "Default batch is larger than the max-variants cap — the runtime will clamp it down."
        )

    return errors, warnings


# ---------------------------------------------------------------------------
# Backend probes ("Test" action)
# ---------------------------------------------------------------------------

PROBE_TIMEOUT_SECONDS = 5.0
"""Per-request timeout for a Settings > Image Gen backend probe."""


@dataclass(frozen=True)
class ImageGenProbeResult:
    """Outcome of one live/filesystem backend probe.

    Attributes:
        ok: Whether the probe found the backend reachable/usable.
        badge: One of the spec's exact short strings -- ``"Reachable"``,
            ``"Reachable (auth unverified)"``, ``"Auth failed"``,
            ``"Unreachable: <category>"`` (category is one of "connection
            refused", "timeout", "HTTP <status>", "blocked by egress
            policy"), ``"Binary found"``, ``"Binary missing or not
            executable"``, ``"Model file missing"``. Never contains
            exception text, URLs, headers, or credentials -- see the module
            docstring's sanitization contract.
    """

    ok: bool
    badge: str


def _guarded_get(
    url: str, *, headers: Mapping[str, str] | None = None
) -> tuple[httpx.Response | None, str | None]:
    """Egress-checked, sanitized GET shared by every network probe.

    Runs the SSRF egress check before any request (trusting only ``url``'s
    own host, per the spec's ``origin_set(url)`` pattern), then issues one
    short, non-redirect-following GET.

    Returns:
        ``(response, None)`` for any HTTP answer -- including 4xx/5xx, which
        still means the server responded -- or ``(None, badge)`` with a
        closed-set ``"Unreachable: <category>"`` badge on failure. Exception
        text is never propagated into the badge: it can carry hosts, ports,
        or embedded secrets (e.g. a malformed-URL error message).
    """
    try:
        check_url_or_raise(url, trusted_origins=origin_set(url))
    except EgressBlockedError:
        return None, "Unreachable: blocked by egress policy"
    try:
        with httpx.Client(timeout=PROBE_TIMEOUT_SECONDS, follow_redirects=False) as client:
            return client.get(url, headers=dict(headers) if headers else None), None
    except httpx.TimeoutException:
        return None, "Unreachable: timeout"
    except Exception:
        # Every other transport/protocol failure (incl. ConnectError):
        # collapse to the generic closed-set category rather than risk
        # echoing raw exception text.
        return None, "Unreachable: connection refused"


def _probe_swarmui(base_url: str) -> ImageGenProbeResult:
    """SwarmUI has no models-listing route -- a plain GET on ``base_url``
    that gets *any* HTTP answer (even 4xx/5xx) means the server responded."""
    _response, blocked_badge = _guarded_get(base_url)
    if blocked_badge is not None:
        return ImageGenProbeResult(ok=False, badge=blocked_badge)
    return ImageGenProbeResult(ok=True, badge="Reachable")


def _probe_reachability_only(base_url: str) -> ImageGenProbeResult:
    """novita/modelstudio/fal: none of these has a confirmed cheap
    authenticated GET (see ``probe_backend``'s docstring) -- unauthenticated
    reachability only, so auth is never actually verified. fal's queue API
    (``queue.fal.run``) has no models-listing route either -- this is a
    plain reachability GET on the configured base, same shape as
    ``_probe_swarmui`` but reporting the honest "auth unverified" badge
    (unlike swarmui, which has no auth concept at all)."""
    _response, blocked_badge = _guarded_get(base_url)
    if blocked_badge is not None:
        return ImageGenProbeResult(ok=False, badge=blocked_badge)
    return ImageGenProbeResult(ok=True, badge="Reachable (auth unverified)")


def _probe_openai_compatible(base_url: str, secret: str | None) -> ImageGenProbeResult:
    """openrouter/together: OpenAI-compatible ``GET {base_url}/models``."""
    url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {secret}"} if secret else None
    response, blocked_badge = _guarded_get(url, headers=headers)
    if blocked_badge is not None:
        return ImageGenProbeResult(ok=False, badge=blocked_badge)
    if not secret:
        return ImageGenProbeResult(ok=True, badge="Reachable (auth unverified)")
    if response.status_code in (401, 403):
        return ImageGenProbeResult(ok=False, badge="Auth failed")
    if 200 <= response.status_code < 300:
        return ImageGenProbeResult(ok=True, badge="Reachable")
    return ImageGenProbeResult(ok=False, badge=f"Unreachable: HTTP {response.status_code}")


def _probe_gemini(base_url: str, secret: str | None) -> ImageGenProbeResult:
    """gemini: authenticated ``GET {base_url}/models``, key in the
    ``x-goog-api-key`` header -- Google's models-listing route is cheap and
    exists on the same base as the image-generation endpoint, so (unlike
    novita/modelstudio/fal) this is a full auth-verified probe, mirroring
    ``_probe_openai_compatible``'s shape with Gemini's own header/auth
    semantics."""
    url = f"{base_url.rstrip('/')}/models"
    headers = {"x-goog-api-key": secret} if secret else None
    response, blocked_badge = _guarded_get(url, headers=headers)
    if blocked_badge is not None:
        return ImageGenProbeResult(ok=False, badge=blocked_badge)
    if not secret:
        return ImageGenProbeResult(ok=True, badge="Reachable (auth unverified)")
    if response.status_code in (401, 403):
        return ImageGenProbeResult(ok=False, badge="Auth failed")
    if 200 <= response.status_code < 300:
        return ImageGenProbeResult(ok=True, badge="Reachable")
    return ImageGenProbeResult(ok=False, badge=f"Unreachable: HTTP {response.status_code}")


def _probe_sd_cpp(form_values: Mapping[str, str]) -> ImageGenProbeResult:
    """sd.cpp is filesystem-only -- no network. Both ``binary_path`` and
    ``model_path`` are operator-owned local config paths (config-set, never
    user-uploaded), so this deliberately skips ``path_validation``
    confinement, matching ``StableDiffusionCppAdapter._resolve_path`` and
    ``listing._path_exists``'s own plain filesystem checks (cf. the
    #862/#867/#884 review-decline precedent for this class of local,
    operator-controlled path)."""
    binary_raw = (form_values.get("binary_path") or "").strip()
    binary_ok = False
    if binary_raw:
        if shutil.which(binary_raw):
            binary_ok = True
        else:
            binary_path = Path(binary_raw).expanduser()
            binary_ok = binary_path.is_file() and os.access(binary_path, os.X_OK)
    if not binary_ok:
        return ImageGenProbeResult(ok=False, badge="Binary missing or not executable")

    model_raw = (form_values.get("model_path") or "").strip()
    if not model_raw or not Path(model_raw).expanduser().is_file():
        return ImageGenProbeResult(ok=False, badge="Model file missing")

    return ImageGenProbeResult(ok=True, badge="Binary found")


def probe_backend(
    backend_id: str, form_values: Mapping[str, str], secret: str | None
) -> ImageGenProbeResult:
    """Run one short Test-action probe for a backend. BLOCKING.

    Performs a blocking filesystem check or network request -- callers (the
    Settings screen's Test action) MUST run this in a thread worker, never
    on the UI/event loop.

    Behavior per backend:

    - ``stable_diffusion_cpp``: filesystem-only, no network -- see
      ``_probe_sd_cpp``.
    - ``swarmui``: plain reachability GET on ``base_url`` -- see
      ``_probe_swarmui``.
    - ``openrouter``/``together``: OpenAI-compatible ``GET
      {base_url}/models``, authenticated when a secret is available -- see
      ``_probe_openai_compatible``.
    - ``novita``/``modelstudio``/``fal``: unauthenticated reachability
      only. Novita's adapter (``novita_image_adapter.py``) only exposes the
      async submit/poll routes (``/v3/async/txt2img``,
      ``/v3/async/task-result``) -- no cheap authenticated GET was
      confirmed in this codebase's adapter, so novita is probed the same
      way as modelstudio rather than guessing at an unconfirmed endpoint.
      fal's queue API (``queue.fal.run``) has no models-listing route
      either -- same reachability-only treatment.
    - ``gemini``: authenticated ``GET {base_url}/models`` with the key in
      ``x-goog-api-key`` -- Google's models-listing route is a cheap,
      confirmed authenticated GET on the same base as image generation, so
      gemini gets a full auth-verified probe like openrouter/together
      rather than reachability-only.

    Every network probe runs the SSRF egress check
    (``Utils.egress.check_url_or_raise``) before any request.

    Args:
        backend_id: One of ``BACKEND_IDS``.
        form_values: The CURRENT editor fields for this backend, keyed by
            ``FIELD_SCHEMA`` ``toml_key`` (e.g. ``"base_url"``,
            ``"binary_path"``, ``"model_path"``). For a field the user
            hasn't touched this session, the caller is responsible for
            falling back to the effective (saved/env) value before calling.
        secret: The pasted-or-effective secret for this backend, or
            ``None`` when neither is available. Ignored by backends that
            don't use one (``stable_diffusion_cpp``, ``swarmui``).

    Returns:
        An ``ImageGenProbeResult`` whose ``badge`` is one of the spec's
        exact strings and never contains exception text, URLs, headers, or
        credentials.

    Raises:
        ValueError: ``backend_id`` is not a member of ``BACKEND_IDS``.
    """
    if backend_id == "stable_diffusion_cpp":
        return _probe_sd_cpp(form_values)

    base_url = (form_values.get("base_url") or "").strip()
    if backend_id == "swarmui":
        return _probe_swarmui(base_url)
    if backend_id in ("openrouter", "together"):
        return _probe_openai_compatible(base_url, secret)
    if backend_id == "gemini":
        return _probe_gemini(base_url, secret)
    if backend_id in ("novita", "modelstudio", "fal"):
        return _probe_reachability_only(base_url)
    raise ValueError(f"unknown backend_id: {backend_id!r}")
