"""Video Gen guided defaults for the Settings hub -- data layer (task-3401.12).

Mirrors ``settings_image_gen_defaults.py``'s pattern (spec:
``Docs/superpowers/specs/2026-07-25-image-gen-settings-page-design.md``)
for the ``[video_generation]`` section: curated per-backend field schema,
backend status rows, the unmerged user-table reader for display values,
draft -> config-write diffing/validation, and the playback-tool probe that
backs the panel's Diagnostics section (AC3). No Textual widgets live here.

Curated surface is deliberately smaller than image's (three backends, and
advanced keys stay config-file-only): ``allowed_extra_params``, sd.cpp's
vae/llm/lora/steps/cfg/sampler/duration/fps defaults, and the retention
fine print are documented as config.toml keys in the advanced-hint line.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from tldw_chatbook.config import _get_effective_config_path
from tldw_chatbook.Video_Generation.config import (
    VideoGenerationConfig,
    _NON_SECRET,
    _resolve_secret,
    _SECRETS,
)


BACKEND_IDS: tuple[str, ...] = (
    "minimax",
    "comfyui",
    "stable_diffusion_cpp",
)

BACKEND_LABELS: dict[str, str] = {
    "minimax": "MiniMax H3 (cloud)",
    "comfyui": "ComfyUI (local server)",
    "stable_diffusion_cpp": "SD.cpp video (local)",
}


@dataclass(frozen=True)
class FieldSpec:
    """One curated per-backend field, driving both the editor form and validation."""

    toml_key: str
    label: str
    kind: str  # "text" | "url" | "path" | "int" | "bool" | "secret"
    min_value: float | None = None


FIELD_SCHEMA: dict[str, tuple[FieldSpec, ...]] = {
    "minimax": (
        FieldSpec("base_url", "Base URL", "url"),
        FieldSpec("default_model", "Default model", "text"),
        FieldSpec("poll_interval_seconds", "Poll interval (seconds)", "int", min_value=1),
        FieldSpec("timeout_seconds", "Timeout (seconds)", "int", min_value=60),
        FieldSpec("allow_uploads", "Allow image uploads (i2v)", "bool"),
        FieldSpec("api_key", "API key", "secret"),
    ),
    "comfyui": (
        FieldSpec("base_url", "Base URL", "url"),
        FieldSpec("default_workflow", "Default workflow", "text"),
        FieldSpec("timeout_seconds", "Timeout (seconds)", "int", min_value=60),
    ),
    "stable_diffusion_cpp": (
        FieldSpec("binary_path", "Binary path", "path"),
        FieldSpec("diffusion_model_path", "Diffusion model path", "path"),
        FieldSpec("timeout_seconds", "Timeout (seconds)", "int", min_value=60),
    ),
}


def _spec_for(backend_id: str, toml_key: str) -> FieldSpec | None:
    for spec in FIELD_SCHEMA.get(backend_id, ()):
        if spec.toml_key == toml_key:
            return spec
    return None


def _is_minimax_configured(cfg: VideoGenerationConfig) -> bool:
    return bool((cfg.minimax_video_api_key or "").strip())


def _is_comfyui_configured(cfg: VideoGenerationConfig) -> bool:
    return bool((cfg.comfyui_base_url or "").strip())


def _is_sd_cpp_configured(cfg: VideoGenerationConfig) -> bool:
    return bool((cfg.sd_cpp_binary_path or "").strip())


_CONFIGURED_CHECKS = {
    "minimax": _is_minimax_configured,
    "comfyui": _is_comfyui_configured,
    "stable_diffusion_cpp": _is_sd_cpp_configured,
}


@dataclass(frozen=True)
class VideoGenBackendRow:
    """One row of the Backends table."""

    backend_id: str
    label: str
    configured: bool
    enabled: bool
    is_default: bool
    key_source: str


def build_backend_rows(cfg: VideoGenerationConfig) -> list[VideoGenBackendRow]:
    """Build one status row per backend from the effective config."""
    enabled_backends = set(cfg.enabled_backends or [])
    key_sources = cfg.key_sources or {}
    rows: list[VideoGenBackendRow] = []
    for backend_id in BACKEND_IDS:
        enabled = backend_id in enabled_backends
        try:
            configured = bool(_CONFIGURED_CHECKS[backend_id](cfg))
        except Exception:
            configured = False
        rows.append(
            VideoGenBackendRow(
                backend_id=backend_id,
                label=BACKEND_LABELS[backend_id],
                configured=configured,
                enabled=enabled,
                is_default=(cfg.default_backend == backend_id),
                key_source=key_sources.get(backend_id, "missing"),
            )
        )
    return rows


def effective_placeholder(cfg: VideoGenerationConfig, backend_id: str, toml_key: str) -> str:
    """Return the resolved effective value for an unset non-secret field.

    Used as the editor's placeholder text so an empty field never hides
    what will actually be used at generation time. Secrets are out of
    scope -- they're never echoed as placeholders.
    """
    flat_field = _NON_SECRET[(backend_id, toml_key)]
    value = getattr(cfg, flat_field, None)
    return "" if value is None else str(value)


def load_user_video_generation_table() -> Mapping[str, Any]:
    """Read the user's OWN ``[video_generation]`` table, UNMERGED with defaults.

    DISPLAY-ONLY (same rule as the image side's
    ``load_user_image_generation_table``): the merged config makes
    never-typed fields look explicitly set. This parses only the on-disk
    file and returns its raw ``[video_generation]`` table. Never raises.
    """
    try:
        config_path = _get_effective_config_path()
    except Exception as exc:
        logger.debug(
            "video_generation: could not resolve config path (error_type={})",
            type(exc).__name__,
        )
        return {}
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "rb") as f:
            parsed = tomllib.load(f)
    except Exception as exc:
        logger.debug(
            "video_generation: could not parse video-generation config (error_type={})",
            type(exc).__name__,
        )
        return {}
    section = parsed.get("video_generation")
    return section if isinstance(section, dict) else {}


def key_source_after_clear(backend_id: str) -> str:
    """What the key-source line would show for ``backend_id`` if its locally
    saved config value were removed right now (env/keyring fallback, or
    ``"missing"``). Backends with no secret field always return ``"missing"``."""
    if backend_id not in _SECRETS:
        return "missing"
    return _resolve_secret(backend_id, {})[2]


def playback_tool_rows() -> list[tuple[str, bool]]:
    """Playback tool availability for the Diagnostics section (AC3).

    Returns:
        ``(tool, found)`` rows for ffmpeg, ffplay, yt-dlp.
    """
    import shutil

    return [(tool, shutil.which(tool) is not None) for tool in ("ffmpeg", "ffplay", "yt-dlp")]


_GLOBAL_DRAFT_KEYS: tuple[str, ...] = (
    "default_backend",
    "enabled_backends",
    "retention",
    "retention_ttl_hours",
    "max_store_mb",
    "confirm_cost_estimate",
)


def canonical_backend_order(backend_ids: Any) -> list[str]:
    """Normalize an ``enabled_backends``-shaped list to ``BACKEND_IDS``'
    canonical order, dropping any unrecognized entries (the list is a set;
    order has no meaning, but Python list equality is order-sensitive)."""
    ids = set(backend_ids or ())
    return [backend_id for backend_id in BACKEND_IDS if backend_id in ids]


@dataclass(frozen=True)
class VideoGenDraftValues:
    """Pending Settings > Video Gen edits.

    Scalar global fields default to ``None`` = "not touched this session"
    (skipped by ``diff_to_sections``). ``enabled_backends`` is the one
    exception (declared list, never ``None``). ``backend_fields`` holds the
    edited raw strings (``backend_id -> toml_key -> raw``; bool fields hold
    real bools); ``cleared_fields`` holds explicit Clear actions.
    """

    default_backend: str | None = None
    enabled_backends: list[str] = field(default_factory=list)
    retention: str | None = None
    retention_ttl_hours: int | None = None
    max_store_mb: int | None = None
    confirm_cost_estimate: bool | None = None
    backend_fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    cleared_fields: dict[str, list[str]] = field(default_factory=dict)


def _coerce_value(spec: FieldSpec | None, raw_value: Any) -> Any:
    """Coerce an edited raw value per its FieldSpec kind (int/bool)."""
    if spec is None:
        return raw_value
    if spec.kind == "int":
        try:
            return int(str(raw_value).strip())
        except (TypeError, ValueError):
            return raw_value  # validate_draft() catches this
    if spec.kind == "bool":
        if isinstance(raw_value, bool):
            return raw_value
        return str(raw_value).strip().lower() in {"true", "1", "yes", "on"}
    return raw_value


def diff_to_sections(
    draft: VideoGenDraftValues, raw_config: Mapping
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Diff a draft against the RAW config mapping -- never a resolved config.

    Same guarantee as the image side: only the draft and the raw
    ``[video_generation]`` TOML table are compared, so an env/keyring
    secret can never be copied into plaintext config.toml. Field-level
    (editing one field never carries a sibling along); emptied-but-set
    keys become deletions, never empty-string sentinels.
    """
    raw_top: Mapping[str, Any] = (raw_config or {}).get("video_generation") or {}
    sections: dict[str, dict[str, Any]] = {}
    deletions: dict[str, list[str]] = {}

    global_diff: dict[str, Any] = {}
    global_deletions: list[str] = []
    for key in _GLOBAL_DRAFT_KEYS:
        value = getattr(draft, key)
        if value is None:
            continue
        if key == "enabled_backends":
            normalized_value = canonical_backend_order(value)
            raw_value = canonical_backend_order(raw_top.get(key))
            if normalized_value != raw_value:
                global_diff[key] = normalized_value
            continue
        if isinstance(value, str) and not value.strip():
            if key in raw_top:
                global_deletions.append(key)
            continue
        raw_value = raw_top.get(key)
        if value != raw_value:
            global_diff[key] = value
    if global_diff:
        sections["video_generation"] = global_diff
    if global_deletions:
        deletions["video_generation"] = global_deletions

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
                continue
            spec = _spec_for(backend_id, toml_key)
            coerced = _coerce_value(spec, raw_value)
            if coerced != raw_backend.get(toml_key):
                backend_diff[toml_key] = coerced
        if backend_diff:
            sections[f"video_generation.{backend_id}"] = backend_diff
        if empty_deletions:
            section_key = f"video_generation.{backend_id}"
            deletions[section_key] = sorted(
                set(deletions.get(section_key, ())) | empty_deletions
            )

    for backend_id, keys in draft.cleared_fields.items():
        if keys:
            section_key = f"video_generation.{backend_id}"
            deletions[section_key] = sorted(
                set(deletions.get(section_key, ())) | set(keys)
            )

    return sections, deletions


_GLOBAL_INT_FIELD_SPECS: tuple[tuple[str, str, int], ...] = (
    ("retention_ttl_hours", "Retention TTL (hours)", 1),
    ("max_store_mb", "Store cap (MB)", 1),
)

_RETENTION_CHOICES = frozenset({"session", "ttl"})


def validate_draft(draft: VideoGenDraftValues) -> tuple[list[str], list[str]]:
    """Validate a draft before it can be saved.

    Returns:
        ``(errors, warnings)``. Errors block the save: default backend must
        be enabled when any backend is; int fields must parse and meet
        minimums; retention must be ``session`` or ``ttl``. Warnings never
        block (e.g. every backend disabled).
    """
    errors: list[str] = []
    warnings: list[str] = []

    if draft.retention is not None and draft.retention not in _RETENTION_CHOICES:
        errors.append(
            f"Retention must be one of {sorted(_RETENTION_CHOICES)}; got {draft.retention!r}."
        )

    for key, label, minimum in _GLOBAL_INT_FIELD_SPECS:
        value = getattr(draft, key)
        if value is None:
            continue
        parsed: int | None
        try:
            parsed = int(str(value).strip())
        except (TypeError, ValueError):
            parsed = None
        if parsed is None or parsed < minimum:
            errors.append(f"{label} must be an integer >= {minimum}.")

    for backend_id, fields in draft.backend_fields.items():
        for toml_key, raw_value in fields.items():
            spec = _spec_for(backend_id, toml_key)
            if spec is not None and spec.kind == "int":
                try:
                    parsed = int(str(raw_value).strip())
                except (TypeError, ValueError):
                    parsed = None
                minimum = spec.min_value if spec.min_value is not None else 1
                if parsed is None or parsed < minimum:
                    errors.append(
                        f"{BACKEND_LABELS.get(backend_id, backend_id)} {spec.label} "
                        f"must be an integer >= {minimum}."
                    )

    if draft.enabled_backends:
        if draft.default_backend is not None and draft.default_backend not in draft.enabled_backends:
            errors.append("The default backend must be one of the enabled backends.")
    elif draft.enabled_backends is not None and not draft.enabled_backends and draft.default_backend:
        warnings.append("No backends are enabled; generation commands will refuse to run.")

    return errors, warnings
