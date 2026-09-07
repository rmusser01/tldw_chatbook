"""Configuration helpers for video generation backends.

Mirrors ``Image_Generation/config.py`` mechanics: nested
``[video_generation]`` TOML (globals + ``[video_generation.<backend>]``
subsections), secret precedence env -> config -> keyring (namespace
``tldw_chatbook_videogen``), and warn-on-unknown-key for the flat-spelling
mistake. Field names carry the ``video_`` infix so the flat dataclass fields
can never collide with the image package's.

Storage/ephemerality knobs (``retention``, ``retention_ttl_hours``,
``max_store_mb``) are consumed by the VideoStore (task-3401.4); they live
here so all ``[video_generation]`` parsing has exactly one owner.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import keyring
from loguru import logger


DEFAULT_BACKEND = "stable_diffusion_cpp"
DEFAULT_MAX_DURATION_SECONDS = 15
DEFAULT_MAX_FPS = 30
DEFAULT_MAX_WIDTH = 2560
DEFAULT_MAX_HEIGHT = 1440
DEFAULT_MAX_PIXELS = 2560 * 1440
DEFAULT_MAX_STEPS = 50
DEFAULT_MAX_PROMPT_LENGTH = 7000  # MiniMax-H3 documented prompt cap
DEFAULT_MAX_REFERENCE_ASSETS = 12  # MiniMax-H3 mixed-input cap
DEFAULT_DOWNLOAD_MAX_MB = 500
DEFAULT_RETENTION = "session"
DEFAULT_RETENTION_TTL_HOURS = 24
DEFAULT_MAX_STORE_MB = 2048
DEFAULT_CONFIRM_COST_ESTIMATE = True

DEFAULT_MINIMAX_VIDEO_BASE_URL = "https://api.minimax.io"
DEFAULT_MINIMAX_VIDEO_MODEL = "MiniMax-H3"
DEFAULT_MINIMAX_VIDEO_POLL_INTERVAL_SECONDS = 10
DEFAULT_MINIMAX_VIDEO_TIMEOUT_SECONDS = 600

DEFAULT_COMFYUI_BASE_URL = "http://127.0.0.1:8188"
DEFAULT_COMFYUI_WORKFLOW = "minimax_h3_t2v.json"
DEFAULT_COMFYUI_TIMEOUT_SECONDS = 1800

DEFAULT_SD_CPP_VIDEO_STEPS = 25
DEFAULT_SD_CPP_VIDEO_CFG_SCALE = 7.5
DEFAULT_SD_CPP_VIDEO_SAMPLER = "euler_a"
DEFAULT_SD_CPP_VIDEO_DEVICE = "auto"
DEFAULT_SD_CPP_VIDEO_DURATION_SECONDS = 3
DEFAULT_SD_CPP_VIDEO_FPS = 16
DEFAULT_SD_CPP_VIDEO_TIMEOUT_SECONDS = 7200

# Secret fields: backend -> (flat_field_name, [env vars in precedence
# order], keyring_backend_id, nested [video_generation.<backend>] TOML key
# the secret is read from/written to). Only minimax takes an API key today
# (ComfyUI is a user-run local server; sd.cpp is a local binary). Precedence
# and the ``_resolve_secret`` fallback rule match the image package exactly.
_SECRETS = {
    "minimax": ("minimax_video_api_key", ["MINIMAX_API_KEY"], "minimax", "api_key"),
}
# Non-secret nested keys: (backend, toml_key) -> flat_field_name
_NON_SECRET = {
    ("minimax", "base_url"):              "minimax_video_base_url",
    ("minimax", "default_model"):         "minimax_video_default_model",
    ("minimax", "poll_interval_seconds"): "minimax_video_poll_interval_seconds",
    ("minimax", "timeout_seconds"):       "minimax_video_timeout_seconds",
    ("minimax", "allow_uploads"):         "minimax_video_allow_uploads",
    ("minimax", "allowed_extra_params"):  "minimax_video_allowed_extra_params",
    ("comfyui", "base_url"):              "comfyui_base_url",
    ("comfyui", "default_model"):         "comfyui_default_model",
    ("comfyui", "default_workflow"):      "comfyui_default_workflow",
    ("comfyui", "timeout_seconds"):       "comfyui_timeout_seconds",
    ("comfyui", "allowed_extra_params"):  "comfyui_allowed_extra_params",
    ("stable_diffusion_cpp", "binary_path"):          "sd_cpp_binary_path",
    ("stable_diffusion_cpp", "diffusion_model_path"): "sd_cpp_diffusion_model_path",
    ("stable_diffusion_cpp", "model_path"):           "sd_cpp_model_path",
    ("stable_diffusion_cpp", "llm_path"):             "sd_cpp_llm_path",
    ("stable_diffusion_cpp", "vae_path"):             "sd_cpp_vae_path",
    ("stable_diffusion_cpp", "lora_paths"):           "sd_cpp_lora_paths",
    ("stable_diffusion_cpp", "device"):               "sd_cpp_device",
    ("stable_diffusion_cpp", "default_steps"):        "sd_cpp_default_steps",
    ("stable_diffusion_cpp", "default_cfg_scale"):    "sd_cpp_default_cfg_scale",
    ("stable_diffusion_cpp", "default_sampler"):      "sd_cpp_default_sampler",
    ("stable_diffusion_cpp", "default_duration_seconds"): "sd_cpp_default_duration_seconds",
    ("stable_diffusion_cpp", "default_fps"):          "sd_cpp_default_fps",
    ("stable_diffusion_cpp", "timeout_seconds"):      "sd_cpp_timeout_seconds",
    ("stable_diffusion_cpp", "allowed_extra_params"): "sd_cpp_allowed_extra_params",
}
_GLOBAL_KEYS = [
    "default_backend", "enabled_backends", "max_duration_seconds", "max_fps",
    "max_width", "max_height", "max_pixels", "max_steps", "max_prompt_length",
    "max_reference_assets", "download_max_mb", "retention",
    "retention_ttl_hours", "max_store_mb", "confirm_cost_estimate",
]

# flat_field_name -> (backend, toml_key), derived by reversing _NON_SECRET
# and _SECRETS -- used only to build a helpful unknown-key warning below,
# never to accept the flat spelling itself (same decision as the image
# package's task-621: warn, don't alias, to avoid two spellings of the same
# setting needing a collision-precedence rule).
_FLAT_MAP: dict[str, tuple[str, str]] = {
    flat_field: (backend, toml_key) for (backend, toml_key), flat_field in _NON_SECRET.items()
}
_FLAT_MAP.update({
    flat_field: (backend, config_key)
    for backend, (flat_field, _env_vars, _kr_id, config_key) in _SECRETS.items()
})

# Known [video_generation.<backend>] subsection names.
_BACKEND_NAMES = set(_SECRETS) | {backend for backend, _toml_key in _NON_SECRET}


def _warn_unknown_top_level_keys(raw: dict) -> None:
    """Warn once per unknown key found directly under ``[video_generation]``.

    Mirrors the image package: a backend field written using its *flat*
    dataclass name straight under ``[video_generation]`` matches nothing and
    is silently ignored -- this surfaces that mistake with the exact nested
    replacement to use. Never raises (config loading must never crash on a
    malformed/unexpected key).
    """
    try:
        for key in raw:
            if not isinstance(key, str):
                continue
            if key in _GLOBAL_KEYS or key in _BACKEND_NAMES:
                continue
            target = _FLAT_MAP.get(key)
            if target is not None:
                backend, toml_key = target
                logger.warning(
                    f"[video_generation] unknown key '{key}' is ignored -- flat backend keys are "
                    f"not read here; use [video_generation.{backend}] {toml_key} = ... instead"
                )
            else:
                logger.warning(f"[video_generation] unknown key '{key}' is ignored")
    except Exception as e:  # never let a malformed section crash config loading
        logger.debug(
            "video_generation unknown-key scan failed (error_type={})",
            type(e).__name__,
        )


def _read_video_generation_toml() -> dict:
    """Return the raw [video_generation] section dict (nested). Patch point in tests."""
    from tldw_chatbook.config import load_settings
    return load_settings().get("video_generation", {}) or {}


def _keyring_get(backend: str):
    """Namespaced keyring lookup; never raises. Patch point in tests."""
    try:
        return keyring.get_password("tldw_chatbook_videogen", backend)
    except Exception as e:  # keyring backend may be unavailable
        logger.debug(
            "keyring lookup failed for videogen/{} (error_type={})",
            backend,
            type(e).__name__,
        )
        return None


def _resolve_secret(backend: str, sub: dict):
    """Resolve one backend's secret and where it came from.

    Returns ``(flat_field_name, value, source)`` where ``source`` is one of
    ``"env:<VAR>"``, ``"config"``, ``"keyring"``, or ``"missing"``.
    """
    field_name, env_vars, kr_id, config_key = _SECRETS[backend]
    for ev in env_vars:                       # 1. env
        v = os.getenv(ev)
        if v:
            return field_name, v, f"env:{ev}"
    sub = sub or {}
    cfg_val = sub.get(config_key)             # 2. config
    if cfg_val and cfg_val != "<API_KEY_HERE>":
        return field_name, cfg_val, "config"
    kr = _keyring_get(kr_id)                  # 3. keyring
    if kr:
        return field_name, kr, "keyring"
    return field_name, None, "missing"


def _load_video_generation_section() -> tuple[dict, dict[str, str]]:
    """Assemble the FLAT mapping the config builder expects, from nested TOML + env + keyring.

    Returns ``(flat, key_sources)``; ``key_sources`` maps every known backend
    id to where its secret was resolved from.
    """
    raw = _read_video_generation_toml()
    if not isinstance(raw, dict):
        raw = {}
    _warn_unknown_top_level_keys(raw)
    flat: dict = {}
    for k in _GLOBAL_KEYS:
        if k in raw:
            flat[k] = raw[k]
    for (backend, toml_key), flat_field in _NON_SECRET.items():
        sub = raw.get(backend) or {}
        if not isinstance(sub, dict):
            sub = {}
        if toml_key in sub:
            flat[flat_field] = sub[toml_key]
    key_sources: dict[str, str] = {backend: "missing" for backend in _BACKEND_NAMES}
    for backend in _SECRETS:
        sub = raw.get(backend) or {}
        if not isinstance(sub, dict):
            sub = {}
        field_name, value, source = _resolve_secret(backend, sub)
        key_sources[backend] = source
        if value:
            flat[field_name] = value
    return flat, key_sources


@dataclass(frozen=True)
class VideoStorePolicy:
    """The only three settings ``VideoStore`` reads. No secrets involved.

    Field names and value normalization match ``VideoGenerationConfig``
    exactly, so the store cannot tell the two apart (it reads all three via
    ``getattr(config, name, default)``).
    """

    retention: str
    retention_ttl_hours: int
    max_store_mb: int


def get_video_store_policy() -> VideoStorePolicy:
    """Read the generated-video retention/capacity policy without any secret.

    ``VideoStore`` is constructed and asked to ``enforce_retention()`` inside
    ``TldwCli.__init__``. Routing that through ``get_video_generation_config()``
    made every single boot resolve the MiniMax API key, whose last resort is
    ``keyring.get_password(...)`` -- a real OS credential-store round trip,
    measured at **18.2 ms** on macOS (11.3 ms of keyring backend discovery +
    the Security.framework ctypes load, then the Keychain query itself), for a
    secret the store never looks at. On a locked keychain that call can block
    or raise a consent dialog during startup. TASK-21111(b).

    Returns:
        The retention mode, TTL and capacity, normalized exactly as
        ``get_video_generation_config`` normalizes them.
    """
    raw = _read_video_generation_toml()
    if not isinstance(raw, dict):
        raw = {}
    return VideoStorePolicy(
        retention=_coerce_choice(
            raw.get("retention"), default=DEFAULT_RETENTION, allowed={"session", "ttl"}
        ),
        retention_ttl_hours=max(
            1, _coerce_int(raw.get("retention_ttl_hours"), DEFAULT_RETENTION_TTL_HOURS)
        ),
        max_store_mb=max(1, _coerce_int(raw.get("max_store_mb"), DEFAULT_MAX_STORE_MB)),
    )


@dataclass(frozen=True)
class VideoGenerationConfig:
    default_backend: str | None
    enabled_backends: list[str]
    max_duration_seconds: int
    max_fps: int
    max_width: int
    max_height: int
    max_pixels: int
    max_steps: int
    max_prompt_length: int
    max_reference_assets: int
    download_max_mb: int
    retention: str
    retention_ttl_hours: int
    max_store_mb: int
    confirm_cost_estimate: bool
    minimax_video_base_url: str | None
    minimax_video_api_key: str | None
    minimax_video_default_model: str | None
    minimax_video_poll_interval_seconds: int
    minimax_video_timeout_seconds: int
    minimax_video_allow_uploads: bool
    minimax_video_allowed_extra_params: list[str]
    comfyui_base_url: str | None
    comfyui_default_model: str | None
    comfyui_default_workflow: str | None
    comfyui_timeout_seconds: int
    comfyui_allowed_extra_params: list[str]
    sd_cpp_binary_path: str | None
    sd_cpp_diffusion_model_path: str | None
    sd_cpp_model_path: str | None
    sd_cpp_llm_path: str | None
    sd_cpp_vae_path: str | None
    sd_cpp_lora_paths: list[str]
    sd_cpp_device: str
    sd_cpp_default_steps: int
    sd_cpp_default_cfg_scale: float
    sd_cpp_default_sampler: str
    sd_cpp_default_duration_seconds: int
    sd_cpp_default_fps: int
    sd_cpp_timeout_seconds: int
    sd_cpp_allowed_extra_params: list[str]
    # backend id -> "env:<VAR>" | "config" | "keyring" | "missing". Purely
    # additive/read-only metadata about where each backend's secret was
    # resolved from (same contract as the image package's key_sources).
    key_sources: dict[str, str] = field(default_factory=dict)


_config_cache: VideoGenerationConfig | None = None


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return default


def _coerce_choice(
    value: Any,
    *,
    default: str,
    allowed: set[str],
) -> str:
    """Normalize a string choice to lowercase and return `default` when invalid."""
    raw = str(value or "").strip().lower()
    if raw in allowed:
        return raw
    return default


def _parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    raw = str(value).strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _get_config_value(section: dict[str, Any], key: str) -> str | None:
    raw = section.get(key)
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def get_video_generation_config(*, reload: bool = False) -> VideoGenerationConfig:
    global _config_cache
    if _config_cache is not None and not reload:
        return _config_cache

    section, key_sources = _load_video_generation_section()

    config = VideoGenerationConfig(
        default_backend=_get_config_value(section, "default_backend") or DEFAULT_BACKEND,
        enabled_backends=_parse_list(section.get("enabled_backends")),
        max_duration_seconds=max(1, _coerce_int(section.get("max_duration_seconds"), DEFAULT_MAX_DURATION_SECONDS)),
        max_fps=max(1, _coerce_int(section.get("max_fps"), DEFAULT_MAX_FPS)),
        max_width=_coerce_int(section.get("max_width"), DEFAULT_MAX_WIDTH),
        max_height=_coerce_int(section.get("max_height"), DEFAULT_MAX_HEIGHT),
        max_pixels=_coerce_int(section.get("max_pixels"), DEFAULT_MAX_PIXELS),
        max_steps=_coerce_int(section.get("max_steps"), DEFAULT_MAX_STEPS),
        max_prompt_length=_coerce_int(section.get("max_prompt_length"), DEFAULT_MAX_PROMPT_LENGTH),
        max_reference_assets=max(1, _coerce_int(section.get("max_reference_assets"), DEFAULT_MAX_REFERENCE_ASSETS)),
        download_max_mb=max(1, _coerce_int(section.get("download_max_mb"), DEFAULT_DOWNLOAD_MAX_MB)),
        retention=_coerce_choice(
            section.get("retention"), default=DEFAULT_RETENTION, allowed={"session", "ttl"},
        ),
        retention_ttl_hours=max(1, _coerce_int(section.get("retention_ttl_hours"), DEFAULT_RETENTION_TTL_HOURS)),
        max_store_mb=max(1, _coerce_int(section.get("max_store_mb"), DEFAULT_MAX_STORE_MB)),
        confirm_cost_estimate=_coerce_bool(section.get("confirm_cost_estimate"), DEFAULT_CONFIRM_COST_ESTIMATE),
        minimax_video_base_url=_get_config_value(section, "minimax_video_base_url")
        or DEFAULT_MINIMAX_VIDEO_BASE_URL,
        minimax_video_api_key=_get_config_value(section, "minimax_video_api_key"),
        minimax_video_default_model=_get_config_value(section, "minimax_video_default_model")
        or DEFAULT_MINIMAX_VIDEO_MODEL,
        minimax_video_poll_interval_seconds=max(
            1,
            _coerce_int(
                section.get("minimax_video_poll_interval_seconds"),
                DEFAULT_MINIMAX_VIDEO_POLL_INTERVAL_SECONDS,
            ),
        ),
        minimax_video_timeout_seconds=_coerce_int(
            section.get("minimax_video_timeout_seconds"),
            DEFAULT_MINIMAX_VIDEO_TIMEOUT_SECONDS,
        ),
        minimax_video_allow_uploads=_coerce_bool(section.get("minimax_video_allow_uploads"), False),
        minimax_video_allowed_extra_params=_parse_list(section.get("minimax_video_allowed_extra_params")),
        comfyui_base_url=_get_config_value(section, "comfyui_base_url") or DEFAULT_COMFYUI_BASE_URL,
        comfyui_default_model=_get_config_value(section, "comfyui_default_model"),
        comfyui_default_workflow=(
            _get_config_value(section, "comfyui_default_workflow")
            or DEFAULT_COMFYUI_WORKFLOW
        ),
        comfyui_timeout_seconds=_coerce_int(
            section.get("comfyui_timeout_seconds"),
            DEFAULT_COMFYUI_TIMEOUT_SECONDS,
        ),
        comfyui_allowed_extra_params=_parse_list(section.get("comfyui_allowed_extra_params")),
        sd_cpp_binary_path=_get_config_value(section, "sd_cpp_binary_path"),
        sd_cpp_diffusion_model_path=_get_config_value(section, "sd_cpp_diffusion_model_path"),
        sd_cpp_model_path=_get_config_value(section, "sd_cpp_model_path"),
        sd_cpp_llm_path=_get_config_value(section, "sd_cpp_llm_path"),
        sd_cpp_vae_path=_get_config_value(section, "sd_cpp_vae_path"),
        sd_cpp_lora_paths=_parse_list(section.get("sd_cpp_lora_paths")),
        sd_cpp_device=_get_config_value(section, "sd_cpp_device") or DEFAULT_SD_CPP_VIDEO_DEVICE,
        sd_cpp_default_steps=_coerce_int(section.get("sd_cpp_default_steps"), DEFAULT_SD_CPP_VIDEO_STEPS),
        sd_cpp_default_cfg_scale=_coerce_float(
            section.get("sd_cpp_default_cfg_scale"), DEFAULT_SD_CPP_VIDEO_CFG_SCALE,
        ),
        sd_cpp_default_sampler=_get_config_value(section, "sd_cpp_default_sampler")
        or DEFAULT_SD_CPP_VIDEO_SAMPLER,
        sd_cpp_default_duration_seconds=max(
            1,
            _coerce_int(section.get("sd_cpp_default_duration_seconds"), DEFAULT_SD_CPP_VIDEO_DURATION_SECONDS),
        ),
        sd_cpp_default_fps=max(
            1, _coerce_int(section.get("sd_cpp_default_fps"), DEFAULT_SD_CPP_VIDEO_FPS),
        ),
        sd_cpp_timeout_seconds=_coerce_int(
            section.get("sd_cpp_timeout_seconds"), DEFAULT_SD_CPP_VIDEO_TIMEOUT_SECONDS,
        ),
        sd_cpp_allowed_extra_params=_parse_list(section.get("sd_cpp_allowed_extra_params")),
        key_sources=key_sources,
    )

    _config_cache = config
    return config


def reset_video_generation_config_cache() -> None:
    global _config_cache
    _config_cache = None


def reset_video_generation_runtime() -> None:
    """Invalidate cached video configuration and adapter instances.

    The registry import stays local so configuration loading remains independent
    from adapter construction at module-import time.
    """
    reset_video_generation_config_cache()
    from tldw_chatbook.Video_Generation.adapter_registry import reset_registry

    reset_registry()
