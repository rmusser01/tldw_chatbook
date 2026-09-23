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

from dataclasses import dataclass, field
from typing import Any


from tldw_chatbook.Media_Generation.config_machinery import ModalityConfigTables, SecretSpec
from tldw_chatbook.Media_Generation.config_machinery import coerce_bool
from tldw_chatbook.Media_Generation.config_machinery import coerce_choice
from tldw_chatbook.Media_Generation.config_machinery import coerce_float
from tldw_chatbook.Media_Generation.config_machinery import coerce_int
from tldw_chatbook.Media_Generation.config_machinery import get_config_value
from tldw_chatbook.Media_Generation.config_machinery import keyring_get as media_keyring_get
from tldw_chatbook.Media_Generation.config_machinery import load_generation_section
from tldw_chatbook.Media_Generation.config_machinery import parse_list
from tldw_chatbook.Media_Generation.config_machinery import (
    resolve_secret,
    warn_unknown_top_level_keys,
)


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



def _read_video_generation_toml() -> dict:
    """Return the raw [video_generation] section dict (nested). Patch point in tests."""
    from tldw_chatbook.config import load_settings
    return load_settings().get("video_generation", {}) or {}







# ADR-176: the mechanics below live in Media_Generation.config_machinery;
# these delegates keep the module-level names as test patch points and
# builder call sites.
_TABLES = ModalityConfigTables(
    section_name="video_generation",
    keyring_namespace="tldw_chatbook_videogen",
    keyring_label="videogen",
    global_keys=tuple(_GLOBAL_KEYS),
    secrets={
        backend: SecretSpec(flat_field, tuple(env_vars), kr_id, config_key)
        for backend, (flat_field, env_vars, kr_id, config_key) in _SECRETS.items()
    },
    non_secret=dict(_NON_SECRET),
)

_coerce_int = coerce_int
_coerce_float = coerce_float
_coerce_bool = coerce_bool
_coerce_choice = coerce_choice
_parse_list = parse_list
_get_config_value = get_config_value


def _warn_unknown_top_level_keys(raw: dict) -> None:
    warn_unknown_top_level_keys(raw, _TABLES)


def _keyring_get(backend: str):
    """Namespaced keyring lookup; never raises. Patch point in tests."""
    return media_keyring_get(backend, _TABLES)


def _resolve_secret(backend: str, sub: dict):
    return resolve_secret(backend, sub, _TABLES, keyring_lookup=_keyring_get)


def _load_video_generation_section() -> tuple[dict, dict[str, str]]:
    return load_generation_section(
        _TABLES, read_toml=_read_video_generation_toml, keyring_lookup=_keyring_get
    )
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
