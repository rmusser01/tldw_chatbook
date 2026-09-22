"""Configuration helpers for image generation backends."""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import keyring
from loguru import logger

from tldw_chatbook.Media_Generation.config_machinery import ModalityConfigTables, SecretSpec
from tldw_chatbook.Media_Generation.config_machinery import coerce_choice
from tldw_chatbook.Media_Generation.config_machinery import coerce_float
from tldw_chatbook.Media_Generation.config_machinery import coerce_int
from tldw_chatbook.Media_Generation.config_machinery import get_config_value
from tldw_chatbook.Media_Generation.config_machinery import keyring_get as media_keyring_get
from tldw_chatbook.Media_Generation.config_machinery import load_generation_section
from tldw_chatbook.Media_Generation.config_machinery import parse_list
from tldw_chatbook.Media_Generation.config_machinery import resolve_secret, warn_unknown_top_level_keys
# ADR-176: the mechanics below live in Media_Generation.config_machinery;


DEFAULT_BACKEND = "stable_diffusion_cpp"
DEFAULT_MAX_WIDTH = 1024
DEFAULT_MAX_HEIGHT = 1024
DEFAULT_MAX_PIXELS = 1024 * 1024
DEFAULT_MAX_STEPS = 50
DEFAULT_MAX_PROMPT_LENGTH = 1000
DEFAULT_INLINE_MAX_BYTES = 4_000_000
DEFAULT_IMAGE_BATCH = 1
DEFAULT_MAX_VARIANTS_PER_MESSAGE = 8

# Task-559 AC1: LLM-composed `/generate-image` conversation-context prompt.
# `context_llm_enabled` is a kill-switch (default on, safe to flip off);
# any failure at call time (no ready provider, exception, timeout, empty
# response) ALSO falls back to the keyword extractor regardless of this
# flag -- see `Chat/console_generate_image.py`.
DEFAULT_CONTEXT_LLM_ENABLED = True
DEFAULT_CONTEXT_LLM_TURNS = 10
DEFAULT_CONTEXT_LLM_TIMEOUT_SECONDS = 15.0

DEFAULT_SD_CPP_STEPS = 25
DEFAULT_SD_CPP_CFG_SCALE = 7.5
DEFAULT_SD_CPP_SAMPLER = "euler_a"
DEFAULT_SD_CPP_DEVICE = "auto"
DEFAULT_SD_CPP_TIMEOUT_SECONDS = 120
DEFAULT_SWARMUI_BASE_URL = "http://127.0.0.1:7801"
DEFAULT_SWARMUI_TIMEOUT_SECONDS = 120
DEFAULT_OPENROUTER_IMAGE_BASE_URL = "https://openrouter.ai/api/v1"
# task-620: the original default ("openai/gpt-image-1", ported verbatim from
# tldw_server) was retired from OpenRouter's catalog and 404s on every
# request. Verified live against OpenRouter's catalog 2026-07-25:
# "google/gemini-2.5-flash-image" exists, is cheap, and is fast -- picked
# over "openai/gpt-5-image-mini" on cost. Re-verify against the live catalog
# before changing this again; OpenRouter's image-model lineup moves.
DEFAULT_OPENROUTER_IMAGE_MODEL = "google/gemini-2.5-flash-image"
DEFAULT_OPENROUTER_IMAGE_TIMEOUT_SECONDS = 120
DEFAULT_NOVITA_IMAGE_BASE_URL = "https://api.novita.ai"
DEFAULT_NOVITA_IMAGE_MODEL = "sd_xl_base_1.0.safetensors"
DEFAULT_NOVITA_IMAGE_TIMEOUT_SECONDS = 180
DEFAULT_NOVITA_IMAGE_POLL_INTERVAL_SECONDS = 2
DEFAULT_TOGETHER_IMAGE_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_TOGETHER_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell-Free"
DEFAULT_TOGETHER_IMAGE_TIMEOUT_SECONDS = 120
DEFAULT_MODELSTUDIO_IMAGE_BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1"
DEFAULT_MODELSTUDIO_IMAGE_MODEL = "qwen-image"
DEFAULT_MODELSTUDIO_IMAGE_REGION = "sg"
DEFAULT_MODELSTUDIO_IMAGE_MODE = "auto"
DEFAULT_MODELSTUDIO_IMAGE_POLL_INTERVAL_SECONDS = 2
DEFAULT_MODELSTUDIO_IMAGE_TIMEOUT_SECONDS = 180
DEFAULT_FAL_IMAGE_BASE_URL = "https://queue.fal.run"
DEFAULT_FAL_IMAGE_MODEL = "fal-ai/flux/schnell"
DEFAULT_FAL_IMAGE_POLL_INTERVAL_SECONDS = 2
DEFAULT_FAL_IMAGE_TIMEOUT_SECONDS = 120
DEFAULT_GEMINI_IMAGE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"
DEFAULT_GEMINI_IMAGE_TIMEOUT_SECONDS = 120
DEFAULT_COMFYUI_IMAGE_BASE_URL = "http://127.0.0.1:8188"
DEFAULT_COMFYUI_IMAGE_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_COMFYUI_IMAGE_CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_COMFYUI_IMAGE_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_COMFYUI_IMAGE_TOTAL_DEADLINE_SECONDS = 1800.0

# Secret fields: backend -> (flat_field_name, [env vars in precedence
# order], keyring_backend_id, nested [image_generation.<backend>] TOML
# key the secret is read from/written to). The nested key is DATA, not a
# hardcoded "api_key" literal (final review CRITICAL fix): swarmui's real
# field is `swarm_token` (matches its flat name AND the Settings > Image
# Gen FIELD_SCHEMA/spec) -- hardcoding "api_key" for every backend meant a
# pasted+saved swarm token landed in config.toml but was never actually
# read back (_resolve_secret always looked for "api_key", which swarmui
# never writes), so it resolved to "missing" forever. `_resolve_secret`
# falls back to "api_key" ONLY when a backend's own config_key is unset,
# for backward compatibility with any config hand-written against the
# pre-fix (undocumented, but functional for every OTHER backend) behavior.
_SECRETS = {
    "swarmui":     ("swarmui_swarm_token",        ["SWARMUI_TOKEN"],                       "swarmui",     "swarm_token"),
    "openrouter":  ("openrouter_image_api_key",   ["OPENROUTER_API_KEY"],                  "openrouter",  "api_key"),
    "novita":      ("novita_image_api_key",       ["NOVITA_API_KEY"],                      "novita",      "api_key"),
    "together":    ("together_image_api_key",     ["TOGETHER_API_KEY"],                    "together",    "api_key"),
    "modelstudio": ("modelstudio_image_api_key",  ["DASHSCOPE_API_KEY", "QWEN_API_KEY"],   "modelstudio", "api_key"),
    "fal":         ("fal_image_api_key",          ["FAL_KEY"],                             "fal",         "api_key"),
    "gemini":      ("gemini_image_api_key",       ["GEMINI_API_KEY", "GOOGLE_API_KEY"],    "gemini",      "api_key"),
}
# Non-secret nested keys: (backend, toml_key) -> flat_field_name
# NOTE: `reference_image_supported_models` is intentionally NOT mapped here —
# reference-image support is deferred (reference_images.py was dropped in Phase 1),
# so the dataclass field correctly defaults to {} until a later phase wires it.
_NON_SECRET = {
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
    ("stable_diffusion_cpp", "timeout_seconds"):      "sd_cpp_timeout_seconds",
    ("stable_diffusion_cpp", "allowed_extra_params"): "sd_cpp_allowed_extra_params",
    ("swarmui", "base_url"):              "swarmui_base_url",
    ("swarmui", "default_model"):         "swarmui_default_model",
    ("swarmui", "timeout_seconds"):       "swarmui_timeout_seconds",
    ("swarmui", "allowed_extra_params"):  "swarmui_allowed_extra_params",
    ("openrouter", "base_url"):              "openrouter_image_base_url",
    ("openrouter", "default_model"):         "openrouter_image_default_model",
    ("openrouter", "timeout_seconds"):       "openrouter_image_timeout_seconds",
    ("openrouter", "allowed_extra_params"):  "openrouter_image_allowed_extra_params",
    ("novita", "base_url"):              "novita_image_base_url",
    ("novita", "default_model"):         "novita_image_default_model",
    ("novita", "timeout_seconds"):       "novita_image_timeout_seconds",
    ("novita", "poll_interval_seconds"): "novita_image_poll_interval_seconds",
    ("novita", "allowed_extra_params"):  "novita_image_allowed_extra_params",
    ("together", "base_url"):              "together_image_base_url",
    ("together", "default_model"):         "together_image_default_model",
    ("together", "timeout_seconds"):       "together_image_timeout_seconds",
    ("together", "allowed_extra_params"):  "together_image_allowed_extra_params",
    ("modelstudio", "base_url"):              "modelstudio_image_base_url",
    ("modelstudio", "default_model"):         "modelstudio_image_default_model",
    ("modelstudio", "region"):                "modelstudio_image_region",
    ("modelstudio", "mode"):                  "modelstudio_image_mode",
    ("modelstudio", "poll_interval_seconds"): "modelstudio_image_poll_interval_seconds",
    ("modelstudio", "timeout_seconds"):       "modelstudio_image_timeout_seconds",
    ("modelstudio", "allowed_extra_params"):  "modelstudio_image_allowed_extra_params",
    ("fal", "base_url"):              "fal_image_base_url",
    ("fal", "default_model"):         "fal_image_default_model",
    ("fal", "poll_interval_seconds"): "fal_image_poll_interval_seconds",
    ("fal", "timeout_seconds"):       "fal_image_timeout_seconds",
    ("gemini", "base_url"):           "gemini_image_base_url",
    ("gemini", "default_model"):      "gemini_image_default_model",
    ("gemini", "timeout_seconds"):    "gemini_image_timeout_seconds",
    ("comfyui", "base_url"):                "comfyui_image_base_url",
    ("comfyui", "request_timeout_seconds"): "comfyui_image_request_timeout_seconds",
    ("comfyui", "connect_timeout_seconds"): "comfyui_image_connect_timeout_seconds",
    ("comfyui", "poll_interval_seconds"):    "comfyui_image_poll_interval_seconds",
    ("comfyui", "total_deadline_seconds"):   "comfyui_image_total_deadline_seconds",
    ("comfyui", "default_seed"):             "comfyui_image_default_seed",
    ("comfyui", "default_steps"):            "comfyui_image_default_steps",
    ("comfyui", "default_sampler"):          "comfyui_image_default_sampler",
}
_GLOBAL_KEYS = [
    "default_backend", "enabled_backends", "max_width", "max_height",
    "max_pixels", "max_steps", "max_prompt_length", "inline_max_bytes",
    "default_batch", "max_variants_per_message",
    "context_llm_enabled", "context_llm_turns", "context_llm_timeout_seconds",
]

# task-621: flat_field_name -> (backend, toml_key), derived by reversing
# _NON_SECRET and _SECRETS (whose secret TOML key is per-backend data, not
# always "api_key" -- see _SECRETS' own comment). Used only to build a
# helpful unknown-key warning below -- never to accept the flat spelling
# itself (decision: warn-on-unknown-key, not flat aliases, to avoid two
# spellings of the same setting needing a collision-precedence rule).
_FLAT_MAP: dict[str, tuple[str, str]] = {
    flat_field: (backend, toml_key) for (backend, toml_key), flat_field in _NON_SECRET.items()
}
_FLAT_MAP.update({
    flat_field: (backend, config_key)
    for backend, (flat_field, _env_vars, _kr_id, config_key) in _SECRETS.items()
})

# Known [image_generation.<backend>] subsection names.
_BACKEND_NAMES = set(_SECRETS) | {backend for backend, _toml_key in _NON_SECRET}



def _read_image_generation_toml() -> dict:
    """Return the raw [image_generation] section dict (nested). Patch point in tests."""
    from tldw_chatbook.config import load_settings
    return load_settings().get("image_generation", {}) or {}







# these delegates keep the module-level names as test patch points and
# builder call sites.
_TABLES = ModalityConfigTables(
    section_name="image_generation",
    keyring_namespace="tldw_chatbook_imagegen",
    keyring_label="imagegen",
    global_keys=tuple(_GLOBAL_KEYS),
    extra_exempt=frozenset({"styles"}),
    secrets={
        backend: SecretSpec(flat_field, tuple(env_vars), kr_id, config_key)
        for backend, (flat_field, env_vars, kr_id, config_key) in _SECRETS.items()
    },
    non_secret=dict(_NON_SECRET),
)

_coerce_int = coerce_int
_coerce_float = coerce_float
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


def _load_image_generation_section() -> tuple[dict, dict[str, str]]:
    return load_generation_section(
        _TABLES, read_toml=_read_image_generation_toml, keyring_lookup=_keyring_get
    )
@dataclass(frozen=True)
class ImageGenerationConfig:
    default_backend: str | None
    enabled_backends: list[str]
    max_width: int
    max_height: int
    max_pixels: int
    max_steps: int
    max_prompt_length: int
    inline_max_bytes: int | None
    default_batch: int
    max_variants_per_message: int
    context_llm_enabled: bool
    context_llm_turns: int
    context_llm_timeout_seconds: float
    sd_cpp_diffusion_model_path: str | None
    sd_cpp_llm_path: str | None
    sd_cpp_binary_path: str | None
    sd_cpp_model_path: str | None
    sd_cpp_vae_path: str | None
    sd_cpp_lora_paths: list[str]
    sd_cpp_allowed_extra_params: list[str]
    sd_cpp_default_steps: int
    sd_cpp_default_cfg_scale: float
    sd_cpp_default_sampler: str
    sd_cpp_device: str
    sd_cpp_timeout_seconds: int
    swarmui_base_url: str | None
    swarmui_default_model: str | None
    swarmui_swarm_token: str | None
    swarmui_allowed_extra_params: list[str]
    swarmui_timeout_seconds: int
    openrouter_image_base_url: str | None
    openrouter_image_api_key: str | None
    openrouter_image_default_model: str | None
    openrouter_image_allowed_extra_params: list[str]
    openrouter_image_timeout_seconds: int
    novita_image_base_url: str | None
    novita_image_api_key: str | None
    novita_image_default_model: str | None
    novita_image_allowed_extra_params: list[str]
    novita_image_timeout_seconds: int
    novita_image_poll_interval_seconds: int
    together_image_base_url: str | None
    together_image_api_key: str | None
    together_image_default_model: str | None
    together_image_allowed_extra_params: list[str]
    together_image_timeout_seconds: int
    modelstudio_image_base_url: str | None
    modelstudio_image_api_key: str | None
    modelstudio_image_default_model: str | None
    modelstudio_image_region: str
    modelstudio_image_mode: str
    modelstudio_image_poll_interval_seconds: int
    modelstudio_image_timeout_seconds: int
    modelstudio_image_allowed_extra_params: list[str]
    fal_image_base_url: str | None
    fal_image_api_key: str | None
    fal_image_default_model: str | None
    fal_image_poll_interval_seconds: int
    fal_image_timeout_seconds: int
    gemini_image_base_url: str | None
    gemini_image_api_key: str | None
    gemini_image_default_model: str | None
    gemini_image_timeout_seconds: int
    comfyui_image_base_url: str
    comfyui_image_request_timeout_seconds: float
    comfyui_image_connect_timeout_seconds: float
    comfyui_image_poll_interval_seconds: float
    comfyui_image_total_deadline_seconds: float
    comfyui_image_default_seed: int | None
    comfyui_image_default_steps: int | None
    comfyui_image_default_sampler: str | None
    reference_image_supported_models: dict[str, list[str]] = field(default_factory=dict)
    # backend id -> "env:<VAR>" | "config" | "keyring" | "missing" (task-1,
    # Settings ▸ Image Gen plan). Purely additive/read-only metadata about
    # where each backend's secret was resolved from -- never affects what's
    # written into the secret fields above.
    key_sources: dict[str, str] = field(default_factory=dict)


_config_cache: ImageGenerationConfig | None = None
_IMAGE_GENERATION_RUNTIME_LOCK = threading.RLock()
_IMAGE_GENERATION_CONFIG_SNAPSHOT = threading.local()
_NO_CONFIG_SNAPSHOT = object()




def _coerce_positive_float(value: Any, default: float) -> float:
    """Return a finite positive float, otherwise the documented default."""
    parsed = _coerce_float(value, default)
    return parsed if math.isfinite(parsed) and parsed > 0 else default


def _optional_int(
    section: dict[str, Any], key: str, *, minimum: int
) -> int | None:
    """Parse an optional integer without silently replacing invalid values."""
    raw = section.get(key)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if isinstance(raw, bool):
        raise ValueError(f"[image_generation.comfyui] {key} must be an integer")
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        raise ValueError(
            f"[image_generation.comfyui] {key} must be an integer"
        ) from None
    if value < minimum:
        raise ValueError(
            f"[image_generation.comfyui] {key} must be at least {minimum}"
        )
    return value


def _optional_string(section: dict[str, Any], key: str) -> str | None:
    """Parse an optional string while rejecting structured TOML values."""
    raw = section.get(key)
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError(f"[image_generation.comfyui] {key} must be a string")
    value = raw.strip()
    return value or None


def normalize_comfyui_image_origin(value: Any) -> str:
    """Return one normalized HTTP(S) origin for the image ComfyUI server.

    Userinfo, paths, queries, fragments, malformed ports, and parser-ambiguous
    whitespace/backslashes are refused because later adapter endpoints must stay
    on exactly this scheme/host/port boundary.
    """
    raw = str(value or "").strip()
    if not raw or any(character.isspace() for character in raw) or "\\" in raw:
        raise ValueError("ComfyUI image base URL must be a valid http(s) origin")
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError:
        raise ValueError(
            "ComfyUI image base URL must be a valid http(s) origin"
        ) from None
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname
    if (
        scheme not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or parsed.netloc.endswith(":")
    ):
        raise ValueError("ComfyUI image base URL must be a valid http(s) origin")
    normalized_host = hostname.lower()
    if ":" in normalized_host:
        normalized_host = f"[{normalized_host}]"
    normalized_port = f":{port}" if port is not None else ""
    return f"{scheme}://{normalized_host}{normalized_port}"




def _parse_mapping_of_lists(value: Any) -> dict[str, list[str]]:
    if value is None:
        return {}
    if isinstance(value, dict):
        parsed = value
    else:
        raw = str(value).strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        if not isinstance(parsed, dict):
            return {}

    result: dict[str, list[str]] = {}
    for raw_key, raw_values in parsed.items():
        key = str(raw_key).strip().lower()
        if not key:
            continue
        if isinstance(raw_values, list):
            items = [str(item).strip() for item in raw_values if str(item).strip()]
        elif raw_values is None:
            items = []
        else:
            raw = str(raw_values).strip()
            if not raw:
                items = []
            else:
                try:
                    candidate = json.loads(raw)
                except Exception:
                    candidate = None
                if isinstance(candidate, list):
                    items = [str(item).strip() for item in candidate if str(item).strip()]
                else:
                    items = [item.strip() for item in raw.split(",") if item.strip()]
        result[key] = items
    return result



def _get_image_generation_config_unlocked(
    *, reload: bool = False
) -> ImageGenerationConfig:
    global _config_cache
    if _config_cache is not None and not reload:
        return _config_cache

    from tldw_chatbook.Utils.Utils import coerce_bool_flag

    section, key_sources = _load_image_generation_section()

    default_backend = _get_config_value(section, "default_backend") or DEFAULT_BACKEND
    enabled_backends = _parse_list(section.get("enabled_backends"))
    if not enabled_backends:
        enabled_backends = []

    inline_max_bytes_raw = _get_config_value(section, "inline_max_bytes")
    inline_max_bytes = DEFAULT_INLINE_MAX_BYTES
    if inline_max_bytes_raw is not None:
        inline_max_bytes = max(1, _coerce_int(inline_max_bytes_raw, DEFAULT_INLINE_MAX_BYTES))

    default_batch = max(1, _coerce_int(section.get("default_batch"), DEFAULT_IMAGE_BATCH))
    max_variants_per_message = max(1, _coerce_int(section.get("max_variants_per_message"), DEFAULT_MAX_VARIANTS_PER_MESSAGE))

    context_llm_enabled = coerce_bool_flag(
        section.get("context_llm_enabled"), DEFAULT_CONTEXT_LLM_ENABLED
    )
    context_llm_turns = max(1, _coerce_int(section.get("context_llm_turns"), DEFAULT_CONTEXT_LLM_TURNS))
    context_llm_timeout_seconds = max(
        0.1, _coerce_float(section.get("context_llm_timeout_seconds"), DEFAULT_CONTEXT_LLM_TIMEOUT_SECONDS)
    )

    config = ImageGenerationConfig(
        default_backend=default_backend,
        enabled_backends=enabled_backends,
        max_width=_coerce_int(section.get("max_width"), DEFAULT_MAX_WIDTH),
        max_height=_coerce_int(section.get("max_height"), DEFAULT_MAX_HEIGHT),
        max_pixels=_coerce_int(section.get("max_pixels"), DEFAULT_MAX_PIXELS),
        max_steps=_coerce_int(section.get("max_steps"), DEFAULT_MAX_STEPS),
        max_prompt_length=_coerce_int(section.get("max_prompt_length"), DEFAULT_MAX_PROMPT_LENGTH),
        inline_max_bytes=inline_max_bytes,
        sd_cpp_diffusion_model_path=_get_config_value(section, "sd_cpp_diffusion_model_path"),
        sd_cpp_llm_path=_get_config_value(section, "sd_cpp_llm_path"),
        sd_cpp_binary_path=_get_config_value(section, "sd_cpp_binary_path"),
        sd_cpp_model_path=_get_config_value(section, "sd_cpp_model_path"),
        sd_cpp_vae_path=_get_config_value(section, "sd_cpp_vae_path"),
        sd_cpp_lora_paths=_parse_list(section.get("sd_cpp_lora_paths")),
        sd_cpp_allowed_extra_params=_parse_list(section.get("sd_cpp_allowed_extra_params")),
        sd_cpp_default_steps=_coerce_int(section.get("sd_cpp_default_steps"), DEFAULT_SD_CPP_STEPS),
        sd_cpp_default_cfg_scale=_coerce_float(section.get("sd_cpp_default_cfg_scale"), DEFAULT_SD_CPP_CFG_SCALE),
        sd_cpp_default_sampler=_get_config_value(section, "sd_cpp_default_sampler") or DEFAULT_SD_CPP_SAMPLER,
        sd_cpp_device=_get_config_value(section, "sd_cpp_device") or DEFAULT_SD_CPP_DEVICE,
        sd_cpp_timeout_seconds=_coerce_int(section.get("sd_cpp_timeout_seconds"), DEFAULT_SD_CPP_TIMEOUT_SECONDS),
        swarmui_base_url=_get_config_value(section, "swarmui_base_url") or DEFAULT_SWARMUI_BASE_URL,
        swarmui_default_model=_get_config_value(section, "swarmui_default_model"),
        swarmui_swarm_token=_get_config_value(section, "swarmui_swarm_token"),
        swarmui_allowed_extra_params=_parse_list(section.get("swarmui_allowed_extra_params")),
        swarmui_timeout_seconds=_coerce_int(section.get("swarmui_timeout_seconds"), DEFAULT_SWARMUI_TIMEOUT_SECONDS),
        openrouter_image_base_url=_get_config_value(section, "openrouter_image_base_url")
        or DEFAULT_OPENROUTER_IMAGE_BASE_URL,
        openrouter_image_api_key=_get_config_value(section, "openrouter_image_api_key"),
        openrouter_image_default_model=_get_config_value(section, "openrouter_image_default_model")
        or DEFAULT_OPENROUTER_IMAGE_MODEL,
        openrouter_image_allowed_extra_params=_parse_list(section.get("openrouter_image_allowed_extra_params")),
        openrouter_image_timeout_seconds=_coerce_int(
            section.get("openrouter_image_timeout_seconds"),
            DEFAULT_OPENROUTER_IMAGE_TIMEOUT_SECONDS,
        ),
        novita_image_base_url=_get_config_value(section, "novita_image_base_url")
        or DEFAULT_NOVITA_IMAGE_BASE_URL,
        novita_image_api_key=_get_config_value(section, "novita_image_api_key"),
        novita_image_default_model=_get_config_value(section, "novita_image_default_model")
        or DEFAULT_NOVITA_IMAGE_MODEL,
        novita_image_allowed_extra_params=_parse_list(section.get("novita_image_allowed_extra_params")),
        novita_image_timeout_seconds=_coerce_int(
            section.get("novita_image_timeout_seconds"),
            DEFAULT_NOVITA_IMAGE_TIMEOUT_SECONDS,
        ),
        novita_image_poll_interval_seconds=max(
            1,
            _coerce_int(
                section.get("novita_image_poll_interval_seconds"),
                DEFAULT_NOVITA_IMAGE_POLL_INTERVAL_SECONDS,
            ),
        ),
        together_image_base_url=_get_config_value(section, "together_image_base_url")
        or DEFAULT_TOGETHER_IMAGE_BASE_URL,
        together_image_api_key=_get_config_value(section, "together_image_api_key"),
        together_image_default_model=_get_config_value(section, "together_image_default_model")
        or DEFAULT_TOGETHER_IMAGE_MODEL,
        together_image_allowed_extra_params=_parse_list(section.get("together_image_allowed_extra_params")),
        together_image_timeout_seconds=_coerce_int(
            section.get("together_image_timeout_seconds"),
            DEFAULT_TOGETHER_IMAGE_TIMEOUT_SECONDS,
        ),
        modelstudio_image_base_url=_get_config_value(section, "modelstudio_image_base_url"),
        modelstudio_image_api_key=_get_config_value(section, "modelstudio_image_api_key"),
        modelstudio_image_default_model=_get_config_value(section, "modelstudio_image_default_model")
        or DEFAULT_MODELSTUDIO_IMAGE_MODEL,
        modelstudio_image_region=_coerce_choice(
            _get_config_value(section, "modelstudio_image_region"),
            default=DEFAULT_MODELSTUDIO_IMAGE_REGION,
            allowed={"sg", "cn", "us"},
        ),
        modelstudio_image_mode=_coerce_choice(
            _get_config_value(section, "modelstudio_image_mode"),
            default=DEFAULT_MODELSTUDIO_IMAGE_MODE,
            allowed={"sync", "async", "auto"},
        ),
        modelstudio_image_poll_interval_seconds=max(
            1,
            _coerce_int(
                section.get("modelstudio_image_poll_interval_seconds"),
                DEFAULT_MODELSTUDIO_IMAGE_POLL_INTERVAL_SECONDS,
            ),
        ),
        modelstudio_image_timeout_seconds=_coerce_int(
            section.get("modelstudio_image_timeout_seconds"),
            DEFAULT_MODELSTUDIO_IMAGE_TIMEOUT_SECONDS,
        ),
        modelstudio_image_allowed_extra_params=_parse_list(section.get("modelstudio_image_allowed_extra_params")),
        fal_image_base_url=_get_config_value(section, "fal_image_base_url") or DEFAULT_FAL_IMAGE_BASE_URL,
        fal_image_api_key=_get_config_value(section, "fal_image_api_key"),
        fal_image_default_model=_get_config_value(section, "fal_image_default_model") or DEFAULT_FAL_IMAGE_MODEL,
        fal_image_poll_interval_seconds=max(
            1,
            _coerce_int(
                section.get("fal_image_poll_interval_seconds"),
                DEFAULT_FAL_IMAGE_POLL_INTERVAL_SECONDS,
            ),
        ),
        fal_image_timeout_seconds=_coerce_int(
            section.get("fal_image_timeout_seconds"),
            DEFAULT_FAL_IMAGE_TIMEOUT_SECONDS,
        ),
        gemini_image_base_url=_get_config_value(section, "gemini_image_base_url") or DEFAULT_GEMINI_IMAGE_BASE_URL,
        gemini_image_api_key=_get_config_value(section, "gemini_image_api_key"),
        gemini_image_default_model=_get_config_value(section, "gemini_image_default_model")
        or DEFAULT_GEMINI_IMAGE_MODEL,
        gemini_image_timeout_seconds=_coerce_int(
            section.get("gemini_image_timeout_seconds"),
            DEFAULT_GEMINI_IMAGE_TIMEOUT_SECONDS,
        ),
        comfyui_image_base_url=normalize_comfyui_image_origin(
            _get_config_value(section, "comfyui_image_base_url")
            or DEFAULT_COMFYUI_IMAGE_BASE_URL
        ),
        comfyui_image_request_timeout_seconds=_coerce_positive_float(
            section.get("comfyui_image_request_timeout_seconds"),
            DEFAULT_COMFYUI_IMAGE_REQUEST_TIMEOUT_SECONDS,
        ),
        comfyui_image_connect_timeout_seconds=_coerce_positive_float(
            section.get("comfyui_image_connect_timeout_seconds"),
            DEFAULT_COMFYUI_IMAGE_CONNECT_TIMEOUT_SECONDS,
        ),
        comfyui_image_poll_interval_seconds=_coerce_positive_float(
            section.get("comfyui_image_poll_interval_seconds"),
            DEFAULT_COMFYUI_IMAGE_POLL_INTERVAL_SECONDS,
        ),
        comfyui_image_total_deadline_seconds=_coerce_positive_float(
            section.get("comfyui_image_total_deadline_seconds"),
            DEFAULT_COMFYUI_IMAGE_TOTAL_DEADLINE_SECONDS,
        ),
        comfyui_image_default_seed=_optional_int(
            section, "comfyui_image_default_seed", minimum=-1
        ),
        comfyui_image_default_steps=_optional_int(
            section, "comfyui_image_default_steps", minimum=1
        ),
        comfyui_image_default_sampler=_optional_string(
            section, "comfyui_image_default_sampler"
        ),
        reference_image_supported_models=_parse_mapping_of_lists(section.get("reference_image_supported_models")),
        key_sources=key_sources,
        default_batch=default_batch,
        max_variants_per_message=max_variants_per_message,
        context_llm_enabled=context_llm_enabled,
        context_llm_turns=context_llm_turns,
        context_llm_timeout_seconds=context_llm_timeout_seconds,
    )

    _config_cache = config
    return config


@contextmanager
def _use_image_generation_config_snapshot(
    config: ImageGenerationConfig,
) -> Iterator[None]:
    """Make one registry-owned config visible to constructors on this thread."""
    previous = getattr(
        _IMAGE_GENERATION_CONFIG_SNAPSHOT, "config", _NO_CONFIG_SNAPSHOT
    )
    _IMAGE_GENERATION_CONFIG_SNAPSHOT.config = config
    try:
        yield
    finally:
        if previous is _NO_CONFIG_SNAPSHOT:
            del _IMAGE_GENERATION_CONFIG_SNAPSHOT.config
        else:
            _IMAGE_GENERATION_CONFIG_SNAPSHOT.config = previous


def get_image_generation_config(*, reload: bool = False) -> ImageGenerationConfig:
    """Return one process-wide config snapshot, serialized with runtime reset."""
    captured = getattr(_IMAGE_GENERATION_CONFIG_SNAPSHOT, "config", None)
    if captured is not None and not reload:
        return captured
    with _IMAGE_GENERATION_RUNTIME_LOCK:
        return _get_image_generation_config_unlocked(reload=reload)


def reset_image_generation_config_cache() -> None:
    global _config_cache
    with _IMAGE_GENERATION_RUNTIME_LOCK:
        _config_cache = None


def reset_image_generation_runtime() -> None:
    """Clear the config cache, then the lazily imported adapter registry."""
    with _IMAGE_GENERATION_RUNTIME_LOCK:
        reset_image_generation_config_cache()
        from tldw_chatbook.Image_Generation.adapter_registry import reset_registry

        reset_registry()
