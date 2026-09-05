"""Fail-closed launch identity admission for llama.cpp snapshot management."""

from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import shutil
import socket
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
    ServerLaunchClaim,
)

from .snapshot_models import (
    COMPATIBILITY_STATE_KEYS,
    CompatibilityEvidence,
    FileIdentity,
    LaunchDescriptor,
    ReadinessObservation,
    replace_descriptor,
)

_SPLIT_MODEL = re.compile(
    r"^(?P<prefix>.+)-(?P<part>\d+)-of-(?P<total>\d+)\.gguf$",
    re.IGNORECASE,
)
_TRUE_VALUES = frozenset({"1", "true", "on", "enabled"})
_FALSE_VALUES = frozenset({"0", "false", "off", "disabled"})
_UNRESOLVED_RUNTIME_VALUES: dict[str, frozenset[str]] = {
    "device": frozenset({"@auto", "auto"}),
    "flash-attn": frozenset({"auto"}),
    "gpu-layers": frozenset({"auto"}),
    "mmproj-device": frozenset({"@auto", "auto"}),
}
_CACHE_TYPES = frozenset(
    {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}
)


@dataclass(frozen=True)
class _ValueOption:
    canonical: str
    value_kind: str
    optional: bool = False


_VALUE_OPTIONS: dict[str, _ValueOption] = {}
_FLAG_OPTIONS: dict[str, tuple[str, str]] = {}


def _values(
    aliases: tuple[str, ...], canonical: str, value_kind: str, *, optional: bool = False
) -> None:
    for alias in aliases:
        _VALUE_OPTIONS[alias] = _ValueOption(canonical, value_kind, optional)


def _flags(
    positive: tuple[str, ...], negative: tuple[str, ...], canonical: str
) -> None:
    for alias in positive:
        _FLAG_OPTIONS[alias] = (canonical, "on")
    for alias in negative:
        _FLAG_OPTIONS[alias] = (canonical, "off")


_values(("-m", "--model"), "model", "path")
_values(("--host",), "host", "text")
_values(("--port",), "port", "port")
_values(("--api-key",), "api-key", "text")
_values(("--api-key-file",), "api-key-file", "path")
_values(("-c", "--ctx-size"), "ctx-size", "nonnegative-int")
_values(("-np", "--parallel"), "parallel", "parallel")
_values(("--keep",), "keep", "keep")
_values(("-ctk", "--cache-type-k"), "cache-type-k", "cache-type")
_values(("-ctv", "--cache-type-v"), "cache-type-v", "cache-type")
_values(("-b", "--batch-size"), "batch-size", "positive-int")
_values(("-ub", "--ubatch-size"), "ubatch-size", "positive-int")
_values(("-fa", "--flash-attn"), "flash-attn", "on-off-auto", optional=True)
_values(("--rope-scaling",), "rope-scaling", "rope-scaling")
_values(("--rope-scale",), "rope-scale", "positive-number")
_values(("--rope-freq-base",), "rope-freq-base", "positive-number")
_values(("--rope-freq-scale",), "rope-freq-scale", "positive-number")
_values(("--yarn-orig-ctx",), "yarn-orig-ctx", "nonnegative-int")
_values(("--yarn-ext-factor",), "yarn-ext-factor", "number")
_values(("--yarn-attn-factor",), "yarn-attn-factor", "number")
_values(("--yarn-beta-slow",), "yarn-beta-slow", "number")
_values(("--yarn-beta-fast",), "yarn-beta-fast", "number")
_values(("-dev", "--device"), "device", "text")
_values(("-ngl", "--gpu-layers", "--n-gpu-layers"), "gpu-layers", "gpu-layers")
_values(("-sm", "--split-mode"), "split-mode", "split-mode")
_values(("-ts", "--tensor-split"), "tensor-split", "number-list")
_values(("-mg", "--main-gpu"), "main-gpu", "nonnegative-int")
_values(("-fit", "--fit"), "fit", "on-off", optional=True)
_values(("-fitt", "--fit-target"), "fit-target", "positive-int-list")
_values(("-fitc", "--fit-ctx"), "fit-ctx", "positive-int")
_values(("-mm", "--mmproj"), "mmproj", "path")
_values(("-mmdev", "--mmproj-device"), "mmproj-device", "text")
_values(("--image-min-tokens",), "image-min-tokens", "nonnegative-int")
_values(("--image-max-tokens",), "image-max-tokens", "nonnegative-int")
_values(("--mtmd-batch-max-tokens",), "mtmd-batch-max-tokens", "positive-int")

_flags(("-cb", "--cont-batching"), ("-nocb", "--no-cont-batching"), "cont-batching")
_flags(("--context-shift",), ("--no-context-shift",), "context-shift")
_flags(("--swa-full",), (), "swa-full")
_flags(("-kvo", "--kv-offload"), ("-nkvo", "--no-kv-offload"), "kv-offload")
_flags(("--mmproj-auto",), ("--no-mmproj", "--no-mmproj-auto"), "mmproj-auto")
_flags(("--mmproj-offload",), ("--no-mmproj-offload",), "mmproj-offload")

_STATE_VALUE_KINDS = {
    option.canonical: option.value_kind
    for option in _VALUE_OPTIONS.values()
    if option.canonical in COMPATIBILITY_STATE_KEYS
}
_STATE_VALUE_KINDS.update(
    {
        "cont-batching": "on-off",
        "context-shift": "on-off",
        "kv-offload": "on-off",
        "mmproj-auto": "on-off",
        "mmproj-offload": "on-off",
        "swa-full": "on-off",
    }
)


_STATE_DEFAULTS: dict[str, str] = {
    "ctx-size": "@model",
    "parallel": "@auto",
    "cont-batching": "on",
    "context-shift": "off",
    "keep": "0",
    "cache-type-k": "f16",
    "cache-type-v": "f16",
    "swa-full": "off",
    "flash-attn": "auto",
    "kv-offload": "on",
    "batch-size": "2048",
    "ubatch-size": "512",
    "rope-scaling": "@model",
    "rope-scale": "@model",
    "rope-freq-base": "@model",
    "rope-freq-scale": "@model",
    "yarn-orig-ctx": "@model",
    "yarn-ext-factor": "@model",
    "yarn-attn-factor": "@model",
    "yarn-beta-slow": "@model",
    "yarn-beta-fast": "@model",
    "device": "auto",
    "gpu-layers": "auto",
    "split-mode": "layer",
    "tensor-split": "@auto",
    "main-gpu": "0",
    "fit": "on",
    "fit-target": "1024",
    "fit-ctx": "4096",
    "mmproj-auto": "on",
    "mmproj-offload": "on",
    "mmproj-device": "auto",
    "image-min-tokens": "@model",
    "image-max-tokens": "@model",
    "mtmd-batch-max-tokens": "1024",
}


_ENV_OPTIONS: dict[str, tuple[str, str]] = {
    "LLAMA_ARG_MODEL": ("model", "path"),
    "LLAMA_ARG_HOST": ("host", "text"),
    "LLAMA_ARG_PORT": ("port", "port"),
    "LLAMA_API_KEY": ("api-key", "text"),
    "LLAMA_ARG_API_KEY_FILE": ("api-key-file", "path"),
    "LLAMA_ARG_CTX_SIZE": ("ctx-size", "nonnegative-int"),
    "LLAMA_ARG_N_PARALLEL": ("parallel", "parallel"),
    "LLAMA_ARG_CONT_BATCHING": ("cont-batching", "bool"),
    "LLAMA_ARG_CONTEXT_SHIFT": ("context-shift", "bool"),
    "LLAMA_ARG_CACHE_TYPE_K": ("cache-type-k", "cache-type"),
    "LLAMA_ARG_CACHE_TYPE_V": ("cache-type-v", "cache-type"),
    "LLAMA_ARG_SWA_FULL": ("swa-full", "bool"),
    "LLAMA_ARG_FLASH_ATTN": ("flash-attn", "on-off-auto"),
    "LLAMA_ARG_KV_OFFLOAD": ("kv-offload", "bool"),
    "LLAMA_ARG_BATCH": ("batch-size", "positive-int"),
    "LLAMA_ARG_UBATCH": ("ubatch-size", "positive-int"),
    "LLAMA_ARG_ROPE_SCALING_TYPE": ("rope-scaling", "rope-scaling"),
    "LLAMA_ARG_ROPE_SCALE": ("rope-scale", "positive-number"),
    "LLAMA_ARG_ROPE_FREQ_BASE": ("rope-freq-base", "positive-number"),
    "LLAMA_ARG_ROPE_FREQ_SCALE": ("rope-freq-scale", "positive-number"),
    "LLAMA_ARG_YARN_ORIG_CTX": ("yarn-orig-ctx", "nonnegative-int"),
    "LLAMA_ARG_YARN_EXT_FACTOR": ("yarn-ext-factor", "number"),
    "LLAMA_ARG_YARN_ATTN_FACTOR": ("yarn-attn-factor", "number"),
    "LLAMA_ARG_YARN_BETA_SLOW": ("yarn-beta-slow", "number"),
    "LLAMA_ARG_YARN_BETA_FAST": ("yarn-beta-fast", "number"),
    "LLAMA_ARG_DEVICE": ("device", "text"),
    "LLAMA_ARG_N_GPU_LAYERS": ("gpu-layers", "gpu-layers"),
    "LLAMA_ARG_SPLIT_MODE": ("split-mode", "split-mode"),
    "LLAMA_ARG_TENSOR_SPLIT": ("tensor-split", "number-list"),
    "LLAMA_ARG_MAIN_GPU": ("main-gpu", "nonnegative-int"),
    "LLAMA_ARG_FIT": ("fit", "on-off"),
    "LLAMA_ARG_FIT_TARGET": ("fit-target", "positive-int-list"),
    "LLAMA_ARG_FIT_CTX": ("fit-ctx", "positive-int"),
    "LLAMA_ARG_MMPROJ": ("mmproj", "path"),
    "LLAMA_ARG_MMPROJ_AUTO": ("mmproj-auto", "bool"),
    "LLAMA_ARG_MMPROJ_OFFLOAD": ("mmproj-offload", "bool"),
    "MTMD_BACKEND_DEVICE": ("mmproj-device", "text"),
    "LLAMA_ARG_IMAGE_MIN_TOKENS": ("image-min-tokens", "nonnegative-int"),
    "LLAMA_ARG_IMAGE_MAX_TOKENS": ("image-max-tokens", "nonnegative-int"),
    "LLAMA_ARG_MTMD_BATCH_MAX_TOKENS": ("mtmd-batch-max-tokens", "positive-int"),
}


_IGNORED_VALUE_OPTIONS = frozenset(
    {
        "-t",
        "--threads",
        "-tb",
        "--threads-batch",
        "-s",
        "--seed",
        "--samplers",
        "--sampler-seq",
        "--sampling-seq",
        "--temp",
        "--temperature",
        "--top-k",
        "--top-p",
        "--min-p",
        "--top-nsigma",
        "--top-n-sigma",
        "--xtc-probability",
        "--xtc-threshold",
        "--typical",
        "--typical-p",
        "--repeat-last-n",
        "--repeat-penalty",
        "--presence-penalty",
        "--frequency-penalty",
        "--dry-multiplier",
        "--dry-base",
        "--dry-allowed-length",
        "--dry-penalty-last-n",
        "--dry-sequence-breaker",
        "--adaptive-target",
        "--adaptive-decay",
        "--dynatemp-range",
        "--dynatemp-exp",
        "--mirostat",
        "--mirostat-lr",
        "--mirostat-ent",
        "-l",
        "--logit-bias",
        "--grammar",
        "--grammar-file",
        "-j",
        "--json-schema",
        "-jf",
        "--json-schema-file",
        "-r",
        "--reverse-prompt",
        "--pooling",
        "-a",
        "--alias",
        "--tags",
        "--embd-normalize",
        "--path",
        "--cors-origins",
        "--cors-methods",
        "--cors-headers",
        "--chat-template-kwargs",
        "-to",
        "--timeout",
        "--sse-ping-interval",
        "--threads-http",
        "--cache-reuse",
        "-lv",
        "--verbosity",
        "--log-verbosity",
        "--log-file",
        "-n",
        "--predict",
        "--n-predict",
        "--reasoning-format",
        "-rea",
        "--reasoning",
        "--reasoning-effort",
        "--reasoning-budget",
        "--reasoning-budget-message",
    }
)
_IGNORED_FLAGS = frozenset(
    {
        "--metrics",
        "--props",
        "--cache-prompt",
        "--no-cache-prompt",
        "-v",
        "--verbose",
        "--log-verbose",
        "--log-disable",
        "--perf",
        "--no-perf",
        "--escape",
        "--no-escape",
        "--ignore-eos",
        "--warmup",
        "--no-warmup",
        "-sp",
        "--special",
        "--spm-infill",
        "--embedding",
        "--embeddings",
        "--rerank",
        "--reranking",
        "--reasoning-preserve",
        "--no-reasoning-preserve",
        "--reuse-port",
        "--cors-credentials",
        "--no-cors-credentials",
    }
)
_IGNORED_ENV = frozenset(
    {
        "LLAMA_ARG_THREADS",
        "LLAMA_ARG_N_PREDICT",
        "LLAMA_ARG_TOP_K",
        "LLAMA_ARG_ENDPOINT_METRICS",
        "LLAMA_ARG_ENDPOINT_PROPS",
        "LLAMA_ARG_CACHE_PROMPT",
        "LLAMA_ARG_CACHE_REUSE",
        "LLAMA_ARG_TIMEOUT",
        "LLAMA_ARG_THREADS_HTTP",
        "LLAMA_ARG_LOG_VERBOSITY",
        "LLAMA_ARG_PERF",
        "LLAMA_ARG_ALIAS",
        "LLAMA_ARG_TAGS",
        "LLAMA_ARG_REASONING",
        "LLAMA_ARG_REASONING_EFFORT",
        "LLAMA_ARG_THINK",
        "LLAMA_ARG_THINK_BUDGET",
        "LLAMA_ARG_REASONING_PRESERVE",
        "LLAMA_ARG_POOLING",
        "LLAMA_ARG_REUSE_PORT",
        "LLAMA_ARG_CORS_ORIGINS",
        "LLAMA_ARG_CORS_METHODS",
        "LLAMA_ARG_CORS_HEADERS",
        "LLAMA_ARG_CORS_CREDENTIALS",
        "LLAMA_ARG_OFFLINE",
        "LLAMA_ARG_LOG_FILE",
        "LLAMA_ARG_LOG_COLORS",
        "LLAMA_ARG_LOG_PREFIX",
        "LLAMA_ARG_LOG_TIMESTAMPS",
    }
)


_UNSUPPORTED_OPTIONS: dict[str, tuple[int, str]] = {
    "--slot-save-path": (1, "Snapshot management owns the slot save path."),
    "--slots": (0, "Snapshot management owns the slots endpoint setting."),
    "--no-slots": (0, "Snapshot management requires the slots endpoint."),
    "--lora": (1, "LoRA configuration is not supported for snapshots."),
    "--lora-scaled": (1, "LoRA configuration is not supported for snapshots."),
    "--lora-init-without-apply": (
        0,
        "LoRA configuration is not supported for snapshots.",
    ),
    "--control-vector": (
        1,
        "Control-vector configuration is not supported for snapshots.",
    ),
    "--control-vector-scaled": (
        1,
        "Control-vector configuration is not supported for snapshots.",
    ),
    "--control-vector-layer-range": (
        2,
        "Control-vector configuration is not supported for snapshots.",
    ),
    "--override-kv": (1, "Model metadata overrides are not supported for snapshots."),
    "--rpc": (1, "RPC configuration is not supported for snapshots."),
    "--models-dir": (1, "Router configuration is not supported for snapshots."),
    "--models-preset": (1, "Router configuration is not supported for snapshots."),
    "--models-max": (1, "Router configuration is not supported for snapshots."),
    "--models-autoload": (0, "Router configuration is not supported for snapshots."),
    "--no-models-autoload": (0, "Router configuration is not supported for snapshots."),
    "--ssl-key-file": (1, "TLS configuration is not supported for snapshots."),
    "--ssl-cert-file": (1, "TLS configuration is not supported for snapshots."),
    "-md": (1, "Speculative decoding is not supported for snapshots."),
    "--model-draft": (1, "Speculative decoding is not supported for snapshots."),
    "-mmu": (1, "Remote projector configuration is not supported for snapshots."),
    "--mmproj-url": (
        1,
        "Remote projector configuration is not supported for snapshots.",
    ),
    "--video-fps": (1, "Video projector overrides are not supported for snapshots."),
    "--video-timestamp-interval": (
        1,
        "Video projector overrides are not supported for snapshots.",
    ),
    "--video-ffmpeg-dir": (
        1,
        "Video projector overrides are not supported for snapshots.",
    ),
    "-mu": (1, "Remote model sources are not supported for snapshots."),
    "--model-url": (1, "Remote model sources are not supported for snapshots."),
    "-hf": (1, "Remote model sources are not supported for snapshots."),
    "-hfr": (1, "Remote model sources are not supported for snapshots."),
    "--hf-repo": (1, "Remote model sources are not supported for snapshots."),
}
_UNSUPPORTED_ENV: dict[str, str] = {
    "LLAMA_ARG_SLOT_SAVE_PATH": "Snapshot management owns the slot save path.",
    "LLAMA_ARG_ENDPOINT_SLOTS": "Snapshot management owns the slots endpoint setting.",
    "LLAMA_ARG_LORA": "LoRA configuration is not supported for snapshots.",
    "LLAMA_ARG_RPC": "RPC configuration is not supported for snapshots.",
    "LLAMA_ARG_MODELS_DIR": "Router configuration is not supported for snapshots.",
    "LLAMA_ARG_MODELS_PRESET": "Router configuration is not supported for snapshots.",
    "LLAMA_ARG_SSL_KEY_FILE": "TLS configuration is not supported for snapshots.",
    "LLAMA_ARG_SSL_CERT_FILE": "TLS configuration is not supported for snapshots.",
    "LLAMA_ARG_API_PREFIX": "A custom API prefix is not supported for snapshots.",
    "LLAMA_ARG_MMPROJ_URL": "Remote projector configuration is not supported for snapshots.",
    "LLAMA_ARG_VIDEO_FPS": "Video projector overrides are not supported for snapshots.",
    "LLAMA_ARG_VIDEO_TIMESTAMP_INTERVAL": "Video projector overrides are not supported for snapshots.",
    "LLAMA_ARG_VIDEO_FFMPEG_DIR": "Video projector overrides are not supported for snapshots.",
    "LLAMA_ARG_MODEL_URL": "Remote model sources are not supported for snapshots.",
    "LLAMA_ARG_HF_REPO": "Remote model sources are not supported for snapshots.",
}


@dataclass
class _ParsedLaunch:
    values: dict[str, str]
    specified: set[str]
    disabled_reason: str | None = None


def _parse_bool(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized in _TRUE_VALUES:
        return "on"
    if normalized in _FALSE_VALUES:
        return "off"
    raise ValueError("invalid boolean")


def _decimal(value: str, *, positive: bool = False) -> str:
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("invalid number") from exc
    if not number.is_finite() or (positive and number <= 0):
        raise ValueError("invalid number")
    rendered = format(number.normalize(), "f")
    return "0" if rendered in {"-0", "-0.0"} else rendered


def _integer(value: str, *, minimum: int | None = None) -> str:
    if re.fullmatch(r"[+-]?\d+", value.strip()) is None:
        raise ValueError("invalid integer")
    number = int(value)
    if minimum is not None and number < minimum:
        raise ValueError("integer out of range")
    return str(number)


def _canonicalize(kind: str, value: str) -> str:
    value = value.strip()
    if kind in {"text", "path"}:
        if not value:
            raise ValueError("empty value")
        return value
    if kind == "bool":
        return _parse_bool(value)
    if kind == "on-off":
        return _parse_bool(value)
    if kind == "on-off-auto":
        result = value.casefold()
        if result in {"auto", "-1"}:
            return "auto"
        return _parse_bool(result)
    if kind == "port":
        rendered = _integer(value, minimum=1)
        if int(rendered) > 65535:
            raise ValueError("port out of range")
        return rendered
    if kind == "parallel":
        rendered = _integer(value)
        if rendered == "-1":
            return "@auto"
        if int(rendered) <= 0:
            raise ValueError("invalid parallel count")
        return rendered
    if kind == "keep":
        return _integer(value, minimum=-1)
    if kind == "nonnegative-int":
        return _integer(value, minimum=0)
    if kind == "positive-int":
        return _integer(value, minimum=1)
    if kind == "gpu-layers":
        lowered = value.casefold()
        return lowered if lowered in {"auto", "all"} else _integer(value, minimum=0)
    if kind == "cache-type":
        lowered = value.casefold()
        if lowered not in _CACHE_TYPES:
            raise ValueError("invalid cache type")
        return lowered
    if kind == "rope-scaling":
        lowered = value.casefold()
        if lowered not in {"none", "linear", "yarn"}:
            raise ValueError("invalid rope scaling")
        return lowered
    if kind == "split-mode":
        lowered = value.casefold()
        if lowered not in {"none", "layer", "row", "tensor"}:
            raise ValueError("invalid split mode")
        return lowered
    if kind == "number":
        return _decimal(value)
    if kind == "positive-number":
        return _decimal(value, positive=True)
    if kind in {"number-list", "positive-int-list"}:
        parts = value.split(",")
        if not parts or any(not part.strip() for part in parts):
            raise ValueError("invalid list")
        parser: Callable[[str], str]
        if kind == "number-list":
            parser = lambda part: _decimal(part, positive=True)
        else:
            parser = lambda part: _integer(part, minimum=1)
        return ",".join(parser(part.strip()) for part in parts)
    raise ValueError("unknown value kind")


def _option_and_inline(token: str) -> tuple[str, str | None]:
    if token.startswith("-") and "=" in token:
        option, value = token.split("=", 1)
        return option, value
    return token, None


def has_owned_slot_options(command: tuple[str, ...], env: Mapping[str, str]) -> bool:
    """Find owned options, consuming values using the pinned admission metadata.

    Known unsupported options remain traversable. Unknown arity cannot establish
    a later conflict; ordinary admission will disable management in that case.
    """
    if any(
        name in env for name in ("LLAMA_ARG_SLOT_SAVE_PATH", "LLAMA_ARG_ENDPOINT_SLOTS")
    ):
        return True
    index = 1
    while index < len(command):
        option, inline = _option_and_inline(command[index])
        if option in {"--slots", "--no-slots", "--slot-save-path"}:
            return True
        value_option = _VALUE_OPTIONS.get(option)
        if value_option is not None:
            arity = 1
            if (
                value_option.optional
                and (index + 1 >= len(command) or command[index + 1].startswith("-"))
                and inline is None
            ):
                arity = 0
        elif option in _IGNORED_VALUE_OPTIONS or option == "--api-prefix":
            arity = 1
        elif option in _UNSUPPORTED_OPTIONS:
            arity = _UNSUPPORTED_OPTIONS[option][0]
        elif option in _FLAG_OPTIONS or option in _IGNORED_FLAGS:
            arity = 0
        else:
            return False
        index += 1 + max(0, arity - int(inline is not None))
    return False


def _parse_command(command: tuple[str, ...]) -> _ParsedLaunch:
    parsed = _ParsedLaunch(values={}, specified=set())
    index = 1
    while index < len(command):
        token = command[index]
        option, inline = _option_and_inline(token)
        if option in _UNSUPPORTED_OPTIONS or option.startswith("--spec-"):
            arity, reason = _UNSUPPORTED_OPTIONS.get(
                option,
                (
                    0 if inline is not None else 1,
                    "Speculative decoding is not supported for snapshots.",
                ),
            )
            if inline is None and index + arity >= len(command) + 1:
                parsed.disabled_reason = "A llama.cpp option is missing its value."
            else:
                parsed.disabled_reason = reason
            return parsed
        if option == "--api-prefix":
            if inline is None:
                if index + 1 >= len(command):
                    parsed.disabled_reason = "A llama.cpp option is missing its value."
                    return parsed
                inline = command[index + 1]
                index += 1
            if inline != "":
                parsed.disabled_reason = (
                    "A custom API prefix is not supported for snapshots."
                )
                return parsed
            index += 1
            continue
        flag = _FLAG_OPTIONS.get(option)
        if flag is not None:
            if inline is not None:
                parsed.disabled_reason = "A llama.cpp flag has an unexpected value."
                return parsed
            canonical, value = flag
            parsed.values[canonical] = value
            parsed.specified.add(canonical)
            index += 1
            continue
        value_option = _VALUE_OPTIONS.get(option)
        if value_option is not None:
            value = inline
            if value is None:
                if value_option.optional and (
                    index + 1 >= len(command) or command[index + 1].startswith("-")
                ):
                    value = "on"
                else:
                    if index + 1 >= len(command):
                        parsed.disabled_reason = (
                            "A llama.cpp option is missing its value."
                        )
                        return parsed
                    value = command[index + 1]
                    index += 1
            try:
                parsed.values[value_option.canonical] = _canonicalize(
                    value_option.value_kind, value
                )
            except ValueError:
                parsed.disabled_reason = (
                    f"Invalid value for llama.cpp option {value_option.canonical}."
                )
                return parsed
            parsed.specified.add(value_option.canonical)
            index += 1
            continue
        if option in _IGNORED_VALUE_OPTIONS:
            if inline is None:
                if index + 1 >= len(command):
                    parsed.disabled_reason = "A llama.cpp option is missing its value."
                    return parsed
                index += 1
            index += 1
            continue
        if option in _IGNORED_FLAGS:
            if inline is not None:
                parsed.disabled_reason = "A llama.cpp flag has an unexpected value."
                return parsed
            index += 1
            continue
        parsed.disabled_reason = f"Unknown llama.cpp option: {option}."
        return parsed
    return parsed


def _apply_environment(parsed: _ParsedLaunch, env: Mapping[str, str]) -> None:
    if parsed.disabled_reason is not None:
        return
    for name, reason in _UNSUPPORTED_ENV.items():
        if name in env and str(env[name]).strip():
            parsed.disabled_reason = reason
            return
    known_names = set(_ENV_OPTIONS) | _IGNORED_ENV | set(_UNSUPPORTED_ENV)
    for name in env:
        if name.startswith("LLAMA_ARG_") and name not in known_names:
            parsed.disabled_reason = f"Unknown llama.cpp environment setting: {name}."
            return
    for name, (canonical, kind) in _ENV_OPTIONS.items():
        if canonical in parsed.specified or name not in env:
            continue
        if canonical in {"api-key", "api-key-file"} and parsed.specified & {
            "api-key",
            "api-key-file",
        }:
            continue
        try:
            parsed.values[canonical] = _canonicalize(kind, str(env[name]))
        except ValueError:
            parsed.disabled_reason = f"Invalid llama.cpp environment setting: {name}."
            return


def _resolve_executable(value: str, env: Mapping[str, str]) -> Path:
    candidate = Path(value).expanduser()
    if candidate.parent == Path(".") and not candidate.is_absolute():
        located = shutil.which(value, path=env.get("PATH"))
        if located is None:
            raise ValueError("runtime")
        candidate = Path(located)
    return candidate.resolve(strict=True)


def _hash_regular_file(path: Path) -> FileIdentity:
    resolved = path.expanduser().resolve(strict=True)
    before = resolved.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("not regular")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        opened = os.fstat(handle.fileno())
    after = resolved.stat()
    snapshot = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if snapshot(before) != snapshot(opened) or snapshot(opened) != snapshot(after):
        raise ValueError("file changed")
    return FileIdentity(
        path=resolved,
        device=opened.st_dev,
        inode=opened.st_ino,
        size_bytes=opened.st_size,
        mtime_ns=opened.st_mtime_ns,
        ctime_ns=opened.st_ctime_ns,
        sha256=digest.hexdigest(),
    )


def _model_files(path: Path) -> tuple[tuple[FileIdentity, ...], str]:
    if path.suffix.casefold() != ".gguf":
        raise ValueError("model gguf")
    match = _SPLIT_MODEL.fullmatch(path.name)
    if match is None:
        identity = _hash_regular_file(path)
        return (identity,), identity.sha256
    part_width = len(match.group("part"))
    total_width = len(match.group("total"))
    total = int(match.group("total"))
    selected_part = int(match.group("part"))
    if total < 1 or selected_part < 1 or selected_part > total:
        raise ValueError("split shard")
    identities = []
    for part in range(1, total + 1):
        shard = path.with_name(
            f"{match.group('prefix')}-{part:0{part_width}d}-of-{total:0{total_width}d}.gguf"
        )
        try:
            identities.append(_hash_regular_file(shard))
        except (OSError, ValueError):
            raise ValueError("split shard") from None
    manifest = "".join(
        f"{part}:{identity.size_bytes}:{identity.sha256}\n"
        for part, identity in enumerate(identities, start=1)
    ).encode("ascii")
    return tuple(identities), hashlib.sha256(manifest).hexdigest()


def _numeric_loopback(host: str) -> str:
    candidate = host.strip()
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        if candidate.casefold() != "localhost":
            raise ValueError("loopback") from None
        try:
            answers = socket.getaddrinfo(
                candidate, 0, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
            )
        except OSError:
            raise ValueError("loopback") from None
        addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
        for answer in answers:
            try:
                resolved = ipaddress.ip_address(answer[4][0])
            except (ValueError, IndexError):
                raise ValueError("loopback") from None
            if not resolved.is_loopback:
                raise ValueError("loopback")
            if resolved not in addresses:
                addresses.append(resolved)
        if not addresses:
            raise ValueError("loopback")
        address = addresses[0]
    if not address.is_loopback:
        raise ValueError("loopback")
    return address.compressed


def _read_key_file(value: str) -> str:
    descriptor: int | None = None
    try:
        path = Path(value).expanduser().absolute()
        named = path.lstat()
        if stat.S_ISLNK(named.st_mode) or not stat.S_ISREG(named.st_mode):
            raise ValueError("key file")
        if named.st_size > 1024 * 1024:
            raise ValueError("key file")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        identity = lambda item: (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        if not stat.S_ISREG(opened.st_mode) or identity(named) != identity(opened):
            raise ValueError("key file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(1024 * 1024 + 1)
        if len(payload) > 1024 * 1024:
            raise ValueError("key file")
        text = payload.decode("utf-8")
    except (OSError, UnicodeError, ValueError):
        raise ValueError("key file") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if line:
            return line
    raise ValueError("key file")


def _first_api_key(value: str) -> str:
    keys = [part.strip() for part in value.split(",") if part.strip()]
    if not keys:
        raise ValueError("api key")
    return keys[0]


def _descriptor(
    *,
    launch_id: str,
    claim: ServerLaunchClaim,
    base_url: str,
    bearer_token: str | None,
    env: Mapping[str, str],
    files: tuple[FileIdentity, ...] = (),
    disabled_reason: str | None,
    state_settings: tuple[tuple[str, str], ...] = (),
    required_runtime_keys: frozenset[str] = frozenset(),
    model_paths: tuple[Path, ...] = (),
    projector_path: Path | None = None,
) -> LaunchDescriptor:
    return LaunchDescriptor(
        launch_id=launch_id,
        claim=claim,
        base_url=base_url,
        bearer_token=bearer_token,
        child_env=env,
        files=files,
        compatibility=None,
        disabled_reason=disabled_reason,
        _state_settings=state_settings,
        _required_runtime_keys=required_runtime_keys,
        _model_paths=model_paths,
        _projector_path=projector_path,
    )


def prepare_launch(
    command: tuple[str, ...],
    env: Mapping[str, str],
    claim: ServerLaunchClaim,
    launch_id: str,
) -> LaunchDescriptor:
    """Capture effective argv/environment and verified pre-readiness identities."""

    if not command or not launch_id or claim.provider != "llamacpp":
        raise ValueError("invalid llama.cpp snapshot launch")
    captured_env = {str(key): str(value) for key, value in env.items()}
    parsed = _parse_command(command)
    _apply_environment(parsed, captured_env)
    if parsed.disabled_reason is not None:
        return _descriptor(
            launch_id=launch_id,
            claim=claim,
            base_url="",
            bearer_token=None,
            env=captured_env,
            disabled_reason=parsed.disabled_reason,
        )

    try:
        host = _numeric_loopback(parsed.values.get("host", "127.0.0.1"))
    except ValueError:
        return _descriptor(
            launch_id=launch_id,
            claim=claim,
            base_url="",
            bearer_token=None,
            env=captured_env,
            disabled_reason="Snapshot management requires a loopback llama.cpp host.",
        )
    try:
        port = _canonicalize("port", parsed.values.get("port", "8080"))
    except ValueError:
        return _descriptor(
            launch_id=launch_id,
            claim=claim,
            base_url="",
            bearer_token=None,
            env=captured_env,
            disabled_reason="Snapshot management requires a valid TCP port.",
        )
    authority = f"[{host}]" if ipaddress.ip_address(host).version == 6 else host
    base_url = f"http://{authority}:{port}"

    bearer_token = parsed.values.get("api-key")
    if bearer_token is not None:
        try:
            bearer_token = _first_api_key(bearer_token)
        except ValueError:
            return _descriptor(
                launch_id=launch_id,
                claim=claim,
                base_url=base_url,
                bearer_token=None,
                env=captured_env,
                disabled_reason="The configured llama.cpp API key is invalid.",
            )
    key_file = parsed.values.get("api-key-file")
    if bearer_token is None and key_file is not None:
        try:
            bearer_token = _read_key_file(key_file)
        except ValueError:
            return _descriptor(
                launch_id=launch_id,
                claim=claim,
                base_url=base_url,
                bearer_token=None,
                env=captured_env,
                disabled_reason="The configured llama.cpp key file is invalid or empty.",
            )

    try:
        executable_path = _resolve_executable(command[0], captured_env)
        executable = _hash_regular_file(executable_path)
    except (OSError, ValueError):
        return _descriptor(
            launch_id=launch_id,
            claim=claim,
            base_url=base_url,
            bearer_token=bearer_token,
            env=captured_env,
            disabled_reason="The llama.cpp runtime identity could not be verified.",
        )
    model_value = parsed.values.get("model")
    if model_value is None:
        return _descriptor(
            launch_id=launch_id,
            claim=claim,
            base_url=base_url,
            bearer_token=bearer_token,
            env=captured_env,
            files=(executable,),
            disabled_reason="The loaded GGUF model identity could not be verified.",
        )
    try:
        model_identities, _model_digest = _model_files(Path(model_value))
    except (OSError, ValueError) as exc:
        reason = (
            "The split GGUF model is missing a required shard."
            if str(exc) == "split shard"
            else "The loaded GGUF model identity could not be verified."
        )
        return _descriptor(
            launch_id=launch_id,
            claim=claim,
            base_url=base_url,
            bearer_token=bearer_token,
            env=captured_env,
            files=(executable,),
            disabled_reason=reason,
        )

    state = dict(_STATE_DEFAULTS)
    for key, value in parsed.values.items():
        if key in COMPATIBILITY_STATE_KEYS:
            state[key] = value
    projector_identity: FileIdentity | None = None
    projector_value = parsed.values.get("mmproj")
    if state["mmproj-auto"] == "on" and projector_value is not None:
        try:
            projector_identity = _hash_regular_file(Path(projector_value))
        except (OSError, ValueError):
            return _descriptor(
                launch_id=launch_id,
                claim=claim,
                base_url=base_url,
                bearer_token=bearer_token,
                env=captured_env,
                files=(executable, *model_identities),
                disabled_reason="The multimodal projector identity could not be verified.",
            )

    required_runtime = {
        key for key in ("flash-attn", "device", "gpu-layers") if state[key] == "auto"
    }
    if state["fit"] == "on":
        required_runtime.update({"batch-size", "ctx-size", "gpu-layers", "ubatch-size"})
    if projector_identity is not None and state["mmproj-device"] == "auto":
        required_runtime.add("mmproj-device")
    state_settings = tuple(sorted(state.items()))
    files = (executable, *model_identities)
    if projector_identity is not None:
        files = (*files, projector_identity)
    return _descriptor(
        launch_id=launch_id,
        claim=claim,
        base_url=base_url,
        bearer_token=bearer_token,
        env=captured_env,
        files=files,
        disabled_reason=None,
        state_settings=state_settings,
        required_runtime_keys=frozenset(required_runtime),
        model_paths=tuple(identity.path for identity in model_identities),
        projector_path=projector_identity.path if projector_identity else None,
    )


def revalidate_files(descriptor: LaunchDescriptor) -> bool:
    """Return whether every admitted file retains its exact filesystem identity."""

    for identity in descriptor.files:
        try:
            current = identity.path.stat()
        except OSError:
            return False
        if not stat.S_ISREG(current.st_mode):
            return False
        if (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        ) != (
            identity.device,
            identity.inode,
            identity.size_bytes,
            identity.mtime_ns,
            identity.ctime_ns,
        ):
            return False
    return True


def finalize_launch(
    descriptor: LaunchDescriptor,
    observation: ReadinessObservation,
) -> LaunchDescriptor:
    """Combine verified launch files/args with one whitelisted readiness result."""

    if descriptor.disabled_reason is not None:
        return descriptor
    if not revalidate_files(descriptor):
        return replace_descriptor(
            descriptor,
            compatibility=None,
            disabled_reason="A launch file changed after snapshot admission.",
        )
    if not observation.build_info.strip():
        return replace_descriptor(
            descriptor,
            compatibility=None,
            disabled_reason="The llama.cpp build identity is unavailable.",
        )
    try:
        observed_model = Path(observation.model_path).expanduser().resolve(strict=False)
    except (OSError, ValueError):
        observed_model = Path()
    if observed_model not in descriptor._model_paths:
        return replace_descriptor(
            descriptor,
            compatibility=None,
            disabled_reason="The observed llama.cpp model does not match this launch.",
        )
    if not observation.slots or any(
        slot.context_size is None for slot in observation.slots
    ):
        return replace_descriptor(
            descriptor,
            compatibility=None,
            disabled_reason="The effective per-slot context size is unavailable.",
        )
    slot_ids = tuple(slot.slot_id for slot in observation.slots)
    if len(slot_ids) != len(set(slot_ids)):
        return replace_descriptor(
            descriptor,
            compatibility=None,
            disabled_reason="The observed llama.cpp slot identity is invalid.",
        )
    state = dict(descriptor._state_settings)
    if state["parallel"] == "@auto":
        state["parallel"] = str(len(observation.slots))
    elif int(state["parallel"]) != len(observation.slots):
        return replace_descriptor(
            descriptor,
            compatibility=None,
            disabled_reason="The observed llama.cpp slot count does not match this launch.",
        )
    runtime_values: dict[str, str] = {}
    for key, value in observation.runtime_values:
        kind = _STATE_VALUE_KINDS.get(key)
        if kind is None:
            return replace_descriptor(
                descriptor,
                compatibility=None,
                disabled_reason=f"The observed {key} setting is not valid launch evidence.",
            )
        try:
            normalized = _canonicalize(kind, value)
            unresolved_values = _UNRESOLVED_RUNTIME_VALUES.get(key)
            if unresolved_values and normalized.casefold() in unresolved_values:
                raise ValueError("unresolved automatic value")
            runtime_values[key] = normalized
        except ValueError:
            return replace_descriptor(
                descriptor,
                compatibility=None,
                disabled_reason=f"The observed {key} setting is invalid.",
            )
    missing = sorted(descriptor._required_runtime_keys - runtime_values.keys())
    if missing:
        return replace_descriptor(
            descriptor,
            compatibility=None,
            disabled_reason=f"The effective {missing[0]} setting is unavailable.",
        )
    for key, value in runtime_values.items():
        if key in descriptor._required_runtime_keys:
            state[key] = value
        elif key in state and state[key] != value:
            return replace_descriptor(
                descriptor,
                compatibility=None,
                disabled_reason=f"The observed {key} setting does not match this launch.",
            )
    state["effective-slot-contexts"] = ",".join(
        f"{slot.slot_id}:{slot.context_size}"
        for slot in sorted(observation.slots, key=lambda item: item.slot_id)
    )
    executable = descriptor.files[0]
    model_identities = tuple(
        identity
        for identity in descriptor.files
        if identity.path in descriptor._model_paths
    )
    if len(model_identities) == 1:
        model_sha256 = model_identities[0].sha256
    else:
        manifest = "".join(
            f"{part}:{identity.size_bytes}:{identity.sha256}\n"
            for part, identity in enumerate(model_identities, start=1)
        ).encode("ascii")
        model_sha256 = hashlib.sha256(manifest).hexdigest()
    projector_sha256 = None
    if descriptor._projector_path is not None:
        projector_sha256 = next(
            identity.sha256
            for identity in descriptor.files
            if identity.path == descriptor._projector_path
        )
    evidence = CompatibilityEvidence(
        model_sha256=model_sha256,
        projector_sha256=projector_sha256,
        runtime_sha256=executable.sha256,
        build_info=observation.build_info,
        state_settings=tuple(sorted(state.items())),
    )
    return replace_descriptor(
        descriptor,
        compatibility=evidence,
        disabled_reason=None,
    )


def compatibility_matches(
    saved: CompatibilityEvidence,
    current: CompatibilityEvidence,
) -> bool:
    """Return whether two complete compatibility identities match exactly."""

    return saved == current
