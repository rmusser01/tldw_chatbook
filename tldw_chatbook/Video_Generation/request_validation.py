"""Shared request validation helpers for video-generation entry points."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from tldw_chatbook.Video_Generation.config import (
    DEFAULT_MAX_DURATION_SECONDS,
    DEFAULT_MAX_FPS,
    DEFAULT_MAX_HEIGHT,
    DEFAULT_MAX_PIXELS,
    DEFAULT_MAX_PROMPT_LENGTH,
    DEFAULT_MAX_REFERENCE_ASSETS,
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_WIDTH,
    get_video_generation_config,
)

#: Choke-point caps for the reference-asset seam. Per-kind size caps follow
#: the MiniMax-H3 documented input limits; local backends inherit the same
#: bounds (their inputs are smaller in practice). Adapters can assume any
#: asset that reaches them has passed these checks.
REFERENCE_IMAGE_MAX_BYTES = 30 * 1024 * 1024
REFERENCE_VIDEO_MAX_BYTES = 50 * 1024 * 1024
REFERENCE_AUDIO_MAX_BYTES = 15 * 1024 * 1024

REFERENCE_IMAGE_KINDS = frozenset({"first_frame", "last_frame", "reference_image"})
REFERENCE_IMAGE_ALLOWED_MIMES = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/heic", "image/heif"}
)
REFERENCE_VIDEO_ALLOWED_MIMES = frozenset({"video/mp4"})
REFERENCE_AUDIO_ALLOWED_MIMES = frozenset({"audio/wav", "audio/x-wav", "audio/mpeg"})

#: Per-kind count caps (MiniMax-H3 documented limits); the total is
#: additionally bounded by ``config.max_reference_assets``.
REFERENCE_KIND_MAX_COUNTS = {
    "first_frame": 1,
    "last_frame": 1,
    "reference_image": 9,
    "reference_video": 3,
    "reference_audio": 3,
}

_RATIO_PATTERN = re.compile(r"^(?:\d{1,2}:\d{1,2}|adaptive)$")


@dataclass(frozen=True)
class VideoGenerationValidationIssue:
    code: str
    message: str
    path: str


def allowed_extra_params_for_backend(backend: str, config: Any) -> set[str]:
    """Return configured passthrough allowlist keys for a video backend."""

    backend_name = str(backend or "").strip().lower()
    attr_by_backend = {
        "minimax": "minimax_video_allowed_extra_params",
        "comfyui": "comfyui_allowed_extra_params",
        "stable_diffusion_cpp": "sd_cpp_allowed_extra_params",
    }
    attr = attr_by_backend.get(backend_name)
    if not attr:
        return set()
    return {str(item).strip() for item in getattr(config, attr, []) or [] if str(item).strip()}


def validate_video_generation_request(
    structured: dict[str, Any],
    *,
    config: Any | None = None,
) -> list[VideoGenerationValidationIssue]:
    """Validate shared video-generation bounds and backend passthrough controls."""

    if config is None:
        config = get_video_generation_config()

    issues: list[VideoGenerationValidationIssue] = []
    prompt = structured.get("prompt")
    max_prompt_length = _positive_int_attr(config, "max_prompt_length", DEFAULT_MAX_PROMPT_LENGTH)
    if isinstance(prompt, str) and len(prompt) > max_prompt_length:
        issues.append(_issue("prompt exceeds max length", "prompt"))

    max_duration = _positive_int_attr(config, "max_duration_seconds", DEFAULT_MAX_DURATION_SECONDS)
    _validate_int_bound(issues, structured.get("duration_seconds"), path="duration_seconds", max_value=max_duration)

    max_fps = _positive_int_attr(config, "max_fps", DEFAULT_MAX_FPS)
    _validate_int_bound(issues, structured.get("fps"), path="fps", max_value=max_fps)

    width = structured.get("width")
    height = structured.get("height")
    max_width = _positive_int_attr(config, "max_width", DEFAULT_MAX_WIDTH)
    max_height = _positive_int_attr(config, "max_height", DEFAULT_MAX_HEIGHT)
    max_pixels = _positive_int_attr(config, "max_pixels", DEFAULT_MAX_PIXELS)

    width_ok = _validate_int_bound(issues, width, path="width", max_value=max_width)
    height_ok = _validate_int_bound(issues, height, path="height", max_value=max_height)
    if width_ok and height_ok and isinstance(width, int) and isinstance(height, int) and width * height > max_pixels:
        issues.append(_issue("video dimensions exceed max pixels", "width,height"))

    ratio = structured.get("ratio")
    if ratio is not None:
        if not isinstance(ratio, str) or not _RATIO_PATTERN.match(ratio.strip().lower()):
            issues.append(_issue("ratio must look like '16:9' or 'adaptive'", "ratio"))

    max_steps = _positive_int_attr(config, "max_steps", DEFAULT_MAX_STEPS)
    _validate_int_bound(issues, structured.get("steps"), path="steps", max_value=max_steps)

    _validate_positive_finite_float(issues, structured.get("cfg_scale"), path="cfg_scale")
    _validate_extra_params(structured, config, issues)
    _validate_reference_assets(structured, config, issues)
    return issues


def _issue(message: str, path: str) -> VideoGenerationValidationIssue:
    return VideoGenerationValidationIssue(
        code="video_params_invalid",
        message=message,
        path=path,
    )


def _positive_int_attr(config: Any, attr: str, default: int) -> int:
    try:
        value = int(getattr(config, attr, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _validate_int_bound(
    issues: list[VideoGenerationValidationIssue],
    value: Any,
    *,
    path: str,
    max_value: int,
) -> bool:
    if value is None:
        return True
    if isinstance(value, bool) or not isinstance(value, int):
        issues.append(_issue(f"{path} must be an integer", path))
        return False
    if value <= 0 or value > max_value:
        issues.append(_issue(f"{path} out of range", path))
        return False
    return True


def _validate_positive_finite_float(
    issues: list[VideoGenerationValidationIssue],
    value: Any,
    *,
    path: str,
) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        issues.append(_issue(f"{path} must be a finite positive number", path))
        return
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        issues.append(_issue(f"{path} must be a finite positive number", path))
        return
    if not math.isfinite(candidate) or candidate <= 0:
        issues.append(_issue(f"{path} must be a finite positive number", path))


def _validate_extra_params(
    structured: dict[str, Any],
    config: Any,
    issues: list[VideoGenerationValidationIssue],
) -> None:
    extra_params = structured.get("extra_params") or {}
    if not extra_params:
        return
    if not isinstance(extra_params, dict):
        issues.append(_issue("extra_params must be an object", "extra_params"))
        return

    backend = str(structured.get("backend") or "").strip().lower()
    allowlist = allowed_extra_params_for_backend(backend, config)
    for key in extra_params:
        if key not in allowlist:
            issues.append(_issue("extra_params key not allowlisted", f"extra_params.{key}"))

    if "cli_args" in extra_params and "cli_args" in allowlist:
        cli_args = extra_params.get("cli_args")
        if not isinstance(cli_args, (list, tuple)):
            issues.append(_issue("cli_args must be a list", "extra_params.cli_args"))


def _validate_reference_assets(
    structured: dict[str, Any],
    config: Any,
    issues: list[VideoGenerationValidationIssue],
) -> None:
    """Choke-point validation for ``VideoGenRequest.reference_assets``.

    Runs only when assets are actually present, so every non-reference
    validation path is untouched. Checks fire independently (they don't
    short-circuit each other) so a caller sees every problem at once.
    """

    assets = structured.get("reference_assets") or ()
    if not assets:
        return

    max_assets = _positive_int_attr(config, "max_reference_assets", DEFAULT_MAX_REFERENCE_ASSETS)
    if len(assets) > max_assets:
        issues.append(_issue(f"reference assets exceed the {max_assets}-asset limit", "reference_assets"))

    kind_counts: dict[str, int] = {}
    for index, asset in enumerate(assets):
        path = f"reference_assets[{index}]"
        kind = getattr(asset, "kind", None)
        if kind not in REFERENCE_KIND_MAX_COUNTS:
            issues.append(_issue(f"unknown reference asset kind {kind!r}", f"{path}.kind"))
            continue
        kind_counts[kind] = kind_counts.get(kind, 0) + 1

        mime_type = getattr(asset, "mime_type", None)
        if kind in REFERENCE_IMAGE_KINDS:
            allowed_mimes = REFERENCE_IMAGE_ALLOWED_MIMES
            max_bytes = REFERENCE_IMAGE_MAX_BYTES
            label = "image"
        elif kind == "reference_video":
            allowed_mimes = REFERENCE_VIDEO_ALLOWED_MIMES
            max_bytes = REFERENCE_VIDEO_MAX_BYTES
            label = "video"
        else:  # reference_audio
            allowed_mimes = REFERENCE_AUDIO_ALLOWED_MIMES
            max_bytes = REFERENCE_AUDIO_MAX_BYTES
            label = "audio"
        if mime_type not in allowed_mimes:
            issues.append(
                _issue(f"reference {label} mime {mime_type!r} is not supported", f"{path}.mime_type")
            )

        # Validate the ACTUAL content bytes, never a caller-supplied length
        # field -- same hardening as the image package's task-686 fix.
        content = getattr(asset, "content", None)
        if content is None or content == b"":
            issues.append(_issue("reference asset has no content bytes", path))
        elif len(content) > max_bytes:
            issues.append(_issue(f"reference {label} exceeds the {max_bytes // (1024 * 1024)}MB limit", path))

    for kind, count in kind_counts.items():
        cap = REFERENCE_KIND_MAX_COUNTS[kind]
        if count > cap:
            issues.append(_issue(f"too many {kind} assets ({count} > {cap})", f"reference_assets.{kind}"))
