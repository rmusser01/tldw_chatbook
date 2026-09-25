"""UI-free avatar resolution for character cards (TASK-32954, spec §3.4).

``resolve_avatar`` decides what a character's avatar should become — read
from a file, generated from the card's own description, or removed — with no
Textual/UI dependency. The Console character-tool service (TASK-32954 Task
3) calls this directly; the Personas screen keeps its own richer UI flow
(style templates, staged thumbnails) untouched.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

AVATAR_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})
AVATAR_IMAGE_SUFFIX_COPY = "PNG, JPG, JPEG, WEBP, or GIF"
AVATAR_MAX_BYTES = 5 * 1024 * 1024
AVATAR_MAX_SIZE_COPY = "5 MB"


@dataclass(frozen=True)
class AvatarOutcome:
    kind: Literal["image", "remove", "failed"]
    image: bytes | None = None
    reason: str = ""


def image_backend_configured() -> str | None:
    """Return the configured image-generation backend name, or ``None``."""
    from tldw_chatbook.Image_Generation.config import get_image_generation_config

    return get_image_generation_config().default_backend or None


def generate_avatar_bytes(prompt: str) -> bytes:
    """Generate one avatar image via the configured backend. Blocking.

    Looked up through this module global at call time (not imported at
    call sites) so tests can monkeypatch ``character_avatar.
    generate_avatar_bytes`` directly.
    """
    from tldw_chatbook.Image_Generation.worker import build_request, run_generation

    backend = image_backend_configured()
    if backend is None:
        raise RuntimeError("no image backend configured")
    result = run_generation(build_request(backend=backend, prompt=prompt))
    if not result.content:
        raise RuntimeError("backend returned no image")
    return bytes(result.content)


def _read_file(path_text: str) -> AvatarOutcome:
    from tldw_chatbook.Character_Chat.persona_visual_identity import (
        _portrait_content_type,
    )
    from tldw_chatbook.Utils.path_validation import validate_path_simple

    try:
        path = validate_path_simple(path_text, require_exists=True)
    except Exception:  # noqa: BLE001 - any validation failure is just a rejected path
        return AvatarOutcome("failed", reason="path rejected")
    if not path.is_file():
        return AvatarOutcome("failed", reason="not a file")
    if path.suffix.lower() not in AVATAR_IMAGE_SUFFIXES:
        return AvatarOutcome("failed", reason=f"use {AVATAR_IMAGE_SUFFIX_COPY}")
    if path.stat().st_size > AVATAR_MAX_BYTES:
        return AvatarOutcome("failed", reason=f"must be {AVATAR_MAX_SIZE_COPY} or smaller")
    data = path.read_bytes()
    if not data or _portrait_content_type(data) is None:
        return AvatarOutcome("failed", reason="not an image")
    return AvatarOutcome("image", image=data)


def _card_prompt(card: Mapping[str, Any]) -> str:
    # No style template: Personas' `_expression_generate_style` lives on the
    # mounted Screen instance (UI state), not reachable from a UI-free
    # helper. Plain (unstyled) composition only.
    from tldw_chatbook.Character_Chat.expression_generation import (
        compose_expression_prompt,
    )

    prompt, _negative, _params = compose_expression_prompt(
        name=str(card.get("name") or ""),
        description=str(card.get("description") or ""),
        personality=str(card.get("personality") or ""),
        state="avatar",
    )
    return prompt


def resolve_avatar(
    request: Mapping[str, Any],
    card: Mapping[str, Any],
    *,
    generate: Callable[[str], bytes] | None = None,
) -> AvatarOutcome:
    """Resolve one avatar request against a character card. Never raises.

    Args:
        request: ``{"source": "file"|"generate"|"remove", ...}``. ``"file"``
            reads ``request["path"]``; ``"generate"`` uses ``request["prompt"]``
            if non-blank, else a prompt composed from ``card``.
        card: The character card fields (``name``, ``description``,
            ``personality``) used to compose a generation prompt when none
            is given.
        generate: Override for the real backend call (tests inject a fake);
            defaults to :func:`generate_avatar_bytes` at call time.

    Returns:
        An :class:`AvatarOutcome` describing the image, removal, or failure.
    """
    source = request.get("source")
    if source == "remove":
        return AvatarOutcome("remove")
    if source == "file":
        return _read_file(str(request.get("path") or ""))
    if source == "generate":
        prompt = str(request.get("prompt") or "").strip() or _card_prompt(card)
        try:
            return AvatarOutcome("image", image=(generate or generate_avatar_bytes)(prompt))
        except Exception as exc:  # noqa: BLE001 - reported, never raised (spec §4.3)
            return AvatarOutcome("failed", reason=f"generation failed ({type(exc).__name__})")
    return AvatarOutcome("failed", reason="unknown avatar source")
