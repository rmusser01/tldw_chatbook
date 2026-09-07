"""Gemini (AI Studio) image-generation backend adapter.

Task 4 of the fal/Gemini/Fireworks image-backends plan (see
``.superpowers/sdd/2026-07-26-imagegen-fal-gemini-fireworks/task-4-brief.md``).
Modeled on ``openrouter_image_adapter.py`` (closest sibling): same config
access pattern, error types, and task-620 404-enrichment shape.

``generationConfig.responseModalities`` is pinned to ``["TEXT", "IMAGE"]`` --
verified live against Google's docs (see task-4-report.md for sources):
``gemini-2.5-flash-image`` is a conversational multimodal model that returns
an empty ``parts`` array (HTTP 200, no error) if only ``["IMAGE"]`` is
requested; both modalities must be requested even though this adapter
discards any text part in the response.
"""

from __future__ import annotations

import base64
import os
import re
from typing import Any

import httpx
from loguru import logger

from tldw_chatbook.Image_Generation.http_client import fetch_json
from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest, ImageGenResult
from tldw_chatbook.Image_Generation.adapters.image_format_utils import (
    decode_base64_image,
    validate_and_convert_image_output,
)
from tldw_chatbook.Image_Generation.capabilities import ResolvedReferenceImage
from tldw_chatbook.Image_Generation.config import (
    DEFAULT_GEMINI_IMAGE_BASE_URL,
    DEFAULT_GEMINI_IMAGE_MODEL,
    DEFAULT_GEMINI_IMAGE_TIMEOUT_SECONDS,
    get_image_generation_config,
)
from tldw_chatbook.Image_Generation.exceptions import ImageBackendUnavailableError, ImageGenerationError
from tldw_chatbook.Image_Generation.request_validation import effective_inline_max_bytes
from tldw_chatbook.Utils.egress import origin_set

# Gemini model ids are path-segment-interpolated into the request URL
# (`.../models/{model}:generateContent`); this charset match keeps a
# misconfigured/malicious id (e.g. containing `/`, `..`, `?`, whitespace)
# from ever reaching URL construction.
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# See module docstring: verified requirement for gemini-2.5-flash-image.
_RESPONSE_MODALITIES = ["TEXT", "IMAGE"]


def _validate_model_id(model: str) -> str:
    """Validate a Gemini model id's charset, returning it stripped.

    Args:
        model: The candidate model id (from the request or config default).

    Returns:
        The stripped model id, unchanged, when it passes validation.

    Raises:
        ImageBackendUnavailableError: If ``model`` is empty or contains any
            character outside ``[A-Za-z0-9._-]`` -- naming the offending id
            so a misconfigured ``default_model`` is diagnosable.
    """
    candidate = (model or "").strip()
    if not candidate or not _MODEL_ID_RE.match(candidate):
        raise ImageBackendUnavailableError(
            f"invalid gemini model id {candidate!r} -- check [image_generation.gemini] default_model"
        )
    return candidate


class GeminiImageAdapter:
    """Image-generation adapter for Google's Gemini (AI Studio) API.

    Implements the ``ImageGenerationAdapter`` protocol (``base.py``):
    ``name``/``supported_formats``/``generate()``. Submits a single-shot
    ``generateContent`` request (no polling) and extracts the first decodable
    inline-data image part from the response. See the module docstring for
    the verified request/response shape and the ``responseModalities``
    requirement.
    """

    name = "gemini"
    supported_formats = {"png", "jpg", "webp"}

    def __init__(self) -> None:
        self._config = get_image_generation_config()

    def generate(self, request: ImageGenRequest) -> ImageGenResult:
        """Generate one image via Gemini's ``models/{model}:generateContent`` endpoint.

        Args:
            request: The normalized generation request. ``request.model``
                overrides the configured default model (validated via
                ``_validate_model_id``); ``request.negative_prompt``, when
                set, is appended to the prompt text; ``request.reference_image``,
                when set, is threaded in as an ``inline_data`` part before the
                text part (see ``_reference_image_part``); ``request.format``
                must be one of ``supported_formats``.

        Returns:
            The decoded, format-converted image result.

        Raises:
            ImageGenerationError: If ``request.format`` is unsupported,
                the request to Gemini fails, Gemini returns no usable image
                (blocked prompt, a non-``STOP`` finish reason, an empty
                response, or undecodable inline data), or the reference
                image violates the engine's bytes-in-memory contract.
            ImageBackendUnavailableError: If the API key, base URL, or the
                resolved model id is missing or fails validation.
        """
        output_format = request.format.lower()
        if output_format not in self.supported_formats:
            raise ImageGenerationError(f"unsupported output format: {output_format}")

        api_key = self._resolve_api_key()
        base_url = self._resolve_base_url()
        model = self._resolve_model(request)
        url = self._generate_content_url(base_url, model)
        payload = self._build_payload(request)

        try:
            data = fetch_json(
                method="POST",
                url=url,
                headers=self._headers(api_key),
                json=payload,
                timeout=self._config.gemini_image_timeout_seconds or DEFAULT_GEMINI_IMAGE_TIMEOUT_SECONDS,
                # url is built from the configured base_url plus a
                # charset-validated model id, not API-returned data, so its
                # host is trusted.
                trusted_origins=origin_set(url),
            )
        except httpx.HTTPStatusError as exc:
            # task-620: a bare httpx status error ("404 Not Found for url
            # '...:generateContent'") doesn't say *why* -- for Gemini this is
            # almost always an invalid/retired model id. Name the model and
            # point at the config key so the user isn't left guessing.
            if exc.response is not None and exc.response.status_code in (400, 404):
                status = exc.response.status_code
                raise ImageGenerationError(
                    f"model {model!r} was rejected by Gemini ({status}) — check "
                    "[image_generation.gemini] default_model"
                ) from exc
            # task-686 (live-UAT observation): a bare httpx 429 doesn't say
            # *why* -- name the CATEGORY (rate limited / quota exhausted)
            # and the model id, never the raw response body (it may carry
            # account-identifying detail our sanitization rightly drops).
            if exc.response is not None and exc.response.status_code == 429:
                raise ImageGenerationError(
                    f"Gemini rate limited or image quota exhausted (free-tier caps apply) "
                    f"for model {model!r} (429)"
                ) from exc
            raise ImageGenerationError(f"Gemini request failed: {exc}") from exc
        except ImageGenerationError:
            raise
        except Exception as exc:
            raise ImageGenerationError(f"Gemini request failed: {exc}") from exc

        content, content_type = self._extract_image(data)
        content, content_type = validate_and_convert_image_output(
            content,
            content_type,
            output_format,
            max_bytes=self._max_output_bytes(),
        )
        return ImageGenResult(content=content, content_type=content_type, bytes_len=len(content))

    def _max_output_bytes(self) -> int:
        return effective_inline_max_bytes(self._config)

    def _resolve_api_key(self) -> str:
        api_key = (self._config.gemini_image_api_key or "").strip()
        if not api_key:
            api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        if not api_key:
            raise ImageBackendUnavailableError("gemini image api key is not configured")
        return api_key

    def _resolve_base_url(self) -> str:
        raw = self._config.gemini_image_base_url or DEFAULT_GEMINI_IMAGE_BASE_URL
        cleaned = str(raw).strip()
        if not cleaned:
            raise ImageBackendUnavailableError("gemini image base URL is not configured")
        if not cleaned.startswith("http://") and not cleaned.startswith("https://"):
            cleaned = f"https://{cleaned}"
        return cleaned.rstrip("/")

    def _resolve_model(self, request: ImageGenRequest) -> str:
        model = request.model or self._config.gemini_image_default_model or DEFAULT_GEMINI_IMAGE_MODEL
        return _validate_model_id(model)

    @staticmethod
    def _generate_content_url(base_url: str, model: str) -> str:
        return f"{base_url}/models/{model}:generateContent"

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        # The key is carried ONLY in this header -- Gemini also accepts a
        # `?key=` query param, but URLs are far more likely to be logged
        # (access logs, error messages, redirect chains) than headers, so
        # the query-param form is deliberately never used here.
        return {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }

    def _build_payload(self, request: ImageGenRequest) -> dict[str, Any]:
        prompt = request.prompt.strip()
        if request.negative_prompt:
            prompt = f"{prompt}\n\nNegative prompt: {request.negative_prompt.strip()}"

        parts: list[dict[str, Any]] = []
        if request.reference_image is not None:
            parts.append(self._reference_image_part(request.reference_image))
        parts.append({"text": prompt})

        return {
            "contents": [{"parts": parts}],
            "generationConfig": {"responseModalities": list(_RESPONSE_MODALITIES)},
        }

    @staticmethod
    def _reference_image_part(reference_image: ResolvedReferenceImage) -> dict[str, Any]:
        """Build the Gemini ``inline_data`` part for a validated reference image.

        Args:
            reference_image: The reference image, already checked by the
                engine's choke point (task-3) for backend capability,
                allowed mime type, size cap, and non-empty content before
                this adapter ever runs.

        Returns:
            The ``{"inline_data": {"mime_type": ..., "data": ...}}`` part,
            placed before the text part in the request body.

        Raises:
            ImageGenerationError: If ``reference_image.content`` is ``None``
                or empty. The engine's contract is bytes-in-memory ONLY --
                ``file_id``/``temp_path`` variants are never accepted by the
                choke-point validator, so this should be unreachable in
                practice; it exists as a defensive contract check, not a
                recoverable code path. Unlike ``file_id``/``temp_path``
                inputs, this adapter never reads from disk.
        """
        content = reference_image.content
        if content is None:
            raise ImageGenerationError(
                "reference image reached the adapter without content bytes "
                "(choke-point contract violation)"
            )
        if not content:
            raise ImageGenerationError("invalid reference image data")

        mime_type = (reference_image.mime_type or "application/octet-stream").split(";", 1)[0].strip().lower()
        return {
            "inline_data": {
                "mime_type": mime_type,
                "data": base64.b64encode(content).decode("ascii"),
            }
        }

    def _extract_image(self, data: Any) -> tuple[bytes, str]:
        # Fix-round-1 (reviewer MINOR finding): a decode failure in one part
        # must not abort the scan -- candidate[1] can still carry a valid
        # image even if candidate[0]'s inline data is corrupt. `saw_undecodable`
        # distinguishes "no inline-data part existed at all" (falls through to
        # the normal blockReason/finishReason mapping) from "at least one did,
        # but every one of them failed to decode" (a clearer, dedicated error).
        saw_undecodable = False
        if isinstance(data, dict):
            candidates = data.get("candidates")
            if isinstance(candidates, list):
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    content = candidate.get("content")
                    if not isinstance(content, dict):
                        continue
                    parts = content.get("parts")
                    if not isinstance(parts, list):
                        continue
                    for part in parts:
                        extracted, undecodable = self._extract_inline_data(part)
                        if extracted is not None:
                            return extracted
                        saw_undecodable = saw_undecodable or undecodable
        if saw_undecodable:
            raise ImageGenerationError("Gemini returned image data that could not be decoded")
        raise ImageGenerationError(self._no_image_message(data))

    def _extract_inline_data(self, part: Any) -> tuple[tuple[bytes, str] | None, bool]:
        """Try to pull a decoded image out of one response ``part``.

        Returns ``(result, undecodable)``: ``result`` is the decoded
        ``(bytes, mime_type)`` on success, else ``None``; ``undecodable`` is
        ``True`` only when the part *had* an inline-data payload that failed
        to decode (malformed base64 / oversize) -- the caller uses this to
        pick a more specific error than the generic no-image mapping when
        every part with inline data failed this way.
        """
        if not isinstance(part, dict):
            return None, False
        # Defensive against both spellings: the canonical REST/JSON response
        # field is camelCase `inlineData`, but proto-JSON also round-trips
        # the original `inline_data` field name.
        inline = part.get("inlineData")
        if not isinstance(inline, dict):
            inline = part.get("inline_data")
        if not isinstance(inline, dict):
            return None, False
        b64data = inline.get("data")
        if not isinstance(b64data, str) or not b64data.strip():
            return None, False
        mime_type = inline.get("mimeType") or inline.get("mime_type") or "image/png"
        try:
            content = decode_base64_image(b64data.strip(), max_bytes=self._max_output_bytes())
        except ImageGenerationError as exc:
            # Never log the b64 payload itself -- only the decoder's generic
            # failure reason (e.g. "invalid base64 image data").
            logger.debug(f"Gemini image adapter: skipping undecodable inline-data part ({exc})")
            return None, True
        return (content, str(mime_type)), False

    @staticmethod
    def _no_image_message(data: Any) -> str:
        """Map a no-image Gemini response to a fixed, sanitized message.

        Never includes response text or the request prompt (only
        `blockReason`/`finishReason` enum-ish values, which come from
        Gemini's own moderation/control fields, not free text).
        """
        if isinstance(data, dict):
            prompt_feedback = data.get("promptFeedback")
            if isinstance(prompt_feedback, dict):
                block_reason = prompt_feedback.get("blockReason")
                if block_reason:
                    return f"Gemini blocked the prompt ({block_reason})"
            candidates = data.get("candidates")
            if isinstance(candidates, list):
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    finish_reason = candidate.get("finishReason")
                    if finish_reason and finish_reason != "STOP":
                        return f"Gemini returned no image ({finish_reason})"
        return "Gemini returned no image"
