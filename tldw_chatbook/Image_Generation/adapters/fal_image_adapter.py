"""fal.ai queue image-generation backend adapter.

Task 6 of the fal/Gemini/Fireworks image-backends plan (see
``.superpowers/sdd/2026-07-26-imagegen-fal-gemini-fireworks/task-6-brief.md``).
Modeled on ``novita_image_adapter.py`` (the closest sibling with a
submit/poll queue shape) for the polling loop, and on
``gemini_image_adapter.py`` (``_validate_model_id`` precedent) for
model-path validation and 404 enrichment.

fal's queue API (verified live 2026-07-25, see task-6-report.md for
sources):

- Submit: ``POST {base}/{model_path}`` (e.g.
  ``https://queue.fal.run/fal-ai/flux/schnell``), ``Authorization: Key
  $FAL_KEY``, returns ``{"request_id": ..., "status_url": ..., ...}``.
- The ``image_size`` field accepts either a string enum OR a generic
  object form ``{"width": int, "height": int}`` -- confirmed against the
  live ``fal-ai/flux/schnell`` schema page (fal.ai/models/fal-ai/flux/schnell/api).
- The "app id" used to build the polling/result URLs is NOT the full
  submitted model path -- it is only the first two ``/``-segments
  (``owner/app``), with any further segments (e.g. a variant like
  ``schnell``) dropped. This is confirmed against fal's own
  ``fal_client`` Python SDK source (``AppId.from_endpoint_id`` in
  ``fal-ai/fal``'s ``projects/fal_client/src/fal_client/client.py``),
  which splits an endpoint id into ``owner``/``alias``/``path`` and uses
  only ``owner/alias`` to build the queue status/result base URL --
  exactly the two-segment rule this adapter's ``_app_id`` implements.
- fal's endpoint grammar also has two "app namespace" prefixes --
  ``workflows/`` and ``comfy/`` -- that shift the owner/app segments one
  position to the right. Re-verified 2026-07-25 against the same SDK file
  (https://github.com/fal-ai/fal/blob/main/projects/fal_client/src/fal_client/client.py):
  ``APP_NAMESPACES = ["workflows", "comfy"]`` (line 718);
  ``AppId.from_endpoint_id`` (lines 743-761) special-cases
  ``parts[0] in APP_NAMESPACES`` to parse ``owner=parts[1]``,
  ``alias=parts[2]``, ``path="/".join(parts[3:])``, ``namespace=parts[0]``;
  and the queue URL builders (e.g. lines 1439-1440) prepend
  ``f"{namespace}/"`` before ``owner/alias`` when a namespace is present.
  So for a namespaced path the kept prefix is THREE segments
  (``namespace/owner/alias``), not two -- ``_app_id`` below implements this
  exactly: ``workflows/fal-ai/some-flow`` (3 segments) is kept whole,
  ``workflows/owner/app/variant`` (4 segments) drops only the trailing
  ``variant``, and a plain (non-namespaced) path keeps the original
  two-segment rule unchanged.

Because the polling/result URLs are self-built from validated inputs
(never taken from the API response), a submit response's own
``status_url`` -- when present -- is used ONLY as a cross-check: a
mismatch names the drift as a loud, sanitized error instead of silently
trusting (and fetching) a vendor-controlled URL.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any
from urllib.parse import urlparse

import httpx

from tldw_chatbook.Image_Generation.http_client import fetch_json
from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest, ImageGenResult
from tldw_chatbook.Image_Generation.adapters.image_format_utils import (
    fetch_image_bytes,
    reference_image_data_url,
    validate_and_convert_image_output,
)
from tldw_chatbook.Image_Generation.config import (
    DEFAULT_FAL_IMAGE_BASE_URL,
    DEFAULT_FAL_IMAGE_MODEL,
    DEFAULT_FAL_IMAGE_POLL_INTERVAL_SECONDS,
    DEFAULT_FAL_IMAGE_TIMEOUT_SECONDS,
    get_image_generation_config,
)
from tldw_chatbook.Image_Generation.exceptions import ImageBackendUnavailableError, ImageGenerationError
from tldw_chatbook.Image_Generation.request_validation import effective_inline_max_bytes
from tldw_chatbook.Utils.egress import origin_set

# fal model paths (e.g. "fal-ai/flux/schnell") are path-segment-interpolated
# directly into the submit URL, and their first two segments are used to
# build the poll/result URLs -- this charset match (plus the segment checks
# in `_validate_model_path`) keeps a misconfigured/malicious path from ever
# reaching URL construction. `/` is allowed (unlike Gemini's flat model
# ids) since fal paths are inherently multi-segment.
_MODEL_PATH_CHARSET_RE = re.compile(r"^[A-Za-z0-9._\-/]+$")

# The submit response's `request_id` is interpolated into the self-built
# poll/result URLs -- charset-validated for the same reason as the model
# path above.
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9-]+$")

# fal's "app namespace" prefixes (see module docstring for the verified
# source): a path starting with one of these shifts owner/alias one segment
# to the right, so `_app_id` must keep three segments instead of two.
_APP_NAMESPACES = frozenset({"workflows", "comfy"})


def _validate_model_path(path: str) -> str:
    """Validate a fal model path's charset and segment shape, returning it stripped.

    Args:
        path: The candidate model path (from the request or config default),
            e.g. ``"fal-ai/flux/schnell"``.

    Returns:
        The stripped path, unchanged, when it passes validation.

    Raises:
        ImageBackendUnavailableError: If ``path`` is empty, contains any
            character outside ``[A-Za-z0-9._-/]``, has a leading/trailing
            ``/``, or has an empty (``a//b``) or ``..`` path segment --
            naming the offending path so a misconfigured ``default_model``
            is diagnosable.
    """
    candidate = (path or "").strip()
    if not candidate or not _MODEL_PATH_CHARSET_RE.match(candidate):
        raise ImageBackendUnavailableError(
            f"invalid fal model path {candidate!r} -- check [image_generation.fal] default_model"
        )
    if candidate.startswith("/") or candidate.endswith("/"):
        raise ImageBackendUnavailableError(
            f"invalid fal model path {candidate!r} -- check [image_generation.fal] default_model"
        )
    for segment in candidate.split("/"):
        if not segment or segment == "..":
            raise ImageBackendUnavailableError(
                f"invalid fal model path {candidate!r} -- check [image_generation.fal] default_model"
            )
    return candidate


def _app_id(model_path: str) -> str:
    """Derive fal's queue "app id" from an already-validated model path.

    For a plain path, only the first two ``/``-segments are used -- any
    further segments (a model variant, e.g. ``schnell`` in
    ``fal-ai/flux/schnell``) are dropped. When the path starts with one of
    fal's namespace prefixes (``workflows/``, ``comfy/`` -- see
    ``_APP_NAMESPACES`` and the module docstring for how this was verified
    against fal's own client SDK), the owner/alias segments are shifted one
    position to the right, so the first THREE segments are kept instead
    (``namespace/owner/alias``); any segments after that are dropped the
    same way.

    Args:
        model_path: An already ``_validate_model_path``-validated path.

    Returns:
        The first two segments joined by ``/`` (e.g. ``"fal-ai/flux"``), or
        the first three when ``model_path`` starts with an app-namespace
        prefix (e.g. ``"workflows/owner/app"``).

    Raises:
        ImageBackendUnavailableError: If ``model_path`` has fewer than two
            ``/``-segments (fewer than three when namespaced).
    """
    segments = model_path.split("/")
    if segments[0] in _APP_NAMESPACES:
        if len(segments) < 3:
            raise ImageBackendUnavailableError(
                f"invalid fal model path {model_path!r} -- a {segments[0]!r}-namespaced "
                "path needs at least three path segments (e.g. "
                f"'{segments[0]}/owner/app') -- check [image_generation.fal] default_model"
            )
        return "/".join(segments[:3])
    if len(segments) < 2:
        raise ImageBackendUnavailableError(
            f"invalid fal model path {model_path!r} -- expected at least two "
            "path segments (e.g. 'owner/app') -- check [image_generation.fal] default_model"
        )
    return "/".join(segments[:2])


class FalImageAdapter:
    """Image-generation adapter for fal.ai's queue-based generation API.

    Implements the ``ImageGenerationAdapter`` protocol (``base.py``):
    ``name``/``supported_formats``/``generate()``. Submits a job, polls its
    status until completion (or timeout), fetches the result payload, then
    downloads the resulting image from its (untrusted, vendor-returned) CDN
    URL. See the module docstring for the verified submit/poll/result URL
    shapes and the app-id derivation rule.
    """

    name = "fal"
    supported_formats = {"png", "jpg", "webp"}
    _PENDING_STATUSES = {"IN_QUEUE", "IN_PROGRESS"}
    _COMPLETED_STATUS = "COMPLETED"

    def __init__(self) -> None:
        self._config = get_image_generation_config()

    def generate(self, request: ImageGenRequest) -> ImageGenResult:
        """Submit a fal generation job, poll it to completion, and fetch the image.

        Args:
            request: The normalized generation request. ``request.model``
                overrides the configured default model path (validated via
                ``_validate_model_path``); ``request.negative_prompt``, when
                set, is appended to the prompt text; ``request.seed``/
                ``request.width``/``request.height``, when set, are passed
                through; ``request.reference_image``, when set, is encoded
                as a data URL and sent as ``image_url``; ``request.format``
                must be one of ``supported_formats``.

        Returns:
            The decoded, format-converted image result.

        Raises:
            ImageGenerationError: If ``request.format`` is unsupported, the
                submit/poll/result requests fail, the job doesn't reach
                ``COMPLETED`` before the configured timeout, the result
                payload has no image URL, or the vendor's ``status_url``
                cross-check (see ``_cross_check_status_url``) detects an
                unexpected queue URL shape.
            ImageBackendUnavailableError: If the API key, base URL, or the
                resolved model path is missing or fails validation.
        """
        output_format = request.format.lower()
        if output_format not in self.supported_formats:
            raise ImageGenerationError(f"unsupported output format: {output_format}")

        api_key = self._resolve_api_key()
        base_url = self._resolve_base_url()
        model_path = self._resolve_model_path(request)
        app_id = _app_id(model_path)

        submit_data = self._submit(base_url, model_path, api_key, request)
        request_id = self._extract_request_id(submit_data)
        status_url = f"{base_url}/{app_id}/requests/{request_id}/status"
        result_url = f"{base_url}/{app_id}/requests/{request_id}"
        self._cross_check_status_url(submit_data, status_url)

        self._poll_until_complete(status_url, api_key)
        result_data = self._fetch_result(result_url, api_key)
        image_url = self._extract_image_url(result_data)

        # image_url came from the vendor's result payload (untrusted, a CDN
        # link) -- no trusted_origins, no Authorization header, fully
        # subject to the egress policy, mirroring novita/openrouter's
        # API-returned-URL fetches.
        content, content_type = fetch_image_bytes(
            image_url,
            timeout=self._timeout(),
            max_bytes=self._max_output_bytes(),
        )
        content, content_type = validate_and_convert_image_output(
            content,
            content_type,
            output_format,
            max_bytes=self._max_output_bytes(),
        )
        return ImageGenResult(content=content, content_type=content_type, bytes_len=len(content))

    def _max_output_bytes(self) -> int:
        return effective_inline_max_bytes(self._config)

    def _timeout(self) -> int:
        return self._config.fal_image_timeout_seconds or DEFAULT_FAL_IMAGE_TIMEOUT_SECONDS

    def _resolve_api_key(self) -> str:
        api_key = (self._config.fal_image_api_key or "").strip()
        if not api_key:
            api_key = (os.getenv("FAL_KEY") or "").strip()
        if not api_key:
            raise ImageBackendUnavailableError("fal image api key is not configured")
        return api_key

    def _resolve_base_url(self) -> str:
        raw = self._config.fal_image_base_url or DEFAULT_FAL_IMAGE_BASE_URL
        cleaned = str(raw).strip()
        if not cleaned:
            raise ImageBackendUnavailableError("fal image base URL is not configured")
        if not cleaned.startswith("http://") and not cleaned.startswith("https://"):
            cleaned = f"https://{cleaned}"
        return cleaned.rstrip("/")

    def _resolve_model_path(self, request: ImageGenRequest) -> str:
        model = request.model or self._config.fal_image_default_model or DEFAULT_FAL_IMAGE_MODEL
        return _validate_model_path(model)

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Key {api_key}",
            "Content-Type": "application/json",
        }

    def _build_submit_payload(self, request: ImageGenRequest) -> dict[str, Any]:
        prompt = request.prompt.strip()
        if request.negative_prompt:
            prompt = f"{prompt}\n\nNegative prompt: {request.negative_prompt.strip()}"

        payload: dict[str, Any] = {"prompt": prompt}
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.width is not None and request.height is not None:
            payload["image_size"] = {"width": request.width, "height": request.height}
        if request.reference_image is not None:
            payload["image_url"] = reference_image_data_url(request.reference_image)
        return payload

    def _submit(
        self, base_url: str, model_path: str, api_key: str, request: ImageGenRequest
    ) -> Any:
        submit_url = f"{base_url}/{model_path}"
        payload = self._build_submit_payload(request)
        try:
            return fetch_json(
                method="POST",
                url=submit_url,
                headers=self._headers(api_key),
                json=payload,
                timeout=self._timeout(),
                # submit_url is built from the configured base_url plus a
                # charset/segment-validated model path, not API-returned
                # data, so its host is trusted.
                trusted_origins=origin_set(submit_url),
            )
        except httpx.HTTPStatusError as exc:
            # task-620: a bare httpx status error doesn't say *why* -- for
            # fal this is almost always an invalid/retired model path. Name
            # the path and point at the config key so the user isn't left
            # guessing.
            if exc.response is not None and exc.response.status_code == 404:
                raise ImageGenerationError(
                    f"model {model_path!r} was rejected by fal (404) -- check "
                    "[image_generation.fal] default_model"
                ) from exc
            # task-686 (live-UAT observation): a bare httpx 403 doesn't say
            # *why* -- name the CATEGORY (locked / out-of-balance account)
            # and where to fix it, never the raw response body (it may
            # carry account-identifying detail our sanitization rightly
            # drops).
            if exc.response is not None and exc.response.status_code == 403:
                raise ImageGenerationError(
                    "fal account locked or out of balance -- top up at "
                    "fal.ai/dashboard/billing (403)"
                ) from exc
            raise ImageGenerationError(f"fal submit request failed: {exc}") from exc
        except ImageGenerationError:
            raise
        except Exception as exc:
            raise ImageGenerationError(f"fal submit request failed: {exc}") from exc

    @staticmethod
    def _extract_request_id(data: Any) -> str:
        if not isinstance(data, dict):
            raise ImageGenerationError("fal submit response was not JSON")
        request_id = data.get("request_id")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ImageGenerationError("fal submit response did not include a request_id")
        request_id = request_id.strip()
        if not _REQUEST_ID_RE.match(request_id):
            raise ImageGenerationError("fal submit response returned an invalid request_id")
        return request_id

    @staticmethod
    def _cross_check_status_url(submit_data: Any, self_built_status_url: str) -> None:
        """Cross-check (never follow) a submit response's own ``status_url``.

        The adapter always polls its own self-built URL -- this only
        verifies the vendor's claimed location still matches the shape this
        adapter assumes, so an undetected API change (or a compromised
        response) can't silently go unnoticed. The vendor URL itself is
        NEVER requested.
        """
        if not isinstance(submit_data, dict):
            return
        vendor_status_url = submit_data.get("status_url")
        if not isinstance(vendor_status_url, str) or not vendor_status_url.strip():
            return
        vendor_status_url = vendor_status_url.strip()
        if vendor_status_url == self_built_status_url:
            return

        self_parsed = urlparse(self_built_status_url)
        vendor_parsed = urlparse(vendor_status_url)
        if (
            self_parsed.scheme == vendor_parsed.scheme
            and self_parsed.netloc == vendor_parsed.netloc
            and self_parsed.path == vendor_parsed.path
        ):
            return

        vendor_origin = (
            f"{vendor_parsed.scheme}://{vendor_parsed.netloc}"
            if vendor_parsed.scheme and vendor_parsed.netloc
            else "unknown"
        )
        # Never include the full vendor URL (only its origin) and never any
        # credentials -- this message may end up in logs/UI.
        raise ImageGenerationError(
            "fal queue URL shape changed -- expected "
            f"{self_built_status_url!r}, vendor sent a different location (origin: {vendor_origin})"
        )

    def _poll_until_complete(self, status_url: str, api_key: str) -> None:
        timeout_seconds = float(self._config.fal_image_timeout_seconds or DEFAULT_FAL_IMAGE_TIMEOUT_SECONDS)
        poll_interval = max(
            1.0, float(self._config.fal_image_poll_interval_seconds or DEFAULT_FAL_IMAGE_POLL_INTERVAL_SECONDS)
        )
        deadline = time.monotonic() + timeout_seconds
        # status_url is self-built from the configured base_url plus
        # validated app_id/request_id -- never from API-returned data -- so
        # its host is trusted.
        trusted = origin_set(status_url)

        while time.monotonic() < deadline:
            try:
                data = fetch_json(
                    method="GET",
                    url=status_url,
                    headers=self._headers(api_key),
                    timeout=timeout_seconds,
                    trusted_origins=trusted,
                )
            except Exception as exc:
                raise ImageGenerationError(f"fal status polling failed: {exc}") from exc

            status = self._extract_status(data)
            if status == self._COMPLETED_STATUS:
                return
            if status in self._PENDING_STATUSES:
                time.sleep(poll_interval)
                continue
            # Anything else (FAILED, an error status, or an unrecognized/
            # missing status) is a hard stop -- only the sanitized status
            # label is included, never the raw response payload.
            raise ImageGenerationError(f"fal task did not complete (status: {status or 'unknown'})")

        raise ImageGenerationError("timed out waiting for fal image task result")

    @staticmethod
    def _extract_status(data: Any) -> str | None:
        if not isinstance(data, dict):
            return None
        value = data.get("status")
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
        return None

    def _fetch_result(self, result_url: str, api_key: str) -> Any:
        try:
            return fetch_json(
                method="GET",
                url=result_url,
                headers=self._headers(api_key),
                timeout=self._timeout(),
                trusted_origins=origin_set(result_url),
            )
        except Exception as exc:
            raise ImageGenerationError(f"fal result request failed: {exc}") from exc

    @staticmethod
    def _extract_image_url(data: Any) -> str:
        if isinstance(data, dict):
            images = data.get("images")
            if isinstance(images, list) and images and isinstance(images[0], dict):
                url = images[0].get("url")
                if isinstance(url, str) and url.strip():
                    return url.strip()
        raise ImageGenerationError("fal result response did not include an image URL")
