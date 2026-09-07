"""MiniMax-H3 video-generation backend adapter (task-3401.3, ADR-044).

Endpoints (verified against platform.minimax.io docs, 2026-08-07):

- Submit: ``POST {base}/v2/video_generation`` with a multimodal
  ``content[]`` array (this adapter submits TEXT-TO-VIDEO only: a single
  ``type=text`` item). Returns ``{"task_id": ...}``.
- Poll: ``GET {base}/v2/query/video_generation/{task_id}`` -- on success
  the task carries the download URL directly at ``task.content.url``.
  These are EXPIRING CDN URLs: download immediately, never persist.
  v1-style fallback: when only a ``file_id`` is present, resolve it via
  ``GET {base}/v1/files/retrieve?file_id=...`` -> ``file.download_url``.
- Cancel/delete: ``DELETE {base}/v2/video_generation/{task_id}`` -- a
  queued task is cancelled (no charge); a running task cannot be
  cancelled (the API errors, so cancel is best-effort); a terminal task's
  record is deleted.

Status vocabulary is parsed case-insensitively because the v1/v2 docs
disagree on casing (``Success``/``Fail`` vs ``succeeded``/``failed``):
pending = preparing/queueing/queued/processing/running; success =
succeeded/success; failure = failed/fail/cancelled/canceled.

Scope guard: reference assets (first/last frame, reference media) are
refused with a pointer to task-3401.8 -- uploading local media requires
the ``allow_uploads`` opt-in and the files/upload flow, which lands there.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any

import httpx

from tldw_chatbook.Image_Generation.adapters.image_format_utils import (
    fetch_image_bytes,
)
from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError
from tldw_chatbook.Image_Generation.http_client import fetch_json
from tldw_chatbook.Utils.egress import origin_set
from tldw_chatbook.Video_Generation.adapters.base import (
    VideoGenRequest,
    VideoGenResult,
)
from tldw_chatbook.Video_Generation.config import (
    DEFAULT_MINIMAX_VIDEO_BASE_URL,
    DEFAULT_MINIMAX_VIDEO_MODEL,
    DEFAULT_MINIMAX_VIDEO_POLL_INTERVAL_SECONDS,
    DEFAULT_MINIMAX_VIDEO_TIMEOUT_SECONDS,
    get_video_generation_config,
)
from tldw_chatbook.Video_Generation.exceptions import (
    VideoBackendUnavailableError,
    VideoGenerationError,
)
from tldw_chatbook.Video_Generation.video_formats import normalize_video_mime

# The submit response's task_id is interpolated into the poll/cancel URLs
# -- charset-validated so a malformed/compromised task id can never reach
# URL construction (same rule as fal's request_id).
_TASK_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# MiniMax-H3 documented bounds (platform.minimax.io/docs/guides/video-generation).
MINIMAX_MIN_DURATION_SECONDS = 4
MINIMAX_MAX_DURATION_SECONDS = 15
MINIMAX_DEFAULT_DURATION_SECONDS = 5

_PENDING_STATUSES = frozenset({"preparing", "queueing", "queued", "processing", "running"})
_SUCCESS_STATUSES = frozenset({"succeeded", "success"})
_FAILURE_STATUSES = frozenset({"failed", "fail", "cancelled", "canceled"})


class MiniMaxVideoAdapter:
    """Video-generation adapter for MiniMax's official H3 task API.

    Implements the ``VideoGenerationAdapter`` protocol. Submits a task,
    polls it to a terminal state (honoring an optional cancellation
    event), then downloads the video bytes from the expiring CDN URL
    immediately.
    """

    name = "minimax"
    supported_formats = {"mp4"}

    def __init__(self) -> None:
        self._config = get_video_generation_config()

    def generate(
        self,
        request: VideoGenRequest,
        *,
        cancel_event: threading.Event | None = None,
    ) -> VideoGenResult:
        """Submit a MiniMax-H3 text-to-video task and fetch the result.

        Args:
            request: The normalized generation request. ``request.model``
                overrides the configured default model; ``duration_seconds``
                is clamp-checked against MiniMax's 4-15s bounds; ``ratio``
                defaults to ``"16:9"`` (T2V requires an explicit ratio);
                width/height map onto the ``768P``/``2K`` resolution tiers.
            cancel_event: Optional cooperative-cancellation event (wired by
                the Console command, task-3401.5). When set, the poll loop
                stops, the remote task is cancelled best-effort (so a queued
                task does not bill to completion), and ``VideoGenerationError``
                is raised.

        Returns:
            The generated video bytes as an mp4 result.

        Raises:
            VideoGenerationError: On unsupported format, reference assets
                (not yet supported), duration outside MiniMax's bounds,
                task failure, timeout, cancellation, or a missing result
                URL. Never contains the API key.
            VideoBackendUnavailableError: If the API key or base URL is
                not configured.
        """
        output_format = request.format.lower()
        if output_format not in self.supported_formats:
            raise VideoGenerationError(f"unsupported output format: {output_format}")
        if request.reference_assets:
            raise VideoGenerationError(
                "MiniMax reference assets (first/last frame, reference media) are "
                "not supported yet -- image-to-video lands with task-3401.8."
            )

        api_key = self._resolve_api_key()
        base_url = self._resolve_base_url()
        model = request.model or self._config.minimax_video_default_model or DEFAULT_MINIMAX_VIDEO_MODEL

        task_id = self._submit(base_url, api_key, model, request)
        task = self._poll_until_terminal(base_url, api_key, task_id, cancel_event)
        download_url = self._extract_download_url(task)
        if download_url is None:
            file_id = self._extract_file_id(task)
            if file_id is None:
                raise VideoGenerationError(
                    "MiniMax task succeeded but returned no download URL or file_id"
                )
            download_url = self._retrieve_download_url(base_url, api_key, file_id)

        # The download URL comes from the vendor's response (a CDN link) --
        # untrusted: no trusted_origins, no Authorization header, fully
        # subject to the egress policy, mirroring the image adapters'
        # API-returned-URL fetches.
        try:
            content, content_type = fetch_image_bytes(
                download_url,
                timeout=self._timeout(),
                max_bytes=self._max_download_bytes(),
            )
        except ImageGenerationError as exc:
            raise VideoGenerationError(f"MiniMax video download failed: {exc}") from exc
        try:
            content_type = normalize_video_mime(content_type)
        except ValueError as exc:
            raise VideoGenerationError(
                "MiniMax video download did not return video/mp4 MIME"
            ) from exc
        if content_type != "video/mp4":
            raise VideoGenerationError(
                "MiniMax video download did not return video/mp4 MIME"
            )
        return VideoGenResult(
            content=content,
            content_type=content_type,
            container="mp4",
            bytes_len=len(content),
            duration_seconds=self._duration(request),
            width=self._task_int(task, "video_width"),
            height=self._task_int(task, "video_height"),
            resolved_model=model,
        )

    # -- configuration ----------------------------------------------------

    def _timeout(self) -> int:
        return self._config.minimax_video_timeout_seconds or DEFAULT_MINIMAX_VIDEO_TIMEOUT_SECONDS

    def _poll_interval(self) -> float:
        return max(
            1.0,
            float(
                self._config.minimax_video_poll_interval_seconds
                or DEFAULT_MINIMAX_VIDEO_POLL_INTERVAL_SECONDS
            ),
        )

    def _max_download_bytes(self) -> int:
        return max(1, int(self._config.download_max_mb)) * 1024 * 1024

    def _resolve_api_key(self) -> str:
        api_key = (self._config.minimax_video_api_key or "").strip()
        if not api_key:
            api_key = (os.getenv("MINIMAX_API_KEY") or "").strip()
        if not api_key:
            raise VideoBackendUnavailableError(
                "MiniMax video api key is not configured -- set MINIMAX_API_KEY, "
                "[video_generation.minimax] api_key, or the keyring entry"
            )
        return api_key

    def _resolve_base_url(self) -> str:
        raw = self._config.minimax_video_base_url or DEFAULT_MINIMAX_VIDEO_BASE_URL
        cleaned = str(raw).strip()
        if not cleaned:
            raise VideoBackendUnavailableError("MiniMax video base URL is not configured")
        if not cleaned.startswith("http://") and not cleaned.startswith("https://"):
            cleaned = f"https://{cleaned}"
        return cleaned.rstrip("/")

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    # -- request shaping ---------------------------------------------------

    @staticmethod
    def _duration(request: VideoGenRequest) -> int:
        duration = (
            request.duration_seconds
            if request.duration_seconds is not None
            else MINIMAX_DEFAULT_DURATION_SECONDS
        )
        if not MINIMAX_MIN_DURATION_SECONDS <= duration <= MINIMAX_MAX_DURATION_SECONDS:
            raise VideoGenerationError(
                f"MiniMax-H3 duration must be {MINIMAX_MIN_DURATION_SECONDS}-"
                f"{MINIMAX_MAX_DURATION_SECONDS}s (got {duration}s)"
            )
        return duration

    @staticmethod
    def _resolution(request: VideoGenRequest) -> str:
        width = request.width or 0
        height = request.height or 0
        return "2K" if max(width, height) > 1366 else "768P"

    def _build_submit_payload(self, request: VideoGenRequest, model: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "content": [{"type": "text", "text": request.prompt.strip()}],
            "duration": self._duration(request),
            "resolution": self._resolution(request),
            # T2V requires an explicit ratio (never "adaptive").
            "ratio": (request.ratio or "16:9"),
        }
        return payload

    # -- submit / poll / cancel -------------------------------------------

    def _submit(
        self, base_url: str, api_key: str, model: str, request: VideoGenRequest
    ) -> str:
        submit_url = f"{base_url}/v2/video_generation"
        try:
            data = fetch_json(
                method="POST",
                url=submit_url,
                headers=self._headers(api_key),
                json=self._build_submit_payload(request, model),
                timeout=self._timeout(),
                # Built from the configured base_url -- user-configured,
                # therefore trusted; never from API-returned data.
                trusted_origins=origin_set(submit_url),
            )
        except ImageGenerationError as exc:
            raise VideoGenerationError(f"MiniMax submit failed: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            raise VideoGenerationError(f"MiniMax submit failed (HTTP {status})") from exc
        except Exception as exc:
            raise VideoGenerationError(f"MiniMax submit failed: {exc}") from exc

        self._raise_on_base_resp_error(data, stage="submit")
        if not isinstance(data, dict):
            raise VideoGenerationError("MiniMax submit response was not JSON")
        task_id = data.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise VideoGenerationError("MiniMax submit response did not include a task_id")
        task_id = task_id.strip()
        if not _TASK_ID_RE.match(task_id):
            raise VideoGenerationError("MiniMax submit response returned an invalid task_id")
        return task_id

    def _poll_until_terminal(
        self,
        base_url: str,
        api_key: str,
        task_id: str,
        cancel_event: threading.Event | None,
    ) -> dict[str, Any]:
        """Poll until a terminal status; return the task payload on success."""
        query_url = f"{base_url}/v2/query/video_generation/{task_id}"
        # Self-built from the configured base_url + validated task_id --
        # trusted host.
        trusted = origin_set(query_url)
        deadline = time.monotonic() + float(self._timeout())

        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                self._cancel_task(base_url, api_key, task_id)
                raise VideoGenerationError("MiniMax video generation cancelled by user")
            try:
                data = fetch_json(
                    method="GET",
                    url=query_url,
                    headers=self._headers(api_key),
                    timeout=self._timeout(),
                    trusted_origins=trusted,
                )
            except (ImageGenerationError, httpx.HTTPStatusError) as exc:
                raise VideoGenerationError(f"MiniMax status polling failed: {exc}") from exc
            except Exception as exc:
                raise VideoGenerationError(f"MiniMax status polling failed: {exc}") from exc

            self._raise_on_base_resp_error(data, stage="query")
            task = self._extract_task(data)
            status = self._extract_status(task)
            if status in _SUCCESS_STATUSES:
                return task
            if status in _PENDING_STATUSES:
                if cancel_event is not None:
                    # Wakes the moment cancellation lands instead of after
                    # the full poll interval.
                    cancel_event.wait(self._poll_interval())
                else:
                    time.sleep(self._poll_interval())
                continue
            # failed/cancelled/unknown: hard stop. The sanitized status label
            # and the task's own error field only -- never the raw payload.
            detail = self._extract_task_error(task)
            raise VideoGenerationError(
                f"MiniMax task did not succeed (status: {status or 'unknown'}){detail}"
            )

        raise VideoGenerationError("timed out waiting for MiniMax video task result")

    def _cancel_task(self, base_url: str, api_key: str, task_id: str) -> None:
        """Best-effort remote cancel/delete (DELETE /v2/video_generation/{id}).

        A queued task is cancelled (no charge); a RUNNING task cannot be
        cancelled (the API errors) -- the local poll loop stops regardless,
        so a failure here is logged, never raised.
        """
        cancel_url = f"{base_url}/v2/video_generation/{task_id}"
        try:
            fetch_json(
                method="DELETE",
                url=cancel_url,
                headers=self._headers(api_key),
                timeout=30.0,
                trusted_origins=origin_set(cancel_url),
            )
        except Exception as exc:
            from loguru import logger

            logger.warning(
                "MiniMax remote task cancel failed; local stop still honored "
                "(error_type={})",
                type(exc).__name__,
            )

    # -- response parsing ---------------------------------------------------

    @staticmethod
    def _raise_on_base_resp_error(data: Any, *, stage: str) -> None:
        """Surface MiniMax's ``base_resp`` error envelope (code + message).

        The message is MiniMax's own status text (never request material and
        never the key), matching the image adapters' sanitization rule.
        """
        if not isinstance(data, dict):
            return
        base_resp = data.get("base_resp")
        if not isinstance(base_resp, dict):
            return
        status_code = base_resp.get("status_code")
        if status_code in (None, 0):
            return
        status_msg = str(base_resp.get("status_msg") or "").strip()
        raise VideoGenerationError(
            f"MiniMax {stage} rejected (code {status_code})"
            + (f": {status_msg}" if status_msg else "")
        )

    @staticmethod
    def _extract_task(data: Any) -> dict[str, Any]:
        """Return the task object from a query response (v2 wrapped or v1 flat)."""
        if not isinstance(data, dict):
            raise VideoGenerationError("MiniMax query response was not JSON")
        task = data.get("task")
        if isinstance(task, dict):
            return task
        return data

    @staticmethod
    def _extract_status(task: dict[str, Any]) -> str:
        value = task.get("status")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
        return ""

    @staticmethod
    def _extract_task_error(task: dict[str, Any]) -> str:
        error = task.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or error.get("status_msg") or "").strip()
            if message:
                return f" -- {message}"
        elif isinstance(error, str) and error.strip():
            return f" -- {error.strip()}"
        return ""

    @staticmethod
    def _extract_download_url(task: dict[str, Any]) -> str | None:
        content = task.get("content")
        if isinstance(content, dict):
            url = content.get("url")
            if isinstance(url, str) and url.strip():
                return url.strip()
        return None

    @staticmethod
    def _extract_file_id(task: dict[str, Any]) -> str | None:
        file_id = task.get("file_id")
        if isinstance(file_id, (str, int)) and str(file_id).strip():
            return str(file_id).strip()
        return None

    @staticmethod
    def _task_int(task: dict[str, Any], key: str) -> int | None:
        value = task.get(key)
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                return None
        return None

    def _retrieve_download_url(self, base_url: str, api_key: str, file_id: str) -> str:
        """v1-style fallback: resolve a file_id to its download URL."""
        retrieve_url = f"{base_url}/v1/files/retrieve"
        try:
            data = fetch_json(
                method="GET",
                url=retrieve_url,
                headers=self._headers(api_key),
                params={"file_id": file_id},
                timeout=self._timeout(),
                trusted_origins=origin_set(retrieve_url),
            )
        except Exception as exc:
            raise VideoGenerationError(f"MiniMax file retrieve failed: {exc}") from exc
        self._raise_on_base_resp_error(data, stage="file retrieve")
        if isinstance(data, dict):
            file_obj = data.get("file")
            if isinstance(file_obj, dict):
                url = file_obj.get("download_url")
                if isinstance(url, str) and url.strip():
                    return url.strip()
        raise VideoGenerationError("MiniMax file retrieve returned no download_url")
