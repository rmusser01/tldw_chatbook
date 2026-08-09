"""ComfyUI HTTP video-generation adapter (task-3401.6, ADR-044).

ComfyUI executes API-format workflow graphs.  This adapter deliberately keeps
model filenames inside those graphs: it resolves a user-selected JSON workflow,
updates only nodes bearing the documented title convention, then queues and
polls the graph through ComfyUI's standard HTTP API.
"""

from __future__ import annotations

import copy
import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from tldw_chatbook.Image_Generation.adapters.image_format_utils import fetch_image_bytes
from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError
from tldw_chatbook.Image_Generation.http_client import (
    _validate_egress_or_raise,
    create_client,
    fetch_json,
)
from tldw_chatbook.Utils.egress import origin_set
from tldw_chatbook.Utils.paths import get_user_data_dir
from tldw_chatbook.Video_Generation.adapters.base import (
    ResolvedReferenceAsset,
    VideoGenRequest,
    VideoGenResult,
)
from tldw_chatbook.Video_Generation.config import (
    DEFAULT_COMFYUI_BASE_URL,
    DEFAULT_COMFYUI_TIMEOUT_SECONDS,
    get_video_generation_config,
)
from tldw_chatbook.Video_Generation.exceptions import (
    VideoBackendUnavailableError,
    VideoGenerationError,
)


_PROMPT_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_SUPPORTED_REFERENCE_KINDS = frozenset({"first_frame", "reference_image"})
_VIDEO_SUFFIX_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
_TITLE_CONTROLS = {
    "prompt": frozenset({"prompt"}),
    "negativeprompt": frozenset({"negative_prompt"}),
    "seed": frozenset({"seed"}),
    "width": frozenset({"width"}),
    "height": frozenset({"height"}),
    "frames": frozenset({"frames"}),
    "fps": frozenset({"fps"}),
    "inputimage": frozenset({"input_image"}),
    "widthheightframes": frozenset({"width", "height", "frames"}),
    "widthheightframesfps": frozenset({"width", "height", "frames", "fps"}),
}


class ComfyUIVideoAdapter:
    """Generate videos by queueing parameterized ComfyUI API workflows.

    The configured ComfyUI base URL is a user-selected local/backend origin.
    Every request built from it therefore passes the host as ``trusted_origins``
    to the shared egress-aware HTTP code.  Workflow output URLs are never
    persisted; downloaded bytes are returned directly to the video store layer.
    """

    name = "comfyui"
    supported_formats = {"mp4", "webm", "mov", "avi", "gif", "webp"}

    def __init__(self) -> None:
        self._config = get_video_generation_config()
        self._available_node_classes: set[str] | None = None

    def generate(
        self,
        request: VideoGenRequest,
        *,
        cancel_event: threading.Event | None = None,
    ) -> VideoGenResult:
        """Queue a ComfyUI workflow, poll history, and return its media bytes.

        Args:
            request: A normalized request. The configured workflow supplies
                its model paths; only documented node titles are mutated.
            cancel_event: Optional cooperative-cancellation event. On cancel,
                ComfyUI is interrupted best-effort and local generation stops.

        Returns:
            The generated video or animated-image content without a saved URL.

        Raises:
            VideoBackendUnavailableError: When the base URL/workflow is not
                usable or the server is missing workflow node classes.
            VideoGenerationError: For malformed workflows, queue/poll/upload/
                download failures, cancellation, timeout, or unsupported output.
        """
        output_format = (request.format or "").lower()
        if output_format not in self.supported_formats:
            raise VideoGenerationError(f"unsupported output format: {output_format}")

        base_url = self._base_url()
        workflow_name = (self._config.comfyui_default_workflow or "wan22_t2v.json").strip()
        workflow = self._load_workflow(workflow_name)
        self._validate_reference_assets(request.reference_assets)
        self._validate_required_nodes(base_url, workflow)

        image_name = self._resolve_uploaded_image(request.reference_assets)
        parameterized = self._parameterize_workflow(workflow, request, image_name)
        prompt_id = self._queue_prompt(base_url, parameterized)
        descriptor = self._poll_for_output(base_url, prompt_id, cancel_event)
        return self._download_output(base_url, descriptor)

    # -- configuration / workflows --------------------------------------

    def _base_url(self) -> str:
        """Return a normalized configured ComfyUI HTTP(S) base URL."""
        raw = str(self._config.comfyui_base_url or DEFAULT_COMFYUI_BASE_URL).strip()
        if not raw:
            raise VideoBackendUnavailableError("ComfyUI base URL is not configured")
        if not raw.startswith(("http://", "https://")):
            raw = f"http://{raw}"
        base_url = raw.rstrip("/")
        if not origin_set(base_url):
            raise VideoBackendUnavailableError("ComfyUI base URL must include a host")
        return base_url

    def _timeout(self) -> int:
        """Return the configured ComfyUI request/poll deadline in seconds."""
        return max(
            1,
            int(self._config.comfyui_timeout_seconds or DEFAULT_COMFYUI_TIMEOUT_SECONDS),
        )

    def _trusted_origins(self, base_url: str) -> frozenset:
        """Return the configured backend host as an egress trusted origin."""
        return origin_set(base_url)

    @staticmethod
    def _shipped_workflow_dir() -> Path:
        """Return the package directory containing read-only workflow assets."""
        return Path(__file__).resolve().parent.parent / "workflows"

    def _load_workflow(self, workflow_name: str) -> dict[str, Any]:
        """Load a safe workflow filename, preferring the user workflow folder.

        Args:
            workflow_name: Bare ``.json`` filename selected by configuration.

        Returns:
            Parsed ComfyUI API graph keyed by node id.

        Raises:
            VideoGenerationError: If the name traverses directories, is not
                JSON, cannot be loaded, or is not an API graph object.
        """
        raw_name = str(workflow_name or "").strip()
        candidate = Path(raw_name)
        if not raw_name.endswith(".json"):
            raise VideoGenerationError("ComfyUI workflow must be a JSON filename")
        if candidate.name != raw_name or raw_name in {".", ".."}:
            raise VideoGenerationError("ComfyUI workflow path is not allowed")

        data_root = get_user_data_dir().resolve()
        workflow_root = data_root / "video_workflows"
        user_candidate = workflow_root / candidate.name
        if user_candidate.is_symlink():
            raise VideoGenerationError("ComfyUI workflow symlink is not allowed")
        try:
            user_candidate.resolve().relative_to(workflow_root)
        except (OSError, ValueError) as exc:
            raise VideoGenerationError("ComfyUI workflow path escapes video_workflows") from exc
        paths = (user_candidate, self._shipped_workflow_dir() / candidate.name)
        selected = next((path for path in paths if path.is_file()), None)
        if selected is None:
            raise VideoGenerationError(f"ComfyUI workflow not found: {candidate.name}")
        try:
            parsed = json.loads(selected.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VideoGenerationError(f"failed to load ComfyUI workflow {candidate.name}: {exc}") from exc
        if not isinstance(parsed, dict) or not all(isinstance(node, dict) for node in parsed.values()):
            raise VideoGenerationError("ComfyUI workflow must be an API JSON object keyed by node id")
        return parsed

    # -- workflow shaping -------------------------------------------------

    def _validate_required_nodes(self, base_url: str, workflow: dict[str, Any]) -> None:
        """Ensure all graph ``class_type`` values are installed before queueing."""
        required = {
            str(node.get("class_type", "")).strip()
            for node in workflow.values()
            if isinstance(node, dict) and str(node.get("class_type", "")).strip()
        }
        if not required:
            raise VideoGenerationError("ComfyUI workflow contains no node class_type values")
        if self._available_node_classes is None:
            try:
                info = fetch_json(
                    method="GET",
                    url=f"{base_url}/object_info",
                    timeout=self._timeout(),
                    trusted_origins=self._trusted_origins(base_url),
                )
            except (ImageGenerationError, httpx.HTTPStatusError) as exc:
                raise VideoBackendUnavailableError(f"ComfyUI object_info failed: {exc}") from exc
            except Exception as exc:
                raise VideoBackendUnavailableError(f"ComfyUI object_info failed: {exc}") from exc
            if not isinstance(info, dict):
                raise VideoBackendUnavailableError("ComfyUI object_info response was not a JSON object")
            self._available_node_classes = {str(name) for name in info}
        missing = sorted(required - self._available_node_classes)
        if missing:
            raise VideoBackendUnavailableError(
                "ComfyUI is missing required workflow node classes: " + ", ".join(missing)
            )

    @staticmethod
    def _title_controls(node: dict[str, Any]) -> set[str]:
        """Return documented controls named by a node's title.

        A connected ComfyUI node can carry several independently mutable
        inputs, such as ``Width Height Frames``. Preserve the conventional
        multi-word labels while requiring an exact normalized title. Unknown
        extra words invalidate the title instead of mutating unrelated nodes.
        """
        meta = node.get("_meta")
        raw_title = meta.get("title") if isinstance(meta, dict) else ""
        compact = re.sub(r"[^a-z0-9]+", "", str(raw_title).lower())
        return set(_TITLE_CONTROLS.get(compact, frozenset()))

    @staticmethod
    def _set_input(inputs: dict[str, Any], fields: tuple[str, ...], value: Any) -> None:
        """Set the first existing unlinked canonical field, leaving links intact."""
        for field in fields:
            if field in inputs and not isinstance(inputs[field], list):
                inputs[field] = value
                return

    def _parameterize_workflow(
        self,
        workflow: dict[str, Any],
        request: VideoGenRequest,
        image_name: str | None,
    ) -> dict[str, Any]:
        """Deep-copy and inject request values into title-addressed graph nodes."""
        graph = copy.deepcopy(workflow)
        frames = None
        if request.duration_seconds is not None and request.fps is not None:
            frames = request.duration_seconds * request.fps

        for node in graph.values():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                continue
            controls = self._title_controls(node)
            if "prompt" in controls:
                self._set_input(inputs, ("text", "prompt"), request.prompt)
            if "negative_prompt" in controls:
                self._set_input(inputs, ("text", "prompt"), request.negative_prompt or "")
            if "seed" in controls and request.seed is not None:
                self._set_input(inputs, ("seed", "noise_seed"), request.seed)
            if "width" in controls and request.width is not None:
                self._set_input(inputs, ("width",), request.width)
            if "height" in controls and request.height is not None:
                self._set_input(inputs, ("height",), request.height)
            if "frames" in controls and frames is not None:
                self._set_input(inputs, ("num_frames", "frames", "length", "video_frames"), frames)
            if "fps" in controls and request.fps is not None:
                self._set_input(inputs, ("fps", "frame_rate"), request.fps)
            if "input_image" in controls and image_name is not None:
                self._set_input(inputs, ("image", "image_name"), image_name)
        return graph

    def _resolve_uploaded_image(
        self,
        assets: tuple[ResolvedReferenceAsset, ...],
    ) -> str | None:
        """Upload one supported in-memory image reference and return its name."""
        if not assets:
            return None
        return self._upload_image(assets[0])

    @staticmethod
    def _validate_reference_assets(assets: tuple[ResolvedReferenceAsset, ...]) -> None:
        """Reject unsupported reference kinds before contacting ComfyUI."""
        if not assets:
            return
        if len(assets) != 1:
            raise VideoGenerationError("ComfyUI supports one image first_frame/reference_image input")
        asset = assets[0]
        if asset.kind not in _SUPPORTED_REFERENCE_KINDS or not asset.mime_type.lower().startswith("image/"):
            raise VideoGenerationError(
                "ComfyUI supports only image first_frame/reference_image inputs"
            )

    # -- transport --------------------------------------------------------

    def _queue_prompt(self, base_url: str, workflow: dict[str, Any]) -> str:
        """POST a graph to ``/prompt`` and return the validated prompt id."""
        try:
            response = fetch_json(
                method="POST",
                url=f"{base_url}/prompt",
                json={"prompt": workflow, "client_id": str(uuid.uuid4())},
                timeout=self._timeout(),
                trusted_origins=self._trusted_origins(base_url),
            )
        except (ImageGenerationError, httpx.HTTPStatusError) as exc:
            raise VideoGenerationError(f"ComfyUI prompt submission failed: {exc}") from exc
        except Exception as exc:
            raise VideoGenerationError(f"ComfyUI prompt submission failed: {exc}") from exc
        if not isinstance(response, dict):
            raise VideoGenerationError("ComfyUI prompt response was not JSON")
        prompt_id = response.get("prompt_id")
        if not isinstance(prompt_id, str) or not _PROMPT_ID_RE.fullmatch(prompt_id):
            raise VideoGenerationError("ComfyUI prompt response did not include a valid prompt_id")
        return prompt_id

    def _poll_for_output(
        self,
        base_url: str,
        prompt_id: str,
        cancel_event: threading.Event | None,
    ) -> dict[str, str]:
        """Poll ``/history/{prompt_id}`` until ComfyUI exposes media output."""
        history_url = f"{base_url}/history/{prompt_id}"
        deadline = time.monotonic() + self._timeout()
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                self._interrupt(base_url)
                raise VideoGenerationError("ComfyUI video generation cancelled by user")
            try:
                history = fetch_json(
                    method="GET",
                    url=history_url,
                    timeout=self._timeout(),
                    trusted_origins=self._trusted_origins(base_url),
                )
            except (ImageGenerationError, httpx.HTTPStatusError) as exc:
                raise VideoGenerationError(f"ComfyUI history polling failed: {exc}") from exc
            except Exception as exc:
                raise VideoGenerationError(f"ComfyUI history polling failed: {exc}") from exc
            descriptor = self._find_output_descriptor(history, prompt_id)
            if descriptor is not None:
                return descriptor
            if cancel_event is not None:
                cancel_event.wait(1.0)
            else:
                time.sleep(1.0)
        raise VideoGenerationError("timed out waiting for ComfyUI video result")

    def _interrupt(self, base_url: str) -> None:
        """Best-effort interrupt of the current ComfyUI execution."""
        try:
            fetch_json(
                method="POST",
                url=f"{base_url}/interrupt",
                timeout=self._timeout(),
                trusted_origins=self._trusted_origins(base_url),
            )
        except Exception:
            # Local cancellation must never wait on or be masked by an
            # already-finished/unreachable ComfyUI server.
            return

    def _upload_image(self, asset: ResolvedReferenceAsset) -> str:
        """Upload one in-memory image through ComfyUI's multipart endpoint."""
        base_url = self._base_url()
        upload_url = f"{base_url}/upload/image"
        trusted = self._trusted_origins(base_url)
        filename = Path(asset.source_name or "input-image").name or "input-image"
        try:
            # ``create_client`` disables automatic redirects. Refusing a
            # redirect rather than blindly following it preserves the shared
            # egress guarantee for this multipart-only call site.
            _validate_egress_or_raise(upload_url, trusted_origins=trusted)
            with create_client(timeout=self._timeout()) as client:
                response = client.post(
                    upload_url,
                    files={"image": (filename, asset.content, asset.mime_type)},
                    data={"overwrite": "true"},
                )
                if response.is_redirect:
                    raise ImageGenerationError("ComfyUI image upload returned a redirect")
                response.raise_for_status()
                data = response.json()
        except (ImageGenerationError, httpx.HTTPError, ValueError) as exc:
            raise VideoGenerationError(f"ComfyUI image upload failed: {exc}") from exc
        except Exception as exc:
            raise VideoGenerationError(f"ComfyUI image upload failed: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("name"), str) or not data["name"].strip():
            raise VideoGenerationError("ComfyUI image upload response did not include a filename")
        name = data["name"].strip()
        subfolder = str(data.get("subfolder") or "").strip("/")
        return f"{subfolder}/{name}" if subfolder else name

    # -- history/output parsing ------------------------------------------

    @staticmethod
    def _safe_execution_message(messages: Any) -> str | None:
        """Extract a short, display-safe execution message from ComfyUI status."""
        candidates: list[str] = []

        def collect(value: Any) -> None:
            if isinstance(value, str):
                candidates.append(value)
            elif isinstance(value, dict):
                for key in ("exception_message", "message", "error", "details"):
                    candidate = value.get(key)
                    if isinstance(candidate, str):
                        candidates.append(candidate)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)

        collect(messages)
        for candidate in candidates:
            cleaned = " ".join(candidate.split())
            if cleaned and cleaned not in {"execution_error", "error"}:
                return cleaned[:240]
        return None

    @classmethod
    def _raise_for_terminal_history_status(cls, entry: dict[str, Any]) -> None:
        """Raise immediately for terminal ComfyUI execution failures."""
        status = entry.get("status")
        if not isinstance(status, dict):
            return
        state = str(status.get("status_str") or status.get("status") or "").strip().lower()
        if state in {"error", "failed", "interrupted", "cancelled", "canceled"}:
            detail = cls._safe_execution_message(status.get("messages"))
            message = "ComfyUI execution failed"
            if detail:
                message = f"{message}: {detail}"
            raise VideoGenerationError(message)

    @staticmethod
    def _is_terminal_success(entry: dict[str, Any]) -> bool:
        """Return whether ComfyUI explicitly completed the prompt successfully."""
        status = entry.get("status")
        if not isinstance(status, dict) or status.get("completed") is not True:
            return False
        state = str(status.get("status_str") or status.get("status") or "").strip().lower()
        return state in {"success", "succeeded", "completed", "complete"}

    @staticmethod
    def _find_output_descriptor(history: Any, prompt_id: str) -> dict[str, str] | None:
        """Find the first supported ComfyUI media descriptor in a history payload."""
        if not isinstance(history, dict):
            return None
        entry = history.get(prompt_id)
        if not isinstance(entry, dict):
            return None
        ComfyUIVideoAdapter._raise_for_terminal_history_status(entry)
        outputs = entry.get("outputs")
        if not isinstance(outputs, dict):
            return None
        for node_output in outputs.values():
            if not isinstance(node_output, dict):
                continue
            for collection in ("videos", "gifs", "images"):
                descriptors = node_output.get(collection)
                if not isinstance(descriptors, list):
                    continue
                for descriptor in descriptors:
                    if not isinstance(descriptor, dict):
                        continue
                    filename = descriptor.get("filename")
                    if not isinstance(filename, str) or not filename.strip():
                        continue
                    suffix = Path(filename).suffix.lower()
                    if suffix not in _VIDEO_SUFFIX_TYPES:
                        continue
                    return {
                        "filename": filename,
                        "subfolder": str(descriptor.get("subfolder") or ""),
                        "type": str(descriptor.get("type") or "output"),
                    }
        if outputs or ComfyUIVideoAdapter._is_terminal_success(entry):
            raise VideoGenerationError("ComfyUI history returned no supported video or animated-image output")
        return None

    def _download_output(self, base_url: str, descriptor: dict[str, str]) -> VideoGenResult:
        """Fetch one selected ``/view`` output with egress and byte limits."""
        view_url = f"{base_url}/view?{urlencode(descriptor)}"
        try:
            content, content_type = fetch_image_bytes(
                view_url,
                timeout=self._timeout(),
                max_bytes=max(1, int(self._config.download_max_mb)) * 1024 * 1024,
                trusted_origins=self._trusted_origins(base_url),
            )
        except ImageGenerationError as exc:
            raise VideoGenerationError(f"ComfyUI output download failed: {exc}") from exc
        normalized_type = (content_type or "").split(";", 1)[0].strip().lower()
        if normalized_type not in _VIDEO_SUFFIX_TYPES.values():
            normalized_type = _VIDEO_SUFFIX_TYPES[Path(descriptor["filename"]).suffix.lower()]
        return VideoGenResult(content=content, content_type=normalized_type, bytes_len=len(content))
