"""ComfyUI HTTP video-generation adapter (task-3401.6, ADR-044).

ComfyUI executes API-format workflow graphs.  This adapter deliberately keeps
model filenames inside those graphs: it resolves a user-selected JSON workflow,
updates only nodes bearing the documented title convention, then queues and
polls the graph through ComfyUI's standard HTTP API.
"""

from __future__ import annotations

import copy
import json
import math
import re
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
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
from tldw_chatbook.Utils.path_validation import validate_path
from tldw_chatbook.Utils.paths import get_user_data_dir
from tldw_chatbook.Video_Generation.adapters.base import (
    ResolvedReferenceAsset,
    VideoGenRequest,
    VideoGenResult,
)
from tldw_chatbook.Video_Generation.config import (
    DEFAULT_COMFYUI_BASE_URL,
    DEFAULT_COMFYUI_TIMEOUT_SECONDS,
    DEFAULT_COMFYUI_WORKFLOW,
    get_video_generation_config,
)
from tldw_chatbook.Video_Generation.exceptions import (
    VideoBackendUnavailableError,
    VideoGenerationError,
)
from tldw_chatbook.Video_Generation.video_formats import (
    canonical_video_extension,
    normalize_video_mime,
    video_container_for_mime,
)


_PROMPT_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_SUPPORTED_REFERENCE_KINDS = frozenset({"first_frame", "reference_image"})
_SUPPORTED_OUTPUT_CLASSES = frozenset({"SaveVideo", "VHS_VideoCombine", "SaveAnimatedWEBP"})
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
    "promptwidthheight": frozenset({"prompt", "width", "height"}),
    "duration": frozenset({"duration"}),
    "nativefps": frozenset({"native_fps"}),
}


@dataclass(frozen=True)
class _PreparedWorkflow:
    """An immutable record of a graph and the facts it will produce."""

    graph: dict[str, Any]
    duration_seconds: float | None
    fps: float | None
    width: int | None
    height: int | None
    resolved_seed: int | None


class ComfyUIVideoAdapter:
    """Generate videos by queueing parameterized ComfyUI API workflows.

    The configured ComfyUI base URL is a user-selected local/backend origin.
    Every request built from it therefore passes the host as ``trusted_origins``
    to the shared egress-aware HTTP code.  Workflow output URLs are never
    persisted; downloaded bytes are returned directly to the video store layer.
    """

    name = "comfyui"
    supported_formats = {"mp4", "webm"}

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
        output_format = request.format
        try:
            canonical_video_extension(output_format)
        except ValueError as exc:
            raise VideoGenerationError("unsupported output format") from exc
        workflow = self._load_workflow(self._workflow_name())
        if self._is_h3_workflow(workflow) and output_format != "mp4":
            raise VideoGenerationError("ComfyUI H3 supports MP4 output only")

        base_url = self._base_url()
        self._validate_reference_assets(request.reference_assets)
        if request.reference_assets and self._is_h3_workflow(workflow):
            raise VideoGenerationError("ComfyUI H3 does not support input image")
        self._validate_required_nodes(base_url, workflow)

        image_name = self._resolve_uploaded_image(request.reference_assets)
        prepared = self._parameterize_workflow(workflow, request, image_name)
        prompt_id = self._queue_prompt(base_url, prepared.graph)
        descriptor = self._poll_for_output(
            base_url, prompt_id, cancel_event, prepared.graph, output_format
        )
        return self._download_output(base_url, descriptor, prepared)

    # -- configuration / workflows --------------------------------------

    def _workflow_name(self) -> str:
        """Return the configured bare workflow filename or the shipped default."""
        return (
            self._config.comfyui_default_workflow or DEFAULT_COMFYUI_WORKFLOW
        ).strip()

    def selected_workflow_is_h3(self) -> bool:
        """Classify the configured graph without contacting ComfyUI."""
        return self._is_h3_workflow(self._load_workflow(self._workflow_name()))

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
        if workflow_root.is_symlink():
            raise VideoGenerationError(
                "ComfyUI workflow path escapes video_workflows"
            )
        user_candidate = workflow_root / candidate.name
        if user_candidate.is_symlink():
            raise VideoGenerationError("ComfyUI workflow symlink is not allowed")
        roots = (workflow_root, self._shipped_workflow_dir())
        validated_paths: list[Path] = []
        for root in roots:
            try:
                validated_paths.append(
                    validate_path(candidate.name, root, redact_paths=True)
                )
            except ValueError as exc:
                raise VideoGenerationError(
                    "ComfyUI workflow path is not allowed"
                ) from exc
            if validated_paths[-1].is_file():
                break
        selected = next((path for path in validated_paths if path.is_file()), None)
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
    def _set_input(inputs: dict[str, Any], fields: tuple[str, ...], value: Any) -> str | None:
        """Set and return the first direct canonical field, if available."""
        for field in fields:
            if field in inputs and not isinstance(inputs[field], list):
                inputs[field] = value
                return field
        return None

    @staticmethod
    def _is_h3_workflow(graph: dict[str, Any]) -> bool:
        """Return whether a graph exposes the MiniMax H3 fixed-control node."""
        return any(
            isinstance(node, dict)
            and node.get("class_type") == "MiniMaxH3ImageToVideo"
            for node in graph.values()
        )

    @staticmethod
    def _direct_value(inputs: dict[str, Any], fields: tuple[str, ...]) -> Any | None:
        """Return the first direct input value from a documented field name."""
        for field in fields:
            value = inputs.get(field)
            if field in inputs and not isinstance(value, list):
                return value
        return None

    @staticmethod
    def _require_injection(
        applied_field: str | None,
        request_field: str,
        expected_title: str,
    ) -> None:
        """Raise an actionable error for a supplied control that was not set."""
        if applied_field is None:
            raise VideoGenerationError(
                f"ComfyUI {request_field} requires a direct {expected_title!r} control"
            )

    @staticmethod
    def _require_direct_value(
        value: Any | None,
        request_field: str,
        expected_title: str,
    ) -> Any:
        """Return a direct value or report the documented title that is missing."""
        if value is None:
            raise VideoGenerationError(
                f"ComfyUI {request_field} requires a direct {expected_title!r} control"
            )
        return value

    @staticmethod
    def _require_number(value: Any, request_field: str, expected_title: str) -> float:
        """Validate a graph default that is expected to be numeric."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise VideoGenerationError(
                f"ComfyUI {request_field} requires a numeric {expected_title!r} control"
            )
        return float(value)

    @staticmethod
    def _require_h3_dimension(value: Any, field: str) -> int:
        """Validate one direct H3 dimension at the graph/request boundary."""
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value % 32
        ):
            raise VideoGenerationError(
                f"ComfyUI H3 {field} must be a positive integer and multiple of 32"
            )
        return value

    def _set_control(
        self,
        graph: dict[str, Any],
        control: str,
        fields: tuple[str, ...],
        value: Any,
    ) -> str | None:
        """Set every eligible exact-title control and return one applied field."""
        applied: str | None = None
        for node in graph.values():
            if not isinstance(node, dict) or control not in self._title_controls(node):
                continue
            inputs = node.get("inputs")
            if isinstance(inputs, dict):
                applied = self._set_input(inputs, fields, value) or applied
        return applied

    def _h3_generation_inputs(self, graph: dict[str, Any]) -> dict[str, Any]:
        """Find the H3 generation node and enforce its exact title contract."""
        expected_class = "MiniMaxH3ImageToVideo"
        expected_controls = {"prompt", "width", "height"}
        generation_nodes = [
            node
            for node in graph.values()
            if isinstance(node, dict) and node.get("class_type") == expected_class
        ]
        if len(generation_nodes) != 1:
            raise VideoGenerationError(
                "ComfyUI prompt 'Prompt Width Height' control requires "
                f"{expected_class}; expected exactly one generation node, found "
                f"{len(generation_nodes)}"
            )

        wrong_classes = sorted(
            {
                str(node.get("class_type") or "<missing>")
                for node in graph.values()
                if isinstance(node, dict)
                and expected_controls <= self._title_controls(node)
                and node.get("class_type") != expected_class
            }
        )
        if wrong_classes:
            raise VideoGenerationError(
                "ComfyUI prompt 'Prompt Width Height' control requires class "
                f"{expected_class}; found {', '.join(wrong_classes)}"
            )

        node = generation_nodes[0]
        inputs = node.get("inputs")
        if (
            not isinstance(inputs, dict)
            or not expected_controls <= self._title_controls(node)
        ):
            raise VideoGenerationError(
                "ComfyUI prompt requires a direct 'Prompt Width Height' control "
                f"on class {expected_class}"
            )
        return inputs

    def _h3_control_inputs(
        self,
        graph: dict[str, Any],
        control: str,
        request_field: str,
        expected_title: str,
        expected_class: str,
    ) -> dict[str, Any]:
        """Return inputs for an exact H3 support control or raise clearly."""
        titled_nodes = [
            node
            for node in graph.values()
            if isinstance(node, dict) and control in self._title_controls(node)
        ]
        wrong_classes = sorted(
            {
                str(node.get("class_type") or "<missing>")
                for node in titled_nodes
                if node.get("class_type") != expected_class
            }
        )
        if wrong_classes:
            raise VideoGenerationError(
                f"ComfyUI {request_field} {expected_title!r} control requires class "
                f"{expected_class}; found {', '.join(wrong_classes)}"
            )
        matching_nodes = [
            node for node in titled_nodes if node.get("class_type") == expected_class
        ]
        if len(matching_nodes) != 1:
            raise VideoGenerationError(
                f"ComfyUI {request_field} {expected_title!r} control requires "
                f"{expected_class}; expected exactly one node, found "
                f"{len(matching_nodes)}"
            )
        inputs = matching_nodes[0].get("inputs")
        if not isinstance(inputs, dict):
            raise VideoGenerationError(
                f"ComfyUI {request_field} requires a direct {expected_title!r} control "
                f"on class {expected_class}"
            )
        return inputs

    @staticmethod
    def _parse_ratio(ratio: str) -> float:
        """Parse a positive numeric ``W:H`` ratio supplied for an H3 graph."""
        match = re.fullmatch(
            r"\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*", ratio
        )
        if match is None:
            raise VideoGenerationError("ComfyUI H3 ratio must be a numeric W:H value")
        width, height = (float(part) for part in match.groups())
        if width <= 0 or height <= 0:
            raise VideoGenerationError("ComfyUI H3 ratio must be a positive numeric W:H value")
        return width / height

    def _parameterize_h3_workflow(
        self,
        graph: dict[str, Any],
        request: VideoGenRequest,
        image_name: str | None,
        resolved_seed: int | None,
    ) -> _PreparedWorkflow:
        """Apply and validate the fixed MiniMax H3 workflow control contract."""
        if request.negative_prompt:
            raise VideoGenerationError("ComfyUI H3 does not support negative prompt")
        if image_name is not None:
            raise VideoGenerationError("ComfyUI H3 does not support input image")
        if request.extra_params:
            raise VideoGenerationError("ComfyUI H3 does not support extra params")
        for field in ("model", "sampler", "steps", "cfg_scale"):
            if getattr(request, field) is not None:
                raise VideoGenerationError(
                    f"ComfyUI H3 does not support {field.replace('_', ' ')}"
                )

        generation_inputs = self._h3_generation_inputs(graph)
        self._require_injection(
            self._set_input(generation_inputs, ("prompt",), request.prompt),
            "prompt",
            "Prompt Width Height",
        )
        for field, value in (("width", request.width), ("height", request.height)):
            if value is not None:
                self._require_h3_dimension(value, field)
                self._require_injection(
                    self._set_input(generation_inputs, (field,), value),
                    field,
                    "Prompt Width Height",
                )

        seed_inputs = self._h3_control_inputs(
            graph, "seed", "seed", "Seed", "RandomNoise"
        )
        if resolved_seed is not None:
            self._require_injection(
                self._set_input(seed_inputs, ("noise_seed", "seed"), resolved_seed),
                "seed",
                "Seed",
            )
        effective_seed = self._require_direct_value(
            self._direct_value(seed_inputs, ("noise_seed", "seed")), "seed", "Seed"
        )
        if isinstance(effective_seed, bool) or not isinstance(effective_seed, int) or effective_seed < 0:
            raise VideoGenerationError("ComfyUI seed requires a non-negative integer 'Seed' control")

        duration_inputs = self._h3_control_inputs(
            graph, "duration", "duration", "Duration", "PrimitiveFloat"
        )
        if request.duration_seconds is not None:
            self._require_injection(
                self._set_input(duration_inputs, ("value", "duration_seconds"), request.duration_seconds),
                "duration",
                "Duration",
            )
        duration = self._require_number(
            self._require_direct_value(
                self._direct_value(duration_inputs, ("value", "duration_seconds")),
                "duration",
                "Duration",
            ),
            "duration",
            "Duration",
        )
        if not math.isfinite(duration) or duration <= 0:
            raise VideoGenerationError(
                "ComfyUI H3 duration must be finite and greater than 0"
            )

        fps_inputs = self._h3_control_inputs(
            graph, "native_fps", "native FPS", "Native FPS", "CreateVideo"
        )
        fps = self._require_number(
            self._require_direct_value(
                self._direct_value(fps_inputs, ("fps", "frame_rate")),
                "native FPS",
                "Native FPS",
            ),
            "native FPS",
            "Native FPS",
        )
        if fps != 24.0:
            raise VideoGenerationError("ComfyUI H3 native FPS control must be 24")
        if request.fps is not None and request.fps != 24:
            raise VideoGenerationError(
                f"ComfyUI H3 native FPS is 24; requested {request.fps} is unsupported"
            )

        width = self._require_h3_dimension(
            self._require_direct_value(
                self._direct_value(generation_inputs, ("width",)),
                "width",
                "Prompt Width Height",
            ),
            "width",
        )
        height = self._require_h3_dimension(
            self._require_direct_value(
                self._direct_value(generation_inputs, ("height",)),
                "height",
                "Prompt Width Height",
            ),
            "height",
        )

        if request.ratio is not None:
            if request.ratio.strip().lower() == "adaptive":
                raise VideoGenerationError("ComfyUI H3 ratio cannot be adaptive")
            expected_ratio = self._parse_ratio(request.ratio)
            actual_ratio = width / height
            if abs(actual_ratio - expected_ratio) / expected_ratio > 0.03 + 1e-12:
                raise VideoGenerationError("ComfyUI H3 ratio is incompatible with effective dimensions")

        save_nodes = [
            node for node in graph.values()
            if isinstance(node, dict) and node.get("class_type") == "SaveVideo"
        ]
        if not save_nodes:
            raise VideoGenerationError("ComfyUI H3 requires a SaveVideo MP4 output control")
        if any(
            not isinstance(node.get("inputs"), dict)
            or str(node["inputs"].get("format", "")).lower() != "mp4"
            for node in save_nodes
        ):
            raise VideoGenerationError("ComfyUI H3 SaveVideo format must be MP4")
        if str(request.format or "").lower() != "mp4":
            raise VideoGenerationError("ComfyUI H3 supports MP4 output only")

        return _PreparedWorkflow(
            graph=graph,
            duration_seconds=duration,
            fps=fps,
            width=width,
            height=height,
            resolved_seed=effective_seed,
        )

    def _parameterize_generic_workflow(
        self,
        graph: dict[str, Any],
        request: VideoGenRequest,
        image_name: str | None,
        resolved_seed: int | None,
    ) -> _PreparedWorkflow:
        """Apply the existing custom-workflow title convention without silent no-ops."""
        effective_width: int | None = None
        effective_height: int | None = None
        effective_duration: float | None = None
        effective_fps: float | None = None
        self._require_injection(
            self._set_control(graph, "prompt", ("text", "prompt"), request.prompt),
            "prompt",
            "Prompt",
        )
        if request.negative_prompt is not None:
            self._require_injection(
                self._set_control(graph, "negative_prompt", ("text", "prompt"), request.negative_prompt),
                "negative prompt",
                "Negative Prompt",
            )
        else:
            self._set_control(graph, "negative_prompt", ("text", "prompt"), "")
        if resolved_seed is not None:
            self._require_injection(
                self._set_control(graph, "seed", ("seed", "noise_seed"), resolved_seed),
                "seed",
                "Seed",
            )
        for field in ("width", "height"):
            value = getattr(request, field)
            if value is not None:
                self._require_injection(
                    self._set_control(graph, field, (field,), value),
                    field,
                    field.title(),
                )
                if field == "width":
                    effective_width = value
                else:
                    effective_height = value
        if request.duration_seconds is not None:
            duration_applied = self._set_control(
                graph, "duration", ("value", "duration_seconds"), request.duration_seconds
            )
            if duration_applied is None and request.fps is not None:
                duration_applied = self._set_control(
                    graph,
                    "frames",
                    ("num_frames", "frames", "length", "video_frames"),
                    request.duration_seconds * request.fps,
                )
            self._require_injection(duration_applied, "duration", "Duration or Frames")
            effective_duration = float(request.duration_seconds)
        if request.fps is not None:
            self._require_injection(
                self._set_control(graph, "fps", ("fps", "frame_rate"), request.fps),
                "fps",
                "FPS",
            )
            effective_fps = float(request.fps)
        if image_name is not None:
            self._require_injection(
                self._set_control(graph, "input_image", ("image", "image_name"), image_name),
                "input image",
                "Input Image",
            )
        if request.extra_params:
            raise VideoGenerationError("ComfyUI custom workflow does not support extra params")
        for field in ("ratio", "model", "sampler", "steps", "cfg_scale"):
            if getattr(request, field) is not None:
                raise VideoGenerationError(
                    f"ComfyUI custom workflow does not support {field.replace('_', ' ')}"
                )
        return _PreparedWorkflow(
            graph=graph,
            duration_seconds=effective_duration,
            fps=effective_fps,
            width=effective_width,
            height=effective_height,
            resolved_seed=resolved_seed,
        )

    def _parameterize_workflow(
        self,
        workflow: dict[str, Any],
        request: VideoGenRequest,
        image_name: str | None,
    ) -> _PreparedWorkflow:
        """Deep-copy and strictly inject request values into a workflow graph."""
        graph = copy.deepcopy(workflow)
        requested_seed = request.seed
        if requested_seed is not None and requested_seed < -1:
            raise VideoGenerationError("ComfyUI seed must be -1 or a non-negative integer")
        resolved_seed = secrets.randbelow(2**63) if requested_seed == -1 else requested_seed
        if self._is_h3_workflow(graph):
            return self._parameterize_h3_workflow(graph, request, image_name, resolved_seed)
        return self._parameterize_generic_workflow(graph, request, image_name, resolved_seed)

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
        graph: dict[str, Any],
        requested_container: str,
    ) -> dict[str, str]:
        """Poll ``/history/{prompt_id}`` until ComfyUI exposes media output."""
        history_url = f"{base_url}/history/{prompt_id}"
        deadline = time.monotonic() + self._timeout()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if cancel_event is not None and cancel_event.is_set():
                self._interrupt(base_url)
                raise VideoGenerationError("ComfyUI video generation cancelled by user")
            try:
                history = fetch_json(
                    method="GET",
                    url=history_url,
                    timeout=remaining,
                    trusted_origins=self._trusted_origins(base_url),
                )
            except (ImageGenerationError, httpx.HTTPStatusError) as exc:
                raise VideoGenerationError(f"ComfyUI history polling failed: {exc}") from exc
            except Exception as exc:
                raise VideoGenerationError(f"ComfyUI history polling failed: {exc}") from exc
            descriptor = self._find_output_descriptor(
                history, prompt_id, graph, requested_container
            )
            if descriptor is not None:
                return descriptor
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wait_seconds = min(1.0, remaining)
            if cancel_event is not None:
                cancel_event.wait(wait_seconds)
            else:
                time.sleep(wait_seconds)
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
        raw_subfolder = data.get("subfolder", "")
        if raw_subfolder is None:
            raw_subfolder = ""
        if not isinstance(raw_subfolder, str):
            raise VideoGenerationError("ComfyUI image upload response included an unsafe path")
        subfolder = raw_subfolder.strip()
        if (
            name in {".", ".."}
            or "/" in name
            or "\\" in name
            or "\x00" in name
            or "\\" in subfolder
            or "\x00" in subfolder
            or (
                subfolder
                and any(
                    part in {"", ".", ".."} for part in subfolder.split("/")
                )
            )
        ):
            raise VideoGenerationError("ComfyUI image upload response included an unsafe path")
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
    def _output_node_ids(graph: dict[str, Any]) -> tuple[str, ...]:
        """Return graph node ids whose classes can persist generated media."""
        return tuple(
            str(node_id)
            for node_id, node in graph.items()
            if node.get("class_type") in _SUPPORTED_OUTPUT_CLASSES
        )

    @staticmethod
    def _find_output_descriptor(
        history: Any,
        prompt_id: str,
        graph: dict[str, Any],
        requested_container: str,
    ) -> dict[str, str] | None:
        """Find supported media only under output nodes declared by the graph."""
        if not isinstance(history, dict):
            return None
        entry = history.get(prompt_id)
        if not isinstance(entry, dict):
            return None
        ComfyUIVideoAdapter._raise_for_terminal_history_status(entry)
        if not ComfyUIVideoAdapter._is_terminal_success(entry):
            return None
        expected_suffix = f".{canonical_video_extension(requested_container)}"
        matches: list[dict[str, str]] = []
        outputs = entry.get("outputs")
        if isinstance(outputs, dict):
            for node_id in ComfyUIVideoAdapter._output_node_ids(graph):
                node_output = outputs.get(node_id)
                if not isinstance(node_output, dict):
                    continue
                for descriptors in node_output.values():
                    if not isinstance(descriptors, list):
                        continue
                    for descriptor in descriptors:
                        if not isinstance(descriptor, dict):
                            continue
                        filename = descriptor.get("filename")
                        subfolder = descriptor.get("subfolder", "")
                        output_type = descriptor.get("type")
                        if not isinstance(filename, str) or not filename.strip():
                            continue
                        if subfolder is None:
                            subfolder = ""
                        if not isinstance(subfolder, str) or output_type != "output":
                            continue
                        filename = filename.strip()
                        suffix = Path(filename).suffix
                        if suffix != expected_suffix:
                            continue
                        matches.append(
                            {
                                "filename": filename,
                                "subfolder": subfolder,
                                "type": output_type,
                            }
                        )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise VideoGenerationError(
                "ComfyUI history returned multiple matching canonical video outputs"
            )
        raise VideoGenerationError(
            "ComfyUI history returned no matching canonical video output"
        )

    def _download_output(
        self,
        base_url: str,
        descriptor: dict[str, str],
        prepared: _PreparedWorkflow,
    ) -> VideoGenResult:
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
        suffix = Path(descriptor["filename"]).suffix
        try:
            container = suffix.removeprefix(".")
            canonical_video_extension(container)
            normalized_type = normalize_video_mime(content_type)
            mime_container = video_container_for_mime(normalized_type)
        except ValueError as exc:
            message = (
                "ComfyUI workflow did not return an MP4 output"
                if self._is_h3_workflow(prepared.graph)
                else "ComfyUI output container and MIME did not agree"
            )
            raise VideoGenerationError(message) from exc
        if mime_container != container:
            message = (
                "ComfyUI workflow did not return an MP4 output"
                if self._is_h3_workflow(prepared.graph)
                else "ComfyUI output container and MIME did not agree"
            )
            raise VideoGenerationError(message)
        return VideoGenResult(
            content=content,
            content_type=normalized_type,
            container=container,
            bytes_len=len(content),
            duration_seconds=prepared.duration_seconds,
            fps=prepared.fps,
            width=prepared.width,
            height=prepared.height,
            resolved_seed=prepared.resolved_seed,
        )
