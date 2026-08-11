"""Strict packaged-workflow ComfyUI adapter for one H3 image edit."""

from __future__ import annotations

import io
import json
import re
import secrets
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from importlib.resources import files
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote

import httpx
from loguru import logger
from PIL import Image, UnidentifiedImageError

from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest, ImageGenResult
from tldw_chatbook.Image_Generation.config import (
    get_image_generation_config,
    normalize_comfyui_image_origin,
)
from tldw_chatbook.Image_Generation.exceptions import (
    ComfyUIImageEditError,
    ComfyUIImageEditPhase,
    ImageGenerationCancelled,
)
from tldw_chatbook.Utils.egress import (
    check_url_or_raise,
    host_of,
    same_origin,
)

H3_IMAGE_EDIT_WORKFLOW_KEY = "minimax_h3_image_edit"
_WORKFLOW_RESOURCE_DIRECTORY = "workflows"
COMFYUI_MAX_JSON_BYTES = 32 * 1024 * 1024
_MAX_SEED = 2**64 - 1
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}\Z")
_SAFE_PROMPT_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_REFERENCE_EXTENSION = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}
_SUPPORTED_PNG_MODES = frozenset({"1", "L", "LA", "P", "RGB", "RGBA"})

_EXPECTED_NODE_CLASSES = {
    "114": "LoadImage",
    "121": "VAELoader",
    "124": "VAEDecode",
    "125": "KSamplerSelect",
    "126": "BasicScheduler",
    "127": "SamplerCustomAdvanced",
    "128": "BasicGuider",
    "129": "UNETLoader",
    "130": "CLIPLoader",
    "131": "RandomNoise",
    "133": "MiniMaxH3ImageToVideo",
    "139": "PrimitiveInt",
    "140": "GetImageSize",
    "141": "ImageScaleToTotalPixels",
    "144": "ImageFromBatch",
    "149": "ResizeImageMaskNode",
    "150": "GetImageSize",
    "165": "SaveImage",
}
_EXPECTED_INPUT_KEYS = {
    "114": frozenset({"image"}),
    "121": frozenset({"vae_name"}),
    "124": frozenset({"samples", "vae"}),
    "125": frozenset({"sampler_name"}),
    "126": frozenset({"denoise", "model", "scheduler", "steps"}),
    "127": frozenset({"guider", "latent_image", "noise", "sampler", "sigmas"}),
    "128": frozenset({"conditioning", "model"}),
    "129": frozenset({"unet_name", "weight_dtype"}),
    "130": frozenset({"clip_name", "device", "type"}),
    "131": frozenset({"noise_seed"}),
    "133": frozenset({"clip", "first_frame", "height", "length", "prompt", "vae", "width"}),
    "139": frozenset({"value"}),
    "140": frozenset({"image"}),
    "141": frozenset({"image", "megapixels", "resolution_steps", "upscale_method"}),
    "144": frozenset({"batch_index", "image", "length"}),
    "149": frozenset(
        {
            "input",
            "resize_type",
            "resize_type.crop",
            "resize_type.height",
            "resize_type.width",
            "scale_method",
        }
    ),
    "150": frozenset({"image"}),
    "165": frozenset({"filename_prefix", "images"}),
}
_EXPECTED_DIRECT_LINKS = {
    "124.samples": ("127", 0),
    "124.vae": ("121", 0),
    "126.model": ("129", 0),
    "127.guider": ("128", 0),
    "127.latent_image": ("133", 1),
    "127.noise": ("131", 0),
    "127.sampler": ("125", 0),
    "127.sigmas": ("126", 0),
    "128.conditioning": ("133", 0),
    "128.model": ("129", 0),
    "133.clip": ("130", 0),
    "133.first_frame": ("114", 0),
    "133.height": ("140", 1),
    "133.length": ("139", 0),
    "133.vae": ("121", 0),
    "133.width": ("140", 0),
    "140.image": ("141", 0),
    "141.image": ("114", 0),
    "144.image": ("124", 0),
    "149.input": ("144", 0),
    "149.resize_type.height": ("150", 1),
    "149.resize_type.width": ("150", 0),
    "150.image": ("114", 0),
    "165.images": ("149", 0),
}


@dataclass(frozen=True)
class _PreparedWorkflow:
    graph: dict[str, Any]
    seed: int
    steps: int
    sampler: str
    width: int
    height: int


class _DeadlineExpired(Exception):
    def __init__(self, phase: ComfyUIImageEditPhase) -> None:
        self.phase = phase


def _load_packaged_workflow(
    workflow_key: str = H3_IMAGE_EDIT_WORKFLOW_KEY,
) -> dict[str, Any]:
    """Load a fresh copy of the one supported packaged workflow."""
    if (
        not isinstance(workflow_key, str)
        or "/" in workflow_key
        or "\\" in workflow_key
        or workflow_key != H3_IMAGE_EDIT_WORKFLOW_KEY
    ):
        raise ValueError("Unsupported packaged workflow key")

    resource = files("tldw_chatbook.Image_Generation").joinpath(
        _WORKFLOW_RESOURCE_DIRECTORY,
        f"{workflow_key}.json",
    )
    try:
        with resource.open("r", encoding="utf-8") as stream:
            graph = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("Packaged workflow is unavailable or invalid") from None

    if not isinstance(graph, dict) or not graph:
        raise ValueError("Packaged workflow must be a nonempty JSON object")
    return deepcopy(graph)


def _is_link(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
    )


def _validate_workflow_topology(graph: dict[str, Any]) -> None:
    try:
        classes = {
            node_id: node["class_type"]
            for node_id, node in graph.items()
            if isinstance(node_id, str) and isinstance(node, dict)
        }
        if classes != _EXPECTED_NODE_CLASSES:
            raise ValueError
        for node_id, expected_keys in _EXPECTED_INPUT_KEYS.items():
            inputs = graph[node_id]["inputs"]
            if not isinstance(inputs, dict) or frozenset(inputs) != expected_keys:
                raise ValueError
        links = {
            f"{node_id}.{name}": tuple(value)
            for node_id, node in graph.items()
            for name, value in node["inputs"].items()
            if _is_link(value)
        }
        if links != _EXPECTED_DIRECT_LINKS:
            raise ValueError
        output_nodes = {
            node_id
            for node_id, node in graph.items()
            if node.get("class_type") == "SaveImage"
        }
        if output_nodes != {"165"}:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ComfyUIImageEditError("packaged_workflow_validation") from None


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        return bool(value)
    return True


def _request_reference_dimensions(request: ImageGenRequest) -> tuple[int, int]:
    reference = request.reference_image
    if reference is None or type(reference.content) is not bytes or not reference.content:
        raise ComfyUIImageEditError("request_validation")
    if reference.mime_type not in _REFERENCE_EXTENSION:
        raise ComfyUIImageEditError("request_validation")
    width, height = reference.width, reference.height
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ComfyUIImageEditError("request_validation")
    return width, height


def _resolve_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ComfyUIImageEditError("request_validation")
    if value == -1:
        return secrets.randbits(64)
    if value < 0 or value > _MAX_SEED:
        raise ComfyUIImageEditError("request_validation")
    return value


def _resolve_steps(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ComfyUIImageEditError("request_validation")
    return value


def _resolve_sampler(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ComfyUIImageEditError("request_validation")
    return value.strip()


def _prepare_workflow(
    request: ImageGenRequest,
    *,
    config: Any | None = None,
    graph: dict[str, Any] | None = None,
    upload_name: str | None = None,
) -> _PreparedWorkflow:
    """Validate and prepare a fresh graph without selecting by title."""
    config = config or get_image_generation_config()
    if request.backend.strip().lower() != "comfyui":
        raise ComfyUIImageEditError("request_validation")
    if not isinstance(request.prompt, str) or not request.prompt.strip():
        raise ComfyUIImageEditError("request_validation")
    if (
        _nonempty(request.negative_prompt)
        or request.cfg_scale is not None
        or _nonempty(request.model)
        or str(request.format or "").strip().lower() != "png"
        or request.width is not None
        or request.height is not None
        or _nonempty(request.extra_params)
    ):
        raise ComfyUIImageEditError("request_validation")
    width, height = _request_reference_dimensions(request)

    prepared = deepcopy(graph if graph is not None else _load_packaged_workflow())
    _validate_workflow_topology(prepared)
    packaged_seed = prepared["131"]["inputs"]["noise_seed"]
    packaged_steps = prepared["126"]["inputs"]["steps"]
    packaged_sampler = prepared["125"]["inputs"]["sampler_name"]
    seed = _resolve_seed(
        request.seed
        if request.seed is not None
        else (
            config.comfyui_image_default_seed
            if config.comfyui_image_default_seed is not None
            else packaged_seed
        )
    )
    steps = _resolve_steps(
        request.steps
        if request.steps is not None
        else (
            config.comfyui_image_default_steps
            if config.comfyui_image_default_steps is not None
            else packaged_steps
        )
    )
    sampler = _resolve_sampler(
        request.sampler
        if request.sampler is not None
        else (config.comfyui_image_default_sampler or packaged_sampler)
    )

    if upload_name is not None:
        prepared["114"]["inputs"]["image"] = upload_name
    prepared["125"]["inputs"]["sampler_name"] = sampler
    prepared["126"]["inputs"]["steps"] = steps
    prepared["131"]["inputs"]["noise_seed"] = seed
    prepared["133"]["inputs"]["prompt"] = request.prompt
    return _PreparedWorkflow(prepared, seed, steps, sampler, width, height)


def _declared_length(response: httpx.Response) -> int | None:
    value = response.headers.get("content-length")
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError("invalid declared response length") from None
    if parsed < 0:
        raise ValueError("invalid declared response length")
    return parsed


def _read_bounded_json(
    response: httpx.Response,
    *,
    allow_empty: bool = False,
    cancel_event: Any = None,
    deadline: float | None = None,
    phase: ComfyUIImageEditPhase = "remote_schema_preflight",
) -> Any:
    """Bound declared and streamed JSON bytes before parsing."""
    _check_stream_control(cancel_event, deadline, phase)
    declared = _declared_length(response)
    if declared is not None and declared > COMFYUI_MAX_JSON_BYTES:
        raise ValueError("JSON response exceeds limit")
    collected = bytearray()
    chunks = iter(response.iter_bytes())
    while True:
        _check_stream_control(cancel_event, deadline, phase)
        try:
            chunk = next(chunks)
        except StopIteration:
            _check_stream_control(cancel_event, deadline, phase)
            break
        _check_stream_control(cancel_event, deadline, phase)
        collected.extend(chunk)
        if len(collected) > COMFYUI_MAX_JSON_BYTES:
            raise ValueError("JSON response exceeds limit")
        _check_stream_control(cancel_event, deadline, phase)
    if not collected and allow_empty:
        return {}
    if not collected:
        raise ValueError("empty JSON response")
    try:
        return json.loads(bytes(collected))
    except (UnicodeError, json.JSONDecodeError, TypeError):
        raise ValueError("invalid JSON response") from None


def _schema_inputs(class_schema: dict[str, Any]) -> dict[str, Any]:
    input_schema = class_schema.get("input")
    if not isinstance(input_schema, dict):
        raise ValueError
    required = input_schema.get("required", {})
    optional = input_schema.get("optional", {})
    if not isinstance(required, dict) or not isinstance(optional, dict):
        raise ValueError
    return {**optional, **required}


def _validate_literal_against_schema(value: Any, input_schema: Any) -> None:
    if not isinstance(input_schema, list) or not input_schema:
        raise ValueError
    input_type = input_schema[0]
    if isinstance(input_type, list):
        if value not in input_type:
            raise ValueError
        return
    if input_type == "INT":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError
        options = input_schema[1] if len(input_schema) > 1 else {}
        if isinstance(options, dict):
            if "min" in options and value < options["min"]:
                raise ValueError
            if "max" in options and value > options["max"]:
                raise ValueError
    elif input_type == "FLOAT" and (
        isinstance(value, bool) or not isinstance(value, (int, float))
    ):
        raise ValueError
    elif input_type == "BOOLEAN" and not isinstance(value, bool):
        raise ValueError
    elif input_type == "STRING" and not isinstance(value, str):
        raise ValueError


def _validate_load_image_upload_schema(input_schema: Any) -> None:
    if not isinstance(input_schema, list) or len(input_schema) < 2:
        raise ValueError
    choices, metadata = input_schema[0], input_schema[1]
    if (
        not isinstance(choices, list)
        or any(not isinstance(choice, str) for choice in choices)
        or not isinstance(metadata, dict)
        or metadata.get("image_upload") is not True
    ):
        raise ValueError


def _concrete_input_type(input_schema: Any) -> str:
    if (
        not isinstance(input_schema, list)
        or not input_schema
        or not isinstance(input_schema[0], str)
        or not input_schema[0]
    ):
        raise ValueError
    return input_schema[0]


def _validate_object_info(prepared: _PreparedWorkflow, schema: Any) -> None:
    """Require exact input availability, link types, literals, and PNG output."""
    try:
        if not isinstance(schema, dict):
            raise ValueError
        for node_id, node in prepared.graph.items():
            class_type = node["class_type"]
            class_schema = schema.get(class_type)
            if not isinstance(class_schema, dict):
                raise ValueError
            accepted_inputs = _schema_inputs(class_schema)
            for input_name, value in node["inputs"].items():
                if input_name not in accepted_inputs:
                    raise ValueError
                input_schema = accepted_inputs[input_name]
                if _is_link(value):
                    source_id, output_index = value
                    source_class = prepared.graph[source_id]["class_type"]
                    source_schema = schema.get(source_class)
                    if not isinstance(source_schema, dict):
                        raise ValueError
                    output_types = source_schema.get("output")
                    if (
                        not isinstance(output_types, list)
                        or output_index < 0
                        or output_index >= len(output_types)
                        or output_types[output_index]
                        != _concrete_input_type(input_schema)
                    ):
                        raise ValueError
                elif node_id == "114" and input_name == "image":
                    _validate_load_image_upload_schema(input_schema)
                else:
                    _validate_literal_against_schema(value, input_schema)
        save_schema = schema["SaveImage"]
        save_inputs = _schema_inputs(save_schema)
        if (
            save_schema.get("output_node") is not True
            or save_inputs["images"][0] != "IMAGE"
            or save_inputs["filename_prefix"][0] != "STRING"
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ComfyUIImageEditError("remote_schema_preflight") from None


def _safe_filename(value: Any, *, suffix: str | None = None) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ValueError
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError
    if suffix is not None and not value.lower().endswith(suffix):
        raise ValueError
    return value


def _safe_subfolder(value: Any) -> str:
    if value in {None, ""}:
        return ""
    if not isinstance(value, str) or "\\" in value:
        raise ValueError
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError
    for part in path.parts:
        _safe_filename(part)
    return path.as_posix()


def _select_output_descriptor(history: Any, prompt_id: str) -> dict[str, str]:
    """Select exactly one safe output descriptor from node 165 only."""
    try:
        record = history[prompt_id]
        outputs = record["outputs"]
        node_output = outputs["165"]
        images = node_output["images"]
        if not isinstance(images, list) or len(images) != 1:
            raise ValueError
        descriptor = images[0]
        if not isinstance(descriptor, dict) or descriptor.get("type") != "output":
            raise ValueError
        return {
            "filename": _safe_filename(descriptor.get("filename"), suffix=".png"),
            "subfolder": _safe_subfolder(descriptor.get("subfolder")),
            "type": "output",
        }
    except (KeyError, TypeError, ValueError):
        raise ComfyUIImageEditError("output_descriptor_validation") from None


def _check_stream_control(
    cancel_event: Any,
    deadline: float | None,
    phase: ComfyUIImageEditPhase,
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise ImageGenerationCancelled()
    if deadline is not None and time.monotonic() >= deadline:
        raise _DeadlineExpired(phase)


def _stream_png(
    response: httpx.Response,
    *,
    max_bytes: int,
    cancel_event: Any = None,
    deadline: float | None = None,
    phase: ComfyUIImageEditPhase = "output_download",
) -> bytes:
    """Read one PNG with declared and actual byte bounds."""
    _check_stream_control(cancel_event, deadline, phase)
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("invalid PNG limit")
    declared = _declared_length(response)
    if declared is not None and declared > max_bytes:
        raise ValueError("PNG response exceeds limit")
    content_type = (response.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
    if content_type != "image/png":
        raise ValueError("invalid PNG content type")
    collected = bytearray()
    chunks = iter(response.iter_bytes())
    while True:
        _check_stream_control(cancel_event, deadline, phase)
        try:
            chunk = next(chunks)
        except StopIteration:
            _check_stream_control(cancel_event, deadline, phase)
            break
        _check_stream_control(cancel_event, deadline, phase)
        collected.extend(chunk)
        if len(collected) > max_bytes:
            raise ValueError("PNG response exceeds limit")
        _check_stream_control(cancel_event, deadline, phase)
    data = bytes(collected)
    if not data.startswith(_PNG_SIGNATURE):
        raise ValueError("invalid PNG signature")
    return data


class ComfyUIImageAdapter:
    """Execute the one packaged H3 edit graph against one trusted origin."""

    name = "comfyui"
    supported_formats = {"png"}

    def __init__(self, *, config: Any | None = None, transport: Any | None = None) -> None:
        self.config = config or get_image_generation_config()
        self.origin = normalize_comfyui_image_origin(
            self.config.comfyui_image_base_url
        )
        self._trusted_host = host_of(self.origin)
        if not self._trusted_host:
            raise ValueError("invalid ComfyUI origin")
        self._client = httpx.Client(transport=transport)

    def _endpoint(self, path: str) -> str:
        if not isinstance(path, str) or not path.startswith("/") or path.startswith("//"):
            raise ValueError("invalid ComfyUI endpoint")
        url = f"{self.origin}{path}"
        if not same_origin(self.origin, url):
            raise ValueError("cross-origin ComfyUI endpoint")
        return url

    def _check_cancelled(self, event: Any) -> None:
        if event is not None and event.is_set():
            raise ImageGenerationCancelled()

    def _remaining(self, deadline: float, phase: ComfyUIImageEditPhase) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _DeadlineExpired(phase)
        return remaining

    def _timeout(self, remaining: float) -> httpx.Timeout:
        request_timeout = min(
            float(self.config.comfyui_image_request_timeout_seconds), remaining
        )
        connect_timeout = min(
            float(self.config.comfyui_image_connect_timeout_seconds), remaining
        )
        return httpx.Timeout(request_timeout, connect=connect_timeout)

    def _send(
        self,
        url: str,
        *,
        method: str,
        deadline: float,
        phase: ComfyUIImageEditPhase,
        cancel_event: Any = None,
        json_body: Any = None,
        files: Any = None,
        params: dict[str, str] | None = None,
        ignore_cancel: bool = False,
    ) -> httpx.Response:
        if not ignore_cancel:
            self._check_cancelled(cancel_event)
        if not same_origin(self.origin, url):
            raise ValueError("cross-origin ComfyUI request")
        check_url_or_raise(url, trusted_origins=frozenset({self._trusted_host}))
        remaining = self._remaining(deadline, phase)
        request = self._client.build_request(
            method,
            url,
            json=json_body,
            files=files,
            params=params,
            timeout=self._timeout(remaining),
        )
        response = self._client.send(request, stream=True, follow_redirects=False)
        if response.status_code < 200 or response.status_code >= 300:
            response.close()
            raise ValueError("ComfyUI request failed")
        return response

    def _request_json(
        self,
        url: str,
        *,
        method: str,
        deadline: float | None = None,
        phase: ComfyUIImageEditPhase = "remote_schema_preflight",
        cancel_event: Any = None,
        json_body: Any = None,
        files: Any = None,
        allow_empty: bool = False,
        ignore_cancel: bool = False,
    ) -> Any:
        if deadline is None:
            deadline = time.monotonic() + float(
                self.config.comfyui_image_total_deadline_seconds
            )
        response = self._send(
            url,
            method=method,
            deadline=deadline,
            phase=phase,
            cancel_event=cancel_event,
            json_body=json_body,
            files=files,
            ignore_cancel=ignore_cancel,
        )
        try:
            return _read_bounded_json(
                response,
                allow_empty=allow_empty,
                cancel_event=None if ignore_cancel else cancel_event,
                deadline=deadline,
                phase=phase,
            )
        finally:
            response.close()

    def _upload_reference(
        self,
        request: ImageGenRequest,
        *,
        deadline: float,
    ) -> str:
        reference = request.reference_image
        if reference is None or reference.content is None:
            raise ValueError("missing reference")
        extension = _REFERENCE_EXTENSION[reference.mime_type]
        opaque_name = f"{secrets.token_hex(16)}{extension}"
        payload = self._request_json(
            self._endpoint("/upload/image"),
            method="POST",
            deadline=deadline,
            phase="source_upload",
            cancel_event=request.cancel_event,
            files={
                "image": (opaque_name, reference.content, reference.mime_type),
                "type": (None, "input"),
                "overwrite": (None, "false"),
            },
        )
        if not isinstance(payload, dict) or payload.get("type") != "input":
            raise ValueError("invalid upload response")
        filename = _safe_filename(payload.get("name"), suffix=extension)
        subfolder = _safe_subfolder(payload.get("subfolder"))
        return f"{subfolder}/{filename}" if subfolder else filename

    def _submit_prompt(
        self,
        graph: dict[str, Any],
        request: ImageGenRequest,
        *,
        deadline: float,
    ) -> str:
        payload = self._request_json(
            self._endpoint("/prompt"),
            method="POST",
            deadline=deadline,
            phase="prompt_submission",
            cancel_event=request.cancel_event,
            json_body={"prompt": graph},
        )
        prompt_id = payload.get("prompt_id") if isinstance(payload, dict) else None
        if not isinstance(prompt_id, str) or not _SAFE_PROMPT_ID.fullmatch(prompt_id):
            raise ValueError("invalid prompt id")
        return prompt_id

    def _history_descriptor(
        self,
        history: Any,
        prompt_id: str,
    ) -> dict[str, str] | None:
        if not isinstance(history, dict):
            raise ValueError("invalid history")
        record = history.get(prompt_id)
        if record is None:
            return None
        if not isinstance(record, dict):
            raise ValueError("invalid history")
        status = record.get("status")
        if status is not None and not isinstance(status, dict):
            raise ValueError("invalid history")
        status = status or {}
        status_text = str(status.get("status_str") or "").lower()
        completed = status.get("completed") is True
        if status_text in {"error", "failed"}:
            raise ComfyUIImageEditError("history_polling")
        outputs = record.get("outputs")
        if isinstance(outputs, dict):
            node_output = outputs.get("165")
            images = node_output.get("images") if isinstance(node_output, dict) else None
            if isinstance(images, list) and any(
                isinstance(item, dict) and item.get("type") == "output"
                for item in images
            ):
                return _select_output_descriptor(history, prompt_id)
        if completed:
            raise ComfyUIImageEditError("output_descriptor_validation")
        return None

    def _poll_history(
        self,
        prompt_id: str,
        request: ImageGenRequest,
        *,
        deadline: float,
    ) -> dict[str, str]:
        history_url = self._endpoint(f"/history/{quote(prompt_id, safe='')}")
        while True:
            history = self._request_json(
                history_url,
                method="GET",
                deadline=deadline,
                phase="history_polling",
                cancel_event=request.cancel_event,
            )
            descriptor = self._history_descriptor(history, prompt_id)
            if descriptor is not None:
                return descriptor
            remaining = self._remaining(deadline, "history_polling")
            interval = min(
                float(self.config.comfyui_image_poll_interval_seconds), remaining
            )
            event = request.cancel_event
            if event is not None:
                if event.wait(interval):
                    raise ImageGenerationCancelled()
            else:
                time.sleep(interval)

    def _download_output(
        self,
        descriptor: dict[str, str],
        request: ImageGenRequest,
        *,
        deadline: float,
        width: int,
        height: int,
    ) -> bytes:
        response = self._send(
            self._endpoint("/view"),
            method="GET",
            deadline=deadline,
            phase="output_download",
            cancel_event=request.cancel_event,
            params=descriptor,
        )
        try:
            data = _stream_png(
                response,
                max_bytes=int(self.config.inline_max_bytes),
                cancel_event=request.cancel_event,
                deadline=deadline,
                phase="output_download",
            )
        finally:
            response.close()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as image:
                    if (
                        image.format != "PNG"
                        or image.mode not in _SUPPORTED_PNG_MODES
                        or image.size != (width, height)
                    ):
                        raise ValueError("invalid PNG properties")
                    image.verify()
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
        except (
            OSError,
            UnidentifiedImageError,
            ValueError,
            Image.DecompressionBombWarning,
            Image.DecompressionBombError,
        ):
            raise ValueError("invalid PNG payload") from None
        return data

    def _delete_pending_prompt_once(self, prompt_id: str) -> None:
        cleanup_deadline = time.monotonic() + min(
            float(self.config.comfyui_image_request_timeout_seconds), 1.0
        )
        try:
            self._request_json(
                self._endpoint("/queue"),
                method="POST",
                deadline=cleanup_deadline,
                phase="history_polling",
                json_body={"delete": [prompt_id]},
                allow_empty=True,
                ignore_cancel=True,
            )
        except Exception:
            return

    @staticmethod
    def _log_failure(error: ComfyUIImageEditError, source: BaseException) -> None:
        logger.bind(
            component="image_edit",
            phase=error.phase,
            error_type=type(source).__name__,
        ).warning("ComfyUI image edit phase failed")

    def generate(self, request: ImageGenRequest) -> ImageGenResult:
        """Execute one edit with pre-upload preflight and prompt-scoped cleanup."""
        prompt_id: str | None = None
        phase: ComfyUIImageEditPhase = "request_validation"
        deadline = time.monotonic() + float(
            self.config.comfyui_image_total_deadline_seconds
        )
        try:
            self._check_cancelled(request.cancel_event)
            prepared = _prepare_workflow(request, config=self.config)

            phase = "remote_schema_preflight"
            schema = self._request_json(
                self._endpoint("/object_info"),
                method="GET",
                deadline=deadline,
                phase=phase,
                cancel_event=request.cancel_event,
            )
            _validate_object_info(prepared, schema)

            phase = "source_upload"
            upload_reference = self._upload_reference(request, deadline=deadline)
            queued_graph = deepcopy(prepared.graph)
            queued_graph["114"]["inputs"]["image"] = upload_reference
            prepared = _PreparedWorkflow(
                queued_graph,
                prepared.seed,
                prepared.steps,
                prepared.sampler,
                prepared.width,
                prepared.height,
            )

            phase = "prompt_submission"
            prompt_id = self._submit_prompt(
                prepared.graph, request, deadline=deadline
            )

            phase = "history_polling"
            descriptor = self._poll_history(prompt_id, request, deadline=deadline)

            phase = "output_download"
            data = self._download_output(
                descriptor,
                request,
                deadline=deadline,
                width=prepared.width,
                height=prepared.height,
            )
            self._check_cancelled(request.cancel_event)
            return ImageGenResult(
                content=data,
                content_type="image/png",
                bytes_len=len(data),
                resolved_seed=prepared.seed,
                resolved_model=None,
                effective_params={
                    "operation": "edit",
                    "workflow_key": H3_IMAGE_EDIT_WORKFLOW_KEY,
                    "width": prepared.width,
                    "height": prepared.height,
                    "steps": prepared.steps,
                    "sampler": prepared.sampler,
                    "format": "png",
                },
            )
        except ImageGenerationCancelled:
            if prompt_id is not None:
                self._delete_pending_prompt_once(prompt_id)
            raise
        except _DeadlineExpired as exc:
            if prompt_id is not None:
                self._delete_pending_prompt_once(prompt_id)
            error = ComfyUIImageEditError(exc.phase)
            self._log_failure(error, exc)
            raise error from None
        except ComfyUIImageEditError as exc:
            self._log_failure(exc, exc)
            raise
        except Exception as exc:
            error = ComfyUIImageEditError(phase)
            self._log_failure(error, exc)
            raise error from None
