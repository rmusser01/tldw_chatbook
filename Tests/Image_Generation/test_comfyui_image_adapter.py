"""Strict, scripted-transport tests for the packaged ComfyUI image adapter."""

from __future__ import annotations

import copy
import io
import json
import re
import threading
import time
import warnings
import zlib
from dataclasses import replace
from typing import Any

import httpx
import pytest
from loguru import logger
from PIL import Image, PngImagePlugin

from Tests.Image_Generation.test_comfyui_workflow_assets import (
    EXPECTED_DIRECT_LINKS,
    EXPECTED_NODE_CLASSES,
)
from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest
from tldw_chatbook.Image_Generation.adapters import comfyui_image_adapter as adapter_module
from tldw_chatbook.Image_Generation.capabilities import ResolvedReferenceImage
from tldw_chatbook.Image_Generation.config import get_image_generation_config
from tldw_chatbook.Image_Generation.exceptions import (
    ComfyUIImageEditError,
    ImageGenerationCancelled,
)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ChunkStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    def __iter__(self):
        yield from self.chunks


class ControlledChunkStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes], on_chunk) -> None:
        self.chunks = chunks
        self.on_chunk = on_chunk

    def __iter__(self):
        for index, chunk in enumerate(self.chunks):
            self.on_chunk(index)
            yield chunk


class GuardedChunkStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.iterated = False

    def __iter__(self):
        self.iterated = True
        yield from self.chunks


class FailingChunkStream(httpx.SyncByteStream):
    def __init__(self, on_read, detail: str) -> None:
        self.on_read = on_read
        self.detail = detail

    def __iter__(self):
        self.on_read()
        raise httpx.ReadTimeout(self.detail)


class BlockingChunkStream(httpx.SyncByteStream):
    def __init__(self, first_chunk: bytes, *, on_block=None) -> None:
        self.first_chunk = first_chunk
        self.on_block = on_block
        self.second_read_started = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()
        self.finished = threading.Event()

    def __iter__(self):
        try:
            yield self.first_chunk
            self.second_read_started.set()
            if self.on_block is not None:
                self.on_block()
            if not self.release.wait(0.3):
                raise httpx.ReadTimeout("sentinel-blocking-stream-fallback")
        finally:
            self.finished.set()

    def close(self) -> None:
        self.closed.set()
        self.release.set()


class AdvancingClock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class TrackingEvent:
    def __init__(self, *, cancel_on_wait: bool = False) -> None:
        self.cancel_on_wait = cancel_on_wait
        self.set_state = False
        self.waits: list[float] = []

    def is_set(self) -> bool:
        return self.set_state

    def wait(self, interval: float) -> bool:
        self.waits.append(interval)
        if self.cancel_on_wait:
            self.set_state = True
        return self.set_state


def _png(width: int = 5, height: int = 4, *, mode: str = "RGB") -> bytes:
    image = Image.new(mode, (width, height), color=0)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _png_with_dimensions(width: int, height: int) -> bytes:
    encoded = bytearray(_png())
    encoded[16:20] = width.to_bytes(4, "big")
    encoded[20:24] = height.to_bytes(4, "big")
    encoded[29:33] = (zlib.crc32(encoded[12:29]) & 0xFFFFFFFF).to_bytes(4, "big")
    return bytes(encoded)


def _reference(*, mime: str = "image/png", width: int = 5, height: int = 4) -> ResolvedReferenceImage:
    data = _png(width, height)
    return ResolvedReferenceImage(
        file_id="runtime-only-id",
        filename="private-source-name.png",
        mime_type=mime,
        width=width,
        height=height,
        bytes_len=len(data),
        content=data,
        temp_path=None,
    )


def _request(**updates: Any) -> ImageGenRequest:
    values: dict[str, Any] = {
        "backend": "comfyui",
        "prompt": "neutral synthetic edit",
        "negative_prompt": None,
        "width": None,
        "height": None,
        "steps": None,
        "cfg_scale": None,
        "seed": None,
        "sampler": None,
        "model": None,
        "format": "png",
        "extra_params": {},
        "reference_image": _reference(),
        "cancel_event": None,
    }
    values.update(updates)
    return ImageGenRequest(**values)


def _config(**updates: Any):
    values = {
        "comfyui_image_base_url": "http://127.0.0.1:8188",
        "comfyui_image_request_timeout_seconds": 6.0,
        "comfyui_image_connect_timeout_seconds": 2.0,
        "comfyui_image_poll_interval_seconds": 0.25,
        "comfyui_image_total_deadline_seconds": 20.0,
        "comfyui_image_default_seed": None,
        "comfyui_image_default_steps": None,
        "comfyui_image_default_sampler": None,
        "inline_max_bytes": 1_000_000,
    }
    values.update(updates)
    return replace(get_image_generation_config(), **values)


_INPUT_TYPES = {
    "LoadImage": {"image": "IMAGE_UPLOAD"},
    "VAELoader": {"vae_name": "STRING"},
    "VAEDecode": {"samples": "LATENT", "vae": "VAE"},
    "KSamplerSelect": {"sampler_name": "STRING"},
    "BasicScheduler": {
        "denoise": "FLOAT",
        "model": "MODEL",
        "scheduler": "STRING",
        "steps": "INT",
    },
    "SamplerCustomAdvanced": {
        "guider": "GUIDER",
        "latent_image": "LATENT",
        "noise": "NOISE",
        "sampler": "SAMPLER",
        "sigmas": "SIGMAS",
    },
    "BasicGuider": {"conditioning": "CONDITIONING", "model": "MODEL"},
    "UNETLoader": {"unet_name": "STRING", "weight_dtype": "STRING"},
    "CLIPLoader": {"clip_name": "STRING", "device": "STRING", "type": "STRING"},
    "RandomNoise": {"noise_seed": "INT"},
    "MiniMaxH3ImageToVideo": {
        "clip": "CLIP",
        "first_frame": "IMAGE",
        "height": "INT",
        "length": "INT",
        "prompt": "STRING",
        "vae": "VAE",
        "width": "INT",
    },
    "PrimitiveInt": {"value": "INT"},
    "GetImageSize": {"image": "IMAGE"},
    "ImageScaleToTotalPixels": {
        "image": "IMAGE",
        "megapixels": "FLOAT",
        "resolution_steps": "INT",
        "upscale_method": "STRING",
    },
    "ImageFromBatch": {"batch_index": "INT", "image": "IMAGE", "length": "INT"},
    "ResizeImageMaskNode": {
        "input": "IMAGE",
        "resize_type": "STRING",
        "resize_type.crop": "STRING",
        "resize_type.height": "INT",
        "resize_type.width": "INT",
        "scale_method": "STRING",
    },
    "SaveImage": {"filename_prefix": "STRING", "images": "IMAGE"},
}

_OUTPUT_TYPES = {
    "LoadImage": ["IMAGE", "MASK"],
    "VAELoader": ["VAE"],
    "VAEDecode": ["IMAGE"],
    "KSamplerSelect": ["SAMPLER"],
    "BasicScheduler": ["SIGMAS"],
    "SamplerCustomAdvanced": ["LATENT", "LATENT"],
    "BasicGuider": ["GUIDER"],
    "UNETLoader": ["MODEL"],
    "CLIPLoader": ["CLIP"],
    "RandomNoise": ["NOISE"],
    "MiniMaxH3ImageToVideo": ["CONDITIONING", "LATENT"],
    "PrimitiveInt": ["INT"],
    "GetImageSize": ["INT", "INT"],
    "ImageScaleToTotalPixels": ["IMAGE"],
    "ImageFromBatch": ["IMAGE"],
    "ResizeImageMaskNode": ["IMAGE", "MASK"],
    "SaveImage": [],
}


def _input_spec(class_type: str, input_name: str) -> list[Any]:
    expected_type = _INPUT_TYPES[class_type][input_name]
    if expected_type == "IMAGE_UPLOAD":
        return [["existing-server-file.png"], {"image_upload": True}]
    if expected_type == "BOOLEAN":
        return ["BOOLEAN"]
    if expected_type == "INT":
        return ["INT", {"min": 0, "max": 2**64 - 1}]
    if expected_type == "FLOAT":
        return ["FLOAT"]
    return [expected_type]


def _object_info(
    graph: dict[str, Any] | None = None,
    *,
    resize_v3: bool = True,
) -> dict[str, Any]:
    graph = graph or adapter_module._load_packaged_workflow()
    result: dict[str, Any] = {}
    for node in graph.values():
        class_type = node["class_type"]
        if class_type in result:
            continue
        required = {
            name: _input_spec(class_type, name) for name in node["inputs"]
        }
        result[class_type] = {
            "input": {"required": required},
            "input_order": {"required": list(required)},
            "output": list(_OUTPUT_TYPES[class_type]),
            "output_node": class_type == "SaveImage",
            "name": class_type,
            "display_name": class_type,
            "category": "test/real-object-info-shape",
        }

    choice_inputs = {
        ("VAELoader", "vae_name"),
        ("UNETLoader", "unet_name"),
        ("UNETLoader", "weight_dtype"),
        ("CLIPLoader", "clip_name"),
        ("CLIPLoader", "device"),
        ("CLIPLoader", "type"),
        ("KSamplerSelect", "sampler_name"),
        ("BasicScheduler", "scheduler"),
        ("ImageScaleToTotalPixels", "upscale_method"),
        ("ResizeImageMaskNode", "resize_type"),
        ("ResizeImageMaskNode", "resize_type.crop"),
        ("ResizeImageMaskNode", "scale_method"),
    }
    for node in graph.values():
        for class_type, input_name in choice_inputs:
            if node["class_type"] == class_type and input_name in node["inputs"]:
                result[class_type]["input"]["required"][input_name] = [
                    [node["inputs"][input_name]]
                ]
    result["SaveImage"]["input"]["required"]["images"] = ["IMAGE"]
    result["SaveImage"]["input"]["required"]["filename_prefix"] = ["STRING"]
    if resize_v3:
        result["ResizeImageMaskNode"]["input"] = {
            "required": {
                "input": [
                    "COMFY_MATCHTYPE_V3",
                    {
                        "template": {
                            "template_id": "input_type",
                            "allowed_types": "IMAGE,MASK",
                        }
                    },
                ],
                "resize_type": [
                    "COMFY_DYNAMICCOMBO_V3",
                    {
                        "options": [
                            {
                                "key": "scale dimensions",
                                "inputs": {
                                    "required": {
                                        "width": ["INT", {"min": 0, "max": 16384}],
                                        "height": ["INT", {"min": 0, "max": 16384}],
                                        "crop": [
                                            "COMBO",
                                            {
                                                "options": ["disabled", "center"],
                                                "default": "center",
                                            },
                                        ],
                                    }
                                },
                            },
                            {
                                "key": "scale by",
                                "inputs": {
                                    "required": {
                                        "multiplier": [
                                            "FLOAT",
                                            {"min": 0.01, "max": 8.0},
                                        ]
                                    }
                                },
                            },
                        ]
                    },
                ],
                "scale_method": [
                    "COMBO",
                    {
                        "options": [
                            "nearest-exact",
                            "bilinear",
                            "area",
                            "bicubic",
                            "lanczos",
                        ],
                        "default": "area",
                    },
                ],
            }
        }
        result["ResizeImageMaskNode"]["input_order"] = {
            "required": ["input", "resize_type", "scale_method"]
        }
        result["ResizeImageMaskNode"]["output"] = ["COMFY_MATCHTYPE_V3"]
        result["ResizeImageMaskNode"]["output_matchtypes"] = ["input_type"]
    return result


def _json_response(payload: Any, *, status: int = 200, headers: dict[str, str] | None = None):
    body = json.dumps(payload, separators=(",", ":")).encode()
    return httpx.Response(status, content=body, headers=headers)


class SuccessfulScript:
    """One deterministic ComfyUI exchange over a real httpx MockTransport."""

    def __init__(self, *, previews: int = 0, output: bytes | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.requests: list[httpx.Request] = []
        self.previews = previews
        self.output = output or _png()
        self.queued_graph: dict[str, Any] | None = None
        self.upload_body = b""
        self.history_calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, request.url.path))
        self.requests.append(request)
        path = request.url.path
        if path == "/object_info":
            return _json_response(_object_info())
        if path == "/upload/image":
            self.upload_body = request.read()
            suffix = ".jpg" if b".jpg" in self.upload_body else ".png"
            return _json_response(
                {"name": f"opaque-upload{suffix}", "subfolder": "", "type": "input"}
            )
        if path == "/prompt":
            self.queued_graph = json.loads(request.read())["prompt"]
            return _json_response({"prompt_id": "opaque-prompt-id"})
        if path == "/history/opaque-prompt-id":
            self.history_calls += 1
            if self.history_calls <= self.previews:
                return _json_response(
                    {
                        "opaque-prompt-id": {
                            "outputs": {
                                "165": {
                                    "images": [
                                        {
                                            "filename": "preview.png",
                                            "subfolder": "temp",
                                            "type": "temp",
                                        }
                                    ]
                                }
                            }
                        }
                    }
                )
            return _json_response(
                {
                    "opaque-prompt-id": {
                        "status": {"completed": True, "status_str": "success"},
                        "outputs": {
                            "165": {
                                "images": [
                                    {
                                        "filename": "edited.png",
                                        "subfolder": "safe/nested",
                                        "type": "output",
                                    }
                                ]
                            }
                        },
                    }
                }
            )
        if path == "/view":
            assert dict(request.url.params) == {
                "filename": "edited.png",
                "subfolder": "safe/nested",
                "type": "output",
            }
            return httpx.Response(
                200,
                content=self.output,
                headers={"content-type": "image/png", "content-length": str(len(self.output))},
            )
        if path == "/queue":
            return _json_response({})
        raise AssertionError(f"unexpected scripted path {path}")


class TrackingResponseStream(httpx.SyncByteStream):
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.closed = threading.Event()

    def __iter__(self):
        yield self.body

    def close(self) -> None:
        self.closed.set()


class BlockingResponseTransport(httpx.BaseTransport):
    def __init__(
        self,
        base: SuccessfulScript,
        path: str,
        *,
        on_block=None,
    ) -> None:
        self.base = base
        self.path = path
        self.on_block = on_block
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()
        self.late_response_created = threading.Event()
        self.late_stream = TrackingResponseStream(b"{}")

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != self.path:
            return self.base(request)
        self.base.calls.append((request.method, request.url.path))
        self.base.requests.append(request)
        self.started.set()
        if self.on_block is not None:
            self.on_block()
        self.release.wait(0.3)
        self.late_response_created.set()
        headers = {"content-type": "image/png"} if self.path == "/view" else {}
        return httpx.Response(200, stream=self.late_stream, headers=headers)

    def close(self) -> None:
        self.closed.set()
        self.release.set()


class BlockingRequestStream(httpx.SyncByteStream):
    def __init__(self, inner, *, on_block=None) -> None:
        self.inner = inner
        self.on_block = on_block
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()
        self.finished = threading.Event()

    def __iter__(self):
        iterator = iter(self.inner)
        try:
            yield next(iterator)
            self.started.set()
            if self.on_block is not None:
                self.on_block()
            self.release.wait(0.3)
            if not self.closed.is_set():
                yield from iterator
        finally:
            self.finished.set()

    def close(self) -> None:
        self.closed.set()
        self.release.set()
        self.inner.close()


class ScriptTransport(httpx.BaseTransport):
    def __init__(self, script) -> None:
        self.script = script

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self.script(request)


def _make_adapter(script, *, config=None):
    return adapter_module.ComfyUIImageAdapter(
        config=config or _config(),
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(script)),
    )


def _make_adapter_with_transport(transport, *, config=None):
    return adapter_module.ComfyUIImageAdapter(
        config=config or _config(),
        client_factory=lambda: httpx.Client(transport=transport),
    )


def _assert_phase(exc: pytest.ExceptionInfo[ComfyUIImageEditError], phase: str) -> None:
    assert exc.value.phase == phase
    assert "neutral synthetic edit" not in str(exc.value)
    assert "private-source-name" not in str(exc.value)


def test_prepare_uses_exact_topology_direct_links_and_id_targets() -> None:
    packaged = adapter_module._load_packaged_workflow()
    before = copy.deepcopy(packaged)

    prepared = adapter_module._prepare_workflow(
        _request(seed=7, steps=12, sampler="res_multistep"),
        config=_config(),
        graph=packaged,
        upload_name="opaque.png",
    )

    assert {node_id: node["class_type"] for node_id, node in prepared.graph.items()} == EXPECTED_NODE_CLASSES
    links = {
        f"{node_id}.{name}": tuple(value)
        for node_id, node in prepared.graph.items()
        for name, value in node["inputs"].items()
        if isinstance(value, list)
    }
    assert links == EXPECTED_DIRECT_LINKS
    assert prepared.graph["114"]["inputs"]["image"] == "opaque.png"
    assert prepared.graph["125"]["inputs"]["sampler_name"] == "res_multistep"
    assert prepared.graph["126"]["inputs"]["steps"] == 12
    assert prepared.graph["131"]["inputs"]["noise_seed"] == 7
    assert prepared.graph["133"]["inputs"]["prompt"] == "neutral synthetic edit"
    assert prepared.graph["165"]["inputs"]["filename_prefix"] == "h3_edit"
    assert packaged == before


@pytest.mark.parametrize(
    "mutate",
    [
        lambda graph: graph.pop("114"),
        lambda graph: graph.__setitem__("114", {**graph["114"], "class_type": "SaveImage"}),
        lambda graph: graph.__setitem__("999", copy.deepcopy(graph["114"])),
        lambda graph: graph["165"]["inputs"].__setitem__("images", ["114", 0]),
        lambda graph: graph.__setitem__(
            "999",
            {"class_type": "LoadImage", "inputs": {"image": "decoy.png"}, "_meta": {"title": "Load Image"}},
        ),
    ],
    ids=["missing", "wrong-class", "unexpected-output", "link-drift", "title-decoy"],
)
def test_prepare_rejects_malformed_topology_without_mutating_input(mutate) -> None:
    graph = adapter_module._load_packaged_workflow()
    mutate(graph)
    before = copy.deepcopy(graph)

    with pytest.raises(ComfyUIImageEditError) as exc:
        adapter_module._prepare_workflow(_request(), config=_config(), graph=graph)

    _assert_phase(exc, "packaged_workflow_validation")
    assert graph == before


def test_prepare_precedence_unset_literals_and_seed_minus_one_resolves_once(monkeypatch) -> None:
    calls: list[int] = []

    def randbits(bits: int) -> int:
        calls.append(bits)
        return 123456

    monkeypatch.setattr(adapter_module.secrets, "randbits", randbits)
    graph = adapter_module._load_packaged_workflow()
    packaged_seed = graph["131"]["inputs"]["noise_seed"]
    retained = adapter_module._prepare_workflow(_request(), config=_config(), graph=graph)
    defaults = adapter_module._prepare_workflow(
        _request(),
        config=_config(
            comfyui_image_default_seed=-1,
            comfyui_image_default_steps=11,
            comfyui_image_default_sampler="res_multistep",
        ),
        graph=graph,
    )
    explicit = adapter_module._prepare_workflow(
        _request(seed=9, steps=13, sampler="res_multistep"),
        config=_config(
            comfyui_image_default_seed=-1,
            comfyui_image_default_steps=11,
            comfyui_image_default_sampler="ignored-default",
        ),
        graph=graph,
    )

    assert retained.seed == packaged_seed
    assert retained.steps == graph["126"]["inputs"]["steps"]
    assert retained.sampler == graph["125"]["inputs"]["sampler_name"]
    assert defaults.seed == 123456
    assert defaults.graph["131"]["inputs"]["noise_seed"] == 123456
    assert explicit.seed == 9
    assert explicit.steps == 13
    assert explicit.sampler == "res_multistep"
    assert calls == [64]


def test_generate_resolves_seed_minus_one_once_for_queued_graph_and_result(monkeypatch) -> None:
    calls: list[int] = []

    def randbits(bits: int) -> int:
        calls.append(bits)
        return 987654

    monkeypatch.setattr(adapter_module.secrets, "randbits", randbits)
    script = SuccessfulScript()

    result = _make_adapter(script).generate(_request(seed=-1))

    assert calls == [64]
    assert result.resolved_seed == 987654
    assert script.queued_graph is not None
    assert script.queued_graph["131"]["inputs"]["noise_seed"] == 987654


@pytest.mark.parametrize(
    "updates",
    [
        {"negative_prompt": "forbidden"},
        {"cfg_scale": 1.0},
        {"model": "forbidden-model"},
        {"format": "jpeg"},
        {"width": 5},
        {"height": 4},
        {"extra_params": {"unknown": True}},
    ],
)
def test_unsupported_controls_fail_before_preflight_or_upload(updates) -> None:
    calls: list[str] = []

    def forbidden(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        raise AssertionError("transport must not be reached")

    with pytest.raises(ComfyUIImageEditError) as exc:
        _make_adapter(forbidden).generate(_request(**updates))

    _assert_phase(exc, "request_validation")
    assert calls == []


def test_object_info_preflight_precedes_upload_and_validates_choices() -> None:
    graph = adapter_module._load_packaged_workflow()
    schema = _object_info(graph)
    sampler = graph["125"]["inputs"]["sampler_name"]
    schema["KSamplerSelect"]["input"]["required"]["sampler_name"] = [["not-the-sampler"]]
    calls: list[str] = []

    def script(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        assert request.url.path == "/object_info"
        return _json_response(schema)

    with pytest.raises(ComfyUIImageEditError) as exc:
        _make_adapter(script).generate(_request(sampler=sampler))

    _assert_phase(exc, "remote_schema_preflight")
    assert calls == ["/object_info"]


def test_object_info_accepts_real_load_image_upload_schema_without_placeholder_choice() -> None:
    prepared = adapter_module._prepare_workflow(_request(), config=_config())
    schema = _object_info(prepared.graph)

    assert prepared.graph["114"]["inputs"]["image"] not in schema["LoadImage"][
        "input"
    ]["required"]["image"][0]
    adapter_module._validate_object_info(prepared, schema)


def test_object_info_accepts_legacy_resize_schema_for_supported_servers() -> None:
    prepared = adapter_module._prepare_workflow(_request(), config=_config())

    adapter_module._validate_object_info(
        prepared,
        _object_info(prepared.graph, resize_v3=False),
    )


@pytest.mark.parametrize("required_kind", ["top-level", "selected-dynamic"])
def test_object_info_rejects_unprovided_server_required_input_before_upload(
    required_kind: str,
) -> None:
    schema = _object_info()
    if required_kind == "top-level":
        schema["BasicScheduler"]["input"]["required"]["future_required"] = ["INT"]
    else:
        schema["ResizeImageMaskNode"]["input"]["required"]["resize_type"][1][
            "options"
        ][0]["inputs"]["required"]["future_required"] = ["INT"]
    calls: list[str] = []

    def script(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return _json_response(schema)

    with pytest.raises(ComfyUIImageEditError) as exc:
        _make_adapter(script).generate(_request())

    _assert_phase(exc, "remote_schema_preflight")
    assert calls == ["/object_info"]


def test_object_info_ignores_optional_hidden_and_unselected_dynamic_inputs() -> None:
    prepared = adapter_module._prepare_workflow(_request(), config=_config())
    schema = _object_info(prepared.graph)
    schema["BasicScheduler"]["input"].setdefault("optional", {})[
        "future_optional"
    ] = ["INT"]
    schema["BasicScheduler"]["input"]["hidden"] = {
        "internal_value": "UNSUPPORTED_HIDDEN"
    }
    schema["ResizeImageMaskNode"]["input"]["required"]["resize_type"][1][
        "options"
    ][1]["inputs"]["required"]["unselected_required"] = ["UNSUPPORTED"]

    adapter_module._validate_object_info(prepared, schema)


def _add_future_dynamic_input(
    schema: dict[str, Any],
    *,
    group: str = "optional",
) -> None:
    schema["BasicScheduler"]["input"].setdefault(group, {})["future_mode"] = [
        "COMFY_DYNAMICCOMBO_V3",
        {
            "options": [
                {
                    "key": "simple",
                    "inputs": {"required": {}},
                },
                {
                    "key": "advanced",
                    "inputs": {
                        "required": {
                            "amount": ["INT", {"min": 1, "max": 10}],
                        },
                        "optional": {
                            "label": ["STRING"],
                        },
                    },
                },
            ]
        },
    ]


def test_object_info_ignores_absent_optional_dynamic_selector() -> None:
    prepared = adapter_module._prepare_workflow(_request(), config=_config())
    schema = _object_info(prepared.graph)
    _add_future_dynamic_input(schema)

    adapter_module._validate_object_info(prepared, schema)


def test_object_info_present_optional_dynamic_requires_selected_branch_fields() -> None:
    prepared = adapter_module._prepare_workflow(_request(), config=_config())
    schema = _object_info(prepared.graph)
    _add_future_dynamic_input(schema)
    prepared.graph["126"]["inputs"]["future_mode"] = "advanced"

    with pytest.raises(ComfyUIImageEditError) as exc:
        adapter_module._validate_object_info(prepared, schema)

    _assert_phase(exc, "remote_schema_preflight")


def test_object_info_accepts_complete_present_optional_dynamic_branch() -> None:
    prepared = adapter_module._prepare_workflow(_request(), config=_config())
    schema = _object_info(prepared.graph)
    _add_future_dynamic_input(schema)
    prepared.graph["126"]["inputs"].update(
        {
            "future_mode": "advanced",
            "future_mode.amount": 3,
        }
    )

    adapter_module._validate_object_info(prepared, schema)


def test_object_info_still_rejects_absent_required_dynamic_selector() -> None:
    prepared = adapter_module._prepare_workflow(_request(), config=_config())
    schema = _object_info(prepared.graph)
    _add_future_dynamic_input(schema, group="required")

    with pytest.raises(ComfyUIImageEditError) as exc:
        adapter_module._validate_object_info(prepared, schema)

    _assert_phase(exc, "remote_schema_preflight")


@pytest.mark.parametrize(
    "damage",
    [
        "match-union",
        "unsupported-match-type",
        "dynamic-option",
        "dynamic-field",
        "unsupported-dynamic-field",
        "output-match-link",
    ],
)
def test_object_info_v3_resize_schema_fails_closed_before_upload(damage: str) -> None:
    schema = _object_info()
    resize = schema["ResizeImageMaskNode"]
    if damage == "match-union":
        resize["input"]["required"]["input"][1]["template"]["allowed_types"] = "MASK"
    elif damage == "unsupported-match-type":
        resize["input"]["required"]["input"][1]["template"]["allowed_types"] = (
            "IMAGE,UNSUPPORTED"
        )
    elif damage == "dynamic-option":
        resize["input"]["required"]["resize_type"][1]["options"][0]["key"] = (
            "different option"
        )
    elif damage == "dynamic-field":
        resize["input"]["required"]["resize_type"][1]["options"][0]["inputs"][
            "required"
        ].pop("height")
    elif damage == "unsupported-dynamic-field":
        resize["input"]["required"]["resize_type"][1]["options"][0]["inputs"][
            "required"
        ]["width"] = ["UNSUPPORTED"]
    else:
        resize["output_matchtypes"] = ["different_template"]
    calls: list[str] = []

    def script(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return _json_response(schema)

    with pytest.raises(ComfyUIImageEditError) as exc:
        _make_adapter(script).generate(_request())

    _assert_phase(exc, "remote_schema_preflight")
    assert calls == ["/object_info"]


@pytest.mark.parametrize(
    "upload_schema",
    [
        [["existing.png"]],
        [["existing.png"], {}],
        [["existing.png"], {"image_upload": False}],
        ["IMAGE", {"image_upload": True}],
    ],
    ids=["missing-metadata", "missing-flag", "false-flag", "not-a-choice-list"],
)
def test_object_info_rejects_malformed_load_image_upload_schema_before_upload(
    upload_schema: list[Any],
) -> None:
    schema = _object_info()
    schema["LoadImage"]["input"]["required"]["image"] = upload_schema
    calls: list[str] = []

    def script(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return _json_response(schema)

    with pytest.raises(ComfyUIImageEditError) as exc:
        _make_adapter(script).generate(_request())

    _assert_phase(exc, "remote_schema_preflight")
    assert calls == ["/object_info"]


@pytest.mark.parametrize(
    "damage",
    ["missing-output", "output-index", "source-type", "target-type"],
)
def test_object_info_rejects_direct_link_output_schema_mismatches_before_upload(
    damage: str,
) -> None:
    schema = _object_info()
    if damage == "missing-output":
        schema["UNETLoader"].pop("output")
    elif damage == "output-index":
        schema["MiniMaxH3ImageToVideo"]["output"] = ["CONDITIONING"]
    elif damage == "source-type":
        schema["VAELoader"]["output"][0] = "IMAGE"
    else:
        schema["VAEDecode"]["input"]["required"]["samples"] = ["IMAGE"]
    calls: list[str] = []

    def script(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return _json_response(schema)

    with pytest.raises(ComfyUIImageEditError) as exc:
        _make_adapter(script).generate(_request())

    _assert_phase(exc, "remote_schema_preflight")
    assert calls == ["/object_info"]


@pytest.mark.parametrize("damage", ["class", "input", "loader", "save-image"])
def test_object_info_rejects_missing_classes_inputs_loader_choices_and_png_output(damage) -> None:
    graph = adapter_module._load_packaged_workflow()
    schema = _object_info(graph)
    if damage == "class":
        schema.pop("VAELoader")
    elif damage == "input":
        schema["BasicScheduler"]["input"]["required"].pop("steps")
    elif damage == "loader":
        schema["UNETLoader"]["input"]["required"]["unet_name"] = [["other"]]
    else:
        schema["SaveImage"]["output_node"] = False
    calls: list[str] = []

    def script(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return _json_response(schema)

    with pytest.raises(ComfyUIImageEditError) as exc:
        _make_adapter(script).generate(_request())

    _assert_phase(exc, "remote_schema_preflight")
    assert calls == ["/object_info"]


def test_bounded_json_rejects_declared_and_streamed_overflow_before_loads(monkeypatch) -> None:
    loads_calls: list[object] = []
    monkeypatch.setattr(adapter_module.json, "loads", lambda value: loads_calls.append(value))
    declared = httpx.Response(
        200,
        headers={"content-length": str(adapter_module.COMFYUI_MAX_JSON_BYTES + 1)},
        content=b"{}",
    )
    streamed = httpx.Response(
        200,
        stream=ChunkStream([b"a" * (adapter_module.COMFYUI_MAX_JSON_BYTES // 2 + 1)] * 2),
    )

    with pytest.raises(ValueError):
        adapter_module._read_bounded_json(declared)
    with pytest.raises(ValueError):
        adapter_module._read_bounded_json(streamed)

    assert loads_calls == []


@pytest.mark.parametrize("encoding", ["gzip", "br"])
def test_json_rejects_non_identity_content_encoding_before_iteration_or_parse(
    monkeypatch,
    encoding: str,
) -> None:
    stream = GuardedChunkStream([b"{}"])
    loads_calls: list[object] = []
    monkeypatch.setattr(adapter_module.json, "loads", lambda value: loads_calls.append(value))
    response = httpx.Response(
        200,
        stream=stream,
        headers={"content-encoding": encoding},
    )

    with pytest.raises(ValueError):
        adapter_module._read_bounded_json(response)

    assert stream.iterated is False
    assert loads_calls == []


def test_json_oversized_single_chunk_never_extends_bounded_buffer_or_parses(
    monkeypatch,
) -> None:
    max_buffer_len = 0
    loads_calls: list[object] = []
    real_bytearray = bytearray

    class TrackingBytearray(real_bytearray):
        def extend(self, value) -> None:
            nonlocal max_buffer_len
            super().extend(value)
            max_buffer_len = max(max_buffer_len, len(self))

    monkeypatch.setattr(adapter_module, "bytearray", TrackingBytearray, raising=False)
    monkeypatch.setattr(adapter_module.json, "loads", lambda value: loads_calls.append(value))
    response = httpx.Response(
        200,
        stream=ChunkStream([b"x" * (adapter_module.COMFYUI_MAX_JSON_BYTES + 1)]),
    )

    with pytest.raises(ValueError):
        adapter_module._read_bounded_json(response)

    assert max_buffer_len == 0
    assert loads_calls == []


def test_json_drip_stream_hits_absolute_deadline_before_prompt_without_delete(
    monkeypatch,
) -> None:
    clock = AdvancingClock()
    monkeypatch.setattr(adapter_module.time, "monotonic", clock.monotonic)
    body = json.dumps(_object_info(), separators=(",", ":")).encode()
    size = len(body) // 4
    chunks = [body[:size], body[size : size * 2], body[size * 2 : size * 3], body[size * 3 :]]
    yielded: list[int] = []
    calls: list[str] = []

    def advance(index: int) -> None:
        yielded.append(index)
        clock.advance(0.4)

    def script(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        assert request.url.path == "/object_info"
        return httpx.Response(200, stream=ControlledChunkStream(chunks, advance))

    with pytest.raises(ComfyUIImageEditError) as exc:
        _make_adapter(
            script,
            config=_config(
                comfyui_image_request_timeout_seconds=0.5,
                comfyui_image_total_deadline_seconds=1.0,
            ),
        ).generate(_request())

    _assert_phase(exc, "remote_schema_preflight")
    assert yielded == [0, 1, 2]
    assert calls == ["/object_info"]
    assert "/queue" not in calls


def test_json_stream_cancellation_after_prompt_stops_and_deletes_once() -> None:
    event = threading.Event()
    yielded: list[int] = []
    calls: list[str] = []
    history = {
        "opaque-prompt-id": {
            "status": {"completed": False, "status_str": "running"},
            "private_detail": "sentinel-private-stream-detail",
        }
    }
    body = json.dumps(history, separators=(",", ":")).encode()
    size = len(body) // 4
    chunks = [body[:size], body[size : size * 2], body[size * 2 : size * 3], body[size * 3 :]]

    def cancel(index: int) -> None:
        yielded.append(index)
        if index == 1:
            event.set()

    def script(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/object_info":
            return _json_response(_object_info())
        if request.url.path == "/upload/image":
            return _json_response({"name": "opaque.png", "subfolder": "", "type": "input"})
        if request.url.path == "/prompt":
            return _json_response({"prompt_id": "opaque-prompt-id"})
        if request.url.path == "/history/opaque-prompt-id":
            return httpx.Response(200, stream=ControlledChunkStream(chunks, cancel))
        if request.url.path == "/queue":
            return _json_response({})
        raise AssertionError("unexpected request")

    with pytest.raises(ImageGenerationCancelled) as exc:
        _make_adapter(script).generate(_request(cancel_event=event))

    assert "sentinel-private-stream-detail" not in str(exc.value)
    assert yielded == [0, 1]
    assert calls.count("/queue") == 1


def test_success_uses_opaque_mime_extension_exact_origin_node165_and_effective_params(monkeypatch) -> None:
    script = SuccessfulScript(previews=2)
    checks: list[tuple[str, frozenset[str]]] = []

    def allow(url: str, *, trusted_origins=frozenset()) -> None:
        checks.append((url, trusted_origins))

    monkeypatch.setattr(adapter_module, "check_url_or_raise", allow)
    result = _make_adapter(script).generate(
        _request(seed=8, steps=14, sampler="res_multistep")
    )

    assert result.content == script.output
    assert result.content_type == "image/png"
    assert result.bytes_len == len(script.output)
    assert result.resolved_seed == 8
    assert result.resolved_model is None
    assert result.effective_params == {
        "operation": "edit",
        "workflow_key": "minimax_h3_image_edit",
        "width": 5,
        "height": 4,
        "steps": 14,
        "sampler": "res_multistep",
        "format": "png",
    }
    assert script.queued_graph is not None
    assert script.queued_graph["114"]["inputs"]["image"] == "opaque-upload.png"
    assert script.queued_graph["131"]["inputs"]["noise_seed"] == result.resolved_seed
    assert script.queued_graph["126"]["inputs"]["steps"] == result.effective_params["steps"]
    assert script.queued_graph["125"]["inputs"]["sampler_name"] == result.effective_params["sampler"]
    assert b"private-source-name" not in script.upload_body
    assert re.search(rb'filename="[0-9a-f]{32}\.png"', script.upload_body)
    assert all(url.startswith("http://127.0.0.1:8188/") for url, _ in checks)
    assert all(trusted == frozenset({"127.0.0.1"}) for _, trusted in checks)
    assert script.history_calls == 3


def test_upload_extension_comes_only_from_validated_mime() -> None:
    script = SuccessfulScript()
    reference = _reference(mime="image/jpeg")
    reference = replace(reference, content=_png(), bytes_len=len(_png()))

    _make_adapter(script).generate(_request(reference_image=reference))

    assert b".jpg" in script.upload_body
    assert b"private-source-name" not in script.upload_body


def test_redirects_are_not_followed() -> None:
    calls: list[str] = []

    def script(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1:8288/object_info"})

    with pytest.raises(ComfyUIImageEditError) as exc:
        _make_adapter(script).generate(_request())

    _assert_phase(exc, "remote_schema_preflight")
    assert calls == ["http://127.0.0.1:8188/object_info"]


def test_cross_origin_private_request_is_rejected_before_transport(monkeypatch) -> None:
    calls: list[str] = []
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: calls.append(str(request.url)))
    )
    image_adapter = adapter_module.ComfyUIImageAdapter(config=_config())

    try:
        monkeypatch.setattr(adapter_module, "same_origin", lambda _a, _b: False)
        with pytest.raises(ValueError):
            image_adapter._request_json(
                client,
                "http://127.0.0.1:8288/object_info",
                method="GET",
            )
    finally:
        client.close()

    assert calls == []


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
def test_locally_owned_httpx_client_closes_for_every_outcome(
    outcome: str,
) -> None:
    instances: list[httpx.Client] = []
    base = SuccessfulScript()

    class TrackingClient(httpx.Client):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.close_calls = 0
            instances.append(self)

        def close(self) -> None:
            self.close_calls += 1
            super().close()

    def script(request: httpx.Request) -> httpx.Response:
        if outcome == "error" and request.url.path == "/object_info":
            return _json_response({}, status=500)
        return base(request)

    def client_factory() -> httpx.Client:
        return TrackingClient(transport=httpx.MockTransport(script))

    event = threading.Event()
    if outcome == "cancel":
        event.set()
    image_adapter = adapter_module.ComfyUIImageAdapter(
        config=_config(),
        client_factory=client_factory,
    )

    if outcome == "success":
        image_adapter.generate(_request(cancel_event=event))
    elif outcome == "cancel":
        with pytest.raises(ImageGenerationCancelled):
            image_adapter.generate(_request(cancel_event=event))
    else:
        with pytest.raises(ComfyUIImageEditError):
            image_adapter.generate(_request(cancel_event=event))

    assert len(instances) == 1
    assert instances[0].close_calls == 1


def test_caller_owned_httpx_client_injection_is_not_supported() -> None:
    client = httpx.Client(transport=httpx.MockTransport(SuccessfulScript()))
    try:
        with pytest.raises(TypeError):
            adapter_module.ComfyUIImageAdapter(config=_config(), client=client)
    finally:
        client.close()


def test_two_generations_use_distinct_usable_factory_clients() -> None:
    clients: list[httpx.Client] = []

    def client_factory() -> httpx.Client:
        client = httpx.Client(transport=httpx.MockTransport(SuccessfulScript()))
        clients.append(client)
        return client

    image_adapter = adapter_module.ComfyUIImageAdapter(
        config=_config(),
        client_factory=client_factory,
    )

    first = image_adapter.generate(_request())
    second = image_adapter.generate(_request())

    assert first.content_type == second.content_type == "image/png"
    assert len(clients) == 2
    assert clients[0] is not clients[1]
    assert all(client.is_closed for client in clients)


def test_multipart_request_stream_is_closed_after_send(monkeypatch) -> None:
    real_client = httpx.Client
    wrapped_streams: list[object] = []

    class ClosingStream(httpx.SyncByteStream):
        def __init__(self, inner) -> None:
            self.inner = inner
            self.closed = False

        def __iter__(self):
            yield from self.inner

        def close(self) -> None:
            self.closed = True
            self.inner.close()

    class TrackingClient(real_client):
        def build_request(self, *args, **kwargs) -> httpx.Request:
            request = super().build_request(*args, **kwargs)
            if request.url.path == "/upload/image":
                wrapped = ClosingStream(request.stream)
                request.stream = wrapped
                wrapped_streams.append(wrapped)
            return request

    monkeypatch.setattr(adapter_module.httpx, "Client", TrackingClient)

    _make_adapter(SuccessfulScript()).generate(_request())

    assert len(wrapped_streams) == 1
    assert wrapped_streams[0].closed is True


@pytest.mark.parametrize(
    ("blocked_path", "phase", "queue_deletes"),
    [
        ("/object_info", "remote_schema_preflight", 0),
        ("/history/opaque-prompt-id", "history_polling", 1),
    ],
)
def test_blocked_response_headers_obey_hard_deadline_and_close_late_response(
    blocked_path: str,
    phase: str,
    queue_deletes: int,
) -> None:
    base = SuccessfulScript()
    transport = BlockingResponseTransport(base, blocked_path)

    started = time.perf_counter()
    with pytest.raises(ComfyUIImageEditError) as exc:
        _make_adapter_with_transport(
            transport,
            config=_config(
                comfyui_image_request_timeout_seconds=5.0,
                comfyui_image_total_deadline_seconds=0.2,
            ),
        ).generate(_request())
    elapsed = time.perf_counter() - started

    _assert_phase(exc, phase)
    assert elapsed < 0.75
    assert transport.started.is_set()
    assert transport.closed.is_set()
    assert transport.late_response_created.wait(0.1)
    assert transport.late_stream.closed.wait(0.1)
    assert base.calls.count(("POST", "/queue")) == queue_deletes


def test_factory_clients_close_blocked_sends_without_thread_accumulation() -> None:
    records: list[tuple[httpx.Client, BlockingResponseTransport]] = []
    baseline_threads = {
        thread.ident
        for thread in threading.enumerate()
        if thread.name == "comfyui-request-send"
    }

    class TrackingClient(httpx.Client):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            super().close()

    def client_factory() -> httpx.Client:
        transport = BlockingResponseTransport(
            SuccessfulScript(),
            "/object_info",
        )
        client = TrackingClient(transport=transport)
        records.append((client, transport))
        return client

    image_adapter = adapter_module.ComfyUIImageAdapter(
        config=_config(
            comfyui_image_request_timeout_seconds=5.0,
            comfyui_image_total_deadline_seconds=0.08,
        ),
        client_factory=client_factory,
    )

    for attempt in range(3):
        started = time.perf_counter()
        with pytest.raises(ComfyUIImageEditError) as exc:
            image_adapter.generate(_request())
        elapsed = time.perf_counter() - started

        _assert_phase(exc, "remote_schema_preflight")
        assert elapsed < 0.5
        client, transport = records[attempt]
        assert client.close_calls == 1
        assert client.is_closed
        assert transport.closed.is_set()
        assert transport.late_response_created.wait(0.1)
        assert transport.late_stream.closed.wait(0.1)

        thread_deadline = time.perf_counter() + 0.1
        while time.perf_counter() < thread_deadline:
            leaked = [
                thread
                for thread in threading.enumerate()
                if thread.name == "comfyui-request-send"
                and thread.ident not in baseline_threads
            ]
            if not leaked:
                break
            time.sleep(0.005)
        assert leaked == []

    assert len(records) == 3
    assert len({id(client) for client, _transport in records}) == 3


def test_blocked_response_headers_recheck_cancellation_and_delete_known_prompt() -> None:
    event = threading.Event()
    base = SuccessfulScript()
    transport = BlockingResponseTransport(
        base,
        "/history/opaque-prompt-id",
        on_block=event.set,
    )

    started = time.perf_counter()
    with pytest.raises(ImageGenerationCancelled):
        _make_adapter_with_transport(transport).generate(
            _request(cancel_event=event)
        )
    elapsed = time.perf_counter() - started

    assert elapsed < 0.25
    assert transport.closed.is_set()
    assert transport.late_response_created.wait(0.1)
    assert transport.late_stream.closed.wait(0.1)
    assert base.calls.count(("POST", "/queue")) == 1


@pytest.mark.parametrize("control", ["deadline", "cancel"])
def test_blocked_multipart_upload_obeys_control_and_closes_request_stream(
    monkeypatch,
    control: str,
) -> None:
    real_client = httpx.Client
    event = threading.Event()
    wrapped_streams: list[BlockingRequestStream] = []

    class TrackingClient(real_client):
        def build_request(self, *args, **kwargs) -> httpx.Request:
            request = super().build_request(*args, **kwargs)
            if request.url.path == "/upload/image":
                wrapped = BlockingRequestStream(
                    request.stream,
                    on_block=event.set if control == "cancel" else None,
                )
                request.stream = wrapped
                wrapped_streams.append(wrapped)
            return request

    monkeypatch.setattr(adapter_module.httpx, "Client", TrackingClient)
    base = SuccessfulScript()
    deadline = 0.08 if control == "deadline" else 5.0

    started = time.perf_counter()
    if control == "cancel":
        with pytest.raises(ImageGenerationCancelled):
            _make_adapter_with_transport(ScriptTransport(base)).generate(
                _request(cancel_event=event)
            )
    else:
        with pytest.raises(ComfyUIImageEditError) as exc:
            _make_adapter_with_transport(
                ScriptTransport(base),
                config=_config(
                    comfyui_image_request_timeout_seconds=5.0,
                    comfyui_image_total_deadline_seconds=deadline,
                ),
            ).generate(_request(cancel_event=event))
        _assert_phase(exc, "source_upload")
    elapsed = time.perf_counter() - started

    assert elapsed < 0.25
    assert len(wrapped_streams) == 1
    assert wrapped_streams[0].started.is_set()
    assert wrapped_streams[0].closed.is_set()
    assert wrapped_streams[0].finished.wait(0.1)
    assert base.calls.count(("POST", "/queue")) == 0
    assert not any(path == "/prompt" for _, path in base.calls)


@pytest.mark.parametrize(
    "descriptor",
    [
        {"filename": "../edited.png", "subfolder": "", "type": "output"},
        {"filename": "edited.jpg", "subfolder": "", "type": "output"},
        {"filename": "edited.png", "subfolder": "../escape", "type": "output"},
        {"filename": "edited.png", "subfolder": "", "type": "temp"},
        {"filename": "edited.png", "subfolder": "safe//ambiguous", "type": "output"},
    ],
)
def test_output_descriptor_is_safe_node165_only_and_query_only(descriptor) -> None:
    history = {
        "opaque-prompt-id": {
            "status": {"completed": True, "status_str": "success"},
            "outputs": {
                "114": {"images": [{"filename": "source.png", "subfolder": "", "type": "output"}]},
                "165": {"images": [descriptor]},
            },
        }
    }

    with pytest.raises(ComfyUIImageEditError) as exc:
        adapter_module._select_output_descriptor(history, "opaque-prompt-id")

    _assert_phase(exc, "output_descriptor_validation")


def test_terminal_execution_error_is_sanitized_and_not_polled_forever() -> None:
    secret_body = "sentinel-private-server-body"
    records: list[str] = []
    sink = logger.add(lambda message: records.append(str(message)))

    def script(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/object_info":
            return _json_response(_object_info())
        if request.url.path == "/upload/image":
            return _json_response({"name": "opaque.png", "subfolder": "", "type": "input"})
        if request.url.path == "/prompt":
            return _json_response({"prompt_id": "opaque-prompt-id"})
        if request.url.path == "/history/opaque-prompt-id":
            return _json_response(
                {
                    "opaque-prompt-id": {
                        "status": {"completed": True, "status_str": "error", "messages": [["error", {"details": secret_body}]]},
                        "outputs": {},
                    }
                }
            )
        raise AssertionError("unexpected request")

    try:
        with pytest.raises(ComfyUIImageEditError) as exc:
            _make_adapter(script).generate(_request(prompt="sentinel-private-instruction"))
    finally:
        logger.remove(sink)

    _assert_phase(exc, "history_polling")
    rendered = "\n".join(records) + str(exc.value)
    assert secret_body not in rendered
    assert "sentinel-private-instruction" not in rendered
    assert "opaque-prompt-id" not in rendered


@pytest.mark.parametrize("kind", ["declared", "actual", "mime", "signature", "decode", "mode", "dimensions"])
def test_png_download_is_bounded_and_validated(kind) -> None:
    good = _png()

    def script(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/view":
            return SuccessfulScript()(request)
        if kind == "declared":
            return httpx.Response(200, content=b"x", headers={"content-type": "image/png", "content-length": "101"})
        if kind == "actual":
            midpoint = len(good) // 2
            return httpx.Response(
                200,
                stream=ChunkStream([good[:midpoint], good[midpoint:]]),
                headers={"content-type": "image/png"},
            )
        if kind == "mime":
            return httpx.Response(200, content=good, headers={"content-type": "image/jpeg"})
        if kind == "signature":
            return httpx.Response(200, content=b"not-a-png", headers={"content-type": "image/png"})
        if kind == "decode":
            return httpx.Response(200, content=PNG_SIGNATURE + b"broken", headers={"content-type": "image/png"})
        if kind == "mode":
            return httpx.Response(200, content=_png(mode="I;16"), headers={"content-type": "image/png"})
        return httpx.Response(200, content=_png(6, 4), headers={"content-type": "image/png"})

    byte_limit = len(good) - 1 if kind == "actual" else 100
    with pytest.raises(ComfyUIImageEditError) as exc:
        _make_adapter(script, config=_config(inline_max_bytes=byte_limit)).generate(_request())

    _assert_phase(exc, "output_download")


def test_concurrent_png_validation_does_not_mutate_warning_filters(
    monkeypatch,
) -> None:
    entered_verify = threading.Event()
    release_verify = threading.Event()
    original_verify = PngImagePlugin.PngImageFile.verify
    worker_errors: list[BaseException] = []
    worker_results = []
    script = SuccessfulScript()

    def blocking_verify(image) -> None:
        entered_verify.set()
        if not release_verify.wait(2):
            raise AssertionError("PNG validation did not resume")
        original_verify(image)

    def generate() -> None:
        try:
            worker_results.append(_make_adapter(script).generate(_request()))
        except BaseException as exc:
            worker_errors.append(exc)

    monkeypatch.setattr(PngImagePlugin.PngImageFile, "verify", blocking_verify)
    original_filters = list(warnings.filters)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", Image.DecompressionBombWarning)
        expected_filters = list(warnings.filters)
        thread = threading.Thread(target=generate)
        thread.start()
        assert entered_verify.wait(2)
        filters_changed = warnings.filters != expected_filters
        unrelated_error = None
        try:
            warnings.warn("unrelated Pillow warning", Image.DecompressionBombWarning)
        except BaseException as exc:
            unrelated_error = exc
        finally:
            release_verify.set()
            thread.join(2)

        assert not thread.is_alive()
        assert filters_changed is False
        assert unrelated_error is None
        assert worker_errors == []
        assert len(worker_results) == 1
        assert worker_results[0].content_type == "image/png"
        assert len(caught) == 1
        assert warnings.filters == expected_filters

    assert warnings.filters == original_filters


def test_png_warning_band_is_rejected_before_full_load(monkeypatch) -> None:
    warning_ceiling = Image.MAX_IMAGE_PIXELS
    assert type(warning_ceiling) is int
    width = 100_000
    height = warning_ceiling // width + 1
    output = _png_with_dimensions(width, height)
    load_calls: list[object] = []
    base = SuccessfulScript()

    def script(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/view":
            return httpx.Response(
                200,
                content=output,
                headers={"content-type": "image/png"},
            )
        return base(request)

    monkeypatch.setattr(
        PngImagePlugin.PngImageFile,
        "load",
        lambda *args, **kwargs: load_calls.append((args, kwargs)),
    )

    with pytest.raises(ComfyUIImageEditError) as exc:
        _make_adapter(script).generate(
            _request(reference_image=_reference(width=width, height=height))
        )

    _assert_phase(exc, "output_download")
    assert load_calls == []


def test_png_external_decompression_warning_is_sanitized(monkeypatch) -> None:
    output = _png()
    base = SuccessfulScript(output=output)
    real_open = Image.open
    open_calls = 0

    def warned_open(*args, **kwargs):
        nonlocal open_calls
        open_calls += 1
        warnings.warn("external Pillow limit", Image.DecompressionBombWarning)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(adapter_module.Image, "open", warned_open)
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with pytest.raises(ComfyUIImageEditError) as exc:
            _make_adapter(base).generate(_request())

    _assert_phase(exc, "output_download")
    assert open_calls == 1


@pytest.mark.parametrize("encoding", ["gzip", "br"])
def test_png_rejects_non_identity_content_encoding_before_iteration_or_pillow(
    monkeypatch,
    encoding: str,
) -> None:
    stream = GuardedChunkStream([_png()])
    pillow_calls: list[object] = []
    monkeypatch.setattr(
        adapter_module.Image,
        "open",
        lambda value: pillow_calls.append(value),
    )
    base = SuccessfulScript()

    def script(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/view":
            base.calls.append((request.method, request.url.path))
            base.requests.append(request)
            return httpx.Response(
                200,
                stream=stream,
                headers={
                    "content-type": "image/png",
                    "content-encoding": encoding,
                },
            )
        return base(request)

    with pytest.raises(ComfyUIImageEditError) as exc:
        _make_adapter(script).generate(_request())

    _assert_phase(exc, "output_download")
    assert stream.iterated is False
    assert pillow_calls == []


def test_png_oversized_single_chunk_never_extends_bounded_buffer_or_parses(
    monkeypatch,
) -> None:
    max_buffer_len = 0
    pillow_calls: list[object] = []
    real_bytearray = bytearray

    class TrackingBytearray(real_bytearray):
        def extend(self, value) -> None:
            nonlocal max_buffer_len
            super().extend(value)
            max_buffer_len = max(max_buffer_len, len(self))

    monkeypatch.setattr(adapter_module, "bytearray", TrackingBytearray, raising=False)
    monkeypatch.setattr(
        adapter_module.Image,
        "open",
        lambda value: pillow_calls.append(value),
    )
    response = httpx.Response(
        200,
        stream=ChunkStream([PNG_SIGNATURE + b"x" * 64]),
        headers={"content-type": "image/png"},
    )

    with pytest.raises(ValueError):
        adapter_module._stream_png(response, max_bytes=16)

    assert max_buffer_len == 0
    assert pillow_calls == []


def test_png_drip_stream_hits_absolute_deadline_and_deletes_prompt_once(
    monkeypatch,
) -> None:
    clock = AdvancingClock()
    monkeypatch.setattr(adapter_module.time, "monotonic", clock.monotonic)
    output = _png()
    size = len(output) // 4
    chunks = [
        output[:size],
        output[size : size * 2],
        output[size * 2 : size * 3],
        output[size * 3 :],
    ]
    yielded: list[int] = []
    base = SuccessfulScript()

    def advance(index: int) -> None:
        yielded.append(index)
        clock.advance(0.4)

    def script(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/view":
            base.calls.append((request.method, request.url.path))
            base.requests.append(request)
            return httpx.Response(
                200,
                stream=ControlledChunkStream(chunks, advance),
                headers={"content-type": "image/png"},
            )
        return base(request)

    with pytest.raises(ComfyUIImageEditError) as exc:
        _make_adapter(
            script,
            config=_config(
                comfyui_image_request_timeout_seconds=0.5,
                comfyui_image_total_deadline_seconds=1.0,
            ),
        ).generate(_request())

    _assert_phase(exc, "output_download")
    assert yielded == [0, 1, 2]
    assert base.calls.count(("POST", "/queue")) == 1


def test_png_stream_cancellation_stops_and_deletes_prompt_once() -> None:
    event = threading.Event()
    output = _png()
    size = len(output) // 4
    chunks = [
        output[:size],
        output[size : size * 2],
        output[size * 2 : size * 3],
        output[size * 3 :],
    ]
    yielded: list[int] = []
    base = SuccessfulScript()

    def cancel(index: int) -> None:
        yielded.append(index)
        if index == 1:
            event.set()

    def script(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/view":
            base.calls.append((request.method, request.url.path))
            base.requests.append(request)
            return httpx.Response(
                200,
                stream=ControlledChunkStream(chunks, cancel),
                headers={"content-type": "image/png"},
            )
        return base(request)

    with pytest.raises(ImageGenerationCancelled):
        _make_adapter(script).generate(_request(cancel_event=event))

    assert yielded == [0, 1]
    assert base.calls.count(("POST", "/queue")) == 1


def test_cancel_before_prompt_id_never_deletes_queue() -> None:
    event = threading.Event()
    event.set()
    calls: list[str] = []

    def script(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        raise AssertionError("cancelled request reached transport")

    with pytest.raises(ImageGenerationCancelled):
        _make_adapter(script).generate(_request(cancel_event=event))

    assert calls == []


@pytest.mark.parametrize(
    ("cancel_after", "forbidden_next", "queue_deletes"),
    [
        ("/object_info", "/upload/image", 0),
        ("/upload/image", "/prompt", 0),
        ("/prompt", "/history/opaque-prompt-id", 1),
        ("/history/opaque-prompt-id", "/view", 1),
    ],
)
def test_cancellation_is_checked_before_each_following_network_phase(
    cancel_after: str,
    forbidden_next: str,
    queue_deletes: int,
) -> None:
    event = threading.Event()
    script = SuccessfulScript(previews=1 if cancel_after == "/history/opaque-prompt-id" else 0)

    def cancel_between_phases(request: httpx.Request) -> httpx.Response:
        response = script(request)
        if request.url.path == "/prompt" and cancel_after == "/prompt":
            body = response.content
            response.close()
            return httpx.Response(
                200,
                stream=ControlledChunkStream([body], lambda _index: event.set()),
            )
        if request.url.path == cancel_after:
            event.set()
        return response

    with pytest.raises(ImageGenerationCancelled):
        _make_adapter(cancel_between_phases).generate(_request(cancel_event=event))

    assert not any(path == forbidden_next for _, path in script.calls)
    assert script.calls.count(("POST", "/queue")) == queue_deletes


def test_prompt_full_id_chunk_cancellation_captures_id_and_deletes_once() -> None:
    event = threading.Event()
    base = SuccessfulScript()
    yielded: list[int] = []

    def cancel(index: int) -> None:
        yielded.append(index)
        if index == 0:
            event.set()

    def script(request: httpx.Request) -> httpx.Response:
        response = base(request)
        if request.url.path != "/prompt":
            return response
        body = response.content
        response.close()
        return httpx.Response(
            200,
            stream=ControlledChunkStream([body, b"forbidden-extra"], cancel),
        )

    with pytest.raises(ImageGenerationCancelled):
        _make_adapter(script).generate(_request(cancel_event=event))

    assert base.calls.count(("POST", "/queue")) == 1
    assert not any(path == "/history/opaque-prompt-id" for _, path in base.calls)
    assert yielded == [0]


def test_prompt_full_id_chunk_deadline_captures_id_and_deletes_once(monkeypatch) -> None:
    clock = AdvancingClock()
    monkeypatch.setattr(adapter_module.time, "monotonic", clock.monotonic)
    base = SuccessfulScript()
    yielded: list[int] = []

    def cross_deadline(index: int) -> None:
        yielded.append(index)
        if index == 0:
            clock.value = 1.1

    def script(request: httpx.Request) -> httpx.Response:
        response = base(request)
        if request.url.path != "/prompt":
            return response
        body = response.content
        response.close()
        return httpx.Response(
            200,
            stream=ControlledChunkStream([body, b"forbidden-extra"], cross_deadline),
        )

    with pytest.raises(ComfyUIImageEditError) as exc:
        _make_adapter(
            script,
            config=_config(comfyui_image_total_deadline_seconds=1.0),
        ).generate(_request())

    _assert_phase(exc, "prompt_submission")
    assert base.calls.count(("POST", "/queue")) == 1
    assert not any(path == "/history/opaque-prompt-id" for _, path in base.calls)
    assert yielded == [0]


def test_prompt_partial_id_chunk_cancellation_has_no_id_and_no_delete() -> None:
    event = threading.Event()
    base = SuccessfulScript()
    yielded: list[int] = []

    def cancel(index: int) -> None:
        yielded.append(index)
        if index == 0:
            event.set()

    def script(request: httpx.Request) -> httpx.Response:
        response = base(request)
        if request.url.path != "/prompt":
            return response
        response.close()
        return httpx.Response(
            200,
            stream=ControlledChunkStream(
                [b'{"prompt_id":"partial', b'-opaque-prompt-id"}'],
                cancel,
            ),
        )

    with pytest.raises(ImageGenerationCancelled):
        _make_adapter(script).generate(_request(cancel_event=event))

    assert base.calls.count(("POST", "/queue")) == 0
    assert not any(path == "/history/opaque-prompt-id" for _, path in base.calls)
    assert yielded == [0]


def test_poll_wait_uses_event_and_cancellation_deletes_exact_prompt_once() -> None:
    event = TrackingEvent(cancel_on_wait=True)
    calls: list[tuple[str, str, bytes]] = []

    def script(request: httpx.Request) -> httpx.Response:
        body = request.read()
        calls.append((request.method, request.url.path, body))
        if request.url.path == "/object_info":
            return _json_response(_object_info())
        if request.url.path == "/upload/image":
            return _json_response({"name": "opaque.png", "subfolder": "", "type": "input"})
        if request.url.path == "/prompt":
            return _json_response({"prompt_id": "opaque-prompt-id"})
        if request.url.path == "/history/opaque-prompt-id":
            return _json_response({})
        if request.url.path == "/queue":
            return _json_response({"deleted": False})
        raise AssertionError("unexpected request")

    with pytest.raises(ImageGenerationCancelled):
        _make_adapter(script).generate(_request(cancel_event=event))

    assert event.waits == [0.25]
    queue_calls = [call for call in calls if call[1] == "/queue"]
    assert len(queue_calls) == 1
    assert json.loads(queue_calls[0][2]) == {"delete": ["opaque-prompt-id"]}
    assert not any(path == "/interrupt" for _, path, _ in calls)


def test_queue_delete_failure_cannot_mask_cancellation() -> None:
    event = TrackingEvent(cancel_on_wait=True)

    def script(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/object_info":
            return _json_response(_object_info())
        if request.url.path == "/upload/image":
            return _json_response({"name": "opaque.png", "subfolder": "", "type": "input"})
        if request.url.path == "/prompt":
            return _json_response({"prompt_id": "opaque-prompt-id"})
        if request.url.path == "/history/opaque-prompt-id":
            return _json_response({})
        if request.url.path == "/queue":
            raise httpx.ConnectError("sentinel-delete-detail")
        raise AssertionError("unexpected request")

    with pytest.raises(ImageGenerationCancelled):
        _make_adapter(script).generate(_request(cancel_event=event))


def test_timeout_uses_monotonic_remaining_time_and_deletes_once(monkeypatch) -> None:
    clock = AdvancingClock()
    monkeypatch.setattr(adapter_module.time, "monotonic", clock.monotonic)
    calls: list[httpx.Request] = []

    def script(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/object_info":
            return _json_response(_object_info())
        if request.url.path == "/upload/image":
            return _json_response({"name": "opaque.png", "subfolder": "", "type": "input"})
        if request.url.path == "/prompt":
            return _json_response({"prompt_id": "opaque-prompt-id"})
        if request.url.path == "/history/opaque-prompt-id":
            clock.value = 1.1
            return _json_response({})
        if request.url.path == "/queue":
            raise httpx.ConnectError("sentinel-timeout-delete-detail")
        raise AssertionError("unexpected request")

    with pytest.raises(ComfyUIImageEditError) as exc:
        _make_adapter(
            script,
            config=_config(
                comfyui_image_total_deadline_seconds=1.0,
                comfyui_image_poll_interval_seconds=0.25,
            ),
        ).generate(_request(cancel_event=TrackingEvent()))

    _assert_phase(exc, "history_polling")
    assert len([request for request in calls if request.url.path == "/queue"]) == 1
    for request in calls:
        timeout = request.extensions.get("timeout")
        if timeout:
            assert max(value for value in timeout.values() if value is not None) <= 1.0


def test_history_read_timeout_after_deadline_deletes_once_and_stays_sanitized(
    monkeypatch,
) -> None:
    clock = AdvancingClock()
    monkeypatch.setattr(adapter_module.time, "monotonic", clock.monotonic)
    base = SuccessfulScript()
    private_detail = "sentinel-private-read-timeout-detail"

    def script(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/history/opaque-prompt-id":
            base.calls.append((request.method, request.url.path))
            base.requests.append(request)
            clock.value = 1.1
            raise httpx.ReadTimeout(private_detail, request=request)
        return base(request)

    with pytest.raises(ComfyUIImageEditError) as exc:
        _make_adapter(
            script,
            config=_config(comfyui_image_total_deadline_seconds=1.0),
        ).generate(_request())

    _assert_phase(exc, "history_polling")
    assert private_detail not in str(exc.value)
    assert base.calls.count(("POST", "/queue")) == 1


@pytest.mark.parametrize(
    ("failing_path", "phase"),
    [
        ("/history/opaque-prompt-id", "history_polling"),
        ("/view", "output_download"),
    ],
)
def test_body_iterator_read_timeout_after_deadline_deletes_known_prompt_once(
    monkeypatch,
    failing_path: str,
    phase: str,
) -> None:
    clock = AdvancingClock()
    monkeypatch.setattr(adapter_module.time, "monotonic", clock.monotonic)
    base = SuccessfulScript()
    private_detail = "sentinel-private-iterator-timeout-detail"

    def expire() -> None:
        clock.value = 1.1

    def script(request: httpx.Request) -> httpx.Response:
        if request.url.path == failing_path:
            base.calls.append((request.method, request.url.path))
            base.requests.append(request)
            headers = {"content-type": "image/png"} if failing_path == "/view" else {}
            return httpx.Response(
                200,
                stream=FailingChunkStream(expire, private_detail),
                headers=headers,
            )
        return base(request)

    with pytest.raises(ComfyUIImageEditError) as exc:
        _make_adapter(
            script,
            config=_config(comfyui_image_total_deadline_seconds=1.0),
        ).generate(_request())

    _assert_phase(exc, phase)
    assert private_detail not in str(exc.value)
    assert base.calls.count(("POST", "/queue")) == 1


def test_body_iterator_read_timeout_after_deadline_before_prompt_has_no_delete(
    monkeypatch,
) -> None:
    clock = AdvancingClock()
    monkeypatch.setattr(adapter_module.time, "monotonic", clock.monotonic)
    private_detail = "sentinel-private-pre-id-iterator-timeout-detail"
    calls: list[str] = []

    def expire() -> None:
        clock.value = 1.1

    def script(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        assert request.url.path == "/object_info"
        return httpx.Response(
            200,
            stream=FailingChunkStream(expire, private_detail),
        )

    with pytest.raises(ComfyUIImageEditError) as exc:
        _make_adapter(
            script,
            config=_config(comfyui_image_total_deadline_seconds=1.0),
        ).generate(_request())

    _assert_phase(exc, "remote_schema_preflight")
    assert private_detail not in str(exc.value)
    assert calls == ["/object_info"]


def test_body_iterator_read_timeout_before_deadline_stays_phase_transport_error(
    monkeypatch,
) -> None:
    clock = AdvancingClock()
    monkeypatch.setattr(adapter_module.time, "monotonic", clock.monotonic)
    base = SuccessfulScript()
    private_detail = "sentinel-private-pre-deadline-iterator-detail"

    def remain_before_deadline() -> None:
        clock.value = 0.5

    def script(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/history/opaque-prompt-id":
            base.calls.append((request.method, request.url.path))
            base.requests.append(request)
            return httpx.Response(
                200,
                stream=FailingChunkStream(remain_before_deadline, private_detail),
            )
        return base(request)

    with pytest.raises(ComfyUIImageEditError) as exc:
        _make_adapter(
            script,
            config=_config(comfyui_image_total_deadline_seconds=1.0),
        ).generate(_request())

    _assert_phase(exc, "history_polling")
    assert private_detail not in str(exc.value)
    assert base.calls.count(("POST", "/queue")) == 0


def test_body_iterator_transport_failure_rechecks_cancellation_before_deadline() -> None:
    event = threading.Event()
    base = SuccessfulScript()
    private_detail = "sentinel-private-cancelled-iterator-detail"

    def cancel() -> None:
        event.set()

    def script(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/history/opaque-prompt-id":
            base.calls.append((request.method, request.url.path))
            base.requests.append(request)
            return httpx.Response(
                200,
                stream=FailingChunkStream(cancel, private_detail),
            )
        return base(request)

    with pytest.raises(ImageGenerationCancelled) as exc:
        _make_adapter(script).generate(_request(cancel_event=event))

    assert private_detail not in str(exc.value)
    assert base.calls.count(("POST", "/queue")) == 1


@pytest.mark.parametrize(
    ("failing_path", "first_chunk", "phase"),
    [
        ("/history/opaque-prompt-id", b"{", "history_polling"),
        ("/view", PNG_SIGNATURE, "output_download"),
    ],
)
def test_blocking_body_read_obeys_hard_deadline_and_closes_stream(
    failing_path: str,
    first_chunk: bytes,
    phase: str,
) -> None:
    stream = BlockingChunkStream(first_chunk)
    base = SuccessfulScript()

    def script(request: httpx.Request) -> httpx.Response:
        if request.url.path == failing_path:
            base.calls.append((request.method, request.url.path))
            base.requests.append(request)
            headers = {"content-type": "image/png"} if failing_path == "/view" else {}
            return httpx.Response(200, stream=stream, headers=headers)
        return base(request)

    started = time.perf_counter()
    with pytest.raises(ComfyUIImageEditError) as exc:
        _make_adapter(
            script,
            config=_config(
                comfyui_image_request_timeout_seconds=5.0,
                comfyui_image_total_deadline_seconds=0.08,
            ),
        ).generate(_request())
    elapsed = time.perf_counter() - started

    _assert_phase(exc, phase)
    assert elapsed < 0.25
    assert stream.second_read_started.is_set()
    assert stream.closed.is_set()
    assert stream.finished.wait(0.1)
    assert base.calls.count(("POST", "/queue")) == 1


def test_blocking_body_read_before_prompt_times_out_without_delete() -> None:
    stream = BlockingChunkStream(b"{")
    calls: list[str] = []

    def script(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        assert request.url.path == "/object_info"
        return httpx.Response(200, stream=stream)

    started = time.perf_counter()
    with pytest.raises(ComfyUIImageEditError) as exc:
        _make_adapter(
            script,
            config=_config(
                comfyui_image_request_timeout_seconds=5.0,
                comfyui_image_total_deadline_seconds=0.08,
            ),
        ).generate(_request())
    elapsed = time.perf_counter() - started

    _assert_phase(exc, "remote_schema_preflight")
    assert elapsed < 0.25
    assert stream.second_read_started.is_set()
    assert stream.closed.is_set()
    assert stream.finished.wait(0.1)
    assert calls == ["/object_info"]


def test_blocking_body_read_wakes_prompt_scoped_cancellation_promptly() -> None:
    event = threading.Event()
    stream = BlockingChunkStream(b"{", on_block=event.set)
    base = SuccessfulScript()

    def script(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/history/opaque-prompt-id":
            base.calls.append((request.method, request.url.path))
            base.requests.append(request)
            return httpx.Response(200, stream=stream)
        return base(request)

    started = time.perf_counter()
    with pytest.raises(ImageGenerationCancelled):
        _make_adapter(script).generate(_request(cancel_event=event))
    elapsed = time.perf_counter() - started

    assert elapsed < 0.25
    assert stream.second_read_started.is_set()
    assert stream.closed.is_set()
    assert stream.finished.wait(0.1)
    assert base.calls.count(("POST", "/queue")) == 1


def test_final_event_check_after_validated_png_makes_cancellation_win() -> None:
    event = TrackingEvent()
    script = SuccessfulScript()

    def set_cancel_after_output(request: httpx.Request) -> httpx.Response:
        response = script(request)
        if request.url.path == "/view":
            event.set_state = True
        return response

    with pytest.raises(ImageGenerationCancelled):
        _make_adapter(set_cancel_after_output).generate(_request(cancel_event=event))

    assert script.calls.count(("POST", "/queue")) == 1
