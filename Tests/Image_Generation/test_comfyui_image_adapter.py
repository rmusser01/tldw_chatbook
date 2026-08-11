"""Strict, scripted-transport tests for the packaged ComfyUI image adapter."""

from __future__ import annotations

import copy
import io
import json
import re
import threading
from dataclasses import replace
from typing import Any

import httpx
import pytest
from loguru import logger
from PIL import Image

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


def _object_info(graph: dict[str, Any] | None = None) -> dict[str, Any]:
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


def _make_adapter(script, *, config=None):
    transport = httpx.MockTransport(script)
    return adapter_module.ComfyUIImageAdapter(config=config or _config(), transport=transport)


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
    image_adapter = _make_adapter(lambda request: calls.append(str(request.url)))
    monkeypatch.setattr(adapter_module, "same_origin", lambda _a, _b: False)

    with pytest.raises(ValueError):
        image_adapter._request_json("http://127.0.0.1:8288/object_info", method="GET")

    assert calls == []


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
        ("/prompt", "/history/opaque-prompt-id", 0),
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
        if request.url.path == cancel_after:
            event.set()
        return response

    with pytest.raises(ImageGenerationCancelled):
        _make_adapter(cancel_between_phases).generate(_request(cancel_event=event))

    assert not any(path == forbidden_next for _, path in script.calls)
    assert script.calls.count(("POST", "/queue")) == queue_deletes


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
