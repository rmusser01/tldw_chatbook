"""ComfyUI video adapter tests (task-3401.6)."""

from __future__ import annotations

import inspect
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Image_Generation.http_client import fetch_json
from tldw_chatbook.Video_Generation.adapters import comfyui_video_adapter as cva
from tldw_chatbook.Video_Generation.adapters.base import ResolvedReferenceAsset
from tldw_chatbook.Video_Generation.exceptions import (
    VideoBackendUnavailableError,
    VideoGenerationError,
)
from tldw_chatbook.Video_Generation.worker import build_request


def _fake_config(**overrides):
    base = {
        "comfyui_base_url": "http://127.0.0.1:8188",
        "comfyui_default_workflow": "wan22_t2v.json",
        "comfyui_timeout_seconds": 30,
        "download_max_mb": 500,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(cva, "get_video_generation_config", lambda: _fake_config())
    return cva.ComfyUIVideoAdapter()


@pytest.fixture
def json_recorder(monkeypatch):
    """Record fetch_json calls, with fake signature matching the real helper."""
    assert set(inspect.signature(fetch_json).parameters) >= {
        "method", "url", "headers", "json", "params", "timeout", "trusted_origins",
    }
    calls: list[dict] = []
    routes: dict[tuple[str, str], object] = {}

    def fake_fetch_json(
        method, url, *, headers=None, json=None, params=None, cookies=None,
        timeout=None, trusted_origins=frozenset(),
    ):
        calls.append({
            "method": method, "url": url, "headers": headers, "json": json,
            "params": params, "cookies": cookies, "timeout": timeout,
            "trusted_origins": trusted_origins,
        })
        for (route_method, fragment), response in routes.items():
            if method == route_method and fragment in url:
                if isinstance(response, list):
                    return response.pop(0)
                return response
        raise AssertionError(f"unscripted {method} {url}")

    monkeypatch.setattr(cva, "fetch_json", fake_fetch_json)
    return calls, routes


def _request(**overrides):
    kwargs = {"backend": "comfyui", "prompt": "a lighthouse in a storm"}
    kwargs.update(overrides)
    return build_request(**kwargs)


def _workflow():
    return {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"}, "_meta": {"title": "Prompt"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "keep"}, "_meta": {"title": "Negative Prompt"}},
        "3": {"class_type": "KSampler", "inputs": {"seed": 0, "latent_image": ["9", 0]}, "_meta": {"title": "Seed"}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 320, "height": 240, "batch_size": 1}, "_meta": {"title": "Width"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 320, "height": 240}, "_meta": {"title": "Height"}},
        "6": {"class_type": "WanVideo", "inputs": {"num_frames": 1, "fps": 8}, "_meta": {"title": "Frames"}},
        "7": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["6", 0]}, "_meta": {"title": "Output"}},
        "8": {"class_type": "LoadImage", "inputs": {"image": "old.png"}, "_meta": {"title": "Input Image"}},
        "9": {"class_type": "OtherNode", "inputs": {"text": "unrelated", "seed": 77}, "_meta": {"title": "Ignore me"}},
        "10": {"class_type": "OtherNode", "inputs": {"fps": 8}, "_meta": {"title": "FPS"}},
    }


def _install_workflow(adapter, monkeypatch, workflow=None):
    monkeypatch.setattr(adapter, "_load_workflow", lambda _name: _workflow() if workflow is None else workflow)


def _object_info_for(workflow):
    return {node["class_type"]: {} for node in workflow.values()}


def test_submits_title_parameterized_workflow_polls_and_downloads(adapter, json_recorder, monkeypatch):
    calls, routes = json_recorder
    graph = _workflow()
    _install_workflow(adapter, monkeypatch, graph)
    monkeypatch.setattr(cva, "fetch_image_bytes", lambda url, **kwargs: (b"video", "video/mp4"))
    routes[("GET", "/object_info")] = _object_info_for(graph)
    routes[("POST", "/prompt")] = {"prompt_id": "job-1"}
    routes[("GET", "/history/job-1")] = [
        {},
        {"job-1": {"outputs": {"7": {"videos": [{"filename": "clip.mp4", "subfolder": "out", "type": "output"}]}}}},
    ]
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    result = adapter.generate(_request(seed=41, width=640, height=360, fps=24, duration_seconds=2))

    assert result.content == b"video"
    assert result.content_type == "video/mp4"
    assert result.bytes_len == 5
    submit = next(call for call in calls if call["method"] == "POST" and call["url"].endswith("/prompt"))
    assert submit["url"] == "http://127.0.0.1:8188/prompt"
    assert submit["json"]["client_id"]
    sent = submit["json"]["prompt"]
    assert sent["1"]["inputs"]["text"] == "a lighthouse in a storm"
    assert sent["2"]["inputs"]["text"] == ""
    assert sent["3"]["inputs"]["seed"] == 41
    assert sent["4"]["inputs"]["width"] == 640
    assert sent["5"]["inputs"]["height"] == 360
    assert sent["6"]["inputs"]["num_frames"] == 48
    assert sent["10"]["inputs"]["fps"] == 24
    assert sent["3"]["inputs"]["latent_image"] == ["9", 0]
    assert sent["9"]["inputs"] == {"text": "unrelated", "seed": 77}
    assert graph["1"]["inputs"]["text"] == "old"
    assert all(call["trusted_origins"] == frozenset({"127.0.0.1"}) for call in calls)
    histories = [call for call in calls if "/history/job-1" in call["url"]]
    assert len(histories) == 2


def test_view_download_has_descriptor_query_and_trusted_origin(adapter, json_recorder, monkeypatch):
    calls, routes = json_recorder
    graph = _workflow()
    _install_workflow(adapter, monkeypatch, graph)
    download_calls = []

    def fake_fetch_bytes(url, **kwargs):
        download_calls.append((url, kwargs))
        return b"animated", "image/webp"

    monkeypatch.setattr(cva, "fetch_image_bytes", fake_fetch_bytes)
    routes[("GET", "/object_info")] = _object_info_for(graph)
    routes[("POST", "/prompt")] = {"prompt_id": "job-2"}
    routes[("GET", "/history/job-2")] = {
        "job-2": {"outputs": {"7": {"gifs": [{"filename": "clip.webp", "subfolder": "", "type": "temp"}]}}}
    }

    result = adapter.generate(_request())

    assert result.content_type == "image/webp"
    assert download_calls == [(
        "http://127.0.0.1:8188/view?filename=clip.webp&subfolder=&type=temp",
        {"timeout": 30, "max_bytes": 500 * 1024 * 1024, "trusted_origins": frozenset({"127.0.0.1"})},
    )]


def test_uploads_image_asset_and_injects_uploaded_filename(adapter, json_recorder, monkeypatch):
    calls, routes = json_recorder
    graph = _workflow()
    _install_workflow(adapter, monkeypatch, graph)
    uploads = []
    monkeypatch.setattr(adapter, "_upload_image", lambda asset: uploads.append(asset) or "upload.png")
    monkeypatch.setattr(cva, "fetch_image_bytes", lambda url, **kwargs: (b"v", "video/mp4"))
    routes[("GET", "/object_info")] = _object_info_for(graph)
    routes[("POST", "/prompt")] = {"prompt_id": "job-3"}
    routes[("GET", "/history/job-3")] = {
        "job-3": {"outputs": {"7": {"videos": [{"filename": "clip.mp4", "subfolder": "", "type": "output"}]}}}
    }
    image = ResolvedReferenceAsset("first_frame", b"png-bytes", "image/png", "source.png")

    adapter.generate(_request(reference_assets=(image,)))

    assert uploads == [image]
    submit = next(call for call in calls if call["method"] == "POST" and call["url"].endswith("/prompt"))
    assert submit["json"]["prompt"]["8"]["inputs"]["image"] == "upload.png"


def test_upload_uses_multipart_endpoint_and_trusted_origin(adapter, monkeypatch):
    requests = []
    egress_checks = []

    class Response:
        is_redirect = False

        def raise_for_status(self):
            return None

        def json(self):
            return {"name": "input.png", "subfolder": "upload"}

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def post(self, url, **kwargs):
            requests.append((url, kwargs))
            return Response()

    monkeypatch.setattr(cva, "create_client", lambda timeout: Client())
    monkeypatch.setattr(
        cva,
        "_validate_egress_or_raise",
        lambda url, *, trusted_origins: egress_checks.append((url, trusted_origins)),
    )
    asset = ResolvedReferenceAsset("first_frame", b"png-bytes", "image/png", "source.png")

    uploaded = adapter._upload_image(asset)

    assert uploaded == "upload/input.png"
    assert requests == [(
        "http://127.0.0.1:8188/upload/image",
        {"files": {"image": ("source.png", b"png-bytes", "image/png")}, "data": {"overwrite": "true"}},
    )]
    assert egress_checks == [
        ("http://127.0.0.1:8188/upload/image", frozenset({"127.0.0.1"}))
    ]


def test_missing_required_classes_fail_before_prompt(adapter, json_recorder, monkeypatch):
    calls, routes = json_recorder
    graph = _workflow()
    _install_workflow(adapter, monkeypatch, graph)
    routes[("GET", "/object_info")] = {"CLIPTextEncode": {}, "KSampler": {}}

    with pytest.raises(VideoBackendUnavailableError, match="EmptyLatentImage.*LoadImage.*OtherNode.*VHS_VideoCombine.*WanVideo"):
        adapter.generate(_request())
    assert not any(call["url"].endswith("/prompt") for call in calls)


def test_cancellation_interrupts_and_stops(adapter, json_recorder, monkeypatch):
    event = threading.Event()
    calls, routes = json_recorder
    graph = _workflow()
    _install_workflow(adapter, monkeypatch, graph)
    routes[("GET", "/object_info")] = _object_info_for(graph)
    routes[("POST", "/prompt")] = {"prompt_id": "job-4"}
    routes[("POST", "/interrupt")] = {}
    event.set()

    with pytest.raises(VideoGenerationError, match="cancelled"):
        adapter.generate(_request(), cancel_event=event)
    interrupt = next(call for call in calls if call["url"].endswith("/interrupt"))
    assert interrupt["method"] == "POST"
    assert interrupt["trusted_origins"] == frozenset({"127.0.0.1"})


def test_workflow_resolution_prefers_user_dir_and_rejects_unsafe_names(adapter, monkeypatch, tmp_path):
    user_dir = tmp_path / "data"
    workflows = user_dir / "video_workflows"
    workflows.mkdir(parents=True)
    (workflows / "chosen.json").write_text('{"user": {"class_type": "User"}}')
    monkeypatch.setattr(cva, "get_user_data_dir", lambda: user_dir)

    assert adapter._load_workflow("chosen.json") == {"user": {"class_type": "User"}}
    assert adapter._load_workflow("wan22_t2v.json")["1"]["class_type"] == "UNETLoader"
    with pytest.raises(VideoGenerationError, match="workflow.*JSON"):
        adapter._load_workflow("chosen.txt")
    with pytest.raises(VideoGenerationError, match="workflow.*path"):
        adapter._load_workflow("../chosen.json")


def test_unsupported_reference_and_bad_output_are_clear(adapter, json_recorder, monkeypatch):
    graph = _workflow()
    _install_workflow(adapter, monkeypatch, graph)
    unsupported = ResolvedReferenceAsset("reference_video", b"video", "video/mp4", "source.mp4")
    with pytest.raises(VideoGenerationError, match="image.*first_frame"):
        adapter.generate(_request(reference_assets=(unsupported,)))

    calls, routes = json_recorder
    monkeypatch.setattr(cva, "fetch_image_bytes", lambda url, **kwargs: (b"v", "video/mp4"))
    routes[("GET", "/object_info")] = _object_info_for(graph)
    routes[("POST", "/prompt")] = {"prompt_id": "job-5"}
    routes[("GET", "/history/job-5")] = {"job-5": {"outputs": {"7": {"images": [{"filename": "plain.png", "subfolder": "", "type": "output"}]}}}}
    with pytest.raises(VideoGenerationError, match="video or animated"):
        adapter.generate(_request())


def test_shipped_workflows_are_api_graphs_with_documented_titles(adapter):
    wan = adapter._load_workflow("wan22_t2v.json")
    svd = adapter._load_workflow("svd_xt_i2v.json")
    titles = {
        str(node.get("_meta", {}).get("title", ""))
        for graph in (wan, svd)
        for node in graph.values()
    }
    assert all(isinstance(node, dict) and node.get("class_type") for graph in (wan, svd) for node in graph.values())
    assert {"Prompt", "Negative Prompt", "Seed", "Width", "Height", "Frames", "FPS", "Input Image"} <= titles
