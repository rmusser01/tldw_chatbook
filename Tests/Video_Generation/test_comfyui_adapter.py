"""ComfyUI video adapter tests (task-3401.6)."""

from __future__ import annotations

import copy
import inspect
import threading
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import httpx
import pytest

from tldw_chatbook.Image_Generation.adapters.image_format_utils import fetch_image_bytes
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
        "comfyui_default_workflow": "minimax_h3_t2v.json",
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


def _custom_workflow():
    return {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"}, "_meta": {"title": "Prompt"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "keep"}, "_meta": {"title": "Negative Prompt"}},
        "3": {"class_type": "KSampler", "inputs": {"seed": 0, "latent_image": ["9", 0]}, "_meta": {"title": "Seed"}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 320, "height": 240, "batch_size": 1}, "_meta": {"title": "Width"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 320, "height": 240}, "_meta": {"title": "Height"}},
        "6": {"class_type": "VideoFrames", "inputs": {"num_frames": 1, "fps": 8}, "_meta": {"title": "Frames"}},
        "7": {"class_type": "VHS_VideoCombine", "inputs": {"images": ["6", 0]}, "_meta": {"title": "Output"}},
        "8": {"class_type": "LoadImage", "inputs": {"image": "old.png"}, "_meta": {"title": "Input Image"}},
        "9": {"class_type": "OtherNode", "inputs": {"text": "unrelated", "seed": 77}, "_meta": {"title": "Ignore me"}},
        "10": {"class_type": "OtherNode", "inputs": {"fps": 8}, "_meta": {"title": "FPS"}},
    }


def _h3_workflow():
    return {
        "gen": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {"prompt": "safe placeholder", "width": 864, "height": 480, "length": ["expr", 1]},
            "_meta": {"title": "Prompt Width Height"},
        },
        "seed": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": 0},
            "_meta": {"title": "Seed"},
        },
        "duration": {
            "class_type": "PrimitiveFloat",
            "inputs": {"value": 5},
            "_meta": {"title": "Duration"},
        },
        "video": {
            "class_type": "CreateVideo",
            "inputs": {"fps": 24, "images": ["frames", 0], "audio": ["audio", 0]},
            "_meta": {"title": "Native FPS"},
        },
        "save": {
            "class_type": "SaveVideo",
            "inputs": {"format": "mp4", "codec": "auto", "video": ["video", 0]},
            "_meta": {"title": "Save Video"},
        },
    }


def _install_workflow(adapter, monkeypatch, workflow=None):
    monkeypatch.setattr(adapter, "_load_workflow", lambda _name: _h3_workflow() if workflow is None else workflow)


def _object_info_for(workflow):
    return {node["class_type"]: {} for node in workflow.values()}


def test_submits_packaged_workflow_through_observed_history_shape(
    adapter, json_recorder, monkeypatch
):
    calls, routes = json_recorder
    graph = adapter._load_workflow("minimax_h3_t2v.json")
    original_prompt = graph["105:104"]["inputs"]["prompt"]

    def fetch_bytes(
        url: str,
        *,
        timeout: int | float,
        headers=None,
        cookies=None,
        max_bytes=None,
        trusted_origins=frozenset(),
    ):
        return b"video", "video/mp4"

    monkeypatch.setattr(cva, "fetch_image_bytes", fetch_bytes)
    routes[("GET", "/object_info")] = _object_info_for(graph)
    routes[("POST", "/prompt")] = {"prompt_id": "job-1"}
    routes[("GET", "/history/job-1")] = [
        {
            "job-1": {
                "outputs": {
                    "preview": {
                        "images": [
                            {
                                "filename": "preview.png",
                                "subfolder": "",
                                "type": "temp",
                            }
                        ]
                    }
                },
                "status": {
                    "completed": False,
                    "status_str": "running",
                    "messages": [],
                },
            }
        },
        {
            "job-1": {
                "outputs": {
                    "92": {
                        "images": [
                            {
                                "filename": "clip.mp4",
                                "subfolder": "video",
                                "type": "output",
                            }
                        ]
                    }
                },
                "status": {
                    "completed": True,
                    "status_str": "success",
                    "messages": [],
                },
            }
        },
    ]
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    result = adapter.generate(_request(seed=41, width=640, height=352, fps=24, duration_seconds=2))

    assert result.content == b"video"
    assert result.content_type == "video/mp4"
    assert result.container == "mp4"
    assert result.bytes_len == 5
    submit = next(call for call in calls if call["method"] == "POST" and call["url"].endswith("/prompt"))
    assert submit["url"] == "http://127.0.0.1:8188/prompt"
    assert submit["json"]["client_id"]
    sent = submit["json"]["prompt"]
    assert sent["105:104"]["inputs"]["prompt"] == "a lighthouse in a storm"
    assert sent["105:15"]["inputs"]["noise_seed"] == 41
    assert sent["105:104"]["inputs"]["width"] == 640
    assert sent["105:104"]["inputs"]["height"] == 352
    assert sent["105:111"]["inputs"]["value"] == 2
    assert sent["105:91"]["inputs"]["fps"] == 24
    assert sent["105:104"]["inputs"]["length"] == ["105:107", 1]
    assert graph["105:104"]["inputs"]["prompt"] == original_prompt
    assert all(call["trusted_origins"] == frozenset({"127.0.0.1"}) for call in calls)
    histories = [call for call in calls if "/history/job-1" in call["url"]]
    assert len(histories) == 2


def test_generic_webm_output_returns_observed_container(
    adapter, json_recorder, monkeypatch
):
    calls, routes = json_recorder
    graph = _custom_workflow()
    _install_workflow(adapter, monkeypatch, graph)
    download_calls = []

    def fake_fetch_bytes(
        url: str,
        *,
        timeout: int | float,
        headers=None,
        cookies=None,
        max_bytes=None,
        trusted_origins=frozenset(),
    ):
        download_calls.append((url, {
            "timeout": timeout,
            "headers": headers,
            "cookies": cookies,
            "max_bytes": max_bytes,
            "trusted_origins": trusted_origins,
        }))
        return b"video", " Video/WebM ; codecs=vp9 "

    # Match the shared helper's transport contract, not the call under test.
    assert tuple(inspect.signature(fake_fetch_bytes).parameters) == tuple(
        inspect.signature(fetch_image_bytes).parameters
    )
    monkeypatch.setattr(cva, "fetch_image_bytes", fake_fetch_bytes)
    routes[("GET", "/object_info")] = _object_info_for(graph)
    routes[("POST", "/prompt")] = {"prompt_id": "job-2"}
    routes[("GET", "/history/job-2")] = {
        "job-2": {
            "outputs": {"7": {"videos": [{"filename": "clip.webm", "subfolder": "", "type": "output"}]}},
            "status": {"completed": True, "status_str": "success", "messages": []},
        }
    }

    result = adapter.generate(
        _request(
            video_format="webm",
            width=1280,
            height=704,
            duration_seconds=6,
            fps=24,
        )
    )

    assert download_calls == [(
        "http://127.0.0.1:8188/view?filename=clip.webm&subfolder=&type=output",
        {"timeout": 30, "headers": None, "cookies": None, "max_bytes": 500 * 1024 * 1024, "trusted_origins": frozenset({"127.0.0.1"})},
    )]
    assert result.content_type == "video/webm"
    assert result.container == "webm"


def test_h3_webm_request_rejects_after_local_graph_load_before_remote_preflight(
    adapter, monkeypatch
):
    effects: list[str] = []
    original_load = adapter._load_workflow
    monkeypatch.setattr(
        adapter,
        "_load_workflow",
        lambda name: effects.append("load") or original_load(name),
    )
    monkeypatch.setattr(adapter, "_base_url", lambda: effects.append("base_url"))
    monkeypatch.setattr(
        adapter,
        "_validate_required_nodes",
        lambda *_args: effects.append("object_info"),
    )
    monkeypatch.setattr(
        adapter,
        "_resolve_uploaded_image",
        lambda *_args: effects.append("upload"),
    )
    monkeypatch.setattr(
        adapter,
        "_queue_prompt",
        lambda *_args: effects.append("queue"),
    )

    with pytest.raises(VideoGenerationError, match="H3.*MP4"):
        adapter.generate(_request(video_format="webm"))

    assert adapter.supported_formats == {"mp4", "webm"}
    assert effects == ["load"]


@pytest.mark.parametrize("observed_type", ["application/octet-stream", None, "video/webm"])
def test_h3_download_requires_observed_mp4_mime(
    adapter, json_recorder, monkeypatch, observed_type
):
    _calls, routes = json_recorder
    graph = _h3_workflow()
    _install_workflow(adapter, monkeypatch, graph)

    def fetch_bytes(
        url: str,
        *,
        timeout: int | float,
        headers=None,
        cookies=None,
        max_bytes=None,
        trusted_origins=frozenset(),
    ):
        return b"video", observed_type

    monkeypatch.setattr(cva, "fetch_image_bytes", fetch_bytes)
    routes[("GET", "/object_info")] = _object_info_for(graph)
    routes[("POST", "/prompt")] = {"prompt_id": "job-mime"}
    routes[("GET", "/history/job-mime")] = {
        "job-mime": {
            "outputs": {
                "save": {
                    "videos": [
                        {"filename": "clip.mp4", "subfolder": "", "type": "output"}
                    ]
                }
            },
            "status": {"completed": True, "status_str": "success", "messages": []},
        }
    }

    with pytest.raises(VideoGenerationError, match="did not return an MP4 output"):
        adapter.generate(_request())


def test_uploads_image_asset_and_injects_uploaded_filename(adapter, json_recorder, monkeypatch):
    calls, routes = json_recorder
    graph = _custom_workflow()
    _install_workflow(adapter, monkeypatch, graph)
    uploads = []
    monkeypatch.setattr(adapter, "_upload_image", lambda asset: uploads.append(asset) or "upload.png")
    def fetch_bytes(
        url: str,
        *,
        timeout: int | float,
        headers=None,
        cookies=None,
        max_bytes=None,
        trusted_origins=frozenset(),
    ):
        return b"v", "video/mp4"

    monkeypatch.setattr(cva, "fetch_image_bytes", fetch_bytes)
    routes[("GET", "/object_info")] = _object_info_for(graph)
    routes[("POST", "/prompt")] = {"prompt_id": "job-3"}
    routes[("GET", "/history/job-3")] = {
        "job-3": {
            "outputs": {"7": {"videos": [{"filename": "clip.mp4", "subfolder": "", "type": "output"}]}},
            "status": {"completed": True, "status_str": "success", "messages": []},
        }
    }
    image = ResolvedReferenceAsset("first_frame", b"png-bytes", "image/png", "source.png")

    adapter.generate(_request(reference_assets=(image,)))

    assert uploads == [image]
    submit = next(call for call in calls if call["method"] == "POST" and call["url"].endswith("/prompt"))
    assert submit["json"]["prompt"]["8"]["inputs"]["image"] == "upload.png"


def test_h3_reference_image_is_rejected_before_remote_side_effects(adapter, monkeypatch):
    effects: list[str] = []
    resolve_uploaded_image = adapter._resolve_uploaded_image
    monkeypatch.setattr(
        adapter,
        "_validate_required_nodes",
        lambda *_args: effects.append("object_info"),
    )
    monkeypatch.setattr(
        adapter,
        "_resolve_uploaded_image",
        lambda assets: effects.append("resolve") or resolve_uploaded_image(assets),
    )
    monkeypatch.setattr(
        adapter,
        "_upload_image",
        lambda _asset: effects.append("upload") or "upload.png",
    )
    image = ResolvedReferenceAsset(
        "first_frame", b"png-bytes", "image/png", "source.png"
    )

    with pytest.raises(VideoGenerationError, match="input image"):
        adapter.generate(_request(reference_assets=(image,)))

    assert effects == []


def test_upload_uses_multipart_endpoint_and_trusted_origin(adapter, monkeypatch):
    egress_checks = []

    response = Mock(spec=httpx.Response)
    response.is_redirect = False
    response.json.return_value = {"name": "input.png", "subfolder": "upload"}
    client = create_autospec(httpx.Client, instance=True)
    client.__enter__.return_value = client
    client.post.return_value = response

    def fake_create_client(timeout=None, *, follow_redirects=False):
        assert timeout == 30
        assert follow_redirects is False
        return client

    monkeypatch.setattr(cva, "create_client", fake_create_client)
    monkeypatch.setattr(
        cva,
        "_validate_egress_or_raise",
        lambda url, *, trusted_origins: egress_checks.append((url, trusted_origins)),
    )
    asset = ResolvedReferenceAsset("first_frame", b"png-bytes", "image/png", "source.png")

    uploaded = adapter._upload_image(asset)

    assert uploaded == "upload/input.png"
    client.post.assert_called_once_with(
        "http://127.0.0.1:8188/upload/image",
        files={"image": ("source.png", b"png-bytes", "image/png")},
        data={"overwrite": "true"},
    )
    assert egress_checks == [
        ("http://127.0.0.1:8188/upload/image", frozenset({"127.0.0.1"}))
    ]


@pytest.mark.parametrize(
    ("name", "subfolder"),
    [
        ("../input.png", "upload"),
        ("nested/input.png", "upload"),
        (r"nested\input.png", "upload"),
        (".", "upload"),
        ("input.png", "../escape"),
        ("input.png", "/absolute"),
        ("input.png", "nested//upload"),
        ("input.png", "nested/./upload"),
        ("input.png", "nested/../upload"),
        ("input.png", r"nested\upload"),
        ("input.png", 0),
    ],
)
def test_upload_rejects_unsafe_server_returned_paths(
    adapter, monkeypatch, name, subfolder
):
    response = Mock(spec=httpx.Response)
    response.is_redirect = False
    response.json.return_value = {"name": name, "subfolder": subfolder}
    client = create_autospec(httpx.Client, instance=True)
    client.__enter__.return_value = client
    client.post.return_value = response
    monkeypatch.setattr(cva, "create_client", lambda **_kwargs: client)
    monkeypatch.setattr(cva, "_validate_egress_or_raise", lambda *_args, **_kwargs: None)
    asset = ResolvedReferenceAsset(
        "first_frame", b"png-bytes", "image/png", "source.png"
    )

    with pytest.raises(VideoGenerationError, match="unsafe path"):
        adapter._upload_image(asset)


def test_upload_accepts_safe_nested_server_subfolder(adapter, monkeypatch):
    response = Mock(spec=httpx.Response)
    response.is_redirect = False
    response.json.return_value = {
        "name": "input.png",
        "subfolder": "nested/upload",
    }
    client = create_autospec(httpx.Client, instance=True)
    client.__enter__.return_value = client
    client.post.return_value = response
    monkeypatch.setattr(cva, "create_client", lambda **_kwargs: client)
    monkeypatch.setattr(cva, "_validate_egress_or_raise", lambda *_args, **_kwargs: None)
    asset = ResolvedReferenceAsset(
        "first_frame", b"png-bytes", "image/png", "source.png"
    )

    assert adapter._upload_image(asset) == "nested/upload/input.png"


def test_missing_required_classes_fail_before_prompt(adapter, json_recorder, monkeypatch):
    calls, routes = json_recorder
    graph = _custom_workflow()
    _install_workflow(adapter, monkeypatch, graph)
    routes[("GET", "/object_info")] = {"CLIPTextEncode": {}, "KSampler": {}}

    with pytest.raises(VideoBackendUnavailableError, match="EmptyLatentImage.*LoadImage.*OtherNode.*VHS_VideoCombine.*VideoFrames"):
        adapter.generate(_request())
    assert not any(call["url"].endswith("/prompt") for call in calls)


def test_cancellation_interrupts_and_stops(adapter, json_recorder, monkeypatch):
    event = threading.Event()
    calls, routes = json_recorder
    graph = _h3_workflow()
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
    assert adapter._load_workflow("minimax_h3_t2v.json")["105:104"]["class_type"] == "MiniMaxH3ImageToVideo"
    with pytest.raises(VideoGenerationError, match="workflow.*JSON"):
        adapter._load_workflow("chosen.txt")
    with pytest.raises(VideoGenerationError, match="workflow.*path"):
        adapter._load_workflow("../chosen.json")


def test_workflow_read_uses_central_path_validation_result(
    adapter, monkeypatch, tmp_path
):
    user_dir = tmp_path / "data"
    workflow_root = user_dir / "video_workflows"
    workflow_root.mkdir(parents=True)
    validated = workflow_root / "validated.json"
    validated.write_text('{"safe": {"class_type": "Validated"}}')
    calls = []
    monkeypatch.setattr(cva, "get_user_data_dir", lambda: user_dir)

    def fake_validate_path(user_path, base_directory, **kwargs):
        calls.append((user_path, base_directory, kwargs))
        return validated

    monkeypatch.setattr(cva, "validate_path", fake_validate_path, raising=False)

    assert adapter._load_workflow("configured.json") == {
        "safe": {"class_type": "Validated"}
    }
    assert calls == [
        (
            "configured.json",
            workflow_root,
            {"redact_paths": True},
        )
    ]


def test_workflow_resolution_rejects_user_workflow_symlink_escape(adapter, monkeypatch, tmp_path):
    user_dir = tmp_path / "data"
    workflows = user_dir / "video_workflows"
    workflows.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text('{"outside": {"class_type": "Outside"}}')
    (workflows / "escape.json").symlink_to(outside)
    monkeypatch.setattr(cva, "get_user_data_dir", lambda: user_dir)

    with pytest.raises(VideoGenerationError, match="symlink"):
        adapter._load_workflow("escape.json")


def test_workflow_resolution_rejects_symlinked_workflow_parent(adapter, monkeypatch, tmp_path):
    user_dir = tmp_path / "data"
    user_dir.mkdir()
    outside = tmp_path / "outside-workflows"
    outside.mkdir()
    (outside / "escape.json").write_text('{"outside": {"class_type": "Outside"}}')
    (user_dir / "video_workflows").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(cva, "get_user_data_dir", lambda: user_dir)

    with pytest.raises(VideoGenerationError, match="escapes"):
        adapter._load_workflow("escape.json")


def test_unsupported_reference_and_bad_output_are_clear(adapter, json_recorder, monkeypatch):
    graph = _h3_workflow()
    _install_workflow(adapter, monkeypatch, graph)
    unsupported = ResolvedReferenceAsset("reference_video", b"video", "video/mp4", "source.mp4")
    with pytest.raises(VideoGenerationError, match="image.*first_frame"):
        adapter.generate(_request(reference_assets=(unsupported,)))

    calls, routes = json_recorder
    def fetch_bytes(
        url: str,
        *,
        timeout: int | float,
        headers=None,
        cookies=None,
        max_bytes=None,
        trusted_origins=frozenset(),
    ):
        return b"v", "video/mp4"

    monkeypatch.setattr(cva, "fetch_image_bytes", fetch_bytes)
    routes[("GET", "/object_info")] = _object_info_for(graph)
    routes[("POST", "/prompt")] = {"prompt_id": "job-5"}
    routes[("GET", "/history/job-5")] = {"job-5": {"outputs": {"save": {"images": [{"filename": "plain.png", "subfolder": "", "type": "output"}]}}}}
    routes[("GET", "/history/job-5")]["job-5"]["status"] = {
        "completed": True,
        "status_str": "success",
        "messages": [],
    }
    with pytest.raises(VideoGenerationError, match="no matching canonical video output"):
        adapter.generate(_request())


@pytest.mark.parametrize(
    "outputs",
    [
        {},
        {"preview": {"images": [{"filename": "partial.png", "subfolder": "", "type": "output"}]}},
    ],
)
def test_terminal_history_failure_is_not_masked_by_empty_or_partial_output(adapter, outputs):
    graph = _h3_workflow()
    history = {
        "job-6": {
            "outputs": outputs,
            "status": {
                "completed": True,
                "status_str": "error",
                "messages": [["execution_error", {"exception_message": "model unavailable"}]],
            },
        }
    }

    with pytest.raises(VideoGenerationError, match="execution failed: model unavailable"):
        adapter._find_output_descriptor(history, "job-6", graph, "mp4")


def test_output_selection_uses_save_video_node_not_preview(adapter):
    graph = _h3_workflow()
    history = {
        "job": {
            "outputs": {
                "preview": {
                    "images": [
                        {"filename": "preview.png", "subfolder": "", "type": "temp"}
                    ]
                },
                "save": {
                    "videos": [
                        {"filename": "clip.mp4", "subfolder": "video", "type": "output"}
                    ]
                },
            },
            "status": {"completed": True, "status_str": "success", "messages": []},
        }
    }

    assert adapter._find_output_descriptor(history, "job", graph, "mp4")["filename"] == "clip.mp4"


def test_save_video_output_accepts_arbitrary_list_collection(adapter):
    graph = _h3_workflow()
    history = {
        "job": {
            "outputs": {
                "save": {
                    "files": [
                        {"filename": "clip.mp4", "subfolder": "video", "type": "output"}
                    ]
                }
            },
            "status": {"completed": True, "status_str": "success", "messages": []},
        }
    }

    assert adapter._find_output_descriptor(history, "job", graph, "mp4") == {
        "filename": "clip.mp4",
        "subfolder": "video",
        "type": "output",
    }


@pytest.mark.parametrize(
    "descriptor",
    [
        None,
        {},
        {"filename": ""},
        {"filename": "clip.mp4", "subfolder": {}},
        {"filename": "clip.mp4", "type": []},
        {"filename": "clip.mp4"},
        {"filename": "still.png", "subfolder": "", "type": "output"},
    ],
)
def test_terminal_output_rejects_malformed_or_unsupported_descriptors(
    adapter, descriptor
):
    graph = _h3_workflow()
    history = {
        "job": {
            "outputs": {"save": {"files": [descriptor]}},
            "status": {"completed": True, "status_str": "success", "messages": []},
        }
    }

    with pytest.raises(VideoGenerationError, match="no matching canonical"):
        adapter._find_output_descriptor(history, "job", graph, "mp4")


@pytest.mark.parametrize("requested", ["mp4", "webm"])
def test_output_selection_uses_only_request_matching_canonical_descriptor(
    adapter, requested
):
    graph = {
        "unrelated": {"class_type": "OtherNode", "inputs": {}},
        "output": {"class_type": "VHS_VideoCombine", "inputs": {}},
    }
    history = {
        "job": {
            "outputs": {
                "unrelated": {
                    "files": [
                        {"filename": f"wrong-node.{requested}", "subfolder": "", "type": "output"}
                    ]
                },
                "output": {
                    "arbitrary_collection": [
                        {"filename": "animated.webp", "subfolder": "", "type": "output"},
                        {"filename": "first.mp4", "subfolder": "", "type": "output"},
                        {"filename": "second.webm", "subfolder": "", "type": "output"},
                        {"filename": "movie.mov", "subfolder": "", "type": "output"},
                    ]
                }
            },
            "status": {"completed": True, "status_str": "success", "messages": []},
        }
    }

    expected = "first.mp4" if requested == "mp4" else "second.webm"
    assert adapter._find_output_descriptor(history, "job", graph, requested)["filename"] == expected


def test_output_selection_waits_for_explicit_terminal_success(adapter):
    graph = {"output": {"class_type": "SaveVideo", "inputs": {}}}
    history = {
        "job": {
            "outputs": {
                "output": {
                    "videos": [
                        {"filename": "partial.mp4", "subfolder": "", "type": "output"}
                    ]
                }
            },
            "status": {"completed": False, "status_str": "running", "messages": []},
        }
    }

    assert adapter._find_output_descriptor(history, "job", graph, "mp4") is None


@pytest.mark.parametrize("reverse_order", [False, True])
def test_output_selection_rejects_multiple_matching_final_outputs_boundedly(
    adapter, reverse_order
):
    node_ids = ["save-a", "save-b"]
    if reverse_order:
        node_ids.reverse()
    graph = {
        node_id: {"class_type": "SaveVideo", "inputs": {}} for node_id in node_ids
    }
    output_items = [
        (
            "save-a",
            {
                "videos": [
                    {"filename": "PRIVATE-A.mp4", "subfolder": "", "type": "output"},
                    {"filename": "temp.mp4", "subfolder": "", "type": "temp"},
                ]
            },
        ),
        (
            "save-b",
            {
                "files": [
                    {"filename": "PRIVATE-B.mp4", "subfolder": "", "type": "output"}
                ]
            },
        ),
    ]
    if reverse_order:
        output_items.reverse()
        for _node_id, collections in output_items:
            collections["files" if "files" in collections else "videos"].reverse()
    history = {
        "job": {
            "outputs": dict(output_items),
            "status": {"completed": True, "status_str": "success", "messages": []},
        }
    }

    with pytest.raises(VideoGenerationError, match="multiple matching canonical") as exc_info:
        adapter._find_output_descriptor(history, "job", graph, "mp4")

    assert "PRIVATE-A" not in str(exc_info.value)
    assert "PRIVATE-B" not in str(exc_info.value)


@pytest.mark.parametrize("reverse_order", [False, True])
def test_output_selection_ignores_temp_descriptor_with_matching_suffix(
    adapter, reverse_order
):
    descriptors = [
        {"filename": "temp.mp4", "subfolder": "", "type": "temp"},
        {"filename": "final.mp4", "subfolder": "video", "type": "output"},
    ]
    if reverse_order:
        descriptors.reverse()
    graph = {"save": {"class_type": "SaveVideo", "inputs": {}}}
    history = {
        "job": {
            "outputs": {"save": {"videos": descriptors}},
            "status": {"completed": True, "status_str": "success", "messages": []},
        }
    }

    assert adapter._find_output_descriptor(history, "job", graph, "mp4") == {
        "filename": "final.mp4",
        "subfolder": "video",
        "type": "output",
    }


def test_terminal_success_without_media_fails_without_waiting(adapter):
    graph = _h3_workflow()
    history = {
        "job-7": {
            "outputs": {
                "preview": {
                    "images": [
                        {"filename": "preview.webp", "subfolder": "", "type": "temp"}
                    ]
                }
            },
            "status": {"completed": True, "status_str": "success", "messages": []},
        }
    }

    with pytest.raises(VideoGenerationError, match="no matching canonical video output"):
        adapter._find_output_descriptor(history, "job-7", graph, "mp4")


def test_poll_timeout_is_bounded_by_remaining_deadline(adapter, monkeypatch):
    adapter._config.comfyui_timeout_seconds = 2
    now = [0.0]
    request_timeouts: list[float] = []
    sleeps: list[float] = []

    def fake_fetch_json(*, timeout, **_kwargs):
        request_timeouts.append(timeout)
        now[0] += min(0.6, timeout)
        return {
            "job-timeout": {
                "outputs": {
                    "preview": {
                        "images": [
                            {
                                "filename": "preview.png",
                                "subfolder": "",
                                "type": "temp",
                            }
                        ]
                    }
                },
                "status": {
                    "completed": False,
                    "status_str": "running",
                    "messages": [],
                },
            }
        }

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(cva, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(cva.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(cva.time, "sleep", fake_sleep)

    with pytest.raises(VideoGenerationError, match="timed out"):
        adapter._poll_for_output(
            "http://127.0.0.1:8188",
            "job-timeout",
            None,
            _h3_workflow(),
            "mp4",
        )

    assert now[0] == pytest.approx(2.0)
    assert request_timeouts == pytest.approx([2.0, 0.4])
    assert sleeps == pytest.approx([1.0])


def test_h3_preparation_applies_request_and_reports_effective_values(adapter):
    prepared = adapter._parameterize_workflow(
        _h3_workflow(),
        _request(seed=41, width=1280, height=704, duration_seconds=6, fps=24, ratio="16:9"),
        None,
    )

    assert prepared.graph["gen"]["inputs"]["prompt"] == "a lighthouse in a storm"
    assert prepared.graph["gen"]["inputs"]["width"] == 1280
    assert prepared.graph["gen"]["inputs"]["height"] == 704
    assert prepared.graph["seed"]["inputs"]["noise_seed"] == 41
    assert prepared.graph["duration"]["inputs"]["value"] == 6
    assert prepared.graph["gen"]["inputs"]["length"] == ["expr", 1]
    assert (prepared.width, prepared.height, prepared.duration_seconds, prepared.fps) == (1280, 704, 6.0, 24.0)
    assert prepared.resolved_seed == 41


@pytest.mark.parametrize("fps", [12, 23, 25, 30])
def test_h3_native_fps_rejects_non_24(adapter, fps):
    with pytest.raises(VideoGenerationError, match="native FPS.*24"):
        adapter._parameterize_workflow(_h3_workflow(), _request(fps=fps), None)


def test_requested_value_without_eligible_control_fails(adapter):
    graph = _h3_workflow()
    graph["gen"]["inputs"]["width"] = ["linked", 0]

    with pytest.raises(VideoGenerationError, match="width.*Prompt Width Height"):
        adapter._parameterize_workflow(graph, _request(width=1280), None)


def test_h3_rejects_incompatible_ratio_and_format(adapter):
    with pytest.raises(VideoGenerationError, match="ratio"):
        adapter._parameterize_workflow(_h3_workflow(), _request(ratio="1:1"), None)
    with pytest.raises(VideoGenerationError, match="MP4"):
        adapter._parameterize_workflow(_h3_workflow(), _request(video_format="webm"), None)


def test_h3_defaults_are_reported_without_modifying_graph(adapter):
    workflow = _h3_workflow()
    original = copy.deepcopy(workflow)

    prepared = adapter._parameterize_workflow(workflow, _request(), None)

    assert (prepared.width, prepared.height, prepared.duration_seconds, prepared.fps, prepared.resolved_seed) == (864, 480, 5.0, 24.0, 0)
    assert workflow == original
    assert prepared.graph is not workflow


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("width", 0),
        ("width", 865),
        ("width", 864.0),
        ("height", -32),
        ("height", 481),
        ("height", 480.0),
    ],
)
def test_h3_rejects_invalid_effective_dimension_defaults(adapter, field, value):
    graph = _h3_workflow()
    graph["gen"]["inputs"][field] = value

    with pytest.raises(
        VideoGenerationError,
        match=rf"{field}.*positive integer.*multiple of 32",
    ):
        adapter._parameterize_workflow(graph, _request(), None)


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_h3_rejects_invalid_effective_duration_defaults(adapter, duration):
    graph = _h3_workflow()
    graph["duration"]["inputs"]["value"] = duration

    with pytest.raises(VideoGenerationError, match="duration.*finite.*greater than 0"):
        adapter._parameterize_workflow(graph, _request(), None)


@pytest.mark.parametrize(
    ("node_id", "field", "value", "match"),
    [
        ("seed", "noise_seed", -1, "seed.*non-negative integer"),
        ("seed", "noise_seed", 1.5, "seed.*non-negative integer"),
        ("video", "fps", 23, "native FPS.*24"),
    ],
)
def test_h3_rejects_invalid_omitted_seed_and_fps_defaults(
    adapter, node_id, field, value, match
):
    graph = _h3_workflow()
    graph[node_id]["inputs"][field] = value

    with pytest.raises(VideoGenerationError, match=match):
        adapter._parameterize_workflow(graph, _request(), None)


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_h3_rejects_invalid_supplied_duration(adapter, duration):
    with pytest.raises(VideoGenerationError, match="duration.*finite.*greater than 0"):
        adapter._parameterize_workflow(
            _h3_workflow(),
            _request(duration_seconds=duration),
            None,
        )


def test_generic_custom_workflow_keeps_documented_title_controls(adapter):
    prepared = adapter._parameterize_workflow(
        _custom_workflow(),
        _request(seed=41, width=1280, height=704, duration_seconds=6, fps=24),
        "safe-input.png",
    )

    assert prepared.graph["1"]["inputs"]["text"] == "a lighthouse in a storm"
    assert prepared.graph["2"]["inputs"]["text"] == ""
    assert prepared.graph["3"]["inputs"]["seed"] == 41
    assert prepared.graph["4"]["inputs"]["width"] == 1280
    assert prepared.graph["5"]["inputs"]["height"] == 704
    assert prepared.graph["6"]["inputs"]["num_frames"] == 144
    assert prepared.graph["10"]["inputs"]["fps"] == 24
    assert prepared.graph["8"]["inputs"]["image"] == "safe-input.png"
    assert (prepared.width, prepared.height, prepared.duration_seconds, prepared.fps) == (1280, 704, 6.0, 24.0)


def test_h3_seed_minus_one_resolves_once_and_rejects_lower_values(adapter, monkeypatch):
    monkeypatch.setattr(cva.secrets, "randbelow", lambda upper: 73)
    prepared = adapter._parameterize_workflow(_h3_workflow(), _request(seed=-1), None)

    assert prepared.graph["seed"]["inputs"]["noise_seed"] == 73
    assert prepared.resolved_seed == 73
    with pytest.raises(VideoGenerationError, match="-1 or a non-negative"):
        adapter._parameterize_workflow(_h3_workflow(), _request(seed=-2), None)


@pytest.mark.parametrize("ratio", ["adaptive", "1:1", "ratio"])
def test_h3_rejects_unsupported_ratios(adapter, ratio):
    with pytest.raises(VideoGenerationError, match="ratio"):
        adapter._parameterize_workflow(_h3_workflow(), _request(ratio=ratio), None)


def test_h3_allows_the_exact_three_percent_ratio_boundary(adapter):
    ratio = f"{864 / 1.03}:480"

    prepared = adapter._parameterize_workflow(_h3_workflow(), _request(ratio=ratio), None)

    assert (prepared.width, prepared.height) == (864, 480)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("width", 0),
        ("width", 1279),
        ("width", 1280.0),
        ("height", -32),
        ("height", 703),
        ("height", 704.0),
    ],
)
def test_h3_rejects_invalid_supplied_dimensions(adapter, field, value):
    with pytest.raises(
        VideoGenerationError,
        match=f"{field}.*positive integer.*multiple of 32",
    ):
        adapter._parameterize_workflow(_h3_workflow(), _request(**{field: value}), None)


@pytest.mark.parametrize(
    ("node_id", "title", "match"),
    [
        ("gen", "Prompt Width Hight", "prompt.*Prompt Width Height"),
        ("duration", "Clip Duration", "duration.*Duration"),
        ("video", "FPS", "native FPS.*Native FPS"),
    ],
)
def test_h3_requires_exact_documented_control_titles(adapter, node_id, title, match):
    graph = _h3_workflow()
    graph[node_id]["_meta"]["title"] = title

    with pytest.raises(VideoGenerationError, match=match):
        adapter._parameterize_workflow(graph, _request(), None)


@pytest.mark.parametrize("duplicate_title", ["Prompt Width Height", "Unrelated"])
def test_h3_rejects_ambiguous_generation_nodes(adapter, duplicate_title):
    graph = _h3_workflow()
    graph["duplicate-generation"] = copy.deepcopy(graph["gen"])
    graph["duplicate-generation"]["_meta"]["title"] = duplicate_title

    with pytest.raises(
        VideoGenerationError,
        match="prompt.*Prompt Width Height.*MiniMaxH3ImageToVideo.*exactly one",
    ):
        adapter._parameterize_workflow(graph, _request(), None)


def test_h3_rejects_wrong_class_generation_title_decoy(adapter):
    graph = _h3_workflow()
    graph["generation-decoy"] = copy.deepcopy(graph["gen"])
    graph["generation-decoy"]["class_type"] = "OtherGenerationNode"

    with pytest.raises(
        VideoGenerationError,
        match="prompt.*Prompt Width Height.*MiniMaxH3ImageToVideo.*OtherGenerationNode",
    ):
        adapter._parameterize_workflow(graph, _request(), None)


@pytest.mark.parametrize(
    ("node_id", "control", "field", "title", "expected_class"),
    [
        ("seed", "seed", "seed", "Seed", "RandomNoise"),
        ("duration", "duration", "duration", "Duration", "PrimitiveFloat"),
        ("video", "native_fps", "native FPS", "Native FPS", "CreateVideo"),
    ],
)
def test_h3_rejects_duplicate_support_controls(
    adapter, node_id, control, field, title, expected_class
):
    graph = _h3_workflow()
    graph[f"duplicate-{control}"] = copy.deepcopy(graph[node_id])

    with pytest.raises(
        VideoGenerationError,
        match=rf"{field}.*{title}.*{expected_class}.*exactly one",
    ):
        adapter._parameterize_workflow(graph, _request(), None)


@pytest.mark.parametrize(
    ("node_id", "control", "field", "title", "expected_class"),
    [
        ("seed", "seed", "seed", "Seed", "RandomNoise"),
        ("duration", "duration", "duration", "Duration", "PrimitiveFloat"),
        ("video", "native_fps", "native FPS", "Native FPS", "CreateVideo"),
    ],
)
def test_h3_rejects_wrong_class_support_title_decoys(
    adapter, node_id, control, field, title, expected_class
):
    graph = _h3_workflow()
    graph[f"decoy-{control}"] = copy.deepcopy(graph[node_id])
    graph[f"decoy-{control}"]["class_type"] = "WrongControlNode"

    with pytest.raises(
        VideoGenerationError,
        match=rf"{field}.*{title}.*{expected_class}.*WrongControlNode",
    ):
        adapter._parameterize_workflow(graph, _request(), None)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("negative_prompt", "safe negative"),
        ("model", "safe-model"),
        ("sampler", "safe-sampler"),
        ("steps", 8),
        ("cfg_scale", 4.0),
        ("extra_params", {"safe": "value"}),
    ],
)
def test_h3_rejects_unsupported_explicit_controls(adapter, field, value):
    with pytest.raises(VideoGenerationError, match=field.replace("_", " ")):
        adapter._parameterize_workflow(_h3_workflow(), _request(**{field: value}), None)


def test_h3_rejects_an_input_image_without_a_documented_control(adapter):
    with pytest.raises(VideoGenerationError, match="input image"):
        adapter._parameterize_workflow(_h3_workflow(), _request(), "safe-input.png")


def test_title_controls_reject_titles_with_unrelated_words(adapter):
    workflow = {
        "preview": {
            "class_type": "Text",
            "inputs": {"text": "preview"},
            "_meta": {"title": "Prompt Preview"},
        },
        "reference": {
            "class_type": "Sampler",
            "inputs": {"seed": 7},
            "_meta": {"title": "Seed Reference"},
        },
        "notes": {
            "class_type": "Notes",
            "inputs": {"width": 320},
            "_meta": {"title": "Output Width Notes"},
        },
        "image_preview": {
            "class_type": "Image",
            "inputs": {"image": "preview.png"},
            "_meta": {"title": "Input Image Preview"},
        },
    }

    assert all(adapter._title_controls(node) == set() for node in workflow.values())
