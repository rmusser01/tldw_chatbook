"""/generate-video helpers: parsing, cost text, blocking generation (task-3401.5)."""

import asyncio
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_generate_video import (
    estimate_video_cost_text,
    is_paid_backend,
    parse_generate_video_args,
    run_video_generation,
)
from tldw_chatbook.Video_Generation.adapters import comfyui_video_adapter as cva
from tldw_chatbook.Video_Generation.video_store import VideoStore


# -- parsing ------------------------------------------------------------------


def test_parse_plain_prompt():
    args = parse_generate_video_args("a kite over the harbor")
    assert args.backend is None
    assert args.prompt == "a kite over the harbor"


def test_parse_backend_token():
    args = parse_generate_video_args(":minimax a kite")
    assert args.backend == "minimax"
    assert args.prompt == "a kite"


def test_parse_bare_colon_stays_prompt():
    args = parse_generate_video_args(": a kite")
    assert args.backend is None
    assert args.prompt == ": a kite"


def test_parse_empty():
    args = parse_generate_video_args("   ")
    assert args.backend is None
    assert args.prompt == ""


# -- cost gate text -------------------------------------------------------------


def test_is_paid_backend():
    assert is_paid_backend("minimax")
    assert is_paid_backend(" MiniMax ")  # case/whitespace tolerant
    assert not is_paid_backend("comfyui")
    assert not is_paid_backend("stable_diffusion_cpp")


def test_estimate_video_cost_text_shapes():
    paid = estimate_video_cost_text("minimax", 6)
    assert "6s" in paid and "billed per generated second" in paid
    local = estimate_video_cost_text("comfyui", 6)
    assert "no per-clip charge" in local


# -- blocking generation --------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry():
    from tldw_chatbook.Video_Generation import adapter_registry as r
    r.reset_registry()
    yield
    r.reset_registry()


def _register_fake(result_content: bytes = b"vid-bytes", **result_kwargs):
    from tldw_chatbook.Video_Generation.adapter_registry import get_registry
    from tldw_chatbook.Video_Generation.adapters.base import VideoGenResult

    class FakeAdapter:
        name = "fakevid"
        supported_formats = {"mp4"}

        def generate(self, request):
            return VideoGenResult(
                content=result_content, content_type="video/mp4",
                bytes_len=len(result_content), **result_kwargs,
            )

    registry = get_registry()
    registry._enabled_backends = ["fakevid"]
    registry._default_backend = "fakevid"
    registry.register_adapter("fakevid", FakeAdapter)


def _register_capturing_comfyui(captured_requests: list) -> None:
    from tldw_chatbook.Video_Generation.adapter_registry import get_registry
    from tldw_chatbook.Video_Generation.adapters.base import VideoGenResult

    class CapturingComfyUIAdapter:
        name = "comfyui"
        supported_formats = {"mp4"}

        def generate(self, request):
            captured_requests.append(request)
            return VideoGenResult(
                content=b"video",
                content_type="video/mp4",
                bytes_len=5,
            )

    registry = get_registry()
    registry._enabled_backends = ["comfyui"]
    registry._default_backend = "comfyui"
    registry.register_adapter("comfyui", CapturingComfyUIAdapter)


def _install_custom_workflow(
    tmp_path: Path, monkeypatch, *, class_type: str = "MiniMaxH3ImageToVideo"
) -> None:
    data_root = tmp_path / "data"
    workflow_dir = data_root / "video_workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "custom-cinematic.json").write_text(
        json.dumps(
            {
                "generation": {
                    "class_type": class_type,
                    "inputs": {},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cva, "get_user_data_dir", lambda: data_root)
    monkeypatch.setattr(
        cva,
        "get_video_generation_config",
        lambda: SimpleNamespace(
            comfyui_default_workflow="custom-cinematic.json",
            comfyui_base_url="http://127.0.0.1:8188",
            comfyui_timeout_seconds=30,
            download_max_mb=500,
        ),
    )
    monkeypatch.setattr(
        cva,
        "fetch_json",
        lambda **_kwargs: pytest.fail("classification must not call the network"),
    )


def test_run_video_generation_saves_and_returns_metadata(tmp_path):
    _register_fake(resolved_model="FakeH3", duration_seconds=6.0, width=1920, height=1080)
    store = VideoStore(root=tmp_path / "gv")
    meta, path = run_video_generation(
        backend="fakevid",
        prompt="A Red Dragon",
        message_id="msg-42",
        video_store=store,
    )
    assert path.read_bytes() == b"vid-bytes"
    assert path.parent.name == "msg-42"
    assert meta.name == "a-red-dragon"
    assert meta.backend == "fakevid"
    assert meta.model == "FakeH3"  # resolved model wins
    assert meta.duration_seconds == 6.0
    assert meta.width == 1920 and meta.height == 1080
    assert store.resolve("msg-42", "a-red-dragon") == path


def test_run_video_generation_cancel_event_threaded_when_supported(tmp_path):
    from tldw_chatbook.Video_Generation.adapter_registry import get_registry
    from tldw_chatbook.Video_Generation.adapters.base import VideoGenResult

    received: list = []

    class CancelAwareAdapter:
        name = "fakevid"
        supported_formats = {"mp4"}

        def generate(self, request, *, cancel_event=None):
            received.append(cancel_event)
            return VideoGenResult(content=b"v", content_type="video/mp4", bytes_len=1)

    registry = get_registry()
    registry._enabled_backends = ["fakevid"]
    registry.register_adapter("fakevid", CancelAwareAdapter)

    event = threading.Event()
    run_video_generation(
        backend="fakevid", prompt="p", message_id="m1",
        cancel_event=event, video_store=VideoStore(root=tmp_path / "gv"),
    )
    assert received == [event]


def test_run_video_generation_unknown_backend_raises(tmp_path):
    from tldw_chatbook.Video_Generation.exceptions import VideoGenerationError

    with pytest.raises(VideoGenerationError, match="not enabled/available"):
        run_video_generation(
            backend="nope", prompt="p", message_id="m1",
            video_store=VideoStore(root=tmp_path / "gv"),
        )


def test_run_video_generation_invalid_request_never_writes(tmp_path):
    _register_fake()
    from tldw_chatbook.Video_Generation.exceptions import VideoGenerationError

    store = VideoStore(root=tmp_path / "gv")
    with pytest.raises(VideoGenerationError, match="Invalid video generation request"):
        run_video_generation(
            backend="fakevid", prompt="p", message_id="m1",
            duration_seconds=999,  # over the configured cap
            video_store=store,
        )
    assert list(store.iter_stored()) == []


def test_custom_named_h3_dispatch_preserves_positive_suffix_and_strips_style_negative(
    tmp_path, monkeypatch
):
    _install_custom_workflow(tmp_path, monkeypatch)
    captured_requests: list = []
    _register_capturing_comfyui(captured_requests)

    meta, _path = run_video_generation(
        backend="comfyui",
        prompt="base prompt, cinematic positive suffix",
        negative_prompt="style-derived negative",
        style_negative_prompt=True,
        message_id="styled-message",
        video_store=VideoStore(root=tmp_path / "videos"),
    )

    assert len(captured_requests) == 1
    assert captured_requests[0].prompt == "base prompt, cinematic positive suffix"
    assert captured_requests[0].negative_prompt is None
    assert meta.prompt == "base prompt, cinematic positive suffix"
    assert meta.negative_prompt == ""


def test_custom_named_h3_dispatch_keeps_explicit_programmatic_negative(
    tmp_path, monkeypatch
):
    _install_custom_workflow(tmp_path, monkeypatch)
    from tldw_chatbook.Video_Generation.adapter_registry import get_registry
    from tldw_chatbook.Video_Generation.exceptions import VideoGenerationError

    registry = get_registry()
    registry._enabled_backends = ["comfyui"]
    registry._default_backend = "comfyui"
    registry.register_adapter("comfyui", cva.ComfyUIVideoAdapter)
    monkeypatch.setattr(
        cva.ComfyUIVideoAdapter,
        "_validate_required_nodes",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        cva.ComfyUIVideoAdapter,
        "_queue_prompt",
        lambda *_args: pytest.fail("explicit negative must be rejected before queue"),
    )

    with pytest.raises(VideoGenerationError, match="negative prompt"):
        run_video_generation(
            backend="comfyui",
            prompt="programmatic prompt",
            negative_prompt="explicit programmatic negative",
            style_negative_prompt=False,
            message_id="programmatic-message",
            video_store=VideoStore(root=tmp_path / "videos"),
        )


def test_custom_non_h3_dispatch_keeps_style_negative(tmp_path, monkeypatch):
    _install_custom_workflow(tmp_path, monkeypatch, class_type="GenericVideoNode")
    captured_requests: list = []
    _register_capturing_comfyui(captured_requests)

    run_video_generation(
        backend="comfyui",
        prompt="base prompt, cinematic positive suffix",
        negative_prompt="style-derived negative",
        style_negative_prompt=True,
        message_id="generic-message",
        video_store=VideoStore(root=tmp_path / "videos"),
    )

    assert captured_requests[0].negative_prompt == "style-derived negative"


@pytest.mark.asyncio
async def test_chat_screen_dispatch_marks_template_negative_as_style_derived(monkeypatch):
    from tldw_chatbook.UI.Screens import chat_screen as chat_screen_module
    from tldw_chatbook.Video_Generation import adapter_registry
    from tldw_chatbook.Video_Generation import video_templates

    captured_dispatch: dict = {}
    appended_messages: list = []

    class FakeStore:
        workspace_context = SimpleNamespace(active_workspace_id="workspace")

        def ensure_session(self, **_kwargs):
            return SimpleNamespace(id="session")

        def append_video_message(self, *args, **kwargs):
            appended_messages.append((args, kwargs))

    class FakeRegistry:
        @staticmethod
        def resolve_backend(backend):
            return object() if backend == "comfyui" else None

    async def fake_to_thread(function, **kwargs):
        captured_dispatch["function"] = function
        captured_dispatch["kwargs"] = kwargs
        return object(), Path("/tmp/generated.mp4")

    async def append_system_message(*_args, **_kwargs):
        return None

    async def sync_ui():
        return None

    monkeypatch.setattr(
        video_templates,
        "get_video_template",
        lambda _name: SimpleNamespace(default_params={"duration_seconds": 5}),
    )
    monkeypatch.setattr(
        video_templates,
        "apply_video_template",
        lambda _template, prompt: (
            f"{prompt}, cinematic positive suffix",
            "style-derived negative",
        ),
    )
    monkeypatch.setattr(
        chat_screen_module,
        "get_video_generation_config",
        lambda: SimpleNamespace(
            default_backend="comfyui",
            comfyui_default_workflow="custom-cinematic.json",
            confirm_cost_estimate=False,
        ),
    )
    monkeypatch.setattr(adapter_registry, "get_registry", lambda: FakeRegistry())
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    fake_screen = SimpleNamespace(
        _session=SimpleNamespace(
            _default_console_session_settings=lambda: object()
        ),
        _ensure_console_chat_store=lambda: FakeStore(),
        _append_native_console_system_message=append_system_message,
        _console_videogen_inflight_sessions=lambda: set(),
        _console_composer_or_none=lambda: None,
        _clear_console_composer_draft=lambda: None,
        _console_videogen_cancel_events=lambda: {},
        _ensure_console_video_store=lambda: object(),
        _sync_native_console_chat_ui=sync_ui,
    )

    await chat_screen_module.ChatScreen._console_command_generate_video(
        fake_screen,
        SimpleNamespace(args="@cinematic base prompt"),
    )

    assert captured_dispatch["function"] is chat_screen_module.run_video_generation
    assert captured_dispatch["kwargs"]["prompt"] == (
        "base prompt, cinematic positive suffix"
    )
    assert captured_dispatch["kwargs"]["negative_prompt"] == "style-derived negative"
    assert captured_dispatch["kwargs"]["style_negative_prompt"] is True
    assert appended_messages
