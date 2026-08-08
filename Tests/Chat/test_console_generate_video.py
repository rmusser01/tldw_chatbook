"""/generate-video helpers: parsing, cost text, blocking generation (task-3401.5)."""

import threading

import pytest

from tldw_chatbook.Chat.console_generate_video import (
    estimate_video_cost_text,
    is_paid_backend,
    parse_generate_video_args,
    run_video_generation,
)
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
