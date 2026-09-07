"""MiniMax-H3 video adapter tests (task-3401.3) -- HTTP layer fully mocked."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Video_Generation.adapters import minimax_video_adapter as mva
from tldw_chatbook.Video_Generation.adapters.base import ResolvedReferenceAsset
from tldw_chatbook.Video_Generation.exceptions import (
    VideoBackendUnavailableError,
    VideoGenerationError,
)
from tldw_chatbook.Video_Generation.worker import build_request


def _fake_config(**overrides):
    base = {
        "minimax_video_api_key": "test-secret-key",
        "minimax_video_base_url": "https://api.minimax.io",
        "minimax_video_default_model": "MiniMax-H3",
        "minimax_video_poll_interval_seconds": 1,
        "minimax_video_timeout_seconds": 30,
        "download_max_mb": 500,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(mva, "get_video_generation_config", lambda: _fake_config())
    return mva.MiniMaxVideoAdapter()


@pytest.fixture
def http_recorder(monkeypatch):
    """Record fetch_json/fetch_image_bytes calls; script responses via `routes`.

    routes maps (method, url-substring) -> response dict (or list of dicts,
    consumed in order) for fetch_json, and ("BYTES", url-substring) ->
    (bytes, content_type) for fetch_image_bytes.
    """
    calls: list[dict] = []
    routes: dict = {}

    def fake_fetch_json(method, url, **kwargs):
        calls.append({"method": method, "url": url, "kwargs": kwargs})
        for (route_method, fragment), value in routes.items():
            if route_method == "BYTES":
                continue
            if method == route_method and fragment in url:
                if isinstance(value, list):
                    return value.pop(0)
                return value
        raise AssertionError(f"unscripted {method} {url}")

    def fake_fetch_bytes(url, **kwargs):
        calls.append({"method": "BYTES", "url": url, "kwargs": kwargs})
        for (route_method, fragment), value in routes.items():
            if route_method == "BYTES" and fragment in url:
                return value
        raise AssertionError(f"unscripted download {url}")

    monkeypatch.setattr(mva, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(mva, "fetch_image_bytes", fake_fetch_bytes)
    return calls, routes


def _request(**overrides):
    kwargs = {"backend": "minimax", "prompt": "a kite over the harbor"}
    kwargs.update(overrides)
    return build_request(**kwargs)


# -- happy path -------------------------------------------------------------


def test_happy_path_submit_poll_download(adapter, http_recorder):
    calls, routes = http_recorder
    routes[("POST", "/v2/video_generation")] = {"task_id": "task-abc123"}
    routes[("GET", "/v2/query/video_generation/")] = {
        "task": {
            "status": "succeeded",
            "content": {"url": "https://cdn.example.com/video.mp4"},
            "video_width": 1920,
            "video_height": 1080,
        }
    }
    routes[("BYTES", "cdn.example.com")] = (b"video-bytes", "video/mp4")

    result = adapter.generate(_request(duration_seconds=6))

    assert result.content == b"video-bytes"
    assert result.content_type == "video/mp4"
    assert result.container == "mp4"
    assert result.bytes_len == len(b"video-bytes")
    assert result.duration_seconds == 6
    assert result.width == 1920 and result.height == 1080
    assert result.resolved_model == "MiniMax-H3"

    submit = calls[0]
    assert submit["method"] == "POST"
    payload = submit["kwargs"]["json"]
    assert payload["model"] == "MiniMax-H3"
    assert payload["content"] == [{"type": "text", "text": "a kite over the harbor"}]
    assert payload["duration"] == 6
    assert payload["ratio"] == "16:9"  # T2V default, never adaptive
    # The configured base host is trusted; the CDN download is NOT.
    assert submit["kwargs"]["trusted_origins"]
    download = calls[-1]
    assert download["method"] == "BYTES"
    assert "trusted_origins" not in download["kwargs"]


def test_poll_waits_through_pending_statuses(adapter, http_recorder, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    calls, routes = http_recorder
    routes[("POST", "/v2/video_generation")] = {"task_id": "task-xyz"}
    routes[("GET", "/v2/query/video_generation/")] = [
        {"task": {"status": "Preparing"}},
        {"task": {"status": "queueing"}},
        {"task": {"status": "PROCESSING"}},
        {"task": {"status": "succeeded", "content": {"url": "https://cdn.example.com/v.mp4"}}},
    ]
    routes[("BYTES", "cdn.example.com")] = (b"v", "video/mp4")

    result = adapter.generate(_request())
    assert result.content == b"v"
    polls = [c for c in calls if c["method"] == "GET" and "/query/" in c["url"]]
    assert len(polls) == 4


def test_resolution_tier_mapping(adapter, http_recorder):
    calls, routes = http_recorder
    routes[("POST", "/v2/video_generation")] = {"task_id": "t1"}
    routes[("GET", "/v2/query/")] = {
        "task": {"status": "succeeded", "content": {"url": "https://cdn.example.com/v.mp4"}}
    }
    routes[("BYTES", "cdn.example.com")] = (b"v", "video/mp4")

    adapter.generate(_request(width=1280, height=720))
    adapter.generate(_request(width=2560, height=1440))
    submits = [c for c in calls if c["method"] == "POST"]
    assert submits[0]["kwargs"]["json"]["resolution"] == "768P"
    assert submits[1]["kwargs"]["json"]["resolution"] == "2K"


@pytest.mark.parametrize("observed_type", [None, "application/octet-stream", "video/webm"])
def test_file_id_fallback_requires_observed_mp4_mime(
    adapter, http_recorder, observed_type
):
    calls, routes = http_recorder
    routes[("POST", "/v2/video_generation")] = {"task_id": "t2"}
    routes[("GET", "/v2/query/")] = {"task_id": "t2", "status": "Success", "file_id": "98765"}
    routes[("GET", "/v1/files/retrieve")] = {
        "file": {"download_url": "https://cdn.example.com/fallback.mp4"}
    }
    routes[("BYTES", "cdn.example.com")] = (b"fallback-bytes", observed_type)

    with pytest.raises(VideoGenerationError, match="video/mp4 MIME"):
        adapter.generate(_request())
    retrieve = next(c for c in calls if "/v1/files/retrieve" in c["url"])
    assert retrieve["kwargs"]["params"] == {"file_id": "98765"}


def test_file_id_fallback_accepts_exact_normalized_mp4_result(adapter, http_recorder):
    calls, routes = http_recorder
    routes[("POST", "/v2/video_generation")] = {"task_id": "t2-positive"}
    routes[("GET", "/v2/query/")] = {
        "task_id": "t2-positive",
        "status": "Success",
        "file_id": "12345",
    }
    routes[("GET", "/v1/files/retrieve")] = {
        "file": {"download_url": "https://cdn.example.com/fallback.mp4"}
    }
    routes[("BYTES", "cdn.example.com")] = (
        b"fallback-bytes",
        " Video/MP4 ; codecs=avc1 ",
    )

    result = adapter.generate(_request())

    assert result.content == b"fallback-bytes"
    assert result.content_type == "video/mp4"
    assert result.container == "mp4"
    retrieve = next(c for c in calls if "/v1/files/retrieve" in c["url"])
    assert retrieve["kwargs"]["params"] == {"file_id": "12345"}


def test_download_accepts_normalized_mp4_mime_with_parameters(adapter, http_recorder):
    _calls, routes = http_recorder
    routes[("POST", "/v2/video_generation")] = {"task_id": "t-params"}
    routes[("GET", "/v2/query/")] = {
        "task": {
            "status": "succeeded",
            "content": {"url": "https://cdn.example.com/params.mp4"},
        }
    }
    routes[("BYTES", "cdn.example.com")] = (
        b"video",
        " Video/MP4 ; codecs=avc1 ",
    )

    result = adapter.generate(_request())

    assert result.content_type == "video/mp4"
    assert result.container == "mp4"


# -- failures ----------------------------------------------------------------


def test_terminal_failure_status_surfaced(adapter, http_recorder):
    calls, routes = http_recorder
    routes[("POST", "/v2/video_generation")] = {"task_id": "t3"}
    routes[("GET", "/v2/query/")] = {
        "task": {"status": "failed", "error": {"message": "content moderated"}}
    }
    with pytest.raises(VideoGenerationError, match="failed.*content moderated"):
        adapter.generate(_request())


def test_base_resp_error_surfaced_without_key(adapter, http_recorder):
    calls, routes = http_recorder
    routes[("POST", "/v2/video_generation")] = {
        "task_id": "",
        "base_resp": {"status_code": 1008, "status_msg": "insufficient balance"},
    }
    with pytest.raises(VideoGenerationError) as exc_info:
        adapter.generate(_request())
    message = str(exc_info.value)
    assert "1008" in message and "insufficient balance" in message
    assert "test-secret-key" not in message


def test_submit_http_error_never_contains_key(adapter, http_recorder, monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("connection reset")

    # Re-patch over the recorder's fake: submit raises at the transport.
    monkeypatch.setattr(mva, "fetch_json", boom)
    with pytest.raises(VideoGenerationError) as exc_info:
        adapter.generate(_request())
    assert "test-secret-key" not in str(exc_info.value)


def test_missing_api_key(monkeypatch):
    monkeypatch.setattr(
        mva,
        "get_video_generation_config",
        lambda: _fake_config(minimax_video_api_key=None),
    )
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    adapter = mva.MiniMaxVideoAdapter()
    with pytest.raises(VideoBackendUnavailableError, match="api key"):
        adapter.generate(_request())


def test_reference_assets_refused_until_3401_8(adapter):
    asset = ResolvedReferenceAsset(kind="first_frame", content=b"png", mime_type="image/png")
    with pytest.raises(VideoGenerationError, match="3401.8"):
        adapter.generate(_request(reference_assets=(asset,)))


def test_duration_bounds_enforced(adapter):
    with pytest.raises(VideoGenerationError, match="4-15s"):
        adapter.generate(_request(duration_seconds=3))
    with pytest.raises(VideoGenerationError, match="4-15s"):
        adapter.generate(_request(duration_seconds=16))


def test_success_without_url_or_file_id_errors(adapter, http_recorder):
    calls, routes = http_recorder
    routes[("POST", "/v2/video_generation")] = {"task_id": "t4"}
    routes[("GET", "/v2/query/")] = {"task": {"status": "succeeded"}}
    with pytest.raises(VideoGenerationError, match="no download URL"):
        adapter.generate(_request())


def test_invalid_task_id_refused(adapter, http_recorder):
    calls, routes = http_recorder
    routes[("POST", "/v2/video_generation")] = {"task_id": "../../evil"}
    with pytest.raises(VideoGenerationError, match="invalid task_id"):
        adapter.generate(_request())


# -- cancellation / timeout ---------------------------------------------------


def test_cancel_event_stops_polling_and_calls_remote_cancel(adapter, http_recorder):
    import threading

    cancel_event = threading.Event()
    calls, routes = http_recorder
    routes[("POST", "/v2/video_generation")] = {"task_id": "t5"}
    routes[("GET", "/v2/query/")] = {"task": {"status": "processing"}}
    routes[("DELETE", "/v2/video_generation/t5")] = {"task_id": "t5", "action": "cancelled"}

    cancel_event.set()  # already cancelled before the first poll iteration
    with pytest.raises(VideoGenerationError, match="cancelled by user"):
        adapter.generate(_request(), cancel_event=cancel_event)

    deletes = [c for c in calls if c["method"] == "DELETE"]
    assert len(deletes) == 1
    assert "/v2/video_generation/t5" in deletes[0]["url"]


def test_cancel_event_landing_mid_poll(adapter, http_recorder, monkeypatch):
    import threading

    cancel_event = threading.Event()
    calls, routes = http_recorder
    routes[("POST", "/v2/video_generation")] = {"task_id": "t6"}

    def pending_then_cancel(method, url, **kwargs):
        if method == "POST":
            return {"task_id": "t6"}
        if method == "GET" and "/query/" in url:
            cancel_event.set()  # user pressed stop while the task was running
            return {"task": {"status": "processing"}}
        if method == "DELETE":
            return {"task_id": "t6", "action": "cancelled"}
        raise AssertionError(f"unexpected {method} {url}")

    # Overlay the ordered fake for poll/delete (re-patches over the recorder).
    monkeypatch.setattr(mva, "fetch_json", pending_then_cancel)
    with pytest.raises(VideoGenerationError, match="cancelled by user"):
        adapter.generate(_request(), cancel_event=cancel_event)


def test_poll_timeout(adapter, http_recorder, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monkeypatch.setattr(
        mva,
        "get_video_generation_config",
        lambda: _fake_config(minimax_video_timeout_seconds=1),
    )
    fast_adapter = mva.MiniMaxVideoAdapter()
    calls, routes = http_recorder
    routes[("POST", "/v2/video_generation")] = {"task_id": "t7"}
    routes[("GET", "/v2/query/")] = {"task": {"status": "processing"}}
    with pytest.raises(VideoGenerationError, match="timed out"):
        fast_adapter.generate(_request())


def test_remote_cancel_failure_is_swallowed(adapter, http_recorder, monkeypatch):
    """A failing DELETE (e.g. task already running) must not mask the local stop."""
    import threading

    cancel_event = threading.Event()
    cancel_event.set()

    def submit_ok_delete_fails(method, url, **kwargs):
        if method == "POST":
            return {"task_id": "t8"}
        if method == "DELETE":
            raise RuntimeError("cannot cancel a running task")
        raise AssertionError(f"unexpected {method} {url}")

    monkeypatch.setattr(mva, "fetch_json", submit_ok_delete_fails)
    with pytest.raises(VideoGenerationError, match="cancelled by user"):
        adapter.generate(_request(), cancel_event=cancel_event)
