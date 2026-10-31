"""MiniMax-H3 video adapter tests (task-3401.3) -- HTTP layer fully mocked.

The Stop tests at the end (TASK-33621.50) replay MiniMax's own responses,
recorded live on 2026-10-30, through the real ``fetch_json`` over an httpx
MockTransport.
"""

import json
from pathlib import Path
import threading
from types import SimpleNamespace

import httpx
import pytest
from loguru import logger

from tldw_chatbook.Image_Generation import http_client as image_http_client
from tldw_chatbook.Video_Generation import exceptions as video_exceptions
from tldw_chatbook.Video_Generation.adapters import minimax_video_adapter as mva
from tldw_chatbook.Video_Generation.adapters.base import ResolvedReferenceAsset
from tldw_chatbook.Video_Generation.exceptions import (
    VideoBackendUnavailableError,
    VideoGenerationError,
)
from tldw_chatbook.Video_Generation.worker import build_request

RECORDED = json.loads(
    (
        Path(__file__).parent / "fixtures" / "minimax_cancel_recorded_2026-10-30.json"
    ).read_text()
)


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
        {
            "task": {
                "status": "succeeded",
                "content": {"url": "https://cdn.example.com/v.mp4"},
            }
        },
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
        "task": {
            "status": "succeeded",
            "content": {"url": "https://cdn.example.com/v.mp4"},
        }
    }
    routes[("BYTES", "cdn.example.com")] = (b"v", "video/mp4")

    adapter.generate(_request(width=1280, height=720))
    adapter.generate(_request(width=2560, height=1440))
    submits = [c for c in calls if c["method"] == "POST"]
    assert submits[0]["kwargs"]["json"]["resolution"] == "768P"
    assert submits[1]["kwargs"]["json"]["resolution"] == "2K"


@pytest.mark.parametrize(
    "observed_type", [None, "application/octet-stream", "video/webm"]
)
def test_file_id_fallback_requires_observed_mp4_mime(
    adapter, http_recorder, observed_type
):
    calls, routes = http_recorder
    routes[("POST", "/v2/video_generation")] = {"task_id": "t2"}
    routes[("GET", "/v2/query/")] = {
        "task_id": "t2",
        "status": "Success",
        "file_id": "98765",
    }
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
    asset = ResolvedReferenceAsset(
        kind="first_frame", content=b"png", mime_type="image/png"
    )
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
    routes[("DELETE", "/v2/video_generation/t5")] = RECORDED[
        "cancel_accepted_just_after_submit"
    ]["body"]

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
            return RECORDED["cancel_accepted_while_running_0s"]["body"]
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


# -- Stop against MiniMax's recorded live cancel responses (TASK-33621.50) ----
#
# Live on 2026-10-30 (fixture provenance): the adapter's DELETE was accepted
# ~1 s after submit and right after a poll read status=running (200,
# action=cancelled, usage stayed empty), and refused 43 s into running (400,
# bad_request_error, code 2013) -- that task kept running and finished billed
# for 4 output seconds. The real fetch_json runs below, so a 4xx raises
# httpx.HTTPStatusError exactly as it did live; only the transport and the
# DNS-resolving egress check are replaced.


class _RecordedMiniMax:
    """Replay one recorded MiniMax exchange; set the Stop on the first poll."""

    def __init__(self, *, poll: str, delete: "str | dict | Exception") -> None:
        self.poll = RECORDED[poll]
        # A str names a recorded exchange; a dict is a synthesized one with
        # the same {http_status, body} shape; an Exception is a transport error.
        self.delete = RECORDED[delete] if isinstance(delete, str) else delete
        self.task_id = self.poll["body"]["task"]["id"]
        self.cancel_event = threading.Event()
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"task_id": self.task_id})
        if request.method == "GET":
            self.cancel_event.set()  # the user presses Stop while it runs
            return httpx.Response(self.poll["http_status"], json=self.poll["body"])
        if request.method == "DELETE":
            if isinstance(self.delete, Exception):
                raise self.delete
            return httpx.Response(self.delete["http_status"], json=self.delete["body"])
        raise AssertionError(f"unscripted {request.method} {request.url}")

    def deletes(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == "DELETE"]


@pytest.fixture
def replay(monkeypatch):
    """Install a recorded exchange under the real fetch_json."""

    def install(**kwargs) -> _RecordedMiniMax:
        script = _RecordedMiniMax(**kwargs)
        monkeypatch.setattr(
            image_http_client,
            "create_client",
            lambda timeout=None, **_kw: httpx.Client(
                transport=httpx.MockTransport(script), timeout=timeout
            ),
        )
        monkeypatch.setattr(
            image_http_client, "_validate_egress_or_raise", lambda *_a, **_k: None
        )
        return script

    return install


@pytest.fixture
def log_lines():
    lines: list[tuple[str, str]] = []
    handler = logger.add(
        lambda message: lines.append(
            (message.record["level"].name, message.record["message"])
        ),
        level="DEBUG",
    )
    yield lines
    logger.remove(handler)


def _stopped(adapter, script) -> VideoGenerationError:
    with pytest.raises(VideoGenerationError) as caught:
        adapter.generate(_request(), cancel_event=script.cancel_event)
    cancelled_type = getattr(video_exceptions, "VideoGenerationCancelled", None)
    assert cancelled_type is not None and isinstance(caught.value, cancelled_type), (
        f"Stop raised {type(caught.value).__name__}, not a typed cancellation "
        "the Console can report as a stop (TASK-33621.45)"
    )
    return caught.value


def test_stop_sends_the_cancel_minimax_accepted_live_and_logs_success(
    adapter, replay, log_lines
):
    """AC#2 + AC#4: the exact request MiniMax accepted live, and the log says so."""
    script = replay(
        poll="poll_running_at_submit", delete="cancel_accepted_while_running_0s"
    )

    stopped = _stopped(adapter, script)

    assert stopped.remote_cancel == "cancelled"
    [delete] = script.deletes()
    recorded = RECORDED["cancel_accepted_while_running_0s"]
    assert (delete.method, str(delete.url)) == (recorded["method"], recorded["url"])
    assert delete.headers["authorization"] == "Bearer test-secret-key"
    assert delete.headers["content-type"] == "application/json"
    # The header names match the request MiniMax accepted live (its value
    # for authorization is redacted in the recording).
    recorded_names = {
        name.removesuffix("(redacted)")
        for name in RECORDED["cancel_request_headers_sent"]
    }
    assert {name.lower() for name in delete.headers.keys()} == recorded_names
    assert delete.content == b""
    assert any(
        level == "INFO" and "cancel confirmed" in text for level, text in log_lines
    ), log_lines


def test_stop_after_rendering_starts_reports_minimax_refused_live(
    adapter, replay, log_lines
):
    """AC#1/#3: MiniMax's live 400 (code 2013) is a refusal -- the task keeps
    running and may be billed -- and the log names the status and code."""
    script = replay(
        poll="query_after_refused_cancel", delete="cancel_refused_while_running_43s"
    )

    stopped = _stopped(adapter, script)

    assert stopped.remote_cancel == "refused"
    assert len(script.deletes()) == 1
    warnings = [text for level, text in log_lines if level == "WARNING"]
    assert any("http_status=400" in t and "error_code=2013" in t for t in warnings), (
        log_lines
    )
    assert all("test-secret-key" not in text for _level, text in log_lines)


def test_stop_never_reports_a_cancel_for_a_record_minimax_deleted(
    adapter, replay, log_lines
):
    """DELETE on a task that finished meanwhile deletes its record (live: 200,
    action=deleted). Nothing was cancelled, so it must not read as cancelled."""
    script = replay(poll="query_after_refused_cancel", delete="delete_on_succeeded")

    stopped = _stopped(adapter, script)

    assert stopped.remote_cancel == "unconfirmed"


def test_stop_with_an_unreachable_cancel_endpoint_is_unconfirmed(
    adapter, replay, log_lines
):
    script = replay(
        poll="poll_running_at_submit",
        delete=httpx.ConnectError("connection refused"),
    )

    stopped = _stopped(adapter, script)

    assert stopped.remote_cancel == "unconfirmed"
    assert any(
        level == "WARNING" and "error_type=ConnectError" in text
        for level, text in log_lines
    ), log_lines


def _synthesized_400(message: str) -> dict:
    """A 400 in the recorded refusal's shape with a different message."""
    refusal = RECORDED["cancel_refused_while_running_43s"]
    body = json.loads(json.dumps(refusal["body"]))
    body["error"]["message"] = message
    return {"http_status": 400, "body": body}


@pytest.mark.parametrize(
    "message",
    [
        # Another MiniMax error code: not the "cannot cancel now" refusal.
        "invalid params, task_id is invalid (1004)",
        # 2013 is MiniMax's generic invalid-params code; without the
        # current-status wording it says nothing about the task's stage.
        "invalid params, task_id is malformed (2013)",
    ],
)
def test_stop_counts_only_minimaxs_current_status_refusal_as_refused(
    adapter, replay, log_lines, message
):
    """Any other 400 must read as unconfirmed, never as "already rendering"."""
    script = replay(poll="poll_running_at_submit", delete=_synthesized_400(message))

    stopped = _stopped(adapter, script)

    assert stopped.remote_cancel == "unconfirmed"
    assert any(
        level == "WARNING" and "unconfirmed" in text and "http_status=400" in text
        for level, text in log_lines
    ), log_lines
