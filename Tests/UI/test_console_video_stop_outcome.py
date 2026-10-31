"""Stop on a MiniMax ``/generate-video`` says what happened to the remote task.

TASK-33621.50: Stop issued MiniMax's cancel, but a task that had started
rendering was refused (HTTP 400, code 2013), kept running and was billed,
while the Console said nothing about it. TASK-33621.45: the same Stop then
read as "Video generation failed (VideoGenerationError)".

The real app drives it end to end: the composer's Stop button, the Console
video controller and the store, the real MiniMax adapter and the real
``fetch_json``. Only the HTTP transport is replaced; it replays MiniMax's own
responses recorded live on 2026-10-30
(``Tests/Video_Generation/fixtures/minimax_cancel_recorded_2026-10-30.json``).
The registry and the video store stay the send-freeze harness's stand-ins:
nothing here reaches them, because a stopped generation produces no video.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from textual.widgets import Button

from Tests.UI.test_console_hook_review_send_freeze import _until
from Tests.UI.test_console_video_send_freeze import (
    _confirm_generate,
    _console_sending_video,
    _video_idle,
    _wait_for_cost_confirm,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Image_Generation import http_client as image_http_client
from tldw_chatbook.Video_Generation.adapters import minimax_video_adapter as mva
from tldw_chatbook.Video_Generation.worker import build_request

pytestmark = pytest.mark.bootstrap_profile

RECORDED = json.loads(
    (
        Path(__file__).parents[1]
        / "Video_Generation"
        / "fixtures"
        / "minimax_cancel_recorded_2026-10-30.json"
    ).read_text()
)


class _RecordedMiniMaxGeneration:
    """The real MiniMax adapter at the Console's generation seam, over a
    recorded exchange. The poll keeps answering "running" until the user's
    Stop wakes the adapter, which then sends its DELETE."""

    def __init__(self, monkeypatch, *, poll: str, delete: str | None) -> None:
        self.poll = RECORDED[poll]
        self.delete = RECORDED[delete] if delete is not None else None
        self.task_id = self.poll["body"]["task"]["id"]
        self.started = threading.Event()
        self.polled = threading.Event()
        self.release = threading.Event()  # the harness's teardown sets it
        self.requests: list[httpx.Request] = []
        monkeypatch.setattr(
            mva,
            "get_video_generation_config",
            lambda: SimpleNamespace(
                minimax_video_api_key="test-secret-key",
                minimax_video_base_url="https://api.minimax.io",
                minimax_video_default_model="MiniMax-H3",
                minimax_video_poll_interval_seconds=1,
                minimax_video_timeout_seconds=30,
                download_max_mb=500,
            ),
        )
        monkeypatch.setattr(
            image_http_client,
            "create_client",
            lambda timeout=None, **_kw: httpx.Client(
                transport=httpx.MockTransport(self._serve), timeout=timeout
            ),
        )
        monkeypatch.setattr(
            image_http_client, "_validate_egress_or_raise", lambda *_a, **_k: None
        )

    def _serve(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"task_id": self.task_id})
        if request.method == "GET":
            self.polled.set()
            return httpx.Response(self.poll["http_status"], json=self.poll["body"])
        if request.method == "DELETE" and self.delete is not None:
            return httpx.Response(self.delete["http_status"], json=self.delete["body"])
        raise AssertionError(f"unscripted {request.method} {request.url}")

    def __call__(self, **kwargs):
        self.started.set()
        request = build_request(
            backend="minimax",
            prompt=kwargs["prompt"],
            duration_seconds=kwargs.get("duration_seconds"),
        )
        return mva.MiniMaxVideoAdapter().generate(
            request, cancel_event=kwargs["cancel_event"]
        )

    def deletes(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == "DELETE"]


def _system_rows(console) -> list[str]:
    store = console._console_chat_store
    return [
        message.content
        for message in store.messages_for_session(store.active_session_id)
        if message.role == ConsoleMessageRole.SYSTEM
    ]


async def _stop_after_first_poll(app, console, backend) -> list[str]:
    await _confirm_generate(app, "enter")
    assert await _until(backend.polled.is_set, 10), "the adapter never polled"
    stop = console.query_one("#console-stop-generation", Button)
    assert await _until(lambda: stop.display and not stop.disabled, 5)
    before = len(_system_rows(console))
    stop.press()
    assert await _until(lambda: _video_idle(console), 10), "Stop never settled"
    assert await _until(lambda: len(_system_rows(console)) > before, 5)
    return _system_rows(console)[before:]


async def test_stop_while_minimax_can_still_cancel_reads_as_a_free_stop(monkeypatch):
    """AC#2 (.50) + AC#1 (.45): MiniMax accepted the cancel live (200,
    action=cancelled), so the row says the task was cancelled and not billed,
    and nothing reads as a failure."""
    backend = _RecordedMiniMaxGeneration(
        monkeypatch,
        poll="poll_running_at_submit",
        delete="cancel_accepted_while_running_0s",
    )
    async with _console_sending_video(monkeypatch, "enter", backend) as (
        app,
        console,
        _cleanups,
    ):
        rows = await _stop_after_first_poll(app, console, backend)

    assert len(backend.deletes()) == 1
    [row] = rows
    assert "stopped" in row and "MiniMax cancelled the remote task" in row, row
    assert "not billed" in row, row
    assert "failed" not in row.lower(), row


async def test_stop_after_minimax_starts_rendering_says_it_may_still_be_billed(
    monkeypatch,
):
    """AC#3 (.50) + AC#1 (.45): MiniMax refused the cancel live 43 s into a
    render (400, code 2013) and billed the clip, so the row says plainly the
    task keeps running and may still be billed -- never that it was
    cancelled, and never that the generation failed."""
    backend = _RecordedMiniMaxGeneration(
        monkeypatch,
        poll="query_after_refused_cancel",
        delete="cancel_refused_while_running_43s",
    )
    async with _console_sending_video(monkeypatch, "enter", backend) as (
        app,
        console,
        _cleanups,
    ):
        rows = await _stop_after_first_poll(app, console, backend)

    assert len(backend.deletes()) == 1
    [row] = rows
    assert "stopped" in row, row
    assert "keeps running" in row and "may still be billed" in row, row
    assert "cancelled the remote task" not in row, row
    assert "failed" not in row.lower(), row


async def test_the_cost_confirm_and_progress_row_say_stop_may_not_prevent_billing(
    monkeypatch,
):
    """AC#3 (.50): before the user confirms, and on the row that says Stop is
    available, the copy says a started render may still be billed."""
    backend = _RecordedMiniMaxGeneration(
        monkeypatch,
        poll="query_after_refused_cancel",
        delete="cancel_refused_while_running_43s",
    )
    async with _console_sending_video(monkeypatch, "enter", backend) as (
        app,
        console,
        _cleanups,
    ):
        confirm = await _wait_for_cost_confirm(app, "enter")
        confirm_copy = confirm.message
        await _confirm_generate(app, "enter")
        assert await _until(backend.polled.is_set, 10)
        progress = [row for row in _system_rows(console) if "Generating video" in row]
        stop = console.query_one("#console-stop-generation", Button)
        assert await _until(lambda: stop.display and not stop.disabled, 5)
        stop.press()
        assert await _until(lambda: _video_idle(console), 10)

    assert "Stop" in confirm_copy and "may still be billed" in confirm_copy, (
        confirm_copy
    )
    [row] = progress
    assert "may still be billed" in row, row


async def test_a_real_minimax_failure_still_reads_as_a_failure(monkeypatch):
    """AC#2 (.45): with no Stop, a backend failure keeps its failure copy."""
    backend = _RecordedMiniMaxGeneration(
        monkeypatch, poll="query_after_refused_cancel", delete=None
    )
    backend.poll = {
        "http_status": 500,
        "body": {"type": "error", "error": {"type": "server_error"}},
    }
    async with _console_sending_video(monkeypatch, "enter", backend) as (
        app,
        console,
        _cleanups,
    ):
        before = len(_system_rows(console))
        await _confirm_generate(app, "enter")
        assert await _until(backend.polled.is_set, 10)
        assert await _until(lambda: _video_idle(console), 10)
        assert await _until(lambda: len(_system_rows(console)) > before + 1, 5)
        rows = _system_rows(console)[before:]

    assert backend.deletes() == []
    assert any("Video generation failed" in row for row in rows), rows
    assert not any("stopped" in row for row in rows), rows
