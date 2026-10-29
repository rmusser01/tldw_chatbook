"""TASK-33621.5/.6: a provider error, then the automatic retry's trace fails.

The send goes out once, the provider answers HTTP 429, and the agent's
in-place retry then cannot reserve its trace call, so it pauses on the
"Trace capture blocked" card before anything else is sent. Every test mounts
the real ``ChatScreen`` with its real controller, store, agent loop, gateway,
provider adapter and durable trace runtime on a file-backed database. Only
the HTTP response is fake, and only the retry's reservation is made to fail,
in one of the two ways it fails in the app:

* ``unknown``: SQLite refuses the reservation transaction ("database is
  locked", reproduced live by holding a write lock across the retry). The
  outcome of the reservation is unknown to the gateway.
* ``refused``: the reservation transaction itself refuses the retry
  (``trace_tool_chain_unavailable``, the refusal the review recorded) and
  rolls back, so nothing was reserved.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_console_blocked_send_recovery import (
    _assistant_texts,
    _frame_rows,
    _mounted_console,
    _send,
    _settle_workers,
    _until,
)
from Tests.UI.test_console_provider_error_copy import (
    HTTP_REPLY,
    answer_http,
    real_provider_adapter,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus
from tldw_chatbook.Chat.console_trace_errors import TraceCallPersistenceError
from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository
from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    TraceCallRecoveryCallout,
)

pytestmark = pytest.mark.bootstrap_profile

DRAFT = "Name one color"


def fail_the_retry_reservation(h, monkeypatch, kind: str) -> list[object]:
    """Make the second trace reservation (the agent's retry) fail as ``kind``."""
    gateway = h.controller.provider_gateway
    factory = gateway._trace_call_boundary_factory
    routes: list[object] = []
    refusing = SimpleNamespace(active=False)
    original_lookup = ConsoleTraceRepository.get_call_by_logical_identity

    def lookup(self, *args, **kwargs):
        # Inside the reservation transaction: the retry finds no previous
        # call, which the factory refuses as ``trace_tool_chain_unavailable``.
        return None if refusing.active else original_lookup(self, *args, **kwargs)

    monkeypatch.setattr(ConsoleTraceRepository, "get_call_by_logical_identity", lookup)

    def flaky(request, resolution, route):
        routes.append(route)
        if len(routes) != 2:
            return factory(request, resolution, route)
        if kind == "unknown":
            raise sqlite3.OperationalError("database is locked")
        if kind == "dispatch_unknown":
            # A reservation whose dispatch read-back failed: the request may
            # have gone out, so this one is the genuine unknown delivery.
            raise TraceCallPersistenceError(
                boundary=SimpleNamespace(dispatch_outcome="unknown")
            )
        refusing.active = True
        try:
            return factory(request, resolution, route)
        finally:
            refusing.active = False

    monkeypatch.setattr(gateway, "_trace_call_boundary_factory", flaky)
    return routes


async def paused_after_429(h, monkeypatch, kind: str, posts: list[str]):
    real_provider_adapter(h, monkeypatch)
    answer_http(monkeypatch, [429, 200], posts)
    routes = fail_the_retry_reservation(h, monkeypatch, kind)
    await _send(h, DRAFT)
    callout = h.console.query_one(TraceCallRecoveryCallout)
    await _until(
        h.pilot,
        lambda: callout.display,
        timeout=30.0,
        what=f"no recovery card: {h.controller.run_state}",
    )
    assert len(posts) == 1, posts  # the 429; the retry was never sent
    assert len(routes) == 2, routes
    return callout


def visible_text(h) -> str:
    return " ".join(" ".join(_frame_rows(h.host)).split())


async def press(h, callout, action: str) -> None:
    button = callout.query_one(f"#console-trace-{action}", Button)
    button.focus()
    await h.pilot.pause(0.2)
    await h.pilot.press("enter")
    await _settle_workers(h)
    await h.console._sync_native_console_chat_ui()
    await h.pilot.pause(0.3)


def header_label(h) -> str:
    return h.console._build_console_workbench_state(
        h.console._build_console_control_state(None)
    ).header.status_label


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "refused"])
async def test_send_without_capture_dispatches_the_turn(tmp_path, monkeypatch, kind):
    """.6 AC#2/#3: no false unknown-delivery panel; the turn is answered."""
    async with _mounted_console(tmp_path, monkeypatch, (160, 52)) as h:
        posts: list[str] = []
        callout = await paused_after_429(h, monkeypatch, kind, posts)

        await press(h, callout, "send-without")
        await _until(
            h.pilot,
            lambda: h.controller.run_state.status is ConsoleRunStatus.COMPLETED,
            timeout=30.0,
            what=f"the send did not complete: {h.controller.run_state}",
        )
        await h.console._sync_native_console_chat_ui()
        await h.pilot.pause(0.3)

        assert len(posts) == 2, posts
        assert _assistant_texts(h) == [HTTP_REPLY]
        assert h.store.dispatch_recovery_for_presentation(h.session.id) is None
        assert "delivery status is unknown" not in visible_text(h)


@pytest.mark.asyncio
async def test_retry_reruns_capture_when_nothing_was_reserved(tmp_path, monkeypatch):
    """.6 AC#2: Retry capture re-runs the captured send and it is answered."""
    async with _mounted_console(tmp_path, monkeypatch, (160, 52)) as h:
        posts: list[str] = []
        callout = await paused_after_429(h, monkeypatch, "refused", posts)

        await press(h, callout, "retry")
        await _until(
            h.pilot,
            lambda: h.controller.run_state.status is ConsoleRunStatus.COMPLETED,
            timeout=30.0,
            what=f"the retry did not complete: {h.controller.run_state}",
        )
        await h.console._sync_native_console_chat_ui()
        await h.pilot.pause(0.3)

        assert len(posts) == 2, posts
        assert _assistant_texts(h) == [HTTP_REPLY]
        assert "delivery status is unknown" not in visible_text(h)
        with h.database.transaction() as cursor:
            states = [
                row[0]
                for row in cursor.execute(
                    "SELECT state FROM console_trace_calls ORDER BY created_at"
                ).fetchall()
            ]
        assert states == ["error", "complete"], states


@pytest.mark.asyncio
async def test_retry_reshows_the_capture_error_when_the_reservation_is_unknown(
    tmp_path, monkeypatch
):
    """.6 AC#2: an unconfirmed reservation keeps the card, never 'unknown delivery'."""
    async with _mounted_console(tmp_path, monkeypatch, (160, 52)) as h:
        posts: list[str] = []
        callout = await paused_after_429(h, monkeypatch, "unknown", posts)

        await press(h, callout, "retry")

        assert len(posts) == 1, posts
        assert callout.display
        status = str(callout.query_one("#console-trace-status", Static).render())
        assert "could not be confirmed" in status, status
        assert h.store.dispatch_recovery_for_presentation(h.session.id) is None
        assert "delivery status is unknown" not in visible_text(h)
        assert header_label(h) == "Blocked"

        await press(h, callout, "send-without")
        await _until(
            h.pilot,
            lambda: h.controller.run_state.status is ConsoleRunStatus.COMPLETED,
            timeout=30.0,
            what=f"the send did not complete: {h.controller.run_state}",
        )
        assert len(posts) == 2, posts

@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["unknown", "refused"])
async def test_cancel_send_discards_the_turn_and_returns_the_draft(
    tmp_path, monkeypatch, kind
):
    """.6 AC#1: Cancel works even when the reservation status is unknown."""
    async with _mounted_console(tmp_path, monkeypatch, (160, 52)) as h:
        posts: list[str] = []
        callout = await paused_after_429(h, monkeypatch, kind, posts)
        h.console._console_composer_or_none().load_draft("")

        await press(h, callout, "cancel")

        assert not callout.display
        assert "Trace reservation status is unavailable" not in visible_text(h)
        assert h.console._console_composer_or_none().draft_text() == DRAFT
        assert len(posts) == 1
        assert h.controller.trace_call_recovery_preparation() is None
        assert h.store.dispatch_recovery_for_presentation(h.session.id) is None
        assert h.controller.run_state.status is ConsoleRunStatus.STOPPED
        assert header_label(h) != "Blocked"
