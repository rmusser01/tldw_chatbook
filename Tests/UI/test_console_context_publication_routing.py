"""Context publication uses current Preparing display routing without losing FULL demand."""

import asyncio
from contextlib import contextmanager
import inspect
import sys
from types import SimpleNamespace

import pytest
from textual.worker import NoActiveWorker, get_current_worker

from Tests.UI.test_console_hook_review_send_freeze import _until
from Tests.UI.test_console_poll_reconciliation import _qualify_warm_capture_sources
from Tests.UI.test_console_received_intent_feedback import (
    _held_received_record,
    _received_console_case,
    _send,
)
from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@contextmanager
def _publication_calls(console):
    """Count original consumers only in the actual publication worker."""
    codes = {
        inspect.unwrap(ChatScreen._sync_console_chat_core_state).__code__: "core",
        ChatScreen._dispatch_active_console_roleplay_refresh.__code__: "roleplay",
        ChatScreen._sync_native_console_transcript.__code__: "transcript",
        ChatScreen._sync_console_control_bar_under_config.__code__: "controls",
    }
    calls = dict.fromkeys(codes.values(), 0)
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if event != "call" or frame.f_code not in codes:
            return
        if frame.f_locals.get("self") is not console:
            return
        try:
            worker = get_current_worker()
        except NoActiveWorker:
            return
        if worker.group == "console-context-publication":
            calls[codes[frame.f_code]] += 1

    sys.setprofile(observe)
    try:
        yield calls
    finally:
        sys.setprofile(previous)


async def _publish(snapshot, controller, session_id):
    """Run the real context reader and wait for its issued display publication."""
    publications = []
    schedule = snapshot.schedule

    def collect(operation, **kwargs):
        worker = schedule(operation, **kwargs)
        if kwargs.get("group") == "console-context-publication":
            publications.append(worker)
        return worker

    snapshot.schedule = collect
    try:
        snapshot.key = snapshot.value = None
        key = snapshot._cache_key(controller, session_id)
        assert key is not None
        await snapshot._refresh(controller, session_id, key)
        assert snapshot.value is not None
        assert publications
        return [await worker.wait() for worker in publications]
    finally:
        snapshot.schedule = schedule


async def test_context_publication_during_received_preparing_uses_display_route(
    monkeypatch,
):
    async with _received_console_case(
        monkeypatch, "context-publication-preparing", durable=True
    ) as case:
        await _qualify_warm_capture_sources(case)
        case.console._stop_console_transcript_sync_timer()
        _send(case, "enter")
        record = await _held_received_record(case)
        case.console._stop_console_transcript_sync_timer()
        console = case.console
        # FULL may defer through replay; wait for its completed token and publishers.
        previous_completion = getattr(console, "_console_full_sync_completed", None)
        await console._sync_native_console_chat_ui()
        snapshot = spend.ConsoleContextReadSnapshot.for_screen(console, max_age=1.0)
        readiness = getattr(console, "_console_readiness_config_projection", None)
        display_groups = {
            "console-context-presentation",
            "console-context-publication",
            "console-readiness-config",
            "console-readiness-publication",
            "console-sync",
        }

        def publication_state():
            completion = getattr(console, "_console_full_sync_completed", None)
            return {
                "full_completed": completion is not None
                and completion is not previous_completion,
                "sync": console._console_sync_in_progress,
                "requested": console._console_sync_requested,
                "maintenance": getattr(
                    console, "_console_sync_maintenance_paused", False
                ),
                "replay": getattr(
                    console, "_console_control_bar_replay_whole_sync", False
                ),
                "readiness_pending": bool(getattr(readiness, "pending", False)),
                "context_pending": snapshot.pending_key is not None,
                "workers": tuple(
                    worker.group
                    for worker in console.workers
                    if worker.node is console
                    and worker.group in display_groups
                    and not worker.is_finished
                ),
            }

        def settled():
            state = publication_state()
            return (
                state["full_completed"]
                and not any(state[name] for name in state if name != "full_completed")
                and console._console_preparing_poll_record() is record
            )

        assert await _until(settled, 5.0), publication_state()
        with _publication_calls(case.console) as calls:
            results = await _publish(snapshot, case.controller, case.session.id)
        assert results and all(
            result is True for result in results
        ), publication_state()
        assert record.request is None
        assert case.provider_calls == []
        assert case.composer.draft_text() == case.draft
        assert not case.probe.release_timed_out
        assert calls["transcript"] > 0
        assert calls["controls"] > 0
        assert calls["core"] == calls["roleplay"] == 0


async def test_context_publication_during_full_sync_preserves_trailing_demand(
    monkeypatch,
):
    async with _received_console_case(
        monkeypatch, "context-publication-overlap", durable=True
    ) as case:
        await _qualify_warm_capture_sources(case)
        case.console._stop_console_transcript_sync_timer()
        await case.console._sync_native_console_chat_ui()
        snapshot = spend.ConsoleContextReadSnapshot.for_screen(
            case.console, max_age=1.0
        )
        await snapshot.warm(case.controller, case.session.id)
        original = case.console._sync_native_console_transcript
        consumed = asyncio.Event()
        release = asyncio.Event()
        entries = []

        async def hold_after_original(*args, **kwargs):
            await original(*args, **kwargs)
            entries.append(True)
            if len(entries) == 1:
                consumed.set()
                await release.wait()

        monkeypatch.setattr(
            case.console, "_sync_native_console_transcript", hold_after_original
        )
        running = asyncio.create_task(case.console._sync_native_console_chat_ui())
        try:
            assert await _until(consumed.is_set, 5.0)
            assert case.console._console_sync_in_progress
            # Isolate the new demand after the running pass consumed the old context.
            case.console._console_sync_requested = False
            await _publish(snapshot, case.controller, case.session.id)
            assert case.console._console_sync_requested
            release.set()
            await running
            assert await _until(
                lambda: len(entries) >= 2
                and not case.console._console_sync_in_progress
                and not case.console._console_sync_requested,
                5.0,
            )
        finally:
            release.set()
            await asyncio.gather(running, return_exceptions=True)


class _CallbackScreen:
    def __init__(self, calls):
        self.calls = calls
        self.app_instance = SimpleNamespace(app_config={})
        self._console_sync_in_progress = False

    def run_worker(self, *args, **kwargs):
        raise AssertionError("Callback-only contract must not schedule a worker")

    async def _sync_native_console_chat_ui(self):
        self.calls.append("captured-full")

    async def _sync_console_poll_display_ui(self):
        self.calls.append("poll")


@pytest.mark.parametrize("callback_kind", ["instance", "bound", "legacy"])
async def test_context_refresh_keeps_captured_full_after_replacement(callback_kind):
    calls = []
    screen = _CallbackScreen(calls)
    if callback_kind == "instance":

        async def captured():
            calls.append("captured-full")

        screen._sync_native_console_chat_ui = captured
    elif callback_kind == "legacy":
        screen._sync_console_poll_display_ui = None
    snapshot = spend.ConsoleContextReadSnapshot.for_screen(screen, max_age=1.0)

    async def replacement():
        calls.append("replacement-full")

    screen._sync_native_console_chat_ui = replacement
    await snapshot.refresh()
    assert calls == ["captured-full"]


async def test_context_refresh_recognizes_unchanged_bound_full_callback():
    calls = []
    screen = _CallbackScreen(calls)
    snapshot = spend.ConsoleContextReadSnapshot.for_screen(screen, max_age=1.0)
    # Each attribute access creates a new bound-method object for the same owner/body.
    assert (
        screen._sync_native_console_chat_ui is not screen._sync_native_console_chat_ui
    )
    await snapshot.refresh()
    assert calls == ["poll"]
