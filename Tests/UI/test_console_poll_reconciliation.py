"""Original mounted polling must not repeatedly prepare unchanged live state."""

import contextlib
import inspect
import sys
from types import CodeType

import pytest

from Tests.UI.test_console_hook_review_send_freeze import _until
from Tests.UI.test_console_received_intent_feedback import (
    _held_received_record,
    _received_console_case,
    _send,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


class _OriginalPreparingPolls:
    """Observe original callbacks without replacing timer or reconciliation."""

    def __init__(self, case, record):
        self.case, self.record = case, record
        self.poll_code = next(
            code
            for code in ChatScreen._start_console_transcript_sync_timer.__code__.co_consts
            if isinstance(code, CodeType) and code.co_name == "_poll_transcript"
        )
        self.full_code = ChatScreen._sync_native_console_chat_ui.__code__
        self.core_code = inspect.unwrap(
            ChatScreen._sync_console_chat_core_state
        ).__code__
        self.active = {}
        self.completed = []

    def _started(self, code, _offset):
        frame = sys._getframe(1)
        if code is self.poll_code and frame.f_locals.get("self") is self.case.console:
            self.active[id(frame)] = {"core_returns": 0, "full_results": []}

    def _returned(self, code, _offset, value):
        frame = sys._getframe(1)
        parent = frame
        while parent is not None and parent.f_code is not self.poll_code:
            parent = parent.f_back
        if parent is None or parent.f_locals.get("self") is not self.case.console:
            return
        row = self.active.get(id(parent))
        if row is None:
            return  # A callback already running when observation began.
        if code is self.core_code:
            row["core_returns"] += 1
        elif code is self.full_code:
            row["full_results"].append(value)
        elif code is self.poll_code:
            case = self.case
            row["held"] = (
                case.probe.entered.is_set()
                and not case.probe.release.is_set()
                and not case.probe.release_timed_out
                and case.store.received_turn_for_session(case.session.id)
                is self.record.received_claim
                and self.record.request is None
            )
            row["deferred"] = bool(
                getattr(case.console, "_console_sync_maintenance_paused", False)
                or getattr(
                    case.console, "_console_control_bar_replay_whole_sync", False
                )
            )
            self.completed.append(row)
            del self.active[id(parent)]

    @contextlib.contextmanager
    def installed(self):
        monitoring = sys.monitoring
        tool = next(value for value in range(6) if monitoring.get_tool(value) is None)
        monitoring.use_tool_id(tool, "original-preparing-polls")
        try:
            monitoring.register_callback(
                tool, monitoring.events.PY_START, self._started
            )
            monitoring.register_callback(
                tool, monitoring.events.PY_RETURN, self._returned
            )
            monitoring.set_local_events(
                tool,
                self.poll_code,
                monitoring.events.PY_START | monitoring.events.PY_RETURN,
            )
            for code in (self.full_code, self.core_code):
                monitoring.set_local_events(tool, code, monitoring.events.PY_RETURN)
            yield self
        finally:
            for code in (self.poll_code, self.full_code, self.core_code):
                monitoring.set_local_events(tool, code, 0)
            monitoring.register_callback(tool, monitoring.events.PY_START, None)
            monitoring.register_callback(tool, monitoring.events.PY_RETURN, None)
            monitoring.free_tool_id(tool)
            self.active.clear()


async def test_ordinary_preparing_polls_do_not_repeat_live_core_reconciliation(
    monkeypatch,
    record_property,
):
    """Count original completed work; a busy/deferred retry is not a valid RED."""
    async with _received_console_case(monkeypatch, "poll-reconciliation") as case:
        _send(case, "enter")
        record = await _held_received_record(case)
        assert case.console._console_transcript_sync_timer is not None
        observed = _OriginalPreparingPolls(case, record)
        with observed.installed():
            completed = await _until(lambda: len(observed.completed) >= 2, 5)
        rows = observed.completed
        record_property("original_preparing_poll_observations", rows)
        assert completed, f"Two original timer callbacks did not complete: {rows!r}"
        assert all(row["held"] for row in rows), "Original Preparing hold expired"
        assert not any(
            row["deferred"] for row in rows
        ), f"Deferred full retry: {rows!r}"
        assert not any(
            result is False for row in rows for result in row["full_results"]
        ), f"Full reconciliation was refused, not redundant: {rows!r}"
        assert case.provider_calls == []
        assert case.composer.draft_text() == case.draft
        assert (
            sum(row["core_returns"] for row in rows) == 0
        ), f"Unchanged Preparing polls repeated original live core reconciliation: {rows!r}"
