"""Warm, stock-source-qualified polling must not repeatedly prepare live state."""

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


async def _qualify_warm_capture_sources(case):
    """Initialize real sources before Send; cold fallback is a separate control."""
    from tldw_chatbook.Chat.console_configuration_preparation import (
        standard_console_configuration_sources,
    )

    app = case.console.app_instance
    trust = await app.ensure_local_skill_trust_service()
    # Publish through the original lazy property after its real async owner
    # has physically completed. Do not inject a ready flag or replace a guard.
    assert app.local_skills_service.trust_service is trust
    assert standard_console_configuration_sources(
        app, case.store, case.controller, session_id=case.session.id
    ), "Warm original configuration sources are not eligible for worker capture"
    assert not case.probe.entered.is_set()


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
    async with _received_console_case(
        monkeypatch, "poll-reconciliation", durable=True
    ) as case:
        await _qualify_warm_capture_sources(case)
        _send(case, "enter")
        record = await _held_received_record(case)
        assert case.console._console_transcript_sync_timer is not None
        observed = _OriginalPreparingPolls(case, record)
        with observed.installed():
            completed = await _until(lambda: len(observed.completed) >= 3, 5)
        rows = observed.completed
        record_property("original_preparing_poll_observations", rows)
        assert completed, f"Three original timer callbacks did not complete: {rows!r}"
        assert all(row["held"] for row in rows), "Original Preparing hold expired"
        assert not any(
            row["deferred"] for row in rows
        ), f"Deferred full retry: {rows!r}"
        assert not any(
            result is False for row in rows for result in row["full_results"]
        ), f"Full reconciliation was refused, not redundant: {rows!r}"
        assert case.provider_calls == []
        assert case.composer.draft_text() == case.draft
        # The fixture selected a workspace after mount, and Send may owe an
        # initial full reconciliation. Permit the first observed poll to
        # settle that work; only subsequent unchanged polls are the target.
        steady_rows = rows[1:]
        assert (
            sum(row["core_returns"] for row in steady_rows) == 0
        ), f"Unchanged Preparing polls repeated original live core reconciliation: {rows!r}"


async def test_in_place_runtime_disable_reaches_next_real_send(monkeypatch):
    """The next driver Send must honor the gate without a test forcing full sync."""
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    async with _received_console_case(
        monkeypatch, "runtime-gate-next-send", durable=True
    ) as case:
        await _qualify_warm_capture_sources(case)
        controller = case.controller
        assert controller._agent_runtime_enabled is True
        assert controller._agent_bridge is not None
        app_config = case.console.app_instance.app_config
        console_config = app_config["console"]
        assert console_config.get("agent_runtime", True) is True
        agent_entries = []
        code = ConsoleChatController._run_agent_reply.__code__

        def started(_code, _offset):
            if sys._getframe(1).f_locals.get("self") is controller:
                agent_entries.append(True)

        monitoring = sys.monitoring
        tool = next(value for value in range(6) if monitoring.get_tool(value) is None)
        monitoring.use_tool_id(tool, "original-runtime-gate-next-send")
        try:
            monitoring.register_callback(tool, monitoring.events.PY_START, started)
            monitoring.set_local_events(tool, code, monitoring.events.PY_START)
            # The supported in-memory kill switch changes in place. Do not
            # save/reload config, rebuild the screen, update the controller,
            # call full sync, or yield before posting the actual next action.
            console_config["agent_runtime"] = False
            _send(case, "enter")
            record = await _held_received_record(case)
            assert case.console.app_instance.app_config is app_config
            assert app_config["console"] is console_config
            assert record.received_intent.selection.agent_runtime_enabled is False
            assert (
                record.received_intent.selection.tool_configuration[
                    "agent_runtime_enabled"
                ]
                is False
            )
            task = record.task
            case.probe.release.set()
            assert await _until(
                lambda: not case.runtime.has_custodied_turns(case.session.id), 15
            )
            task.result()
            assert len(case.provider_calls) == 1
            messages = case.store.messages_for_session(case.session.id)
            assert any(
                message.role is ConsoleMessageRole.ASSISTANT
                and message.status == "complete"
                and message.content == "received intent reply"
                for message in messages
            )
            assert agent_entries == [], "Disabled runtime still entered the agent route"
            assert case.probe.retired()
        finally:
            case.probe.release.set()
            monitoring.set_local_events(tool, code, 0)
            monitoring.register_callback(tool, monitoring.events.PY_START, None)
            monitoring.free_tool_id(tool)
