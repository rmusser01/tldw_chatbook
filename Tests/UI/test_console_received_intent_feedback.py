"""Actual Console input must enter runtime custody before native preparation."""

import asyncio
import contextlib
import json
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from textual.app import App
from textual.screen import Screen
from textual.widgets import Button

from Tests.Chat.test_console_configuration_worker_lifetime import (
    _OriginalConfigurationWorkspaceRead,
)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_configuration_worker_feedback import (
    _NaturalInputFrame,
    _VisibleNavigatedConsoleHarness,
    _update_contains,
)
from Tests.UI.test_console_hook_review_send_freeze import (
    _key,
    _task_factory,
    _until,
)
from Tests.UI.test_console_native_chat_flow import (
    _configure_native_ready_console,
    _persist_console_provider_config,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("route", ["enter", "send-button"])
async def test_actual_send_receives_intent_before_original_configuration_read(
    route, monkeypatch
):
    async with _received_console_case(monkeypatch, f"admission-{route}") as case:
        _send(case, route)
        record = await _held_received_record(case)
        assert case.composer.draft_text() == case.draft
        assert case.runtime.has_custodied_turns(case.session.id)
        assert case.store.preparation_for_session(case.session.id) is None
        assert record.store is case.store and record.session_id == case.session.id


class _NaturalPreparingFrame:
    """Observe the supplied natural button frame, never request a repaint."""

    def __init__(self, case):
        self.case = case
        self.code = App._display.__code__
        self.refresh_code = Screen._compositor_refresh.__code__
        self.action_at = self.frame_at = self.feedback_frame_at = None
        self.while_held = False
        self.pending_user_frame_at = self.pending_user_identity = None
        self.pending_user_while_held = False
        self.painted = threading.Event()

    def _returned(self, code, _offset, _value):
        if code is not self.code or self.action_at is None:
            return
        display_at = time.perf_counter()
        frame = sys._getframe(1)
        if (
            frame.f_locals.get("self") is not self.case.host
            or frame.f_locals.get("screen") is not self.case.console
            or self.case.host.screen is not self.case.console
        ):
            return
        parent = frame.f_back
        while parent is not None and parent.f_code is not self.refresh_code:
            parent = parent.f_back
        if parent is None:
            return
        button = self.case.console.query_one("#console-send-message", Button)
        chip = self.case.console.query_one("#console-run-chip")
        if self.feedback_frame_at is None and _update_contains(
            frame.f_locals.get("renderable"), chip.region, "Sending"
        ):
            self.feedback_frame_at = time.perf_counter()
        if self.frame_at is None and _update_contains(
            frame.f_locals.get("renderable"), button.region, "Preparing"
        ):
            self.frame_at = time.perf_counter()
            self.while_held = not self.case.probe.release.is_set()
            self.painted.set()
        self._pending_user_frame(frame.f_locals.get("renderable"), display_at)

    def _pending_user_frame(self, update, display_at):
        """Read exact pending-row cells; never include composer or stored echo."""
        if self.pending_user_frame_at is not None:
            return
        case = self.case
        module = sys.modules.get(
            "tldw_chatbook.UI.Console_Modules.send_acknowledgement"
        )
        if module is None:
            return
        ack = vars(case.console).get(module.ACK_ATTRIBUTE)
        if type(ack) is not module.ConsoleSendAcknowledgement:
            return
        pending = vars(ack).get("_pending", {}).get(case.session.id)
        models = sys.modules["tldw_chatbook.Chat.console_chat_models"]
        if (
            type(pending) is not module._PendingSend
            or pending.session_id != case.session.id
            or case.store.active_session_id != case.session.id
            or vars(case.console).get("_console_runtime_ref") is not case.runtime
            or vars(case.runtime).get("_chat_store") is not case.store
            or case.runtime.view is not case.console
            or type(pending.row) is not models.ConsoleChatMessage
            or pending.row.role is not models.ConsoleMessageRole.USER
            or pending.row.status != "pending"
            or pending.row.content != case.draft
        ):
            return
        widget = next(
            iter(case.console.query(f"#console-message-{pending.row.id}")), None
        )
        if widget is None or not widget.is_attached:
            return
        transcript = case.console.query_one("#console-native-transcript")
        region = widget.region.intersection(transcript.region).intersection(
            case.console.region
        )
        if region and _update_contains(update, region, case.draft):
            self.pending_user_frame_at = display_at
            self.pending_user_identity = (id(ack), id(pending.token), pending.row.id)
            self.pending_user_while_held = not case.probe.release.is_set()

    @contextlib.contextmanager
    def installed(self):
        monitoring = sys.monitoring
        tool = next(value for value in range(6) if monitoring.get_tool(value) is None)
        monitoring.use_tool_id(tool, "received-natural-preparing-frame")
        try:
            monitoring.register_callback(
                tool, monitoring.events.PY_RETURN, self._returned
            )
            monitoring.set_local_events(tool, self.code, monitoring.events.PY_RETURN)
            yield self
        finally:
            monitoring.set_local_events(tool, self.code, 0)
            monitoring.register_callback(tool, monitoring.events.PY_RETURN, None)
            monitoring.free_tool_id(tool)


@contextlib.asynccontextmanager
async def _received_console_case(monkeypatch, name, *, durable=False):
    """Reuse the qualified mounted setup with exact reader/runtime teardown."""
    from tldw_chatbook import config

    app = _build_test_app(user_data_dir=config.get_user_data_dir())
    database = None
    if durable:
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

        # Real file-backed persistence is required for the production worker
        # route; the general in-memory UI fixture deliberately declines it.
        database = CharactersRAGDB(
            config.get_chachanotes_db_path(), client_id="received-navigation"
        )
        app.chachanotes_db = database
    if durable:
        # Exercise real saved admission with only the remote adapter replaced.
        # llama.cpp readiness also probes a live server, absent in this test.
        _persist_console_provider_config(
            app,
            provider="openai",
            model="gpt-4o",
            provider_settings={"api_key": "synthetic-test-key"},
        )
        app.chat_api_provider_value = "openai"
        app.chat_api_model_value = "gpt-4o"
    else:
        _configure_native_ready_console(app)
    registry = app.workspace_registry_service
    workspace_id = f"received-{name}"
    registry.create_workspace(workspace_id=workspace_id, name=f"Received {name}")
    registry.set_active_workspace(workspace_id)
    provider_calls = []

    def reply(**kwargs):
        provider_calls.append(kwargs)
        return "received intent reply"

    monkeypatch.setattr("tldw_chatbook.Chat.Chat_Functions.chat_api_call", reply)
    host = _VisibleNavigatedConsoleHarness(app)
    runtime = probe = None
    try:
        with _task_factory("eager"):
            async with host.run_test(size=(120, 40)) as pilot:
                assert await _until(
                    lambda: isinstance(host.screen, ChatScreen)
                    and host.screen.is_mounted
                    and host.screen._console_attach_reconciled
                    and not host.screen._console_attach_reconcile_running,
                    10,
                )
                console = host.screen
                runtime = console._console_runtime()
                controller = console._ensure_console_chat_controller()
                store = controller.store
                session = store.ensure_session()
                assert session.workspace_id == workspace_id
                assert store.active_session_id == session.id
                assert console._console_visible_draft_session_id == session.id
                composer = console._console_composer_or_none()
                draft = "An exact received draft"
                composer.load_draft(draft)
                composer.focus()
                await pilot.pause()
                assert host.focused is composer
                assert console._console_composer_or_none() is composer
                assert store.active_session_id == session.id
                assert console._console_visible_draft_session_id == session.id
                assert composer.draft_text() == store.session_draft(session.id) == draft
                for selector in (
                    "#console-command-visible-text",
                    "#console-send-message",
                ):
                    region = console.query_one(selector).region
                    assert region and region.overlaps(console.region)
                probe = _OriginalConfigurationWorkspaceRead(registry)
                case = SimpleNamespace(
                    host=host,
                    console=console,
                    runtime=runtime,
                    controller=controller,
                    store=store,
                    session=session,
                    composer=composer,
                    draft=draft,
                    probe=probe,
                    provider_calls=provider_calls,
                )
                with probe.installed():
                    try:
                        yield case
                    finally:
                        probe.release.set()
                        controller.stop_active_run()
                        host.workers.cancel_group(console, "console-hook-send-review")
                        tasks = tuple(
                            row.task
                            for row in runtime._turn_custody.values()
                            if row.session_id == session.id and row.task is not None
                        )
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        if tasks:
                            await asyncio.wait_for(
                                asyncio.gather(*tasks, return_exceptions=True), 15
                            )
                        assert await _until(lambda: not console._hooks._busy, 15)
                        if probe.entered.is_set():
                            assert await _until(probe.retired, 5)
    finally:
        if probe is not None:
            probe.release.set()
        try:
            if runtime is not None:
                await runtime.dispose()
        finally:
            if database is not None:
                with database.quiesce_connections(timeout_seconds=5):
                    pass
                assert database.registered_connection_count() == 0


def _send(case, route):
    if route == "enter":
        _key(case.host, "enter", "\r")
    else:
        case.console.query_one("#console-send-message", Button).press()


async def _held_received_record(case):
    # Import inside the test path so the original RED still collects before
    # the implementation's detached intent model exists.
    from tldw_chatbook.Chat.console_received_intent import ConsoleReceivedTurnIntent

    assert await _until(case.probe.entered.is_set, 10)
    assert case.probe.live_at_entry and not case.probe.release_timed_out
    claim = case.store.received_turn_for_session(case.session.id)
    assert claim is not None
    record = case.runtime._turn_custody[claim.request_id]
    assert record.received_claim is claim and record.request is None
    assert isinstance(record.received_intent, ConsoleReceivedTurnIntent)
    assert record.received_intent.inputs.draft == case.draft
    assert record.task is not None and not record.task.done()
    assert case.provider_calls == []
    return record


@pytest.mark.parametrize("route", ["enter", "send-button"])
async def test_actual_send_paints_preparing_and_accepts_input_within_100ms(
    route, monkeypatch, record_property
):
    """Headless supplied frames qualify readiness, not physical terminal flush."""
    async with _received_console_case(monkeypatch, f"feedback-{route}") as case:
        preparing = _NaturalPreparingFrame(case)
        typing = _NaturalInputFrame(case.host, case.console, case.composer, case.probe)
        loop = asyncio.get_running_loop()
        stop = threading.Event()
        receipt_observed = threading.Event()

        def release_independently():
            try:
                while not case.probe.entered.wait(0.01):
                    if stop.is_set():
                        return
                typing.sent_at = time.perf_counter()
                loop.call_soon_threadsafe(_key, case.host, "Z", "Z")
                deadline = time.perf_counter() + 0.5
                typing.painted.wait(0.5)
                preparing.painted.wait(max(0, deadline - time.perf_counter()))
                receipt_observed.wait(max(0, deadline - time.perf_counter()))
            finally:
                case.probe.release.set()

        releaser = threading.Thread(target=release_independently)
        with preparing.installed(), typing.installed():
            releaser.start()
            try:
                # All Send-triggered preparation remains inside this interval.
                preparing.action_at = time.perf_counter()
                _send(case, route)
                record = await _held_received_record(case)
                ack = vars(case.console).get("_console_send_ack")
                pending = vars(ack)["_pending"][case.session.id]
                assert pending.admitted and pending.handoff_ids is not None
                received_pending_identity = (id(ack), id(pending.token), pending.row.id)
                assert record.received_intent.inputs.draft == pending.row.content
                receipt_observed.set()
                assert await _until(case.probe.release.is_set, 2)
                record_property(
                    "send_to_preparing_frame_seconds",
                    None
                    if preparing.frame_at is None
                    else preparing.frame_at - preparing.action_at,
                )
                for label, observed_at in (
                    ("input_mutation_seconds", typing.mutated_at),
                    ("input_frame_seconds", typing.frame_at),
                ):
                    record_property(
                        label,
                        None if observed_at is None else observed_at - typing.sent_at,
                    )
                record_property(
                    "send_to_pending_user_frame_seconds",
                    None
                    if preparing.pending_user_frame_at is None
                    else preparing.pending_user_frame_at - preparing.action_at,
                )
                assert preparing.pending_user_frame_at is not None
                assert preparing.pending_user_identity == received_pending_identity
                assert preparing.pending_user_while_held
                record_property("pending_user_linked_to_received_session", True)
                record_property("headless_supplied_frame_only", True)
                record_property("held_reader_budget_seconds", 0.5)
                record_property(
                    "send_to_run_feedback_frame_seconds",
                    None
                    if preparing.feedback_frame_at is None
                    else preparing.feedback_frame_at - preparing.action_at,
                )
                assert preparing.feedback_frame_at is not None
                assert preparing.feedback_frame_at - preparing.action_at <= 0.1
                assert preparing.frame_at is not None and preparing.while_held
                assert preparing.frame_at - preparing.action_at <= 0.1
                assert typing.mutated_while_held and typing.frame_while_held
                assert typing.mutated_at - typing.sent_at <= 0.1
                assert typing.frame_at - typing.sent_at <= 0.1
            finally:
                stop.set()
                case.probe.release.set()
                releaser.join(timeout=2)
                assert not releaser.is_alive()


def _install_received_wire_reply(monkeypatch, case):
    def reply(**kwargs):
        case.provider_calls.append(kwargs)
        if kwargs.get("streaming"):
            chunk = {
                "choices": [
                    {
                        "delta": {"content": "received intent reply"},
                        "finish_reason": None,
                    }
                ]
            }
            return iter(["data: " + json.dumps(chunk) + "\n\n", "data: [DONE]\n\n"])
        return {"choices": [{"message": {"content": "received intent reply"}}]}

    monkeypatch.setattr("tldw_chatbook.Chat.Chat_Functions.chat_api_call", reply)


def _assert_pressed_reply(case):
    assert len(case.provider_calls) == 1
    payload = case.provider_calls[0]["messages_payload"]
    assert any(
        row.get("role") == "user" and row.get("content") == case.draft
        for row in payload
    )
    assert any(
        message.role.value == "assistant"
        and message.status == "complete"
        and message.content == "received intent reply"
        for message in case.store.messages_for_session(case.session.id)
    )


@pytest.mark.parametrize("hidden_owner", [False, True])
async def test_pressed_capture_survives_identical_retyping_before_promotion(
    monkeypatch,
    hidden_owner,
):
    async with _received_console_case(
        monkeypatch, f"identical-retype-{hidden_owner}", durable=True
    ) as case:
        _install_received_wire_reply(monkeypatch, case)
        successor = case.store.create_session(title="Other draft owner", activate=False)
        other_draft = "Untouched other draft"
        case.store.set_session_draft(successor.id, other_draft)
        _send(case, "enter")
        record = await _held_received_record(case)
        intent = record.received_intent
        before = intent.inputs.draft_revision
        case.composer.clear_draft()
        case.composer.insert_text(case.draft)

        # The later authored draft is distinct even when its text is identical.
        assert case.store.session_draft(case.session.id) == case.draft
        assert (
            case.store.session_input_snapshot(case.session.id).draft_revision > before
        )
        assert case.composer.draft_text() == case.draft
        assert record.received_intent is intent and record.request is None
        if hidden_owner:
            case.console._session._capture_console_draft_switch_snapshot()
            case.store.switch_session(successor.id)
            case.console._session._sync_console_session_draft()
            assert case.console._console_visible_draft_session_id == successor.id
            assert case.composer.draft_text() == other_draft
            assert case.store.session_draft(case.session.id) == case.draft
        case.probe.release.set()
        assert await _until(
            lambda: not case.runtime.has_custodied_turns(case.session.id), 15
        )
        _assert_pressed_reply(case)
        assert case.store.received_turn_for_session(case.session.id) is None
        assert case.store.session_draft(case.session.id) == case.draft
        assert case.store.session_draft(successor.id) == other_draft
        assert case.store.received_turn_for_session(successor.id) is None
        if hidden_owner:
            assert case.store.active_session_id == successor.id
            assert case.console._console_visible_draft_session_id == successor.id
            assert case.composer.draft_text() == other_draft
        else:
            assert case.composer.draft_text() == case.draft


async def test_enter_sends_pressed_body_after_clear_and_type_before_paint_dispatch(
    monkeypatch,
):
    from tldw_chatbook.UI.Console_Modules import send_acknowledgement

    original_paint = send_acknowledgement.paint_acknowledgement
    paint_entered, paint_release = asyncio.Event(), asyncio.Event()

    async def held_original_paint(screen, session_id):
        paint_entered.set()
        await paint_release.wait()
        return await original_paint(screen, session_id)

    monkeypatch.setattr(
        send_acknowledgement, "paint_acknowledgement", held_original_paint
    )
    async with _received_console_case(
        monkeypatch, "clear-before-paint", durable=True
    ) as case:
        _install_received_wire_reply(monkeypatch, case)
        try:
            _send(case, "enter")
            assert await _until(paint_entered.is_set, 10)
            pending = case.console._console_pending_send
            assert pending is not None
            assert pending.session_id == case.session.id
            assert pending.stash.text == case.draft
            assert not case.probe.entered.is_set()
            assert case.provider_calls == []
            newer = "Newer unsent draft"
            case.composer.clear_draft()
            case.composer.insert_text(newer)
            assert await _until(
                lambda: case.store.session_draft(case.session.id) == newer, 5
            )
            assert case.composer.draft_text() == newer
            paint_release.set()
            record = await _held_received_record(case)
            assert record.received_intent.inputs.draft == case.draft
            assert case.store.session_draft(case.session.id) == newer
            case.probe.release.set()
            assert await _until(
                lambda: not case.runtime.has_custodied_turns(case.session.id), 15
            )
            _assert_pressed_reply(case)
            assert all(
                newer not in str(row)
                for row in case.provider_calls[0]["messages_payload"]
            )
            assert case.store.received_turn_for_session(case.session.id) is None
            assert case.store.session_draft(case.session.id) == newer
            assert case.composer.draft_text() == newer
        finally:
            paint_release.set()
            case.probe.release.set()


async def test_navigation_after_receipt_keeps_original_turn_and_newer_session_draft(
    monkeypatch,
):
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    async with _received_console_case(monkeypatch, "navigation", durable=True) as case:
        successor = case.store.create_session(
            title="Other received owner", activate=False
        )
        case.store.set_session_draft(successor.id, "Untouched other draft")
        _send(case, "enter")
        record = await _held_received_record(case)
        claim = record.received_claim
        task = record.task

        # This harness pushes an uninstalled view, so pop follows Textual's
        # real remove/unmount path. Cached production tabs instead suspend.
        assert case.host.screen is case.console
        assert not case.host.is_screen_installed(case.console)
        await case.host.pop_screen()
        # Textual leaves is_mounted=True after removal; attachment and runtime
        # ownership are the live witnesses (also covered by sync_outlives_screen).
        assert await _until(
            lambda: case.console not in case.host.screen_stack
            and not case.console.is_attached
            and case.runtime.view is None,
            5,
        )
        case.store.switch_session(successor.id)
        assert case.runtime._turn_custody[record.turn_id] is record
        assert case.store.received_turn_for_session(case.session.id) is claim
        assert record.request is None and not record.task.done()
        assert not case.probe.retired()

        case.probe.release.set()
        assert await _until(
            lambda: not case.runtime.has_custodied_turns(case.session.id), 15
        )
        task.result()
        assert len(case.provider_calls) == 1
        assert any(
            message.role is ConsoleMessageRole.USER and message.content == case.draft
            for message in case.store.messages_for_session(case.session.id)
        )
        assert case.store.session_draft(successor.id) == "Untouched other draft"
        assert case.store.received_turn_for_session(successor.id) is None


async def test_switch_settlement_send_refuses_stale_draft_without_consumption(
    monkeypatch,
):
    from tldw_chatbook.UI.Console_Modules.wiring import receive_console_visible_intent

    async with _received_console_case(monkeypatch, "switch-settlement") as case:
        successor = case.store.create_session(title="Next draft owner", activate=False)
        case.store.set_session_draft(successor.id, "Other saved draft")
        case.console._session._capture_console_draft_switch_snapshot()
        case.store.switch_session(successor.id)
        case.composer.insert_text(" new typing")
        visible = case.composer.draft_text()
        captured = case.composer.capture_draft_for_send()
        previous = case.store.session_draft(case.session.id)
        assert visible != previous

        result = receive_console_visible_intent(
            case.console, visible, case.session.id, captured
        )

        assert result == ""
        assert case.store.received_turn_for_session(case.session.id) is None
        assert case.store.received_turn_for_session(successor.id) is None
        assert case.store.session_draft(case.session.id) == previous
        assert case.store.session_draft(successor.id) == "Other saved draft"
        assert case.composer.draft_text() == visible
        assert case.provider_calls == []


class _NaturalRunFeedbackFrames(_NaturalPreparingFrame):
    """Read existing compositor updates for the visible run indicator."""

    def __init__(self, case):
        super().__init__(case)
        self.frames = []

    def _returned(self, code, _offset, _value):
        if code is not self.code:
            return
        frame = sys._getframe(1)
        if (
            frame.f_locals.get("self") is not self.case.host
            or frame.f_locals.get("screen") is not self.case.console
            or self.case.host.screen is not self.case.console
        ):
            return
        parent = frame.f_back
        while parent is not None and parent.f_code is not self.refresh_code:
            parent = parent.f_back
        if parent is None:
            return
        for selector in ("#console-run-chip", "#console-status-collapsed-copy"):
            widget = self.case.console.query_one(selector)
            for marker in ("Sending", "Waiting for reply", "Thinking", "Streaming"):
                if _update_contains(
                    frame.f_locals.get("renderable"), widget.region, marker
                ):
                    entry = (marker, selector)
                    if not self.frames or self.frames[-1] != entry:
                        self.frames.append(entry)


@pytest.mark.parametrize("route,collapsed", [("enter", False), ("send-button", True)])
async def test_send_feedback_survives_preparation_and_provider_wait(
    route, collapsed, monkeypatch
):
    """Only the external adapter is held; admission and saving stay real."""
    from tldw_chatbook.Widgets.Console.console_status_chips import ConsoleStatusChips

    async with _received_console_case(
        monkeypatch, f"progress-{route}", durable=True
    ) as case:
        chips = case.console.query_one("#console-status-chips", ConsoleStatusChips)
        chips.set_collapsed(collapsed)
        entered, release = threading.Event(), threading.Event()

        def reply(**kwargs):
            case.provider_calls.append(kwargs)
            entered.set()
            if not release.wait(15):
                raise RuntimeError("test provider wait timed out")
            if kwargs.get("streaming"):
                chunk = {
                    "choices": [
                        {
                            "delta": {"content": "received intent reply"},
                            "finish_reason": None,
                        }
                    ]
                }
                return iter(["data: " + json.dumps(chunk) + "\n\n", "data: [DONE]\n\n"])
            return {"choices": [{"message": {"content": "received intent reply"}}]}

        monkeypatch.setattr("tldw_chatbook.Chat.Chat_Functions.chat_api_call", reply)
        frames = _NaturalRunFeedbackFrames(case)
        selector = (
            "#console-status-collapsed-copy" if collapsed else "#console-run-chip"
        )
        with frames.installed():
            try:
                _send(case, route)
                await _held_received_record(case)
                assert await _until(lambda: ("Sending", selector) in frames.frames, 2)
                case.probe.release.set()
                assert await _until(entered.is_set, 15)
                assert await _until(
                    lambda: ("Waiting for reply", selector) in frames.frames, 5
                ), frames.frames
                feedback = case.console.query_one(selector)
                assert feedback.display and feedback.parent.display
                assert "Waiting for reply" in str(feedback.render())
                assert not release.is_set()
                assert case.composer.draft_text() == ""
                release.set()
                assert await _until(
                    lambda: not case.runtime.has_custodied_turns(case.session.id), 15
                )
                assert await _until(
                    lambda: not case.console.query_one("#console-run-chip").display, 5
                )
                assert len(case.provider_calls) == 1
                assert any(
                    message.role.value == "assistant"
                    and message.status == "complete"
                    and message.content == "received intent reply"
                    for message in case.store.messages_for_session(case.session.id)
                )
                assert case.store.received_turn_for_session(case.session.id) is None
                assert (
                    str(
                        case.console.query_one(
                            "#console-status-collapsed-copy"
                        ).render()
                    )
                    == "Status hidden"
                )
            finally:
                release.set()
                case.probe.release.set()


async def test_stopping_received_send_clears_feedback_and_keeps_draft(monkeypatch):
    async with _received_console_case(monkeypatch, "feedback-stop") as case:
        frames = _NaturalRunFeedbackFrames(case)
        with frames.installed():
            _send(case, "enter")
            await _held_received_record(case)
            assert await _until(
                lambda: ("Sending", "#console-run-chip") in frames.frames, 2
            )
            case.controller.stop_active_run()
            case.probe.release.set()
            assert await _until(
                lambda: not case.runtime.has_custodied_turns(case.session.id), 15
            )
            assert await _until(
                lambda: not case.console.query_one("#console-run-chip").display, 5
            )
            assert case.composer.draft_text() == case.draft
            assert case.store.session_draft(case.session.id) == case.draft
            assert case.provider_calls == []


async def test_unrelated_acknowledgement_does_not_admit_a_blocked_pressed_capture(
    monkeypatch,
):
    from tldw_chatbook.UI.Console_Modules.send_acknowledgement import (
        acknowledgement_for,
    )
    from tldw_chatbook.UI.Console_Modules.wiring import receive_console_visible_intent

    async with _received_console_case(monkeypatch, "unrelated-ack") as case:
        inputs = case.store.session_input_snapshot(case.session.id)
        stash = case.composer.capture_draft_for_send()
        ack = acknowledgement_for(case.console)
        token = ack.begin(case.session.id, case.draft)
        assert token is not None
        case.composer._send_blocked = True
        try:
            assert (
                receive_console_visible_intent(
                    case.console,
                    case.draft,
                    case.session.id,
                    stash,
                    _captured_inputs=inputs,
                )
                is None
            )
            assert case.store.received_turn_for_session(case.session.id) is None
            assert not case.runtime.has_custodied_turns(case.session.id)
            assert not case.probe.entered.is_set()
        finally:
            ack.release(token)
