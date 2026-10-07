"""Actual Console input must enter runtime custody before native preparation."""

import asyncio
import contextlib
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
    from tldw_chatbook import config

    app = _build_test_app(user_data_dir=config.get_user_data_dir())
    _configure_native_ready_console(app)
    registry = app.workspace_registry_service
    workspace_id = f"received-intent-{route}"
    registry.create_workspace(
        workspace_id=workspace_id, name=f"Received intent {route}"
    )
    provider_calls = []

    def reply(**kwargs):
        provider_calls.append(kwargs)
        return "received intent reply"

    # Keep all original admission, capture and storage bodies.
    monkeypatch.setattr("tldw_chatbook.Chat.Chat_Functions.chat_api_call", reply)
    host = _VisibleNavigatedConsoleHarness(app)
    runtime = None
    probe = None
    try:
        with _task_factory("eager"):
            async with host.run_test(size=(120, 40)) as pilot:
                assert await _until(
                    lambda: isinstance(host.screen, ChatScreen)
                    and host.screen.is_mounted,
                    10,
                )
                console = host.screen
                runtime = console._console_runtime()
                controller = console._ensure_console_chat_controller()
                store = controller.store
                session = store.ensure_session()
                session.workspace_id = workspace_id
                composer = console._console_composer_or_none()
                original_draft = "Receive this before checked preparation"
                composer.load_draft(original_draft)
                composer.focus()
                await pilot.pause()
                assert host.focused is composer
                draft_region = console.query_one("#console-command-visible-text").region
                assert draft_region and draft_region.overlaps(console.region)
                probe = _OriginalConfigurationWorkspaceRead(registry)

                with probe.installed():
                    try:
                        if route == "enter":
                            _key(host, "enter", "\r")
                        else:
                            console.query_one("#console-send-message", Button).press()
                        assert await _until(
                            probe.entered.is_set, 10
                        ), "stock Send did not reach original configuration Workspace SQL"
                        assert probe.live_at_entry and not probe.release_timed_out
                        assert not probe.release.is_set()
                        assert provider_calls == []
                        assert composer.draft_text() == original_draft

                        # Existing API: this fails on the original delayed receipt,
                        # independently of any new intent class or method.
                        claim = store.received_turn_for_session(session.id)
                        assert (
                            claim is not None
                        ), "Send reached native configuration before received admission"
                        record = runtime._turn_custody[claim.request_id]
                        assert record.received_claim is claim
                        assert record.store is store and record.session_id == session.id
                        assert runtime.has_custodied_turns(session.id)
                        assert record.task is not None and not record.task.done()
                        assert record.request is None
                        assert store.preparation_for_session(session.id) is None
                    finally:
                        # Release the actual native body before any await/teardown,
                        # including when the intended original-API assertion is RED.
                        probe.release.set()
                        controller.stop_active_run()
                        host.workers.cancel_group(console, "console-hook-send-review")
                        assert await _until(lambda: not console._hooks._busy, 15)
                        if probe.entered.is_set():
                            assert await _until(
                                probe.retired, 5
                            ), "original configuration reader did not physically retire"
    finally:
        if probe is not None:
            probe.release.set()
        if runtime is not None:
            await runtime.dispose()


class _NaturalPreparingFrame:
    """Observe the supplied natural button frame, never request a repaint."""

    def __init__(self, case):
        self.case = case
        self.code = App._display.__code__
        self.refresh_code = Screen._compositor_refresh.__code__
        self.action_at = self.frame_at = None
        self.while_held = False
        self.painted = threading.Event()

    def _returned(self, code, _offset, _value):
        if code is not self.code or self.action_at is None or self.frame_at is not None:
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
        button = self.case.console.query_one("#console-send-message", Button)
        if _update_contains(
            frame.f_locals.get("renderable"), button.region, "Preparing"
        ):
            self.frame_at = time.perf_counter()
            self.while_held = not self.case.probe.release.is_set()
            self.painted.set()

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
                    and host.screen.is_mounted,
                    10,
                )
                console = host.screen
                runtime = console._console_runtime()
                controller = console._ensure_console_chat_controller()
                store = controller.store
                session = store.ensure_session()
                session.workspace_id = workspace_id
                composer = console._console_composer_or_none()
                draft = "An exact received draft"
                composer.load_draft(draft)
                composer.focus()
                await pilot.pause()
                assert host.focused is composer
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
                await _held_received_record(case)
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
                record_property("headless_supplied_frame_only", True)
                record_property("held_reader_budget_seconds", 0.5)
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


async def test_identical_retyping_synchronously_invalidates_unpromoted_received_intent(
    monkeypatch,
):
    async with _received_console_case(monkeypatch, "identical-retype") as case:
        _send(case, "enter")
        record = await _held_received_record(case)
        intent = record.received_intent
        before = intent.inputs.draft_revision
        case.composer.clear_draft()
        case.composer.insert_text(case.draft)

        # No await between authored mutation and observing the domain witness.
        assert case.store.session_draft(case.session.id) == case.draft
        assert (
            case.store.session_input_snapshot(case.session.id).draft_revision > before
        )
        assert case.composer.draft_text() == case.draft
        assert record.received_intent is intent and record.request is None
        case.probe.release.set()
        assert await _until(
            lambda: not case.runtime.has_custodied_turns(case.session.id), 15
        )
        assert case.provider_calls == []
        assert case.store.received_turn_for_session(case.session.id) is None
        assert case.store.session_draft(case.session.id) == case.draft
        assert case.composer.draft_text() == case.draft


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
