"""Actual Console input must enter runtime custody before native preparation."""

import asyncio
import contextlib
import gc
import json
import os
import sys
import threading
import time
import warnings
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

#: TASK-38718: run feedback is the status line's "run" segment since
#: TASK-35160 (c5ceca1247) retired the run chip and the Status collapsible.
_RUN_FEEDBACK = "#console-status-text"


def _run_segment(console) -> tuple[bool, str]:
    """Whether the status line shows the run segment, and its text."""
    from tldw_chatbook.Widgets.Console.console_status_chips import (
        ConsoleStatusChips,
    )

    strip = console.query_one("#console-status-chips", ConsoleStatusChips)
    return strip.segment_visible("run"), strip.segment_text("run")


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
        # Loop-thread CPU at the action and at the Sending frame (TASK-38948).
        self.action_cpu = self.feedback_frame_cpu = None
        self.while_held = self.feedback_while_held = False
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
        chip = self.case.console.query_one(_RUN_FEEDBACK)
        if self.feedback_frame_at is None and _update_contains(
            frame.f_locals.get("renderable"), chip.region, "Sending"
        ):
            self.feedback_frame_at = time.perf_counter()
            self.feedback_frame_cpu = time.thread_time()
            self.feedback_while_held = not self.case.probe.release.is_set()
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


class _TimedInputFrame(_NaturalInputFrame):
    """Also stamp the event loop thread's CPU at the mutation and the frame.

    ``key_cpu`` is stamped when the loop starts handling the typed key, so the
    loop's CPU between the key and each observation can be held to the budget.
    The stamps ride the timestamps' setters: wrapping ``_returned`` would add
    a frame to the ``sys._getframe(1)`` read it makes of the observed call.
    """

    key_cpu = mutated_cpu = frame_cpu = None

    @property
    def mutated_at(self):
        return self._mutated_at

    @mutated_at.setter
    def mutated_at(self, value):
        self._mutated_at = value
        if value is not None:
            self.mutated_cpu = time.thread_time()

    @property
    def frame_at(self):
        return self._frame_at

    @frame_at.setter
    def frame_at(self, value):
        self._frame_at = value
        if value is not None:
            self.frame_cpu = time.thread_time()


#: How long the held provider read waits for both paints (TASK-38948).
_HELD_READER_SECONDS = 5.0


def _assert_send_budget(record_property, label, start, end, loop_cpu, budget=0.1):
    """Hold the loop's CPU to the budget at any load; wall time on a free host.

    TASK-38948 (AC#2). ``loop_cpu`` is the event loop thread's CPU time in
    the interval. It is held to the budget at every load, so work added on
    the loop fails even on a loaded runner. Wall time is held to the budget
    only while the 1-minute load average is under the CPU count. At or over
    that, a wall-clock miss is reported, not failed: a ``<label>_host_load_miss``
    property and a warning carrying the load and the loop's CPU and off-CPU
    time. A blocking wait added before the paint is therefore caught only on
    a host that is not oversubscribed.
    """
    elapsed = end - start
    record_property(f"{label}_loop_cpu_seconds", loop_cpu)
    lag = (
        f"the loop ran {loop_cpu * 1000:.0f} ms on the CPU and was off it "
        f"{(elapsed - loop_cpu) * 1000:.0f} ms"
    )
    assert loop_cpu <= budget, (
        f"{label}: the event loop spent {loop_cpu * 1000:.0f} ms of CPU against "
        f"{budget * 1000:.0f} ms ({elapsed * 1000:.0f} ms wall)"
    )
    if elapsed <= budget:
        return
    cores = os.cpu_count() or 1
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else 0.0
    assert load >= cores, (
        f"{label}: {elapsed * 1000:.0f} ms against {budget * 1000:.0f} ms at "
        f"load {load:.1f} on {cores} CPUs; {lag}"
    )
    record_property(f"{label}_host_load_miss", load)
    warnings.warn(
        f"{label} missed {budget * 1000:.0f} ms ({elapsed * 1000:.0f} ms) at load "
        f"{load:.0f} on {cores} CPUs, so wall time was not enforced; {lag}",
        stacklevel=2,
    )


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


@pytest.mark.parametrize("route", ["enter", "send-button", "original-handler"])
async def test_stock_received_image_refused_enter_preserves_custody(
    route, monkeypatch, record_property
):
    """Stock readers keep the image and draft through a refusal, without contact."""
    import copy
    from textual.widgets import Static
    from Tests.UI.test_console_send_strip_codes import _image
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_received_dispatch import stock_native_methods
    from tldw_chatbook.UI.Screens.chat_screen import _CONSOLE_RECEIVED_SCREEN_SELECTORS

    async with _received_console_case(monkeypatch, f"image-refused-{route}") as case:
        assert stock_native_methods(case.console, _CONSOLE_RECEIVED_SCREEN_SELECTORS)
        image = _image()
        assert case.store.add_pending_attachment(case.session.id, image)
        refusal = case.console._console_attachment_blocked_reason()
        assert "local-model can't accept images" in refusal
        from tldw_chatbook.UI.Console_Modules.prompt_queue import (
            ConsolePromptDispatchResult,
            ConsolePromptDispatchStatus,
            ConsolePromptQueueUIController,
        )

        results = []
        monitoring = sys.monitoring
        tool = next(value for value in range(6) if monitoring.get_tool(value) is None)
        monitoring.use_tool_id(tool, "stock-image-refusal-result")
        code = ConsolePromptQueueUIController.dispatch.__code__

        def returned(_code, _offset, value):
            if isinstance(value, ConsolePromptDispatchResult):
                results.append(value)

        monitoring.register_callback(tool, monitoring.events.PY_RETURN, returned)
        monitoring.set_local_events(tool, code, monitoring.events.PY_RETURN)
        record = task = None
        try:
            if route == "original-handler":
                await case.console.handle_console_send_message(
                    Button.Pressed(
                        case.console.query_one("#console-send-message", Button)
                    )
                )
            else:
                _send(case, route)
            assert await _until(
                lambda: case.probe.entered.is_set()
                or any(
                    row.role is ConsoleMessageRole.SYSTEM and row.content == refusal
                    for row in case.store.messages_for_session(case.session.id)
                ),
                15,
            )
            if case.probe.entered.is_set():
                record = await _held_received_record(case)
                claim, task = record.received_claim, record.task
                assert record.received_intent.inputs.attachments == (image,)
                assert record.session_id == case.session.id
                assert case.store.received_turn_for_session(case.session.id) is claim
                case.probe.release.set()
                assert await _until(case.probe.retired, 5)
            else:
                # A natural Enter frame may paint the existing blocker before
                # receipt. Observe the real dispatch result, never fake a gate
                # or force a held configuration phase that did not happen.
                assert await _until(lambda: bool(results), 5)
                assert results[-1].status is ConsolePromptDispatchStatus.REFUSED
                assert results[-1].detail == refusal
                assert not case.probe.entered.is_set()
                # This attempt never entered received preparation. Stop only
                # its still-unentered native-read observation before the later
                # independent Context preview (which may use a UI DB reader).
                native_tool = next(
                    value
                    for value in range(6)
                    if monitoring.get_tool(value) == "configuration-original-workspace"
                )
                monitoring.set_local_events(native_tool, case.probe.code, 0)
                case.probe.release.set()
            assert await _until(
                lambda: not case.runtime.has_custodied_turns(case.session.id), 15
            )
        finally:
            monitoring.set_local_events(tool, code, 0)
            monitoring.register_callback(tool, monitoring.events.PY_RETURN, None)
            monitoring.free_tool_id(tool)
        rows = case.store.messages_for_session(case.session.id)
        system = [row for row in rows if row.role is ConsoleMessageRole.SYSTEM]
        assert system and type(system[-1].content) is str  # noqa: E721 -- exact plain payload/terminal type contract
        assert system[-1].content == refusal
        assert case.provider_calls == []
        assert not any(row.role is ConsoleMessageRole.USER for row in rows)
        assert case.store.pending_attachments(case.session.id) == [image]
        assert (
            case.composer.draft_text()
            == case.store.session_draft(case.session.id)
            == case.draft
        )
        assert case.store.received_turn_for_session(case.session.id) is None
        assert case.store.preparation_for_session(case.session.id) is None
        if record is not None:
            assert record.request is None and record.task is None
            assert record.inputs.durable_accepted is False
            assert task.done()
            assert (
                type(task.exception()) is RuntimeError
            )  # exact image preflight terminal
            assert (
                str(task.exception())
                == "The selected model cannot accept this attachment; draft kept."
            )
        else:
            assert (
                route != "original-handler"
            ), "Original stock handler must reach checked received-image preflight"
        record_property(
            "pr3330_refusal_owner",
            "checked_received_preflight"
            if record is not None
            else "coded_dispatch_before_receipt",
        )
        ack = vars(case.console).get("_console_send_ack")
        if ack is not None:
            assert case.session.id not in ack._pending

        def painted():
            return [
                str(widget.content)
                for widget in case.console.query_one(
                    "#console-native-transcript"
                ).query(Static)
            ]

        assert await _until(
            lambda: any(refusal in text for text in painted())
            and not any("Sending…" in text for text in painted()),
            10,
        )
        copy.deepcopy(rows)
        assert (
            await case.controller.build_context_snapshot("", session_id=case.session.id)
            is not None
        )


@pytest.mark.parametrize("route", ["enter", "send-button"])
async def test_actual_send_paints_sending_and_accepts_input_within_100ms(
    route, monkeypatch, record_property
):
    """Headless supplied frames qualify readiness, not physical terminal flush."""
    async with _received_console_case(monkeypatch, f"feedback-{route}") as case:
        # Finish mount-only discovery before timing Send. This app factory's
        # synthetic policy keeps that discovery on the compatibility path.
        assert await _until(
            lambda: not any(
                worker.group == "console-skill-discovery" and not worker.is_finished
                for worker in case.console.workers
            ),
            10,
        )
        preparing = _NaturalPreparingFrame(case)
        typing = _TimedInputFrame(case.host, case.console, case.composer, case.probe)
        loop = asyncio.get_running_loop()
        stop = threading.Event()
        receipt_observed = threading.Event()

        def press_z():
            typing.key_cpu = time.thread_time()
            _key(case.host, "Z", "Z")

        def release_independently():
            try:
                while not case.probe.entered.wait(0.01):
                    if stop.is_set():
                        return
                typing.sent_at = time.perf_counter()
                loop.call_soon_threadsafe(press_z)
                # The reader stays held until both paints and the receipt
                # are seen, or this long. It is the test's patience, not a
                # budget: the budgets below come from the frames' own
                # timestamps. 0.5 s released before a slow frame at load
                # 105 (TASK-38948).
                deadline = time.perf_counter() + _HELD_READER_SECONDS
                typing.painted.wait(_HELD_READER_SECONDS)
                preparing.painted.wait(max(0, deadline - time.perf_counter()))
                receipt_observed.wait(max(0, deadline - time.perf_counter()))
            finally:
                case.probe.release.set()

        releaser = threading.Thread(target=release_independently)
        with preparing.installed(), typing.installed():
            releaser.start()
            try:
                # Start with no collection owed: a gen-2 pass over the worker's
                # accumulated heap read as 483 ms of loop CPU in the interval
                # late in a mixed run (TASK-38948 review r12t-rev1). A full
                # collection zeroes CPython's long-lived pending count, so no
                # automatic full pass can run within the short interval. No
                # freeze: unfreezing would empty the product's own frozen set.
                gc.collect()
                # All Send-triggered preparation remains inside this interval.
                preparing.action_at = time.perf_counter()
                preparing.action_cpu = time.thread_time()
                _send(case, route)
                record = await _held_received_record(case)
                ack = vars(case.console).get("_console_send_ack")
                pending = vars(ack)["_pending"][case.session.id]
                assert pending.admitted and pending.handoff_ids is not None
                received_pending_identity = (id(ack), id(pending.token), pending.row.id)
                assert record.received_intent.inputs.draft == pending.row.content
                receipt_observed.set()
                assert await _until(case.probe.release.is_set, _HELD_READER_SECONDS + 2)
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
                record_property("held_reader_budget_seconds", _HELD_READER_SECONDS)
                record_property(
                    "send_to_run_feedback_frame_seconds",
                    None
                    if preparing.feedback_frame_at is None
                    else preparing.feedback_frame_at - preparing.action_at,
                )
                # A late Sending frame is waited for, so the held-read check
                # below names it instead of reading None.
                await _until(
                    lambda: preparing.feedback_frame_at is not None,
                    _HELD_READER_SECONDS,
                )
                assert preparing.feedback_frame_at is not None
                # A loaded host only reports wall misses; the Sending frame
                # must still land before the held read is released.
                assert preparing.feedback_while_held, (
                    "Sending painted only after the held read was released"
                )
                assert preparing.frame_at is not None and preparing.while_held
                # Preparing follows the immediate Sending frame at real receipt.
                assert typing.mutated_while_held and typing.frame_while_held
            finally:
                stop.set()
                case.probe.release.set()
                releaser.join(timeout=_HELD_READER_SECONDS + 2)
                assert not releaser.is_alive()
        for label, start, end, loop_cpu in (
            (
                "send_to_run_feedback_frame",
                preparing.action_at,
                preparing.feedback_frame_at,
                preparing.feedback_frame_cpu - preparing.action_cpu,
            ),
            (
                "input_mutation",
                typing.sent_at,
                typing.mutated_at,
                typing.mutated_cpu - typing.key_cpu,
            ),
            (
                "input_frame",
                typing.sent_at,
                typing.frame_at,
                typing.frame_cpu - typing.key_cpu,
            ),
        ):
            _assert_send_budget(record_property, label, start, end, loop_cpu)


#: TASK-39166: patience for a released durable turn to settle. At load
#: 100-110 one took over the 15 s these tests used to allow; the wait ends as
#: soon as the turn's own task does, so only a stuck turn uses all of it.
_SETTLE_SECONDS = 60.0


async def _turn_settles(case, task):
    """Wait for the released turn's task, then for its custody to clear."""
    done, _pending = await asyncio.wait({task}, timeout=_SETTLE_SECONDS)
    assert done, (
        f"the released turn's task did not finish within {_SETTLE_SECONDS:g} s: "
        f"{task!r}; provider calls={len(case.provider_calls)}"
    )
    assert await _until(
        lambda: not case.runtime.has_custodied_turns(case.session.id), _SETTLE_SECONDS
    ), f"the turn finished ({task!r}) but its custody never cleared"


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
        await _turn_settles(case, record.task)
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
            assert await _until(paint_entered.is_set, 10), "the held ack paint never ran"
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
            ), "the newer typed draft never reached the session's store"
            assert case.composer.draft_text() == newer
            paint_release.set()
            record = await _held_received_record(case)
            assert record.received_intent.inputs.draft == case.draft
            assert case.store.session_draft(case.session.id) == newer
            case.probe.release.set()
            await _turn_settles(case, record.task)
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
        ), "the popped Console never detached from the screen stack and runtime"
        case.store.switch_session(successor.id)
        assert case.runtime._turn_custody[record.turn_id] is record
        assert case.store.received_turn_for_session(case.session.id) is claim
        assert record.request is None and not record.task.done()
        assert not case.probe.retired()

        case.probe.release.set()
        await _turn_settles(case, task)
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
        widget = self.case.console.query_one(_RUN_FEEDBACK)
        for marker in ("Sending", "Waiting for reply", "Thinking", "Streaming"):
            if _update_contains(frame.f_locals.get("renderable"), widget.region, marker):
                entry = (marker, _RUN_FEEDBACK)
                if not self.frames or self.frames[-1] != entry:
                    self.frames.append(entry)


@pytest.mark.parametrize("route", ["enter", "send-button"])
async def test_send_feedback_survives_preparation_and_provider_wait(route, monkeypatch):
    """Only the external adapter is held; admission and saving stay real.

    TASK-38718: the collapsed-Status arm is retired with the Status
    collapsible (TASK-35160); the status line is always shown.
    """
    async with _received_console_case(
        monkeypatch, f"progress-{route}", durable=True
    ) as case:
        entered, release = threading.Event(), threading.Event()

        def reply(**kwargs):
            case.provider_calls.append(kwargs)
            entered.set()
            # The test releases in its finally; this only ends a stray thread.
            if not release.wait(_SETTLE_SECONDS):
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
        selector = _RUN_FEEDBACK
        with frames.installed():
            try:
                _send(case, route)
                record = await _held_received_record(case)
                assert await _until(
                    lambda: ("Sending", selector) in frames.frames, 2
                ), f"Sending never painted while the read was held: {frames.frames}"
                case.probe.release.set()
                await _until(
                    lambda: entered.is_set() or record.task.done(), _SETTLE_SECONDS
                )
                assert entered.is_set(), (
                    "the turn never reached the provider: "
                    + (f"it ended first ({record.task!r})" if record.task.done()
                       else f"not within {_SETTLE_SECONDS:g} s")
                )
                assert await _until(
                    lambda: ("Waiting for reply", selector) in frames.frames, 5
                ), frames.frames
                feedback = case.console.query_one(selector)
                assert feedback.display and feedback.parent.display
                shown, text = _run_segment(case.console)
                assert shown and "Waiting for reply" in text, text
                assert not release.is_set()
                assert case.composer.draft_text() == ""
                release.set()
                await _turn_settles(case, record.task)
                assert await _until(
                    lambda: not _run_segment(case.console)[0], 5
                ), "the run segment never hid after the turn settled"
                assert len(case.provider_calls) == 1
                assert any(
                    message.role.value == "assistant"
                    and message.status == "complete"
                    and message.content == "received intent reply"
                    for message in case.store.messages_for_session(case.session.id)
                )
                assert case.store.received_turn_for_session(case.session.id) is None
            finally:
                release.set()
                case.probe.release.set()


async def test_stopping_received_send_clears_feedback_and_keeps_draft(monkeypatch):
    async with _received_console_case(monkeypatch, "feedback-stop") as case:
        frames = _NaturalRunFeedbackFrames(case)
        with frames.installed():
            _send(case, "enter")
            await _held_received_record(case)
            assert await _until(lambda: ("Sending", _RUN_FEEDBACK) in frames.frames, 2)
            case.controller.stop_active_run()
            case.probe.release.set()
            assert await _until(
                lambda: not case.runtime.has_custodied_turns(case.session.id), 15
            )
            assert await _until(lambda: not _run_segment(case.console)[0], 5)
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


async def test_delayed_stock_receipt_keeps_original_view_generation(monkeypatch):
    from tldw_chatbook.UI.Console_Modules.wiring import receive_console_visible_intent

    class Successor:
        def console_view_hooks(self):
            return {}

        def _project_console_received_input(self, *args, **kwargs):
            raise AssertionError("old pressed draft reached successor composer")

    async with _received_console_case(monkeypatch, "delayed-view-generation") as case:
        original_generation = case.console._console_runtime_attachment_generation
        stash = case.composer.capture_draft_for_send()
        captured = case.store.session_input_snapshot(case.session.id)
        successor = Successor()
        successor_generation = case.runtime.attach_view(successor)
        assert successor_generation != original_generation
        turn_id = receive_console_visible_intent(
            case.console, case.draft, case.session.id, stash, _captured_inputs=captured
        )
        assert turn_id
        record = case.runtime._turn_custody[turn_id]
        assert record.received_intent.view_attachment_generation == original_generation
        case.runtime._project_received_input(record)
