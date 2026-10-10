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
    assert await _until(
        lambda: case.console._console_attach_reconciled
        and not case.console._console_attach_reconcile_running,
        5,
    ), "Original attachment workflow did not finish before stable Send qualification"


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
        self.eligibility_code = ChatScreen._console_preparing_poll_record.__code__
        self.tab_code = ChatScreen._sync_console_native_session_tabs.__code__
        self.transcript_code = ChatScreen._sync_native_console_transcript.__code__
        self.roleplay_code = (
            ChatScreen._dispatch_active_console_roleplay_refresh.__code__
        )
        self.active = {}
        self.completed = []

    def _started(self, code, _offset):
        frame = sys._getframe(1)
        if code is self.poll_code and frame.f_locals.get("self") is self.case.console:
            self.active[id(frame)] = {
                "core_returns": 0,
                "roleplay_returns": 0,
                "tab_returns": 0,
                "transcript_returns": 0,
                "full_results": [],
                "eligibility_refusals": {},
                "eligible_returns": 0,
            }

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
        elif code is self.tab_code:
            row["tab_returns"] += 1
        elif code is self.transcript_code:
            row["transcript_returns"] += 1
        elif code is self.roleplay_code:
            row["roleplay_returns"] += 1
        elif code is self.eligibility_code:
            if value is self.record:
                row["eligible_returns"] += 1
            else:
                refusals = row["eligibility_refusals"]
                refusals[frame.f_lineno] = refusals.get(frame.f_lineno, 0) + 1
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

    def qualified_indexes(self):
        return [
            index
            for index, row in enumerate(self.completed)
            if row["held"]
            and not row["deferred"]
            and row["full_results"]
            and all(value is True for value in row["full_results"])
            and row["tab_returns"] > 0
            and row["transcript_returns"] > 0
        ]

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
            for code in (
                self.full_code,
                self.core_code,
                self.tab_code,
                self.transcript_code,
                self.roleplay_code,
                self.eligibility_code,
            ):
                monitoring.set_local_events(tool, code, monitoring.events.PY_RETURN)
            yield self
        finally:
            for code in (
                self.poll_code,
                self.full_code,
                self.core_code,
                self.tab_code,
                self.transcript_code,
                self.roleplay_code,
                self.eligibility_code,
            ):
                monitoring.set_local_events(tool, code, 0)
            monitoring.register_callback(tool, monitoring.events.PY_START, None)
            monitoring.register_callback(tool, monitoring.events.PY_RETURN, None)
            monitoring.free_tool_id(tool)
            self.active.clear()


async def test_ordinary_preparing_polls_do_not_repeat_live_core_reconciliation(
    monkeypatch,
    record_property,
):
    """Report deferred rows separately from completed, unchanged poll work."""
    from tldw_chatbook.UI.Console_Modules import poll_cadence

    # Lever L2: the routine full pass still lands every
    # CONSOLE_POLL_FULL_SYNC_INTERVAL_SECONDS (Tests/UI/test_console_poll_
    # cadence.py proves that cadence). Pin it out of this short window so the
    # observed rows measure only unchanged routine polls, deterministically.
    monkeypatch.setattr(poll_cadence, "CONSOLE_POLL_FULL_SYNC_INTERVAL_SECONDS", 3600.0)
    async with _received_console_case(
        monkeypatch, "poll-reconciliation", durable=True
    ) as case:
        await _qualify_warm_capture_sources(case)
        _send(case, "enter")
        record = await _held_received_record(case)
        assert case.console._console_transcript_sync_timer is not None
        observed = _OriginalPreparingPolls(case, record)
        with observed.installed():
            completed = await _until(lambda: len(observed.qualified_indexes()) >= 2, 5)
        rows = observed.completed
        qualified = observed.qualified_indexes()
        deferred = [
            index
            for index, row in enumerate(rows)
            if row["deferred"] or any(value is False for value in row["full_results"])
        ]
        record_property("original_preparing_poll_observations", rows)
        record_property("qualified_poll_indexes", qualified)
        record_property("deferred_poll_indexes", deferred)
        assert all(row["held"] for row in rows), "Original Preparing hold expired"
        assert completed, (
            f"Two healthy original polls did not complete: qualified={qualified!r}, "
            f"deferred={deferred!r}, rows={rows!r}"
        )
        assert case.provider_calls == []
        assert case.composer.draft_text() == case.draft
        # The fixture selected a workspace after mount, and Send may owe an
        # initial full reconciliation. Permit the first healthy poll to
        # settle that work. Deferred callbacks remain recorded above, but
        # do not count their necessary retry work as redundant preparation.
        steady_rows = [rows[index] for index in qualified[1:]]
        assert (
            sum(row["core_returns"] for row in steady_rows) == 0
        ), f"Unchanged Preparing polls repeated original live core reconciliation: {rows!r}"
        assert sum(row["roleplay_returns"] for row in steady_rows) == 0, rows


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


@contextlib.contextmanager
def _original_transition_events(bindings, observe):
    """Observe original code locally; keep no frames and replace no callbacks."""
    from functools import partial

    captured = []
    by_code = {}
    for label, owner, name in bindings:
        binding = getattr(owner, name)
        function = inspect.unwrap(binding)
        chain = []
        current = binding
        while True:
            chain.append(
                (
                    current,
                    current.__code__,
                    current.__globals__,
                    current.__defaults__,
                    current.__kwdefaults__,
                    getattr(current, "__wrapped__", None),
                )
            )
            if not hasattr(current, "__wrapped__"):
                break
            current = current.__wrapped__
        captured.append((owner, name, binding, chain))
        assert function.__code__ not in by_code
        by_code[function.__code__] = (label, function.__globals__)
    invalid = []
    count = 0

    def event(kind, code, _offset, *values):
        nonlocal count
        frame = None
        try:
            count += 1
            assert count <= 4096, "original_transition_observer_overflow"
            label, namespace = by_code[code]
            frame = sys._getframe(1)
            assert frame.f_code is code and frame.f_globals is namespace
            observe(kind, label, frame, values[0] if values else None)
        except BaseException as error:
            if len(invalid) < 16:
                invalid.append((type(error).__name__, str(error)))
        finally:
            del frame

    monitoring = sys.monitoring
    tool = next(value for value in range(6) if monitoring.get_tool(value) is None)
    monitoring.use_tool_id(tool, "original-polling-transitions")
    events = (
        monitoring.events.PY_START,
        monitoring.events.PY_YIELD,
        monitoring.events.PY_RETURN,
    )
    try:
        for kind in events:
            monitoring.register_callback(tool, kind, partial(event, kind))
        for code in by_code:
            monitoring.set_local_events(tool, code, events[0] | events[1] | events[2])
        yield invalid
    finally:
        for code in by_code:
            monitoring.set_local_events(tool, code, 0)
        for kind in events:
            monitoring.register_callback(tool, kind, None)
        monitoring.free_tool_id(tool)
        for owner, name, binding, chain in captured:
            assert (
                getattr(owner, name) is binding
            ), "original_transition_binding_changed"
            for function, code, namespace, defaults, kwdefaults, wrapped in chain:
                assert function.__code__ is code and function.__globals__ is namespace
                assert function.__defaults__ is defaults
                assert function.__kwdefaults__ is kwdefaults
                assert getattr(function, "__wrapped__", None) is wrapped
        assert not invalid, invalid


def _transition_parent(frame, code, owner):
    """Return only a transient original ancestor; callers must not retain it."""
    parent = frame.f_back
    for _ in range(40):
        if parent is None:
            return None
        if parent.f_code is code and parent.f_locals.get("self") is owner:
            return parent
        parent = parent.f_back
    return None


def _assert_original_configuration_custody(case):
    import threading
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Chat.console_turn_context import (
        resolve_turn_tool_policy_profile_id,
    )

    probe = case.probe
    assert inspect.unwrap(type(probe.registry).get_workspace).__code__ is probe.code
    assert resolve_turn_tool_policy_profile_id.__code__ is probe.caller_code
    assert probe.entered.is_set() and probe.live_at_entry
    assert not probe.release.is_set() and not probe.release_timed_out
    assert probe.thread is not threading.current_thread()
    with storage._lock:
        assert probe.lease in storage._live_leases
        assert (
            probe.registry.db._maintenance_participant.connections.get(probe.connection)
            is probe.lease
        )
        assert probe.lease.resource_thread is probe.thread
    assert not probe.retired()


@pytest.mark.parametrize(
    "preparing_display", [False, True], ids=["original-demand", "preparing-display"]
)
async def test_full_request_during_poll_await_replays_with_current_owner(
    monkeypatch, record_property, preparing_display
):
    """An original session activation cannot lose its full request in a poll."""
    import asyncio
    from textual.timer import Timer
    from textual.worker import Worker
    from textual.worker_manager import WorkerManager
    from textual.widgets import Button
    from tldw_chatbook.UI.Console_Modules import poll_cadence
    from tldw_chatbook.Widgets.Console.console_session_surface import (
        ConsoleSessionSurface,
    )

    # Lever L2: this control's subject is the poll's ORIGINAL full pass, so
    # route every tick there. The light pass's own replay contract is
    # Tests/UI/test_console_poll_cadence.py's late-full controls.
    monkeypatch.setattr(poll_cadence, "CONSOLE_POLL_FULL_SYNC_INTERVAL_SECONDS", 0.0)
    async with _received_console_case(
        monkeypatch, f"late-full-poll-{preparing_display}", durable=True
    ) as case:
        await _qualify_warm_capture_sources(case)
        successor = case.store.create_session(
            title="Current polling owner",
            workspace_id=case.session.workspace_id,
            settings=case.session.settings,
            activate=False,
        )
        case.store.set_session_draft(successor.id, "Current owner draft")
        _send(case, "enter")
        record = await _held_received_record(case)
        _assert_original_configuration_custody(case)
        console = case.console
        assert await _until(
            lambda: not console._console_sync_in_progress
            and console._console_session_tabs_sync_calls == 0
            and not getattr(console, "_console_control_bar_replay_whole_sync", False),
            5,
        ), "Original full reconciliation never became available for the poll hold"
        if preparing_display:
            successful_display = []

            def completed_display(kind, label, frame, value):
                if (
                    kind == sys.monitoring.events.PY_RETURN
                    and frame.f_locals.get("self") is console
                    and frame.f_locals.get("poll_record") is record
                    and value is True
                    and asyncio.current_task()
                    is console._console_transcript_sync_timer._task
                ):
                    successful_display.append(True)

            with _original_transition_events(
                [("completed_display", ChatScreen, "_sync_native_console_chat_ui")],
                completed_display,
            ):
                assert await _until(
                    lambda: bool(successful_display)
                    and not console._console_sync_in_progress,
                    5,
                ), "No original natural Preparing display completed before the hold"
        surface = console.query_one("#console-session-surface", ConsoleSessionSurface)
        lock = surface._session_sync_lock
        assert type(lock) is asyncio.Lock and not lock.locked()
        assert type(case.host.workers) is WorkerManager
        timer = console._console_transcript_sync_timer
        assert type(timer) is Timer
        poll = timer._callback
        assert (
            poll.__code__
            in ChatScreen._start_console_transcript_sync_timer.__code__.co_consts
        )
        full_code = ChatScreen._sync_native_console_chat_ui.__code__
        tab_code = ChatScreen._sync_console_native_session_tabs.__code__
        held = {}
        publications, replays, full_results, late_full_tasks = [], [], [], []
        activation_code = type(
            console._session
        )._activate_native_console_session.__code__
        activation = None
        replay_code = ChatScreen._run_coalesced_control_bar_sync.__code__
        held_return, coalesced = {}, {}
        replay_entries = []
        owned_deferred = set()

        def pending_state():
            # Plain fields only: diagnostic observation cannot advance the UI.
            return {
                "requested": console._console_sync_requested,
                "replay": getattr(
                    console, "_console_control_bar_replay_whole_sync", False
                ),
                "scheduled": console._console_control_bar_sync_scheduled,
                "in_progress": console._console_sync_in_progress,
                "paused": vars(console).get("_console_sync_maintenance_paused", False),
                "active": case.store.active_session_id,
                "visible": console._console_visible_draft_session_id,
                "same_store": console._console_chat_store is case.store,
                "same_view": case.runtime.view is console,
                "activation_done": activation is not None and activation.done(),
            }

        def observe(kind, label, frame, value):
            if label == "surface" and frame.f_locals.get("self") is surface:
                tabs = _transition_parent(frame, tab_code, console)
                if tabs is None:
                    return
                if kind == sys.monitoring.events.PY_YIELD and not held:
                    full = _transition_parent(frame, full_code, console)
                    poll_frame = _transition_parent(frame, poll.__code__, console)
                    tick = _transition_parent(frame, Timer._tick.__code__, timer)
                    if full is None or poll_frame is None or tick is None:
                        return
                    # This is the admitted full pass, not its coalesced tab-only path.
                    if "visit" not in full.f_locals:
                        return
                    display_record = full.f_locals.get("poll_record")
                    if preparing_display:
                        assert display_record is record
                    assert tabs.f_locals["store"] is case.store
                    assert frame.f_locals["active_session_id"] == case.session.id
                    assert any(
                        row is case.session for row in frame.f_locals["sessions"]
                    )
                    assert type(value) is asyncio.Future and not value.done()
                    assert value.get_loop() is asyncio.get_running_loop()
                    assert any(waiter is value for waiter in lock._waiters)
                    assert asyncio.current_task() is timer._task
                    held.update(
                        full=id(full),
                        tabs=id(tabs),
                        task=asyncio.current_task(),
                        future=value,
                        preparing_display=display_record is record,
                    )
                elif (
                    kind == sys.monitoring.events.PY_RETURN
                    and held
                    and id(tabs) == held["tabs"]
                    and not held.get("tabs_returned")
                    and asyncio.current_task() is held["task"]
                ):
                    publications.append(
                        (
                            id(tabs),
                            frame.f_locals["active_session_id"],
                            tuple(frame.f_locals["sessions"]),
                        )
                    )
            elif label == "tabs" and kind == sys.monitoring.events.PY_RETURN and held:
                if id(frame) == held["tabs"] and asyncio.current_task() is held["task"]:
                    held["tabs_returned"] = True
            elif label == "coalesced" and frame.f_locals.get("self") is console:
                if kind == sys.monitoring.events.PY_START:
                    coalesced.clear()
                    # FULL finally may consume requested before PY_RETURN;
                    # its retained replay flag still owns this continuation.
                    if held_return.get("replay") or owned_deferred:
                        state = pending_state()
                        assert len(replay_entries) < 64
                        replay_entries.append(state)
                        coalesced.update(
                            frame=id(frame),
                            task=asyncio.current_task(),
                            eligible=state["replay"]
                            and not state["in_progress"]
                            and not state["paused"],
                        )
                elif kind == sys.monitoring.events.PY_RETURN:
                    coalesced.clear()
            elif label == "worker" and kind == sys.monitoring.events.PY_RETURN and held:
                parent = _transition_parent(frame, full_code, console)
                immediate = (
                    parent is not None
                    and id(parent) == held["full"]
                    and asyncio.current_task() is held["task"]
                    and not held.get("returned")
                )
                delayed = _transition_parent(frame, replay_code, console)
                deferred = (
                    delayed is not None
                    and id(delayed) == coalesced.get("frame")
                    and asyncio.current_task() is coalesced.get("task")
                    and coalesced.get("eligible") is True
                )
                if not (immediate or deferred):
                    return
                assert type(value) is Worker
                work, task = value._work, value._task
                # The same FULL task also starts unrelated display workers.
                # Select the issued FULL body before asserting its route.
                if not inspect.iscoroutine(work) or work.cr_code is not full_code:
                    return
                assert frame.f_locals["self"] is case.host.workers
                assert value.node is console
                assert value.group == "console-sync"
                assert (
                    type(task) is asyncio.Task
                    and task.get_loop() is asyncio.get_running_loop()
                )
                replays.append(
                    (value, task, work, "immediate" if immediate else "coalesced")
                )
            elif label == "full" and kind == sys.monitoring.events.PY_START and held:
                if (
                    frame.f_locals.get("self") is console
                    and _transition_parent(frame, activation_code, console._session)
                    is not None
                ):
                    late_full_tasks.append(asyncio.current_task())
            elif label == "full" and kind == sys.monitoring.events.PY_RETURN:
                if frame.f_locals.get("self") is console:
                    if (
                        held
                        and id(frame) == held["full"]
                        and asyncio.current_task() is held["task"]
                        and not held.get("returned")
                    ):
                        held["returned"] = True
                        held_return.update(pending_state(), result=value)
                    task = asyncio.current_task()
                    if (
                        value is False
                        and getattr(
                            console, "_console_control_bar_replay_whole_sync", False
                        )
                        and any(
                            task is issued for _worker, issued, _work, _route in replays
                        )
                    ):
                        owned_deferred.add(task)
                    full_results.append((task, value, case.store.active_session_id))

        bindings = [
            ("poll", timer, "_callback"),
            ("full", ChatScreen, "_sync_native_console_chat_ui"),
            ("tabs", ChatScreen, "_sync_console_native_session_tabs"),
            ("surface", ConsoleSessionSurface, "sync_sessions"),
            ("tick", Timer, "_tick"),
            ("worker", WorkerManager, "_new_worker"),
            ("coalesced", ChatScreen, "_run_coalesced_control_bar_sync"),
            ("activation", type(console._session), "_activate_native_console_session"),
        ]
        await lock.acquire()
        held_lock = True
        try:
            with _original_transition_events(bindings, observe):
                assert await _until(
                    lambda: bool(held), 5
                ), "No original natural poll reached the real tab lock"
                _assert_original_configuration_custody(case)
                assert not held["task"].done() and not held["future"].done()
                assert (
                    not console._console_sync_requested
                ), "Unrelated full demand preceded the intervention"
                activation = asyncio.create_task(
                    console._session._activate_native_console_session(successor.id)
                )
                assert await _until(
                    lambda: case.store.active_session_id == successor.id
                    and activation in late_full_tasks
                    and console._console_sync_requested,
                    5,
                ), "Original activation did not leave full demand behind the suspended poll"
                assert not activation.done() and console._console_sync_in_progress
                assert case.runtime.view is console
                _assert_original_configuration_custody(case)
                lock.release()
                held_lock = False
                assert await _until(lambda: bool(replays), 5), (
                    "Pending demand reached neither original full-replay owner",
                    held_return,
                    replay_entries,
                    pending_state(),
                )
                assert await _until(
                    lambda: activation.done()
                    and any(
                        value is True
                        and owner == successor.id
                        and any(
                            task is issued for _worker, issued, _work, _route in replays
                        )
                        for task, value, owner in full_results
                    )
                    and not console._console_sync_in_progress
                    and not console._console_sync_requested
                    and not getattr(
                        console, "_console_control_bar_replay_whole_sync", False
                    ),
                    5,
                ), "Full demand did not settle on the current session owner"
                activation.result()
                same_call = [row for row in publications if row[0] == held["tabs"]]
                assert same_call and same_call[0][1] == case.session.id
                if held["preparing_display"]:
                    assert held_return["result"] is False
                    assert (
                        len(same_call) == 1
                    ), "Display caller continued after its owner changed"
                else:
                    assert any(
                        owner == successor.id and any(s is successor for s in sessions)
                        for _call, owner, sessions in same_call[1:]
                    ), "Original FULL tab caller failed its owner replay"
                assert all(
                    worker._task is task and worker._work is work
                    for worker, task, work, _route in replays
                )
                assert console.query_one(
                    f"#console-session-tab-{successor.id}", Button
                ).has_class("console-session-tab-active")
                assert console._console_visible_draft_session_id == successor.id
                assert case.composer.draft_text() == "Current owner draft"
                assert console._console_session_tabs_sync_calls == 0
                _assert_original_configuration_custody(case)
                record_property(
                    "late_full_held_preparing_display", held["preparing_display"]
                )
                record_property("late_full_original_replay_workers", len(replays))
                record_property(
                    "same_original_tab_caller_owners", [row[1] for row in same_call]
                )
        finally:
            # Preserve failure-path flags before fixture release or cancellation.
            import json

            try:
                record_property(
                    "late_full_replay_observation",
                    json.dumps(
                        {
                            "held_return": held_return,
                            "coalesced_entries": replay_entries,
                            "issued_routes": [
                                route for _worker, _task, _work, route in replays
                            ],
                            "full_results": [
                                {
                                    "result": result,
                                    "active": owner,
                                    "held_task": task is held.get("task"),
                                    "issued_task": any(
                                        task is issued for _w, issued, _c, _r in replays
                                    ),
                                }
                                for task, result, owner in full_results
                            ],
                            "current": pending_state(),
                        },
                        sort_keys=True,
                    ),
                )
            finally:
                # Only the acquisition made by this test may be released here.
                if held_lock:
                    lock.release()
                case.probe.release.set()
                if activation is not None:
                    if not activation.done():
                        activation.cancel()
                    await asyncio.wait_for(
                        asyncio.gather(activation, return_exceptions=True), 15
                    )


async def test_poll_terminal_transition_settles_while_background_turn_remains_active(
    monkeypatch, record_property
):
    """Use two received saved turns; no synthetic run state or attention service."""
    import asyncio
    import sqlite3
    import threading
    from textual.widgets import Button
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleMessageRole,
        FEEDBACK_ACTIVE_RUN_STATUSES,
    )
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Chat.conversation_local_marks_service import (
        ConversationLocalMarksService,
    )
    from tldw_chatbook.DB.ChaChaNotes_DB import (
        CharactersRAGDB,
        TransactionContextManager,
    )
    from tldw_chatbook.UI.Console_Modules.fleet import ConsoleFleetLifecycleController
    from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
    from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

    async with _received_console_case(
        monkeypatch, "terminal-background-poll", durable=True
    ) as case:
        await _qualify_warm_capture_sources(case)
        console, runtime, store, controller = (
            case.console,
            case.runtime,
            case.store,
            case.controller,
        )
        # Preserve the real saved direct-provider route; only its remote adapter is held.
        console.app_instance.app_config["console"]["agent_runtime"] = False
        entered, release, provider_retired = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        timed_out = []
        calls_lock = threading.Lock()

        def reply(**kwargs):
            with calls_lock:
                case.provider_calls.append(kwargs)
                index = len(case.provider_calls)
            if index == 1:
                entered.set()
                try:
                    if not release.wait(15):
                        timed_out.append(True)
                    return "Background polling reply completed."
                finally:
                    provider_retired.set()
            assert index == 2
            return "Viewed polling reply completed."

        monkeypatch.setattr("tldw_chatbook.Chat.Chat_Functions.chat_api_call", reply)
        marks = runtime._console_local_marks_service()
        assert type(marks) is ConversationLocalMarksService
        database = console.app_instance.chachanotes_db
        assert type(database) is CharactersRAGDB and marks.db is database
        assert str(database.db_path) != ":memory:"
        participant = database._maintenance_participant
        assert participant.repository() is database
        assert participant.path == database.db_path
        assert participant.owner_id == "db.chachanotes.primary"
        transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
        viewed = None
        received, acknowledgements, fulls, tails = {}, [], [], []
        received_tasks = {}
        terminal_core_passes = set()
        native_acks = {}
        full_code = ChatScreen._sync_native_console_chat_ui.__code__
        transcript_code = ChatScreen._sync_native_console_transcript.__code__
        ack_code = ConsoleRuntime.acknowledge_rendered_terminal_receipts.__code__
        poll_code = next(
            code
            for code in ChatScreen._start_console_transcript_sync_timer.__code__.co_consts
            if isinstance(code, CodeType) and code.co_name == "_poll_transcript"
        )

        def terminal_message():
            if viewed is None:
                return None
            # Do not call messages_for_session here: that original getter can
            # materialize stream buffers and schedule persistence. Observe only.
            return next(
                (
                    message
                    for message in store._messages_by_session.get(viewed.id, ())
                    if message.role is ConsoleMessageRole.ASSISTANT
                    and message.status == "complete"
                    and message.content == "Viewed polling reply completed."
                    and getattr(message.metadata, "terminal_receipt_id", "")
                ),
                None,
            )

        def background_live():
            record = received.get(case.session.id)
            return (
                entered.is_set()
                and not release.is_set()
                and not timed_out
                and record is not None
                and runtime._turn_custody.get(record.turn_id) is record
                and record.task is not None
                and not record.task.done()
                and controller.run_state_for(case.session.id).status
                in FEEDBACK_ACTIVE_RUN_STATUSES
            )

        def observe(kind, label, frame, value):
            if kind != sys.monitoring.events.PY_RETURN:
                return
            owner = frame.f_locals.get("self")
            if label == "received" and owner is runtime:
                record = frame.f_locals["record"]
                assert (
                    value == record.turn_id
                    and runtime._turn_custody.get(value) is record
                )
                assert (
                    record.store is store
                    and record.received_intent is frame.f_locals["intent"]
                )
                assert (
                    type(record.task) is asyncio.Task
                    and record.task.get_loop() is asyncio.get_running_loop()
                )
                assert record.received_intent.selection.agent_runtime_enabled is False
                received[record.session_id] = record
                received_tasks[record.session_id] = record.task
            elif (
                label == "core" and owner is console and terminal_message() is not None
            ):
                full = _transition_parent(frame, full_code, console)
                if full is not None and background_live():
                    terminal_core_passes.add((asyncio.current_task(), id(full)))
            elif label == "full" and owner is console:
                key = (asyncio.current_task(), id(frame))
                reconciled = key in terminal_core_passes
                terminal_core_passes.discard(key)
                if (
                    value is True
                    and terminal_message() is not None
                    and background_live()
                ):
                    fulls.append(reconciled)
            elif (
                label == "ack_transaction" and type(owner) is TransactionContextManager
            ):
                if owner.db is not database:
                    return
                ack_frame = _transition_parent(
                    frame,
                    ConversationLocalMarksService.acknowledge_console_unseen.__code__,
                    marks,
                )
                if ack_frame is None:
                    return
                assert (
                    frame.f_locals["exc_type"] is None
                    and owner.is_outermost_transaction
                )
                assert not owner.borrows_native_transaction
                connection, cursor = owner.conn, owner.cursor
                assert isinstance(connection, sqlite3.Connection) and isinstance(
                    cursor, sqlite3.Cursor
                )
                assert (
                    cursor.connection is connection
                    and vars(database._local).get("conn") is connection
                )
                assert not sqlite3.Connection.in_transaction.__get__(connection)
                with storage._lock:
                    lease = participant.connections.get(connection)
                    assert lease in storage._live_leases
                    assert lease.resource_thread is threading.current_thread()
                    operations = tuple(
                        operation
                        for operation in storage._operations
                        if operation.participant is participant
                        and operation.thread is threading.current_thread()
                    )
                    assert (
                        len(operations) == 1
                        and operations[0].lease in storage._live_leases
                    )
                pair = (
                    ack_frame.f_locals["conversation_id"],
                    ack_frame.f_locals["receipt_id"],
                )
                assert len(native_acks) < 8
                native_acks[pair] = operations[0]
            elif label == "ack" and owner is marks and value is True:
                message = terminal_message()
                if message is None:
                    return
                pair = (frame.f_locals["conversation_id"], frame.f_locals["receipt_id"])
                if pair != (
                    viewed.persisted_conversation_id,
                    message.metadata.terminal_receipt_id,
                ):
                    return
                runtime_frame = _transition_parent(frame, ack_code, runtime)
                display_frame = _transition_parent(frame, transcript_code, console)
                assert runtime_frame is not None and display_frame is not None
                assert runtime_frame.f_locals["view"] is console
                assert (
                    runtime_frame.f_locals["attachment_generation"]
                    == console._console_runtime_attachment_generation
                )
                assert pair in runtime_frame.f_locals["rendered"]
                assert pair in display_frame.f_locals["rendered_receipts"]
                # The original SQL committed, then its counted scope retired
                # before this service returned its exact successful deletion.
                assert pair in native_acks
                with storage._lock:
                    assert native_acks[pair] not in storage._operations
                    assert native_acks[pair].lease not in storage._live_leases
                assert background_live()
                assert console._console_transcript_sync_timer is not None
                acknowledgements.append(pair)
            elif label in {"invalidate", "stop", "survivor"}:
                expected = {
                    "invalidate": console._workspace,
                    "stop": console,
                    "survivor": console._fleet,
                }[label]
                parent = frame.f_back
                if (
                    owner is not expected
                    or parent is None
                    or parent.f_code is not poll_code
                    or parent.f_locals.get("self") is not console
                ):
                    return
                assert release.is_set() and provider_retired.is_set()
                assert (
                    not runtime.has_custodied_turns()
                    and controller.in_flight_run_count() == 0
                )
                if label == "invalidate":
                    assert console._workspace._console_persisted_rows_cache is None
                if label in {"stop", "survivor"}:
                    assert console._console_transcript_sync_timer is None
                tails.append(label)

        bindings = [
            ("received", ConsoleRuntime, "accept_received_intent"),
            ("timer_factory", ChatScreen, "_start_console_transcript_sync_timer"),
            ("core", ChatScreen, "_sync_console_chat_core_state"),
            ("full", ChatScreen, "_sync_native_console_chat_ui"),
            ("transcript", ChatScreen, "_sync_native_console_transcript"),
            ("rendered_ack", ConsoleRuntime, "acknowledge_rendered_terminal_receipts"),
            ("ack", ConversationLocalMarksService, "acknowledge_console_unseen"),
            ("ack_transaction", TransactionContextManager, "_exit_transaction"),
            (
                "invalidate",
                ConsoleWorkspaceController,
                "_invalidate_console_persisted_rows_cache",
            ),
            ("stop", ChatScreen, "_stop_console_transcript_sync_timer"),
            (
                "survivor",
                ConsoleFleetLifecycleController,
                "_maybe_start_console_fleet_survivor_tick",
            ),
        ]
        try:
            with _original_transition_events(bindings, observe):
                _send(case, "enter")
                original = await _held_received_record(case)
                _assert_original_configuration_custody(case)
                assert received[case.session.id] is original
                case.probe.release.set()
                assert await _until(entered.is_set, 10)
                assert await _until(case.probe.retired, 5)
                assert background_live()
                viewed = store.create_session(
                    title="Viewed terminal owner",
                    workspace_id=case.session.workspace_id,
                    settings=case.session.settings,
                    activate=False,
                )
                await asyncio.wait_for(
                    console._session._activate_native_console_session(viewed.id), 15
                )
                assert await _until(
                    lambda: console._console_visible_draft_session_id == viewed.id, 5
                )
                assert store.active_session_id == viewed.id and runtime.view is console
                case.composer.load_draft("Complete the viewed polling turn")
                case.composer.focus()
                _send(case, "enter")
                assert await _until(lambda: viewed.id in received, 10)
                viewed_record = received[viewed.id]
                assert viewed_record.session_id != original.session_id
                assert (
                    viewed_record.received_intent.inputs.draft
                    == "Complete the viewed polling turn"
                )
                assert viewed_record.store is original.store is store
                assert await _until(
                    lambda: terminal_message() is not None
                    and acknowledgements
                    and any(fulls)
                    and not runtime.has_custodied_turns(viewed.id),
                    15,
                ), "Viewed terminal state did not fully reconcile and acknowledge while background stayed live"
                received_tasks[viewed.id].result()
                message = terminal_message()
                pair = (
                    viewed.persisted_conversation_id,
                    message.metadata.terminal_receipt_id,
                )
                assert acknowledgements == [pair]
                assert message.id in transcript.mounted_message_content_ids()
                assert pair not in marks.list_console_unseen_marks()
                assert background_live() and not timed_out
                assert console._console_transcript_sync_timer is not None
                assert console._console_transcript_poll_needed()
                assert console.query_one(
                    f"#console-session-tab-{viewed.id}", Button
                ).has_class("console-session-tab-active")
                assert "●" in str(
                    console.query_one(
                        f"#console-session-tab-{case.session.id}", Button
                    ).label
                )
                release.set()
                assert await _until(lambda: not runtime.has_custodied_turns(), 15)
                received_tasks[case.session.id].result()
                assert await _until(
                    lambda: tails == ["invalidate", "stop", "survivor"], 5
                )
                assert console._console_transcript_sync_timer is None
                assert not timed_out and provider_retired.is_set()
                assert len(case.provider_calls) == 2
                record_property(
                    "viewed_terminal_exact_ack_while_background_live", acknowledgements
                )
                record_property("original_final_poll_tail", tails)
        finally:
            release.set()
            case.probe.release.set()
            tasks = tuple(received_tasks.values())
            if tasks:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), 15
                )
