"""Temporary observation of original tab startup; no callable replacement."""

import asyncio
import inspect
import json
import sys
import time

import pytest


def _chain(coroutine):
    rows = []
    seen = set()
    while coroutine is not None and id(coroutine) not in seen and len(rows) < 7:
        seen.add(id(coroutine))
        code = getattr(coroutine, "cr_code", None) or getattr(coroutine, "gi_code", None)
        frame = getattr(coroutine, "cr_frame", None) or getattr(coroutine, "gi_frame", None)
        rows.append({
            "kind": type(coroutine).__name__,
            "function": getattr(code, "co_qualname", None),
            "line": getattr(frame, "f_lineno", None),
        })
        coroutine = getattr(coroutine, "cr_await", None) or getattr(coroutine, "gi_yieldfrom", None)
    return rows


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if item.path.name != "test_console_tab_publication_read_barrier.py":
        yield
        return

    from Tests.UI import test_console_session_tab_close as helpers
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    targets = (
        (ConsoleSessionController, "_report_session_close_failure"),
        (ConsoleChatController, "finalize_session_close"),
        (ChatScreen, "_sync_native_console_chat_ui"),
        (helpers, "_show_tabs"),
        (helpers, "_await_tabs"),
    )
    bindings = tuple(
        (owner, name, function, function.__code__, function.__defaults__, function.__kwdefaults__)
        for owner, name in targets
        for function in (inspect.getattr_static(owner, name),)
    )
    names = {code: name for _, name, _, code, _, _ in bindings}
    monitor = sys.monitoring
    slot = next(slot for slot in range(6) if monitor.get_tool(slot) is None)
    monitor.use_tool_id(slot, "tab-startup-readonly-probe")
    started = time.monotonic()
    screen = None
    count = 0
    full_count = 0
    full_frames = {}
    show_context = {}

    def current():
        return all(
            inspect.getattr_static(owner, name) is function
            and function.__code__ is code
            and function.__defaults__ is defaults
            and function.__kwdefaults__ is kwdefaults
            for owner, name, function, code, defaults, kwdefaults in bindings
        )

    def snapshot(console):
        namespace = vars(console)
        store = namespace.get("_console_chat_store")
        facts = {
            "flags": {
                name: namespace.get(name)
                for name in (
                    "_console_sync_in_progress", "_console_sync_requested",
                    "_console_control_bar_replay_whole_sync",
                    "_console_sync_maintenance_paused", "_console_attach_reconciled",
                    "_console_attach_view_started", "_console_attach_sync_complete",
                    "_console_attach_reconcile_running", "_console_session_tabs_sync_calls",
                    "_console_warm_refusals", "_closing", "_closed",
                )
                if type(namespace.get(name)) in (bool, int, type(None))
            },
            "store_identity": id(store),
            "active_session_id": None if store is None else store.active_session_id,
            "store_session_ids": [] if store is None else [session.id for session in store.sessions()][:20],
            "dom_session_ids": sorted(helpers._open_tab_ids(console)),
        }
        projection = namespace.get("_console_readiness_config_projection")
        if projection is not None:
            facts["readiness"] = {
                "pending": projection.pending,
                "key_session": None if projection.key is None else projection.key[5],
                "has_value": projection.value is not None,
            }
        workers = []
        for worker in console.workers:
            if worker.node is not console or worker.is_finished:
                continue
            if len(workers) >= 10:
                break
            task = worker._task
            workers.append({
                "group": worker.group,
                "state": worker.state.name,
                "task_done": None if task is None else task.done(),
                "await_chain": [] if task is None else _chain(task.get_coro()),
            })
        facts["workers"] = workers
        # Include the current exact caller chain; no frame locals are emitted.
        task = asyncio.current_task()
        facts["caller_chain"] = [] if task is None else _chain(task.get_coro())
        return facts

    def emit(event, frame, value=None):
        nonlocal screen, count, full_count
        code = frame.f_code
        name = names[code]
        if name == "_report_session_close_failure" and event == "start":
            error = frame.f_locals["exc"]
            chain, trace = [], error.__traceback__
            while trace is not None:
                chain.append({"function": trace.tb_frame.f_code.co_qualname, "file": trace.tb_frame.f_code.co_filename, "line": trace.tb_lineno})
                trace = trace.tb_next
            print("TAB_CLOSE_FAILURE " + json.dumps({"type": type(error).__name__, "trace": chain, "bindings_unchanged": current()}), file=sys.stderr, flush=True)
            return
        if name == "finalize_session_close" and event == "start":
            controller = frame.f_locals["self"]
            ticket = frame.f_locals["ticket"]
            from tldw_chatbook.Chat.console_preparation_reads import preparation_reads_for
            reads = preparation_reads_for(controller._preparation_reads, ticket.session_id)
            print("TAB_FINALIZE_READS " + json.dumps({"count": len(reads), "session_id": ticket.session_id, "reads": [{"session_id": read.session_id, "creator": type(read.creator).__name__, "callback": getattr(read.callback, "__qualname__", type(read.callback).__name__), "task_done": read.task.done(), "producer_done": read._producer is not None and read._producer.done(), "retired": read.retired.done()} for read in reads]}), file=sys.stderr, flush=True)
            return
        if name == "_show_tabs" and event == "start":
            screen = frame.f_locals["console"]
            show_context[id(frame)] = sorted(frame.f_locals["expected"])
        console = frame.f_locals.get("self") if name == "_sync_native_console_chat_ui" else frame.f_locals.get("console")
        if console is not screen or screen is None or count >= 64:
            return
        if name == "_sync_native_console_chat_ui":
            if event == "start":
                full_count += 1
                if full_count > 18:
                    return
                full_frames[id(frame)] = full_count
            if id(frame) not in full_frames:
                return
        count += 1
        row = {
            "event": event, "function": name, "line": frame.f_lineno,
            "elapsed": round(time.monotonic() - started, 4),
            "bindings_unchanged": current(),
            "full_invocation": full_frames.get(id(frame)),
            "expected": sorted(frame.f_locals["expected"]) if "expected" in frame.f_locals else show_context.get(id(frame)),
        }
        if name == "_sync_native_console_chat_ui":
            row["entry_decisions"] = {
                key: frame.f_locals[key]
                for key in ("held", "_preparing_poll", "prepared")
                if key in frame.f_locals and type(frame.f_locals[key]) is bool
            }
        if event == "return":
            row["return"] = value if type(value) in (bool, type(None)) else type(value).__name__
        elif event == "unwind":
            row["exception_type"] = type(value).__name__
        row.update(snapshot(console))
        print("TAB_STARTUP_PROBE " + json.dumps(row, sort_keys=True), file=sys.stderr, flush=True)
        if event in {"return", "unwind"}:
            full_frames.pop(id(frame), None)
            show_context.pop(id(frame), None)

    def start(code, offset):
        emit("start", sys._getframe(1))

    def returned(code, offset, value):
        emit("return", sys._getframe(1), value)

    def unwind(code, offset, error):
        emit("unwind", sys._getframe(1), error)

    callbacks = {
        monitor.events.PY_START: start,
        monitor.events.PY_RETURN: returned,
    }
    mask = 0
    for event, callback in callbacks.items():
        monitor.register_callback(slot, event, callback)
        mask |= event
    for code in names:
        monitor.set_local_events(slot, code, mask)
    try:
        yield
    finally:
        for code in names:
            monitor.set_local_events(slot, code, 0)
        for event in callbacks:
            monitor.register_callback(slot, event, None)
        monitor.free_tool_id(slot)
        print("TAB_STARTUP_PROBE " + json.dumps({
            "event": "observer_removed", "rows": count,
            "final_snapshot": "not read after application exit",
            "bindings_unchanged": current(),
        }, sort_keys=True), file=sys.stderr, flush=True)
