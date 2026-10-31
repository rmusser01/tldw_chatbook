"""Separate scalar native-read wall/CPU spans from uninstrumented comparisons."""

from collections import defaultdict
import dis
import inspect
import sys
import threading
import time


def install(report, stage):
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
    from tldw_chatbook.Backup_Recovery import config_participants as participants
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as display
    from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
    from tldw_chatbook.UI.Screens.home_screen import HomeScreen

    targets = {}
    for function, name in (
        (display.ConsoleReadinessConfigProjection.for_screen.__func__, "read_current"),
        (
            inspect.unwrap(
                ConsoleWorkspaceController.prepare_console_workspace_reconciliation
            ),
            "read",
        ),
    ):
        code = next(
            c
            for c in function.__code__.co_consts
            if inspect.iscode(c) and c.co_name == name
        )
        targets[code] = code.co_qualname
    functions = (
        storage._acquire_storage,
        storage._scope,
        bootstrap.startup_permission,
        inspect.unwrap(participants.operation),
        display._checked_display_status,
        display._run_checked_display_sync,
    )
    names = {fn.__code__: fn.__qualname__ for fn in functions} | targets
    points = {
        fn.__code__: fn.__qualname__
        for fn in (
            HomeScreen._home_content_seam_call,
            HomeScreen.on_screen_suspend,
            HomeScreen.on_screen_resume,
        )
    }
    worker_code = next(
        code
        for code in HomeScreen._home_content_seam_call.__code__.co_consts
        if inspect.iscode(code) and code.co_name == "invoke_seam_in_worker"
    )
    points[worker_code] = worker_code.co_qualname
    lines = {code: list(dis.findlinestarts(code)) for code in names}
    active = defaultdict(int)
    starts = defaultdict(list)
    rows = []
    edges = []
    monitor = sys.monitoring
    tool = next(i for i in range(1, 6) if monitor.get_tool(i) is None)
    monitor.use_tool_id(tool, "native-preparation-wall-cpu")

    def enter(code, offset):
        if code in points:
            edges.append([stage(), points[code], "start", time.perf_counter()])
            return
        thread = threading.get_ident()
        if code in targets:
            active[thread] += 1
        if active[thread] or code.co_name in (
            "_checked_display_status",
            "_run_checked_display_sync",
        ):
            starts[thread, code].append(
                (time.perf_counter(), time.thread_time(), stage())
            )

    def leave(code, offset, result):
        if code not in names and code not in points:
            return  # PY_UNWIND is global-only in Python 3.12.
        if code in points:
            edges.append([stage(), points[code], "return", time.perf_counter()])
            return
        thread = threading.get_ident()
        pending = starts[thread, code]
        if pending:
            wall, cpu, label = pending.pop()
            line = next(
                (
                    line
                    for instruction, line in reversed(lines[code])
                    if instruction <= offset
                ),
                None,
            )
            rows.append(
                [
                    label,
                    names[code],
                    thread,
                    wall,
                    (time.perf_counter() - wall) * 1000,
                    (time.thread_time() - cpu) * 1000,
                    line,
                    result
                    if type(result) in (bool, type(None))
                    else type(result).__name__,
                ]
            )
        if code in targets:
            active[thread] -= 1

    monitor.register_callback(tool, monitor.events.PY_START, enter)
    monitor.register_callback(tool, monitor.events.PY_RESUME, enter)
    monitor.register_callback(tool, monitor.events.PY_RETURN, leave)
    monitor.register_callback(tool, monitor.events.PY_YIELD, leave)
    monitor.register_callback(tool, monitor.events.PY_UNWIND, leave)
    monitor.set_events(tool, monitor.events.PY_UNWIND)
    for code in names.keys() | points.keys():
        if code in points:
            monitor.set_local_events(
                tool,
                code,
                monitor.events.PY_START | monitor.events.PY_RETURN,
            )
            continue
        monitor.set_local_events(
            tool,
            code,
            monitor.events.PY_START
            | monitor.events.PY_RESUME
            | monitor.events.PY_RETURN
            | monitor.events.PY_YIELD,
        )

    def finish():
        monitor.set_events(tool, 0)
        for code in names.keys() | points.keys():
            monitor.set_local_events(tool, code, 0)
        for event in (
            monitor.events.PY_START,
            monitor.events.PY_RESUME,
            monitor.events.PY_RETURN,
            monitor.events.PY_YIELD,
            monitor.events.PY_UNWIND,
        ):
            monitor.register_callback(tool, event, None)
        monitor.free_tool_id(tool)
        report["native_wait_spans"] = rows
        report["native_wait_home_edges"] = edges

    return finish
