"""Count original native entries only while the direct coalesced callback runs."""

import inspect
import os
import sys
import threading
import time


def install(report, stage):
    from tldw_chatbook.Backup_Recovery import config_participants
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    callback = ChatScreen._run_coalesced_control_bar_sync.__code__
    operation = inspect.unwrap(config_participants.operation).__code__
    native_open = None
    if os.name == "nt":
        from tldw_chatbook.Utils import windows_files

        native_open = windows_files._native().open_handle.__func__.__code__
    main = threading.get_ident()
    starts, rows, main_entries = [], [], []
    entries = opens = 0
    monitor = sys.monitoring
    tool = next(i for i in range(1, 6) if monitor.get_tool(i) is None)
    monitor.use_tool_id(tool, "direct-control-bar-native-entry")

    def enter(code, offset):
        nonlocal entries, opens
        if threading.get_ident() != main:
            return
        if code is callback:
            starts.append((stage(), time.perf_counter(), entries, opens))
        elif code is operation:
            main_entries.append([stage(), time.perf_counter(), bool(starts)])
            entries += 1
        elif code is native_open:
            opens += 1

    def leave(code, offset, result):
        if code is not callback or threading.get_ident() != main or not starts:
            return
        label, started, before_entries, before_opens = starts.pop()
        rows.append(
            [
                label,
                started,
                (time.perf_counter() - started) * 1000,
                entries - before_entries,
                opens - before_opens,
            ]
        )

    monitor.register_callback(tool, monitor.events.PY_START, enter)
    monitor.register_callback(tool, monitor.events.PY_RETURN, leave)
    monitor.register_callback(tool, monitor.events.PY_UNWIND, leave)
    monitor.set_events(tool, monitor.events.PY_UNWIND)
    for code in (callback, operation, native_open):
        if code is not None:
            monitor.set_local_events(
                tool,
                code,
                monitor.events.PY_START
                | (monitor.events.PY_RETURN if code is callback else 0),
            )

    def finish():
        monitor.set_events(tool, 0)
        for code in (callback, operation, native_open):
            if code is not None:
                monitor.set_local_events(tool, code, 0)
        for event in (
            monitor.events.PY_START,
            monitor.events.PY_RETURN,
            monitor.events.PY_UNWIND,
        ):
            monitor.register_callback(tool, event, None)
        monitor.free_tool_id(tool)
        report["control_bar_spans"] = rows
        report["main_config_entry_events"] = main_entries

    return finish
