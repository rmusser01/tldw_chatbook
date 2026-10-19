"""Identity-preserving synchronous return spans, separate from plain timings."""

from collections import defaultdict
import asyncio
import inspect
import sys
import threading
import time


def install(report, stage):
    from textual.screen import Screen
    from textual.css.stylesheet import Stylesheet
    from tldw_chatbook import config
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.UI.Console_Modules import config_sync
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        ConsoleReadinessConfigProjection,
    )
    from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
    from tldw_chatbook.Chat.console_preparation_reads import run_preparation_read
    from textual import _win_sleep

    functions = [
        Screen._on_screen_resume,
        Stylesheet.update_nodes,
        config.load_settings,
        config._config_file_posture,
        ChatScreen._build_console_provider_selection,
        ChatScreen._sync_console_cost_chip,
        ChatScreen._build_console_inspector_state,
        ChatScreen.on_screen_resume,
        config_sync.run_console_presentation_sync,
        config_sync.run_console_config_sync,
    ]
    names = {fn.__code__: fn.__qualname__ for fn in functions}
    asynchronous = [
        ChatScreen._sync_native_console_chat_ui,
        ChatScreen._sync_console_native_session_tabs,
        ChatScreen._sync_native_console_transcript,
        ChatScreen._reconcile_console_after_attach,
        config_sync.prepare_console_presentation_sync,
        ConsoleReadinessConfigProjection._refresh,
        ConsoleReadinessConfigProjection.warm,
        inspect.unwrap(
            ConsoleWorkspaceController.prepare_console_workspace_reconciliation
        ),
        run_preparation_read,
    ]
    async_names = {fn.__code__: fn.__qualname__ for fn in asynchronous}
    worker_names = {}
    for function, nested in (
        (run_preparation_read, "invoke"),
        (ConsoleReadinessConfigProjection.for_screen.__func__, "read_current"),
        (
            inspect.unwrap(
                ConsoleWorkspaceController.prepare_console_workspace_reconciliation
            ),
            "read",
        ),
        (_win_sleep.sleep, "wait_inner"),
    ):
        for code in function.__code__.co_consts:
            if inspect.iscode(code) and code.co_name == nested:
                worker_names[code] = code.co_qualname
    timeline = []
    starts = defaultdict(list)
    rows = []
    main = threading.get_ident()
    monitor = sys.monitoring
    tool = next(i for i in range(1, 6) if monitor.get_tool(i) is None)
    monitor.use_tool_id(tool, "console-return-spans")

    def enter(code, offset):
        if code in worker_names:
            timeline.append(
                [
                    stage(),
                    worker_names[code],
                    "start",
                    time.perf_counter(),
                    threading.get_ident(),
                ]
            )
            return
        if threading.get_ident() == main:
            if code in async_names:
                timeline.append(
                    [
                        stage(),
                        async_names[code],
                        "start",
                        time.perf_counter(),
                        id(asyncio.current_task()),
                    ]
                )
            else:
                starts[code].append((time.perf_counter(), stage()))

    def leave(code, offset, result):
        if code in worker_names:
            timeline.append(
                [
                    stage(),
                    worker_names[code],
                    "return",
                    time.perf_counter(),
                    threading.get_ident(),
                ]
            )
            return
        if threading.get_ident() == main:
            if code in async_names:
                timeline.append(
                    [
                        stage(),
                        async_names[code],
                        "return",
                        time.perf_counter(),
                        id(asyncio.current_task()),
                    ]
                )
            elif starts[code]:
                began, label = starts[code].pop()
                rows.append([label, names[code], (time.perf_counter() - began) * 1000])

    def yielded(code, offset, result):
        if threading.get_ident() == main:
            timeline.append(
                [
                    stage(),
                    async_names[code],
                    "yield",
                    time.perf_counter(),
                    id(asyncio.current_task()),
                ]
            )

    def resumed(code, offset):
        if threading.get_ident() == main:
            timeline.append(
                [
                    stage(),
                    async_names[code],
                    "resume",
                    time.perf_counter(),
                    id(asyncio.current_task()),
                ]
            )

    monitor.register_callback(tool, monitor.events.PY_START, enter)
    monitor.register_callback(tool, monitor.events.PY_RETURN, leave)
    monitor.register_callback(tool, monitor.events.PY_YIELD, yielded)
    monitor.register_callback(tool, monitor.events.PY_RESUME, resumed)
    for code in names:
        monitor.set_local_events(
            tool, code, monitor.events.PY_START | monitor.events.PY_RETURN
        )
    for code in async_names:
        monitor.set_local_events(
            tool,
            code,
            monitor.events.PY_START
            | monitor.events.PY_RETURN
            | monitor.events.PY_YIELD
            | monitor.events.PY_RESUME,
        )
    for code in worker_names:
        monitor.set_local_events(
            tool, code, monitor.events.PY_START | monitor.events.PY_RETURN
        )

    def finish():
        for code in names.keys() | async_names.keys() | worker_names.keys():
            monitor.set_local_events(tool, code, 0)
        monitor.register_callback(tool, monitor.events.PY_START, None)
        monitor.register_callback(tool, monitor.events.PY_RETURN, None)
        monitor.register_callback(tool, monitor.events.PY_YIELD, None)
        monitor.register_callback(tool, monitor.events.PY_RESUME, None)
        monitor.free_tool_id(tool)
        report["return_sync_spans"] = rows
        report["return_async_timeline"] = timeline

    return finish
