"""Diagnostic-only attachment spans; delegates every call without changing results."""

import functools
import inspect
import time
import asyncio
import sys
import threading
import types


def install(report, stage):
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.UI.Console_Modules.console_spend_projection import (
        ConsoleReadinessConfigProjection,
    )

    events = report["attachment_spans"] = []
    epoch = time.perf_counter()

    def record(name, edge, instance, result=None):
        current = stage()
        if not current.startswith("console-resume-"):
            return
        screen = getattr(instance, "screen", instance)
        from textual.worker import get_current_worker, NoActiveWorker

        try:
            group = get_current_worker().group
        except NoActiveWorker:
            group = None
        events.append(
            {
                "stage": current,
                "ms": (time.perf_counter() - epoch) * 1000,
                "name": name,
                "edge": edge,
                "result": result if isinstance(result, bool) else None,
                "sync": getattr(screen, "_console_sync_in_progress", None),
                "replay": getattr(
                    screen, "_console_control_bar_replay_whole_sync", None
                ),
                "attached": getattr(screen, "_console_attach_reconciled", None),
                "worker": group,
            }
        )

    def wrap(cls, name):
        original = getattr(cls, name)
        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def observed(self, *args, **kwargs):
                record(name, "begin", self)
                result = await original(self, *args, **kwargs)
                record(name, "end", self, result)
                return result
        else:

            @functools.wraps(original)
            def observed(self, *args, **kwargs):
                record(name, "begin", self)
                result = original(self, *args, **kwargs)
                record(name, "end", self, result)
                return result

        setattr(cls, name, observed)

    for name in (
        "on_screen_resume",
        "_start_console_attach_reconciliation",
        "_reconcile_console_after_attach",
        "_sync_native_console_chat_ui",
        "_sync_console_native_session_tabs",
        "_sync_console_rail_and_controls",
        "_sync_native_console_transcript",
        "_current_console_rail_state",
        "_request_console_control_bar_sync",
        "_run_coalesced_control_bar_sync",
        "_handle_console_composer_draft_edit",
        "_sync_console_workbench_actions_from_draft",
        "_sync_console_settings_summary",
        "_sync_console_cost_chip",
    ):
        wrap(ChatScreen, name)
    for name in ("warm", "_refresh"):
        wrap(ConsoleReadinessConfigProjection, name)

    # Observe queue delay without replacing original callbacks or retaining frames.
    from tldw_chatbook.Chat.console_preparation_reads import run_preparation_read

    outer = run_preparation_read.__code__
    inner = next(
        c
        for c in outer.co_consts
        if isinstance(c, types.CodeType) and c.co_name == "invoke"
    )
    monitor = sys.monitoring
    tool = next(i for i in range(1, 6) if monitor.get_tool(i) is None)
    monitor.use_tool_id(tool, "attachment-native-queue")
    native = report["preparation_spans"] = []

    def boundary(code, offset, *args):
        current = stage()
        if not current.startswith("console-resume-"):
            return
        frame = sys._getframe(1)
        local = frame.f_locals
        if code is outer:
            callback = local["callback"]
            task = asyncio.current_task()
            edge = "requested"
        else:
            read = local["read"]
            callback = read.callback
            task = read.task
            edge = "finished" if args else "entered"
        name = getattr(callback, "__qualname__", type(callback).__qualname__)
        native.append(
            {
                "stage": current,
                "ms": (time.perf_counter() - epoch) * 1000,
                "name": name,
                "task": id(task),
                "edge": edge,
                "thread": threading.get_ident(),
            }
        )
        del frame, local, callback, task

    monitor.register_callback(tool, monitor.events.PY_START, boundary)
    monitor.register_callback(tool, monitor.events.PY_RETURN, boundary)
    monitor.set_local_events(tool, outer, monitor.events.PY_START)
    monitor.set_local_events(
        tool, inner, monitor.events.PY_START | monitor.events.PY_RETURN
    )
