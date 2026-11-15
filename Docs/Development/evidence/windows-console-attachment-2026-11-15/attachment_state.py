"""Bounded, content-free snapshots of the original attachment wait owners."""

import asyncio
import inspect
import time


def capture(report, stage, edge, screen):
    """Record identities and suspended code locations without invoking owners."""
    flags = (
        "_console_attach_reconciled",
        "_console_attach_sync_complete",
        "_console_attach_runtime_reconciled",
        "_console_attach_view_started",
        "_console_attach_reconcile_running",
        "_console_resume_after_reconcile",
        "_console_attach_resume_in_progress",
        "_console_sync_in_progress",
        "_console_sync_requested",
        "_console_control_bar_replay_whole_sync",
        "_console_sync_maintenance_paused",
        "_console_attach_reconcile_retry_exhausted",
    )
    state = vars(screen) if screen is not None else {}
    tasks = []
    for task in asyncio.all_tasks():
        current, chain, seen = task.get_coro(), [], set()
        while current is not None and id(current) not in seen and len(chain) < 24:
            seen.add(id(current))
            if inspect.iscoroutine(current):
                frame, code, waiting = (
                    current.cr_frame,
                    current.cr_code,
                    current.cr_await,
                )
            elif inspect.isgenerator(current):
                frame, code, waiting = (
                    current.gi_frame,
                    current.gi_code,
                    current.gi_yieldfrom,
                )
            else:
                chain.append(
                    {"awaitable_type": type(current).__qualname__, "id": id(current)}
                )
                break
            chain.append(
                {
                    "function": code.co_qualname,
                    "id": id(current),
                    "line": None if frame is None else frame.f_lineno,
                }
            )
            current = waiting
        if any("console" in item.get("function", "").lower() for item in chain):
            tasks.append(
                {
                    "task_id": id(task),
                    "done": task.done(),
                    "cancelled": task.cancelled(),
                    "await_chain": chain,
                }
            )
    report.setdefault("attachment_state", []).append(
        {
            "stage": stage,
            "edge": edge,
            "wall_seconds": time.perf_counter(),
            "screen_id": None if screen is None else id(screen),
            "phase_flags": {
                name: state.get(name) for name in flags if type(state.get(name)) is bool
            },
            "visit_generation": state.get("_console_attach_visit_generation"),
            "attachment_generation": state.get(
                "_console_runtime_attachment_generation"
            ),
            "tasks": tasks,
        }
    )
