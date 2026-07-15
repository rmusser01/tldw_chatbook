"""Worker-group hygiene for ChatScreen (TASK-228).

Root cause being guarded against: Textual's ``run_worker(..., exclusive=True)``
cancels every other worker in the same group, and calls without ``group=`` all
share the default group. The Console send worker and the UI-sync re-kick both
ran ungrouped-exclusive, so an overlapping sync silently cancelled in-flight
streams (probe-confirmed on textual 8.2.7; live symptoms: vision sends stalling
with no token/no row/no error, truncated assistant persists, and a permanently
stuck [streaming] suffix).
"""

import ast
import asyncio
from pathlib import Path

import pytest

CHAT_SCREEN_PATH = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook" / "UI" / "Screens" / "chat_screen.py"
)


def _call_name(node: ast.Call) -> str:
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")


def _exclusive_worker_sites(tree: ast.AST):
    """Yield (lineno, has_group) for every exclusive worker declaration.

    Covers both spellings — ``run_worker(..., exclusive=True)`` calls and
    ``@work(exclusive=True)`` decorators — and fails CLOSED: a non-literal
    ``exclusive=`` value (variable, expression, positional) is treated as
    exclusive, because a guard that skips what it can't prove is a guard
    that certifies an invariant the file doesn't hold.
    """
    for node in ast.walk(tree):
        calls: list[ast.Call] = []
        if isinstance(node, ast.Call) and _call_name(node) == "run_worker":
            calls.append(node)
        elif isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            calls.extend(
                dec for dec in node.decorator_list
                if isinstance(dec, ast.Call) and _call_name(dec) == "work"
            )
        for call in calls:
            keywords = {kw.arg: kw.value for kw in call.keywords if kw.arg}
            has_kwargs_spread = any(kw.arg is None for kw in call.keywords)
            exclusive = keywords.get("exclusive")
            if exclusive is None and not has_kwargs_spread:
                continue  # exclusive defaults to False
            if isinstance(exclusive, ast.Constant):
                if exclusive.value is not True:
                    continue
            # non-Constant exclusive= or a **kwargs spread: fail closed.
            yield call.lineno, "group" in keywords


def test_every_exclusive_worker_on_chat_screen_names_a_group():
    """An exclusive worker without group= joins the default group and cancels
    (or is cancelled by) every other ungrouped exclusive worker on the screen
    — including the Console send worker mid-stream. Never ship one."""
    tree = ast.parse(CHAT_SCREEN_PATH.read_text(encoding="utf-8"))
    ungrouped = [
        lineno for lineno, has_group in _exclusive_worker_sites(tree)
        if not has_group
    ]
    assert ungrouped == [], (
        "exclusive worker (run_worker or @work) without an explicit group= at "
        f"chat_screen.py lines {ungrouped} — these share Textual's default "
        "worker group and silently cancel each other (see TASK-228)."
    )


def test_console_run_and_sync_workers_use_disjoint_groups():
    """Pin the separation this fix exists for: the sync kicks must never share
    a group with the run workers. The names-a-group guard alone would pass if
    someone put a sync kick into group="console-run" — and the collision this
    branch fixed would silently return."""
    tree = ast.parse(CHAT_SCREEN_PATH.read_text(encoding="utf-8"))
    RUN_COROUTINES = {
        "_submit_console_native_draft",
        "_retry_console_message",
        "_regenerate_console_message",
        "_continue_console_message",
    }
    SYNC_COROUTINE = "_sync_native_console_chat_ui"
    run_groups: set[str] = set()
    sync_groups: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _call_name(node) == "run_worker"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        target = _call_name(first) if isinstance(first, ast.Call) else ""
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        group = keywords.get("group")
        group_name = group.value if isinstance(group, ast.Constant) else None
        if target in RUN_COROUTINES:
            run_groups.add(group_name)
        elif target == SYNC_COROUTINE:
            sync_groups.add(group_name)
    assert run_groups == {"console-run"}, run_groups
    assert sync_groups == {"console-sync"}, sync_groups


class TestTextualExclusiveGroupSemantics:
    """Pin the Textual semantics the fix relies on, on the installed version."""

    @staticmethod
    def _make_app(results: dict):
        from textual.app import App

        class Probe(App):
            async def long_run(self):
                try:
                    await asyncio.sleep(3)
                    results["run"] = "completed"
                except asyncio.CancelledError:
                    results["run"] = "cancelled"
                    raise

            async def quick_sync(self):
                results["sync"] = "ran"

        return Probe()

    @pytest.mark.asyncio
    async def test_ungrouped_exclusive_workers_cancel_each_other(self):
        """The footgun itself: default-group exclusive collision."""
        results: dict = {}
        app = self._make_app(results)
        async with app.run_test():
            app.run_worker(app.long_run(), exclusive=True)
            await asyncio.sleep(0.2)
            app.run_worker(app.quick_sync(), exclusive=True)
            await asyncio.sleep(0.2)
        assert results.get("run") == "cancelled"
        assert results.get("sync") == "ran"

    @pytest.mark.asyncio
    async def test_distinct_groups_do_not_cancel_each_other(self):
        """The fix premise: a console-sync kick must not touch console-run."""
        results: dict = {}
        app = self._make_app(results)
        async with app.run_test():
            run_worker = app.run_worker(
                app.long_run(), exclusive=True, group="console-run"
            )
            await asyncio.sleep(0.2)
            app.run_worker(app.quick_sync(), exclusive=True, group="console-sync")
            await asyncio.sleep(0.2)
            assert results.get("sync") == "ran"
            assert results.get("run") is None  # still running, NOT cancelled
            run_worker.cancel()
