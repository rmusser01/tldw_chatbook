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


def _exclusive_run_worker_calls(tree: ast.AST):
    """Yield (lineno, has_group) for every exclusive=True run_worker call."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "run_worker":
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        exclusive = keywords.get("exclusive")
        if not (isinstance(exclusive, ast.Constant) and exclusive.value is True):
            continue
        yield node.lineno, "group" in keywords


def test_every_exclusive_worker_on_chat_screen_names_a_group():
    """An exclusive run_worker without group= joins the default group and
    cancels (or is cancelled by) every other ungrouped exclusive worker on the
    screen — including the Console send worker mid-stream. Never ship one."""
    tree = ast.parse(CHAT_SCREEN_PATH.read_text(encoding="utf-8"))
    ungrouped = [
        lineno for lineno, has_group in _exclusive_run_worker_calls(tree)
        if not has_group
    ]
    assert ungrouped == [], (
        "run_worker(exclusive=True) without an explicit group= at "
        f"chat_screen.py lines {ungrouped} — these share Textual's default "
        "worker group and silently cancel each other (see TASK-228)."
    )


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
