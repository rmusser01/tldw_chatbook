"""Regression guard for the Textual MRO double-dispatch trap (TASK-31418).

Textual's ``MessagePump._get_dispatch_methods`` walks the MRO and calls EVERY
distinct implementation of a lifecycle handler for one event. A subclass that
both overrides ``on_unmount`` AND calls ``super().on_unmount()`` therefore runs
the base body twice. The repo convention is: a subclass handler for an
MRO-dispatched lifecycle event does NOT call ``super().on_*()``.

Two guards:
- a runtime count that pins the base ``on_unmount`` firing exactly once under
  the no-super convention (it would be two with an explicit ``super()`` call);
- an AST scan that fails if any screen/modal ``on_unmount`` re-introduces a
  ``super().on_unmount()`` call, so a regression fails loudly instead of
  hiding until a non-idempotent base teardown makes it a double-teardown bug.

See ``backlog/docs/lessons-textual.md`` for the mechanism and the probe.
"""

from __future__ import annotations

import ast
import pathlib

from textual.app import App
from textual.screen import Screen

_PKG_ROOT = pathlib.Path(__file__).resolve().parents[2] / "tldw_chatbook"


async def test_base_on_unmount_fires_once_under_mro():
    """The base ``on_unmount`` runs once per unmount when the subclass omits super()."""
    counts = {"base": 0, "child": 0}

    class _Base(Screen):
        def on_unmount(self) -> None:
            counts["base"] += 1

    class _Child(_Base):
        def on_unmount(self) -> None:
            # Convention under test: NO super().on_unmount().
            counts["child"] += 1

    app: App = App()
    async with app.run_test() as pilot:
        await app.push_screen(_Child())
        await pilot.pause()
        app.pop_screen()
        await pilot.pause()

    assert counts["child"] == 1
    # Exactly one base fire from Textual's MRO walk. It would be 2 if
    # _Child.on_unmount called super().on_unmount() (the TASK-31418 bug).
    assert counts["base"] == 1


def _reintroduced_super_on_unmount() -> list[str]:
    offenders: list[str] = []
    for path in _PKG_ROOT.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name != "on_unmount":
                continue
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "on_unmount"
                    and isinstance(sub.func.value, ast.Call)
                    and isinstance(sub.func.value.func, ast.Name)
                    and sub.func.value.func.id == "super"
                ):
                    offenders.append(f"{path.relative_to(_PKG_ROOT.parent)}:{sub.lineno}")
    return offenders


def test_no_screen_reintroduces_super_on_unmount():
    """No on_unmount handler may call super().on_unmount() (double-fires the base)."""
    offenders = _reintroduced_super_on_unmount()
    assert offenders == [], (
        "super().on_unmount() re-introduced under Textual MRO dispatch — this "
        "double-fires the base teardown (TASK-31418). Remove the explicit call "
        "(the dispatcher already invokes the base), or, if the base target is a "
        "plain method reachable only via super(), rename it so it is not a "
        "dispatched handler. Offending sites: " + ", ".join(offenders)
    )
