"""Regression guard for the Textual MRO double-dispatch trap, mount side (TASK-31822).

Mount-side twin of ``Tests/UI/test_on_unmount_mro_convention.py`` (TASK-31418).
Textual's ``MessagePump._get_dispatch_methods`` walks the MRO and calls EVERY
distinct implementation of a lifecycle handler for one event. A subclass that
both overrides ``on_mount`` AND calls ``super().on_mount()`` therefore runs
the base body twice. The repo convention is: a subclass handler for an
MRO-dispatched lifecycle event does NOT call ``super().on_*()``.

Two guards:
- a runtime count that pins the base ``on_mount`` firing exactly once under
  the no-super convention (it would be two with an explicit ``super()`` call);
- an AST scan that fails if any screen/modal/widget ``on_mount`` re-introduces
  a ``super().on_mount()`` call, so a regression fails loudly instead of
  hiding until a non-idempotent base mount handler makes it a double-fire bug.

A ``super().on_mount()`` is only ever safe when the base target is NOT itself
a separately MRO-dispatched handler (e.g. a plain method reachable only via
the explicit call). ``_ALLOWED_OFFENDERS`` below is the escape hatch for that
case; it is empty because the TASK-31822 audit found none -- every live
``super().on_mount()`` in the repo resolved to a base whose ``on_mount`` is
defined in its own class ``__dict__`` (SafeModalDismissMixin or
LibraryAdaptiveReaderShell), so all 19 were redundant and removed. The one
genuine run-once-and-callable need in the repo (``BaseWizard``) uses the
plain-method pattern (``_post_mount_hook()``) instead of ``super()``, so it
never appears in this scan.

See ``backlog/docs/lessons-textual.md`` for the mechanism and the probe.
"""

from __future__ import annotations

import ast
import pathlib

from textual.app import App
from textual.screen import Screen

_PKG_ROOT = pathlib.Path(__file__).resolve().parents[2] / "tldw_chatbook"

# Escape hatch for a genuinely load-bearing super().on_mount() call -- a base
# target that is NOT itself a separately MRO-dispatched handler. Format:
# "relative/path.py:lineno" -> reason. Empty: the TASK-31822 audit found no
# such site (see module docstring).
_ALLOWED_OFFENDERS: dict[str, str] = {}


async def test_base_on_mount_fires_once_under_mro():
    """The base ``on_mount`` runs once per mount when the subclass omits super()."""
    counts = {"base": 0, "child": 0}

    class _Base(Screen):
        def on_mount(self) -> None:
            counts["base"] += 1

    class _Child(_Base):
        def on_mount(self) -> None:
            # Convention under test: NO super().on_mount().
            counts["child"] += 1

    app: App = App()
    async with app.run_test() as pilot:
        await app.push_screen(_Child())
        await pilot.pause()

    assert counts["child"] == 1
    # Exactly one base fire from Textual's MRO walk. It would be 2 if
    # _Child.on_mount called super().on_mount() (the TASK-31822 bug).
    assert counts["base"] == 1


def _reintroduced_super_on_mount() -> list[str]:
    offenders: list[str] = []
    for path in _PKG_ROOT.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name != "on_mount":
                continue
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "on_mount"
                    and isinstance(sub.func.value, ast.Call)
                    and isinstance(sub.func.value.func, ast.Name)
                    and sub.func.value.func.id == "super"
                ):
                    site = f"{path.relative_to(_PKG_ROOT.parent)}:{sub.lineno}"
                    if site not in _ALLOWED_OFFENDERS:
                        offenders.append(site)
    return offenders


def test_no_screen_reintroduces_super_on_mount():
    """No on_mount handler may call super().on_mount() (double-fires the base)."""
    offenders = _reintroduced_super_on_mount()
    assert offenders == [], (
        "super().on_mount() re-introduced under Textual MRO dispatch — this "
        "double-fires the base mount handler (TASK-31822). Remove the explicit "
        "call (the dispatcher already invokes the base), or, if the base "
        "target is a plain method reachable only via super(), rename it so it "
        "is not a dispatched handler (see BaseWizard._post_mount_hook for the "
        "pattern), or add a justified entry to _ALLOWED_OFFENDERS. Offending "
        "sites: " + ", ".join(offenders)
    )
