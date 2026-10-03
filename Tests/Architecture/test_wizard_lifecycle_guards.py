"""TASK-34100.1 AC#3: wizard liveness and background-work rules, pinned by AST.

**Liveness.** In Textual 8.2.8 ``is_mounted`` is sticky. ``_is_mounted`` is
set True once and never cleared, so a removed step and a popped screen both
still report ``is_mounted is True`` (backlog/docs/lessons-textual.md, "is_mounted
never goes False"). Every liveness guard that read it was a no-op on the
teardown it was written for. The first-run wizard alone had 43 such reads on
2026-10-02, and they kept producing post-teardown crashes.
``is_attached``, which walks ``_parent`` to the DOM root, is the predicate
that goes False. This pins every module under ``UI/Wizards/`` at zero
``.is_mounted`` reads, comments excluded.

**Background work.** A Textual worker exits the whole app on an uncaught
error unless it is started with ``exit_on_error=False``, and that default is
how a failing Next quit the app during setup (TASK-33621.14). Every
first-run worker now starts through ``first_run_step_guard.run_wizard_worker``
or the ``wizard_work`` decorator. Both always pass ``exit_on_error=False`` and
report an escaped error on the wizard's pinned status strip. This pins that
no first-run module calls ``run_worker`` or ``@work`` directly.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_WIZARDS = _REPO / "tldw_chatbook" / "UI" / "Wizards"
_HELPER = _WIZARDS / "first_run_step_guard.py"


def _first_run_modules() -> list[Path]:
    modules = [_WIZARDS / "FirstRunSetupWizard.py"]
    modules += sorted(_WIZARDS.glob("first_run_*.py"))
    return [path for path in modules if path != _HELPER]


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_no_wizard_module_reads_is_mounted() -> None:
    offenders = [
        f"{path.relative_to(_REPO)}:{node.lineno}"
        for path in sorted(_WIZARDS.rglob("*.py"))
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.Attribute) and node.attr == "is_mounted"
    ]

    assert offenders == [], (
        "`is_mounted` never goes False in Textual 8, so it cannot tell a "
        "removed widget from a live one. Use `is_attached`:\n  "
        + "\n  ".join(offenders)
    )


def test_first_run_modules_start_no_worker_outside_the_shared_helper() -> None:
    offenders = []
    for path in _first_run_modules():
        for node in ast.walk(_tree(path)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run_worker"
            ):
                offenders.append(f"{path.name}:{node.lineno} run_worker(...)")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for deco in node.decorator_list:
                    target = deco.func if isinstance(deco, ast.Call) else deco
                    name = (
                        target.id
                        if isinstance(target, ast.Name)
                        else getattr(target, "attr", "")
                    )
                    if name == "work":
                        offenders.append(f"{path.name}:{node.lineno} @work")

    assert offenders == [], (
        "Start first-run background work with "
        "first_run_step_guard.run_wizard_worker / @wizard_work, which never "
        "exits the app and reports on the pinned strip:\n  "
        + "\n  ".join(offenders)
    )


def test_the_shared_helper_never_lets_a_worker_exit_the_app() -> None:
    calls = [
        node
        for node in ast.walk(_tree(_HELPER))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_worker"
    ]

    assert len(calls) == 1, "the helper must be the one run_worker call site"
    flags = {kw.arg: kw.value for kw in calls[0].keywords}
    assert isinstance(flags.get("exit_on_error"), ast.Constant)
    assert flags["exit_on_error"].value is False
