# test_console_moved_seam_guard.py
# Description: AST guard -- no test calls a Console seam that decomposition
# moved off `ChatScreen` (task-14920 AC#4).
#
# The shape this file exists to stop: the Console decomposition programme moves
# a private method from `ChatScreen` onto one of the `UI/Console_Modules`
# controllers. Tests that call it directly (this repo's long-standing
# direct-call convention for Console behaviour) then raise
# `AttributeError: 'ChatScreen' object has no attribute '_x'`. That is loud
# when the test runs -- but two such tests
# (`test_console_save_as_savers_confirm_at_success_severity`,
# `test_console_settings_save_fires_success_toast`) were WRITTEN after their
# seams had already moved, merged red, and sat unnoticed for four days inside a
# file whose pass count nobody had ever read. One of them was masking a real
# product defect (the Save-as-Chatbook confirmation was still firing at
# "information" severity while FB-07 had moved the other three to "success").
#
# This is checked by AST + the live classes rather than by grep because the
# question "does `ChatScreen` still have this attribute?" is only answerable
# against the imported class, and because grep cannot tell
# `controller._foo()` (correct) from `console._foo()` (broken).

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from pathlib import Path

import pytest

from tldw_chatbook.UI.Console_Modules.left_rail import ConsoleLeftRail
from tldw_chatbook.UI.Console_Modules.message import ConsoleMessageController
from tldw_chatbook.UI.Console_Modules.prompts import ConsolePromptsController
from tldw_chatbook.UI.Console_Modules.retrieval import ConsoleRetrievalController
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
from tldw_chatbook.UI.Console_Modules.transcript import ConsoleTranscriptRegion
from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

TESTS_DIR = Path(__file__).resolve().parent.parent

CONSOLE_CONTROLLERS = (
    ConsoleSessionController,
    ConsoleMessageController,
    ConsoleWorkspaceController,
    ConsoleTranscriptRegion,
    ConsolePromptsController,
    ConsoleRetrievalController,
    ConsoleLeftRail,
)

CONTROLLER_CLASS_NAMES = frozenset(c.__name__ for c in CONSOLE_CONTROLLERS)

# Identifiers that are plainly bound to a controller (or to the module under
# test), not to a `ChatScreen`. Calling a moved seam on one of these is the
# CORRECT spelling, so it must not be reported.
CONTROLLER_VARIABLE_NAMES = frozenset(
    {
        "controller",
        "session_controller",
        "message_controller",
        "workspace_controller",
        "transcript_controller",
        "prompts_controller",
        "retrieval_controller",
        "left_rail",
        "rail",
        "region",
    }
)


@dataclass(frozen=True)
class Violation:
    """One call to a seam that no longer exists on `ChatScreen`."""

    path: str
    line: int
    call: str
    owner: str

    def __str__(self) -> str:  # pragma: no cover - failure formatting only
        return (
            f"{self.path}:{self.line}  {self.call}  -- moved to {self.owner}; "
            f"call it through the controller (e.g. `screen._session.<name>()`)"
        )


def moved_console_seams() -> dict[str, str]:
    """Private callables that live on a Console controller and NOT on ChatScreen.

    Returns:
        Mapping of method name to the controller class that owns it. A name
        that `ChatScreen` still exposes (directly or through a delegator) is
        excluded, because calling it on the screen is still correct.
    """
    moved: dict[str, str] = {}
    for controller in CONSOLE_CONTROLLERS:
        for name, value in vars(controller).items():
            if not name.startswith("_") or name.startswith("__"):
                continue
            if not (inspect.isfunction(value) or inspect.iscoroutinefunction(value)):
                continue
            if hasattr(ChatScreen, name):
                continue
            moved.setdefault(name, controller.__name__)
    return moved


def _seams_bound_from_a_controller(tree: ast.AST) -> set[str]:
    """Seam names this module binds straight off a controller class.

    A double that declares ``_confirm_fleet_loss =
    ConsoleSessionController._confirm_fleet_loss`` is running the
    controller's OWN implementation, so calling it on that double is the
    correct route by definition -- reporting it would be a false positive.
    Detected structurally rather than by adding the double's variable name
    to an allow-list: a name list cannot tell a borrowed implementation
    from a `ChatScreen` subclass that happens to be called ``harness``.
    """
    bound: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if not isinstance(value, ast.Attribute) or not isinstance(value.value, ast.Name):
            continue
        if value.value.id not in CONTROLLER_CLASS_NAMES:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == value.attr:
                bound.add(value.attr)
    return bound


def scan_tree(tree: ast.AST, path: str, moved: dict[str, str]) -> list[Violation]:
    """Report every `<name>.<moved seam>(...)` call in one parsed test module."""
    violations: list[Violation] = []
    borrowed = _seams_bound_from_a_controller(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or not isinstance(func.value, ast.Name):
            continue
        base = func.value.id
        if base in CONTROLLER_CLASS_NAMES or base in CONTROLLER_VARIABLE_NAMES:
            continue
        if func.attr in borrowed:
            continue
        owner = moved.get(func.attr)
        if owner is None:
            continue
        violations.append(
            Violation(
                path=path,
                line=node.lineno,
                call=f"{base}.{func.attr}()",
                owner=owner,
            )
        )
    return violations


def test_moved_seam_inventory_is_not_empty():
    """A sweep over an empty rule set would pass vacuously forever."""
    moved = moved_console_seams()
    assert moved, (
        "No private callable was found that lives on a Console controller but "
        "not on ChatScreen -- the guard below would then check nothing."
    )


def test_no_test_calls_a_console_seam_that_moved_off_chatscreen():
    """Every test-suite call site addresses a seam that still exists."""
    moved = moved_console_seams()
    violations: list[Violation] = []
    scanned = 0
    for path in sorted(TESTS_DIR.rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        scanned += 1
        violations.extend(
            scan_tree(tree, str(path.relative_to(TESTS_DIR.parent)), moved)
        )
    assert scanned > 100, f"Only {scanned} test modules parsed; the sweep did not run."
    ordered = sorted(violations, key=lambda violation: (violation.path, violation.line))
    assert not violations, "Calls to seams that moved off ChatScreen:\n" + "\n".join(
        str(violation) for violation in ordered
    )


def test_rule_fires_on_the_task_14920_shape():
    """The exact shape that shipped red must be reported."""
    moved = moved_console_seams()
    seam = sorted(moved)[0]
    source = f"async def test_thing():\n    await console.{seam}(message_id)\n"
    found = scan_tree(ast.parse(source), "<synthetic>", moved)
    assert [violation.call for violation in found] == [f"console.{seam}()"]


@pytest.mark.parametrize(
    "template",
    [
        # the correct spelling: through the owning controller instance
        "async def test_thing():\n    await controller.{seam}(message_id)\n",
        # a direct class-level patch/reference target
        "def test_thing(monkeypatch):\n"
        "    monkeypatch.setattr(ConsoleSessionController, '{seam}', stub)\n",
        # an attribute chain, not a bare name -- already the controller route
        "async def test_thing():\n    await screen._session.{seam}(message_id)\n",
    ],
    ids=["controller-var", "class-attr", "controller-chain"],
)
def test_rule_does_not_fire_on_correct_spellings(template: str):
    """Calls that already route through the controller must not be reported."""
    moved = moved_console_seams()
    seam = sorted(moved)[0]
    assert scan_tree(ast.parse(template.format(seam=seam)), "<synthetic>", moved) == []
