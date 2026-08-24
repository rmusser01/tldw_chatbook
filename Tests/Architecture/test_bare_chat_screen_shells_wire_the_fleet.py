"""A bypassed-``__init__`` ChatScreen shell that sets its store must wire ``_fleet``.

`ChatScreen.__init__` installs `_fleet` via
`Console_Modules/wiring.build_console_controllers`, so a
`ChatScreen.__new__(ChatScreen)` shell never has one. That stays invisible until
the shell touches a seam that reaches it — and one of the most ordinary lines a
Console test writes does. `screen._console_chat_store = store` is a property
whose setter calls `self._console_runtime().set_chat_store(...)`, which builds
the chat controller's kwargs, which reads
`self._fleet._console_wake_user_priority`.

The result is a test that dies while being *set up*, with an `AttributeError`
naming an attribute the test file never mentions. TASK-21381 found 115 of them
across 8 files, and the count only grew because each new bare-screen helper was
written by copying one that already worked — before production grew the
dependency.

This guard is the ratchet: a new shell that assigns the store without wiring a
fleet controller fails here, naming its own function, rather than 20 tests
failing later on an unrelated-looking attribute.

There are two ways to satisfy this. Wire a fleet controller with
`Tests/UI/console_controller_stubs.stub_fleet_controller`, whose raiser defaults
keep the shell fail-loud at any seam it has not wired; or hand the shell its own
`_console_runtime_ref`, which `_console_runtime()` returns verbatim so the
kwargs build never happens. The guard accepts either, because the invariant is
"do not let the store setter build a runtime this shell cannot satisfy", not
"call this one helper".
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]

#: Functions allowed to build a store-setting shell without a fleet controller.
#: Shrink-only: an entry may be removed when its function is fixed, never added.
#: Empty on purpose — every known case was repaired in TASK-21381.
ALLOWLIST: frozenset[str] = frozenset()


def _is_chat_screen_new(node: ast.AST) -> bool:
    """``ChatScreen.__new__(...)``, however the module spells the attribute."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "__new__"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ChatScreen"
    )


def _sets_console_chat_store(node: ast.AST) -> bool:
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    return any(
        isinstance(t, ast.Attribute) and t.attr == "_console_chat_store"
        for t in targets
    )


def _supplies_own_runtime(node: ast.AST) -> bool:
    """``screen._console_runtime_ref = ...`` — the other way to be safe.

    `_console_runtime()` returns a pre-set `_console_runtime_ref` verbatim and
    only calls `ensure_console_runtime` when it finds none. A shell that
    supplies its own runtime therefore never reaches the kwargs build, so it
    needs no fleet controller at all. `Tests/UI/test_console_native_chat_flow.py`
    does exactly this, and it is arguably the cleaner of the two fixes.
    """
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    return any(
        isinstance(t, ast.Attribute) and t.attr == "_console_runtime_ref"
        for t in targets
    )


def _calls_fleet_stub(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    return name == "stub_fleet_controller"


def _offending_functions(tree: ast.AST, rel: str) -> list[str]:
    out: list[str] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = list(ast.walk(fn))
        if not any(_is_chat_screen_new(n) for n in body):
            continue
        if not any(_sets_console_chat_store(n) for n in body):
            continue
        if any(_calls_fleet_stub(n) or _supplies_own_runtime(n) for n in body):
            continue
        out.append(f"{rel}::{fn.name}")
    return out


def test_no_bare_chat_screen_shell_sets_its_store_without_a_fleet_controller() -> None:
    """Fail on any shell that assigns the store without satisfying the runtime.

    Scans every test module that mentions both spellings, and reports the
    offending functions by name rather than the first one found -- a sweep that
    stops at one violation makes a multi-file regression take as many rounds to
    clear as it has files.

    Raises:
        AssertionError: If any function outside `ALLOWLIST` builds a shell with
            `ChatScreen.__new__` and assigns `_console_chat_store` without
            either wiring a fleet controller or supplying `_console_runtime_ref`.
    """
    offenders: list[str] = []
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        # Cheap reject before paying for a parse: both spellings must appear.
        if "ChatScreen.__new__" not in text or "_console_chat_store" not in text:
            continue
        rel = path.relative_to(TESTS_ROOT.parent).as_posix()
        offenders.extend(_offending_functions(ast.parse(text), rel))

    new = sorted(set(offenders) - ALLOWLIST)
    assert not new, (
        "These functions build a ChatScreen shell with __new__ and assign "
        "_console_chat_store without wiring a fleet controller. The assignment "
        "is a property whose setter reaches _fleet._console_wake_user_priority, "
        "so the shell will die during setup with an AttributeError that names "
        "neither this function nor the behaviour under test. Either call "
        "Tests.UI.console_controller_stubs.stub_fleet_controller(screen) before "
        "the assignment, or give the shell its own _console_runtime_ref.\n  " + "\n  ".join(new)
    )


def test_the_allowlist_does_not_name_a_function_that_is_already_clean() -> None:
    """Refuse allowlist entries that no longer describe a violation.

    A ratchet that keeps stale entries stops ratcheting: the list grows a
    reputation for being noise, and the next real entry is waved through with
    it. Anything listed must still be a genuine violation, or the list should
    shrink.

    Raises:
        AssertionError: If an `ALLOWLIST` entry names a function that no longer
            violates the rule.
    """
    if not ALLOWLIST:
        return
    live: set[str] = set()
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "ChatScreen.__new__" not in text or "_console_chat_store" not in text:
            continue
        rel = path.relative_to(TESTS_ROOT.parent).as_posix()
        live.update(_offending_functions(ast.parse(text), rel))
    stale = sorted(ALLOWLIST - live)
    assert not stale, f"allowlist entries no longer violate; remove them: {stale}"
