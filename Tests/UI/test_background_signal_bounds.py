# test_background_signal_bounds.py
# Description: AST guard -- no unbounded wait on a background-set signal (task-14912).
#
# The rule this file enforces, and why, is documented in
# Tests/UI/background_signals.py. Short version: `await some_event.wait()` where
# only background work can set `some_event` is an UNBOUNDED wait. If that work
# raises, a fire-and-forget task swallows the exception, the signal never
# arrives, and the test hangs forever. Under this repo's
# `timeout_method = thread` a hung test kills the WHOLE pytest process, so every
# test after it in the file silently never runs -- the file's pass count becomes
# a lie.
#
# This is checked by AST rather than grep because grep cannot tell an unbounded
# `await ev.wait()` from one already wrapped in `asyncio.wait_for`, cannot tell
# an `asyncio.Event` from a Textual `Worker` or a `RetainedPushOperation` (both
# of which expose a `.wait()` that re-raises, so neither can strand), and cannot
# see whether a spawn precedes the wait in the same scope.

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

TESTS_UI_DIR = Path(__file__).resolve().parent

# Calls that start background work the test does not await inline.
SPAWN_CALLS = frozenset({"create_task", "ensure_future", "run_worker"})

# Wrappers that already impose a bound. The three `*_background_*` /
# `wait_for_signal` names are Tests/UI/background_signals.py's helpers; the
# underscore-prefixed spellings are the pre-task-14912 aliases still re-exported
# from test_screen_navigation.py.
BOUNDING_CALLS = frozenset(
    {
        "wait_for",
        "wait_for_signal",
        "_wait_for_signal",
        "wait_for_background_signal",
        "_wait_for_background_signal",
        "await_background_task",
        "_await_background_task",
    }
)


@dataclass(frozen=True)
class Violation:
    """One unbounded `await <event>.wait()`."""

    path: str
    line: int
    receiver: str
    function: str
    reason: str

    def __str__(self) -> str:  # pragma: no cover - failure formatting only
        return (
            f"{self.path}:{self.line}  await {self.receiver}.wait()  "
            f"in {self.function}()  [{self.reason}]"
        )


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _receiver(node: ast.Call) -> str:
    func = node.func
    return ast.unparse(func.value) if isinstance(func, ast.Attribute) else ""


def _is_bounding(node: ast.Call) -> bool:
    """Whether this call imposes a timeout on whatever it wraps."""
    name = _call_name(node)
    if name in BOUNDING_CALLS:
        return True
    receiver = _receiver(node)
    if name in {"timeout", "timeout_at"} and receiver.endswith("asyncio"):
        return True
    if name == "wait" and receiver.endswith("asyncio"):
        # asyncio.wait({...}, timeout=...) -- bounded only with an explicit timeout
        return any(kw.arg == "timeout" for kw in node.keywords)
    return False


def _event_expressions(tree: ast.AST) -> tuple[set[str], set[str]]:
    """Every expression, and every attribute name, assigned an ``asyncio.Event``.

    Returns:
        (full expressions, bare attribute names). The second set exists because
        a fake declares ``self.started = asyncio.Event()`` and the test awaits
        ``service.started.wait()`` -- different text, same object.
    """
    full: set[str] = set()
    attrs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not isinstance(value, ast.Call) or _call_name(value) != "Event":
            continue
        source = ast.unparse(value.func)
        if "asyncio" not in source and source != "Event":
            continue  # threading.Event().wait(timeout=...) is already bounded
        for target in targets:
            if isinstance(target, ast.Name):
                full.add(target.id)
            elif isinstance(target, ast.Attribute):
                full.add(ast.unparse(target))
                attrs.add(target.attr)
    return full, attrs


class _FunctionScan(ast.NodeVisitor):
    """Scan one function body, never descending into nested definitions."""

    def __init__(self) -> None:
        self.spawn_lines: list[int] = []
        self.awaited_waits: list[tuple[int, str, bool]] = []
        self._bounded = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]
    visit_Lambda = visit_FunctionDef  # type: ignore[assignment]

    def visit_Call(self, node: ast.Call) -> None:
        if _call_name(node) in SPAWN_CALLS:
            self.spawn_lines.append(node.lineno)
        if _is_bounding(node):
            self._bounded += 1
            self.generic_visit(node)
            self._bounded -= 1
            return
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        bounded = any(
            isinstance(item.context_expr, ast.Call) and _is_bounding(item.context_expr)
            for item in node.items
        )
        if bounded:
            self._bounded += 1
            self.generic_visit(node)
            self._bounded -= 1
            return
        self.generic_visit(node)

    visit_AsyncWith = visit_With  # type: ignore[assignment]

    def visit_Await(self, node: ast.Await) -> None:
        value = node.value
        if (
            isinstance(value, ast.Call)
            and _call_name(value) == "wait"
            and isinstance(value.func, ast.Attribute)
            and not _is_bounding(value)
        ):
            self.awaited_waits.append(
                (node.lineno, _receiver(value), self._bounded > 0)
            )
        self.generic_visit(node)


def _matches_event(receiver: str, full: set[str], attrs: set[str]) -> bool:
    if receiver in full:
        return True
    tail = receiver.rsplit(".", 1)[-1]
    return "." in receiver and tail in attrs


def _module_level_functions(tree: ast.AST) -> set[int]:
    """Ids of functions defined directly at module level.

    Only those are pytest test bodies. A ``test_``-prefixed METHOD is a service
    fake's stub (``ToolTestHubService.test_hub_tool``), and a stub awaiting a
    release the test sets is the inverse, safe shape.
    """
    return {
        id(node)
        for node in getattr(tree, "body", [])
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def scan_tree(tree: ast.AST, path: str) -> list[Violation]:
    """Return every unbounded wait on a background-set ``asyncio.Event``."""
    full, attrs = _event_expressions(tree)
    if not full and not attrs:
        return []
    top_level = _module_level_functions(tree)
    violations: list[Violation] = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        scan = _FunctionScan()
        for statement in function.body:
            scan.visit(statement)
        is_test_body = function.name.startswith("test_") and id(function) in top_level
        for line, receiver, bounded in scan.awaited_waits:
            if bounded or not _matches_event(receiver, full, attrs):
                continue
            spawned_before = any(spawn <= line for spawn in scan.spawn_lines)
            if spawned_before:
                reason = "background work was spawned in this scope before the wait"
            elif is_test_body:
                reason = "a test body may only wait on a signal with a bound"
            else:
                # A stub awaiting a release the TEST sets is the inverse shape
                # and cannot strand the run: the setter is the test itself.
                continue
            violations.append(
                Violation(
                    path=path,
                    line=line,
                    receiver=receiver,
                    function=function.name,
                    reason=reason,
                )
            )
    return sorted(violations, key=lambda v: (v.path, v.line))


def scan_directory(directory: Path) -> list[Violation]:
    """Scan every Python module under ``directory``."""
    violations: list[Violation] = []
    for path in sorted(directory.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        violations.extend(scan_tree(ast.parse(source), str(path.relative_to(directory))))
    return violations


def test_no_unbounded_waits_on_background_signals():
    """No Tests/UI wait on a background-set Event may be unbounded.

    Enumerated by AST, not grep -- an asserted sweep is exactly what task-14912
    exists to distrust. Fix a failure by routing the wait through
    ``Tests/UI/background_signals.py``:

      * the test owns the task -> ``wait_for_background_signal(ev, task, what=...)``
        (re-raises the exception the task swallowed)
      * the product owns the work -> ``wait_for_signal(ev, what=...)``
        (timeout-only, but still a named failure instead of a dead process)
    """
    violations = scan_directory(TESTS_UI_DIR)
    assert not violations, (
        "unbounded waits on a signal only background work can set "
        f"({len(violations)}):\n  " + "\n  ".join(str(v) for v in violations)
    )


def test_rule_detects_the_incident_shape():
    """The AST rule must actually fire on the task-3316 shape it exists to ban."""
    source = (
        "import asyncio\n"
        "async def test_thing():\n"
        "    started = asyncio.Event()\n"
        "    task = asyncio.create_task(screen.reload())\n"
        "    await started.wait()\n"
    )
    violations = scan_tree(ast.parse(source), "<synthetic>")
    assert [(v.line, v.receiver) for v in violations] == [(5, "started")]


@pytest.mark.parametrize(
    "source",
    [
        # already bounded by the shared helper
        "import asyncio\n"
        "async def test_thing():\n"
        "    started = asyncio.Event()\n"
        "    task = asyncio.create_task(screen.reload())\n"
        "    await wait_for_background_signal(started, task, what='reload')\n",
        # already bounded by asyncio.wait_for
        "import asyncio\n"
        "async def test_thing():\n"
        "    started = asyncio.Event()\n"
        "    task = asyncio.create_task(screen.reload())\n"
        "    await asyncio.wait_for(started.wait(), timeout=5)\n",
        # a stub waiting on a release the test body sets -- the inverse shape
        "import asyncio\n"
        "release = asyncio.Event()\n"
        "async def stub():\n"
        "    await release.wait()\n",
        # a Textual Worker / retained-operation handle re-raises, it cannot strand
        "import asyncio\n"
        "gate = asyncio.Event()\n"
        "async def test_thing():\n"
        "    worker = widget.run_worker(thing())\n"
        "    await worker.wait()\n",
        # a `test_`-prefixed METHOD is a service fake's stub, not a pytest test
        "import asyncio\n"
        "class Fake:\n"
        "    def __init__(self):\n"
        "        self.test_gate = asyncio.Event()\n"
        "    async def test_hub_tool(self):\n"
        "        await self.test_gate.wait()\n",
    ],
    ids=["helper", "wait_for", "stub-release", "worker-handle", "fake-method"],
)
def test_rule_does_not_fire_on_safe_shapes(source: str):
    """Shapes that cannot strand a run must not be reported."""
    assert scan_tree(ast.parse(source), "<synthetic>") == []
