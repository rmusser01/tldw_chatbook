"""TASK-33620.1.12: every production agent loop holds its tool calls during a stop.

``LoopDeps.emergency_stop_refusal`` defaults to ``None`` (no stop wired), so
the pure loop's own tests stay simple -- and so a second production
``LoopDeps(...)`` built without it would run tools through an active
emergency stop, with nothing failing. This pins that every construction in
the package passes the probe, and passes something other than ``None``.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "tldw_chatbook"


def _loop_deps_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name == "LoopDeps":
            yield node


def _constructions() -> list[tuple[str, ast.Call]]:
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "LoopDeps(" not in text:
            continue
        for call in _loop_deps_calls(ast.parse(text)):
            found.append((f"{path.relative_to(PACKAGE.parent)}:{call.lineno}", call))
    return found


def test_the_package_builds_loop_deps_somewhere():
    """Guard the guard: a scan that finds nothing proves nothing."""
    assert _constructions(), "no LoopDeps(...) construction found in tldw_chatbook"


def test_every_loop_deps_construction_wires_the_emergency_stop():
    unwired = []
    for where, call in _constructions():
        keyword = next(
            (kw for kw in call.keywords if kw.arg == "emergency_stop_refusal"), None
        )
        if keyword is None or (
            isinstance(keyword.value, ast.Constant) and keyword.value.value is None
        ):
            unwired.append(where)
    assert not unwired, (
        "LoopDeps(...) built without an emergency_stop_refusal probe: its run "
        f"would execute tools during an emergency stop: {unwired}"
    )
