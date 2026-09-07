"""TASK-1349 AC#2: no predicate-named function in the Watchlists modules
may have a side effect.

The bug this guards against: a function named like a pure query
(`_is_*`/`_has_*`/`_can_*`/`*_is_blocked`) that actually calls `self.notify`,
posts a message, or writes the DB. Such a function is safe until someone wires
it into a render path -- this codebase already shipped `provider_is_configured()`
writing an `eval_models` row from `compose()`, so opening the Evals screen
mutated the DB on every fresh install. The name gave no warning.

`_content_toggle_is_blocked` (a predicate that notified) was renamed to the
action `_refuse_content_toggle_off_read_tab` to fix exactly this; this test
keeps the whole module family clean going forward -- a new side-effecting
predicate fails here rather than waiting to be discovered from a render path.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# A repo-root-relative walk so the test is independent of the CWD pytest runs in.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_WATCHLISTS_DIRS = [
    _REPO_ROOT / "tldw_chatbook" / "UI" / "Watchlists_Modules",
    _REPO_ROOT / "tldw_chatbook" / "UI" / "Screens" / "watchlists_collections_screen.py",
]

# Names that read as a pure boolean query.
_PREDICATE_PREFIXES = ("_is_", "_has_", "_can_", "_should_", "is_", "has_", "can_")
_PREDICATE_SUFFIXES = ("_is_blocked", "_is_allowed", "_is_open", "_is_active")

# Attribute/call names that mutate observable state.
_SIDE_EFFECTS = {
    "notify",
    "post_message",
    "execute",
    "executemany",
    "commit",
    "_schedule_layout_persist",
}


def _python_files():
    for target in _WATCHLISTS_DIRS:
        if target.is_dir():
            # Recursive: a predicate in a future nested subpackage must be
            # covered too (Qodo, TASK-1349).
            yield from target.rglob("*.py")
        elif target.is_file():
            yield target


def _is_predicate_name(name: str) -> bool:
    return name.startswith(_PREDICATE_PREFIXES) or any(
        name.endswith(s) for s in _PREDICATE_SUFFIXES
    )


def _side_effects_in(node: ast.AST) -> set[str]:
    hits: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            if sub.func.attr in _SIDE_EFFECTS:
                hits.add(sub.func.attr)
    return hits


def test_no_predicate_named_function_in_watchlists_has_a_side_effect():
    offenders = []
    for path in _python_files():
        # Explicit UTF-8: these sources carry non-ASCII glyphs (e.g. the
        # `▸` collapsed-region marker), so a default-locale read could
        # UnicodeDecodeError or mis-decode on some runners (Qodo, TASK-1349).
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _is_predicate_name(node.name):
                continue
            effects = _side_effects_in(node)
            if effects:
                offenders.append(
                    f"{path.name}:{node.lineno} {node.name} -> {sorted(effects)}"
                )
    assert not offenders, (
        "predicate-named functions must be pure; rename to an action (verb) or "
        "move the side effect to the caller (TASK-1349):\n  "
        + "\n  ".join(offenders)
    )
