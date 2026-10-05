"""``NotesSyncRuntimeOwner._publish`` is the only place a root status is built.

TASK-32633 fix round 1 (review Minor 2). The dated healthy label reads the
publication's own time (``NotesSyncRootRuntimeSnapshot.published_at``) and
falls back to a bare "✓ Up to date" only when a snapshot carries none. That
fallback is reachable from hand-built snapshots in tests and from nowhere in
production -- as long as every production snapshot comes out of ``_publish``,
which stamps the time. A second construction site would bring the
unqualified label back silently; this test makes it loud.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PACKAGE = Path(__file__).resolve().parents[2] / "tldw_chatbook"
_RUNTIME = "tldw_chatbook/Notes/notes_sync_runtime.py"

pytestmark = pytest.mark.unit


def _construction_sites() -> list[tuple[str, str]]:
    """Every ``NotesSyncRootRuntimeSnapshot(...)`` call: (file, enclosing def)."""

    sites: list[tuple[str, str]] = []
    for path in sorted(_PACKAGE.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "NotesSyncRootRuntimeSnapshot(" not in source:
            continue
        tree = ast.parse(source)
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name != "NotesSyncRootRuntimeSnapshot":
                continue
            owner = node
            while owner in parents and not isinstance(
                owner, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                owner = parents[owner]
            enclosing = (
                owner.name
                if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef))
                else "<module>"
            )
            sites.append((path.relative_to(_PACKAGE.parent).as_posix(), enclosing))
    return sites


def test_publish_is_the_only_root_snapshot_construction_site() -> None:
    sites = _construction_sites()
    assert sites == [(_RUNTIME, "_publish")], (
        "A NotesSyncRootRuntimeSnapshot is built outside _publish: "
        f"{sites}. Route it through _publish so it carries published_at, or "
        "the healthy label loses its time silently."
    )


def test_publish_stamps_the_publication_time() -> None:
    source = (_PACKAGE.parent / _RUNTIME).read_text(encoding="utf-8")
    publish = source[source.index("    async def _publish(") :]
    publish = publish[: publish.index("\n    def ")]
    assert "published_at=time.time()" in publish
