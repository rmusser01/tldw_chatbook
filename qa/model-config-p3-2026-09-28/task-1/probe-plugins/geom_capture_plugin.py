"""Scratch probe plugin (TASK-33003.1 review round 1, finding 7).

Wraps ``App.run_test`` so that, as each existing test's app context exits,
the active screen's Collapsible / CollapsibleTitle / Select / displayed
focusable geometry is appended to ``$GEOM_OUT/<worker>.jsonl``. Running the
same composer test files in the base and head trees and diffing the dumps
reuses every existing harness instead of hand-mounting each composer.
Tests do not change; fixtures and isolation run exactly as usual.
"""

import json
import os
from contextlib import asynccontextmanager

import textual.app

_ORIG = textual.app.App.run_test
_OUT = os.environ["GEOM_OUT"]
_NODE = {"id": None, "n": 0}


def pytest_runtest_setup(item):
    _NODE["id"] = item.nodeid
    _NODE["n"] = 0


def _shown(widget) -> bool:
    node = widget
    while node is not None and hasattr(node, "display"):
        if not node.display or not node.visible:
            return False
        node = node.parent
    return True


def _dump(app) -> None:
    from textual.widgets import Collapsible, Select
    from textual.widgets._collapsible import CollapsibleTitle

    screen = app.screen
    rows = []
    for widget in screen.walk_children(with_self=False):
        kind = None
        if isinstance(widget, (Collapsible, CollapsibleTitle, Select)):
            kind = type(widget).__mro__[0].__name__
            for base in (CollapsibleTitle, Collapsible, Select):
                if isinstance(widget, base):
                    kind = base.__name__
                    break
        elif widget.focusable:
            kind = "focusable"
        if kind is None or not _shown(widget):
            continue
        region = widget.region
        parent = widget.parent
        parent_region = getattr(parent, "region", None)
        rows.append(
            {
                "kind": kind,
                "type": type(widget).__name__,
                "id": widget.id,
                "path": " > ".join(
                    f"{type(n).__name__}#{n.id}" if n.id else type(n).__name__
                    for n in reversed(list(widget.ancestors_with_self)[:4])
                ),
                "region": list(region),
                "zero": region.area == 0,
                "hclip": bool(
                    parent_region is not None
                    and parent_region.area
                    and region.area
                    and (region.x < parent_region.x or region.right > parent_region.right)
                ),
                "sib": [id(s) for s in []],
                "_pid": id(parent),
                "_layer": widget.styles.layer or "",
            }
        )
    # sibling overlap among shown, non-zero, same-parent, same-layer widgets
    by_parent = {}
    for row in rows:
        if not row["zero"]:
            by_parent.setdefault((row["_pid"], row["_layer"]), []).append(row)
    for group in by_parent.values():
        for i, a in enumerate(group):
            ra = textual.geometry.Region(*a["region"])
            for b in group[i + 1 :]:
                rb = textual.geometry.Region(*b["region"])
                if ra.overlaps(rb):
                    a.setdefault("overlaps", []).append(b["path"])
    for row in rows:
        row.pop("_pid")
        row.pop("sib")
    _NODE["n"] += 1
    record = {
        "node": _NODE["id"],
        "n": _NODE["n"],
        "screen": type(screen).__name__,
        "size": list(app.size),
        "rows": rows,
    }
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    with open(os.path.join(_OUT, f"{worker}.jsonl"), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


import textual.geometry  # noqa: E402


@asynccontextmanager
async def _run_test(self, *args, **kwargs):
    async with _ORIG(self, *args, **kwargs) as pilot:
        try:
            yield pilot
        finally:
            try:
                await pilot.pause()
                _dump(self)
            except Exception as exc:  # probe must never change the test result
                worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
                with open(os.path.join(_OUT, f"{worker}.err"), "a") as fh:
                    fh.write(f"{_NODE['id']}: {type(exc).__name__}: {exc}\n")


textual.app.App.run_test = _run_test
