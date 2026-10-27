"""String lookups on the app object only go down (TASK-36942.4).

Architecture review 2026-10-25 F1: code reaches application services with
``getattr(app, "<name>")`` / ``hasattr(app, "<name>")`` instead of an explicit
service, so a rename or removal fails at runtime, not at import. This counts
those literal-name lookups per top-level package (top-level modules count under
``<root>``) and pins the counts in ``app_lookup_census.tsv``. A receiver is
app-like when its last name segment is ``app``, ``app_instance``, ``tldw_app``,
``_app`` or ``_app_instance`` (``app``, ``self.app_instance``, ``screen.app``).

Non-literal lookups such as ``getattr(app, name)`` are not counted (about 45,
e.g. ``Event_Handlers/LLM_Management_Events/server_lifecycle.py``), so they
bypass both the count and the private-name rule.

* A count above its pin fails: reach the service explicitly instead.
* A count below its pin fails too: lower the pin in the same change.
* A lookup of an underscore-prefixed app attribute that is not in
  ``app_private_attrs.txt`` fails. That list is frozen; do not add to it.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1] / "tldw_chatbook"
CENSUS = HERE / "app_lookup_census.tsv"
PRIVATE_ALLOWED = HERE / "app_private_attrs.txt"
_RECEIVERS = {"app", "app_instance", "tldw_app", "_app", "_app_instance"}


def _receiver(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _scan() -> tuple[Counter[str], set[str]]:
    counts: Counter[str] = Counter()
    private: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        rel = path.relative_to(PACKAGE)
        pkg = rel.parts[0] if len(rel.parts) > 1 else "<root>"
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("getattr", "hasattr")
                and len(node.args) > 1
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)
                and _receiver(node.args[0]) in _RECEIVERS
            ):
                continue
            counts[pkg] += 1
            name = node.args[1].value
            if name.startswith("_") and not name.startswith("__"):
                private.add(name)
    return counts, private


def _pins() -> dict[str, int]:
    rows = (l.split("\t") for l in CENSUS.read_text().splitlines() if l.strip())
    return {pkg: int(n) for pkg, n in rows}


_COUNTS, _PRIVATE = _scan()
_PINS = _pins()


@pytest.mark.unit
@pytest.mark.parametrize("pkg", sorted(set(_COUNTS) | set(_PINS)))
def test_app_string_lookups_match_their_pin(pkg: str) -> None:
    count, pin = _COUNTS.get(pkg, 0), _PINS.get(pkg, 0)
    assert count <= pin, (
        f"{pkg}: {count} literal getattr/hasattr lookups on the app object "
        f"(pinned {pin}, +{count - pin}). Reach the service explicitly; do NOT "
        f"raise the pin ({CENSUS.name})."
    )
    assert count >= pin, (
        f"{pkg}: {count} lookups, pinned {pin}. Lower the pin to {count} in "
        f"{CENSUS.name} so the real measurement is what's pinned."
    )


@pytest.mark.unit
def test_no_new_private_app_attribute_lookups() -> None:
    allowed = set(PRIVATE_ALLOWED.read_text().split())
    new = sorted(_PRIVATE - allowed)
    assert not new, (
        f"New lookup of private app attribute(s) {new}. Private names are "
        f"frozen in {PRIVATE_ALLOWED.name}; expose a public service instead."
    )
    stale = sorted(allowed - _PRIVATE)
    assert not stale, (
        f"{stale} no longer looked up; remove from {PRIVATE_ALLOWED.name}."
    )
