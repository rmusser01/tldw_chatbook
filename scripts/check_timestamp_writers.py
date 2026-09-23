#!/usr/bin/env python3
"""Guard the UTC-timestamp write contract (ADR-173 / TASK-32803.1 AC#2/#3).

ADR-173 makes ``tldw_chatbook/Utils/timestamps.py`` the one sanctioned producer
of stored UTC timestamps (canonical millisecond ``Z`` shape,
``YYYY-MM-DDTHH:MM:SS.mmmZ``) and reader of every shape on disk.

The predicate is the **emitted format**, not the writer's naivety. That
distinction is the bug this guard shipped with: it matched only naive calls and
its own docstring declared ``datetime.now(timezone.utc).isoformat()`` "fine",
when ADR-173 names that exact expression as a drifting shape to eliminate
(:14) and mandates a ``Z`` suffix (:46). The census was therefore empty and the
check printed OK while 117 live writers emitted a non-canonical shape. It is a
data bug, not a style nit: ``'+'`` is 0x2B and ``'Z'`` is 0x5A, so for the same
instant a ``+00:00`` row always sorts *before* the canonical one and a
``WHERE ts >= '<...>Z'`` cutoff silently excludes it.

Kinds matched:

1. ``utcnow`` — ``datetime.utcnow()``, deprecated in 3.12 and always **naive**;
   never correct here. Forbidden outright (never censused): fix it to
   ``utc_now_iso()`` (canonical string) or ``datetime.now(timezone.utc)``
   (aware datetime).

2. ``naive_now_iso`` — ``datetime.now().isoformat()`` with no timezone
   argument returns *local* time and serialises it as if it were the stored
   shape: the "naive local time stored as UTC" latent bug ADR-173 describes.

3. ``offset_now_iso`` — ``datetime.now(<tz>).isoformat()``, which emits
   ``+00:00`` where ADR-173 mandates ``Z``. A call whose result is immediately
   ``.replace("+00:00", "Z")``-ed is *not* counted: that is the conforming
   idiom the shared helper uses. The exemption requires the replacement to
   actually run -- exactly two arguments (``replace("+00:00", "Z", 0)`` and
   ``count=0`` replace nothing yet matched an ``args[:2]`` check), against a
   UTC-spelled ``now()`` (against another zone the replacement is a no-op and
   the value still ships ``+05:30``).

4. ``strftime_iso`` — a hand-rolled ISO-8601 ``strftime`` shape
   (``"%Y-%m-%dT%H:%M:%S.%f"``, ``"%Y%m%dT%H%M%SZ"``, …). Only ``T``-separated
   date-time formats match; human-display formats (``"%Y-%m-%d %H:%M"``,
   ``"%B %d, %Y"``) are out of scope and deliberately do not.

Kinds 2-4 are ratcheted, not gated: the occurrences that already exist are
pinned in ``scripts/timestamp_writer_census.tsv`` and only a new or grown one
fails, so TASK-32901 can migrate them to the shared helper incrementally.

Known remaining gap: ``isoformat().replace("+00:00", "Z")`` at *microsecond*
precision is treated as conforming here, but is variable-width (the fraction
vanishes at whole seconds) and so is not strictly canonical either. Widening to
that is a larger census and belongs with the migration, not with this guard.

The census is keyed on ``module<TAB>symbol<TAB>kind<TAB>count`` (the enclosing
function/method qualname, not line numbers, so ordinary edits do not churn it).
A key that is new, or whose count has grown, fails; removing occurrences is
always allowed — regenerate with ``--write`` to shrink the census.

Usage:
    python scripts/check_timestamp_writers.py            # verify (preflight/CI)
    python scripts/check_timestamp_writers.py --write    # regenerate the census
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOT = REPO_ROOT / "tldw_chatbook"
CENSUS = Path(__file__).resolve().parent / "timestamp_writer_census.tsv"

KIND_UTCNOW = "utcnow"
KIND_NAIVE_NOW_ISO = "naive_now_iso"
KIND_OFFSET_NOW_ISO = "offset_now_iso"
KIND_STRFTIME_ISO = "strftime_iso"

#: Kinds that are pinned in the census and fail only when new or grown.
RATCHETED_KINDS = (KIND_NAIVE_NOW_ISO, KIND_OFFSET_NOW_ISO, KIND_STRFTIME_ISO)

#: A ``strftime`` format that is an ISO-8601 ``T``-separated date-time, i.e. a
#: machine/stored shape. Display formats (space- or comma-separated) do not
#: match, which is why this is a regex on the format and not "any strftime".
ISO_STRFTIME = re.compile(r"%Y-?%m-?%dT")

#: Files excluded from the scan: the shared helper (its aware datetime use is
#: the sanctioned one) and this guard itself.
_EXCLUDE = {
    PRODUCTION_ROOT / "Utils" / "timestamps.py",
}

Occurrence = tuple[str, str, str]  # (module, symbol, kind)

#: Spellings of UTC accepted as the ``tz`` argument of ``now(...)``.
#: ``ZoneInfo("UTC")`` is deliberately not among them -- it is equivalent but
#: not statically distinguishable from ``ZoneInfo("Asia/Kolkata")``, and the
#: safe direction for a ratcheted census is one extra pinned row, never a
#: silent exemption. Nothing in the package writes it.
_UTC_SPELLINGS = {"utc", "UTC"}


def _replaces_offset(call: ast.Call) -> bool:
    """Whether a ``.replace(...)`` call really rewrites ``+00:00`` to ``Z``.

    Args:
        call: The ``.replace(...)`` call wrapping an ``isoformat()`` result.

    Returns:
        True only for exactly the two-argument canonical form. A third
        argument -- ``replace("+00:00", "Z", 0)``, or ``count=0`` -- caps the
        replacements and can leave the offset untouched, and the check
        inspected only ``args[:2]``, so such a call bought a full exemption
        while still emitting ``+00:00``.
    """
    if call.keywords or len(call.args) != 2:
        return False
    return [
        arg.value for arg in call.args if isinstance(arg, ast.Constant)
    ] == ["+00:00", "Z"]


def _emits_utc_offset(receiver: ast.AST) -> bool:
    """Whether ``receiver.isoformat()`` can produce the ``+00:00`` being replaced.

    Args:
        receiver: The expression ``isoformat()`` is called on.

    Returns:
        False only for a ``now(<non-UTC tz>)`` receiver, where the
        replacement is a no-op and the value still ships its own offset
        (``+05:30``) -- the very ``offset_now_iso`` shape the census exists
        to track. True otherwise, including for receivers this guard does not
        classify at all (a bare name, ``astimezone(...)``), which the
        ``offset_now_iso`` predicate never matches anyway.
    """
    if not (
        isinstance(receiver, ast.Call)
        and isinstance(receiver.func, ast.Attribute)
        and receiver.func.attr == "now"
    ):
        return True
    args = [*receiver.args, *(keyword.value for keyword in receiver.keywords)]
    if len(args) != 1:
        return False
    tz = args[0]
    if isinstance(tz, ast.Attribute):
        return tz.attr in _UTC_SPELLINGS
    return isinstance(tz, ast.Name) and tz.id in _UTC_SPELLINGS


class _Visitor(ast.NodeVisitor):
    """Collect timestamp-writer occurrences with their enclosing qualname."""

    def __init__(self, module: str) -> None:
        self.module = module
        self._stack: list[str] = []
        self.hits: Counter[Occurrence] = Counter()
        #: ids of ``.isoformat()`` calls whose result is immediately
        #: ``.replace("+00:00", "Z")``-ed. Populated when the *outer* replace
        #: call is visited, which ``generic_visit`` always reaches before its
        #: own receiver.
        self._canonicalised: set[int] = set()
        #: `id()` of every Attribute node that is a Call's `func`, so
        #: `visit_Attribute` can count the BARE references only and not
        #: double-count what `visit_Call` already saw.
        self._called: set[int] = set()

    @property
    def _symbol(self) -> str:
        return ".".join(self._stack) if self._stack else "<module>"

    def _enter(self, node: ast.AST) -> None:
        self._stack.append(node.name)  # type: ignore[attr-defined]
        self.generic_visit(node)
        self._stack.pop()

    visit_FunctionDef = _enter
    visit_AsyncFunctionDef = _enter
    visit_ClassDef = _enter

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            self._called.add(id(func))
        # datetime.utcnow()  (attr == "utcnow", any receiver)
        if isinstance(func, ast.Attribute) and func.attr == "utcnow":
            self.hits[(self.module, self._symbol, KIND_UTCNOW)] += 1

        # <...>.isoformat().replace("+00:00", "Z") is the conforming idiom:
        # exempt the inner call before we descend into it. The exemption must
        # only cover a replacement that actually RUNS -- see _replaces_offset.
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "replace"
            and isinstance(func.value, ast.Call)
            and isinstance(func.value.func, ast.Attribute)
            and func.value.func.attr == "isoformat"
            and _replaces_offset(node)
            and _emits_utc_offset(func.value.func.value)
        ):
            self._canonicalised.add(id(func.value))

        # <...>.now(...).isoformat() — naive (local time stored as UTC) or
        # aware (emits "+00:00" where ADR-173 mandates "Z").
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "isoformat"
            and isinstance(func.value, ast.Call)
            and isinstance(func.value.func, ast.Attribute)
            and func.value.func.attr == "now"
            and id(node) not in self._canonicalised
        ):
            now = func.value
            kind = (
                KIND_NAIVE_NOW_ISO
                if not now.args and not now.keywords
                else KIND_OFFSET_NOW_ISO
            )
            self.hits[(self.module, self._symbol, kind)] += 1

        # <...>.strftime("<ISO-8601 T-separated format>")
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "strftime"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and ISO_STRFTIME.search(node.args[0].value)
        ):
            self.hits[(self.module, self._symbol, KIND_STRFTIME_ISO)] += 1

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # A BARE `datetime.utcnow` reference -- no `()`. This is how a
        # pydantic field spells it: `Field(default_factory=datetime.utcnow)`.
        # Matching calls only reported "0 sites ... OK" over a live
        # occurrence (`tldw_api/chat_loop_schemas.py`, tier-2 review S06).
        # Equally forbidden: the value it produces is the same naive one.
        if node.attr == "utcnow" and id(node) not in self._called:
            self.hits[(self.module, self._symbol, KIND_UTCNOW)] += 1
        self.generic_visit(node)


def scan_tree() -> Counter[Occurrence]:
    hits: Counter[Occurrence] = Counter()
    for path in sorted(PRODUCTION_ROOT.rglob("*.py")):
        if path in _EXCLUDE:
            continue
        module = path.relative_to(REPO_ROOT).with_suffix("").as_posix()
        try:
            with warnings.catch_warnings():
                # Some scanned files carry pre-existing invalid-escape strings;
                # their SyntaxWarning is not this guard's concern.
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        visitor = _Visitor(module)
        visitor.visit(tree)
        hits.update(visitor.hits)
    return hits


def read_census() -> Counter[Occurrence]:
    census: Counter[Occurrence] = Counter()
    if not CENSUS.exists():
        return census
    for lineno, raw in enumerate(CENSUS.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            print(f"FAIL: {CENSUS.name}:{lineno}: expected 4 TAB-separated fields")
            raise SystemExit(2)
        module, symbol, kind, count = parts
        census[(module, symbol, kind)] += int(count)
    return census


def write_census(hits: Counter[Occurrence]) -> None:
    """Overwrite the committed census with `hits`, discarding the old pins.

    Destructive by design and only reached via ``--write``: every previous
    pin is replaced, so a shape that regressed since the last regeneration is
    silently re-baselined. Read the rows the check named before running it.

    Args:
        hits: The occurrences to pin, as scanned from the current tree.
    """
    lines = [
        "# Timestamp-writer census (ADR-173 / TASK-32803.1). Every row emits a",
        "# shape other than the canonical YYYY-MM-DDTHH:MM:SS.mmmZ. Ratchet:",
        "# counts only shrink as TASK-32901 adopts Utils/timestamps.py.",
        "# Regenerate with: python scripts/check_timestamp_writers.py --write",
        "# module\tsymbol\tkind\tcount",
    ]
    for (module, symbol, kind), count in sorted(hits.items()):
        lines.append(f"{module}\t{symbol}\t{kind}\t{count}")
    CENSUS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Scan the package and compare it against the committed census.

    Returns:
        0 when clean (or when ``--write`` regenerated the census), 1 when a
        ``datetime.utcnow()`` site exists or a ratcheted shape is new or has
        grown. A malformed census row exits 2 from :func:`read_census`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="regenerate the census from the tree"
    )
    args = parser.parse_args()

    hits = scan_tree()

    if args.write:
        write_census(hits)
        total = sum(hits.values())
        print(f"wrote {CENSUS.relative_to(REPO_ROOT)}: {total} occurrence(s)")
        return 0

    census = read_census()

    # datetime.utcnow() is forbidden outright — it should never be censused.
    utcnow = sorted(k for k in hits if k[2] == KIND_UTCNOW)
    # Any ratcheted occurrence is new/grown when it exceeds its census count.
    grown = sorted(
        k
        for k, n in hits.items()
        if k[2] in RATCHETED_KINDS and n > census.get(k, 0)
    )

    totals = {
        kind: sum(n for k, n in hits.items() if k[2] == kind)
        for kind in RATCHETED_KINDS
    }
    print(
        f"timestamp writers: {len(utcnow)} datetime.utcnow() site(s), "
        + ", ".join(f"{totals[kind]} {kind}" for kind in RATCHETED_KINDS)
        + f" occurrence(s) ({sum(census.values())} pinned)."
    )

    if not (utcnow or grown):
        print("check_timestamp_writers: OK")
        return 0

    if utcnow:
        print(
            f"\nFAIL: {len(utcnow)} datetime.utcnow() call(s) — deprecated and "
            "naive; ADR-173 forbids it. Use utc_now_iso() (canonical string) or "
            "datetime.now(timezone.utc):"
        )
        for module, symbol, _ in utcnow:
            print(f"    {module}\t{symbol}")

    if grown:
        print(
            f"\nFAIL: {len(grown)} new/grown non-canonical timestamp write(s). "
            "ADR-173 mandates one stored shape, YYYY-MM-DDTHH:MM:SS.mmmZ: a "
            "naive now() stores LOCAL time as UTC, now(tz).isoformat() emits "
            "'+00:00' (which sorts BEFORE the same instant's 'Z' and is dropped "
            "by a >= cutoff), and a hand-rolled ISO strftime is a shape of its "
            "own. Produce the timestamp with "
            "Utils.timestamps.utc_now_iso()/to_utc_iso() instead:"
        )
        for module, symbol, kind in grown:
            print(f"    {module}\t{symbol}\t(now {hits[(module, symbol, kind)]}, "
                  f"pinned {census.get((module, symbol, kind), 0)})")
        print(
            f"  If a site is genuinely displaying LOCAL time (not storing a "
            f"timestamp), that is out of scope — but do not add it to "
            f"{CENSUS.name}; refactor it to be unambiguous."
        )

    return 1


if __name__ == "__main__":
    sys.exit(main())
