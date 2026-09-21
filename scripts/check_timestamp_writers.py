#!/usr/bin/env python3
"""Guard the UTC-timestamp write contract (ADR-173 / TASK-32803.1 AC#2/#3).

ADR-173 makes ``tldw_chatbook/Utils/timestamps.py`` the one sanctioned producer
of stored UTC timestamps (canonical millisecond ``Z`` shape) and reader of every
shape on disk. This guard keeps new code from re-introducing the two write
idioms the review found were actively wrong:

1. ``datetime.utcnow()`` — deprecated in 3.12 and always **naive**; it is never
   correct here. Forbidden outright: fix it to ``utc_now_iso()`` (canonical
   string) or ``datetime.now(timezone.utc)`` (aware datetime).

2. Naive ``datetime.now().isoformat()`` — ``datetime.now()`` with no timezone
   argument returns *local* time, and ``.isoformat()`` then serialises that
   local instant as if it were the stored shape. That is exactly the
   "naive local time stored as UTC" latent bug ADR-173 describes. New
   occurrences are forbidden; the ones that already exist are pinned in
   ``scripts/timestamp_writer_census.tsv`` as a ratchet that only shrinks as
   TASK-32803.5 migrates them to the shared helper. (An *aware*
   ``datetime.now(timezone.utc).isoformat()`` does not match and is fine.)

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
import sys
import warnings
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOT = REPO_ROOT / "tldw_chatbook"
CENSUS = Path(__file__).resolve().parent / "timestamp_writer_census.tsv"

KIND_UTCNOW = "utcnow"
KIND_NAIVE_NOW_ISO = "naive_now_iso"

#: Files excluded from the scan: the shared helper (its aware datetime use is
#: the sanctioned one) and this guard itself.
_EXCLUDE = {
    PRODUCTION_ROOT / "Utils" / "timestamps.py",
}

Occurrence = tuple[str, str, str]  # (module, symbol, kind)


class _Visitor(ast.NodeVisitor):
    """Collect timestamp-writer occurrences with their enclosing qualname."""

    def __init__(self, module: str) -> None:
        self.module = module
        self._stack: list[str] = []
        self.hits: Counter[Occurrence] = Counter()

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
        # datetime.utcnow()  (attr == "utcnow", any receiver)
        if isinstance(func, ast.Attribute) and func.attr == "utcnow":
            self.hits[(self.module, self._symbol, KIND_UTCNOW)] += 1
        # <...>.now().isoformat()  with a NAIVE .now() (no tz arg)
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "isoformat"
            and isinstance(func.value, ast.Call)
            and isinstance(func.value.func, ast.Attribute)
            and func.value.func.attr == "now"
            and not func.value.args
            and not func.value.keywords
        ):
            self.hits[(self.module, self._symbol, KIND_NAIVE_NOW_ISO)] += 1
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
    lines = [
        "# Timestamp-writer census (ADR-173 / TASK-32803.1). Ratchet: counts",
        "# only shrink as TASK-32803.5 adopts Utils/timestamps.py. Regenerate",
        "# with: python scripts/check_timestamp_writers.py --write",
        "# module\tsymbol\tkind\tcount",
    ]
    for (module, symbol, kind), count in sorted(hits.items()):
        lines.append(f"{module}\t{symbol}\t{kind}\t{count}")
    CENSUS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
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
    # A naive_now_iso occurrence is new/grown when it exceeds its census count.
    grown = sorted(
        k
        for k, n in hits.items()
        if k[2] == KIND_NAIVE_NOW_ISO and n > census.get(k, 0)
    )

    naive_total = sum(n for k, n in hits.items() if k[2] == KIND_NAIVE_NOW_ISO)
    print(
        f"timestamp writers: {len(utcnow)} datetime.utcnow() site(s), "
        f"{naive_total} naive datetime.now().isoformat() occurrence(s) "
        f"({sum(census.values())} pinned)."
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
            f"\nFAIL: {len(grown)} new/grown naive datetime.now().isoformat() "
            "occurrence(s). datetime.now() with no tz returns LOCAL time and "
            "serialises it as if UTC (ADR-173). Produce the timestamp with "
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
