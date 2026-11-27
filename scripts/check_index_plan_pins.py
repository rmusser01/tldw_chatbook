#!/usr/bin/env python3
"""Guard: adding a database index must stop at a query-plan decision.

TASK-21126 and TASK-21593. None of this repo's SQLite databases ever runs
``ANALYZE`` -- grep the ``DB/`` modules and you will not find one -- so no
user's database has a ``sqlite_stat1``, and SQLite's planner works from
default row-count estimates rather than from your data. In that state an
index can be perfect for a query and still be *ignored*:

* TASK-21126 added ``(chunk_engine_version, media_id) WHERE deleted = 0``,
  a textbook covering index for the query it was written for. Measured with
  no stats: **118.8 ms without it, 120.2 ms with it.** Five megabytes of
  disk, a schema migration, and a green "the index exists" test, for 1%.
  The shipped index had to lead with a *redundant* ``deleted`` column to be
  chosen at all -- then 23.4 ms.
* TASK-21593 reproduced it across the whole media DB: the bare
  ``(last_modified DESC, id DESC) WHERE deleted = 0 AND is_trash = 0``
  index for the Library page query is likewise never picked, while
  ``(deleted, is_trash, last_modified DESC, id DESC)`` takes the same query
  from 19.5 ms to 0.07 ms.

Neither of those is visible to `SELECT name FROM sqlite_master WHERE
type='index'`, and neither is reliably visible to timing alone -- 118.8 ->
120.2 ms reads as noise. The only thing that says *why* is the
``EXPLAIN QUERY PLAN`` string, captured on a database with no
``sqlite_stat1``.

So this checker makes the decision mechanical rather than remembered. It
scans the schema-defining SQL for every ``CREATE INDEX`` name and requires
each one to appear in ``scripts/index_plan_pin_census.tsv`` as one of:

``plan-pinned``
    A test names this index inside a file that also runs
    ``EXPLAIN QUERY PLAN`` **and** asserts ``sqlite_stat1`` is absent. That
    combination is the evidence; either half alone is not. (A plan captured
    on a fixture that ran ``ANALYZE`` is not the plan production runs --
    TASK-15469 hit the same wall from the other side, where ``ANALYZE`` on a
    small dev database flipped a good plan back to a scan.)

``pre-convention``
    Predates this guard. No pin required, and none is being demanded
    retroactively -- the point is that the NEXT index cannot be added
    without someone typing one of these two words.

``dropped``
    A historical migration still creates it, but a later one drops it, so no
    plan is claimed (TASK-42544: v56's superseded response-revision index,
    dropped in ChaChaNotes v89). Only versioned migration scripts count
    (``<family>_v<from>_to_v<to>*.sql``), and each database family is its
    own sequence: index names repeat across databases (``idx_sync_log_ts``
    is in ChaChaNotes, Media and Prompts) and step numbers overlap. Every
    family that creates the name must drop it later in the same family, by
    ``<to>``, then by position when the drop and the create share one file;
    a shared step in two files has no order, so it fails. A drop in a
    family that never creates the name is ignored. No other source may
    still create it: a fresh-schema ``.py`` string or an unversioned
    ``.sql``. The recovery layer's frozen historical catalogs
    (``_RECOVERY_CATALOGS``, listed by path) quote shipped declarations and
    do not count. A drop-and-re-create, such as v81->v82's canvas sequence
    index, stays live and needs a plan pin.

A ``CREATE INDEX`` whose name has no census row fails the check, as does a
census row naming an index nothing creates any more. Nothing is auto-added:
the script prints the exact lines to paste, and choosing between the two
statuses is the decision this file exists to force.

**The expectation is not derived from the thing it guards** (TASK-19045's
rule): the index names come from the SQL text, and the census is only ever
read.

WHAT THIS CANNOT SEE, measured rather than assumed:

* An index whose NAME is not a literal in the DDL string --
  ``f"CREATE INDEX {name} ..."``. No such construction exists in ``DB/``
  today; the scan and a live ``sqlite_master`` agree exactly.
* Whether a ``plan-pinned`` test's assertion is any *good*. It checks that
  the index name, ``EXPLAIN QUERY PLAN`` and a ``sqlite_stat1`` absence
  assertion are in the same file -- not that they are in the same test.
  ``Tests/DB/test_media_db_schema_v9.py`` is the worked example of what a
  real pin looks like: assert the index name IS in the plan, assert
  ``TEMP B-TREE`` is NOT, and keep a negative control proving the shape you
  rejected is still not chosen.

  What it *does* now refuse to read as evidence is a mention that says the
  opposite. The first version accepted any ``idx_...`` token anywhere in a
  qualifying file, and measured against the tree that admitted two indexes
  the file only ever asserts are **absent** from a plan --
  ``idx_media_deleted`` and ``idx_keywords_deleted``, both named solely by
  a ``#`` comment quoting the pre-fix plan and by ``assert "..." not in
  plan``. Either census row could have been flipped to ``plan-pinned`` and
  passed CI on evidence that the planner does not choose the index. See
  ``_plan_evidence_strings``.
* Indexes created outside ``DB/`` (e.g. a vector store's own schema).

Usage:  python scripts/check_index_plan_pins.py
Exits 0 when the census covers the tree, 1 otherwise.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_DIR = REPO_ROOT / "tldw_chatbook" / "DB"
TESTS_DIR = REPO_ROOT / "Tests"
CENSUS = Path(__file__).resolve().parent / "index_plan_pin_census.tsv"

PLAN_PINNED = "plan-pinned"
PRE_CONVENTION = "pre-convention"
DROPPED = "dropped"
VALID_STATUSES = {PLAN_PINNED, PRE_CONVENTION, DROPPED}

_CREATE_INDEX = re.compile(
    r"CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
# ``(?!IF\b)``: ``f"DROP INDEX IF EXISTS {name}"`` names nothing; without
# the guard the optional group backtracks and records an index ``IF``.
_DROP_INDEX = re.compile(
    r"DROP\s+INDEX\s+(?:IF\s+EXISTS\s+)?(?!IF\b)([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
#: A versioned migration script: ``<family>_v<from>_to_v<to>[_<what>].sql``.
_MIGRATION_STEP = re.compile(r"^(.*?)_v(\d+)_to_v(\d+)(?=[_.])")
#: The recovery layer's frozen historical catalogs (not live schema), by
#: repository path. A new module is live until it is listed here on review.
_RECOVERY_CATALOGS = frozenset(
    f"tldw_chatbook/DB/{name}.py"
    for name in (
        "recovery_core",
        "recovery_core_schema",
        "recovery_dreams_guardian",
        "recovery_goal_schema",
        "recovery_operations",
        "recovery_response_rules",
        "recovery_review_boundary_schema",
        "recovery_sqlite",
        "recovery_workflow_schema",
    )
)
_INDEX_IDENTIFIER = re.compile(r"\b(?:idx|uq)_[A-Za-z0-9_]+\b")

#: The two halves that together make a plan assertion trustworthy.
_PLAN_CALL = "EXPLAIN QUERY PLAN"
_NO_STATS = "sqlite_stat1"

# One left-to-right pass: a ``/*----`` banner is a block comment, not a
# ``--`` line comment that eats its ``----*/`` close (and with it the code up
# to the next ``*/``; that hid three ChaChaNotes_DB.py creations).
_SQL_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)


def _strip_sql_comments(text: str) -> str:
    """Drop ``--`` and ``/* */`` comments before looking for DDL.

    Not cosmetic: ``ChaChaNotes_DB.py`` carries the literal SQL comment
    ``-- Create index for feedback queries`` inside a schema string, and the
    first draft of this checker duly reported an index named ``for``.
    """
    return _SQL_COMMENT.sub(" ", text)


def _sql_texts(path: Path) -> list[str]:
    """Every SQL-looking string literal in a Python module.

    Read through ``ast`` rather than as raw text so that prose in a ``#``
    comment -- for instance this task's own DDL comment, which quotes the
    rejected ``CREATE INDEX`` shapes with their measurements -- can never be
    mistaken for a declaration. That is not hypothetical: the comment beside
    ``_ACTIVE_MEDIA_INDEX_MIGRATION_SQL`` names an index that is
    deliberately NOT created.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _statements(pattern: re.Pattern[str]):
    """Yield ``(name, path, offset)`` for each ``pattern`` match under ``DB/``."""
    sources: list[tuple[Path, list[str]]] = []
    for py in sorted(DB_DIR.rglob("*.py")):
        sources.append((py, _sql_texts(py)))
    for sql in sorted(DB_DIR.rglob("*.sql")):
        sources.append((sql, [sql.read_text(encoding="utf-8")]))
    for path, texts in sources:
        for text in texts:
            for match in pattern.finditer(_strip_sql_comments(text)):
                yield match.group(1), path, match.start()


def declared_indexes(pattern: re.Pattern[str] = _CREATE_INDEX) -> dict[str, set[str]]:
    """Find every index the schema sources under ``DB/`` create.

    Args:
        pattern: The statement to look for; ``_DROP_INDEX`` finds the drops.

    Returns:
        Index name -> the repository-relative source paths whose SQL
        declares it. A name created by more than one file (a fresh-schema
        string and a migration step, typically) maps to both.
    """
    found: dict[str, set[str]] = {}
    for name, path, _offset in _statements(pattern):
        found.setdefault(name, set()).add(path.relative_to(REPO_ROOT).as_posix())
    return found


def _step(path: Path) -> tuple[str, int] | None:
    """``(family, version it ends at)`` for a migration script, else None."""
    match = _MIGRATION_STEP.match(path.name)
    return (match[1], int(match[3])) if path.suffix == ".sql" and match else None


def undropped_rows(census: dict[str, tuple[str, str]]) -> list[tuple[str, str]]:
    """Check every ``dropped`` census row against the migration order.

    Args:
        census: The parsed census.

    Returns:
        ``(name, reason)`` for each ``dropped`` row whose index is still live:
        a non-recovery source outside the versioned steps still creates it,
        no versioned step creates it, or some family that creates it does not
        drop it after its last create in that same family.
    """
    statements: dict[str, list[tuple[str, tuple[str, int] | None, str, int]]] = {}
    for kind, pattern in (("create", _CREATE_INDEX), ("drop", _DROP_INDEX)):
        for name, path, offset in _statements(pattern):
            rel = path.relative_to(REPO_ROOT).as_posix()
            statements.setdefault(name, []).append((kind, _step(path), rel, offset))
    failures = []
    for name in sorted(n for n, (status, _note) in census.items() if status == DROPPED):
        rows = statements.get(name, [])
        live = sorted(
            {
                rel
                for kind, step, rel, _offset in rows
                if kind == "create" and step is None and rel not in _RECOVERY_CATALOGS
            }
        )
        families = sorted({step[0] for kind, step, _rel, _o in rows if kind == "create" and step})
        if live:
            failures.append((name, "still created by " + ", ".join(live)))
        elif not families:
            failures.append((name, "no versioned migration step creates it"))
        for family in families if not live else ():
            versioned = [(kind, step[1], rel, offset) for kind, step, rel, offset in rows if step and step[0] == family]
            last_create = max(v for kind, v, _r, _o in versioned if kind == "create")
            drops = [v for kind, v, _r, _o in versioned if kind == "drop"]
            if not drops or max(drops) < last_create:
                failures.append((name, f"{family}: no step drops it after v{last_create}"))
            elif max(drops) == last_create:
                same = [(kind, rel, offset) for kind, v, rel, offset in versioned if v == last_create]
                if len({rel for _k, rel, _o in same}) > 1:
                    failures.append((name, f"{family}: v{last_create} creates and drops it in different files"))
                elif max(o for k, _r, o in same if k == "drop") < max(o for k, _r, o in same if k == "create"):
                    failures.append((name, f"{family}: v{last_create} creates it again after its drop"))
    return failures


def read_census() -> dict[str, tuple[str, str]]:
    """Parse ``index_plan_pin_census.tsv``.

    Blank lines and ``#`` comments are skipped; the note column is optional.

    Returns:
        Index name -> ``(status, note)``, where status is one of
        ``VALID_STATUSES`` and note is ``""`` when the row omits it.

    Raises:
        SystemExit: With code 1, after printing the offending line, when the
            census file is missing, a row has fewer than two TAB-separated
            columns, a row's status is outside ``VALID_STATUSES``, or an
            index name appears twice. Malformed input is a hard stop rather
            than a skipped row: a census that silently drops what it cannot
            parse is a guard that reports success for rows nobody read.
    """
    if not CENSUS.exists():
        print(f"FAIL: census file missing: {CENSUS.relative_to(REPO_ROOT)}")
        raise SystemExit(1)
    rows: dict[str, tuple[str, str]] = {}
    for lineno, raw in enumerate(CENSUS.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            print(
                f"FAIL: {CENSUS.name}:{lineno}: expected TAB-separated "
                f"<name>\\t<status>[\\t<note>], got: {line!r}"
            )
            raise SystemExit(1)
        name, status = parts[0].strip(), parts[1].strip()
        note = parts[2].strip() if len(parts) > 2 else ""
        if status not in VALID_STATUSES:
            print(
                f"FAIL: {CENSUS.name}:{lineno}: status must be one of "
                f"{sorted(VALID_STATUSES)}, got {status!r}"
            )
            raise SystemExit(1)
        if name in rows:
            print(f"FAIL: {CENSUS.name}:{lineno}: duplicate entry for {name!r}")
            raise SystemExit(1)
        rows[name] = (status, note)
    return rows


def _plan_evidence_strings(tree: ast.AST) -> list[str]:
    """String literals in a test module that can carry POSITIVE plan evidence.

    Two kinds of occurrence are dropped, because both are a claim that an
    index is *not* what the planner uses:

    * **docstrings and ``#`` comments** -- prose *about* an index, not an
      assertion *on* one. ``test_media_db_schema_v9.py`` names
      ``idx_media_deleted`` in a docstring and in two comments quoting the
      pre-fix plan it replaced;
    * the **member operand of a ``not in`` comparison** --
      ``assert "idx_media_deleted" not in plan`` is evidence the planner
      does NOT choose it, and counting it as a pin inverts the guard.

    A name survives if it appears even once outside those positions, so a
    file that asserts an index IS in one plan and is NOT in another still
    pins it.

    Args:
        tree: The parsed module.

    Returns:
        Every qualifying string literal, in no particular order.
    """
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstrings.add(id(body[0].value))

    negated: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        for offset, operator in enumerate(node.ops):
            member = operands[offset]
            if isinstance(operator, ast.NotIn) and isinstance(member, ast.Constant):
                negated.add(id(member))

    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and id(node) not in negated
    ]


def plan_pinning_files() -> dict[str, set[str]]:
    """Find the test files that plan-pin each index, with stats absent.

    A file qualifies only if it contains both ``EXPLAIN QUERY PLAN`` and a
    ``sqlite_stat1`` mention; within a qualifying file an index counts as
    pinned only where ``_plan_evidence_strings`` admits the name.

    Returns:
        Index name -> the repository-relative test files that pin it. Names
        that no qualifying file mentions positively are simply absent.
    """
    pins: dict[str, set[str]] = {}
    if not TESTS_DIR.exists():
        return pins
    for path in sorted(TESTS_DIR.rglob("test_*.py")):
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if _PLAN_CALL not in source or _NO_STATS not in source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        for text in _plan_evidence_strings(tree):
            stripped = _strip_sql_comments(text)
            names = set(_CREATE_INDEX.findall(stripped)) | set(
                _INDEX_IDENTIFIER.findall(text)
            )
            # Literal index names need not use the conventional idx_/uq_ prefix.
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
                names.add(text)
            for name in names:
                pins.setdefault(name, set()).add(rel)
    return pins


def main() -> int:
    """Check the census against the tree and report every discrepancy.

    Four failure classes are reported together rather than short-circuiting
    on the first: an index created but absent from the census, a census row
    naming an index nothing creates any more, a row recorded
    ``plan-pinned`` that no test file positively pins, and a row recorded
    ``dropped`` whose index is still live (see ``undropped_rows``).

    Returns:
        0 when the census covers the tree exactly, 1 otherwise.
    """
    declared = declared_indexes()
    census = read_census()
    pins = plan_pinning_files()

    missing = sorted(set(declared) - set(census))
    stale = sorted(set(census) - set(declared))
    unpinned = sorted(
        name
        for name, (status, _note) in census.items()
        if status == PLAN_PINNED and name in declared and name not in pins
    )
    undropped = undropped_rows(census)

    print(
        f"index plan pins: {len(declared)} index names declared under "
        f"tldw_chatbook/DB, {len(census)} census rows, "
        f"{sum(1 for s, _ in census.values() if s == PLAN_PINNED)} plan-pinned."
    )

    if not (missing or stale or unpinned or undropped):
        print("check_index_plan_pins: OK")
        return 0

    if missing:
        print(
            f"\nFAIL: {len(missing)} index(es) are created but absent from "
            f"{CENSUS.relative_to(REPO_ROOT)}.\n"
            "  Adding an index is a query-plan decision, not a schema detail: "
            "with no sqlite_stat1\n"
            "  the planner may simply never choose it (TASK-21126 shipped 5 MB "
            "of dead index that way).\n"
            "  Capture EXPLAIN QUERY PLAN on a corpus with sqlite_stat1 ABSENT, "
            "then paste ONE of:"
        )
        for name in missing:
            where = ", ".join(sorted(declared[name]))
            print(f"    {name}\t{PLAN_PINNED}\t<test file> ({where})")
            print(f"    {name}\t{PRE_CONVENTION}\t<why no plan pin> ({where})")

    if stale:
        print(
            f"\nFAIL: {len(stale)} census row(s) name an index nothing creates "
            "any more. Delete them:"
        )
        for name in stale:
            print(f"    {name}\t{census[name][0]}")

    if unpinned:
        print(
            f"\nFAIL: {len(unpinned)} index(es) are recorded as "
            f"{PLAN_PINNED!r} but no test file names them alongside both "
            f"{_PLAN_CALL!r} and a {_NO_STATS!r} assertion:"
        )
        for name in unpinned:
            print(f"    {name}   (recorded note: {census[name][1] or '-'})")
        print(
            "  Either add the pin (see Tests/DB/test_media_db_schema_v9.py for "
            f"the worked example) or change the row to {PRE_CONVENTION!r} with "
            "a reason."
        )

    if undropped:
        print(
            f"\nFAIL: {len(undropped)} index(es) are recorded as {DROPPED!r} but "
            "are still live (a dropped index needs a later versioned DROP INDEX "
            "and nothing that creates it again):"
        )
        for name, reason in undropped:
            print(f"    {name}   ({reason})")

    return 1


if __name__ == "__main__":
    sys.exit(main())
