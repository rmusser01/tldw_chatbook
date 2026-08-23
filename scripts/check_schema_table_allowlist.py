#!/usr/bin/env python3
"""Guard: every table a ChaChaNotes migration CREATEs must be allowlisted.

TASK-20971. ``DB/sql_validation.py``'s ``VALID_TABLES['chachanotes']`` is a
hand-maintained allowlist; a table name that is missing from it is rejected by
``validate_table_name`` and every generic CRUD helper that routes through it
raises unconditionally. TASK-864 found nine of ~47 tables listed. TASK-19568
repaired the entry after it went stale again. TASK-19057 broke it **fourteen
and a half hours later** by adding two tables in a v44->v45 migration.

Why a second guard, when ``Tests/DB/test_sql_validation.py::
TestChachanotesValidTablesMatchesLiveSchema`` already pins this: that test is
correct and it did go red -- it just went red *after* the merge. The full suite
takes hours and has produced no CI verdict since 2026-06-26 (see
``.github/workflows/derived-artifacts.yml``), so in practice a migration author
sees only the tests they run, and which tests those are is decided by
geography. Measured on the TASK-19057 branch: the author updated
``Tests/ChaChaNotesDB/test_index_census.py`` -- an equally hand-maintained
literal -- because it lives beside the migration test they wrote and a
directory run turned it red; and they updated three schema-version pins under
``Tests/DB/`` that a grep for the version constant finds. ``VALID_TABLES`` is
reachable by neither route: it names no schema version, and nothing else in
``Tests/DB/test_sql_validation.py`` mentions the feature. Nothing connected
"I added a table" to "update the allowlist" except a human remembering.

This checker makes that connection mechanical, at authoring time, in
``scripts/preflight.sh`` (~ms, stdlib-only, no dependency install, no database
built).

**The expectation is NOT derived from the thing it guards.** TASK-19045
established the rule for the index census: a census that re-derives its
expectation from the artifact it checks is the identity function on exactly
the defect class it exists to catch, which is why ``VALID_TABLES`` is
deliberately not generated from the schema. The independent source of truth
used here is the **schema-defining SQL text itself**:

* ``tldw_chatbook/DB/migrations/chachanotes_*.sql`` -- read as raw text, and
* the SQL string literals inside ``tldw_chatbook/DB/ChaChaNotes_DB.py``
  (``_FULL_SCHEMA_SQL_V4`` and the inline ``_migrate_from_vX_to_vY`` scripts),
  extracted through ``ast`` so that prose in a ``#`` comment saying "the base
  CREATE TABLE as well" can never be mistaken for a table declaration.

Every ``CREATE TABLE <name>`` found there is a name the schema declares.
``VALID_TABLES`` is only ever *read* here, never used to decide what the
expected set is. Measured at the commit that fixed TASK-20971, against a real
fully-migrated ``CharactersRAGDB(":memory:")``: the static scan and the live
``sqlite_master`` set are **identical** -- 69 substantive tables, symmetric
difference empty -- and the static scan reproduces the drift the runtime pin
reported (``actor_pack_persona_intents``, ``actor_portable_identities``)
without opening a database.

WHAT THIS CANNOT SEE (measured, not assumed -- each case was run against a
fixture tree using this exact script). None of these shapes exists in the
``chachanotes`` sources today, which is why the static scan and a live
``sqlite_master`` agree exactly; they are the ways that could stop being true,
and they matter because an authoring-time guard is trusted whether or not it
covers a creation style:

* **DDL whose table name is not a literal.** ``"CREATE TABLE {} ...".format(n)``,
  ``f"CREATE TABLE {n} ..."`` and ``"CREATE TABLE " + n`` all hide the name from
  the regex. A dead-end fragment like the literal half of the last example
  (``"CREATE TABLE IF NOT EXISTS "``, with the name arriving later via ``+``)
  used to make the regex backtrack out of its optional ``IF NOT EXISTS``
  group and report a phantom table named ``IF``; ``CREATE_TABLE_RE`` now
  matches that clause as an all-or-nothing unit (TASK-20971 Qodo round), so
  this shape simply is not reported, like the rest of this bullet. Keep
  schema DDL a literal string.
* **A ``.sql`` file whose name does not match a glob in ``SCHEMAS``.**
  ``migrations/add_sync_fields_to_notes.sql`` is exactly that shape today (it
  is unreferenced, and its two tables happen to be declared inline in
  ``ChaChaNotes_DB.py`` as well, so nothing is currently missed).
* **A table created from a ``.py`` module not listed in ``SCHEMAS``**, or from
  DDL loaded from a file at runtime and ``executescript``-ed.
* **``DROP TABLE`` and ``ALTER TABLE ... RENAME TO``.** The scan is
  append-only: it reads every historical ``CREATE TABLE`` and has no notion of
  a table being removed or renamed later. Retiring a table would put this
  checker (which would demand the name stay allowlisted) in direct conflict
  with the runtime pin (which would demand it be removed) -- one of the two
  must be taught about the removal in the same commit, and until then there is
  no combination of ``VALID_TABLES`` contents that satisfies both.

The first three fail *safe* in the sense that the missed table simply is not
reported -- the runtime pin still catches it, i.e. the guard degrades to the
status quo rather than lying. The backstop for all of them is
``Tests/DB/test_schema_table_allowlist_guard.py::
test_shipped_static_scan_matches_the_live_chachanotes_schema``, which asserts
this scan equals a live fully-migrated database name for name; if a future
migration uses one of these shapes, that test is what says so.

Scope: ``chachanotes`` only, deliberately. The ``media`` and ``prompts``
entries of the same allowlist have drifted in both directions and are owned by
TASK-19867; wiring them in here before that task lands would make preflight
red for everyone on day one. Extending this checker is a matter of adding rows
to ``SCHEMAS`` below once those literals describe reality -- but the source
files for those two schemas are ``Client_Media_DB_v2.py`` /``Prompts_DB.py``,
whose rebuild-and-rename steps (``ChunkingTemplates_v7``) need an explicit
decision first, so do it under TASK-19867, not as a drive-by.

Usage:  ./scripts/check_schema_table_allowlist.py
Exit 0 when the allowlist matches the declared schema, 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Iterable, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SQL_VALIDATION = REPO_ROOT / "tldw_chatbook" / "DB" / "sql_validation.py"
MIGRATIONS_DIR = REPO_ROOT / "tldw_chatbook" / "DB" / "migrations"

#: ``CREATE [VIRTUAL|TEMP|TEMPORARY] TABLE [IF NOT EXISTS] [schema.]name``.
#: The modifier is captured rather than skipped so virtual/temp declarations can
#: be classified instead of silently swallowed by an over-broad pattern.
#:
#: The ``IF NOT EXISTS`` clause is matched as an all-or-nothing unit --
#: ``(?:IF\s+NOT\s+EXISTS\s+|(?!IF\s+NOT\s+EXISTS\b))`` -- rather than a bare
#: ``(?:...)?``. A plain optional group backtracks: when the clause is present
#: but the text ends right after it (the interpolated-DDL shape in the module
#: docstring's WHAT THIS CANNOT SEE section, e.g. the literal half of
#: ``"CREATE TABLE IF NOT EXISTS " + table_name``), the engine falls back to
#: matching zero-width and then reads the leftover word ``IF`` as the table
#: name. The alternation's second branch is a negative lookahead for the same
#: clause, so it only succeeds when the clause genuinely is not there --
#: closing the backtrack path without needing atomic groups (unavailable in
#: Python's ``re`` before 3.11; this script also runs under 3.9).
CREATE_TABLE_RE = re.compile(
    r"\bCREATE\s+(?:(?P<modifier>VIRTUAL|TEMP|TEMPORARY)\s+)?TABLE\s+"
    r"(?:IF\s+NOT\s+EXISTS\s+|(?!IF\s+NOT\s+EXISTS\b))"
    r"(?:[A-Za-z_][A-Za-z_0-9]*\s*\.\s*)?"
    r"[\"'`\[]?(?P<name>[A-Za-z_][A-Za-z_0-9]*)",
    re.IGNORECASE,
)


class SchemaSources(NamedTuple):
    """Where one database's schema is declared, and how to read it."""

    #: Key into ``VALID_TABLES``.
    key: str
    #: ``.sql`` files read as raw text.
    sql_globs: tuple[str, ...]
    #: ``.py`` files whose *string literals* are read (comments excluded).
    python_files: tuple[Path, ...]


SCHEMAS: tuple[SchemaSources, ...] = (
    SchemaSources(
        key="chachanotes",
        sql_globs=("chachanotes_*.sql",),
        python_files=(REPO_ROOT / "tldw_chatbook" / "DB" / "ChaChaNotes_DB.py",),
    ),
    # TASK-19867 owns adding "media" and "prompts" here; see the module
    # docstring for why they are not enabled yet.
)


class Declaration(NamedTuple):
    """One ``CREATE TABLE`` occurrence, with where it was found."""

    name: str
    origin: str


def _sql_fragments(path: Path) -> Iterable[tuple[str, str]]:
    """Yield ``(sql_text, origin_label)`` pairs for one schema source file.

    ``.sql`` files are schema top-to-bottom, so the whole file is one fragment.
    ``.py`` files are read through ``ast``: only string constants are returned,
    which is what makes the scan immune to prose. ``ChaChaNotes_DB.py`` and
    ``Client_Media_DB_v2.py`` both contain ``#`` comments that say
    "CREATE TABLE" in a sentence; a raw-text scan of those two files reports
    three phantom tables (``IF``, ``column``, ``as``). An AST scan reports zero.

    Args:
        path: A ``.sql`` or ``.py`` schema source file.

    Yields:
        ``(text, origin)`` where ``origin`` is a human-readable ``path:line``.

    Raises:
        OSError: If the file cannot be read.
        SyntaxError: If a ``.py`` source cannot be parsed.
    """
    text = path.read_text(encoding="utf-8")
    label = path.relative_to(REPO_ROOT).as_posix()
    if path.suffix != ".py":
        yield text, label
        return
    tree = ast.parse(text, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value, f"{label}:{node.lineno}"


def _strip_sql_comments(text: str) -> str:
    """Remove ``--`` line comments and ``/* ... */`` block comments from SQL.

    Without this, a migration comment like ``-- CREATE TABLE ghost(...)`` or
    a historical note inside a ``/* ... */`` block reads as a real
    declaration to ``CREATE_TABLE_RE`` and fails preflight/CI spuriously --
    the same phantom-table failure mode the ``.py`` side already avoids by
    scanning through ``ast`` instead of raw text (see ``_sql_fragments``).

    A comment opener inside a string literal is not a comment: SQL escapes a
    quote by doubling it (``'it''s'``), so this walks the text tracking
    whether it is inside a ``'``/``"``/`` ` `` -delimited literal and only
    treats ``--``/``/*`` as comment openers outside of one -- a stripper that
    mangles a real string literal (e.g. a ``DEFAULT`` value containing
    ``--``) would be worse than the bug it fixes.

    Args:
        text: Raw SQL text -- a whole ``.sql`` file, or one SQL string
            literal pulled out of a ``.py`` source by ``_sql_fragments``.

    Returns:
        ``text`` with every comment removed and every string literal intact.
        A line comment is replaced by a bare newline (so a line's contents
        after ``--`` are gone but the newline, and therefore any ``lineno``
        computed from later text, is unaffected); a block comment -- even one
        spanning multiple lines -- is removed entirely.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in ("'", '"', "`"):
            out.append(ch)
            i += 1
            while i < n:
                out.append(text[i])
                if text[i] == ch:
                    if i + 1 < n and text[i + 1] == ch:
                        # Doubled quote: an escaped quote character, still
                        # inside the string literal.
                        out.append(text[i + 1])
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            continue
        if text[i : i + 2] == "--":
            j = text.find("\n", i)
            if j == -1:
                i = n
            else:
                out.append("\n")
                i = j + 1
            continue
        if text[i : i + 2] == "/*":
            j = text.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _is_substantive(name: str) -> bool:
    """Whether a declared table is one ``VALID_TABLES`` is meant to allowlist.

    Mirrors the filter in ``Tests/DB/test_sql_validation.py``
    (``_live_chachanotes_table_names``) exactly, so the two guards cannot
    disagree about what counts:

    * ``sqlite_sequence`` -- SQLite's own AUTOINCREMENT bookkeeping.
    * anything containing ``_fts`` -- the FTS5 virtual tables and their
      ``_data``/``_idx``/``_docsize``/``_config`` shadows, written to only by
      triggers, never through a helper that calls ``validate_table_name``.
    """
    return name != "sqlite_sequence" and "_fts" not in name


def declared_tables(schema: SchemaSources) -> dict[str, list[str]]:
    """Every substantive table name the schema sources declare.

    Args:
        schema: The source registration to scan.

    Returns:
        Mapping of table name -> the origins that declare it, sorted.
    """
    files = [
        path
        for pattern in schema.sql_globs
        for path in sorted(MIGRATIONS_DIR.glob(pattern))
    ]
    files.extend(schema.python_files)

    found: dict[str, set[str]] = {}
    for path in files:
        for text, origin in _sql_fragments(path):
            for match in CREATE_TABLE_RE.finditer(_strip_sql_comments(text)):
                if match.group("modifier"):
                    # VIRTUAL (the FTS tables) and TEMP tables are not
                    # allowlist targets; _is_substantive drops the FTS names
                    # anyway, and a TEMP table has no persistent identity.
                    continue
                name = match.group("name")
                if _is_substantive(name):
                    found.setdefault(name, set()).add(origin)
    return {name: sorted(origins) for name, origins in found.items()}


def allowlisted_tables(key: str) -> set[str]:
    """Read one ``VALID_TABLES`` entry without importing the package.

    ``sql_validation.py`` is parsed, not imported. Importing it pulls in
    ``tldw_chatbook``, which loads config, touches the filesystem, and emits
    log lines -- none of which a derived-artifact checker should do, and all of
    which would break the stdlib-only, no-install contract the other four
    preflight checks keep.

    This is a *read* of the artifact under test. It is not the source of the
    expectation: the expected set comes from ``declared_tables``.

    Args:
        key: A ``VALID_TABLES`` key, e.g. ``"chachanotes"``.

    Returns:
        The allowlisted table names.

    Raises:
        SystemExit: If the literal cannot be located or evaluated -- a silent
            empty set here would turn the guard off.
    """
    tree = ast.parse(SQL_VALIDATION.read_text(encoding="utf-8"), filename=str(SQL_VALIDATION))
    for node in tree.body:
        # ``VALID_TABLES = {...}`` (ast.Assign, possibly multiple/chained
        # targets) and ``VALID_TABLES: dict[str, set[str]] = {...}``
        # (ast.AnnAssign, exactly one target) are both semantics-preserving
        # ways to write this literal; a refactor from the first to the second
        # must not make this guard unable to find it.
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if value is None:  # bare "VALID_TABLES: dict[...]" with no RHS yet
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "VALID_TABLES"
            for target in targets
        ):
            continue
        try:
            table_map = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:  # pragma: no cover - defensive
            raise SystemExit(
                f"::error::VALID_TABLES in {SQL_VALIDATION} is no longer a "
                f"plain literal and cannot be read statically ({exc}). This "
                f"guard reads it with ast.literal_eval on purpose (no import, "
                f"no database); either keep it a literal or update this script."
            ) from exc
        if key not in table_map:
            raise SystemExit(
                f"::error::VALID_TABLES has no {key!r} entry in {SQL_VALIDATION}."
            )
        return set(table_map[key])
    raise SystemExit(f"::error::no module-level VALID_TABLES assignment in {SQL_VALIDATION}.")


def _report(schema: SchemaSources, declared: dict[str, list[str]], allowed: set[str]) -> bool:
    """Print the verdict for one schema. Returns True when it passes."""
    unlisted = sorted(set(declared) - allowed)
    phantom = sorted(allowed - set(declared))

    if not unlisted and not phantom:
        print(
            f"{schema.key}: {len(declared)} declared tables, all present in "
            f"VALID_TABLES['{schema.key}']."
        )
        return True

    if unlisted:
        print(
            f"::error::{len(unlisted)} table(s) created by the {schema.key} "
            f"schema are missing from VALID_TABLES['{schema.key}'] in "
            f"tldw_chatbook/DB/sql_validation.py:"
        )
        for name in unlisted:
            print(f"  - {name}   (declared in {', '.join(declared[name])})")
        print()
        print(
            "Paste these lines into the VALID_TABLES["
            f'"{schema.key}"] set (it is sorted; keep it sorted):'
        )
        for name in unlisted:
            print(f'        "{name}",')
        print()
        print(
            "A table that is not allowlisted is rejected by "
            "validate_table_name(), so every generic CRUD helper that routes "
            "through it raises unconditionally for that table."
        )

    if phantom:
        print(
            f"::error::{len(phantom)} name(s) in VALID_TABLES['{schema.key}'] "
            "are not created by any migration or schema script:"
        )
        for name in phantom:
            print(f"  - {name}")
        print(
            "Remove them, or -- if the table is created somewhere this checker "
            "does not scan -- add that file to SCHEMAS in this script so the "
            "guard keeps covering it."
        )
    return False


def main(argv: list[str] | None = None) -> int:
    """Run the allowlist check for every schema in ``SCHEMAS`` (or a subset).

    Args:
        argv: Command-line arguments, excluding the program name (e.g.
            ``["--schema", "chachanotes"]``). ``None`` (the default) makes
            ``argparse`` read from ``sys.argv`` itself.

    Returns:
        ``0`` if every selected schema's declared tables match its
        ``VALID_TABLES`` entry exactly (see ``_report``); ``1`` if any
        schema has an unlisted table, a phantom allowlist entry, or an empty
        scan (a moved/renamed schema source, which would otherwise pass this
        guard vacuously).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schema",
        action="append",
        dest="schemas",
        choices=[schema.key for schema in SCHEMAS],
        help="limit the check to one VALID_TABLES key; repeatable",
    )
    args = parser.parse_args(argv)

    selected = [
        schema
        for schema in SCHEMAS
        if args.schemas is None or schema.key in args.schemas
    ]
    if not selected:  # pragma: no cover - argparse choices make this unreachable
        print("::error::no schema selected", file=sys.stderr)
        return 1

    ok = True
    for schema in selected:
        declared = declared_tables(schema)
        if not declared:
            print(
                f"::error::no CREATE TABLE statements found for {schema.key}. "
                "The schema sources moved; update SCHEMAS in "
                "scripts/check_schema_table_allowlist.py. (An empty scan would "
                "otherwise pass this guard vacuously.)"
            )
            ok = False
            continue
        ok = _report(schema, declared, allowlisted_tables(schema.key)) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
