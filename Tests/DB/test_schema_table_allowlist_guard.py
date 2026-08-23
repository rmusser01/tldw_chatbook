"""The authoring-time half of the ``VALID_TABLES`` guard bites (TASK-20971).

``scripts/check_schema_table_allowlist.py`` exists because the runtime pin in
``test_sql_validation.py`` reports only after a merge: it was repaired by
TASK-19568 at 2026-08-22 00:16 -0700 and broken again by TASK-19057 at 14:51
the same day. A guard whose own failure mode is "silently stops failing" is
worse than none, so this module pins the properties that make it a guard:

1. it fails when a migration creates a table the allowlist does not name;
2. it fails when the allowlist names a table nothing creates;
3. it refuses to pass vacuously when it can find no ``CREATE TABLE`` at all
   (the shape a moved-sources change would otherwise take); and
4. **its expectation is not derived from the allowlist it guards.** TASK-19045
   established the rule for the index census: a census that re-derives its
   expectation from the artifact it checks is the identity function on exactly
   the defect class it exists to catch. The test below proves independence
   directly -- it holds the schema SQL fixed and mutates only the allowlist,
   and the verdict changes. An identity check could not do that.

The shipped-tree assertion at the end is the belt to that braces: the static
scan must agree, name for name, with a real fully-migrated
``CharactersRAGDB(":memory:")``. Two independent oracles that agree are
evidence; one oracle asserted against itself is not.

PR #1983's Qodo round found three more traps in the checker itself, each
pinned below: raw ``.sql`` text was scanned without stripping SQL comments,
so a migration comment containing ``CREATE TABLE`` failed CI spuriously
(``test_sql_line_comment_does_not_produce_a_phantom_table``,
``test_sql_block_comment_does_not_produce_a_phantom_table``, plus a direct
unit test of the stripper proving it does not mangle a string literal that
happens to contain ``--`` or ``/*``); the same bare ``CREATE_TABLE_RE``
backtracked into reading the word ``IF`` as a phantom table name for
interpolated DDL (``test_if_not_exists_backtracking_does_not_produce_a_phantom_if_table``);
and ``allowlisted_tables()`` recognised only ``ast.Assign``, so annotating
``VALID_TABLES: dict[str, set[str]] = {...}`` -- a semantics-preserving
refactor -- broke the guard (``test_allowlisted_tables_reads_annotated_assignment``,
plus a companion pinning that a genuinely-missing assignment still fails
loudly rather than passing vacuously).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "scripts" / "check_schema_table_allowlist.py"


def _load_checker():
    """Import the checker by path (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location("_check_schema_table_allowlist", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_tree(tmp_path: Path, *, migration_sql: str, allowlisted: list[str]) -> object:
    """Build a miniature repo the checker can run against, and rebind it there.

    The checker resolves its paths from ``REPO_ROOT`` at import time, so a
    freshly loaded copy is repointed at ``tmp_path``. This keeps the mutation
    tests off the real tree entirely -- no Edit-and-restore dance, and no way
    for a failed test to leave the repo dirty.
    """
    migrations = tmp_path / "tldw_chatbook" / "DB" / "migrations"
    migrations.mkdir(parents=True)
    (migrations / "chachanotes_v1_to_v2_fixture.sql").write_text(migration_sql, encoding="utf-8")

    db_module = tmp_path / "tldw_chatbook" / "DB" / "ChaChaNotes_DB.py"
    # A Python source whose *comment* claims a table, to pin the AST-vs-text
    # distinction: a raw-text scan of the real ChaChaNotes_DB.py reports three
    # phantom tables from prose exactly like this line.
    db_module.write_text(
        textwrap.dedent(
            '''\
            # Prose that mentions CREATE TABLE phantom_from_a_comment on purpose.
            _SCHEMA = """
            CREATE TABLE IF NOT EXISTS from_python_literal(id INTEGER PRIMARY KEY);
            """
            '''
        ),
        encoding="utf-8",
    )

    body = "".join(f'        "{name}",\n' for name in allowlisted)
    (tmp_path / "tldw_chatbook" / "DB" / "sql_validation.py").write_text(
        f'VALID_TABLES = {{\n    "chachanotes": {{\n{body}    }},\n}}\n',
        encoding="utf-8",
    )

    module = _load_checker()
    module.REPO_ROOT = tmp_path
    module.MIGRATIONS_DIR = migrations
    module.SQL_VALIDATION = tmp_path / "tldw_chatbook" / "DB" / "sql_validation.py"
    module.SCHEMAS = (
        module.SchemaSources(
            key="chachanotes",
            sql_globs=("chachanotes_*.sql",),
            python_files=(db_module,),
        ),
    )
    return module


_FIXTURE_SQL = textwrap.dedent(
    """\
    CREATE TABLE IF NOT EXISTS from_sql_migration(id INTEGER PRIMARY KEY);
    CREATE VIRTUAL TABLE from_sql_migration_fts USING fts5(body);
    """
)
_FIXTURE_TABLES = ["from_python_literal", "from_sql_migration"]


def test_passes_when_every_created_table_is_allowlisted(tmp_path, capsys):
    module = _fake_tree(tmp_path, migration_sql=_FIXTURE_SQL, allowlisted=_FIXTURE_TABLES)
    assert module.main([]) == 0
    out = capsys.readouterr().out
    assert "2 declared tables" in out
    # The FTS5 virtual table is not an allowlist target and must not be
    # counted; the filter has to match the runtime pin's exactly.
    assert "from_sql_migration_fts" not in out


def test_bites_when_a_migration_adds_an_unlisted_table(tmp_path, capsys):
    """AC: adding a CREATE TABLE without touching the allowlist fails."""
    sql = _FIXTURE_SQL + "CREATE TABLE brand_new_table(id INTEGER PRIMARY KEY);\n"
    module = _fake_tree(tmp_path, migration_sql=sql, allowlisted=_FIXTURE_TABLES)
    assert module.main([]) == 1
    out = capsys.readouterr().out
    assert "brand_new_table" in out
    assert "chachanotes_v1_to_v2_fixture.sql" in out
    # It must tell the author what to paste, not merely that they are wrong.
    assert '        "brand_new_table",' in out


def test_bites_when_the_allowlist_names_a_table_nothing_creates(tmp_path, capsys):
    module = _fake_tree(
        tmp_path,
        migration_sql=_FIXTURE_SQL,
        allowlisted=_FIXTURE_TABLES + ["retired_table"],
    )
    assert module.main([]) == 1
    assert "retired_table" in capsys.readouterr().out


def test_verdict_is_not_derived_from_the_allowlist_it_guards(tmp_path, capsys):
    """The expectation comes from the SQL, so the allowlist cannot self-satisfy.

    Same schema sources in both halves; only ``VALID_TABLES`` moves. A checker
    that re-derived its expectation from ``VALID_TABLES`` would return the same
    verdict for both, which is the TASK-19045 identity-function failure.
    """
    sql = _FIXTURE_SQL + "CREATE TABLE brand_new_table(id INTEGER PRIMARY KEY);\n"

    incomplete = _fake_tree(tmp_path / "a", migration_sql=sql, allowlisted=_FIXTURE_TABLES)
    assert incomplete.main([]) == 1
    capsys.readouterr()

    complete = _fake_tree(
        tmp_path / "b",
        migration_sql=sql,
        allowlisted=_FIXTURE_TABLES + ["brand_new_table"],
    )
    assert complete.main([]) == 0


def test_comment_prose_is_never_read_as_a_table_declaration(tmp_path, capsys):
    """The .py side is scanned via ast, so ``# ... CREATE TABLE x`` is inert.

    Without this, a raw-text scan of the real ChaChaNotes_DB.py /
    Client_Media_DB_v2.py reports the phantom tables ``IF``, ``column`` and
    ``as`` -- and a guard that reports phantoms gets muted.
    """
    module = _fake_tree(tmp_path, migration_sql=_FIXTURE_SQL, allowlisted=_FIXTURE_TABLES)
    assert module.main([]) == 0
    assert "phantom_from_a_comment" not in capsys.readouterr().out


def test_sql_line_comment_does_not_produce_a_phantom_table(tmp_path):
    """AC: a ``-- CREATE TABLE ghost(...)`` comment is not a declaration.

    Unlike the ``.py`` side (immune via ``ast``, see the test above), the
    ``.sql`` side reads raw file text, so before Qodo's round-2 review this
    regex-scanned comment prose exactly like a real ``CREATE TABLE`` and
    failed CI/preflight spuriously.
    """
    sql = (
        "-- CREATE TABLE ghost_table(id INTEGER PRIMARY KEY);\n"
        "CREATE TABLE real_table(id INTEGER PRIMARY KEY);\n"
    )
    module = _fake_tree(
        tmp_path, migration_sql=sql, allowlisted=["real_table", "from_python_literal"]
    )
    declared = module.declared_tables(module.SCHEMAS[0])
    assert "ghost_table" not in declared
    assert "real_table" in declared


def test_sql_block_comment_does_not_produce_a_phantom_table(tmp_path):
    """AC: a ``/* ... */`` comment -- including a multi-line one -- is inert too."""
    sql = (
        "/* legacy note:\n"
        "CREATE TABLE ghost_table(id INTEGER PRIMARY KEY);\n"
        "kept for history */\n"
        "CREATE TABLE real_table(id INTEGER PRIMARY KEY);\n"
    )
    module = _fake_tree(
        tmp_path, migration_sql=sql, allowlisted=["real_table", "from_python_literal"]
    )
    declared = module.declared_tables(module.SCHEMAS[0])
    assert "ghost_table" not in declared
    assert "real_table" in declared


def test_strip_sql_comments_preserves_string_literals():
    """The stripper must not treat ``--``/``/*`` inside a string as a comment opener.

    A comment-stripper that mangles real SQL (e.g. eating a DEFAULT value
    that happens to contain ``--``) would be worse than the phantom-table bug
    it fixes.
    """
    module = _load_checker()
    text = (
        "CREATE TABLE t (\n"
        "    note TEXT DEFAULT 'a -- not a comment',\n"
        "    tag TEXT DEFAULT '/* also not a comment */'\n"
        ");\n"
        "-- real comment CREATE TABLE ghost(id INTEGER);\n"
    )
    stripped = module._strip_sql_comments(text)
    assert "a -- not a comment" in stripped
    assert "/* also not a comment */" in stripped
    assert "ghost" not in stripped


def test_strip_sql_comments_handles_escaped_quotes_in_strings():
    """SQL escapes a quote by doubling it; that doubled quote must not end the string early."""
    module = _load_checker()
    text = "INSERT INTO t VALUES ('it''s -- not a comment');\n-- actual comment\n"
    stripped = module._strip_sql_comments(text)
    assert "it''s -- not a comment" in stripped
    assert "actual comment" not in stripped


def test_if_not_exists_backtracking_does_not_produce_a_phantom_if_table(tmp_path):
    """Interpolated DDL must not surface a phantom table literally named ``IF``.

    ``sql = "CREATE TABLE IF NOT EXISTS " + table_name + " (...)"`` is one of
    the documented WHAT THIS CANNOT SEE shapes: the real table name is hidden
    behind string concatenation, so the checker is allowed to simply miss it
    (fail safe, like the rest of that list). Before Qodo's round-2 review, the
    bare optional ``(?:IF\\s+NOT\\s+EXISTS\\s+)?`` group backtracked out of a
    dead-end match and read the leftover word ``IF`` as the table name
    instead -- an active false positive, not a miss.
    """
    module = _fake_tree(tmp_path, migration_sql=_FIXTURE_SQL, allowlisted=_FIXTURE_TABLES)
    db_module = module.SCHEMAS[0].python_files[0]
    db_module.write_text(
        'sql = "CREATE TABLE IF NOT EXISTS " + table_name + " (id INTEGER PRIMARY KEY)"\n',
        encoding="utf-8",
    )
    declared = module.declared_tables(module.SCHEMAS[0])
    assert "IF" not in declared


def test_allowlisted_tables_reads_annotated_assignment(tmp_path):
    """A semantics-preserving refactor to an annotated assignment must not break the guard.

    ``allowlisted_tables()`` used to filter on ``ast.Assign`` only; if
    ``sql_validation.py`` were ever refactored to
    ``VALID_TABLES: dict[str, set[str]] = {...}`` (``ast.AnnAssign``), the
    checker could no longer find the literal at all.
    """
    module = _load_checker()
    module.SQL_VALIDATION = tmp_path / "sql_validation.py"
    module.SQL_VALIDATION.write_text(
        'VALID_TABLES: dict[str, set[str]] = {\n    "chachanotes": {"a", "b"},\n}\n',
        encoding="utf-8",
    )
    assert module.allowlisted_tables("chachanotes") == {"a", "b"}


def test_allowlisted_tables_still_reads_the_plain_assignment(tmp_path):
    """The original, un-annotated form (``ast.Assign``) keeps working."""
    module = _load_checker()
    module.SQL_VALIDATION = tmp_path / "sql_validation.py"
    module.SQL_VALIDATION.write_text(
        'VALID_TABLES = {\n    "chachanotes": {"a", "b"},\n}\n',
        encoding="utf-8",
    )
    assert module.allowlisted_tables("chachanotes") == {"a", "b"}


def test_allowlisted_tables_still_fails_loudly_when_the_assignment_is_gone(tmp_path):
    """Accepting AnnAssign must not weaken the loud-failure case for a real 'it's gone'.

    A silent empty set here would turn the guard off, so a file with no
    ``VALID_TABLES`` assignment at all -- annotated or not -- must still
    raise ``SystemExit``, exactly as it did before this fix.
    """
    module = _load_checker()
    module.SQL_VALIDATION = tmp_path / "sql_validation.py"
    module.SQL_VALIDATION.write_text("SOMETHING_ELSE = 1\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        module.allowlisted_tables("chachanotes")


def test_empty_scan_fails_instead_of_passing_vacuously(tmp_path, capsys):
    """Moving the schema sources must break the guard loudly, not silently."""
    module = _fake_tree(tmp_path, migration_sql="-- nothing here\n", allowlisted=[])
    module.SCHEMAS[0].python_files[0].write_text("# no SQL at all\n", encoding="utf-8")
    assert module.main([]) == 1
    assert "no CREATE TABLE statements found" in capsys.readouterr().out


def test_shipped_static_scan_matches_the_live_chachanotes_schema():
    """Both oracles, on the real tree, must name the same tables.

    This is what licenses the fast static checker to stand in for the runtime
    pin at authoring time. If a future migration creates a table somewhere the
    scanner does not look, this fails here rather than silently narrowing the
    guard.
    """
    from Tests.DB.test_sql_validation import _live_chachanotes_table_names

    module = _load_checker()
    (schema,) = module.SCHEMAS
    declared = set(module.declared_tables(schema))
    live = _live_chachanotes_table_names()
    assert declared == live, (
        "static CREATE TABLE scan and the live schema disagree: "
        f"only-in-scan={sorted(declared - live)}, only-in-live={sorted(live - declared)}"
    )


def test_checker_is_stdlib_only_and_never_imports_the_package():
    """The contract every preflight check keeps, asserted two ways.

    Statically: no import reaches ``tldw_chatbook``. Importing it would load
    config, touch the filesystem, emit log lines, and -- fatally for a
    derived-artifact checker -- make the guard depend on the package being
    installed. It also matters for independence: the allowlist is read by
    parsing ``sql_validation.py``, never by importing it.

    Dynamically: the script still exits 0 under ``-I -S``, which suppresses
    site-packages entirely, so nothing outside the stdlib is in play.
    """
    import ast

    tree = ast.parse(CHECKER.read_text(encoding="utf-8"), filename=str(CHECKER))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert "tldw_chatbook" not in imported, imported
    assert imported <= set(sys.stdlib_module_names), imported - set(sys.stdlib_module_names)

    completed = subprocess.run(
        [sys.executable, "-I", "-S", str(CHECKER)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "all present in VALID_TABLES['chachanotes']" in completed.stdout
