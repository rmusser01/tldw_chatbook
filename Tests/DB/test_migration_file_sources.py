"""No decorative ``DB/migrations/*.sql`` files (task-19565).

Why this exists: at the 2026-08-21 lane's census, 12 of 26 migration ``.sql``
files were decorative — the migration executed an embedded Python constant,
the on-disk twin was never opened, alignment lived only in a comment, and 9
of them were pinned as shipped wheel content by the packaging test, which
made them look authoritative to anyone reading the repo. No test compared
any file to its constant, so they could (and in four checked cases, HAD)
silently diverged: ``chachanotes_v19_to_v20``'s twin carried the ``ADD
COLUMN`` the constant executed separately, and an editor changing the file
would have changed nothing at all.

The single rule this module enforces, chosen in task-19565 and applied per
module according to its architecture:

    every ``DB/migrations/*.sql`` is LOAD-BEARING — it is either
    (a) read at runtime by its owning module (ChaChaNotes file-backed
        steps, Workflows' f-string chain, Library_Collections,
        Subscriptions), or
    (b) a pinned reference/audit twin that a TEST reads, executes, or
        byte-compares (the four Workspace constants' twins, the six
        agent_runs standalone-migration audit records).

A file matching neither bucket is decorative and this test fails. The
``Keep this runner SQL aligned with`` twin-marker comment is likewise
banned: an alignment that lives only in a comment is not alignment.
"""

from __future__ import annotations

import re
from pathlib import Path

from tldw_chatbook.DB.Workflows_DB import WorkflowsDB

REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = REPO_ROOT / "tldw_chatbook" / "DB" / "migrations"
TESTS_DIR = REPO_ROOT / "Tests"
DB_PACKAGE_DIR = REPO_ROOT / "tldw_chatbook" / "DB"

#: Matches ``Path(__file__).parent / "migrations" / "<name>.sql"`` (and the
#: ``with_name`` variant) — the same source shape the packaging test's
#: ``RUNTIME_MIGRATION_READ`` derives wheel requirements from, kept
#: deliberately in sync with it.
RUNTIME_READ_RE = re.compile(r'"migrations"\s*/\s*"([^"\n]+\.sql)"')

#: Modules whose source may open migration files at runtime.
RUNTIME_READER_MODULES = (
    "ChaChaNotes_DB.py",
    "Library_Collections_DB.py",
    "Subscriptions_DB.py",
    "private_sqlite_helper_entry.py",
)


def _runtime_read_paths() -> dict[str, set[str]]:
    """Map each reader module to the migration filenames it opens at runtime."""
    readers: dict[str, set[str]] = {}
    for module_name in RUNTIME_READER_MODULES:
        source = (DB_PACKAGE_DIR / module_name).read_text(encoding="utf-8")
        readers[module_name] = set(RUNTIME_READ_RE.findall(source))
    # Workflows_DB builds names dynamically:
    # ``f"workflows_v{source_version}_to_v{source_version + 1}.sql"`` for
    # every step below its schema version. Derive them exactly as the
    # runtime does, from the class attribute, never by listing.
    readers["Workflows_DB.py"] = {
        f"workflows_v{version}_to_v{version + 1}.sql"
        for version in range(0, WorkflowsDB.SCHEMA_VERSION)
    }
    return readers


def _test_referenced_filenames() -> set[str]:
    """Every ``*.sql`` filename any test file names in a string literal."""
    referenced: set[str] = set()
    sql_literal_re = re.compile(r'"([A-Za-z0-9_./-]+\.sql)"')
    for test_path in TESTS_DIR.rglob("*.py"):
        for name in sql_literal_re.findall(test_path.read_text(encoding="utf-8")):
            referenced.add(Path(name).name)
    return referenced


class TestNoDecorativeMigrationFiles:
    """Every migrations/*.sql must be runtime-executed or test-pinned."""

    def test_every_migration_file_is_load_bearing(self):
        """A file no module reads and no test names is decorative: fail.

        Adding a new migration file? Either point the owning step at it (the
        ChaChaNotes file-backed pattern), or pin it with a test that reads,
        executes, or byte-compares it (the Workspace twin / agent_runs
        audit-record pattern). A third copy that merely sits in the wheel
        is exactly the defect class task-19565 closed.
        """
        on_disk = {path.name for path in MIGRATIONS_DIR.glob("*.sql")}
        assert on_disk, "the migrations directory disappeared?"
        runtime_read = set().union(*_runtime_read_paths().values())
        test_referenced = _test_referenced_filenames()
        decorative = sorted(on_disk - runtime_read - test_referenced)
        assert not decorative, (
            f"Decorative migration files — no production module reads them "
            f"and no test pins them, so they can silently diverge from what "
            f"the app actually executes (task-19565): {decorative}. Either "
            f"make the owning step execute the file, pin it with a test, or "
            f"delete it."
        )

    def test_every_runtime_read_file_exists(self):
        """A runtime read of a missing file is a wheel-death (task-19860)."""
        on_disk = {path.name for path in MIGRATIONS_DIR.glob("*.sql")}
        missing = sorted(
            (module, name)
            for module, names in _runtime_read_paths().items()
            for name in names
            if name not in on_disk
        )
        assert not missing, (
            f"DB modules read migration files that do not exist: {missing}. "
            f"A wheel built from this tree dies at migration time."
        )

    def test_chachanotes_files_are_all_runtime_executed(self):
        """ChaChaNotes migrations are file-as-source, with no exceptions.

        Since v26 the ChaChaNotes runner executes on-disk migrations;
        task-19565 converted the last constant-backed steps (v16->v31 era)
        and deleted the two twins whose steps carry Python recovery logic
        (v41->v42, v42->v43 proofs). A chachanotes_*.sql that is only
        test-referenced would reintroduce the decorative twin this task
        removed -- if a future step needs Python-owned logic, keep the
        logic in the step and do not ship an unread .sql twin for it.
        """
        on_disk = {
            path.name
            for path in MIGRATIONS_DIR.glob("*.sql")
            if path.name.startswith("chachanotes_")
        }
        chachanotes_reads = _runtime_read_paths()["ChaChaNotes_DB.py"]
        not_executed = sorted(on_disk - chachanotes_reads)
        assert not not_executed, (
            f"chachanotes_*.sql files exist that ChaChaNotes_DB.py never "
            f"opens: {not_executed}. The module's convention (task-19565) is "
            f"file-as-source with no decorative twins."
        )

    def test_no_twin_marker_comments_remain(self):
        """``Keep this runner SQL aligned with`` comments are banned.

        That comment marked a decorative twin kept aligned by hand -- the
        exact mechanism that failed (four diverged pairs measured at this
        task's start). Workspaces keep pinned twins, but their alignment is
        enforced by byte-comparison tests, not by a comment.
        """
        offenders = []
        for path in sorted(DB_PACKAGE_DIR.glob("*.py")):
            source = path.read_text(encoding="utf-8")
            if "Keep this runner SQL aligned with" in source:
                offenders.append(path.name)
        assert not offenders, (
            f"{offenders} still carry 'Keep this runner SQL aligned with' "
            f"comments. Alignment-by-comment is not alignment (task-19565): "
            f"execute the file, pin it with a comparison test, or delete it."
        )
