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
        Subscriptions, the recovery dreams/guardian schema policies), or
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

import pytest

from tldw_chatbook.DB.Workflows_DB import WorkflowsDB

REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = REPO_ROOT / "tldw_chatbook" / "DB" / "migrations"
TESTS_DIR = REPO_ROOT / "Tests"
DB_PACKAGE_DIR = REPO_ROOT / "tldw_chatbook" / "DB"

#: Matches every ``"migrations"`` access shape the DB package uses to open
#: a migration script (TASK-38601): the ``Path(__file__).parent /
#: "migrations"`` and ``Path(__file__).with_name("migrations")`` forms, the
#: ``/`` on the same line or continued onto the next, and quoted or
#: f-string filenames. A strict superset of the packaging test's
#: ``RUNTIME_MIGRATION_READ`` (which only needs the ChaChaNotes ``parent``
#: shape). An f-string hit is a TEMPLATE — it carries ``{`` braces and is
#: materialized against the on-disk directory by
#: ``_materialize_template`` instead of counting as a concrete filename
#: (a brace-bearing name would poison the existence check).
RUNTIME_READ_RE = re.compile(r'"migrations"\)?\s*/\s*f?"([^"\n]+\.sql)"')


def _reader_module_sources() -> dict[str, str]:
    """Every DB module that opens a ``migrations/*.sql``, discovered not listed.

    Scanning the whole package keeps the reader set honest as modules are
    added or renamed. The hand list this test shipped with (TASK-38601) was
    already stale: ``recovery_dreams_guardian.py`` and
    ``recovery_operations.py`` both read migrations and neither was on it,
    so five dreams steps, the guardian step, and the library_collections
    v4->v5 step were flagged decorative while being executed at runtime.
    """
    sources: dict[str, str] = {}
    for module_path in sorted(DB_PACKAGE_DIR.glob("*.py")):
        source = module_path.read_text(encoding="utf-8")
        if RUNTIME_READ_RE.search(source):
            sources[module_path.name] = source
    return sources


def _materialize_template(template: str, on_disk: set[str]) -> set[str]:
    """On-disk filenames an f-string read template can produce at runtime.

    Every interpolation in these reader loops is a schema version number
    (``f"dreams_v{version}_to_v{version + 1}.sql"``), so each brace group
    widens to ``\\d+`` — deliberately conservative: a template can only
    ever claim files whose interpolated parts are numeric.
    """
    parts = re.split(r"\{[^{}]*\}", template)
    pattern = re.compile(r"\d+".join(re.escape(part) for part in parts))
    return {name for name in on_disk if pattern.fullmatch(name)}


def _runtime_read_paths() -> dict[str, set[str]]:
    """Map each reader module to the migration filenames it opens at runtime."""
    on_disk = {path.name for path in MIGRATIONS_DIR.glob("*.sql")}
    readers: dict[str, set[str]] = {}
    for module_name, source in _reader_module_sources().items():
        literals: set[str] = set()
        templates: set[str] = set()
        for name in RUNTIME_READ_RE.findall(source):
            (templates if "{" in name else literals).add(name)
        names = set(literals)
        if module_name != "Workflows_DB.py":
            for template in templates:
                names |= _materialize_template(template, on_disk)
        readers[module_name] = names
    # Workflows_DB builds names dynamically:
    # ``f"workflows_v{source_version}_to_v{source_version + 1}.sql"`` for
    # every step below its schema version. Derived exactly as the runtime
    # does, from the class attribute — stricter than any-version template
    # matching, so this module is excluded from materialization above.
    readers.setdefault("Workflows_DB.py", set()).update(
        {
            f"workflows_v{version}_to_v{version + 1}.sql"
            for version in range(0, WorkflowsDB.SCHEMA_VERSION)
        }
    )
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

    def test_reader_shape_detection_covers_all_real_source_shapes(self):
        """Unit proof of the widened detection (TASK-38601).

        Every shape the DB package actually uses must be recognized — the
        audit's false positives all came from shapes the old regex could
        not see — and non-reads must not match.
        """
        literal_shapes = {
            r'Path(__file__).parent / "migrations" / "a.sql"': "a.sql",
            r'Path(__file__).with_name("migrations") / "b.sql"': "b.sql",
            '(Path(__file__).with_name("migrations")\n'
            '                     / "c.sql").read_text(encoding="utf-8")': "c.sql",
        }
        for shape, expected in literal_shapes.items():
            (captured,) = RUNTIME_READ_RE.findall(shape)
            assert captured == expected
        template_shape = (
            r'Path(__file__).with_name("migrations") '
            r'/ f"dreams_v{version}_to_v{version + 1}.sql"'
        )
        (template,) = RUNTIME_READ_RE.findall(template_shape)
        assert "{" in template  # recognized as a template, not a name
        assert _materialize_template(
            template, {"dreams_v1_to_v2.sql", "dreams_v9_to_v10.sql", "x.sql"}
        ) == {"dreams_v1_to_v2.sql", "dreams_v9_to_v10.sql"}
        for non_read in (
            'directory = "migrations"',  # no filename opened
            'other / "not_migrations" / "a.sql"',  # different directory
            'read_text(encoding="utf-8")',  # no path shape at all
        ):
            assert RUNTIME_READ_RE.findall(non_read) == []

    def test_guard_still_flags_a_genuinely_decorative_file(
        self, tmp_path, monkeypatch
    ):
        """End-to-end proof the widened detection did not blind the guard.

        Points the live-directory scan at a scratch directory holding one
        file no module reads and no test pins, then runs the real
        load-bearing assertion against it. The fake name is assembled from
        two fragments on purpose: this file's own literals feed
        ``_test_referenced_filenames``, and a contiguous quoted name would
        pin the fake as test-referenced and hollow out the proof.
        """
        fake_name = "chachanotes_v99_to_v100_" + "fake.sql"
        (tmp_path / fake_name).write_text("-- decorative\n", encoding="utf-8")
        monkeypatch.setattr(
            "Tests.DB.test_migration_file_sources.MIGRATIONS_DIR", tmp_path
        )
        with pytest.raises(AssertionError, match=re.escape(fake_name)):
            self.test_every_migration_file_is_load_bearing()
