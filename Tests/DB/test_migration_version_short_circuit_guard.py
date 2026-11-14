# test_migration_version_short_circuit_guard.py
# Description: Regression coverage for task-19866 (no migration test may
#              open with a version assertion that switches off its body)
"""
A migration test that asserts ``db.schema_version == N`` (or any comparison
that stops holding when the schema advances past ``N``) BEFORE its
substantive assertions is not a test -- it is a switch. Once the schema
moves past ``N``, the equality fails and every column/index/row-survival
assertion below it stops running; the suite stays green because the file
still "passes", and the migration it was written to protect is silently no
longer checked by anything. TASK-19568 removed the then-known instances
and its reviewer found the same shape already latent in the newest
migration test file -- a check scoped to "the file we know about"
structurally cannot see the file added last week. This module is the guard
both the reviewer and Qodo proposed.

What is flagged, in every ``Tests/**/test_*migration*.py`` module:

1. **Hazard asserts**: an ``assert`` comparing a per-database version read
   (``X.schema_version``, ``X.get_schema_version()``, a
   ``PRAGMA user_version`` read, or a module helper that wraps one) against
   an integer literal with an operator that stops holding when the version
   advances (``==``, ``!=``, ``<``, ``<=``), positioned before a later
   substantive assertion in the same function body.
   ``>=``/``>`` floor checks are deliberately NOT flagged: a floor stays
   true as the version advances, so it cannot switch the body off. A
   dynamic comparison against the class's own ``_CURRENT_SCHEMA_VERSION``
   is likewise not a literal and self-adjusts.
2. **Skip guards**: an ``if`` over a version expression whose body calls
   ``pytest.skip`` (or the module aliases it), positioned before a later
   substantive assertion -- the same switch, expressed as a skip so the
   suite reports no failure at all.
3. **Helper evasion**: a call to a module-level helper whose body contains
   a hazard assert, positioned before a later substantive assertion -- the
   same switch, one frame away.

The escape hatch, documented per AC#5: a version assertion that is the
SUBJECT of the test remains possible. The distinguishing mechanism is
positional -- a version assertion is legitimate when no substantive
assertion follows it in the body (the test's whole purpose is that the
version reached/moved to that value), and deliberate era-pinned sanity
checks that must precede the state they enable can be allowlisted below,
each with a written reason.

The guard is structural (AST) rather than textual, so reformatting, line
wrapping or an intervening comment cannot evade it, and it reports every
offender, not just the first.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "Tests"

#: The migration-test modules the guard covers.
MIGRATION_TEST_GLOB = "test_*migration*.py"

#: Comparison operators that stop holding when the schema version ADVANCES
#: past the literal -- the "switch" operators. ``>=``/``>`` floors survive
#: a version bump and are not hazards.
_HAZARD_OPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE)

#: Deliberate, era-pinned asserts that must run BEFORE the state they
#: enable, each with its written reason (AC#5/AC#6). Keyed by
#: ``(posix path, function name, assert source text)`` so line shifts do
#: not invalidate entries -- the source text is the assertion's identity.
ALLOWLIST: dict[tuple[str, str, str], str] = {
    (
        "Tests/Evals/test_evals_db_v5_to_v6_migration.py",
        "test_opening_a_v5_database_drops_the_inert_version_columns",
        "assert SCHEMA_VERSION == 7",
    ): (
        "Deliberate suite pin, self-documented in the source: the suite "
        "tests the v5 -> v6 hop plus the v7 results index and MUST stop passing (and "
        "be updated) when EvalsDB's schema advances; the comment on the "
        "line says exactly this."
    ),
    (
        "Tests/DB/test_chachanotes_notes_organization_migration.py",
        "test_real_v57_reopen_backfills_stable_unique_uuid4_ids",
        "assert CharactersRAGDB._CURRENT_SCHEMA_VERSION == 58",
    ): (
        "Bootstrap-context pin: the assert verifies that the historical "
        "bootstrap (chachanotes_db_at_version(path, 58, ...)) has pinned the "
        "class constant, and it MUST run inside that with-block -- after the "
        "patch exits, the constant returns to the live version, so the "
        "assert cannot move below the later function-level assertions. It "
        "is stable under schema bumps (the bootstrap pins 58 regardless of "
        "the live version)."
    ),
    (
        "Tests/Scheduling/test_migrations.py",
        "test_v7_migration_dedupes_existing_duplicates_keeping_newest_by_updated_at",
        "assert db.get_schema_version() == 6",
    ): (
        "Mid-state rollback-target pin: the assert must run BEFORE seeding "
        "the duplicate rows (possible only below v7, where the UNIQUE index "
        "does not exist yet), so it cannot move below the substantive "
        "assertions. The suite tests the v6 -> v7 hop and is updated with "
        "the chain."
    ),
    (
        "Tests/Scheduling/test_migrations.py",
        "test_v7_migration_dedupe_handles_mixed_offset_updated_at",
        "assert db.get_schema_version() == 6",
    ): (
        "Same mid-state rollback-target pin as its sibling dedupe test: "
        "must precede the duplicate seeding it enables."
    ),
    (
        "Tests/Scheduling/test_migrations.py",
        "test_v7_migration_dedupe_keeps_the_true_newest_within_one_second",
        "assert db.get_schema_version() == 6",
    ): (
        "Same mid-state rollback-target pin as its sibling dedupe tests: "
        "must precede the duplicate seeding it enables."
    ),
}


# ---------------------------------------------------------------------------
# The scanner
# ---------------------------------------------------------------------------


def _pragma_version_readers(tree: ast.Module) -> set[str]:
    """Names of module-level helpers that read ``PRAGMA user_version``."""
    readers: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and any(
                isinstance(a, ast.Constant)
                and isinstance(a.value, str)
                and "user_version" in a.value
                for a in sub.args
            ):
                readers.add(node.name)
                break
    return readers


def _is_version_expression(node: ast.AST, pragma_readers: set[str]) -> bool:
    """Whether an expression reads a database schema version."""
    if isinstance(node, ast.Attribute) and node.attr == "schema_version":
        return True
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr in {
            "get_schema_version",
        }:
            return True
        if isinstance(node.func, ast.Name) and node.func.id in pragma_readers:
            return True
        return any(
            isinstance(a, ast.Constant)
            and isinstance(a.value, str)
            and "user_version" in a.value
            for a in node.args
        )
    if isinstance(node, (ast.Name, ast.Attribute)):
        name = node.id if isinstance(node, ast.Name) else node.attr
        return name == "SCHEMA_VERSION" or name.endswith("_SCHEMA_VERSION")
    return False


def _hazard_assert(
    node: ast.AST, pragma_readers: set[str]
) -> ast.Assert | None:
    """Return ``node`` if it is an assert that can switch off on a bump."""
    if not (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)):
        return None
    test = node.test
    sides = [test.left, *test.comparators]
    has_version = any(_is_version_expression(side, pragma_readers) for side in sides)
    has_int_literal = any(
        isinstance(side, ast.Constant) and isinstance(side.value, int)
        for side in sides
    )
    if (
        has_version
        and has_int_literal
        and any(isinstance(op, _HAZARD_OPS) for op in test.ops)
    ):
        return node
    return None


def _is_substantive_assert(node: ast.AST, pragma_readers: set[str]) -> bool:
    return isinstance(node, ast.Assert) and _hazard_assert(node, pragma_readers) is None


def _version_condition(node: ast.AST, pragma_readers: set[str]) -> bool:
    """Whether a comparison expression involves a version read."""
    if not isinstance(node, ast.Compare):
        return False
    sides = [node.left, *node.comparators]
    return any(_is_version_expression(side, pragma_readers) for side in sides)


def _calls_pytest_skip(branch: ast.AST) -> bool:
    for sub in ast.walk(branch):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
            if sub.func.attr == "skip":
                return True
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
            if "skip" in sub.func.id:
                return True
    return False


def scan_migration_tests(root: Path) -> list[str]:
    """Scan every migration test module and report each short-circuit.

    Args:
        root: The repository root (the ``Tests/`` tree beneath it is
            scanned for ``test_*migration*.py`` modules).

    Returns:
        A list of human-readable findings, one per offender, naming the
        file, the function, the line and the shape. Empty means green.
    """
    findings: list[str] = []
    for path in sorted((root / "Tests").rglob(MIGRATION_TEST_GLOB)):
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            findings.append(f"{path.relative_to(root)}: unparseable ({exc})")
            continue
        pragma_readers = _pragma_version_readers(tree)

        # Module-level helpers whose body carries a hazard assert: calling
        # one before the substantive assertions is the helper evasion.
        hazard_helpers: dict[str, ast.FunctionDef] = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and any(
                _hazard_assert(sub, pragma_readers) is not None
                for sub in ast.walk(node)
            )
        }

        relative = path.relative_to(root).as_posix()
        lines = source.splitlines()

        def _assert_text(lineno: int) -> str:
            """Normalized, comment-free source of the assertion's line."""
            return " ".join(
                re.sub(r"\s*#.*$", "", lines[lineno - 1]).split()
            )
        for function in [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]:
            substantive_lines = [
                sub.lineno
                for sub in ast.walk(function)
                if _is_substantive_assert(sub, pragma_readers)
            ]
            if not substantive_lines:
                continue

            def _has_later_substantive(lineno: int) -> bool:
                return any(line > lineno for line in substantive_lines)

            for sub in ast.walk(function):
                hazard = _hazard_assert(sub, pragma_readers)
                if hazard is not None and _has_later_substantive(hazard.lineno):
                    text = _assert_text(hazard.lineno)
                    key = (relative, function.name, text)
                    if key not in ALLOWLIST:
                        findings.append(
                            f"{relative}:{hazard.lineno} in {function.name}: "
                            f"version assertion before substantive assertions "
                            f"({text})"
                        )
                if (
                    isinstance(sub, ast.If)
                    and _version_condition(sub.test, pragma_readers)
                    and _calls_pytest_skip(sub)
                    and _has_later_substantive(sub.lineno)
                ):
                    findings.append(
                        f"{relative}:{sub.lineno} in {function.name}: "
                        "pytest.skip version guard before substantive assertions"
                    )
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id in hazard_helpers
                    and sub.func.id != function.name
                    and _has_later_substantive(sub.lineno)
                ):
                    helper = hazard_helpers[sub.func.id]
                    findings.append(
                        f"{relative}:{sub.lineno} in {function.name}: call to "
                        f"'{sub.func.id}' (a version-assert helper, defined at "
                        f"line {helper.lineno}) before substantive assertions"
                    )
    return findings


def test_no_migration_test_opens_with_a_version_short_circuit():
    """AC#1/AC#4: the guard over the real tree, reporting every offender."""
    findings = scan_migration_tests(REPO_ROOT)
    assert findings == [], (
        "These migration tests carry a schema-version assertion (or skip "
        "guard, or version-assert helper call) positioned BEFORE their "
        "substantive assertions -- a version bump switches the rest of "
        "each body off while the file still 'passes'. Move the version "
        "assertion after the substantive assertions, convert a "
        "'was fully migrated' pin to a >= floor, or allowlist it here "
        "with a written reason:\n  " + "\n  ".join(findings)
    )


def test_allowlisted_entries_still_exist_and_carry_reasons():
    """AC#6: every allowlist entry names a real assertion and a reason."""
    for (relative, function, text), reason in ALLOWLIST.items():
        assert reason.strip(), (relative, function, text)
        path = REPO_ROOT / relative
        assert path.exists(), relative
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function
        ]
        assert functions, f"{relative}: allowlisted function {function} is gone"
        lines = source.splitlines()
        found = any(
            " ".join(re.sub(r"\s*#.*$", "", lines[sub.lineno - 1]).split()) == text
            for fn in functions
            for sub in ast.walk(fn)
            if isinstance(sub, ast.Assert) and sub.lineno <= len(lines)
        )
        assert found, (
            f"{relative}: allowlisted assertion {text!r} in {function} no "
            "longer exists -- remove the stale allowlist entry"
        )


# ---------------------------------------------------------------------------
# The mutation evidence (AC#3): distinct evasion shapes, each caught
# ---------------------------------------------------------------------------


def _scan_source(source: str, tmp_path: Path) -> list[str]:
    """Write one synthetic migration module and scan it."""
    tests = tmp_path / "Tests" / "Migrations"
    tests.mkdir(parents=True)
    (tests / "test_synthetic_migration.py").write_text(source, encoding="utf-8")
    return scan_migration_tests(tmp_path)


def test_catches_a_literal_equality_short_circuit(tmp_path):
    """The original shape: ``assert db.schema_version == N`` up front."""
    source = (
        "def test_example(db):\n"
        "    assert db.schema_version == 44\n"
        "    assert db.get_columns('notes') == ['id', 'title']\n"
    )
    findings = _scan_source(source, tmp_path)
    assert any("version assertion before substantive" in f for f in findings), findings


def test_catches_a_leq_short_circuit(tmp_path):
    """AC#3 evasion shape: ``<=`` rather than ``==``."""
    source = (
        "def test_example(db):\n"
        "    assert db.schema_version <= 44\n"
        "    assert db.get_columns('notes') == ['id', 'title']\n"
    )
    findings = _scan_source(source, tmp_path)
    assert any("version assertion before substantive" in f for f in findings), findings


def test_catches_a_not_equal_short_circuit(tmp_path):
    """AC#3 evasion shape: ``!=`` rather than ``==``."""
    source = (
        "def test_example(db):\n"
        "    assert db.schema_version != 45\n"
        "    assert db.get_columns('notes') == ['id', 'title']\n"
    )
    findings = _scan_source(source, tmp_path)
    assert any("version assertion before substantive" in f for f in findings), findings


def test_catches_a_pytest_skip_guard(tmp_path):
    """AC#3 evasion shape: the switch expressed as a skip."""
    source = (
        "import pytest\n"
        "def test_example(db):\n"
        "    if db.schema_version != 44:\n"
        "        pytest.skip('requires v44')\n"
        "    assert db.get_columns('notes') == ['id', 'title']\n"
    )
    findings = _scan_source(source, tmp_path)
    assert any("pytest.skip version guard" in f for f in findings), findings


def test_catches_a_version_assert_helper_called_from_the_body(tmp_path):
    """AC#3 evasion shape: the switch hidden in a helper one frame away."""
    source = (
        "def _require_v44(db):\n"
        "    assert db.schema_version == 44\n"
        "def test_example(db):\n"
        "    _require_v44(db)\n"
        "    assert db.get_columns('notes') == ['id', 'title']\n"
    )
    findings = _scan_source(source, tmp_path)
    assert any("version-assert helper" in f for f in findings), findings


def test_catches_a_pragma_user_version_helper_read(tmp_path):
    """A helper that wraps ``PRAGMA user_version`` counts as a version
    read, so the switch cannot hide behind an indirection."""
    source = (
        "def _version(conn):\n"
        "    return conn.execute('PRAGMA user_version').fetchone()[0]\n"
        "def test_example(conn):\n"
        "    assert _version(conn) == 44\n"
        "    assert conn.tables() == ['notes']\n"
    )
    findings = _scan_source(source, tmp_path)
    assert any("version assertion before substantive" in f for f in findings), findings


def test_a_reformatted_short_circuit_is_still_caught(tmp_path):
    """AC#2: the guard is AST-based -- line wrapping cannot evade it."""
    source = (
        "def test_example(db):\n"
        "    assert (\n"
        "        db.schema_version\n"
        "        == 44\n"
        "    )\n"
        "    assert db.get_columns('notes') == ['id', 'title']\n"
    )
    findings = _scan_source(source, tmp_path)
    assert any("version assertion before substantive" in f for f in findings), findings


def test_reports_every_offender_not_just_the_first(tmp_path):
    """AC#4: two offenders, two findings."""
    source = (
        "def test_one(db):\n"
        "    assert db.schema_version == 44\n"
        "    assert db.cols() == []\n"
        "def test_two(db):\n"
        "    assert db.schema_version == 45\n"
        "    assert db.cols() == []\n"
    )
    findings = _scan_source(source, tmp_path)
    assert len(findings) == 2, findings


# ---------------------------------------------------------------------------
# What stays legal (AC#5)
# ---------------------------------------------------------------------------


def test_a_version_assertion_that_is_the_subject_stays_possible(tmp_path):
    """A test whose whole purpose is that the version advanced: no
    substantive assertion follows, so nothing can be switched off."""
    source = (
        "def test_migration_advances(db):\n"
        "    db.migrate()\n"
        "    assert db.schema_version == 45\n"
    )
    findings = _scan_source(source, tmp_path)
    assert findings == [], findings


def test_a_version_assertion_after_the_substantive_ones_stays_possible(tmp_path):
    """Trailing version checks run after everything they might have
    switched off -- the hazard is positional, not the assertion itself."""
    source = (
        "def test_example(db):\n"
        "    assert db.get_columns('notes') == ['id', 'title']\n"
        "    assert db.schema_version == 45\n"
    )
    findings = _scan_source(source, tmp_path)
    assert findings == [], findings


def test_a_floor_check_is_not_a_switch(tmp_path):
    """``>=``/``>`` stay true as the version advances, so they cannot turn
    a body off -- the future-proof form is legal wherever it fits."""
    source = (
        "def test_example(db):\n"
        "    assert db.schema_version >= 44\n"
        "    assert db.get_columns('notes') == ['id', 'title']\n"
    )
    findings = _scan_source(source, tmp_path)
    assert findings == [], findings


def test_a_dynamic_comparison_with_the_current_constant_is_not_a_switch(tmp_path):
    """``== SomeDB._CURRENT_SCHEMA_VERSION`` is not a literal: it
    self-adjusts on every bump and cannot go stale."""
    source = (
        "def test_example(db):\n"
        "    assert db.schema_version == db._CURRENT_SCHEMA_VERSION\n"
        "    assert db.get_columns('notes') == ['id', 'title']\n"
    )
    findings = _scan_source(source, tmp_path)
    assert findings == [], findings
