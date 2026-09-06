"""Cover ``scripts/check_index_plan_pins.py`` -- the index query-plan guard.

The checker is wired into ``scripts/preflight.sh`` and the required CI job,
which exercises exactly one path: the current tree, passing. That says
nothing about what it does with a missing row, a stale row, a malformed
census, or -- the one that matters -- a row recorded ``plan-pinned`` whose
"evidence" is a test asserting the planner does **not** choose the index.

That last case was real. The first version of ``plan_pinning_files``
accepted any ``idx_...`` token anywhere in a qualifying file, so measured
against the shipped tree it reported ``idx_media_deleted`` and
``idx_keywords_deleted`` as plan-pinned; both are named in
``Tests/DB/test_media_db_schema_v9.py`` only by a ``#`` comment quoting the
pre-fix plan and by ``assert "..." not in plan``. Flipping either census
row to ``plan-pinned`` would have passed CI on evidence of the opposite of
what the status claims. ``test_a_negative_assertion_is_not_a_pin`` and
``test_the_real_tree_does_not_pin_an_index_it_asserts_is_absent`` are the
two halves of that regression: one synthetic, one against the real files.

**Naming discipline for this file.** It contains ``EXPLAIN QUERY PLAN`` and
``sqlite_stat1``, so the checker treats it as a qualifying file like any
other -- the first draft of these tests used real index names in its
fixtures and duly made itself pin ``idx_media_deleted``, which is the
defect under test. Every synthetic index here is therefore ``idx_zz_*``,
and a real index name may appear only in a docstring, a comment, or the
member side of a ``not in``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_index_plan_pins.py"

PLAN_HEADER = '''
"""A qualifying test module: it runs EXPLAIN QUERY PLAN with no stats."""

def _plan(conn, sql):
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name='sqlite_stat1'"
    ).fetchone() is None
    return " | ".join(r[0] for r in conn.execute("EXPLAIN QUERY PLAN " + sql))
'''


@pytest.fixture()
def checker() -> ModuleType:
    """Load the script as a module (it is not importable by name)."""
    spec = importlib.util.spec_from_file_location("_check_index_plan_pins", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def repo(tmp_path: Path, checker: ModuleType, monkeypatch: pytest.MonkeyPatch):
    """Point the checker at a throwaway repository layout.

    Returns a helper that writes a DB source file, a test file and a census
    and then runs ``main()``, so each test states a whole scenario.
    """
    db_dir = tmp_path / "tldw_chatbook" / "DB"
    tests_dir = tmp_path / "Tests"
    db_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)
    census = tmp_path / "index_plan_pin_census.tsv"

    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "DB_DIR", db_dir)
    monkeypatch.setattr(checker, "TESTS_DIR", tests_dir)
    monkeypatch.setattr(checker, "CENSUS", census)

    class Repo:
        module = checker
        root = tmp_path
        db = db_dir
        tests = tests_dir
        census_path = census

        @staticmethod
        def write_db(name: str, body: str) -> None:
            (db_dir / name).write_text(body, encoding="utf-8")

        @staticmethod
        def write_test(name: str, body: str) -> None:
            (tests_dir / name).write_text(body, encoding="utf-8")

        @staticmethod
        def write_census(*rows: str) -> None:
            census.write_text(
                "# a census\n" + "\n".join(rows) + "\n", encoding="utf-8"
            )

    return Repo


# ---------------------------------------------------------------------------
# declared_indexes: what counts as a declaration
# ---------------------------------------------------------------------------
def test_a_sql_comment_inside_a_schema_string_is_not_an_index(repo):
    """The first draft of this checker reported an index named ``for``."""
    repo.write_db(
        "Schema_DB.py",
        'SCHEMA = """\n'
        "-- Create index for feedback queries\n"
        "CREATE INDEX idx_zz_feedback ON messages(feedback);\n"
        '"""\n',
    )
    assert set(repo.module.declared_indexes()) == {"idx_zz_feedback"}


def test_a_python_comment_naming_a_rejected_shape_is_not_a_declaration(repo):
    """``_ACTIVE_MEDIA_INDEX_MIGRATION_SQL`` carries exactly this comment."""
    repo.write_db(
        "Media_DB.py",
        "# Measured and REJECTED: CREATE INDEX idx_zz_rejected ON Media(x)\n"
        'MIGRATION = "CREATE INDEX idx_zz_recent ON Media(deleted, id)"\n',
    )
    assert set(repo.module.declared_indexes()) == {"idx_zz_recent"}


def test_a_sql_file_declares_its_indexes(repo):
    repo.write_db("step.sql", "CREATE UNIQUE INDEX IF NOT EXISTS uq_thing ON t(a);")
    declared = repo.module.declared_indexes()
    assert set(declared) == {"uq_thing"}
    assert declared["uq_thing"] == {"tldw_chatbook/DB/step.sql"}


# ---------------------------------------------------------------------------
# read_census: malformed input is a hard stop, not a silent skip
# ---------------------------------------------------------------------------
def test_a_missing_census_file_exits_nonzero(repo):
    with pytest.raises(SystemExit) as exc:
        repo.module.read_census()
    assert exc.value.code == 1


@pytest.mark.parametrize(
    "row",
    [
        "idx_zz_lonely",  # no status column
        "idx_zz_bad\tprobably-fine",  # status outside the vocabulary
    ],
)
def test_a_malformed_census_row_exits_nonzero(repo, row):
    repo.write_census(row)
    with pytest.raises(SystemExit) as exc:
        repo.module.read_census()
    assert exc.value.code == 1


def test_a_duplicated_census_row_exits_nonzero(repo):
    repo.write_census("idx_zz_a\tpre-convention\tone", "idx_zz_a\tplan-pinned\ttwo")
    with pytest.raises(SystemExit) as exc:
        repo.module.read_census()
    assert exc.value.code == 1


def test_comments_and_blank_lines_are_skipped_and_notes_are_optional(repo):
    repo.census_path.write_text(
        "# header\n\nidx_zz_a\tpre-convention\n   \nidx_zz_b\tplan-pinned\twhy\n",
        encoding="utf-8",
    )
    assert repo.module.read_census() == {
        "idx_zz_a": ("pre-convention", ""),
        "idx_zz_b": ("plan-pinned", "why"),
    }


# ---------------------------------------------------------------------------
# plan_pinning_files: only POSITIVE evidence is a pin
# ---------------------------------------------------------------------------
def test_a_positive_plan_assertion_is_a_pin(repo):
    repo.write_test(
        "test_good.py",
        PLAN_HEADER
        + '\nNAME = "idx_zz_recent"\n'
        "def test_plan(conn):\n"
        "    assert NAME in _plan(conn, 'SELECT 1')\n",
    )
    assert repo.module.plan_pinning_files()["idx_zz_recent"] == {
        "Tests/test_good.py"
    }


@pytest.mark.parametrize("operator, expected", [("in", True), ("not in", False)])
def test_literal_index_names_without_conventional_prefix_are_recognized(
    repo, operator, expected
):
    repo.write_test(
        "test_nonprefixed.py",
        PLAN_HEADER
        + "\ndef test_plan(conn):\n"
        + f"    assert 'zz_character_search_revision' {operator} _plan(conn, 'SELECT 1')\n",
    )
    assert (
        "zz_character_search_revision" in repo.module.plan_pinning_files()
    ) is expected


def test_a_positive_unique_plan_assertion_is_a_pin(repo):
    repo.write_test(
        "test_unique.py",
        PLAN_HEADER
        + '\nNAME = "uq_zz_identity"\n'
        "def test_plan(conn):\n"
        "    assert NAME in _plan(conn, 'SELECT 1')\n",
    )
    assert repo.module.plan_pinning_files()["uq_zz_identity"] == {
        "Tests/test_unique.py"
    }


def test_a_negative_assertion_is_not_a_pin(repo):
    """The reported defect, in its smallest form.

    ``assert "idx_x" not in plan`` is evidence the planner does NOT choose
    ``idx_x``. The first version read it as a pin.
    """
    repo.write_test(
        "test_negative.py",
        PLAN_HEADER
        + "\ndef test_plan(conn):\n"
        "    plan = _plan(conn, 'SELECT 1')\n"
        "    assert 'idx_zz_absent' not in plan\n",
    )
    assert "idx_zz_absent" not in repo.module.plan_pinning_files()


def test_a_comment_only_mention_is_not_a_pin(repo):
    repo.write_test(
        "test_comment.py",
        PLAN_HEADER
        + "\ndef test_plan(conn):\n"
        "    # The pre-fix plan was: SEARCH ... USING INDEX idx_zz_absent\n"
        "    assert _plan(conn, 'SELECT 1')\n",
    )
    assert "idx_zz_absent" not in repo.module.plan_pinning_files()


def test_a_docstring_only_mention_is_not_a_pin(repo):
    repo.write_test(
        "test_docstring.py",
        PLAN_HEADER
        + "\ndef test_plan(conn):\n"
        '    """Explains why idx_zz_absent was the old plan."""\n'
        "    assert _plan(conn, 'SELECT 1')\n",
    )
    assert "idx_zz_absent" not in repo.module.plan_pinning_files()


def test_a_file_that_asserts_both_ways_still_pins_the_positive_name(repo):
    """One negative mention must not disqualify a genuinely pinned index."""
    repo.write_test(
        "test_both.py",
        PLAN_HEADER
        + "\ndef test_plan(conn):\n"
        "    plan = _plan(conn, 'SELECT 1')\n"
        "    assert 'idx_zz_recent' in plan\n"
        "    assert 'idx_zz_recent' not in 'something else'\n",
    )
    assert "idx_zz_recent" in repo.module.plan_pinning_files()


@pytest.mark.parametrize("drop", ["EXPLAIN QUERY PLAN", "sqlite_stat1"])
def test_a_file_missing_either_half_of_the_evidence_pins_nothing(repo, drop):
    """A plan captured after ANALYZE is not the plan production runs."""
    repo.write_test(
        "test_half.py",
        (PLAN_HEADER + '\nNAME = "idx_zz_recent"\n').replace(drop, "XXX"),
    )
    assert repo.module.plan_pinning_files() == {}


# ---------------------------------------------------------------------------
# main(): the three failure classes and the clean run
# ---------------------------------------------------------------------------
def _declare(repo, *names: str) -> None:
    repo.write_db(
        "Schema_DB.py",
        "\n".join(f'S{i} = "CREATE INDEX {n} ON t(a)"' for i, n in enumerate(names))
        + "\n",
    )


def test_a_clean_tree_exits_zero(repo, capsys):
    _declare(repo, "idx_zz_a")
    repo.write_census("idx_zz_a\tpre-convention\tno plan captured")
    assert repo.module.main() == 0
    assert "check_index_plan_pins: OK" in capsys.readouterr().out


def test_a_declared_index_absent_from_the_census_fails(repo, capsys):
    _declare(repo, "idx_zz_a", "idx_zz_new")
    repo.write_census("idx_zz_a\tpre-convention\tno plan captured")
    assert repo.module.main() == 1
    out = capsys.readouterr().out
    assert "idx_zz_new" in out and "plan-pinned" in out and "pre-convention" in out


def test_a_census_row_for_an_index_nothing_creates_fails(repo, capsys):
    _declare(repo, "idx_zz_a")
    repo.write_census(
        "idx_zz_a\tpre-convention\tno plan captured", "idx_zz_gone\tpre-convention\tstale"
    )
    assert repo.module.main() == 1
    assert "idx_zz_gone" in capsys.readouterr().out


def test_a_plan_pinned_row_with_no_pinning_test_fails(repo, capsys):
    _declare(repo, "idx_zz_a")
    repo.write_census("idx_zz_a\tplan-pinned\tTests/test_nothing.py")
    assert repo.module.main() == 1
    assert "idx_zz_a" in capsys.readouterr().out


def test_a_plan_pinned_row_backed_only_by_a_negative_assertion_fails(repo, capsys):
    """The whole point: the guard must reject evidence of the opposite.

    Before the fix this returned 0 -- the census could claim ``plan-pinned``
    while the only test naming the index proved the planner ignores it.
    """
    _declare(repo, "idx_zz_absent")
    repo.write_test(
        "test_negative.py",
        PLAN_HEADER
        + "\ndef test_plan(conn):\n"
        "    plan = _plan(conn, 'SELECT 1')\n"
        "    # pre-fix this was SEARCH ... USING INDEX idx_zz_absent\n"
        "    assert 'idx_zz_absent' not in plan\n",
    )
    repo.write_census("idx_zz_absent\tplan-pinned\tTests/test_negative.py")
    assert repo.module.main() == 1
    assert "idx_zz_absent" in capsys.readouterr().out


def test_a_plan_pinned_row_backed_by_a_real_pin_passes(repo):
    _declare(repo, "idx_zz_recent")
    repo.write_test(
        "test_real.py",
        PLAN_HEADER
        + '\nNAME = "idx_zz_recent"\n'
        "def test_plan(conn):\n"
        "    assert NAME in _plan(conn, 'SELECT 1')\n",
    )
    repo.write_census("idx_zz_recent\tplan-pinned\tTests/test_real.py")
    assert repo.module.main() == 0


# ---------------------------------------------------------------------------
# ...and the same claim against the real tree, not only a synthetic one
# ---------------------------------------------------------------------------
def test_the_real_tree_does_not_pin_an_index_it_asserts_is_absent(checker):
    """Measured on the shipped files, so a revert is caught here too.

    ``Tests/DB/test_media_db_schema_v9.py`` qualifies (it runs EXPLAIN QUERY
    PLAN and asserts ``sqlite_stat1`` is absent) and names both of these,
    every time in a comment, a docstring, or a ``not in`` assertion.
    """
    pins = checker.plan_pinning_files()
    assert "idx_media_deleted" not in pins
    assert "idx_keywords_deleted" not in pins
    # ...while every index the census really does record as plan-pinned is
    # still backed by a file, or the shipped checker would be failing CI.
    census = checker.read_census()
    declared = checker.declared_indexes()
    pinned = {
        name
        for name, (status, _note) in census.items()
        if status == checker.PLAN_PINNED and name in declared
    }
    assert pinned, "the census records no plan-pinned index; this test is vacuous"
    assert pinned <= set(pins), sorted(pinned - set(pins))
