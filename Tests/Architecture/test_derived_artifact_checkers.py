"""TASK-19572: the derived-artifact checkers must teach the fix, not just fail.

`check_persistent_diagnostic_inventory.py` used to fail with a single sentence
naming no file, no owner and no call site. Four separate burn-down tasks each
hand-rebuilt a ~30-line diff probe to find out what had drifted. These tests pin
the promoted report: which rows moved, by how much, and what to run next.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts import check_backlog_task_ids as backlog_ids
from scripts import check_persistent_diagnostic_inventory as inventory


def _inventory(**overrides) -> dict:
    base = {
        "schema_version": 2,
        "scope": "tldw_chatbook/**/*.py",
        "classification_rules": {"TASK-492": {"prefixes": ["tldw_chatbook/Chat/"]}},
        "reviewed_exclusions": [],
        "summary": {
            "owner_files": 2,
            "task_492_calls": 3,
            "task_494_calls": 4,
            "persistent_sink_files": 1,
        },
        "owners": [
            {
                "path": "tldw_chatbook/Chat/a.py",
                "owner": "TASK-492",
                "reason": "r",
                "call_count": 3,
                "diagnostic_digest": "aaaaaaaaaaaaaaaaaaaa",
            },
            {
                "path": "tldw_chatbook/b.py",
                "owner": "TASK-494",
                "reason": "r",
                "call_count": 4,
                "diagnostic_digest": "bbbbbbbbbbbbbbbbbbbb",
            },
        ],
        "persistent_sink_topology": [
            {
                "path": "tldw_chatbook/Logging_Config.py",
                "sinks": [
                    {
                        "method": "add",
                        "digest": "cccccccccccccccc",
                        "kind": "loguru_sink",
                        "scope": "configure_application_logging",
                    }
                ],
            }
        ],
    }
    base.update(overrides)
    return base


def _diff(committed: dict, rebuilt: dict) -> str:
    return inventory.render_diff(json.dumps(committed), rebuilt)


def test_added_diagnostic_names_the_file_and_the_delta():
    committed = _inventory()
    rebuilt = copy.deepcopy(committed)
    rebuilt["owners"][1]["call_count"] = 6
    rebuilt["owners"][1]["diagnostic_digest"] = "dddddddddddddddddddd"
    rebuilt["summary"]["task_494_calls"] = 6

    report = _diff(committed, rebuilt)

    assert "tldw_chatbook/b.py" in report
    assert "4/bbbbbbbbbbbbbbbbbbbb -> 6/dddddddddddddddddddd" in report
    assert "+2 diagnostic call(s)" in report
    assert "task_494_calls: 4 -> 6" in report
    assert "--write" in report


def test_reworded_diagnostic_is_distinguished_from_an_added_one():
    """Same count, different digest: the message, level or arguments changed --
    the case that actually matters for privacy review."""
    committed = _inventory()
    rebuilt = copy.deepcopy(committed)
    rebuilt["owners"][0]["diagnostic_digest"] = "eeeeeeeeeeeeeeeeeeee"

    report = _diff(committed, rebuilt)

    assert "same count, content changed" in report
    assert "tldw_chatbook/Chat/a.py" in report


def test_files_gained_and_lost_are_reported_on_their_own_sides():
    committed = _inventory()
    rebuilt = copy.deepcopy(committed)
    rebuilt["owners"] = [
        rebuilt["owners"][0],
        {
            "path": "tldw_chatbook/c.py",
            "owner": "TASK-494",
            "reason": "r",
            "call_count": 1,
            "diagnostic_digest": "ffffffffffffffffffff",
        },
    ]

    report = _diff(committed, rebuilt)

    assert "only in committed" in report and "tldw_chatbook/b.py" in report
    assert "only in rebuild" in report and "tldw_chatbook/c.py" in report


def test_a_new_persistent_sink_file_is_called_out():
    """A new disk sink is the highest-consequence drift this artifact tracks."""
    committed = _inventory()
    rebuilt = copy.deepcopy(committed)
    rebuilt["persistent_sink_topology"].append(
        {
            "path": "tldw_chatbook/Utils/secret_dump.py",
            "sinks": [
                {
                    "method": "open_private_text_append",
                    "digest": "1111111111111111",
                    "kind": "open_private_text_append",
                    "scope": "dump",
                }
            ],
        }
    )

    report = _diff(committed, rebuilt)

    assert "NEW persistent sink file" in report
    assert "tldw_chatbook/Utils/secret_dump.py" in report
    assert "open_private_text_append" in report


def test_a_changed_sink_entry_shows_both_sides():
    committed = _inventory()
    rebuilt = copy.deepcopy(committed)
    rebuilt["persistent_sink_topology"][0]["sinks"][0]["scope"] = "some_other_scope"

    report = _diff(committed, rebuilt)

    assert "changed sinks" in report
    assert "configure_application_logging" in report
    assert "some_other_scope" in report


def test_sink_multiplicity_increase_is_named_with_counts():
    """Qodo PR #1947 finding 1: `_sink_lines()` used to collapse each file's
    sink list into a dict keyed by (scope, kind, method, digest), so a SECOND,
    byte-identical sink call (the same handler installed twice) shared the
    same key and silently vanished from the comparison -- a pure multiplicity
    drift, the highest-consequence class this artifact tracks, reported as no
    change at all (or worse, as "only serialization differs"). The fix must
    name the count explicitly, not merely say something changed."""
    committed = _inventory()
    rebuilt = copy.deepcopy(committed)
    duplicate_sink = dict(rebuilt["persistent_sink_topology"][0]["sinks"][0])
    rebuilt["persistent_sink_topology"][0]["sinks"].append(duplicate_sink)

    report = _diff(committed, rebuilt)

    assert "only its serialization differs" not in report
    assert "changed sinks" in report
    assert "tldw_chatbook/Logging_Config.py (1 -> 2 entries)" in report
    assert (
        "configure_application_logging: loguru_sink.add (cccccccccccccccc): "
        "1 -> 2  (+1)" in report
    )


def test_sink_multiplicity_decrease_is_named_too():
    """The symmetric case: deleting one of two identical sink calls must also
    be named with the old/new count, not silently absorbed."""
    committed = _inventory()
    duplicate_sink = dict(committed["persistent_sink_topology"][0]["sinks"][0])
    committed["persistent_sink_topology"][0]["sinks"].append(duplicate_sink)
    rebuilt = copy.deepcopy(committed)
    del rebuilt["persistent_sink_topology"][0]["sinks"][1]

    report = _diff(committed, rebuilt)

    assert "tldw_chatbook/Logging_Config.py (2 -> 1 entries)" in report
    assert (
        "configure_application_logging: loguru_sink.add (cccccccccccccccc): "
        "2 -> 1  (-1)" in report
    )


def test_metadata_drift_is_not_silent():
    """The checker compares the whole encoded file, so a changed classification
    rule fails it too. Without this section the report would list zero rows and
    read as a false alarm."""
    committed = _inventory()
    rebuilt = copy.deepcopy(committed)
    rebuilt["classification_rules"]["TASK-492"]["prefixes"].append("tldw_chatbook/DB/")

    report = _diff(committed, rebuilt)

    assert "classification_rules" in report
    assert "tldw_chatbook/DB/" in report


def test_serialization_only_drift_says_so():
    """Identical content, different bytes (hand-edited whitespace or key order)
    -- otherwise the report is empty and looks like a checker bug."""
    committed = _inventory()
    report = inventory.render_diff(
        json.dumps(committed, indent=8), copy.deepcopy(committed)
    )
    assert "only its serialization differs" in report


def test_unparseable_committed_file_is_explained():
    report = inventory.render_diff("{not json", _inventory())
    assert "not valid JSON" in report
    assert "--write" in report


def test_report_always_ends_with_the_next_command():
    """Every exit path hands the reader the same one command."""
    committed = _inventory()
    rebuilt = copy.deepcopy(committed)
    rebuilt["owners"][0]["call_count"] = 99
    assert inventory.NEXT_STEPS in _diff(committed, rebuilt)


def test_the_changed_row_note_admits_re_indentation():
    """The row note used to offer only "reworded / re-levelled / new args".

    A count-preserving digest change also happens when a call merely shifts
    nesting level, and sending a reviewer looking for a rewording that is not
    there is how a gate loses its credibility.
    """
    committed = _inventory()
    rebuilt = copy.deepcopy(committed)
    rebuilt["owners"][0]["diagnostic_digest"] = "eeeeeeeeeeeeeeeeeeee"

    report = _diff(committed, rebuilt)

    assert "same count, content changed" in report
    assert "re-indented" in report
    assert "--statements" in report


def test_next_steps_sends_the_reader_to_statements_not_git_diff():
    """`git diff` is the wrong recovery tool for this artifact and the trailer
    must not recommend it: the digest covers indentation, so a moved call reads
    as changed and a line diff buries it in unrelated edits."""
    assert "--statements" in inventory.NEXT_STEPS
    assert "--since" in inventory.NEXT_STEPS
    assert "git diff" not in inventory.NEXT_STEPS.replace("Do NOT reach for `git diff`", "")


_MOVED_BEFORE = """
import logging
logger = logging.getLogger(__name__)

def handler(exc):
    if exc:
        logger.warning(
            "wake delivery ledger stamp failed (exception_type={})",
            type(exc).__name__,
        )
"""

_MOVED_AFTER = """
import logging
logger = logging.getLogger(__name__)

def handler(exc):
    logger.warning(
        "wake delivery ledger stamp failed (exception_type={})",
        type(exc).__name__,
    )
"""


def test_a_re_indented_statement_is_reported_as_needing_no_review():
    """The incident this mode was built for: TASK-19572's pre-merge review found
    console_fleet_wake.py's row changed inside a 328-line diff in which not one
    diagnostic statement had actually changed -- only two had been re-indented
    when the surrounding code was refenced."""
    report = inventory.render_statement_diff(_MOVED_BEFORE, _MOVED_AFTER, "m.py")

    assert "moved/re-indented only: 1" in report
    assert "removed: 0" in report
    assert "added: 0" in report
    assert "NO review needed" in report


def test_an_added_statement_is_printed_in_full_for_reading():
    """The whole point: the pin cannot carry statement text, and the
    interpolation check needs it."""
    after = _MOVED_AFTER + """
def other(path):
    logger.error(f"export failed for {path}")
"""
    report = inventory.render_statement_diff(_MOVED_AFTER, after, "m.py")

    assert "added: 1" in report
    assert 'logger.error(f"export failed for {path}")' in report
    assert "interpolate user content, a secret, a path, or a URL" in report


def test_a_removed_statement_is_shown_too():
    report = inventory.render_statement_diff(_MOVED_AFTER, "x = 1\n", "m.py")

    assert "removed: 1" in report
    assert "wake delivery ledger stamp failed" in report


def test_no_statement_change_says_the_pin_was_already_stale():
    """A listed row with nothing to show means the drift predates the base --
    the TASK-19572 finding that two rows rode into a pin unexamined."""
    report = inventory.render_statement_diff(_MOVED_AFTER, _MOVED_AFTER, "m.py")

    assert "no diagnostic statement changed" in report
    assert "widen the base revision" in report


def test_statement_digests_match_the_ones_the_pin_moves():
    """The report is only trustworthy if its keys are the gate's keys."""
    entries = inventory._statement_entries(_MOVED_AFTER, "m.py")
    diagnostics, _ = inventory.scan_source(_MOVED_AFTER, filename="m.py")

    assert [e["digest"] for e in entries] == [d["digest"] for d in diagnostics]


def test_duplicate_task_ids_are_caught_in_both_namespaces(tmp_path):
    """A hand-edited rename can leave the filename unique and the frontmatter
    colliding, which is what the backlog CLI actually resolves on."""
    (tmp_path / "task-42 - First.md").write_text("id: TASK-42\n", encoding="utf-8")
    (tmp_path / "task-42.1 - Second.md").write_text("id: task-42\n", encoding="utf-8")
    (tmp_path / "task-7 - Fine.md").write_text("id: TASK-7\n", encoding="utf-8")

    by_filename, by_frontmatter = backlog_ids.duplicate_ids(tmp_path)

    assert by_filename == {}
    # Paths outside the repo are reported in full: two scratch buckets can both
    # be named "tasks", and a basename would make the rows indistinguishable.
    assert by_frontmatter == {
        "task-42": sorted(
            [
                (tmp_path / "task-42 - First.md").resolve().as_posix(),
                (tmp_path / "task-42.1 - Second.md").resolve().as_posix(),
            ]
        )
    }
    assert backlog_ids.main(["--tasks-dir", str(tmp_path)]) == 1


def test_duplicate_task_ids_are_caught_across_buckets(tmp_path):
    """Archiving a colliding file must not hide it: the id is still taken.

    Upstream Backlog.md reissues an archived task's id to the next
    ``task create``, so this is how collisions are actually born here -- and
    scanning only ``backlog/tasks`` let TASK-2157 sit on dev with a green guard
    while ``backlog task 2157`` resolved to two files.
    """
    tasks = tmp_path / "tasks"
    archive = tmp_path / "archive" / "tasks"
    tasks.mkdir()
    archive.mkdir(parents=True)
    (tasks / "task-9 - Live.md").write_text("id: TASK-9\n", encoding="utf-8")
    (archive / "task-9 - Archived.md").write_text("id: TASK-9\n", encoding="utf-8")

    by_filename, by_frontmatter = backlog_ids.duplicate_ids(tasks, archive)

    assert set(by_filename) == {"task-9"}
    assert set(by_frontmatter) == {"task-9"}
    assert backlog_ids.main(["--tasks-dir", str(tasks), "--tasks-dir", str(archive)]) == 1


def test_default_scope_is_every_bucket_the_cli_resolves():
    """Pin the scope: narrowing it back to tasks/ is the bug this guard had."""
    assert {
        path.relative_to(backlog_ids.REPO_ROOT).as_posix() for path in backlog_ids.TASK_DIRS
    } == {"backlog/tasks", "backlog/completed", "backlog/archive/tasks"}


def test_an_absent_optional_bucket_is_not_an_error(tmp_path):
    """A project with no completed/ or archive/ still passes."""
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "task-1 - A.md").write_text("id: TASK-1\n", encoding="utf-8")

    assert backlog_ids.main(["--tasks-dir", str(tasks), "--tasks-dir", str(tmp_path / "absent")]) == 0


def test_unique_task_ids_pass(tmp_path):
    (tmp_path / "task-1 - A.md").write_text("id: TASK-1\n", encoding="utf-8")
    (tmp_path / "task-2 - B.md").write_text("id: TASK-2\n", encoding="utf-8")
    assert backlog_ids.duplicate_ids(tmp_path) == ({}, {})
    assert backlog_ids.main(["--tasks-dir", str(tmp_path)]) == 0


def test_repo_relative_accepts_a_relative_path_inside_the_repo():
    assert inventory._repo_relative(
        "scripts/check_backlog_task_ids.py"
    ) == Path("scripts/check_backlog_task_ids.py")


def test_repo_relative_accepts_an_absolute_path_inside_the_repo():
    absolute = inventory.REPO_ROOT / "scripts" / "check_backlog_task_ids.py"
    assert inventory._repo_relative(str(absolute)) == Path(
        "scripts/check_backlog_task_ids.py"
    )


def test_repo_relative_rejects_an_absolute_path_outside_the_repo():
    """Qodo PR #1947 finding 4: this used to be an unguarded
    `Path.relative_to(REPO_ROOT)` that raised `ValueError` for exactly this
    input -- an absolute path outside the repo, a routine CLI slip."""
    assert inventory._repo_relative("/etc/hosts") is None


def test_repo_relative_rejects_relative_traversal_out_of_the_repo():
    """Finding 3: a relative path containing enough `..` segments to walk out
    of REPO_ROOT must not be read silently."""
    outside = "/".join([".."] * 12) + "/etc/hosts"
    assert inventory._repo_relative(outside) is None


def test_statements_on_an_absolute_path_outside_the_repo_does_not_crash(capsys):
    """The bug: `_run_statements()` called `path.relative_to(REPO_ROOT)` with
    no exception handling, so this exact input killed the recovery tool the
    failure report tells developers to run with a traceback instead of an
    error message."""
    assert inventory._run_statements(["/etc/hosts"], None) == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "does not resolve inside the repository" in captured.err


def test_statements_on_relative_traversal_does_not_read_outside_the_repo(capsys):
    outside = "/".join([".."] * 12) + "/etc/hosts"
    assert inventory._run_statements([outside], None) == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "does not resolve inside the repository" in captured.err


def test_statements_still_works_for_a_real_in_repo_path():
    """The fix must not regress the ordinary case: a relative in-repo path."""
    assert inventory._run_statements(["scripts/check_backlog_task_ids.py"], None) == 0
