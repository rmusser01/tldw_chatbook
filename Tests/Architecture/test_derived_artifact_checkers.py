"""TASK-19572: the derived-artifact checkers must teach the fix, not just fail.

`check_persistent_diagnostic_inventory.py` used to fail with a single sentence
naming no file, no owner and no call site. Four separate burn-down tasks each
hand-rebuilt a ~30-line diff probe to find out what had drifted. These tests pin
the promoted report: which rows moved, by how much, and what to run next.
"""

from __future__ import annotations

import copy
import json

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


def test_duplicate_task_ids_are_caught_in_both_namespaces(tmp_path):
    """A hand-edited rename can leave the filename unique and the frontmatter
    colliding, which is what the backlog CLI actually resolves on."""
    (tmp_path / "task-42 - First.md").write_text("id: TASK-42\n", encoding="utf-8")
    (tmp_path / "task-42.1 - Second.md").write_text("id: task-42\n", encoding="utf-8")
    (tmp_path / "task-7 - Fine.md").write_text("id: TASK-7\n", encoding="utf-8")

    by_filename, by_frontmatter = backlog_ids.duplicate_ids(tmp_path)

    assert by_filename == {}
    assert by_frontmatter == {"task-42": ["task-42 - First.md", "task-42.1 - Second.md"]}
    assert backlog_ids.main(["--tasks-dir", str(tmp_path)]) == 1


def test_unique_task_ids_pass(tmp_path):
    (tmp_path / "task-1 - A.md").write_text("id: TASK-1\n", encoding="utf-8")
    (tmp_path / "task-2 - B.md").write_text("id: TASK-2\n", encoding="utf-8")
    assert backlog_ids.duplicate_ids(tmp_path) == ({}, {})
    assert backlog_ids.main(["--tasks-dir", str(tmp_path)]) == 0
