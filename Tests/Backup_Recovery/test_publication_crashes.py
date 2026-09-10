"""Durable recovery records never infer successful publication from intent."""

import json
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "damage", ["truncated", "gap", "duplicate", "unknown", "linked"]
)
def test_damaged_journal_is_retained_for_recovery(tmp_path, damage):
    from tldw_chatbook.Backup_Recovery.journal import Journal

    journal = Journal(tmp_path, "op")
    journal.record("prepared", {"generation": "g1", "mode": "isolated"})
    record = journal.root / "000000.json"
    if damage == "truncated":
        record.write_bytes(b'{"version":1,')
    elif damage == "gap":
        record.rename(journal.root / "000001.json")
    elif damage == "duplicate":
        record.write_bytes(b'{"version":1,"version":1}')
    elif damage == "unknown":
        value = json.loads(record.read_bytes())
        value["event"] = "invented_success"
        record.write_text(json.dumps(value))
    else:
        os.link(record, journal.root / "linked")
    before = {p.name: p.read_bytes() for p in journal.root.iterdir()}
    assert Journal(tmp_path, "op").recover() == "recovery_required"
    assert before == {p.name: p.read_bytes() for p in journal.root.iterdir()}


@pytest.mark.parametrize("boundary", ["prepared", "publication_started"])
def test_process_exit_after_durable_record_retains_startup_fence(tmp_path, boundary):
    from tldw_chatbook.Backup_Recovery.bootstrap import startup_permission
    from tldw_chatbook.Backup_Recovery.control_records import register_pending
    from tldw_chatbook.Backup_Recovery.journal import Journal

    config = tmp_path / "broken.toml"
    config.write_bytes(b"broken [")
    register_pending(
        tmp_path / "bootstrap", "op", ("scope",), tmp_path / "control", (config,)
    )
    script = """
import os, sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery.journal import Journal
j = Journal(Path(sys.argv[1]), "op")
j.record("prepared", {"generation":"g1", "mode":"isolated"})
if sys.argv[2] == "publication_started":
    j.record("publication_started", {})
os._exit(73)
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), boundary],
        timeout=20,
        capture_output=True,
        check=False,
    )
    assert child.returncode == 73, child.stderr.decode()
    expected = "prepared" if boundary == "prepared" else "recovery_required"
    assert Journal(tmp_path, "op").recover() == expected
    assert startup_permission(config, tmp_path / "bootstrap") == (
        False,
        "recovery_pending",
    )
    assert config.read_bytes() == b"broken ["


def test_replacement_cannot_start_before_verified_rollback(tmp_path):
    from tldw_chatbook.Backup_Recovery.journal import Journal

    journal = Journal(tmp_path, "op")
    journal.record("prepared", {"generation": "g1", "mode": "replace"})
    with pytest.raises(ValueError, match="rollback_required"):
        journal.record("publication_started", {})


def test_artifact_identity_recognizes_rename_without_completion_record(tmp_path):
    from tldw_chatbook.Backup_Recovery.journal import Journal, observe_artifact
    from tldw_chatbook.Backup_Recovery.native_files import publish_new

    candidate = tmp_path / "candidate"
    candidate.mkdir(mode=0o700)
    (candidate / "data").write_bytes(b"durable new bytes")
    (candidate / "empty").mkdir()
    target = tmp_path / "target"
    journal = Journal(tmp_path, "op")
    journal.record(
        "prepared",
        {
            "generation": "g1",
            "mode": "isolated",
            "artifacts": [
                {
                    "logical_id": "data",
                    "candidate": observe_artifact(candidate),
                    "target": str(target),
                    "previous": None,
                    "retained": None,
                }
            ],
        },
    )
    journal.record("publication_started", {})
    publish_new(candidate, target)
    assert journal.artifact_states() == {"data": "published"}
    assert Journal(tmp_path, "op").recover() == "recovery_required"
    (target / "data").write_bytes(b"unreviewed change")
    assert journal.artifact_states() == {"data": "uncertain"}


def test_unrecorded_directory_child_invalidates_artifact(tmp_path):
    from tldw_chatbook.Backup_Recovery.journal import Journal, observe_artifact

    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    (source / "one").write_bytes(b"one")
    journal = Journal(tmp_path, "op")
    journal.record(
        "prepared",
        {
            "generation": "g",
            "mode": "isolated",
            "artifacts": [
                {
                    "logical_id": "tree",
                    "candidate": observe_artifact(source),
                    "target": str(tmp_path / "target"),
                    "previous": None,
                    "retained": None,
                }
            ],
        },
    )
    (source / "two").write_bytes(b"two")
    assert journal.artifact_states() == {"tree": "uncertain"}


def test_incomplete_publication_remains_recovery_required(tmp_path):
    from tldw_chatbook.Backup_Recovery.journal import Journal

    journal = Journal(tmp_path, "operation-1")
    journal.record("prepared", {"generation": "g1", "mode": "isolated"})
    journal.record("publication_started", {})
    assert Journal(tmp_path, "operation-1").recover() == "recovery_required"


def test_prepared_record_survives_fresh_reader(tmp_path):
    from tldw_chatbook.Backup_Recovery.journal import Journal

    Journal(tmp_path, "operation-1").record(
        "prepared", {"generation": "g1", "mode": "isolated"}
    )
    assert Journal(tmp_path, "operation-1").recover() == "prepared"


@pytest.mark.parametrize("event", ["committed", "artifact_published", "unknown"])
def test_event_cannot_skip_preparation(tmp_path, event):
    from tldw_chatbook.Backup_Recovery.journal import Journal

    journal = Journal(tmp_path, "operation-1")
    with pytest.raises(ValueError, match="journal_transition_invalid"):
        journal.record(event, {})


def test_journal_refuses_secret_fields(tmp_path):
    from tldw_chatbook.Backup_Recovery.journal import Journal

    with pytest.raises(ValueError, match="journal_evidence_invalid"):
        Journal(tmp_path, "operation-1").record(
            "prepared",
            {"generation": "g1", "mode": "isolated", "password": "never-persist"},
        )
    assert all(
        b"never-persist" not in path.read_bytes() for path in tmp_path.rglob("*.json")
    )


def test_preparation_refuses_retained_path_without_previous_object(tmp_path):
    from tldw_chatbook.Backup_Recovery.journal import Journal, observe_artifact

    candidate = tmp_path / "candidate"
    candidate.write_bytes(b"candidate bytes")
    journal = Journal(tmp_path, "op")
    with pytest.raises(ValueError, match="journal_evidence_invalid"):
        journal.record(
            "prepared",
            {
                "generation": "g",
                "mode": "isolated",
                "artifacts": [
                    {
                        "logical_id": "data",
                        "candidate": observe_artifact(candidate),
                        "target": str(tmp_path / "target"),
                        "previous": None,
                        "retained": str(tmp_path / "unexpected-retained"),
                    }
                ],
            },
        )
    assert candidate.read_bytes() == b"candidate bytes"
    assert not (journal.root / "000000.json").exists()


def test_tree_observation_refuses_earlier_child_changed_during_later_read(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import journal

    candidate = tmp_path / "candidate"
    candidate.mkdir(mode=0o700)
    first = candidate / "a"
    second = candidate / "b"
    first.write_bytes(b"original")
    second.write_bytes(b"later child")
    later_identity = second.stat().st_dev, second.stat().st_ino
    original_read = os.read
    changed = False

    def read(fd, size):
        nonlocal changed
        info = os.fstat(fd)
        if not changed and (info.st_dev, info.st_ino) == later_identity:
            # Sorted traversal already read/closed a. In-place mutation does
            # not change the directory names or its mtime, and keeps a's size.
            first.write_bytes(b"modified")
            changed = True
        return original_read(fd, size)

    monkeypatch.setattr(journal.os, "read", read)
    with pytest.raises(ValueError, match="artifact_changed"):
        journal.observe_artifact(candidate)
    assert changed
    assert first.read_bytes() == b"modified"
    assert second.read_bytes() == b"later child"
