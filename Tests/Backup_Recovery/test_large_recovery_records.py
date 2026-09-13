"""Valid file collections must fit both staging and durable recovery records."""

import json
from pathlib import Path
from textwrap import indent
from threading import Event

import pytest

from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Backup_Recovery.journal import Journal, observe_artifact
from tldw_chatbook.Backup_Recovery.publication import _descriptor
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.staging import stage_restore


def test_small_recovery_record_does_not_allocate_the_maximum_read_buffer(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.limits import RECOVERY_RECORD_BYTES
    from tldw_chatbook.Backup_Recovery.native_files import (
        create_private_file,
        pinned_directory,
    )

    data = b'{"version":1}'
    with create_private_file(tmp_path / "record.json") as fd:
        bootstrap.os.write(fd, data)
    original = bootstrap.os.read
    requested = []

    def read(fd, count):
        requested.append(count)
        return original(fd, count)

    monkeypatch.setattr(bootstrap.os, "read", read)
    with pinned_directory(tmp_path) as parent:
        assert bootstrap._read(parent, "record.json", max_bytes=RECOVERY_RECORD_BYTES) == {"version": 1}
    assert max(requested) <= 64 * 1024


def test_private_record_read_handles_short_reads_and_checks_exact_eof(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.native_files import (
        create_private_file,
        pinned_directory,
    )

    data = b'{"version":1}'
    with create_private_file(tmp_path / "record.json") as fd:
        bootstrap.os.write(fd, data)
    original = bootstrap.os.read
    monkeypatch.setattr(bootstrap.os, "read", lambda fd, count: original(fd, min(count, 3)))
    with pinned_directory(tmp_path) as parent:
        assert bootstrap._read(parent, "record.json", max_bytes=len(data)) == {"version": 1}


def test_private_record_growth_beyond_limit_is_refused(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery.native_files import (
        create_private_file,
        pinned_directory,
    )

    data = b'{"version":1}'
    path = tmp_path / "record.json"
    with create_private_file(path) as fd:
        bootstrap.os.write(fd, data)
    original = bootstrap.os.read
    grew = False

    def read(fd, count):
        nonlocal grew
        if not grew:
            with path.open("ab") as output:
                output.write(b" ")
            grew = True
        return original(fd, count)

    monkeypatch.setattr(bootstrap.os, "read", read)
    with pinned_directory(tmp_path) as parent, pytest.raises(ValueError, match="oversized_record"):
        bootstrap._read(parent, "record.json", max_bytes=len(data))


def test_collection_descriptor_and_journal_survive_reopening(tmp_path):
    def many_files(doc):
        template = doc["files"][0]
        doc["files"] = [
            dict(template, logical_id=f"file{i}", relative_path=f"note{i}.txt", payload=f"payload/{i}")
            for i in range(1800)
        ]
        doc["dependency_groups"][0]["members"] = [row["logical_id"] for row in doc["files"]]

    archive = sealed(tmp_path, mutate=many_files)
    plan = plan_restore(
        archive, mode="isolated", destinations={"root": tmp_path / "restored"}, target=None
    )
    candidate = stage_restore(archive, plan, tmp_path / "stage", Event())
    descriptor = candidate / "candidate.json"
    assert descriptor.stat().st_size > 1024**2
    _descriptor(candidate, plan)

    # Exercise the actual durable writer, reader and flush after an app restart.
    artifacts = [
        {
            "logical_id": row["logical_id"],
            "candidate": observe_artifact(Path(row["candidate"])),
            "target": row["destination"],
            "previous": None,
            "retained": None,
        }
        for row in json.loads(descriptor.read_bytes())["artifacts"]
    ]
    (tmp_path / "control").mkdir(mode=0o700)
    journal = Journal(tmp_path / "control", "collection")
    journal.record("prepared", {"generation": "g1", "mode": "isolated", "artifacts": artifacts})
    assert (journal.root / "000000.json").stat().st_size > 1024**2
    reopened = Journal(tmp_path / "control", "collection")
    with reopened._locked(exclusive=True) as parent:
        reopened._flush_records(parent)
        assert len(reopened._records(parent)[0].evidence["artifacts"]) == 1801


def test_bootstrap_limit_stays_small_and_recovery_limit_is_bounded(tmp_path):
    from tldw_chatbook.Backup_Recovery.bootstrap import MAX_RECORD, _read
    from tldw_chatbook.Backup_Recovery.limits import RECOVERY_RECORD_BYTES
    from tldw_chatbook.Backup_Recovery.native_files import (
        create_private_file,
        pinned_directory,
    )
    from tldw_chatbook.Utils.platform_files import os

    with create_private_file(tmp_path / "record.json") as fd:
        os.ftruncate(fd, MAX_RECORD + 1)
    with pinned_directory(tmp_path) as parent, pytest.raises(ValueError, match="oversized_record"):
        _read(parent, "record.json")
    with (tmp_path / "record.json").open("r+b") as output:
        output.truncate(RECOVERY_RECORD_BYTES + 1)
    with pinned_directory(tmp_path) as parent, pytest.raises(ValueError, match="oversized_record"):
        _read(parent, "record.json", max_bytes=RECOVERY_RECORD_BYTES)


def test_journal_rejects_oversized_record_before_append(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import journal as source

    (tmp_path / "control").mkdir(mode=0o700)
    journal = Journal(tmp_path / "control", "bounded")
    monkeypatch.setattr(source, "RECOVERY_RECORD_BYTES", 32)
    with pytest.raises(ValueError, match="journal_record_limit"):
        journal.record("prepared", {"generation": "g", "mode": "isolated"})
    assert not list(journal.root.glob("*.json"))
    monkeypatch.setattr(source, "RECOVERY_RECORD_BYTES", 1024**2)
    journal.record("prepared", {"generation": "g", "mode": "isolated"})


def test_publication_reserves_events_before_changing_targets(tmp_path, monkeypatch):
    from Tests.Backup_Recovery.test_publication_finalization import installed
    from tldw_chatbook.Backup_Recovery import journal as source

    monkeypatch.setattr(source, "MAX_EVENTS", 10)
    with pytest.raises(ValueError, match="recovery_event_budget"), installed(tmp_path):
        pytest.fail("a restore without room for rollback was published")
    assert not (tmp_path / "isolated" / "config" / "config.toml").exists()
    events = [json.loads(path.read_bytes())["event"] for path in (tmp_path / "control").rglob("[0-9]*.json")]
    assert "publication_started" not in events


def test_large_collection_completes_actual_isolated_publication(tmp_path):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run
    from Tests.Backup_Recovery.test_isolated_restore import _RESTORE

    extra = """
 doc['owners'].append(dict(owner_id='external.files', schema_version=1, capabilities=[]))
 doc['directories'].append(dict(doc['directories'][0], logical_id='external', root_id='external', synthetic=False))
 template=doc['files'][0]
 for index in range(1800):
  doc['files'].append(dict(template,logical_id=f'external:file{index}',root_id='external',parent_id='external',relative_path=f'note{index}.txt',owner_id='external.files',payload=f'payload/extra{index}'))
 doc['dependency_groups'].append(dict(group_id='external-group',members=['external']+[row['logical_id'] for row in doc['files'][1:]],complete=True))
"""
    script = _RESTORE.replace("archive=sealed(base,", extra + "archive=sealed(base,")
    script = script.replace("'root':dest/'config',", "'external':dest/'research','root':dest/'config',")
    script += "\nassert len(list((dest/'research').glob('note*.txt')))==1800\n"
    script += "assert (dest/'research'/'note1799.txt').read_bytes()==b'[general]\\nusers_name=\"original\"\\n'\n"
    script = (
        "from pathlib import Path\n"
        "from Tests.Backup_Recovery.thread_diagnostics import observe_threads\n"
        "stop_stacks=observe_threads(Path.home()/'large-isolated-stacks.log',interval=30)\n"
        "print('LARGE_ISOLATED_STARTED',flush=True)\ntry:\n"
        + indent(script, " ")
        + "\nfinally:stop_stacks()\nprint('LARGE_ISOLATED_COMPLETED',flush=True)\n"
    )
    _run(tmp_path, "complete", "large-isolated", script=script, timeout=180)
