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


def test_large_restore_observer_preserves_calls_errors_and_restores_originals(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import tldw_chatbook.Backup_Recovery as package
    from Tests.Backup_Recovery.thread_diagnostics import observe_large_restore

    token = object()
    failure = ValueError("private-exception-text")
    received = []
    path = tmp_path / "timings.json"

    def original(*args, **kwargs):
        received.append((args, kwargs))
        return token

    def failing(*args, **kwargs):
        raise failure

    staging = SimpleNamespace(require_capacity=original, _copy=failing, stage_restore=original)
    publication = SimpleNamespace(publish_candidate=original, finalize_candidate=original)
    owner = SimpleNamespace(validate=original)
    monkeypatch.setattr(package, "staging", staging)
    monkeypatch.setattr(package, "publication", publication)
    monkeypatch.setattr(package, "recovery_files", SimpleNamespace(_RawDeclaration=owner))
    stop = observe_large_restore(path)
    try:
        assert staging.require_capacity(token, private_argument=token) is token  # nosec B101
        assert received == [((token,), {"private_argument": token})]  # nosec B101
        with pytest.raises(ValueError) as caught:
            staging._copy(token)
        assert caught.value is failure  # nosec B101
        assert staging.stage_restore() is token  # nosec B101
        assert publication.publish_candidate() is token  # nosec B101
        assert publication.finalize_candidate() is token  # nosec B101
    finally:
        stop()
    data = json.loads(path.read_text())
    assert data["calls"]["copy"]["failed"] == 1  # nosec B101
    assert data["calls"]["copy"]["completed"] == 1  # nosec B101
    assert data["calls"]["copy"]["active"] == 0  # nosec B101
    assert [row["phase"] for row in data["phases"]] == [  # nosec B101
        "stage_started", "stage_finished", "publication_started", "publication_finished",
        "finalization_started", "finalization_finished",
    ]
    assert "private" not in path.read_text()  # nosec B101
    assert staging.require_capacity is original and staging._copy is failing  # nosec B101
    assert owner.validate is original and publication.publish_candidate is original  # nosec B101


def test_large_restore_observer_is_bounded_and_records_in_progress_call(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import tldw_chatbook.Backup_Recovery as package
    from Tests.Backup_Recovery.thread_diagnostics import observe_large_restore

    path = tmp_path / "timings.json"
    active = []

    def copy():
        active.append(json.loads(path.read_text())["calls"]["copy"]["active"])

    def noop():
        pass
    staging = SimpleNamespace(require_capacity=noop, _copy=copy, stage_restore=noop)
    monkeypatch.setattr(package, "staging", staging)
    monkeypatch.setattr(package, "publication", SimpleNamespace(publish_candidate=noop, finalize_candidate=noop))
    monkeypatch.setattr(package, "recovery_files", SimpleNamespace(_RawDeclaration=SimpleNamespace(validate=noop)))
    stop = observe_large_restore(path)
    try:
        staging._copy()
        for _ in range(2000):
            staging.require_capacity()
    finally:
        stop()
    data = json.loads(path.read_text())
    assert active == [1]  # nosec B101
    assert data["calls"]["require_capacity"]["completed"] == 2000  # nosec B101
    assert data["calls"]["require_capacity"]["elapsed_seconds"] >= 0  # nosec B101
    assert len(data["calls"]) == 6 and len(data["phases"]) <= 12  # nosec B101
    assert path.stat().st_size < 4096  # nosec B101


def _large_restore_script(script):
    return (
        "import sys\nfrom pathlib import Path\n"
        "from Tests.Backup_Recovery.thread_diagnostics import observe_threads,observe_large_restore\n"
        "stop_stacks=observe_threads(Path.home()/'large-isolated-stacks.log',interval=30)\n"
        "stop_timings=None\ntry:\n"
        " stop_timings=observe_large_restore(Path.home()/'large-isolated-timings.log')\n"
        " print('LARGE_ISOLATED_STARTED',flush=True)\n"
        + indent(script, " ")
        + "\nfinally:\n"
        " _large_pending_error=sys.exc_info()[1]\n _large_cleanup_error=None\n"
        " for _large_stop in (stop_timings,stop_stacks):\n"
        "  if _large_stop is None:continue\n"
        "  try:_large_stop()\n"
        "  except BaseException as _large_error:\n"
        "   if _large_cleanup_error is None:_large_cleanup_error=_large_error\n"
        " if _large_pending_error is None and _large_cleanup_error is not None:raise _large_cleanup_error\n"
        "print('LARGE_ISOLATED_COMPLETED',flush=True)\n"
    )


@pytest.mark.parametrize("write_failure,stack_failure", [(True, False), (False, True), (True, True)])
def test_large_child_preserves_original_error_across_observer_cleanup(
    tmp_path, monkeypatch, write_failure, stack_failure
):
    from Tests.Backup_Recovery import thread_diagnostics as diagnostics
    from tldw_chatbook.Backup_Recovery import staging

    stopped = []
    original_copy = staging._copy
    original_error = ValueError("original restore failure")

    def stop_stacks():
        stopped.append(True)
        if stack_failure:
            raise RuntimeError("stack observer cleanup failure")

    real_write = diagnostics._write

    def write(path, records):
        if write_failure:
            raise OSError("diagnostic output unavailable")
        real_write(tmp_path / "timings.log", records)

    monkeypatch.setattr(diagnostics, "_write", write)
    monkeypatch.setattr(diagnostics, "observe_threads", lambda *a, **k: stop_stacks)
    with pytest.raises(ValueError) as caught:
        # Execute only the fixed test harness and literal test body.
        exec(_large_restore_script("raise original_error"), {"original_error": original_error})  # noqa: S102 # nosec B102
    assert caught.value is original_error  # nosec B101
    assert stopped == [True] and staging._copy is original_copy  # nosec B101


@pytest.mark.parametrize("stack_failure", [False, True])
def test_large_child_cleans_stack_observer_if_timing_setup_fails(monkeypatch, stack_failure):
    from Tests.Backup_Recovery import thread_diagnostics as diagnostics

    stopped = []
    original_error = ValueError("timing setup failure")

    def stop_stacks():
        stopped.append(True)
        if stack_failure:
            raise RuntimeError("stack observer cleanup failure")

    def start_timings(path):
        raise original_error

    monkeypatch.setattr(diagnostics, "observe_threads", lambda *a, **k: stop_stacks)
    monkeypatch.setattr(diagnostics, "observe_large_restore", start_timings)
    with pytest.raises(ValueError) as caught:
        exec(_large_restore_script("raise AssertionError('body ran')"))  # noqa: S102 # nosec B102
    assert caught.value is original_error and stopped == [True]  # nosec B101


@pytest.mark.parametrize("write_failure", [False, True])
def test_large_child_still_fails_for_observer_error_without_restore_error(
    tmp_path, monkeypatch, write_failure
):
    from Tests.Backup_Recovery import thread_diagnostics as diagnostics
    from tldw_chatbook.Backup_Recovery import staging

    stopped = []
    original_copy = staging._copy

    def stop_stacks():
        stopped.append(True)
        if not write_failure:
            raise RuntimeError("stack observer cleanup failure")

    def write(path, records):
        raise OSError("diagnostic output unavailable")

    if write_failure:
        monkeypatch.setattr(diagnostics, "_write", write)
    else:
        real_write = diagnostics._write
        monkeypatch.setattr(diagnostics, "_write", lambda path, rows: real_write(tmp_path / "timings.log", rows))
    monkeypatch.setattr(diagnostics, "observe_threads", lambda *a, **k: stop_stacks)
    expected = "large_restore_diagnostic_write_failed" if write_failure else "stack observer cleanup failure"
    with pytest.raises(RuntimeError, match=expected):
        exec(_large_restore_script("pass"))  # noqa: S102 # nosec B102
    assert stopped == [True] and staging._copy is original_copy  # nosec B101


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
    script = _large_restore_script(script)
    _run(tmp_path, "complete", "large-isolated", script=script, timeout=180)
