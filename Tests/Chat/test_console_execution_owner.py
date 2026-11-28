"""The executing process, not its parent or a PID file, owns the run store."""

import subprocess
import sys

from tldw_chatbook.Chat.console_execution_owner import (
    ConsoleExecutionBusy,
    ConsoleExecutionOwner,
)


def test_other_process_cannot_execute_the_same_store(tmp_path):
    path = tmp_path / "agent_runs.db"
    program = (
        "import sys\n"
        "from pathlib import Path\n"
        "from tldw_chatbook.Chat.console_execution_owner import "
        "ConsoleExecutionOwner, ConsoleExecutionBusy\n"
        "try:\n"
        "    ConsoleExecutionOwner.acquire(Path(sys.argv[1]))\n"
        "except ConsoleExecutionBusy:\n"
        "    raise SystemExit(19)\n"
        "raise SystemExit(0)\n"
    )
    with ConsoleExecutionOwner.acquire(path):
        result = subprocess.run(
            [sys.executable, "-c", program, str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    assert result.returncode == 19, result.stderr
    with ConsoleExecutionOwner.acquire(path):
        pass


import os

import pytest

from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionUnavailable


def test_unowned_disk_constructor_cannot_create_or_sweep(tmp_path):
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    path = tmp_path / "agent_runs.db"
    with pytest.raises(ConsoleExecutionUnavailable):
        AgentRunsDB(path)
    assert not path.exists()


def test_read_only_open_never_migrates_or_sweeps_and_rejects_writes(tmp_path):
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    path = tmp_path / "agent_runs.db"
    with ConsoleExecutionOwner.acquire(path) as owner:
        db = AgentRunsDB(path, execution_owner=owner)
        run_id = db.create_run(conversation_id="c", agent_kind="primary")
        with db.connection() as conn:
            before = tuple(conn.execute("SELECT * FROM schema_version"))
        reader = AgentRunsDB.open_read_only(path)
        try:
            assert reader.get_run(run_id)["status"] == "running"
            with pytest.raises(ConsoleExecutionUnavailable):
                reader.reconcile_orphaned_runs()
            with pytest.raises(ConsoleExecutionUnavailable):
                reader.create_run(conversation_id="c", agent_kind="primary")
            with pytest.raises(ConsoleExecutionUnavailable):
                reader._initialize_schema()
            with pytest.raises(ConsoleExecutionUnavailable):
                reader.vacuum()
            with reader.connection() as conn:
                assert tuple(conn.execute("SELECT * FROM schema_version")) == before
                import sqlite3

                with pytest.raises(sqlite3.OperationalError):
                    conn.execute("DELETE FROM agent_runs")
            assert db.get_run(run_id)["status"] == "running"
        finally:
            reader.close()
            db.close()


def test_closed_owner_cannot_write_or_reconcile_with_cached_connection(tmp_path):
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    path = tmp_path / "agent_runs.db"
    owner = ConsoleExecutionOwner.acquire(path)
    db = AgentRunsDB(path, execution_owner=owner)
    run_id = db.create_run(conversation_id="c", agent_kind="primary")
    owner.close()
    try:
        with pytest.raises(ConsoleExecutionUnavailable):
            db.set_status(run_id, "done")
        with pytest.raises(ConsoleExecutionUnavailable):
            db.reconcile_orphaned_runs()
    finally:
        db.close()


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory"])
def test_store_alias_or_non_file_is_refused(tmp_path, kind):
    path = tmp_path / "agent_runs.db"
    original = tmp_path / "original.db"
    original.touch(mode=0o600)
    if kind == "symlink":
        path.symlink_to(original)
    elif kind == "hardlink":
        os.link(original, path)
    else:
        path.mkdir()
    with pytest.raises(ConsoleExecutionUnavailable):
        ConsoleExecutionOwner.acquire(path)


def test_unsupported_lock_is_unavailable(monkeypatch, tmp_path):
    import errno

    import tldw_chatbook.Chat.console_execution_owner as module

    def unsupported(*args):
        raise OSError(errno.ENOTSUP, "unsupported")

    monkeypatch.setattr(module.fcntl, "flock", unsupported)
    with pytest.raises(ConsoleExecutionUnavailable):
        ConsoleExecutionOwner.acquire(tmp_path / "agent_runs.db")


def test_actual_holder_death_releases_lease_even_if_lock_file_remains(tmp_path):
    import signal

    path = tmp_path / "agent_runs.db"
    program = """
import sys
from pathlib import Path
from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionOwner
owner = ConsoleExecutionOwner.acquire(Path(sys.argv[1]))
print('ready', flush=True)
sys.stdin.readline()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", program, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises(ConsoleExecutionBusy):
            ConsoleExecutionOwner.acquire(path)
        child.send_signal(signal.SIGKILL)
        child.wait(timeout=10)
        with ConsoleExecutionOwner.acquire(path):
            pass
        assert (tmp_path / ".agent_runs.db.execution.lock").exists()
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)


def test_surviving_executing_child_retains_lease_after_parent_exit(tmp_path):
    path = tmp_path / "agent_runs.db"
    child_program = """
import sys
from pathlib import Path
from tldw_chatbook.Chat.console_execution_owner import ConsoleExecutionOwner
with ConsoleExecutionOwner.acquire(Path(sys.argv[1])):
    print('ready', flush=True)
    sys.stdin.readline()
print('retired', flush=True)
"""
    parent_program = 'import subprocess, sys\nsubprocess.Popen([sys.executable, "-c", sys.argv[1], sys.argv[2]])'
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_program, child_program, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert parent.stdout.readline().strip() == "ready"
        assert parent.wait(timeout=10) == 0
        with pytest.raises(ConsoleExecutionBusy):
            ConsoleExecutionOwner.acquire(path)
        parent.stdin.write("retire\n")
        parent.stdin.flush()
        assert parent.stdout.readline().strip() == "retired"
        with ConsoleExecutionOwner.acquire(path):
            pass
    finally:
        parent.communicate(timeout=10)


def test_replaced_lock_or_store_revokes_existing_owner(tmp_path):
    path = tmp_path / "agent_runs.db"
    path.touch(mode=0o600)
    with ConsoleExecutionOwner.acquire(path) as owner:
        replacement = tmp_path / "replacement.db"
        replacement.touch(mode=0o600)
        replacement.replace(path)
        with pytest.raises(ConsoleExecutionUnavailable):
            owner.require_current(path)


def test_user_owned_parent_symlink_is_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ConsoleExecutionUnavailable):
        ConsoleExecutionOwner.acquire(alias / "agent_runs.db")


def test_owned_legacy_mode_is_secured_only_after_lease_admission(tmp_path):
    import stat

    path = tmp_path / "agent_runs.db"
    path.write_bytes(b"legacy")
    path.chmod(0o644)
    with ConsoleExecutionOwner.acquire(path):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert path.read_bytes() == b"legacy"


def test_owner_close_waits_for_physical_sqlite_operation(tmp_path):
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    path = tmp_path / "agent_runs.db"
    owner = ConsoleExecutionOwner.acquire(path)
    db = AgentRunsDB(path, execution_owner=owner)
    try:
        with db.transaction() as conn:
            owner.close()
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; from pathlib import Path; "
                        "from tldw_chatbook.Chat.console_execution_owner import *\n"
                        "try: ConsoleExecutionOwner.acquire(Path(sys.argv[1]))\n"
                        "except ConsoleExecutionBusy: raise SystemExit(19)"
                    ),
                    str(path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert probe.returncode == 19, probe.stderr
            conn.execute("INSERT INTO schema_version(version) VALUES (12345)")
        with ConsoleExecutionOwner.acquire(path):
            pass
    finally:
        db.close()
        owner.close()


def test_read_only_keeps_legacy_file_mode_contents_and_schema(tmp_path):
    import hashlib
    import stat

    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    path = tmp_path / "agent_runs.db"
    with ConsoleExecutionOwner.acquire(path) as owner:
        db = AgentRunsDB(path, execution_owner=owner)
        run_id = db.create_run(conversation_id="saved", agent_kind="primary")
        db.close()
    path.chmod(0o644)
    before = (hashlib.sha256(path.read_bytes()).digest(), path.stat().st_mtime_ns)
    reader = AgentRunsDB.open_read_only(path)
    try:
        assert reader.get_run(run_id)["status"] == "running"
        with pytest.raises(ConsoleExecutionUnavailable):
            reader.mark_wake_delivered([])
        with (
            pytest.raises(ConsoleExecutionUnavailable),
            reader.automatic_work.transaction(),
        ):
            pass
    finally:
        reader.close()
    assert (
        hashlib.sha256(path.read_bytes()).digest(),
        path.stat().st_mtime_ns,
    ) == before
    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_missing_store_is_not_created_by_lease_or_read_only_open(tmp_path):
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    path = tmp_path / "agent_runs.db"
    with ConsoleExecutionOwner.acquire(path):
        assert not path.exists()
    with pytest.raises(OSError):
        AgentRunsDB.open_read_only(path)
    assert not path.exists()


def test_unregistered_owner_cannot_confer_execution_authority(tmp_path):
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.Utils.private_paths import open_private_lock_stream

    path = tmp_path / "agent_runs.db"
    with open_private_lock_stream(tmp_path / ".agent_runs.db.execution.lock") as stream:
        owner = ConsoleExecutionOwner(path, stream)
        with pytest.raises(ConsoleExecutionUnavailable):
            AgentRunsDB(path, execution_owner=owner)
    assert not path.exists()
