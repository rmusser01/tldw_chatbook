"""Native lifecycle of the elected writer's separately enrolled SQLite resource."""

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows writer lifecycle")


def test_opener_refuses_replacement_between_committed_receipt_and_local_binding(monkeypatch, tmp_path):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.DB import profile_writer as writer
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.profile_writer_selection import writer_file_identity
    from tldw_chatbook.Utils.platform_files import os as files

    profile = config.get_user_data_dir()
    path = profile / "agent_runs.db"
    original = writer._exchange
    observed = []

    def replace_after_open(*args, **kwargs):
        receipt = original(*args, **kwargs)
        if args[1] == "call" and args[2]["operation"] == "agent_runs.open":
            observed.append(tuple(receipt["database"]))
            writer.drain_profile_writer(profile)
            replacement = tmp_path / "replacement.sqlite"
            with closing(sqlite3.connect(path)) as source, closing(sqlite3.connect(replacement)) as target:
                source.backup(target)
            files.chmod(replacement, 0o600)
            os.replace(replacement, path)
        return receipt

    try:
        with monkeypatch.context() as changes:
            changes.setattr(writer, "_exchange", replace_after_open)
            with pytest.raises(RecoveryRequired, match="database_changed"):
                AgentRunsDB(path, reconcile_on_init=False)
        current = AgentRunsDB(path, reconcile_on_init=False)
        try:
            assert observed and current._writer_file != observed[0]
            assert current._writer_file == writer_file_identity(path)
        finally:
            current.close()
    finally:
        writer.drain_profile_writer(profile)


def _wait_file(path, process, timeout=12):
    deadline = time.monotonic() + timeout
    while not path.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(.025)
    assert path.exists(), (path.name, process.poll())


def _instrumented_service(profile, probe, mode):
    from tldw_chatbook.Backup_Recovery import participants, storage_admission as storage
    from tldw_chatbook.DB import profile_writer as writer
    from tldw_chatbook.DB.private_sqlite import _ordinary_connections

    accept = writer.pipe.accept
    require_retired = writer._require_retired

    def ready(*args):
        (probe / "elected").touch()
        return accept(*args)

    def observe(startup, idle=None):
        require_retired(startup, idle)
        if idle is not None:
            connection, = (conn for conn, lease in _ordinary_connections.items() if lease is idle)
            assert sqlite3.Connection.in_transaction.__get__(connection) is False
            assert idle.resource_thread is threading.current_thread()
            assert not storage._operations and not storage._raw_operations
            event = [id(connection), id(idle), threading.get_native_id(), len(storage._live_leases)]
            with (probe / "idle.jsonl").open("a") as log:
                print(json.dumps(event), file=log)

    # Apply the close hook only after stock initialization installed the real
    # SQLite resource. Importing the repository on the listener thread could
    # otherwise run config startup outside the very lifecycle under test.
    from tldw_chatbook.DB.profile_writer_repository import AgentRunsWriterDatabase

    retire = AgentRunsWriterDatabase.retire

    def retire_observed(owner):
        if owner.database is None:
            return retire(owner)
        current = threading.get_native_id()
        assert owner.lease.resource_thread.native_id == current
        if mode != "success" and (probe / "inject").exists():
            database = owner.database
            original_close = type(database).close

            def failed_close(db):
                assert db is database and owner.lease.resource_thread.native_id == threading.get_native_id()
                (probe / "close-entered").touch()
                end = time.monotonic() + 8
                while not (probe / "release-close").exists() and time.monotonic() < end:
                    time.sleep(.025)
                assert (probe / "release-close").exists()
                if mode == "raises":
                    raise sqlite3.OperationalError("injected native close failure")
                # An original close declines a real live borrowed transaction
                # on its locally closed participant. Its apparent None result
                # must never be mistaken for positive handle retirement.
                participants.repository_participant(db).close_admission()
                owner.connection.execute("BEGIN")
                original_close(db)
                assert sqlite3.Connection.in_transaction.__get__(owner.connection)

            type(database).close = failed_close
        retire(owner)
        (probe / "closed").write_text(str(current))

    writer.pipe.accept = ready
    writer._require_retired = observe
    AgentRunsWriterDatabase.retire = retire_observed
    return writer._serve(profile)


def _exercise(probe: Path, mode: str):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission
    from tldw_chatbook.Backup_Recovery.control_records import UNBOUND_NAMESPACE, admission_authority
    from tldw_chatbook.DB import profile_writer as writer
    from tldw_chatbook.DB.profile_writer_selection import writer_file_identity
    from tldw_chatbook.Utils.platform_files import os as files

    profile = config.get_user_data_dir()
    database = profile / "agent_runs.db"
    run = "from Tests.DB.test_profile_writer_connection import _instrumented_service as f; "
    source = "import sys; from pathlib import Path; " + run + "raise SystemExit(f(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]))"
    diagnostic = probe / "service.log"
    with diagnostic.open("w") as log:
        server = subprocess.Popen(
            [sys.executable, "-B", "-c", source, str(profile), str(probe), mode],
            cwd=Path(__file__).resolve().parents[2], stdout=log, stderr=log,
        )
    host = None
    try:
        _wait_file(probe / "elected", server)
        host = writer.start_profile_writer(profile)
        writer.call_profile_writer(profile, "agent_runs.open", {})

        def insert(name):
            return writer.call_profile_writer(profile, "agent_runs.create_agent_definition", {
                "defn": {"name": name, "instructions": "inspect", "description": "check"},
            })

        first = insert("original")
        # All committed work is visible during idle, and the provider may
        # wait indefinitely with exactly startup + one non-transactional lease.
        insert("second")
        events = [json.loads(line) for line in (probe / "idle.jsonl").read_text().splitlines()]
        assert len(events) >= 3 and len({tuple(event) for event in events}) == 1 and events[0][-1] == 2
        authority = admission_authority(bootstrap.default_bootstrap_root())
        if mode == "success":
            replacement = profile / "replacement.sqlite"
            with closing(sqlite3.connect(database)) as source_db, closing(sqlite3.connect(replacement)) as new:
                source_db.backup(new)
                new.execute("UPDATE agent_definitions SET description='restored' WHERE id=?", (first,))
                new.commit()
            files.chmod(replacement, 0o600)
            old_identity = writer_file_identity(database)
            storage_admission._shutdown()
            with authority.maintenance((UNBOUND_NAMESPACE,), 5):
                assert int((probe / "closed").read_text()) == events[0][2]
                os.replace(replacement, database)  # Windows refuses replacement with the old WAL handle live.
            assert old_identity != writer_file_identity(database)
            after = insert("after-restore")
            with closing(sqlite3.connect(database)) as observer:
                assert observer.execute("SELECT description FROM agent_definitions WHERE id=?", (first,)).fetchone() == ("restored",)
                assert observer.execute("SELECT name FROM agent_definitions WHERE id=?", (after,)).fetchone() == ("after-restore",)
                observer.execute("DROP INDEX idx_agent_definitions_name")
            insert("after-schema-change")
            with closing(sqlite3.connect(database)) as observer:
                assert observer.execute("SELECT name FROM sqlite_schema WHERE name='idx_agent_definitions_name'").fetchone()
            subprocess.run(["icacls", str(database), "/grant", "*S-1-1-0:(R)"],
                           check=True, capture_output=True)
            try:
                with pytest.raises(bootstrap.RecoveryRequired, match="database_unsafe"):
                    insert("unsafe-insert")
            finally:
                files.chmod(database, 0o600)
            assert server.wait(timeout=8) != 0
            insert("after-acl-repair")
            with closing(sqlite3.connect(database)) as observer:
                assert observer.execute("SELECT id FROM agent_definitions WHERE name='unsafe-insert'").fetchone() is None
        else:
            (probe / "inject").touch()
            storage_admission._shutdown()

            def capture():
                with authority.maintenance((UNBOUND_NAMESPACE,), 10):
                    return True

            with ThreadPoolExecutor(2) as executor:
                waiting = executor.submit(capture)
                _wait_file(probe / "close-entered", server)
                # The actual separate native owner still pins exclusivity here.
                assert not waiting.done() and server.poll() is None
                rejected = executor.submit(insert, "must-never-commit")
                time.sleep(.1)
                assert not rejected.done() and not waiting.done()
                (probe / "release-close").touch()
                assert server.wait(timeout=8) != 0
                with pytest.raises((writer.ProfileWriterOutcomeUnknown, bootstrap.RecoveryRequired)):
                    rejected.result(timeout=8)
                assert waiting.result(timeout=8)
            assert not (probe / "closed").exists()
            # The real registered host can elect a replacement after process
            # death. The failed original command and its close were not replayed.
            after = insert("after-loss")
            with closing(sqlite3.connect(database)) as observer:
                assert observer.execute("SELECT id FROM agent_definitions WHERE name='must-never-commit'").fetchone() is None
                assert observer.execute("SELECT name FROM agent_definitions WHERE id=?", (after,)).fetchone() == ("after-loss",)
    finally:
        if host is not None:
            host.close()
        if server.poll() is None:
            server.kill()
        server.wait(timeout=5)


@pytest.mark.parametrize("mode", ("success", "raises", "transaction"))
def test_original_thread_idle_connection_drains_before_file_replacement_or_owner_loss(tmp_path, mode):
    from tldw_chatbook.Utils.platform_files import os as files

    private = tmp_path / "owner-home"
    config = private / "config"
    files.mkdir(private, 0o700)
    files.mkdir(config, 0o700)
    env = dict(os.environ, HOME=str(private), USERPROFILE=str(private), LOGURU_LEVEL="ERROR",
               XDG_CONFIG_HOME=str(config), TLDW_CONFIG_PATH=str(config / "config.toml"))
    source = (
        "import sys; from pathlib import Path; from Tests.DB.test_profile_writer_connection import _exercise; "
        "_exercise(Path(sys.argv[1]),sys.argv[2])"
    )
    result = subprocess.run([sys.executable, "-B", "-c", source, str(private), mode],
                            cwd=Path(__file__).resolve().parents[2], env=env,
                            capture_output=True, text=True, timeout=55)
    if result.returncode:
        diagnostic = private / "service.log"
        tail = diagnostic.read_text(errors="replace") if diagnostic.exists() else ""
        pytest.fail(result.stderr.split("Traceback (most recent call last):")[-1][-1600:]
                    + (tail.split("Traceback (most recent call last):")[-1][-1000:] if "Traceback" in tail else ""))
