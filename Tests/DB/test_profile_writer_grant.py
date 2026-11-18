"""Real independent-process exclusion for original caller-owned FULL transactions."""

import os
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest

from .test_profile_writer_connection import _wait_file


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows profile writer grant")


def _caller(probe, mode):
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.profile_writer import ProfileWriterOutcomeUnknown
    from tldw_chatbook.DB.profile_writer_grant import _local_agent_transaction
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.DB.profile_writer_grant import local_agent_transaction

    db = AgentRunsDB(config.get_user_data_dir() / "agent_runs.db", reconcile_on_init=False)
    with pytest.raises(RecoveryRequired, match="source_changed"):
        local_agent_transaction(db, "goal.checkpoint")  # A label alone grants no authority.
    prior = None
    if mode == "borrowed":
        with db.connection() as prior:
            pass
    storage._shutdown()
    try:
        try:
            with _local_agent_transaction(db, "goal.checkpoint") as grant:
                with db.connection() as connection:
                    grant.bind(connection)
                    previous = int(connection.execute("PRAGMA synchronous").fetchone()[0])
                    connection.execute("PRAGMA synchronous=FULL")
                    connection.execute("BEGIN IMMEDIATE")
                    try:
                        connection.execute("UPDATE agent_definitions SET description='local' WHERE name='original'")
                        (probe / "in-transaction").write_text(str(os.getpid()))
                        deadline = time.monotonic() + 20
                        while not (probe / "finish").exists() and time.monotonic() < deadline:
                            time.sleep(.025)
                        assert (probe / "finish").exists()
                        if mode == "uncertain":
                            grant.uncertain()
                            raise RuntimeError("simulated unconfirmed original receipt")
                        connection.commit()
                    except BaseException:
                        connection.rollback()
                        raise
                    finally:
                        connection.execute(f"PRAGMA synchronous={previous}")
                grant.retire(connection)
        except ProfileWriterOutcomeUnknown:
            assert mode in ("writer-crash", "uncertain")
            (probe / "unknown").touch()
        if mode == "borrowed":
            assert db._thread_local.conn is prior and not prior.in_transaction
        elif mode != "uncertain":
            with pytest.raises(sqlite3.ProgrammingError):
                sqlite3.Connection.in_transaction.__get__(connection)
        if mode == "uncertain":
            assert (probe / "unknown").exists()
            time.sleep(18)  # The coordinator observes custody and terminates this exact process.
    finally:
        db.close()


def _coordinator(probe, mode):
    import _winapi
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission
    from tldw_chatbook.Backup_Recovery.admission import AdmissionTimeout
    from tldw_chatbook.Backup_Recovery.control_records import UNBOUND_NAMESPACE, admission_authority
    from tldw_chatbook.DB import profile_writer as writer
    from Tests.DB.test_profile_writer_transport import _wait_for_owned_server

    profile = config.get_user_data_dir()
    database = profile / "agent_runs.db"
    host = writer.start_profile_writer(profile)
    worker = None
    process = caller_process = None
    log = (probe / "caller.log").open("w")
    try:
        writer.call_profile_writer(profile, "agent_runs.open", {})

        def insert(name):
            return writer.call_profile_writer(profile, "agent_runs.create_agent_definition", {
                "defn": {"name": name, "instructions": "inspect"},
            })

        first = insert("original")
        _, process = _wait_for_owned_server(profile, terminate=True)
        authority = admission_authority(bootstrap.default_bootstrap_root())
        storage_admission._shutdown()
        program = "import sys; from pathlib import Path; from Tests.DB.test_profile_writer_grant import _caller; _caller(Path(sys.argv[1]),sys.argv[2])"
        worker = subprocess.Popen([sys.executable, "-B", "-c", program, str(probe), mode],
                                  cwd=Path(__file__).resolve().parents[2], stdout=log, stderr=log)
        _wait_file(probe / "in-transaction", worker)
        # A venv launcher and its real interpreter need not have the same PID.
        # Pin the writer-grant pipe's actual original server for crash tests.
        if mode in ("client-crash", "uncertain"):
            from tldw_chatbook.DB.profile_writer_grant import _guard_name
            from tldw_chatbook.DB.profile_writer_selection import writer_file_identity

            name = _guard_name(writer_file_identity(database))
            end = time.monotonic() + 4
            while True:
                try:
                    with writer.pipe.client(name) as guarded:
                        pid = writer.pipe.server_pid(guarded)
                        assert pid == int((probe / "in-transaction").read_text())
                        caller_process = _winapi.OpenProcess(_winapi.SYNCHRONIZE | 1, False, pid)
                    break
                except OSError as error:
                    if error.winerror != 231 or time.monotonic() >= end:
                        raise
                    time.sleep(.025)
        with closing(sqlite3.connect(database)) as independent:
            assert independent.execute("SELECT description FROM agent_definitions WHERE id=?", (first,)).fetchone() == ("",)
        with pytest.raises(AdmissionTimeout):
            with authority.maintenance((UNBOUND_NAMESPACE,), .2):
                pytest.fail("exclusive recovery entered the live original transaction")
        if mode == "writer-crash":
            _winapi.TerminateProcess(process, 57)
            assert _winapi.WaitForSingleObject(process, 5000) == _winapi.WAIT_OBJECT_0
            with pytest.raises(bootstrap.RecoveryRequired, match="transaction_active"):
                insert("during-transaction")
            _, successor = _wait_for_owned_server(profile, terminate=True)
            try:
                _winapi.TerminateProcess(successor, 58)
                assert _winapi.WaitForSingleObject(successor, 5000) == _winapi.WAIT_OBJECT_0
            finally:
                _winapi.CloseHandle(successor)
            with pytest.raises(bootstrap.RecoveryRequired, match="transaction_active"):
                insert("during-transaction")
        if mode == "client-crash":
            _winapi.TerminateProcess(caller_process, 59)
            assert _winapi.WaitForSingleObject(caller_process, 5000) == _winapi.WAIT_OBJECT_0
        elif mode == "uncertain":
            (probe / "finish").touch()
            _wait_file(probe / "unknown", worker)
            with pytest.raises(bootstrap.RecoveryRequired, match="transaction_active"):
                insert("during-transaction")
            with pytest.raises(AdmissionTimeout):
                with authority.maintenance((UNBOUND_NAMESPACE,), .2):
                    pytest.fail("exclusive recovery entered unconfirmed custody")
            _winapi.TerminateProcess(caller_process, 60)
            assert _winapi.WaitForSingleObject(caller_process, 5000) == _winapi.WAIT_OBJECT_0
        else:
            with ThreadPoolExecutor(1) as executor:
                queued = executor.submit(insert, "ordered") if mode != "writer-crash" else None
                if queued is not None:
                    time.sleep(.15)
                    assert not queued.done()
                (probe / "finish").touch()
                if queued is not None:
                    assert queued.result(timeout=12)
        code = worker.wait(timeout=12)
        if mode not in ("client-crash", "uncertain"):
            assert code == 0, (probe / "caller.log").read_text(errors="replace")[-1600:]
        assert insert("after-transaction")
        with authority.maintenance((UNBOUND_NAMESPACE,), 4), closing(sqlite3.connect(database)) as db:
            expected = "" if mode in ("client-crash", "uncertain") else "local"
            assert db.execute("SELECT description FROM agent_definitions WHERE id=?", (first,)).fetchone() == (expected,)
            assert db.execute("SELECT id FROM agent_definitions WHERE name='during-transaction'").fetchone() is None
            with closing(sqlite3.connect(":memory:")) as backup:
                db.backup(backup)
                backup.execute("UPDATE agent_definitions SET description='restored' WHERE id=?", (first,))
                backup.commit()
                backup.backup(db)
        assert insert("after-restore")
        with closing(sqlite3.connect(database)) as db:
            assert db.execute("SELECT description FROM agent_definitions WHERE id=?", (first,)).fetchone() == ("restored",)
    finally:
        if worker is not None and worker.poll() is None:
            worker.kill()
            worker.wait(timeout=5)
        if process is not None:
            _winapi.CloseHandle(process)
        if caller_process is not None:
            _winapi.CloseHandle(caller_process)
        log.close()
        host.close()


@pytest.mark.parametrize("mode", ("new", "borrowed", "client-crash", "writer-crash", "uncertain"))
def test_local_transaction_reserves_physical_file_across_writer_or_caller_loss(tmp_path, mode):
    from tldw_chatbook.Utils.platform_files import os as files

    private = tmp_path / "grant-home"
    config = private / "config"
    files.mkdir(private, 0o700)
    files.mkdir(config, 0o700)
    env = dict(os.environ, HOME=str(private), USERPROFILE=str(private), LOGURU_LEVEL="ERROR",
               XDG_CONFIG_HOME=str(config), TLDW_CONFIG_PATH=str(config / "config.toml"))
    program = "import sys; from pathlib import Path; from Tests.DB.test_profile_writer_grant import _coordinator; _coordinator(Path(sys.argv[1]),sys.argv[2])"
    result = subprocess.run([sys.executable, "-B", "-c", program, str(private), mode],
                            cwd=Path(__file__).resolve().parents[2], env=env,
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        tail = private / "caller.log"
        pytest.fail(result.stderr.split("Traceback (most recent call last):")[-1][-1500:]
                    + (tail.read_text(errors="replace")[-1200:] if tail.exists() else ""))
