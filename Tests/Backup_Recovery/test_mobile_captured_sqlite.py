"""Real encrypted capture reads retain native authority and physical custody."""

import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import Event

import pytest

from Tests.Backup_Recovery.test_mobile_consolidation_lifecycle import _capture_fixture


def _fingerprints(root):
    return {
        str(path.relative_to(root)): (
            path.stat().st_mode,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in root.rglob("*")
        if path.is_file()
    }


@contextmanager
def _observe_physical_retirement(monkeypatch):
    import errno
    import sys
    import threading

    from tldw_chatbook.Backup_Recovery import (
        crypto,
        mobile_captured_sqlite,
        storage_admission,
    )
    from tldw_chatbook.DB import private_sqlite

    close = storage_admission._CaptureFileDescriptors.close_descriptor
    retire = crypto._CryptoJob.retire
    connect = private_sqlite.connect_private_sqlite
    descriptors, jobs, references, dispatches = [], [], [], []

    def close_descriptor(custody, fd):
        is_read = type(custody.scope) is mobile_captured_sqlite._CapturedSQLite
        before = storage_admission.os.fstat(fd) if is_read else None
        result = close(custody, fd)
        if is_read:
            try:
                after = storage_admission.os.fstat(fd)
            except OSError as error:
                assert error.errno == errno.EBADF
            else:
                assert (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
            descriptors.append((fd, before.st_dev, before.st_ino))
        return result

    def retire_job(job):
        result = retire(job)
        if type(job.scope) is mobile_captured_sqlite._CapturedSQLite:
            assert job.process.poll() is not None
            assert all(not worker.is_alive() for worker in job.workers)
            assert all(
                pipe.closed
                for pipe in (job.process.stdin, job.process.stdout, job.process.stderr)
            )
            jobs.append(job)
        return result

    def reference(owner_id, *args, **kwargs):
        connection = connect(owner_id, *args, **kwargs)
        if owner_id == "recovery.validation_schema":
            references.append(connection)
        return connection

    def dispatch(frame, event, arg):
        if event == "call" and frame.f_globals.get("__name__", "").startswith(
            ("tldw_chatbook.LLM_Calls.", "tldw_chatbook.Tools.")
        ):
            dispatches.append((frame.f_globals["__name__"], frame.f_code.co_name))

    old_profile, old_thread_profile = sys.getprofile(), threading.getprofile()
    with monkeypatch.context() as observe:
        observe.setattr(
            storage_admission._CaptureFileDescriptors,
            "close_descriptor",
            close_descriptor,
        )
        observe.setattr(crypto._CryptoJob, "retire", retire_job)
        observe.setattr(private_sqlite, "connect_private_sqlite", reference)
        sys.setprofile(dispatch)
        threading.setprofile(dispatch)
        try:
            yield
        finally:
            sys.setprofile(old_profile)
            threading.setprofile(old_thread_profile)
    assert descriptors
    assert len(jobs) == 4
    assert len(references) >= 6
    for connection in references:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
    assert dispatches == []


@pytest.mark.parametrize("distinct_selectors", (False, True))
def test_actual_captured_sqlite_reads_preserve_frozen_schemas_and_bytes(
    tmp_path, monkeypatch, distinct_selectors
):
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )

    with _capture_fixture(
        tmp_path, monkeypatch, distinct_selectors=distinct_selectors
    ) as values:
        session, journal, _, _, preview, inventory, _ = values
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=output,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            before = _fingerprints(tmp_path / "profile")
            with (
                _observe_physical_retirement(monkeypatch),
                captured.sqlite_reads(
                    password=b"test-private-mobile-rollback", cancel=Event()
                ) as reads,
            ):
                assert reads.versions == (
                    ("legacy", "chat", 81),
                    ("legacy", "workspaces", 8),
                    ("legacy", "runs", 25),
                    ("canonical", "chat", 82),
                    ("canonical", "workspaces", 9),
                    ("canonical", "runs", 26),
                )
                columns, rows = reads.page("legacy", "chat", "messages", limit=100)
                assert "content" in columns
                assert "private fixture prompt" in [
                    row[columns.index("content")] for row in rows
                ]
                _, blobs = reads.page(
                    "legacy", "chat", "console_trace_artifacts", limit=100
                )
                assert b'"private trace artifact"' in {
                    value for row in blobs for value in row if isinstance(value, bytes)
                }
                _, receipts = reads.page(
                    "canonical", "chat", "mobile_consolidation_batches", limit=100
                )
                assert receipts == ()
                from concurrent.futures import ThreadPoolExecutor
                from copy import copy

                with pytest.raises(ValueError, match="inactive"):
                    copy(reads).page("legacy", "chat", "messages")
                with (
                    ThreadPoolExecutor(max_workers=1) as executor,
                    pytest.raises((ValueError, RuntimeError), match="inactive"),
                ):
                    executor.submit(reads.page, "legacy", "chat", "messages").result()
                with (
                    pytest.raises(ValueError, match="inactive"),
                    copy(captured).sqlite_reads(
                        password=b"test-private-mobile-rollback", cancel=Event()
                    ),
                ):
                    raise AssertionError("copied capture gained authority")
                connections = tuple(
                    lease.connection
                    for lease in reads.resources
                    if hasattr(lease, "connection") and lease.connection is not None
                )
                assert len(connections) == 6
                with pytest.raises(ValueError, match="limit"):
                    reads.page("legacy", "chat", "messages", limit=1001)
                with pytest.raises(ValueError, match="table"):
                    reads.page("legacy", "chat", "messages; DELETE FROM messages")
            assert reads.resources == []
            for connection in connections:
                with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                    connection.execute("SELECT 1")
            with pytest.raises(ValueError, match="inactive"):
                reads.page("legacy", "chat", "messages")
            assert not tuple(output.glob(".mobile-read-*"))
            assert _fingerprints(tmp_path / "profile") == before


def _physical_fault(root, phase, monkeypatch):
    """Child-only fault: retain quarantine until process exit, never fake cleanup."""
    import json
    import subprocess
    import sys
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery import (
        archive_reader,
        crypto,
        mobile_captured_sqlite,
        sqlite_validation,
        storage_admission,
    )
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )
    from tldw_chatbook.DB import private_sqlite

    reached = {}
    native_close = archive_reader.os.close
    native_connect = private_sqlite.connect_private_sqlite
    native_restrict = sqlite_validation._restrict_connection
    native_acquire = archive_reader.acquire
    native_popen = crypto.subprocess.Popen
    native_retire_staging = mobile_captured_sqlite._retire_staging
    native_retire_output = crypto._CryptoOutput.retire
    native_sqlite = storage_admission.sqlite3
    processes = []

    def acquire(source, *args, **kwargs):
        scope = getattr(storage_admission._local, "mobile_sqlite_scope", None)
        if scope is not None:
            reached["scope"] = scope
            reached["armed"] = True
            selected = (
                source
                if phase == "acquire"
                else crypto._package_resource_root() / "age_worker.py"
            )
            info = selected.stat()
            reached["selected"] = info.st_dev, info.st_ino
        return native_acquire(source, *args, **kwargs)

    def retire_staging(scope, path, identity):
        if phase == "cleanup_close" and "fd" not in reached:
            reached["cleaning"] = True
            reached["selected"] = identity
        try:
            return native_retire_staging(scope, path, identity)
        finally:
            reached["cleaning"] = False

    def retire_output(owner):
        if phase == "output_close" and "fd" not in reached:
            reached["output_fd"] = owner.parent
            reached["selected"] = owner.identity
            reached["retiring_output"] = True
        try:
            return native_retire_output(owner)
        finally:
            reached["retiring_output"] = False

    def close(fd):
        if (
            reached.get("retiring_output")
            and fd == reached.get("output_fd")
            and "fd" not in reached
        ):
            reached["fd"] = fd
            raise OSError("injected exact output parent close")
        if (
            reached.get("armed")
            and (phase in {"acquire", "helper"} or reached.get("cleaning"))
            and "fd" not in reached
        ):
            info = archive_reader.os.fstat(fd)
            if (info.st_dev, info.st_ino) == reached["selected"]:
                reached["fd"] = fd
                raise OSError("injected reached native close")
        return native_close(fd)

    def popen(*args, **kwargs):
        process = native_popen(*args, **kwargs)
        if reached.get("armed"):
            processes.append(process)
            if phase == "decrypt_pipe" and args[0][-1] == "decrypt":
                pipe = process.stdin
                reached["pipe"] = pipe
                reached["fd"] = pipe.fileno()
                info = archive_reader.os.fstat(pipe.fileno())
                reached["selected"] = info.st_dev, info.st_ino

                def fail_pipe_close():
                    raise OSError("injected actual decrypt stdin close")

                monkeypatch.setattr(pipe, "close", fail_pipe_close)
        return process

    def connect(owner_id, *args, **kwargs):
        connection = native_connect(owner_id, *args, **kwargs)
        if (
            reached.get("armed")
            and phase == "reference_close"
            and owner_id == "recovery.validation_schema"
        ):
            reached["connection"] = connection
        return connection

    def restrict(connection, *args, **kwargs):
        if (
            reached.get("armed")
            and phase == "validation_setup"
            and any(
                getattr(lease, "connection", None) is connection
                for lease in reached["scope"].resources
            )
        ):
            reached["connection"] = connection
            raise ValueError("injected restriction setup failure before yield")
        return native_restrict(connection, *args, **kwargs)

    class NativeClose:
        @staticmethod
        def close(connection):
            if connection is reached.get("connection"):
                reached["close_reached"] = True
                raise sqlite3.OperationalError("injected before native SQLite close")
            return native_sqlite.Connection.close(connection)

    with (
        pytest.raises(RuntimeError, match="capture_resources_not_retired"),
        _capture_fixture(root, monkeypatch) as values,
    ):
        session, journal, _, _, preview, inventory, authority = values
        output = root / "archives"
        output.mkdir(mode=0o700)
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=output,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            monkeypatch.setattr(
                mobile_captured_sqlite, "_retire_staging", retire_staging
            )
            monkeypatch.setattr(crypto._CryptoOutput, "retire", retire_output)
            monkeypatch.setattr(archive_reader, "acquire", acquire)
            monkeypatch.setattr(archive_reader.os, "close", close)
            monkeypatch.setattr(crypto.subprocess, "Popen", popen)
            monkeypatch.setattr(private_sqlite, "connect_private_sqlite", connect)
            monkeypatch.setattr(sqlite_validation, "_restrict_connection", restrict)
            monkeypatch.setattr(
                storage_admission,
                "sqlite3",
                SimpleNamespace(**{**vars(native_sqlite), "Connection": NativeClose}),
            )
            with captured.sqlite_reads(
                password=b"test-private-mobile-rollback", cancel=Event()
            ) as reads:
                assert phase in {"read_close", "cleanup_close", "output_close"}
                reads.page("legacy", "chat", "messages")
                if phase == "read_close":
                    reached["connection"] = reads._connections["legacy", "chat"][0]
    scope = reached["scope"]
    assert scope.resources
    assert all(path.exists() for path, _ in scope.roots)
    if phase != "cleanup_close":
        assert any(tuple(path.rglob("input")) for path, _ in scope.roots)
    proof = {
        "phase": phase,
        "scope": id(scope),
        "staging": [str(path) for path, _ in scope.roots],
    }
    if "fd" in reached:
        actual = archive_reader.os.fstat(reached["fd"])
        assert (actual.st_dev, actual.st_ino) == reached["selected"]
        proof.update(fd=reached["fd"], device=actual.st_dev, inode=actual.st_ino)
    else:
        assert reached["close_reached"]
        assert reached["connection"].execute("SELECT 1").fetchone() == (1,)
        proof["connection"] = id(reached["connection"])
    assert all(process.poll() is not None for process in processes)
    proof["children"] = [process.pid for process in processes]
    jobs = [
        resource
        for resource in scope.resources
        if isinstance(resource, crypto._CryptoJob)
    ]
    assert all(not worker.is_alive() for job in jobs for worker in job.workers)
    proof["threads"] = [worker.ident for job in jobs for worker in job.workers]
    assert storage_admission._failed_capture_holds
    # A different process must still be unable to obtain actual maintenance.
    probe = """from pathlib import Path
from Tests.real_profile_guard import install
install()
from tldw_chatbook.Backup_Recovery.control_records import admission_authority
from tldw_chatbook.Backup_Recovery.admission import AdmissionError
import sys
try:
 with admission_authority(Path(sys.argv[1])).maintenance(('bootstrap.unbound','mobile.profile'),0.1): pass
except AdmissionError:
 sys.exit(23)
raise AssertionError('live fault released native maintenance')
"""
    result = subprocess.run(
        [sys.executable, "-c", probe, str(authority.control_root.parent)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 23, result.stdout + result.stderr
    import os

    proof["control_root"] = str(authority.control_root.parent)
    Path(os.environ["TLDW_R_FAULT_PROOF"]).write_text(json.dumps(proof))


@pytest.mark.parametrize(
    "phase",
    [
        "acquire",
        "helper",
        "decrypt_pipe",
        "validation_setup",
        "reference_close",
        "read_close",
        "cleanup_close",
        "output_close",
    ],
)
def test_actual_resource_fault_keeps_native_exclusion_until_process_exit(
    tmp_path, phase
):
    import json
    import os
    import subprocess
    import sys

    child_root = tmp_path / "child"
    child_root.mkdir(mode=0o700)
    proof_path = child_root / "proof.json"
    environment = dict(
        os.environ, TLDW_R_FAULT_PHASE=phase, TLDW_R_FAULT_PROOF=str(proof_path)
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "--tb=short",
            "--show-capture=no",
            "--basetemp",
            str(child_root / "pytest"),
            "Tests/Backup_Recovery/test_mobile_captured_sqlite.py::test_native_fault_child",
        ],
        env=environment,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    proof = json.loads(proof_path.read_text())
    assert proof["phase"] == phase
    print(json.dumps(proof))
    from tldw_chatbook.Backup_Recovery.control_records import admission_authority

    with admission_authority(Path(proof["control_root"])).maintenance(
        ("bootstrap.unbound", "mobile.profile"), 2
    ):
        pass


def test_native_fault_child(tmp_path, monkeypatch):
    import os

    phase = os.environ.get("TLDW_R_FAULT_PHASE")
    if phase is None:
        pytest.skip("only executed in a private native-quarantine child")
    _physical_fault(tmp_path, phase, monkeypatch)
    # The proof precedes physical process retirement; pytest teardown cannot
    # retry deliberately uncertain descriptors or clear the native hold.
    os._exit(0)


def test_read_staging_substitution_is_retained_not_removed(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import mobile_captured_sqlite, native_files
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )

    with _capture_fixture(tmp_path, monkeypatch) as values:
        session, journal, _, _, preview, inventory, _ = values
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=output,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            changed = []
            cleaning = []
            native_close = native_files._close_owned
            native_retire = mobile_captured_sqlite._retire_staging

            def retire(*args):
                cleaning.append(True)
                try:
                    return native_retire(*args)
                finally:
                    cleaning.pop()

            monkeypatch.setattr(mobile_captured_sqlite, "_retire_staging", retire)
            try:
                with (
                    pytest.raises(ValueError, match="staging_changed"),
                    captured.sqlite_reads(
                        password=b"test-private-mobile-rollback", cancel=Event()
                    ) as reads,
                ):
                    root, identity = reads.roots[0]

                    def substitute_after_root_close(fd, custody):
                        before = native_files.os.fstat(fd)
                        native_close(fd, custody)
                        if (
                            cleaning
                            and (before.st_dev, before.st_ino) == identity
                            and not changed
                        ):
                            retained = root.with_name(root.name + "-retained")
                            root.rename(retained)
                            root.mkdir(mode=0o700)
                            changed.append((root, retained, root.stat().st_ino))

                    monkeypatch.setattr(
                        native_files, "_close_owned", substitute_after_root_close
                    )
                assert len(changed) == 1
                assert changed[0][0].stat().st_ino == changed[0][2]
            finally:
                # Restore only this test's named substitution after witnessing
                # retention; no descriptor retry or admission flag manipulation.
                for foreign, retained, inode in changed:
                    if foreign.exists():
                        assert foreign.stat().st_ino == inode
                        foreign.rmdir()
                    retained.rename(foreign)


@pytest.mark.parametrize("distinct_selectors", (False, True))
def test_native_read_authority_and_integrity_controls(
    tmp_path, monkeypatch, distinct_selectors
):
    from copy import copy

    from tldw_chatbook.Backup_Recovery import (
        bootstrap,
        control_records,
        crypto,
        mobile_captured_sqlite,
    )
    from tldw_chatbook.Backup_Recovery.inventory import discover
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        _preview_selected_mobile,
    )
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )

    with _capture_fixture(tmp_path, monkeypatch) as values:
        session, journal, source, _, _, _, authority = values
        target = source.with_name("target.toml") if distinct_selectors else source
        selectors = tuple(dict.fromkeys((source, target)))
        candidate_selector = source.with_name("candidate.toml")
        for selector in (*selectors[1:], candidate_selector):
            selector.write_bytes(source.read_bytes())
            selector.chmod(0o600)
            control_records.bind_profile(
                authority.control_root.parent,
                selector,
                ("mobile.profile",),
                authority.control_root,
                session=session,
            )
        with session._discovery_reads(private_sqlite=True):
            inventory = discover(selectors)
            preview = _preview_selected_mobile(source, target)
        assert {source, target, candidate_selector} <= {
            row.selector for row in preview.selector_provenance
        }
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=output,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            before = _fingerprints(tmp_path / "profile")
            cancelled = Event()
            cancelled.set()
            with (
                pytest.raises(InterruptedError, match="cancelled"),
                captured.sqlite_reads(
                    password=b"test-private-mobile-rollback", cancel=cancelled
                ),
            ):
                pytest.fail("cancelled scope admitted")
            with (
                pytest.raises(crypto.CryptoError),
                captured.sqlite_reads(password=b"incorrect password", cancel=Event()),
            ):
                pytest.fail("unauthenticated scope admitted")
            assert not tuple(output.glob(".mobile-read-*"))
            cancellation = Event()
            with captured.sqlite_reads(
                password=b"test-private-mobile-rollback", cancel=cancellation
            ) as reads:
                for name, replacement in (
                    ("session", copy(session)),
                    ("journal", copy(journal)),
                    ("claim", copy(captured.claim)),
                ):
                    with monkeypatch.context() as fault:
                        fault.setattr(captured, name, replacement)
                        with pytest.raises((ValueError, RuntimeError)):
                            reads.page("legacy", "chat", "messages")
                with monkeypatch.context() as fault:
                    fault.setattr(journal, "operation_id", "foreign-operation")
                    with pytest.raises((ValueError, RuntimeError)):
                        reads.page("legacy", "chat", "messages")
                for selector in (*selectors, candidate_selector):
                    retained = selector.with_suffix(".retained")
                    original = selector.read_bytes()
                    selector.rename(retained)
                    selector.write_bytes(original)
                    selector.chmod(0o600)
                    try:
                        with pytest.raises(
                            bootstrap.RecoveryRequired, match="selector"
                        ):
                            reads.page("legacy", "chat", "messages")
                    finally:
                        selector.unlink()
                        retained.rename(selector)
                root = authority.control_root.parent
                previous = set(root.glob("pending-*"))
                control_records.register_pending(
                    root,
                    "foreign-pending",
                    ("mobile.profile",),
                    journal.root.parent,
                    selectors,
                )
                try:
                    with pytest.raises(
                        bootstrap.RecoveryRequired, match="pending_scope"
                    ):
                        reads.page("legacy", "chat", "messages")
                finally:
                    for path in set(root.glob("pending-*")) - previous:
                        path.unlink()
                with monkeypatch.context() as fault:
                    fault.setattr(mobile_captured_sqlite, "_PAGE_BYTES", 1)
                    with pytest.raises(ValueError, match="page_limit"):
                        reads.page("legacy", "chat", "messages")
                cancellation.set()
                with pytest.raises(InterruptedError, match="cancelled"):
                    reads.page("legacy", "chat", "messages")
                cancellation.clear()
                staged = reads._connections["legacy", "chat"][3]
                with staged.open("ab") as stream:
                    stream.write(b"changed staged member")
                with pytest.raises(ValueError, match="candidate_changed"):
                    reads.page("legacy", "chat", "messages")
            assert not tuple(output.glob(".mobile-read-*"))
            assert _fingerprints(tmp_path / "profile") == before
            # Corrupt authenticated selected bytes only while the actual member
            # extractor is streaming. Native archive verification still executes.
            extract = mobile_captured_sqlite._extract
            chunks = mobile_captured_sqlite.archive_reader._member_chunks
            extracting = []
            corrupted = []

            def altered_chunks(*args):
                for chunk in chunks(*args):
                    if extracting and not corrupted and chunk:
                        corrupted.append(True)
                        yield bytes((chunk[0] ^ 1,)) + chunk[1:]
                    else:
                        yield chunk

            def actual_extract(*args):
                extracting.append(True)
                try:
                    yield from extract(*args)
                finally:
                    extracting.pop()

            with monkeypatch.context() as fault:
                fault.setattr(mobile_captured_sqlite, "_extract", actual_extract)
                fault.setattr(
                    mobile_captured_sqlite.archive_reader,
                    "_member_chunks",
                    altered_chunks,
                )
                with (
                    pytest.raises(ValueError, match="payload_changed"),
                    captured.sqlite_reads(
                        password=b"test-private-mobile-rollback", cancel=Event()
                    ),
                ):
                    pytest.fail("corrupt selected payload admitted")
            assert corrupted
            assert not tuple(output.glob(".mobile-read-*"))
            assert _fingerprints(tmp_path / "profile") == before


@pytest.mark.parametrize("boundary", ("validation", "page"))
def test_authenticated_candidate_stays_bound_to_actual_sqlite_pin(
    tmp_path, monkeypatch, boundary
):
    from contextlib import contextmanager

    from tldw_chatbook.Backup_Recovery import mobile_captured_sqlite
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )

    with _capture_fixture(tmp_path, monkeypatch) as values:
        session, journal, _, _, preview, inventory, _ = values
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=output,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            reached = []
            original = mobile_captured_sqlite.open_recovery_validation

            @contextmanager
            def substitute_before_validation(owner_id, candidate, **kwargs):
                if not reached:
                    authentic = candidate.read_bytes()
                    changed = authentic.replace(
                        b"private fixture prompt", b"changed fixture prompt"
                    )
                    assert changed != authentic
                    candidate.write_bytes(changed)
                    reached.append(candidate.stat().st_ino)
                with original(owner_id, candidate, **kwargs) as opened:
                    yield opened

            if boundary == "validation":
                monkeypatch.setattr(
                    mobile_captured_sqlite,
                    "open_recovery_validation",
                    substitute_before_validation,
                )
                with (
                    pytest.raises(ValueError, match="candidate_changed"),
                    captured.sqlite_reads(
                        password=b"test-private-mobile-rollback", cancel=Event()
                    ),
                ):
                    pytest.fail(
                        "unauthenticated candidate reached a validated read scope"
                    )
                assert reached
            else:
                with captured.sqlite_reads(
                    password=b"test-private-mobile-rollback", cancel=Event()
                ) as reads:
                    candidate = reads._connections["legacy", "chat"][3]
                    authentic = candidate.read_bytes()
                    info = candidate.stat()
                    pins = [
                        fd
                        for fd in reads.custody.fds
                        if (
                            mobile_captured_sqlite.os.fstat(fd).st_dev,
                            mobile_captured_sqlite.os.fstat(fd).st_ino,
                        )
                        == (info.st_dev, info.st_ino)
                    ]
                    assert len(pins) == 1
                    retained = candidate.with_suffix(".retained")
                    candidate.rename(retained)
                    candidate.write_bytes(authentic)
                    candidate.chmod(0o600)
                    changed = authentic.replace(
                        b"private fixture prompt", b"changed fixture prompt"
                    )
                    assert changed != authentic
                    retained.write_bytes(changed)
                    assert (
                        mobile_captured_sqlite.os.fstat(pins[0]).st_ino
                        == retained.stat().st_ino
                    )
                    assert candidate.stat().st_ino != retained.stat().st_ino
                    assert (
                        hashlib.sha256(candidate.read_bytes()).digest()
                        == hashlib.sha256(authentic).digest()
                    )
                    with pytest.raises(ValueError, match="candidate_changed"):
                        reads.page("legacy", "chat", "messages")
            assert not tuple(output.glob(".mobile-read-*"))


def test_plaintext_output_never_uses_substituted_staging_parent(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from tldw_chatbook.Backup_Recovery import crypto
    from tldw_chatbook.Backup_Recovery.mobile_consolidation_apply import (
        _captured_mobile,
    )

    with _capture_fixture(tmp_path, monkeypatch) as values:
        session, journal, _, _, preview, inventory, _ = values
        output = tmp_path / "archives"
        output.mkdir(mode=0o700)
        with _captured_mobile(
            session,
            journal,
            preview,
            inventory.scope_digest,
            destination=output,
            password=b"test-private-mobile-rollback",
            cancel=Event(),
        ) as captured:
            violations = []

            def run_case(boundary, kind):
                foreign = tmp_path / (boundary + "-" + kind)
                foreign.mkdir(mode=0o700)
                reached = {}
                native_open = crypto._open_regular
                native_link = crypto.os.link

                def substitute(parent):
                    retained = output / (parent.name + "-retained")
                    parent.rename(retained)
                    if kind == "symlink":
                        parent.symlink_to(foreign, target_is_directory=True)
                    else:
                        foreign.rename(parent)
                    reached.update(parent=parent, retained=retained)

                @contextmanager
                def open_input(path, **kwargs):
                    with native_open(path, **kwargs) as stream:
                        if (
                            path.name == "input"
                            and kwargs.get("_custody") is not None
                            and not reached
                        ):
                            reached["scope"] = kwargs["_custody"].scope
                            reached["operation"] = path.parent
                            if boundary == "allocation":
                                substitute(path.parent)
                        yield stream

                def publish(source, target, **kwargs):
                    if (
                        boundary == "publication"
                        and reached.get("operation") is not None
                        and not reached.get("parent")
                        and Path(target).name == "decrypted.zip"
                    ):
                        substitute(reached["operation"])
                        replacement = (
                            foreign if kind == "symlink" else reached["parent"]
                        )
                        # An unrelated existing file must never be adopted by
                        # path-based publication after the parent is replaced.
                        (replacement / Path(source).name).write_bytes(b"foreign marker")
                        reached["allowed"] = {Path(source).name: b"foreign marker"}
                    return native_link(source, target, **kwargs)

                try:
                    with monkeypatch.context() as fault:
                        fault.setattr(crypto, "_open_regular", open_input)
                        fault.setattr(crypto.os, "link", publish)
                        with (
                            pytest.raises((OSError, ValueError, RuntimeError)),
                            captured.sqlite_reads(
                                password=b"test-private-mobile-rollback", cancel=Event()
                            ),
                        ):
                            pytest.fail("substituted output parent admitted")
                    assert reached.get("parent") is not None
                    replacement = foreign if kind == "symlink" else reached["parent"]
                    actual = {
                        p.name: p.read_bytes()
                        for p in (replacement.iterdir() if replacement.exists() else ())
                        if p.is_file()
                    }
                    if actual != reached.get("allowed", {}):
                        violations.append((boundary, kind, sorted(actual)))
                    if not replacement.exists():
                        violations.append((boundary, kind, "foreign directory removed"))
                    identity = reached["retained"].stat()
                    pins = [
                        fd
                        for fd in reached["scope"].custody.fds
                        if (crypto.os.fstat(fd).st_dev, crypto.os.fstat(fd).st_ino)
                        == (identity.st_dev, identity.st_ino)
                    ]
                    if not pins:
                        violations.append(
                            (boundary, kind, "owned parent pin not retained")
                        )
                finally:
                    if reached.get("parent") is not None:
                        parent = reached["parent"]
                        if parent.is_symlink():
                            parent.unlink()
                        elif parent.exists():
                            parent.rename(foreign)
                        if parent.parent.exists():
                            reached["retained"].rename(parent)
                            # Restore the exact test-owned original before retrying
                            # name verification, never an uncertain descriptor close.
                            reached["scope"].retire()

            for boundary in ("allocation", "publication"):
                for kind in ("symlink", "directory"):
                    run_case(boundary, kind)
            assert violations == []
