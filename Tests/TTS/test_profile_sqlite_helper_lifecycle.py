"""Real process contenders and retained TTS helper lifecycle regressions."""

from __future__ import annotations

import asyncio
import json
import os
import select
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from tldw_chatbook.DB import private_sqlite_process as process
from tldw_chatbook.DB.private_sqlite_protocol import FileIdentity
from tldw_chatbook.TTS import profile_migration_namespace as namespace
from tldw_chatbook.TTS import profile_schema
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
from tldw_chatbook.TTS.profile_sqlite_policy import require_native_close_policy_support
from tldw_chatbook.TTS.profile_store_lock import ProfileStoreLease, ProfileStoreLockMode


def _other_process_can_begin_write(path):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import sqlite3, sys
connection = sqlite3.connect(sys.argv[1], timeout=0.05, isolation_level=None)
try:
    connection.execute('BEGIN IMMEDIATE')
except sqlite3.OperationalError as error:
    print(error.sqlite_errorcode)
else:
    print(0)
    connection.rollback()
finally:
    connection.close()
""",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    code = int(result.stdout.strip())
    assert code in {0, sqlite3.SQLITE_BUSY}
    return code == 0


@pytest.mark.asyncio
async def test_closing_one_repository_preserves_sibling_writer(tmp_path):
    path = tmp_path / "profiles.sqlite3"
    first = TTSProfileRepository(path)
    second = TTSProfileRepository(path)
    await first.open()
    await second.open()
    await second._submit_operation(lambda c: c.execute("BEGIN IMMEDIATE"))
    try:
        assert not _other_process_can_begin_write(path)
        await first.close()
        assert not _other_process_can_begin_write(path)
    finally:
        await second._submit_operation(lambda c: c.rollback())
        await second.close()


@pytest.mark.asyncio
async def test_partial_initial_sidecar_pin_refusal_preserves_sibling_writer(tmp_path):
    path = tmp_path / "profiles.sqlite3"
    sibling, rejected = TTSProfileRepository(path), TTSProfileRepository(path)
    await sibling.open()
    await sibling._submit_operation(lambda c: c.execute("BEGIN IMMEDIATE"))
    shm = Path(f"{path}-shm")
    assert not _other_process_can_begin_write(path)
    shm.chmod(0o644)
    try:
        with pytest.raises(ProfileRepositoryError):
            await rejected.open()
        assert rejected._connection is None
        assert not _other_process_can_begin_write(path)
    finally:
        shm.chmod(0o600)
        await rejected.close()
        await sibling._submit_operation(lambda c: c.rollback())
        await sibling.close()


def test_exact_live_policy_precedes_first_sql(tmp_path, monkeypatch):
    path = tmp_path / "profiles.sqlite3"
    seed = profile_schema.open_profile_store(path)
    seed.close()
    events = []
    factory = sqlite3.connect

    class TracingConnection(sqlite3.Connection):
        def setconfig(self, option, enabled=True):
            super().setconfig(option, enabled)
            events.append(("set", option, enabled))

        def getconfig(self, option):
            result = super().getconfig(option)
            events.append(("get", option, result))
            return result

        def execute(self, sql, parameters=()):
            events.append(("sql", sql))
            return super().execute(sql, parameters)

    def tracing_factory(*args, **kwargs):
        return factory(*args, **kwargs, factory=TracingConnection)

    monkeypatch.setattr(sqlite3, "connect", tracing_factory)
    connection = profile_schema.open_exact_current_profile_store(path)
    try:
        first_sql = next(
            index for index, event in enumerate(events) if event[0] == "sql"
        )
        assert events[:first_sql] == [
            ("set", sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, True),
            ("get", sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE, True),
        ]
    finally:
        connection.close()


def test_live_policy_does_not_change_initialization_or_immutable_evidence(
    tmp_path, monkeypatch
):
    path = tmp_path / "profiles.sqlite3"
    configured = []
    factory = sqlite3.connect

    class Observed(sqlite3.Connection):
        def setconfig(self, option, enabled=True):
            configured.append(self)
            return super().setconfig(option, enabled)

    def observed(*args, **kwargs):
        return factory(*args, **kwargs, factory=Observed)

    monkeypatch.setattr(sqlite3, "connect", observed)
    initialized = profile_schema.open_profile_store(path)
    assert initialized not in configured
    initialized.close()
    for _ in range(2):
        current = profile_schema.open_exact_current_profile_store(path)
        assert current._connection in configured
        current.close()
    assert len(configured) == 2
    assert Path(f"{path}-wal").exists()
    assert Path(f"{path}-shm").exists()


def test_post_init_size_binding_is_rechecked_before_publication(tmp_path, monkeypatch):
    path = tmp_path / "profiles.sqlite3"
    profile_schema.open_profile_store(path).close()
    expected = profile_schema.capture_post_init_profile_store_authority(path)
    stream = profile_schema._stream_exact_store_metadata_evidence

    def grow_after_evidence(connection):
        result = stream(connection)
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                "import sys; f=open(sys.argv[1], 'ab'); f.write(bytes(4096)); f.close()",
                str(path),
            ],
            check=True,
            timeout=5,
        )
        return result

    monkeypatch.setattr(
        profile_schema, "_stream_exact_store_metadata_evidence", grow_after_evidence
    )
    opened = None
    try:
        with pytest.raises(ProfileRepositoryError):
            opened = profile_schema.open_exact_current_profile_store(
                path, expected_post_init_authority=expected
            )
    finally:
        if opened is not None:
            opened.close()


@pytest.mark.parametrize("reject", ["set", "get"])
@pytest.mark.parametrize("close_fails", [False, True])
def test_rejected_actual_policy_retains_unused_handle_until_close(
    tmp_path, monkeypatch, reject, close_fails
):
    path = tmp_path / "profiles.sqlite3"
    profile_schema.open_profile_store(path).close()
    used = dict(process.HELPER_ADMISSION._used)
    events = []
    factory = sqlite3.connect

    class RejectedConnection(sqlite3.Connection):
        def setconfig(self, option, enabled=True):
            if reject == "set":
                raise sqlite3.NotSupportedError("test rejection")
            super().setconfig(option, enabled)

        def getconfig(self, option):
            return False if reject == "get" else super().getconfig(option)

        def execute(self, sql, parameters=()):
            events.append("sql")
            return super().execute(sql, parameters)

        def close(self):
            events.append("close")
            if close_fails and events.count("close") == 1:
                raise sqlite3.OperationalError("test close failure")
            super().close()

    def actual_factory(database, *args, **kwargs):
        if database == ":memory:":
            return factory(database, *args, **kwargs)
        return factory(database, *args, **kwargs, factory=RejectedConnection)

    monkeypatch.setattr(sqlite3, "connect", actual_factory)
    require_native_close_policy_support()
    if close_fails:
        with pytest.raises(profile_schema.ExactProfileStoreCleanupError) as caught:
            profile_schema.open_exact_current_profile_store(path)
        owner = caught.value.connection
        assert owner._helper.cleanup_state == "still_owned"
        assert process.HELPER_ADMISSION._used["retained"] == used["retained"] + 1
        owner.close()
    else:
        with pytest.raises(ProfileRepositoryError) as caught:
            profile_schema.open_exact_current_profile_store(path)
        assert caught.value.code == "runtime_unsupported"
    assert "sql" not in events
    assert process.HELPER_ADMISSION._used == used


@pytest.mark.asyncio
async def test_normal_cleanup_rolls_back_and_checkpoints_once(tmp_path):
    from uuid import uuid4

    from tldw_chatbook.TTS.profile_types import TTSProfileDraft

    path = tmp_path / "profiles.sqlite3"
    repository = TTSProfileRepository(path)
    await repository.open()
    await repository.create_profile(
        TTSProfileDraft("Committed", "openai", "tts-1", "alloy", "wav", 1.0, {}),
        uuid4(),
    )
    statements = []
    pending = "UPDATE tts_generation_profiles SET display_name='Pending', normalized_name='pending'"

    def stage(connection):
        connection.set_trace_callback(statements.append)
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(pending)

    await repository._submit_operation(stage)
    await repository.close()
    assert statements == [
        "BEGIN IMMEDIATE",
        pending,
        "ROLLBACK",
        "PRAGMA main.wal_checkpoint(PASSIVE)",
    ]
    reopened = TTSProfileRepository(path)
    await reopened.open()
    try:
        assert [
            p.display_name for p in (await reopened.list_profiles()).value.profiles
        ] == ["Committed"]
    finally:
        await reopened.close()


@pytest.mark.parametrize(
    ("body_type", "close_type"),
    [
        (asyncio.CancelledError, sqlite3.OperationalError),
        (KeyboardInterrupt, sqlite3.OperationalError),
        (SystemExit, sqlite3.OperationalError),
        (ValueError, asyncio.CancelledError),
        (ValueError, KeyboardInterrupt),
        (ValueError, SystemExit),
        (asyncio.CancelledError, SystemExit),
    ],
)
def test_live_open_cleanup_preserves_control_and_charged_owner(
    tmp_path, monkeypatch, body_type, close_type
):
    """Failed native close must not replace control or reap its live proof."""
    path = tmp_path / "profiles.sqlite3"
    profile_schema.open_profile_store(path).close()
    used = dict(process.HELPER_ADMISSION._used)
    body_error, close_error = body_type("body"), close_type("close")
    expected = body_error if not isinstance(body_error, Exception) else close_error
    real_connect = sqlite3.connect
    owners, native_flags = [], []
    fail_close = True

    class RetainedConnection(sqlite3.Connection):
        def close(self):
            native_flags.append(
                self.getconfig(sqlite3.SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE)
            )
            if fail_close:
                raise close_error
            super().close()

    def connect(*args, **kwargs):
        return real_connect(*args, **kwargs, factory=RetainedConnection)

    def interrupt_validation(connection, **kwargs):
        owners.append(connection)
        raise body_error

    monkeypatch.setattr(sqlite3, "connect", connect)
    monkeypatch.setattr(profile_schema, "_validate_schema", interrupt_validation)
    try:
        with pytest.raises(BaseException) as caught:
            profile_schema.open_exact_current_profile_store(path)
        assert len(owners) == 1
        owner = owners[0]
        assert not owner._sqlite_closed and not owner._proof_lost
        assert owner._helper.cleanup_state == "still_owned"
        assert owner._helper._child.poll() is None
        assert native_flags == [True]
        assert process.HELPER_ADMISSION._used == {
            "transient": used["transient"],
            "retained": used["retained"] + 1,
        }
        assert caught.value is expected
    finally:
        fail_close = False
        for owner in owners:
            owner.close()
    assert native_flags == [True, True]
    assert owner._sqlite_closed and owner._helper.cleanup_state == "reaped"
    assert process.HELPER_ADMISSION._used == used


def test_reused_hostile_live_signal_keeps_prior_owners_without_stale_handoff(
    tmp_path, monkeypatch
):
    from tldw_chatbook.TTS.profile_errors import (
        ProfileMigrationCleanupError,
        _migration_cleanup_owner,
    )

    class HostileSignal(BaseException):
        def __getattribute__(self, name):
            if name.startswith("_profile_") or name == "__dict__":
                pytest.fail("signal attribute getter ran")
            return super().__getattribute__(name)

        def __setattr__(self, name, value):
            if name.startswith("_profile_") or name == "__dict__":
                pytest.fail("signal attribute setter ran")
            super().__setattr__(name, value)

    path = tmp_path / "profiles.sqlite3"
    profile_schema.open_profile_store(path).close()
    signal = HostileSignal()
    metadata = BaseException.__dict__["__dict__"].__get__(signal, BaseException)
    earlier_owner = object()
    migration = ProfileMigrationCleanupError(earlier_owner)
    metadata["_profile_migration_cleanup_error"] = migration
    used = dict(process.HELPER_ADMISSION._used)
    real_connect = sqlite3.connect
    owners, carriers = [], []
    fail_close = True

    class RetainedConnection(sqlite3.Connection):
        def close(self):
            if fail_close:
                raise sqlite3.OperationalError("owned close failure")
            super().close()

    def connect(*args, **kwargs):
        return real_connect(*args, **kwargs, factory=RetainedConnection)

    def interrupt(connection, **kwargs):
        owners.append(connection)
        raise signal

    monkeypatch.setattr(sqlite3, "connect", connect)
    monkeypatch.setattr(profile_schema, "_validate_schema", interrupt)
    try:
        for attempt in range(2):
            with pytest.raises(BaseException) as caught:
                profile_schema.open_exact_current_profile_store(path)
            assert caught.value is signal
            carrier = profile_schema._exact_profile_store_cleanup_error(signal)
            assert carrier.connection is owners[attempt]
            carriers.append(carrier)
            assert metadata["_profile_exact_cleanup_history"] == tuple(carriers[:-1])
            assert _migration_cleanup_owner(signal) is earlier_owner
            assert metadata["_profile_migration_cleanup_error"] is migration
            assert all(not owner._sqlite_closed for owner in owners)
            assert all(owner._helper._child.poll() is None for owner in owners)

        fail_close = False
        with pytest.raises(BaseException) as caught:
            profile_schema.open_exact_current_profile_store(path)
        assert caught.value is signal and owners[-1]._sqlite_closed
        assert profile_schema._exact_profile_store_cleanup_error(signal) is None
        assert metadata["_profile_exact_cleanup_history"] == tuple(carriers)

        # Reset current to a prior carrier to make the early-refusal control
        # independently sensitive to stale adoption, before a live owner exists.
        metadata["_profile_exact_cleanup_error"] = carriers[-1]

        def refuse_before_live(*args, **kwargs):
            raise signal

        monkeypatch.setattr(process.HelperLease, "start", refuse_before_live)
        with pytest.raises(BaseException) as caught:
            profile_schema.open_exact_current_profile_store(path)
        assert caught.value is signal and len(owners) == 3
        assert profile_schema._exact_profile_store_cleanup_error(signal) is None
        assert metadata["_profile_exact_cleanup_history"] == tuple(carriers)
        assert _migration_cleanup_owner(signal) is earlier_owner
        assert process.HELPER_ADMISSION._used == {
            "transient": used["transient"],
            "retained": used["retained"] + 2,
        }
    finally:
        fail_close = False
        for owner in owners:
            owner.close()
    assert process.HELPER_ADMISSION._used == used


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "checkpoint", [(0, 1), (0, "1", 1), (2, 1, 1), (0, 1, 2), (0, -1, 0)]
)
async def test_invalid_checkpoint_retains_complete_cleanup_owner(tmp_path, checkpoint):
    repository = TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()
    connection = repository._connection
    execute = connection.execute

    class Result:
        def fetchone(self):
            return checkpoint

    def invalid(sql, parameters=()):
        if sql == "PRAGMA main.wal_checkpoint(PASSIVE)":
            return Result()
        return execute(sql, parameters)

    connection.execute = invalid
    try:
        with pytest.raises(Exception, match="operation_failed"):
            await repository.close()
        assert repository._connection is connection
        assert repository._lease is not None
        assert repository._executor is not None
    finally:
        connection.execute = execute
        await repository.close()


@pytest.mark.asyncio
async def test_helper_reap_retry_does_not_reuse_closed_native_handle(tmp_path):
    repository = TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()
    owner = repository._connection
    original = owner._helper.close

    def fail_reap():
        raise process.HelperCleanupError()

    owner._helper.close = fail_reap
    try:
        with pytest.raises(ProfileRepositoryError):
            await repository.close()
        assert owner._sqlite_closed
        assert repository._connection is owner
        assert repository._executor is not None
    finally:
        owner._helper.close = original
    await repository.close()
    assert repository._connection is None
    assert repository._executor is None


@pytest.mark.asyncio
async def test_pinned_reader_allows_partial_checkpoint_and_committed_recovery(tmp_path):
    path = tmp_path / "profiles.sqlite3"
    writer, reader = TTSProfileRepository(path), TTSProfileRepository(path)
    await writer.open()
    await reader.open()
    from uuid import uuid4

    from tldw_chatbook.TTS.profile_types import TTSProfileDraft

    draft = TTSProfileDraft("Committed", "openai", "tts-1", "alloy", "wav", 1.0, {})
    await reader._submit_operation(lambda c: c.execute("BEGIN"))
    await reader._submit_operation(
        lambda c: c.execute("SELECT * FROM tts_generation_profiles").fetchall()
    )
    await writer.create_profile(draft, uuid4())
    checkpoint = []
    owner = writer._connection
    execute = owner.execute

    def observe(sql, parameters=()):
        result = execute(sql, parameters)
        if sql == "PRAGMA main.wal_checkpoint(PASSIVE)":
            row = result.fetchone()
            checkpoint.append(tuple(row))

            class Result:
                def fetchone(self):
                    return row

            return Result()
        return result

    owner.execute = observe
    try:
        await writer.close()
        assert len(checkpoint) == 1
        assert checkpoint[0][1] > checkpoint[0][2] >= 0
        assert Path(f"{path}-wal").exists()
    finally:
        await reader.close()
        await writer.close()
    reopened = TTSProfileRepository(path)
    await reopened.open()
    try:
        result = await reopened._submit_operation(
            lambda c: c.execute(
                "SELECT display_name FROM tts_generation_profiles"
            ).fetchall()
        )
        assert [row[0] for row in result.value] == ["Committed"]
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_failed_restore_export_keeps_live_owner_before_checkpoint(tmp_path):
    path, candidate = tmp_path / "profiles.sqlite3", tmp_path / "candidate.sqlite3"
    profile_schema.open_profile_store(candidate).close()
    repository = TTSProfileRepository(path)
    await repository.open()
    owner = repository._connection
    export = owner.export_restore_authority
    publication = tmp_path / ".profiles.sqlite3.migration-publication.json"

    def substitute(*, deadline):
        publication.write_bytes(b"foreign publication evidence")
        publication.chmod(0o600)
        return export(deadline=deadline)

    owner.export_restore_authority = substitute
    statements = []
    await repository._submit_operation(
        lambda c: c.set_trace_callback(statements.append)
    )
    try:
        with pytest.raises(ProfileRepositoryError):
            await repository.restore_from(candidate)
        assert not any("wal_checkpoint" in sql.lower() for sql in statements)
        assert repository._connection is owner
        assert not owner._sqlite_closed
        assert repository._lease is not None
        assert repository._executor is not None
        assert publication.read_bytes() == b"foreign publication evidence"
        with pytest.raises(ProfileRepositoryError):
            await repository.close()
    finally:
        owner.export_restore_authority = export
        publication.rename(tmp_path / "preserved-export-refusal-evidence")
        await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code",
    [
        sqlite3.SQLITE_BUSY,
        sqlite3.SQLITE_LOCKED,
        sqlite3.SQLITE_IOERR,
        sqlite3.SQLITE_BUSY | (1 << 8),
    ],
)
async def test_checkpoint_exception_requires_exact_busy(tmp_path, code):
    repository = TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()
    owner = repository._connection
    execute = owner.execute
    statements = []

    def checkpoint(sql, parameters=()):
        statements.append(sql)
        if sql == "PRAGMA main.wal_checkpoint(PASSIVE)":
            error = sqlite3.OperationalError("test checkpoint failure")
            error.sqlite_errorcode = code
            raise error
        return execute(sql, parameters)

    owner.execute = checkpoint
    try:
        if code == sqlite3.SQLITE_BUSY:
            await repository.close()
            assert repository._connection is None
        else:
            with pytest.raises(ProfileRepositoryError):
                await repository.close()
            assert repository._connection is owner
            assert repository._lease is not None
            assert repository._executor is not None
        assert statements == ["PRAGMA main.wal_checkpoint(PASSIVE)"]
    finally:
        owner.execute = execute
        await repository.close()


@pytest.mark.asyncio
async def test_cancelled_open_keeps_helper_capacity_until_worker_settles(
    tmp_path, monkeypatch
):
    path = tmp_path / "profiles.sqlite3"
    profile_schema.open_profile_store(path).close()
    entered, release, acquired = threading.Event(), threading.Event(), threading.Event()
    configure = profile_schema.configure_native_close_policy
    used = dict(process.HELPER_ADMISSION._used)

    def barrier(connection):
        configure(connection)
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(profile_schema, "configure_native_close_policy", barrier)
    repository = TTSProfileRepository(path)
    opening = asyncio.create_task(repository.open())
    assert await asyncio.to_thread(entered.wait, 5)
    opening.cancel()
    await asyncio.sleep(0)
    assert not opening.done()
    assert process.HELPER_ADMISSION._used == {
        "retained": used["retained"] + 1,
        "transient": used["transient"] + 1,
    }

    def reserve_all():
        with process.HELPER_ADMISSION.reserve(
            retained=4,
            transient=1,
            deadline=process.OperationDeadline(time.monotonic() + 5),
        ):
            acquired.set()

    waiter = asyncio.create_task(asyncio.to_thread(reserve_all))
    try:
        assert not acquired.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await opening
        assert repository._connection is not None
        assert not acquired.is_set()
        await repository.close()
        await waiter
        assert acquired.is_set()
        assert process.HELPER_ADMISSION._used == used
    finally:
        release.set()
        await repository.close()


def test_latch_wakes_retained_waiters_without_blocking_transient_work():
    admission = process.HelperAdmission()
    waiting, refused = threading.Event(), threading.Event()
    failures = []
    original_wait = admission._condition.wait

    def observed_wait(timeout=None):
        waiting.set()
        return original_wait(timeout)

    admission._condition.wait = observed_wait

    def waiter():
        try:
            with admission.reserve(
                retained=1,
                transient=1,
                deadline=process.OperationDeadline(time.monotonic() + 5),
            ):
                failures.append("latched waiter admitted")
        except process.TTSAdmissionLatchedError:
            refused.set()

    with admission.reserve(
        retained=4, transient=0, deadline=process.OperationDeadline(None)
    ):
        thread = threading.Thread(target=waiter)
        thread.start()
        assert waiting.wait(1)
        admission.latch_tts_proof_loss()
        assert refused.wait(1)
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert not failures

        # Unrelated transient ownership is independent of this retained latch.
        def transient():
            with admission.reserve(
                retained=0, transient=1, deadline=process.OperationDeadline(None)
            ):
                return True

        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(transient).result(timeout=1)


@pytest.mark.asyncio
async def test_pre_live_helper_death_releases_capacity_and_allows_retry(
    tmp_path, monkeypatch
):
    path = tmp_path / "profiles.sqlite3"
    profile_schema.open_profile_store(path).close()
    used = dict(process.HELPER_ADMISSION._used)
    exchange = process.HelperLease._exchange

    def die_before_initial_reply(self, frame, operation, deadline):
        if operation == "tts_exact_current":
            self._child.kill()
            self._child.wait(timeout=5)
        return exchange(self, frame, operation, deadline)

    monkeypatch.setattr(process.HelperLease, "_exchange", die_before_initial_reply)
    repository = TTSProfileRepository(path)
    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.open()
        assert caught.value.code == "operation_failed"
        assert repository._connection is None and repository._lease is None
        assert process.HELPER_ADMISSION._used == used
        assert not process.HELPER_ADMISSION.tts_proof_lost
        monkeypatch.setattr(process.HelperLease, "_exchange", exchange)
        await repository.open()
    finally:
        await repository.close()
    assert process.HELPER_ADMISSION._used == used


@pytest.mark.parametrize("field", ["dev", "ino", "mode", "uid", "gid"])
def test_typed_namespace_parent_rejects_security_metadata_substitution(tmp_path, field):
    tmp_path.chmod(0o700)
    current = tmp_path.stat()
    expected = FileIdentity.from_stat(current)
    altered = replace(expected, **{field: getattr(expected, field) + 1})
    adapter = namespace.HelperNamespaceIdentity(altered)
    assert not namespace._same_parent(current, adapter)
    assert not namespace._same_parent_with_link_delta(current, adapter, link_delta=0)
    original = namespace.HelperNamespaceIdentity(expected)
    assert namespace._same_parent(current, original)
    shifted = namespace.HelperNamespaceIdentity(
        replace(expected, nlink=expected.nlink + 1)
    )
    assert not namespace._same_parent(current, shifted)
    assert namespace._same_parent_with_link_delta(current, shifted, link_delta=-1)


def test_expired_before_helper_dispatch_keeps_healthy_owner(tmp_path):
    program = """
import asyncio, sys, time
from pathlib import Path
import Tests.conftest
from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
from tldw_chatbook.DB.private_sqlite_process import HELPER_ADMISSION, OperationDeadline, HelperTimeoutError
repository = TTSProfileRepository(Path(sys.argv[1]))
async def main():
    await repository.open()
    owner = repository._connection
    def expired():
        try:
            owner.export_restore_authority(deadline=OperationDeadline(time.monotonic() - 1))
        except HelperTimeoutError:
            pass
        else:
            raise AssertionError("expired export was dispatched")
    await asyncio.wrap_future(repository._executor.submit(expired))
    assert not HELPER_ADMISSION.tts_proof_lost
    await repository.close()
asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path / "profiles.sqlite3")],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_expired_live_open_retains_bounded_cleanup_without_publication(
    tmp_path, monkeypatch
):
    path = tmp_path / "profiles.sqlite3"
    profile_schema.open_profile_store(path).close()
    used = dict(process.HELPER_ADMISSION._used)
    deadline = process.OperationDeadline(time.monotonic() + 1)
    metadata = profile_schema._stream_exact_store_metadata_evidence

    def expire_after_metadata(connection):
        result = metadata(connection)
        threading.Event().wait(max(0, deadline.expires_at - time.monotonic()) + 0.01)
        return result

    monkeypatch.setattr(
        profile_schema, "_stream_exact_store_metadata_evidence", expire_after_metadata
    )
    with pytest.raises(profile_schema.ExactProfileStoreCleanupError) as caught:
        profile_schema.open_exact_current_profile_store(path, deadline=deadline)
    owner = caught.value.connection
    assert not owner._proof_lost
    assert not process.HELPER_ADMISSION.tts_proof_lost
    started = time.monotonic()
    owner.close()
    assert time.monotonic() - started < 8
    assert process.HELPER_ADMISSION._used == used


def test_latch_between_worker_submission_and_reservation_is_restart_required(tmp_path):
    program = """
import asyncio, sys
from pathlib import Path
import Tests.conftest
from tldw_chatbook.TTS import profile_repository as module
from tldw_chatbook.TTS import profile_schema
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.DB.private_sqlite_process import HELPER_ADMISSION
path = Path(sys.argv[1])
profile_schema.open_profile_store(path).close()
repository = module.TTSProfileRepository(path)
original = module.open_exact_current_profile_store
def latched(*args, **kwargs):
    HELPER_ADMISSION.latch_tts_proof_loss()
    return original(*args, **kwargs)
module.open_exact_current_profile_store = latched
async def main():
    try:
        await repository.open()
    except ProfileRepositoryError as error:
        assert error.code == "restart_required", error.code
    else:
        raise AssertionError("late latch admitted")
    assert repository._connection is None and repository._lease is None
    await repository.close()
asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path / "profiles.sqlite3")],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


_APP_EXIT_CHILD = """
import asyncio, json, sys
from pathlib import Path
import Tests.conftest
sys.path.insert(0, str(Path.cwd() / "packages/tldw_profile_core/src"))
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.DB.private_sqlite_process import HELPER_ADMISSION
from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
import tldw_chatbook.app as app_module

app_module.get_tts_profiles_db_path = lambda: Path(sys.argv[1])
app = _build_test_app()
state, owner_count = sys.argv[2], int(sys.argv[3])
siblings = []
statements = []

if state == "partial":
    from tldw_chatbook.TTS import profile_repository as repository_module
    original_revalidate = repository_module.revalidate_exact_current_profile_store
    def publication_barrier(connection, path):
        connection.set_trace_callback(statements.append)
        print(json.dumps({"phase": "ready"}), file=sys.__stdout__, flush=True)
        assert sys.stdin.readline().strip() == "substituted"
        connection._helper._child.kill()
        connection._helper._child.wait(timeout=5)
        return original_revalidate(connection, path)
    repository_module.revalidate_exact_current_profile_store = publication_barrier

async def exercise():
    ensured = await app._ensure_tts_profile_repository()
    repository = app._tts_profile_repository
    assert repository is not None
    if state == "partial":
        assert ensured is None
        assert repository._helper_restart_required
    else:
        assert ensured is repository
    for _ in range(owner_count - 1):
        sibling = TTSProfileRepository(Path(sys.argv[1]))
        await sibling.open()
        siblings.append(sibling)
    def establish(connection):
        connection.execute("CREATE TABLE shutdown_gate (value INTEGER, payload BLOB)")
        connection.execute("INSERT INTO shutdown_gate VALUES (1, X'01')")
        if state == "read":
            connection.execute("BEGIN")
            connection.outstanding = connection.execute("SELECT value FROM shutdown_gate UNION ALL SELECT 2")
            assert connection.outstanding.fetchone()[0] == 1
        elif state == "write":
            connection.execute("PRAGMA cache_size=16")
            connection.execute("BEGIN IMMEDIATE")
            before = Path(str(repository._database_path) + "-wal").stat().st_size
            connection.executemany("INSERT INTO shutdown_gate VALUES (2, zeroblob(65536))", [() for _ in range(128)])
            after = Path(str(repository._database_path) + "-wal").stat().st_size
            assert after > before + 1024 * 1024, (before, after)
        connection.set_trace_callback(statements.append)
    if state != "partial":
        await repository._submit_operation(establish)
        print(json.dumps({"phase": "ready"}), file=sys.__stdout__, flush=True)
        assert sys.stdin.readline().strip() == "substituted"
    owner = repository._connection
    helper = owner._helper
    if state != "partial":
        helper._child.kill()
        helper._child.wait(timeout=5)
    for sibling in siblings:
        sibling._connection._helper._child.kill()
        sibling._connection._helper._child.wait(timeout=5)
        try:
            await sibling._submit_operation(lambda connection: connection.execute("SELECT 1"))
        except ProfileRepositoryError as error:
            assert error.code == "restart_required", error.code
        else:
            raise AssertionError("sibling lost proof accepted")
    try:
        await repository._submit_operation(lambda connection: connection.execute("SELECT 1"))
    except ProfileRepositoryError as error:
        assert error.code == "restart_required", error.code
    else:
        raise AssertionError("dead helper authorized live use")
    for _ in range(2):
        try:
            await app._close_owned_tts_resources()
        except ProfileRepositoryError as error:
            assert error.code == "restart_required", error.code
        else:
            raise AssertionError("terminal close succeeded")
    assert not statements
    assert repository._connection is owner
    assert repository._lease is not None
    assert repository._executor is not None
    assert HELPER_ADMISSION._used["retained"] == owner_count, HELPER_ADMISSION._used
    assert 0 <= HELPER_ADMISSION._used["transient"] <= 4
    print(json.dumps({"phase": "retained"}), file=sys.__stdout__, flush=True)
    assert sys.stdin.readline().strip() == "exit"

async def main():
    original = app._close_owned_tts_resources
    calls = []
    async def observed_cleanup():
        calls.append(True)
        return await original()
    app._close_owned_tts_resources = observed_cleanup
    async with app.run_test(size=(100, 30)):
        await exercise()
    assert len(calls) >= 3, "real app unmount did not run retained cleanup"

asyncio.run(main())
"""


def _phase(child, expected, stderr):
    if not select.select([child.stdout], [], [], 30)[0]:
        stderr.seek(0)
        pytest.fail(f"owned child phase timed out: {stderr.read()[-8000:]}")
    line = child.stdout.readline()
    if not line:
        stderr.seek(0)
        pytest.fail(f"owned child exited {child.poll()}: {stderr.read()[-8000:]}")
    assert json.loads(line) == {"phase": expected}


def test_loss_before_transaction_commit_never_rolls_back(tmp_path):
    program = """
import asyncio, sys
from pathlib import Path
import Tests.conftest
from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
repository = TTSProfileRepository(Path(sys.argv[1]))
async def main():
    await repository.open()
    statements = []
    sibling = TTSProfileRepository(Path(sys.argv[1]))
    await sibling.open()
    def transaction(connection):
        connection.set_trace_callback(statements.append)
        def lose():
            helper = connection._helper
            helper._child.kill()
            helper._child.wait(timeout=5)
        repository._worker_transaction(connection, lose, operation_kind="create", immediate=True)
    try:
        await repository._submit_operation(transaction)
    except ProfileRepositoryError as error:
        assert error.code == "restart_required", error.code
    else:
        raise AssertionError("lost proof transaction published")
    assert statements == ["BEGIN IMMEDIATE"], statements
    assert (await sibling._submit_operation(lambda c: c.execute("SELECT 1").fetchone()[0])).value == 1
    await sibling.close()
    for _ in range(5):
        other = TTSProfileRepository(Path(sys.argv[1]))
        try:
            await other.open()
        except ProfileRepositoryError as error:
            assert error.code == "restart_required", error.code
        else:
            raise AssertionError("new admission bypassed proof-loss latch")
        assert other._executor is None
    try:
        await repository.close()
    except ProfileRepositoryError as error:
        assert error.code == "restart_required"
    else:
        raise AssertionError("lost proof closed")
asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path / "profiles.sqlite3")],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode", ["read", "write"])
def test_helper_death_during_real_reference_blob_lifetime(tmp_path, mode):
    program = """
import asyncio, hashlib, io, sys, wave
from pathlib import Path
from uuid import uuid4
import Tests.conftest
from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
from tldw_chatbook.TTS.profile_types import TTSProfileDraft
from tldw_chatbook.TTS.profile_reference_types import CanonicalTTSCloneReference, TTSCloneRecipeRequirement
from tldw_chatbook.TTS.profile_schema import ExactProfileStoreProofLostError
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
repository = TTSProfileRepository(Path(sys.argv[1]))
mode = sys.argv[2]
events = []
stream = io.BytesIO()
with wave.open(stream, "wb") as writer:
    writer.setnchannels(1)
    writer.setsampwidth(2)
    writer.setframerate(16000)
    writer.writeframes(bytes(32000))
payload = stream.getvalue()
canonical = CanonicalTTSCloneReference(payload, "test reference", hashlib.sha256(payload).hexdigest(), len(payload), 1000, 16000, 1, "pcm_s16le")
draft = TTSProfileDraft("Blob", "openai", "test-model", "alloy", "wav", 1.0, {})
requirement = TTSCloneRecipeRequirement("test-recipe", 1, "test-model")
profile_id = uuid4()
async def main():
    opened = await repository.open()
    if mode == "read":
        await repository.create_profile_with_reference(draft, profile_id, canonical, requirement, expected_generation=opened.generation)
    owner = repository._connection
    original = owner.blobopen
    lost = False
    class Blob:
        def __init__(self, blob): self.blob = blob
        def __len__(self): return len(self.blob)
        def tell(self): return self.blob.tell()
        def death(self):
            nonlocal lost
            if not lost:
                lost = True
                events.append("helper_died_with_blob_open")
                owner._helper._child.kill()
                owner._helper._child.wait(timeout=5)
        def read(self, amount):
            if mode == "read": self.death()
            return self.blob.read(amount)
        def write(self, content):
            if mode == "write": self.death()
            return self.blob.write(content)
        def close(self):
            events.append("blob_close")
            self.blob.close()
    owner.blobopen = lambda *args, **kwargs: Blob(original(*args, **kwargs))
    revalidate = repository._worker_revalidate_exact_authority
    def observed(*args, **kwargs):
        try:
            return revalidate(*args, **kwargs)
        except ExactProfileStoreProofLostError:
            events.append("proof_loss_observed")
            raise
    repository._worker_revalidate_exact_authority = observed
    try:
        if mode == "read":
            await repository.get_reference(profile_id, expected_revision=2, expected_generation=opened.generation)
        else:
            await repository.create_profile_with_reference(draft, profile_id, canonical, requirement, expected_generation=opened.generation)
    except ProfileRepositoryError as error:
        assert error.code == "restart_required", (error.code, events)
    else:
        raise AssertionError("lost proof operation published")
    observed_at = events.index("proof_loss_observed")
    assert events.index("helper_died_with_blob_open") < events.index("blob_close") < observed_at, events
    assert "blob_close" not in events[observed_at:], events
    assert repository._connection is owner
    assert repository._lease is not None
    assert repository._executor is not None
asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path / "profiles.sqlite3"), mode],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("phase", ["initial", "directory_accessor", "native_close"])
def test_close_first_observes_helper_loss_and_retains_terminal_owner(tmp_path, phase):
    program = """
import asyncio, os, sys
from pathlib import Path
import Tests.conftest
from tldw_chatbook.TTS import profile_repository as module
from tldw_chatbook.TTS import profile_schema as schema
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.DB.private_sqlite_process import HELPER_ADMISSION
async def main():
    repository = module.TTSProfileRepository(Path(sys.argv[1]))
    if sys.argv[2] == "directory_accessor":
        incoming = Path(sys.argv[1]).with_name("incoming.sqlite3")
        schema.open_profile_store(incoming).close()
    await repository.open()
    if sys.argv[2] == "directory_accessor":
        await repository.restore_from(incoming)
    owner, lease, executor = repository._connection, repository._lease, repository._executor
    events = []
    await asyncio.wrap_future(executor.submit(owner._connection.set_trace_callback, events.append))
    lose_proof = owner._lose_proof
    def observed_loss():
        events.append("observed_loss")
        return lose_proof()
    owner._lose_proof = observed_loss
    def kill():
        owner._helper._child.kill()
        owner._helper._child.wait(timeout=5)
    phase = sys.argv[2]
    if phase == "initial":
        kill()
    else:
        method = "verified_parent_fd" if phase == "directory_accessor" else "close"
        original = getattr(owner, method)
        if phase == "directory_accessor":
            assert repository._reusable_tombstones, "real restore must require settlement"
        def lose_at_phase(*args, **kwargs):
            events.append(phase)
            kill()
            return original(*args, **kwargs)
        setattr(owner, method, lose_at_phase)
    for attempt in range(2):
        try:
            await repository.close()
        except ProfileRepositoryError as error:
            outcome = (error.code, repository._helper_restart_required)
            assert outcome == ("restart_required", True), (phase, attempt, outcome, events)
            assert error.__cause__ is None
            assert str(repository._active_database_path) not in str(error)
        else:
            raise AssertionError("lost-proof close reported success")
        assert repository._connection is owner and not owner._sqlite_closed
        assert repository._lease is lease and lease.acquired
        assert repository._executor is executor and not repository._executor_shutdown
        assert repository._exact_authority_quarantined
        assert HELPER_ADMISSION.tts_proof_lost
        assert os.fstat(owner._parent_fd).st_nlink > 0
    observed = events.index("observed_loss")
    assert events[observed + 1:] == [], events
asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path / "profiles.sqlite3"), phase],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("phase", ["policy", "metadata", "repository", "control"])
def test_helper_loss_during_repository_publication_retains_owner(tmp_path, phase):
    program = """
import asyncio, sys
from pathlib import Path
import Tests.conftest
from tldw_chatbook.TTS import profile_repository as module
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS import profile_schema as schema
from tldw_chatbook.DB.private_sqlite_process import HELPER_ADMISSION
repository = module.TTSProfileRepository(Path(sys.argv[1]))
phase = sys.argv[2]
class ControlFlow(BaseException):
    pass
signal = ControlFlow()
owners = []
initialize = schema._ExactCurrentProfileConnection.__init__
def capture(self, *args, **kwargs):
    initialize(self, *args, **kwargs)
    owners.append(self)
    if phase == "policy":
        self._helper._child.kill()
        self._helper._child.wait(timeout=5)
schema._ExactCurrentProfileConnection.__init__ = capture
metadata = schema._stream_exact_store_metadata_evidence
def after_metadata(connection):
    result = metadata(connection)
    if phase in {"metadata", "control"}:
        connection._helper._child.kill()
        connection._helper._child.wait(timeout=5)
        if phase == "control":
            raise signal
    return result
schema._stream_exact_store_metadata_evidence = after_metadata
original = module.revalidate_exact_current_profile_store
def lose(connection, path):
    if phase == "repository":
        connection._helper._child.kill()
        connection._helper._child.wait(timeout=5)
    return original(connection, path)
module.revalidate_exact_current_profile_store = lose
async def main():
    try:
        await repository.open()
    except ControlFlow as error:
        assert phase == "control" and error is signal
    except ProfileRepositoryError as error:
        assert phase != "control"
        assert error.code == "restart_required", error.code
    else:
        raise AssertionError("partial owner published")
    assert repository._connection is not None
    assert repository._lease is not None
    assert repository._executor is not None
    assert repository._helper_restart_required
    assert repository._connection is owners[-1]
    assert not owners[-1]._sqlite_closed
    assert HELPER_ADMISSION._used == {"retained": 1, "transient": 0}
    assert HELPER_ADMISSION.tts_proof_lost
    try:
        await repository.close()
    except ProfileRepositoryError as error:
        assert error.code == "restart_required", error.code
    else:
        raise AssertionError("terminal partial owner closed")
asyncio.run(main())
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path / "profiles.sqlite3"), phase],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("exit_kind", ["orderly", "abrupt"])
@pytest.mark.parametrize(
    "state,owner_count",
    [(state, count) for state in ("idle", "read", "write") for count in (1, 2)]
    + [("partial", 1)],
)
def test_actual_app_retained_repository_exit_preserves_foreign_cohort(
    tmp_path, exit_kind, state, owner_count
):
    path = tmp_path / "profiles.sqlite3"
    seed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import Tests.conftest; from pathlib import Path; from tldw_chatbook.TTS.profile_schema import open_profile_store; import sys; open_profile_store(Path(sys.argv[1])).close()",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert seed.returncode == 0, seed.stderr
    with (tmp_path / "child-stderr.txt").open("w+") as stderr:
        child = subprocess.Popen(
            [sys.executable, "-c", _APP_EXIT_CHILD, str(path), state, str(owner_count)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
        )
        foreign = []
        try:
            _phase(child, "ready", stderr)
            for suffix in ("-wal", "-shm"):
                selected = Path(f"{path}{suffix}")
                selected.rename(tmp_path / f"original{suffix}")
                content = b"foreign cohort sentinel" + suffix.encode()
                with selected.open("xb") as stream:
                    stream.write(content)
                selected.chmod(0o600)
                observer = os.open(selected, os.O_RDONLY)
                foreign.append((selected, observer, os.fstat(observer), content))
            child.stdin.write("substituted\n")
            child.stdin.flush()
            _phase(child, "retained", stderr)
            with pytest.raises(ProfileRepositoryError):
                ProfileStoreLease(
                    path, ProfileStoreLockMode.EXCLUSIVE, timeout_seconds=0.1
                ).acquire()
            if exit_kind == "orderly":
                child.stdin.write("exit\n")
                child.stdin.flush()
                assert child.wait(timeout=15) == 0
            else:
                child.kill()
                assert child.wait(timeout=5) == -signal.SIGKILL
            exclusive = ProfileStoreLease(
                path, ProfileStoreLockMode.EXCLUSIVE, timeout_seconds=0.1
            ).acquire()
            exclusive.release()
            for selected, observer, identity, content in foreign:
                assert selected.stat().st_ino == identity.st_ino
                assert os.fstat(observer).st_nlink == 1
                assert os.pread(observer, len(content) + 1, 0) == content
            recovery_dir = tmp_path / "recovery"
            recovery_dir.mkdir(mode=0o700)
            recovery = recovery_dir / "original.sqlite3"
            shutil.copyfile(path, recovery)
            shutil.copyfile(tmp_path / "original-wal", Path(f"{recovery}-wal"))
            recovered = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    "import sqlite3, sys; c=sqlite3.connect(sys.argv[1]); rows=c.execute('SELECT COUNT(*) FROM tts_generation_profiles' if sys.argv[2] == 'partial' else 'SELECT value, hex(payload) FROM shutdown_gate').fetchall(); assert rows == ([(0,)] if sys.argv[2] == 'partial' else [(1, '01')]), rows; assert c.execute('PRAGMA integrity_check').fetchone() == ('ok',); c.close()",
                    str(recovery),
                    state,
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            assert recovered.returncode == 0, recovered.stderr
            for selected, observer, identity, content in foreign:
                assert selected.stat().st_ino == identity.st_ino
                assert os.fstat(observer).st_nlink == 1
                assert os.pread(observer, len(content) + 1, 0) == content
        finally:
            if child.poll() is None:
                child.kill()  # Contain only this failed test's captured child.
                child.wait(timeout=5)
            for _selected, observer, _identity, _content in foreign:
                os.close(observer)
            child.stdin.close()
            child.stdout.close()
