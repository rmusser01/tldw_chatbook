"""Real migration/publication native lifetime evidence in private processes."""

import pytest

from Tests.TTS.test_profile_repository_maintenance import _run_private_child


async def _recovery_close_child(root, after):
    import os
    import time
    import types

    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_profile_migration_publication import _store
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import profile_migration_recovery as recovery
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository

    active = root / "profiles.sqlite"
    _store(active, version=2, marker="historical-v2")
    original_parent = recovery.private_paths._open_verified_parent
    original_close = os.close
    selected = []
    observed_holds = []
    calls = []
    recovery.private_paths = types.SimpleNamespace(**vars(recovery.private_paths))
    recovery.os = types.SimpleNamespace(**vars(os))

    def parent(*args, **kwargs):
        result = original_parent(*args, **kwargs)
        selected.append(result[0])
        observed_holds.extend(storage._holds.values())
        return result

    def close(fd):
        if selected and fd == selected[0]:
            calls.append(fd)
            if after:
                original_close(fd)
            raise OSError("recovery parent close outcome unknown")
        return original_close(fd)

    recovery.private_paths._open_verified_parent = parent
    recovery.os.close = close
    repo = TTSProfileRepository(active)
    with pytest.raises(ProfileRepositoryError):
        await repo.open()
    assert selected and calls, [
        repr(n.body_error) for n in repo._migration_native_operations
    ]
    hold = observed_holds[0]
    repo._maintenance_close_admission()
    drained = await repo._maintenance_drain(time.monotonic() + 3)
    storage._shutdown()
    native = _probe(hold.authority.control_root, hold.names)
    assert native == "blocked", (drained, native, "native parent uncertainty escaped")
    assert not drained


@pytest.mark.parametrize("after", [False, True])
def test_initialize_recovery_parent_uncertainty_keeps_native_exclusion(tmp_path, after):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_migration_native_maintenance import _recovery_close_child
asyncio.run(_recovery_close_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(after),
    )


async def _publication_close_child(root, after, restore):
    import os
    import time
    import types

    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_profile_migration_publication import _store
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import profile_migration_publication as publication
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository

    active = root / "profiles.sqlite"
    repo = TTSProfileRepository(active)
    if restore:
        await repo.open()
        candidate = root / "historical.sqlite"
        _store(candidate, version=2, marker="historical-v2")
    else:
        _store(active, version=2, marker="historical-v2")
    actual_open = publication._open_exact
    actual_close = os.close
    selected = []
    calls = []
    observed_holds = []
    publication.os = types.SimpleNamespace(**vars(os))

    def open_exact(identity, **kwargs):
        result = actual_open(identity, **kwargs)
        if not selected:
            selected.append(result[1])
            observed_holds.extend(storage._holds.values())
        return result

    def close(fd):
        if selected and fd == selected[0] and not calls:
            calls.append(fd)
            if after:
                actual_close(fd)
            raise OSError("publication file close outcome unknown")
        return actual_close(fd)

    publication._open_exact = open_exact
    publication.os.close = close
    with pytest.raises(ProfileRepositoryError):
        if restore:
            await repo.restore_from(candidate)
        else:
            await repo.open()
    assert selected and calls, [
        repr(n.body_error) for n in repo._migration_native_operations
    ]
    from tldw_chatbook.TTS.profile_migration_journal import (
        PROFILE_MIGRATION_CANDIDATE_LEAVES,
        ProfileMigrationPublicationSlot,
    )

    preserved = active.with_name(
        PROFILE_MIGRATION_CANDIDATE_LEAVES[ProfileMigrationPublicationSlot.ACTIVE]
    )
    assert preserved.exists(), "compound cleanup removed unresolved native source"
    hold = observed_holds[0]
    repo._maintenance_close_admission()
    drained = await repo._maintenance_drain(time.monotonic() + 3)
    storage._shutdown()
    native = _probe(hold.authority.control_root, hold.names)
    assert native == "blocked", (drained, native, "publication uncertainty escaped")
    assert not drained


@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize("restore", [False, True])
def test_publication_native_uncertainty_keeps_native_exclusion(
    tmp_path, after, restore
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_migration_native_maintenance import _publication_close_child
asyncio.run(_publication_close_child(Path(sys.argv[1]), sys.argv[2] == 'True', sys.argv[3] == 'True'))
""",
        str(after),
        str(restore),
    )


async def _pause_publication_child(root, phase, restore):
    import asyncio
    import time

    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import profile_repository as repository
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    active = root / "profiles.sqlite"
    repo = repository.TTSProfileRepository(active)
    if restore:
        await repo.open()
        candidate = root / "historical.sqlite"
        reference = _historical_v3(candidate)
        incoming_bytes = candidate.read_bytes()
    else:
        reference = _historical_v3(active)
    actual_publish = repository.publish_profile_migration
    pauses = []
    observed = []
    prior_bytes = []
    prepared_references = []

    def publish(**kwargs):
        original_hook = kwargs.get("stage_hook")

        def hook(stage):
            if original_hook is not None:
                original_hook(stage)
            observed.append(stage.value)
            if stage.value == "preflight":
                prior_bytes.append(active.read_bytes())
                import sqlite3

                for artifact in (
                    kwargs["active_candidate"],
                    *kwargs["backup_candidates"],
                ):
                    connection = sqlite3.connect(
                        artifact._path.as_uri() + "?mode=ro&immutable=1", uri=True
                    )
                    try:
                        payload = connection.execute(
                            "SELECT wav_bytes FROM tts_profile_clone_references"
                        ).fetchone()[0]
                        assert payload == reference.wav_bytes
                        prepared_references.append(payload)
                    finally:
                        connection.close()
            if stage.value == phase:
                pauses.append(storage._begin_local_pause())
                from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
                from tldw_chatbook.TTS.profile_migration_journal import (
                    ProfileMigrationPublicationSlot,
                )
                from tldw_chatbook.TTS.profile_migration_publication import (
                    prepare_profile_migration_artifact,
                )

                with pytest.raises(RecoveryRequired):
                    prepare_profile_migration_artifact(
                        active, slot=ProfileMigrationPublicationSlot.ACTIVE
                    )

        kwargs["stage_hook"] = hook
        return actual_publish(**kwargs)

    repository.publish_profile_migration = publish
    result = None
    try:
        result = await (repo.restore_from(candidate) if restore else repo.open())
    except ProfileRepositoryError as error:
        result = error.code
    assert len(pauses) == 1, observed
    assert len(prior_bytes) == 1 and len(prepared_references) == 2
    assert result == (
        "unavailable"
        if phase == "ponr"
        else "restore_failed"
        if restore
        else "migration_failed"
    )
    repo._maintenance_close_admission()
    drained = await repo._maintenance_drain(time.monotonic() + 3)
    import json
    import sqlite3

    from tldw_chatbook.TTS.profile_migration_journal import (
        parse_profile_migration_journal,
    )
    from tldw_chatbook.TTS.profile_migration_recovery import (
        recover_profile_migration_publication,
    )

    journal = active.with_name("." + active.name + ".migration-publication.json")
    phase_before = (
        parse_profile_migration_journal(journal.read_bytes()).phase
        if journal.exists()
        else None
    )

    def inspect():
        connection = sqlite3.connect(active.as_uri() + "?mode=ro&immutable=1", uri=True)
        try:
            return connection.execute("PRAGMA user_version").fetchone()[
                0
            ], connection.execute("PRAGMA application_id").fetchone()[0]
        finally:
            connection.close()

    assert drained
    assert phase_before == ("publishing" if phase == "ponr" else None)
    assert active.read_bytes() == prior_bytes[0]
    before = await asyncio.wrap_future(repo._executor.submit(inspect))
    await asyncio.wrap_future(repo._executor.submit(pauses[0].resume))
    recovered = await asyncio.wrap_future(
        repo._executor.submit(recover_profile_migration_publication, active)
    )
    assert not await asyncio.wrap_future(
        repo._executor.submit(recover_profile_migration_publication, active)
    )
    after = await asyncio.wrap_future(repo._executor.submit(inspect))
    assert not journal.exists()
    # Recovery converges to exactly prior or completely migrated authority.
    assert after == before
    assert after[0] == (4 if restore else 3)
    assert active.read_bytes() == prior_bytes[0]
    assert recovered is (phase == "ponr")
    if restore:
        assert candidate.read_bytes() == incoming_bytes
    (root / "pause-proof.json").write_text(
        json.dumps(
            {
                "phase": phase,
                "restore": restore,
                "result": str(result),
                "drained": drained,
                "journal": phase_before,
                "before": before,
                "recovered": recovered,
                "after": after,
            }
        )
    )
    await repo.close()


@pytest.mark.parametrize("phase", ["preflight", "ponr"])
@pytest.mark.parametrize("restore", [False, True])
def test_observe_actual_publication_pause_boundary(tmp_path, phase, restore):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_migration_native_maintenance import _pause_publication_child
asyncio.run(_pause_publication_child(Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'True'))
""",
        phase,
        str(restore),
    )
    print((tmp_path / "pause-proof.json").read_text())


async def _publication_reader_child(root, after, restore):
    import time

    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_profile_migration_publication import _store
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import profile_migration_publication as publication
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
    from tldw_chatbook.TTS.profile_migration_journal import (
        PROFILE_MIGRATION_CANDIDATE_LEAVES,
        ProfileMigrationPublicationSlot,
    )
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository

    active = root / "profiles.sqlite"
    repo = TTSProfileRepository(active)
    if restore:
        await repo.open()
        candidate = root / "historical.sqlite"
        _store(candidate, version=2, marker="historical-v2")
    else:
        _store(active, version=2, marker="historical-v2")
    original = publication.connect_private_sqlite_descriptor
    opened = []
    calls = []
    held = []

    class Reader:
        def __init__(self, native):
            object.__setattr__(self, "native", native)

        def __getattr__(self, name):
            return getattr(self.native, name)

        def __setattr__(self, name, value):
            setattr(self.native, name, value)

        def close(self):
            calls.append(self.native)
            if after:
                self.native.close()
            raise OSError("descriptor reader close uncertainty")

    def connect(*args, **kwargs):
        native = original(*args, **kwargs)
        opened.append(native)
        held.extend(storage._holds.values())
        return Reader(native)

    publication.connect_private_sqlite_descriptor = connect
    with pytest.raises(ProfileRepositoryError):
        await (repo.restore_from(candidate) if restore else repo.open())
    assert len(opened) == len(calls) == 1
    # A failed child reader must remain associated with its compound namespace,
    # even if the existing privateSQLite lease independently retains the child.
    preserved = active.with_name(
        PROFILE_MIGRATION_CANDIDATE_LEAVES[ProfileMigrationPublicationSlot.ACTIVE]
    )
    assert preserved.exists(), "publication discarded unresolved reader source"
    repo._maintenance_close_admission()
    drained = await repo._maintenance_drain(time.monotonic() + 3)
    storage._shutdown()
    native = _probe(held[0].authority.control_root, held[0].names)
    assert not drained and native == "blocked", (drained, native)


@pytest.mark.parametrize("after", [False, True])
@pytest.mark.parametrize("restore", [False, True])
def test_descriptor_reader_uncertainty_preserves_compound_source(
    tmp_path, after, restore
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_migration_native_maintenance import _publication_reader_child
asyncio.run(_publication_reader_child(Path(sys.argv[1]), sys.argv[2] == 'True', sys.argv[3] == 'True'))
""",
        str(after),
        str(restore),
    )


def _mismatch_child(root):
    from Tests.TTS.test_profile_migration_publication import _store
    from tldw_chatbook.TTS.profile_migration_native import _migration_native
    from tldw_chatbook.TTS.profile_migration_recovery import (
        recover_profile_migration_publication,
    )

    first = root / "first.sqlite"
    second = root / "second.sqlite"
    _store(first, version=2, marker="first")
    original = _store(second, version=2, marker="second")
    before = second.stat()
    with _migration_native((first,)) as operation:
        leases = tuple(operation.leases)
        with pytest.raises(ValueError, match="migration_native_source_mismatch"):
            recover_profile_migration_publication(second, _native=operation)
        assert tuple(operation.leases) == leases
        assert not operation.descriptors and not operation.pending
    assert second.read_bytes() == original
    assert second.stat().st_mtime_ns == before.st_mtime_ns


def test_foreign_helper_source_refuses_before_io(tmp_path):
    _run_private_child(
        tmp_path,
        """
import sys
from pathlib import Path
from Tests.TTS.test_profile_migration_native_maintenance import _mismatch_child
_mismatch_child(Path(sys.argv[1]))
""",
    )


def _historical_v3(path):
    """Run original v0->v3 DDL and insert valid historical reference data."""
    from Tests.TTS.test_profile_reference_repository import REFERENCE_A, _canonical
    from Tests.TTS.test_profile_schema import _build_candidate_version

    connection = _build_candidate_version(path, 3)
    try:
        connection.execute(
            "UPDATE tts_generation_profiles SET provider_id='audio_cpp', model_id='clone-model', voice_id=NULL, response_format='wav'"
        )
        profile_id = connection.execute(
            "SELECT profile_id FROM tts_generation_profiles"
        ).fetchone()[0]
        canonical = _canonical(sample=7)
        connection.execute(
            """INSERT INTO tts_profile_clone_references (
            profile_id, reference_id, wav_bytes, reference_text, sha256, byte_length,
            duration_ms, sample_rate_hz, channels, sample_encoding, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                profile_id,
                str(REFERENCE_A),
                canonical.wav_bytes,
                canonical.reference_text,
                canonical.sha256,
                canonical.byte_length,
                canonical.duration_ms,
                canonical.sample_rate_hz,
                canonical.channels,
                canonical.sample_encoding,
                "2026-08-10T12:00:00.000000Z",
                "2026-08-10T12:00:00.000000Z",
            ),
        )
        connection.commit()
    finally:
        connection.close()
    path.chmod(0o600)
    return canonical


def _descriptor_fault_child(root, role, after, recovery_mode=False):
    import os
    import sqlite3
    import time
    import types

    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_profile_migration_publication import _store
    from Tests.TTS.test_profile_migration_recovery import (
        _publication_fixture,
        _write_journal,
    )
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.DB import private_sqlite as private
    from tldw_chatbook.TTS import profile_migration_publication as publication
    from tldw_chatbook.TTS import profile_migration_recovery as recovery
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
    from tldw_chatbook.TTS.profile_migration_journal import (
        ProfileMigrationPublicationSlot,
    )

    if recovery_mode:
        _, active, artifacts, destinations = _publication_fixture(
            root, slots=("active",)
        )
        original_bytes = active.read_bytes()
        journal = _write_journal(
            publication, active, artifacts, destinations, phase="prepared"
        )
        call = lambda: recovery.recover_profile_migration_publication(active)
    else:
        active = root / ".profile-migration-active.candidate.sqlite3"
        original_bytes = _store(active, version=4, marker="standalone")
        call = lambda: publication.prepare_profile_migration_artifact(
            active, slot=ProfileMigrationPublicationSlot.ACTIVE
        )
    allocated = []
    observed = []
    calls = []
    private.os = types.SimpleNamespace(**vars(os))
    original_dup, original_close = os.dup, os.close
    original_connect = private._SQLITE_CONNECT
    original_registered = private._connect_registered_sqlite

    def duplicate(fd):
        value = original_dup(fd)
        observed.extend(storage._holds.values())
        allocated.append(value)
        if role == "unreturned_dup":
            raise OSError("allocated duplicate before return error")
        return value

    private.os.dup = duplicate

    def close(fd):
        if allocated and fd == allocated[0] and not calls:
            calls.append(fd)
            if after:
                original_close(fd)
            raise OSError("duplicate close uncertainty")
        return original_close(fd)

    if role == "dup_close":
        private.os.close = close
    native_connections = []
    if role == "constructor":

        class Constructor(sqlite3.Connection):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                native_connections.append(self)
                raise OSError("native custom constructor allocated then failed")

        def connect(*args, **kwargs):
            return original_connect(*args, factory=Constructor, **kwargs)

        private._SQLITE_CONNECT = connect
    elif role == "registered_return":

        def registered(*args, **kwargs):
            value = original_registered(*args, **kwargs)
            native_connections.append(value)
            raise OSError("registered connector returned native before wrapper failed")

        private._connect_registered_sqlite = registered
    elif role == "reader":

        def connect(*args, **kwargs):
            class Reader(sqlite3.Connection):
                def close(self):
                    calls.append(self)
                    if after:
                        super().close()
                    raise OSError("recovery native reader close uncertainty")

            native = original_connect(*args, factory=Reader, **kwargs)
            native_connections.append(native)
            return native

        private._SQLITE_CONNECT = connect
    with pytest.raises((ProfileRepositoryError, OSError)):
        call()
    assert allocated and observed
    assert active.read_bytes() == original_bytes
    if recovery_mode:
        assert journal.exists()
    if role == "dup_close":
        assert calls == [allocated[0]]
        if after:
            with pytest.raises(OSError):
                os.fstat(allocated[0])
        else:
            assert os.fstat(allocated[0]).st_ino == active.stat().st_ino
    if role == "constructor":
        assert native_connections[0].execute("PRAGMA user_version").fetchone()[0] == 4
    if role == "registered_return":
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            native_connections[0].execute("SELECT 1")
    pause = storage._begin_local_pause()
    try:
        assert not pause.drain(time.monotonic() + 0.01)
    finally:
        pause.resume()
    storage._shutdown()
    assert _probe(observed[0].authority.control_root, observed[0].names) == "blocked"


@pytest.mark.parametrize(
    "role,after,recovery_mode",
    [
        ("dup_close", False, False),
        ("dup_close", True, False),
        ("unreturned_dup", False, False),
        ("constructor", False, False),
        ("registered_return", False, False),
        ("reader", False, True),
        ("reader", True, True),
    ],
)
def test_descriptor_actual_native_outcomes(tmp_path, role, after, recovery_mode):
    _run_private_child(
        tmp_path,
        """
import sys
from pathlib import Path
from Tests.TTS.test_profile_migration_native_maintenance import _descriptor_fault_child
_descriptor_fault_child(Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'True', sys.argv[4] == 'True')
""",
        role,
        str(after),
        str(recovery_mode),
    )


async def _restore_source_child(root, occurrence, after):
    import time

    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import profile_repository as repository
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    active = root / "profiles.sqlite"
    candidate = root / "incoming.sqlite"
    _historical_v3(candidate)
    candidate_bytes = candidate.read_bytes()
    repo = repository.TTSProfileRepository(active)
    await repo.open()
    original = repository.connect_private_sqlite
    matching = []
    calls = []
    observed = []

    class Source:
        def __init__(self, connection):
            object.__setattr__(self, "connection", connection)

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def __setattr__(self, name, value):
            setattr(self.connection, name, value)

        def close(self):
            calls.append(self.connection)
            if len(calls) == 1:
                if after:
                    self.connection.close()
                raise OSError("restore candidate source close uncertainty")
            self.connection.close()

    def connect(owner, path, **kwargs):
        native = original(owner, path, **kwargs)
        if owner == "tts.profile_restore_stage" and path == candidate:
            matching.append(native)
            if len(matching) == occurrence:
                observed.extend(storage._holds.values())
                return Source(native)
        return native

    repository.connect_private_sqlite = connect
    with pytest.raises(ProfileRepositoryError):
        await repo.restore_from(candidate)
    assert len(matching) >= occurrence and calls
    operations = tuple(repo._migration_native_operations)
    assert len(operations) == 1
    operation = operations[0]
    assert set(operation.sources) == {active, candidate}
    assert any(
        item.path == candidate and item.close_attempted and not item.closed
        for item in operation.source_connections
    )
    repo._maintenance_close_admission()
    drained = await repo._maintenance_drain(time.monotonic() + 3)
    assert candidate.read_bytes() == candidate_bytes
    storage._shutdown()
    native = _probe(observed[0].authority.control_root, observed[0].names)
    assert not drained and native == "blocked", (drained, native, len(calls))


@pytest.mark.parametrize("occurrence", [1, 2, 3])
@pytest.mark.parametrize("after", [False, True])
def test_restore_candidate_source_close_cannot_be_normalized_into_drain(
    tmp_path, occurrence, after
):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_migration_native_maintenance import _restore_source_child
asyncio.run(_restore_source_child(Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == 'True'))
""",
        str(occurrence),
        str(after),
    )


async def _v3_restore_success_child(root, corrupt):
    import sqlite3

    from Tests.TTS.test_profile_schema import PROFILE_ID
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository

    candidate = root / "incoming.sqlite"
    reference = _historical_v3(candidate)
    if corrupt:
        connection = sqlite3.connect(candidate)
        try:
            connection.execute(
                "UPDATE tts_profile_clone_references SET sha256 = ?", ("0" * 64,)
            )
            connection.commit()
        finally:
            connection.close()
    before = candidate.read_bytes()
    repo = TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    try:
        if corrupt:
            with pytest.raises(ProfileRepositoryError):
                await repo.restore_from(candidate)
            assert (await repo.list_profiles()).value.profiles == ()
        else:
            result = await repo.restore_from(candidate)
            assert result.value.profile_count == 1
            profile = await repo.get_profile(PROFILE_ID)
            restored = await repo.get_reference(
                PROFILE_ID,
                expected_revision=profile.value.revision,
                expected_generation=repo.generation,
            )
            assert restored.value.wav_bytes == reference.wav_bytes
            assert restored.value.reference_text == reference.reference_text
        assert candidate.read_bytes() == before
    finally:
        await repo.close()


@pytest.mark.parametrize("corrupt", [False, True])
def test_genuine_v3_reference_public_restore(tmp_path, corrupt):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_migration_native_maintenance import _v3_restore_success_child
asyncio.run(_v3_restore_success_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(corrupt),
    )


def _partial_nonwal_child(root, foreign, fault=""):
    import sqlite3

    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_profile_schema import _build_candidate_version
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import profile_schema as schema

    active = root / "profiles.sqlite"
    fixture = _build_candidate_version(active, 4)
    fixture.close()
    active.chmod(0o600)
    before = active.read_bytes()
    original = schema.connect_private_sqlite
    observed, natives, closes = [], [], []
    sentinel = active.with_name(active.name + "-wal")

    class Live:
        def __init__(self, native):
            self.native = native

        def __getattr__(self, name):
            return getattr(self.native, name)

        def execute(self, statement, *args):
            cursor = self.native.execute(statement, *args)
            if statement == "PRAGMA journal_mode" and foreign:
                sentinel.write_bytes(b"foreign sidecar stays exact")
                sentinel.chmod(0o600)
            if statement == "PRAGMA journal_mode":
                if fault == "unknown_mode":
                    raise OSError("mode outcome unavailable")
                if fault == "foreign_main":
                    active.rename(active.with_name("original.sqlite"))
                    active.write_bytes(b"foreign main stays exact")
                    active.chmod(0o600)
            return cursor

        def close(self):
            closes.append(self.native)
            if fault == "close_before":
                raise OSError("close outcome unavailable")
            self.native.close()
            if fault == "close_after":
                raise OSError("close outcome unavailable")

    def connect(*args, **kwargs):
        native = original(*args, **kwargs)
        natives.append(native)
        observed.extend(storage._holds.values())
        return Live(native)

    schema.connect_private_sqlite = connect
    expected = (
        schema.ExactProfileStoreCleanupError
        if foreign or fault
        else schema.ExactProfileStoreNotCurrentError
    )
    with pytest.raises(expected):
        schema.open_exact_current_profile_store(active)
    assert natives
    if foreign:
        assert not closes
        assert sentinel.read_bytes() == b"foreign sidecar stays exact"
    elif fault:
        assert len(closes) == (1 if fault.startswith("close_") else 0)
    else:
        assert len(closes) == 1
        with pytest.raises(sqlite3.ProgrammingError):
            natives[0].execute("SELECT 1")
    assert active.read_bytes() == (
        b"foreign main stays exact" if fault == "foreign_main" else before
    )
    storage._shutdown()
    admission = _probe(observed[0].authority.control_root, observed[0].names)
    assert admission == ("blocked" if foreign or fault else "entered")


@pytest.mark.parametrize("foreign", [False, True])
def test_partial_nonwal_exact_open_closes_only_proven_namespace(tmp_path, foreign):
    _run_private_child(
        tmp_path,
        """
import sys
from pathlib import Path
from Tests.TTS.test_profile_migration_native_maintenance import _partial_nonwal_child
_partial_nonwal_child(Path(sys.argv[1]), sys.argv[2] == 'True')
""",
        str(foreign),
    )


@pytest.mark.parametrize(
    "fault", ["unknown_mode", "foreign_main", "close_before", "close_after"]
)
def test_partial_nonwal_native_uncertainty_remains_excluded(tmp_path, fault):
    _run_private_child(
        tmp_path,
        """
import sys
from pathlib import Path
from Tests.TTS.test_profile_migration_native_maintenance import _partial_nonwal_child
_partial_nonwal_child(Path(sys.argv[1]), False, sys.argv[2])
""",
        fault,
    )


async def _initialize_remaining_child(root, role, after):
    import os
    import time
    import types

    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import profile_repository as repository
    from tldw_chatbook.TTS import profile_schema as schema
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    active = root / "profiles.sqlite"
    _historical_v3(active)
    observed, calls, matching = [], [], []

    class Source:
        def __init__(self, native):
            object.__setattr__(self, "native", native)

        def __getattr__(self, name):
            return getattr(self.native, name)

        def __setattr__(self, name, value):
            setattr(self.native, name, value)

        def close(self):
            calls.append(self.native)
            if len(calls) == 1:
                if after:
                    self.native.close()
                raise OSError("initialization source close uncertainty")
            self.native.close()

    if role == "version":
        original = repository.connect_private_sqlite

        def connect(owner, path, **kwargs):
            native = original(owner, path, **kwargs)
            if owner == "tts.profile_migration_backup" and path == active:
                matching.append(native)
                if len(matching) == 1:
                    observed.extend(storage._holds.values())
                    return Source(native)
            return native

        repository.connect_private_sqlite = connect
    elif role == "initialize":
        original = repository.open_profile_store

        def connect(*args, **kwargs):
            native = original(*args, **kwargs)
            observed.extend(storage._holds.values())
            return Source(native)

        repository.open_profile_store = connect
    else:
        original_capture = repository.capture_post_init_profile_store_authority
        enabled = []
        original_open, original_close = os.open, os.close
        native_os = schema.os
        capture_os = types.SimpleNamespace(**vars(os))

        def opening(*args, **kwargs):
            fd = original_open(*args, **kwargs)
            if enabled:
                matching.append(fd)
                observed.extend(storage._holds.values())
            return fd

        def closing(fd):
            if enabled and matching and fd == matching[0]:
                calls.append(fd)
                if after:
                    original_close(fd)
                raise OSError("postinit descriptor close uncertainty")
            original_close(fd)

        def capture(*args, **kwargs):
            enabled.append(True)
            schema.os = capture_os
            try:
                return original_capture(*args, **kwargs)
            finally:
                enabled.clear()
                schema.os = native_os

        capture_os.open, capture_os.close = opening, closing
        repository.capture_post_init_profile_store_authority = capture
    repo = repository.TTSProfileRepository(active)
    with pytest.raises(ProfileRepositoryError) as caught:
        await repo.open()
    assert calls, (
        str(caught.value),
        [
            (str(n.body_error), [repr(e) for e in n.errors])
            for n in repo._migration_native_operations
        ],
        matching,
    )
    repo._maintenance_close_admission()
    drained = await repo._maintenance_drain(time.monotonic() + 3)
    storage._shutdown()
    native = _probe(observed[0].authority.control_root, observed[0].names)
    assert not drained and native == "blocked", (role, drained, native, len(calls))


@pytest.mark.parametrize("role", ["version", "initialize", "postinit"])
@pytest.mark.parametrize("after", [False, True])
def test_initialize_remaining_native_outcomes_are_retained(tmp_path, role, after):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_migration_native_maintenance import _initialize_remaining_child
asyncio.run(_initialize_remaining_child(Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'True'))
""",
        role,
        str(after),
    )


async def _handoff_parent_child(root, role, after):
    import os
    import time
    import types

    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import profile_repository as repository
    from tldw_chatbook.TTS import profile_schema as schema
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    active = root / "profiles.sqlite"
    repo = repository.TTSProfileRepository(active)
    if role == "revalidate":
        await repo.open()
    module = schema if role == "revalidate" else repository
    original_parent = module.private_paths._open_verified_parent
    original_close = os.close
    observed, selected, calls = [], [], []
    enabled = [True]
    module.private_paths = types.SimpleNamespace(**vars(module.private_paths))
    module.os = types.SimpleNamespace(**vars(module.os))

    def parent(*args, **kwargs):
        result = original_parent(*args, **kwargs)
        if enabled and not selected:
            selected.append(result[0])
            observed.extend(storage._holds.values())
        return result

    def close(fd):
        if enabled and selected and fd == selected[0]:
            calls.append(fd)
            if after:
                original_close(fd)
            raise OSError("handoff parent close uncertainty")
        original_close(fd)

    module.private_paths._open_verified_parent = parent
    module.os.close = close
    with pytest.raises(ProfileRepositoryError):
        if role == "revalidate":
            await repo.list_profiles()
        else:
            await repo.open()
    enabled.clear()
    assert calls
    repo._maintenance_close_admission()
    drained = await repo._maintenance_drain(time.monotonic() + 3)
    storage._shutdown()
    native = _probe(observed[0].authority.control_root, observed[0].names)
    assert not drained and native == "blocked", (role, drained, native)


@pytest.mark.parametrize("role", ["shared", "revalidate"])
@pytest.mark.parametrize("after", [False, True])
def test_handoff_parent_native_close_does_not_escape_owner(tmp_path, role, after):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_migration_native_maintenance import _handoff_parent_child
asyncio.run(_handoff_parent_child(Path(sys.argv[1]), sys.argv[2], sys.argv[3] == 'True'))
""",
        role,
        str(after),
    )


def _standalone_outcome_child(root, mode):
    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from Tests.TTS.test_profile_schema import _build_candidate_version
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.TTS import profile_schema as schema
    from tldw_chatbook.TTS.profile_migration_recovery import (
        recover_profile_migration_publication,
    )

    active = root / "profiles.sqlite"
    fixture = _build_candidate_version(active, 4)
    fixture.close()
    active.chmod(0o600)
    before = active.read_bytes()
    observed, calls = [], []
    if mode == "refusal":
        pause = storage._begin_local_pause()
        try:
            with pytest.raises(RecoveryRequired):
                schema.capture_post_init_profile_store_authority(active)
            with pytest.raises(RecoveryRequired):
                recover_profile_migration_publication(active)
        finally:
            pause.resume()
        assert not any(
            getattr(lease, "native_owner", None).__class__.__name__
            == "_MigrationNativeState"
            for lease in storage._live_leases
        )
    elif mode == "success":
        result = schema.capture_post_init_profile_store_authority(active)
        assert result.file_identity.st_ino == active.stat().st_ino
        assert recover_profile_migration_publication(active) is False
    else:

        class Stop(BaseException):
            pass

        signal = Stop("cleanup control flow")
        original = storage.StorageLease.close

        def close(lease):
            owner = getattr(lease, "native_owner", None)
            if (
                owner is not None
                and owner.__class__.__name__ == "_MigrationNativeState"
                and not calls
            ):
                calls.append(lease)
                observed.extend(storage._holds.values())
                raise signal
            original(lease)

        storage.StorageLease.close = close
        with pytest.raises(Stop) as caught:
            schema.capture_post_init_profile_store_authority(active)
        assert caught.value is signal
        assert calls
        storage._shutdown()
        assert (
            _probe(observed[0].authority.control_root, observed[0].names) == "blocked"
        )
    assert active.read_bytes() == before


@pytest.mark.parametrize("mode", ["refusal", "success", "control"])
def test_standalone_migration_outcome_preserves_retirement_and_control_flow(
    tmp_path, mode
):
    _run_private_child(
        tmp_path,
        """
import sys
from pathlib import Path
from Tests.TTS.test_profile_migration_native_maintenance import _standalone_outcome_child
_standalone_outcome_child(Path(sys.argv[1]), sys.argv[2])
""",
        mode,
    )


async def _historical_blob_child(root, after):
    import asyncio
    import sqlite3
    import time

    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import profile_repository as repository
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    active = root / "profiles.sqlite"
    _historical_v3(active)
    before = active.read_bytes()
    original = repository.connect_private_sqlite
    blobs, parents, observed = [], [], []

    class Blob:
        def __init__(self, native):
            self.native = native

        def __getattr__(self, name):
            return getattr(self.native, name)

        def __len__(self):
            return len(self.native)

        def close(self):
            if after:
                self.native.close()
            raise OSError("historical blob close uncertainty")

    def connect(owner, path, **kwargs):
        connection = original(owner, path, **kwargs)
        if owner == "tts.profile_migration_backup":
            real_blobopen = type(connection).blobopen

            def blobopen(parent, *args, **options):
                blob = real_blobopen(parent, *args, **options)
                blobs.append(blob)
                parents.append(parent)
                observed.extend(storage._holds.values())
                return Blob(blob)

            type(connection).blobopen = blobopen
        return connection

    repository.connect_private_sqlite = connect
    repo = repository.TTSProfileRepository(active)
    with pytest.raises(ProfileRepositoryError):
        await repo.open()
    assert blobs and parents

    def prove_closed():
        for blob in blobs:
            with pytest.raises(sqlite3.ProgrammingError):
                len(blob)
        for parent in parents:
            with pytest.raises(sqlite3.ProgrammingError):
                parent.execute("SELECT 1")

    await asyncio.wrap_future(repo._executor.submit(prove_closed))
    assert active.read_bytes() == before
    repo._maintenance_close_admission()
    assert await repo._maintenance_drain(time.monotonic() + 3)
    storage._shutdown()
    assert _probe(observed[0].authority.control_root, observed[0].names) == "entered"


@pytest.mark.parametrize("after", [False, True])
def test_historical_migration_blob_is_retired_by_original_parent(tmp_path, after):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_migration_native_maintenance import _historical_blob_child
asyncio.run(_historical_blob_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(after),
    )


async def _restore_source_allocation_child(root, refused):
    import asyncio
    import time

    from Tests.DB.test_sqlite_source_pin_lifetime import _probe
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS import profile_repository as repository
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    candidate = root / "incoming.sqlite"
    _historical_v3(candidate)
    before = candidate.read_bytes()
    repo = repository.TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    original = repository.connect_private_sqlite
    observed, allocated, pauses, entered = [], [], [], []

    def connect(owner, path, **kwargs):
        if owner == "tts.profile_restore_stage" and not entered:
            entered.append(True)
            observed.extend(storage._holds.values())
            if refused:
                pauses.append(storage._begin_local_pause())
                return original(owner, path, **kwargs)
            allocated.append(original(owner, path, **kwargs))
            raise OSError("native restore reader allocated before wrapper return")
        return original(owner, path, **kwargs)

    repository.connect_private_sqlite = connect
    with pytest.raises(ProfileRepositoryError):
        await repo.restore_from(candidate)
    assert entered and candidate.read_bytes() == before
    repo._maintenance_close_admission()
    drained = await repo._maintenance_drain(time.monotonic() + 3)
    assert drained is refused
    if refused:
        assert not repo._migration_native_operations
        await asyncio.wrap_future(repo._executor.submit(pauses[0].resume))
    else:
        assert allocated and any(
            item.pending
            for operation in repo._migration_native_operations
            for item in operation.source_connections
        )
    storage._shutdown()
    assert _probe(observed[0].authority.control_root, observed[0].names) == (
        "entered" if refused else "blocked"
    )


@pytest.mark.parametrize("refused", [False, True])
def test_restore_source_allocation_distinguishes_original_refusal(tmp_path, refused):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_migration_native_maintenance import _restore_source_allocation_child
asyncio.run(_restore_source_allocation_child(Path(sys.argv[1]), sys.argv[2] == 'True'))
""",
        str(refused),
    )


async def _repeated_revalidation_child(root):
    import os
    import time

    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.TTS.profile_repository import TTSProfileRepository

    repo = TTSProfileRepository(root / "profiles.sqlite")
    await repo.open()
    owner = repo._connection
    retained = set(owner.native_descriptors)
    original = owner._observe_revalidation_parent_descriptor
    observed = []

    def opening(*args, **kwargs):
        fd = original(*args, **kwargs)
        observed.append(fd)
        return fd

    owner._observe_revalidation_parent_descriptor = opening
    for _ in range(10):
        await repo.list_profiles()
        assert set(owner.native_descriptors) == retained
        assert not owner.pending and not owner.attempted_descriptors
    assert len(observed) > len(set(observed)), (
        "actual OS descriptor reuse not exercised"
    )
    for fd in set(observed):
        with pytest.raises(OSError):
            os.fstat(fd)
    repo._maintenance_close_admission()
    pause = storage._begin_local_pause()
    try:
        assert await repo._maintenance_drain(time.monotonic() + 3)
    finally:
        pause.resume()
    await repo._maintenance_resume()
    await repo.list_profiles()
    await repo.close()


def test_repeated_current_revalidation_reuses_fds_and_resumes_source(tmp_path):
    _run_private_child(
        tmp_path,
        """
import asyncio, sys
from pathlib import Path
from Tests.TTS.test_profile_migration_native_maintenance import _repeated_revalidation_child
asyncio.run(_repeated_revalidation_child(Path(sys.argv[1])))
""",
    )


def test_recovery_leaf_rejects_foreign_parent_operation_before_open(tmp_path):
    _run_private_child(
        tmp_path,
        """
import os, sys
from pathlib import Path
import pytest
from tldw_chatbook.TTS.profile_migration_native import _migration_native
from tldw_chatbook.TTS.profile_migration_recovery import _open_leaf
root = Path(sys.argv[1])
active = root / 'active.sqlite'
foreign = root / 'foreign'
foreign.mkdir(mode=0o700)
other = foreign / active.name
other.write_bytes(b'foreign exact bytes')
other.chmod(0o600)
parent = os.open(foreign, os.O_RDONLY | os.O_DIRECTORY)
try:
    with _migration_native((active,)) as native:
        count = len(native.leases)
        with pytest.raises(ValueError, match='migration_native_parent_mismatch'):
            _open_leaf(parent, other.name, _native=native)
        assert len(native.leases) == count and not native.descriptors
    assert other.read_bytes() == b'foreign exact bytes'
finally:
    os.close(parent)
""",
    )
