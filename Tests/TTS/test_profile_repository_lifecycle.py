"""Lifecycle tests for the serialized TTS profile repository."""

from __future__ import annotations

import asyncio
import gc
import importlib
import os
import sqlite3
import threading
import traceback
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, cast
from uuid import UUID

import pytest

import tldw_chatbook.DB.private_sqlite as private_sqlite
from tldw_chatbook.TTS.migrations.v0_to_v1 import migrate as _raw_migrate_v0_to_v1
from tldw_chatbook.TTS.migrations.v1_to_v2 import migrate as _raw_migrate_v1_to_v2
from tldw_chatbook.TTS.migrations.v2_to_v3 import migrate as _raw_migrate_v2_to_v3
from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError
from tldw_chatbook.TTS.profile_migration_candidate import (
    ProfileMigrationBoundary,
    step_profile_migration_candidate,
)
from tldw_chatbook.TTS.profile_schema import (
    CURRENT_PROFILE_SCHEMA_VERSION,
    encode_profile,
    open_profile_store,
    validate_profile_candidate,
)
from tldw_chatbook.TTS.profile_store_lock import (
    ProfileStoreLease,
    ProfileStoreLockMode,
)
from tldw_chatbook.TTS.profile_types import (
    ProfileBackupReceipt,
    ProfileRepositoryState,
    ProfileRestoreReceipt,
    ProfileStoreResult,
    TTSGenerationProfile,
    TTSProfileDraft,
)


class _ControlFlow(BaseException):
    """A test-only control-flow signal."""


class _RecordingConnection:
    def __init__(
        self,
        events: list[tuple[str, int]],
        *,
        close_error: BaseException | None = None,
    ) -> None:
        self.events = events
        self.close_error = close_error
        self.closed = False

    def execute(self, _sql: str) -> tuple[object, ...]:
        return ()

    def close(self) -> None:
        self.events.append(("connection.close", threading.get_ident()))
        if self.close_error is not None:
            raise self.close_error
        self.closed = True


class _CloseFailingSQLiteProxy:
    def __init__(self, connection: sqlite3.Connection, secret: str) -> None:
        self.connection = connection
        self.secret = secret
        self.fail_close = True
        self.close_calls = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.connection, name)

    @property
    def row_factory(self) -> object:
        return self.connection.row_factory

    @row_factory.setter
    def row_factory(self, value: object) -> None:
        self.connection.row_factory = cast(Any, value)

    def close(self) -> None:
        self.close_calls += 1
        if self.fail_close:
            raise RuntimeError(self.secret)
        self.connection.close()


class _RecordingLease:
    def __init__(
        self,
        events: list[tuple[str, int]],
        *,
        acquire_error: BaseException | None = None,
        release_error: BaseException | None = None,
    ) -> None:
        self.events = events
        self.acquire_error = acquire_error
        self.release_error = release_error
        self.acquired = False

    def acquire(self) -> _RecordingLease:
        self.events.append(("lease.acquire", threading.get_ident()))
        if self.acquire_error is not None:
            raise self.acquire_error
        self.acquired = True
        return self

    def release(self) -> None:
        self.events.append(("lease.release", threading.get_ident()))
        self.acquired = False
        if self.release_error is not None:
            raise self.release_error


class _SequencedCloseConnection:
    def __init__(
        self,
        events: list[tuple[str, int]],
        label: str,
        close_errors: list[BaseException | None],
        *,
        before_close: Callable[[], None] | None = None,
    ) -> None:
        self.events = events
        self.label = label
        self.close_errors = close_errors
        self.before_close = before_close
        self.close_calls = 0
        self.closed = False

    def execute(self, _sql: str) -> tuple[object, ...]:
        return ()

    def close(self) -> None:
        if self.before_close is not None:
            self.before_close()
        self.events.append((f"{self.label}.close", threading.get_ident()))
        self.close_calls += 1
        error = self.close_errors.pop(0) if self.close_errors else None
        if error is not None:
            raise error
        self.closed = True


class _SequencedReleaseLease:
    def __init__(
        self,
        events: list[tuple[str, int]],
        label: str,
        release_errors: list[BaseException | None],
    ) -> None:
        self.events = events
        self.label = label
        self.release_errors = release_errors
        self.acquire_calls = 0
        self.release_calls = 0
        self.acquired = False

    def acquire(self) -> _SequencedReleaseLease:
        self.events.append((f"{self.label}.acquire", threading.get_ident()))
        self.acquire_calls += 1
        self.acquired = True
        return self

    def release(self) -> None:
        self.events.append((f"{self.label}.release", threading.get_ident()))
        self.release_calls += 1
        error = self.release_errors.pop(0) if self.release_errors else None
        if error is not None:
            raise error
        self.acquired = False


class _ObservedProfileStoreLease(ProfileStoreLease):
    def __init__(
        self,
        database_path: Path,
        mode: ProfileStoreLockMode,
        events: list[tuple[str, int]],
        label: str,
    ) -> None:
        super().__init__(database_path, mode)
        self.events = events
        self.label = label

    def acquire(self) -> ProfileStoreLease:
        self.events.append((f"{self.label}.acquire", threading.get_ident()))
        return super().acquire()

    def release(self) -> None:
        self.events.append((f"{self.label}.release", threading.get_ident()))
        super().release()


class _RecordingExecutor:
    def __init__(
        self,
        max_workers: int,
        events: list[tuple[str, int]],
    ) -> None:
        self.events = events
        self.max_workers = max_workers
        self.shutdown_calls = 0
        self._delegate = RealThreadPoolExecutor(max_workers=max_workers)
        self.events.append(("executor.construct", threading.get_ident()))

    def submit(
        self,
        function: Any,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Future[Any]:
        return self._delegate.submit(function, *args, **kwargs)

    def shutdown(
        self,
        wait: bool = True,
        *,
        cancel_futures: bool = False,
    ) -> None:
        self.shutdown_calls += 1
        self.events.append(("executor.shutdown", threading.get_ident()))
        self._delegate.shutdown(wait=wait, cancel_futures=cancel_futures)


def _repository_module() -> ModuleType:
    try:
        return importlib.import_module("tldw_chatbook.TTS.profile_repository")
    except ModuleNotFoundError:
        pytest.fail("TTS profile repository lifecycle is not implemented")
    raise AssertionError("unreachable")


def _repository(database_path: Path) -> Any:
    return _repository_module().TTSProfileRepository(database_path)


def _draft(name: str) -> TTSProfileDraft:
    return TTSProfileDraft(
        display_name=name,
        provider_id="audio_cpp",
        model_id="supertonic",
        voice_id=None,
        response_format="wav",
        speed=1.0,
        options={},
    )


async def _create_profile_store(path: Path, *names: str) -> None:
    repository = _repository(path)
    await repository.open()
    try:
        for index, name in enumerate(names, start=1):
            await repository.create_profile(
                _draft(name),
                UUID(f"00000000-0000-4000-8000-{index:012d}"),
            )
    finally:
        await repository.close()


def _build_populated_v1_store_at(path: Path) -> None:
    """Build an honest, populated v1 store the repository has never opened.

    Runs the module's own v0->v1 migration directly on a raw connection --
    never by monkeypatching ``CURRENT_PROFILE_SCHEMA_VERSION`` back to 1 --
    then inserts one profile row through the real ``encode_profile`` codec,
    so the resulting file is exactly what a pre-slice user's store would be.
    """

    connection = sqlite3.connect(path)
    try:
        _raw_migrate_v0_to_v1(connection)
        now = datetime(2026, 7, 26, 12, 34, 56, 123456, tzinfo=UTC)
        profile = TTSGenerationProfile(
            profile_id=UUID("00000000-0000-4000-8000-000000000099"),
            display_name="Legacy",
            normalized_name="legacy",
            provider_id="audio_cpp",
            model_id="supertonic",
            voice_id=None,
            response_format="wav",
            speed=1.0,
            options={},
            revision=1,
            created_at=now,
            updated_at=now,
        )
        connection.execute(
            """
            INSERT INTO tts_generation_profiles (
                profile_id, display_name, normalized_name, provider_id, model_id,
                voice_id, response_format, speed, options_json, revision,
                created_at, updated_at
            ) VALUES (
                :profile_id, :display_name, :normalized_name, :provider_id,
                :model_id, :voice_id, :response_format, :speed, :options_json,
                :revision, :created_at, :updated_at
            )
            """,
            encode_profile(profile),
        )
        connection.commit()
    finally:
        connection.close()


def _build_populated_v2_store_at(path: Path, *, display_name: str = "Legacy") -> None:
    _build_populated_v1_store_at(path)
    connection = sqlite3.connect(path)
    try:
        _raw_migrate_v1_to_v2(connection)
        connection.execute(
            "UPDATE tts_generation_profiles SET display_name = ?, normalized_name = ?",
            (display_name, display_name.casefold()),
        )
        connection.commit()
    finally:
        connection.close()


def _build_populated_v3_store_at(path: Path, *, display_name: str = "Legacy") -> None:
    _build_populated_v2_store_at(path, display_name=display_name)
    connection = sqlite3.connect(path)
    try:
        _raw_migrate_v2_to_v3(connection)
        connection.commit()
    finally:
        connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_version", [1, 2, 3])
async def test_repository_open_recovers_before_access_and_publishes_exact_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_version: int,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    if source_version == 1:
        _build_populated_v1_store_at(database_path)
    elif source_version == 2:
        _build_populated_v2_store_at(database_path)
    else:
        _build_populated_v3_store_at(database_path)
    events: list[str] = []
    real_recover = module.recover_profile_migration_publication
    real_connect = module.connect_private_sqlite
    real_open = module.open_profile_store
    real_schema_connect = module._profile_schema.connect_private_sqlite

    def tracked_recover(path: Path) -> bool:
        events.append("recover")
        return real_recover(path)

    def tracked_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        events.append("sqlite")
        return real_connect(*args, **kwargs)

    def tracked_open(*args: object, **kwargs: object) -> sqlite3.Connection:
        events.append("open")
        return real_open(*args, **kwargs)

    def tracked_schema_connect(
        *args: object,
        **kwargs: object,
    ) -> sqlite3.Connection:
        events.append("schema_sqlite")
        return real_schema_connect(*args, **kwargs)

    monkeypatch.setattr(
        module, "recover_profile_migration_publication", tracked_recover
    )
    monkeypatch.setattr(module, "connect_private_sqlite", tracked_connect)
    monkeypatch.setattr(module, "open_profile_store", tracked_open)
    monkeypatch.setattr(
        module._profile_schema, "connect_private_sqlite", tracked_schema_connect
    )

    repository = module.TTSProfileRepository(database_path)
    await repository.open()
    await repository.close()

    assert "recover" in events
    with sqlite3.connect(database_path) as active:
        assert active.execute("PRAGMA user_version").fetchone() == (4,)
    if source_version <= 2:
        with sqlite3.connect(
            database_path.with_name("profiles.sqlite3.pre-v3.sqlite3")
        ) as pre_v3:
            assert pre_v3.execute("PRAGMA user_version").fetchone() == (2,)
            assert (
                _stored_profile_name(
                    database_path.with_name("profiles.sqlite3.pre-v3.sqlite3")
                )
                == "Legacy"
            )
    with sqlite3.connect(
        database_path.with_name("profiles.sqlite3.pre-v4.sqlite3")
    ) as pre_v4:
        assert pre_v4.execute("PRAGMA user_version").fetchone() == (3,)


@pytest.mark.asyncio
async def test_repository_open_never_migrates_the_active_store_in_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    _build_populated_v3_store_at(database_path)
    active_before = database_path.read_bytes()
    real_open = module.open_profile_store

    def guarded_open(path: Path, **kwargs: object) -> sqlite3.Connection:
        if path == database_path and not kwargs.get("must_exist"):
            assert database_path.read_bytes() == active_before
            raise AssertionError("active store must not be opened for migration")
        return real_open(path, **kwargs)

    monkeypatch.setattr(module, "open_profile_store", guarded_open)
    repository = module.TTSProfileRepository(database_path)
    await repository.open()
    await repository.close()

    with sqlite3.connect(database_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (4,)


@pytest.mark.asyncio
async def test_restore_v2_uses_transactional_publication_with_exact_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    candidate_path = tmp_path / "incoming.sqlite3"
    await _create_profile_store(database_path, "Current")
    _build_populated_v2_store_at(candidate_path, display_name="Restored")
    publications = 0
    real_publish = module.publish_profile_migration

    def tracked_publish(**kwargs: object) -> None:
        nonlocal publications
        publications += 1
        real_publish(**kwargs)

    monkeypatch.setattr(module, "publish_profile_migration", tracked_publish)
    repository = module.TTSProfileRepository(database_path)
    await repository.open()
    try:
        await repository.restore_from(candidate_path)
    finally:
        await repository.close()

    assert publications == 1
    assert _stored_profile_name(database_path) == "Restored"
    pre_v3 = database_path.with_name("profiles.sqlite3.pre-v3.sqlite3")
    pre_v4 = database_path.with_name("profiles.sqlite3.pre-v4.sqlite3")
    assert _stored_profile_name(pre_v3) == "Restored"
    assert _stored_profile_name(pre_v4) == "Restored"
    with sqlite3.connect(pre_v3) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (2,)
    with sqlite3.connect(pre_v4) as connection:
        assert connection.execute("PRAGMA user_version").fetchone() == (3,)


@pytest.mark.asyncio
async def test_three_sequential_older_restores_reuse_fixed_nonzero_tombstones(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    await _create_profile_store(database_path, "Current")
    candidates = tuple(tmp_path / f"incoming-{index}.sqlite3" for index in range(3))
    for index, candidate in enumerate(candidates):
        _build_populated_v2_store_at(candidate, display_name=f"Restored {index}")
    repository = _repository(database_path)
    await repository.open()
    retained_inode_set: set[int] | None = None
    try:
        for index, candidate in enumerate(candidates):
            await repository.restore_from(candidate)
            assert _stored_profile_name(database_path) == f"Restored {index}"
            private_namespace = tuple(
                path
                for path in tmp_path.iterdir()
                if path == database_path
                or path.name.startswith(".profile-migration-")
                or ".pre-v" in path.name
            )
            assert private_namespace
            if index == 1:
                retained_inode_set = {path.stat().st_ino for path in private_namespace}
            elif index == 2:
                assert {
                    path.stat().st_ino for path in private_namespace
                } == retained_inode_set
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_restore_close_restart_restore_reuses_zero_tombstones(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    first = tmp_path / "incoming-first.sqlite3"
    second = tmp_path / "incoming-second.sqlite3"
    third = tmp_path / "incoming-third.sqlite3"
    await _create_profile_store(database_path, "Current")
    _build_populated_v2_store_at(first, display_name="First")
    _build_populated_v2_store_at(second, display_name="Second")
    _build_populated_v2_store_at(third, display_name="Third")

    repository = _repository(database_path)
    await repository.open()
    await repository.restore_from(first)
    await repository.close()
    tombstones = tuple(tmp_path.glob(".profile-migration-*.tombstone"))
    assert tombstones
    assert all(path.stat().st_size == 0 for path in tombstones)
    retained_inode_set: set[int] | None = None
    for index, (candidate, expected_name) in enumerate(
        ((second, "Second"), (third, "Third"))
    ):
        restarted = _repository(database_path)
        await restarted.open()
        try:
            await restarted.restore_from(candidate)
            assert _stored_profile_name(database_path) == expected_name
        finally:
            await restarted.close()
        namespace = tuple(
            path
            for path in tmp_path.iterdir()
            if path == database_path
            or path.name.startswith(".profile-migration-")
            or ".pre-v" in path.name
        )
        if index == 0:
            retained_inode_set = {path.stat().st_ino for path in namespace}
        else:
            assert {path.stat().st_ino for path in namespace} == retained_inode_set
        assert all(
            path.stat().st_size == 0
            for path in tmp_path.glob(".profile-migration-*.tombstone")
        )


@pytest.mark.asyncio
async def test_close_refuses_hardlinked_known_tombstone_and_retries_safely(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    incoming = tmp_path / "incoming.sqlite3"
    await _create_profile_store(database_path, "Current")
    _build_populated_v2_store_at(incoming, display_name="Restored")
    repository = _repository(database_path)
    await repository.open()
    await repository.restore_from(incoming)
    tombstone = tmp_path / ".profile-migration-active-rollback.tombstone"
    alias = tmp_path / "foreign-alias.sqlite3"
    os.link(tombstone, alias)
    private_bytes = alias.read_bytes()

    with pytest.raises(ProfileRepositoryError, match="operation_failed"):
        await repository.close()

    assert alias.read_bytes() == private_bytes
    assert tombstone.read_bytes() == private_bytes
    assert repository._connection is not None
    assert repository._lease is not None and repository._lease.acquired is True
    await _assert_exclusive_lease_blocked(database_path)

    alias.unlink()
    await repository.close()
    assert tombstone.stat().st_size == 0


@pytest.mark.asyncio
async def test_close_refuses_substituted_known_tombstone_and_retries_safely(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    incoming = tmp_path / "incoming.sqlite3"
    await _create_profile_store(database_path, "Current")
    _build_populated_v2_store_at(incoming, display_name="Restored")
    repository = _repository(database_path)
    await repository.open()
    await repository.restore_from(incoming)
    tombstone = tmp_path / ".profile-migration-active-rollback.tombstone"
    retained = tmp_path / "retained-exact-tombstone"
    foreign = b"foreign replacement bytes"
    tombstone.rename(retained)
    tombstone.write_bytes(foreign)

    with pytest.raises(ProfileRepositoryError, match="operation_failed"):
        await repository.close()

    assert tombstone.read_bytes() == foreign
    assert retained.stat().st_size > 0
    assert repository._connection is not None
    assert repository._lease is not None and repository._lease.acquired is True
    await _assert_exclusive_lease_blocked(database_path)

    foreign_leaf = tmp_path / "preserved-foreign-tombstone"
    tombstone.rename(foreign_leaf)
    retained.rename(tombstone)
    await repository.close()
    assert foreign_leaf.read_bytes() == foreign
    assert tombstone.stat().st_size == 0


@pytest.mark.asyncio
async def test_close_settlement_failure_retains_authority_for_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    incoming = tmp_path / "incoming.sqlite3"
    await _create_profile_store(database_path, "Current")
    _build_populated_v2_store_at(incoming, display_name="Restored")
    repository = _repository(database_path)
    await repository.open()
    await repository.restore_from(incoming)
    module = _repository_module()
    original_prepare = module.prepare_reusable_tombstone
    fail_once = True

    def transient_prepare(*args: object, **kwargs: object) -> os.stat_result:
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise OSError("injected settle failure")
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(module, "prepare_reusable_tombstone", transient_prepare)
    with pytest.raises(ProfileRepositoryError, match="operation_failed"):
        await repository.close()

    assert repository._connection is not None
    assert repository._lease is not None and repository._lease.acquired is True
    await _assert_exclusive_lease_blocked(database_path)
    await repository.close()
    assert all(
        path.stat().st_size == 0
        for path in tmp_path.glob(".profile-migration-*.tombstone")
    )


@pytest.mark.asyncio
async def test_restart_rejects_unknown_nonzero_tombstone_without_modifying_it(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    first = tmp_path / "incoming-first.sqlite3"
    second = tmp_path / "incoming-second.sqlite3"
    await _create_profile_store(database_path, "Current")
    _build_populated_v2_store_at(first, display_name="First")
    _build_populated_v2_store_at(second, display_name="Second")
    repository = _repository(database_path)
    await repository.open()
    await repository.restore_from(first)
    await repository.close()
    tombstone = tmp_path / ".profile-migration-active-rollback.tombstone"
    foreign = b"unknown nonzero restart bytes"
    tombstone.write_bytes(foreign)

    restarted = _repository(database_path)
    await restarted.open()
    try:
        with pytest.raises(ProfileRepositoryError, match="restore_failed"):
            await restarted.restore_from(second)
        assert tombstone.read_bytes() == foreign
        assert _stored_profile_name(database_path) == "First"
    finally:
        with pytest.raises(ProfileRepositoryError, match="operation_failed"):
            await restarted.close()
        assert tombstone.read_bytes() == foreign
        tombstone.unlink()
        await restarted.close()


@pytest.mark.asyncio
async def test_nonzero_tombstone_hardlink_blocks_reuse_without_touching_alias(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    first = tmp_path / "incoming-first.sqlite3"
    second = tmp_path / "incoming-second.sqlite3"
    await _create_profile_store(database_path, "Current")
    _build_populated_v2_store_at(first, display_name="First")
    _build_populated_v2_store_at(second, display_name="Second")
    repository = _repository(database_path)
    await repository.open()
    try:
        await repository.restore_from(first)
        tombstone = tmp_path / ".profile-migration-active-rollback.tombstone"
        alias = tmp_path / "foreign-alias.sqlite3"
        os.link(tombstone, alias)
        retained = alias.read_bytes()

        with pytest.raises(ProfileRepositoryError, match="restore_failed"):
            await repository.restore_from(second)

        assert alias.read_bytes() == retained
        assert tombstone.read_bytes() == retained
        assert _stored_profile_name(database_path) == "First"
    finally:
        alias = tmp_path / "foreign-alias.sqlite3"
        alias.unlink(missing_ok=True)
        await repository.close()


@pytest.mark.asyncio
async def test_prepublication_migration_failure_retains_bounded_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    _build_populated_v3_store_at(database_path)
    active_before = database_path.read_bytes()
    real_step = module.step_profile_migration_candidate

    def fail_step(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("private migration failure")

    monkeypatch.setattr(module, "step_profile_migration_candidate", fail_step)
    repository = module.TTSProfileRepository(database_path)
    with pytest.raises(ProfileRepositoryError, match="migration_failed"):
        await repository.open()
    assert database_path.read_bytes() == active_before
    assert not tuple(tmp_path.glob(".profile-migration-*.candidate.sqlite3"))
    tombstones = tuple(tmp_path.glob(".profile-migration-*.tombstone"))
    assert tombstones
    assert all(path.read_bytes() for path in tombstones)
    active_tombstone = tmp_path / ".profile-migration-active-candidate.tombstone"
    retained_inode = active_tombstone.stat().st_ino

    monkeypatch.setattr(module, "step_profile_migration_candidate", real_step)
    with pytest.raises(ProfileRepositoryError, match="migration_failed"):
        await repository.open()
    await repository.close()

    assert active_tombstone.stat().st_ino == retained_inode


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cancel_stage", "published_name", "recovery_count"),
    [
        ("preflight", "Original", 0),
        ("ponr", "Candidate", 1),
    ],
)
async def test_restore_control_flow_settles_publication_before_redelivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_stage: str,
    published_name: str,
    recovery_count: int,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    candidate_path = tmp_path / "incoming.sqlite3"
    await _create_profile_store(database_path, "Original")
    await _create_profile_store(candidate_path, "Candidate")
    real_publish = module.publish_profile_migration
    cancellation = _ControlFlow()

    def cancel_publication(**kwargs: object) -> None:
        repository_hook = kwargs.pop("stage_hook", None)

        def stage_hook(stage: object) -> None:
            if cancel_stage == "preflight" and stage.value == cancel_stage:
                raise cancellation
            if repository_hook is not None:
                repository_hook(stage)
            if cancel_stage == "ponr" and stage.value == cancel_stage:
                raise cancellation

        real_publish(**kwargs, stage_hook=stage_hook)

    monkeypatch.setattr(module, "publish_profile_migration", cancel_publication)
    repository = module.TTSProfileRepository(database_path)
    await repository.open()
    try:
        with pytest.raises(_ControlFlow) as caught:
            await repository.restore_from(candidate_path)

        assert caught.value is cancellation
        assert repository.state is ProfileRepositoryState.OPEN
        assert _stored_profile_name(database_path) == published_name
        assert (
            len(tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3")))
            == recovery_count
        )
        assert not tuple(tmp_path.glob("*.migration-publication.json"))
        assert not tuple(tmp_path.glob("*.rollback.sqlite3"))
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_restore_post_ponr_failure_restores_active_and_retains_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    candidate_path = tmp_path / "incoming.sqlite3"
    await _create_profile_store(database_path, "Original")
    await _create_profile_store(candidate_path, "Candidate")
    real_publish = module.publish_profile_migration

    def fail_after_active_replace(**kwargs: object) -> None:
        repository_hook = kwargs.pop("stage_hook", None)

        def stage_hook(stage: object) -> None:
            if repository_hook is not None:
                repository_hook(stage)
            if stage.value == "active_replaced":
                raise OSError("publication fault")

        real_publish(**kwargs, stage_hook=stage_hook)

    monkeypatch.setattr(module, "publish_profile_migration", fail_after_active_replace)
    repository = module.TTSProfileRepository(database_path)
    await repository.open()
    try:
        with pytest.raises(ProfileRepositoryError, match="restore_failed"):
            await repository.restore_from(candidate_path)

        assert repository.state is ProfileRepositoryState.OPEN
        assert _stored_profile_name(database_path) == "Original"
        recoveries = tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
        assert len(recoveries) == 1
        validate_profile_candidate(recoveries[0])
        assert not tuple(tmp_path.glob("*.migration-publication.json"))
        assert not tuple(tmp_path.glob("*.rollback.sqlite3"))
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_restore_deadline_during_candidate_copy_preserves_active_without_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    candidate_path = tmp_path / "incoming.sqlite3"
    await _create_profile_store(database_path, "Original")
    await _create_profile_store(candidate_path, "Candidate")
    real_migrate = module.migrate_profile_store_to_candidate
    expired = False

    def clock() -> float:
        return 10.0 if expired else 0.0

    def expire_during_copy(*args: object, **kwargs: object) -> object:
        nonlocal expired
        expired = True
        return real_migrate(*args, **kwargs)

    monkeypatch.setattr(module, "_monotonic", clock)
    monkeypatch.setattr(
        module, "migrate_profile_store_to_candidate", expire_during_copy
    )
    repository = module.TTSProfileRepository(database_path)
    await repository.open()
    try:
        with pytest.raises(ProfileRepositoryError, match="restore_failed"):
            await repository.restore_from(candidate_path, timeout_seconds=5)

        assert repository.state is ProfileRepositoryState.OPEN
        assert _stored_profile_name(database_path) == "Original"
        assert not tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
        assert not tuple(tmp_path.glob("*.migration-publication.json"))
    finally:
        await repository.close()


def test_restore_candidate_uses_path_free_version_stepper_without_touching_active(
    tmp_path: Path,
) -> None:
    active_path = tmp_path / "active.sqlite3"
    restore_path = tmp_path / "restore-candidate.sqlite3"
    active = open_profile_store(active_path)
    active.close()
    active_before = active_path.read_bytes()
    _build_populated_v1_store_at(restore_path)
    restore = sqlite3.connect(restore_path)
    restore.row_factory = sqlite3.Row
    observed: list[ProfileMigrationBoundary] = []

    def consume_boundary(snapshot, request) -> None:
        observed.append(request.kind)
        destination = private_sqlite.open_profile_migration_boundary_destination(
            tmp_path / f"restore-{request.kind.value}.sqlite3",
            schema_version=request.schema_version,
        )
        snapshot.backup_to(destination)

    result = step_profile_migration_candidate(
        restore,
        boundary_sink=consume_boundary,
    )

    assert result.source_version == 1
    assert observed == [
        ProfileMigrationBoundary.PRE_V3,
        ProfileMigrationBoundary.PRE_V4,
    ]
    assert active_path.read_bytes() == active_before
    assert not tuple(tmp_path.glob("active.sqlite3-*"))
    migrated = sqlite3.connect(restore_path)
    try:
        assert migrated.execute("PRAGMA user_version").fetchone()[0] == 4
    finally:
        migrated.close()


def _test_online_backup(source_path: Path, destination_path: Path) -> None:
    source = sqlite3.connect(source_path)
    destination = sqlite3.connect(destination_path)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    os.chmod(destination_path, 0o600)


def _stored_profile_name(path: Path) -> str:
    connection = sqlite3.connect(path)
    try:
        return cast(
            str,
            connection.execute(
                "SELECT display_name FROM tts_generation_profiles"
            ).fetchone()[0],
        )
    finally:
        connection.close()


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_exact_version_under_exclusive_publishes_v2_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    _build_populated_v2_store_at(database_path)
    backup_path = module._v2_migration_backup_path(database_path)
    modes: list[ProfileStoreLockMode] = []
    real_lease = module.ProfileStoreLease

    def tracked_lease(
        path: Path, mode: ProfileStoreLockMode, **kwargs: object
    ) -> ProfileStoreLease:
        modes.append(mode)
        return real_lease(path, mode, **kwargs)

    monkeypatch.setattr(module, "ProfileStoreLease", tracked_lease)

    repository = module.TTSProfileRepository(database_path)
    await repository.open()
    await repository.close()

    assert modes == [
        ProfileStoreLockMode.SHARED,
        ProfileStoreLockMode.EXCLUSIVE,
        ProfileStoreLockMode.SHARED,
    ]
    assert backup_path.is_file()
    backup = sqlite3.connect(backup_path)
    try:
        assert backup.execute("PRAGMA user_version").fetchone()[0] == 2
    finally:
        backup.close()


@pytest.mark.asyncio
async def test_v2_migration_reuses_equivalent_retained_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    _build_populated_v2_store_at(database_path)
    backup_path = module._v2_migration_backup_path(database_path)
    _test_online_backup(database_path, backup_path)

    def unexpected_backup(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("equivalent retained backup must be reused")

    monkeypatch.setattr(module, "backup_open_connections_to_private", unexpected_backup)

    repository = module.TTSProfileRepository(database_path)
    await repository.open()
    await repository.close()

    assert _stored_profile_name(backup_path) == "Legacy"


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_v2_migration_source_close_failure_blocks_migration_context_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    _build_populated_v2_store_at(database_path)
    real_connect = module.connect_private_sqlite
    source_proxy: _CloseFailingSQLiteProxy | None = None
    matching_connects = 0

    def close_failing_connect(
        owner_id: str, database: object, **kwargs: object
    ) -> sqlite3.Connection:
        nonlocal matching_connects, source_proxy
        connection = real_connect(owner_id, database, **kwargs)
        if (
            owner_id == "tts.profile_migration_backup"
            and Path(cast(str | os.PathLike[str], database)) == database_path
            and kwargs.get("read_only") is True
        ):
            matching_connects += 1
            if matching_connects == 2:
                source_proxy = _CloseFailingSQLiteProxy(
                    connection,
                    "PRIVATE source close detail",
                )
                return cast(sqlite3.Connection, source_proxy)
        return connection

    monkeypatch.setattr(module, "connect_private_sqlite", close_failing_connect)
    repository = module.TTSProfileRepository(database_path)

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.open()

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert caught.value.code == "migration_failed"
    assert "PRIVATE" not in repr(caught.value)
    source = sqlite3.connect(database_path)
    try:
        assert source.execute("PRAGMA user_version").fetchone()[0] == 2
    finally:
        source.close()
    assert source_proxy is not None
    assert repository._connection is source_proxy
    assert repository._lease is not None
    assert repository._lease.mode is ProfileStoreLockMode.EXCLUSIVE
    source_proxy.fail_close = False
    await repository.close()


@pytest.mark.asyncio
async def test_post_copy_source_close_failure_retains_handle_and_exclusive_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    _build_populated_v3_store_at(database_path)
    active_before = database_path.read_bytes()
    real_connect = module.connect_private_sqlite
    matching_connects = 0
    source_proxy: _CloseFailingSQLiteProxy | None = None

    def close_failing_second_connect(
        owner_id: str,
        database: object,
        **kwargs: object,
    ) -> sqlite3.Connection:
        nonlocal matching_connects, source_proxy
        connection = real_connect(owner_id, database, **kwargs)
        if (
            owner_id == "tts.profile_migration_backup"
            and Path(cast(str | os.PathLike[str], database)) == database_path
            and kwargs.get("read_only") is True
        ):
            matching_connects += 1
            if matching_connects == 2:
                source_proxy = _CloseFailingSQLiteProxy(
                    connection,
                    "PRIVATE post-copy close detail",
                )
                return cast(sqlite3.Connection, source_proxy)
        return connection

    monkeypatch.setattr(module, "connect_private_sqlite", close_failing_second_connect)
    repository = module.TTSProfileRepository(database_path)
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.open()

    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert source_proxy is not None and source_proxy.close_calls == 2
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository._connection is source_proxy
    assert repository._lease is not None
    assert repository._lease.mode is ProfileStoreLockMode.EXCLUSIVE
    assert repository._lease.acquired is True
    assert database_path.read_bytes() == active_before
    await _assert_exclusive_lease_blocked(database_path)

    source_proxy.fail_close = False
    await repository.close()
    assert await asyncio.to_thread(_try_exclusive_lease, database_path) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("restore_stage_open", (1, 2, 3))
async def test_restore_stage_close_failure_retains_source_and_exclusive_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    restore_stage_open: int,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    real_connect = module.connect_private_sqlite
    matching_connects = 0
    source_proxy: _CloseFailingSQLiteProxy | None = None

    def close_failing_restore_source(
        owner_id: str,
        database: object,
        **kwargs: object,
    ) -> sqlite3.Connection:
        nonlocal matching_connects, source_proxy
        connection = real_connect(owner_id, database, **kwargs)
        if (
            owner_id == "tts.profile_restore_stage"
            and Path(cast(str | os.PathLike[str], database)) == candidate
            and kwargs.get("read_only") is True
        ):
            matching_connects += 1
            if matching_connects == restore_stage_open:
                source_proxy = _CloseFailingSQLiteProxy(
                    connection,
                    f"PRIVATE restore source {restore_stage_open}",
                )
                return cast(sqlite3.Connection, source_proxy)
        return connection

    monkeypatch.setattr(module, "connect_private_sqlite", close_failing_restore_source)
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.restore_from(candidate)

    _assert_safe_error(caught.value, "restore_failed", str(candidate))
    assert source_proxy is not None
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository._connection is source_proxy
    assert repository._lease is not None
    assert repository._lease.mode is ProfileStoreLockMode.EXCLUSIVE
    assert repository._lease.acquired is True
    await _assert_exclusive_lease_blocked(database_path)

    source_proxy.fail_close = False
    await repository.close()
    assert await asyncio.to_thread(_try_exclusive_lease, database_path) is None


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_populated_v1_store_upgrades_under_exclusive_lease_through_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-slice v1 store must upgrade via the exclusive-guarded path.

    ``open_profile_store`` transparently upgrades a populated v1 store in
    place, but that write must never happen while only the repository's
    ordinary shared lease is held (:meth:`_worker_open_if_proven_current`'s
    contract). Opening through the real lease-guarded repository layer --
    not the schema module's unit-test seam -- must route the upgrade
    through :meth:`_worker_initialize_store`'s exclusive lease first, and
    only then hand back a shared-leased connection for ordinary use.
    """

    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    _build_populated_v1_store_at(database_path)

    recorded_modes: list[ProfileStoreLockMode] = []
    real_lease_cls = module.ProfileStoreLease

    def recording_lease_factory(
        lease_path: Path,
        mode: ProfileStoreLockMode,
        **kwargs: object,
    ) -> ProfileStoreLease:
        recorded_modes.append(mode)
        return real_lease_cls(lease_path, mode, **kwargs)

    monkeypatch.setattr(module, "ProfileStoreLease", recording_lease_factory)

    repository = module.TTSProfileRepository(database_path)
    opened = await repository.open()
    try:
        assert opened == ProfileStoreResult(generation=1, value=None)
        page = await repository.list_profiles()
        assert [profile.display_name for profile in page.value.profiles] == ["Legacy"]
    finally:
        await repository.close()

    assert recorded_modes == [
        ProfileStoreLockMode.SHARED,
        ProfileStoreLockMode.EXCLUSIVE,
        ProfileStoreLockMode.SHARED,
    ]

    check = sqlite3.connect(database_path)
    try:
        assert (
            check.execute("PRAGMA user_version").fetchone()[0]
            == CURRENT_PROFILE_SCHEMA_VERSION
        )
        assert (
            check.execute("SELECT COUNT(*) FROM tts_generation_profiles").fetchone()[0]
            == 1
        )
    finally:
        check.close()


@pytest.mark.asyncio
async def test_current_store_open_uses_continuous_shared_proof_and_live_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery and schema inspection precede the shared live handle."""

    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    await _create_profile_store(database_path, "Current")

    recorded_modes: list[ProfileStoreLockMode] = []
    real_lease_cls = module.ProfileStoreLease

    def recording_lease_factory(
        lease_path: Path,
        mode: ProfileStoreLockMode,
        **kwargs: object,
    ) -> ProfileStoreLease:
        recorded_modes.append(mode)
        return real_lease_cls(lease_path, mode, **kwargs)

    monkeypatch.setattr(module, "ProfileStoreLease", recording_lease_factory)

    repository = module.TTSProfileRepository(database_path)
    await repository.open()
    try:
        page = await repository.list_profiles()
        assert [profile.display_name for profile in page.value.profiles] == ["Current"]
    finally:
        await repository.close()

    assert recorded_modes == [ProfileStoreLockMode.SHARED]


@pytest.mark.asyncio
async def test_two_current_store_readers_open_concurrently(tmp_path: Path) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    await _create_profile_store(database_path, "Current")
    first = _repository_module().TTSProfileRepository(database_path)
    second = _repository_module().TTSProfileRepository(database_path)

    await first.open()
    try:
        await second.open()
        assert first._lease is not None and second._lease is not None
        assert first._lease.mode is ProfileStoreLockMode.SHARED
        assert second._lease.mode is ProfileStoreLockMode.SHARED
    finally:
        await second.close()
        await first.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_version", (None, 3))
async def test_initialized_store_reopens_with_exact_live_authority(
    tmp_path: Path,
    initial_version: int | None,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    if initial_version == 3:
        _build_populated_v3_store_at(database_path)
    repository = _repository(database_path)

    await repository.open()
    try:
        authority = cast(Any, repository._connection)
        assert authority.selected == database_path.resolve(strict=False)
        assert authority.file_fd >= 0
        assert set(authority.sidecar_fds) == {"-wal", "-shm"}
        assert repository._lease is not None
        assert repository._lease.mode is ProfileStoreLockMode.SHARED
    finally:
        await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_version", (None, 3))
async def test_post_initialization_handoff_rejects_distinct_valid_v4_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    initial_version: int | None,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    foreign_path = tmp_path / "foreign.sqlite3"
    detached_path = tmp_path / "detached.sqlite3"
    await _create_profile_store(foreign_path, "Foreign")
    foreign_bytes = foreign_path.read_bytes()
    if initial_version == 3:
        _build_populated_v3_store_at(database_path, display_name="Original")
    real_initialize = module.TTSProfileRepository._worker_initialize_store

    def initialize_then_swap(
        self: Any,
        active_path: Path,
        *,
        allow_create: bool = True,
    ) -> object:
        authority = real_initialize(
            self,
            active_path,
            allow_create=allow_create,
        )
        os.replace(active_path, detached_path)
        os.replace(foreign_path, active_path)
        return authority

    monkeypatch.setattr(
        module.TTSProfileRepository,
        "_worker_initialize_store",
        initialize_then_swap,
    )
    repository = module.TTSProfileRepository(database_path)

    with pytest.raises(ProfileRepositoryError, match="operation_failed"):
        await repository.open()

    assert database_path.read_bytes() == foreign_bytes
    assert _stored_profile_name(database_path) == "Foreign"
    assert detached_path.exists()
    await repository.close()


@pytest.mark.asyncio
async def test_normal_exact_open_never_serializes_or_selects_reference_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    await _create_profile_store(database_path, "Current")
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            """
            INSERT INTO tts_profile_clone_references (
                profile_id, reference_id, wav_bytes, reference_text, sha256,
                byte_length, duration_ms, sample_rate_hz, channels,
                sample_encoding, created_at, updated_at, recipe_id,
                recipe_revision
            ) VALUES (?, ?, zeroblob(?), ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
            """,
            (
                "00000000-0000-4000-8000-000000000001",
                "00000000-0000-4000-8000-000000000099",
                8 * 1024 * 1024,
                "large private reference",
                "0" * 64,
                8 * 1024 * 1024,
                1000,
                24000,
                1,
                "pcm_s16le",
                "2026-01-01T00:00:00.000000+00:00",
                "2026-01-01T00:00:00.000000+00:00",
            ),
        )
        connection.commit()
    finally:
        connection.close()

    real_connect = module._profile_schema.connect_private_sqlite

    class NoSerializeOrBlobProxy:
        def __init__(self, owned: sqlite3.Connection) -> None:
            self.connection = owned

        def __getattr__(self, name: str) -> Any:
            if name == "serialize":
                raise AssertionError("normal exact open must not serialize")
            return getattr(self.connection, name)

        @property
        def row_factory(self) -> object:
            return self.connection.row_factory

        @row_factory.setter
        def row_factory(self, value: object) -> None:
            self.connection.row_factory = cast(Any, value)

        def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
            normalized = " ".join(sql.lower().split())
            if "select" in normalized and "wav_bytes" in normalized:
                assert "length(wav_bytes)" in normalized
            return self.connection.execute(sql, *args)

    def proxied_connect(
        owner_id: str,
        database: object,
        **kwargs: object,
    ) -> sqlite3.Connection:
        opened = real_connect(owner_id, database, **kwargs)
        if owner_id == "tts.profile_store":
            return cast(sqlite3.Connection, NoSerializeOrBlobProxy(opened))
        return opened

    monkeypatch.setattr(
        module._profile_schema, "connect_private_sqlite", proxied_connect
    )
    repository = module.TTSProfileRepository(database_path)
    await repository.open()
    await repository.close()


def test_exact_open_metadata_evidence_is_incremental_and_bounded(
    tmp_path: Path,
) -> None:
    module = _repository_module()._profile_schema
    database_path = tmp_path / "profiles.sqlite3"
    connection = module.open_profile_store(database_path)
    try:
        created = "2026-01-01T00:00:00.000000Z"
        connection.executemany(
            """
            INSERT INTO tts_generation_profiles (
                profile_id, display_name, normalized_name, provider_id,
                model_id, voice_id, response_format, speed, options_json,
                revision, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    f"00000000-0000-4000-8000-{index:012d}",
                    f"Profile {index}",
                    f"profile {index}",
                    "audiocpp",
                    "model",
                    "voice",
                    "wav",
                    1.0,
                    "{}",
                    1,
                    created,
                    created,
                )
                for index in range(2_000)
            ),
        )
        connection.commit()

        class BoundedCursor:
            def __init__(self, cursor: sqlite3.Cursor) -> None:
                self.cursor = cursor

            def __iter__(self) -> object:
                return iter(self.cursor)

            def fetchone(self) -> object:
                return self.cursor.fetchone()

            def fetchall(self) -> object:
                raise AssertionError("metadata proof must not fetchall")

        class BoundedConnection:
            def execute(self, sql: str, *args: object) -> BoundedCursor:
                normalized = " ".join(sql.lower().split())
                if "select" in normalized and "wav_bytes" in normalized:
                    assert "length(wav_bytes)" in normalized
                return BoundedCursor(connection.execute(sql, *args))

            def serialize(self) -> bytes:
                raise AssertionError("metadata proof must not serialize")

        digest, counts = module._stream_exact_store_metadata_evidence(
            cast(sqlite3.Connection, BoundedConnection())
        )
    finally:
        connection.close()

    assert type(digest) is bytes and len(digest) == 32
    assert counts == (2_000, 0, 0)


@pytest.mark.asyncio
async def test_current_shared_proof_has_no_publisher_gap_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    await _create_profile_store(database_path, "Current")
    real_open = module.open_exact_current_profile_store
    contender_errors: list[ProfileRepositoryError | None] = []

    def observed_open(*args: object, **kwargs: object) -> sqlite3.Connection:
        contender_errors.append(_try_exclusive_lease(database_path))
        return real_open(*args, **kwargs)

    monkeypatch.setattr(module, "open_exact_current_profile_store", observed_open)
    repository = module.TTSProfileRepository(database_path)
    await repository.open()
    try:
        assert len(contender_errors) == 1
        assert contender_errors[0] is not None
        assert contender_errors[0].code == "lock_timeout"
    finally:
        await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement_version", (3, 4))
async def test_current_shared_proof_rejects_active_substitution_during_exact_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_version: int,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    retained_original = tmp_path / "retained-original.sqlite3"
    replacement = tmp_path / "replacement.sqlite3"
    await _create_profile_store(database_path, "Original")
    if replacement_version == 3:
        _build_populated_v3_store_at(replacement)
    else:
        await _create_profile_store(replacement, "Foreign current")
    original_before = database_path.read_bytes()
    replacement_before = replacement.read_bytes()
    real_connect = module._profile_schema.connect_private_sqlite
    swapped = False

    def swap_before_path_open(
        owner_id: str,
        database: object,
        **kwargs: object,
    ) -> sqlite3.Connection:
        nonlocal swapped
        if owner_id == "tts.profile_store" and not swapped:
            os.replace(database_path, retained_original)
            os.replace(replacement, database_path)
            swapped = True
        return real_connect(owner_id, database, **kwargs)

    monkeypatch.setattr(
        module._profile_schema,
        "connect_private_sqlite",
        swap_before_path_open,
    )
    repository = module.TTSProfileRepository(database_path)
    try:
        with pytest.raises(ProfileRepositoryError):
            await repository.open()

        assert swapped is True
        assert repository.state is ProfileRepositoryState.UNAVAILABLE
        assert retained_original.read_bytes() == original_before
        assert database_path.read_bytes() == replacement_before
        check = sqlite3.connect(database_path)
        try:
            assert (
                check.execute("PRAGMA user_version").fetchone()[0]
                == replacement_version
            )
        finally:
            check.close()
        assert not tuple(tmp_path.glob("*.pre-v3.sqlite3"))
        assert not tuple(tmp_path.glob("*.pre-v4.sqlite3"))
    finally:
        await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("journal_suffix", ("", "-wal", "-shm", "-journal"))
async def test_current_shared_proof_rejects_publication_artifact_inserted_during_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    journal_suffix: str,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    await _create_profile_store(database_path, "Current")
    active_before = database_path.read_bytes()
    inserted = tmp_path / (
        f".profiles.sqlite3.migration-publication.json{journal_suffix}"
    )
    evidence = b"foreign publication evidence"
    real_connect = module._profile_schema.connect_private_sqlite
    inserted_once = False

    def insert_before_path_open(
        owner_id: str,
        database: object,
        **kwargs: object,
    ) -> sqlite3.Connection:
        nonlocal inserted_once
        if owner_id == "tts.profile_store" and not inserted_once:
            inserted.write_bytes(evidence)
            inserted.chmod(0o600)
            inserted_once = True
        return real_connect(owner_id, database, **kwargs)

    monkeypatch.setattr(
        module._profile_schema,
        "connect_private_sqlite",
        insert_before_path_open,
    )
    repository = module.TTSProfileRepository(database_path)
    try:
        with pytest.raises(ProfileRepositoryError):
            await repository.open()

        assert inserted_once is True
        assert repository.state is ProfileRepositoryState.UNAVAILABLE
        assert database_path.read_bytes() == active_before
        assert inserted.read_bytes() == evidence
    finally:
        await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("gap", ("after_open", "after_snapshot", "before_open"))
async def test_current_shared_proof_rechecks_publication_namespace_at_late_gaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gap: str,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    journal = tmp_path / ".profiles.sqlite3.migration-publication.json"
    evidence = b"late foreign publication evidence"
    await _create_profile_store(database_path, "Current")
    active_before = database_path.read_bytes()
    real_connect = module._profile_schema.connect_private_sqlite
    real_revalidate = module.revalidate_exact_current_profile_store
    inserted = False

    def insert_once() -> None:
        nonlocal inserted
        if not inserted:
            journal.write_bytes(evidence)
            journal.chmod(0o600)
            inserted = True

    class GapProxy:
        def __init__(self, connection: sqlite3.Connection) -> None:
            self.connection = connection

        def __getattr__(self, name: str) -> Any:
            return getattr(self.connection, name)

        @property
        def row_factory(self) -> object:
            return self.connection.row_factory

        @row_factory.setter
        def row_factory(self, value: object) -> None:
            self.connection.row_factory = cast(Any, value)

        def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
            result = self.connection.execute(sql, *args)
            if gap == "after_open" and sql == "PRAGMA query_only = ON":
                insert_once()
            if gap == "after_snapshot" and "ORDER BY profile_id" in sql:
                insert_once()
            return result

    def proxied_connect(
        owner_id: str,
        database: object,
        **kwargs: object,
    ) -> sqlite3.Connection:
        connection = real_connect(owner_id, database, **kwargs)
        if owner_id == "tts.profile_store":
            return cast(sqlite3.Connection, GapProxy(connection))
        return connection

    def final_revalidate(connection: sqlite3.Connection, path: Path) -> None:
        if gap == "before_open":
            insert_once()
        real_revalidate(connection, path)

    monkeypatch.setattr(
        module._profile_schema,
        "connect_private_sqlite",
        proxied_connect,
    )
    monkeypatch.setattr(
        module, "revalidate_exact_current_profile_store", final_revalidate
    )
    repository = module.TTSProfileRepository(database_path)
    try:
        with pytest.raises(ProfileRepositoryError):
            await repository.open()

        assert inserted is True
        assert repository.state is ProfileRepositoryState.UNAVAILABLE
        assert database_path.read_bytes() == active_before
        assert journal.read_bytes() == evidence
    finally:
        await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("orphan_suffix", ("-wal", "-shm"))
async def test_current_shared_proof_refuses_orphan_sqlite_sidecar_without_migration(
    tmp_path: Path,
    orphan_suffix: str,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    await _create_profile_store(database_path, "Current")
    active_before = database_path.read_bytes()
    orphan = Path(f"{database_path}{orphan_suffix}")
    orphan.write_bytes(b"foreign orphan sidecar")
    orphan.chmod(0o600)
    blocker = ProfileStoreLease(database_path, ProfileStoreLockMode.SHARED)
    blocker.acquire()
    repository = _repository_module().TTSProfileRepository(database_path)
    try:
        with pytest.raises(ProfileRepositoryError, match="lock_timeout"):
            await repository.open()
        assert database_path.read_bytes() == active_before
        assert orphan.read_bytes() == b"foreign orphan sidecar"
    finally:
        blocker.release()
        await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("sidecar_suffix", ("-wal", "-shm"))
async def test_current_shared_proof_retains_exact_sqlite_sidecar_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sidecar_suffix: str,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    first = module.TTSProfileRepository(database_path)
    await first.open()
    await first.create_profile(
        _draft("Current"),
        UUID("00000000-0000-4000-8000-000000000088"),
    )
    sidecar = Path(f"{database_path}{sidecar_suffix}")
    retained_sidecar = tmp_path / f"retained{sidecar_suffix}"
    real_connect = module._profile_schema.connect_private_sqlite
    substituted = False

    def substitute_sidecar_after_path_open(
        owner_id: str,
        database: object,
        **kwargs: object,
    ) -> sqlite3.Connection:
        nonlocal substituted
        connection = real_connect(owner_id, database, **kwargs)
        if owner_id == "tts.profile_store" and not substituted:
            replacement = tmp_path / f"foreign{sidecar_suffix}"
            replacement.write_bytes(sidecar.read_bytes())
            replacement.chmod(0o600)
            os.replace(sidecar, retained_sidecar)
            os.replace(replacement, sidecar)
            substituted = True
        return connection

    monkeypatch.setattr(
        module._profile_schema,
        "connect_private_sqlite",
        substitute_sidecar_after_path_open,
    )
    second = module.TTSProfileRepository(database_path)
    try:
        with pytest.raises(ProfileRepositoryError):
            await second.open()
        assert substituted is True
        assert second.state is ProfileRepositoryState.UNAVAILABLE
    finally:
        await second.close()
        foreign = tmp_path / f"quarantined-foreign{sidecar_suffix}"
        os.replace(sidecar, foreign)
        os.replace(retained_sidecar, sidecar)
        await first.close()


@pytest.mark.asyncio
async def test_current_shared_proof_rejects_main_swap_with_live_wal_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    replacement = tmp_path / "replacement.sqlite3"
    retained = tmp_path / "retained.sqlite3"
    first = module.TTSProfileRepository(database_path)
    await first.open()
    await first.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000077"),
    )
    await _create_profile_store(replacement, "Foreign current")
    replacement_before = replacement.read_bytes()
    real_connect = module._profile_schema.connect_private_sqlite
    swapped = False

    def swap_main_after_sidecar_pin(
        owner_id: str,
        database: object,
        **kwargs: object,
    ) -> sqlite3.Connection:
        nonlocal swapped
        if owner_id == "tts.profile_store" and not swapped:
            os.replace(database_path, retained)
            os.replace(replacement, database_path)
            swapped = True
        return real_connect(owner_id, database, **kwargs)

    monkeypatch.setattr(
        module._profile_schema,
        "connect_private_sqlite",
        swap_main_after_sidecar_pin,
    )
    second = module.TTSProfileRepository(database_path)
    try:
        with pytest.raises(ProfileRepositoryError):
            await second.open()
        assert swapped is True
        assert database_path.read_bytes() == replacement_before
        assert second.state is ProfileRepositoryState.UNAVAILABLE
    finally:
        await second.close()
        foreign = tmp_path / "quarantined-foreign.sqlite3"
        os.replace(database_path, foreign)
        os.replace(retained, database_path)
        await first.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("read", "create", "update", "delete"))
async def test_live_operation_fence_rejects_valid_v4_main_replacement_before_sql(
    tmp_path: Path,
    operation: str,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    replacement = tmp_path / "replacement.sqlite3"
    detached = tmp_path / "detached.sqlite3"
    await _create_profile_store(replacement, "Foreign current")
    await _create_profile_store(database_path, "Original")
    replacement_before = replacement.read_bytes()
    repository = _repository(database_path)
    await repository.open()
    profile_id = UUID("00000000-0000-4000-8000-000000000001")
    os.replace(database_path, detached)
    os.replace(replacement, database_path)

    try:
        with pytest.raises(ProfileRepositoryError, match="operation_failed"):
            if operation == "read":
                await repository.list_profiles()
            elif operation == "create":
                await repository.create_profile(
                    _draft("New"),
                    UUID("00000000-0000-4000-8000-000000000067"),
                )
            elif operation == "update":
                await repository.update_profile(
                    profile_id,
                    1,
                    _draft("Updated"),
                    expected_generation=repository.generation,
                )
            else:
                await repository.delete_profile(
                    profile_id,
                    expected_generation=repository.generation,
                )

        assert repository.state is ProfileRepositoryState.UNAVAILABLE
        assert database_path.read_bytes() == replacement_before
    finally:
        quarantined = tmp_path / "quarantined-foreign.sqlite3"
        os.replace(database_path, quarantined)
        os.replace(detached, database_path)
        await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("sidecar_suffix", ("-wal", "-shm"))
async def test_sidecar_authority_loss_quarantines_without_sqlite_cleanup(
    tmp_path: Path,
    sidecar_suffix: str,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    await _create_profile_store(database_path, "Original")
    repository = _repository(database_path)
    await repository.open()
    authority = repository._connection
    sidecar = Path(f"{database_path}{sidecar_suffix}")
    detached = tmp_path / f"detached{sidecar_suffix}"
    foreign = tmp_path / f"foreign{sidecar_suffix}"
    foreign_bytes = b"foreign sidecar bytes must remain unchanged"
    foreign.write_bytes(foreign_bytes)
    foreign.chmod(0o600)
    os.replace(sidecar, detached)
    os.replace(foreign, sidecar)

    with pytest.raises(ProfileRepositoryError, match="operation_failed"):
        await repository.list_profiles()

    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository._connection is authority
    assert repository._lease is not None and repository._lease.acquired is True
    assert sidecar.read_bytes() == foreign_bytes
    await _assert_exclusive_lease_blocked(database_path)

    with pytest.raises(ProfileRepositoryError, match="operation_failed"):
        await repository.close()
    assert repository._connection is authority
    assert repository._lease is not None and repository._lease.acquired is True
    assert sidecar.read_bytes() == foreign_bytes

    quarantined_foreign = tmp_path / f"quarantined-foreign{sidecar_suffix}"
    os.replace(sidecar, quarantined_foreign)
    os.replace(detached, sidecar)
    await repository.close()
    assert repository._connection is None
    assert repository._lease is None
    assert quarantined_foreign.read_bytes() == foreign_bytes
    assert await asyncio.to_thread(_try_exclusive_lease, database_path) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("target_suffix", ("", "-wal", "-shm"))
async def test_direct_close_quarantines_substituted_exact_store_before_sqlite_cleanup(
    tmp_path: Path,
    target_suffix: str,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    await _create_profile_store(database_path, "Original")
    repository = _repository(database_path)
    await repository.open()
    authority = repository._connection
    target = Path(f"{database_path}{target_suffix}")
    detached = tmp_path / f"detached{target_suffix or '-main'}"
    foreign = tmp_path / f"foreign{target_suffix or '-main'}"
    foreign_bytes = b"foreign exact-store bytes must remain unchanged"
    foreign.write_bytes(foreign_bytes)
    foreign.chmod(0o600)
    os.replace(target, detached)
    os.replace(foreign, target)

    with pytest.raises(ProfileRepositoryError, match="operation_failed"):
        await repository.close()

    assert repository._connection is authority
    assert repository._lease is not None and repository._lease.acquired is True
    assert target.read_bytes() == foreign_bytes
    await _assert_exclusive_lease_blocked(database_path)

    quarantined_foreign = tmp_path / f"quarantined{target_suffix or '-main'}"
    os.replace(target, quarantined_foreign)
    os.replace(detached, target)
    await repository.close()
    assert quarantined_foreign.read_bytes() == foreign_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ("replace", "chmod"))
async def test_live_authority_revalidates_configured_parent_before_sql_or_close(
    tmp_path: Path,
    mutation: str,
) -> None:
    store_parent = tmp_path / "store"
    store_parent.mkdir(mode=0o700)
    database_path = store_parent / "profiles.sqlite3"
    await _create_profile_store(database_path, "Original")
    repository = _repository(database_path)
    await repository.open()
    authority = repository._connection
    detached_parent = tmp_path / "detached-store"
    foreign_parent = tmp_path / "foreign-store"

    if mutation == "replace":
        os.replace(store_parent, detached_parent)
        store_parent.mkdir(mode=0o700)
        (store_parent / "profiles.sqlite3").write_bytes(b"foreign parent bytes")
    else:
        store_parent.chmod(0o777)

    with pytest.raises(ProfileRepositoryError, match="operation_failed"):
        await repository.list_profiles()
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository._connection is authority
    assert repository._lease is not None and repository._lease.acquired is True
    with pytest.raises(ProfileRepositoryError, match="operation_failed"):
        await repository.close()
    assert repository._connection is authority

    if mutation == "replace":
        os.replace(store_parent, foreign_parent)
        os.replace(detached_parent, store_parent)
        assert (
            foreign_parent / "profiles.sqlite3"
        ).read_bytes() == b"foreign parent bytes"
    else:
        store_parent.chmod(0o700)
    await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("sidecar_suffix", ("-wal", "-shm"))
async def test_precommit_sidecar_loss_retains_uncommitted_transaction_without_rollback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sidecar_suffix: str,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    await _create_profile_store(database_path)
    repository = module.TTSProfileRepository(database_path)
    await repository.open()
    authority = cast(Any, repository._connection)
    sidecar = Path(f"{database_path}{sidecar_suffix}")
    detached = tmp_path / f"detached{sidecar_suffix}"
    foreign = tmp_path / f"foreign{sidecar_suffix}"
    foreign_bytes = b"foreign precommit sidecar bytes"
    foreign.write_bytes(foreign_bytes)
    foreign.chmod(0o600)
    real_revalidate = module.revalidate_exact_current_profile_store
    swapped = False

    def replace_sidecar_before_commit(
        connection: sqlite3.Connection,
        path: Path,
    ) -> None:
        nonlocal swapped
        if connection.in_transaction and not swapped:
            os.replace(sidecar, detached)
            os.replace(foreign, sidecar)
            swapped = True
        real_revalidate(connection, path)

    monkeypatch.setattr(
        module,
        "revalidate_exact_current_profile_store",
        replace_sidecar_before_commit,
    )
    with pytest.raises(ProfileRepositoryError, match="operation_failed"):
        await repository.create_profile(
            _draft("Must remain uncommitted"),
            UUID("00000000-0000-4000-8000-000000000045"),
        )

    assert swapped is True
    assert authority.in_transaction is True
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository._connection is authority
    assert repository._lease is not None and repository._lease.acquired is True
    assert sidecar.read_bytes() == foreign_bytes

    quarantined_foreign = tmp_path / f"quarantined-foreign{sidecar_suffix}"
    os.replace(sidecar, quarantined_foreign)
    os.replace(detached, sidecar)
    await repository.close()
    assert quarantined_foreign.read_bytes() == foreign_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize("artifact_kind", ("publication", "rollback"))
async def test_live_operation_fence_rejects_inserted_journal_artifact(
    tmp_path: Path,
    artifact_kind: str,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    await _create_profile_store(database_path)
    repository = _repository(database_path)
    await repository.open()
    artifact = (
        tmp_path / ".profiles.sqlite3.migration-publication.json"
        if artifact_kind == "publication"
        else Path(f"{database_path}-journal")
    )
    artifact.write_bytes(b"foreign journal evidence")
    artifact.chmod(0o600)

    try:
        with pytest.raises(ProfileRepositoryError, match="operation_failed"):
            await repository.list_profiles()
        assert repository.state is ProfileRepositoryState.UNAVAILABLE
        assert artifact.read_bytes() == b"foreign journal evidence"
        with pytest.raises(ProfileRepositoryError, match="operation_failed"):
            await repository.close()
        assert repository._connection is not None
        assert repository._lease is not None and repository._lease.acquired is True
        assert artifact.read_bytes() == b"foreign journal evidence"
    finally:
        if artifact.exists():
            artifact.unlink()
        await repository.close()


@pytest.mark.asyncio
async def test_write_transaction_revalidates_authority_immediately_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    replacement = tmp_path / "replacement.sqlite3"
    detached = tmp_path / "detached.sqlite3"
    await _create_profile_store(replacement, "Foreign current")
    await _create_profile_store(database_path)
    replacement_before = replacement.read_bytes()
    repository = module.TTSProfileRepository(database_path)
    await repository.open()
    real_revalidate = module.revalidate_exact_current_profile_store
    swapped = False

    def swap_inside_transaction(
        connection: sqlite3.Connection,
        path: Path,
    ) -> None:
        nonlocal swapped
        if connection.in_transaction and not swapped:
            os.replace(database_path, detached)
            os.replace(replacement, database_path)
            swapped = True
        real_revalidate(connection, path)

    monkeypatch.setattr(
        module,
        "revalidate_exact_current_profile_store",
        swap_inside_transaction,
    )
    try:
        with pytest.raises(ProfileRepositoryError, match="operation_failed"):
            await repository.create_profile(
                _draft("Must remain uncommitted"),
                UUID("00000000-0000-4000-8000-000000000044"),
            )
        assert swapped is True
        assert repository.state is ProfileRepositoryState.UNAVAILABLE
        assert database_path.read_bytes() == replacement_before
    finally:
        quarantined = tmp_path / "quarantined-foreign.sqlite3"
        os.replace(database_path, quarantined)
        os.replace(detached, database_path)
        await repository.close()


@pytest.mark.asyncio
async def test_read_transaction_revalidates_authority_before_returning_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    replacement = tmp_path / "replacement.sqlite3"
    detached = tmp_path / "detached.sqlite3"
    await _create_profile_store(database_path, "Detached result")
    await _create_profile_store(replacement, "Foreign current")
    replacement_before = replacement.read_bytes()
    repository = module.TTSProfileRepository(database_path)
    await repository.open()
    real_revalidate = module.revalidate_exact_current_profile_store
    swapped = False

    def swap_after_read(
        connection: sqlite3.Connection,
        path: Path,
    ) -> None:
        nonlocal swapped
        if connection.in_transaction and not swapped:
            os.replace(database_path, detached)
            os.replace(replacement, database_path)
            swapped = True
        real_revalidate(connection, path)

    monkeypatch.setattr(
        module,
        "revalidate_exact_current_profile_store",
        swap_after_read,
    )
    try:
        with pytest.raises(ProfileRepositoryError, match="operation_failed"):
            await repository.list_profiles()
        assert swapped is True
        assert repository.state is ProfileRepositoryState.UNAVAILABLE
        assert database_path.read_bytes() == replacement_before
    finally:
        quarantined = tmp_path / "quarantined-foreign.sqlite3"
        os.replace(database_path, quarantined)
        os.replace(detached, database_path)
        await repository.close()


@pytest.mark.asyncio
async def test_authority_mismatch_close_retains_connection_and_shared_lease(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    replacement = tmp_path / "replacement.sqlite3"
    detached = tmp_path / "detached.sqlite3"
    await _create_profile_store(database_path, "Original")
    await _create_profile_store(replacement, "Foreign current")
    replacement_before = replacement.read_bytes()
    repository = _repository(database_path)
    await repository.open()
    authority = cast(Any, repository._connection)
    close_proxy = _CloseFailingSQLiteProxy(
        cast(sqlite3.Connection, authority._connection),
        str(tmp_path / "secret-close-error"),
    )
    authority._connection = close_proxy
    os.replace(database_path, detached)
    os.replace(replacement, database_path)

    with pytest.raises(ProfileRepositoryError, match="operation_failed"):
        await repository.list_profiles()

    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository._connection is authority
    assert repository._lease is not None
    assert repository._lease.mode is ProfileStoreLockMode.SHARED
    assert repository._lease.acquired is True
    await _assert_exclusive_lease_blocked(database_path)

    with pytest.raises(ProfileRepositoryError, match="operation_failed"):
        await repository.close()
    assert close_proxy.close_calls == 0
    assert database_path.read_bytes() == replacement_before
    assert repository._connection is authority
    assert repository._lease is not None and repository._lease.acquired is True
    quarantined = tmp_path / "quarantined-foreign.sqlite3"
    os.replace(database_path, quarantined)
    os.replace(detached, database_path)
    close_proxy.fail_close = False
    await repository.close()
    assert repository._connection is None
    assert repository._lease is None
    assert await asyncio.to_thread(_try_exclusive_lease, database_path) is None


@pytest.mark.asyncio
async def test_restore_fences_live_authority_before_checkpoint_or_publication(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    replacement = tmp_path / "replacement.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    detached = tmp_path / "detached.sqlite3"
    await _create_profile_store(database_path, "Original")
    await _create_profile_store(replacement, "Foreign current")
    await _create_profile_store(candidate, "Restore candidate")
    replacement_before = replacement.read_bytes()
    candidate_before = candidate.read_bytes()
    repository = _repository(database_path)
    await repository.open()
    os.replace(database_path, detached)
    os.replace(replacement, database_path)

    try:
        with pytest.raises(ProfileRepositoryError, match="operation_failed"):
            await repository.restore_from(candidate)
        assert repository.state is ProfileRepositoryState.UNAVAILABLE
        assert database_path.read_bytes() == replacement_before
        assert candidate.read_bytes() == candidate_before
    finally:
        quarantined = tmp_path / "quarantined-foreign.sqlite3"
        os.replace(database_path, quarantined)
        os.replace(detached, database_path)
        await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("journal_suffix", ("", "-wal", "-shm", "-journal"))
async def test_journal_present_under_shared_reader_never_runs_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    journal_suffix: str,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    await _create_profile_store(database_path, "Current")
    journal = tmp_path / f".profiles.sqlite3.migration-publication.json{journal_suffix}"
    journal.write_bytes(b"private journal evidence")
    journal.chmod(0o600)
    active_before = database_path.read_bytes()
    journal_before = journal.read_bytes()
    recovery_calls = 0

    def forbidden_recovery(_path: Path) -> bool:
        nonlocal recovery_calls
        recovery_calls += 1
        raise AssertionError("recovery ran without exclusive ownership")

    blocker = ProfileStoreLease(database_path, ProfileStoreLockMode.SHARED)
    blocker.acquire()
    monkeypatch.setattr(
        module, "recover_profile_migration_publication", forbidden_recovery
    )
    repository = module.TTSProfileRepository(database_path)
    try:
        with pytest.raises(ProfileRepositoryError, match="lock_timeout"):
            await repository.open()
        assert recovery_calls == 0
        assert database_path.read_bytes() == active_before
        assert journal.read_bytes() == journal_before
    finally:
        blocker.release()
        await repository.close()


@pytest.mark.asyncio
async def test_v3_open_blocked_by_shared_lease_never_recovers_or_falls_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    _build_populated_v3_store_at(database_path)
    active_before = database_path.read_bytes()
    recovery_calls = 0
    real_recover = module.recover_profile_migration_publication

    def tracked_recovery(path: Path) -> bool:
        nonlocal recovery_calls
        recovery_calls += 1
        return real_recover(path)

    blocker = ProfileStoreLease(database_path, ProfileStoreLockMode.SHARED)
    blocker.acquire()
    monkeypatch.setattr(
        module, "recover_profile_migration_publication", tracked_recovery
    )
    repository = module.TTSProfileRepository(database_path)
    try:
        with pytest.raises(ProfileRepositoryError, match="lock_timeout"):
            await repository.open()

        assert recovery_calls == 0
        assert repository.state is ProfileRepositoryState.UNAVAILABLE
        assert database_path.read_bytes() == active_before
        assert not database_path.with_name("profiles.sqlite3.pre-v4.sqlite3").exists()
        assert not tuple(tmp_path.glob(".profile-migration-*.candidate.sqlite3"))
    finally:
        blocker.release()
        await repository.close()


def _assert_safe_error(
    error: ProfileRepositoryError,
    code: str,
    *secrets: str,
) -> None:
    assert type(error) is ProfileRepositoryError
    assert error.code == code
    assert str(error) == f"TTS profile repository failed: {code}"
    assert error.__cause__ is None
    assert error.__context__ is None
    visible = " ".join(
        (
            str(error),
            repr(error),
            "".join(traceback.format_exception(error)),
            *(str(note) for note in getattr(error, "__notes__", ())),
        )
    )
    for secret in secrets:
        assert secret not in visible


def _tainted_repository_error(
    code: str,
    secret: str,
) -> ProfileRepositoryError:
    try:
        raise RuntimeError(secret)
    except RuntimeError:
        try:
            raise ProfileRepositoryError(code)
        except ProfileRepositoryError as error:
            return error


def _try_exclusive_lease(database_path: Path) -> ProfileRepositoryError | None:
    contender = ProfileStoreLease(
        database_path,
        ProfileStoreLockMode.EXCLUSIVE,
        timeout_seconds=0.05,
        check_interval_seconds=0.005,
    )
    try:
        contender.acquire()
    except ProfileRepositoryError as error:
        return error
    try:
        return None
    finally:
        contender.release()


async def _assert_exclusive_lease_blocked(database_path: Path) -> None:
    error = await asyncio.to_thread(_try_exclusive_lease, database_path)
    assert error is not None
    _assert_safe_error(error, "lock_timeout", str(database_path))


async def _wait_thread_event(event: threading.Event) -> None:
    assert await asyncio.to_thread(event.wait, 5.0)


async def _run_in_new_loop_thread(
    awaitable_factory: Callable[[], Awaitable[Any]],
) -> tuple[Any | None, BaseException | None]:
    outcomes: list[tuple[Any | None, BaseException | None]] = []

    async def invoke() -> Any:
        return await awaitable_factory()

    def runner() -> None:
        try:
            outcomes.append((asyncio.run(invoke()), None))
        except BaseException as error:
            outcomes.append((None, error))

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    await asyncio.to_thread(thread.join, 5.0)
    assert not thread.is_alive()
    assert len(outcomes) == 1
    return outcomes[0]


def _install_fake_store(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    events: list[tuple[str, int]],
    connection: _RecordingConnection,
    *,
    leases: list[_RecordingLease] | None = None,
) -> list[_RecordingLease]:
    recorded_leases = leases if leases is not None else []

    def lease_factory(
        _database_path: Path,
        mode: ProfileStoreLockMode,
        **_kwargs: object,
    ) -> _RecordingLease:
        lease = _RecordingLease(events)
        recorded_leases.append(lease)
        return lease

    def opener(
        _database_path: Path,
        *,
        must_exist: bool = False,
    ) -> Any:
        del must_exist
        events.append(("store.open", threading.get_ident()))
        return connection

    def fake_proven_current(self: Any, path: Path) -> tuple[Any, Any]:
        lease = module.ProfileStoreLease(path, ProfileStoreLockMode.SHARED)
        opened: Any = None
        try:
            lease.acquire()
            opened = module.open_profile_store(path, must_exist=True)
            return lease, opened
        except BaseException as primary:
            cleanup: BaseException | None = None
            if opened is not None:
                try:
                    opened.close()
                except BaseException as error:
                    cleanup = error
                    self._connection = opened
                    self._lease = lease
                    self._active_database_path = path
            if cleanup is None:
                try:
                    lease.release()
                except BaseException as error:
                    cleanup = error
                    self._lease = lease
                    self._active_database_path = path
            if cleanup is not None and not isinstance(cleanup, Exception):
                raise cleanup
            raise primary

    monkeypatch.setattr(module, "ProfileStoreLease", lease_factory)
    monkeypatch.setattr(
        module.TTSProfileRepository,
        "_worker_initialize_store",
        lambda _self, _path, **_kwargs: None,
    )
    monkeypatch.setattr(
        module.TTSProfileRepository,
        "_worker_open_if_proven_current",
        fake_proven_current,
    )
    monkeypatch.setattr(module, "open_profile_store", opener)
    return recorded_leases


def _bypass_startup_ownership_for_open_cleanup_test(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
) -> None:
    """Keep legacy cleanup tests focused on the shared-handle phase."""

    def fake_proven_current(self: Any, path: Path) -> tuple[Any, Any]:
        lease = module.ProfileStoreLease(path, ProfileStoreLockMode.SHARED)
        opened: Any = None
        try:
            lease.acquire()
            opened = module.open_profile_store(path, must_exist=True)
            return lease, opened
        except BaseException as primary:
            cleanup: BaseException | None = None
            if opened is not None:
                try:
                    opened.close()
                except BaseException as error:
                    cleanup = error
                    self._connection = opened
                    self._lease = lease
                    self._active_database_path = path
            if cleanup is None:
                try:
                    lease.release()
                except BaseException as error:
                    cleanup = error
                    self._lease = lease
                    self._active_database_path = path
            if cleanup is not None and not isinstance(cleanup, Exception):
                raise cleanup
            raise primary

    monkeypatch.setattr(
        module.TTSProfileRepository,
        "_worker_initialize_store",
        lambda _self, _path, **_kwargs: None,
    )
    monkeypatch.setattr(
        module.TTSProfileRepository,
        "_worker_open_if_proven_current",
        fake_proven_current,
    )


def _phase_threads(
    events: list[tuple[str, int]],
    phase: str,
) -> list[int]:
    return [thread_id for name, thread_id in events if name == phase]


def test_constructor_is_pure_and_starts_initial_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "nested" / "profiles.sqlite3"
    entries_before = tuple(tmp_path.iterdir())

    def forbidden(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("constructor performed lazy lifecycle work")

    monkeypatch.setattr(module, "ThreadPoolExecutor", forbidden)
    monkeypatch.setattr(module, "ProfileStoreLease", forbidden)
    monkeypatch.setattr(module, "open_profile_store", forbidden)

    repository = module.TTSProfileRepository(database_path)

    assert tuple(tmp_path.iterdir()) == entries_before
    assert repository.state is ProfileRepositoryState.CLOSED
    assert repository.generation == 0
    assert repository.terminal is False
    assert repository._executor is None
    assert repository._connection is None
    assert repository._lease is None
    assert repository._active_database_path is None
    assert repository._store_established is False
    assert repository._owner_loop is None
    assert repository._lifecycle_lock is None


def test_constructor_rejects_non_path_without_exposing_value(tmp_path: Path) -> None:
    secret = str(tmp_path / "secret-profile-store.sqlite3")

    with pytest.raises(ProfileRepositoryError) as caught:
        _repository_module().TTSProfileRepository(secret)

    _assert_safe_error(caught.value, "operation_failed", secret)


@pytest.mark.asyncio
async def test_open_uses_one_worker_for_lease_connection_sql_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    loop_thread = threading.get_ident()
    events: list[tuple[str, int]] = []
    sql_traces: list[tuple[int, str]] = []

    class RecordingLease(ProfileStoreLease):
        def acquire(self) -> ProfileStoreLease:
            events.append(("lease.acquire", threading.get_ident()))
            return super().acquire()

        def release(self) -> None:
            events.append(("lease.release", threading.get_ident()))
            super().release()

    def traced_open(
        path: Path,
        *,
        must_exist: bool = False,
    ) -> sqlite3.Connection:
        events.append(("store.open", threading.get_ident()))
        connection = open_profile_store(path, must_exist=must_exist)
        connection.set_trace_callback(
            lambda statement: sql_traces.append((threading.get_ident(), statement))
        )
        return connection

    monkeypatch.setattr(module, "ProfileStoreLease", RecordingLease)
    monkeypatch.setattr(module, "open_profile_store", traced_open)
    repository = module.TTSProfileRepository(database_path)

    opened = await repository.open()
    assert repository._store_established is True

    def query(connection: sqlite3.Connection) -> int:
        events.append(("operation", threading.get_ident()))
        return cast(
            int,
            connection.execute(
                "SELECT count(*) FROM tts_generation_profiles"
            ).fetchone()[0],
        )

    result = await repository._submit_operation(query)
    closed = await repository.close()

    worker_threads = {
        thread_id
        for phase, thread_id in events
        if phase in {"lease.acquire", "store.open", "operation", "lease.release"}
    }
    assert opened == ProfileStoreResult(generation=1, value=None)
    assert result == ProfileStoreResult(generation=1, value=0)
    assert closed == ProfileStoreResult(generation=2, value=None)
    assert len(worker_threads) == 1
    assert loop_thread not in worker_threads
    assert sql_traces
    assert {thread_id for thread_id, _statement in sql_traces} == worker_threads
    assert [phase for phase, _thread_id in events].index("lease.acquire") < [
        phase for phase, _thread_id in events
    ].index("store.open")


@pytest.mark.asyncio
async def test_open_rejects_configured_symlink_drift_before_publishing_active_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    active_path = tmp_path / "active.sqlite3"
    alternate_path = tmp_path / "alternate.sqlite3"
    configured_path = tmp_path / "configured.sqlite3"
    await _create_profile_store(active_path, "Active")
    await _create_profile_store(alternate_path, "Alternate")
    configured_path.symlink_to(active_path)
    real_open = module.open_exact_current_profile_store

    def drifting_open(
        path: Path,
    ) -> sqlite3.Connection:
        connection = real_open(path)
        configured_path.unlink()
        configured_path.symlink_to(alternate_path)
        return connection

    monkeypatch.setattr(module, "open_exact_current_profile_store", drifting_open)
    repository = _repository(configured_path)

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.open()

    _assert_safe_error(caught.value, "operation_failed", str(configured_path))
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository.generation == 1
    assert repository._connection is None
    assert repository._lease is None
    assert repository._active_database_path is None
    assert await asyncio.to_thread(_try_exclusive_lease, active_path) is None
    assert await asyncio.to_thread(_try_exclusive_lease, alternate_path) is None
    await repository.close()


@pytest.mark.asyncio
async def test_open_and_close_are_idempotent_and_shutdown_once_off_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    executors: list[_RecordingExecutor] = []

    def executor_factory(max_workers: int) -> _RecordingExecutor:
        executor = _RecordingExecutor(max_workers, events)
        executors.append(executor)
        return executor

    monkeypatch.setattr(module, "ThreadPoolExecutor", executor_factory)
    leases = _install_fake_store(
        monkeypatch,
        module,
        events,
        connection,
    )
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    loop_thread = threading.get_ident()

    first_open = await repository.open()
    second_open = await repository.open()
    first_close = await repository.close()
    second_close = await repository.close()

    assert first_open == second_open == ProfileStoreResult(generation=1, value=None)
    assert (
        first_close
        == second_close
        == ProfileStoreResult(
            generation=2,
            value=None,
        )
    )
    assert repository.state is ProfileRepositoryState.CLOSED
    assert repository.generation == 2
    assert repository.terminal is True
    assert len(executors) == 1
    assert executors[0].max_workers == 1
    assert executors[0].shutdown_calls == 1
    assert len(leases) == 1
    assert _phase_threads(events, "lease.acquire") == _phase_threads(
        events, "store.open"
    )
    assert _phase_threads(events, "connection.close") == _phase_threads(
        events, "lease.release"
    )
    assert _phase_threads(events, "lease.acquire") == _phase_threads(
        events, "connection.close"
    )
    assert _phase_threads(events, "executor.shutdown")[0] != loop_thread

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.open()
    _assert_safe_error(caught.value, "terminal", str(tmp_path))
    assert len(executors) == 1


@pytest.mark.asyncio
async def test_close_before_open_is_terminal_without_creating_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    executor_calls = 0

    def forbidden_executor(_max_workers: int) -> Any:
        nonlocal executor_calls
        executor_calls += 1
        raise AssertionError

    monkeypatch.setattr(module, "ThreadPoolExecutor", forbidden_executor)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")

    first = await repository.close()
    second = await repository.close()

    assert first == second == ProfileStoreResult(generation=1, value=None)
    assert repository.state is ProfileRepositoryState.CLOSED
    assert repository.generation == 1
    assert repository.terminal is True
    assert executor_calls == 0
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.open()
    _assert_safe_error(caught.value, "terminal")
    assert executor_calls == 0


@pytest.mark.asyncio
async def test_concurrent_open_calls_share_one_attempt_and_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    open_started = threading.Event()
    allow_open = threading.Event()
    leases = _install_fake_store(
        monkeypatch,
        module,
        events,
        connection,
    )

    def blocked_open(
        _database_path: Path,
        *,
        must_exist: bool = False,
    ) -> Any:
        del must_exist
        events.append(("store.open", threading.get_ident()))
        open_started.set()
        assert allow_open.wait(5.0)
        return connection

    monkeypatch.setattr(module, "open_profile_store", blocked_open)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    second_started = asyncio.Event()

    first_task = asyncio.create_task(repository.open())

    async def second_open() -> ProfileStoreResult[None]:
        second_started.set()
        return cast(ProfileStoreResult[None], await repository.open())

    second_task = asyncio.create_task(second_open())
    await _wait_thread_event(open_started)
    await second_started.wait()
    allow_open.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert first == second == ProfileStoreResult(generation=1, value=None)
    assert len(leases) == 1
    assert len(_phase_threads(events, "store.open")) == 1
    assert repository.generation == 1
    await repository.close()


@pytest.mark.asyncio
async def test_active_open_rejects_foreign_loop_before_joining_shared_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    open_started = threading.Event()
    allow_open = threading.Event()
    open_calls = 0
    _install_fake_store(monkeypatch, module, events, connection)

    def blocked_open(
        _database_path: Path,
        *,
        must_exist: bool = False,
    ) -> Any:
        del must_exist
        nonlocal open_calls
        open_calls += 1
        open_started.set()
        assert allow_open.wait(5.0)
        return connection

    monkeypatch.setattr(module, "open_profile_store", blocked_open)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    main_open = asyncio.create_task(repository.open())
    await _wait_thread_event(open_started)

    try:
        result, error = await _run_in_new_loop_thread(repository.open)

        assert result is None
        assert isinstance(error, ProfileRepositoryError)
        _assert_safe_error(error, "invalid_state", "event loop", repr(main_open))
        assert main_open.done() is False
        assert repository.generation == 1
        assert open_calls == 1
    finally:
        allow_open.set()
        await main_open
        await repository.close()


@pytest.mark.asyncio
async def test_operation_caller_paths_reject_foreign_loop_without_worker_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()
    foreign_work_ran = threading.Event()
    admission = repository._admit_operation(lambda _connection: "main-result")

    async def admit_from_foreign_loop() -> None:
        repository._admit_operation(lambda _connection: foreign_work_ran.set())

    caller_paths: tuple[Callable[[], Awaitable[Any]], ...] = (
        admit_from_foreign_loop,
        lambda: repository._submit_operation(
            lambda _connection: foreign_work_ran.set()
        ),
        lambda: repository._publish_operation(admission),
    )

    try:
        for caller_path in caller_paths:
            result, error = await _run_in_new_loop_thread(caller_path)
            assert result is None
            assert isinstance(error, ProfileRepositoryError)
            _assert_safe_error(error, "invalid_state", "event loop")
        assert foreign_work_ran.is_set() is False
        assert await repository._publish_operation(admission) == ProfileStoreResult(
            generation=1,
            value="main-result",
        )
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_terminal_close_rejects_foreign_loop_but_stays_idempotent_on_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()
    closed = await repository.close()

    result, error = await _run_in_new_loop_thread(repository.close)

    assert result is None
    assert isinstance(error, ProfileRepositoryError)
    _assert_safe_error(error, "invalid_state", "event loop")
    assert await repository.close() == closed
    assert repository.generation == 2


@pytest.mark.asyncio
async def test_overlapping_failed_open_calls_share_attempt_before_later_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    first_attempt_started = threading.Event()
    release_first_failure = threading.Event()
    attempts = 0
    failures_enabled = True
    leases = _install_fake_store(
        monkeypatch,
        module,
        events,
        connection,
    )

    def controlled_open(
        _database_path: Path,
        *,
        must_exist: bool = False,
    ) -> Any:
        del must_exist
        nonlocal attempts
        attempts += 1
        events.append(("store.open", threading.get_ident()))
        if failures_enabled:
            if attempts == 1:
                first_attempt_started.set()
                assert release_first_failure.wait(5.0)
            raise ProfileRepositoryError("schema_corrupt")
        return connection

    monkeypatch.setattr(module, "open_profile_store", controlled_open)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")

    async def capture_open_error() -> ProfileRepositoryError:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.open()
        return caught.value

    first_task = asyncio.create_task(capture_open_error())
    await _wait_thread_event(first_attempt_started)
    second_invoked = asyncio.Event()

    async def overlapping_open() -> ProfileRepositoryError:
        second_invoked.set()
        return await capture_open_error()

    second_task = asyncio.create_task(overlapping_open())
    await second_invoked.wait()
    release_first_failure.set()
    first_error, second_error = await asyncio.gather(first_task, second_task)

    assert first_error is not second_error
    _assert_safe_error(first_error, "schema_corrupt")
    _assert_safe_error(second_error, "schema_corrupt")
    assert attempts == 1
    assert len(leases) == 1
    assert len(_phase_threads(events, "lease.acquire")) == 1
    assert len(_phase_threads(events, "store.open")) == 1
    assert len(set(_phase_threads(events, "store.open"))) == 1
    assert leases[0].acquired is False
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository.generation == 1

    failures_enabled = False
    retried = await repository.open()

    assert retried == ProfileStoreResult(generation=2, value=None)
    assert attempts == 2
    assert len(leases) == 2
    assert repository.state is ProfileRepositoryState.OPEN
    assert repository.generation == 2
    await repository.close()


@pytest.mark.asyncio
async def test_cancelled_overlapping_open_waits_for_shared_attempt_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    attempt_started = threading.Event()
    release_failure = threading.Event()
    attempts = 0
    _install_fake_store(monkeypatch, module, events, connection)

    def controlled_open(
        _database_path: Path,
        *,
        must_exist: bool = False,
    ) -> Any:
        del must_exist
        nonlocal attempts
        attempts += 1
        attempt_started.set()
        assert release_failure.wait(5.0)
        raise ProfileRepositoryError("schema_corrupt")

    monkeypatch.setattr(module, "open_profile_store", controlled_open)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    first_task = asyncio.create_task(repository.open())
    await _wait_thread_event(attempt_started)
    second_invoked = asyncio.Event()

    async def overlapping_open() -> ProfileStoreResult[None]:
        second_invoked.set()
        return cast(ProfileStoreResult[None], await repository.open())

    second_task = asyncio.create_task(overlapping_open())
    await second_invoked.wait()
    second_task.cancel()
    loop = asyncio.get_running_loop()
    cancellation_checkpoint: asyncio.Future[bool] = loop.create_future()

    def observe_after_cancellation_delivery() -> None:
        loop.call_soon(cancellation_checkpoint.set_result, second_task.done())

    loop.call_soon(observe_after_cancellation_delivery)

    cancelled_before_settlement = await cancellation_checkpoint
    release_failure.set()
    assert cancelled_before_settlement is False
    with pytest.raises(ProfileRepositoryError) as first_error:
        await first_task
    with pytest.raises(asyncio.CancelledError):
        await second_task

    _assert_safe_error(first_error.value, "schema_corrupt")
    assert attempts == 1
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository.generation == 1
    await repository.close()


@pytest.mark.asyncio
async def test_invalid_existing_store_fails_closed_and_retry_can_recover(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TABLE unrelated(value TEXT)")
    connection.commit()
    connection.close()
    invalid_bytes = database_path.read_bytes()
    repository = _repository(database_path)

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.open()

    _assert_safe_error(caught.value, "schema_partial", str(database_path))
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository.generation == 1
    assert repository.terminal is False
    assert database_path.read_bytes() == invalid_bytes
    executor = repository._executor

    database_path.unlink()
    retried = await repository.open()

    assert retried == ProfileStoreResult(generation=2, value=None)
    assert repository.state is ProfileRepositoryState.OPEN
    assert repository.generation == 2
    assert repository._executor is executor
    await repository.close()


@pytest.mark.asyncio
async def test_failed_open_cleans_partial_lease_and_maps_hostile_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    secret = str(tmp_path / "secret-store.sqlite3")
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    leases = _install_fake_store(
        monkeypatch,
        module,
        events,
        connection,
    )

    def hostile_open(
        _database_path: Path,
        *,
        must_exist: bool = False,
    ) -> Any:
        del must_exist
        events.append(("store.open", threading.get_ident()))
        raise RuntimeError(secret)

    monkeypatch.setattr(module, "open_profile_store", hostile_open)
    repository = module.TTSProfileRepository(Path(secret))

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.open()

    _assert_safe_error(caught.value, "operation_failed", secret)
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository.generation == 1
    assert repository._executor is not None
    assert len(leases) == 1
    assert leases[0].acquired is False
    assert _phase_threads(events, "lease.acquire") == _phase_threads(
        events, "lease.release"
    )

    def healthy_open(
        _database_path: Path,
        *,
        must_exist: bool = False,
    ) -> Any:
        del must_exist
        events.append(("store.open", threading.get_ident()))
        return connection

    monkeypatch.setattr(module, "open_profile_store", healthy_open)
    retried = await repository.open()

    assert retried == ProfileStoreResult(generation=2, value=None)
    assert len(leases) == 2
    await repository.close()


@pytest.mark.asyncio
async def test_failed_open_retains_lease_until_retry_cleanup_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    _bypass_startup_ownership_for_open_cleanup_test(monkeypatch, module)
    secret = str(tmp_path / "residual-lease-failure")
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    leases: list[_SequencedReleaseLease] = []
    open_calls = 0

    def lease_factory(
        _database_path: Path,
        mode: ProfileStoreLockMode,
        **_kwargs: object,
    ) -> _SequencedReleaseLease:
        assert mode is ProfileStoreLockMode.SHARED
        release_errors: list[BaseException | None]
        if not leases:
            release_errors = [RuntimeError(secret), RuntimeError(secret), None]
        else:
            release_errors = []
        lease = _SequencedReleaseLease(
            events,
            f"lease-{len(leases) + 1}",
            release_errors,
        )
        leases.append(lease)
        return lease

    def controlled_open(
        _database_path: Path,
        *,
        must_exist: bool = False,
    ) -> Any:
        del must_exist
        nonlocal open_calls
        open_calls += 1
        events.append((f"store.open-{open_calls}", threading.get_ident()))
        if open_calls == 1:
            raise RuntimeError(secret)
        return connection

    monkeypatch.setattr(module, "ProfileStoreLease", lease_factory)
    monkeypatch.setattr(module, "open_profile_store", controlled_open)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")

    try:
        with pytest.raises(ProfileRepositoryError) as first_error:
            await repository.open()
        _assert_safe_error(first_error.value, "operation_failed", secret)
        assert repository._lease is leases[0]
        assert leases[0].acquired is True
        assert leases[0].release_calls == 1

        with pytest.raises(ProfileRepositoryError) as cleanup_error:
            await repository.open()
        _assert_safe_error(cleanup_error.value, "operation_failed", secret)
        assert repository.state is ProfileRepositoryState.UNAVAILABLE
        assert repository._lease is leases[0]
        assert leases[0].acquired is True
        assert leases[0].release_calls == 2
        assert len(leases) == 1
        assert open_calls == 1

        retried = await repository.open()

        assert retried == ProfileStoreResult(generation=3, value=None)
        assert leases[0].release_calls == 3
        assert leases[0].acquired is False
        assert repository._lease is leases[1]
        assert leases[1].acquired is True
        assert open_calls == 2
        phases = [phase for phase, _thread_id in events]
        assert max(
            index for index, phase in enumerate(phases) if phase == "lease-1.release"
        ) < phases.index("lease-2.acquire")
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_retry_cleans_residual_connection_before_acquiring_new_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    _bypass_startup_ownership_for_open_cleanup_test(monkeypatch, module)
    secret = str(tmp_path / "residual-connection-failure")
    events: list[tuple[str, int]] = []
    first_connection = _SequencedCloseConnection(
        events,
        "connection-1",
        [RuntimeError(secret), None],
    )
    second_connection = _SequencedCloseConnection(events, "connection-2", [])
    connections = [first_connection, second_connection]
    leases: list[_SequencedReleaseLease] = []
    open_calls = 0

    def lease_factory(
        _database_path: Path,
        mode: ProfileStoreLockMode,
        **_kwargs: object,
    ) -> _SequencedReleaseLease:
        assert mode is ProfileStoreLockMode.SHARED
        lease = _SequencedReleaseLease(
            events,
            f"lease-{len(leases) + 1}",
            [],
        )
        leases.append(lease)
        return lease

    def controlled_open(
        _database_path: Path,
        *,
        must_exist: bool = False,
    ) -> Any:
        nonlocal open_calls
        assert must_exist is True
        connection = connections[open_calls]
        open_calls += 1
        events.append((f"store.open-{open_calls}", threading.get_ident()))
        return connection

    monkeypatch.setattr(module, "ProfileStoreLease", lease_factory)
    monkeypatch.setattr(module, "open_profile_store", controlled_open)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()
    with repository._state_lock:
        repository._state = ProfileRepositoryState.UNAVAILABLE

    try:
        with pytest.raises(ProfileRepositoryError) as cleanup_error:
            await repository.open()
        _assert_safe_error(cleanup_error.value, "operation_failed", secret)
        assert repository.state is ProfileRepositoryState.UNAVAILABLE
        assert repository._connection is first_connection
        assert first_connection.closed is False
        assert first_connection.close_calls == 1
        assert repository._lease is leases[0]
        assert leases[0].acquired is True
        assert leases[0].release_calls == 0
        assert len(leases) == 1
        assert open_calls == 1

        retried = await repository.open()

        assert retried == ProfileStoreResult(generation=3, value=None)
        assert first_connection.closed is True
        assert first_connection.close_calls == 2
        assert repository._connection is second_connection
        assert len(leases) == 2
        assert open_calls == 2
        phases = [phase for phase, _thread_id in events]
        assert phases.index("connection-1.close") < phases.index("lease-1.release")
        assert phases.index("lease-1.release") < phases.index("lease-2.acquire")
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_open_recreates_structured_error_without_hostile_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    secret = str(tmp_path / "secret-open-context.sqlite3")
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    incoming = _tainted_repository_error("schema_corrupt", secret)
    _install_fake_store(monkeypatch, module, events, connection)

    def hostile_open(
        _database_path: Path,
        *,
        must_exist: bool = False,
    ) -> Any:
        del must_exist
        raise incoming

    monkeypatch.setattr(module, "open_profile_store", hostile_open)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.open()

    assert caught.value is not incoming
    _assert_safe_error(caught.value, "schema_corrupt", secret)
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository.generation == 1
    await repository.close()


@pytest.mark.asyncio
async def test_open_rejects_missing_connection_and_releases_partial_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    leases = _install_fake_store(
        monkeypatch,
        module,
        events,
        connection,
    )

    def missing_connection(
        _database_path: Path,
        *,
        must_exist: bool = False,
    ) -> Any:
        del must_exist
        events.append(("store.open", threading.get_ident()))
        return None

    monkeypatch.setattr(module, "open_profile_store", missing_connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.open()

    _assert_safe_error(caught.value, "operation_failed")
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository.generation == 1
    assert len(leases) == 1
    assert leases[0].acquired is False
    assert _phase_threads(events, "lease.acquire") == _phase_threads(
        events, "lease.release"
    )
    await repository.close()


@pytest.mark.asyncio
async def test_cancelled_open_settles_worker_and_publishes_consistent_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    open_started = threading.Event()
    allow_open = threading.Event()
    _install_fake_store(monkeypatch, module, events, connection)

    def blocked_open(
        _database_path: Path,
        *,
        must_exist: bool = False,
    ) -> Any:
        del must_exist
        events.append(("store.open", threading.get_ident()))
        open_started.set()
        assert allow_open.wait(5.0)
        return connection

    monkeypatch.setattr(module, "open_profile_store", blocked_open)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    open_task = asyncio.create_task(repository.open())
    await _wait_thread_event(open_started)

    open_task.cancel()
    allow_open.set()
    with pytest.raises(asyncio.CancelledError):
        await open_task

    assert repository.state is ProfileRepositoryState.OPEN
    assert repository.generation == 1
    assert repository._connection is connection
    result = await repository._submit_operation(lambda _connection: "usable")
    assert result == ProfileStoreResult(generation=1, value="usable")
    await repository.close()


@pytest.mark.asyncio
async def test_normal_submission_checks_state_before_mismatched_expected_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository._submit_operation(
            lambda _connection: "closed",
            expected_generation=999,
        )
    _assert_safe_error(caught.value, "closed")

    await repository.open()
    with repository._state_lock:
        repository._state = ProfileRepositoryState.RESTORING
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository._submit_operation(
            lambda _connection: "restoring",
            expected_generation=999,
        )
    _assert_safe_error(caught.value, "restoring")

    with repository._state_lock:
        repository._state = ProfileRepositoryState.UNAVAILABLE
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository._submit_operation(
            lambda _connection: "unavailable",
            expected_generation=999,
        )
    _assert_safe_error(caught.value, "unavailable")

    with repository._state_lock:
        repository._state = ProfileRepositoryState.OPEN
    await repository.close()
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository._submit_operation(
            lambda _connection: "terminal",
            expected_generation=999,
        )
    _assert_safe_error(caught.value, "terminal")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("incoming_code", "expected_code"),
    [
        ("missing", "missing"),
        ("hostile-invalid-code", "operation_failed"),
    ],
)
async def test_operation_publication_recreates_structured_error_without_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    incoming_code: str,
    expected_code: str,
) -> None:
    module = _repository_module()
    secret = str(tmp_path / "secret-operation-context.sqlite3")
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    incoming = _tainted_repository_error("missing", secret)
    incoming.code = incoming_code
    _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()

    def hostile_operation(_connection: Any) -> None:
        raise incoming

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository._submit_operation(hostile_operation)

    assert caught.value is not incoming
    _assert_safe_error(caught.value, expected_code, secret)
    await repository.close()


@pytest.mark.asyncio
async def test_worker_preflight_and_publication_both_reject_stale_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()
    first_started = threading.Event()
    release_first = threading.Event()
    second_ran = False

    def first_operation(_connection: Any) -> str:
        first_started.set()
        assert release_first.wait(5.0)
        return "old-result"

    def second_operation(_connection: Any) -> str:
        nonlocal second_ran
        second_ran = True
        return "must-not-run"

    first = repository._admit_operation(first_operation)
    await _wait_thread_event(first_started)
    second = repository._admit_operation(second_operation)
    with repository._state_lock:
        repository._generation += 1
        repository._state = ProfileRepositoryState.RESTORING
    release_first.set()

    with pytest.raises(ProfileRepositoryError) as first_error:
        await repository._publish_operation(first)
    with pytest.raises(ProfileRepositoryError) as second_error:
        await repository._publish_operation(second)

    _assert_safe_error(first_error.value, "stale")
    _assert_safe_error(second_error.value, "stale")
    assert second_ran is False
    assert repository.generation == 2
    await repository.close()


@pytest.mark.asyncio
async def test_cancelled_operation_remains_registered_until_worker_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()
    operation_started = threading.Event()
    finish_operation = threading.Event()
    writes: list[str] = []

    def write_like_operation(_connection: Any) -> str:
        operation_started.set()
        assert finish_operation.wait(5.0)
        writes.append("committed")
        return "unpublished"

    operation_task = asyncio.create_task(
        repository._submit_operation(write_like_operation)
    )
    await _wait_thread_event(operation_started)
    operation_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation_task

    with repository._state_lock:
        assert len(repository._pending_futures) == 1

    close_admitted = asyncio.Event()
    original_finish_close = repository._finish_close

    async def observed_finish_close(*args: Any, **kwargs: Any) -> None:
        close_admitted.set()
        await original_finish_close(*args, **kwargs)

    monkeypatch.setattr(repository, "_finish_close", observed_finish_close)
    close_task = asyncio.create_task(repository.close())
    await close_admitted.wait()

    assert repository.state is ProfileRepositoryState.CLOSED
    assert repository.terminal is True
    assert close_task.done() is False
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository._submit_operation(lambda _connection: "too-late")
    _assert_safe_error(caught.value, "terminal")

    finish_operation.set()
    await close_task

    assert writes == ["committed"]
    with repository._state_lock:
        assert not repository._pending_futures


@pytest.mark.asyncio
async def test_caller_cancellation_wins_when_close_cancels_same_queued_future(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()
    first_started = threading.Event()
    finish_first = threading.Event()
    second_ran = False

    def first_operation(_connection: Any) -> None:
        first_started.set()
        assert finish_first.wait(5.0)

    def second_operation(_connection: Any) -> None:
        nonlocal second_ran
        second_ran = True

    repository._admit_operation(first_operation)
    await _wait_thread_event(first_started)
    second = repository._admit_operation(second_operation)
    publication_started = asyncio.Event()

    async def publish_second() -> asyncio.CancelledError:
        publication_started.set()
        try:
            await repository._publish_operation(second)
        except asyncio.CancelledError as error:
            return error
        raise AssertionError("caller cancellation was not preserved")

    operation_task = asyncio.create_task(publish_second())
    await publication_started.wait()
    second.future.add_done_callback(
        lambda _future: operation_task.cancel("caller-cancellation")
    )
    close_task = asyncio.create_task(repository.close())

    try:
        cancellation = await operation_task
        assert type(cancellation) is asyncio.CancelledError
        assert cancellation.args == ("caller-cancellation",)
        assert operation_task.cancelling() == 1
        assert second_ran is False
    finally:
        finish_first.set()
        await close_task


@pytest.mark.asyncio
async def test_worker_future_cancellation_without_caller_cancellation_is_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()
    first_started = threading.Event()
    finish_first = threading.Event()
    second_ran = False

    def first_operation(_connection: Any) -> None:
        first_started.set()
        assert finish_first.wait(5.0)

    def second_operation(_connection: Any) -> None:
        nonlocal second_ran
        second_ran = True

    repository._admit_operation(first_operation)
    await _wait_thread_event(first_started)
    second = repository._admit_operation(second_operation)
    publication_started = asyncio.Event()

    async def publish_second() -> ProfileStoreResult[None]:
        publication_started.set()
        return cast(
            ProfileStoreResult[None],
            await repository._publish_operation(second),
        )

    operation_task = asyncio.create_task(publish_second())
    await publication_started.wait()
    assert second.future.cancel()

    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await operation_task
        _assert_safe_error(caught.value, "stale")
        assert second_ran is False
    finally:
        finish_first.set()
        await repository.close()


@pytest.mark.asyncio
async def test_cancelled_operation_late_failure_is_retrieved_without_loop_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()
    operation_started = threading.Event()
    finish_operation = threading.Event()
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled_contexts: list[dict[str, Any]] = []

    def capture_unhandled(
        _loop: asyncio.AbstractEventLoop,
        context: dict[str, Any],
    ) -> None:
        unhandled_contexts.append(context)

    loop.set_exception_handler(capture_unhandled)
    try:

        def late_failure(_connection: Any) -> None:
            operation_started.set()
            assert finish_operation.wait(5.0)
            raise RuntimeError("late-worker-failure")

        operation_task = asyncio.create_task(repository._submit_operation(late_failure))
        await _wait_thread_event(operation_started)
        operation_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation_task

        close_task = asyncio.create_task(repository.close())
        finish_operation.set()
        await close_task
        del operation_task
        gc.collect()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(previous_handler)

    assert not [
        context
        for context in unhandled_contexts
        if context.get("message") == "Future exception was never retrieved"
    ]


@pytest.mark.asyncio
async def test_close_cancels_queued_work_before_draining_running_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events)
    _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()
    first_started = threading.Event()
    finish_first = threading.Event()
    second_ran = False

    def first_operation(_connection: Any) -> None:
        first_started.set()
        assert finish_first.wait(5.0)
        events.append(("first.finished", threading.get_ident()))

    def second_operation(_connection: Any) -> None:
        nonlocal second_ran
        second_ran = True

    first = repository._admit_operation(first_operation)
    await _wait_thread_event(first_started)
    second = repository._admit_operation(second_operation)
    close_admitted = asyncio.Event()
    original_finish_close = repository._finish_close

    async def observed_finish_close(*args: Any, **kwargs: Any) -> None:
        close_admitted.set()
        await original_finish_close(*args, **kwargs)

    monkeypatch.setattr(repository, "_finish_close", observed_finish_close)
    close_task = asyncio.create_task(repository.close())
    await close_admitted.wait()

    assert repository.state is ProfileRepositoryState.CLOSED
    assert repository.generation == 2
    assert repository.terminal is True
    assert second.future.cancelled()
    assert close_task.done() is False

    finish_first.set()
    await close_task

    assert first.future.done()
    assert second_ran is False
    phases = [phase for phase, _thread_id in events]
    assert phases.index("first.finished") < phases.index("connection.close")
    assert phases.index("connection.close") < phases.index("lease.release")


@pytest.mark.asyncio
async def test_close_failure_retains_real_shared_lease_and_connection_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    _bypass_startup_ownership_for_open_cleanup_test(monkeypatch, module)
    database_path = tmp_path / "profiles.sqlite3"
    secret = str(tmp_path / "secret-close-error")
    events: list[tuple[str, int]] = []
    connection = _RecordingConnection(events, close_error=RuntimeError(secret))
    executors: list[_RecordingExecutor] = []
    leases: list[_ObservedProfileStoreLease] = []

    def executor_factory(max_workers: int) -> _RecordingExecutor:
        executor = _RecordingExecutor(max_workers, events)
        executors.append(executor)
        return executor

    def lease_factory(
        path: Path,
        mode: ProfileStoreLockMode,
        **_kwargs: object,
    ) -> _ObservedProfileStoreLease:
        lease = _ObservedProfileStoreLease(
            path,
            mode,
            events,
            f"lease-{len(leases) + 1}",
        )
        leases.append(lease)
        return lease

    monkeypatch.setattr(module, "ThreadPoolExecutor", executor_factory)
    monkeypatch.setattr(module, "ProfileStoreLease", lease_factory)
    monkeypatch.setattr(
        module,
        "open_profile_store",
        lambda _path, **_kwargs: connection,
    )
    repository = module.TTSProfileRepository(database_path)
    await repository.open()

    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.close()

        _assert_safe_error(caught.value, "operation_failed", secret)
        assert repository.state is ProfileRepositoryState.CLOSED
        assert repository.terminal is True
        assert repository._connection is connection
        assert repository._lease is leases[0]
        assert connection.closed is False
        assert leases[0].acquired is True
        assert not _phase_threads(events, "lease-1.release")
        assert executors[0].shutdown_calls == 1
        await _assert_exclusive_lease_blocked(database_path)
        assert await repository.close() == ProfileStoreResult(
            generation=2,
            value=None,
        )
        assert executors[0].shutdown_calls == 1
    finally:
        connection.close_error = None
        if not connection.closed:
            connection.close()
        if leases and leases[0].acquired:
            await asyncio.to_thread(leases[0].release)
        repository._connection = None
        repository._lease = None

    assert await asyncio.to_thread(_try_exclusive_lease, database_path) is None


@pytest.mark.asyncio
async def test_close_preserves_connection_cleanup_control_and_retains_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    events: list[tuple[str, int]] = []
    signal = _ControlFlow()
    connection = _RecordingConnection(events, close_error=signal)
    executors: list[_RecordingExecutor] = []

    def executor_factory(max_workers: int) -> _RecordingExecutor:
        executor = _RecordingExecutor(max_workers, events)
        executors.append(executor)
        return executor

    monkeypatch.setattr(module, "ThreadPoolExecutor", executor_factory)
    leases = _install_fake_store(monkeypatch, module, events, connection)
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    await repository.open()

    try:
        with pytest.raises(_ControlFlow) as caught:
            await repository.close()

        assert caught.value is signal
        assert repository._connection is connection
        assert repository._lease is leases[0]
        assert connection.closed is False
        assert leases[0].acquired is True
        assert not _phase_threads(events, "lease.release")
        assert executors[0].shutdown_calls == 1
        assert repository.state is ProfileRepositoryState.CLOSED
        assert repository.terminal is True
    finally:
        connection.close_error = None
        if not connection.closed:
            connection.close()
        if leases[0].acquired:
            leases[0].release()
        repository._connection = None
        repository._lease = None


@pytest.mark.asyncio
async def test_online_backup_serializes_with_write_and_publishes_valid_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    destination = tmp_path / "backup.sqlite3"
    created_at = datetime(2026, 7, 27, 10, 11, 12, 123456, tzinfo=UTC)
    repository = _repository_module().TTSProfileRepository(
        database_path,
        _clock=lambda: created_at,
    )
    await repository.open()
    await repository.create_profile(
        _draft("Before"),
        UUID("00000000-0000-4000-8000-000000000001"),
    )
    commit_started = threading.Event()
    finish_commit = threading.Event()
    real_commit = repository._commit_transaction

    def controlled_commit(connection: sqlite3.Connection) -> None:
        commit_started.set()
        assert finish_commit.wait(5.0)
        real_commit(connection)

    monkeypatch.setattr(repository, "_commit_transaction", controlled_commit)
    write = asyncio.create_task(
        repository.create_profile(
            _draft("After"),
            UUID("00000000-0000-4000-8000-000000000002"),
        )
    )
    await _wait_thread_event(commit_started)
    backup = asyncio.create_task(repository.backup_to(destination))
    await asyncio.sleep(0)

    try:
        assert backup.done() is False
        finish_commit.set()
        write_result, backup_result = await asyncio.gather(write, backup)

        validate_profile_candidate(destination)
        snapshot = open_profile_store(destination, must_exist=True)
        try:
            profile_count = snapshot.execute(
                "SELECT COUNT(*) FROM tts_generation_profiles"
            ).fetchone()[0]
        finally:
            snapshot.close()
        assert write_result.generation == 1
        assert backup_result == ProfileStoreResult(
            generation=1,
            value=ProfileBackupReceipt(
                created_at=created_at,
                byte_count=destination.stat().st_size,
            ),
        )
        assert profile_count == 2
    finally:
        finish_commit.set()
        await repository.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX snapshot mode contract")
def test_standalone_snapshot_validation_is_read_only_and_side_effect_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    snapshot = tmp_path / "snapshot.sqlite3"
    connection = open_profile_store(snapshot)
    connection.close()
    repository = module.TTSProfileRepository(tmp_path / "active.sqlite3")
    real_connect = module.connect_private_sqlite
    observed_options: list[dict[str, object]] = []

    def observe_connect(
        owner_id: str,
        database: str | os.PathLike[str],
        **options: object,
    ) -> sqlite3.Connection:
        if owner_id == "tts.profile_snapshot":
            observed_options.append(dict(options))
        return real_connect(owner_id, database, **options)

    monkeypatch.setattr(module, "connect_private_sqlite", observe_connect)

    repository._worker_validate_standalone_snapshot(snapshot)

    assert observed_options == [
        {
            "must_exist": True,
            "read_only": True,
            "immutable": True,
            "isolation_level": None,
        }
    ]
    assert not Path(f"{snapshot}-journal").exists()
    assert not Path(f"{snapshot}-wal").exists()
    assert not Path(f"{snapshot}-shm").exists()


@pytest.mark.asyncio
async def test_backup_and_restore_path_filesystem_checks_run_only_on_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    destination = tmp_path / "backup.sqlite3"
    loop_thread = threading.get_ident()
    path_events: list[tuple[str, int]] = []
    real_canonical = module._canonical_database_path
    real_backup_destination = module._validate_backup_destination
    real_restore_candidate = module._validate_restore_candidate_path

    def traced_canonical(path: Path, failure_code: str) -> Path:
        path_events.append(("canonical", threading.get_ident()))
        return real_canonical(path, failure_code)

    def traced_backup_destination(
        path: object,
        active_path: Path,
    ) -> Any:
        path_events.append(("backup", threading.get_ident()))
        return real_backup_destination(path, active_path)

    def traced_restore_candidate(
        path: object,
        active_path: Path,
    ) -> Any:
        path_events.append(("restore", threading.get_ident()))
        return real_restore_candidate(path, active_path)

    monkeypatch.setattr(module, "_canonical_database_path", traced_canonical)
    monkeypatch.setattr(
        module,
        "_validate_backup_destination",
        traced_backup_destination,
    )
    monkeypatch.setattr(
        module,
        "_validate_restore_candidate_path",
        traced_restore_candidate,
    )
    repository = module.TTSProfileRepository(database_path)
    await repository.open()
    try:
        path_events.clear()
        await repository.backup_to(destination)

        assert path_events
        assert any(name == "backup" for name, _thread_id in path_events)
        assert len({thread_id for _name, thread_id in path_events}) == 1
        assert loop_thread not in {thread_id for _name, thread_id in path_events}

        path_events.clear()
        await repository.restore_from(destination)

        assert path_events
        assert any(name == "restore" for name, _thread_id in path_events)
        assert len({thread_id for _name, thread_id in path_events}) == 1
        assert loop_thread not in {thread_id for _name, thread_id in path_events}
    finally:
        await repository.close()


def test_backup_destination_applies_central_path_safety_validation(
    tmp_path: Path,
) -> None:
    module = _repository_module()
    destination = tmp_path / "backup;unsafe.sqlite3"

    with pytest.raises(ProfileRepositoryError) as caught:
        module._validate_backup_destination(
            destination,
            tmp_path / "profiles.sqlite3",
        )

    _assert_safe_error(caught.value, "backup_failed", str(destination))


def test_restore_candidate_applies_central_path_safety_validation(
    tmp_path: Path,
) -> None:
    module = _repository_module()
    candidate = tmp_path / "candidate;unsafe.sqlite3"
    candidate.write_bytes(b"candidate")

    with pytest.raises(ProfileRepositoryError) as caught:
        module._validate_restore_candidate_path(
            candidate,
            tmp_path / "profiles.sqlite3",
        )

    _assert_safe_error(caught.value, "restore_failed", str(candidate))


def test_restore_online_backup_checks_deadline_between_page_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    monotonic_values = iter((0.0, 2.0))
    observed_owner: str | None = None

    def checked_backup(
        owner_id: str,
        _source: sqlite3.Connection,
        _destination: sqlite3.Connection,
        *,
        progress_guard: Callable[[], None] | None = None,
    ) -> None:
        nonlocal observed_owner
        observed_owner = owner_id
        assert progress_guard is not None
        progress_guard()

    monkeypatch.setattr(module, "_monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        module,
        "backup_open_connections_to_private",
        checked_backup,
    )

    with pytest.raises(ProfileRepositoryError) as caught:
        repository._worker_online_backup(
            cast(sqlite3.Connection, object()),
            cast(sqlite3.Connection, object()),
            deadline=1.0,
        )

    _assert_safe_error(caught.value, "restore_failed")
    assert observed_owner == "tts.profile_backup"


def test_restore_integrity_check_interrupts_and_clears_progress_handler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    monotonic_values = iter((0.0, 2.0))
    progress_handlers: list[tuple[object, int]] = []

    class Connection:
        progress_handler: Callable[[], int] | None = None

        def set_progress_handler(
            self,
            handler: Callable[[], int] | None,
            opcode_interval: int,
        ) -> None:
            self.progress_handler = handler
            progress_handlers.append((handler, opcode_interval))

        def execute(self, _statement: str) -> tuple[tuple[str], ...]:
            assert self.progress_handler is not None
            if self.progress_handler() != 0:
                raise sqlite3.OperationalError
            return (("ok",),)

    monkeypatch.setattr(module, "_monotonic", lambda: next(monotonic_values))
    connection = Connection()

    with pytest.raises(ProfileRepositoryError) as caught:
        repository._worker_require_full_integrity(
            cast(sqlite3.Connection, connection),
            deadline=1.0,
        )

    _assert_safe_error(caught.value, "restore_failed")
    assert callable(progress_handlers[0][0])
    assert progress_handlers[0][1] > 0
    assert progress_handlers[-1] == (None, 0)


def test_restore_checkpoint_caps_busy_wait_to_remaining_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    repository = module.TTSProfileRepository(tmp_path / "profiles.sqlite3")
    now = 0.0
    observed_checkpoint_timeout: int | None = None
    timeout_updates: list[int] = []
    progress_handlers: list[tuple[object, int]] = []
    events: list[tuple[str, int]] = []
    lease = _RecordingLease(events)
    lease.acquired = True

    class Cursor:
        def __init__(self, row: tuple[int, ...] | None = None) -> None:
            self.row = row

        def fetchone(self) -> tuple[int, ...] | None:
            return self.row

    class Connection:
        busy_timeout = 5_000

        def execute(self, statement: str) -> Cursor:
            nonlocal now, observed_checkpoint_timeout
            normalized = statement.strip()
            if normalized == "PRAGMA busy_timeout":
                return Cursor((self.busy_timeout,))
            if normalized.startswith("PRAGMA busy_timeout = "):
                self.busy_timeout = int(normalized.rsplit(" ", 1)[1])
                timeout_updates.append(self.busy_timeout)
                return Cursor()
            if normalized == "PRAGMA wal_checkpoint(TRUNCATE)":
                observed_checkpoint_timeout = self.busy_timeout
                now = 0.3
                return Cursor((0, 0, 0))
            raise AssertionError

        def set_progress_handler(
            self,
            handler: Callable[[], int] | None,
            opcode_interval: int,
        ) -> None:
            progress_handlers.append((handler, opcode_interval))

    monkeypatch.setattr(module, "_monotonic", lambda: now)
    connection = Connection()
    repository._connection = cast(sqlite3.Connection, connection)
    repository._lease = cast(ProfileStoreLease, lease)

    with pytest.raises(ProfileRepositoryError) as caught:
        repository._worker_close_for_restore(0.25)

    _assert_safe_error(caught.value, "restore_failed")
    assert observed_checkpoint_timeout is not None
    assert observed_checkpoint_timeout <= 250
    assert timeout_updates[-1] == 5_000
    assert callable(progress_handlers[0][0])
    assert progress_handlers[-1] == (None, 0)
    assert repository._connection is connection
    assert repository._lease is lease
    assert lease.acquired is True


def test_restore_sidecar_checks_deadline_between_filesystem_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    wal_path = database_path.with_name(f"{database_path.name}-wal")
    wal_path.write_bytes(b"must remain")
    repository = module.TTSProfileRepository(database_path)
    repository._active_database_path = database_path
    now = 0.0
    real_lstat = Path.lstat

    def expiring_lstat(path: Path) -> os.stat_result:
        nonlocal now
        try:
            return real_lstat(path)
        finally:
            if path.name.endswith("-journal"):
                now = 2.0

    monkeypatch.setattr(module, "_monotonic", lambda: now)
    monkeypatch.setattr(Path, "lstat", expiring_lstat)

    with pytest.raises(ProfileRepositoryError) as caught:
        repository._worker_remove_live_sidecars(deadline=1.0)

    _assert_safe_error(caught.value, "restore_failed")
    assert wal_path.read_bytes() == b"must remain"


def test_restore_sidecar_swap_after_admission_preserves_foreign_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    wal_path = Path(f"{database_path}-wal")
    detached = tmp_path / "detached-wal"
    foreign = tmp_path / "foreign-wal"
    wal_path.write_bytes(b"admitted wal bytes")
    wal_path.chmod(0o600)
    expected = wal_path.stat()
    foreign_bytes = b"foreign wal bytes must remain"
    foreign.write_bytes(foreign_bytes)
    foreign.chmod(0o600)
    repository = module.TTSProfileRepository(database_path)
    repository._active_database_path = database_path
    repository._restore_sidecar_identities = {"-wal": expected}
    repository._restore_parent_authority = module.ParentAuthority(tmp_path.stat())
    real_lstat = Path.lstat
    swapped = False

    def swap_after_lstat(path: Path) -> os.stat_result:
        nonlocal swapped
        observed = real_lstat(path)
        if path == wal_path and not swapped:
            os.replace(wal_path, detached)
            os.replace(foreign, wal_path)
            swapped = True
        return observed

    monkeypatch.setattr(Path, "lstat", swap_after_lstat)
    with pytest.raises(ProfileRepositoryError, match="restore_failed"):
        repository._worker_remove_live_sidecars(deadline=module._read_monotonic() + 1.0)

    assert swapped is True
    assert wal_path.read_bytes() == foreign_bytes
    assert detached.read_bytes() == b"admitted wal bytes"


@pytest.mark.asyncio
async def test_online_backup_rejects_live_lock_and_sidecar_targets(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    repository = _repository(database_path)
    await repository.open()
    migration_backup = _repository_module()._v2_migration_backup_path(database_path)
    migration_backup.write_bytes(b"retained-v2-downgrade-snapshot")
    reserved = (
        database_path,
        database_path.with_name(f"{database_path.name}.lock"),
        migration_backup,
        *(
            database_path.with_name(f"{database_path.name}{suffix}")
            for suffix in (
                "-wal",
                "-shm",
                "-journal",
            )
        ),
    )

    try:
        before = {path: path.read_bytes() for path in reserved if path.is_file()}
        for destination in reserved:
            with pytest.raises(ProfileRepositoryError) as caught:
                await repository.backup_to(destination)
            _assert_safe_error(caught.value, "backup_failed", str(destination))
        assert {path: path.read_bytes() for path in before} == before
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_online_backup_rejects_symlink_and_hardlink_aliases_to_live_store(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    symlink_path = tmp_path / "symlink.sqlite3"
    hardlink_path = tmp_path / "hardlink.sqlite3"
    retained_alias = tmp_path / "retained-alias.sqlite3"
    repository = _repository(database_path)
    await repository.open()
    migration_backup = _repository_module()._v2_migration_backup_path(database_path)
    migration_backup.write_bytes(b"retained-v2-downgrade-snapshot")
    symlink_path.symlink_to(database_path)
    os.link(migration_backup, retained_alias)

    try:
        live_bytes = database_path.read_bytes()
        for destination in (symlink_path, retained_alias):
            with pytest.raises(ProfileRepositoryError) as caught:
                await repository.backup_to(destination)
            _assert_safe_error(
                caught.value,
                "backup_failed",
                str(database_path),
                str(destination),
            )
        assert database_path.read_bytes() == live_bytes
        assert symlink_path.is_symlink()
        assert retained_alias.stat().st_ino == migration_backup.stat().st_ino
        assert migration_backup.read_bytes() == b"retained-v2-downgrade-snapshot"
        os.link(database_path, hardlink_path)
        with pytest.raises(ProfileRepositoryError, match="operation_failed"):
            await repository.backup_to(hardlink_path)
        assert repository.state is ProfileRepositoryState.UNAVAILABLE
        assert hardlink_path.stat().st_ino == database_path.stat().st_ino
    finally:
        hardlink_path.unlink(missing_ok=True)
        await repository.close()


@pytest.mark.asyncio
async def test_online_backup_replace_failure_preserves_destination_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    destination = tmp_path / "existing-backup.sqlite3"
    destination.write_bytes(b"trusted-existing-destination")
    secret = str(tmp_path / "secret-backup-replace")
    repository = _repository(database_path)
    await repository.open()
    before = destination.read_bytes()

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError(secret)

    monkeypatch.setattr(module.os, "replace", fail_replace)
    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.backup_to(destination)

        _assert_safe_error(
            caught.value,
            "backup_failed",
            secret,
            str(database_path),
            str(destination),
        )
        assert destination.read_bytes() == before
        assert not tuple(tmp_path.glob(f".{destination.name}.*.backup"))
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_restore_replaces_store_with_validated_candidate_and_safe_receipt(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    restored_at = datetime(2026, 7, 27, 12, 13, 14, 654321, tzinfo=UTC)
    await _create_profile_store(candidate, "Candidate one", "Candidate two")
    repository = _repository_module().TTSProfileRepository(
        database_path,
        _clock=lambda: restored_at,
    )
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )

    try:
        result = await repository.restore_from(candidate)

        assert result == ProfileStoreResult(
            generation=2,
            value=ProfileRestoreReceipt(
                restored_at=restored_at,
                profile_count=2,
                assignment_count=0,
            ),
        )
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository.generation == 2
        page = await repository.list_profiles()
        assert [profile.display_name for profile in page.value.profiles] == [
            "Candidate one",
            "Candidate two",
        ]
        recoveries = tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
        assert len(recoveries) == 1
        validate_profile_candidate(recoveries[0])
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_pre_restore_loaded_mutations_reject_same_identity_replacement_before_enqueue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    profile_id = UUID("00000000-0000-4000-8000-000000000001")
    await _create_profile_store(candidate, "Replacement")
    repository = _repository(database_path)
    await repository.open()
    loaded = await repository.create_profile(
        _draft("Original"),
        profile_id,
    )
    real_submit_operation = repository._submit_operation
    all_paused = asyncio.Event()
    resume_admission = asyncio.Event()
    paused_count = 0
    mutation_tasks: list[asyncio.Task[object]] = []

    async def pause_mutation_admission(
        operation: Callable[[sqlite3.Connection], object],
        *,
        expected_generation: int | None = None,
    ) -> ProfileStoreResult[object]:
        nonlocal paused_count
        if expected_generation == loaded.generation:
            paused_count += 1
            if paused_count == 3:
                all_paused.set()
            await resume_admission.wait()
        return await real_submit_operation(
            operation,
            expected_generation=expected_generation,
        )

    monkeypatch.setattr(
        repository,
        "_submit_operation",
        pause_mutation_admission,
    )

    try:
        mutation_tasks = [
            asyncio.create_task(
                repository.update_profile(
                    profile_id,
                    loaded.value.revision,
                    _draft("Updated"),
                    expected_generation=loaded.generation,
                )
            ),
            asyncio.create_task(
                repository.delete_profile(
                    profile_id,
                    expected_generation=loaded.generation,
                )
            ),
            asyncio.create_task(
                repository.create_profile(
                    _draft("Duplicate"),
                    expected_generation=loaded.generation,
                )
            ),
        ]
        await asyncio.wait_for(all_paused.wait(), timeout=1.0)

        restored = await repository.restore_from(candidate)
        assert restored.generation == 2
        executor = repository._executor
        assert executor is not None
        real_executor_submit = executor.submit
        stale_worker_submissions = 0
        reject_stale_submission = True

        def forbid_stale_worker_submission(
            function: Callable[..., object],
            /,
            *args: object,
            **kwargs: object,
        ) -> Future[object]:
            nonlocal stale_worker_submissions
            if reject_stale_submission and function == repository._worker_operation:
                stale_worker_submissions += 1
                raise AssertionError("stale mutation reached worker submission")
            return cast(Future[object], real_executor_submit(function, *args, **kwargs))

        monkeypatch.setattr(
            executor,
            "submit",
            forbid_stale_worker_submission,
        )
        resume_admission.set()

        for mutation in mutation_tasks:
            with pytest.raises(ProfileRepositoryError) as caught:
                await mutation
            _assert_safe_error(caught.value, "stale")

        assert stale_worker_submissions == 0
        reject_stale_submission = False
        replacement = await repository.get_profile(profile_id)
        assert replacement.generation == 2
        assert replacement.value.display_name == "Replacement"
        assert replacement.value.revision == loaded.value.revision == 1
        page = await repository.list_profiles()
        assert page.value.total == 1
        assert page.value.profiles == (replacement.value,)
    finally:
        resume_admission.set()
        if mutation_tasks:
            await asyncio.gather(*mutation_tasks, return_exceptions=True)
        await repository.close()


@pytest.mark.asyncio
async def test_restore_online_backup_includes_committed_candidate_wal_rows(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    candidate_repository = _repository(candidate)
    await candidate_repository.open()
    await candidate_repository.create_profile(
        _draft("Committed in WAL"),
        UUID("00000000-0000-4000-8000-000000000077"),
    )
    assert candidate.with_name(f"{candidate.name}-wal").is_file()
    repository = _repository(database_path)
    await repository.open()

    try:
        result = await repository.restore_from(candidate)

        assert result.value.profile_count == 1
        page = await repository.list_profiles()
        assert [profile.display_name for profile in page.value.profiles] == [
            "Committed in WAL"
        ]
    finally:
        await repository.close()
        await candidate_repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "timeout",
    [None, True, 0, -1, float("inf"), float("-inf"), float("nan"), "5"],
)
async def test_restore_rejects_invalid_timeout_before_lifecycle_or_file_mutation(
    tmp_path: Path,
    timeout: object,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    before = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }

    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.restore_from(
                candidate,
                timeout_seconds=timeout,  # type: ignore[arg-type]
            )

        _assert_safe_error(caught.value, "restore_failed", str(candidate))
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository.generation == 1
        assert {
            path.name: path.read_bytes()
            for path in tmp_path.iterdir()
            if path.is_file()
        } == before
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_restore_rejects_live_reserved_and_alias_candidates_on_worker(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    repository = _repository(database_path)
    await repository.open()
    hardlink = tmp_path / "hardlink-candidate.sqlite3"
    symlink = tmp_path / "symlink-candidate.sqlite3"
    symlink.symlink_to(database_path)
    candidates = (
        database_path,
        database_path.with_name(f"{database_path.name}.lock"),
        *(
            database_path.with_name(f"{database_path.name}{suffix}")
            for suffix in (
                "-wal",
                "-shm",
                "-journal",
            )
        ),
        symlink,
    )

    try:
        for candidate in candidates:
            with pytest.raises(ProfileRepositoryError) as caught:
                await repository.restore_from(candidate)
            _assert_safe_error(
                caught.value,
                "restore_failed",
                str(candidate),
                str(database_path),
            )
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository.generation == 1 + len(candidates)
        assert not tuple(tmp_path.glob("*.restore-stage.sqlite3"))
        assert not tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
        os.link(database_path, hardlink)
        with pytest.raises(ProfileRepositoryError, match="restore_failed"):
            await repository.restore_from(hardlink)
        assert repository.state is ProfileRepositoryState.UNAVAILABLE
    finally:
        hardlink.unlink(missing_ok=True)
        await repository.close()


@pytest.mark.asyncio
async def test_corrupt_restore_candidate_preserves_live_store_and_reopens_advanced(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate-secret.sqlite3"
    candidate.write_bytes(b"not a sqlite database")
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )
    await repository._submit_operation(
        lambda connection: connection.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        ).fetchall()
    )
    before_bytes = database_path.read_bytes()

    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.restore_from(candidate)

        _assert_safe_error(
            caught.value,
            "schema_corrupt",
            str(candidate),
            str(database_path),
        )
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository.generation == 2
        assert database_path.read_bytes() == before_bytes
        assert not tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
        assert not tuple(tmp_path.glob("*.restore-stage.sqlite3"))
        restored = await repository.list_profiles()
        assert [profile.display_name for profile in restored.value.profiles] == [
            "Original"
        ]
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_restore_quiescence_timeout_advances_without_file_or_lease_mutation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    operation_started = threading.Event()
    release_operation = threading.Event()

    def blocked_operation(_connection: sqlite3.Connection) -> str:
        operation_started.set()
        assert release_operation.wait(5.0)
        return "old-result"

    old = repository._admit_operation(blocked_operation)
    await _wait_thread_event(operation_started)
    connection = repository._connection
    lease = repository._lease
    before = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }

    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.restore_from(candidate, timeout_seconds=0.01)

        _assert_safe_error(caught.value, "restore_failed")
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository.generation == 2
        assert repository._connection is connection
        assert repository._lease is lease
        assert {
            path.name: path.read_bytes()
            for path in tmp_path.iterdir()
            if path.is_file()
        } == before
        assert not tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
        assert not tuple(tmp_path.glob("*.restore-stage.sqlite3"))
    finally:
        release_operation.set()

    with pytest.raises(ProfileRepositoryError) as stale:
        await repository._publish_operation(old)
    _assert_safe_error(stale.value, "stale")
    assert (await repository.list_profiles()).generation == 2
    await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("variant", "expected_code"),
    [
        ("partial", "schema_corrupt"),
        ("unsupported", "schema_unsupported"),
        ("domain", "corrupt_data"),
        ("foreign_key", "schema_corrupt"),
    ],
)
async def test_invalid_restore_candidates_preserve_original_and_rebind_open(
    tmp_path: Path,
    variant: str,
    expected_code: str,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / f"{variant}-candidate.sqlite3"
    if variant == "partial":
        partial = sqlite3.connect(candidate)
        partial.execute("CREATE TABLE unexpected(value TEXT)")
        partial.execute(f"PRAGMA user_version = {CURRENT_PROFILE_SCHEMA_VERSION}")
        partial.close()
    else:
        await _create_profile_store(candidate, "Candidate")
        hostile = sqlite3.connect(candidate)
        if variant == "unsupported":
            hostile.execute(
                f"PRAGMA user_version = {CURRENT_PROFILE_SCHEMA_VERSION + 1}"
            )
        elif variant == "domain":
            hostile.execute("UPDATE tts_generation_profiles SET revision = 0")
        else:
            timestamp = "2026-07-27T12:00:00.000000Z"
            hostile.execute(
                """
                INSERT INTO character_tts_assignments (
                    source, authority_id, character_id, profile_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "local",
                    "authority",
                    "character",
                    "00000000-0000-4000-8000-999999999999",
                    timestamp,
                    timestamp,
                ),
            )
        hostile.commit()
        hostile.close()

    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )
    await repository._submit_operation(
        lambda connection: connection.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        ).fetchall()
    )
    before_bytes = database_path.read_bytes()

    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.restore_from(candidate)

        _assert_safe_error(
            caught.value,
            expected_code,
            str(candidate),
            str(database_path),
        )
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository.generation == 2
        assert database_path.read_bytes() == before_bytes
        assert not tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
        page = await repository.list_profiles()
        assert [profile.display_name for profile in page.value.profiles] == ["Original"]
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_restore_cancels_queued_old_write_and_suppresses_running_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    commit_started = threading.Event()
    release_commit = threading.Event()
    second_body_ran = False
    real_commit = repository._commit_transaction
    real_create = repository._worker_create_profile
    commit_calls = 0

    def controlled_commit(connection: sqlite3.Connection) -> None:
        nonlocal commit_calls
        commit_calls += 1
        if commit_calls == 1:
            commit_started.set()
            assert release_commit.wait(5.0)
        real_commit(connection)

    def observed_create(
        connection: sqlite3.Connection,
        draft: TTSProfileDraft,
        profile_id: UUID | None,
    ) -> Any:
        nonlocal second_body_ran
        if draft.display_name == "Queued old":
            second_body_ran = True
        return real_create(connection, draft, profile_id)

    monkeypatch.setattr(repository, "_commit_transaction", controlled_commit)
    monkeypatch.setattr(repository, "_worker_create_profile", observed_create)
    running = asyncio.create_task(
        repository.create_profile(
            _draft("Running old"),
            UUID("00000000-0000-4000-8000-000000000010"),
        )
    )
    await _wait_thread_event(commit_started)
    queued = asyncio.create_task(
        repository.create_profile(
            _draft("Queued old"),
            UUID("00000000-0000-4000-8000-000000000011"),
        )
    )
    await asyncio.sleep(0)
    restore = asyncio.create_task(repository.restore_from(candidate))
    for _ in range(100):
        if repository.state is ProfileRepositoryState.RESTORING:
            break
        await asyncio.sleep(0)

    assert repository.state is ProfileRepositoryState.RESTORING
    assert repository.generation == 2
    assert restore.done() is False
    release_commit.set()
    running_outcome, queued_outcome, restore_outcome = await asyncio.gather(
        running,
        queued,
        restore,
        return_exceptions=True,
    )

    assert isinstance(running_outcome, ProfileRepositoryError)
    _assert_safe_error(running_outcome, "stale")
    assert isinstance(queued_outcome, ProfileRepositoryError)
    _assert_safe_error(queued_outcome, "stale")
    assert second_body_ran is False
    assert restore_outcome.generation == 2
    assert repository.state is ProfileRepositoryState.OPEN
    page = await repository.list_profiles()
    assert [profile.display_name for profile in page.value.profiles] == ["Candidate"]
    await repository.close()


@pytest.mark.asyncio
async def test_recovery_backup_failure_cleans_stage_and_rebinds_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    secret = str(tmp_path / "secret-recovery-failure")
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )

    def fail_recovery(_restored_at: datetime, _deadline: float) -> Path:
        raise RuntimeError(secret)

    monkeypatch.setattr(repository, "_worker_create_recovery_backup", fail_recovery)
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.restore_from(candidate)

    _assert_safe_error(caught.value, "restore_failed", secret, str(database_path))
    assert repository.state is ProfileRepositoryState.OPEN
    assert repository.generation == 2
    assert not tuple(tmp_path.glob("*.restore-stage.sqlite3"))
    assert not tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
    page = await repository.list_profiles()
    assert [profile.display_name for profile in page.value.profiles] == ["Original"]
    await repository.close()


@pytest.mark.asyncio
async def test_post_replace_shared_reacquire_failure_is_unavailable_with_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    real_lease_type = module.ProfileStoreLease
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )

    def lease_factory(
        path: Path,
        mode: ProfileStoreLockMode,
        **kwargs: object,
    ) -> Any:
        if mode is ProfileStoreLockMode.SHARED:
            return _RecordingLease(
                [],
                acquire_error=ProfileRepositoryError("lock_timeout"),
            )
        return real_lease_type(path, mode, **kwargs)

    monkeypatch.setattr(module, "ProfileStoreLease", lease_factory)
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.restore_from(candidate)

    _assert_safe_error(caught.value, "restore_failed", str(database_path))
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository._connection is None
    assert repository._lease is None
    assert len(tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))) == 1
    authoritative = open_profile_store(database_path, must_exist=True)
    try:
        names = [
            row[0]
            for row in authoritative.execute(
                "SELECT display_name FROM tts_generation_profiles"
            )
        ]
    finally:
        authoritative.close()
    assert names == ["Candidate"]
    await repository.close()


@pytest.mark.asyncio
async def test_post_replace_long_lived_reopen_failure_is_unavailable_without_blank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    secret = str(tmp_path / "secret-reopen-failure")
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )
    real_open = module.open_profile_store
    real_exact_open = module.open_exact_current_profile_store
    real_publish = module.publish_profile_migration
    publication_complete = False

    def tracked_publish(**kwargs: object) -> None:
        nonlocal publication_complete
        real_publish(**kwargs)
        publication_complete = True

    def injected_open(
        path: Path,
        *,
        must_exist: bool = False,
        check_deadline: Callable[[], None] | None = None,
    ) -> sqlite3.Connection:
        if must_exist and publication_complete:
            raise RuntimeError(secret)
        return real_open(
            path,
            must_exist=must_exist,
            check_deadline=check_deadline,
        )

    monkeypatch.setattr(module, "publish_profile_migration", tracked_publish)
    monkeypatch.setattr(module, "open_profile_store", injected_open)
    monkeypatch.setattr(
        module,
        "open_exact_current_profile_store",
        lambda path: (
            (_ for _ in ()).throw(RuntimeError(secret))
            if publication_complete
            else real_exact_open(path)
        ),
    )
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.restore_from(candidate)

    _assert_safe_error(caught.value, "restore_failed", secret, str(database_path))
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository._connection is None
    assert repository._lease is None
    assert database_path.stat().st_size > 0
    assert len(tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))) == 1
    authoritative = real_open(database_path, must_exist=True)
    try:
        assert [
            row[0]
            for row in authoritative.execute(
                "SELECT display_name FROM tts_generation_profiles"
            )
        ] == ["Candidate"]
    finally:
        authoritative.close()
    await repository.close()


@pytest.mark.asyncio
async def test_restore_handoff_opens_newer_valid_store_that_wins_before_shared_rebind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    real_lease_type = module.ProfileStoreLease
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    newer = tmp_path / "newer.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    await _create_profile_store(newer, "Newer one", "Newer two")
    repository = _repository(database_path)
    await repository.open()
    replaced_during_handoff = False

    class HandoffLease(real_lease_type):
        def release(self) -> None:
            nonlocal replaced_during_handoff
            super().release()
            if (
                self.mode is ProfileStoreLockMode.EXCLUSIVE
                and not replaced_during_handoff
            ):
                os.replace(newer, database_path)
                replaced_during_handoff = True

    monkeypatch.setattr(module, "ProfileStoreLease", HandoffLease)

    try:
        result = await repository.restore_from(candidate)

        assert replaced_during_handoff is True
        assert result.value.profile_count == 2
        assert result.value.assignment_count == 0
        assert repository.state is ProfileRepositoryState.OPEN
        page = await repository.list_profiles()
        assert [profile.display_name for profile in page.value.profiles] == [
            "Newer one",
            "Newer two",
        ]
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_restore_handoff_rejects_domain_invalid_authoritative_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    real_lease_type = module.ProfileStoreLease
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    invalid_winner = tmp_path / "invalid-winner.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    await _create_profile_store(invalid_winner, "Invalid winner")
    invalid = sqlite3.connect(invalid_winner, isolation_level=None)
    invalid.execute("UPDATE tts_generation_profiles SET revision = 0")
    invalid.close()
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )
    replaced_during_handoff = False

    class InvalidHandoffLease(real_lease_type):
        def release(self) -> None:
            nonlocal replaced_during_handoff
            super().release()
            if (
                self.mode is ProfileStoreLockMode.EXCLUSIVE
                and not replaced_during_handoff
            ):
                os.replace(invalid_winner, database_path)
                replaced_during_handoff = True

    monkeypatch.setattr(module, "ProfileStoreLease", InvalidHandoffLease)

    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.restore_from(candidate)

        _assert_safe_error(caught.value, "restore_failed", str(database_path))
        assert replaced_during_handoff is True
        assert repository.state is ProfileRepositoryState.UNAVAILABLE
        assert repository.generation == 2
        assert repository._connection is None
        assert repository._lease is None
        assert repository._active_database_path is None
        recoveries = tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
        assert len(recoveries) == 1
        validate_profile_candidate(recoveries[0])
        live = sqlite3.connect(database_path)
        try:
            assert (
                live.execute("SELECT revision FROM tts_generation_profiles").fetchone()[
                    0
                ]
                == 0
            )
        finally:
            live.close()

        with pytest.raises(ProfileRepositoryError) as retry_error:
            await repository.open()
        _assert_safe_error(retry_error.value, "corrupt_data", str(database_path))
        assert repository.state is ProfileRepositoryState.UNAVAILABLE
        assert repository.generation == 3
        assert repository._connection is None
        assert repository._lease is None
        assert repository._store_established is True

        for suffix in ("-wal", "-shm", "-journal"):
            try:
                database_path.with_name(f"{database_path.name}{suffix}").unlink()
            except FileNotFoundError:
                pass
        recovered_copy = tmp_path / "recovered-copy.sqlite3"
        recovered_copy.write_bytes(recoveries[0].read_bytes())
        os.replace(recovered_copy, database_path)
        reopened = await repository.open()
        assert reopened == ProfileStoreResult(generation=4, value=None)
        page = await repository.list_profiles()
        assert [profile.display_name for profile in page.value.profiles] == ["Original"]
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_restore_preclose_failure_does_not_reopen_domain_invalid_shared_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    secret = str(tmp_path / "secret-clock")
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )
    original_connection = repository._connection
    original_lease = repository._lease
    await repository._submit_operation(
        lambda connection: connection.execute(
            "UPDATE tts_generation_profiles SET revision = 0"
        )
    )
    monkeypatch.setattr(
        repository,
        "_clock",
        lambda: (_ for _ in ()).throw(RuntimeError(secret)),
    )

    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.restore_from(candidate)

        _assert_safe_error(caught.value, "restore_failed", secret, str(database_path))
        assert repository.state is ProfileRepositoryState.UNAVAILABLE
        assert repository.generation == 2
        assert repository._connection is original_connection
        assert repository._lease is original_lease
        assert repository._lease is not None
        assert repository._lease.mode is ProfileStoreLockMode.SHARED
        assert repository._lease.acquired is True
        assert repository._active_database_path == database_path.resolve()
        assert not tuple(tmp_path.glob("*.restore-stage.sqlite3"))
        assert not tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
        await _assert_exclusive_lease_blocked(database_path)
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_established_store_retry_does_not_recreate_missing_postreplace_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    real_lease_type = module.ProfileStoreLease
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )
    removed_during_handoff = False

    class RemovingHandoffLease(real_lease_type):
        def release(self) -> None:
            nonlocal removed_during_handoff
            super().release()
            if (
                self.mode is ProfileStoreLockMode.EXCLUSIVE
                and not removed_during_handoff
            ):
                for target in (
                    database_path,
                    *(
                        database_path.with_name(f"{database_path.name}{suffix}")
                        for suffix in ("-wal", "-shm", "-journal")
                    ),
                ):
                    try:
                        target.unlink()
                    except FileNotFoundError:
                        pass
                removed_during_handoff = True

    monkeypatch.setattr(module, "ProfileStoreLease", RemovingHandoffLease)

    try:
        with pytest.raises(ProfileRepositoryError) as restore_error:
            await repository.restore_from(candidate)
        _assert_safe_error(restore_error.value, "restore_failed", str(database_path))
        assert removed_during_handoff is True
        assert repository.state is ProfileRepositoryState.UNAVAILABLE
        assert repository._store_established is True
        assert database_path.exists() is False
        recoveries = tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
        assert len(recoveries) == 1
        validate_profile_candidate(recoveries[0])

        with pytest.raises(ProfileRepositoryError) as retry_error:
            await repository.open()
        _assert_safe_error(retry_error.value, "missing", str(database_path))
        assert repository.state is ProfileRepositoryState.UNAVAILABLE
        assert repository.generation == 3
        assert repository._store_established is True
        assert database_path.exists() is False
        assert not any(
            database_path.with_name(f"{database_path.name}{suffix}").exists()
            for suffix in ("-wal", "-shm", "-journal")
        )

        database_path.write_bytes(recoveries[0].read_bytes())
        reopened = await repository.open()
        assert reopened == ProfileStoreResult(generation=4, value=None)
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository._store_established is True
        page = await repository.list_profiles()
        assert [profile.display_name for profile in page.value.profiles] == ["Original"]
    finally:
        await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("boundary", "expected_mode"),
    [
        ("recovery_source", ProfileStoreLockMode.EXCLUSIVE),
        ("post_rebind", ProfileStoreLockMode.SHARED),
    ],
)
async def test_restore_live_connection_close_failure_retains_protecting_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    expected_mode: ProfileStoreLockMode,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    secret = str(tmp_path / f"secret-{boundary}-close")
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )
    real_open = module.open_profile_store
    real_exact_open = module.open_exact_current_profile_store
    real_counts = repository._worker_store_counts
    real_publish = module.publish_profile_migration
    publication_complete = False
    proxies: list[_CloseFailingSQLiteProxy] = []

    def tracked_publish(**kwargs: object) -> None:
        nonlocal publication_complete
        real_publish(**kwargs)
        publication_complete = True

    def injected_open(
        path: Path,
        *,
        must_exist: bool = False,
        check_deadline: Callable[[], None] | None = None,
    ) -> sqlite3.Connection:
        connection = real_open(
            path,
            must_exist=must_exist,
            check_deadline=check_deadline,
        )
        should_proxy = must_exist and (
            (boundary == "recovery_source" and not publication_complete)
            or (boundary == "post_rebind" and publication_complete)
        )
        if should_proxy:
            proxy = _CloseFailingSQLiteProxy(connection, secret)
            proxies.append(proxy)
            return cast(sqlite3.Connection, proxy)
        return connection

    def injected_counts(
        connection: sqlite3.Connection,
        *,
        deadline: float | None = None,
    ) -> tuple[int, int]:
        if boundary == "post_rebind" and any(connection is proxy for proxy in proxies):
            raise RuntimeError(secret)
        return real_counts(connection, deadline=deadline)

    def injected_exact_open(path: Path, **kwargs: object) -> sqlite3.Connection:
        connection = real_exact_open(path, **kwargs)
        if boundary == "post_rebind" and publication_complete:
            proxy = _CloseFailingSQLiteProxy(connection, secret)
            proxies.append(proxy)
            return cast(sqlite3.Connection, proxy)
        return connection

    monkeypatch.setattr(module, "publish_profile_migration", tracked_publish)
    monkeypatch.setattr(module, "open_profile_store", injected_open)
    monkeypatch.setattr(
        module,
        "open_exact_current_profile_store",
        injected_exact_open,
    )
    monkeypatch.setattr(repository, "_worker_store_counts", injected_counts)

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.restore_from(candidate)

    _assert_safe_error(caught.value, "restore_failed", secret, str(database_path))
    assert len(proxies) == 1
    proxy = proxies[0]
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository._connection is proxy
    assert repository._lease is not None
    assert repository._lease.mode is expected_mode
    assert repository._lease.acquired is True
    await _assert_exclusive_lease_blocked(database_path)

    proxy.fail_close = False
    await repository.close()
    assert await asyncio.to_thread(_try_exclusive_lease, database_path) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("boundary", "expected_mode"),
    [
        ("exclusive_release", ProfileStoreLockMode.EXCLUSIVE),
        ("rebound_release", ProfileStoreLockMode.SHARED),
    ],
)
async def test_restore_release_failure_retains_residual_lease_for_later_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    expected_mode: ProfileStoreLockMode,
) -> None:
    module = _repository_module()
    real_lease_type = module.ProfileStoreLease
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    secret = str(tmp_path / f"secret-{boundary}")
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    real_counts = repository._worker_store_counts
    real_publish = module.publish_profile_migration
    publication_complete = False
    retained: list[Any] = []

    class ControlledReleaseLease(real_lease_type):
        fail_release = True

        def release(self) -> None:
            should_fail = (
                boundary == "exclusive_release"
                and self.mode is ProfileStoreLockMode.EXCLUSIVE
            ) or (
                boundary == "rebound_release"
                and self.mode is ProfileStoreLockMode.SHARED
            )
            if should_fail and self.fail_release:
                retained.append(self)
                raise RuntimeError(secret)
            super().release()

    def tracked_publish(**kwargs: object) -> None:
        nonlocal publication_complete
        real_publish(**kwargs)
        publication_complete = True

    def injected_counts(
        connection: sqlite3.Connection,
        *,
        deadline: float | None = None,
    ) -> tuple[int, int]:
        if boundary == "rebound_release" and publication_complete:
            raise RuntimeError(secret)
        return real_counts(connection, deadline=deadline)

    monkeypatch.setattr(module, "ProfileStoreLease", ControlledReleaseLease)
    monkeypatch.setattr(module, "publish_profile_migration", tracked_publish)
    monkeypatch.setattr(repository, "_worker_store_counts", injected_counts)

    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.restore_from(candidate)

    _assert_safe_error(caught.value, "restore_failed", secret, str(database_path))
    assert retained
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository._connection is None
    assert repository._lease is retained[-1]
    assert repository._lease.mode is expected_mode
    assert repository._lease.acquired is True
    await _assert_exclusive_lease_blocked(database_path)

    repository._lease.fail_release = False
    await repository.close()
    assert await asyncio.to_thread(_try_exclusive_lease, database_path) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["clock", "checkpoint", "connection_close"])
async def test_restore_preclose_failure_keeps_usable_original_shared_pair_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    secret = str(tmp_path / f"secret-{boundary}")
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )
    original_connection = repository._connection
    original_lease = repository._lease
    proxy: _CloseFailingSQLiteProxy | None = None

    if boundary == "clock":
        monkeypatch.setattr(
            repository,
            "_clock",
            lambda: (_ for _ in ()).throw(RuntimeError(secret)),
        )
    elif boundary == "connection_close":
        assert original_connection is not None
        proxy = _CloseFailingSQLiteProxy(original_connection, secret)
        repository._connection = cast(sqlite3.Connection, proxy)
        original_connection = repository._connection
    else:
        assert original_connection is not None
        delegate = original_connection

        class CheckpointFailingProxy:
            def __getattr__(self, name: str) -> Any:
                return getattr(delegate, name)

            def execute(
                self,
                sql: str,
                parameters: object = (),
            ) -> Any:
                if sql == "PRAGMA wal_checkpoint(TRUNCATE)":
                    raise RuntimeError(secret)
                return delegate.execute(sql, parameters)

        repository._connection = cast(sqlite3.Connection, CheckpointFailingProxy())
        original_connection = repository._connection

    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.restore_from(candidate)

        _assert_safe_error(caught.value, "restore_failed", secret, str(database_path))
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository.generation == 2
        assert repository._connection is original_connection
        assert repository._lease is original_lease
        assert repository._lease is not None
        assert repository._lease.mode is ProfileStoreLockMode.SHARED
        page = await repository.list_profiles()
        assert [profile.display_name for profile in page.value.profiles] == ["Original"]
    finally:
        if proxy is not None:
            proxy.fail_close = False
        await repository.close()


@pytest.mark.asyncio
async def test_restore_fsyncs_recovery_directory_entry_before_publication_ponr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    events: list[tuple[str, Path | None]] = []
    real_fsync_file = module._fsync_file
    real_fsync_directory = module._fsync_directory
    real_publish = module.publish_profile_migration

    def observed_fsync_file(path: Path) -> None:
        events.append(("file", path))
        real_fsync_file(path)

    def observed_fsync_directory(path: Path) -> None:
        events.append(("directory", path))
        real_fsync_directory(path)

    def observed_publish(**kwargs: object) -> None:
        repository_hook = kwargs.pop("stage_hook", None)

        def stage_hook(stage: object) -> None:
            if repository_hook is not None:
                repository_hook(stage)
            if stage.value == "ponr":
                events.append(("ponr", None))

        real_publish(**kwargs, stage_hook=stage_hook)

    monkeypatch.setattr(module, "_fsync_file", observed_fsync_file)
    monkeypatch.setattr(module, "_fsync_directory", observed_fsync_directory)
    monkeypatch.setattr(module, "publish_profile_migration", observed_publish)

    try:
        await repository.restore_from(candidate)

        recovery_file_index = next(
            index
            for index, (kind, path) in enumerate(events)
            if kind == "file" and path.name.endswith(".recovery.sqlite3")
        )
        ponr_index = next(
            index for index, (kind, _path) in enumerate(events) if kind == "ponr"
        )
        assert events[recovery_file_index + 1] == ("directory", tmp_path)
        assert recovery_file_index < ponr_index
    finally:
        await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "clock",
    [
        lambda: float("inf"),
        lambda: (_ for _ in ()).throw(RuntimeError("hostile-monotonic")),
    ],
)
async def test_restore_hostile_monotonic_fails_before_lifecycle_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clock: Callable[[], float],
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    before = {
        path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()
    }
    monkeypatch.setattr(module, "_monotonic", clock)

    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.restore_from(candidate)

        _assert_safe_error(caught.value, "restore_failed", str(candidate))
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository.generation == 1
        assert {
            path.name: path.read_bytes()
            for path in tmp_path.iterdir()
            if path.is_file()
        } == before
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_cancelled_restore_settles_transition_before_propagating_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    worker_started = threading.Event()
    release_worker = threading.Event()
    real_restore = repository._worker_restore

    def blocked_restore(*args: object, **kwargs: object) -> ProfileRestoreReceipt:
        worker_started.set()
        assert release_worker.wait(5.0)
        return real_restore(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(repository, "_worker_restore", blocked_restore)
    restore_task = asyncio.create_task(repository.restore_from(candidate))
    await _wait_thread_event(worker_started)
    restore_task.cancel("caller-cancelled")
    await asyncio.sleep(0)
    assert restore_task.done() is False

    release_worker.set()
    with pytest.raises(asyncio.CancelledError) as caught:
        await restore_task

    assert caught.value.args == ("caller-cancelled",)
    assert repository.state is ProfileRepositoryState.OPEN
    assert repository.generation == 2
    page = await repository.list_profiles()
    assert [profile.display_name for profile in page.value.profiles] == ["Candidate"]
    await repository.close()


@pytest.mark.asyncio
async def test_backup_and_restore_reject_foreign_loop_without_lifecycle_mutation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    destination = tmp_path / "backup.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()

    try:
        for operation in (
            lambda: repository.backup_to(destination),
            lambda: repository.restore_from(candidate),
        ):
            result, error = await _run_in_new_loop_thread(operation)
            assert result is None
            assert isinstance(error, ProfileRepositoryError)
            _assert_safe_error(error, "invalid_state")
            assert repository.state is ProfileRepositoryState.OPEN
            assert repository.generation == 1
        assert destination.exists() is False
        assert not tuple(tmp_path.glob("*.restore-stage.sqlite3"))
    finally:
        await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "terminal", "expected_code"),
    [
        (ProfileRepositoryState.CLOSED, False, "closed"),
        (ProfileRepositoryState.RESTORING, False, "restoring"),
        (ProfileRepositoryState.UNAVAILABLE, False, "unavailable"),
        (ProfileRepositoryState.CLOSED, True, "terminal"),
    ],
)
async def test_backup_and_restore_preserve_non_open_lifecycle_errors(
    tmp_path: Path,
    state: ProfileRepositoryState,
    terminal: bool,
    expected_code: str,
) -> None:
    store_directory = tmp_path / "profile-store"
    repository = _repository(store_directory / "profiles.sqlite3")
    repository._state = state
    repository._terminal = terminal

    for operation in (
        lambda: repository.backup_to(store_directory / "backup.sqlite3"),
        lambda: repository.restore_from(store_directory / "candidate.sqlite3"),
    ):
        with pytest.raises(ProfileRepositoryError) as caught:
            await operation()
        _assert_safe_error(caught.value, expected_code, str(store_directory))

    assert repository.state is state
    assert repository.generation == 0
    assert repository._connection is None
    assert repository._lease is None
    assert repository._active_database_path is None
    assert store_directory.exists() is False


@pytest.mark.asyncio
async def test_restore_runs_full_integrity_check_before_publishing_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    statements: list[str] = []
    real_connect = module.sqlite3.connect

    def traced_connect(
        database: object,
        *args: object,
        **kwargs: object,
    ) -> sqlite3.Connection:
        connection = real_connect(database, *args, **kwargs)  # type: ignore[arg-type]
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(module.sqlite3, "connect", traced_connect)
    try:
        await repository.restore_from(candidate)

        assert "PRAGMA integrity_check" in statements
        assert repository.state is ProfileRepositoryState.OPEN
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_restore_full_integrity_failure_preserves_original_before_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )

    def fail_integrity(
        _connection: sqlite3.Connection,
        *,
        deadline: float | None = None,
    ) -> None:
        del deadline
        raise ProfileRepositoryError("schema_corrupt")

    monkeypatch.setattr(repository, "_worker_require_full_integrity", fail_integrity)
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.restore_from(candidate)

    _assert_safe_error(
        caught.value, "schema_corrupt", str(candidate), str(database_path)
    )
    assert repository.state is ProfileRepositoryState.OPEN
    assert repository.generation == 2
    assert not tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
    page = await repository.list_profiles()
    assert [profile.display_name for profile in page.value.profiles] == ["Original"]
    await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "checkpoint_row",
    [
        (0, True, 0),
        (0, 1, 0),
        (0, 0, 1),
        (0, 0),
        ("0", 0, 0),
    ],
)
async def test_restore_requires_exact_completed_truncate_checkpoint_evidence(
    tmp_path: Path,
    checkpoint_row: tuple[object, ...],
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    original_connection = repository._connection
    original_lease = repository._lease
    assert original_connection is not None

    class CheckpointCursor:
        def fetchone(self) -> tuple[object, ...]:
            return checkpoint_row

    class CheckpointProxy:
        def __getattr__(self, name: str) -> Any:
            return getattr(original_connection, name)

        def execute(self, sql: str, parameters: object = ()) -> Any:
            if sql == "PRAGMA wal_checkpoint(TRUNCATE)":
                return CheckpointCursor()
            return original_connection.execute(sql, parameters)

    proxy = cast(sqlite3.Connection, CheckpointProxy())
    repository._connection = proxy
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.restore_from(candidate)

    _assert_safe_error(caught.value, "restore_failed", str(database_path))
    assert repository.state is ProfileRepositoryState.OPEN
    assert repository.generation == 2
    assert repository._connection is proxy
    assert repository._lease is original_lease
    assert (await repository.list_profiles()).value.total == 0
    await repository.close()


@pytest.mark.asyncio
async def test_restore_refuses_live_rollback_journal_without_deleting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    journal = database_path.with_name(f"{database_path.name}-journal")
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )
    real_remove_sidecars = repository._worker_remove_live_sidecars
    real_rebind = repository._worker_rebind_current_store
    journal_seen_before_rebind = False

    def inject_journal(*, deadline: float) -> None:
        journal.touch()
        real_remove_sidecars(deadline=deadline)

    def observed_rebind() -> None:
        nonlocal journal_seen_before_rebind
        journal_seen_before_rebind = journal.exists()
        real_rebind()

    monkeypatch.setattr(repository, "_worker_remove_live_sidecars", inject_journal)
    monkeypatch.setattr(repository, "_worker_rebind_current_store", observed_rebind)
    with pytest.raises(ProfileRepositoryError) as caught:
        await repository.restore_from(candidate)

    _assert_safe_error(caught.value, "restore_failed", str(database_path))
    assert journal_seen_before_rebind is True
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    assert repository._connection is None
    assert repository._lease is None
    assert journal.exists()
    check = sqlite3.connect(database_path)
    try:
        assert check.execute(
            "SELECT display_name FROM tts_generation_profiles"
        ).fetchall() == [("Original",)]
    finally:
        check.close()
    assert not tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
    await repository.close()


@pytest.mark.asyncio
async def test_backup_rejects_configured_symlink_drift_without_replacing_active_lock(
    tmp_path: Path,
) -> None:
    active_path = tmp_path / "active.sqlite3"
    alternate_path = tmp_path / "alternate.sqlite3"
    configured_path = tmp_path / "configured.sqlite3"
    await _create_profile_store(active_path, "Active")
    await _create_profile_store(alternate_path, "Alternate")
    configured_path.symlink_to(active_path)
    repository = _repository(configured_path)
    await repository.open()
    active_lock = active_path.with_name(f"{active_path.name}.lock")
    lock_identity = (active_lock.stat().st_dev, active_lock.stat().st_ino)
    lock_bytes = active_lock.read_bytes()
    active_bytes = active_path.read_bytes()
    alternate_bytes = alternate_path.read_bytes()
    configured_path.unlink()
    configured_path.symlink_to(alternate_path)

    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.backup_to(active_lock)

        _assert_safe_error(
            caught.value,
            "backup_failed",
            str(active_path),
            str(configured_path),
        )
        assert (active_lock.stat().st_dev, active_lock.stat().st_ino) == lock_identity
        assert active_lock.read_bytes() == lock_bytes
        assert active_path.read_bytes() == active_bytes
        assert alternate_path.read_bytes() == alternate_bytes
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository.generation == 1
        page = await repository.list_profiles()
        assert [profile.display_name for profile in page.value.profiles] == ["Active"]
        await _assert_exclusive_lease_blocked(active_path)
        assert await asyncio.to_thread(_try_exclusive_lease, alternate_path) is None
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_backup_rechecks_configured_symlink_after_worker_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_path = tmp_path / "active.sqlite3"
    alternate_path = tmp_path / "alternate.sqlite3"
    configured_path = tmp_path / "configured.sqlite3"
    destination = tmp_path / "backup.sqlite3"
    await _create_profile_store(active_path, "Active")
    await _create_profile_store(alternate_path, "Alternate")
    configured_path.symlink_to(active_path)
    repository = _repository(configured_path)
    await repository.open()
    real_backup = repository._worker_online_backup

    def drifting_backup(
        source: sqlite3.Connection,
        target: sqlite3.Connection,
    ) -> None:
        real_backup(source, target)
        configured_path.unlink()
        configured_path.symlink_to(alternate_path)

    monkeypatch.setattr(repository, "_worker_online_backup", drifting_backup)
    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.backup_to(destination)

        _assert_safe_error(caught.value, "backup_failed", str(configured_path))
        assert destination.exists() is False
        assert not tuple(tmp_path.glob(f".{destination.name}.*.backup"))
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository.generation == 1
        page = await repository.list_profiles()
        assert [profile.display_name for profile in page.value.profiles] == ["Active"]
        await _assert_exclusive_lease_blocked(active_path)
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_restore_rejects_configured_symlink_drift_without_mutating_either_store(
    tmp_path: Path,
) -> None:
    active_path = tmp_path / "active.sqlite3"
    alternate_path = tmp_path / "alternate.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    configured_path = tmp_path / "configured.sqlite3"
    await _create_profile_store(active_path, "Active")
    await _create_profile_store(alternate_path, "Alternate")
    await _create_profile_store(candidate, "Candidate")
    configured_path.symlink_to(active_path)
    repository = _repository(configured_path)
    await repository.open()
    active_lock = active_path.with_name(f"{active_path.name}.lock")
    lock_identity = (active_lock.stat().st_dev, active_lock.stat().st_ino)
    active_bytes = active_path.read_bytes()
    alternate_bytes = alternate_path.read_bytes()
    configured_path.unlink()
    configured_path.symlink_to(alternate_path)

    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.restore_from(candidate)

        _assert_safe_error(
            caught.value,
            "restore_failed",
            str(active_path),
            str(alternate_path),
            str(configured_path),
        )
        assert (active_lock.stat().st_dev, active_lock.stat().st_ino) == lock_identity
        assert active_path.read_bytes() == active_bytes
        assert alternate_path.read_bytes() == alternate_bytes
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository.generation == 2
        assert not tuple(tmp_path.glob("*.restore-stage.sqlite3"))
        assert not tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
        page = await repository.list_profiles()
        assert [profile.display_name for profile in page.value.profiles] == ["Active"]
        alternate = open_profile_store(alternate_path, must_exist=True)
        try:
            alternate_name = alternate.execute(
                "SELECT display_name FROM tts_generation_profiles"
            ).fetchone()[0]
        finally:
            alternate.close()
        assert alternate_name == "Alternate"
        await _assert_exclusive_lease_blocked(active_path)
        assert await asyncio.to_thread(_try_exclusive_lease, alternate_path) is None
    finally:
        await repository.close()


@pytest.mark.asyncio
async def test_restore_rechecks_symlink_drift_after_quiescing_old_worker_io(
    tmp_path: Path,
) -> None:
    active_path = tmp_path / "active.sqlite3"
    alternate_path = tmp_path / "alternate.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    configured_path = tmp_path / "configured.sqlite3"
    await _create_profile_store(active_path, "Active")
    await _create_profile_store(alternate_path, "Alternate")
    await _create_profile_store(candidate, "Candidate")
    configured_path.symlink_to(active_path)
    repository = _repository(configured_path)
    await repository.open()
    original_connection = repository._connection
    original_lease = repository._lease
    operation_started = threading.Event()
    release_operation = threading.Event()

    def blocked_operation(_connection: sqlite3.Connection) -> None:
        operation_started.set()
        assert release_operation.wait(5.0)

    admission = repository._admit_operation(blocked_operation)
    await _wait_thread_event(operation_started)
    restore = asyncio.create_task(repository.restore_from(candidate))
    for _ in range(100):
        if repository.state is ProfileRepositoryState.RESTORING:
            break
        await asyncio.sleep(0)
    assert repository.state is ProfileRepositoryState.RESTORING
    configured_path.unlink()
    configured_path.symlink_to(alternate_path)
    release_operation.set()

    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await restore

        _assert_safe_error(caught.value, "restore_failed", str(configured_path))
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository.generation == 2
        assert repository._connection is original_connection
        assert repository._lease is original_lease
        assert not tuple(tmp_path.glob("*.restore-stage.sqlite3"))
        assert not tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
        page = await repository.list_profiles()
        assert [profile.display_name for profile in page.value.profiles] == ["Active"]
        await _assert_exclusive_lease_blocked(active_path)
        with pytest.raises(ProfileRepositoryError) as stale:
            await repository._publish_operation(admission)
        _assert_safe_error(stale.value, "stale")
    finally:
        release_operation.set()
        await repository.close()


@pytest.mark.asyncio
async def test_restore_expired_immediately_after_exclusive_acquire_does_not_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _repository_module()
    real_lease_type = module.ProfileStoreLease
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    followup_backup = tmp_path / "followup-backup.sqlite3"
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )
    now = 0.0
    candidate_copy_called = False

    class DelayedExclusiveLease(real_lease_type):
        def acquire(self) -> ProfileStoreLease:
            nonlocal now
            result = super().acquire()
            if self.mode is ProfileStoreLockMode.EXCLUSIVE:
                now = 11.0
            return result

    def unexpected_candidate_copy(*_args: object, **_kwargs: object) -> object:
        nonlocal candidate_copy_called
        candidate_copy_called = True
        raise AssertionError("candidate copy must not run after the deadline")

    monkeypatch.setattr(module, "_monotonic", lambda: now)
    monkeypatch.setattr(module, "ProfileStoreLease", DelayedExclusiveLease)
    monkeypatch.setattr(
        module,
        "migrate_profile_store_to_candidate",
        unexpected_candidate_copy,
    )

    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.restore_from(candidate, timeout_seconds=10.0)

        _assert_safe_error(caught.value, "restore_failed", str(database_path))
        assert candidate_copy_called is False
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository.generation == 2
        assert repository._active_database_path == database_path.resolve()
        assert not tuple(tmp_path.glob("*.restore-stage.sqlite3"))
        assert not tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
        page = await repository.list_profiles()
        assert [profile.display_name for profile in page.value.profiles] == ["Original"]
        now = 0.0
        await repository.backup_to(followup_backup)
        validate_profile_candidate(followup_backup)
        await _assert_exclusive_lease_blocked(database_path)
    finally:
        await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("with_pending_operation", [False, True])
@pytest.mark.parametrize("timing_failure", ["exception", "non_finite"])
async def test_restore_post_admission_timing_failure_settles_original_store_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_pending_operation: bool,
    timing_failure: str,
) -> None:
    module = _repository_module()
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    secret = str(tmp_path / f"secret-{timing_failure}")
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )
    original_connection = repository._connection
    original_lease = repository._lease
    operation_started = threading.Event()
    release_operation = threading.Event()
    admission: Any = None

    if with_pending_operation:

        def blocked_operation(_connection: sqlite3.Connection) -> None:
            operation_started.set()
            assert release_operation.wait(5.0)

        admission = repository._admit_operation(blocked_operation)
        await _wait_thread_event(operation_started)

    timing_calls = 0

    def sequenced_monotonic() -> float:
        nonlocal timing_calls
        timing_calls += 1
        if timing_calls <= 2:
            return 0.0
        if timing_failure == "exception":
            raise RuntimeError(secret)
        return float("inf")

    monkeypatch.setattr(module, "_monotonic", sequenced_monotonic)
    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.restore_from(candidate, timeout_seconds=10.0)

        _assert_safe_error(caught.value, "restore_failed", secret)
        assert timing_calls == 3
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository.generation == 2
        assert repository._connection is original_connection
        assert repository._lease is original_lease
        assert repository._lease is not None
        assert repository._lease.mode is ProfileStoreLockMode.SHARED
        assert repository._lease.acquired is True
        assert not tuple(tmp_path.glob("*.restore-stage.sqlite3"))
        assert not tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
    finally:
        release_operation.set()
        if admission is not None:
            with pytest.raises(ProfileRepositoryError) as stale:
                await repository._publish_operation(admission)
            _assert_safe_error(stale.value, "stale")
        await repository.close()


@pytest.mark.asyncio
async def test_restore_worker_submission_failure_settles_original_store_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "profiles.sqlite3"
    candidate = tmp_path / "candidate.sqlite3"
    secret = str(tmp_path / "secret-submit")
    await _create_profile_store(candidate, "Candidate")
    repository = _repository(database_path)
    await repository.open()
    await repository.create_profile(
        _draft("Original"),
        UUID("00000000-0000-4000-8000-000000000099"),
    )
    original_connection = repository._connection
    original_lease = repository._lease
    executor = repository._executor
    assert executor is not None
    real_submit = executor.submit

    def rejecting_submit(
        function: Callable[..., object],
        /,
        *args: object,
        **kwargs: object,
    ) -> Future[object]:
        if function == repository._worker_restore:
            raise RuntimeError(secret)
        return real_submit(function, *args, **kwargs)

    monkeypatch.setattr(executor, "submit", rejecting_submit)
    try:
        with pytest.raises(ProfileRepositoryError) as caught:
            await repository.restore_from(candidate)

        _assert_safe_error(caught.value, "restore_failed", secret)
        assert repository.state is ProfileRepositoryState.OPEN
        assert repository.generation == 2
        assert repository._connection is original_connection
        assert repository._lease is original_lease
        assert repository._active_database_path == database_path.resolve()
        assert not tuple(tmp_path.glob("*.restore-stage.sqlite3"))
        assert not tuple(tmp_path.glob("*.pre-restore-*.recovery.sqlite3"))
        page = await repository.list_profiles()
        assert [profile.display_name for profile in page.value.profiles] == ["Original"]
    finally:
        await repository.close()
