"""Crash-safe managed files for Local Collections captures."""

from __future__ import annotations

import os
import stat
import threading
import hashlib
from pathlib import Path

import pytest

from tldw_chatbook.DB.Library_Collections_DB import LibraryCollectionsDB
from tldw_chatbook.Library.collections_capture_models import (
    CaptureSaveRequest,
    CollectionsCaptureError,
)
from tldw_chatbook.Library.collections_capture_repository import (
    CollectionsCaptureRepository,
)
from tldw_chatbook.Library.collections_offline_store import CollectionsOfflineStore
from tldw_chatbook.Utils.private_paths import (
    PrivateFileWritePrecondition,
    atomic_private_write_bytes,
)


def _repository(path: Path, authority: str = "local:profile-a"):
    database = LibraryCollectionsDB(path)
    return database, CollectionsCaptureRepository(database, authority_key=authority)


def _capture(repository: CollectionsCaptureRepository, suffix: str = "one"):
    outcome = repository.save_capture(
        CaptureSaveRequest(
            authority_key=repository.authority_key,
            submitted_url=f"https://example.org/{suffix}",
            title=f"Capture {suffix}",
        )
    )
    assert outcome.capture is not None
    return outcome.capture


def _store(
    repository: CollectionsCaptureRepository,
    root: Path,
    *,
    fingerprint: str = "a1b2c3d4",
    max_copy_bytes: int = 1024,
    max_authority_bytes: int = 4096,
) -> CollectionsOfflineStore:
    return CollectionsOfflineStore(
        repository,
        data_root=root,
        authority_fingerprint=fingerprint,
        max_copy_bytes=max_copy_bytes,
        max_authority_bytes=max_authority_bytes,
    )


def _digest(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def test_save_open_and_delete_private_copy(tmp_path: Path) -> None:
    database, repository = _repository(tmp_path / "collections.db")
    capture = _capture(repository)
    store = _store(repository, tmp_path / "private")

    saved = store.save_copy(capture.identity, b"offline body", "text/plain")

    assert saved.state == "ready"
    assert saved.size == len(b"offline body")
    assert store.open_copy(capture.identity) == b"offline body"
    path = store.authority_root / saved.file_id / f"{saved.file_id}.bin"
    if os.name == "posix":
        assert stat.S_IMODE(store.authority_root.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(
            (store.authority_root / ".lifecycle.lock").stat().st_mode
        ) == 0o600

    result = store.delete_copy(capture.identity)
    assert result.success is True
    assert path.exists() is False
    detail = repository.get_detail(capture.identity)
    assert detail is not None and detail.offline_copy is None
    database.close()


def test_copy_and_authority_quota_count_staging_reservations(tmp_path: Path) -> None:
    database, repository = _repository(tmp_path / "collections.db")
    first = _capture(repository, "first")
    second = _capture(repository, "second")
    third = _capture(repository, "third")
    store = _store(
        repository,
        tmp_path / "private",
        max_copy_bytes=60,
        max_authority_bytes=100,
    )

    with pytest.raises(CollectionsCaptureError) as caught:
        store.save_copy(first.identity, b"x" * 61, "text/plain")
    assert caught.value.reason == "offline_copy_too_large"

    ready = store.save_copy(first.identity, b"r" * 40, "text/plain")
    assert ready.state == "ready"
    staging = repository.reserve_offline_copy(
        second.identity,
        reserved_size=60,
        media_type="text/plain",
        content_hash=_digest(b"x" * 60),
        max_copy_bytes=60,
        max_authority_bytes=100,
    )
    assert staging.state == "staging"
    with pytest.raises(CollectionsCaptureError) as caught:
        repository.reserve_offline_copy(
            third.identity,
            reserved_size=1,
            media_type="text/plain",
            content_hash=_digest(b"x"),
            max_copy_bytes=60,
            max_authority_bytes=100,
        )
    assert caught.value.reason == "offline_authority_quota_exceeded"
    database.close()


def test_default_limits_admit_exact_boundaries(tmp_path: Path) -> None:
    database, repository = _repository(tmp_path / "collections.db")
    copy_limit = 50 * 1024 * 1024
    authority_limit = 1024 * 1024 * 1024
    captures = [_capture(repository, f"quota-{index}") for index in range(23)]

    for capture in captures[:20]:
        repository.reserve_offline_copy(
            capture.identity,
            reserved_size=copy_limit,
            media_type="application/octet-stream",
            content_hash=_digest(b"reservation"),
        )
    repository.reserve_offline_copy(
        captures[20].identity,
        reserved_size=authority_limit - (20 * copy_limit),
        media_type="application/octet-stream",
        content_hash=_digest(b"reservation"),
    )
    with pytest.raises(CollectionsCaptureError) as caught:
        repository.reserve_offline_copy(
            captures[21].identity,
            reserved_size=1,
            media_type="application/octet-stream",
            content_hash=_digest(b"reservation"),
        )
    assert caught.value.reason == "offline_authority_quota_exceeded"
    with pytest.raises(CollectionsCaptureError) as caught:
        repository.reserve_offline_copy(
            captures[22].identity,
            reserved_size=copy_limit + 1,
            media_type="application/octet-stream",
            content_hash=_digest(b"reservation"),
        )
    assert caught.value.reason == "offline_copy_too_large"
    database.close()


def test_concurrent_quota_reservations_admit_only_one(tmp_path: Path) -> None:
    path = tmp_path / "collections.db"
    database_a, repo_a = _repository(path)
    database_b, repo_b = _repository(path)
    first = _capture(repo_a, "first")
    second = _capture(repo_a, "second")
    barrier = threading.Barrier(3)
    outcomes: list[str] = []

    def reserve(repository, identity) -> None:
        barrier.wait()
        try:
            repository.reserve_offline_copy(
                identity,
                reserved_size=60,
                media_type="text/plain",
                content_hash=_digest(b"x" * 60),
                max_copy_bytes=60,
                max_authority_bytes=100,
            )
            outcomes.append("saved")
        except CollectionsCaptureError as exc:
            outcomes.append(exc.reason)

    threads = [
        threading.Thread(target=reserve, args=(repo_a, first.identity)),
        threading.Thread(target=reserve, args=(repo_b, second.identity)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)

    assert sorted(outcomes) == ["offline_authority_quota_exceeded", "saved"]
    database_a.close()
    database_b.close()


def test_reconcile_does_not_fail_a_live_publication(tmp_path: Path) -> None:
    path = tmp_path / "collections.db"
    database_a, repository_a = _repository(path)
    database_b, repository_b = _repository(path)
    capture = _capture(repository_a)
    store_a = _store(repository_a, tmp_path / "private")
    store_b = _store(repository_b, tmp_path / "private")
    publication_started = threading.Event()
    release_publication = threading.Event()
    saved: list[object] = []
    failures: list[BaseException] = []

    def block_before_publish() -> None:
        publication_started.set()
        assert release_publication.wait(timeout=10)

    def publish() -> None:
        try:
            saved.append(store_a.save_copy(capture.identity, b"live", "text/plain"))
        except BaseException as exc:  # noqa: BLE001 - preserve thread failure
            failures.append(exc)

    store_a._before_publish = block_before_publish
    thread = threading.Thread(target=publish)
    thread.start()
    assert publication_started.wait(timeout=10)

    recovery = store_b.reconcile_batch(limit=1)
    release_publication.set()
    thread.join(timeout=10)

    assert recovery.processed == 0
    assert failures == []
    assert len(saved) == 1
    assert store_b.open_copy(capture.identity) == b"live"
    database_a.close()
    database_b.close()


def test_hard_delete_refuses_live_publication_without_tombstoning(
    tmp_path: Path,
) -> None:
    path = tmp_path / "collections.db"
    database_a, repository_a = _repository(path)
    database_b, repository_b = _repository(path)
    capture = _capture(repository_a)
    store = _store(repository_a, tmp_path / "private")
    publication_started = threading.Event()
    release_publication = threading.Event()
    failures: list[BaseException] = []

    def block_before_publish() -> None:
        publication_started.set()
        assert release_publication.wait(timeout=10)

    def publish() -> None:
        try:
            store.save_copy(capture.identity, b"live", "text/plain")
        except BaseException as exc:  # noqa: BLE001 - preserve thread failure
            failures.append(exc)

    store._before_publish = block_before_publish
    thread = threading.Thread(target=publish)
    thread.start()
    assert publication_started.wait(timeout=10)

    with pytest.raises(CollectionsCaptureError) as caught:
        repository_b.hard_delete(
            capture.identity,
            expected_revision=capture.revision,
        )
    assert caught.value.reason == "offline_copy_busy"
    assert caught.value.retryable is True
    assert repository_b.get_detail(capture.identity) is not None

    release_publication.set()
    thread.join(timeout=10)
    assert failures == []
    database_a.close()
    database_b.close()


def test_store_rejects_unsafe_authority_root_and_symlink(tmp_path: Path) -> None:
    database, repository = _repository(tmp_path / "collections.db")
    with pytest.raises(CollectionsCaptureError) as caught:
        _store(repository, tmp_path / "private", fingerprint="../escape")
    assert caught.value.reason == "invalid_authority_fingerprint"

    private = tmp_path / "private"
    archives = private / "collections_archives"
    archives.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (archives / "a1b2c3d4").symlink_to(outside, target_is_directory=True)
    with pytest.raises(CollectionsCaptureError) as caught:
        _store(repository, private)
    assert caught.value.reason == "offline_store_unavailable"
    database.close()


@pytest.mark.parametrize("relative_path", ["/tmp/outside.bin", "../outside.bin"])
def test_open_rejects_tampered_relative_path(
    tmp_path: Path,
    relative_path: str,
) -> None:
    database, repository = _repository(tmp_path / "collections.db")
    capture = _capture(repository)
    store = _store(repository, tmp_path / "private")
    saved = store.save_copy(capture.identity, b"private", "text/plain")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    with database.transaction() as connection:
        connection.execute(
            "UPDATE collection_capture_offline_files SET relative_path = ? "
            "WHERE authority_key = ? AND file_id = ?",
            (relative_path, repository.authority_key, saved.file_id),
        )

    with pytest.raises(CollectionsCaptureError) as caught:
        store.open_copy(capture.identity)

    assert caught.value.reason == "offline_copy_unavailable"
    assert outside.read_bytes() == b"outside"
    detail = repository.get_detail(capture.identity)
    assert detail is not None and detail.offline_copy is not None
    assert detail.offline_copy.state == "failed"
    database.close()


def test_open_rejects_symlinked_copy_directory(tmp_path: Path) -> None:
    database, repository = _repository(tmp_path / "collections.db")
    capture = _capture(repository)
    store = _store(repository, tmp_path / "private")
    saved = store.save_copy(capture.identity, b"private", "text/plain")
    file_root = store.authority_root / saved.file_id
    moved = tmp_path / "moved-copy"
    file_root.rename(moved)
    file_root.symlink_to(moved, target_is_directory=True)

    with pytest.raises(CollectionsCaptureError) as caught:
        store.open_copy(capture.identity)

    assert caught.value.reason == "offline_copy_unavailable"
    assert (moved / f"{saved.file_id}.bin").read_bytes() == b"private"
    detail = repository.get_detail(capture.identity)
    assert detail is not None and detail.offline_copy is not None
    assert detail.offline_copy.state == "failed"
    database.close()


def test_authorities_cannot_read_each_others_files(tmp_path: Path) -> None:
    path = tmp_path / "collections.db"
    database_a, repository_a = _repository(path, "local:profile-a")
    database_b, repository_b = _repository(path, "local:profile-b")
    capture_a = _capture(repository_a, "a")
    capture_b = _capture(repository_b, "b")
    store_a = _store(repository_a, tmp_path / "private", fingerprint="a1b2c3d4")
    store_b = _store(repository_b, tmp_path / "private", fingerprint="b1c2d3e4")
    copy_a = store_a.save_copy(capture_a.identity, b"authority a", "text/plain")
    copy_b = store_b.save_copy(capture_b.identity, b"authority b", "text/plain")

    with pytest.raises(CollectionsCaptureError) as caught:
        store_b.open_copy(capture_a.identity)
    assert caught.value.reason == "authority_mismatch"
    assert store_a.open_copy(capture_a.identity) == b"authority a"
    assert store_b.open_copy(capture_b.identity) == b"authority b"
    assert copy_a.file_id != copy_b.file_id
    assert store_a.authority_root != store_b.authority_root
    database_a.close()
    database_b.close()


def test_open_detects_hash_or_size_tampering(tmp_path: Path) -> None:
    database, repository = _repository(tmp_path / "collections.db")
    capture = _capture(repository)
    store = _store(repository, tmp_path / "private")
    saved = store.save_copy(capture.identity, b"original", "text/plain")
    path = store.authority_root / saved.file_id / f"{saved.file_id}.bin"
    path.write_bytes(b"tampered")
    path.chmod(0o600)

    with pytest.raises(CollectionsCaptureError) as caught:
        store.open_copy(capture.identity)
    assert caught.value.reason == "offline_copy_unavailable"
    detail = repository.get_detail(capture.identity)
    assert detail is not None and detail.offline_copy is not None
    assert detail.offline_copy.state == "failed"
    assert detail.offline_copy.failure_reason == "offline_integrity_failed"
    assert path.exists() is False
    database.close()


def test_open_rejects_file_that_grew_past_copy_limit(tmp_path: Path) -> None:
    database, repository = _repository(tmp_path / "collections.db")
    capture = _capture(repository)
    store = _store(repository, tmp_path / "private", max_copy_bytes=8)
    saved = store.save_copy(capture.identity, b"12345678", "text/plain")
    path = store.authority_root / saved.file_id / f"{saved.file_id}.bin"
    path.write_bytes(b"123456789")
    path.chmod(0o600)

    with pytest.raises(CollectionsCaptureError) as caught:
        store.open_copy(capture.identity)
    assert caught.value.reason == "offline_copy_unavailable"
    detail = repository.get_detail(capture.identity)
    assert detail is not None and detail.offline_copy is not None
    assert detail.offline_copy.state == "failed"
    assert path.exists() is False
    database.close()


def test_reconcile_recovers_post_publish_and_fails_pre_publish_crashes(
    tmp_path: Path,
) -> None:
    database, repository = _repository(tmp_path / "collections.db")
    before = _capture(repository, "before")
    after = _capture(repository, "after")
    store = _store(repository, tmp_path / "private")

    store._before_publish = lambda: (_ for _ in ()).throw(SystemExit())
    with pytest.raises(SystemExit):
        store.save_copy(before.identity, b"before", "text/plain")
    store._before_publish = lambda: None
    store._after_publish = lambda: (_ for _ in ()).throw(SystemExit())
    with pytest.raises(SystemExit):
        store.save_copy(after.identity, b"after", "text/plain")
    store._after_publish = lambda: None

    first = store.reconcile_batch(limit=1)
    second = store.reconcile_batch(limit=1)

    assert first.processed == 1 and first.has_more is True
    assert second.processed == 1
    before_detail = repository.get_detail(before.identity)
    after_detail = repository.get_detail(after.identity)
    assert before_detail is not None
    assert before_detail.offline_copy is not None
    assert before_detail.offline_copy.state == "failed"
    assert after_detail is not None
    assert after_detail.offline_copy is not None
    assert after_detail.offline_copy.state == "ready"
    database.close()


@pytest.mark.parametrize("replacement", [b"aftEr", b"a"])
def test_reconcile_rejects_corrupted_or_truncated_staged_publication(
    tmp_path: Path,
    replacement: bytes,
) -> None:
    database, repository = _repository(tmp_path / "collections.db")
    capture = _capture(repository)
    store = _store(repository, tmp_path / "private")
    store._after_publish = lambda: (_ for _ in ()).throw(SystemExit())
    with pytest.raises(SystemExit):
        store.save_copy(capture.identity, b"after", "text/plain")
    detail = repository.get_detail(capture.identity)
    assert detail is not None and detail.offline_copy is not None
    staged = detail.offline_copy
    target = store.authority_root / staged.file_id / f"{staged.file_id}.bin"
    target.write_bytes(replacement)
    target.chmod(0o600)

    outcome = store.reconcile_batch(limit=1)

    assert outcome.processed == 1
    detail = repository.get_detail(capture.identity)
    assert detail is not None and detail.offline_copy is not None
    assert detail.offline_copy.state == "failed"
    assert detail.offline_copy.failure_reason == "offline_integrity_failed"
    assert target.exists() is False
    database.close()


def test_reconcile_fails_oversized_staged_publication(tmp_path: Path) -> None:
    database, repository = _repository(tmp_path / "collections.db")
    capture = _capture(repository)
    store = _store(repository, tmp_path / "private", max_copy_bytes=8)
    staged = repository.reserve_offline_copy(
        capture.identity,
        reserved_size=8,
        media_type="text/plain",
        content_hash=_digest(b"12345678"),
        max_copy_bytes=8,
        max_authority_bytes=4096,
    )
    file_root = store.authority_root / staged.file_id
    file_root.mkdir(mode=0o700)
    target = file_root / f"{staged.file_id}.bin"
    atomic_private_write_bytes(
        target,
        b"123456789",
        application_owned_directory=file_root,
        target_precondition=PrivateFileWritePrecondition.missing(),
    )

    outcome = store.reconcile_batch(limit=1)

    assert outcome.processed == 1
    detail = repository.get_detail(capture.identity)
    assert detail is not None and detail.offline_copy is not None
    assert detail.offline_copy.state == "failed"
    assert detail.offline_copy.failure_reason == "offline_integrity_failed"
    assert target.exists() is False
    database.close()


@pytest.mark.parametrize("unlink_before_recovery", [False, True])
def test_reconcile_completes_interrupted_copy_purge(
    tmp_path: Path,
    unlink_before_recovery: bool,
) -> None:
    database, repository = _repository(tmp_path / "collections.db")
    capture = _capture(repository)
    store = _store(repository, tmp_path / "private")
    saved = store.save_copy(capture.identity, b"purging", "text/plain")
    path = store.authority_root / saved.file_id / f"{saved.file_id}.bin"
    repository.begin_offline_copy_purge(
        capture.identity,
        file_id=saved.file_id,
        expected_revision=saved.revision,
    )
    if unlink_before_recovery:
        path.unlink()

    outcome = store.reconcile_batch(limit=1)

    assert outcome.processed == 1
    detail = repository.get_detail(capture.identity)
    assert detail is not None and detail.offline_copy is None
    assert path.exists() is False
    database.close()


def test_reconcile_marks_missing_ready_and_completes_capture_purge(
    tmp_path: Path,
) -> None:
    database, repository = _repository(tmp_path / "collections.db")
    missing = _capture(repository, "missing")
    purged = _capture(repository, "purged")
    store = _store(repository, tmp_path / "private")
    missing_copy = store.save_copy(missing.identity, b"missing", "text/plain")
    purged_copy = store.save_copy(purged.identity, b"purged", "text/plain")
    (store.authority_root / missing_copy.file_id / f"{missing_copy.file_id}.bin").unlink()
    store.hard_delete(purged.identity, expected_revision=purged.revision)
    assert repository.get_detail(purged.identity) is None
    assert (
        store.authority_root / purged_copy.file_id / f"{purged_copy.file_id}.bin"
    ).exists() is False
    with database.connection() as connection:
        assert connection.execute(
            "SELECT 1 FROM collection_capture_items WHERE authority_key = ? "
            "AND capture_id = ?",
            (repository.authority_key, purged.identity.capture_id),
        ).fetchone() is None

    outcomes = []
    for _ in range(6):
        outcome = store.reconcile_batch(limit=1)
        outcomes.append(outcome)
        if not outcome.has_more:
            break

    missing_detail = repository.get_detail(missing.identity)
    assert missing_detail is not None
    assert missing_detail.offline_copy is not None
    assert missing_detail.offline_copy.state == "failed"
    assert repository.get_detail(purged.identity) is None
    purged_path = (
        store.authority_root / purged_copy.file_id / f"{purged_copy.file_id}.bin"
    )
    assert purged_path.exists() is False
    with database.connection() as connection:
        assert connection.execute(
            "SELECT 1 FROM collection_capture_items WHERE authority_key = ? "
            "AND capture_id = ?",
            (repository.authority_key, purged.identity.capture_id),
        ).fetchone() is None
        cursor = connection.execute(
            "SELECT cursor_kind, cursor_value FROM collection_capture_scavenge_state "
            "WHERE authority_key = ?",
            (repository.authority_key,),
        ).fetchone()
    assert cursor is not None
    assert any(outcome.processed == 1 for outcome in outcomes)
    database.close()


@pytest.mark.parametrize("relative_path", ["/tmp/outside.bin", "../outside.bin"])
def test_hard_delete_finishes_when_relative_metadata_is_tampered(
    tmp_path: Path,
    relative_path: str,
) -> None:
    database, repository = _repository(tmp_path / "collections.db")
    capture = _capture(repository)
    store = _store(repository, tmp_path / "private")
    saved = store.save_copy(capture.identity, b"private", "text/plain")
    target = store.authority_root / saved.file_id / f"{saved.file_id}.bin"
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    with database.transaction() as connection:
        connection.execute(
            "UPDATE collection_capture_offline_files SET relative_path = ? "
            "WHERE authority_key = ? AND file_id = ?",
            (relative_path, repository.authority_key, saved.file_id),
        )

    result = store.hard_delete(
        capture.identity,
        expected_revision=capture.revision,
    )

    assert result.success is True
    assert target.exists() is False
    assert outside.read_bytes() == b"outside"
    with database.connection() as connection:
        assert connection.execute(
            "SELECT 1 FROM collection_capture_items WHERE authority_key = ? "
            "AND capture_id = ?",
            (repository.authority_key, capture.identity.capture_id),
        ).fetchone() is None
    database.close()


def test_reconcile_removes_abandoned_atomic_temporary_file(tmp_path: Path) -> None:
    database, repository = _repository(tmp_path / "collections.db")
    capture = _capture(repository)
    store = _store(repository, tmp_path / "private")
    saved = store.save_copy(capture.identity, b"ready", "text/plain")
    file_root = store.authority_root / saved.file_id
    temporary = file_root / f".{saved.file_id}.deadbeef.tmp"
    atomic_private_write_bytes(
        temporary,
        b"abandoned",
        application_owned_directory=file_root,
        target_precondition=PrivateFileWritePrecondition.missing(),
    )

    outcome = store.reconcile_batch(limit=1)

    assert outcome.processed == 1
    assert temporary.exists() is False
    assert store.open_copy(capture.identity) == b"ready"
    database.close()


def test_reconcile_cursor_never_processes_more_than_batch_limit(tmp_path: Path) -> None:
    database, repository = _repository(tmp_path / "collections.db")
    store = _store(repository, tmp_path / "private")
    for index in range(5):
        capture = _capture(repository, f"bounded-{index}")
        store.save_copy(capture.identity, f"body-{index}".encode(), "text/plain")

    outcomes = []
    for _ in range(6):
        outcome = store.reconcile_batch(limit=2)
        outcomes.append(outcome)
        assert outcome.processed <= 2
        if not outcome.has_more:
            break

    assert sum(outcome.processed for outcome in outcomes) == 5
    file_cursors = [
        outcome.cursor_value
        for outcome in outcomes
        if outcome.cursor_kind == "files" and outcome.processed
    ]
    assert file_cursors == sorted(file_cursors)
    assert outcomes[-1].has_more is False
    database.close()


def test_reconcile_rejects_batch_larger_than_fixed_maximum(tmp_path: Path) -> None:
    database, repository = _repository(tmp_path / "collections.db")
    store = _store(repository, tmp_path / "private")

    with pytest.raises(CollectionsCaptureError) as caught:
        store.reconcile_batch(limit=101)

    assert caught.value.reason == "invalid_reconcile_limit"
    database.close()
