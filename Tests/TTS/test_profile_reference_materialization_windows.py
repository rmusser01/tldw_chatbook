"""Windows contracts for operation-scoped clone-reference materialization."""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from tldw_chatbook.TTS import profile_reference_materialization as module
from tldw_chatbook.TTS.profile_reference_materialization import (
    TTSCloneMaterializationError,
    TTSCloneReferenceMaterializer,
)
from tldw_chatbook.TTS.profile_reference_types import (
    TTSCloneReference,
    TTSCloneReferenceSummary,
)
from tldw_chatbook.TTS.windows_artifact_fs import (
    WindowsArtifactError,
    WindowsFileIdentity,
)


_NOW = datetime(2026, 8, 14, tzinfo=UTC)
_PRIVATE_TEXT = "PRIVATE WINDOWS CLONE TRANSCRIPT"


def _reference() -> TTSCloneReference:
    wav_bytes = b"windows-clone-reference-wav"
    return TTSCloneReference(
        summary=TTSCloneReferenceSummary(
            reference_id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
            byte_length=len(wav_bytes),
            duration_ms=250,
            sample_rate_hz=24_000,
            channels=1,
            sample_encoding="pcm_s16le",
            created_at=_NOW,
            updated_at=_NOW,
        ),
        reference_text=_PRIVATE_TEXT,
        sha256=hashlib.sha256(wav_bytes).hexdigest(),
        wav_bytes=wav_bytes,
    )


class _FakeWindowsHandle:
    def __init__(self, filesystem: _FakeWindowsFilesystem, path: Path, kind: str):
        self.filesystem = filesystem
        self.path = path
        self.kind = kind
        self.identity = self._current_identity()
        self.closed = False
        self.deleted = False
        self.locked = False
        self.close_failures = 0

    def _current_identity(self) -> WindowsFileIdentity:
        info = self.path.stat()
        return WindowsFileIdentity(
            volume_serial_number=info.st_dev,
            file_id=info.st_ino.to_bytes(16, "little", signed=False),
            kind=self.kind,  # type: ignore[arg-type]
            reparse_tag=0,
        )

    @property
    def privacy_posture(self) -> str:
        return "windows_account_protected"

    def verify_private_acl(self) -> bool:
        return not self.closed

    def read(self, count: int, *, offset: int = 0) -> bytes:
        return self.path.read_bytes()[offset : offset + count]

    def lock_exclusive_nonblocking(self) -> None:
        key = self.identity.file_id
        if key in self.filesystem.locks and not self.locked:
            raise WindowsArtifactError("busy")
        self.filesystem.locks.add(key)
        self.locked = True

    def unlock(self) -> None:
        if self.locked:
            self.filesystem.locks.discard(self.identity.file_id)
            self.locked = False

    def delete_exact(self) -> None:
        if self.closed or self.deleted:
            return
        if not self.path.exists() or self._current_identity() != self.identity:
            raise WindowsArtifactError("changed")
        self.deleted = True

    def close(self) -> None:
        if self.closed:
            return
        if self.close_failures:
            self.close_failures -= 1
            raise WindowsArtifactError("cleanup_failed", cleanup_owner=self)  # type: ignore[arg-type]
        if self.deleted and self.path.exists():
            if self.kind == "directory":
                self.path.rmdir()
            else:
                self.path.unlink()
        self.closed = True


class _FakeWindowsFilesystem:
    def __init__(self) -> None:
        self.handles: list[_FakeWindowsHandle] = []
        self.locks: set[bytes] = set()
        self.fail_asset_creation = False
        self.owner_close_failures = 0
        self.root_close_failures = 0

    def _handle(self, path: Path, kind: str) -> _FakeWindowsHandle:
        handle = _FakeWindowsHandle(self, path, kind)
        self.handles.append(handle)
        return handle

    def protect_private_directory(self, path: Path) -> _FakeWindowsHandle:
        return self._handle(Path(path), "directory")

    def pin_directory_no_reparse(self, path: Path) -> _FakeWindowsHandle:
        return self._handle(Path(path), "directory")

    def create_private_directory(self, path: Path) -> _FakeWindowsHandle:
        selected = Path(path)
        selected.mkdir()
        handle = self._handle(selected, "directory")
        if selected.name.startswith("clone-v1-"):
            handle.close_failures = self.owner_close_failures
        else:
            handle.close_failures = self.root_close_failures
        return handle

    def create_private_file(
        self,
        path: Path,
        data: bytes,
        *,
        read_only: bool = False,
    ) -> _FakeWindowsHandle:
        if data and self.fail_asset_creation:
            raise WindowsArtifactError("unavailable")
        selected = Path(path)
        selected.write_bytes(data)
        return self._handle(selected, "file")

    def open_file_no_reparse(
        self,
        path: Path,
        *,
        writable: bool = False,
    ) -> _FakeWindowsHandle:
        del writable
        return self._handle(Path(path), "file")


@pytest.fixture
def windows_filesystem(monkeypatch: pytest.MonkeyPatch) -> _FakeWindowsFilesystem:
    filesystem = _FakeWindowsFilesystem()
    monkeypatch.setattr(module, "_windows_artifact_filesystem", filesystem)
    return filesystem


@pytest.mark.asyncio
async def test_windows_materialization_is_private_locked_and_exact(
    tmp_path: Path,
    windows_filesystem: _FakeWindowsFilesystem,
) -> None:
    materializer = TTSCloneReferenceMaterializer(tmp_path / "runtime")

    handle = await materializer.materialize(_reference())

    assert handle.voice_ref.read_bytes() == _reference().wav_bytes
    assert handle.reference_text == _PRIVATE_TEXT
    assert _PRIVATE_TEXT not in repr(handle)
    assert not any(
        _PRIVATE_TEXT in path.read_text(errors="ignore")
        for path in tmp_path.rglob("*.*")
    )
    assert all(
        owner.verify_private_acl()
        for owner in windows_filesystem.handles
        if not owner.closed
    )
    record = handle._record
    assert record.lock_handle.locked
    with pytest.raises(WindowsArtifactError) as busy:
        contender = windows_filesystem.open_file_no_reparse(
            handle.voice_ref.parent / "owner.lock"
        )
        contender.lock_exclusive_nonblocking()
    assert busy.value.code == "busy"
    assert await handle.validated_voice_ref() == handle.voice_ref

    await handle.aclose()
    assert not handle.voice_ref.parent.exists()
    assert not materializer.owns(handle)
    await materializer.close()


@pytest.mark.asyncio
async def test_windows_asset_substitution_is_preserved_and_rejected(
    tmp_path: Path,
    windows_filesystem: _FakeWindowsFilesystem,
) -> None:
    materializer = TTSCloneReferenceMaterializer(tmp_path / "runtime")
    handle = await materializer.materialize(_reference())
    asset = handle.voice_ref
    asset.unlink()
    asset.write_bytes(_reference().wav_bytes)

    with pytest.raises(TTSCloneMaterializationError) as validation:
        await handle.validated_voice_ref()
    with pytest.raises(TTSCloneMaterializationError) as cleanup:
        await handle.aclose()

    assert validation.value.code == "unavailable"
    assert cleanup.value.code == "cleanup_failed"
    assert asset.read_bytes() == _reference().wav_bytes
    assert materializer.owns(handle)


@pytest.mark.asyncio
async def test_windows_root_substitution_is_preserved_and_rejected(
    tmp_path: Path,
    windows_filesystem: _FakeWindowsFilesystem,
) -> None:
    root = tmp_path / "runtime"
    moved = tmp_path / "moved-runtime"
    materializer = TTSCloneReferenceMaterializer(root)
    handle = await materializer.materialize(_reference())
    owner_name = handle.voice_ref.parent.name
    asset_name = handle.voice_ref.name
    root.rename(moved)
    replacement = root / owner_name
    replacement.mkdir(parents=True)
    (replacement / "owner.lock").write_bytes(b"")
    replacement_asset = replacement / asset_name
    replacement_asset.write_bytes(_reference().wav_bytes)

    with pytest.raises(TTSCloneMaterializationError) as validation:
        await handle.validated_voice_ref()
    with pytest.raises(TTSCloneMaterializationError) as cleanup:
        await handle.aclose()

    assert validation.value.code == "unavailable"
    assert cleanup.value.code == "cleanup_failed"
    assert replacement_asset.read_bytes() == _reference().wav_bytes


@pytest.mark.asyncio
async def test_windows_cleanup_failure_retains_exact_owner_for_retry(
    tmp_path: Path,
    windows_filesystem: _FakeWindowsFilesystem,
) -> None:
    materializer = TTSCloneReferenceMaterializer(tmp_path / "runtime")
    handle = await materializer.materialize(_reference())
    handle._record.owner_handle.close_failures = 1

    with pytest.raises(TTSCloneMaterializationError) as first:
        await handle.aclose()

    assert first.value.code == "cleanup_failed"
    assert materializer.owns(handle)
    await handle.aclose()
    assert not materializer.owns(handle)
    assert not handle.voice_ref.parent.exists()
    await materializer.close()


@pytest.mark.asyncio
async def test_windows_materializer_close_retries_retained_owner(
    tmp_path: Path,
    windows_filesystem: _FakeWindowsFilesystem,
) -> None:
    materializer = TTSCloneReferenceMaterializer(tmp_path / "runtime")
    handle = await materializer.materialize(_reference())
    handle._record.owner_handle.close_failures = 1

    with pytest.raises(TTSCloneMaterializationError) as first:
        await materializer.close()

    assert first.value.code == "cleanup_failed"
    assert materializer.owns(handle)
    await materializer.close()
    assert not materializer.owns(handle)
    assert not handle.voice_ref.parent.exists()


@pytest.mark.asyncio
async def test_windows_orphan_sweep_removes_only_unlocked_recognized_owner(
    tmp_path: Path,
    windows_filesystem: _FakeWindowsFilesystem,
) -> None:
    root = tmp_path / "runtime"
    root.mkdir()
    orphan = root / ("clone-v1-" + "1" * 32)
    orphan.mkdir()
    (orphan / "owner.lock").write_bytes(b"")
    (orphan / ("asset-" + "2" * 32 + ".wav")).write_bytes(b"orphan")
    live = root / ("clone-v1-" + "3" * 32)
    live.mkdir()
    live_lock_path = live / "owner.lock"
    live_lock_path.write_bytes(b"")
    (live / ("asset-" + "4" * 32 + ".wav")).write_bytes(b"live")
    live_lock = windows_filesystem.open_file_no_reparse(live_lock_path, writable=True)
    live_lock.lock_exclusive_nonblocking()

    materializer = TTSCloneReferenceMaterializer(root)
    handle = await materializer.materialize(_reference())

    assert not orphan.exists()
    assert live.exists()
    await handle.aclose()
    await materializer.close()
    live_lock.unlock()
    live_lock.close()


@pytest.mark.asyncio
async def test_windows_unexpected_owner_entry_blocks_cleanup(
    tmp_path: Path,
    windows_filesystem: _FakeWindowsFilesystem,
) -> None:
    materializer = TTSCloneReferenceMaterializer(tmp_path / "runtime")
    handle = await materializer.materialize(_reference())
    foreign = handle.voice_ref.parent / "foreign.txt"
    foreign.write_text("keep", encoding="utf-8")

    with pytest.raises(TTSCloneMaterializationError) as raised:
        await handle.aclose()

    assert raised.value.code == "cleanup_failed"
    assert foreign.read_text(encoding="utf-8") == "keep"
    assert materializer.owns(handle)


@pytest.mark.asyncio
async def test_windows_cancellation_joins_creation_and_removes_publication(
    tmp_path: Path,
    windows_filesystem: _FakeWindowsFilesystem,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    real_create = module._create_windows_materialization_sync

    def blocked_create(*args: object, **kwargs: object):
        entered.set()
        assert release.wait(timeout=2.0)
        return real_create(*args, **kwargs)

    monkeypatch.setattr(module, "_create_windows_materialization_sync", blocked_create)
    materializer = TTSCloneReferenceMaterializer(tmp_path / "runtime")
    operation = asyncio.create_task(materializer.materialize(_reference()))
    assert await asyncio.to_thread(entered.wait, 2.0)
    operation.cancel()
    await asyncio.sleep(0)
    assert not operation.done()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await operation

    assert tuple((tmp_path / "runtime").iterdir()) == ()
    await materializer.close()


@pytest.mark.asyncio
async def test_windows_cancelled_cleanup_failure_stays_materializer_owned(
    tmp_path: Path,
    windows_filesystem: _FakeWindowsFilesystem,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    real_create = module._create_windows_materialization_sync
    real_cleanup = module._cleanup_materialization_sync

    def blocked_create(*args: object, **kwargs: object):
        entered.set()
        assert release.wait(timeout=2.0)
        return real_create(*args, **kwargs)

    def failed_cleanup(_record: object) -> None:
        raise OSError("PRIVATE cleanup detail")

    monkeypatch.setattr(module, "_create_windows_materialization_sync", blocked_create)
    monkeypatch.setattr(module, "_cleanup_materialization_sync", failed_cleanup)
    materializer = TTSCloneReferenceMaterializer(tmp_path / "runtime")
    operation = asyncio.create_task(materializer.materialize(_reference()))
    assert await asyncio.to_thread(entered.wait, 2.0)
    operation.cancel()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await operation

    assert len(materializer._active) == 1
    monkeypatch.setattr(module, "_cleanup_materialization_sync", real_cleanup)
    await materializer.close()
    assert tuple((tmp_path / "runtime").iterdir()) == ()


@pytest.mark.asyncio
async def test_windows_partial_creation_cleanup_failure_is_retained(
    tmp_path: Path,
    windows_filesystem: _FakeWindowsFilesystem,
) -> None:
    windows_filesystem.fail_asset_creation = True
    windows_filesystem.owner_close_failures = 1
    materializer = TTSCloneReferenceMaterializer(tmp_path / "runtime")

    with pytest.raises(TTSCloneMaterializationError) as raised:
        await materializer.materialize(_reference())

    assert raised.value.code == "cleanup_failed"
    assert len(materializer._pending_windows_cleanup) == 1
    await materializer.close()
    assert materializer._pending_windows_cleanup == []
    assert tuple((tmp_path / "runtime").iterdir()) == ()


@pytest.mark.asyncio
async def test_windows_prepare_close_failure_is_retried_before_creation(
    tmp_path: Path,
    windows_filesystem: _FakeWindowsFilesystem,
) -> None:
    windows_filesystem.root_close_failures = 1
    materializer = TTSCloneReferenceMaterializer(tmp_path / "runtime")

    with pytest.raises(TTSCloneMaterializationError) as first:
        await materializer.materialize(_reference())

    assert first.value.code == "cleanup_failed"
    assert len(materializer._pending_windows_cleanup) == 1
    windows_filesystem.root_close_failures = 0
    handle = await materializer.materialize(_reference())
    assert materializer._pending_windows_cleanup == []
    await handle.aclose()
    await materializer.close()


@pytest.mark.asyncio
async def test_platform_is_unsupported_only_when_neither_backend_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "_POSIX_SUPPORTED", False)
    monkeypatch.setattr(module, "_windows_artifact_filesystem", None)
    materializer = TTSCloneReferenceMaterializer(tmp_path / "runtime")

    with pytest.raises(TTSCloneMaterializationError) as raised:
        await materializer.materialize(_reference())

    assert raised.value.code == "unsupported"
    await materializer.close()


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows handles")
@pytest.mark.asyncio
async def test_native_windows_materialization_lifecycle(tmp_path: Path) -> None:
    materializer = TTSCloneReferenceMaterializer(tmp_path / "runtime")

    handle = await materializer.materialize(_reference())

    assert await handle.validated_voice_ref() == handle.voice_ref
    assert handle.voice_ref.read_bytes() == _reference().wav_bytes
    assert _PRIVATE_TEXT not in repr(handle)
    await handle.aclose()
    assert not handle.voice_ref.parent.exists()
    await materializer.close()
