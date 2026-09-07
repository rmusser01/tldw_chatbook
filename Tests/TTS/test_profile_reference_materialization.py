from __future__ import annotations

import asyncio
import hashlib
import os
import stat
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from tldw_chatbook.TTS import profile_reference_materialization as module
from tldw_chatbook.TTS.profile_reference_materialization import (
    TTSCloneMaterializationError,
    TTSCloneReferenceMaterialization,
    TTSCloneReferenceMaterializer,
)
from tldw_chatbook.TTS.profile_reference_types import (
    TTSCloneReference,
    TTSCloneReferenceSummary,
)

fcntl = pytest.importorskip(
    "fcntl",
    reason="POSIX profile-reference materialization contracts require fcntl",
)

_NOW = datetime(2026, 8, 10, tzinfo=UTC)
_PRIVATE_TEXT = "PRIVATE MATERIALIZATION TRANSCRIPT"
_PRIVATE_PATH = "/private/user/reference.wav"


def _reference() -> TTSCloneReference:
    wav_bytes = b"canonical-reference-wav"
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


def _exception_graph(error: BaseException) -> str:
    seen: set[int] = set()
    pending: list[BaseException] = [error]
    rendered: list[str] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        rendered.extend((str(current), repr(current), repr(current.args)))
        rendered.extend(getattr(current, "__notes__", ()))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    return "\n".join(rendered)


@pytest.mark.asyncio
async def test_materializer_is_lazy_and_publishes_one_private_opaque_owner(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime" / "clone-references"
    materializer = TTSCloneReferenceMaterializer(root)
    assert not root.exists()

    handle = await materializer.materialize(_reference())

    assert root.is_dir()
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert handle.voice_ref.read_bytes() == _reference().wav_bytes
    assert handle.reference_text == _PRIVATE_TEXT
    assert stat.S_IMODE(handle.voice_ref.stat().st_mode) == 0o600
    assert handle.voice_ref.parent.name.startswith("clone-v1-")
    assert _PRIVATE_TEXT not in str(handle.voice_ref)
    assert str(_reference().summary.reference_id) not in str(handle.voice_ref)
    assert _reference().sha256 not in str(handle.voice_ref)
    assert _PRIVATE_TEXT not in repr(handle)
    assert str(handle.voice_ref) not in repr(handle)
    assert materializer.owns(handle)

    await handle.aclose()
    assert not handle.voice_ref.parent.exists()
    assert not materializer.owns(handle)
    await materializer.close()


@pytest.mark.asyncio
async def test_live_owner_lock_excludes_an_independent_process_lock(
    tmp_path: Path,
) -> None:
    materializer = TTSCloneReferenceMaterializer(tmp_path / "runtime")
    handle = await materializer.materialize(_reference())
    lock_path = handle.voice_ref.parent / "owner.lock"
    contender = os.open(lock_path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(contender)
        await materializer.close()


@pytest.mark.asyncio
async def test_close_is_idempotent_seals_creation_and_removes_active_owners(
    tmp_path: Path,
) -> None:
    materializer = TTSCloneReferenceMaterializer(tmp_path / "runtime")
    handle = await materializer.materialize(_reference())

    await materializer.close()
    await materializer.close()

    assert not handle.voice_ref.parent.exists()
    with pytest.raises(TTSCloneMaterializationError) as caught:
        await materializer.materialize(_reference())
    assert caught.value.code == "closed"


@pytest.mark.asyncio
async def test_first_use_removes_only_recognized_unlocked_orphan(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime"
    root.mkdir(mode=0o700)
    orphan = root / ("clone-v1-" + "1" * 32)
    orphan.mkdir(mode=0o700)
    (orphan / "owner.lock").write_bytes(b"")
    os.chmod(orphan / "owner.lock", 0o600)
    asset = orphan / ("asset-" + "2" * 32 + ".wav")
    asset.write_bytes(_reference().wav_bytes)
    os.chmod(asset, 0o600)

    materializer = TTSCloneReferenceMaterializer(root)
    handle = await materializer.materialize(_reference())

    assert not orphan.exists()
    await handle.aclose()
    await materializer.close()


@pytest.mark.asyncio
async def test_first_use_removes_only_exact_empty_interrupted_publication(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime"
    root.mkdir(mode=0o700)
    interrupted = root / ("clone-v1-" + "8" * 32)
    interrupted.mkdir(mode=0o700)
    unknown = root / (".clone-staging-" + "9" * 32)
    unknown.mkdir(mode=0o700)

    materializer = TTSCloneReferenceMaterializer(root)
    handle = await materializer.materialize(_reference())

    assert not interrupted.exists()
    assert unknown.exists()
    await handle.aclose()
    await materializer.close()


@pytest.mark.asyncio
async def test_first_use_preserves_live_locked_recognized_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime"
    root.mkdir(mode=0o700)
    live = root / ("clone-v1-" + "3" * 32)
    live.mkdir(mode=0o700)
    lock_path = live / "owner.lock"
    lock_path.write_bytes(b"")
    os.chmod(lock_path, 0o600)
    owner_fd = os.open(lock_path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    fcntl.flock(owner_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        materializer = TTSCloneReferenceMaterializer(root)
        handle = await materializer.materialize(_reference())
        assert live.exists()
        await handle.aclose()
        await materializer.close()
    finally:
        os.close(owner_fd)


@pytest.mark.asyncio
async def test_first_use_preserves_unknown_symlink_and_unrecognized_entries(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime"
    root.mkdir(mode=0o700)
    unknown = root / "old-clone-output"
    unknown.mkdir()
    target = tmp_path / "do-not-touch"
    target.mkdir()
    marker = target / "marker"
    marker.write_text("keep")
    link = root / ("clone-v1-" + "4" * 32)
    link.symlink_to(target, target_is_directory=True)
    hostile = root / ("clone-v1-" + "5" * 32)
    hostile.mkdir(mode=0o700)
    (hostile / "owner.lock").symlink_to(marker)

    materializer = TTSCloneReferenceMaterializer(root)
    handle = await materializer.materialize(_reference())

    assert unknown.exists()
    assert link.is_symlink()
    assert hostile.exists()
    assert marker.read_text() == "keep"
    await handle.aclose()
    await materializer.close()


@pytest.mark.asyncio
async def test_first_use_preserves_recognized_owner_with_hardlinked_asset(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime"
    root.mkdir(mode=0o700)
    hostile = root / ("clone-v1-" + "6" * 32)
    hostile.mkdir(mode=0o700)
    lock = hostile / "owner.lock"
    lock.write_bytes(b"")
    os.chmod(lock, 0o600)
    external = tmp_path / "shared-reference.wav"
    external.write_bytes(_reference().wav_bytes)
    os.chmod(external, 0o600)
    linked_asset = hostile / ("asset-" + "7" * 32 + ".wav")
    os.link(external, linked_asset)

    materializer = TTSCloneReferenceMaterializer(root)
    handle = await materializer.materialize(_reference())

    assert hostile.exists()
    assert linked_asset.exists()
    assert external.read_bytes() == _reference().wav_bytes
    await handle.aclose()
    await materializer.close()


@pytest.mark.asyncio
async def test_first_use_preserves_fifo_and_removes_empty_interrupted_owner(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime"
    root.mkdir(mode=0o700)
    fifo_owner = root / ("clone-v1-" + "8" * 32)
    fifo_owner.mkdir(mode=0o700)
    fifo_lock = fifo_owner / "owner.lock"
    fifo_lock.write_bytes(b"")
    os.chmod(fifo_lock, 0o600)
    fifo_asset = fifo_owner / ("asset-" + "9" * 32 + ".wav")
    os.mkfifo(fifo_asset, mode=0o600)
    incomplete = root / ("clone-v1-" + "a" * 32)
    incomplete.mkdir(mode=0o700)

    materializer = TTSCloneReferenceMaterializer(root)
    handle = await materializer.materialize(_reference())

    assert fifo_owner.exists()
    assert stat.S_ISFIFO(fifo_asset.lstat().st_mode)
    assert not incomplete.exists()
    await handle.aclose()
    await materializer.close()


def test_orphan_sweep_never_opens_a_fifo_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "runtime"
    root.mkdir(mode=0o700)
    owner = root / ("clone-v1-" + "b" * 32)
    owner.mkdir(mode=0o700)
    lock = owner / "owner.lock"
    lock.write_bytes(b"")
    os.chmod(lock, 0o600)
    asset_name = "asset-" + "c" * 32 + ".wav"
    os.mkfifo(owner / asset_name, mode=0o600)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    real_open = os.open

    def guarded_open(file: object, flags: int, *args: object, **kwargs: object):
        if file == asset_name:
            raise AssertionError("sweep opened an unqualified FIFO")
        return real_open(file, flags, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", guarded_open)
    try:
        module._sweep_orphans(root_fd)
    finally:
        os.close(root_fd)

    assert owner.exists()
    assert stat.S_ISFIFO((owner / asset_name).lstat().st_mode)


@pytest.mark.asyncio
async def test_concurrent_first_materializations_share_exactly_one_sweep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_prepare = module._prepare_runtime_root_sync
    calls = 0
    guard = threading.Lock()

    def counted_prepare(*args: object, **kwargs: object):
        nonlocal calls
        with guard:
            calls += 1
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(module, "_prepare_runtime_root_sync", counted_prepare)
    materializer = TTSCloneReferenceMaterializer(tmp_path / "runtime")

    first, second = await asyncio.gather(
        materializer.materialize(_reference()),
        materializer.materialize(_reference()),
    )

    assert calls == 1
    assert first.voice_ref != second.voice_ref
    await asyncio.gather(first.aclose(), second.aclose())
    await materializer.close()


def test_operation_file_and_directory_open_policies_require_nofollow() -> None:
    assert module._DIRECTORY_FLAGS & os.O_NOFOLLOW
    assert module._FILE_FLAGS & os.O_NOFOLLOW


@pytest.mark.asyncio
async def test_current_user_permissive_root_is_narrowed_before_use(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime"
    root.mkdir(mode=0o755)

    materializer = TTSCloneReferenceMaterializer(root)
    handle = await materializer.materialize(_reference())

    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    await handle.aclose()
    await materializer.close()


@pytest.mark.asyncio
async def test_symlink_root_fails_closed_without_touching_target(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    marker = target / "marker"
    marker.write_text("keep")
    root = tmp_path / "runtime"
    root.symlink_to(target, target_is_directory=True)
    materializer = TTSCloneReferenceMaterializer(root)

    with pytest.raises(TTSCloneMaterializationError) as caught:
        await materializer.materialize(_reference())

    assert caught.value.code == "unavailable"
    assert marker.read_text() == "keep"
    await materializer.close()


@pytest.mark.asyncio
async def test_owner_qualification_failure_preserves_existing_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "runtime"
    root.mkdir(mode=0o700)
    marker = root / "unknown"
    marker.write_text("keep")
    root_identity = (root.stat().st_dev, root.stat().st_ino)
    monkeypatch.setattr(
        module,
        "_owned_by_effective_user",
        lambda info: (info.st_dev, info.st_ino) != root_identity,
    )
    materializer = TTSCloneReferenceMaterializer(root)

    with pytest.raises(TTSCloneMaterializationError):
        await materializer.materialize(_reference())

    assert marker.read_text() == "keep"
    await materializer.close()


@pytest.mark.asyncio
async def test_root_substitution_after_sweep_fails_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "runtime"
    moved = tmp_path / "prepared-runtime"
    replacement_marker = root / "replacement-marker"
    real_create = module._create_materialization_sync

    def substitute_then_create(
        selected_root: Path,
        expected_identity: tuple[int, int] | None,
        wav_bytes: bytes,
    ):
        selected_root.rename(moved)
        selected_root.mkdir(mode=0o700)
        replacement_marker.write_text("keep")
        return real_create(selected_root, expected_identity, wav_bytes)

    monkeypatch.setattr(module, "_create_materialization_sync", substitute_then_create)
    materializer = TTSCloneReferenceMaterializer(root)

    with pytest.raises(TTSCloneMaterializationError) as caught:
        await materializer.materialize(_reference())

    assert caught.value.code == "unavailable"
    assert replacement_marker.read_text() == "keep"
    assert not tuple(moved.iterdir())
    await materializer.close()


@pytest.mark.asyncio
async def test_cancellation_joins_worker_and_removes_late_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    real_create = module._create_materialization_sync
    loop = asyncio.get_running_loop()

    def blocked_create(*args: object, **kwargs: object):
        loop.call_soon_threadsafe(entered.set)
        asyncio.run_coroutine_threadsafe(release.wait(), loop).result()
        return real_create(*args, **kwargs)

    monkeypatch.setattr(module, "_create_materialization_sync", blocked_create)
    materializer = TTSCloneReferenceMaterializer(tmp_path / "runtime")
    operation = asyncio.create_task(materializer.materialize(_reference()))
    await entered.wait()
    operation.cancel()
    await asyncio.sleep(0)
    assert not operation.done()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await operation

    root = tmp_path / "runtime"
    assert not root.exists() or tuple(root.iterdir()) == ()
    await materializer.close()


@pytest.mark.asyncio
async def test_two_materializers_cannot_sweep_a_concurrently_publishing_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "runtime"
    first_lock_entered = threading.Event()
    release_first_lock = threading.Event()
    real_lock = module._lock_exclusive_nonblocking
    first_call = True

    def block_first_lock(descriptor: int) -> None:
        nonlocal first_call
        if first_call:
            first_call = False
            first_lock_entered.set()
            assert release_first_lock.wait(timeout=2.0)
        real_lock(descriptor)

    monkeypatch.setattr(module, "_lock_exclusive_nonblocking", block_first_lock)
    first = TTSCloneReferenceMaterializer(root)
    second = TTSCloneReferenceMaterializer(root)
    first_task = asyncio.create_task(first.materialize(_reference()))
    second_task: asyncio.Task[TTSCloneReferenceMaterialization] | None = None
    first_handle = None
    second_handle = None
    try:
        assert await asyncio.to_thread(first_lock_entered.wait, 2.0)
        second_task = asyncio.create_task(second.materialize(_reference()))
        await asyncio.sleep(0.01)
        assert not second_task.done()
        release_first_lock.set()
        first_handle = await first_task
        second_handle = await second_task

        assert (await first_handle.validated_voice_ref()).read_bytes() == (
            _reference().wav_bytes
        )
        assert (await second_handle.validated_voice_ref()).read_bytes() == (
            _reference().wav_bytes
        )
    finally:
        release_first_lock.set()
        if not first_task.done():
            first_task.cancel()
            await asyncio.gather(first_task, return_exceptions=True)
        if second_task is not None and not second_task.done():
            second_task.cancel()
            await asyncio.gather(second_task, return_exceptions=True)
        if first_handle is not None:
            await first_handle.aclose()
        if second_handle is not None:
            await second_handle.aclose()
        await first.close()
        await second.close()


@pytest.mark.asyncio
async def test_validated_voice_ref_rejects_lexical_root_substitution(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime"
    moved = tmp_path / "moved-runtime"
    materializer = TTSCloneReferenceMaterializer(root)
    handle = await materializer.materialize(_reference())
    original_path = handle.voice_ref

    root.rename(moved)
    replacement_owner = root / original_path.parent.name
    replacement_owner.mkdir(parents=True, mode=0o700)
    (replacement_owner / "owner.lock").write_bytes(b"")
    os.chmod(replacement_owner / "owner.lock", 0o600)
    replacement_asset = replacement_owner / original_path.name
    replacement_asset.write_bytes(b"attacker replacement")
    os.chmod(replacement_asset, 0o600)

    with pytest.raises(TTSCloneMaterializationError) as caught:
        await handle.validated_voice_ref()
    assert caught.value.code == "unavailable"
    assert replacement_asset.read_bytes() == b"attacker replacement"

    await handle.aclose()
    assert replacement_asset.exists()
    await materializer.close()


@pytest.mark.asyncio
async def test_close_during_creation_joins_and_cleans_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    real_create = module._create_materialization_sync
    loop = asyncio.get_running_loop()

    def blocked_create(*args: object, **kwargs: object):
        loop.call_soon_threadsafe(entered.set)
        asyncio.run_coroutine_threadsafe(release.wait(), loop).result()
        return real_create(*args, **kwargs)

    monkeypatch.setattr(module, "_create_materialization_sync", blocked_create)
    materializer = TTSCloneReferenceMaterializer(tmp_path / "runtime")
    operation = asyncio.create_task(materializer.materialize(_reference()))
    await entered.wait()
    closing = asyncio.create_task(materializer.close())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    await closing

    with pytest.raises(TTSCloneMaterializationError) as caught:
        await operation
    assert caught.value.code == "closed"
    root = tmp_path / "runtime"
    assert not root.exists() or tuple(root.iterdir()) == ()


@pytest.mark.asyncio
async def test_substituted_asset_is_preserved_and_cleanup_fails_safely(
    tmp_path: Path,
) -> None:
    materializer = TTSCloneReferenceMaterializer(tmp_path / "runtime")
    handle = await materializer.materialize(_reference())
    asset = handle.voice_ref
    asset.unlink()
    target = tmp_path / "private-target"
    target.write_text("keep")
    asset.symlink_to(target)

    with pytest.raises(TTSCloneMaterializationError) as caught:
        await handle.aclose()

    assert caught.value.code == "cleanup_failed"
    assert target.read_text() == "keep"
    assert asset.is_symlink()
    with pytest.raises(TTSCloneMaterializationError):
        await materializer.close()


@pytest.mark.asyncio
async def test_raw_worker_failure_is_normalized_outside_exception_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_private(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(f"{_PRIVATE_PATH} {_PRIVATE_TEXT}")

    monkeypatch.setattr(module, "_create_materialization_sync", fail_private)
    materializer = TTSCloneReferenceMaterializer(tmp_path / "runtime")

    with pytest.raises(TTSCloneMaterializationError) as caught:
        await materializer.materialize(_reference())

    rendered = _exception_graph(caught.value)
    assert caught.value.code == "unavailable"
    assert _PRIVATE_PATH not in rendered
    assert _PRIVATE_TEXT not in rendered
    assert _reference().sha256 not in rendered
    assert repr(_reference().wav_bytes) not in rendered
    await materializer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target",
    ["_lock_exclusive_nonblocking", "write", "fsync"],
)
async def test_creation_syscall_failures_are_private_and_leave_no_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    def fail_private(*_args: object, **_kwargs: object) -> object:
        raise OSError(f"{_PRIVATE_PATH} {_PRIVATE_TEXT} {_reference().sha256}")

    if target == "_lock_exclusive_nonblocking":
        monkeypatch.setattr(module, target, fail_private)
    else:
        monkeypatch.setattr(module.os, target, fail_private)
    root = tmp_path / "runtime"
    materializer = TTSCloneReferenceMaterializer(root)

    with pytest.raises(TTSCloneMaterializationError) as caught:
        await materializer.materialize(_reference())

    rendered = _exception_graph(caught.value)
    assert caught.value.code == "unavailable"
    assert _PRIVATE_PATH not in rendered
    assert _PRIVATE_TEXT not in rendered
    assert _reference().sha256 not in rendered
    assert not root.exists() or tuple(root.iterdir()) == ()
    await materializer.close()


@pytest.mark.asyncio
async def test_cleanup_worker_failure_is_private_and_retriable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = TTSCloneReferenceMaterializer(tmp_path / "runtime")
    handle = await materializer.materialize(_reference())
    real_cleanup = module._cleanup_materialization_sync

    def fail_private(*_args: object, **_kwargs: object) -> object:
        raise OSError(f"{_PRIVATE_PATH} {_PRIVATE_TEXT}")

    monkeypatch.setattr(module, "_cleanup_materialization_sync", fail_private)
    with pytest.raises(TTSCloneMaterializationError) as caught:
        await handle.aclose()
    assert caught.value.code == "cleanup_failed"
    assert _PRIVATE_PATH not in _exception_graph(caught.value)
    assert _PRIVATE_TEXT not in _exception_graph(caught.value)

    monkeypatch.setattr(module, "_cleanup_materialization_sync", real_cleanup)
    await materializer.close()
    assert not handle.voice_ref.parent.exists()


@pytest.mark.asyncio
async def test_cleanup_retries_after_failure_following_entry_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materializer = TTSCloneReferenceMaterializer(tmp_path / "runtime")
    handle = await materializer.materialize(_reference())
    real_fsync = os.fsync
    failed = False

    def fail_first_cleanup_fsync(descriptor: int) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError(f"{_PRIVATE_PATH} cleanup fsync")
        real_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", fail_first_cleanup_fsync)
    with pytest.raises(TTSCloneMaterializationError) as caught:
        await handle.aclose()
    assert caught.value.code == "cleanup_failed"
    assert materializer.owns(handle)

    monkeypatch.setattr(module.os, "fsync", real_fsync)
    await handle.aclose()
    assert not materializer.owns(handle)
    assert not handle.voice_ref.parent.exists()
    await materializer.close()


@pytest.mark.asyncio
async def test_concurrent_handle_close_has_one_exact_terminal_cleanup(
    tmp_path: Path,
) -> None:
    materializer = TTSCloneReferenceMaterializer(tmp_path / "runtime")
    handle = await materializer.materialize(_reference())

    await asyncio.gather(handle.aclose(), handle.aclose())

    assert not materializer.owns(handle)
    assert not handle.voice_ref.parent.exists()
    await materializer.close()


@pytest.mark.asyncio
async def test_non_posix_platform_fails_with_explicit_safe_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "_POSIX_SUPPORTED", False)
    materializer = TTSCloneReferenceMaterializer(tmp_path / "runtime")

    with pytest.raises(TTSCloneMaterializationError) as caught:
        await materializer.materialize(_reference())

    assert caught.value.code == "unsupported"
    assert not (tmp_path / "runtime").exists()
    await materializer.close()
