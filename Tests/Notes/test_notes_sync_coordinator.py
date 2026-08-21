from __future__ import annotations

import importlib
import stat
import threading
from pathlib import Path
from types import SimpleNamespace

import portalocker
import pytest

from tldw_chatbook.Notes.file_notes_session_owner import SessionBinding


def _coordinator_module():
    return importlib.import_module("tldw_chatbook.Notes.notes_sync_coordinator")


@pytest.mark.parametrize("existing_inside_candidate", [False, True])
def test_candidate_rejects_both_lasting_root_overlap_directions(
    tmp_path: Path,
    existing_inside_candidate: bool,
) -> None:
    module = _coordinator_module()
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    candidate, existing = (
        (outer, inner) if existing_inside_candidate else (inner, outer)
    )

    with pytest.raises(module.RootAdmissionError, match="lasting_root_overlap"):
        module.validate_candidate_root(candidate, lasting_roots=(existing,))


@pytest.mark.parametrize("existing_inside_candidate", [False, True])
def test_candidate_overlap_uses_filesystem_identity_for_case_aliases(
    tmp_path: Path,
    existing_inside_candidate: bool,
) -> None:
    module = _coordinator_module()
    outer = tmp_path / "MixedCase"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    alias_outer = tmp_path / "mixedcase"
    if not alias_outer.exists():
        pytest.skip("filesystem is case-sensitive")
    candidate, existing = (
        (alias_outer, inner)
        if existing_inside_candidate
        else (alias_outer / "inner", outer)
    )

    with pytest.raises(module.RootAdmissionError, match="lasting_root_overlap"):
        module.validate_candidate_root(candidate, lasting_roots=(existing,))


@pytest.mark.parametrize("binding_inside_candidate", [False, True])
def test_candidate_rejects_both_file_notes_overlap_directions(
    tmp_path: Path,
    binding_inside_candidate: bool,
) -> None:
    module = _coordinator_module()
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    candidate, bound = (outer, inner) if binding_inside_candidate else (inner, outer)
    binding = SessionBinding(root_key=str(bound.resolve()), generation=1)

    with pytest.raises(module.RootAdmissionError, match="file_notes_overlap"):
        module.validate_candidate_root(candidate, file_notes_binding=binding)


def test_candidate_rejects_sensitive_root_conflict_without_disclosing_path(
    tmp_path: Path,
) -> None:
    module = _coordinator_module()
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    private = candidate / "PRIVATE-state"

    with pytest.raises(module.RootAdmissionError) as raised:
        module.validate_candidate_root(
            candidate,
            private_conflict=lambda _root: private,
        )

    assert raised.value.reason_code == "private_path_overlap"
    assert "PRIVATE" not in str(raised.value)


def test_candidate_rejects_symlink_missing_and_unsupported_write_capability(
    tmp_path: Path,
) -> None:
    module = _coordinator_module()
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(module.RootAdmissionError, match="root_link_or_reparse"):
        module.validate_candidate_root(linked)
    with pytest.raises(module.RootAdmissionError, match="root_offline"):
        module.validate_candidate_root(tmp_path / "missing")
    with pytest.raises(
        module.RootAdmissionError, match="writable_filesystem_unsupported"
    ):
        module.validate_candidate_root(real, write_supported=False)


def test_candidate_rejects_windows_reparse_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _coordinator_module()
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda _path: SimpleNamespace(
            st_mode=stat.S_IFDIR,
            st_file_attributes=module._REPARSE_ATTRIBUTE,
        ),
    )

    with pytest.raises(module.RootAdmissionError, match="root_link_or_reparse"):
        module.validate_candidate_root(root)


def test_missing_root_is_exposed_as_offline_without_lock_authority(
    tmp_path: Path,
) -> None:
    module = _coordinator_module()
    coordinator = module.NotesSyncRootCoordinator(tmp_path / "locks")

    admission = coordinator.try_acquire(tmp_path / "missing")

    assert admission.state is module.RootAdmissionState.OFFLINE
    assert admission.reason_code == "root_offline"
    assert admission.lease is None
    assert not admission.can_plan and not admission.can_write


def test_root_disappearing_after_validation_becomes_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _coordinator_module()
    root = tmp_path / "root"
    root.mkdir()
    coordinator = module.NotesSyncRootCoordinator(tmp_path / "locks")
    monkeypatch.setattr(
        coordinator,
        "_digest",
        lambda _root: (_ for _ in ()).throw(
            module.RootAdmissionError("root_unavailable")
        ),
    )

    admission = coordinator.try_acquire(root)

    assert admission.state is module.RootAdmissionState.OFFLINE
    assert admission.reason_code == "root_unavailable"


def test_owner_private_digest_lock_and_passive_authority(tmp_path: Path) -> None:
    module = _coordinator_module()
    root = tmp_path / "PRIVATE-root"
    root.mkdir()
    lock_directory = tmp_path / "locks"
    first = module.NotesSyncRootCoordinator(lock_directory)
    second = module.NotesSyncRootCoordinator(lock_directory)

    owner = first.try_acquire(root)
    passive = second.try_acquire(root)

    assert owner.state is module.RootAdmissionState.OWNER
    assert owner.label == "Active in this process"
    assert owner.can_watch and owner.can_plan and owner.can_write
    assert passive.state is module.RootAdmissionState.PASSIVE
    assert passive.label == "Passive in this process"
    assert not passive.can_watch and not passive.can_plan and not passive.can_write
    for operation in ("watch", "plan", "write"):
        with pytest.raises(module.RootAuthorityError, match="passive_process"):
            passive.require_authority(operation)

    lock_files = list(lock_directory.glob("*.lock"))
    assert len(lock_files) == 1
    assert lock_files[0].stem == owner.root_digest
    assert len(lock_files[0].stem) == 64
    int(lock_files[0].stem, 16)
    assert "PRIVATE-root" not in lock_files[0].name
    assert "PRIVATE-root" not in repr(owner)
    if hasattr(stat, "S_IMODE"):
        assert stat.S_IMODE(lock_directory.stat().st_mode) == 0o700
        assert stat.S_IMODE(lock_files[0].stat().st_mode) == 0o600

    first.release(owner.lease)


def test_case_aliases_share_one_os_lock_identity(tmp_path: Path) -> None:
    module = _coordinator_module()
    root = tmp_path / "MixedCase"
    root.mkdir()
    alias = tmp_path / "mixedcase"
    if not alias.exists():
        pytest.skip("filesystem is case-sensitive")
    lock_directory = tmp_path / "locks"
    first = module.NotesSyncRootCoordinator(lock_directory)
    second = module.NotesSyncRootCoordinator(lock_directory)

    owner = first.try_acquire(root)
    passive = second.try_acquire(alias)

    assert owner.root_digest == passive.root_digest
    assert passive.state is module.RootAdmissionState.PASSIVE
    first.release(owner.lease)


def test_lock_backend_failure_is_rejected_not_reported_as_passive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _coordinator_module()
    root = tmp_path / "root"
    root.mkdir()
    coordinator = module.NotesSyncRootCoordinator(tmp_path / "locks")

    monkeypatch.setattr(
        module.portalocker,
        "lock",
        lambda *_args: (_ for _ in ()).throw(
            portalocker.exceptions.LockException("PRIVATE backend failure")
        ),
    )
    admission = coordinator.try_acquire(root)

    assert admission.state is module.RootAdmissionState.REJECTED
    assert admission.reason_code == "lock_unavailable"
    assert "PRIVATE" not in repr(admission)


def test_close_admission_settles_before_unlock_and_blocks_new_work(
    tmp_path: Path,
) -> None:
    module = _coordinator_module()
    root = tmp_path / "root"
    root.mkdir()
    lock_directory = tmp_path / "locks"
    owner = module.NotesSyncRootCoordinator(lock_directory)
    contender = module.NotesSyncRootCoordinator(lock_directory)
    admission = owner.try_acquire(root)
    events: list[str] = []

    def settle() -> None:
        events.append("settle_started")
        assert not admission.can_watch
        with pytest.raises(module.RootAuthorityError, match="admission_closed"):
            admission.require_authority("write")
        assert owner.try_acquire(root).state is module.RootAdmissionState.PASSIVE
        assert contender.try_acquire(root).state is module.RootAdmissionState.PASSIVE
        events.append("settle_finished")

    owner.close_admission(admission.lease, settle)
    events.append("closed")

    assert events == ["settle_started", "settle_finished", "closed"]
    assert contender.try_acquire(root).state is module.RootAdmissionState.OWNER


def test_awaitable_settlement_never_releases_before_it_runs(tmp_path: Path) -> None:
    module = _coordinator_module()
    root = tmp_path / "root"
    root.mkdir()
    lock_directory = tmp_path / "locks"
    owner = module.NotesSyncRootCoordinator(lock_directory)
    contender = module.NotesSyncRootCoordinator(lock_directory)
    admission = owner.try_acquire(root)

    async def settle_later() -> None:
        raise AssertionError("must not be silently discarded")

    with pytest.raises(module.RootCoordinatorError, match="settlement_not_completed"):
        owner.close_admission(admission.lease, settle_later)

    assert contender.try_acquire(root).state is module.RootAdmissionState.PASSIVE
    owner.release(admission.lease)


def test_release_is_idempotent_and_revokes_authority(tmp_path: Path) -> None:
    module = _coordinator_module()
    root = tmp_path / "root"
    root.mkdir()
    coordinator = module.NotesSyncRootCoordinator(tmp_path / "locks")
    admission = coordinator.try_acquire(root)

    coordinator.release(admission.lease)
    coordinator.release(admission.lease)

    assert not admission.can_write
    with pytest.raises(module.RootAuthorityError, match="admission_closed"):
        admission.require_authority("write")


def test_replacing_live_lock_file_revokes_old_owner_before_new_owner(
    tmp_path: Path,
) -> None:
    module = _coordinator_module()
    root = tmp_path / "root"
    root.mkdir()
    lock_directory = tmp_path / "locks"
    first = module.NotesSyncRootCoordinator(lock_directory)
    second = module.NotesSyncRootCoordinator(lock_directory)
    old_owner = first.try_acquire(root)
    lock_path = next(lock_directory.glob("*.lock"))
    lock_path.rename(lock_directory / "displaced.lock")
    lock_path.touch(mode=0o600)

    assert not old_owner.can_write
    new_owner = second.try_acquire(root)

    assert new_owner.state is module.RootAdmissionState.OWNER
    assert new_owner.can_write
    assert not old_owner.can_write
    first.release(old_owner.lease)
    second.release(new_owner.lease)


def test_replacing_live_lock_directory_revokes_old_owner_and_old_coordinator(
    tmp_path: Path,
) -> None:
    module = _coordinator_module()
    root = tmp_path / "root"
    root.mkdir()
    lock_directory = tmp_path / "locks"
    first = module.NotesSyncRootCoordinator(lock_directory)
    old_owner = first.try_acquire(root)
    lock_directory.rename(tmp_path / "displaced-locks")
    lock_directory.mkdir(mode=0o700)

    assert not old_owner.can_write
    stale_attempt = first.try_acquire(root)
    second = module.NotesSyncRootCoordinator(lock_directory)
    new_owner = second.try_acquire(root)

    assert stale_attempt.state is module.RootAdmissionState.REJECTED
    assert stale_attempt.reason_code == "lock_unavailable"
    assert new_owner.state is module.RootAdmissionState.OWNER
    assert new_owner.can_write
    assert not old_owner.can_write
    first.release(old_owner.lease)
    second.release(new_owner.lease)


def test_recreating_root_inode_revokes_old_owner_before_new_root_owner(
    tmp_path: Path,
) -> None:
    module = _coordinator_module()
    root = tmp_path / "root"
    root.mkdir()
    lock_directory = tmp_path / "locks"
    first = module.NotesSyncRootCoordinator(lock_directory)
    second = module.NotesSyncRootCoordinator(lock_directory)
    old_owner = first.try_acquire(root)
    root.rename(tmp_path / "displaced-root")
    root.mkdir()

    assert not old_owner.can_write
    new_owner = second.try_acquire(root)

    assert new_owner.state is module.RootAdmissionState.OWNER
    assert new_owner.root_digest != old_owner.root_digest
    assert new_owner.can_write
    assert not old_owner.can_write
    first.release(old_owner.lease)
    second.release(new_owner.lease)


def test_acquire_rechecks_lock_file_identity_after_os_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _coordinator_module()
    root = tmp_path / "root"
    root.mkdir()
    lock_directory = tmp_path / "locks"
    coordinator = module.NotesSyncRootCoordinator(lock_directory)
    real_lock = module.portalocker.lock

    def lock_then_replace(handle, flags) -> None:
        real_lock(handle, flags)
        lock_path = next(lock_directory.glob("*.lock"))
        lock_path.rename(lock_directory / "displaced.lock")
        lock_path.touch(mode=0o600)

    monkeypatch.setattr(module.portalocker, "lock", lock_then_replace)

    admission = coordinator.try_acquire(root)

    assert admission.state is module.RootAdmissionState.REJECTED
    assert admission.reason_code == "lock_unavailable"
    assert admission.lease is None


def test_release_waits_for_close_admission_settlement(tmp_path: Path) -> None:
    module = _coordinator_module()
    root = tmp_path / "root"
    root.mkdir()
    lock_directory = tmp_path / "locks"
    owner = module.NotesSyncRootCoordinator(lock_directory)
    contender = module.NotesSyncRootCoordinator(lock_directory)
    admission = owner.try_acquire(root)
    settlement_started = threading.Event()
    allow_settlement = threading.Event()
    release_finished = threading.Event()
    errors: list[BaseException] = []

    def settle() -> None:
        settlement_started.set()
        assert allow_settlement.wait(3.0)

    def close_owner() -> None:
        try:
            owner.close_admission(admission.lease, settle)
        except BaseException as exc:
            errors.append(exc)

    def release_owner() -> None:
        try:
            owner.release(admission.lease)
            release_finished.set()
        except BaseException as exc:
            errors.append(exc)

    close_thread = threading.Thread(target=close_owner)
    release_thread = threading.Thread(target=release_owner)
    close_thread.start()
    assert settlement_started.wait(3.0)
    release_thread.start()

    assert not release_finished.wait(0.1)
    assert contender.try_acquire(root).state is module.RootAdmissionState.PASSIVE
    allow_settlement.set()
    close_thread.join(3.0)
    release_thread.join(3.0)

    assert not close_thread.is_alive()
    assert not release_thread.is_alive()
    assert release_finished.is_set()
    assert errors == []
    replacement = contender.try_acquire(root)
    assert replacement.state is module.RootAdmissionState.OWNER
    contender.release(replacement.lease)


def test_concurrent_close_admission_settles_once(tmp_path: Path) -> None:
    module = _coordinator_module()
    root = tmp_path / "root"
    root.mkdir()
    owner = module.NotesSyncRootCoordinator(tmp_path / "locks")
    admission = owner.try_acquire(root)
    settlement_started = threading.Event()
    allow_settlement = threading.Event()
    calls = 0
    errors: list[BaseException] = []

    def settle() -> None:
        nonlocal calls
        calls += 1
        settlement_started.set()
        assert allow_settlement.wait(3.0)

    def close_owner() -> None:
        try:
            owner.close_admission(admission.lease, settle)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=close_owner) for _index in range(2)]
    threads[0].start()
    assert settlement_started.wait(3.0)
    threads[1].start()
    allow_settlement.set()
    for thread in threads:
        thread.join(3.0)

    assert calls == 1
    assert all(not thread.is_alive() for thread in threads)
    assert errors == []


def test_waiting_release_does_not_block_settlement_admission_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _coordinator_module()
    root = tmp_path / "root"
    root.mkdir()
    owner = module.NotesSyncRootCoordinator(tmp_path / "locks")
    admission = owner.try_acquire(root)
    settlement_started = threading.Event()
    release_waiting = threading.Event()
    outcomes: list[object] = []
    errors: list[BaseException] = []
    real_take_handle = module.RootLease._take_handle

    def observed_take_handle(lease):
        release_waiting.set()
        return real_take_handle(lease)

    monkeypatch.setattr(module.RootLease, "_take_handle", observed_take_handle)

    def settle() -> None:
        settlement_started.set()
        assert release_waiting.wait(3.0)
        outcomes.append(owner.try_acquire(root).state)

    def close_owner() -> None:
        try:
            owner.close_admission(admission.lease, settle)
        except BaseException as exc:
            errors.append(exc)

    def release_owner() -> None:
        try:
            owner.release(admission.lease)
        except BaseException as exc:
            errors.append(exc)

    close_thread = threading.Thread(target=close_owner, daemon=True)
    release_thread = threading.Thread(target=release_owner, daemon=True)
    close_thread.start()
    assert settlement_started.wait(3.0)
    release_thread.start()
    close_thread.join(1.0)
    release_thread.join(1.0)

    assert outcomes == [module.RootAdmissionState.PASSIVE]
    assert not close_thread.is_alive()
    assert not release_thread.is_alive()
    assert errors == []


@pytest.mark.parametrize(
    "lock_error",
    [
        portalocker.exceptions.AlreadyLocked(),
        portalocker.exceptions.LockException(),
    ],
)
def test_acquire_bounds_cleanup_failure_without_path_disclosure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lock_error: Exception,
) -> None:
    module = _coordinator_module()
    root = tmp_path / "root"
    root.mkdir()
    coordinator = module.NotesSyncRootCoordinator(tmp_path / "locks")

    class FailingClose:
        def close(self) -> None:
            raise OSError("PRIVATE/path/lock")

    monkeypatch.setattr(
        module,
        "open_private_text_append_stream",
        lambda *_args, **_kwargs: FailingClose(),
    )
    monkeypatch.setattr(
        module.portalocker,
        "lock",
        lambda *_args: (_ for _ in ()).throw(lock_error),
    )

    admission = coordinator.try_acquire(root)

    expected = (
        module.RootAdmissionState.PASSIVE
        if isinstance(lock_error, portalocker.exceptions.AlreadyLocked)
        else module.RootAdmissionState.REJECTED
    )
    assert admission.state is expected
    assert "PRIVATE" not in repr(admission)


def test_released_owner_label_does_not_claim_active(tmp_path: Path) -> None:
    module = _coordinator_module()
    root = tmp_path / "root"
    root.mkdir()
    coordinator = module.NotesSyncRootCoordinator(tmp_path / "locks")
    admission = coordinator.try_acquire(root)

    coordinator.release(admission.lease)

    assert admission.label == "Inactive in this process"
