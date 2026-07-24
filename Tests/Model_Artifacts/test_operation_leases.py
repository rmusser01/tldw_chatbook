from __future__ import annotations

from pathlib import Path

import portalocker
import pytest

import tldw_chatbook.Model_Artifacts.leases as leases_module
from tldw_chatbook.Model_Artifacts.leases import (
    ArtifactLeaseCancelledError,
    ArtifactLeaseError,
    ArtifactLeaseKey,
    ArtifactLeaseTimeoutError,
    ArtifactOperationLease,
    ArtifactOperationLeaseSet,
    LeaseMode,
)


def key(
    artifact_id: str = "parakeet-v2",
    revision: str = "rev-a",
    variant: str = "int8",
) -> ArtifactLeaseKey:
    return ArtifactLeaseKey(artifact_id, revision, variant)


def test_lease_key_rejects_empty_and_reserved_separator() -> None:
    with pytest.raises(ValueError, match="artifact_id"):
        ArtifactLeaseKey("", "rev-a", "int8")
    with pytest.raises(ValueError, match="revision"):
        ArtifactLeaseKey("model", " ", "int8")
    with pytest.raises(ValueError, match="reserved separator"):
        ArtifactLeaseKey("model", "rev-a", "int8\u001funsafe")


def test_lock_filename_is_deterministic_and_opaque(tmp_path: Path) -> None:
    unsafe = ArtifactLeaseKey("../../model", "refs/rev", "int8/windows")
    first = ArtifactOperationLease(tmp_path, unsafe, LeaseMode.SHARED)
    second = ArtifactOperationLease(tmp_path, unsafe, LeaseMode.SHARED)

    assert first.lock_path == second.lock_path
    assert first.lock_path.parent == tmp_path
    assert first.lock_path.suffix == ".lock"
    assert ".." not in first.lock_path.name
    assert "model" not in first.lock_path.name
    assert "windows" not in first.lock_path.name


def test_exclusive_times_out_while_shared_lease_is_open(tmp_path: Path) -> None:
    with ArtifactOperationLease(tmp_path, key(), LeaseMode.SHARED):
        with pytest.raises(ArtifactLeaseTimeoutError):
            ArtifactOperationLease(
                tmp_path,
                key(),
                LeaseMode.EXCLUSIVE,
                timeout_seconds=0.05,
                check_interval_seconds=0.005,
            ).acquire()


def test_single_lease_does_not_retry_after_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Clock:
        def __init__(self) -> None:
            self._values = iter((0.0, 0.5, 1.01))

        def monotonic(self) -> float:
            return next(self._values)

        def sleep(self, seconds: float) -> None:
            pass

    attempts = 0

    def lock_once_contended(handle: object, flags: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise portalocker.exceptions.AlreadyLocked("lease is held", fh=handle)

    monkeypatch.setattr(leases_module, "time", Clock())
    monkeypatch.setattr(portalocker, "lock", lock_once_contended)
    lease = ArtifactOperationLease(
        tmp_path,
        key(),
        LeaseMode.SHARED,
        timeout_seconds=1.0,
        check_interval_seconds=0.5,
    )

    with pytest.raises(ArtifactLeaseTimeoutError):
        lease.acquire()

    assert attempts == 1
    assert lease.acquired is False


def test_lease_set_does_not_start_later_key_after_total_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Clock:
        def __init__(self) -> None:
            self.calls = 0

        def monotonic(self) -> float:
            self.calls += 1
            return 0.0 if self.calls == 1 else 1.01

        def sleep(self, seconds: float) -> None:
            pass

    attempts = 0
    opened: list[Path] = []
    keys = [
        ArtifactLeaseKey("a-root", "rev", "int8"),
        ArtifactLeaseKey("b-dependency", "rev", "fp32"),
    ]
    lease_set = ArtifactOperationLeaseSet(
        tmp_path,
        keys,
        LeaseMode.SHARED,
        timeout_seconds=1.0,
    )

    class Handle:
        def close(self) -> None:
            pass

    def record_open(path: Path, *args: object, **kwargs: object) -> Handle:
        opened.append(path)
        return Handle()

    def record_lock(handle: object, flags: object) -> None:
        nonlocal attempts
        attempts += 1

    monkeypatch.setattr(leases_module, "time", Clock())
    monkeypatch.setattr(Path, "open", record_open)
    monkeypatch.setattr(portalocker, "lock", record_lock)
    monkeypatch.setattr(portalocker, "unlock", lambda handle: None)

    with pytest.raises(ArtifactLeaseTimeoutError):
        lease_set.acquire()

    assert attempts == 1
    assert len(opened) == 1
    assert lease_set.acquired is False


def test_non_contention_lock_failure_raises_stable_error_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = portalocker.exceptions.LockException("backend failed")
    attempts = 0

    def fail_lock(handle: object, flags: object) -> None:
        nonlocal attempts
        attempts += 1
        raise failure

    monkeypatch.setattr(portalocker, "lock", fail_lock)
    lease = ArtifactOperationLease(
        tmp_path,
        key(),
        LeaseMode.SHARED,
        timeout_seconds=0,
    )

    with pytest.raises(
        ArtifactLeaseError,
        match="failed acquiring shared lease for parakeet-v2",
    ) as exc_info:
        lease.acquire()

    assert type(exc_info.value) is ArtifactLeaseError
    assert exc_info.value.__cause__ is failure
    assert attempts == 1
    assert lease.acquired is False


@pytest.mark.parametrize("phase", ["mkdir", "open"])
def test_setup_failure_raises_stable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    failure = OSError(f"{phase} failed")

    if phase == "mkdir":
        monkeypatch.setattr(
            Path, "mkdir", lambda *args, **kwargs: (_ for _ in ()).throw(failure)
        )
    else:
        monkeypatch.setattr(
            Path, "open", lambda *args, **kwargs: (_ for _ in ()).throw(failure)
        )

    lease = ArtifactOperationLease(tmp_path, key(), LeaseMode.SHARED)

    with pytest.raises(
        ArtifactLeaseError,
        match="failed preparing shared lease for parakeet-v2",
    ) as exc_info:
        lease.acquire()

    assert exc_info.value.__cause__ is failure
    assert lease.acquired is False


def test_unexpected_backend_lock_failure_raises_stable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = OSError("unexpected backend failure")

    monkeypatch.setattr(
        portalocker,
        "lock",
        lambda handle, flags: (_ for _ in ()).throw(failure),
    )
    lease = ArtifactOperationLease(tmp_path, key(), LeaseMode.SHARED)

    with pytest.raises(
        ArtifactLeaseError,
        match="failed acquiring shared lease for parakeet-v2",
    ) as exc_info:
        lease.acquire()

    assert exc_info.value.__cause__ is failure
    assert lease.acquired is False


def test_acquire_preserves_timeout_when_handle_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_failure = OSError("close failed")

    class Handle:
        def close(self) -> None:
            raise close_failure

    handle = Handle()
    contention = portalocker.exceptions.AlreadyLocked("lease is held", fh=handle)
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: handle)
    monkeypatch.setattr(
        portalocker,
        "lock",
        lambda current_handle, flags: (_ for _ in ()).throw(contention),
    )
    lease = ArtifactOperationLease(
        tmp_path,
        key(),
        LeaseMode.SHARED,
        timeout_seconds=0,
    )

    with pytest.raises(ArtifactLeaseTimeoutError) as exc_info:
        lease.acquire()

    assert exc_info.value.__cause__ is contention
    assert any(
        "close failed" in note for note in getattr(exc_info.value, "__notes__", ())
    )
    assert lease.acquired is False


@pytest.mark.parametrize("unlock_fails", [False, True])
def test_release_raises_stable_error_without_masking_unlock_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unlock_fails: bool,
) -> None:
    unlock_failure = OSError("unlock failed")
    close_failure = OSError("close failed")

    class Handle:
        def close(self) -> None:
            raise close_failure

    handle = Handle()
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: handle)
    monkeypatch.setattr(portalocker, "lock", lambda current_handle, flags: None)
    if unlock_fails:
        monkeypatch.setattr(
            portalocker,
            "unlock",
            lambda current_handle: (_ for _ in ()).throw(unlock_failure),
        )
    else:
        monkeypatch.setattr(portalocker, "unlock", lambda current_handle: None)
    lease = ArtifactOperationLease(tmp_path, key(), LeaseMode.SHARED).acquire()

    with pytest.raises(
        ArtifactLeaseError,
        match=(
            "failed releasing shared lease"
            if unlock_fails
            else "failed closing shared lease"
        ),
    ) as exc_info:
        lease.release()

    assert exc_info.value.__cause__ is (
        unlock_failure if unlock_fails else close_failure
    )
    if unlock_fails:
        assert any(
            "close failed" in note for note in getattr(exc_info.value, "__notes__", ())
        )
    assert lease.acquired is False


def test_release_does_not_attach_close_failure_to_ambient_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_failure = OSError("close failed")
    ambient_error = ValueError("unrelated caller error")

    class Handle:
        def close(self) -> None:
            raise close_failure

    handle = Handle()
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: handle)
    monkeypatch.setattr(portalocker, "lock", lambda current_handle, flags: None)
    monkeypatch.setattr(portalocker, "unlock", lambda current_handle: None)
    lease = ArtifactOperationLease(tmp_path, key(), LeaseMode.SHARED).acquire()

    try:
        raise ambient_error
    except ValueError:
        with pytest.raises(
            ArtifactLeaseError,
            match="failed closing shared lease",
        ) as exc_info:
            lease.release()

    assert exc_info.value.__cause__ is close_failure
    assert getattr(ambient_error, "__notes__", ()) == ()
    assert lease.acquired is False


def test_context_close_releases_lock(tmp_path: Path) -> None:
    shared = ArtifactOperationLease(tmp_path, key(), LeaseMode.SHARED)
    with shared:
        assert shared.acquired is True

    assert shared.acquired is False
    with ArtifactOperationLease(
        tmp_path,
        key(),
        LeaseMode.EXCLUSIVE,
        timeout_seconds=0.2,
    ) as exclusive:
        assert exclusive.acquired is True


@pytest.mark.parametrize("lease_kind", ["single", "set"])
def test_context_preserves_body_error_when_release_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lease_kind: str,
) -> None:
    if lease_kind == "single":
        lease = ArtifactOperationLease(tmp_path, key(), LeaseMode.SHARED)
    else:
        lease = ArtifactOperationLeaseSet(
            tmp_path,
            [key()],
            LeaseMode.SHARED,
        )
    body_error = ValueError("body failed")
    cleanup_error = RuntimeError("cleanup failed")
    cleanup_error.add_note("lower-level cleanup detail")
    original_release = lease.release

    def release_with_failure() -> None:
        original_release()
        raise cleanup_error

    monkeypatch.setattr(lease, "release", release_with_failure)

    with pytest.raises(ValueError, match="body failed") as exc_info:
        with lease:
            raise body_error

    assert exc_info.value is body_error
    assert lease.acquired is False
    notes = getattr(body_error, "__notes__", ())
    assert any(
        "lease context cleanup failed" in note and str(cleanup_error) in note
        for note in notes
    )
    assert "lower-level cleanup detail" in notes


@pytest.mark.parametrize("lease_kind", ["single", "set"])
def test_context_raises_cleanup_error_when_body_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lease_kind: str,
) -> None:
    if lease_kind == "single":
        lease = ArtifactOperationLease(tmp_path, key(), LeaseMode.SHARED)
    else:
        lease = ArtifactOperationLeaseSet(
            tmp_path,
            [key()],
            LeaseMode.SHARED,
        )
    cleanup_error = RuntimeError("cleanup failed")
    original_release = lease.release

    def release_with_failure() -> None:
        original_release()
        raise cleanup_error

    monkeypatch.setattr(lease, "release", release_with_failure)

    with pytest.raises(RuntimeError, match="cleanup failed") as exc_info:
        with lease:
            pass

    assert exc_info.value is cleanup_error
    assert lease.acquired is False


def test_cancelled_acquire_closes_unowned_handle(tmp_path: Path) -> None:
    lease = ArtifactOperationLease(
        tmp_path,
        key(),
        LeaseMode.SHARED,
        cancelled=lambda: True,
    )

    with pytest.raises(ArtifactLeaseCancelledError):
        lease.acquire()

    assert lease.acquired is False


def test_mid_contention_cancellation_stops_retries_and_closes_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    seen_handles: list[object] = []

    def cancelled() -> bool:
        return attempts >= 2

    def contend(handle: object, flags: object) -> None:
        nonlocal attempts
        attempts += 1
        seen_handles.append(handle)
        raise portalocker.exceptions.AlreadyLocked(
            "lease is held",
            fh=handle,
        )

    monkeypatch.setattr(portalocker, "lock", contend)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    lease = ArtifactOperationLease(
        tmp_path,
        key(),
        LeaseMode.SHARED,
        timeout_seconds=1.0,
        check_interval_seconds=0.01,
        cancelled=cancelled,
    )

    with pytest.raises(ArtifactLeaseCancelledError):
        lease.acquire()

    assert attempts == 2
    assert len({id(handle) for handle in seen_handles}) == 1
    assert getattr(seen_handles[0], "closed") is True
    assert lease.acquired is False


def test_double_acquire_is_rejected(tmp_path: Path) -> None:
    lease = ArtifactOperationLease(tmp_path, key(), LeaseMode.SHARED)
    with lease:
        with pytest.raises(ArtifactLeaseError, match="already acquired"):
            lease.acquire()


@pytest.mark.parametrize("lease_kind", ["single", "set"])
@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        pytest.param("timeout_seconds", float("nan"), id="timeout-nan"),
        pytest.param("timeout_seconds", float("inf"), id="timeout-positive-inf"),
        pytest.param("timeout_seconds", float("-inf"), id="timeout-negative-inf"),
        pytest.param("timeout_seconds", -1.0, id="timeout-negative"),
        pytest.param("check_interval_seconds", float("nan"), id="interval-nan"),
        pytest.param(
            "check_interval_seconds",
            float("inf"),
            id="interval-positive-inf",
        ),
        pytest.param(
            "check_interval_seconds",
            float("-inf"),
            id="interval-negative-inf",
        ),
        pytest.param("check_interval_seconds", 0.0, id="interval-zero"),
        pytest.param("check_interval_seconds", -1.0, id="interval-negative"),
    ],
)
def test_lease_constructors_reject_invalid_timing_values(
    tmp_path: Path,
    lease_kind: str,
    field_name: str,
    value: float,
) -> None:
    kwargs = {field_name: value}

    with pytest.raises(ValueError, match=field_name):
        if lease_kind == "single":
            ArtifactOperationLease(tmp_path, key(), LeaseMode.SHARED, **kwargs)
        else:
            ArtifactOperationLeaseSet(
                tmp_path,
                [key()],
                LeaseMode.SHARED,
                **kwargs,
            )


@pytest.mark.parametrize("lease_kind", ["single", "set"])
def test_lease_constructors_reject_plain_string_mode(
    tmp_path: Path,
    lease_kind: str,
) -> None:
    with pytest.raises(TypeError, match="mode must be a LeaseMode"):
        if lease_kind == "single":
            ArtifactOperationLease(
                tmp_path,
                key(),
                "shared",  # type: ignore[arg-type]
            )
        else:
            ArtifactOperationLeaseSet(
                tmp_path,
                [key()],
                "shared",  # type: ignore[arg-type]
            )


def test_lease_set_sorts_and_deduplicates_keys(tmp_path: Path) -> None:
    keys = [
        ArtifactLeaseKey("vad", "rev-b", "fp32"),
        ArtifactLeaseKey("parakeet", "rev-a", "int8"),
        ArtifactLeaseKey("vad", "rev-b", "fp32"),
    ]
    lease_set = ArtifactOperationLeaseSet(
        tmp_path,
        keys,
        LeaseMode.SHARED,
    )

    assert lease_set.keys == (
        ArtifactLeaseKey("parakeet", "rev-a", "int8"),
        ArtifactLeaseKey("vad", "rev-b", "fp32"),
    )


def test_partial_set_failure_releases_already_acquired_keys(tmp_path: Path) -> None:
    first = ArtifactLeaseKey("a-root", "rev", "int8")
    blocked = ArtifactLeaseKey("z-vad", "rev", "fp32")

    with ArtifactOperationLease(tmp_path, blocked, LeaseMode.EXCLUSIVE):
        with pytest.raises(ArtifactLeaseTimeoutError):
            ArtifactOperationLeaseSet(
                tmp_path,
                [blocked, first],
                LeaseMode.SHARED,
                timeout_seconds=0.05,
                check_interval_seconds=0.005,
            ).acquire()

        with ArtifactOperationLease(
            tmp_path,
            first,
            LeaseMode.EXCLUSIVE,
            timeout_seconds=0.2,
        ) as recovered:
            assert recovered.acquired is True


def test_empty_lease_set_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        ArtifactOperationLeaseSet(tmp_path, [], LeaseMode.SHARED)


def test_lease_set_release_unwinds_all_keys_and_raises_first_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease_set = ArtifactOperationLeaseSet(
        tmp_path,
        [
            ArtifactLeaseKey("a", "rev", "int8"),
            ArtifactLeaseKey("b", "rev", "int8"),
            ArtifactLeaseKey("c", "rev", "int8"),
        ],
        LeaseMode.SHARED,
    ).acquire()
    release_order: list[str] = []
    first_error = RuntimeError("release failed for c")
    second_error = RuntimeError("release failed for b")
    release_errors = {"c": first_error, "b": second_error}
    original_release = ArtifactOperationLease.release

    def release_with_failures(lease: ArtifactOperationLease) -> None:
        release_order.append(lease.key.artifact_id)
        original_release(lease)
        if error := release_errors.get(lease.key.artifact_id):
            raise error

    try:
        with monkeypatch.context() as patch:
            patch.setattr(ArtifactOperationLease, "release", release_with_failures)
            with pytest.raises(RuntimeError) as exc_info:
                lease_set.release()

        assert exc_info.value is first_error
        assert release_order == ["c", "b", "a"]
        assert lease_set.acquired is False
    finally:
        lease_set.release()


def test_acquire_preserves_timeout_when_rollback_release_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = ArtifactLeaseKey("a-root", "rev", "int8")
    blocked = ArtifactLeaseKey("z-vad", "rev", "fp32")
    cleanup_error = RuntimeError("cleanup failed for a-root")
    original_release = ArtifactOperationLease.release

    def release_with_failure(lease: ArtifactOperationLease) -> None:
        original_release(lease)
        if lease.key == first:
            raise cleanup_error

    lease_set = ArtifactOperationLeaseSet(
        tmp_path,
        [blocked, first],
        LeaseMode.SHARED,
        timeout_seconds=0.05,
        check_interval_seconds=0.005,
    )
    with ArtifactOperationLease(tmp_path, blocked, LeaseMode.EXCLUSIVE):
        with monkeypatch.context() as patch:
            patch.setattr(ArtifactOperationLease, "release", release_with_failure)
            with pytest.raises(ArtifactLeaseTimeoutError) as exc_info:
                lease_set.acquire()

    assert lease_set.acquired is False
    assert any(
        str(cleanup_error) in note for note in getattr(exc_info.value, "__notes__", ())
    )
