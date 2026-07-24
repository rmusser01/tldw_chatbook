from __future__ import annotations

from pathlib import Path

import pytest

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


def test_double_acquire_is_rejected(tmp_path: Path) -> None:
    lease = ArtifactOperationLease(tmp_path, key(), LeaseMode.SHARED)
    with lease:
        with pytest.raises(ArtifactLeaseError, match="already acquired"):
            lease.acquire()


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
