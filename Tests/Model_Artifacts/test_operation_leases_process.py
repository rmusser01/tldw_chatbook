from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from Tests.Model_Artifacts.lease_processes import hold_one, hold_set
from tldw_chatbook.Model_Artifacts.leases import (
    ArtifactLeaseKey,
    ArtifactLeaseTimeoutError,
    ArtifactOperationLease,
    LeaseMode,
)


pytestmark = pytest.mark.integration
RawKey = tuple[str, str, str]


@contextmanager
def holding_process(
    target: Callable[..., None],
    args: tuple[object, ...],
) -> Iterator[multiprocessing.Process]:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(target=target, args=(*args, ready, release))
    process.start()
    try:
        assert ready.wait(10.0), (
            f"child failed to acquire lease, exit={process.exitcode}"
        )
        yield process
    finally:
        release.set()
        process.join(10.0)
        if process.is_alive():
            process.terminate()
            process.join(5.0)
        if process.is_alive():
            process.kill()
            process.join(5.0)
        assert process.is_alive() is False
        assert process.exitcode == 0


def test_two_spawned_processes_hold_shared_lease(tmp_path: Path) -> None:
    raw: RawKey = ("parakeet", "rev-a", "int8")
    with holding_process(
        hold_one,
        (str(tmp_path), raw, LeaseMode.SHARED.value),
    ):
        with holding_process(
            hold_one,
            (str(tmp_path), raw, LeaseMode.SHARED.value),
        ):
            with pytest.raises(ArtifactLeaseTimeoutError):
                ArtifactOperationLease(
                    tmp_path,
                    ArtifactLeaseKey(*raw),
                    LeaseMode.EXCLUSIVE,
                    timeout_seconds=0.1,
                    check_interval_seconds=0.01,
                ).acquire()


def test_spawned_exclusive_lease_blocks_shared_until_normal_release(
    tmp_path: Path,
) -> None:
    raw: RawKey = ("parakeet", "rev-a", "int8")
    with holding_process(
        hold_one,
        (str(tmp_path), raw, LeaseMode.EXCLUSIVE.value),
    ):
        with pytest.raises(ArtifactLeaseTimeoutError):
            ArtifactOperationLease(
                tmp_path,
                ArtifactLeaseKey(*raw),
                LeaseMode.SHARED,
                timeout_seconds=0.1,
                check_interval_seconds=0.01,
            ).acquire()

    with ArtifactOperationLease(
        tmp_path,
        ArtifactLeaseKey(*raw),
        LeaseMode.SHARED,
        timeout_seconds=2.0,
        check_interval_seconds=0.02,
    ) as lease:
        assert lease.acquired is True


def test_idle_root_and_dependency_set_blocks_deletion(tmp_path: Path) -> None:
    raw_keys: tuple[RawKey, ...] = (
        ("parakeet", "rev-a", "int8"),
        ("vad", "rev-vad", "fp32"),
    )
    with holding_process(
        hold_set,
        (str(tmp_path), raw_keys, LeaseMode.SHARED.value),
    ):
        for raw in raw_keys:
            with pytest.raises(ArtifactLeaseTimeoutError):
                ArtifactOperationLease(
                    tmp_path,
                    ArtifactLeaseKey(*raw),
                    LeaseMode.EXCLUSIVE,
                    timeout_seconds=0.1,
                    check_interval_seconds=0.01,
                ).acquire()


def test_forced_exclusive_process_death_releases_root_and_dependency_set(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    never_release = context.Event()
    raw_keys: tuple[RawKey, ...] = (
        ("parakeet", "rev-a", "int8"),
        ("vad", "rev-vad", "fp32"),
    )
    process = context.Process(
        target=hold_set,
        args=(
            str(tmp_path),
            raw_keys,
            LeaseMode.EXCLUSIVE.value,
            ready,
            never_release,
        ),
    )
    process.start()
    try:
        assert ready.wait(10.0)

        for raw in raw_keys:
            with pytest.raises(ArtifactLeaseTimeoutError):
                ArtifactOperationLease(
                    tmp_path,
                    ArtifactLeaseKey(*raw),
                    LeaseMode.SHARED,
                    timeout_seconds=0.1,
                    check_interval_seconds=0.01,
                ).acquire()

        process.terminate()
        process.join(10.0)
        assert process.is_alive() is False

        for raw in raw_keys:
            with ArtifactOperationLease(
                tmp_path,
                ArtifactLeaseKey(*raw),
                LeaseMode.SHARED,
                timeout_seconds=2.0,
                check_interval_seconds=0.02,
            ) as lease:
                assert lease.acquired is True
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5.0)
        if process.is_alive():
            process.kill()
            process.join(5.0)
        assert process.is_alive() is False


def test_forced_process_death_releases_root_and_dependency_set(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    never_release = context.Event()
    raw_keys: tuple[RawKey, ...] = (
        ("parakeet", "rev-a", "int8"),
        ("vad", "rev-vad", "fp32"),
    )
    process = context.Process(
        target=hold_set,
        args=(
            str(tmp_path),
            raw_keys,
            LeaseMode.SHARED.value,
            ready,
            never_release,
        ),
    )
    process.start()
    try:
        assert ready.wait(10.0)

        process.terminate()
        process.join(10.0)
        assert process.is_alive() is False

        for raw in raw_keys:
            with ArtifactOperationLease(
                tmp_path,
                ArtifactLeaseKey(*raw),
                LeaseMode.EXCLUSIVE,
                timeout_seconds=2.0,
                check_interval_seconds=0.02,
            ) as lease:
                assert lease.acquired is True
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5.0)
        if process.is_alive():
            process.kill()
            process.join(5.0)
        assert process.is_alive() is False
