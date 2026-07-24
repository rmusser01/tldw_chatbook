"""Spawn targets used by cross-platform artifact lease tests."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from tldw_chatbook.Model_Artifacts.leases import (
    ArtifactLeaseKey,
    ArtifactOperationLease,
    ArtifactOperationLeaseSet,
    LeaseMode,
)


class EventLike(Protocol):
    """Spawn-safe event operations used by the child targets."""

    def set(self) -> None: ...

    def wait(self, timeout: float | None = None) -> bool: ...


def _key(raw: tuple[str, str, str]) -> ArtifactLeaseKey:
    return ArtifactLeaseKey(*raw)


def hold_one(
    lock_root: str,
    raw_key: tuple[str, str, str],
    mode: str,
    ready: EventLike,
    release: EventLike,
) -> None:
    with ArtifactOperationLease(
        Path(lock_root),
        _key(raw_key),
        LeaseMode(mode),
        timeout_seconds=5.0,
    ):
        ready.set()
        release.wait(30.0)


def hold_set(
    lock_root: str,
    raw_keys: tuple[tuple[str, str, str], ...],
    mode: str,
    ready: EventLike,
    release: EventLike,
) -> None:
    with ArtifactOperationLeaseSet(
        Path(lock_root),
        [_key(raw) for raw in raw_keys],
        LeaseMode(mode),
        timeout_seconds=5.0,
    ):
        ready.set()
        release.wait(30.0)
