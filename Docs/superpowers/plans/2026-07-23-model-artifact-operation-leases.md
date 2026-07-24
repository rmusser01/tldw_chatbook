# Cross-Platform Model Artifact Operation Leases Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove and land the cross-platform shared/exclusive operation-lease
primitive that later protects resident model artifacts and their dependencies.

**Architecture:** Wrap the exactly pinned `portalocker==3.2.0` low-level
shared/exclusive API behind Chatbook-owned typed leases. Lock filenames are
SHA-256 derivations of validated opaque artifact identities, acquisition is
non-blocking with monotonic timeout/cancellation polling, and multi-artifact
sets acquire in canonical order with reverse rollback. Spawned native process
tests are the gate: failure on Windows, macOS, or Linux blocks TASK-507 and
requires an ADR-025 amendment instead of a platform-specific fallback.

**Tech Stack:** Python 3.11+, `portalocker==3.2.0`, standard-library
dataclasses/enums/hashlib/multiprocessing/pathlib/time, pytest, GitHub Actions.

## Global Constraints

- Implement only TASK-505. Do not add artifact descriptors, downloads,
  activation records, model browsers, STT providers, or inference workers.
- The selected dependency is exactly `portalocker==3.2.0`, distributed as a
  `py3-none-any` wheel under BSD-3-Clause.
- Shared leases represent resident model loads. Exclusive leases represent
  mutation/deletion ownership.
- Lock ownership must be tied to the open OS handle so normal close and process
  termination release it without PID files, timestamps, or stale-lock cleanup.
- Acquisition uses non-blocking OS attempts, `time.monotonic()`, a bounded
  timeout, and an optional cancellation callback.
- Artifact identities are opaque data. They never become path components.
- Lease sets sort by `(artifact_id, revision, variant)`, acquire in that order,
  and release in reverse order after success or partial failure.
- Locks are advisory on POSIX. Every Chatbook artifact reader/writer must use
  this API; network-filesystem guarantees are outside TASK-505.
- Native Windows, macOS, and Linux tests are mandatory. A local pass on only one
  platform is insufficient to mark the task Done.
- ADR required: yes
- ADR path:
  `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`
- Reason: ADR-025 already mandates cross-process shared/exclusive operation
  leases with automatic crash release and blocks the artifact core on this
  proof.

## File Structure

### New production files

- `tldw_chatbook/Model_Artifacts/__init__.py` — public artifact-lease exports.
- `tldw_chatbook/Model_Artifacts/leases.py` — lease identities, errors,
  single-handle acquisition, ordered lease sets, timeout/cancellation, and
  release.

### Modified packaging files

- `pyproject.toml` — add the exact base dependency `portalocker==3.2.0`.
- `requirements.txt` — mirror the authoritative base dependency for developer
  convenience.

### New tests

- `Tests/Model_Artifacts/__init__.py` — test package marker.
- `Tests/Model_Artifacts/lease_processes.py` — importable spawn targets that
  hold single leases or lease sets without importing application code.
- `Tests/Model_Artifacts/test_operation_leases.py` — identity, timeout,
  cancellation, normal release, ordering, and partial rollback.
- `Tests/Model_Artifacts/test_operation_leases_process.py` — native
  shared/shared, shared/exclusive, idle residency, dependency-set, and forced
  process-death proofs.

### Modified CI and CI tests

- `.github/workflows/test.yml` — focused three-OS lease qualification job.
- `Tests/CI/test_github_actions_test_workflow.py` — workflow-shape regression.

### New documentation

- `backlog/docs/model-artifact-operation-leases.md` — selected primitive,
  supported semantics, evidence commands, limitations, dependency/license, and
  TASK-507 gate.

---

### Task 1: Implement one typed operation lease

**Files:**

- Create: `tldw_chatbook/Model_Artifacts/__init__.py`
- Create: `tldw_chatbook/Model_Artifacts/leases.py`
- Create: `Tests/Model_Artifacts/__init__.py`
- Create: `Tests/Model_Artifacts/test_operation_leases.py`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`

**Interfaces:**

- Consumes: `portalocker.lock()`, `portalocker.unlock()`,
  `portalocker.LockFlags`, an opaque `ArtifactLeaseKey`, and an optional
  `Callable[[], bool]` cancellation check.
- Produces: `ArtifactLeaseKey`, `LeaseMode`, `ArtifactOperationLease`,
  `ArtifactLeaseError`, `ArtifactLeaseTimeoutError`, and
  `ArtifactLeaseCancelledError`.

- [ ] **Step 1: Write failing single-lease tests**

```python
# Tests/Model_Artifacts/test_operation_leases.py
from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest


def lease_api() -> ModuleType:
    """Import the wished-for API inside the test so RED is a test failure."""

    return importlib.import_module("tldw_chatbook.Model_Artifacts.leases")


def key(
    api: ModuleType,
    artifact_id: str = "parakeet-v2",
    revision: str = "rev-a",
    variant: str = "int8",
):
    return api.ArtifactLeaseKey(artifact_id, revision, variant)


def test_lease_key_rejects_empty_and_reserved_separator() -> None:
    api = lease_api()
    with pytest.raises(ValueError, match="artifact_id"):
        api.ArtifactLeaseKey("", "rev-a", "int8")
    with pytest.raises(ValueError, match="revision"):
        api.ArtifactLeaseKey("model", " ", "int8")
    with pytest.raises(ValueError, match="reserved separator"):
        api.ArtifactLeaseKey("model", "rev-a", "int8\u001funsafe")


def test_lock_filename_is_deterministic_and_opaque(tmp_path: Path) -> None:
    api = lease_api()
    unsafe = api.ArtifactLeaseKey("../../model", "refs/rev", "int8/windows")
    first = api.ArtifactOperationLease(tmp_path, unsafe, api.LeaseMode.SHARED)
    second = api.ArtifactOperationLease(tmp_path, unsafe, api.LeaseMode.SHARED)

    assert first.lock_path == second.lock_path
    assert first.lock_path.parent == tmp_path
    assert first.lock_path.suffix == ".lock"
    assert ".." not in first.lock_path.name
    assert "model" not in first.lock_path.name
    assert "windows" not in first.lock_path.name


def test_exclusive_times_out_while_shared_lease_is_open(tmp_path: Path) -> None:
    api = lease_api()
    with api.ArtifactOperationLease(tmp_path, key(api), api.LeaseMode.SHARED):
        with pytest.raises(api.ArtifactLeaseTimeoutError):
            api.ArtifactOperationLease(
                tmp_path,
                key(api),
                api.LeaseMode.EXCLUSIVE,
                timeout_seconds=0.05,
                check_interval_seconds=0.005,
            ).acquire()


def test_context_close_releases_lock(tmp_path: Path) -> None:
    api = lease_api()
    shared = api.ArtifactOperationLease(tmp_path, key(api), api.LeaseMode.SHARED)
    with shared:
        assert shared.acquired is True

    assert shared.acquired is False
    with api.ArtifactOperationLease(
        tmp_path,
        key(api),
        api.LeaseMode.EXCLUSIVE,
        timeout_seconds=0.2,
    ) as exclusive:
        assert exclusive.acquired is True


def test_cancelled_acquire_closes_unowned_handle(tmp_path: Path) -> None:
    api = lease_api()
    lease = api.ArtifactOperationLease(
        tmp_path,
        key(api),
        api.LeaseMode.SHARED,
        cancelled=lambda: True,
    )

    with pytest.raises(api.ArtifactLeaseCancelledError):
        lease.acquire()

    assert lease.acquired is False


def test_double_acquire_is_rejected(tmp_path: Path) -> None:
    api = lease_api()
    lease = api.ArtifactOperationLease(tmp_path, key(api), api.LeaseMode.SHARED)
    with lease:
        with pytest.raises(api.ArtifactLeaseError, match="already acquired"):
            lease.acquire()
```

```python
# Tests/Model_Artifacts/__init__.py
"""Model artifact tests."""
```

- [ ] **Step 2: Run the focused tests and verify the missing module failure**

Run:

```bash
pytest Tests/Model_Artifacts/test_operation_leases.py -v
```

Expected: all six tests collect and fail from `lease_api()` with
`ModuleNotFoundError: No module named 'tldw_chatbook.Model_Artifacts'`.

- [ ] **Step 3: Pin and install the selected dependency**

Add to the base `dependencies` list in `pyproject.toml`:

```toml
    "portalocker==3.2.0",
```

Add beside the other base dependencies in `requirements.txt`:

```text
portalocker==3.2.0
```

Install the updated editable package:

```bash
python -m pip install -e .
```

Expected: installation succeeds and reports `portalocker==3.2.0`.

- [ ] **Step 4: Implement the complete single-lease API**

```python
# tldw_chatbook/Model_Artifacts/leases.py
"""Cross-platform operation leases for immutable model artifacts."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import BinaryIO

import portalocker


_IDENTITY_SEPARATOR = "\x1f"


class ArtifactLeaseError(RuntimeError):
    """Base error for model-artifact lease operations."""


class ArtifactLeaseTimeoutError(ArtifactLeaseError):
    """Raised when an incompatible lease remains held past the deadline."""


class ArtifactLeaseCancelledError(ArtifactLeaseError):
    """Raised when the caller cancels while waiting for a lease."""


class LeaseMode(str, Enum):
    """Supported interprocess lease modes."""

    SHARED = "shared"
    EXCLUSIVE = "exclusive"


@dataclass(frozen=True, order=True)
class ArtifactLeaseKey:
    """Opaque identity of one immutable artifact version."""

    artifact_id: str
    revision: str
    variant: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("artifact_id", self.artifact_id),
            ("revision", self.revision),
            ("variant", self.variant),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            if _IDENTITY_SEPARATOR in value:
                raise ValueError(f"{field_name} contains the reserved separator")

    @property
    def canonical_identity(self) -> str:
        """Return the unambiguous identity used only as hash input."""

        return _IDENTITY_SEPARATOR.join(
            (self.artifact_id, self.revision, self.variant)
        )


class ArtifactOperationLease:
    """Own one shared or exclusive OS-backed artifact lock."""

    def __init__(
        self,
        lock_root: Path,
        key: ArtifactLeaseKey,
        mode: LeaseMode,
        *,
        timeout_seconds: float = 5.0,
        check_interval_seconds: float = 0.05,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be nonnegative")
        if check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be positive")
        self._lock_root = Path(lock_root)
        self.key = key
        self.mode = mode
        self.timeout_seconds = timeout_seconds
        self.check_interval_seconds = check_interval_seconds
        self._cancelled = cancelled
        self._handle: BinaryIO | None = None

    @property
    def lock_path(self) -> Path:
        """Return the contained lock-file path for this opaque identity."""

        digest = hashlib.sha256(self.key.canonical_identity.encode("utf-8")).hexdigest()
        return self._lock_root / f"{digest}.lock"

    @property
    def acquired(self) -> bool:
        """Return whether this object currently owns its OS-backed lock."""

        return self._handle is not None

    def acquire(self) -> ArtifactOperationLease:
        """Acquire the lease or raise a stable timeout/cancellation error."""

        if self._handle is not None:
            raise ArtifactLeaseError("lease is already acquired")

        self._lock_root.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+b")
        flags = portalocker.LockFlags.NON_BLOCKING
        flags |= (
            portalocker.LockFlags.SHARED
            if self.mode is LeaseMode.SHARED
            else portalocker.LockFlags.EXCLUSIVE
        )
        deadline = time.monotonic() + self.timeout_seconds

        try:
            while True:
                if self._cancelled is not None and self._cancelled():
                    raise ArtifactLeaseCancelledError(
                        f"lease acquisition cancelled for {self.key.artifact_id}"
                    )
                try:
                    portalocker.lock(handle, flags)
                except portalocker.exceptions.LockException as exc:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ArtifactLeaseTimeoutError(
                            f"timed out acquiring {self.mode.value} lease "
                            f"for {self.key.artifact_id}"
                        ) from exc
                    time.sleep(min(self.check_interval_seconds, remaining))
                    continue
                self._handle = handle
                return self
        except BaseException:
            handle.close()
            raise

    def release(self) -> None:
        """Release the OS lock and close its owning handle idempotently."""

        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            portalocker.unlock(handle)
        finally:
            handle.close()

    def __enter__(self) -> ArtifactOperationLease:
        return self.acquire()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()
```

```python
# tldw_chatbook/Model_Artifacts/__init__.py
"""Shared managed model-artifact contracts."""

from .leases import (
    ArtifactLeaseCancelledError,
    ArtifactLeaseError,
    ArtifactLeaseKey,
    ArtifactLeaseTimeoutError,
    ArtifactOperationLease,
    LeaseMode,
)

__all__ = [
    "ArtifactLeaseCancelledError",
    "ArtifactLeaseError",
    "ArtifactLeaseKey",
    "ArtifactLeaseTimeoutError",
    "ArtifactOperationLease",
    "LeaseMode",
]
```

- [ ] **Step 5: Run the focused tests**

After GREEN, refactor the test module to the direct production imports shown by
the public interface list above, remove `lease_api()`, and change `key(api, ...)`
back to `key(...)`. This is test-only cleanup; do not change behavior.

Run:

```bash
pytest Tests/Model_Artifacts/test_operation_leases.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Commit the single-lease API**

```bash
git add pyproject.toml requirements.txt \
  tldw_chatbook/Model_Artifacts/__init__.py \
  tldw_chatbook/Model_Artifacts/leases.py \
  Tests/Model_Artifacts/__init__.py \
  Tests/Model_Artifacts/test_operation_leases.py
git commit -m "feat: add model artifact operation lease"
```

### Task 2: Add canonical multi-artifact lease sets

**Files:**

- Modify: `tldw_chatbook/Model_Artifacts/leases.py`
- Modify: `tldw_chatbook/Model_Artifacts/__init__.py`
- Modify: `Tests/Model_Artifacts/test_operation_leases.py`

**Interfaces:**

- Consumes: the Task 1 single-lease API.
- Produces:
  `ArtifactOperationLeaseSet(lock_root, keys, mode, timeout_seconds=...,
  check_interval_seconds=..., cancelled=...)`.

- [ ] **Step 1: Add failing ordering and rollback tests**

Append:

```python
# Tests/Model_Artifacts/test_operation_leases.py
def lease_set_type():
    """Resolve the wished-for type inside a test so RED is a test failure."""

    from tldw_chatbook.Model_Artifacts import leases

    return leases.ArtifactOperationLeaseSet


def test_lease_set_sorts_and_deduplicates_keys(tmp_path: Path) -> None:
    ArtifactOperationLeaseSet = lease_set_type()
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
    ArtifactOperationLeaseSet = lease_set_type()
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
    ArtifactOperationLeaseSet = lease_set_type()
    with pytest.raises(ValueError, match="at least one"):
        ArtifactOperationLeaseSet(tmp_path, [], LeaseMode.SHARED)
```

- [ ] **Step 2: Run the new tests and verify the import failure**

Run:

```bash
pytest Tests/Model_Artifacts/test_operation_leases.py -v
```

Expected: all three new tests collect and fail at runtime with
`AttributeError` because `ArtifactOperationLeaseSet` does not exist.

- [ ] **Step 3: Implement ordered set acquisition and reverse rollback**

Append to `leases.py`:

```python
class ArtifactOperationLeaseSet:
    """Acquire one mode over a canonical immutable artifact closure."""

    def __init__(
        self,
        lock_root: Path,
        keys: list[ArtifactLeaseKey] | tuple[ArtifactLeaseKey, ...],
        mode: LeaseMode,
        *,
        timeout_seconds: float = 5.0,
        check_interval_seconds: float = 0.05,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        ordered_keys = tuple(sorted(set(keys)))
        if not ordered_keys:
            raise ValueError("lease set requires at least one artifact key")
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be nonnegative")
        self._lock_root = Path(lock_root)
        self.keys = ordered_keys
        self.mode = mode
        self.timeout_seconds = timeout_seconds
        self.check_interval_seconds = check_interval_seconds
        self._cancelled = cancelled
        self._leases: list[ArtifactOperationLease] = []

    @property
    def acquired(self) -> bool:
        """Return whether the complete key set is currently held."""

        return len(self._leases) == len(self.keys)

    def acquire(self) -> ArtifactOperationLeaseSet:
        """Acquire every key in order under one total timeout budget."""

        if self._leases:
            raise ArtifactLeaseError("lease set is already acquired")
        deadline = time.monotonic() + self.timeout_seconds
        try:
            for key in self.keys:
                remaining = max(0.0, deadline - time.monotonic())
                lease = ArtifactOperationLease(
                    self._lock_root,
                    key,
                    self.mode,
                    timeout_seconds=remaining,
                    check_interval_seconds=self.check_interval_seconds,
                    cancelled=self._cancelled,
                )
                lease.acquire()
                self._leases.append(lease)
        except BaseException:
            self.release()
            raise
        return self

    def release(self) -> None:
        """Release all acquired keys in reverse order."""

        while self._leases:
            self._leases.pop().release()

    def __enter__(self) -> ArtifactOperationLeaseSet:
        return self.acquire()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()
```

Add `ArtifactOperationLeaseSet` to the import and `__all__` lists in
`tldw_chatbook/Model_Artifacts/__init__.py`.

- [ ] **Step 4: Run the complete unit file**

After GREEN, replace `lease_set_type()` with a direct
`ArtifactOperationLeaseSet` import and remove the three local assignments.

Run:

```bash
pytest Tests/Model_Artifacts/test_operation_leases.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit ordered lease sets**

```bash
git add tldw_chatbook/Model_Artifacts/leases.py \
  tldw_chatbook/Model_Artifacts/__init__.py \
  Tests/Model_Artifacts/test_operation_leases.py
git commit -m "feat: add ordered artifact lease sets"
```

### Task 3: Prove native process compatibility and crash release

**Files:**

- Create: `Tests/Model_Artifacts/lease_processes.py`
- Create: `Tests/Model_Artifacts/test_operation_leases_process.py`

**Interfaces:**

- Consumes: `multiprocessing.get_context("spawn")`, single leases, and lease
  sets.
- Produces: native-OS evidence that shared/shared succeeds, shared/exclusive
  conflicts, root/dependency lease sets remain held while idle, and process
  death releases ownership.

- [ ] **Step 1: Add importable spawn targets**

```python
# Tests/Model_Artifacts/lease_processes.py
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
```

- [ ] **Step 2: Add native process proofs**

```python
# Tests/Model_Artifacts/test_operation_leases_process.py
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
        assert ready.wait(10.0), f"child failed to acquire lease, exit={process.exitcode}"
        yield process
    finally:
        release.set()
        process.join(10.0)
        if process.is_alive():
            process.terminate()
            process.join(5.0)
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
```

- [ ] **Step 3: Run both files locally**

Run:

```bash
pytest Tests/Model_Artifacts/test_operation_leases.py \
  Tests/Model_Artifacts/test_operation_leases_process.py -v
```

Expected: 50 passed, including stable backend/cleanup errors, hard deadline
enforcement, and both shared-to-exclusive and exclusive-to-shared process
proofs.

- [ ] **Step 4: Commit native process proofs**

```bash
git add Tests/Model_Artifacts/lease_processes.py \
  Tests/Model_Artifacts/test_operation_leases_process.py
git commit -m "test: prove artifact lease process semantics"
```

### Task 4: Add the mandatory native CI gate

**Files:**

- Modify: `.github/workflows/test.yml`
- Modify: `Tests/CI/test_github_actions_test_workflow.py`

**Interfaces:**

- Consumes: GitHub-hosted `ubuntu-latest`, `macos-latest`, and
  `windows-latest` runners with Python 3.11.
- Produces: one focused required job that installs the authoritative package
  metadata and executes only TASK-505 lease proofs.

- [ ] **Step 1: Add a failing workflow-shape regression**

Append:

```python
# Tests/CI/test_github_actions_test_workflow.py
def _artifact_lease_job_block() -> str:
    workflow = _workflow_text()
    start = workflow.index("  artifact-lease-spike:")
    end = workflow.index("  integration-tests:", start)
    return workflow[start:end]


def test_artifact_lease_spike_runs_natively_on_three_operating_systems() -> None:
    block = _artifact_lease_job_block()

    assert "ubuntu-latest" in block
    assert "macos-latest" in block
    assert "windows-latest" in block
    assert 'python-version: [\"3.11\"]' in block
    assert "pip install -e ." in block
    assert "Tests/Model_Artifacts/test_operation_leases_process.py" in block
```

- [ ] **Step 2: Run the regression and verify it fails**

Run:

```bash
pytest Tests/CI/test_github_actions_test_workflow.py::test_artifact_lease_spike_runs_natively_on_three_operating_systems -v
```

Expected: FAIL with `ValueError: substring not found`.

- [ ] **Step 3: Add the focused CI job before `integration-tests`**

```yaml
  artifact-lease-spike:
    name: Artifact leases - Python ${{ matrix.python-version }} - ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.11"]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
        cache: pip

    - name: Install lease test dependencies
      run: |
        python -m pip install --upgrade pip setuptools wheel
        pip install -e .
        pip install -r requirements-test.txt

    - name: Prove cross-platform operation leases
      run: |
        pytest Tests/Model_Artifacts/test_operation_leases.py \
          Tests/Model_Artifacts/test_operation_leases_process.py -v
```

- [ ] **Step 4: Run CI-shape and lease tests**

Run:

```bash
pytest Tests/CI/test_github_actions_test_workflow.py \
  Tests/Model_Artifacts/test_operation_leases.py \
  Tests/Model_Artifacts/test_operation_leases_process.py -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the native CI gate**

```bash
git add .github/workflows/test.yml \
  Tests/CI/test_github_actions_test_workflow.py
git commit -m "ci: gate model artifact lease semantics"
```

### Task 5: Record the primitive and close the prerequisite gate

**Files:**

- Create: `backlog/docs/model-artifact-operation-leases.md`
- Modify:
  `backlog/tasks/task-505 - Prove-cross-platform-model-artifact-operation-leases.md`

**Interfaces:**

- Consumes: the committed local tests and successful native CI matrix.
- Produces: durable primitive documentation and a completed prerequisite that
  unblocks TASK-507.

- [ ] **Step 1: Write the selected-primitive documentation**

```markdown
# Model artifact operation leases

## Selected primitive

Chatbook uses `portalocker==3.2.0` behind
`tldw_chatbook.Model_Artifacts.leases`. Portalocker is a BSD-3-Clause,
`py3-none-any` package. Its Windows implementation uses `LockFileEx`, which
supports distinct shared and exclusive modes; its POSIX implementation uses
advisory `flock`.

Python's `msvcrt.locking` was not selected directly because Microsoft documents
`LK_RLCK` as equivalent to `LK_LOCK`, so it does not supply the shared-reader
contract required by ADR-025.

## Chatbook contract

- `ArtifactLeaseKey` values are opaque and hashed into lock filenames.
- `LeaseMode.SHARED` protects a resident root or dependency artifact.
- `LeaseMode.EXCLUSIVE` protects mutation and deletion.
- Acquisition is non-blocking with monotonic timeout and cancellation polling.
- Lease sets acquire sorted unique identities and release in reverse order.
- The open OS handle owns the lock. Close and process death release it.
- No PID files, timestamps, or stale-lock cleanup are part of correctness.

## Qualification gate

The focused `artifact-lease-spike` GitHub Actions job installs the package from
`pyproject.toml` and runs the same spawn-process suite on `ubuntu-latest`,
`macos-latest`, and `windows-latest`.

The suite proves:

1. two spawned processes can hold the same shared lease;
2. an exclusive lease is blocked by a shared lease;
3. an idle resident root/VAD lease set blocks deletion of both artifacts;
4. forced process termination releases the complete lease set; and
5. partial set acquisition rolls back already acquired leases.

TASK-507 remains blocked unless every native job passes for the same commit.

## Scope and limitations

- POSIX locks are advisory; all Chatbook artifact access must use this API.
- This qualification covers local filesystems on supported operating systems.
- Network/distributed filesystem semantics are not claimed.
- Lock files contain no authoritative state and may remain after release.

## Sources

- [Portalocker 3.2.0](https://pypi.org/project/portalocker/3.2.0/)
- [Portalocker API](https://portalocker.readthedocs.io/en/latest/portalocker.html)
- [Windows LockFileEx](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-lockfileex)
- [Python fcntl locking](https://docs.python.org/3/library/fcntl.html)
- [Microsoft `_locking`](https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/locking)
```

- [ ] **Step 2: Run the complete local verification**

Run:

```bash
pytest Tests/Model_Artifacts/ \
  Tests/CI/test_github_actions_test_workflow.py -v
git diff --check
```

Expected: all selected tests pass and `git diff --check` produces no output.

- [ ] **Step 3: Push the branch and verify all native jobs**

Run:

```bash
git push -u origin HEAD
gh pr view --json url >/dev/null || gh pr create --draft --fill
gh pr checks --watch
```

Expected: the `artifact-lease-spike` matrix passes on Ubuntu, macOS, and
Windows. If any matrix entry fails, leave TASK-505 In Progress, keep TASK-507
blocked, record the failure in the task notes, and amend ADR-025 before trying a
different primitive.

- [ ] **Step 4: Complete Backlog hygiene only after the native gate passes**

Run:

```bash
backlog task edit 505 \
  --check-ac 1 --check-ac 2 --check-ac 3 \
  --check-ac 4 --check-ac 5 --check-ac 6 \
  --notes "Implemented the portalocker 3.2.0-backed shared/exclusive lease API, canonical multi-artifact lease sets, timeout and cancellation behavior, partial-acquisition rollback, and native spawn-process proofs. The Ubuntu, macOS, and Windows artifact-lease-spike jobs passed for the same commit. ADR-025 remains the governing decision; TASK-507 may now start." \
  -s Done
```

Expected: TASK-505 is `Done`, all six acceptance criteria are checked, and
Implementation Notes describe the chosen primitive and native evidence.

- [ ] **Step 5: Commit documentation and completed task state**

```bash
git add backlog/docs/model-artifact-operation-leases.md \
  "backlog/tasks/task-505 - Prove-cross-platform-model-artifact-operation-leases.md"
git commit -m "docs: record model artifact lease proof"
```

## Execution stop condition

Do not start TASK-507 in the same implementation run merely because local
tests pass. TASK-505 completes only after the three native operating-system
jobs pass for one commit; TASK-507 receives a separate plan and review gate.
