# Model artifact operation leases

## Selected primitive

Chatbook uses `portalocker==3.2.0` behind
`tldw_chatbook.Model_Artifacts.leases`. The dependency is pinned in both
`pyproject.toml` and `requirements.txt`.

Portalocker 3.2.0 publishes a `py3-none-any` wheel, requires Python 3.9 or
newer, and uses the BSD-3-Clause license. Its only platform dependency is
`pywin32>=226` when `platform_system == "Windows"`.

The pinned implementation has different operating-system backends:

- On Windows, the default `MsvcrtLocker` uses Win32 `LockFileEx` for shared
  locks and `msvcrt.locking` for exclusive locks. Omitting
  `LOCKFILE_EXCLUSIVE_LOCK` gives the shared mode; non-blocking requests add
  `LOCKFILE_FAIL_IMMEDIATELY`.
- On POSIX systems, the default backend applies advisory `fcntl.flock` locks
  with shared, exclusive, and non-blocking flags.

Direct use of Python's `msvcrt.locking` was rejected. Its `LK_RLCK` and
`LK_NBRLCK` modes are documented with the same exclusive locking behavior as
`LK_LOCK` and `LK_NBLCK`; they do not provide the shared-reader contract
required by ADR-025. Portalocker supplies that missing Windows shared mode
through `LockFileEx`.

## Chatbook contract

- `ArtifactLeaseKey` identifies one immutable artifact by artifact ID,
  revision, and variant. Its canonical value is opaque outside this module and
  is SHA-256 hashed into a contained lock filename.
- `LeaseMode.SHARED` protects a resident root or dependency artifact.
- `LeaseMode.EXCLUSIVE` protects mutation and deletion.
- Each attempt uses a non-blocking OS lock with a monotonic total timeout,
  bounded polling, and an optional cancellation callback.
- `ArtifactOperationLeaseSet` sorts and deduplicates identities before
  acquisition, applies one timeout budget to the complete set, and releases in
  reverse order.
- Partial set acquisition rolls back every lease already acquired. Release
  continues across cleanup failures and reports the first failure after all
  held leases have been attempted.
- The open file handle owns the OS lock. Explicit close, context exit, and
  process death release it.
- Lock files contain no authoritative state. PID files, timestamps, stale-lock
  detection, and stale-lock cleanup are not part of correctness.

All artifact readers and writers must use this API. In particular, the
long-lived STT worker must hold shared leases for the root model and every
loaded dependency for the complete resident-model lifetime, including idle
reuse. Install, activation, replacement, and deletion paths must take the
corresponding exclusive leases.

## Qualification gate

The focused `artifact-lease-spike` GitHub Actions job installs Chatbook and its
test dependencies, then runs these exact files on Python 3.11:

- `Tests/Model_Artifacts/test_operation_leases.py`
- `Tests/Model_Artifacts/test_operation_leases_process.py`

The job has native matrix entries for `ubuntu-latest`, `macos-latest`, and
`windows-latest`. The stable `Artifact Lease Gate` check succeeds only when the
complete matrix succeeds, and the repository's test-summary job depends on
that gate.

Together, the unit and spawn-process tests prove:

1. two spawned processes can hold the same shared lease;
2. an exclusive lease is blocked by a shared lease;
3. an idle root-and-VAD lease set blocks deletion of both artifacts;
4. normal close and forced process termination release the held leases;
5. lease-set acquisition is ordered and uses one timeout budget; and
6. partial acquisition failure releases every previously acquired lease.

### Current status

Local macOS verification passes. Native qualification is still pending until
the Ubuntu, macOS, and Windows matrix entries all pass for one final commit.
Do not treat local success, the presence of the workflow, or a subset of the
matrix as cross-platform proof.

TASK-507 remains blocked while that native result is pending. Any native
failure leaves TASK-505 in progress and requires the primitive or ADR-025 to be
revisited before artifact-core implementation starts.

## Scope and limitations

- POSIX `flock` locks are advisory. A code path that bypasses the lease API can
  still read, replace, or delete a file.
- Qualification covers local filesystems on the supported Windows, macOS, and
  Linux runners only.
- Network, distributed, FUSE, container-volume, and remote-filesystem locking
  semantics are not claimed. Such storage requires separate qualification.
- Lock files may remain after release or process death. Their existence does
  not mean a lease is held, and deleting them is not a recovery mechanism.
- This primitive coordinates cooperating Chatbook processes; it is not an
  authorization, integrity, or sandbox boundary.
- The `py3-none-any` wheel tag describes package distribution compatibility,
  not proof of identical filesystem-lock behavior on every Python-supported
  operating system.

## Sources

- [Portalocker 3.2.0 package metadata and wheel](https://pypi.org/project/portalocker/3.2.0/)
- [Portalocker v3.2.0 platform implementation](https://github.com/wolph/portalocker/blob/v3.2.0/portalocker/portalocker.py)
- [Portalocker v3.2.0 package configuration](https://github.com/wolph/portalocker/blob/v3.2.0/pyproject.toml)
- [Portalocker v3.2.0 license](https://github.com/wolph/portalocker/blob/v3.2.0/LICENSE)
- [Python 3.11 `msvcrt.locking`](https://docs.python.org/3.11/library/msvcrt.html#msvcrt.locking)
- [Windows `LockFileEx`](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-lockfileex)
- [Python 3.11 `fcntl.flock`](https://docs.python.org/3.11/library/fcntl.html#fcntl.flock)
- [ADR-025](../decisions/025-shared-stt-artifacts-and-runtime-routing.md)
- [TASK-505](../tasks/task-505%20-%20Prove-cross-platform-model-artifact-operation-leases.md)
- [TASK-507](../tasks/task-507%20-%20Build-shared-model-artifact-descriptors-and-lifecycle.md)
