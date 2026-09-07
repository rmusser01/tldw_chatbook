# ADR-125: Isolate private SQLite file checks from live lock ownership

Status: Accepted
Date: 2026-09-07
Related Task: TASK-31942
Extends: ADR-029
Preserves: ADR-028, ADR-051, ADR-121, ADR-124

## Context

Canvas qualification exposed owned child SIGBUS failures in SQLite WAL code.
A deterministic real-SQLite multiprocess regression established that private
`-shm` inspection cancels a live connection's writer exclusion: another writer
can begin while the first remains in a transaction. Ordinary SQLite connection
creation preserves exclusion. This proves a lock-ownership defect, not every
historical crash's precise interleaving.

POSIX closes affect the process's other locks on the same inode, including when
the descriptor was opened only for privacy validation. SQLite documents the
[close-related locking hazard](https://sqlite.org/howtocorrupt.html#posix_advisory_locks_canceled_by_a_separate_thread_doing_close_).
ADR-029's security checks cannot be removed to avoid it.

## Decision

Perform live SQLite file/sidecar descriptor validation, hardening and retained
source proof in bounded operation-owned helper processes, not in the process
holding the live SQL connections. Keep connection factories, SQL transactions,
WAL behavior and repository ownership unchanged.

Batch each database with its fixed sidecar inventory. Retained backup pins use
helper leases and exact identity rechecks, not a one-time permission stamp.
No original-inode descriptors return to the parent. Use bounded private IPC,
closed operations, explicit timeout/error refusal, captured child ownership and
guaranteed cleanup reporting. No listener, daemon, generic SQL/callable RPC,
unbounded retention or global SQLite connection registry is introduced.

The live TTS exact-current proof requires one fixed metadata-only validator
operation in its helper. It retains original proof descriptors, invokes the
existing immutable current-schema/domain/metadata checks, and returns only
bounded proof status/identity. It never loads reference BLOBs, copies the whole
database or moves live transactions out of the repository. This preserves
ADR-051's metadata-focused startup while avoiding parent-side live proof closes.
The approved written design includes this fixed supporting operation.

Repository restore receives an explicit revalidated parent/sidecar identity
handoff before closing the live proof. Tombstone settlement retains a verified
parent-directory handle, not original database/sidecar FDs. Both existing
repository consumers must migrate with the wrapper; absent fields must not
silently bypass cleanup.

Lost live TTS proof is terminal: retain the SQLite handle, SHARED store lease and
owning worker, report restart required, and do not force-close or mint replacement
proof. Close retry fails promptly; process exit releases retained authority.
New TTS admission is latched off after such a failure while healthy siblings
remain usable. The user approved this availability tradeoff on 2026-09-07; it is not a claim of
successful or bounded-time in-process SQLite cleanup. Early failures before a
live connection exists use ordinary resource cleanup.

Bound eight live helpers as four retained TTS owner permits plus four transient
slots. Quarantined owners retain their permits. Reserve each transient operation's
whole helper envelope before launch; nested calls use that reservation. TTS
initialization reserves its retained and transient capacity atomically. Existing
absolute operation deadlines govern admission and IPC together; SQLite lock-wait
timeouts remain separate. Canceled async waiters do not abandon shielded workers.

Local raw artifact descriptors remain permissible only for genuinely closed,
exclusively owned migration/publication/recovery artifacts. Their actual owner
must close all SQLite views before raw descriptors and retain authority on close
failure. A shared lease or `immutable=1` does not establish that precondition.
Existing publication/recovery finalizers require correction and close-failure
tests; this ordering is a requirement, not an assertion about current code.

All ADR-029 no-follow/type/owner/link/identity/mode checks and read-only exceptions
remain binding. Windows ACL verification remains unclaimed. No schema, journal,
native dependency, Canvas permission or release policy changes are implied.

## Alternatives

- Retaining all local descriptors requires a complete cross-owner connection
  registry; partial tracking is unsafe and indefinite retention is unbounded.
- Metadata-only checks alone miss wrong-mode hardening and proof-descriptor
  lifetime. They are not a complete correction.
- Copying TTS evidence can read/persist up to 576 MiB at ordinary startup and
  conflicts with the existing metadata-focused contract.
- Disabling WAL or background maintenance removes expected behavior without
  correcting lock ownership.

## Consequences and qualification

Connection acquisition gains process/IPC cost. Batch validation, explicit
resource bounds and measured startup/TTS/Canvas qualification are required;
existing performance ceilings are not raised. Unavailable helpers refuse the
affected operation rather than falling back to unsafe local checks.

Real lock-preservation regressions cover WAL/rollback modes, sibling handles,
backup release, partial failures and TTS shared-reader cleanup. Privacy, package,
import-isolation and helper lifetime gates accompany targeted integration tests.
Restore authority handoff, tombstone settlement, capacity saturation, shared
deadlines and explicit terminal-retention/process-exit behavior are required
regressions. Normal leak-free cycles and terminal quarantine are reported
separately.
Canvas Mermaid stays disabled until its independent release gate passes.

## Links

- [Detailed design](../../Docs/superpowers/specs/2026-09-07-sqlite-lock-safe-private-validation-design.md)
- [TASK-31942](<../tasks/task-31942 - Resolve-native-SQLite-crash-blocking-Canvas-qualification.md>)
- [ADR-029](029-local-private-data-boundary.md)
- [ADR-028](028-character-tts-generation-profile-ownership.md)
- [ADR-051](051-private-tts-clone-reference-assets.md)
