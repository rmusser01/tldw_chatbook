# Lock-safe private SQLite validation

Date: 2026-09-07
Status: Revised after design review; helper-process approach approved, revised contract awaiting approval
Task: TASK-31942
ADR: [ADR-125](../../../backlog/decisions/125-lock-safe-private-sqlite-validation.md)

## Purpose and evidence

Repair private-file inspection without canceling locks held by live SQLite
connections. Keep Canvas Mermaid disabled until the corrective work is reviewed
and its release qualification is rerun successfully.

On Python 3.12.11 / SQLite 3.49.1 / macOS 26.5.2, a real connection holding
`BEGIN IMMEDIATE` excluded a separately exec'd writer with `SQLITE_BUSY`.
Calling `connect_private_sqlite`, or its actual artifact validator on `-shm`,
then admitted that writer while the original connection remained in a
transaction. A normal SQLite connection preserved exclusion. Raw main-file and
WAL-file closes preserved this particular WAL write-lock test; that is not
evidence that main-file closes are safe for rollback locks.

The isolated cause is a non-SQLite `close()` of the shared-memory inode.
[SQLite documents this POSIX lock hazard](https://sqlite.org/howtocorrupt.html#posix_advisory_locks_canceled_by_a_separate_thread_doing_close_).
It is consistent with the captured native WAL recovery/frame-lookup SIGBUS,
but does not establish every historical crash's precise interleaving.

## Preserved contracts

- ADR-029's descriptor-based no-follow, regular-file, single-link,
  current-effective-user, identity and private-mode checks remain in force.
  Existing eligible wrong-mode files are still hardened or explicitly refused.
- Retain configured/custom parent handling, read-only source mode preservation,
  missing-file policy and optional-sidecar generation/refusal behavior.
- Keep SQL, transactions, connection factories, WAL concurrency, backup and
  restore ownership at their existing callers. No journal-mode or mmap workaround,
  SQLite replacement, schema migration, full database copy or new dependency.
- Windows retains its explicit unverified ACL posture; do not claim this POSIX
  correction qualifies Windows permissions or locking.
- No network listener, daemon, generic SQL/callable execution service, settings,
  environment enable switch, global SQLite registry or unbounded FD retention.
- No user database, provider, process or shared Python environment is used by
  tests. Repository pytest pre-import isolation remains mandatory.

## 1. Operation-owned helper boundary

Use a fresh exec'd helper for one private-file preparation batch or one retained
proof lease. The parent never opens, duplicates or receives original database or
sidecar file descriptors for these checks. Closing the helper's descriptors
therefore cannot cancel the parent's POSIX locks.

One preparation request covers the selected DB and its three fixed suffixes
`-wal`, `-shm`, `-journal`; do not launch four interpreters. The helper uses the
same extracted leaf validation implementation, including descriptor-relative
parent checks and optional-sidecar revalidation. It is not a second, weaker
implementation of privacy policy. Directory descriptors may remain local where
needed; they are not the SQLite file inode.

Parent policy admission remains authoritative: registered owner, target kind,
read-only/must-exist flags and source-mode policy are validated before launch.
The helper validates a closed request schema too. It accepts no caller-chosen
suffix, executable, import name, SQL statement, callback or environment patch.
Normal memory databases do not launch a helper.

The launcher uses the current interpreter and installed, package-owned helper
code, without a shell or user working-directory import search. General file
validation loads only standard-library and dependency-leaf privacy modules, not
application startup, config, keyring, providers, logging sinks or model refresh.
Packaging tests must exercise an installed wheel, not only checkout file paths.

## 2. IPC and failure ownership

Communication uses inherited private pipes, never a listening socket. Paths
travel only in bounded private request frames, not command-line arguments or
logs. No database contents, exception messages or raw traceback are returned.

Protocol version 1 uses length-prefixed JSON frames, a maximum of 64 KiB per
request or response, closed per-operation fields and one outstanding request per
lease. Identity projections carry explicit integer device/inode, mode, uid/gid,
link count, size and nanosecond timestamps where the existing comparison needs
them; no lossy floating-point identity conversion. Parent and helper independently
check shapes and limits before use. Unknown fields, versions, operations,
malformed lengths and oversized output fail closed.

File operations are `prepare`, `pin_source`, `recheck_source` and `close`.
A source lease binds one initial path/identity; recheck never accepts a replacement
target. Success returns only bounded status and identity metadata. Existing
privacy statuses and source-free reasons remain distinct from helper-unavailable,
protocol and timeout failures. None triggers local unsafe validation as fallback.
Permission changes already safely completed before a later failure may remain;
do not widen modes to simulate rollback. No helper operation writes database
content. New-file precreation retains the existing private empty-file behavior.

Each launch/control round trip has a five-second deadline. Close gets one second
to exit normally, then the owner terminates and, if necessary, kills and reaps
only its captured child with a further two-second cleanup bound. Control-flow
exceptions remain primary. A failed cleanup is reported, never silently called
successful. The helper cannot launch descendants; stderr is discarded and errors
are encoded through the bounded protocol.
Pass an explicit parent-local absolute monotonic deadline through the private
seam, source leases and TTS adapters. Consume it before forwarding SQLite kwargs;
it is not SQLite's separate lock-wait `timeout`. Admission, launch and every IPC
wait use the lesser of the remaining operation budget and their own maximum;
retries never reset the operation budget. No Python callback crosses IPC. Existing
deadline guards remain active during parent-side SQL and publication. Preserve
the current five-second default TTS backup budget and earlier caller deadlines.

Cancellation preserves the existing worker-ownership contract: canceling a
shielded async waiter does not abandon its still-running repository worker or
release that worker's helper. A worker-observed cancellation/deadline stops new
work and enters owned cleanup. Cleanup may consume its separately stated bound
after the operation deadline; report that distinction, and never claim all
resources were released merely because the caller stopped waiting.

EOF releases helper-owned descriptors and ends the helper. Retained helpers also
check their original parent identity during idle polling at least once per second
so a leaked inherited pipe cannot orphan them. A lease is not reused across fork
or interpreter/process replacement. It remains valid for its owning operation
or connection lifetime, not an arbitrary idle duration that could invalidate a
legitimate long backup or idle TTS repository.

At most eight helpers may be alive per owning Chatbook process: at most four
retained TTS proof owners, with four slots reserved for transient work. Idle TTS
repositories cannot borrow transient capacity. Terminally quarantined TTS owners
continue to consume their owner permit even after their child is reaped.

Reserve a transient operation's complete helper envelope atomically before it
opens any helper: one slot for a normal preparation, two for a source-pin
backup/copy/restore (one pin plus one sequential preparation). TTS admission
reserves one retained permit and one transient preparation slot together; it
must not hold one while waiting for the other. Nested internal calls borrow the
owning operation's reservation, not acquire more global capacity. A reentrant
callback that exceeds that envelope refuses before launching or mutating its
nested target; it cannot wait while holding an insufficient reservation. Verify
these envelopes against every migrated caller; a larger required envelope is a
design gate, not permission to increase the eight-child ceiling.

Acquisition uses the shared operation deadline, with a five-second maximum.
Release child capacity only after reaping and unused reserved capacity when the
operation settles. A child that cannot be reaped remains charged to its owner.
This is bounded admission accounting, not a connection registry, daemon or pool.
Saturation rejects only the unadmitted operation; it does not invalidate healthy
proofs or require those proofs to release capacity needed by their own work.

## 3. Normal connections and online backup pins

`connect_private_sqlite` retains its caller-owned SQLite return value and custom
factory behavior. Batch preparation completes in the helper before the parent
calls SQLite. Preserve the existing expected-identity check immediately before
opening; the helper does not claim to eliminate SQLite's existing pathname-open
race. Private validation does not reopen an original inode locally afterward.

`_PinnedSQLiteSource` retains selected path and exact identity plus a helper
lease instead of parent-held original file FDs. The helper pins the source and
parent for the whole enclosing backup/copy/restore operation. Every existing
`_reverify_source` boundary checks retained descriptors, current named identity,
parent authority and privacy through that lease. Lease death is an identity/
operation failure, not a stale cached success.

The caller still executes SQLite's online backup with its current callbacks and
transaction guards. Borrowed source/destination connections stay borrowed.
Backup exceptions and cancellation close only owned SQLite handles and the helper
lease. Preserve restore safety snapshots, rollback/indeterminate reporting and
the existing exclusive destination protocol; do not redesign restore journaling.

## 4. Live TTS exact-current proof

This is the one additional fixed validator operation needed to close the live
descriptor hazard. `open_exact_current_profile_store` currently retains main and
WAL/SHM proof FDs beside a live connection. Two repositories can hold SHARED
leases, so closing one wrapper does not prove another connection is absent.

Move those original-inode pins into an operation-owned helper lease. A fixed
`tts_exact_current` initializer opens one helper-owned immutable evidence view
through its own pinned descriptor and runs the same current-version, schema,
domain-row and incremental metadata validators. This operation is not arbitrary
SQL RPC and never accepts SQL, serialized Python objects or callable names.
Only existing bounded status/identity facts cross IPC; no rows, reference text,
audio bytes or profile contents are returned or logged.

Keep the evidence SQL read-only and metadata-focused under ADR-051: no BLOB
selection, full serialization, snapshot file or up-to-576-MiB startup copy.
Use the existing one-million metadata-row ceiling and 576-MiB artifact ceiling,
without increasing either. Initial fixed validation has a 30-second maximum,
with bounded cancellation/cleanup; subsequent identity rechecks use the normal
five-second deadline. The validator runs off the UI event loop. Its imports must
be isolated to the existing pure schema/domain validators and standard library;
extract leaf code where necessary, with no application/config initialization.
It opens only its own fixed immutable evidence view, never recursively launches
helpers, and closes that SQLite view before releasing any of its file pins.
After initial validation, it retains only the pins needed for identity checks.

After proof, the parent opens and validates its live SQLite connection through
the normal private seam. Preserve query-only-before-admission, exact version,
post-init identity, sidecar-cohort and namespace checks. If live SQLite creates
previously absent sidecars, a fixed `tts_pin_sidecars` command captures only that
lease's original path cohort; mixed or substituted cohorts remain refused.
`tts_recheck` performs the existing pinned-parent/main/sidecar comparisons before
each currently guarded use. Never silently rebuild proof after helper death or
retarget a lease to a new inode.

The wrapper retains the helper lease instead of raw original-inode FDs or a local
immutable evidence connection. It may retain a separately verified directory-only
parent FD for the concrete namespace consumers below. Normal close revalidates
authority, settles required tombstones, closes its SQLite connection, then
closes/reaps the helper and releases directory/owner resources. If SQLite close
fails while proof remains available, retain the same quarantined owner/lease for
the existing cleanup retry; do not publish a usable repository. Partial setup
failures must preserve locks held by an independent live sibling.

### Repository authority handoff

There are two external consumers of the current wrapper's descriptor fields;
replacing the fields without migrating these callers is not compatible:

- `_worker_close_for_restore` captures parent and WAL/SHM identities before
  checkpoint/close. Introduce a fixed `tts_export_restore_authority` operation
  that revalidates the existing lease and returns an immutable, bounded identity
  record tied to the same repository generation and cohort. Preserve the current
  pre-checkpoint capture and post-checkpoint revalidation. Before closing, the
  repository installs this record for exact sidecar cleanup after acquiring its
  exclusive restore lease. Missing/failed export refuses restore before close;
  no empty-dictionary `getattr` fallback. Exported metadata is only authority for
  the existing exact namespace comparisons, not permission to remove a changed
  target or to keep using SQL after helper loss.
- `_worker_cleanup` uses the parent FD to settle reusable tombstones. The wrapper
  retains a parent directory FD opened through the existing verified-parent seam
  and matched to the helper's parent authority before admission. An explicit
  accessor rechecks the helper and local directory identity before settlement;
  never silently skip settlement because a private field disappeared. Directory
  handles remain parent-local; no database/sidecar FD is transferred or reopened.

Keep uid/gid, full mode and exact integer identity fields required by
`ParentAuthority` and post-init comparisons. Use an explicit validated metadata
type at the IPC adapter, rather than inventing incomplete `os.stat_result`
objects. Migrate the internal identity consumers together while preserving the
public `expected_identity` contract for existing callers.

### Lost proof: explicit terminal quarantine

A dead helper or irrecoverably failed proof channel once a live SQLite handle
exists, including partial setup, cannot be repaired by merely restatting the
pathname: that would mint new proof
without the original pins. This revision deliberately chooses terminal retention,
not an unproven teardown-only reopen. Mark the repository unavailable with a
source-free **restart required** reason, reject further use and automatic reopen,
and retain its SQLite handle, SHARED store lease, directory authority and owning
worker until process exit. Close/retry returns the same terminal failure promptly;
it neither issues SQL nor pretends resources were released. Reap the dead child
independently where possible. An ordinary namespace mismatch with a healthy
helper still supports the existing exact-authority restoration and close retry.

Retain terminal owners through existing application/repository ownership, never
drop them to trigger SQLite finalizers. Latch new TTS repository admission off
for this process after a live owner loses proof; already healthy siblings remain
usable. Together with the four-owner permit limit this prevents repeated reopen
attempts from accumulating quarantined connections or workers. No polling worker,
automatic process restart, force-close, foreign-sidecar deletion or global
all-SQLite registry is introduced. If proof fails before a live connection exists,
ordinary helper/directory cleanup releases admission; it need not latch TTS off.

This is an explicit availability tradeoff requiring written-design approval:
safe in-process cleanup is not promised after loss of irreplaceable proof. Other
processes may remain unable to obtain the exclusive store lease until this
Chatbook process exits. Tests must show prompt terminal errors, bounded retention,
foreign-cohort preservation and release after owned process exit, not label
terminal quarantine a successful close or a leak-free normal lifecycle.

## 5. Exclusive migration artifacts and descriptor views

The generic local descriptor-view path is no longer valid for a live store.
Remove the now-unused live-store descriptor owner from normal admission.
Local immutable descriptor views remain only for explicitly closed publication,
recovery or opaque migration-candidate lifetimes under existing exclusive
repository ownership. Document and test that precondition at real callers;
`immutable=1`, a filename or owner-id string alone is not proof of exclusivity.

For these exclusive artifacts, borrow the existing verified descriptor during
SQLite open instead of creating and immediately closing an unnecessary duplicate.
Close all owned SQLite handles before raw artifact FDs. Preserve retained
ownership when a close fails, exact namespace/content rechecks and the existing
callback-no-escaped-handle contract. Do not add a global all-SQLite registry.
In particular, publication `_immutable_validate` and recovery
`_validate_authoritative_targets` currently have `finally` paths that close raw
FDs even if SQLite close raises. Replace those paths with explicit retained
ownership on close failure; current code is not evidence that the required
ordering already holds.

Inventory all original-inode closes in `DB/private_sqlite.py` and its descriptor
consumers in TTS schema/publication/recovery/namespace/repository modules. Each
must have either helper-process ownership or a tested closed/exclusive artifact lifetime.
Directory-only closes do not require file-inode isolation. An unproven live
caller is a blocking gap, not a grandfathered exception.

## 6. Verification and performance gates

1. Promote the diagnostic into deterministic real-SQLite regressions. Preserve
   a strict `SQLITE_BUSY` oracle, normal SQLite positive control and a deliberately
   isolated raw-close negative control. Cover WAL writer/read-snapshot exclusion
   and rollback-journal locks, including another thread and another process.
2. Exercise correct and wrong-mode main/sidecar files, read-only/must-exist opens,
   actual connection factories, replacement/symlink/hardlink/wrong-owner failures,
   optional generations, helper crash/timeout/EOF/oversized IPC and cleanup.
3. Hold an independent connection's transaction across actual backup source-pin
   release and failure paths. Verify borrowed handles remain usable and committed
   rows/integrity stay exact. Cover restore rejection and existing recovery rules.
4. Exercise two real TTS repositories, sibling lock preservation on close and
   partial failure, substituted parent/main/sidecar identities, metadata-only
   startup, post-init binding and exclusive migration/recovery. Cover real restore
   with retained sidecars, refusal when authority export fails, and tombstone
   settlement at close. Inject SQLite-close failures in exclusive validators and
   assert their raw pins remain owned.
5. Kill a retained helper in an owned test process: assert subsequent use/close
   fails promptly, no SQL cleanup or automatic re-admission occurs, repeated
   retries do not grow resources, healthy sibling locks survive, foreign files
   remain unchanged, and exclusive access becomes possible after process exit.
   Qualify both ordinary application shutdown (including interpreter finalizers)
   and abrupt owned-process exit; a kernel-only exit test does not prove orderly
   shutdown safe. Unsafe finalization is a blocking gate, not permission to force
   process termination or weaken foreign-cohort checks.
   Separately prove normal/early-failure cycles reap helpers and release FDs.
6. Saturate retained and transient admission using barriers, including simultaneous
   TTS opens and backup/restore. Admitted work must have enough reserved capacity
   to finish; excess work refuses before mutation. Exercise nested-envelope
   refusal, failed reaping, partial-frame IPC deadlines and canceled async waiters
   whose workers still own resources. A short caller budget must not become
   repeated five-second waits. Verify parent exit releases proof helpers. Missing
   permissions or OS support are failures/declared gaps, never passing evidence.
7. Run targeted DB/TTS/static/package checks and the exact previously failing
   actual Canvas child workflows. Then obtain independent correction review
   before resuming all required Canvas candidate/admitted qualification gates.
   No full repository suite, PR, push or merge is authorized by this design.

Measure full helper launch/batch latency, representative threaded app startup,
TTS metadata-only open and actual child workflows against the unchanged baseline
with identical synthetic data. Five bare interpreter launches measured a 21.34ms
median here; that is a launch floor, not an implementation benchmark. Existing
startup/import/UI guard ceilings must stay unchanged. If end-to-end performance
fails them, revise this design rather than add a persistent pool, skip privacy
checks or increase budgets unilaterally.

## Rejected alternatives and scope

An all-connection FD-retention registry must understand raw/borrowed handles,
custom factories, GC and inode generations; an incomplete one is unsafe and
unbounded retention leaks resources. A metadata-only fast path does not handle
wrong-mode hardening or live proof closes, so it is not the repair. Full TTS
evidence copying conflicts with metadata-focused startup and can read/persist
hundreds of MiB unnecessarily. Disabling WAL, maintenance or fail-closed transport
would hide symptoms instead of repairing lock ownership.

This change does not add user-facing TTS features, migrate data, change Canvas
semantics, qualify historical crash causes or provide a general process service.
Helper-based TTS validation is a fixed supporting operation, not a transfer of
the repository's transaction or lifecycle authority.
