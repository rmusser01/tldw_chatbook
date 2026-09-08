# Native maintenance admission and publication

TASK-31987 implements the foundation in [ADR-126](../decisions/126-complete-local-backup-and-recovery.md). It does not expose Complete backup, isolated restore, or destructive replacement.

## Participating owners

Use one stable, private `Admission(control_root)` outside every managed target. The root must be an existing trusted-parent path with no symlink components. The constructor creates the final directory when absent, but never reconstructs missing/corrupt state in an existing directory. Every subsequent acquisition verifies the control directory identity and native qualification. Lock files remain permanently named and are never replaced or removed by this protocol.

Register installed owner declarations with `register(namespace, roots)`. Roots must exist and have verified file/directory identity. A future destination is first created through private staging. Shared paths, symlinks, hardlinks and overlapping parent/child roots acquire the same ordered set of namespace locks. The registry keeps historical path/inode alias evidence across replacement and remapping, conservatively retaining old associations even if an inode is later reused. A busy shared-alias registration refuses with `namespace_busy`; callers release their existing work and retry from a new scope preview.

Enter `normal(namespaces)` before opening an owner and keep the context until its transactions and connections retire. Declare all namespaces for cross-owner work up front. Nested admission on a thread is refused. Owners must preserve drafts and finish transactions themselves; the coordinator neither kills workers nor rolls back transactions. Exiting the context is the owner's statement that retirement completed.

`maintenance(namespaces, timeout, *, cancel=None)` closes admission gates, then waits for established owner holds without retaining a registry lock. Once owners retire it acquires shared registry authority, rechecks the alias set, and holds that scope through the context. A timeout or per-call `threading.Event` cancellation releases acquired holds and raises a sanitized `AdmissionTimeout` or `AdmissionCancelled`. Gate and lease ordering is deterministic; ordinary growth within a registered store does not change its namespace.

`incompatible(namespaces)` represents **positively known incompatible activity** for its OS-held lifetime. Maintenance refuses this evidence with `known_incompatible_client`. This adapter does not make arbitrary older clients cooperative. `Utils.instance_lock.InstanceLockStatus` remains advisory and fail-open; neither it nor a PID scan proves that all legacy/external writers are absent.

## Mapping reservations and interrupted work

`remap(namespace, roots, timeout, *, cancel=None)` first persists a version-1 pending operation token and proposed roots under a short registry-exclusive hold. Both old/new aliases are reserved before that hold is released. It then takes ordered gates and owner leases, drains established work without the registry lock, verifies the reservation and root identities, and publishes the final mapping under registry authority.

Timeout, cancellation or process death after reservation leaves `pending`, `proposed` and historical aliases intact. Both generations are retained and affected admission refuses `remap_recovery_required`; there is no silent clearing, PID-based completion inference or automatic retry. Task 18's local control/journal reconciliation must inspect the operation token, old/proposed locators and actual filesystem generations under exclusive admission, establish a verified commit/rollback outcome, and only then publish reconciled registry state. This task intentionally does not provide an unsafe general-purpose clear method. Bootstrap associations and durable restore-operation/activation fences remain later tasks; death of an ordinary maintenance holder releases its OS locks and is not itself a durable restore fence.

## Qualified native operations

`qualified_for(operation, root)` reads the installed versioned `native_qualification.json` and current pinned native identity. It never creates probe files or promotes support at runtime. The shipped evidence is restricted to Darwin 25.5.0, arm64, Python 3.12.11, APFS mount flags 76583040, and these operations:

- `publish_new`, `publish_file`, `publish_directory`: native `renameatx_np(RENAME_EXCL)` with pinned parents, including regular files and empty/populated directories.
- `admission`: the cooperative process protocol above, stable native `flock` objects and private durable registry publication.

Different OS/architecture/Python/filesystem/flags return unavailable. `isolated_restore`, `replacement` and any unknown operation remain unavailable even on the evidenced host. Later release qualification owns broad capability promotion. The matching native identity is rechecked on the descriptors used for publication/control locking, not inferred solely from a prior pathname lookup.

`publish_new(staged, destination)` checks native qualification, owned regular-object identity, single-link files, same-volume placement and a no-follow private directory tree. It flushes files (`fsync` plus Darwin `F_FULLFSYNC`) and directories before publication, then flushes destination/source parent directory entries. Existing targets of every kind are refused by the atomic native primitive. There is no exists-then-replace or copy/delete/hardlink fallback. A flush failure after rename is an ambiguous result: preserve actual names and bytes for the caller's journal reconciliation, never delete an observed destination or retry with overwrite. Directory publication is a primitive capability; it does not establish isolated-profile validation or whole-installation replacement support.

Callers own and freeze staged bytes before publication and journal their operation intent. Native identity checks do not promise exclusion of arbitrary same-user out-of-protocol modification. `create_private_directory` and `create_private_file` are low-level no-follow, exclusive owner-private staging helpers, not publication or whole-operation capability claims. Their output remains owned by the calling persistence adapter and must be inventoried there.

## Native evidence

The required tests execute without native skips on the identity above. Four concurrent publisher processes produce one new artifact and three `FileExistsError` results. Independent admission processes demonstrate inode replacement, alias and nested-root exclusion, SQLite commit/connection retirement, ordered contention, timeout/cancellation, known-incompatible activity, holder death and retained remap reservation evidence. A child exiting immediately after native rename leaves the published generation inspectable and the stage name absent. This is process-crash/OS-flush evidence, not a simulated physical power-loss or disk-controller test.

Focused command: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Backup_Recovery/test_admission.py Tests/Backup_Recovery/test_native_files.py -q`.

The source census records admission's durable control producers as unsupported for capture/relocation until the control owner cohort qualifies. The wheel/sdist resource contract includes the installed native qualification record. No developer configuration, databases, credentials or keyring were used as fixtures.
