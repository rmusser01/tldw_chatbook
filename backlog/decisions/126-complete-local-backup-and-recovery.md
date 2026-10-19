# ADR-126: Complete local backup and recovery

Status: Accepted — revision 4 approved by the user on 2026-09-07
Date: 2026-09-07

2026-09-12 user correction: the Go implementation/build/delivery choice in decision
6 is superseded by the [Python encryption correction](../../Docs/superpowers/specs/2026-09-12-python-backup-encryption-design.md)
(TASK-32561). The existing age v1 format, isolated worker, integrity checks and
recovery boundaries remain. See the [verification record](../../Docs/Development/backup-python-verification-2026-09-12.md).

2026-09-12 platform correction (TASK-32562), explicitly requested by the user:
the existing recovery feature must operate on macOS, Linux and Windows. Use the
[cross-platform correction spec](../../Docs/superpowers/specs/2026-09-12-cross-platform-backup-correction.md).
Native APFS, ext4 and local NTFS contracts replace exact OS/Python patch matching;
actual identity acquisition and every operation still enforce supported volumes,
privacy, containment, no-overwrite publication, locking and persistence barriers.
POSIX callers retain the standard library operations. Windows callers use a local
Python ctypes filesystem interface with native handles and ACL checks. No global
standard-library monkeypatch, new language, archive format or encryption dependency
is introduced. Real installed product tests on each platform are required before
claiming completion; native primitive tests alone are insufficient.

The correction is verified by actual installed backup/restore/replacement/rollback
workflows on all three platforms and final Windows owner support tests. Exact
revisions, artifact receipts and remaining merge limitations are recorded in the
[cross-platform verification record](../../Docs/Development/backup-cross-platform-verification-2026-09-12.md).

Revision: 4 — incorporates the fourth user-requested design review.

2026-09-15 user-authorized architecture migration (TASK-32628): expose selectable
data groups within the existing Python backup set and recovery coordinator.
Everything remains the default; shared physical stores remain indivisible.
Restore replaces reviewed selected/required groups after a verified safety copy
and preserves unselected stored data. Configuration support is distinct from
permission to replace Settings. Version-one archives remain readable. The
[selectable-group spec](../../Docs/superpowers/specs/2026-09-15-selectable-backup-groups-design.md)
and [implementation plan](../../Docs/superpowers/plans/2026-09-15-selectable-backup-groups.md)
define the migration and its cross-platform verification. This amendment does
not authorize record merging, scheduling, cloud storage, or unrelated changes.

Refinement explicitly approved by the user on 2026-09-15 after static review:
use existing registered directory authority for an absent redundant
child alias only after historical canonical-path, native parent-identity and
per-profile ownership checks. Raw registry/profile records and historical alias
tokens remain unchanged, and the absent alias adds no actual source authority.
This also accepts external absence of that same redundant child; missing roots
without such already owned coverage still refuse. The selectable-group spec
records admission, fresh recovery and finalization requirements for this case.

2026-09-29 owner-approved amendment (PERF-07/PERF-08, TASK-33266/33267):
ordinary storage admission may reuse the allowed result of an unmodified
derivation while per-call `lstat` stamps of every walked chain and the
records/registry/selector content stay identical. See "PERF-07/PERF-08
amendment" at the end of this ADR.

2026-10-03 owner-approved amendment (TASK-34100.17): provider keys a user chose
to keep in the OS keychain are Chatbook-owned keyring values for backup and
isolated restore. See "TASK-34100.17 amendment" at the end of this ADR.

Task: [TASK-31978](../tasks/task-31978%20-%20Design-complete-local-backup-and-restore.md)

Design: [Complete local backup and restore](../../Docs/superpowers/specs/2026-09-07-complete-local-backup-restore-design.md)

Implementation: [Roadmap and component plans](../../Docs/superpowers/plans/2026-09-07-complete-local-backup-restore.md)

Extends: ADR-004, ADR-029, ADR-030, ADR-036, ADR-059, ADR-060

Supersedes: None. Existing selective Chatbook export and storage-settings Save
contracts remain unchanged.

Allocation: 126 was free across 425 locally available branch/remote refs and 68
worktrees on 2026-09-07. Recheck upstream/open-branch allocations before merge.

## Context

Users need an exposed, complete recovery workflow covering Chatbook-owned local
data, both replacement and isolated-profile restoration, and explicitly selected
external files/models. Existing selective Chatbook bundles are not a lossless
installation image; the legacy bulk database backup covers only three databases.
Configuration and databases can live outside default roots. Other stores own
assets, recoverable journals, indexes, credentials, and device-local operational state.

Current profile instance locking is advisory. Individual SQLite snapshots do not
coordinate multiple stores and referenced files. A full restore spans filesystems
and cannot be one SQLite or portable filesystem transaction. Normal startup can
migrate databases, start workers, or erase temporary media before a restore is
known safe. A copied profile can alias original databases or reuse shared credentials.

Portable credential exclusion also conflicts with exact rollback: a safety copy
cannot remove the credentials needed to recreate the old installation state.
Device-bound journals and permissions contain useful recovery data but cannot grant
new filesystem or execution authority when imported.

## Decision

1. Add an explicit versioned recovery archive and app-owned recovery service,
   exposed through canonical F9 Settings, first-run setup, and a dependency-light
   pre-bootstrap recovery launcher. Keep selective content import/export separate.

2. Use a declared storage-owner inventory and explicit profile list. Default coverage
   includes identified app-owned durable data and configured custom storage. External
   folders, models, and currently available temporary media are optional. Unknown or
   unavailable required data blocks a complete result. Partial archives are labeled
   partial and cannot perform installation replacement in v1. Server data is excluded.
   Valid current configuration is required for complete backup-source discovery, not
   archive inspection or isolated restore. Restore destination discovery uses the
   archive and independently verified local targets; damaged config never causes a
   silent fallback to default database paths. Unverifiable replacement targets block
   replacement while inspection and new-destination recovery remain available.
   Revalidate selected owners, effective mappings, shared aliases, dependencies, and
   required assets after maintenance admission closes and writers drain. Scope or
   budget changes require release, renewed preview, and ordered reacquisition; normal
   growth within approved owners is captured within rechecked limits. Reconcile the
   final manifest with the fenced capture inventory before writers resume. Discovery-
   time counts cannot establish completeness.

3. Coordinate protocol-aware persistence participants through enforced maintenance
   admission over stable logical namespaces outside replaced data. Verified file/path
   identity establishes shared-store aliases; replacement of an inode does not create
   a new unlocked namespace. Reserve both namespaces during remapping, acquire locks
   deterministically, and drain without circular waits. For ordinary backup, resume
   after coherent capture, before packaging. Replacement/later rollback remains fenced
   through encrypted rollback verification, publication, and installed validation.
   Known incompatible activity blocks maintenance. Arbitrary legacy/external processes
   do not honor the protocol and are outside its exclusion guarantee; PID scans are
   not proof of their absence. Native safety qualification remains required. External
   folders receive per-file consistency, not a whole-folder point-in-time guarantee.

4. Use a ZIP64 container with a manifest, dependency groups, schema versions, sizes,
   digests, coverage, and relocation metadata. Inspect/execute only a completed private
   source copy or qualified immutable snapshot bound to the preview by digest; a
   pinned handle alone does not prevent in-place modification. Enforce source-copy,
   decrypted-container, header, extraction, and crypto budgets before/during streaming.
   Authenticated bytes remain untrusted application input. Owner-specific schema
   allowlists, trusted_schema=OFF, restricted functions/authorizers, and SQL execution
   budgets apply before and during staged migration. Installed migration SQL must
   not activate unvalidated imported schema; no unrestricted-connection fallback.
   Represent selected folder topology, empty directories, and versioned supported
   directory metadata explicitly in the manifest. Directory/file names share bounded
   collision/containment checks. Restore parents before children and metadata last;
   unsupported metadata requires an explicit partial-metadata result. New archives
   and rollback copies publish atomically to new files only, never overwrite an
   existing destination, and reject protected source/control aliases or unexcluded
   output-inside-source layouts. Check-then-replace publication is insufficient.

5. Exclude managed credentials by default using typed owner adapters on staged
   copies, including qualified treatment of SQLite secret remnants. Explicit export
   of supported readable credentials requires encryption. Arbitrary user content is
   not promised secret-free. Environment contents and unrelated keychain entries are
   not exported; existing memory-only Sync key boundaries remain intact.

6. Encrypt the whole recovery container with age v1 passphrase encryption using a
   small bundled helper built from the official age Go library. Do not invent a
   cryptographic format or repurpose config-value encryption for large files. Secrets
   pass through anonymous pipes, never command arguments/environment or persistent
   request files. Qualify streaming, resource limits, packaging, interoperability,
   dependency licensing, and every advertised platform as an early delivery gate.
   Define the helper protocol, integrity/version checks, release/update ownership,
   and wheel/source/editable install behavior before dependent feature implementation.
   Release installs must not unexpectedly need Go or download/resolve a helper at
   backup time. Missing/unqualified helpers fail capability preflight before passwords
   or maintenance; change the ADR if the selected integration cannot qualify. There
   is no plaintext fallback when encryption is requested or required.

7. Restore through private staging, immutable target mappings, qualified publication
   primitives, and a durable operation journal outside replacement targets. Check
   pending recovery before ordinary configuration fallback, migrations, cleanup, or
   service composition. Durable associations in a fixed bootstrap admission directory
   make custom control roots discoverable by every supported launch route, independent
   of the convenience catalog and restored config. Register before publication and
   clear only after a durable verified outcome. Ambiguous interruption blocks affected
   storage and preserves both generations. Other profiles may launch only if intact
   evidence proves their namespaces disjoint. Cross-volume replacement is recoverable,
   not globally atomic, and is enabled only on qualified platforms/filesystems.
   Before clearing an operation fence, persist its installed generation's independent
   activation-required state. Completing recovery is not permission to start work.

8. Replacement first creates and verifies an encrypted local rollback archive of the
   exact stored-data snapshot and supported captured managed credentials.
   The user supplies a rollback password before any live mutation. Portable export
   redaction does not apply to this artifact. Never overwrite shared keyring entries.
   Retain recovery copies until explicit deletion; protect unresolved evidence. A
   later rollback preserves intervening changes with another verified recovery copy.
   Replacement previews restore, retire into rollback storage, and preserve-outside-
   scope sets. Archive absence alone cannot authorize deletion; unknown files or
   unsupported old-owner mappings block publication until reviewed. Owners retire
   obsolete managed objects only after verified rollback, and final inventory checks
   reject accidentally active newer objects/sidecars outside the desired generation.
   Optional exclusions preserve only independent content. Invalidate affected query
   caches and disable dependent projections before restored sources become queryable;
   omitted indexes are retired into rollback or quarantined through their owners.
   Under ADR-030, restored and retained indexes need verified source identity/state,
   schema/embedding compatibility, and active-source reconciliation before retrieval.
   Missing provenance leaves retrieval unavailable with explicit prerequisites; no
   rebuild starts automatically. Apply the same contract to isolated restore and
   later rollback, with shared-index scope expansion/refusal and no activation bypass.
   Capture readable affected Chatbook-owned keyring values and scope mappings; old
   references alone do not guarantee later credential recovery. If an existing scope
   no longer matches, restore into a new scope through the owning adapter rather
   than overwrite shared credentials. Journal those references/scopes. Disclose
   unsupported/unreadable credentials before replacement and require acknowledgement
   of the missing credential coverage if proceeding. Exact local stored-data recovery
   does not imply that revoked tokens, expired sessions, or environment credentials
   become available; no provider authentication check runs automatically.

9. Isolated restoration uses a new data namespace, dedicated config, explicit fresh
   process launch, owner-aware path remapping, and new device/credential scopes. A
   small owner-private recovery catalog makes the profile reopenable without hot
   switching or building a general profile manager. Reject writable aliases to the
   original profile. Shared settings and ambient credential references do not silently
   become active in the recovered profile.

10. Restore external folders only to newly created destinations in v1. Original
    overwrite is deferred. Retain included temporary media under a durable recovered
    asset owner with stable catalog/reference identity and transcript resolution.
    Recovered assets enter baseline subsequent backups even when temporary-media
    capture is off. Explicit deletion/cleanup rechecks references and recovery holds;
    unexpected missing bytes render a missing-media state instead of resolving a
    different file. Explicit asset deletion journals a versioned owner tombstone and
    payload retirement, retaining references marked intentionally deleted. Validated
    tombstones round-trip without requiring deleted bytes or making a backup partial;
    missing files or failed digests never imply intentional deletion. Restored deleted
    references show Deleted recovered media and cannot revive unrelated retained bytes.
    Temporary-store TTL/startup sweeps cannot remove committed recovered assets.
    Store models inertly; restoration does not launch/download them.

11. Restore operational definitions and history without restoring active authority.
    Schedules, queues, agents, model processes, network activity, tool permissions,
    sync bindings, claims, cursors, and pending journals stay paused/quarantined until
    existing owners complete explicit review/reconciliation. This defines the explicit
    recovery exception anticipated by ADR-060 without adding those fields to ordinary
    Chatbook export or activating imported device-local journals.
    A durable per-generation activation record is authoritative for every supported
    launch route, including later ordinary/headless launches. Per-owner approvals
    do not grant other owners permission. Completed-operation fence clearing, UI
    closure, report rebuilding, or imported approval claims cannot clear this state.
    Missing/corrupt state for a known restored generation keeps capabilities inactive.
    Every restore/rollback creates a new locally reviewed generation.

12. Distinguish Archive verified, Restoration validated, Opened successfully, and
    Needs setup. Release requires real data round trips, both destinations, startup
    independence, process/crash fault injection, credential isolation, and a first
    open with no unintended execution or network activity.

Second-review evidence additionally covers lock continuity across inode changes,
known incompatible participants, restore/retire/preserve reconciliation, in-place
archive mutation, resource failure before manifest parsing, schema-trigger attacks
during installed migrations, custom-root discovery through normal launchers, scoped
startup blocking, separate maintenance intervals, packaged helper availability, and
recovered-media deletion/re-backup. These extend the existing release gate rather
than claiming those runtime checks have already been performed.

Third-review evidence adds recovery without parseable current configuration,
activation persistence after ordinary relaunch and operation-fence clearing, changed
or unavailable keyring values, publication races against an existing good backup,
and empty-directory/metadata round trips. These are contract corrections; they do
not enlarge the feature into server credential repair or general filesystem backup.

Fourth-review evidence covers scope changes and in-scope growth between preview and
maintenance, stale/incompatible projections across restore and rollback, and explicit
media-deletion tombstones versus unexpected payload loss. Require final captured
coverage, unavailable stale retrieval until reconciliation, no automatic rebuild,
and complete tombstone round trips with interruption recovery. These remain required
implementation checks, not runtime evidence claimed by this documentation change.

## Alternatives considered

| Alternative | Reason not selected |
| --- | --- |
| Extend selective Chatbook export into total recovery | Requires every data domain to maintain a lossless logical serializer and still leaves settings, assets, device state, and exact rollback unsolved. |
| Copy the application directories | Misses custom paths and coordinated writes, can capture unrelated links, and cannot prove total coverage. |
| Restore each database from the Settings screen | Cannot safely publish a whole installation, survive broken startup, or coordinate cross-store rollback. |
| Rename a data folder to make a second profile | Config/custom paths and credentials can still alias the original installation. |
| Use credential-excluded export as the rollback copy | Cannot recreate the exact pre-restore credential/config state. |
| Restore imported permissions/journals unchanged | Transfers device-local authority and can replay writes against changed files or servers. |
| Custom chunked archive encryption | Adds unnecessary cryptographic protocol design and interoperability risk; use a maintained standard file-encryption implementation. |
| Overwrite original external folders in v1 | Requires independent conflict, metadata, multi-volume recovery, and external-writer contracts beyond app-owned replacement. |

## Consequences

This is a new recovery subsystem, not an extension of legacy Settings handlers.
It introduces storage-owner capture/relocation declarations, maintenance admission,
a recovery catalog/journal plus independent bootstrap admission records, an encryption
helper packaging dependency, a recovered-media owner, and a pre-bootstrap recovery
path. New schemas follow normal migration/version rules. Helper delivery and the
maintenance/bootstrap protocol must be qualified before dependent feature slices.

The main UI remains usable for inspection/progress, but coherent capture can pause
writes for the duration of snapshot work. Space is required for staging, encrypted
output, and retained rollback. Password loss prevents decrypting an encrypted
backup/rollback; passwords are not persisted automatically. Private plaintext
working files may exist during capture/restore, and no forensic-erasure claim is made.

Complete means complete for the shown profile inventory and selected content, not
proof that an unknown custom profile or remote server has been backed up. Restored
data is intentionally inactive until reviewed, so recovery is not a clone of live
execution state. The design preserves this distinction in UI and verification.

The approved implementation is decomposed into component plans and atomic tasks.
This decision does not claim the feature implemented or authorize a full test sweep.
Platform qualification can restrict destructive replacement while inspection and
safe extraction remain available.

## Related contracts

- [ADR-004](004-settings-storage-defaults-restart-boundary.md): ordinary path Save is restart-required and does not relocate live stores.
- [ADR-029](029-local-private-data-boundary.md): private files, checked paths, and metadata-only diagnostics.
- [ADR-030](030-derived-index-lifecycle-and-atomic-media-migrations.md): authoritative media lifecycle, derived-index reconciliation, and query-cache invalidation.
- [ADR-036](036-application-service-composition-lifecycle.md): one application composition root and memory-only Sync dataset keys.
- [ADR-021](021-file-backed-notes-disk-authority-and-recovery.md): File Notes filesystem authority and independent recovery ownership.
- [ADR-059](059-notes-folder-import-and-device-local-sync-ownership.md), [ADR-060](060-notes-sync-round-trip-and-interoperability-constraints.md): device-local Notes sync and explicit paused recovery boundary.
- [ADR-069](069-console-project-instruction-local-state-and-preflight.md): imported/local context state does not grant tool authority.
- [ADR-051](051-private-tts-clone-reference-assets.md): private voice reference assets and repository-owned validation.
- [age v1 format](https://age-encryption.org/v1) and [official implementation](https://github.com/FiloSottile/age): standard encrypted-container boundary selected here.

## Concrete chat source compound lifetimes (Task10 phase10)

An admitted dictionary mutation can commit its core row before a pause refuses its
history publication. Independent raw and SQLite scopes therefore do not establish
a coherent source boundary. The actual installed Persona/dictionary/citation source
pairs may preadmit their existing core and raw scopes and expose both discoveries
on the same PID, Thread and Task only after exact source/config/profile/DB/sidecar
identity and both gates are rechecked. Acquisition races unwind before body effects.
No transferable token, owner-string flag, arbitrary callback or post-pause member
addition is introduced. Factories bind the actual configured services before their
constructor loads; custom/no-file/memory/subclass sources remain ordinary and
unqualified. Existing native borrowers retain their actual owner-thread lifetime.

Dictionary history RLock and the narrow Persona/compatibility-cache path lock cover
body, result/cache restoration and native retirement. Canonical citation migration
retains its existing SQL claim/generation fencing and concurrent behavior: a new
shared citation lock deadlocked the real stale-generation test and was rejected.
That SQL fencing does not serialize unrelated raw cache writes. Exact service and
migration companions are preselected together without changing ADR-024 provenance,
ADR-037 persona identity, schema or migration/replay policy.

A failed mixed publication restores usable prior cache and retains sticky process
failure evidence; it does not roll back a committed DB. Native total_changes is
conservative evidence of possible effects (including rolled-back statements), not
proof of durable commit. Benign no-change optimistic conflicts remain usable.
Unrelated later reads/successes never clear mixed uncertainty. No new journal or
automatic reconciliation is added; restart is not proof of repair, and subsequent
source/archive validation must assess actual durable consistency.

The exact Chat_Dictionary_Lib parser/import/export/listing routes may finish fixed
external file IO across pause only through actual qualified ordinary native leases
and checked lexical/resolved/parent identities. Default export reads/materializes
its record under retained core admission before fixing its output name; no write
precedes complete member admission. External paths never enlarge capture ownership
and uncooperative external editors are outside the maintenance guarantee. Ordinary
unsupported platforms stay explicitly unqualified. No None lease grants paired
pause authority. Copy metadata uses retained descriptors; Darwin's missing Python
fchflags binding is filled narrowly by libc.fchflags(int, uint32_t), with explicit
ctypes signatures/use_errno and checked results. Only EOPNOTSUPP/ENOTSUP is ignored,
as in shutil.copystat; missing descriptor support cannot silently qualify fallback
path IO. Owned temporary/native uncertainty remains retained on failed retirement.

The concrete dictionary scope executor job reserves pending work before dispatch,
uses actual worker-thread admission, cancels queued entry atomically, and waits for
actual completion under running/repeated or independent executor cancellation.
Only a newly opened native worker connection is closed on its source thread;
preexisting worker borrowers remain live. Startup plus pending/result bookkeeping
composition remains distinct from source-only native retirement evidence. Full
application safe points, startup release/reacquire, responder and Complete product
qualification remain unavailable until the rest of Task10 is implemented/reviewed.

## Concrete MCP local persistence lifetimes (Task10 phase11)

Actual LocalMCPStore, ConfiguredServerTargetStore, UnifiedMCPContextStore,
MCPPermissionStore and MCPExecutionLog constructor/read/RMW operations may bind
only their canonical selected data files to the actual config module/profile.
Admission precedes default selector IO; later source validation uses the selected
config cache without new folder effects. Custom/subclass/unqualified sources keep
ordinary behavior; an installed source cannot retarget or demote to bypass a gate.
The target class lock and other source instance locks cover complete operation,
native retirement and caller timestamp publication. This adds no cross-process
CRUD serialization promise or imported permission/launch authority.

Permission read corruption recovery fixes active/temp/backup before mutation.
History fixes active/.1 and both exact private temporaries/parent before migration,
rotation or append. Only installed history extends existing private-helper discovery
to those fixed binary-read, atomic-write, append-stream and secure-parent routes;
actual PID/Thread/Task/native checks and existing private posture algorithms remain.
No generic callback, caller-selected owner, arbitrary companion, directory scan or
unqualified/None native hold grants post-pause authority. Owned publication updates
its destination expectation from the positively published inode, never from a new
path lookup; a foreign replacement or recreated consumed temporary is preserved.

A failed partial-generation publication or uncertain explicit native close leaves
sticky source failure even after unrelated successful operations. Caller timestamp
restoration does not claim disk rollback. There is no new automatic repair/journal;
actual source/archive validation remains necessary. Corruption reset and JSONL
migration are ordinary runtime behavior, never archive inspection/capture readers.

The five sources do not qualify the surrounding MCP runtime. Permission downgrade
marker then best-effort history append, tool/process work followed by execution
history, lifecycle/discovery followed by runtime snapshots, mutable service context
followed by persistence, and credential value/index operations need their later
exact service/job boundaries. A cancelled awaiter is not native worker/transport
completion. Source-only private test children retire only their own quiescent
startup for independent native evidence; production startup/responder/Complete
qualification remains unavailable pending whole Task10 implementation and review.

### Task10 phase12 — concrete Persona Visual lifetimes (rulings70–73)

Actual Persona Visual repository/core operations, runtime asset file readers,
issued authoring workspaces, imported review candidates and immutable publication
now participate through `Backup_Recovery/persona_visual_participants.py`. This
private bridge reuses storage pending/native leases and existing core operations;
it adds no generic owner/coordinator or transferable callback authority. Source
selection validates the exact loaded config, registered canonical core repository
and profile. Candidate issuance uses actual object identity, complete dataclass
shape and native candidate/marker/member identities; copied fields or tokens do
not create installed authority. Custom roots retain their ordinary contracts.

Each actual producer fixes all selected inputs, UUID outputs and private directory
paths before effects. Native gates and identities are rechecked before activating
the compound scope. Native descriptors are strongly adopted immediately and held
until a positively successful close. Partial construction and uncertain closes
remain visible to local drain and independent native maintenance. The private
helper extension is limited to `secure_private_directory` on selected exact dirs.
Archive external read/validation and fixed candidate extraction remain separately
admitted phases. Runtime decode reads bounded bytes under its actual file scope.

Installed publication source/profile overlap is permitted only for exact current
core graph locators/metadata or actual issued candidate members. Source and output
candidate trees remain disjoint; existing destination candidates refuse before
effects. The concrete Personas UI publication worker retains its exact bound local
Persona source before visual scopes so the existing late eligibility guard can
finish after pause. A real public-guard/UI concurrency schedule proved that adding
a profile-global visual mutex here inverted Persona and visual locks. Only the
authenticated actual publication producer omits that new mutex: unique output
paths, immutable source checks and existing optimistic core activation remain its
correctness boundary. Workspace, import and cleanup serialization is unchanged.

The existing `_drain_to_thread` now waits for actual callback completion despite
independent Task/Future cancellation; this is lifetime bookkeeping, never IO
permission. Concrete Persona job reservations span selectors, source/native work,
result/cache/cleanup handling and new source-thread DB borrower retirement.
Preexisting DB borrowers remain owned by their existing source. Dirty drafts and
failed cleanup references are retained by nondestructive maintenance inspection.
Successful cleanup retires only the matched issued candidate's filesystem blocker
after reference checks and positive native retirement; uncertain native failures
and unrelated blockers cannot be erased by another successful operation.

This establishes source and concrete Persona UI evidence only. Shared Visual
Identity candidate/generation/publication lifetimes, other app/headless owners and
aggregate responder/startup composition still require their own Task10 work.
Required-state clear continues to produce a reviewable, unpublishable dirty draft;
clearing an optional state preserves existing valid publication behavior. No new
visual feature, capture reader, archive capability or Complete claim is introduced.

### Task10 phase13 — Shared Visual Identity source/native lifetimes (rulings74–75)

`Backup_Recovery/visual_identity_participants.py` binds the actual loaded config,
canonical profile, registered CharactersRAGDB, original candidate and fixed source
and publication members. The existing storage pending/native and core operations
carry publication through file rename, optimistic core activation and positive
native retirement. This is a separate concrete source bridge; the Persona Visual
bridge and generic callback lifetime do not grant this source permission. Existing
candidate and Samira seed locks remain; no new global source mutex is introduced.

Public publication, cleanup, runtime readers and the config builtin seed caller
reserve before selectors, native allocation and candidate publication flags.
Filesystem package sources are finite and read-only. Custom package roots and
non-filesystem resources keep ordinary contracts without across-pause authority.
Seeding preserves its two commits: a card remains usable after pack failure.
Public injectable `atomic_replace` and repository guards retain ordinary behavior.
Builtin/shared pack saves retain their fork and binding-version rules.

Cleanup keeps its relpath API. Only the exact module-issued relpath object, original
repository/source association and captured directory/member native identities may
reconcile that publication's blocker. An equal rebuilt string may use ordinary
fresh admission and existing reference/namespace checks; it does not inherit an
installed scope or clear a source failure. Foreign native members remain intact.
Only verified unreferenced cleanup followed by positive native retirement clears
the matching failure. Partial allocation and close-before/close-after uncertainty
retain strong native state and maintenance exclusion, including committed results.

The actual canonical UI restoration method may append only missing canonical
path-free rows under the original candidate lock. Issuance validates the entire old
graph and exact appended metadata; copied, changed and unstageable candidates
cannot refresh issuance. Restoration alone is an unsaved draft. Every retained
placeholder needs staged replacement bytes before publication; clearing remains
the existing explicit author action. A private copy-surviving refusal marker adds
no authority: only the registry's original object/source relationship does.

Concrete Personas reaction jobs retain pending state through queued/running
cancellation, native callback completion, result/cache handling and retirement of
new source-thread DB borrowers. Borrowed connections remain with their owner.
Inspection reports pending/dirty/failure state without navigation discard. External
`_generate_visual_identity_assets_admitted` provider work is still unqualified:
its UI pending lifetime is evidence of outstanding work, not runtime retirement.
App/headless aggregation, responder/startup release and remaining persistence
cohorts require Task10 continuation and whole-task review. No Complete promotion.

Rulings76–77 close actual public callback borrowing found during phase13 review.
The publication's `atomic_replace` call and both repository `publication_guard`
invocations suppress Shared Visual and core discovery only while evaluating the
public callback. Exact loaded producer code/globals authenticate these narrow
callsites; original pending/native/transaction/lease registries stay retained.
Discovery is restored in `finally` before rename verification or original core
activation. Ordinary callbacks can perform fresh admitted work before pause; they
cannot borrow the original core operation for unrelated work after pause. The
actual internal filesystem guard requires no privileged callback registry.

Ruling78 permits only three original same-repository result-read edges to retain
their validated core operation: `activate_pack`/`publish_version` to
`get_active_actor_pack`, and `get_active_actor_pack` to `list_version_assets`.
Both exact loaded producer/callee codes, globals, receiver, source and current
operation participant must match. Public guards still suppress discovery,
including their truth-value evaluation. Original result materialization therefore
finishes without admitting an independent new source operation after pause.
If source failure and borrower retirement both fail, the UI retains the original
exception, exact cleanup token and truthful result on the retirement error; it
does not turn an unrelated error into publication success or clear uncertainty.

### Task10 phase14 — TTS repository lifecycle foundation (ruling79)

The existing TTSProfileRepository owner loop, lifecycle/state locks and serialized
executor now supply a private reversible maintenance boundary. Ordinary first-open,
CRUD and restore requests reserve before their lifecycle/native entry. Construction
remains pure. Sealing admission preserves the generation of admitted work; actual
concurrent worker futures and explicit owner-loop result-publication futures settle
before the source worker closes its exact connection and ProfileStoreLease. A
quiescent executor may remain for later reuse. Only positive native cleanup permits
CLOSED and a generation advance. Definitive public close remains terminal with its
existing quarantine and ordinary cleanup-failure behavior.

Sealing and resumption each register the exact repository in the existing local
raw-operation set before cleanup or new allocation can drop native SQLite leases.
That strong registration is pending/failure evidence, never filesystem authority or
a cross-process lease. Timeout and caller cancellation retain the owned transition;
uncertain close, including an error after native close, retains source state and
blocks local drain. Resume rechecks the original configured path, canonical path,
parent and main-file identities on the worker before and after open. It never
automatically opens an originally closed, unavailable or terminal repository.
Cancelled successful resume settles native open and admission before redelivery;
failed resume retains its source blocker and cannot be cleared by unrelated retry.

This foundation introduces no installed across-pause IO capability, generic callback
authority or source/config binding. Exact migration/backup/reference native scopes,
residual resources, app-owned source registration, dirty editors and runtime jobs
remain immediate Task10 work. Ordinary domain backup/restore behavior is preserved;
finite additional-file selection and native uncertainty are not qualified by this
lifecycle alone. Independent native tests distinguish the TTS profile lock from the
shared admission lock and keep diagnostic-child startup evidence separate from
production startup. No repository, startup, responder or Complete promotion follows.

### Task10 phase14b — outer TTS backup native retention (ruling80)

The actual repository now retains each outer `backup_to` operation before native
allocation, with its source connection, configured/active path, admitted generation,
selected destination, temporary inode, parent/file descriptors, destination SQLite
connection and independent ordinary storage lease. Explicit attempted-close state
is separate from positive retirement. Before/after-close uncertainty preserves the
actual resources and local blocker through public terminal close, cancelled callers,
and unrelated successful backups. Public errors remain sanitized and definitive
close keeps its existing contract. This is retention, not an installed source binding
or a new across-pause capability.

Outer cleanup checks original parent/main identity and never removes a substituted
file or an unproven remaining SQLite sidecar. Normal SQLite backup leaves a journal
until native close; that existing close remains allowed against the unchanged main
and parent. Exact delegated journal allocation/substitution/close authority still
requires the immediate phase14c source/native work. File fsync and parent directory
fsync descriptors belong to the retained outer record. A completed rename and its
receipt remain distinct from directory durability, positive native retirement and
the eventual public result, including an observed rename followed by an exception.

The mandatory next phase owns `validate_profile_candidate`'s source FD, private
snapshot directory/file, copy FD, upgrade/read SQLite and reference readers, plus
`_worker_validate_standalone_snapshot`'s additional immutable reader and the checked
backup helper's native source pins. Existing pause refusal occurs at subordinate
SQLite admission after validation has already opened/copied files; it is not proof
of admission before file effects. Finite original source selection must precede
those effects before any full backup/source qualification. Migration, restore,
reference BLOB/materializer/bundle and actual app/runtime owners remain Task10 work.
No startup release, responder, full repository or Complete qualification follows.


### Task10 phase14c — ordinary candidate validation native job (ruling81)

The standalone synchronous `validate_profile_candidate` API has no repository
receiver and authenticates no installed source. It now registers its actual call
in the existing raw-operation set before ordinary admission, callbacks or selectors.
Each new native allocation obtains ordinary admission. It creates no operation token,
callback permission or directory-descendant authority; a later pause refuses new
allocation even when another installed operation called the helper.

The concrete job retains supplied/resolved source and observed source identity,
source/copy descriptors, returned upgrade/read SQLite connections, private snapshot
and directory identities, parent pins, ordinary leases and body/cleanup association.
Resources become owned immediately on return, before subsequent metadata probes.
Attempted close and positive native retirement are distinct. Independent closes
continue after errors with original first control-flow precedence; uncertain native
state, allocation return, substituted namespace or unproven sidecars retain the job
and actual leases. No uncertain close is retried or foreign namespace removed.
Parent-pinned removal and mode changes apply only to the observed private cohort.
Positive ordinary cleanup removes only that job. Another successful validation
cannot retire older uncertainty. Disposable-copy-only historical migration, schema4,
source bytes, deadline and public sanitized error contracts remain unchanged.

Explicit missing-parent-pin primitives select the pre-existing ordinary path with
identity checks; actual pin, IO or admission failures never downgrade. Every acquired
lease is retained, including real native leases on a host missing only a pin primitive.
Host-simulated portability is not Windows-native, source or capture qualification.

This is the bounded ordinary native repair, not whole candidate/repository ownership:
internal historical migration/BLOB lifetimes, delegated SQLite source pins and journals,
the repository's additional immutable snapshot reader, configured app source binding,
finite preselection, restore/publication/materializer/runtime and all-owner startup
remain immediate Task10 work. Successful schema inspection never proves those resources
retired. No Complete/replacement/responder/startup capability or task AC is promoted.

### Task10 phase14d — configured TTS source and delegated pin ownership (rulings82–88)

Bind only the original repository created by actual app composition to its original
loaded configuration source/functions, selected profile and exact database path.
Check source identity before ordinary admission effects, on the worker with its
admitted generation, and before publishing results. Preserve harmless config/cache
changes, pure standalone constructors, ordinary custom behavior and definitive
cleanup of original resources after a source mismatch. This binding grants no
across-pause IO capability and does not register whole-app/runtime coverage.

The shared checked SQLite source-pin helper retains concrete jobs through preflight,
final source pin and native cleanup. Ordinary exact source/parent leases remain
separate from existing validated capture leases. The latter associate the job with
the same capture scope and preserve native maintenance quarantine on failed
retirement. No owner policy, source/staging scope, registry or callback permission
is widened. Existing private preflight helpers receive the job explicitly; the
trusted-directory verifier's optional private open/close observers preserve their
default native calls and existing raw/visual attribution. The concrete job marks
allocation only after admission, associates actual returned descriptors immediately,
and retains unreturned allocation failures separately from later successful opens. Independent close attempts retain
original body/cleanup association and never retry an uncertain descriptor number.

A disposable, unpublished ordinary backup validation can finish at the spec's
completed boundary with its truthful refusal after pause blocks allocation. Actual
worker/result completion and positive temporary/native retirement are required;
failed ownership remains a blocker. Successful user operation is not the only
completed boundary. This decision grants no new validation IO after pause and
waives none of the native journal/BLOB/outer-reader/migration/restore/runtime graph.
Task10 remains incomplete and startup/responder/Complete/replacement unavailable.


Ruling87 distinguishes confirmed native rejection from unknown provider outcome
with one explicit per-call outcome record at the original native-open boundary.
Only rejection by the exact original native primitive proves no descriptor; its
returned FD is recorded before raw registration or outer wrapper work. Known
wrapper failures can therefore retire their actual FDs. Substituted/delegated
unreturned failures retain uncertainty, regardless of exception type or later
successful opens. Existing raw/visual attribution and trust/scope rules remain.
This corrects the demonstrated trusted-alias regression: default TTS profile paths
can remain lexical, while custom TTS database selections resolve canonically.
Ordinary alias copies and actual adapter capture must both positively retire;
no normal supported alias is waived as an unqualified-platform limitation.


Ruling88 carries the same per-call outcome into the existing SQLite artifact-open
primitive, preserving its non-raw/visual attribution and unrelated callers. Concrete
preflight and final pins attach known returned FDs before wrapper errors, recognize
only original-native rejection as absence, and retain unknown substituted outcomes.
This repairs a confirmed disappearing-optional-sidecar retirement regression against
BASE. Real ordinary/capture copies cover all three suffixes; actual replacement retry
retires, while a successful retry after an unknown allocation cannot erase that
older uncertainty. Privacy, no-follow, generation/absence retries and source identity
checks remain unchanged. No new capture/source/descendant permission is introduced.

### Task10 phase14e — outer snapshot and native journal ownership (rulings89–91)

Use the existing concrete `_BackupNativeState` for both ordinary public backup and
pre-restore recovery backup. Associate nested snapshot validation uncertainty with
that exact outer pathname before cleanup. The existing SQLite admission registry
continues to retain native readers; this adds no second connection registry or
capture authority. Each ordinary allocation is admitted before marking its pending
outcome. Returned native descriptors and exact path/parent identities are recorded
immediately; genuinely unreturned allocations retain the actual outer lease and
unknown state. No error class, subsequent success or cancelled caller proves native
retirement.

The recovery caller explicitly owns its `tts.profile_recovery` destination and uses
`backup_open_connections_to_private` with the existing literal policy. Hidden helper
close warnings can no longer let this caller report a successful restore. Keep the
original source-close failure handoff to the repository's exclusive profile lease;
its exact wrapper and lock retry behavior remains mandatory phase14f work.

Observe the exact temporary `-journal` during the existing ordinary native progress
and completion boundaries, and recheck it before native close. First observed native
identity is preserved; observed replacement refuses close before SQLite can remove a
foreign pathname. Native SQLite retires its own journal normally. No manual residual
sidecar deletion or permission for directory descendants is introduced. Callback
errors retain the original body/control-flow signal together with namespace errors;
existing migration authority quarantine retains its public unavailable contract.
This is cooperative coordination, not atomic hostile-process substitution detection
before the first observation or inside a native primitive.

Publication/receipt and durability remain distinct from positive resource retirement.
Partial/uncertain outcomes retain exact outer ownership after unrelated success,
caller cancellation and terminal repository close. Ordinary completed refusals must
still retire positively. Exact current wrapper/dual native handles/ProfileStoreLease,
BLOBs, migration/restore/rebind/quarantine, materialization/bundles, dirty UI/service,
voice/model/audio/process and app/headless/startup aggregation remain Task10; no
Complete/replacement/responder capability or acceptance criterion is promoted.

Ruling92 associates the newly created existing `_CandidateValidationJob` with the
actual outer record through private `_outer_job` before first admission/callback or
native effect. Cleanup consults that exact job's uncertainty; standalone validation
and positively completed refusals keep their ordinary behavior. This is ownership
association only, with no registry scan or new source/IO/capture permission.

Ruling93 adds one private per-call `_SQLiteAdmissionOutcome`, consumed and removed
at the original SQLite admission boundary before preflight/constructor/factory
entry. Only refusal at that concrete boundary can clear the current outer pending
connection attempt. An identical sanitized error raised by a substituted connector
after opening a native handle remains unknown. Memory, foreign-source, descriptor,
capture and custom-factory branches retain their original behavior; factories never
receive this record. Older uncertainty cannot be cleared by a later successful call.

Ruling94 stops the existing standalone candidate lease-retirement loop immediately
when its existing error recorder marks uncertainty. A real close-then-wrapper-error
previously left the candidate job locally retained while releasing every native
hold. Keep the remaining actual leases, original error and ordinary positive release;
no retry, registry or interface is added. Outer association is additional protection,
not a substitute for standalone native exclusion.

### Task10 phase14f — exact TTS connection and profile-lock retirement (rulings95–99)

The existing exact-current wrapper is created before pin allocation and retained
through ordinary source leases in existing strong lease membership. Separate native
attempts retain their returned descriptors, dual SQLite handles, source identities,
unknown outcomes and cleanup errors. The existing exact cleanup exception carries
partial ownership to the repository. Native close uncertainty cannot be healed by
retrying a descriptor number, a closed property, later successful allocation or a
terminal repository result. Remaining ordinary leases survive uncertain release.
Standalone wrapper close revalidates the original WAL/SHM namespace before its first
live SQLite close, preserving the repository's foreign-namespace quarantine.

Original verifier traversal uses its existing private-path open/close attribution;
main and sidecar opens retain their original direct OS attribution. The retained
parent's final close remains the original direct OS close. Per-attempt native
rejection evidence and the existing SQLite admission outcome distinguish positively
refused allocations from substituted providers that allocate then raise. A later
optional-sidecar result cannot clear an earlier unknown attempt.

Each ProfileStoreLease acquisition owns a separate native outcome and ordinary
source holds before lock-file open. Its primary/residual compatibility slots, public
retry behavior and error precedence remain. Successful original native close can
retire despite an unlock error; unknown open/close remains independently retained
through closed-field normalization or later successful acquisition. This introduces
no second mutex, native registry, source binding or callback/capture permission.

Real reference BLOBs remain owned by their original SQLite parent. Source-worker
native observations prove parent close retires returned and unreturned BLOBs after
read/write close faults; failed parent close retains actual native exclusion. The
existing sanitized BLOB error codes and first control-flow identity are preserved;
secondary ordinary error detail remains intentionally unmapped (ruling98). No BLOB
registry or diagnostic exception payload is added. These host-specific native tests
are not Windows qualification, whole TTS runtime drain or startup release. Migration,
restore publication/rebind, reference materialization, bundles, voices, dirty UI,
service/audio/model/process and app/headless aggregation remain Task10 obligations.


### Task10 phase14g: live TTS migration and restore outcomes (rulings100–108)

Actual migration/restore/publication/recovery operations retain a concrete ordinary
native record on existing source leases and repository membership. Exact source,
parent and validated recovery-row associations constrain helper use; original opaque
namespace and destination policies remain decisive. Scalar standalone helpers own
actual supplied sources, while returned-FD namespace helpers preserve their existing
destination owner. No ambient callback/capture permission, duplicate global native
registry or second mutex is added.

A private descriptor-connector outcome records duplication and actual native SQLite
return before subsequent wrapper/finally failures, because verified-descriptor SQLite
bypasses ordinary path admission. Source readers and initialized connections retain
constructor and first-close outcomes before later configuration/callback failures.
Original public source cleanup retries remain compatible but cannot heal native
uncertainty; positive first cleanup retires, including historical BLOB children under
their original native parent. New final cleanup propagates original control flow by
identity when no body error exists, preserving original publication body precedence.

Shared-open scalar parent checks use the same ordinary record. Existing exact-current
revalidation observes its bounded parent reopen on the held exact owner without new
admission during close. Traversal/final primitive attribution and positive descriptor
reuse stay intact. Proven query-only DELETE-mode partial construction may close only
after exact parent/main/security/publication checks and explicit WAL/SHM absence;
full-current WAL authority and foreign-namespace quarantine remain unchanged.

The existing schema-version-aware migration reference validator qualifies genuine v3
restore before migration; no schema or archive validation policy changes. Public open
maps errors outside its catch while retaining the original reservation-finally timing.
Actual pause evidence preserves preflight or durable publishing bytes and recovers
convergently after ordinary resume. A journal is not capture readiness, and this phase
grants no PONR continuation authority. TTS materializer/bundle/voice/runtime/dirty UI,
app/headless aggregation and startup release remain mandatory Task10 obligations.

### Task10 phase14h: private clone materializer ownership (rulings109–112)

Retain actual materializer native outcomes on ordinary source leases and its
original opaque reference records. Reserve before queued workers; retain known
returned descriptors before outer constructor errors and distinguish original
native rejection from unreturned allocation. Every new allocation still obtains
ordinary admission. Only exact selected root/owner/lock/asset members and known
parents belong to direct helpers; secure-directory traversal observers preserve
original trust/attribution and do not confer child or capture authority.

First failed native close remains unresolved, including after actual closure.
Close distinct known resources independently; never retry ambiguous numeric FDs,
release remaining holds after uncertain lease retirement, or substitute registry
size/closed fields for positive native retirement. Preserve original no-body
control flow and existing body precedence. Failed-create cleanup checks earliest
returned-native identities before removal, preserving substituted/renamed bytes
and unknown siblings. Namespace residue remains a full safe-point blocker.

Bind only the original default factory materializer to its original configured
source/functions/profile/root. Recheck before async admission, worker/native IO,
handle publication and resume; source drift cannot retarget existing cleanup.
Explicit custom constructors retain ordinary behavior without source qualification.
The existing factory selector may create/harden base/profile config directories
before binding; no clone sweep, model/adapter work or capture is added by binding.

Reversible maintenance closes new admission and waits for actual work but never
deletes a live consumer's WAV. The existing generation-to-response cleanup transfer
retains the reference until its native cleanup succeeds, including failures after
response closure is marked. Whole service/backend/process/consumer shutdown and
runtime-root inventory remain separate mandatory Task10 composition work.

The first globally injected traversal fault was subsequently found to hit storage
admission instead of the materializer's own selected-directory traversal. Only the
corrected runtime-inode fault and observer-disabled counterfactual establish that
shared edge; neither is an exported-BASE claim. Native POSIX evidence does not
promote complete backup/replacement, Windows support, or startup/runtime readiness.

### Task10 phase14i: retained voice bundle operations (rulings113–114)

The original bundle service retains each concrete native worker operation on
ordinary storage leases. Fixed runtime and selected caller paths/parents receive
ordinary admission before their IO; runtime admission alone cannot authorize a
selected external file. Every new native allocation rechecks exact source and
path admission. Only the original finite operation/member/temp selection edges
can extend an operation's selected names. Traversal observers remain outcome
observers, not descendant permission.

Native descriptor and buffered-stream creators record returned resources before
outer wrappers can fail. Buffered member streams use `closefd=False`; the concrete
operation is the sole descriptor closer. Unknown stream ownership retains its
associated descriptor to avoid later writes through a reused number. The first
ambiguous close/allocation or namespace residue remains a blocker, independent
of public sanitization and task completion. No descriptor close is replayed;
distinct known resources retire independently. New allocations stop once an
operation has uncertain ownership. Uncertain lease release retains remaining
holds. Failed-create disposal requires the earliest observed original native
inode and matching current private empty directory, preserving substitutions.

Reversible service maintenance closes admission and waits for actual calls,
queued/native workers, retained repository calls, and result publication. Pending
opaque inspection sessions survive unchanged, including their existing expiry
and consumption rules. Terminal sealing/session disposal remains separate.
A successful export acknowledgement survives late native cleanup failure only
when the original publisher recorded its exact destination/inode after both
existing durability/convergence checks. A successful fingerprint boolean is not
publication evidence. Native uncertainty independently blocks maintenance;
existing body/control outcomes retain their precedence. Published bytes and the
intentional pre-publication private temporary residue are never pathname-deleted
for readiness.

Only the actual original lazy app factory can bind a configured bundle source.
Checks retain original app/service/config selectors, selected profile/root,
configured repository, dependency/profile/coordinator receivers and original
bound mutation fence. Custom constructor/selector/profile/fence routes remain
ordinary and unqualified. Source checks do not invoke custom accessors. Binding
is pure; the preceding ordinary app selectors/repository open retain their IO.
Native cleanup keeps its original descriptors and namespace after source drift.

This cohort qualifies concrete POSIX native behavior on the execution host.
Runtime-root inventory classification (including empty roots and foreign residue),
whole service/backend/process and dirty UI/consumer safe points, app/startup
release, shared voice paths and whole-Task10 review remain required. No Complete
backup, replacement, startup release, or other-platform qualification is added.


### Task10 phase14j ruling118 — original concurrent first initialization

The same-process first authority initializer reserves its resolved root on its actual pending acquisition record under the existing coordinator condition. Followers wait outside the lock and recheck original cancellation and provenance before proceeding. The initializer runs original startup/native qualification and authority construction outside the coordinator lock and releases its reservation in finally. Same-thread recursion refuses rather than waiting for itself. Incomplete, failed, crashed or foreign-process evidence retains the original fail-closed behavior; this does not repair or retry durable uncertainty, add a registry/mutex, or qualify whole-app startup. Required evidence includes deterministic marker-before-register follower ordering and the supported uninitialized128-sample service case. Exact shared-source BASE reproduces the prior follower refusal at startup_permission; unchanged external dependencies do not make that a whole-repository BASE run.


### Task10 phase14j rulings115–117 — profile evidence and Library work

Original profile-service admission/drain/resume counts actual PID/Thread/Task callers through final evidence/result work; synchronous evidence legitimately runs on a worker thread. Original same-receiver nested calls can settle while admission closes, preserving artifact coordinator before consumer fence lock order. Fresh sample FD/codec work always reacquires ordinary selected file and required-parent admission before metadata/allocation. Optional private native outcomes preserve known returned resources before late errors, unknown allocations and first uncertain native/lease closes independently of content validity. Original body controls win over cleanup controls while both outcomes stay accounted. Default standalone/shared Settings validators keep their existing API/behavior.

The original lazy app profile-service factory binds its exact original repository/config/profile/dependency/coordinator and static no-shadow identities through profile_source. It checks before admission, new native work, final evidence/result publication and resume. Custom/prebinding/proxy sources are not replayed or promoted; a source relationship does not qualify artifact runtime work.

Profile Library readiness preserves exact mounted typed input, submitted conflict or maintenance-refused drafts, review handles and their original service sessions. Maintenance neither dismisses nor saves/discards them. It closes new action/page admission, defers late page/search publication, and retains actual result/cleanup lifetimes. Reopened-conflict Cancel preserves submitted text; original refresh/retry Save resolves it. Already submitted text is retained before stale action-target refusal. Sanitized export retains actual queued/running writer and native stream through repeated cancellation, original overwrite bytes/result and first uncertain cleanup; caller destinations are not baseline resources. Service/file completion alone cannot retire the remaining Speech caller's queued/post-thread UI work. Whole-app composition and all other Task10 debts remain open.


### Task10 phase14j ruling119 — selected export before truncation

A reached output-parent replacement after original path validation redirected the sanitized writer. The exact existing export operation now snapshots selected identities after ordinary admission, owns/verifies the parent descriptor, opens an existing output without truncation and verifies the selected regular-file FD before truncating it. Missing output uses exclusive creation beneath the verified parent; this preserves ordinary new/overwrite behavior without replacing or deleting existing output. Stream and parent/target FDs have explicit separate owning closers and retained first/unknown outcomes. Later pathname replacement cannot redirect writes through the already selected FD. This is one caller-selected writer, not a generic filesystem capability or baseline resource.


### Task10 phase14j ruling120 — ordinary platform compatibility

Pinned export is selected upfront from original primitive identity, required flags and original dir_fd support. When those primitives are unavailable before allocation, retain the original ordinary pathname stream route and its complete operation/stream outcomes; Library maintenance remains explicitly unqualified. A failed or unknown pinned allocation never selects fallback. OS Independent ordinary use is preserved without a Windows/native-platform qualification claim, mutation of global capability metadata or use of fallback as a capture capability.


### Task10: finite loose voice operations

The unfinished recursive Python frame/code/closure/MRO authority machinery is
replaced by finite scopes at installed voice constructors and file operations.
Original revision4 section6 limits coordination to declared owners and
protocol-aware entrypoints; arbitrary out-of-protocol same-user modification is
outside the guarantee. Actual source binding, containment, native descriptor
settlement and related audio/JSON operation consistency remain required.

Existing service, registry, profile, repository and runtime ordering remains in
place. Accepted nested operations may finish while fresh intake is paused; failed
partial publication or uncertain native close keeps capture blocked. Whole live
app/capture and standalone-root inventory coverage remain separate qualification
requirements. The prior uncommitted proposal and tests were preserved before
replacement; this decision adds no execution/restore feature or global attestation
framework.


### PERF-07/PERF-08 amendment — reusable admission evidence (TASK-33266, TASK-33267)

Status: **Approved by the owner on 2026-09-29.** The owner approved the direction
as decisions D1 and D2 of the
[2026-09-27 structural perf audit](../../qa/perf-structural-audit-2026-09-27/report.md)
(§4 PERF-07, PERF-08), then settled the mechanism with a criterion: choose
whichever option aligns with the planned Python 3.13 then 3.14 migration.

Held descriptors and per-call `lstat` stamps are equally version-neutral, but
only stamps meet the invariant (see below). So the amendment uses stamps, plus
the version-alignment requirements in "Python 3.13 and 3.14" at the end of this
section.

**Deviation from the approved wording.** D1/D2 said ancestors would be re-checked
"via held handles". Research showed that cannot work. `fstat` on a held
descriptor describes the inode that was opened, so it cannot see a directory
renamed away and replaced at the same path. Holding one directory descriptor per
registered root (about 100) also risks macOS's 256-descriptor soft limit.
Ancestors are therefore re-checked with a per-call `lstat` of every path
component. The guarantee is the one D1/D2 asked for; only the mechanism differs.
Held descriptors stay where I/O already goes through them (raw pins, `openat`).

**Context.** Every outermost guarded call re-derives admission from disk:
- about 245 `open()` calls per DB transaction;
- `registry.json` read 7 times;
- a directory walk from `/` for every chain;
- all of it serialized per bootstrap root inside `_Acquisition.initializing`,
  which caps unrelated DBs at about 200 transactions/s together.

This re-derivation is not what detects a cooperating backup or restore. The
live hold's shared lease lock already blocks them, and the 10 Hz monitor and
`pause_requested` observe pauses. What the re-read does detect is changes to:
- control records;
- the registry;
- the config selector;
- directory posture.

The amendment keeps detecting exactly those.

**Decision.** Ordinary storage admission (`acquire_storage`) may reuse the
*allowed* result of an unmodified full derivation only while all of these hold:

1. **Same hold and namespaces.** The same live `_Hold` for `(pid, bootstrap
   root)` is still ready, not stopped, and has no error. Its namespaces are
   identical.
2. **Same epoch.** The in-process admission epoch is unchanged. In-process
   writers of records, the registry and the selector advance it. This is
   hardening; the stamps below are the correctness mechanism.
3. **Same stamps, re-observed on every call.**
   - *Posture:* `lstat` values `(dev, ino, type, permission bits, owner)` for
     every component from `/`, for every chain the derivation walked. That
     includes every verified ancestor and the admitted path's own chain and leaf
     state.
   - *Content:* `(dev, ino, size, mtime_ns, ctime_ns)` for:
     - the bootstrap and admission directories;
     - `registry.json`, `registry.lock` and the enrollment marker;
     - every control record file (a directory stamp misses an in-place edit);
     - the config selector file.
4. **Per-call gates unchanged.** Every in-memory gate is still evaluated on
   every call, with the same reason codes:
   - local pause and participant closure;
   - fork;
   - pid/thread/task provenance;
   - execution selection and maintenance thread;
   - hold readiness.

The lease is counted before a final re-observation of every stamp and the
epoch, which keeps today's "counted before the final permission read" ordering:
a change landing between the first look and the count is still seen. The reuse
path never enters `initializing`. (The first implementation re-read only
content stamps after counting, and stamped the selector only when bound; both
were narrowed to this text after review on #2919.)

**Recording rule.** Evidence is recorded only when all of these hold:
- the unmodified derivation *allowed* the call;
- the complete stamp set was identical just before and just after that
  derivation;
- every content change time is at least a settle margin (1 s) older than the
  first stamp. This is git's "racily clean" rule for coarse filesystem clocks.

Evidence is never recorded with:
- pending records;
- absence-proved roots;
- a symlink anywhere in a chain;
- unqualified native storage;
- startup reacquisition;
- a maintenance or capture session.

**Refusal.** Any mismatch runs the unmodified derivation. It remains the only
source of refusals and reason codes. Evidence is process-local and never
persisted. It is discarded with its hold, on fork, and when the epoch advances.

**Preserved invariant.** Reuse never admits what the unmodified derivation
would refuse at that moment. Each of these changes a stamp or trips a per-call
gate, and so goes back through the full derivation before dependent I/O:
- a moved, replaced or re-permissioned admitted directory or verified ancestor;
- a storage pause or participant closure;
- a registry, binding or selection change.

The derivation then decides exactly as it does today. It refuses some of these,
such as a group-writable ancestor, a pending record or a registry intent. It
admits others, such as an admitted data directory renamed away and recreated
at the same path, which the private-path checks at open time then govern. The
implementation's oracle test pins that reuse and derivation agree on every
mutation in its catalog.

**D2 corollary (PERF-07).** `get_user_data_dir` may memoize its verified
directory *inside* the guarded body, after the handshake. The memo is keyed on:
- the config cache object and its generation;
- source and selector;
- HOME / default data base;
- `data_dir` and `users_name`;
- bound/unbound.

It is re-checked every call against:
- posture stamps from `/` to the directory, which must still be 0700 and owned
  by the effective user;
- the default-root ambiguity inputs;
- the lock file's stamp.

Any mismatch runs today's create/harden/refuse path. The "no process-scope
caching" note in `resolve_sensitive_context` is amended to this key.

**Unchanged.** All of these stay as they are:
- the native gate, lease and incompatible-lock protocol;
- maintenance;
- pending and activation semantics;
- the module attributes that tests and thread diagnostics wrap by name.

Out-of-protocol modification stays outside the guarantee, and detection is never
weaker than today.

**Required evidence before merge.**
- A differential oracle test: reuse verdict equals full-derivation verdict after
  each mutation in a catalog. The catalog includes:
  - ancestor chmod;
  - ancestor rename+recreate;
  - symlink swap;
  - leaf replace;
  - a pending record or registry replace from a subprocess;
  - the registry intent file;
  - an in-place record edit;
  - marker replace;
  - selector edit;
  - bootstrap-root deletion;
  - fallback-root creation.
- A dependency-completeness trace test.
- A full `Tests/Backup_Recovery` run with the settle margin at 0.
- Benchmarks for PERF-08 AC#3/#4.

**Python 3.13 and 3.14.** The owner plans to move from 3.12 to 3.13 and then
3.14. The reuse path must not need rework at either step:

- **Version-neutral fields.** Stamps use only `stat` fields that have the same
  POSIX meaning on 3.12, 3.13 and 3.14: `st_dev`, `st_ino`, `st_mode`, `st_uid`,
  `st_size`, `st_mtime_ns` and `st_ctime_ns`. They are compared as Python ints,
  never packed to a fixed width; on Windows `st_ino` can be up to 128 bits
  since 3.12.
- **No raw `st_ctime` on Windows.** Since 3.12 it holds the creation time,
  deprecated, and a future release changes it to change time. A Windows stamp
  must take change time from the repository's `platform_files` /
  `windows_files` shim. Windows still keeps the full derivation until that is
  verified (below).
- **Free-threading safe.** Free-threaded builds are officially supported from
  3.14 (PEP 779). All shared evidence state (`hold.evidence`, the epoch) is
  read and written only under the existing coordinator `_lock`. Nothing may
  rely on the GIL making dict or int operations atomic.
- **Tested on the supported versions.** The oracle and trace-completeness tests
  must run on each Python version the project supports when this lands.

**Open until measured.**
- Windows keeps the full derivation until its `stat` cost and NTFS directory
  change-time behaviour are verified.
- The visual native scope's `_check_path` interaction is unverified. Reuse is
  disabled inside that scope unless it is shown harmless.

### TASK-34200 amendment — file identity is path + inode, not the device

Status: **Approved by the owner on 2026-10-03** (chosen over a volume-UUID
identity and over filing only).

**Context.** The owner's Mac rebooted on 2026-09-28 and its data volume came
back with a different `st_dev` (16777234 → 16777230). The admission registry
pinned `inode:<st_dev>:<st_ino>`. So the bootstrap marker, the same file with
the same inode at the same path, no longer matched its own registration, and
every start of the app, on any profile, failed closed with "Recovery required:
recovery_scope_uncertain". No recovery operation was pending. The recovery CLI
has no re-enroll for this; the owner's machine was unblocked by moving the
registry aside.

**Decision.** Identity tokens are the path tokens plus `inode:<st_ino>`
(`bootstrap.inode_token`). Every comparison against stored tokens goes through
`bootstrap.identity_view`, which reads a token recorded before this amendment
(`inode:<dev>:<ino>`) as `inode:<ino>`. Existing registries therefore keep
matching across a renumbering; new registrations record the device-free form.

**Consequences.** Replacing or copying a file at a registered path still
produces a new inode, so that fence is unchanged (pinned by
`Tests/Backup_Recovery/test_admission_device_renumber.py`). One case is weaker:
a different volume mounted at the same path whose file has the same inode number
would now match. That needs a volume swap at the same mount point and an inode
collision, and it was accepted. Matches are now a superset of before, and every
comparison a false inode match could flip fails closed: admission groups more
namespaces (more locking), and the activation, publication, replacement,
storage-admission and later-rollback scope checks refuse. Nothing that was
refused before is admitted. A malformed stored token is never normalized, so it
cannot match. The PERF-08 per-call posture stamps keep
`st_dev`: they never outlive a process, so a renumbering cannot reach them.

### TASK-34100.17 amendment — first-run setup confirmation and keychain-held provider keys (2026-10-03)

Source: the [first-run setup shape spec](../../Docs/superpowers/specs/2026-10-03-first-run-setup-shape-design.md), approved by the owner on
2026-10-03, and [TASK-34100.17](../tasks/task-34100.17%20-%20Owner-approved-design-spec-for-the-setup-flow-Quick-track-tldw-server-re-run-dashboard-Say-hello.md).

**Confirmed.** Restore stays reachable from first-run setup (decision 1: "exposed
through canonical F9 Settings, first-run setup, and a dependency-light
pre-bootstrap recovery launcher"). It stays on Welcome, and plain-text setup names
the recovery command.

**Amendment.** Provider keys a user chose to keep in the OS keychain (an opt-in; the
default store stays config.toml) are Chatbook-owned keyring values under decisions 5
and 8: excluded from portable export by default, and captured in local rollback
archives through a typed owner adapter. Under decision 9, isolated restore remaps the
persisted `credential_scope_id`, so a restored profile never reads or overwrites
another profile's keys.

### TASK-33560 amendment — ordinary raw and witness evidence reuse (2026-10-06)

Status: **Accepted implementation direction; targeted qualification recorded, independent implementation review pending.**
Independent preflight, meaningful warm-path RED, and a retained-owner actual-app
idle baseline preceded production changes. This
extends the owner-approved PERF-07/PERF-08 stamp model above. The already approved
one-second maintenance probe interval remains unchanged; no new polling or
pause-latency tradeoff is proposed.

**Problem.** `acquire_storage` already reuses ordinary admission evidence, but
MCP selection also calls the shared `generation_witnesses._witnesses` reader.
That reader independently re-derives source scope, startup permission, paired
control records and registry state. Raw operations separately derive pause
groups, parent pin chains and config companion scope. Reusing only a store
getter or storage acquisition leaves those original paths active.

**Decision.** The same live `_Hold` may keep bounded, process-local positive
evidence for these exact ordinary derivations, using the existing `_Evidence`
posture/content fields, settle margin and epoch. Each entry is tied to its
actual hold, namespace tuple, bootstrap root, selector and selected paths.
Results containing mutable records or witnesses are defensively copied; a
consumer cannot change cached authority by editing a returned list or dict.

The permitted derivations are:

- The complete shared `_witnesses(path, lease)` result, including
  `_source_scope_admitted` and `_paired_witnesses`. The actual lease's execution
  context and all existing provenance/selection gates remain per-call checks.
  Dependencies include every fixed control record and registry input, all
  registry roots needed for source relevance/containment, the complete chains
  of every consulted historical `path:` token (their resolution is an input
  even when registry bytes are unchanged), the selector, and every relevant
  activation store/generation chain and `required.json` read. A stage whose
  historical/alias dependencies cannot qualify stays on its original fresh path.
  This caches passive evidence, never activation approval or tool permission.
- The registry/group derivation in `Admission.pause_requested`. Its dependencies
  include all registered roots, including foreign namespaces, and every input
  to absence/overlap determination. The actual registry lock and every gate's
  fresh open, identity observations and nonblocking flock remain per-call.
  A cached group never substitutes for a current native contention result.
- A successful raw parent-chain proof. Each operation still owns a physical
  parent descriptor and its original cleanup/uncertainty lifetime. Reuse needs
  the complete chain to match before and after descriptor acquisition, and the
  descriptor's identity/posture to match the fully derived parent. A held
  descriptor or leaf-only check does not prove current pathname ancestry.
- Config companion metadata derived under the existing registry shared lock.
  Every current member's inode/type/link checks, foreign overlap checks, source
  selection, parent posture and original lock lifetime remain in place.

Fresh history temporary children may reuse only confirmed directory containment
within the same ordinary group, while their own complete chain/leaf state is
observed on the current call and after its independent lease is counted. The
directory itself must receive two full positive containment derivations under
the complete selector/group evidence; an admitted file never proves its parent.
`read_recent` can migrate legacy content, so its
temporary creation and publication authority is never omitted on a warm read.

**Recording and fallback.** Two full positive derivations must bracket identical
complete dependencies/stamps at an unchanged epoch, with the existing one-second
content settle margin. Evidence is read/published under the coordinator lock;
filesystem work stays outside it. No reuse is recorded for pending/intent state,
absence-proved roots, symlinked chains, unqualified storage, startup reacquisition,
maintenance/capture, or Windows. Any stamp, epoch, hold or dependency mismatch
runs the original derivation with its original result and refusal reason.
Allocation or uncertain-close failures retain their original resource custody;
they do not select a weaker fallback. All counted-before-final-check ordering
and independent participant/lease lifetimes remain unchanged.

**Required evidence.** Actual warm readers for all five MCP sources must reach
their bodies while skipping the expensive derivations. Differential mutation
oracles must compare exact results/reasons against reuse disabled, including the
  existing catalog, foreign roots/historical path retargeting, activation
  requirements, leaf replacement, fallback creation and fresh history members.
  A dependency-completeness trace
must cover every skipped input, with an omitted-dependency negative control.
Native gate contention must still request pause after a warm false observation;
physical FD identity, cancellation and retirement must remain observable.
A task-owned paired real-app boot/settled-idle probe must bill unchanged work at
the existing cadence and demonstrate at least a further 50% idle-open reduction.
Original completed PR performance budgets are retained, not replayed as this
task's measurement. New verification is targeted to the changed paths.


**TASK-33560 qualification checkpoint (2026-10-06).** The retained-owner actual
app pair uses the same frozen script and unchanged real timers: 884 to 286
native opens in ten seconds (67.65% lower rate), ten native maintenance probes
and forty credential polls in both runs. The intermediate 574-open result failed
the further-50% requirement and is preserved. The earlier unbound probes are
setup evidence only. A near-worst-phase real native pause was noticed in 0.976 s,
began actual local pause in 1.070 s and entered exclusive maintenance in 1.109 s
at the already owner-approved one-second interval. Fifty-two new safety controls
passed; two additional exact permission parse-pause controls preserve the
original `storage_locally_paused` refusal and unchanged bytes with reuse enabled
and disabled. A related selection remains non-green (six passed, one default-
False setup failure), and nested permission RMW completion across pause is not
claimed. The [portable report and manifest](../../Docs/superpowers/qa/2026-10-06-task33560-raw-evidence/report.md)
bind raw setup/failure history, exact source snapshots, measurement scope and
unchanged baseline static diagnostics. These targeted checks do not renew the
original completed PR budgets or certify broader platform/capture coverage.


### TASK-34408 clarification — bootstrap directory creation durability (2026-10-04)

Registration must durably establish newly created bootstrap ancestry before it
publishes a pending fence. Use the existing intent-before-mutation pattern: an
exclusive private `.tldw-bootstrap-directory-create.pending` in the pinned parent
is flushed before mkdir; mkdir, its parent barrier and intent retirement use
that same native parent handle. Any surviving intent in the ancestry refuses a
retry, including one using different case or a Windows short-name alias. No
permission change can reinterpret an interrupted creation as completed.

A fixed private `ancestry-settled.json` records the bootstrap directory identity
and canonical path only after creation settles, before any new pending record.
It cannot be established over an unresolved legacy pending record. Readers
validate it as local control evidence. New publication reflushes the pending
record and its containing directory; unchanged system ancestors need no write
access. Legacy pending evidence without this receipt retains the previous
required ancestor barriers, and failures remain refusals. No recovery evidence
is automatically repaired or removed after an interrupted creation.

Receipt identity follows the TASK-34200 amendment: Windows persists its NTFS
volume serial and inode; POSIX persists the inode and canonical path with a null
device field, because POSIX device numbers can change across a reboot. Readers
compare the receipt with the actual pinned directory, not an earlier path probe.

This corrects the Windows system-directory flush failure while preserving the
existing native identity, privacy and required persistence contracts. See
`Docs/superpowers/plans/2026-10-04-windows-recovery-verification-fixes.md`.

## Dreams, Guardian and capture provenance ownership (2026-10-12)

The PR2921 integration declares optional Dreams v6 and Guardian v2 stores with
exact selected default/custom paths, participant drains and installed schema
catalogs. Absent unused optional stores remain absent. Supported populated Dreams
v1-v5 predecessors and Guardian v1 (migrated to v2 per ADR204) require exact admission, rollback/reopen and
unknown/future/hybrid refusal before their catalogs are accepted. Collections v5
and Subscriptions v3 preserve conservative URL lineage across capture, relocation
and restore. Recovery cannot turn a legacy unmarked URL into USER_ENTERED.

Restored Dreams/Guardian automatic behavior is inert until current activation
review accepts the selected owners; restored settings/rules alone grant no runtime
authority. Ordinary startup need not open Subscriptions to reconstruct source
provenance: the source store and actual fetch boundary own that decision. New
finite thread work drains under existing maintenance owners and retires only its
own acquired native handles. No namespace exemption, borrowed-handle closure or
weakened schema validator is authorized by adding these owners.


## Current-owner Send integration (2026-10-13)

Finite callback admission covers the complete synchronous body, before exact newly acquired worker handles retire. Current Goal, Dreams, Guardian, RAG, borrowed/open-transaction, memory and custom owner boundaries remain; no foreign or mass closure is permitted. Issued native futures must settle through repeated cancellation before their creator closes, with actual ordinary commit outcomes retained for recovery. Preserve current schema/recovery catalogs and Rules helper physical capacity/deadline ownership. This introduces no schema or new lease authority.

Raw pin derivation consults the existing ordinary-hold eligibility only for optional evidence. Ineligible evidence is omitted; actual operation holds, fresh native checks and descriptor retirement remain unchanged. Windows batching/security projection is a separate prerequisite requiring actual native qualification; POSIX controls never establish Windows acceptance.

### 2026-10-13 current Console canonical MCP observation

An accepted installed MCP raw operation retains its exact canonical-path storage lease alongside destination leases through its full callback. Nested recovery selection may consume only that operation-owned canonical lease after original source, participant, process, thread, task and live native custody checks; selected recovery-destination authority cannot substitute for canonical authority. Source and parent proofs execute outside the coordinator lock with custody rechecked afterward. New/later/foreign reads still acquire admission, and all actual path, generation, source and pin checks remain. This reconciles the accepted catalog maintenance lifetime under ADR230; it grants no cached permission or Windows qualification.


### Current checked Console presentation — 2026-10-13

A finite original configuration operation may issue detached display-only data tagged with its checked generation/path and exact screen/session/settings/database owner. The operation and native handles retire before synchronous UI consumption; the result carries no lease, permission, dispatch or persistence authority. Original live action/configuration callers remain fresh. The current config_sync owner may use this issued warm presentation only while its source/participant/owner, original one-second age and publication checks hold before and after rendering. Cold owners defer to the existing coalesced refresh; exact expired pending refreshes request retry. Custom/unqualified direct reads retain the original checked native route, with nonblocking maintenance/REBUILD/FILE and error policy preserved.

Pending-only Inspector data uses current selected-session in-memory decision facts with exact owner fencing; it never claims old provider, scope, evidence, recipe, readiness or action facts are fresh. First persistence may recognize only the original exact Session/host/incarnation transition from its completed full predecessor. Unknown fields stay explicit, Review retains live authority, and full refresh/retry remains required. Rules phase-only chip and bound-controller Stop remain independent of these reads. Pricing/context metadata may be detached only under directly consumed stock source checks; replacement/custom routes preserve their existing behavior. Finite reads retain physical custody through repeated/all-task cancellation and shutdown.


## Current PR3023 compact lifecycle reconciliation (2026-10-13)

### TASK-34406 clarification: finite compact model setup reads

The compose and mount lifecycle paths in the stock compact model selector must avoid running the public checked provider-settings reader on the UI loop. They use one finite background invocation of the original public reader, retain its native checks, and apply the returned options and defaults only to the same mounted widget, App, configuration mapping, selection and source generation. The compose phase may yield controls from the current in-memory options while the checked refresh is pending. The original explicit provider-change action and custom reader contract remain fresh.

Capture definition-time stock reader metadata and reject source or owner changes before invocation and before publication. Retain and drain the exact worker on cancellation, repeated cancellation or unmount, including physical resource retirement, before accepting another lifecycle result. Do not publish over user choices made during the wait. Preserve the existing default-provider resolution, model fallback, Select-change suppression, temperature and control-bar synchronization. No permission or native posture verdict is cached; no TTL, timer, source guard, startup or Send budget changes.

Acceptance requires genuine held original-reader evidence for both lifecycle paths, actual UI-loop progress during the hold, normal worker and native retirement, current-source and owner controls, and the original provider/default-selection tests. Cross-platform and whole-app performance gates remain separate.


#### Compact model display inputs and transient cache replacement

The stock App retains its constructor settings mapping while ordinary original settings reloads may replace the global cache. Object identity with that transient cache, or its last source marker, is therefore not an eligibility proof for a finite stock provider read. Exact plain App mappings and nested display/default values may select that read under the existing defining callback/class/lookup, current lexical configuration identity, parent, loop and owner fences. These input values grant no native authority; the original public callback retains fresh permission, path and posture checks for each invocation.

Capture and recheck the exact App mapping and its plain inputs without rebinding either App or cache. Custom mapping classes or nested coercions, customized readers/bodies/classes and unsupported sources retain the direct lifecycle contract. A preinstalled exact plain dict is display data under the same rule; it cannot grant storage or Send permission. Mid-read mapping, user selection, source or generation changes still refuse publication, and the exact original callback Task still drains before retirement.




## Current PR3023 Collections finite setup and reconciliation

### TASK-34406 clarification: finite deferred Collections capture setup

The stock post-ready timer currently calls the synchronous Collections capture initializer on the UI loop. The source-current contained whole probe records this original caller during the idle UI pause; detached setup must preserve the fresh path, data-root, schema, offline-store and legacy checks. This amendment refines this ADR's finite callback lifetime and preserves ADR113 Local/Server authority selection. ADR085 concerns the activity receipt switcher and does not govern this initializer.

Only the original deferred timer with the exact existing stock file-backed LibraryCollectionsDB may select one finite background build. Keep constructor and synchronous first-use/custom/memory routes unchanged. The callback invokes the captured original native readers and builders, returns detached pieces, and physically retires only its newly created worker connection/resources on that same worker; existing borrowers remain live. The UI loop constructs the non-native scope/service, publishes fields and performs original authority activation only after exact App, deferred scope, database, original defining source bindings, selected configuration path/generation and runtime owner still match. A first-use winner, replacement, changed source or shutdown cannot be overwritten.

Retain the exact initializer Task and native callback through cancellation, repeated cancellation and failure. Capture shutdown seals new initializer admission and deactivates the current scope before cancelling/draining the initializer, and completes the existing extraction shutdown before creator-resource close. A cancelled Task alone cannot prove native retirement. Late timer calls cannot start after this boundary; no result or authority survives retirement for another initialization. Generic unmount cancellation occurring later is insufficient.

Keep the original timer interval, source/permission checks, reconciliation behavior and direct APIs. No configuration or native permission cache, longer TTL, constructor-wide guard, await-held native scope or performance ceiling change is introduced. Original Collections app-wiring contracts and genuine held native build/source/owner/first-use/cancellation/shutdown controls establish acceptance. Whole-app startup, Send and platform gates remain independent.


The current reconciliation owner also retains its two original ordered callbacks (interruption then offline batch limit25) under the same finite read primitive. Capture shutdown closes admission and drains that owner set before extraction shutdown/creator resource close. Direct source callbacks, exact captured database and operation-owned scopes remain unchanged.


### Current Character finite callback custody — 2026-10-13

The existing Character controller retains each issued exact file-backed CharactersRAGDB callback through physical return before cancellation reaches its caller. It reuses the Console concurrent-Future preparation lifetime around the same operation_owned_connection and _core_operation scopes as the existing DB offload; only newly acquired handles close, and existing borrowed connections/transactions remain caller-owned. No global DB offload cancellation behavior changes. Memory, subclass and custom database routes keep their original offload and callback arguments, errors and values. The controller and current view-worker owner retain the issued lifetime; an asyncio Task cancellation alone is not native retirement.

The accepted TASK33620.15.1 run-active one-second _scope_checked cache and _scope_invalidations remain authoritative, including unsettled observations and invalidation during refresh. Display-only cadence and paired/inner batching are separate qualifications: preserve original ambient midpoint checks on the loop, direct/custom reads, complete source checks before adopting stock bodies and all post-retirement owner/publication fences. No permission cache, new service, longer deadline or whole-Send performance claim follows from a finite callback reduction.

The exact file-backed Character scope may keep its existing before/after metadata reads on one finite owned handle. The intervening ambient identity check still runs on the issuing event loop through a bounded concurrent Future, with the source ten-second handoff limit, cancellation of expired checks and the final original post-return validation. This neither caches metadata nor changes admission/transaction scope. Memory, subclass and custom DB routes retain their preceding two-await callback behavior; current run-active one-second scope reuse and invalidation remain unchanged.

The stock active display may carry that same original display-owner closure through refresh_if_scope_changed into the existing finite inner batch. This does not establish a two-second memo on the active path or alter the original one-second _scope_checked/invalidation policy. Recheck original definitions after run/owner callbacks and scope awaits before optional keyword dispatch; dynamically customized no-argument callbacks retain the original route.

TASK-33620.15.6 (2026-10-19): the stock display, active and idle, may also decide inside that one inner batch. The batch receives the displayed fingerprint, ends after its first fresh pair when that still matches, and otherwise loads and fence-reads as before, so a changed check opens one finite handle (one ADR-125 helper) instead of a probe handle and a batch handle. Publication waits for the callback's retirement and advances the generation as _begin would. Outcomes keep the two-callback route's semantics: owner drift or a changed qualified body during the read publishes nothing; a scope that moved during the load, and a first read that did not decide for a current owner, take the original refresh and probe routes; dynamically customized no-argument callbacks retain the original route. No metadata cache, new coordinator, admission or retirement change follows; forced, direct and custom routes are unchanged.

The accepted Character display memo is optional stock presentation only: two-second same-owner reuse never changes explicit/custom fresh actions. Current run-active one-second scope reuse/invalidation remains the preceding owner and is not replaced by the older source removal. Display work holds its existing lock through finite callback retirement, rechecks the captured source/ambient owner after awaits, and cannot establish a memo from error, cancellation, displaced publication or an unrelated prior successful state. No private asyncio coordinator is needed around the existing physically owned await.

### 2026-10-13 — Literal Windows argv codec prerequisite (TASK-34406)

Frozen source TASK-34404 AC13 and commit `7f0b51a172` supplied a narrow
prerequisite for valid Windows executable paths containing literal `\x`.
That task is absent from the current branch; only this outcome is reconciled
under TASK-34406. The current original private writer reproduces a
`ConfigSerializationError` for the literal `\x64` path, while the ordinary
path control succeeds (current-writer RED r2: one failure/one pass).

The three existing config serialization consumers use the original TOML
codec for ordinary values, including its one-argument fault seam. Only an
exact builtin string containing literal `\x` selects the special string
encoder; the helper is imported on that first affected use. Keys, other
values/codecs, argv spelling and consent fingerprints are unchanged. The
writer's original parse-back, atomic private admission, selected-source
checks, publication/cache ownership and error boundaries remain mandatory.
This does not extend the writer to promise fidelity for unrelated legacy
control-character behavior, normalize arguments, or add a dependency.

Cross-platform codec and original writer/Hook Save controls are distinct
from actual Windows native execution/retirement evidence. The remaining
source-only TASK-34404 outcomes are not completed by this amendment.

### 2026-10-13 — Current Change Review marker read reconciliation (TASK-34406 AC20/42)

A real SQLite authorizer on the current bridge reproduces reads of unrelated
legacy run fields and child/step payloads for both file and memory stores,
while primary ordering, anchors and durable snapshots already match. Add the
source's read-only primary ID/assistant-anchor query to the existing
AgentRunsDB connection owner. An exact stock `list_runs` definition selects
that projection; legacy adapters, subclasses and replaced callbacks keep
their preceding `list_runs(..., include_superseded=False)` route. No schema,
status, history or native admission policy changes.

The marker reader captures the selected DB and existing render callback,
keeps caller-owned `operation_owned_connection` custody and preserves the
optional snapshot-error fallback. Shared-loop offloading and late-publication
fences require separate current-owner native/cancellation qualification
before integration; SQL projection alone does not establish that lifetime.

The original-return native control separately reproduces zero shared-loop
progress while the actual stock marker SELECT is held. The existing marker
projection may prepare one exact stock file-backed read through
`run_preparation_read`, observed by that projection and the current runtime.
The callback owns its finite core operation and retires only the connection
it acquired; a prior same-worker borrower and a replaced foreign cache remain
owned by their original callers. Source/body/default, DB/local/path, runtime,
coordinator, conversation and publication-revision changes refuse publication.
Cancellation, including repeated cancellation, waits for physical retirement.

The existing transcript caller awaits preparation before capturing messages,
then uses the current shared screen-closure predicate. Custom and memory
readers retain their preceding direct contract; the current send-ack overlay
and all subsequent transcript work stay in their existing order. No storage
permission, broad cache, timer cadence, shutdown deadline or native platform
claim is introduced.


### Current PR3023 browser finite pair reconciliation

The current flat persisted browser retains its global/Default scopes, generic-character query filter, pagination, partial-error/custom-service ABI and newer archive/membership/cache-token publication checks. Actual original-source qualification reproduces two physical opens for the stock pair, changed-source publication and captured-handle retirement gaps; original App shutdown also returns while its issued browser callback is still native-live. Only the qualified stock pair shares one finite counted worker interval. Its existing controller preparation-read owner retains the private executor Future through cancellation, and its captured Notes handle retires after operation exit without touching caller borrowers or a replacement cache. Definition-time consumed producer records select or decline this optional path; custom/memory readers keep the previous route. The shared Notes close policy is extracted with its ordinary default error/rollback behavior preserved; strict captured retirement is internal to this finite reader. No global DB cancellation contract, new service, schema, deadline or authority change.

### 2026-10-13 — Native metadata prerequisite test plan (TASK34406/34563.36)

Root released test-only qualification preparation before any Windows optimization selection. Keep current Windows full derivation, actual owner/security projection, ordinary-hold exclusion and accepted TASK34408 fixed creation-intent durability unchanged. Exact source invalid-ancestor and ancestor-replacement controls are distinct from newly authored standalone fresh descriptor/absence/order/native retirement/uncertain-close controls. Source OPT74 native counters assume both one-name validation and TokenOwner queries; retain them unchanged for their later stage, never label them a current-security/full-ancestor baseline. Source binding-tree custody controls require their original missing owner and cannot be silently imported as facade tests. An independently reviewed exact test snapshot and actual supported Windows execution precede any product prerequisite release. Portable collection/skips establish no native qualification; elevated original-token policy and acquisition reuse need separate review.

### 2026-10-14 — Fresh Windows metadata prerequisite, before optimization selection

The independently reviewed supported Windows capability run38515921962 keeps
the original73 controls passing and records12 failures solely for absent
metadata APIs/retained-close type. These are capability failures, not evidence
that any new native metadata behavior ran. Under TASK34406/34563.36 add only
fresh named scalar and batch metadata observations with the current TokenUser
security projection and copied exact self-relative descriptor bytes. Both APIs
use the same checked native handle custody. Full component chains remain
validated before native access; one-name validation is not selected.

Every newly acquired handle incarnation enters custody before metadata/NTFS
validation can fail. Retire each once, including failures before publication;
a false or raising CloseHandle retains the exact native owner, numeric handle,
known identity, path and close error in the defining exception, without retry.
A retained numeric handle alone never authorizes later cleanup. Ordinary
stat/fstat, their custom callback arguments and existing error behavior remain
unchanged. No process-wide closer or authority cache is introduced.

Keep current fixed creation-intent durability, full derivation, Windows reuse
exclusions, TokenUser policy and all ordinary security guards. TokenOwner,
scoped acquisition reuse and OPT09/36/74/task22 require separate release. The
scalar and batch uncertainty tests must each exercise real protected-HANDLE
failure plus injected before/after/raising outcomes, with exact incarnation
cleanup and no blind retry. Independent frozen review and actual Windows
behavior/refusal/retirement qualification precede optimization selection.

### 2026-10-14 — Current metadata optimization preselection tests

The separately reviewed metadata prerequisite passed105 actual Windows controls
on exact snapshot e1b9c7e30617ee9c1c9c94a930ee438b5b68baf6. This releases only
current full-ancestor counter and config-observation test preparation for
TASK34563.36/.28. Preserve current TokenUser, opener-captured identity and exact
five-field config posture projection. Original source TokenOwner counters,
six-field descriptor stamps and absent reuse helpers are historical contracts,
not current baseline fixtures. Native birth observers must be passive and
assert only after owned cleanup. The config proposal must preserve ordinary
error/custom/duplicate/order/memo behavior while retaining and propagating any
new metadata-close uncertainty before fallback or further work. Exact test
review and supported-native baseline precede any product selection or a
bounded amendment for that boundary. No reuse, elevated policy, OPT09/task22
or optimization product is released by this test-preparation note.

### 2026-10-14 — Connected Collections defining ownership

Move only the existing connected Collections construction/deferred setup,
finite read, publication/reconciliation and shutdown chain into actual
app_collections_wiring functions and a real CollectionsServiceWiringMixin.
ServiceWiringMixin inherits the original callable descriptors and original
module importing/patch slots hold the actual implementations. Keep the current
setup source checker in its original owner and enroll only source records
already consumed by the selected route; constructor-only/custom functions do
not gain a blanket refusal predicate. Original live collaborator slots, bound
partials, defaults, scopes, result fields, transactions and borrowed ownership
remain with the same App/database/runtime.

The already recorded defining-module pointer is retained in the captured setup
source. Validate it before the retained request/currentness entry reads a task,
exception class, namespace or proof reader. The issued initializer uses that
captured original module through its existing await/refusal/finally path, while
reading the module's collaborator slots live. Preserve exact obsolete/optional
unavailable/cancellation categories and physical read retirement before setup
clearance or creator shutdown. There is no new storage authority, task owner,
closer, retry, fallback, service or source policy.

The exact connected proposal4be7147c and scoped independent reviewf17aa6ea
release the four-file current integration after declared plan recording. Actual
stock held-pointer negatives and3 prospective passes qualify guard ordering,
not whole native/latency acceptance. Existing native metadata and all withheld
optimization policies remain separate; original affected cancellation, source,
custom, borrowed, first-use and shutdown controls precede final acceptance.

### Current PR3023 original receipt hydration and fleet recovery lifetime

TASK34406 AC53/54 retain the actual current runtime hydration service, authority token and runs owner. Exact stock file-backed Runs and standard marks/Characters storage use existing operation_owned_connection scopes around the complete synchronous hydration callback; pre-existing borrowers and transactions remain owned by their caller. Memory/custom runs retain their preceding close branch. The existing runtime preparation-read set retains the original callback through repeated cancellation, and the original hydration Task remains referenced until physical callback and task retirement.

Fleet recovery stays with its existing coordinator and run_owned_db_call API. Runtime disposal selects only the exact stock coordinator already installed in the controller's ordinary _fleet_wake field; custom presentation facades acquire no required getter, method or refusal policy. The same captured stock coordinator is synchronously fenced before the first disposal await, then carried to the inner join before store/creator cleanup and the outer finally backstop. Controller shutdown also joins its exact recovery task before its no-stream return; recorded waiter cancellation is delivered after cleanup. Disposed new/deferred scheduling, post-native-await steps and locked final readiness/error publication remain fenced. Current loop task-factory, original grace, errors and generic DB callback policy are unchanged. Original7 source controls plus4 actual early-worktree-close/replaced-controller/facade/getter controls qualify this scoped change; unrelated incomplete slots fixtures remain negative evidence. No generic custody registry, storage authority cache or optimization selection follows.

### Current PR3023 original citation-count callback lifetime

TASK34406 AC55 retains the already-issued Console citation-count Worker through its original database callback. Only the exact stock CitationTraceRepository with exact file-backed CharactersRAGDB shields the captured run_owned_db_call Task through repeated cancellation, consumes its original outcome, then delivers cancellation. The existing generic DB callback/operation-owned connection and query argument ABI remain unchanged; custom, subclass and memory routes keep the preceding direct await. Existing generation, signature, eligibility and repository-token publication checks remain current. Newly acquired native handles retire in their original worker scope, borrowers remain usable, and original callback errors retain identity. Existing host group membership already includes this Worker; it now waits on its actual producer rather than relying on synthetic group controls. No global database cancellation policy or custody registry is added.

### Current PR3023 original retrieval-display callback lifetime

TASK34406 AC56 opts only stock file-backed retrieval display into the existing finite read owner. The synchronous conversation, world-book and dictionary producers keep their exact current callback, database and original operation-owned connection policy. Retain and shield the already issued coroutine/Task through repeated waiter cancellation, consume its original completion/error, then deliver cancellation before later reads, cache writes or UI publication. The shared resolver's optional retain_display_reads selector defaults off; ordinary prompt/default calls, custom getters/services, subclass and memory routes preserve direct cancellation and the existing error boundary. Qualification is checked inside the original workspace registry try, so getter failures retain their prior category. No generic DB cancellation change, new owner registry, source authority, timeout, retry or cross-owner close is introduced. Creator and borrowed connection retirement remain with their original synchronous scopes. Exact c287 three-source proposal and original314 per-case reconciliation authorize saved-source qualification; parent summaries without retained raw native child receipts remain limited evidence.

### Current run-log preparation lifetime

The run-log context optimization remains inside the original synchronous acquire_storage scope. The exact acquired lease and native hold retire before bind returns or its nonfatal error boundary completes. No lease spans an await, no fallback retries an uncertain close, and source-sharing fallback retains the original native operation. This is the existing finite lifetime contract, not a new admission or recovery authority.

### Context display database affinity in current PR3023 integration

The disposable Context reader uses its finite worker only with known stock repository and persistence collaborators whose actual databases are file-backed. Memory databases and unknown/custom collaborators run the same captured probe and projection on the issuing thread, preserving their existing connection and borrowed transaction affinity. Keep current owner checks around that read. Supplied global policy, including explicit None, remains an input; an unset policy still uses the original optional reader. This fallback adds no permission, cache, native authority or general DB cancellation policy.

Exact source and controlled qualification are recorded in the Context affinity proposal and prospective result binding in the PR3023 evidence directory. They include real generated-prefix memory, explicit policy, custom borrowed transactions and the original file-backed held/repeated-cancellation controls. Saved real-path qualification precedes final integration acceptance.

### Complete Eval persistence safe point during current participant drain

The installed Eval participant asks its existing persistence_safe_point callback for readiness outside the coordinator lock. This preserves the complete effective-config, explicit-profile-overrides and failed-save contract, including an explicit override equal to a shipped default. The original admission, pending operation, hold and post-callback source/closed-state checks remain unchanged. Callback errors retain their original identity, and custom or uninstalled loaders keep their existing route. No partial mirror, native authority or new refusal policy is introduced.

The original override-only dirty cases and the scoped error, admission, selector, writer and coordinator controls qualify this correction. The selector fixture exercises actual source rejection at the original scope boundary both while closed and after resume; gate closure alone grants no capture authority. Saved real-path results precede final integration acceptance.

### Character ambient cancellation transport in current integration

Both existing worker-to-issuer ambient handoffs transport the original BaseException through their existing concurrent Future wrapper. Supported accessor cancellation retains its exact object and cause, reaches the original coroutine error boundary, and retires the actual native callback before creator cleanup. Existing reader-change handling, custom and memory routes, source checks, publication policy and ten-second timeout remain unchanged. No general cancellation or retry policy is added.

The current native cancellation failures and custom comparison remain distinct from the reviewed candidate results. The candidate passes the corrected three cases and thirteen original error, source, borrowed and native lifetime controls. Saved real-path qualification precedes final acceptance.

### One pause probe for a repeated current stock hold

The final raw scope check may share a False pause query only across the same exact stock hold, authority and namespace object in that finite operation. Every counted lease remains live and independently retires through its original path. Admission records its own original query at definition; Raw rechecks its callback, code, globals, defaults, closure and class lookup/dictionary identities on each occurrence. A custom or changed callback follows the original dynamic per-lease route. No pause result survives the operation, and no permission, native evidence or recovery authority is cached.

The original six stock/custom/mapping/mid-call/maintenance controls and three default/lookup controls qualify the bounded selection. Their native claims are limited to the original asserted boundaries; neither every physical close nor Windows work is inferred from lease-set cleanup. The complete Eval persistence safe-point correction remains intact. Saved source qualification precedes final integration acceptance.

### Historical display task creation failure retains its issued callback

The existing stock file-backed historical display loader captures the same database callback and arguments before task creation. If the loop factory raises after issuing that exact coroutine, the loader recovers and joins only Tasks owning that coroutine, without cancelling the inner to_thread operation. It consumes the issued outcome before re-raising the original creation error outside the existing business read-error handler. Only an unissued coroutine is closed. The defining owner's pending flag resets in finally after known work retirement, so retry cannot overtake a live callback.

This local recovery follows the existing view-worker pattern and adds no generic task registry, database cancellation policy, source permission or timeout. Original successful creation, repeated waiter cancellation, borrowed connection, currentness, cache and publication behavior remain. The real App/view-drain factory cases and original five controls qualify these bounded paths; they do not establish universal immediate-terminal or arbitrary custom factory recovery. Saved real-path qualification precedes final integration acceptance.

### Checked standalone cost derivation — 2026-10-14

TASK34406 AC5/51 routes the existing zero-argument cost builder through the existing checked synchronous presentation owner. Select the current session and store once in the original order; ownerless state clears before configuration/pricing work. The original derivation body and optional error boundary remain. A before-body or post-body refusal retains the last accepted display, cache classification and two outer pricing memo references, so refused work cannot change the real TTL timer or seed a later memo hit. The existing coalesced refresh selects a fresh pair on retry. Custom zero-argument builders and the original ten-second timer keep their dispatch contract.

Original direct-entry and actual armed-TTL controls reproduced blocked loop progress under held configuration locks. Candidate d0d88f61 and independently reviewed original/new controls require genuine installed-path qualification with original bounds, real post-check refusal, ownerless Timer.stop/task join and fresh-owner/source replay. This extends the existing checked display contract; it adds no authority, general rollback or scheduler. Full branch/current-base/native/latency acceptance remains open.

### Exact owned descriptor retirement — current Task17 integration

Known descriptor cleanup uses only the existing exact issued raw operation, active process/thread/task ownership and registered FD membership. It does not re-admit new work through source/parent checks after revocation. Voice/visual precedence and the preceding checked fallback for unrelated, copied, foreign or inactive ownership remain. Uncertain known custody still reaches the original close owner, which refuses native retry and retains resources/leases. Subsequent open/read/write/publication gates remain fresh. The current source review and original ownership/count/uncertainty plus affected lock/posture controls qualify this bounded seam; no general native retirement, platform or whole-Send timing claim follows.

### Current draft-cost settings freshness

Current cost R3 correction before qualification: independently reviewed source77069a2f/a3fc4cd2 fixes two reached draft-estimate regressions in R2 while preserving checked warm behavior, original getter/error/ownerless/zero-arg ABI and three-reference refusal rollback. Only an existing same complete stock owner/source/current-Runtime projection with settings-revision-only drift selects the existing fresh nonblocking native config owner. No fallback follows presentationFalse; actual native post-body full-key/closing/Runtime/disposed checks retain prior display and request existing fresh retry. Qualify exact original two draft regressions, original four causal/ownerless/checked refusal cases and seven real held-settings/native postbody/lifecycle controls; all original bounds/native/Timer/error/source gates retained. Root source and TEST reviews precede guarded temporary installation only of a3fc, freshfa612 baseline and explicit current Task-governance rebind with all79AC unchanged. No task/source/nativeall/platform/base/latency/wholebranch completion. ADR required:no new ADR; existing126/230 clarified for this same-owner native selection. Original active-edit Mock failure reproduces on baseline and remains separate.


### TASK-34406 AC77: successful settled missing companion metadata

A complete successful result that the exact profile record is absent may use the existing ordinary POSIX hold-owned derived-metadata protocol. This is passive negative metadata, not admission authority. The original registry shared lock is still acquired on every call; successful complete derivations must bracket identical selector, registry, member, source and ancestor inputs at the same epoch and original one-second settle margin. The original temporary companion guard must close successfully before a missing result is published. The API still returns None and the caller still acquires both original fallback leases. Pending publication, uncertain cleanup, unsupported storage and Windows remain ineligible.

Only absence of the fixed creation-intent inputs already validated by selector evidence may participate in those stamps. A missing registered root, uncertain observation or arbitrary missing dependency does not qualify. Original fresh-result branches retain their member inode/type/link, overlap, selected-source, parent-posture and lock-lifetime checks at the same boundaries. The original missing branch returns passive None before the found-record-only companion-guard checks; absence metadata must preserve that branch and cannot create a companion guard or admission authority. Both independent fallback acquisitions and their original current native scope, source and custody checks still run on every call. This clarification adds no found-record-only checks to the missing route. Publication, profile enrollment, selector/registry changes and callback or defining-body changes invalidate evidence and retain the original fresh result/refusal route. Two complete successful observations are required even when the observed profile record is missing; a cold refusal never proves invalidation of engaged evidence.

The original six controls ran unchanged on actual Darwin and Linux before this clarification: each reached3PASS/3FAIL, with one avoided-record-walk RED and separate original post-retirement guard/reason-specific reference mismatches. Later controls must observe an active guard inside its operation, keep original retirement assertions after exit, preserve the actual earlier pre-body refusal and establish engaged negative evidence before mutation. No latency/census/platform-wide or task completion follows from the baseline. Existing authority, storage and lock-order boundaries are unchanged; no new ADR or cache is introduced.

### TASK-34563.28 / OPT36: current shipping config posture adapter

Under Task34563.28, the existing synchronous config user-data-directory posture helper may combine one eligible stock Windows observation through the already-qualified fresh metadata API. Its result remains exactly the current ordered five-field posture tuple, including duplicates/missing entries; copied descriptors are never memo/authority fields. The original bound verification, configured/default selection, guarded resolver, memo key/timing/brackets, locks and admission/native custody remain unchanged. POSIX and known custom callback/facade routes retain their original scalar invocation and error behavior.

Only a finite captured original definition/binding record may select the optional batch. It retains no native or permission result and introduces no generalized source service or acquisition reuse. Direct customization declines before optional work; a consumed source change within a live batch/projection invalidates that whole observation after original retirement, so no mixed tuple is published. This is not an atomic transaction across calls or immunity to arbitrary Python tampering.

The exact stock descriptor representability PermissionError13/marker may reach the original scalar observation only after the fresh producer has drained all metadata-owned births. Ordinary OSError remains unavailable. The defining retained metadata-close error is an intentional exception to that broad OSError-to-None policy: retain and propagate its exact failed native owners and original exception chain before fallback, None conversion, resolver retry or further work. Failed/raising close is not retried, and a numeric handle alone is never cleanup authority. Original owned cleanup still runs; the refusal forbids subsequent observation/acquisition/resolver work, not required retirement. Independently reviewed current-source causal, custom/held-source, actual descriptor and protected/raised-close controls plus supported-native results precede release. Historical prototype/counter/source results retain their original scope; no TokenOwner/elevation, Windows acquisition reuse, OPT74, whole-Send or latency acceptance follows.


### Workspace kernel source and layout compatibility (PR3023, 2026-10-15)

The existing optional finite stock run-log probe follows Workspace's current Kernel-first declaration under `backlog/decisions/228-sqlite-store-kernel.md`; that ADR's Proposed status is unchanged. Its finite source metadata must bind both effective Workspace lookups and the actual declaring Kernel methods, consumed local overrides/global hooks, wrapped/code/default identities and fixed class layout. It must not index inherited methods as if they were local definitions or copy older connection methods back into Workspace.

Cold and cached stock admission recognize only the canonical inherited `is_memory_db=False` layout, with no local field/method shadow and the same exact instance fields/native owner. Memory/custom/subclass/replaced source, body/default/global or class/instance shadow routes retain the original direct callback once with original return/error and lifetime. Generic source/function-current checks, Kernel implementation, `db.base` provenance, transactions, finite worker and owned/borrowed/uncertain retirement contracts remain unchanged; no permission or cleanup authority comes from this metadata.

The rebase incident is a saved-path collection `KeyError('_get_connection')` at Workspace's old own-class registry, before any test body; inherited-False cached eligibility is a separate source-supported mismatch, not evidence of a native leak. Existing cold/cached native and custom controls plus a small inherited-source refusal family qualify the repair. No whole-platform, latency or branch acceptance follows.


### AgentRuns inherited kernel qualification — PR3023 current integration

After TASK-33132 moves AgentRuns onto SQLiteStoreKernel, finite run-log and
Change Review reads retain the incoming MRO, db.base provenance and transaction
policy. Their existing exact stock qualification records the actual inherited
and defining kernel callbacks, own connection hooks, BaseDB connector and
consumed module globals. Fixed factory slots are shared as data by the two
existing readers; this adds no general MRO registry or new authority. Changed
sources decline before capture or invalidate an issued read through its existing
obsolete-result path, after the acquired handle retires. Borrowed, custom and
memory routes retain their preceding lifetimes and errors. Workspace's deferred
default and the kernel/AgentRuns immediate default remain unchanged; explicit
transaction keywords still win. Existing ADR-230 governs preparation ownership.


### Default-only Console binding display (TASK34406 AC22, Linux census follow-up)

Under [ADR028](028-settings-workspaces-category-and-folder-roots.md), Default
cannot have runtime folder bindings. The exact source-current stock file-backed
registry may therefore derive an empty tuple for the Console display read-view
and a wholly Default availability snapshot without opening a native callback.
This is immutable presentation policy, not a cached permission or DB observation.
Definition-owned policy methods, normalizer, Default identity and plain receiver
lookup must remain current; custom, memory, changed-source, named and mixed reads
keep their preceding route. Public/direct Default listers still freshly remove
legacy bindings. Keep the existing worker, generation, selected registry/database,
runtime, publication and cancellation checks, and recheck policy before accepting
the derived snapshot. A previously issued native callback still retires through
its original owner. No TTL, admission, native authority, deadline or census budget
changes follow from this exception.

### TASK-34406.2 — Admit explicit run writers and saved history at use (2026-10-16)

A stock Console run whose actual injected stock writer has an explicit root
admits that root before binding, instead of resolving an unrelated global
history root. Initial config/profile, database, workspace and permission admission
and physical worker/resource lifetimes still apply. Exact writer class, original
path/bind callbacks and consumed root must qualify; implicit, custom, changed or
stale public writers retain the global entry route. No admission decision is cached.

Conversation-scope saved-run search selects its current global root while the
profile/workspace scope is held, then retains history admission through directory
probes and actual segment loading. The canonical resolver consumes that admitted
root explicitly; a missing root never triggers a second implicit selection.
Custom readers retain their preceding one-argument ABI under the same scope.
This deliberately moves refusal for an unrelated global history source to its
actual search; refusal for the live writer remains before bind/provider entry.
Prompt-history persistence, audit order and durable checkpoints are unchanged.

Rejected: simply skipping global entry admission without guarding older-history
loads, or admitting one root and then reselecting another. Qualify real native
refusal/retirement and matched first/later sends; a successful reply alone does
not establish the broad latency or platform targets. See TASK-34406.2 and the
companion ADR-126/ADR-230 amendment.

The explicit writer root may be passively qualified from already loaded stock
source objects before entry and included in the initial admission observation.
That single execution scope still freshly admits every actual source before the
run body starts; it avoids re-admitting the same profile/config/database sources
in a nested scope. Qualification never imports or calls an optional source before
admission. Missing or custom source objects keep the admitted global fallback.
