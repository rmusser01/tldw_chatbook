# ADR-126: Complete local backup and recovery

Status: Accepted â€” revision 4 approved by the user on 2026-09-07
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

Revision: 4 â€” incorporates the fourth user-requested design review.

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

### Task10 phase12 â€” concrete Persona Visual lifetimes (rulings70â€“73)

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

### Task10 phase13 â€” Shared Visual Identity source/native lifetimes (rulings74â€“75)

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

Rulings76â€“77 close actual public callback borrowing found during phase13 review.
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

### Task10 phase14 â€” TTS repository lifecycle foundation (ruling79)

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

### Task10 phase14b â€” outer TTS backup native retention (ruling80)

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


### Task10 phase14c â€” ordinary candidate validation native job (ruling81)

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

### Task10 phase14d â€” configured TTS source and delegated pin ownership (rulings82â€“88)

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

### Task10 phase14e â€” outer snapshot and native journal ownership (rulings89â€“91)

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

### Task10 phase14f â€” exact TTS connection and profile-lock retirement (rulings95â€“99)

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


### Task10 phase14g: live TTS migration and restore outcomes (rulings100â€“108)

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

### Task10 phase14h: private clone materializer ownership (rulings109â€“112)

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

### Task10 phase14i: retained voice bundle operations (rulings113â€“114)

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


### Task10 phase14j ruling118 â€” original concurrent first initialization

The same-process first authority initializer reserves its resolved root on its actual pending acquisition record under the existing coordinator condition. Followers wait outside the lock and recheck original cancellation and provenance before proceeding. The initializer runs original startup/native qualification and authority construction outside the coordinator lock and releases its reservation in finally. Same-thread recursion refuses rather than waiting for itself. Incomplete, failed, crashed or foreign-process evidence retains the original fail-closed behavior; this does not repair or retry durable uncertainty, add a registry/mutex, or qualify whole-app startup. Required evidence includes deterministic marker-before-register follower ordering and the supported uninitialized128-sample service case. Exact shared-source BASE reproduces the prior follower refusal at startup_permission; unchanged external dependencies do not make that a whole-repository BASE run.


### Task10 phase14j rulings115â€“117 â€” profile evidence and Library work

Original profile-service admission/drain/resume counts actual PID/Thread/Task callers through final evidence/result work; synchronous evidence legitimately runs on a worker thread. Original same-receiver nested calls can settle while admission closes, preserving artifact coordinator before consumer fence lock order. Fresh sample FD/codec work always reacquires ordinary selected file and required-parent admission before metadata/allocation. Optional private native outcomes preserve known returned resources before late errors, unknown allocations and first uncertain native/lease closes independently of content validity. Original body controls win over cleanup controls while both outcomes stay accounted. Default standalone/shared Settings validators keep their existing API/behavior.

The original lazy app profile-service factory binds its exact original repository/config/profile/dependency/coordinator and static no-shadow identities through profile_source. It checks before admission, new native work, final evidence/result publication and resume. Custom/prebinding/proxy sources are not replayed or promoted; a source relationship does not qualify artifact runtime work.

Profile Library readiness preserves exact mounted typed input, submitted conflict or maintenance-refused drafts, review handles and their original service sessions. Maintenance neither dismisses nor saves/discards them. It closes new action/page admission, defers late page/search publication, and retains actual result/cleanup lifetimes. Reopened-conflict Cancel preserves submitted text; original refresh/retry Save resolves it. Already submitted text is retained before stale action-target refusal. Sanitized export retains actual queued/running writer and native stream through repeated cancellation, original overwrite bytes/result and first uncertain cleanup; caller destinations are not baseline resources. Service/file completion alone cannot retire the remaining Speech caller's queued/post-thread UI work. Whole-app composition and all other Task10 debts remain open.


### Task10 phase14j ruling119 â€” selected export before truncation

A reached output-parent replacement after original path validation redirected the sanitized writer. The exact existing export operation now snapshots selected identities after ordinary admission, owns/verifies the parent descriptor, opens an existing output without truncation and verifies the selected regular-file FD before truncating it. Missing output uses exclusive creation beneath the verified parent; this preserves ordinary new/overwrite behavior without replacing or deleting existing output. Stream and parent/target FDs have explicit separate owning closers and retained first/unknown outcomes. Later pathname replacement cannot redirect writes through the already selected FD. This is one caller-selected writer, not a generic filesystem capability or baseline resource.


### Task10 phase14j ruling120 â€” ordinary platform compatibility

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


### PERF-07/PERF-08 amendment â€” reusable admission evidence (TASK-33266, TASK-33267)

Status: **Approved by the owner on 2026-09-29.** The owner approved the direction
as decisions D1 and D2 of the
[2026-09-27 structural perf audit](../../qa/perf-structural-audit-2026-09-27/report.md)
(Â§4 PERF-07, PERF-08), then settled the mechanism with a criterion: choose
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

### TASK-34200 amendment â€” file identity is path + inode, not the device

Status: **Approved by the owner on 2026-10-03** (chosen over a volume-UUID
identity and over filing only).

**Context.** The owner's Mac rebooted on 2026-09-28 and its data volume came
back with a different `st_dev` (16777234 â†’ 16777230). The admission registry
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

### TASK-34100.17 amendment â€” first-run setup confirmation and keychain-held provider keys (2026-10-03)

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
### TASK-34404 amendment â€” native Windows ordinary admission evidence

Status: Owner-authorized implementation, 2026-10-04; merge requires native
differential and complete-conversation receipts.

Windows ordinary acquisitions use the same confirmed-evidence mechanism and
count-before-final-observation ordering as POSIX. Every observation uses the
repository Windows facade, whose stat values come from a freshly opened
non-reparse local-NTFS handle: actual file identity, owner SID and ordered DACL
projection, exact owner/DACL descriptor bytes in posture stamps, and
FILE_BASIC_INFO.ChangeTime in content st_ctime_ns. CPython's Windows
creation-time st_ctime is never evidence. Posture and content for the same
named path use one fresh stat handle; descriptor bytes are immutable snapshot
data, not cached permission or path decisions. Ancestor ChangeTime is not
posture evidence: unrelated directory content churn does not change ownership
or privacy and would prevent confirmation during normal app activity.

The per-call volume check uses qualified_for("admission", root.parent), which
reads the current native volume identity and refuses FILE_READ_ONLY_VOLUME,
remote/unsupported filesystems or unavailable qualification. It does not
persist permission or path decisions. The settle rule, evidence confirmation,
epoch, pause, provenance, maintenance gates and complete fallback derivation
are unchanged. New native tests compare verdicts after actual ACL edits,
in-place selector edits with restored mtime, directory namespace changes with
restored mtime, and directory replacement; they also mutate after the lease is
counted and before the final observation, and count native opens for both routes.

This closes the earlier Windows-only full-derivation restriction when the
native tests pass. Native ownership correction is a separate boundary below;
a successful admission observation alone never authorizes a foreign SQLite file.


### TASK-34404 amendment â€” proven private custody under Windows TokenOwner

Native Windows Server 2022 evidence (run 37230513912, job 111518996121)
shows TokenUser=runneradmin, TokenOwner=BUILTIN Administrators. The facade-created
private parent and main database have the actual TokenUser owner. SQLite's
default-created WAL/SHM have the actual TokenOwner owner and inherit an explicit
full-access ACE naming that exact TokenUser, plus SYSTEM/Administrators grants.
The previous projection mapped these sidecars to uid 0 and ordinary reopening
refused them as wrong_owner. Ordinary local Windows has TokenOwner=TokenUser.

Actual TokenUser ownership retains the existing policy. An alternative native
owner satisfies private-user custody only when all of these are measured:
- the actual owner SID equals the current process token's TokenOwner SID;
- that SID is already an administrative principal trusted by the existing policy;
- the freshly read DACL has only ordinary allow ACEs and an applicable full-access
  ACE naming the exact TokenUser SID (OWNER RIGHTS is insufficient);
- the conservative DACL projection has no public/shared exposure bits.

TokenOwner is queried afresh for each security observation. Immutable descriptor
decoding is cached only by descriptor bytes, directory kind, TokenUser and current
TokenOwner; no token owner, permission or filesystem decision is cached.
Raw owner SID and descriptor bytes remain native evidence: uid 1000 is the
existing compatibility category, now expressing that narrowly proven custody.
Files owned by a foreign SID, default-owner files lacking the exact user ACE,
admin-only or OWNER RIGHTS-only ACLs, denied user grants and shared ACLs do not
satisfy this rule. The process token and actual file owner are never changed.
Ordinary user-owned file hardening and trusted system directory traversal retain
their existing policies.


The Windows evidence snapshot shares fresh parent handles only inside one
observation. Descent opens each component once. A second bottom-up pass reopens
every named parent-to-child association and reads current owner/DACL/content
from those validation handles; each child's binding is checked before its
parent's binding. A replaced, renamed or vanished ancestor therefore cannot
remain invisible behind an older pinned parent. Any identity mismatch raises
ESTALE and runs full derivation. Only needed parent handles survive descent,
all are closed before the snapshot returns, and no pin is kept for later calls.
A native regression replaces the data directory after child validation closes
and proves the subsequent parent binding check catches it.


A counted ordinary acquisition observes the union of selector evidence and all
related-path evidence in one fresh snapshot. Common posture/content dependencies
are read once and mapped back to every immutable evidence entry before comparison;
no entry's dependency is omitted. This avoids repeated traversal of common native
ancestors when one config operation admits lock, backup and temporary siblings.
The final epoch comparison remains after that observation. An unavailable optional
snapshot first retires the counted token and then runs full derivation, which remains
the only source of refusal. The native oracle changes a related sibling after
counting and proves rederivation without a leaked count; actual parent-replacement,
owner/ACL, control-record and pause/maintenance oracles retain their verdict checks.

### TASK-34406 amendment â€” presentation config lock entry

Console presentation may request nonblocking entry to the existing configuration
operation. If either the rebuild or file lock is held by another thread, entry
raises the code-owned `ConfigOperationBusy` category, releases any earlier lock
and acquisition resources, and neither enters the projection body nor changes
configuration cache or failure state. The Console coalesces one existing trailing
refresh and recomputes current state when entry succeeds. No selected path,
permission, policy or checked storage lifetime is cached by this contract.

Once both locks are acquired, presentation retains the same continuously checked
native operation until all synchronous projection work and nested configuration
readers retire. Default callers and explicit actions keep waiting for the locks;
maintenance, cancellation, selector changes and body/cleanup errors keep their
existing refusal and propagation semantics. A busy category raised inside the
projection body is an error, not an entry deferral. This narrow contract addresses
native-probe samples showing the UI blocked at the real configuration lock wait;
moving its checked body outside admission or replacing it with cached authority
would weaken the existing lifetime contract and is not adopted.

The third native probe also sampled independent credential polling loading
provider-readiness configuration on the UI thread. Synchronous Console
presentation entry points may share one disposable configuration mapping loaded
by a finite owned worker. Publication requires the same configuration generation
and lexical selector, app/database, session, workspace and settings revision as
the captured request. A cold or changed owner defers the presentation body;
expiry retains only that same owner's already rendered data while one worker
checks current configuration. Cancellation retires the worker's native resources
before its owner finishes. No cached mapping grants file or provider authority.

The live configuration getter remains the default for explicit actions, provider
selection and controller send wiring outside those synchronous presentation
scopes. Readiness itself still recomputes credential expiry and current shared
connection evidence from the disposable mapping; neither verdict is cached.
Once an admitted projection body enters, its continuous checked config lifetime
still encloses every nested synchronous config operation. This preserves the
existing recovery and send contracts while removing repeated presentation IO.

### TASK-34405 amendment â€” finite database callback counted intervals (2026-10-04)

A qualified `run_owned_db_call` owns one complete synchronous callback interval
for its exact installed, file-backed CharactersRAGDB, AgentRunsDB or WorkspaceDB
receiver. The existing `_core_operation` encloses the callback inside the existing
`operation_owned_connection` boundary. Same-receiver nested repository reads and
transactions therefore reuse one validated participant operation; every actual SQL
read and metadata/ambient publication fence remains in place. The operation exits
before retirement of a newly opened worker connection, because explicit close
must not run while its counted borrower remains live.

This interval grants only the receiver's existing bounded descendant scope. A
different repository instance, including another receiver at the same pathname,
requires independent ordinary admission and cannot inherit the original receiver's
operation. Fresh callbacks refuse while admission is closed; an already counted
callback may finish its same-receiver reads while pause waits for its operation and
native handles to retire. Native allocation leases remain separately counted.
Awaiter cancellation does not shorten the synchronous callback or retire its
resources early. Pre-existing borrowed handles and transactions remain owned by
the original caller; memory, subclasses and custom owners retain their prior route.
No persistent callback executor, checked-result cache or maintenance capability is
introduced. The measured alternative of retaining separate operations around each
metadata read repeated native admission within the same finite callback without
adding a freshness boundary; complete callback counting preserves those SQL fences
while eliminating that repeated admission work.

Console workspace availability uses that same finite callback boundary with its
captured registry service and database receiver. Qualified reads share the counted
interval without a redundant inner owned close; direct and unqualified adapter
reads retain their original ownership route. Publication rechecks the captured
service/database identities and request generation after the await. A redirected
registry database aborts remaining workspace reads and the current receiver is
captured afresh; no old receiver's cleanup may be redirected to the new receiver.

### TASK-34406 amendment â€” checked presentation source and defaults handoff (2026-10-04)

A finite readiness mapping carries the configuration generation and actual
validated raw selected path observed inside its checked config operation both
after entry and after loading and copying the mapping. Both tags must equal the
requested source, and publication still rechecks the complete UI owner. Ambient
selector equality before and after an await is insufficient: retargeting A to B
and back to A does not advance the config generation.

Disposable presentation scopes reuse established session settings and cannot
perform pristine-default convergence from their mapping, including while it is
expired. After the worker source and UI owner checks succeed, one explicit
handoff outside that scope may converge an eligible session from the freshly
checked mapping. Existing provenance, pristine-state and explicit-default
generation gates remain in force. The publication key is recaptured after that
handoff's own settings revision change. Unscoped actions continue to obtain live
configuration and retain their existing convergence behavior. This separates
display reuse from the authoritative session update without introducing a cached
send or file authority.

The Agent rail and fleet may likewise share one disposable historical projection
for their current owner. Its finite owned database callback derives from the exact
captured AgentRunsDB receiver, with the existing counted callback interval and
connection retirement. Publication rechecks bridge, receiver, profile/config,
session/workspace, conversation and process-local run identity. It never writes
the bridge's authoritative historical cache or redirects a captured read to a
replacement database. Live fleet and live summary precedence remain unchanged;
direct bridge history/action reads retain their existing contract. Changed
historical content requests only the existing coalesced Agent-section repaint.
The pure mode-bar presentation may use the same disposable readiness scope;
controller core-state/provider/runtime publication and sends remain live.

### TASK-34404 amendment â€” bootstrap barriers cover changed directory entries (2026-10-04)

Pending publication revalidates and fully synchronizes its existing pending record
and pinned bootstrap directory. It does not reopen and synchronize every unchanged
ancestor of that directory. Registration already synchronizes each newly created
directory entry through its exact pinned containing directory before returning,
then synchronizes the pending record and bootstrap directory. If an exclusive
creation loses after an absent observation, registration synchronizes that same
containing directory before accepting the competing name; the existing no-follow
owner and private-root checks still validate the resulting object. Required barrier
failures propagate and leave recovery evidence fenced. No directory flush failure
is converted to success, and no ownership-based or cached path decision selects
which required barrier may be omitted.

The ordinary Windows whole-HEAD control reproduced three restore failures before
this correction. A call-through failed-handle receipt identified the unchanged,
SYSTEM-owned home-parent directory (`C:\Users`, projected mode 0744), rather than
an operation-created directory or bootstrap root. The root barrier had succeeded.
A read-only same-object rights probe showed that neither write nor append access
could reopen that protected ancestor. Requesting only append access or enabling
backup privileges therefore does not correct the durability boundary. The native
Windows primitive remains unchanged: checked same-object reopen followed by normal
`NtFlushBuffersFileEx` flags zero, preserving directory metadata and device cache
synchronization. Publication retains all record, created-entry, rename, installed
metadata and terminal fence barriers. Removing the unbounded unchanged-ancestor
walk also removes its drive-empty `Path("/")` comparison on Windows; publication
no longer needs a hard-coded volume-root termination rule.

Native regression evidence covers each newly created ancestor's containing-entry
barrier, a competing creation, propagation of required creation and pending-root
barrier failures, and real flags-zero Windows barriers. The exact three original
app restore controls and supported OS qualification remain required evidence.

### Exact bootstrap creation intents and failed-barrier retries (TASK-34404)

The narrower publication barrier must also preserve a directory entry created by
an earlier failed attempt. A real Windows control created an ancestor successfully,
then denied its containing-directory reopen with an exact-user rights2|4 deny ACE.
After restoring that isolated test ACL, retrying registration omitted the prior
entry barrier. The original HEAD publication ancestor walk supplied it. Its three
mounted success controls therefore establish ordinary restore behavior, not retry
durability qualification.

Before each missing-directory creation, write a bounded private intent in its
existing pinned containing directory and synchronize both intent bytes and the
containing directory. The name is derived only from the local lexical child;
strict contents bind that child and the actual parent device/inode. Hold the
actual intent file's native exclusive lock through creation, containing-entry
synchronization, removal and removal synchronization. Successful creation cannot
forget its intent before its required barrier succeeds.

A retry checks only the exact intent names for its own lexical ancestor chain,
including an already existing chain. Under the actual file lock it validates the
private regular record, named file binding, strict contents and current pinned
parent identity before reestablishing that entry barrier. Existing resulting
children retain their current owner/no-follow/private-root checks; no directory
is deleted, replaced or permission-repaired. Live creators, incomplete/corrupt,
foreign, moved or substituted records and changed parents refuse. Missing child
names may be created only from the caller-derived local chain under this same
finite intent boundary. No serialized locator grants authority, and unchanged
ancestors without an exact creation intent receive no durability barrier.


### TASK-34561 amendment: finite fresh paired-witness metadata observation (2026-10-04)
An installed Windows MCP permission read repeated current-generation selection eighteen times. Its nineteen witness reads each independently reread control records three times and registry twice. The measured 3,003 native opens are governed read work, not provider latency.

Each witness query may read full validated control records and registry once and share those values among its existing source-scope, startup and paired-generation decisions. The observation belongs only to that finite call and its actual admitted lease execution context. Public startup/source-scope readers continue to perform fresh native reads. No mapping, absence, selected path or empty witness is retained for a later call, callback or turn.

The reader must preserve all record-schema, unknown-record, pending-intent, cross-profile overlap, historical-inode, held-namespace, paired-generation and required-owner checks. Fresh named metadata at observation completion must still match the physical identities and change stamps observed while reading; a pending record inserted during the read or a changed/missing/replaced record refuses publication. This fence retains native no-reparse/private-owner validation and applies on Windows, Linux and macOS. Native leases, source guards and caller-method identities are unchanged.

This is an ADR amendment because metadata sharing changes the recovery reader's observation boundary. Retaining repeated full reads was rejected after actual native attribution proved their cost; cross-call witness caching and omission of empty-generation checks remain rejected because later recovery evidence must be visible immediately.

### TASK-34561 amendment: admitted MCP generation observations retain exact custody

A real seeded permission read-modify-write starts before local pause and must finish under its existing native custody. Its repeated bound-source checks still read fresh paired-generation evidence. They must not request a new ordinary storage acquisition after pause, nor borrow a selected-generation lease for a different canonical path. An installed MCP raw operation retains an exact canonical observation lease before acceptance when canonical and selected paths differ; otherwise its already admitted exact canonical lease serves that observation. Only the same source's active issued operation, current process/thread/task, installed participant mapping and positively live native leases may supply this retained observation. A foreign source, child task, sibling path, retired operation or unqualified native hold cannot inherit it.

The retained lease changes no execution selection or path authority. Each observation still checks its exact canonical execution context and reads fresh validated generation evidence, with no cross-call witness, absence or permission cache. Bound source and native parent proof run outside the coordinator lock; pure exact identity, active ownership, closure and pause predicates are rechecked under the lock before accepting an operation or changing participant state. The corrected native pause test seeds actual policy bytes, exposing a pre-existing no-op test gap and the prior attempt to reacquire storage during accepted work.

Participant closure, drain and resume are coordinator bookkeeping over exact installed source identities and counted native work. They do not obtain fresh file authority, create an observation acquisition or validate generation metadata while a pause is active. Closure may run on the maintenance thread while an accepted source operation retains its own canonical observer. Drain tests exact counters and persistence readiness; resume remains refused during process pause. Every actual source admission/use retains its fresh out-of-lock bound-source and native parent proof, so opening the bookkeeping gate cannot authorize a stale or retargeted source.


The TASK-34561 finite-observation completion fence must verify native containing ancestors as well as the selected control entries. On Windows, retain all ancestor receipts from the same fresh bottom-up named tree and apply the existing private-parent owner and writable/sticky policy; observing but discarding those receipts is insufficient. On POSIX, hold completion pins live while reopening descendants before ancestors through the verified native path walk and compare the actual named physical identities to their held identities. A containing ancestor may be renamed without changing descendant inode metadata, so held-descriptor fstat alone is not a named-binding proof. These checks retain the existing platform trust rules and add no authority/cache boundary.


### Checked sparse Console display policy sharing (TASK-34561 AC9)

A screen-owned readiness read may capture the immutable sparse global Console policy
from the same actual CLI bootstrap mapping while its existing native config operation
is live, between the original before/after checked source tags. Its context-estimate
presentation reader may share only the successfully published mapping's exact source,
generation and full profile/session/workspace/settings owner, with the existing
one-second refresh and cancellation/retirement rules. Cold explicit modal display
warming waits for that checked read, without opening a second config lifetime.

The shared argument belongs only to the named context-control presentation method.
It is never a lease, permission decision, source proof for dispatch, or substitute for
live context-control actions or provider preparation. Those paths continue independent
fresh global-policy reads. Context cache identity also includes the published sparse
policy so a fresh changed mapping invalidates estimates; live owner identity remains
separate so first publication does not masquerade as a changed modal owner. Reject
pending reads after source/generation/profile/session/workspace/settings changes.


### TASK-34561 amendment: finite MCP owner classification and nested selection

Immutable owner classification may use the actual installed class/kind mapping, checked against its existing bound source type, without re-reading generation evidence merely to choose that owner's fixed member names. It grants no file, payload or permission authority. Source admission retains fresh initial and final bound-source proof, native custody, exact path membership and producer identity checks. An exact same-source nested MCP raw scope may name its selected path from the full raw check that just returned in that synchronous call; its selected_read/path/writing checks and the subsequent full raw check remain in force. No witness value or selected path is cached across calls, workers, awaits or operations, and every raw check still observes fresh source/generation/native state.

Creation evidence completeness: _ensure consumes exact ancestor creation intents before ordinary admission. The cache now stamps those exact paths absent; present markers force the unchanged full initializer, preserving its refusal or completed durable retry. Genuine Windows warm/full verdict divergence RED2 precedes this change; final root bundle32passed2 ordinary capability skips13.29s includes both new controls and the original creation/admission controls. No marker contents are parsed or trusted by optional evidence, no security/durability check is bypassed. Native macOS completeness qualification remains pending.


### Bounded body-free Console display observations (TASK-34561)

Ordinary subagent badge counts and Character browser scope presentation may retain their successful process-local observation for at most two seconds. This display-only cadence follows the existing subagent count TTL and reduces actual finite callbacks rather than weakening native checks inside them. Counts remain body-free and use one exact captured AgentRunsDB receiver. A bridge or receiver/library change, visible row set, run target or live child identity change invalidates immediately; a pending read cannot redirect to a replacement receiver or publish under its owner.

The separately named Character presentation facade belongs only to general screen synchronization. Its owner includes config generation/selected source, app/config mapping, exact database, controller generation, active store/session/workspace/conversation binding and current character/open conversation. A queued waiter recomputes this owner after acquiring its async lock. Each fresh observation still performs both metadata pairs and its loop-side midpoint. Publication from a display-triggered refresh checks the captured ambient owner after awaited native work. Stable same-owner successful observations alone may be memoized; failure, cancellation or changed generation/owner cannot establish freshness. External database changes become visible within the bounded display interval.

The memo grants no file, permission or dispatch authority. Existing direct refresh, action scope capture, final commit and midpoint validation always run their fresh native checks. A changed live run/child/owner is not delayed by the TTL. Cross-callback native/permission caching and retaining the 0.2-second active badge polling were rejected: the former loses fresh authority, while measured actual display-only callbacks make the latter a major preparation cost. Existing whole-send and heartbeat ratchets remain unchanged.


### TASK-34561 amendment: finite external MCP catalog worker and loop projection

The exact standard UnifiedMCPControlPlaneService / LocalMCPControlService /
LocalMCPStore catalog route reads one fresh catalog bundle in a finite worker.
The bundle includes profile definitions, discovery snapshots and runtime state;
no permission or checked-source observation survives for another call. Native
store admission remains independent on the worker with its original source and
path checks. Governance and the actual caller-loop client/plugin connection
projection remain on that loop. Custom services and subclasses keep their
existing service route and behavior.

Capture the exact local service and store before starting the worker and reject
replacement before publication. Recheck governance before projecting the result.
The existing async producer interval stays live until the already-started worker
settles, including on repeated awaiter cancellation. Cancellation is propagated
only after the worker's native scope has retired; worker errors during retirement
do not replace that cancellation. No coroutine is moved to another event loop,
no new maintenance or native authority is borrowed from the caller, and no idle
worker resources or permission cache are introduced.

The prior asynchronous method performed two complete blocking store loads on
its caller loop. Offloading the entire local-service call was rejected because
its governance and live connection observations belong to that loop. One finite
blocking bundle read preserves the current store contract while removing both
that blocking loop work and the duplicate load. Cancellation that immediately
released the producer while its native worker remained live was rejected as an
incomplete retirement boundary.


### TASK-34405 amendment: repository issued state and fresh native path proof

The shared storage coordinator synchronizes exact issued operation membership, installed repository identity, captured path scope, live lease and accepted native hold. It must not remain held while a repository boundary performs fresh parent observation or path resolution. Each full source check validates its pure issued state under the coordinator, performs the unchanged fresh native observation outside it, and validates the exact captured state again before returning. Callers that reuse, restore or publish operation discovery perform a final pure fence inside their short coordinator interval. No observed path or permission value survives into a later call.

Already counted same-owner work may complete during pause while its exact lease/hold remains live. New owners and independent repositories still require ordinary admission and refuse after pause; removed operations, leases or installed participants, changed source/path and unqualified native holds refuse even when native observation started earlier. Pure source checks do not grant native or maintenance authority. Moving guards into a worker while retaining the shared coordinator and temporarily unlocking callers through private RLock methods were rejected: the former still blocks the UI and the latter obscures producer ownership and final state races.


### Exact live-empty fleet display precedence (TASK-34561)

The fleet rail shares the existing summary live-over-historical contract: an exact real bridge and AgentRunsDB may render its process-owned setup marker or its bound running primary snapshot even when its children are empty. Nonempty fleet handles remain first, including survivors from a preceding turn; inline live summaries render from the same current snapshot. A live setup marker deliberately suppresses the previous terminal turn, as the bridge already documents. Unknown or restored bridges lacking that process-local ownership, terminal runs and custom bridge fallbacks continue fresh bounded durable history projection.

This is display source selection, not storage admission or a permission verdict. It removes repeated native historical queries and prevents previous durable children from being attributed to a known current empty run. Skipping history for arbitrary non-idle/custom snapshots or deleting terminal history was rejected because those states do not establish this live source contract.


### TASK-34561 amendment: fresh async Console definition maximum preparation

Actual asynchronous Console submission and continuation paths may prepare their
MCP definition maximum through a separately named asynchronous capture entry.
Only exact standard service/local/store sources qualify for the worker split.
The blocking worker reads the real permission payload once and local catalog
bundle once, with each source retaining independent original raw admission and
fresh source checks. Missing/refused/changed source evidence freezes an
empty maximum. A global kill switch freezes an empty maximum. Default permission
profile resolution, definition hashing, builtin raw-name exclusions and the
existing changed-definition marker/audit semantics remain in force. Any blocking
audit work retains the same finite producer and worker retirement boundary.

Governance, actual builtin inventory and UI-owned snapshot composition remain on
the original caller loop. Exact standard default and screen builders accept the
explicit detached maximum only for that same capture. No checked payload, maximum
or permission verdict is retained for another request. Preserve true synchronous
entry points and injected/custom provider contracts through their previous route.
The maximum is a frozen narrowing ceiling; existing fresh invocation and final
dispatch permission gates remain required.

Capture the exact controller/app/store/session, owning workspace and settings
revision, provider callback/config ownership, configuration module/generation and
lexical selection, and MCP service/local/source receivers before awaiting work.
Publication rechecks those owners and the actual raw source bindings observed by
the worker. An await cannot redirect accepted work to a replacement receiver or
turn actor. Refuse changed actor ownership and discard changed or uncertain source
results. Cancellation retains the original counted producer until all started
worker/native scopes retire, including repeated cancellation and worker errors.

Native Windows measurement of the previous capture gives three complete payload
loads, 5,428 opens and 0.655 seconds unprofiled. A diagnostic two-payload join
preserves hashes with 3,525 opens and 0.424 seconds; both exceed the repository
worker threshold. A synchronous join alone was therefore rejected as the UI
repair. Offloading the full snapshot or creating another event loop was rejected
because UI composition and actual inventory ownership belong to the existing
loop. Silently changing synchronous/custom callbacks to an async contract or
using a cross-request permission cache was also rejected.


Display-owner completion clarification: exact session identity and original config mapping references are retained alongside their identity values, preventing field-equal session replacement or identity reuse from inheriting a memo. Readiness default convergence may alter only its own settings revision slot; every source/app/session/workspace/owner field remains compared. Repeated cancellation cannot settle the checked-read completion event or release display coalescing before its finite native worker retires. Custom bridge count callbacks preserve their declared behavior; only the exact standard bridge uses the captured direct AgentRunsDB query. None of these observations supplies action or dispatch authority.


TASK-34405 Settings writer completion clarification (2026-10-04): the exact standard ChatPersistenceService generation-settings and same-DB ConsoleContextRepository policy writers are finite Notes callbacks. Capture the original bound writer, persistence adapter and exact file-backed Notes receiver before scheduling; retain one counted repository interval through the complete synchronous CAS callback, then retire only its newly created worker handle after interval exit. Recheck captured source bindings before/after native work and before accepting results. The settings drain retains its native worker through repeated cancellation until actual retirement; existing caller-owned handles, custom/subclass/memory routes, reconciliation and session publication fences remain intact. A global lease clear or a fixture that bypasses the actual persistence writer was rejected because it would hide an ordinary application handle lifetime defect. No cache, cross-source authority or new maintenance capability is introduced.

Async maximum preparation retains the original permission reader's corruption
recovery (backup plus default ask), rather than treating that recovered payload
as a persisted deny or bypassing its native source checks. Downgrade marking still
precedes best-effort audit-log creation. A private captured lazy log receiver
resolves against the admitted local path, checks the exact service/store/cache
owner before pure cache publication, and cannot redirect an audit through a
replacement service. Existing synchronous audit callers keep their original
lazy receiver contract. The catalog worker qualification is specifically the
exact installed local service/store, so an inherited Unified facade retains its
own override dispatch; the separate async maximum route qualifies its standard
Unified reader contract and preserves custom/subclass synchronous routes.


### Rejected raw-parent observation grouping (TASK-34404)

The native permission scope has one named parent. Applying a full tree snapshot increased actual opens1903 to1973 and security reads924 to987, retaining7full checks,14fresh witnesses and13source selections. The prospective1100 bound failed. Keep the original fresh named stat and pin comparison; no grouped raw-parent helper or cross-call observation is introduced. This confirms the earlier isolated6-to12-open observation recorded in the implementation plan.

Settings source capture includes the actual writer body: private expected receivers supplied only by the qualified async route reject body-entry identity drift and anchor every standard generation CAS read/update and context policy transaction to the strong captured Notes receiver. The captured original same-DB policy writer cannot redirect through a replaced repository. Before/after equality alone was rejected after an actual A-to-B-to-A native control wrote B while publishing success for A. Default synchronous/custom calls keep their existing arguments; these private expected receivers grant no admission or permission authority.


### Exact checked warm Console display scope (TASK-34561 AC18)

A successfully published standard readiness mapping may render synchronously
within its original one-second age without opening another enclosing native
configuration lifetime. The actual standard finite reader alone issues the
detached display proof after both existing checked source tags match. Its exact
module, installed raw participant/state, factory reader/loader, mapping and full
owner/source key plus original UI loop/thread remain retained and compared.
Issuance, exact active projection/mapping, installed source identity, closure and
local pause, current source/owner, loop/thread and age are checked before and
after the synchronous body. Injected, custom, replaced, cold, expired or changed
state keeps the original native route or existing cold deferral. Failure remains
visible, including original body and cleanup error precedence.

This proof carries no raw operation, native lease, path authority or lock. Every
actual nested guarded disk reader or writer still enters its original fresh
scope; live actions and dispatch do not consume this rendering proof. The
continuous admitted config lifetime described above remains the default for
genuine native work and unqualified callbacks. Maintenance/replay ownership and
the projection worker cancellation/retirement contract remain unchanged.

The distinction addresses measured redundant enclosing display admissions,
without changing the original one-second age or whole-send/heartbeat budgets.
Unconditional wrapper removal, injected flags, retired-operation borrowing and
permission/native caches would lose those boundaries and are rejected.

Startup initializing publication clarification (TASK-34404 AC8): an exact pause-owned startup readmission intentionally has no repository operation while its coordinator pause remains active. After the outside-lock check, the publication lock retains actual issued pending membership and PID/Thread/Task/operation identity plus cancellation. Only the exact _StartupReacquisition with path=None and operation=None rechecks its original class metadata check under that lock. This preserves the current pause, startup worker, retired source selection and original native gate contract; ordinary callers, subclasses, unissued copies and stale/canceled actors cannot claim this route. A blanket operation=None pause refusal was rejected after all three original mounted native controls failed at that new fence despite matching actual custody. No file path, dispatch permission or new maintenance capability is conferred.


AC16 exact asynchronous source completion clarification: standard finite worker
qualification requires real bound methods with the original installed function
and the exact captured service/local/store/permission/log receiver. A borrowed
method from another actual receiver retains the previous custom route. Exact
registry binding objects, governance/manifest owners, arbitrary callback objects
and the actual app configuration mapping remain strong references across each
await; bound callback comparison names its function and receiver. One ephemeral
captured-source keeper spans standard catalog composition's separate kill,
external catalog and effective-state reads, retaining its original async catalog
callback. Lazy permission/log publication updates only its own new retained slot
and immediately rechecks all prior owners. This ledger grants no native source
or permission authority and expires at composition completion. Standard workers
require original permission marker and log append callbacks; overrides keep
synchronous caller-loop compatibility.

The actual UI async boundary also retains attachment IDs alongside strong
attachment objects before waiting. Any in-place ID mutation refuses custody
while preserving the draft. The same screen/controller/store/session/runtime,
settings, draft/stash/prefill/evidence and configuration ownership is rechecked
before the original synchronous runtime acceptance, without another await.

AC16 nested source contract clarification: callable provenance is captured at
its defining module/class completion, including the public permission-store and
execution-log descriptors, permission kill getter, local external-catalog and
projection callbacks, and the local store load used by the joined reader. Exact
original bound receiver and descriptor identities are checked before selecting a
standard finite worker and retained across its await. Lazy absent permission/log
owners require the same original class contracts before construction; publication
updates only their newly owned callback slots and cannot accept unrelated drift.
Custom getters, descriptors, projections and load callbacks keep the preceding
consumer contract, including the existing provider worker route and synchronous
maximum/local routes.

The private standard bundle input retains both the original guarded bundle and
its exact original bound guarded load. It grants no path or permission authority;
both methods still enter their existing fresh native guards. Proxy, foreign or
subclass receiver inputs refuse, while default public/custom calls continue to
resolve load dynamically. A post-publication check alone was rejected because an
already admitted bundle body can resolve a replaced nested callback before that
check. The finite catalog worker retains exact local/store source bindings and
projection/governance owners through retirement; inherited Unified facade and
custom catalog dispatch remain compatible. No native scope, read or downgrade
RMW is cached or skipped.

The standard provider composition chooser also requires the original real bound
Unified async catalog callback, retained at its defining module completion. An
explicit custom catalog callback retains its preceding late inventory-receiver
lookup and worker contract. This extra qualification belongs only to catalog
composition; synchronous maximum/local preparation does not depend on that
unrelated async callback. It neither accepts changed standard-source publication
nor constrains legitimate custom callback receiver management.

Only provider composition captures the async catalog callback in its ephemeral
ledger. Its original class descriptor is checked before a dynamic callback
lookup across any finite worker await. Maximum and local preparation do not
dereference the unrelated async callback, preserving custom descriptor contracts
on those routes. Returning an equal original bound callback from a changed class
property does not satisfy standard provider publication provenance.

TASK-34404 AC9 fresh finite visual metadata clarification: grouping an original
fixed set of selected-file and carrying-parent observations may use the existing
Windows admission snapshot only when its defining original class/function and
actual native facade/receiver bindings remain exact. Each observation is fresh,
checks every original identity and retires its native handles before acceptance
or complete scalar fallback. It grants no source, path, lease or permission
authority and stores no reusable metadata. The existing 4,096-byte descriptor
cap and custom/POSIX scalar contracts remain unchanged.

Physical snapshot close uncertainty requires native proof before implementation.
The planned control protects a real issued metadata handle from CloseHandle
without replacing guarded functions. If that proves an unretired handle, the
existing visual state's raw/capture exclusion must retain its exact handle and
uncertainty before any fallback or attempt retirement; an error alone is not
proof of physical close. The metadata optimization and any checked-close repair
remain pending this evidence. Immutable Samira resources and two-commit seed
semantics remain governed by ADR-067; no deferral or deadline increase applies.

AC9 original native qualification is complete before implementation:
samira-observation-red.xml has three genuine failures (17,053 opens against
12,000, actual source replacement accepted and actual Windows pause accepted),
five original positive/path/custom/large-DACL passes, and one explicitly
unqualified missing integrated snapshot boundary. The independent original
snapshot control, samira-observation-close-red.xml, proves that it returned
metadata while the same protected physical HANDLE and file identity remained
live. Fixture-only flag removal and positive close were verified; guarded
functions, eight source hashes and original deadlines stayed unchanged.

The approved checked-close repair tracks each actual opened handle incarnation
with its captured native owner and fresh identity. Each close is attempted once.
All remaining handles are retired even on body/ENOTSUP errors; any uncertain
close raises the specific defining exception carrying failed incarnations.
The actual issued visual state retains these handles and uncertainty in its
existing raw/source exclusion before fallback or scope retirement. It never
retries a failed close. Ordinary unavailable metadata can fall back only after
positive retirement. This refines existing ADR-126 finite custody; no admission
cap, other WindowsOS operation, seed semantics or deadline changes apply.

AC9 review refinements, native proof before implementation: the actual held
standard snapshot accepted a retargeted stat binding, made 59 replacement
callback calls and returned original bytes (binding-red, one genuine failure).
After initial stock qualification, recheck its exact bindings before every
scalar fallback, including successful, ENOTSUP/OSError and ValueError exits;
mid-read drift refuses without invoking replacements. Preinstalled custom
facades retain the original scalar contract.

The admitted metadata control uses a real enrolled native profile and installed
_observe_stamps inside the issued visual source. Its protected physical HANDLE
survived while bytes were returned and raw/source state retired (one genuine
failure, 4.91 seconds). Ordinary metadata errors keep prior evidence fallback;
only the exact defining close-uncertainty exception passes through selector,
path, reused evidence, candidate observation and public acquisition wrappers.
The actual visual state retains its failed incarnations before scope retirement.

The original opener's real junction/reparse refusal also leaves its protected
pre-return HANDLE alive without reporting close uncertainty. Check only that
validation-failure close; retain the exact native owner/handle (identity may be
unknown if info failed), preserve the original validation error on successful
close, and transfer failed incarnations through the snapshot's existing ledger.
Fixture-owned physical cleanup was positively verified; all guarded functions,
source hashes and deadlines stayed unchanged. Prior authority-path and absent-
handle observer errors were setup gaps, never authority RED. Existing ADR-126
applies; no other native close site, admission cap or deadline changes apply.



### AC9 scalar close uncertainty refinement (2026-10-04)

ADR required: no new ADR. ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md. Reason: preserve the existing finite native resource retirement boundary for the original custom scalar route.

The actual custom-facade scalar reader delegated to the original Windows rejected-junction opener. Its genuine protected HANDLE remained alive after the defining metadata-close exception, but the issued visual state/raw record was removed with uncertainty false (one native RED, 3.77 seconds; guards/network zero, captured source hashes unchanged, physical fixture cleanup verified). Before implementation, scope the repair to catching that exact defining close exception around the existing scalar observation and passing its actual opened-handle records to the existing uncertainty retainer. Ordinary custom/POSIX errors and successful custom/native/large-DACL fallback contracts retain their current behavior. Verify the new native negative with the existing custom positive, standard rejected-open, large-DACL, and unsupported/body-error controls; preserve the original startup 20/45 second bounds.

### TASK-34404 amendment: scoped run-log source admission (2026-10-04)

The original Console bridge already selects a run's explicit workspace/scratch
root and supplies a RunLogWriter with a real scratch access scope. The installed
AgentService guard must freshly admit that same source under agents.history,
before its first run/history mutation, rather than discover an unrelated global
workspace/sandbox root a second time. Existing family source observations and
the original recovery/native execution gates remain in place.

Only the exact installed service, definition-time original run_turn and writer
class/methods, original root selector, and real bound receivers qualify. Capture
the injected writer, immutable Path object, access-scope and publisher objects
before admission. A finite current PID/Thread/Task metadata record retains these
strong owners until the original turn exits; it carries no lease, permission,
resolved sensitive context or reusable verdict. Fresh current-root admission
is independently required. Custom/subclass/borrowed/overridden producers retain
the complete original selection and call contract.

Recheck the captured producer/source before the original turn's first effect,
before and after access-scope entry, and before binding/publication. Bind consumes
the captured root; it never recaptures a changed mutable field as authority. A
changed writer/root/callback/module/actor refuses or disables logging at the
existing boundary. Publication checks occur outside the optional publisher's
exception-swallowing block. The record resets in finally and never transfers to
another run or surviving worker. Scratch authority, native storage containment,
sensitive-path checks, hidden .agent-runs, legacy migration/no-overwrite and
physical retirement remain governed by their existing original checks.

Meaningful native RED evidence: an actual private AgentRunsDB, exact service and
writer, real scratch snapshots/leases and scripted provider reproduced one
unrelated global selector and no actual-root extra admission; actual scope entry
retargeted the root/publisher and was allowed to bind. Three expected failures
and one unchanged custom-selector pass completed in 28.65s, source hashes stable,
without guarded-callable replacement. Counter reductions are mechanism evidence;
the original whole-Send/UI limits and three-platform verification remain required.

The finite record has an issued live bit cleared on exit, so copied contexts cannot use retired source metadata. Exact stock partial callback bindings retain function, positional tuple and keyword-owner objects; in-place keyword retargeting refuses. Invocation consumes captured binding values, with no arbitrary callable introspection. An actual in-place partial publisher retarget reproduced another native RED in8.23s with original source unchanged. Unsupported callable objects or unbounded partial bindings preserve the original custom route.


Scoped dispatch amendment: native late-return metadata faults reproduced three
failures in 17.39s after actual DB creation, source checking and containment.
The qualified route invokes retained original bind, under-bind and migration
functions with the checked receiver, including a first-effect migration fence.
The default/custom route retains its dynamic calls. Writer selection consumes
the captured writer after another service fence. Source owner fields are frozen;
only the issued live bit retires in finally.

A copied-context native worker reproduced writer deactivation in 8.38s.
Foreign PID/Thread/Task metadata is inapplicable and supplies neither roots nor
admission: original worker, scratch and storage guards reacquire independently.
Same-actor retired metadata refuses, and explicit scope entry always checks the
full actor/live fence. Original child/survivor append and close remain unchanged.

### TASK-34404 AC10 cold raw-sensitive-input finite config operation (2026-10-04)

ADR required: yes, clarification of this existing interface. A new synchronous reader grouping must preserve finite native source, actor and publication custody; it supplies no permission or maintenance capability.

Two source-passive native controls ran serially in separate fresh private processes. The original stock cold context made 7,176 Windows Native.open_handle calls; enclosing the same original resolver in the supported `config_participants.operation(config)` made 2,627. All 20 original user-directory bodies ran, fresh raw checks rose from 40 to 123, the normalized exact derived pathset matched, nine production hashes/protected callbacks stayed unchanged, the seeded SQLite handle physically closed, and all ordinary/pending/core/raw/retiring resources reached zero. Warm grouping increased opens from 356 to 377. These finite hypothesis receipts are not whole Send timings, a whole-HEAD baseline or cross-host performance completion.

Only an actual natural `_raw_inputs` memo miss may add the finite operation. The initial original key read and warm memo return remain outside it. Every original getter, native source/path check, additive failure counter, whole environment/cwd and strong cache-object/generation/source key stays intact. No public accessor or resolved-path/permission algorithm is replaced. The original resolved SensitivePathContext remains invocation-local; unresolved raw tuples retain their existing memo contract.

Exact stock qualification is refusal-only metadata. Retain direct original functions/accessors/helpers at their defining module completion before helper lazy import, require installed module/globals and concrete function identity, keep strong callback/source/actor owners, and compare by `is`. A config guard wrapper is owned by its actual defining `config_participants` globals; copied metadata does not turn those globals into config. The original wrapper continues checking its original config body and installed source. Preinstalled custom/bound/borrowed/proxy and remote/stub paths retain their preceding ungrouped zero-argument callback route. Standard callbacks/source/pause drift after qualification refuses without invoking replacement callbacks under the added scope or silently retrying custom fallback. All getters still run their own guards; admitted source identity is freshly checked at both operation edges.

The sensitive module owns its memo, not config rollback. Evaluate the original final whole-key/failure comparison while the finite operation remains active, retain eligibility locally, and defer publication until the finite operation has exited with positively retired resources and final source/callback fences hold. A failed entry, body or final custody check publishes nothing from that build and must not remove an independently published successor. Ordinary accessor failures and environment/cwd drift retain their additive return and non-memoization semantics. A private optional refusal checkpoint before each original DB accessor may protect dynamic callback lookup without altering preceding no-argument/custom call shapes.

Native TDD must precede production: stock cold ≤5,000 actual opens with all 20 getters and literal protected paths; unchanged warm one-getter/two-check hit; original custom/first-import/bound/proxy/helper routes; mid-read callback/source/pause refusal; failed final exit leaving no memo; environment/cwd and failure non-memoization; actual finite native retirement and original POSIX/stdlib worker contracts. All integrated Send/UI/open/helper budgets and deadlines remain unchanged.

Cold-input wrapper contract clarification (2026-10-05): the first candidate
failed actual config import before test collection because the original cached
selector has no function globals (exit 4, 4.687s, unchanged source). Definition-
time metadata must retain this concrete original lru wrapper and its original
wrapped function/code/globals. Qualification compares those exact objects;
wrapper invocation and cache behavior remain original. Arbitrary unwrap results
never become authority; unsupported custom objects use the preceding route,
and drift after stock qualification refuses. Ordinary function-body mutation
coverage remains pending its separate native RED, with no completion claim.

Ordinary reader body clarification (2026-10-05): two real native code-only
mutations retained the original function/global identity but wrongly entered or
completed the cold stock scope (2 failures, 8.04s; seeded SQLite physically
closed, native/hash/guard/body/hook cleanup passed). Definition-time ordinary
reader and direct-owner metadata must also retain exact code objects. Pre-first-
helper-import body changes preserve the original ungrouped custom route;
post-qualification changes refuse before invocation or memo publication. This
completes the existing refusal-only metadata contract without unwrapping current
callbacks, changing original getters or accepting a cached permission verdict.
Original key, failure, final-custody/publication and deadline semantics remain.

### Cold tuple and pause publication refinement (2026-10-05)

The stock cold input grouping must retain the defining original DB-name tuple
as well as the original reader functions/codes. Two real native controls failed
after a preinstalled or actual mid-build tuple addition (source-drift bundle
17.39s total, unchanged source, exact native/issued boundary, original guards and
zero finite census). A custom name tuple takes the preceding ungrouped route;
tuple drift after qualification refuses before added reader invocation or memo
publication. Per-lookup checkpoint membership is the captured original set, not
the just-read name/callback pair. Original additive custom/no-argument behavior
and every original getter remain.

Both original pause/final-pause controls reached their declared barriers in the
17-case candidate (15 PASS / 2 FAIL, 126.60s), but published the new memo. An
installed native config operation may legitimately retire after local pause;
that existing protocol is not changed. This leaf must separately check current
publication availability. Retain invocation-local refusal metadata for the actual
admitted participant/state, original weak source reference/selection and actual
coordinator lock. Exact definition-time private helper functions/codes capture
and recheck this metadata. After physical operation retirement, the final local/
participant pause and same installed participant check shares the coordinator
interval with the unresolved memo write. Full source/path observations stay
outside the lock; in-lock checks repeat only pure actor/binding/cache/key fields.
This record holds no operation/lease/verdict and grants no permission. Failure
publishes nothing and does not erase an independently completed successor.

The previous scope/guards, raw source/recovery protocol, custom/remote contracts,
natural warm hit, cold <=5000 guard, original integrated budgets/deadlines and all
native refusal/publication assertions remain unchanged. Bounded native GREEN and
final cross-host/Console/UAT evidence are still required.


Scoped run-log body binding clarification (TASK-34404 AC10, 2026-10-05): the
revised actual native two-case RED executed both same-identity changed bodies
after qualification (two failures, 11.55 seconds; original provider completed,
physical SQLite close and final zero native census, guarded callbacks/source
unchanged). Earlier scoped fixture precondition failures are not product RED.
Before implementation, retain definition-time exact stock service wrapper/body,
writer method and selector code/globals as well as their function/module owners.
Finite supported function/MethodType/partial callbacks retain code/globals and
strong bound receivers. Recognize the actual contextmanager factory only by
original stdlib helper code/globals and its one exact wrapped-function closure;
retain the wrapped body without arbitrary unwrap or recursive authority.
Preinstalled changed stock bodies preserve fallback; issued-source body drift
refuses. Check at actual captured dispatch after the caller's original check
return, and retain publication refusal outside optional observer handling.
Actual fresh root admission, native retirement, custom/default/foreign-worker/
survivor contracts and existing budgets/deadlines remain. This is refusal-only
finite metadata, not permission or a resolved sensitive-context cache.


### TASK-34561 AC23 finite stock MCP composition precheck (2026-10-05)

ADR required: yes; this extends the existing finite source-custody interface.
The ordinary controller and exact stock provider currently read the same kill
switch at composition entry. The provider must still perform its original fresh
native read before catalog composition and its separate fresh effective
permission read afterward. Only the unchanged original factory, constructor,
composition methods, projection methods, service callback and native snapshot
pipeline may omit the preceding duplicate controller read. Customized classes,
factories, callbacks and preinstalled changed bodies retain the original
controller precheck and no-argument compose call.

One issued composition record retains the actual original factory, service and
captured native source owners. It carries callback provenance, no permission
verdict. Exact defining code/globals/defaults/keyword-defaults/closures and
receiver/source identity must remain current before construction, inside the
provider's existing producer lifetime, and after every composition await.
Drift after qualification refuses without invoking a replacement under this
route. Tool invocation retains its independent fresh kill/permission checks.
Global worker guards, physical retirement, deadlines and integrated budgets
remain unchanged.

The original native nine-case RED was 2 expected failures and 7 unchanged
custom/source-control passes. The first candidate is a deliberate verification
checkpoint: a newly identified same-function service-body gap must reproduce
RED and be repaired before this route is described as qualified. Constructor
and await retarget controls, original custom signatures, fresh killed/unkilled
composition and native physical retirement are required evidence.

AC23 callable provenance refinement before production: the omitted service getter and permission property, and the two known defining permission guard bodies, must retain definition-time code/globals/default/keyword-default-item/closure-cell identity. Constructor owned-profile defaults are authority-bearing inputs even when callable object/code identities do not change. Changed or unsupported customized wrappers retain the original controller precheck; a body change after actual Native worker entry refuses publication. Qualify new real-source Native controls before repairing the optimization. This captures no permission verdict and preserves generic worker guards, original budgets and fresh producer reads.

### Qualified MCP composition inner-await fences

The same issued finite controller composition token must validate full captured callable metadata at each existing stock catalog post-await boundary, before the next source read. Eventual refusal after the whole catalog returns is insufficient. The exact stock private route receives the same token and captured-source keeper; custom and ordinary no-argument composition keep their declared prechecks. An inventory exception handler must not swallow qualified metadata refusal. This adds no authority verdict cache and keeps each actual checked native read.


### Fresh storage admission proof outside the coordinator (TASK-34406, 2026-10-05)

Three genuine Windows causal regressions hold the unchanged original native open during the first scope read, final scope read, and nested operation path check. Each positively observes the read actor own the shared coordinator, a different real Thread unable to acquire it without blocking, and that Thread's original independent lease close delayed until the read resumes. Native source and guard identities stay current; all seeded SQLite objects, native holds, acquisitions and operations physically retire before the expected assertion fails. No whole-Send timing saving is inferred from this held control.

Fresh records, registry, binding, fingerprint, containment, startup-permission and installed operation-path observations must execute outside the shared coordinator. Pure metadata validation/publication remains synchronized. Before each observation retain the exact pending acquisition, actor, original installed operation/lease/path/hold and selected root/config/path; after it recheck those same owners and accepted pause/cancellation semantics. A checked per-call scope record captures the existing hold and original startup readmission metadata. If the live-hold continuation branch supplies authority, publication must recheck that exact hold still exists with its original names and has not begun retirement. An independently valid saved binding does not depend on an unrelated holder remaining live. The record grants no permission and never crosses another call or actor.

Acquisitions and leases remain counted before the final fresh scope observation. Final revalidation repeats fresh startup permission and scope, followed by pure synchronized current-owner fences. Evidence reuse keeps its original per-call native observation, metadata epoch and same hold/evidence/path-entry identities; native operation checks move outside its coordinator intervals with exact postproof metadata checks. Source/selection drift, pause/cancel, retired continuation owner, changed operation/path/lease or evidence owner refuses and retires only the exact counted token. Custom/unqualified and pause-owned startup readmission behavior stays intact. No namespace widening, permission cache, skipped native proof, changed guard, timeout or performance ceiling is permitted.


### Finite inner Character display refresh (TASK-34406, 2026-10-05)

The actual original-source Windows work-count control completes the original
three paired metadata readers, recent groups/details, native connection retirement
and private cleanup, then fails because a changed display uses four owned DB
callbacks instead of two. A pre-existing same-worker connection reproduces the
same duplication while remaining borrowed. Fourteen owner-drift, custom-source
and repeated-cancellation controls pass on the original source. This is causal
leaf evidence; existing Send/startup/helper limits remain pending.

Keep the initial change-detection callback physically retired before the original
REFRESHING publication. Only the exact stock file-backed Character display route
may combine its inner initial pair, bounded recent groups/details and final pair
in one existing owned DB callback. Every original paired metadata read and its
loop-side midpoint remain; the original post-pair ambient check remains on the
loop. No GUI publication occurs inside that callback. After physical callback
retirement recheck source bodies, exact receiver, ambient scope, generation and
the captured presentation owner before publishing. Ordinary groups errors still
require the final pair; metadata errors retain their distinct recovery outcome.
Repeated cancellation drains the same callback before releasing its coalescing
lock. Pre-existing worker connections retain original borrowed ownership.

Definition-time original method/service metadata qualifies only that finite read
seam. Preinstalled custom service, callback, subclass, memory or foreign receiver
uses its preceding route. Source/body drift after qualification refuses before
replacement invocation or publication. Direct actions keep their original fresh
reads. No persistent worker, permission verdict, result cache, widened root,
changed guard or relaxed time/helper/open budget is introduced. Actual native
GREEN, refusal/error controls and original whole/platform evidence are required
before accepting the optimization.


### Fresh Windows binding ancestry observation (TASK-34406, 2026-10-05)

Two actual unchanged binding calls observe the same enrolled twelve-root tree
and retire their original native handles, then fail the work-count control at
152 opens each against structural bound63 for24 resolved nodes. The earlier
wrong-wrapper setup and synthetic wrong-parent-policy controls are excluded.
This justifies grouping fresh metadata observations within one binding call;
it supplies no cached permission or cross-call observation.

Only the original defining WindowsOS tree-reader body and exact bound receiver
may use the existing stat_many_for_admission API to observe the union of actual
effective resolved roots and ancestors. Retain definition-time class/function,
code/globals/defaults/closure and instance-bound-method identity. Customized or
unsupported readers retain the original scalar/pinned-directory route. Drift
after choosing the batch refuses before dispatch or result publication; it
never switches to an unknown reader under that chosen interval. Original
bottom-up named identity checks, descriptor policy and uncertain-close custody
remain in the existing API. Every new binding invocation obtains a new tree.

The original parent policy remains exact, including trusted owner checks,
shared writable ancestor refusal, allowed sticky ancestors and stricter refusal
of a shared sticky immediate parent for the missing .bootstrap-reader leaf.
Drive-root empty-component refusal and every original POSIX loop statement
remain. Native permission/custom/body-drift and protected-HANDLE close controls
are required alongside the count GREEN. This leaf does not qualify whole
Send/startup improvements or change any native/helper/responsiveness budget.


## Finite Workspace composite ownership refinement (2026-10-05)

ADR required: yes
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: the optional private composite boundary crosses Workspace services and DB connection ownership. It retains the existing admission and retirement policy.

Four source-qualified native cold controls demonstrated two physical Workspace SQLite handles for one `ChangeReviewConsentService.admit_turn`, `status`, `WorkspaceFileInspector._current_scope` capture, or `LocalWorkspaceRegistryService.save_runtime_binding` call. Every expected result, actual close, ordinary lease retirement, monitoring retirement and source check passed before the work-count assertion failed. The three active borrowed-transaction controls passed. This is a handle-lifetime regression introduced by making the public readers own their new connection: an outer lazy `operation_owned_connection` cannot retain a handle that the first public reader opens and closes itself. The native driver took 30.25 seconds; this receipt establishes work count, not a performance improvement.

For these four exact stock composites, one private context may enter the original `operation_owned_connection(database)` and original counted `database.connection()` before the first Workspace query, while preserving the original service lock, query order and write transaction. The default Change Review capability read remains before this context because it reads configuration, not Workspace SQL. FileInspector retires the context after its two registry reads and before its existing filesystem identity checks. Save-binding retains its existing explicit transaction and checks captured owner/source immediately after `conn.execute`, inside that transaction before commit, as well as before helpers and later reads.

Qualification is restricted to the original concrete registry/WorkspaceDB classes and defining-module callbacks, static descriptors, original lookup, real bound receiver, module/file/spec origin, and function code/globals/defaults/keyword defaults/closure cells. This covers the actual connection and transaction decorator hierarchy, held/getter/close methods, the original owned-cleanup helper, registry query methods, clock, metadata serializer and mutation-generation callback. A preinstalled custom receiver, lookup, clock, serializer, reader, subclass, memory DB or nonstandard binding input retains the preceding route. A qualified owner or source change during admission/body/retirement refuses before a later query or successful result. Custom pure Consent/Inspector paths do not import the DB subsystem solely to attempt this optimization.

The context captures the exact DB, original thread-local cache/path and process/thread/task actor. Existing borrowers, including uncommitted transactions, remain caller-owned. Only a connection newly created by this finite context is retired by the original owned-cleanup helper. A failed physical close is not treated as retirement; retained native custody remains installed. Body exceptions keep precedence over a cleanup error, with that cleanup error chained. Original guard/source/uncertain-close failures never fall back.

A distinct private entry exception lets only the additional optional connection's ordinary SQLite entry failure retain each callsite's existing unavailable/storage-error conversion after successful cleanup and final fences. It does not retry a failed read, swallow custom reader exceptions, or convert a guard/uncertain-close failure into an unavailable result. Existing public reader query-error conversions remain intact.

Rejected alternatives: reverting owned public reader cleanup (reintroduces leaked cold workers); raising work-count limits; a resolved/permission cache; a generic global helper-depth change; a proxy DB or replaced reader/guard; grouping configuration reads into the Workspace interval; and grouping unqualified whole UI/provider visits. Retry/set-active/runtime-turn/WorkspaceFiles-visit variants remain separate follow-up candidates until their own actual boundaries and native controls qualify.

Acceptance requires the unchanged seven original native work-count/borrower controls, the additional source/custom/SQL-error/write-rollback controls, and appropriate existing Workspace service tests. Pure compilation, metadata/closure and cleanup-order controls do not establish native acceptance. No whole-probe budget or freshness/custody policy changes.

Review qualification: candidate4958 is a temporary native-control checkpoint, not an accepted implementation. Exact-cache retarget and custom Consent/Inspector lookup findings must reproduce with physically retired test-owned resources before a repair is accepted. Final production must retire only its positively identified newly created captured handle through the original core-closing boundary, preserve borrowed and foreign handles, and decline custom service lookup/descriptors before optional selection.


## Finite initial Console receipt preparation (2026-10-05)

Reason: explicitly define the original receipt initializer's asynchronous startup and same-App/database publication contract. No persistent preparation service, readiness capability, storage/schema change, permission cache or new authority is introduced.

Proposal before any managed production edit:
- The original resolved initial Chat route prepares only its durable receipt store before constructing its Console screen. Other destinations and original synchronous/custom/headless Runtime APIs keep their preceding routes.
- Only exact stock Runtime/helper/reader defining metadata qualifies the optional worker route. Preinstalled custom functions/instance shadows, subclasses and memory/custom database receivers retain their original route. After selection, queued/body/default/class drift refuses before replacement invocation; there is no custom fallback inside the selected interval.
- Source qualification for publication is scoped only to the selected finite worker through a thread-local proof restored in its original finally; preinstalled class/instance/subclass wrappers may continue delegating to the original synchronous reader. The proof contains the exact Runtime and its captured source-current check, and is not a service, cache, authority or reusable task.
- The original initializer captures the actual App, ChaChaNotes owner/path and marks-service owner before construction. Publication must still belong to that same owner after native construction, and every newly created refused initialization connection retires on its original source thread. Existing service/native borrowers are not adopted or closed.
- The initial startup task holds one finite child task; repeated cancellation drains the same actual callback before propagating cancellation or releasing startup custody. Normal exception stays visible, and cancellation does not publish a screen. Runtime disposal closes admission first and serializes with the original initializer lock; no disposed Runtime publishes storage.
- After actual callback retirement, the initial route rechecks exact App/Runtime, startup task/loop/thread, profile/source, screen stack, current tab, and shutdown/initial-push state before the original screen construction and push. A newer destination is never overwritten by a stale initial push.
- Live first Send retains every original bridge, storage, permission, provider, capture and readiness guard. Receipt preparation makes no claim of permission readiness. All original startup, Send, heartbeat, helper and native-open limits remain unchanged.

Native qualification plan: retain the original cold held-query causal RED first; then real queued reader replacement and same-function code/default drift, in-body App/DB/marks/generation drift, repeated cancellation while actual native SQL is held, disposal during the same held callback, warm borrowed original connection, preinstalled custom/instance/subclass/memory fallbacks, and normal first composition/key/Send/disposal controls. Controls observe original code/native handles; no native calls, guards, waits or budgets are replaced.

Evidence-only candidate is not installed or Native-qualified. Root owns task/plan/ADR registration and serialized actual runs.

Actual original-effect qualification: original-cold-receipts-native-red-6 completes31.375s with current production/test sources. The original Inspector/bridge/receipt/schema chain runs on the main Thread and actual shared loop. Holding its positively identified admitted native SQLite connection prevents the original loop callback; after release, normal Runtime disposal physically closes the exact handle and retires its original StorageLease/participant registration. Source/currentness, original five-cell close closure, global monitoring zero, no invalid observation and monitoring retirement all qualify before the responsiveness assertion. Attempts1-5 fail observer prerequisites and are excluded. Artificial hold and inclusive native spans establish this synchronous blocking site, not normal latency or whole-budget savings.

Rejected alternatives: precreating a persistent store outside startup; a detached readiness service/task; moving UI bridge construction to a worker; weakening admission or changing synchronous custom/public ABI; increasing original budgets; or accepting a captured result after navigation/owner/source drift. Final acceptance requires actual native normal, drift, borrowed, custom, cancellation and disposal controls, then unchanged original whole/platform and live first-Send evidence.


## Captured Workspace composite retirement correction

ADR required: yes; refine the existing ADR126 amendment before applying this fix.

Actual captured-gap native controls produced six genuine failures and one
borrower pass under the unchanged original guard/source/physical-close checks.
Retargeting the stock DB cache after its original held getter returned leaked
the composite's new A handle; a replacement holding foreign B also caused the
lazy owned-helper exit to close B. Four custom service lookup/registry descriptor
routes took the optional A scope while their original B/B reads remained intact.

For only these four optional composites, replace their newly added lazy
operation_owned_connection scope with retirement of the exact newly created
handle. The original counted connection interval still spans the entire finite
callback and exits before retirement. The original public-reader ownership
scopes remain. Record and qualify the original core_closing wrapper/body,
defining namespace/file/spec/origin/alias, code/defaults/closures, and captured
process/Thread/Task. Its original actor/lease/active-work checks govern close.

An original borrower is never closed. An identified new handle is closed once
through that original custody boundary, then positive native closed evidence is
required. Only its still-matching slot in the captured old cache is cleared after
positive retirement. Replacement caches and foreign handles are neither changed
nor adopted; no DB field is restored. Failed/denied/uncertain close retains the
handle's custody and error, without retry. Body exceptions keep precedence, with
retirement failure chained. Owner/source checks still refuse publication.

The earlier amendment's promise to use the original lazy owned helper for this
optional outer scope is superseded here. An ExitStack callback ahead of that
helper would avoid foreign close only on success: on denied/uncertain A close,
the helper would still re-read B or retry A. Clearing/restoring caches, changing
the generic helper or weakening custody was rejected.

Definition-time Consent and Inspector lookup and _registry descriptor records
decline custom selection before the optional scope. Mid-scope checks test those
records before invoking a changed lookup. Existing custom dispatch is preserved.

Acceptance remains the actual 21 native controls and adjacent original service
tests. Pure metadata/error/cleanup controls do not establish native acceptance.
No budgets, guard policy, generic helper or public service interface change.


## Stock run-turn finite admission refinement (before production)

ADR required: no new ADR; existing ADR126 finite same-callback source custody applies.

The corrected passive original-code native observer records nine execution-scope starts for five distinct actual owner/path keys inside one stock guarded run_turn. It observes the original contextmanager generator without replacing production callbacks. The same source-qualified run_turn finishes, its native worker SQLite handle physically closes, storage/monitoring retire, and unchanged source checks pass before the work-count assertion fails. All five custom-selector/root/method/source/pause controls pass. Native red-1 is excluded for the callback adapter's one-frame observer mistake; red-2 is the genuine causal result,44.015seconds. This establishes duplicate work, not exclusive time savings.

For only an already captured qualified stock scoped log source, append its agents.history root to the first original execution source set. Keep the original scoped_log_source qualification and body within that single original execution context. Remove only the second whole-source-set execution entry from this stock path. No authority is reused beyond this synchronous callback. The unqualified scoped=None selector still executes inside its preceding first admission and retains its second fresh log-root admission; independent worker/model guards, generic guard code, native owners and permission/pause policies remain.

Acceptance: the same six native controls must pass with one scope start per actual stock owner/path; the custom selector must retain its original arg-free call and duplicate set route, all retarget/pause controls must refuse provider/log effects and retire resources. Then run the appropriate original scoped log/agent activation compatibility nodes and unchanged whole/platform limits. No tests, bounds, source qualifiers or coordinator checks are relaxed.


## Exact-stock Console browser finite Notes ownership (2026-10-05)

ADR required: yes, amendment of this existing decision.
Task: TASK-34405 AC9. Plan: Docs/superpowers/plans/2026-10-04-console-performance-fixes.md.
Reason: the optional two-scope callback crosses the Console controller, original local conversation service and Notes native ownership boundaries.

The actual Windows ordinary-owner controls establish two original global/Default reads using two separately closed Notes handles. Midpoint service/database/body redirection reaches a changed reader before refusal. Three actual same-worker cache replacement controls leave the newly created A connection live; both foreign-B variants also close the original caller B and accept rows. All17 defining-source, SQL, observer and test-owned cleanup prerequisites pass before eight behavior failures; nine custom/memory/borrower/error controls pass. The first expanded launch failed Windows argv length before child creation and is excluded; the corrected inline transport preserves the complete child AST and original45s bound.

Only the definition-time qualified, exact local ChatConversationService and file-backed CharactersRAGDB may group the two existing stock browser queries in one complete synchronous counted Notes interval. Capture original bound readers, database/path, module/namespace/file/spec/origin and callable code/default/closure metadata before optional selection. Preinstalled custom lookup, instance/subclass/asynchronous/custom-body/memory routes retain the preceding callback and error ABI. The original query order, filters, limit/offset, normalization and ordinary partial-error behavior remain.

The worker captures its actual local object, preexisting registered handle and newly created native connection. Recheck exact service/DB/callable/source/cache/actor custody before each original read, after admission, and after retirement. App/controller fields are checked on the captured caller loop through the existing bounded Future rendezvous pattern; expired queued checks are cancelled. Source/owner drift refuses a stock result before invoking a changed second reader or accepting any rows.

Exit the counted interval before retiring only the positively identified worker-created A through the original Notes core-closing boundary and existing rollback/WAL/close policy. Clear only a still-matching captured cache slot after positive native retirement; preserve replacement caches, caller B, borrowers and their quiescence records. The generic operation_owned_connection lazy exit is not an outer wrapper for this added optional scope. Capture/check the defining retirement source rather than calling through a mutable replacement. Failed/refused/uncertain retirement remains visible, and original body errors retain cleanup failure as secondary evidence. No connection or permission result survives this finite callback; each later invocation independently queries again.

Before acceptance, the same17 unchanged controls must pass, followed by appropriate original browser/read-retirement/source/custom/transaction compatibility checks and actual whole/platform budgets. A one-handle callback pass makes no total helper or responsiveness claim. No guard, source check, native lease, permission policy, timer or performance ceiling is relaxed.

## Consistent empty scope in shared Agent history presentation

The existing Agent section and fleet share one issued, finite historical display state. Their conversation key uses the same normalization for an unpersisted Chat (`None` and `""` denote the empty rail scope). A sibling fleet derivation returning no rows does not invalidate that state's matching source and full local owner key. It clears a mismatched state immediately and never starts an empty-scope worker on its own.

Publication retains the original captured database, owned callback retirement, fresh full-key comparison, and cancellation behavior. Sharing this display state grants no tool, Send, or file permission and reuses no admission proof. The original SQL readers and direct/custom/memory routes remain unchanged. This repairs inconsistent normalization and premature state clearing; whole-probe budgets require separate original-source evidence.


## Pending same-owner expired display deferral

ADR required: yes, amendment to the existing checked warm Console display scope.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: narrow clarification of the existing presentation/source boundary; no new
scope, source capability, timer lifetime or reusable authority is introduced.

An exact standard readiness projection with its genuine issued display proof
must pass all existing current source, mapping, owner/session/settings key,
installed participant, pause, UI Thread/Loop and active-projection checks before
this scheduling distinction applies. If its original real-clock age has expired
and its refresh scheduling bit was already literally True at synchronous
display entry and remains True, checked display
refuses the synchronous body and requests the existing deferred UI retry. It
does not open a second enclosing native configuration operation while that
refresh is pending. The bit is a refusal-only scheduling hint; it is never
evidence of freshness, permission, a live native lease or successful retirement.

The existing one-second (or shorter configured display-age) comparison remains
unchanged. A mismatched publication timestamp retains its original fallback.
Expired nonpending display, direct live calls and custom/unqualified projection
or reader paths retain the original native route or cold deferral. Changed
source/owner/participant/pause keeps its original earlier refusal. Any actual
nested guarded disk reader or writer still enters its own fresh native scope.

The original projection schedules and owns its finite reader. Queued refreshes
are covered before `_read_request` is installed by the async body; requiring
that later record here would reopen the synchronous UI fallback during the
queued interval. Cancellation retains pending/settled ownership until the actual
native callback retires. This change does not alter that worker or publication
contract, capture any native verdict, or carry a scope across an await.

The existing delayed control-bar/replay request owns the unfinished presentation.
An unchanged successful mapping refresh issues a new proof/clock without creating
an additional publication worker. The already requested retry then checks and
renders that fresh proof. If still pending it defers again through the same
existing timer; screen teardown retains the original refusal/no-retry behavior.
No new timer or timer-rate change is introduced.

TASK34406 AC18 native prerequisite `readiness-pending-native-red-1` qualified the
original mechanism: while the source-current actual reader task remained pending
after its native operation retired, the original UI performed one enclosing
config admission and two storage admissions and rendered rather than retrying.
The receipt completed all seven observed spans with no invalid/source/global
event/retirement gap and zero ordinary/pending/core/raw/retiring/raw-state census. One intentional startup lease remained separately reported. Direct nonpending native entry,
the unchanged genuine-expiry control and admitted repeated-cancellation custody
passed. This is bounded causal evidence, not a whole Send/startup performance
result. Production acceptance still requires the repaired native controls and
existing original compatibility checks.


### Pending refresh entry distinction (TASK34406 AC18)

The previous pending-expiry proposal is not accepted as compatible:
`readiness-pending-native-green-1` passed its three added controls but failed the
unchanged genuine-expiry control. A direct decorated presentation call can start
its own refresh before entering its original native body. Its preceding native
fallback contract must remain.

Capture the existing refresh-pending bit at the beginning of the original
synchronous `ConsoleReadinessConfigProjection.run`, before that call schedules
any refresh. Install this refusal-only fact only around the same existing active
display body and restore its prior value in `finally`, including nested calls,
body errors and cancellation. It contains no mapping, receiver, Task, native
lease, source verdict or authority. No fact is accepted as display freshness.

After every original genuine-proof/current-source/owner/participant/Thread/Loop/
active-display/key fence, retain timestamp mismatch's original native fallback.
On genuine clock expiry, defer only if a refresh is still pending AND it was
already pending before the current synchronous display entry. A direct call
that starts its own refresh retains the original native route. Already queued
and running refreshes are both covered; `_read_request` is not required because
the queued original Task has not installed it yet.

All original clocks, TTL, fresh display acceptance, custom/unqualified fallback,
nested live guarded disk scopes, reader retirement/cancellation and UI retry/
maintenance/replay behavior remain. Equal-result publication continues to issue
its fresh proof without another publication worker; the already requested
control-bar timer/replay retries that proof. This scheduling distinction grants
no permission and does not reuse or extend native authority across an await.

Keep v1's source-qualified compatibility failure as RED evidence. Repaired
acceptance requires the unchanged original genuine-expiry node, actual running
and queued preexisting-reader controls, direct native/cancellation positives and
nested/error restoration controls. No whole performance acceptance is implied.

### TASK-34406 clarification: Default workspace binding presentation

ADR028 forbids named folder bindings in Default; its private Chat scratch is a
separate authority. The exact stock runtime-binding lister removes any legacy
Default rows and always returns an empty tuple. A disposable presentation may
derive that immutable empty-binding policy without invoking housekeeping only
after qualifying the defining original service, database, Default constant,
normalizer, lister and cleanup source, and the current profile, installed
participant, registry/database and generation owner. This is a policy derivation,
not a cached database or permission verdict. Memory, custom and unsupported
sources retain their original public route.

The public lister, direct scope reads and live actions retain fresh legacy-row
cleanup. Named or mixed availability retains its original finite native reader,
filesystem status, one-second TTL and cancellation drain. Only a wholly Default
display may derive its existing empty result before starting a native worker;
publication must recheck the original current-owner and source fences. An owner
or source change after selection refuses publication. No connection, admission
proof or action permission survives this derivation, and no timer, timeout,
native guard or performance ceiling changes.

Actual native source/custody controls must first reproduce the repeated cold
Default display connection, then verify its removal alongside a direct native
cleanup positive, real named availability and source/owner/custom/cancellation
controls. The whole-app and platform budgets remain independent acceptance gates.


## Fresh Library initialization within the existing creation owner

TASK34406 AC19 moves the stock new-profile Library UNKNOWN stamp into the
existing exclusive private config creation described by ADR076. Both existing
creation branches retain `create_private_text`, application-owned directory
selection, no-follow/private owner checks, flush/barriers, competing-name
handling and the original refusal/failure contract. Only successful creation
and posture reporting may add the creation value to the returned mapping and
publish the original first-profile/bootstrap success facts.

Programmatic/default merge templates remain unchanged. Definition-time plain
source eligibility may select a creation-only appendix; customization declines
that optimization and keeps the original App compatibility stamp. No schema,
second store, permission cache, native verdict, new profile-origin flag or
deferred writer is introduced. Existing file/custom-parent/cache/generation/
encryption and source ownership policies remain.

The test factory may use the canonical existing conditional settings mutation
for an explicitly documented returning-profile fiction. It requires current
path-bound creation facts and the exact untouched stock document, then checks
source/path/document again under the original transaction lock. A scalar save
without a precondition cannot meet this contract. It must not nest another
interprocess write lock, reinterpret a sticky global flag as path origin, or
rewrite an edited/custom/preexisting/explicit profile. This fixture write is
not a user-startup performance saving. Original normal writer publication and
partial-failure status remain visible.

Actual native creation/old-profile/refusal/origin/reopen checks determine
acceptance. Original startup/Send/heartbeat/helper/open limits are unchanged
and remain independent gates.


### TASK-34406 clarification: successful Character display publication (2026-10-06)

An unchanged Character display observation may reuse its existing short presentation lifetime immediately after the same invocation has successfully published the original stock recent-groups refresh. A scalar generation receipt is produced only after the original finite batch publishes, its defining source bindings remain current, and the original generation still owns the result. The presentation caller additionally requires the expected generation, the captured original refresh binding, the current screen/app/store/session/source owner and a successful fingerprint before setting the existing memo timestamp.

A prior successful state alone does not establish this receipt. Entry refusal, source drift, superseding generation, failure and cancellation cannot establish a memo. Public `refresh` still returns `None`; custom, unsupported, memory and direct routes retain their original fresh behavior. This does not lengthen the existing display TTL or grant permission to a live action.


### TASK-34406 clarification: live pending facts during configuration refresh

The real cold-owner regression in inspector-cold-live-pending-native-v4-red-1 reached the original checked worker with two actual raw leases, the real legacy approval card and in-memory count1. All source, monitor, worker, round, native custody and fixture retirement prerequisites passed. Its sole final oracle failure recorded three facts: Inspector count0, missing pending row paint, and disabled Review. The original denial and count0 after release passed. Earlier gate/cleanup failures remain excluded.

Retain the original guarded full presentation and its one-second checked mapping age, pending/retry behavior and return value. Only after that full body defers may a previously complete state for this exact view/runtime/App/controller/store/actual selected-session/host/config selector/revision be reused as the unchanged base for a new pending-only fragment. Exact owner references include the config mapping, data owners, loop and Thread; changed attachment, session, source path, settings/workspace binding or runtime generation declines. Only configuration generation may differ from the prior complete base; it must remain unchanged across this synchronous fragment.

Call the original in-memory count and kind readers with their original individual non-reentrant locks. No outer lock, config disk read, ensure, worker, permission verdict cache, await or new timer is introduced. Check definition-time original callable/code/globals/default/closure metadata and static receiver lookup before entry and between readers. The two Runtime property fget anchors belong at their defining module completion, so a preinstalled customized property declines without invocation. New field descriptors cannot redirect these instance-only owner/read fields. Malformed optional metadata, unknown bases or receivers decline safely to the existing retry.

Retain only normalized display inputs for the recipe and launch title; custom non-string titles retain their original full-render coercion but are not retained for later partial coercion. Update Approvals/count/hasPending, Live work, recipe approvals, and the existing Review affordance coherently. Preserve provider, retrieval, scope and unrelated actions. Re-read actual count and kinds, then recheck ownership before and after publication; drift rejects acceptance and requests the existing coalesced retry. Original action/Send/approval authority gates and subsequent full publication remain fresh. No source body or automatic context is exposed.

This refines ADR126 and preserves ADR085 App-owned runtime/view attachment and ADR210 current region/action ownership. It adds no visual tokens, action, keybinding or schema.


### TASK-34406 clarification: finite compact model setup reads

The compose and mount lifecycle paths in the stock compact model selector must avoid running the public checked provider-settings reader on the UI loop. They use one finite background invocation of the original public reader, retain its native checks, and apply the returned options and defaults only to the same mounted widget, App, configuration mapping, selection and source generation. The compose phase may yield controls from the current in-memory options while the checked refresh is pending. The original explicit provider-change action and custom reader contract remain fresh.

Capture definition-time stock reader metadata and reject source or owner changes before invocation and before publication. Retain and drain the exact worker on cancellation, repeated cancellation or unmount, including physical resource retirement, before accepting another lifecycle result. Do not publish over user choices made during the wait. Preserve the existing default-provider resolution, model fallback, Select-change suppression, temperature and control-bar synchronization. No permission or native posture verdict is cached; no TTL, timer, source guard, startup or Send budget changes.

Acceptance requires genuine held original-reader evidence for both lifecycle paths, actual UI-loop progress during the hold, normal worker and native retirement, current-source and owner controls, and the original provider/default-selection tests. Cross-platform and whole-app performance gates remain separate.


#### Compact model display inputs and transient cache replacement

The stock App retains its constructor settings mapping while ordinary original settings reloads may replace the global cache. Object identity with that transient cache, or its last source marker, is therefore not an eligibility proof for a finite stock provider read. Exact plain App mappings and nested display/default values may select that read under the existing defining callback/class/lookup, current lexical configuration identity, parent, loop and owner fences. These input values grant no native authority; the original public callback retains fresh permission, path and posture checks for each invocation.

Capture and recheck the exact App mapping and its plain inputs without rebinding either App or cache. Custom mapping classes or nested coercions, customized readers/bodies/classes and unsupported sources retain the direct lifecycle contract. A preinstalled exact plain dict is display data under the same rule; it cannot grant storage or Send permission. Mid-read mapping, user selection, source or generation changes still refuse publication, and the exact original callback Task still drains before retirement.


### TASK-34406 clarification: finite deferred Collections capture setup

The stock post-ready timer currently calls the synchronous Collections capture initializer on the UI loop. The source-current contained whole probe records this original caller during the idle UI pause; detached setup must preserve the fresh path, data-root, schema, offline-store and legacy checks. This amendment refines this ADR's finite callback lifetime and preserves ADR113 Local/Server authority selection. ADR085 concerns the activity receipt switcher and does not govern this initializer.

Only the original deferred timer with the exact existing stock file-backed LibraryCollectionsDB may select one finite background build. Keep constructor and synchronous first-use/custom/memory routes unchanged. The callback invokes the captured original native readers and builders, returns detached pieces, and physically retires only its newly created worker connection/resources on that same worker; existing borrowers remain live. The UI loop constructs the non-native scope/service, publishes fields and performs original authority activation only after exact App, deferred scope, database, original defining source bindings, selected configuration path/generation and runtime owner still match. A first-use winner, replacement, changed source or shutdown cannot be overwritten.

Retain the exact initializer Task and native callback through cancellation, repeated cancellation and failure. Capture shutdown seals new initializer admission and deactivates the current scope before cancelling/draining the initializer, and completes the existing extraction shutdown before creator-resource close. A cancelled Task alone cannot prove native retirement. Late timer calls cannot start after this boundary; no result or authority survives retirement for another initialization. Generic unmount cancellation occurring later is insufficient.

Keep the original timer interval, source/permission checks, reconciliation behavior and direct APIs. No configuration or native permission cache, longer TTL, constructor-wide guard, await-held native scope or performance ceiling change is introduced. Original Collections app-wiring contracts and genuine held native build/source/owner/first-use/cancellation/shutdown controls establish acceptance. Whole-app startup, Send and platform gates remain independent.


### TASK-34406 clarification: pending-only display at first persistence

The source-qualified first-persistence control now observes the same Session's
ordinary None-to-durable-conversation publication while the original checked
configuration reader owns two native leases. Every other owner/incarnation,
host, binding-revision, Workspace, data, mapping and settings fence remains
current. The original complete-base comparison correctly refuses the changed
conversation; the live count is one, but Inspector and pinned counts remain
stale. Original denial, callback, lease, observer and native process retirement
all pass. Evidence: first-persistence-coalesced-observer-native-1, exact receipt
df9a8354b386257ee65325bfa74771ee7b951e084cdae6f6c9dcf393b10cbc25.

Under ADR-085 creator ownership and ADR-210 strict owned Inspector content,
retain the complete-base key and original full-call deferral/retry contract.
Add a separate explicit pending-only model/screen/pinned-summary display only
for that positively identified first persistence of the same exact Session.
It requires a valid prior complete base and every existing fence, permitting
only config generation and None-to-nonempty durable identity to differ, while
the binding revision remains unchanged. The old complete base is rejected and
none of its provider, recipe, scope, retrieval, evidence, readiness or action
values is transplanted. Missing base, replaced Session with the same ID,
changed incarnation/host/runtime/profile/data/attachment/settings/Workspace,
custom readers and any other revision change retain their prior refusal.

Read only original current selected-session in-memory count and registered
decision kinds, with their existing separate non-reentrant locks. Recheck
count/kinds and the complete current owner around publication; drift refuses
acceptance and requests the existing retry. Retain defining-source/code,
globals/defaults/closure and static-receiver qualification. Never hold an outer
host/controller lock, invoke native config/DB I/O for the fragment, or establish
new permission or Send authority from display data.

The explicit pending-only marker survives equality, cleaning, owned-content
projection and pinned summary normalization. It is distinct from unknown or
colliding row IDs and may not use full-state defaults. Existing owned rows show
the current literal approval count and decision copy; unrelated Where, Scope,
Sources, provider, retrieval, evidence, recipe and run readiness say Refreshing.
Review alone may be available from current approval/question/confirmation kinds
and uses its existing live card handler. Approval-first priority is preserved;
questions/confirmations do not inflate approval count. An empty count/kind set
clears Review without inferring Ready or No active work. Existing stable row IDs,
strict ownership, token classes, reconciliation and focus recovery remain.

This partial observation never becomes the complete pending-display base,
configuration proof, persistent state or a cached decision. A later original
fresh full publication restores complete content normally. Acceptance requires
the actual first-persistence control, marker and no-foreign-field/action checks,
owner/source/fact drift refusals, current decision-kind transitions, and the
unchanged six original mounted journeys and cold/retirement controls. Whole
startup/Send and platform performance budgets remain separate and unchanged.


### TASK-34406: finite Character view work retires before creator close

The actual shipping resume worker and supported Console host must retain their
issued finite Character callback until native resources have physically
retired. App or host exit cannot establish healthy creator cleanup from a
cancelled or terminal logical Worker alone. Two original native routes reached
exit with the exact callback, operation and lease still live; their subsequent
settled-close correctly refused, and all final native/source cleanup controls
passed after release.

For the source-qualified standard UI resume, use the existing owned Character
presentation facade with explicit force-fresh entry. This bypasses only its
display memo; retain the original initial metadata pair, inner refresh, final
owner publication checks and repeated-cancel physical callback drain. Direct
actions and custom/subclass/memory/overridden callbacks retain their original
ABI, TTL and fresh reads.

App shutdown and the supported Console test host share only a bounded finite
view-worker drain. Capture the exact manager, owning nodes, same-loop Tasks and
issued workers before cancellation; include Character resume refresh alongside
existing sync and navigation work. Close host intake through original Textual
shutdown and drain the captured callbacks before host return or creator close.
Do not infer physical retirement from Worker state or adopt an unrelated
Runtime/database/worker. Runtime disposal and creator acceptance keep their
existing owners and ordering, including borrower and foreign-source refusal.

Original source, actor, actual Future/native handle/operation/lease retirement,
error priority, repeated cancellation and timing limits remain mandatory.
Explicit captured work references last until actual retirement; disposal
refuses late publication. This refines existing ADR-085 view/App ownership and
ADR-126 finite custody; no permission cache or new storage authority is added.

### TASK-34406: retained Character picker reads own their finite callback

An exact file-backed Characters database selected by the Character context
picker retains the same identity checks before and after its read. Its finite
worker must retire a newly acquired connection on that same worker before the
selection publishes. An already borrowed connection or transaction remains
owned by its caller. Cancelling the selection, including repeated cancellation,
must await the actual read and its cleanup before propagating cancellation.

Use the existing owned database callback for this retained stock route. Custom
and memory databases and the unbound picker route keep their preceding APIs.
This completes the existing finite ownership contract; it changes no storage
authority, profile check, timeout or global cleanup policy. Tests hold the
original card reader with its actual SQLite handle and observe both completion
and cancellation, including a borrowed transaction.

### TASK-34406: default prompt-history selection starts in its actual worker

Constructing the app-shared default PromptHistory is IO-free. An omitted path
means unresolved default selection; an explicitly supplied path and the app's
custom synchronous history factory retain their preceding APIs. The actual
history callback resolves the default once while the existing load/append lock
owns the sink, then retains that lexical source.path. It never follows a later
profile retarget. Every subsequent job still performs the fresh original raw
source-selection and native checks, and a default source whose fixed path no
longer qualifies as the current default refuses rather than demoting to custom.

Only the exact default-history instance defers the FileJob selection handshake.
Its creator still owns the original pending acquisition, creator pid/thread/task
identity, one-shot dispatch state and loop result bookkeeping. Pending registration
grants no IO authority. The actual worker resolves and qualifies the source and
checks its installed closed gate before the unchanged unbound history body and
fresh raw scope. Explicit-path/custom histories, note templates and sidebar state
retain their preceding creator-time selection and refusal timing.

Queued cancellation prevents all source entry, including the deferred resolver.
Running and repeatedly cancelled jobs retain the actual callback through default
resolution, source IO, native retirement and loop bookkeeping. A fixed path or the
private dispatch flag grants no permission and supplies no cached witness. Raw
profile, parent identity, source demotion, pause and foreign-owner checks remain
fresh. Capture inventory continues to derive the prompt-history leaf from the
selected profile; an unstarted sink has no registered raw participant. Original
startup and Send budgets remain unchanged and require separate native receipts.


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


### Finite Change Review transcript reads (2026-10-07)

The existing async transcript refresh awaits one stock marker read through the existing finite preparation-read owner. Both original anchor/snapshot queries share the captured AgentRuns database and one counted worker interval. The original public synchronous/custom/memory route remains available. Runtime disposal observes physical read retirement; repeated cancellation cannot abandon the worker. The worker closes only its newly acquired exact connection after leaving the counted interval, preserving a borrowed connection or a foreign replacement. The loop rechecks Runtime, coordinator, bridge, database, selected conversation and publication revision before caching; stale results publish nothing and the next existing refresh retries. No new scheduler, timer, permission cache, or performance-limit change is introduced.


### TASK-34406: stock Console skill trust setup belongs to App shutdown

The Console's original local-context call enters the scope policy gate before
the local content-source wrapper evaluates its lazy trust factory. That factory
performs the original trust-service setup synchronously; constructing the local
facade lazily does not keep that setup off the shared loop. The existing async
ensure retains its exact physical builder callback and singleflight lock through
repeated cancellation. The stock Console route must actually use that ensure and
must keep the issued work in the original App's shutdown custody.

Select only the definition-time original Console controller, exact App class and
inherited service slots, scope facade, local wrapper/content-source functions,
factory lambda code/globals/defaults/closure, original worker dispatch/wait, and
current configuration source/publication tags. Reuse the existing private finite
metadata checker after qualifying its defining source. Malformed keys, containers,
functions and module metadata decline without invoking foreign callbacks. Exact
instance, facade, local-service, lock, captured policy/server collaborators and
loop/thread owners remain current through the callback and loop publication.
Custom fetch/getter/scheduling/source compositions and services already ready at
selection retain the preceding direct route. No source check performs setup IO.

There is no existing post-policy setup hook in the scope `_call`. The selected
Console call therefore passes one private preparation tuple through the original
`get_context` to `_call`, which awaits it immediately after its unchanged scope
policy gate. It then invokes the unchanged local getter and body. Both policy
checks retain their original count, order, errors and fresh state; no permission
verdict is captured or reused. A denied scope policy starts no builder. The
rejected controller-first prefetch would have run setup before a denial that
originally prevented it. Ordinary callers do not pass the private tuple.

The preparation uses the exact App's original worker manager, with group
`console-skill-trust-setup`, nonexclusive work and the existing retained ensure
body. Name the two original mount/resume discovery workers
`console-skill-discovery`, preserving their existing nonexclusive work. Add both
groups to the existing finite manager-wide Console shutdown selection. Captured
issued Tasks, including detached discovery nodes and the App setup node, remain
owned until the original callback and Task physically settle before Runtime
disposal and creator close. A cancelled or terminal logical Worker is insufficient.

An App ready/injected winner installed during setup retains the original ensure
semantics. A ready local trust slot installed during the held setup remains the
original local getter's fresh winner; setup never assigns the local slot. The
subsequent context is produced afresh by the original local body, without a
prepared context, policy result or permission cache. Changed source or remaining
owner inputs refuse stale setup publication. Direct/custom actions keep their
original APIs and live checks.

Acceptance requires the source/refusal controls on real defining classes and
held original-builder controls on the real App: denied builder count zero,
cold/ready/custom routes, fresh injected winners, exact App manager selection,
repeated cancellation, physical callback/Future/lock retirement, original
shutdown/Runtime/creator close, original observer/source guards, and global drain.
The import-only controls do not establish App or native lifetime acceptance.
Original whole startup and Send performance limits remain independent and unchanged.
This is a narrow refinement of existing ADR-126 finite callback custody, with no
new security authority, service framework, runtime dependency or permission TTL.

The checked ensure uses the existing private physical preparation-read producer,
not a detachable inner asyncio Task. Before the first await it registers the
same handle with the captured original ConsoleRuntime preparation-read set.
The exact plain Runtime/App identity, undisposed state and absence of an issued
Console shutdown task are admission and publication fences. After shutdown
begins the stock route refuses; it cannot fall back to synchronous lazy setup.
Worker validation uses captured plain fields and source metadata only; loop-only
owner validation remains before issue and after physical completion. This closes
the gap after the shutdown worker snapshot and before Runtime disposal. Custom
and non-Console ensure callers retain their existing contracts.


### Qualified Console pricing presentation (2026-10-07)

An original held-model-reader control observed synchronous Console pricing gap-fill enter the models.dev enablement config reader on the UI thread. The existing finite readiness worker may prepare an immutable detached pricing catalog for its exact checked display owner. Only stock catalog construction, configured-price resolution and usage arithmetic qualify. The display catalog is passed explicitly to the current-price and historical-total calculations; it is never installed globally and never participates in Send or capability/vision caches. Direct, subclassed, replaced and injected catalog readers retain their existing native route.

The existing checked display source, owner, settings revision and one-second lifetime govern publication and use. Source replacement refuses the projection before the display body and uses the existing nonblocking native retry path. Semantic pricing changes invalidate display publication and historical totals; equal metadata preserves memo reuse. Hand-maintained direct/pattern prices and local-provider zero rates keep precedence over upstream gap-fill. Disabled or absent upstream data retains honest unknown pricing. No busy lock is interpreted as disabled. Existing capture, permission, persistence and native-lifetime gates remain unchanged.


### Passive Console refresh does not gate transcript publication (2026-10-07)

A source-current completed Send persisted its reply while the general UI refresh
waited for disposable readiness data before reaching transcript publication. A
held-original-reader regression reproduced this dependency after the native
configuration scope had retired, separating it from live authority lock refusal.

Keep the original fresh core, retrieval, tab and roleplay work before transcript
publication. Roleplay materializes message projections and must still run first.
Publish the transcript once before any cold passive rail refusal. Transcript and
control presentation already schedule their checked reads through `inputs()`;
the general refresh must not synchronously await those reads as a prerequisite.
Explicit modal warming and live Send/action checks retain their existing behavior.
The post-transcript rail check, current-owner refusal and delayed full-refresh
retry remain. Pending presentation cannot grant authority or publish old-owner
settings. This changes scheduling within the existing display boundary, with no
new worker, timer, cache, dependency or relaxed performance limit.


### Context presentation must retire before host exit (2026-10-07)

The original supported-host control reached shutdown with the exact Context
Notes callback, connection, operation and lease still live. Host capture omitted
its issued `console-context-presentation` Worker, and cancellation made the
awaiting task terminal before its executor callback retired. Include that group
in the existing finite view-worker drain. For the stock controller and stock
file-backed Notes database only, retain the existing `run_owned_db_call` in a
private standard Task and drain it through repeated awaiting-task cancellation.
Consume its outcome before delivering the original cancellation. The private
Task introduces no configurable task-factory callback; generic database calls,
custom/subclass/memory routes, fresh native admission, borrowed handles and
existing owner/publication fences retain their contracts. This completes the
existing finite callback lifetime; no authority cache or new storage API.


### Deferred Collections setup ownership (2026-10-07)

The stock deferred Collections initializer owns one finite preparation callback and its newly acquired thread-local connection. It retains that callback through cancellation and shutdown, and checks its original owner and source before publication on the issuing loop. Original synchronous first use and custom/memory composition remain compatible. Database and archive operations retain their separate original admission scopes: wrapping both in a database-only operation incorrectly rejects archive access. Borrowed connections remain owned by their caller. This completes existing callback custody under ADR113/126 without adding permission or cached authority.


### TASK-34601 amendment — cheaper native security observations (2026-10-08)

Owner-directed Send-latency work. Every guarantee in the two TASK-34404 sections
above is kept; only how the same facts are read changes.

- **Descriptor acquisition.** Owner and DACL bytes come from one
  `NtQuerySecurityObject(OWNER|DACL)` on the same freshly opened handle, into a
  call-local buffer (re-queried at the reported size when the descriptor grew).
  `GetSecurityInfo`, used before, also read the PARENT's descriptor to synthesize
  INHERITED_ACE bits and a group SID for objects without SE_DACL_AUTO_INHERITED —
  every directory this application creates — at 40–54 µs instead of 3–4 µs. The
  owner SID, ordered ACE types/masks/trustees and every projection are identical
  (`test_object_descriptor_projects_like_the_former_getsecurityinfo_route`); posture
  stamps now hold the object's own stored descriptor bytes, compared only against
  stamps read the same way. Non-self-relative bytes fail closed.
- **TokenOwner.** TokenOwner is queried afresh for each security observation whose
  projection can depend on it: an administrative owner (`_SYSTEM_SIDS`) other than
  TokenUser. For every other owner the projection is identical for any TokenOwner,
  so no token read is spent. Nothing about TokenOwner is cached. Immutable
  descriptor decoding is cached by descriptor bytes, directory kind and TokenUser
  and yields the owner SID, ordered ACEs and mode; the uid projection is computed
  per observation from those and the fresh TokenOwner when it matters.
- **Snapshot bookkeeping.** `stat_many_for_admission` builds its closed node set
  once (case-folded keys, first spelling kept, same depth-then-name order) and
  validates each component once. Opens, the two passes, pins, ESTALE and
  uncertain-close custody are unchanged.
- **Qualification.** `qualified_for` still reads `native_qualification.json` on
  every call; the pydantic parse is memoized by the exact text read.

Measured (observer-free, native Windows, interleaved): a 40-node admission
snapshot 10.2–10.9 ms → 4.8–5.1 ms; warm Send to provider entry 6.0 s → 4.5 s.


### TASK-34601 amendment — one path fence per acquisition; change-notified Windows evidence (2026-10-08)

Owner-approved relaxations of the per-call re-observation rules above, made for
Console Send latency. The full derivation remains the only source of refusals and
reason codes; every in-memory gate (pause, provenance, epoch, hold, maintenance,
lexical path, cancellation) still runs at every bracket.

**One operation-path fence per acquisition.** An acquisition nested in a
repository operation used to repeat the native operation-path fence (a
drive-root reparse-refusing walk to the parent plus resolution) at each of its
three or four bracket checks. It now performs that native fence once per path,
at the first bracket, before the lease is counted; later brackets of the same
acquisition repeat only the in-memory provenance and lexical-path checks. The
narrowing applies only inside one `acquire_storage` call: long-lived
`_Acquisition` reservations elsewhere never pass a path. Repository operations
and every other `_check_operation` caller keep their own fences.

**Change-notified evidence reuse (Windows).** A warm reuse may skip the full
re-observation of confirmed evidence, and the per-call `qualified_for` walk, only
when all of these hold:
- overlapped `ReadDirectoryChangesW` notifications cover every directory of the
  evidence tree (posture-only directories: names, attributes, security; parents
  of content paths: also size and write/creation times), each opened by name under
  its pinned parent after the same single-component reparse-refusing walk and
  verified to be the walked object, and issued from one process-lifetime thread
  (Windows cancels a thread's pending I/O when it exits);
- the watch was armed BEFORE the full observation that confirmed these exact
  evidence objects, that observation found every content file with one link, and
  no write-capable by-id reopen ran in this process meanwhile (a native-mutation
  generation; by-id writes notify no directory);
- the watch is still quiet when checked after the lease is counted and after the
  drive root (which no parent directory can watch) is re-stamped directly; a
  content path on a drive whose root is not posture-stamped disables watching;
- the confirmation is younger than a 0.5 s backstop.

Any signal, overflow, cancelled or failed notification, arm failure, backstop
expiry, generation change, changed evidence identity, hold or epoch runs the
original full observation, re-arming first so a change during it is never lost;
a full observation that finds any change un-verifies every watch of the hold.
Each hold keeps at most eight watched evidence tuples (LRU). No watch is armed or
used during a local pause; all are released when a pause begins and when their
hold retires. POSIX is unchanged.

Accepted consequences, each bounded by the 0.5 s backstop: changes that notify no
watched directory -- writes, security or timestamp changes and in-place reparse
points made by ANOTHER process through a handle opened by file id; data written
through a still-open handle until it closes; a hard link to a content file created
in another directory; a volume turning read-only. While watches are held, Windows
refuses to rename any ancestor of a watched directory, including from other
processes; the app's own maintenance releases them first.

Measured (native Windows, instrumented pause probe, same scenario): native opens
per warm Send 60,296/53,591 at the earlier receipt → 37,900/38,334 with the other
TASK-34601 changes and L4 off → 20,285/19,623 with L4 on; the probe's native-open
budget, failing before, passes.


### TASK-34601 amendment: finite stock MCP admission proof (2026-10-09)

Status: **Not adopted after qualification.** The following describes the tested
alternative, not the current admission contract. It reduced empty-scope native
work but did not establish a whole-Send benefit; original product/tests restored.
Full results and limits are recorded in the linked plan.

An exact installed stock pending MCP preparation may select its source once,
retain resident identity/config/closed/pause/custody gates through lock and member
setup, then perform one fresh full source and parent proof before publishing a
usable operation. Recheck issued attempt, binding/participant identity, closed/pause
and exact lease custody before yield. No proof callback may use an active operation
before this proof completes. Custom locks and supported changed callbacks preserve
their original route; internal private observers follow the revised boundary.

This revises pre-lock recovery refusal: a recovery change after initial selection
may now acquire the source mutex and create an inactive State before refusal.
Setup before the final proof is metadata/resource preparation only. Actual body,
read/write/default/corruption/backup/publication effects remain unreachable until
that proof succeeds, and every existing effect and final result-publication gate
stays fresh. Concurrent invalid conditions may report a different refusal first.
Exact retirement and uncertain native ownership remain unchanged.

The alternative of retaining four complete observations before an empty body was
rejected for the experiment because repeated admission work is measured and its
effect boundary can be stated directly. Merely deleting a check while claiming
identical refusal ordering remains rejected. Cross-call witness reuse, authority
caching and activating the operation before its sole fresh proof are rejected.

Plan: [finite MCP source admission](../../Docs/superpowers/plans/2026-10-09-finite-mcp-source-admission.md).
Task: TASK-34601 AC20. Preserves ADR225 and the existing pending acquisition owner.
