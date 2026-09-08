# ADR-126: Complete local backup and recovery

Status: Accepted — revision 4 approved by the user on 2026-09-07
Date: 2026-09-07

Revision: 4 — incorporates the fourth user-requested design review.

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
