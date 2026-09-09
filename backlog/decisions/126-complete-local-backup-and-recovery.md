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
