# Backup and recovery

This document describes the ADR-126 complete local backup and restore subsystem: inventory and data groups, admission/quiescing, capture and encrypted archives, replace/isolated restore with journaled publication and encrypted rollback, recovered-media review, and the startup fence. "Recovery" elsewhere (notes-sync crash recovery, server-sync account recovery, fleet worktree recovery) belongs to those subsystems' own docs.

## Authoritative files

| Concern | File | Key symbols |
| --- | --- | --- |
| Service | `Backup_Recovery/recovery_service.py` | `RecoveryService` — one worker composing all engines: backup/restore previews, inspection, extraction, rollback, recovery-copies, profiles |
| CLI + startup unlock | `Backup_Recovery/launcher.py` | `python -m tldw_chatbook.Backup_Recovery` (`inspect / extract / restore / recover / profiles / copies / consolidate-mobile`); `startup_preflight` / `startup_unlock` (master-password prompt, rekey, reset); `minimal_recovery` host |
| Admission | `Backup_Recovery/storage_admission.py`, `admission.py` | `admit_startup()` (called before runtime imports — the fence), `MaintenanceSession`, capture scopes/leases; nested admission forbidden |
| Participants | `Backup_Recovery/participants.py` | `run_finite_local_worker`, installed persistence participants, exact-repository-type → owner-id mapping, participant-coverage requirement |
| Inventory & groups | `Backup_Recovery/inventory.py`, `data_groups.py` | `discover` (no guessed targets on config failure), the 16 `BACKUP_GROUPS` and owner→group map |
| Capture & archives | `capture_service.py`, `capture.py`, `archive_writer.py`, `archive_reader.py` | coherent capture, strict-manifest archive publication, bounded inspection (`CompressionReviewRequired`) |
| Encryption | `crypto.py`, `age_worker.py` | package-owned **age v1**; isolated `python -I` single-scrypt pipe worker; no PATH lookup, no plaintext fallback |
| Restore | `restore_plan.py`, `publication.py`, `journal.py`, `replacement.py`, `later_rollback.py`, `isolated_restore.py`, `inert_extraction.py` | reviewed plans (restore/retire/preserve/issues), journaled publication ("never releases admission"), replace-mode with held original + encrypted rollback, separate-profile restore, inert bytes-only extraction |
| Credentials & copies | `credentials.py`, `rollback_credentials.py`, `recovery_copies.py` | credential-aware staged copies, `--acknowledge-credential-issue`, retained rollback copies |
| Post-restore gating | `activation.py`, `recovered_media*.py` | `execution_allowed` per-owner review gating (imported by app.py), recovered-media review (keep/delete) |
| Profiles | `profile_catalog.py`, `profile_open.py`, `profile_paths.py` | restored-profile locators, open/acknowledge, effective path resolution |
| Schema catalogs | `DB/recovery_core_schema.py`, `recovery_core.py`, `recovery_operations.py` | exact installed `sqlite_schema` SQL per core DB (validation compares actual schemas, not version labels); per-owner adapters; ~20 subsystem `recovery.py` modules feed inventory |
| UI | `UI/Screens/backup_restore_screen.py` (+state), `app_lifecycle.py` | the in-app Backup & Restore screen; `action_backup_restore` entry (Settings Storage + Overview); `request_recovery_restart` guarded quit → CLI restart |
| Legacy helpers | `Backup_Recovery/settings_backup_helpers.py` | the retired settings-backup UI's surviving contracts (named-DB manifest publication, character-card serialization) kept for tests/TTS-profile backup |

Control root is fixed at `<config dir>/recovery/control` (`service_storage.py`); there is no `[backup]` config section — destinations are chosen per operation.

## Backup creation (dataflow)

1. Settings → Storage "Backup & Restore" (or the Overview button) pushes the screen.
2. Preview → `inventory.discover` over the selected configs; group resolution via the owner→group map (unknown owners get no group).
3. `start_backup` → admission: a `MaintenanceSession` with participant coverage checks → coherent capture (SQLite through lock-safe checked capture).
4. Optional age-v1 encryption via the isolated worker.
5. `write_archive` stages payloads under strict manifest names and publishes atomically; publishing inside a selected external source requires an explicit exclusion.

## Restore (dataflow)

1. `start_inspection` → bounded archive inspection (groups, profiles, scope).
2. `preview_restore(mode = isolated | replace)` → a reviewed plan (`restore/retire/preserve/issues/safety_scope`); the CLI prints it and requires typing `restore`.
3. Replace mode additionally requires verified capability, an explicit `--target-config`, typed confirmation, and a **fresh rollback password**.
4. `start_restore` → journal evidence → `finalize_candidate` (fsync, retire originals); the original capture is held and an encrypted rollback archive created. Unselected stored data is **preserved**, not deleted (ADR-126 rev 4).
5. Post-restore: `activation.execution_allowed` gates features per owner until restored generations are reviewed; recovered media lands under `<user_data>/recovered_media` with a keep/delete review flow.

Interrupted operations recover through `recover <operation-id> --finish|--rollback|--abort` with typed confirmation; rollback needs the rollback password; retained copies are listable/inspectable.

## Startup fence and unlock

`admit_startup()` runs before runtime/config imports in `app.py`, `config.py`, `Web_Server/serve.py`, and the `Web_Server` and `RAG_Search` package initializers. Importing the `TTS` package is inert; its `__getattr__` calls the same fence before resolving a requested runtime export. In these entry paths, enrollment failure is a hard `SystemExit("Recovery required: …")`. When config encryption is enabled with a verifier, `startup_unlock` prompts pre-TUI: a wrong-but-readable password offers rekey; an empty one resets encrypted config values (keys removed; chats/notes untouched). Served children refuse the unlock with a plain-language error rather than hanging. Non-unlock recovery reasons get the minimal recovery host. A recovery restart (app → CLI handoff) checks no running operation and no unsaved editors, quits guarded, and spawns a fresh process into the minimal recovery host.

## Failure behaviors and boundaries

| Case | Behavior |
| --- | --- |
| Config parse failure at inventory | No guessed targets — discovery yields nothing |
| FIFO/non-regular config | Refused (non-blocking open, regular-file check) |
| Journal/publication crash | Admission never released by record labels; recovery resumes via `recover` |
| Restore without review | `execution_allowed` blocks feature execution per owner |
| Extract request | Inert bytes to a chosen directory — never profile/schema restoration |
| Fresh-install securing failure | `~/.tldw_cli-data` fallback (ADR-127) with a plain-language diagnostic |
| `run_finite_local_worker` cancel | Only that worker's file-backed caches retire; in-transaction/foreign handles survive |

## Governing decisions

ADR-126 (`126-complete-local-backup-and-recovery.md`, rev 4 with amendments), ADR-127 (`127-fresh-install-private-data-root-recovery.md`), ADR-125 (`125-lock-safe-private-sqlite-validation.md`). Specs/plans: the `2026-09-07-complete-local-backup-restore*` series, `2026-09-15-selectable-backup-groups*`, `2026-09-12-python-backup-encryption*` (under `Docs/superpowers/{specs,plans}/`). Tests: `Tests/Backup_Recovery/` (331 files).

## Verified gotchas

1. The schema catalogs are **exact-SQL**: requalification compares actual installed schema text, not version labels — relabeling is forbidden by the module header.
2. `run_finite_local_worker` closes only caches that worker acquired; exact repository classes confer authority ("subclasses confer no authority").
3. `settings_backup_helpers.py` documents a **retired** UI kept alive by tests and the TTS-profile backup contract — the user-facing surface is the Backup & Restore screen.
4. Restoring an old DB does not auto-activate its contents — imported history stays inert until fresh live claims are allocated.
5. The Settings Storage card warns that path changes apply on next launch (active DB handles keep current paths) — backup/restore works on the admitted set, not mid-flight handles.

## Related docs

- [database-layer.md](./database-layer.md) — the stores being captured
- [server-sync.md](./server-sync.md) — the separate server-sync account/key recovery
- [notes-sync.md](./notes-sync.md) — the notes-sync recovery executor

## Legacy mobile history preview (ADR-244)

`mobile_consolidation.py` observes the three exact legacy files under
`profile_paths.user_data_dir(config) / "mobile"`, independently of canonical DB
overrides. The service and `consolidate-mobile --preview` CLI return immutable
schema/count/identity/dependency metadata and capture/rollback scope. The read-only
private SQLite role uses native checked snapshots, exact catalogs and bounded
validation; it never constructs or reconciles a live legacy store. Unknown
content, unsupported schemas, missing references and unsafe aliases block review.
Physical dictionary/persona asset dependencies without a qualified legacy owner
binding block; they are never discarded or guessed from path strings. Historical
note file locators can remain admitted database metadata. Capturing those rows
does not capture the external note files or grant active Notes Sync authority;
actual startup candidate behavior requires separate qualification.
Multiple known configs sharing one physical source require destination review.
`identity_conflicts` lists exact overlapping destination primary-key tuples;
collisions require explicit mapping and never establish equivalent content.

Safe fixed source-claim/lock declarations produced by native mobile discovery
stay in the full inventory while remaining outside automatic payload-alias
grouping. Native creation therefore preserves their reviewed scope across
selector aliases. Other recovery controls and unsafe declarations retain ordinary
classification; live claim custody still rejects substituted objects.

The private captured-SQLite reader rechecks all fixed-owner alias declarations
against authenticated producer and member IDs, including dependencies and shared
groups. Included aliases require fresh native physical identity plus matching
captured size/hash before deterministic representative selection. All-unused
members retain absence; mixed or unproved records refuse. This changes member
selection only, retaining the reader's existing custody and retirement owners.

The existing chat, workspace and AgentRuns owners declare `:legacy-mobile`
logical records. Installed run-log discovery reads the explicitly reviewed legacy
stores for retained artifact roots. Normal maintenance, WAL capture, credential
policy, directory metadata and candidate validation remain authoritative. Legacy
review failures remain blocking even for partial capture. The original stores,
active chains, unfinished wake claims and undelivered results remain unchanged.

`CanonicalConsoleStores.identity` hashes native-normalized absolute target paths;
it is stable while expected absent parents or databases are created. It grants no
ownership, alias waiver or write permission. `destination_witnesses` separately
records each target and its complete ancestor chain as immutable tuples of
`(absolute_path, device_or_none, inode_or_none, mode_or_none)`. A maintenance-owned
apply must validate and journal every expected creation/replacement transition
before accepting new witnesses. Unexpected changes require review. The source
fingerprint includes strict physical evidence, admitted logical row digests and
referenced artifact digests. Preview itself creates neither source nor target
parents. No mobile import, profile cutover or authority activation occurs here.


Installed consolidation support qualifies exact Chat82/Workspace9/AgentRuns26
catalogs and their file-backed native migrations. Read-only validation still
accepts qualified Chat81/Workspace8/AgentRuns24 or25 sources, including every
previously admitted AgentRuns25 physical layout; migration creates a separately
qualified successor. The Task7 mobile fixture deliberately retains those legacy
source heads while canonical fixtures use installed heads. Candidate migration
authorization permits only the fixed owner tables/indexes and native version
stamps, and continues rejecting unknown catalogs. This installs receipts and
quarantine consumers; the preview still performs no import or cutover.


The Subscription database owner separates inert class/schema imports from its
actual threshold settings read. That read retains the native configuration
getter and admission; recovery does not need to initialize application settings
merely to import the owner.

## Local mobile consolidation service and CLI (ADR-244 R115–R121)

`RecoveryService.preview_mobile_consolidation(source_config, target_config)` and
`consolidate-mobile --preview --source-config … --target-config …` remain
observational. `start_mobile_consolidation(..., destination_confirmed=True,
rollback_password=...)` and the mutually exclusive CLI `--apply` enter the actual
local maintenance/capture/import path. The CLI obtains a fresh preview, prints
its fixed metadata, requires typing `consolidate`, then prompts for the encrypted
rollback password. This is a local operator flow; no HTTP migration endpoint or
mobile server startup cutover is installed by these methods.

Required canonical owners must already be capturable. For a new canonical
configuration, ordinary application startup with `--config PATH`, followed by
physical retirement, initializes the installed Core/Workspace owners before a
fresh preview. Configured storage parents must be valid and admitted. Start does
not silently initialize required targets from an earlier absence preview, create
legacy source databases, or waive Complete inventory. Optional canonical Runs
absence uses its installed journaled creation path. Missing required captures
refuse before preparation. Credential review failures remain blocking; an offline
profile selected before startup is a separate supported configuration.

`mobile_enrollment._start_session` keeps the complete initial observational
inventory, native namespace registration, all-overlap plus unbound exclusion,
strict current review, compatible profile publication, and strict postpublication
review in that order. New bindings cover actual per-selector native roots and
proved complete containers; existing bindings are not broadened. Selector bytes,
physical source and destination witnesses, the entire candidate cohort and user
scope must remain equal to the review. Changed scope returns a fresh-review
refusal before preparation. Native PreviewScope and held DiscoveryScope carry
pre-admission accounting and physical custody through their installed readers;
ordinary preview does not acquire maintenance or captured-input authority.

The public path finishes inside the same actual native coordinator frame.
`mobile_committed` binds current native owner receipts, original source/capture
references, canonical targets and exact pending publication. No-data completion
uses real native target ownership and retirement without fabricated import
receipts. The original source claim and stable flock remain held until the durable
terminal and exact witnessed pending retirement finish. A durable terminal with
interrupted pointer retirement still requires explicit recovery. Successful
owner transactions alone do not establish public completion.

`require_mobile_cutover(source_dir, CanonicalConsoleStores(...))` is an
observational ordinary-storage check. It requires the exact committed claim,
retained evidence, owner completion records and physical target continuity.
After a legitimate installed configuration-owner save, it accepts only a current
persisted native binding and unchanged source routes, stores and reviewed
per-selector candidate membership. Raw edits, new candidates, redirects,
substituted ancestors or changes during the check refuse. This postterminal
observer does not relax pending/import/finish/rollback selector checks or grant
any owner write authority.

`start_recovery(operation_id, action=...)` and CLI `recover ... --finish` resume
only the selected existing operation and direction. A missing pending pointer is
admitted only for qualified committed-terminal reconciliation. Pre-cutover
`--rollback` authenticates the original canonical archive with the supplied
password and restores its coherent SQLite snapshot plus authenticated metadata
and absence state through native owners. It retains displaced current objects;
it does not reconstruct uncaptured original WAL bytes/inodes or change legacy
source/capture bytes. Durable intent/observation/stage evidence reconciles rename
gaps. Rollback continuation keeps strict physical/selector/claim bindings while
the original source fingerprint remains an operation reference, not a claim of a
fresh live semantic review. A completed apply is not authorization for later
replacement rollback.

Materialization exhaustion uses the fixed `mobile_materialization_limit` message,
never arbitrary exception bodies. Preserve source and recovery files and inspect
the actual available recovery action. A failed first call and a later successful
explicit continuation are separate outcomes. Fresh standalone processes now
qualify first-call no-data CLI completion and cold config reconnect, plus populated
unbound enrollment with the original allowances unchanged after service close.
Their real HOME/XDG environment remains intact; historical pytest epochs retain
their separate sandbox scope. Ordinary same-target history/Goal/Discard consumers,
a mounted new manual turn, absent/present-marker Notes startup and full native
included-artwork restoration also have targeted evidence, detailed below.
Imported-history holds remain enforced throughout. Targeted integrations now
qualify the restored-generation observer and recovery under the current accounting.
The original-BASE review and scoped accounting correction review are complete.
The canonical mobile API now routes through the retained Console owner and
canonical read services; the separate mobile engine has been retired. Its original
ownership review and both scoped correction reviews are complete. The Chats-first
phone UI and full retained-child/browser qualification remain Tasks 10–11.


## Native mobile consolidation admission (ADR-244/125/126)

`MaintenanceSession.prepare_mobile_consolidation(journal, *, source_config,
target_config, selections, approved_scope, source_fingerprint,
destination_witnesses, destination_sidecars, reviewed_preview)` prepares an internal in-place
operation. It accepts checked config selectors and the reviewed native inventory digest and preview witnesses.
Both selectors must have compatible, positive ADR-126 enrollment; their own
selected roots must cover their respective source/canonical paths. The actual
session holds the unbound gate and every selected namespace exclusively after
cooperative OS leases retire. Missing enrollment refuses with close-old-instance
guidance; a schema, PID, heartbeat, advisory execution lock or newly registered
namespace cannot establish old-writer retirement.

Preparation rechecks source evidence read-only and writes `mobile_prepared` into
the existing private hash-chain journal, then registers the existing typed pending
record before any canonical write. The prepared evidence fixes selector paths and
fingerprints, inventory/source digests, bootstrap root, namespaces and the three
ordered native owner targets. Each target has an independent full ancestor/object
presence, device, inode and mode chain. The normalized canonical path hash remains
metadata only. Reopening requires the original review, exact pending/control
association and selectors, and current witnesses reconstructed only from journaled
native creation transitions.

`session.mobile_consolidation_owner(journal, owner_id)` yields one installed
ChaChaNotes, Workspace or AgentRuns owner (`db.chachanotes.primary`,
`db.workspaces`, `db.agent_runs`). Its private grant is bound to that exact live
session, PID, thread, repository instance/type, target, witnesses and journal
operation. Ordinary owner opens inside maintenance still refuse; capture still
reads source and writes only staging. Missing canonical parents and databases use
pinned native exclusive creation: `mobile_path_intended` fixes an operation-unique
private sibling, `mobile_path_staged` durably witnesses its empty inode and parent,
and native no-replace publication precedes `mobile_path_observed`. Recovery accepts
only that witnessed inode at exactly temp or final. Pre-witness objects remain
unknown, retained and refused. Changed objects and foreign operations refuse. This path does not adopt an
unexpected object or replace an existing object. Existing destination DBs and
their fixed SQLite sidecars use the qualified mode transition described below.

`mobile_owner_opened` and `mobile_owner_retired` bracket actual native handle
ownership. Scope exit closes physical SQLite handles before releasing the separate
AgentRuns execution descriptor. Close failures or retained execution users keep
the actual maintenance resources quarantined; observer cancellation is no evidence
of retirement. After process interruption the same opened owner can be reacquired
under native exclusion (and the exact AgentRuns lock) before new owner progress.
Neither journal labels nor retirement establish that an import batch committed.

RecoveryService derives the available mobile recovery actions from the actual
native journal and pending state. Durable pending records override cached worker
success and continue fencing startup, including corrupt or mismatched
journal/pending records. Public finish and pre-cutover rollback compose the
private provenance, captures, owner receipts and terminal checks described above;
a private owner transaction or retirement record alone never releases the fence.

Before native owner construction, `mobile_mode_intended` records an exact existing
owned regular file witness, and `mobile_mode_observed` accepts only the same
path/device/inode changed to 0600. All three DB owners use this transition;
AgentRuns first wins its stable execution flock. Interrupted transitions remain
pending. Fresh native exclusion can finish only the original inode at its reviewed
mode or journaled 0600 mode; AgentRuns reacquires its real execution flock first.
Foreign modes/inodes refuse. Source bytes and permissions are unchanged.

`destination_sidecars` is separate immutable preview evidence: for each canonical
DB it records exactly `-wal`, `-shm` and `-journal` presence/path/device/inode/mode,
bound to that DB's strict ancestor evidence and native parent qualification.
Preview refuses unsafe objects, physical aliases and name collisions, including
absent main/sidecar names. It neither expands the canonical namespace hash nor
changes the three permanent destination witness chains.

The native scope revalidates this fixed cohort before hardening and factory
handoff. Mode changes on an existing sidecar receive the same strict before/after
journal evidence as main files. SQLite owns subsequent optional sidecar generations;
they are not frozen as permanent DB objects. Bounded `mobile_sidecars_observed`
records observe the cohort after actual connection retirement, or after native
exclusion and the run execution lock are reacquired for an interrupted owner.
These observations permit same-operation reopening through the normal native
optional-sidecar policy; they never establish a committed batch.

Private SQLite helper wire protocol v2 adds a strict `require_private_mode` field.
Only the actual live bound mobile grant selects it in the connection path after
journaled preparation. The helper then refuses a needed mode change, including a
new sidecar generation, without calling `fchmod`. After helper preparation, the
native connection seam rechecks the exact reviewed/journaled cohort immediately
before SQLite allocation; an already-private replacement generation also refuses.
The helper does not create optional sidecars. Transient-generation handling starts
only after admitted SQLite allocation. Ordinary callers retain native hardening;
read-only `preserve_source_mode` keeps its original meaning. Old/mixed
wire versions and missing fields fail closed. This protocol is separate from
Canvas and HTTP protocols.


## Private mobile capture and recovery lifecycle

`mobile_consolidation_apply._captured_mobile(session, journal, preview,
approved_scope, *, destination, password, cancel)` supplies a concrete private
lifecycle to the fixed importer. `_resume_captured_mobile` reconstructs the
original checked review from typed records and revalidates it under fresh native
exclusion. The captured object's `consolidate(password=..., cancel=...)` now runs
completion in that same native lifetime; the public service owns the local entry
and recovery actions. Neither a captured object nor caller-supplied records grant
ordinary write access before durable terminal publication and pointer retirement.

The immutable preview includes every reviewed candidate selector, content digest
and strict physical witness. Native preparation rejects a changed candidate set,
config or source. Frozen preview values alone grant no authority: before durable
preparation, native selection re-derives and compares source/destination paths,
no-data status, source witnesses, schemas/counts/mappings/dependencies and capture
scopes. Separate explicit credential metadata may differ. On resume, only original
destination/provenance witnesses governed by the typed journal and advisory
collision hints are normalized; native source fields remain exact, and the existing
journal checks still qualify creation, modes and transient sidecars.
A durable claim binds the original source fingerprint, canonical
namespace and witnesses, operation and native control locator. A stable,
source-addressed never-unlinked flock remains physically held for the lifecycle;
another config/control cannot import that source under another operation.

Capture uses the complete installed adapter census. Mobile discovery uses the
existing bounded private SQLite read copies, preventing census reads from creating
live WAL sidecars after a strict reviewed-absent witness. Own-pending reads require
the actual live native session and exact typed operation/journal, selectors,
namespace and pending association. Ordinary, foreign, copied or retired scopes
retain the normal pending refusal.

Legacy source history/artifacts and complete canonical rollback are fixed native
partitions captured under that same exclusive session, without intervening owner
writes. This preserves the native archive's one-version-per-owner contract:
legacy81/8/(24 or25) and canonical82/9/26 have the same actual owner IDs.
Generic mixed owner versions still refuse. Each archive uses native encrypted publication and
verification, with typed intended/staged/verified identities. Interrupted
publication can finish only the durable verified ciphertext inode at its exact
temporary or final name. After authentication and the verified record, recovery
retires only the native writer staging directory with the exact durable parent
witness; foreign replacements remain untouched. Empty sources record a native no-data proof, create no
legacy database and have no source archive. C2 must verify purpose-specific rows,
references and artifact bytes before import and cutover.

`_preview_mobile_credential_issues` performs native disposable capture under the
same qualified session before actual capture intent. It returns bounded body-free
issues and grants no consent or publication. Explicitly reviewed manual-only issue
sets are immutable per partition on `preview.credential_acknowledgements`, typed
preparation and capture intent. Actual capture re-exports and checks exact issues
and captured records. `credential_issues` and `manual_recovery_required` on the
private lifecycle disclose the limitation; coherent archives do not establish
automatic credential rollback readiness. Destination confirmation never implies
credential consent. Missing, unreadable or unsupported required material refuses.
The existing generic excluded citation marker also refuses inside the exact native
mobile partition, including when another captured credential produces an explicitly
acknowledged manual-recovery issue. Such issues never skip mobile coverage
validation. Generic backup marker handling is unchanged.

ADR126's native unused-default citation rule is shared with the citation
provisioning invariant in `DB/citation_fingerprint_policy.py`. Only an exact
validated default reference with all fingerprint history absent and a positively
missing backend value may omit a record. Present values still require native
manual recovery review. Export and staged archive reference verification reject
omitted required keys and uncertain catalog/context/predicate/backend evidence.
No key is provisioned by capture or verification.

The returned `source` (optional) and `canonical` archive objects identify encrypted
native output files. Their `manifest_bytes` are authenticated native manifests;
consumers must use the native archive acquisition/decryption path before reading
payloads. `owner(fixed_owner_id)` opens actual installed owners only while the
same operation's source claim, archives and session remain valid. Owner closure
and descriptor retirement remain physical requirements; ambiguous close retains
the native maintenance quarantine. A prepared-only crash can register its exact
pending fence after full revalidation; source claims and captures do not establish
committed import batches.


Reviewed mobile source row content is hashed with native SQLite scalar type tags
in `mobile_consolidation._sqlite_row_digest`: NULL, integer, real (`float.hex`),
text, and BLOB (hex bytes). Thus a native `agent_runs.result` BLOB and text with
the same Python display cannot share a reviewed row digest. Unknown scalar types
refuse inspection. Metadata digests, namespace IDs, physical witnesses and capture
formats are unchanged. A pre-correction source proof that no longer matches is
refused by the native preparation/recovery review; there is no lossy fallback or
silent proof relabeling. Full mapped import and public cutover remain separate
qualification gates.

## Private imported-history file custody

`MaintenanceSession.mobile_history_files(captured, *, password, cancel)` opens the
private file capability for the actual live `_CapturedMobile` operation. It
rechecks the same journal, source claim, captures and original selector content
and physical witnesses. The single prepared-selector proof lives in
`control_records._mobile_pending`, reached through the captured claim and every
native mobile DB owner admission/transaction. It re-observes the complete reviewed
source/target/candidate cohort and refuses same-byte object replacement or a
changed cohort, including when the source contains no history files. The file
capability uses this shared proof; ordinary non-mobile admission is unchanged.
Copied, overlapping, mutated or retired scopes refuse.
The capability derives its only location from the canonical selector:
`user_data_dir(target_config)/agent_history/imports/<destination-identity>/<source-fingerprint>`.
Each authenticated source root contributes a SHA-256 root-ID component beneath
that location. Relative paths, empty directories, original framed-record IDs and
payload bytes are preserved. No caller supplies a destination or file list.

`publish()` returns immutable `_HistoryMember` associations (`logical_id`,
`root_id`, `original_root`, `relative_path`, `location`, `size`, `sha256`). These
are file contracts for the subsequent real owner mappings; they are not committed
DB receipts. `mobile_history_intended`, `mobile_history_staged`,
`mobile_history_observed` and `mobile_history_verified` separately record fixed
location, parent witness, operation-private temporary name, complete-tree identity
and content. Publication uses the existing qualified native flush and no-replace
rename. Same-operation recovery accepts the exact witnessed tree at temporary XOR
final location. Unknown pre-witness objects, occupied names, symlinks, foreign
replacements and conflicting evidence remain retained and refuse publication.

The new path reuses `_CaptureFileDescriptors` for extraction, staging and capacity
pins, recursive tree flush, native publication and qualification, artifact
observation, and private member reads. Optional private custody parameters on the
existing helpers leave ordinary callers' validation and publication rules intact.
Native per-component `O_NOFOLLOW` traversal stays intact; custody reuses close
bookkeeping, not bootstrap's trusted-system-symlink resolver.
An ambiguous native close is never retried; the existing resource owner retains
the maintenance quarantine. Successful method return requires resource retirement.
`read_member(logical_id, *, offset=0, limit=65536)` requires verified same-operation
file evidence, exact member identity and current full-tree/payload byte checks;
it returns at most 1 MiB. It is a maintenance-only reader, not runtime UI admission.

`Agents.recovery._RunLogs.discover_for_stores` owns the partition distinction:
ordinary canonical discovery (`stores is None`) includes `agent_history`; the
explicit legacy-store discovery used by `mobile_inventory` does not. This holds
when source and target selectors coincide and does not exclude caller-selected
paths or substrings. An existing imported tree without its canonical AgentRuns
owner is evidence of use and refuses the native unused-owner declaration.

The canonical archive is captured **before import**. A missing `agent_history`
root is represented by native producer-inventory `unused` evidence; existing
`restore_groups.selected_absent_target_ids` selects its subsequently discovered
descendants for exact local rollback. An existing tree is captured with ordinary
native directory/file metadata and bytes. B must qualify complete same-operation
rollback after file publication plus a real committed DB prefix using that
canonical capture and native witnessed ownership. There is no second restore
engine, name-only deletion, or inference that file verification means a DB commit.
Cross-profile `agents.history` relocation remains refused by native
`staging._config_targets` (`owner_relocation_unverified`). This increment does not
claim public terminal recovery or cross-profile restoration.

A-chat and A-runs still own actual Chat/Workspace/Run/Goal identity mappings,
committed receipts, all-owner preflight and complete reference/byte closure.
A-runs must connect the runtime history consumers to actual canonical receipts
and run/conversation ownership, preserve original-ID filtering and bounded paging,
and retain existing membership/tool gates. Imported history never grants scratch,
workspace binding, hook, tool or permission authority. Public apply, cutover and
terminal recovery remain separate B qualification gates.

### Private captured SQLite reads (R)

`_CapturedMobile.sqlite_reads(password=..., cancel=...)` issues the private
`mobile_captured_sqlite._CapturedSQLite` context from the actual live maintenance
operation. It requalifies fixed native partition declarations and exact logical
members, authenticates the encrypted captures, and extracts only Chat, Workspace
and AgentRuns snapshots. Legacy 81/8/(24 or25) and canonical 82/9/26 remain
distinct. Native installed catalog, version, integrity, foreign-key and domain checks run on
restricted disposable descriptor-bound connections before row access; no live
repository constructor, migration or destination connection participates.

Both fixed-purpose archives finish authenticated member extraction and final
sealed-byte verification before any retained candidate/row-reader SQLite
connection opens. Preparation retains
at most six member/candidate metadata jobs. Each readonly connection then starts
its original native restriction budget under the same custody and ExitStack;
archive decryption and extraction do not consume an already-open connection's
deadline. Scope, descriptor, named-parent, hash, schema and retirement checks still
run at their original boundaries, with a scope check before each open.

`versions` identifies the available fixed members. `page(purpose, family, table,
*, offset=0, limit=100)` returns immutable column names and native SQLite scalar
rows, bounded to 1000 rows and 16 MiB. `purpose` is legacy/canonical and `family` is
chat/workspaces/runs; these select already authenticated declarations and grant
no caller-supplied owner or path authority. The scope owns all connections and
rechecks live native authority at the read boundary. Before and after initial
validation and each page, it hashes the retained candidate descriptor actually
supplied to SQLite and verifies that strict nofollow traversal still names that
same file. An authentic replacement pathname cannot mask a changed retained file.

Optional native archive/crypto custody reaches acquisition before yield,
including the packaged helper, child pipes and threads. Custody-mode plaintext
allocation and no-clobber publication use the exact retained private parent
through descriptor-relative native operations. Parent substitution prevents
retirement and retains the actual parent pin; cleanup never adopts that foreign
replacement. Ordinary helper defaults remain separate. Candidate and installed
reference connections use the existing capture lease. Uncertain retirement
retains staged material and real maintenance exclusion; it never retries an
ambiguous descriptor. Existing journal/bootstrap/selector guard lifetimes remain
their native owners' responsibility and are not a new all-control-FD guarantee.

### Private typed planning (I)

The same authenticated reader exposes a no-argument `schema_catalog()` that
batches immutable purpose/family/table/column metadata, with complete scope and
retained-object checks around the operation. Its native cancellation, step,
deadline, catalog and custody limits remain unchanged.
`mobile_plan_manifest.py` assigns every exact installed table and column a fixed
handler/disposition; populated unsupported obligations and required pending
handlers refuse completion. The 230 literal table specifications cover the union
of the admitted versions, rather than an assumed count for every partition.

`identity_plan()` reserves immutable typed identities across the actual captured
Chat, Workspace and AgentRuns members. Unused genuine IDs survive; collisions
use native-valid deterministic IDs. Global memory/fleet sequence positions retain
complete source order after the canonical allocator frontier. Profile-qualified
citation composites remain original typed dependencies pending T's native
namespace proof. Authenticated optional canonical AgentRuns absence supplies
empty occupancy, without creating a target.

Legacy AgentRuns24 has exactly one native version-bound absence:
`console_command_receipts`. The private `_runs24_receipts_absent` predicate uses
the actual active scope's native-validated version/member plus retained-candidate
resource/hash checks. Manifest coverage excludes only that exact absent table;
typed identity and A-runs history enumeration consume it as absent. Physical
`schema_catalog()` metadata remains unchanged and public `page()` still refuses
the nonexistent table. AgentRuns25 receipts remain required real history. Unknown
catalogs, unexpected24 tables, missing25 tables and declared/observed version
mismatches still refuse. No source bytes/schema/version are migrated or repaired,
no table/row is synthesized, and whole optional-member absence stays separate.

`workspace_plan()` preserves original rows and projects named Workspace IDs,
collision-safe names, memberships, RAG note references, settings and exact folder
binding locators/metadata through those shared reservations. Default selection
remains canonical. Default consent and RAG sidecars are compared in both
directions, including native disabled/unscoped absence and ordinary-empty RAG;
equivalent source rows never replace canonical Default sidecars. Malformed
binding metadata, ambiguous Default content,
unproven external Media references and nonempty handoff manifests produce exact
body-free blockers. The DB-wide assistant-default backfill frontier is not copied;
completion-state differences block only native-eligible imported Workspaces.
Binding holds are a proposed set, never persisted or activated by this plan.

`authority_reference()` proves existing Persona/permission-profile references
using the same canonical captured raw owner/member and complete bytes, installed
strict semantics and actual source/target ownership plus generation/activation
provenance. Selector payloads are decoded by the native pure config parser only
after bounded, pinned, custodied byte acquisition. It imports no raw authority.
Newly owned streams, members and parent
pins use actual custody; inherited native control readers retain their existing
lifetimes. Evals' private passive metadata path recognizes only the actual issued
mobile discovery scope for current `recovery_pending`; ordinary startup and
profile-launch consumers still refuse that fence.

Content, normalized trace, Sync and AgentRuns planning compose the fixed native
plan consumed by `mobile_native_operation`. Planning creates no imported rows,
authority holds, batch receipts or published history and confers no public
apply/cutover success. The complete composed plan is checked before owner writes.

### Private content graph planning (C1)

`content_graph_plan()` on that same captured reader projects the fixed core
content and association graph, message bodies/embedded attachments/branches,
Canvas revision graphs, and response-rule backing records. It accepts no paths,
SQL, callbacks or writer. Original rows remain separate; only declared typed
references relocate. Native `(note://...)` target capture groups map through the
Note identity reservations, preserving display text and every other body byte.
Native backlink regeneration must retain the mapped meaning on the next save.
Historical Notes locators/hash/state accompany the complete captured DB body and
do not cause filesystem probes or activation; actual dictionary/visual file
requirements remain guarded.

Native unique names and organization sync identities are preserved, with exact
conflict blockers instead of merging or renaming. NUL-bearing source names in
SQLite NOCASE domains refuse as unproved comparison inputs. Managed Note folder
owner protection and unproved global rule scopes remain exact dependencies.
Canvas projections retain immutable source/hash/sequence/origin facts and expose
native `CanvasImportBatch` consumer inputs. Rule definitions, validation,
fixtures, drafts and historical source/digests retain their meaning under the
installed typed readers. The pure imported-state projection follows
`prepare_imported_rules`: enabled source bindings become disabled with the native
revision increment, and pending assessments become cancelled in both state
representations. Canonical owners and original snapshots are untouched.

This is a partial plan. Conversation structures are inputs to C2; every final
conversation field and the five named message metadata/generation fields remain
pending C2. Dictionary attachment/unresolved rows require the actual C2 metadata
and native derivation proof before use. C3 must supply native memory/context
proofs and compose complete C. `require_complete()` refuses; T, S, A-runs,
all-owner/integrated review and owning writes remain separate required gates.
Later startup qualification must exercise Notes' actual paused/review migration
and cutover-marker behavior. These private projections do not authorize import,
rule execution, folder access or public cutover.

C1 currently refuses the ordinary collision between two constructor-seeded
`Default Assistant` character cards: the installed character-name BINARY UNIQUE
constraint has no approved consolidation seed-reuse exception. Distinct-name
private fixtures prove supported transformations only; they do not prove an
untouched default-profile happy path. Captured originals remain intact. Public
cutover guidance must qualify a supported preoperation conflict-resolution path
before presenting this case as migratable; the planner never renames or merges
cards. Historical non-NULL Note locators also retain the installed BINARY partial
unique-index conflict behavior, including deleted rows, without probing paths.

### Private conversation settings planning (C2a)

`content_settings_plan()` consumes the captured C1 graph and its typed identities.
It validates all conversation settings, the five pending message metadata/state
fields, and capture/library policy rows through fixed installed readers. Native
metadata namespaces and discriminated message variants reject unknown meaning;
provider/thinking bodies and prompt text remain opaque. Typed workflow, actor,
visual, binding and authority references carry explicit successor obligations.
The combined dependency closure rejects settings whose C1 inputs were blocked.

Local character IDs and decimal assistant IDs map together. A non-NULL local
assistant authority remains an explicit pending column with an original-value T
proof slot, absent from projected values until both native contexts are proved.
Explicit NULL stays NULL; remote identities keep their namespace. Persona
references require actual captured authority coverage and a later C2c proof.

C2a also retains the native metadata-to-dictionary attachment/unresolved derivation
for C1's owned index rows. Live video and Media references require exact physical
member proof; the plan does not invent an asset reader or publisher. This remains
a pure partial projection: `require_complete()` always refuses pending C2b/C2c,
C3, T, S and A-runs obligations, including when their source families are empty.
There are no imported rows, instruction activation, dispatch, holds or receipts.

### Private workflow ledger planning (C2b)

`content_workflow_plan()` consumes the captured C2a settings plan, C1 dependencies
and I identities. Fixed native handlers cover the 76 columns of dispatch
checkpoints, hook and machine-followup receipts, fleet progress, generation
metadata, trajectory metadata and transcript annotations. Declared references
map through their typed domains; provider parameters, prompts, quotes and message
bodies retain their meaning. Receipt operation/turn/settlement/chain identities
remain opaque host history. Native unique-key collisions refuse explicitly.

Checkpoint admission uses the captured active branch and native owner/serializer
predicates. Valid on-path rows retain pending import-quarantine obligations.
Active continuation beside a checkpoint, terminal owners, stale versions/states
and stranded pointers produce exact prewrite conflicts with originals retained;
the planner never reconciles, deletes a checkpoint or invents a historical
version. Checkpoint-free accepted/dispatched and valid continuation history is
supported separately. For checkpoint-free conversations, multiple nondeleted
assistant owners with active continuations on the captured selected path produce
`duplicate_active_path_owner` before any C2b proof is discharged. Active
continuations on other historical branches do not count toward this conflict.
Stored machine receipts omit acceptance-time source-message
versions: their historical ancestry is readable, but no fresh admission proof is
reconstructed from the current message version.

Trajectory handlers recognize fixed native event grammars and declared retrieval
references. A capture-failure hash remains associated with its original source
event; absent source events do not permit rehashing or fresh verification claims.
Unknown grammar, ephemeral turn associations and unproved physical/dependency
references refuse exactly. Transcript message anchors validate conversation
membership even when the separate message-id column is NULL. Fleet sequence
mapping consumes I's global ordered frontier, preserving delivery quarantine.

The immutable plan exposes retained source rows, body-free blockers, provenance
and explicit successor proofs. C2a fields retain their ownership; unresolved C2c,
C3, T, S and A-runs work remains noninsertable. `require_complete()` always refuses,
even for empty remaining families. This API performs no owning-row writes,
dispatch/recovery, holds, receipts, file publication or instruction activation.

### Private logical actor planning and complete C2 composition (C2c)

`content_actor_plan()` consumes the real captured workflow plan and completes the
246-field C2 ownership matrix. Fixed handlers transform the 116 columns of the
portable Actor registry, Buddy profiles and bindings, Persona Visual graph, and
shared Visual Identity graph. Native decoders validate original and projected
associations; only declared typed references change. Original manifests, hashes,
portable UUIDs, revisions and historical source associations remain retained.
Generation provenance maps only declared references and revalidates the native
manifest identity, with the exact source JSON retained.

Logical visual rows retain exact pending physical asset/manifest proofs. The
current native legacy asset admission guard refuses populated asset stores before
captured planning; data-only logical tests do not establish captured asset support
or physical UI readability. Persona references use the existing captured authority
API and retain its member/provenance digests. That API does not prove a requested
Persona revision, so revision-bound bindings remain noninsertable dependencies.
No new capture owner, reader, publisher or live configuration authority is added.

Complete C2 carries C1 transitive blockers and the C2a authority slots forward.
A nonNULL local character authority remains an omitted, pending T column with
its original association; genuine NULL remains NULL. Dictionary proof discharge
rechecks the retained C2a native derivation against original metadata and source
indices, plus mapped metadata and projected indices. C2c successor discharge
preserves native actor and historical emote associations: actor-only fallback
needs no selected asset, legacy expressions must belong to their recorded
character, and selected visual assets must match their exact pack/version graph.
A no-selection snapshot may retain a matching pack/version pair without claiming
asset proof; partial pairs and cross-owner expressions refuse. The immutable
plan still always refuses C3, T, S, A-runs and integrated completion, including
empty families. It cannot authorize writes, holds, receipts, dispatch or activation.


### Private generated memory and complete C composition (C3)

`content_plan()` composes the actual C1, C2 settings/workflow/actor and C3
projections into the fixed 61-table/580-column ownership contract. The final
97 C3 columns cover context policy, auxiliary accounting, generated memories,
scopes/selections and the five character-search families. All original rows
remain private alongside mapped rows; original prefix, prompt and unit
associations are retained independently of the required mapped proof. C1's
relocated Note content remains the final body consumed by later native frontiers.

Memory mapping checks the exact source lineage, selected variant, attachment
bytes and native unit provenance before producing the mapped native persisted
prefix digest. Runtime and persisted source digest forms are checked separately.
Native manual continuation snapshots use their fixed tool-call/result identity
grammar and actual retained checkpoint bodies; provider call IDs stay opaque.
Changed/missing historical inputs refuse at the exact memory dependency rather
than synthesizing a proof from a digest. Sealed prior-range references retain
revision/scope ownership and ordered coverage, cycles refuse, and invalid scope
or memory dependencies close over selections. Global selection sequence mapping
uses I's reserved destination frontier; reset/history status and accounting are
preserved without replaying a call or selecting anything in a live session.

Character-search source rows contribute typed digests and captured-manifest
associations, never imported index/state rows. Destination singleton authority
and native monotonic/index effects remain destination-owned; unresolved source
or destination builds refuse without expiry repair. Existing legacy summary
text and mapped boundary associations discharge the exact C2 successor.

The full C value remains unconditionally noninsertable, including empty families:
T, S, A-runs, physical/revision/authority dependencies and integrated/write gates
remain explicit. It grants no rows, receipts, holds, publication, dispatch or
public cutover. Ordinary seeded-default Character name collisions still refuse;
renamed isolated positive fixtures do not establish an untouched-default import.

Each C builder requests `content_schema_catalog()` with its nonempty unique
tuple of literal installed Chat C table names. The live reader queries exactly
those source and canonical table columns; `schema_catalog()` retains its full
catalog behavior. Full schema/version admission at open and all retained-member
pre/post hash, name, ancestor, claim/journal, cancellation and resource checks
remain in force, including the unqueried Workspaces and Runs members. This
removes repeated irrelevant metadata queries across the five nested C builders;
it introduces no cache, reader renewal or change to the original SQL budgets.


### Private normalized trace planning (T)

`_CapturedSQLite.trace_inputs()` and `trace_plan()` accept no caller paths, SQL,
callbacks or authority. `mobile_trace_inputs.py` uses the existing retained native
connections through a read-only transaction adapter, including the native trace
reader, already-normalized legacy reader, citation context validators and local
character repository. All six members retain their complete before/after custody,
metadata, name, scope, journal and resource checks; cancellation and resource checks
also occur between bounded tuple batches and native calls. No connection is reopened
and no SQL restriction is relaxed.

`mobile_trace_plan.py` assigns all 34 T tables and 331 columns their fixed normalized,
operational, GC or citation disposition. Original rows and bytes accompany the
mapped rows. Only declared typed references and the native header
`adapter_defaults_json.system_composition` revision grammar relocate. Materialized
envelope, artifact, audit and legacy run-tag bytes remain exact. Native artifact
length/digest, frozen policy, component and state validators qualify their respective
fields; per-call closure additionally covers reverse-owned responses/events, header
components, surface prefixes/replacements, semantic live messages/attachments,
policy bindings and frozen masks. A newly lost input produces an exact call blocker;
original native unavailability and uncertainty remain unchanged.

Opaque nonempty trace run groups have their own deterministic reservations, separate
from AgentRuns. Historical semantic message/conversation locators share live identity
reservations and include canonical historical occupancy. The actual native viewer
projection checks mapped semantic keys, selection, duplicate behavior and order;
unpreservable ties and canonical idempotency-key collisions refuse the affected
input. Keys are never rewritten or deduplicated. Operational epochs remain canonical;
source provenance survives without adopting maintenance authority.

`mobile_trace_authority.py` discharges only the exact accepted C local-character
slot after captured source ownership, actual joined-row predicate, full caller
eligibility and accepted mapped conversation/card/reservation proof against canonical
context. Owned inactive history remains inactive; foreign or missing ownership stays
blocked, and explicit None remains None. No canonical identity is overwritten.
Citation SQL no-fingerprint-history is only an inert-context disposition: it proves
neither credential UNUSED nor backend coverage, key readability, namespace adoption
or permissions. Populated unproved citation composites retain exact blockers.

`console_trace_read_budget.py` supplies one private nonrenewable 16 MiB allowance
across the captured source and canonical trace input/proof/materialization operation.
The same object passes through the actual native read-used codecs, semantic projection,
redaction, legacy and reconstruction paths. Reservations precede decoding, cloning,
base64 and repeated-reference expansion; exhaustion is a distinct body-free exception
that cannot become historical uncertainty. Ordinary callers retain their default
unbudgeted behavior. This representation allowance is not a heap measurement or a
large-profile capacity promise, and all existing SQLite limits remain in force.

Independent disposable native-reader tests qualify mapped output after capture
retirement and restoration of private test isolation. Their native constructors do
not prove import-write authority or preservation of constructor-generated audit
creation timestamps. An eventual owning writer must also qualify native artifact
reuse behavior when identical bytes need distinct planned IDs. The current plan
preserves every timestamp consumed by readers and never inserts rows. The standalone
T plan remains noninsertable; its fixed S/integrated composition is described below.
AgentRuns, owner writes and public cutover remain separate gates.

### Private fixed sync and integrated planning (S)

`_CapturedSQLite.sync_plan()` composes the native trace/content plans with the eight
fixed `sync_log` columns and nine installed entity frontiers. The current proof is
planned for the native target INSERT; captured current rows are never replayed as
additional target proofs. Live messages above version one require their unique
immediate predecessor, using the native continuation, generation-state, thinking
and envelope hash rules. Deleted owners retain body-free tombstones and their
captured base hash, or the explicitly marked native legacy reconstruction. No
proof recreates deleted content. Conversation archive-only bumps retain their
last emitted source frontier; ambiguous current proofs, missing predecessors,
foreign payloads and surplus retained rows refuse their exact owner. For the three
latest-only entities, the selected log row's actual version must match the native
owner (literal one for world-book entries); the planner never repairs a foreign
proof version. The conversation archive exception remains separate.

Native hard-deleted `world_book_entries` retain each original proof and parent
association. Validated integer IDs from live rows and body-free tombstones in both
partitions share one reservation domain. Only identity occupancy is deduplicated;
repeated deletions and a recreated live ID do not merge proof rows. Every tombstone
needs its accepted mapped `world_book` parent. The fixed native payload grammar,
CAST-as-text identity correspondence and signed SQLite bounds reject malformed
history before any mapped output.

The outer S or integrated operation owns one existing `_TraceReadBudget` and
passes that same instance privately through Sync and T. Its remaining allowance
covers S accumulation, decode, hash and serialization and every newly introduced
integrated metadata transform, index/container, fixed-point pass, repeated input
projection and result. Reservations precede expansion, including the integrated
A-duty row accumulation; existing R/I/C callers retain their original behavior.
Standalone Sync and trace planning still construct their ordinary budget. The
returned immutable plans expose no budget or renewed allowance. This bounds the representation work receiving that
allowance, not all pre-existing R/I/C allocations or Python heap usage; the original
SQLite cancellation, deadline and step bounds independently remain in force.

`integrated_plan()` obtains its own actual S/T/C plan and composes the fixed manifest
and Workspaces. It passes the exact identity object from that active chain into the
private Workspace builder, which validates its actual type, source fingerprint and
both authenticated manifest digests against the current capture. The public
no-argument Workspace builder retains its full native identity construction. There
is no cached plan or caller-supplied identity argument on the captured API. All
Workspace table reads, native defaults, raw persona/permission references, binding
checks and six-member custody checks remain in place.

The combined plan retains originals, field ownership, exact refusal records and
mapped rows. Native T proof fills only C's pending local-character authority field.
Fixed historical metadata closure checks the accepted Workspace binding, Canvas
revision/message association and original opaque Canvas token, or the accepted
trace call turn and its typed message map for a fork alias. These historical facts
confer no execution authority. Source Sync control, routing, grants, cursors,
receipts and reconciliation state remain the manifest's populated refusals.

T retains each qualified call's exact original input incidence, including visited
messages and attachments, as immutable tuples bound to its original/mapped call
and both authenticated manifests. Retention is charged before accumulation on the
same T/S budget. Final composition compares those inputs to the actual qualified
T/C projections; only the separately checked fixed fork metadata transformation
may change a projected value. Missing, ambiguous, lost or changed inputs refuse
that call without rejecting unrelated calls. Original absent historical inputs
remain absent; incidence does not create a new requirement or replay a native graph.

Canvas closure uses C's actual native batch and complete original revision
ownership. A later lost revision removes its document and dependent cards/hints;
unrelated complete documents survive. Integration repeats semantic, forward and
native-input closure to a fixed point. It retains only discharges and Sync
frontiers whose exact owners remain accepted, while keeping original proof
provenance. S and integrated planning recheck all six captured members and their
resource restrictions before returning. Nested component plans remain diagnostic
proofs, never insertion capabilities.

C-qualified dispatch checkpoints, hook/machine followups and fleet delivery rows
retain their exact known quarantine requirement and an unresolved per-row native
owner duty. This preserves the mapped projection and its dependencies; it does not
prove that a native quarantine consumer has run. Other unclosed S semantics still
refuse, including an unknown requirement or a known tag on the wrong table.

Every S and integrated `require_complete()` unconditionally refuses. Typed pending
duties identify native Run/Goal handlers, all-owner preflight before any file or data
effect, complete reference/artifact closure, atomic binding holds and actual passive
workflow quarantine, native writer lifecycle/creation timestamps/distinct artifact
IDs, receipts and public cutover. A quarantine tag is an obligation, never evidence
of native consumer behavior. Public restore must still qualify durable native Sync
continuation, deletion privacy and zero provider/tool dispatch. These planners
perform no imported writes, publication, holds, receipts, provisioning or activation.


### Pure Run/Goal all-owner preflight

The actual issued captured SQLite scope also exposes the fixed no-argument
`runs_preflight()`. It composes the original integrated proof with fixed A-runs
transformations in `mobile_runs_preflight.py`, `mobile_runs_goals.py`,
`mobile_runs_snapshots.py`, and `mobile_runs_history.py`. The outer immutable
projection retains the exact original integrated/I/S proof objects as diagnostics;
its additional native uniqueness domains never replace the original reservations.
All six captured members are checked before and after composition. The original
16 MiB materialization allowance is shared through integrated planning and new
Run/Goal, operational-history, identity, history, and closure work; it is neither
renewed nor a measurement of Python heap usage.

Goal requests use native typed JSON validation and canonical serialization after
mapping their actual Workspace/binding references. Request hashes are recomputed
for that exact transformed request and both hashes are retained. Checkpoint
fingerprints of full original iteration results, artifact review digests, evidence
manifests, verifier hashes, bodies, accounting, and delivery/status fields remain
historical facts. They do not prove current files, permission, freshness, or result
approval. Removed Goals retain their erased request and non-resumable accounting.
Native Goal-coupled `goal:<id>` work-submission identities remain coupled to the
mapped Goal; arbitrary opaque launch/submission strings are not parsed as IDs.
`mobile_runs_goal_budget.py` admits installed model defaults, derived roots/sets,
path representations, internal canonical/report serialization (including nested
reports), and caller clones/encoding before native validation or serialization.
Native models remain the acceptance and output authority.

Source runtime-owner and command-receipt rows are retained separately as original
operational history. The canonical singleton is unchanged. Original command
owner epochs, clocks, sessions, targets and fingerprints never become current
command tickets. A later owner must provide passive historical storage and exclude
these records from active receipt capacity, invalidation, pruning and dispatch.
Definitions retain their exact names, enabled/deleted state and content; live-name
collisions refuse. A Run's historical definition name/fingerprint does not prove
identity with a currently installed definition or grant its advertised tools.

Private RunLog qualification consumes the scope's own sealed source capture under
its existing custody. Its fixed source-selector path preserves native owned-file,
nofollow, original preview fingerprint/witness and 1 MiB admission, including
ordinary 0644 modes. The same budget admits the bounded read before it occurs and
TOML lexical/container/implicit-prefix expansion before the installed decoder;
model clones are separately charged before construction. Checked configuration
is parsed once, with the installed native RunLog name/fallback function shared by
discovery and proof. No caller selector or cross-operation cache is accepted.
It checks every selected member's CRC/hash/size/topology and complete native record framing, including trailing bytes, and proves recorded Run
ownership from actual captured parent/conversation relations. Raw bytes and their
original IDs remain unchanged. Partial, foreign, unknown or ambiguous histories
produce exact prewrite blockers. This seam neither publishes files nor relaxes the
separate HistoryFiles/SQLite lifetime exclusion. A RunLog digest cannot prove Git
objects or worktree authority.

Header admission accounts for native split/partition tokens and dictionary fields
before the decoder. File-owning primary roots remain separate from framed Run
membership: empty segments and segments containing only descendants retain their
actual owner. A MANIFEST supersession declaration can prove only a recorded
historical reference to a distinct source primary in the same conversation,
including a still-running prior Run. It proves no successful supersession or file
ownership. Declared segments must be regular native `logs.[0-9]+.txt` members with
independently qualified framing; a stale manifest need not enumerate every log.

Retained change notes may legitimately reference snapshots already pruned by the
native retention owner. The outer snapshot domain reserves both partitions' live
IDs and retained non-null locators, preserving NULL and exact per-note provenance.
It creates no synthetic snapshot. Collision with an existing base target refuses.
The actual `sqlite_sequence` entry and required mapped frontier are provenance,
with an unresolved native allocator admission duty before later effects. No source
sequence is copied. Self-contained note excerpts survive; full diff/revert access
still requires the exact original snapshot/root/Git bytes and native availability
semantics. Physical worktree dependencies without captured proof refuse.

Outer closure includes actual Run/Goal dependencies and the original qualified
trace/Canvas inputs. Losing an originally present required step or Goal child
removes its owning candidate and propagates through the existing fixed-point
closure; originally absent/partial history is not fabricated. Final Sync frontiers
and semantic discharges retain only accepted owners. Final qualified history,
its owner/segment dependencies, operational mapped references and row duties close
against those same accepted owners. Separate historical observations retain source
facts with mapped associations cleared; global and passive retention duties stay.
Shallow observation/operational replacements admit their expanded field and
constructor dictionaries, result fields, and surrounding closure collections before
construction. Historical bodies remain shared references during this accounting.
All native holds, passive
history, current binding/authority/advertising gates, file access and atomic
rows/maps/receipts remain explicit pending duties. `require_complete()` always
refuses, including on saved or replaced plan data. This foundation performs no
imported writes, file publication, hold/receipt installation, provisioning, provider
dispatch or public cutover. Every later writer must acquire a fresh complete
all-owner preflight before its first file or data effect (ADR-244 R98–R104).

The private Run preflight selects native RunLog directory names once. It charges
the selected environment/config value before native `str`/`strip`, component and
set construction, including native TOML scalar/container representations, then
passes the same immutable names and live captured source scope through both the
nested mobile preview and inventory discovery. Native default discovery keeps its
ordinary behavior. These local names are data; source capture, native topology
and session custody remain the read authority.

## Same-operation native input and reader retirement (ADR-244 R107)

`_CapturedMobile.consolidate(password=..., cancel=...)` is a private synchronous
coordinator. A denial-only actual-session journal association prevents a second
attempt or reissued lifecycle from reopening readers or renewing the allowance;
fresh native maintenance admission is required after the session ends. It derives
complete native input internally with `_plan_runs` and one original `_TraceReadBudget`, retaining cohort, schema/member/manifest and immutable
row proof references only in its call frame. No saved plan or returned readiness
object can be passed to an owner. All pure `require_complete()` guards still refuse.

`mobile_native_operation` checks actual session scopes, retained resources,
uncertain-close flags and staging retirement before native owner admission can
create paths or write owner-open journal events. The original source claim and
maintenance locks stay held. `_CapturedMobile.owner` and the existing native
owner admission share this check; the native `_MobileScope` also retains the exact
captured lifecycle and revalidates it at subsequent native transaction admission.
Selectors, source location, target physical witnesses, capture artifacts and
cancellation are checked through existing native owners. This authenticates the
prepared snapshots under exclusion; it is not a fresh live-row fingerprint read.

R107 established the same-operation coordinator and reader-retirement boundary;
the later fixed-owner delivery implements the native writers and consumers inside
that coordinator. `_consolidate` derives and qualifies the complete supported
Chat/Workspace/Runs input, references, lifecycle, consumer duties, receipts and
prospective file readability in one frame and on one original allowance before
its first publication or owner effect. All native readers retire before publication
and owner admission. Pure `require_complete()` guards still refuse; saved plans and
readiness objects remain non-authoritative.

The fixed Chat, Workspace and Runs transactions atomically commit rows, typed
identity maps, body-free provenance, exact authority holds and separate completion
receipts. Runs also commits the passive `history.command` master/children, and its
receipt validates that complete component. Recovery reads actual committed receipts
first: a valid receipt skips exactly, a qualified absent receipt may be written,
and partial or mismatched state refuses. Current consumers enforce definition
advertising filters, imported checkpoint quarantine with explicit Discard only,
Fleet/delivery exclusion, Goal and chain mutation holds, local Goal payload privacy,
and ordinary RunLog access through receipt-bound native roots/members and original-ID
filtering (see `Backup_Recovery/mobile_native_operation.py`,
`DB/mobile_consolidation_receipts.py`, `DB/mobile_imported_run_log.py` and
`DB/mobile_imported_goal_privacy.py`).

A standalone Workspace fault check before R130 exited after its actual transaction
committed and before owner retirement. A fresh process resumed the original
captures, skipped all committed Workspace insertion, and verified exact receipts,
rows, the nonempty binding/exclusion and its unreviewed authority hold. That
historical private preterminal proof does not establish current-code recovery,
public completion or graceful teardown of the deliberately terminated process.
Fresh private recovery under R130 currently exhausts the unchanged native 16 MiB
allowance at final receipt materialization; its early terminal check returns
successfully. The failed operation remains preserved. Both production migration
entry points use the public coupled pair. The fresh R145 public Workspace proof
now exits after the same durable commit and finishes the same operation in a new
process, preserving exact receipts, Workspace rows, exclusions, holds and captures
without replay. It does not establish private-path parity or graceful crash teardown.

The local public coordinator now composes terminal verification and pending
retirement around these native owners, as described above. The current public
accounting envelope is 4 GiB, with the original native 16 MiB sub-bound.
Explicit human approval resolved the historical R129/R130 authorization blockers.
The direct-restored-generation observer and early terminal representation pricing
are implemented and pass their focused public integration checks. R146 explicitly
changes the complete ordinary cutover coordinator to one nonrenewable 64 MiB
allowance (512 bytes reserved at issuance); generic trace/Notes and private native
16 MiB limits remain unchanged. Late charges, source and execution holds remain. R131 readonly builtin preservation and R134 native capture-source
namespace composition received specific human approval. Their native component
checks pass. R138 factors duplicate fixed-mobile verification with native
component and generic-default controls. R139 adds a local reusable buffer for
explicitly budgeted native digest reads; R141 reuses only that byte workspace
inside one native capture scope, with unchanged per-file verification and limits.
The fresh R141 integration completes ordinary and held later preview, all three
source-verification phases, and native source-scope retirement. It reaches the
staging-entry sentinel exactly once with no retained resources, current capture
scope or SQLite staging directory, and no nested allowance. Target main/WAL/shm
identity, metadata and bytes remain exact from stage entry through final cleanup.
A separate full integration establishes actual `stage_restore`, including native
preserved-builtin candidate and dependency validation. Its first attempt then
stopped at the prepared source-binding guard before SafetyCopy capture. R142 now
keeps original profile containment first and admits only authenticated unchanged
preserve-only builtin members through an existing held native covering namespace
(`mobile_builtin_preservation._mobile_safety_members`). Each of the three native
checks freshly authenticates the source journal, archive, current binding and
member witnesses on one original 16 MiB validation allowance.

R143 holds each member parent through its current-name reopen within that one
check; every leaf stat and expected manifest frame remains. R144 checks each
distinct existing write Path once per check while retaining all logical owner
mappings and restore effects. These local representations retain native exclusion
and later source/target checks; they do not promise global filesystem atomicity.
Genuine components now complete all three checks with exact and broader,
unprefixed, multi-root namespaces, retire native resources and preserve measured
main/WAL/shm facts. A separate full native integration also completes actual
SafetyCopy capture, authentication, publication and restoration. The new verified
SafetyCopy retains the deliberate post-import note edit while the restored target
returns to its captured contents; final source/package checks and guards pass.
That integration preceded R129 and observed the ordinary mobile gate refuse the
restored generation with native `RecoveryRequired("mobile_object_changed")`.
R129 observer support now passes genuine postrestore checks, ordinary data edits
and native selected-theme saves with current paired fingerprints and unchanged
DB generations. Real source-claim, activation, manifest, plan, inode, schema and
pending tampering refuses, and restored evidence passes again. The fresh included
artwork case also passes its restored gate after actual SafetyCopy/publication. Earlier parse
failures, the staging-entry sentinel and retained prepared failures remain
separate evidence. Source/package preservation and the
expected native capture-source namespace addition remain recorded; earlier target
hash differences include the fixture's deliberate later note edit and are not
evidence of a preview write. R135 ephemeral scratch composition received explicit
human approval and now passes native identity, worker, cancellation and process
controls plus a real mounted App manual turn on the public-produced target.
Targeted public-produced-target evidence also covers ordinary
RunLog/command/trace APIs, native Goal privacy (including stored request-body
erasure) with retained accounting/holds through reopen, and genuine Console
hydration/controller Discard with no provider dispatch and unchanged provenance.
The separate mounted qualification uses the real New Chat/Send path, native
shutdown and ordinary reopen with one scripted generation and no tool dispatch.
Imported identity-scoped holds and accounting remain unchanged. It uses the
native character-estimate tokenizer fallback, not real-provider or hardware proof.
The absent-marker Notes/Personas operator flow now proves a real App rename,
public import and two cold App starts on the same target. The default assistant
remains ID 1 and imported linked notes appear as paused candidates requiring
migration review, without activating file sync or changing external files.
The present-marker flow also passes after the R137 receipt-backed extension:
new unrepresented imported links become disjoint paused candidates while the
existing marker, roots, bindings and reviews are preserved. Two cold App starts
retain the same mapped ID/version/digest and default character, report no added
review warning, and physically close all seven observed Notes handles, including
the callback's separate store. Task8 and the API/client cutover are implemented.
Task11 records connected served qualification separately; physical phone/VPN
operation remains unverified.

Native-owner FIX1 retains every actual runtime directory iterator on
`mobile_imported_run_log._PinnedHistory` before enumeration. Iterator retirement
precedes descriptor and lease retirement. An uncertain close strongly retains the
iterator, pins and native file leases in quarantine, without retry or a GC-based
release. Healthy and exhausted iterators retire normally.

`mobile_native_rows._prepare_rows` also qualifies native detached-owner replay
before publication or owner opening. Proved original conversation mappings admit
an attached insertion, all effective-owner guarded descendants (including response
links and nested owner attachment), then the native one-way detach with the exact
source timestamp. Dependency edges and the conversation's unique attached slot
are checked for the whole schedule on the original allowance. The same actual
Chat transaction executes these transitions and verifies the original final rows
before committing their existing receipt; temporary attached states never enter
receipt provenance. Empty detached graphs remain supported and unproved originals
retain their precise pre-effect refusal under ADR-244 R109.

Cold native Media construction imports the pure `Chunking._template_conversion` migration helper. The Chunking package resolves its existing runtime exports lazily so this pure import does not accidentally acquire configuration startup admission. Actual runtime export access retains the installed engine/configuration defaults and gates.


R146 changes only the complete ordinary `require_mobile_cutover` coordinator envelope from16MiB to64MiB, with512 reserved at issuance through a private no-argument issuer. Restored and original-generation branches share that same single nonrenewable allowance and retain every journal/plan/manifest/physical/schema/selector/activation/capture/history recheck. This is an explicit bound change, not a new native child or storage permission. Default trace/Notes/private native16MiB and public4GiB/coupled-native16MiB remain. B1's first restored gate exhausted its original16 during native plan loading; its successful preceding restoration/SafetyCopy facts remain separate from the later R146 positive gate. Fresh R146 integration passes both ordinary restoration and included-artwork restoration; the latter leaves 2,718,945 units after its full gate. This measured case is not a whole-engine/RSS or universal-fit guarantee.

## Served integration qualification boundary (Task11)

The genuine fresh NoData fixture selects its config and canonical roots before
public cutover and before either serving parent or app child starts. R158 makes
the serving parent's canonical, media and legacy-source selections use the same
authoritative atomic config snapshot as the native producer. R159 lets an absent
optional tokenizer mapping remain absent through first Send and continuation;
fixture directory seeding is not fresh-profile qualification.

Actual cold Create/immediate Send exposed `preview_sqlite_changed` when ordinary
completion verification copied live WAL databases. R160 gives original and
authenticated-restored completion reads the fixed `recovery.mobile.completion.read`
role: must-exist `mode=ro`, deferred read transactions, existing restrictions and
receipt/schema/generation checks. SQLite shared-memory coordination is allowed;
SQL writes, checkpoints and migrations are not. Preview and maintenance still
use their strict private copies and unchanged-source checks.

The accounting-only completion context borrows the coordinator's same 64 MiB
allowance and custody. Installed schema references and the prepaid reusable
64 KiB digest workspace consume that allowance. Actual registered ordinary
connections close through the native admitted wrapper; uncertain physical close
retains the lease and accounting. This context grants no preview or capture
storage authority. Live concurrent-write, authenticated restored-generation,
strict-copy and physical-retirement evidence are recorded separately in the
[Task11 qualification packet](../superpowers/qa/2026-10-25-shared-console-mobile-supervision/README.md).
