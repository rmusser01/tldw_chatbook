# Backup and recovery

This document describes the ADR-126 complete local backup and restore subsystem: inventory and data groups, admission/quiescing, capture and encrypted archives, replace/isolated restore with journaled publication and encrypted rollback, recovered-media review, and the startup fence. "Recovery" elsewhere (notes-sync crash recovery, server-sync account recovery, fleet worktree recovery) belongs to those subsystems' own docs.

## Authoritative files

| Concern | File | Key symbols |
| --- | --- | --- |
| Service | `Backup_Recovery/recovery_service.py` | `RecoveryService` — one worker composing all engines: backup/restore previews, inspection, extraction, rollback, recovery-copies, profiles |
| CLI + startup unlock | `Backup_Recovery/launcher.py` | `python -m tldw_chatbook.Backup_Recovery` (`inspect / extract / restore / recover / profiles / copies`); `startup_preflight` / `startup_unlock` (master-password prompt, rekey, reset); `minimal_recovery` host |
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

`admit_startup()` runs before runtime/config imports (literally the first lines of `app.py` and `Web_Server/serve.py`, and several package `__init__`s — `Web_Server`, `RAG_Search`, `TTS`); enrollment failure is a hard `SystemExit("Recovery required: …")`. When config encryption is enabled with a verifier, `startup_unlock` prompts pre-TUI: a wrong-but-readable password offers rekey; an empty one resets encrypted config values (keys removed; chats/notes untouched). Served children refuse the unlock with a plain-language error rather than hanging. Non-unlock recovery reasons get the minimal recovery host. A recovery restart (app → CLI handoff) checks no running operation and no unsaved editors, quits guarded, and spawns a fresh process into the minimal recovery host.

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
External note/dictionary/persona asset references without a qualified legacy
owner binding also block; they are never discarded or guessed from path strings.
Multiple known configs sharing one physical source require destination review.
`identity_conflicts` lists exact overlapping destination primary-key tuples;
collisions require explicit mapping and never establish equivalent content.

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
accepts qualified Chat81/Workspace8/AgentRuns25 sources, including every
previously admitted AgentRuns25 physical layout; migration creates a separately
qualified successor. The Task7 mobile fixture deliberately retains those legacy
source heads while canonical fixtures use installed heads. Candidate migration
authorization permits only the fixed owner tables/indexes and native version
stamps, and continues rejecting unknown catalogs. This installs receipts and
quarantine consumers; the preview still performs no import or cutover.


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

RecoveryService reports this typed mode as `recovery_required` with no finish,
rollback or abort action. Durable pending records override cached worker success
and continue fencing startup, including corrupt or mismatched journal/pending
records. The private lifecycle below supplies provenance, source claims and native captures.
The importer must add complete mapping/reference and byte closure, native owner
batch receipts, validated terminal recovery and cutover before any public apply or terminal fence-release route is exposed.

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
exclusion. Neither API exposes completion, cutover or a terminal recovery action.

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
legacy81/8/25 and canonical82/9/26 have the same actual owner IDs. Generic mixed
owner versions still refuse. Each archive uses native encrypted publication and
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
and AgentRuns snapshots. Legacy 81/8/25 and canonical 82/9/26 remain distinct. Native
installed catalog, version, integrity, foreign-key and domain checks run on
restricted disposable descriptor-bound connections before row access; no live
repository constructor, migration or destination connection participates.

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

This is extraction only. Typed identities/dispositions, Chat memory/dispatch,
normalized traces/citations, nine-entity sync and integrated pure-plan validation
remain later increments. Reads create no imported rows, authority holds, batch
receipts or published history and confer no public apply/cutover success.
