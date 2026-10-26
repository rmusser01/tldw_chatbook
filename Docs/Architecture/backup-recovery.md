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
destination_witnesses, destination_sidecars)` prepares an internal in-place
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
pinned native exclusive creation, with `mobile_path_intended` before creation and
`mobile_path_observed` before accepting the new witness. Changed objects, foreign
operations and unresolved creation intents refuse. This path does not adopt an
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
records. The importer must add checked preview provenance, source claims, complete
mapping/reference and byte closure, native owner batch receipts, validated recovery
and cutover before any public apply or terminal fence-release route is exposed.

Before native owner construction, `mobile_mode_intended` records an exact existing
owned regular file witness, and `mobile_mode_observed` accepts only the same
path/device/inode changed to 0600. All three DB owners use this transition;
AgentRuns first wins its stable execution flock. Interrupted transitions remain
pending and refuse adoption on reopen. Source bytes and permissions are unchanged.

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
