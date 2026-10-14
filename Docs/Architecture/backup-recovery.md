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
