---
id: TASK-40137
title: "Investigate whether a restored runtime.sync_state lets Notes and Chat sync resume as the old device after a cross-machine restore"
status: Done
assignee:
- '@claude'
created_date: 2026-11-10 17:13
updated_date: 2026-11-13 03:35
labels:
- follow-up
- security
- backup
- recovery
priority: medium
references:
- tldw_chatbook/Sync_Interop/recovery.py
- tldw_chatbook/Sync_Interop/sync_state_repository.py
- tldw_chatbook/Backup_Recovery/activation.py
- Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 13 of the 2026-10-23 burn-down from TASK-39148's owner-approved design, `Docs/superpowers/specs/2026-11-10-personal-context-backup-key-custody-design.md`. Owner decision 2026-11-10 (AskUserQuestion): "Approve all 11". Checked on dev ae9f9e8d21, 2026-11-10; line numbers are dev's.

**Why.** TASK-39148's design requires that a restored device never act as the old device towards the server (spec constraint 1). It could not trace whether general Sync honours that.

What is captured:
- Backup captures `tldw_chatbook_sync_state.db` as `runtime.sync_state` (`Sync_Interop/recovery.py:79-90`).
- That includes the shared `sync_profile_state.device_id`, dataset cursors, and the Sync V2 outbox and remote heads (`:39` and the schema at `:21-46`).

What restore does:
- Every restore mints a new `installation_id` (`Backup_Recovery/isolated_restore.py:96-98`, `replacement.py:1629-1630`).
- Every restore binds all activation-required owners (`Backup_Recovery/publication.py:592-604`), including `runtime.sync_state`.
- Server credentials come back into fresh recovery scopes when remappable (`credentials.py:1231-1302`), or the user enters them again.

What is gated:
- Server Sync V2 calls are gated after a restore. `_sync_call` holds `execution_scope(("config", "runtime.sync_state"), …)` and raises `sync_activation_required` (`Sync_Interop/server_sync_service.py:51-95`, `:85-88`), checked with `git show ae9f9e8d21:…`.
- It wraps `send_changes`, `get_changes`, `run_v2_dry_run`, `push_v2_envelopes`, `pull_v2_envelopes`, the Notes organization bootstrap, the Personal Context link calls, and the conflict and recovery-bundle calls.
- `Sync_Interop/local_first_sync_service.py` uses the same decorator. `LocalFirstSyncService.sync_once` (`:70-71`) syncs Notes, Chat and Personal Context in one call.
- Activation approval is per owner and generation (`Backup_Recovery/activation.py:351-384`), so every Sync V2 call needs both `config` and `runtime.sync_state` approved.
- That includes the Personal Context link calls (`Personal_Context/link_service.py:170`, `:730` → `server_sync_service.py:710-712`, `:811-813`).

**Unknown.**
- What approves `config` and `runtime.sync_state` for a restored generation. No approver naming either was found; the approvers found are Notes, RAG, Skills, MCP and models, each for its own owner.
- Whether that approval is a plain approval. After a cross-machine replace restore, with credentials entered again, Notes and Chat sync would then push and pull with the restored device id and cursors. The restored device would act as the old one, and two devices could share one identity.
- Whether any Notes or Chat sync path reaches the server without going through `_sync_call`.

**Why a ruling is needed.** A fresh Personal Context first link after a restore cannot pass `_sync_call` unless something approves `config` and `runtime.sync_state`. Approving them must not let Notes and Chat resume as the old device.

**Scope.** This task produces findings and a ruling. It does not implement a gate, a rebind or a review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The task notes record, with file:line evidence, which Notes and Chat sync paths go through the `_sync_call` activation gate and which do not, and what approves `config` and `runtime.sync_state` after a restore
- [x] #2 The task notes record, with a test or probe run through the scratch wrapper, whether Notes and Chat sync send the old device id and cursors once that gate is approved after a cross-machine replace restore
- [x] #3 The notes give the ruling: who may approve `config` and `runtime.sync_state` after a restore, and through what review, such that Notes and Chat sync never resume as the old device after a cross-machine restore, and how a fresh Personal Context first link then passes `_sync_call`; where the ruling needs an owner decision, that decision is recorded verbatim with its date
- [x] #4 If current behaviour lets Notes and Chat resume as the old device, the notes propose the guard (gate, rebind or reset after restore) as a follow-up; if not, the guard that prevents it is named and a test pins it
<!-- AC:END -->

## Implementation Plan

1. Re-locate every cited line on base 26b2e677dd. List every production call of the Sync v2 client and every production entry point into Notes and Chat sync, and check each against `_sync_call` / `_sync_execution_scope`.
2. List every `ActivationStore.approve` call site and the owners it can approve, to answer what approves `config` and `runtime.sync_state` after a restore.
3. Probe through the wrapper (scratch HOME, `Tests.network_guard`, a recording fake client): seed `runtime.sync_state` as the old device left it, record the restore's real activation binding, and observe each route (sync push, sync pull, Prepare dry run, Notes organization bootstrap) unapproved, with `config` and `runtime.sync_state` approved (with and without a dataset key), and ordinary.
4. Name the guard that holds today and pin its unpinned half (no reviewer approves the two shared owners) with a small census test in a lane-run file; red under a temporary approver edit.
5. Write the findings, the ruling options with their consequences, and the proposed follow-up guard in the Implementation Notes. The ruling itself goes to the owner (NEEDS_CONTEXT).

## Implementation Notes

**Summary.** Today a restored profile never syncs Notes or Chat again, as the old device or any other: every server Sync v2 call needs `config` and `runtime.sync_state` approved, and no code approves either. Once both are approved, every route that reaches the server sends the restored `device_id`, cursors and queued outbox. So the restored device would act as the old one. Who may approve the two owners, and through what review, needs an owner ruling (AC#3 unticked; options below). Line numbers are base 26b2e677dd.

### AC#1: which paths go through the gate, and what approves the two owners

**The gate.** `_sync_execution_scope` (`Sync_Interop/server_sync_service.py:52-95`) holds `execution_scope(("config", "runtime.sync_state"), path)` for each source the call observes (`:86-87`) and raises `PermissionError("sync_activation_required")` (`:89`). `_sync_call` (`:98-108`) wraps a method in it.

**Every production call of a Sync v2 client method is in `server_sync_service.py`** (`:292, 303, 341, 376, 600, 620, 636, 650, 670, 774, 798, 822, 854, 889, 904, 922, 1020, 1110, 1176, 1205`), inside these `@_sync_call` methods:
- `send_changes` `:285`, `get_changes` `:294`;
- `bootstrap_notes_organization_profile` `:309`, `run_v2_dry_run` `:525`;
- `push_v2_envelopes` `:928`, `pull_v2_envelopes` `:1050`;
- the recovery-bundle, restore-manifest and conflict calls `:862, 892, 911, 1165, 1182`;
- Personal Context: `bootstrap_personal_context_link` `:740`, `complete_personal_context_link` `:841`, first-link push and pull `:1034, 1042, 1137, 1159`.

The undecorated helpers `_push_v2_envelopes` (`:980`) and `_pull_v2_envelopes` (`:1093`) are reached only from gated methods (`:971, 1040, 1048, 1084, 1150, 1163`).

**Notes and Chat entry points, all gated:**
- `LocalFirstSyncService.sync_once` is `@_sync_call(delegate=True, local=True)` (`local_first_sync_service.py:70-71`). It reads the device id (`:116`), pull cursor (`:211`) and outbox (`:220`) from `runtime.sync_state`, then pushes (`:301`) and pulls (`:399`).
- `pull_notes_organization_history` (`:543`) is not decorated, but it reaches the server only through the gated `pull_v2_envelopes` (`:592`), with the stored device id and cursor (`:578-587`).
- `SyncScopeService.send_changes` / `get_changes` are `@_sync_call(delegate=True)` (`sync_scope_service.py:319, 332`; no product caller). `prepare_sync_v2_profile_mode` holds the scope around `run_v2_dry_run` (`:393`).
- UI: Settings › Sync **Manual run** (`settings_screen.py:12671` → `ManualSyncControlService.run_once`). It first runs Notes organization enrollment (`manual_sync_control.py:234` → `notes_organization_sync_service.py:840` bootstrap, `:914` history pull), then `sync_once` (`:285`). Settings' **Prepare sync mode** (`settings_screen.py:26866`) runs the dry run. No automatic or background Notes/Chat Sync v2 run exists: these are the only callers of `run_once` and `sync_once`.

**Paths that bypass `_sync_call`:**
- `Widgets/Settings_Widgets/server_switch_modal.py:231-235` POSTs `{}` to `/api/v1/sync/send` to test a typed token. It sends no device id, cursor or envelope.
- `Sync_Interop/notes_m1_flow.py:35, 55` calls the client directly, but `NotesM1SyncFlow` has no production constructor.
- `upload_sync_v2_attachment` has no caller.

**Nothing approves `config` or `runtime.sync_state` after a restore.**
- Both are required owners of every restored generation. `config` is `_Config(_RawDeclaration)` (`Backup_Recovery/config_adapter.py:136`, default `activation_required=True` at `recovery_files.py:41`). `runtime.sync_state` is a `_SQLiteDeclaration` (`Sync_Interop/recovery.py:79-90`; default `True` at `DB/recovery_operations.py:76`). `finalize_candidate` binds every activation-required owner, whichever groups were restored (`publication.py:592-604`).
- `ActivationStore.allowed` (`activation.py:372-383`) is false without that owner's approval record, and `activation_permission` checks it (`:133`). `ActivationStore.approve` (`:351-370`) is the only writer of approvals. It has six call sites, none for these owners:
  - `MCP/recovery_activation.py:733`: `mcp.local`, `mcp.permissions`, `mcp.context`, `mcp.targets` (`:22-23`, filter `:621`);
  - `Notes/recovery_review.py:318`: `notes.file_notes` (`Notes/file_notes_service.py:345, 352`);
  - `Plugins/backup.py:354`: `plugins`;
  - `RAG_Search/activation.py:389`: `rag.definitions`, `rag.projections`, `db.rag_indexing` (`:267`);
  - `RAG_Search/model_recovery.py:330`: `models.artifacts`;
  - `Skills_Interop/recovery_activation.py:459`: `skills`.
- Side effect: `config` also gates model-catalog auto-refresh (`app_feature_glue.py:559, 629`) and native workspace change tracking (`Workspaces/change_tracking.py:358`). After any restore, including on the same device, those stay off as well.

### AC#2: what is sent once the gate is approved (probe)

**Restore keeps the old identity.**
- `runtime.sync_state` is copied as bytes (`Sync_Interop/recovery.py:68-76`), and `_SQLiteDeclaration.relocate` only validates (`DB/recovery_operations.py:110-120`).
- `Tests/Backup_Recovery/test_complete_roundtrip.py:1045-1047` already asserts that the sync profile row comes back unchanged in a fresh installation.
- The new `installation_id` (`isolated_restore.py:98`, `replacement.py:1630`) never reaches Sync v2. The device id is server-issued and stored in `sync_profile_state`.
- Device registration carries only a `device_id` string, with no proof of possession (`tldw_api/sync_schemas.py:808-819`). The server cannot tell a restored copy from the original.

**Probe setup.** The probe ran through the wrapper: `$FU/run.sh <wt> pcb-probe python driver.py`. Each case ran in a fresh child with scratch HOME, XDG and `TLDW_CONFIG_PATH`, and with `Tests.network_guard`. A recording fake client records the first server call and stops.
- `runtime.sync_state` was seeded as the old device left it: `device_id='old-device'`, `dataset_cursors={'sync_v2': '41'}`, the matching pull cursor, and one note envelope queued by `old-device`.
- The restore's real binding was then recorded: every activation-required owner from `install_adapters()`, through `bind_activation`.

| State | push (sync_once) | pull (sync_once) | Prepare (dry run) | Notes organization bootstrap |
|---|---|---|---|---|
| bound, unapproved | refused `sync_activation_required`, no client call | same | same | same |
| `config` + `runtime.sync_state` approved, dataset key present | push `device_id=old-device`, `last_known_cursor=41`, envelope from `old-device` | pull `device_id=old-device`, `cursor=41` | `register_sync_v2_device(device_id=old-device)` | `get_sync_v2_profile(device_id=old-device)` |
| approved, no dataset key | refused locally: "dataset key is required" | same | `register_sync_v2_device(device_id=old-device)` | `get_sync_v2_profile(device_id=old-device)` |
| ordinary (no restore binding) | as approved | as approved | as approved | as approved |

The only other owners left unapproved in the approved rows were the ones the six reviewers own. The gate checks only the two shared owners.

**Cross-machine nuance.** Encrypted Notes and Chat envelopes need a dataset key. `app.sync_v2_dataset_keys` starts empty (`app_service_wiring.py:3126`) and is filled only from the keyring-held Personal Context staging key after a completed link (`app.py:2653-2664`). That key does not exist on another machine. So after a cross-machine restore with the gate approved:
- **Without a key (the "no dataset key" row):** encrypted push and pull refuse locally (`local_first_sync_service.py:143`, `manual_sync_control.py:148`).
- **Still sent as the old device:**
  - Prepare re-registers the restored `device_id` (`server_sync_service.py:597, 622`), pushes an empty batch and pulls from the restored cursor (`:673`).
  - Notes organization enrollment reuses the restored device id (`:455`, `:341`, `:376`) and pulls history from the restored cursor.
  - The Personal Context first link registers the restored shared `device_id` with the new machine's wrapping key (`:793-800`).
- **On the same machine and OS account,** where the key survives, encrypted push and pull also go out as the old device, including the old device's queued outbox.
- Pulls exclude the device's own changes (`include_own_changes=False`, `local_first_sync_service.py:415`; `excluded_device_id`, `server_sync_service.py:688`). So if the original device is still running, each copy can miss the other's changes.

### AC#3: the ruling (owner decision needed; not decided here)

A finding the ruling must also cover:
- The spec's plan for TASK-40139 is that a fresh Personal Context link "rebinds Personal Context's device without changing Notes and Chat sync state". That plan cannot hold as the data is shaped today.
- With no active workspace, Personal Context and Notes/Chat share one `sync_profile_state` row. Personal Context uses `workspace_scope=None` (`server_sync_service.py:793`, `link_service.py:1014-1028`); Settings uses the active workspace or `None` (`settings_screen.py:12135-12142`).
- So rebinding Personal Context to a new device id changes Notes and Chat's identity too. Keeping the row unchanged means the link re-registers the old device id.

Options, with what each does:
- **A. A sync-identity review that rebinds.** One reviewed step, in Backup & Restore or Settings › Sync, approves `config` and `runtime.sync_state` together. In the same step it clears the restored server identity for every server profile: device id, dataset cursors, pull cursors, remote heads, Personal Context link state, and the pending outbox (counts shown first). `SyncStateRepository.clear_server_profile_state` (`sync_state_repository.py:914`) is a starting point.
  - The next sync registers a new device (`device_id=None`). A fresh Personal Context first link then passes `_sync_call` and seeds that device with no binding conflict.
  - Never acts as the old device, on any machine.
  - Changes the old device queued but never sent are dropped from the outbox (the local rows remain; whether they are queued again is unchecked).
  - Forces a full bootstrap and re-pull. The old device's registration stays on the server, and Chatbook cannot remove it.
- **B. Approve as-is, on the user's word** ("same device, or the old one is retired"). Sync resumes with the old device id, cursors and outbox.
  - Breaks spec constraint 1 whenever that word is wrong, and the client cannot check it.
  - Two live copies would push under one id and hide each other's changes (`include_own_changes=False`).
  - A Personal Context link would register the old device id with a new wrapping key.
- **C. Split by restore kind.** A rollback that only puts back the original's own pre-restore bytes approves automatically. Today a rollback rebinds too (`replacement.py:2584-2595`, in `_rollback_replacement`), so even a rolled-back failed restore keeps sync off. Any forward restore goes through A.
  - Rollbacks stay seamless; this needs a journal signal for "rollback to own state".
- **D. Split the owners.** `config` gets its own review, which also turns model-catalog refresh and change tracking back on. `runtime.sync_state` is approved only through A.
  - Separates the consumers that are not sync from device identity.

Only A, used alone or as the forward-restore path of C or D, keeps constraint 1 on every machine. B does not.

**Owner ruling, 2026-11-12 (AskUserQuestion, recorded by the coordinator):** "Rebind review + separate config review (Recommended)". The option as put to the owner: "Sync identity: one reviewed step approves both owners and clears the restored server identity (device id, cursors, Personal Context link, pending outbox, counts shown first), so the next sync registers as a new device and re-pulls. `config` gets its own simple review, which also turns model-catalog refresh and change tracking back on; today those stay off after any restore. Never acts as the old device."

So, option A plus option D:
- `runtime.sync_state` is approved only by the sync-identity review. That review approves `config` and `runtime.sync_state` together and clears the restored server identity for every server profile. Because Personal Context and Notes/Chat share one `sync_profile_state` row when no workspace is active, the rebind covers both. A fresh Personal Context first link afterwards passes `_sync_call` on the newly issued device.
- `config` can also be approved alone by its own review, which brings back model-catalog refresh and change tracking without touching sync identity.
- The sync-identity review is folded into TASK-40139 (its criteria were amended the same day); the config review is TASK-40711. The pin `test_no_reviewer_approves_the_shared_sync_owners` must then allow exactly these two approvers.

### AC#4: the guard today, and the guard to propose

**Today's guard.** Notes and Chat do not resume as the old device today, because `_sync_execution_scope` needs both owners approved and nothing approves them. Two tests pin it:
- **The gate.** Existing `Tests/Backup_Recovery/test_activation_network_sync.py::test_network_sync_checks_actual_source_before_client_or_replay` (its `inactive` and `config_only` states refuse before any client is built). 43 passed at head (`-n 4`, load 74). That file is in no PR lane.
- **No reviewer approves them.** New `Tests/Architecture/test_backup_owner_inventory.py::test_no_reviewer_approves_the_shared_sync_owners`, in a lane-run file. It pins the six `approve` call sites with their owner arguments, and checks that the owner sets they draw from exclude `config` and `runtime.sync_state`.
  - Red under two temporary edits, each reverted: an extra `store.approve(..., "runtime.sync_state")` in `Plugins/backup.py` failed the call-site census ("Left contains 1 more item"); adding `"config"` to RAG's `_REVIEW_OWNERS` failed the disjointness check.

**The guard to propose (follow-up).** The moment anything approves the two owners, sync resumes as the old device (the AC#2 probe). The proposed follow-up guard is option A's rebind/reset, which must land with whichever approver the ruling chooses. It belongs before or inside TASK-40139.

Files: `Tests/Architecture/test_backup_owner_inventory.py`. The probe scripts are scratch-only, outside the repo.

