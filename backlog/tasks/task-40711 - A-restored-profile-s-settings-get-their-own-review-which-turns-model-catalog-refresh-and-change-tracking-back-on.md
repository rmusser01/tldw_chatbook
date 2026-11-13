---
id: TASK-40711
title: "A restored profile's settings get their own review, which turns model-catalog refresh and change tracking back on"
status: Done
assignee:
- '@claude'
created_date: 2026-11-13 03:35
updated_date: 2026-11-13 15:58
labels:
- follow-up
- backup
- recovery
- security
priority: medium
references:
- tldw_chatbook/Backup_Recovery/activation.py
- Tests/Architecture/test_backup_owner_inventory.py
- backlog/decisions/126-complete-local-backup-and-recovery.md
dependencies:
- TASK-40137
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down from TASK-40137's findings and owner ruling. Checked on dev 26b2e677dd, 2026-11-12.

**Why.** Every restored generation requires the shared `config` owner to be approved before its consumers run, and nothing in the app approves it (TASK-40137's notes list the six `ActivationStore.approve` call sites: MCP, `notes.file_notes`, plugins, RAG, models and skills). So after any restore, even onto the same device or after a rollback, model-catalog auto-refresh and native workspace change tracking stay off for good.

**Owner ruling, 2026-11-12 (TASK-40137):** "Rebind review + separate config review (Recommended)". `config` gets its own simple review, which brings those consumers back. Approving `config` this way must not approve `runtime.sync_state`; server sync identity is approved only by the sync-identity review in TASK-40139.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a restore that publishes `config`, the user can approve `config` on its own through a reviewed step in the existing post-restore review surface; until then model-catalog auto-refresh and native workspace change tracking stay off with a named reason; tests pin both states
- [x] #2 That review approves only `config`: server sync calls stay refused with `sync_activation_required` until the sync-identity review approves `runtime.sync_state`; a test pins it
- [x] #3 After approval, model-catalog auto-refresh and native workspace change tracking run again; a test pins each
- [x] #4 `test_no_reviewer_approves_the_shared_sync_owners` (or its successor) allows exactly this review and the sync-identity review to approve `config`, and nothing else; it is shown red with a stray approver added
- [x] #5 The User Guide's backup and restore page says what this review turns back on and that it does not resume sync
<!-- AC:END -->

## Implementation Plan

1. `RecoveryService.approve_config(generation)` (`Backup_Recovery/recovery_service.py`, beside `current_requirements`): under the selected profile's storage lease, re-read the selected profile's activation witness, refuse (`config_review_changed`) unless it is still the reviewed generation and requires `config`, then `ActivationStore.approve(generation, "config")`. Nothing else is approved.
2. Backup & Restore ▸ Restored profiles ▸ Current profile (the existing post-restore review surface): while `config` is pending, show a settings review that names what stays off (model-catalog auto-refresh, workspace change tracking) and that sync is not resumed, with an **Approve settings** button bound to the shown generation. Approval runs on a worker and refreshes the list.
3. Tests: a real-binding child-process test (states pending/approved) pins catalog refresh and change-tracking capture off then on, and server sync still refused with `sync_activation_required`; a mounted screen test pins the review copy and the button's wiring; the owner-inventory pin gains this one `config` approver, shown red with a stray approver.
4. `Docs/Backup-and-Recovery.md` (the backup user page; there is no `Docs/User_Guide/` page for it): say what the review turns back on and that it does not resume sync.

## Implementation Notes

**Summary.** Backup & Restore ▸ **Restored profiles** ▸ **Current profile** (the existing post-restore review surface) now carries a settings review while the current generation's `config` is pending: a **Settings review** note saying model-catalog auto-refresh and workspace change tracking stay off until approval, and that approving does not resume server sync, then **Approve settings**. The button calls the new `RecoveryService.approve_config(generation)` on a worker, which approves `config` alone and nothing else. Server sync keeps refusing with `sync_activation_required` until `runtime.sync_state` is approved, which only TASK-40139's sync-identity review may do.

**Approach and decisions.**
- `approve_config` (`Backup_Recovery/recovery_service.py`, beside `current_requirements`) holds the selected profile's storage lease, re-reads that profile's activation witness, and refuses with `config_review_changed` unless the witness is still the generation the summary showed, requires `config`, and is a live paired witness (`_witnesses`). Then it calls `ActivationStore.approve(generation, "config")`. It approves the selected profile's own witness only, not other witnesses that share its storage. In that rare case `config` stays refused, which fails closed.
- Placement and copy follow the nearest approvers (no ruling covered them): the button sits under the Current profile summary it reviews, and its label follows **Approve local model** / **Approve RAG owners**. The "named reason" is that note, plus `config` in the summary's pending owners (`current_requirements()["pending_owners"]`). The consumers' own refusals are unchanged: catalog scheduling returns False, and change tracking's capture raises "native initialization source unavailable".
- Model-catalog auto-refresh is scheduled only at startup or setup completion (`app.py`), so after approval it runs from the next start. The note and the success message say so. Change tracking passes its gate at once.
- The pin keeps its name, `test_no_reviewer_approves_the_shared_sync_owners`, so TASK-40139's parallel edit of it still merges. It gains one row, `("tldw_chatbook/Backup_Recovery/recovery_service.py", "'config'")`, appended at the tail, and its docstring says the sync-identity review adds its own row.

**Tests and red/green evidence** (load 16 to 40 on 18 CPUs):
- **The real binding (AC#1–#3).** `Tests/Architecture/test_backup_owner_inventory.py::test_the_config_review_turns_config_consumers_back_on_and_never_sync[pending|approved]` is lane-run (architecture-guards). It runs a child with a scratch HOME and the network guard. It binds `config` and `runtime.sync_state` for a real selector, then drives the real `RecoveryService`, startup catalog scheduling and run, the native change-tracking capture gate, and `ServerSyncService.get_changes`.
  - Red before the fix: `[approved]` failed with `AttributeError: approve_config`.
  - Red with `approve_config` temporarily also approving `runtime.sync_state`: an assertion failed (`pending_owners` was `()`, not `('runtime.sync_state',)`).
  - Green after; serial 2.7 s and 4.5 s.
- **The screen (AC#1).** `Tests/UI/test_backup_restore_setup_entry.py::test_restored_profiles_settings_review_approves_config_alone` is in a whole-file UI census entry, so no census edit was needed. It stands in only the service's current-profile methods.
  - Red against HEAD's screen: `AssertionError: no settings review`.
  - Green after; serial about 3 s (`-p no:xdist --timeout=180 --durations=0`, load 17).
- **The pin (AC#4).** Red with a temporary stray `store.approve(witness["generation"], "config")` in `Plugins/backup.py`: "Left contains 1 more item: ('tldw_chatbook/Plugins/backup.py', \"'config'\")". Reverted; green.

**Runs.**
- `Tests/Architecture/test_backup_owner_inventory.py`, serial: 28 passed.
- Backup UI and recovery covering files, `-n 4`: 119 passed. One failed: `test_backup_restore_screen.py::test_later_credential_omissions_remain_visible_during_keyboard_review`, a keyboard test of the later-rollback credential review, not the profiles view. It passed in the first `-n 4` run and 3 of 3 serial.
- Preflight passed; the markup check is OK.
- Size ratchets: `test_screen_size_ratchet`, `test_module_size_ratchet` passed; neither lists the touched modules.

**Red at dev base 26b2e677dd, not this task's.** Each failed the same way with the base tree's tests and package (`git archive 26b2e677dd`, in scratch). The `_run` children still imported this worktree's package (the wrapper's `PYTHONPATH`), but their failing path does not reach the changed code:
- `test_timer_path_static_update_inventory.py` (2), from `project_index_binding_card.py`'s new `_poll` clock;
- `test_activation_startup.py` (17), `RecoveryRequired: replacement_identity_unverified` on a config import after binding;
- `test_backup_restore_composition.py` (2 failed, 1 error);
- `test_backup_known_profiles.py` (1).

**Docs.** `Docs/Backup-and-Recovery.md` › Setup after restore. It is the backup user page; there is no `Docs/User_Guide/` page for it, as TASK-40132's notes also say.

**Files.**
- `tldw_chatbook/Backup_Recovery/recovery_service.py`
- `tldw_chatbook/UI/Screens/backup_restore_screen.py`
- `Tests/Architecture/test_backup_owner_inventory.py`
- `Tests/UI/test_backup_restore_setup_entry.py`
- `Docs/Backup-and-Recovery.md`

