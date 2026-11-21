---
id: TASK-42113
title: "A Replace at the default config location fails for pet cases, and the pet cannot start after one"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 17:26
updated_date: 2026-11-20 21:37
labels:
- follow-up
- backup
- pets
priority: high
references:
- tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage.py
- tldw_chatbook/Backup_Recovery/control_records.py
- Tests/Backup_Recovery/test_created_owner_rollback.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-42022 (round 18), from temporary probes with a scratch HOME. TASK-42000 (round 17) made pet settings a per-profile file in the config folder and added it to backups. Its tests used only a separate config folder.

At the default config location (`~/.config/tldw_cli`) the config folder holds `recovery-bootstrap`, so it is never a root; each config-folder file is bound as its own root. The probes showed:

- A Replace from a backup with no pet file, while the profile has one (`[pets-none]`), fails at activation with `activation_profile_outside_maintenance` (`control_records.py` ~601) and rolls back (`replacement_rolled_back`).
- `[pets-missing]` and `[pets-dated-only]` fail with `replacement_destination_parent_required`.
- After a successful default-location Replace, the footer pet fails with `RecoveryRequired: storage_scope_not_enrolled`: a pet file created after the binding falls under no root.

So a restore at the default location, where most users keep their profile, can fail for common backups since TASK-42000.

**2026-11-20, coordinator decisions:**
- Steps B to D (pet creation and the pet start at the default location) move to TASK-42115.
- After review rev2: deleted-file-root tolerance needs one design across all root consumers, the recovery journal included, so step A (the retire fix in `_bind_activation` and `bootstrap._binding`) moves to TASK-42115 too. That work is parked on `fu/r18-deleted-roots-wip`.

This task keeps the default-location reproduction (refusal tests that pin the fail-safe behaviour) and the notes on other owners.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The failures above are reproduced at the default config location (scratch HOME) with evidence, and committed tests pin that a pet Replace there refuses fail-safe: refused (and rolled back), nothing created, the pets kept, and the app starts
- [x] #3 The notes say, with evidence, whether other config-folder file owners hit the same failures at the default location; any that do are fixed here or have their own task
- [x] #4 The default-location tests run on a PR or dev-push lane, with their time recorded
<!-- AC:END -->

(#2, step A's retire fix, moved to TASK-42115: coordinator decision 2026-11-20, after review rev2.)

## Implementation Plan

1. Reproduce every case of `test_created_owner_rollback.py` with the profile at the default config location (scratch HOME, `RecoveryService(default_control_root())`), and name the code and destination behind each refusal.
2. Find the root cause and the design the default location already has for config-folder files.
3. Fix it where config-folder files are bound or admitted, keeping the registry and activation invariants; stop with NEEDS_CONTEXT if the fix needs a design decision. (Every fix moved to TASK-42115; see the outcome below.)

## Implementation Notes

**Summary.** The default-location failures are reproduced and pinned as fail-safe refusals. Every fix moved to TASK-42115 (coordinator decisions, 2026-11-20). The investigation below is the evidence for #1 and #3; it was first written while this task was blocked on the design choice.

**Reproduction (AC#1, by temporary probe).** A temporary `location` parameter in `test_created_owner_rollback.py` ran every case at the default location (scratch HOME; `TLDW_CONFIG_PATH` set to `~/.config/tldw_cli/config.toml`, as the F9 default-profile workflow does; `RecoveryService(default_control_root())`). All ten default cases failed; the separate-folder cases are unchanged. The `LOCATION` input is now committed in the scenario (TASK-42114). Temporary wrappers on `recovery_service.issue_code` and `publication.finalize_candidate`, and a temporary edit that added the destination to the refusal, named each failure:

| Case | Failure at the default location |
|---|---|
| `pets-missing`, `pets-present`, `pets-dated-only`, `pets-ungrouped` | `replacement_destination_parent_required`, destination `~/.config/tldw_cli`, for `tamagotchi_pets.json` or its dated copy |
| `pets-none` | `finalize_candidate` → `_bind_activation`: `activation_profile_outside_maintenance`, then `replacement_rolled_back` |
| `databases` | `replacement_destination_parent_required` for `config.toml.bak` (`config.history`) |
| `tools` | `replacement_destination_parent_required` for `config_backup_20260101_000000.toml` and `rag_pipelines.toml` |
| `shared-config-folder` | `replacement_destination_parent_required` for `note_templates.json` |
| `library`, `projects` | `replacement_config_container_unverified`, the third refusal in `replacement._first_profile_container`: the first binding of a Replace without the Settings group; no config-folder file is involved |

- A control, `[pets-none]` without the target's own pet file so nothing is retired, completed. Its pet start afterwards failed with `RecoveryRequired: storage_scope_not_enrolled` (`raw.scope` → `acquire_storage` → `_scope`).
- On a never-restored default-location profile (no binding record, recovery bootstrap present), the pet starts, lists and saves. So AC#3's second half already holds; the start failure needs a bound profile.

**Root cause.** The default config folder holds `recovery-bootstrap`, so it can never be a root. At the default location, the design supports only two kinds of file beside `config.toml`:
- the config route's own members: the lock, `.bak` and snapshots, admitted as companions of the selector;
- the three fixed leaves in `_CONFIG_CAPTURE_LEAVES` (`ui_state.toml`, `recent_emojis.json`, `runtime_policy.json`). For these, admission uses `config_participants.sibling_selector` and `companion_guard`; capture uses `_config_capture_sources` and `_config_capture_item`, so they are never enrolled as roots; Replace publication uses `_config_publication_scopes`, which projects exclusion onto the config selector; later rollback uses `_config_file_relations` and `_config_file_retirement_scopes`.

Every other config-folder owner falls back to per-file roots:
- Capture registers each existing file as its own root.
- A Replace that creates one needs a destination parent root, and the config folder cannot be one (`replacement_destination_parent_required`).
- A Replace that retires one leaves an absent root that no enrolled folder covers, which `_bind_activation` refuses (`activation_profile_outside_maintenance`).
- The pet route admits the config folder itself (`raw_participants.scope`: `admission_paths` starts with the parent for `pet`; `settings_file_participants.preflight` leases each dated copy). That folder is under no root at the default location, so a bound profile's pet cannot start.

The pet route was designed for a folder of its own (the old shared `~/.config/tldw_chatbook`). TASK-42000 moved the files beside `config.toml` without adapting it; on a separate config folder the folder is a root, so it worked there.

**AC#4.** `config.history` (`config.toml.bak`, `config_backup_*.toml`), `rag.definitions` (`rag_pipelines.toml`) and `notes.templates` (`note_templates.json`) hit the same creation refusal at the default location (table above). Those owners lived in the config folder before TASK-42000, so this predates it. Retiring them, and creating them later from the app, were not probed.

**Outcome.** The fix designs moved to TASK-42115 in two steps:
- after the first report: creation and the pet start (steps B to D);
- after review rev2: deleted-file-root tolerance, including step A.

Three per-site attempts (finalization, exact-path ownership, crash recovery) each exposed another root consumer; they are on `fu/r18-deleted-roots-wip` (tip `ee6d0ac966`, not pushed). This branch keeps no deleted-file-root tolerance. At the default location a pet-retiring Replace refuses and rolls back, as before.

**#1 tests.** `test_created_owner_rollback.py::test_a_default_location_pet_replace_refuses_and_keeps_the_pets[pets-none-activation_profile_outside_maintenance|pets-missing-replacement_destination_parent_required]`. Each runs the case at the default location (the scenario's `LOCATION` input, `RecoveryService(default_control_root())`).
- It names the refusal: an observer on `publication.finalize_candidate` sees activation's refusal, which `replacement_rolled_back` hides, and an observer on `recovery_service.issue_code` sees the service's own.
- It then checks that nothing was created, the profile's pets are kept, the old shared folder is unchanged, and a fresh start's permission and admission pass.
- These tests pin today's fail-safe refusals rather than a fix. They pass on this branch and were never red; TASK-42115's criteria turn them into completions.

**Lane.** The tests are in `test_created_owner_rollback.py`, in the Backup_Recovery owner-suites dev-push lane: 2 passed in 49 s at `-n 2`, load 4-5.
