---
id: TASK-42382
title: "Port PR #3682 (make-active workspace activation) and lane its Workspaces and UI tests"
status: Done
assignee:
- '@claude'
created_date: 2026-11-22 20:15
updated_date: 2026-11-22 20:15
labels:
- workspaces
- tests
- ci
priority: medium
references:
- tldw_chatbook/Widgets/workspace_create_modal.py
- Tests/Workspaces/test_workspace_create_modal.py
- Tests/Workspaces/test_console_workspace_create_handler.py
- Tests/Workspaces/test_console_workspace_reconcile.py
- Tests/UI/test_post_release_workspaces_library_depth.py
- Tests/UI/test_console_new_workspace.py
- .github/workflows/derived-artifacts.yml
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #3682 ("Activate a 'make active' workspace before its create dialog closes") came from a Claude cloud session with no task file. It was 141+ commits behind `dev` and had red CI, so the owner had it ported onto a fresh branch off `dev` (round 21, 2026-11-22).

Console, Library and Settings activate a new workspace in the create dialog's result callback. Textual queues that callback with `call_next` on the opener's message pump, so the opener can resume first. In the process-backed Console, the previous workspace then stayed active for about a second after the dialog closed.

The workspace tests that cover this path ran in no PR lane, and several were red on `dev` unnoticed: three Console stub tests had drifted from the worker contract, and two UI files died at setup locally on the unselected profile.

The coordinator authorized this one task as an exception to the round's no-new-task rule.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With "make active" ticked and no profile interview, the new workspace is active before the create dialog dismisses, including after a successful folder-binding retry. The interview path, an unticked box, and Escape after a partial create leave activation to the caller.
- [x] #2 Any exception from that early activation stays inside the dialog and is logged at debug level. The caller's own activation call meets the failure and reports it under its own policy, as on `dev`.
- [x] #3 The Console create-handler and resume-reconcile stub tests expect the worker contract `dev` ships since d043e22438 (the sync callable, not a coroutine) and pass.
- [x] #4 `Tests/UI/test_post_release_workspaces_library_depth.py` runs without a temporary marker locally, and it still catches the early-activation RuntimeError regression.
- [x] #5 The Workspaces create, handler and reconcile files run in a required PR lane, and both UI files run in the UI census.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Cherry-pick 009cba1cd7 with `-x` onto a branch off `dev` ddd3c02052, and show its ordering test red with the modal reverted.
2. Fix the port's own regressions and review findings, each red then green.
3. Repair the drifted stub tests, and give the Library depth file the profile marker its siblings use.
4. Lane the files, measured with each step's flags.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Commits** (on `fu/r21-activate`, stacked on the three Tool Pack ports; the coordinator's restack rewrites these shas):
- `5a85950d22`: cherry-pick of 009cba1cd7.
- `c11f34caeb`: catch every early-activation exception.
- `014451cee0`: handler stub drift.
- `4a5a728d60`: the Library depth file's private-profile markers.
- `7227690c5a`: lanes.
- The review-fix commit that adds this file.

**Evidence** (macOS, CPython 3.12.11; xdist `-n 4` unless marked serial):

*The port*
- `test_make_active_takes_effect_before_the_dialog_closes[make-active]` fails with the modal reverted to `dev` (1 failed / 26 passed) and passes with the port.

*The early-activation catch*
- The cherry-pick caught only `WorkspaceRegistryServiceError`. Library's callback catches every exception there, so `test_create_workspace_recomposes_after_activation_failure` went from green on `dev` to red under the port.
- `test_any_early_activation_failure_is_left_to_the_caller` fails on the cherry-pick (a RuntimeError escapes) and passes after the fix.
- The review made the swallowed error visible: `logger.opt(exception=True).debug(...)`.

*Binding retry (review)*
- `test_binding_retry_activates_only_when_the_retry_closes_the_dialog[retry, escape]`.
- `[retry]` fails with the early activation removed from the success path; `[escape]` fails when the cancel path activates.

*Stub drift*
- Since d043e22438, `run_worker` receives `_sync_native_console_chat_ui` itself.
- `test_console_workspace_create_handler.py::test_make_active_runs_full_console_sequence` was red on `dev`; the file is now 6 of 6.
- The review found the same drift in `test_console_workspace_reconcile.py`. Three tests there were red on `dev` (`test_cross_screen_change_runs_full_sequence_in_order`, `test_no_active_session_activates_registry_workspace`, `test_stale_active_session_id_is_repaired_before_core_sync`); the file is now 9 of 9.
- A probe with the product reverted to the coroutine call fails the handler test at the worker step.
- The stubs' dead `"ui-sync-sentinel"` return is gone.

*Library depth file*
- `dev`'s version: 15 of 15 fail at setup locally with `RecoveryRequired: raw_source_selection_changed`.
- With `@private_profile_test` on each test: 15 passed with no temporary marker.
- With the early-activation catch narrowed back to the cherry-pick's form: 1 failed / 14 passed (the RuntimeError test).
- The private-profile marker stays (coordinator decision): it matches the sibling Workspaces UI tests.

*Comments*
- The reconcile docstring (`ConsoleWorkspaceController._reconcile_console_session_with_registry`), `ChatScreen.on_screen_resume`'s comment, the reconcile test module's docstring and `_activate_before_dismiss` no longer claim that only a cross-screen change writes the registry alone, or that the result callback always runs after the opener resumes.

**Lanes** (serial, each step's own flags):
- **`dreams-tracking`** ("Run the Dreams tracking tests and unlaned riders", `--timeout=180`; display name changed from "Dreams tracking tests"; the required summary check keeps its name). The riders run after `Tests/Dreams/test_track_service.py` and before `Tests/Tool_Packs` (TASK-42191), which must stay last:
  - `Tests/Workspaces/test_console_workspace_create_handler.py`: 6 tests, 3.5 s (load 15-17);
  - `Tests/Workspaces/test_console_workspace_reconcile.py`: 9 tests, 3.4 s (load 14-21);
  - `Tests/Workspaces/test_workspace_create_modal.py`: 28 tests then, 66.6 s (load 15-17). It now has 30 with the review's retry cases.
  - The whole step, 735 tests, took 413.2 s (load 8-18).
  - The job ran median 6.7 / max 8.0 min of its 20 over 22 PR runs.
- **UI census tail** (`-p no:xdist --timeout=180 --durations=0`):
  - `Tests/UI/test_console_new_workspace.py`, whole: 3 tests, 28.9 s (load 13-15). Its private-profile markers come from the PR #3684 port.
  - `Tests/UI/test_post_release_workspaces_library_depth.py`: 114.1 s whole, over the 30 s per-entry budget, so it goes in as 14 test-function nodes. Each takes 6.4-8.7 s; the two-theme function takes 15.4 s.
  - `MINIMUM_FILES` 2457 → 2472. There is no re-deal; the coordinator owns the shard balance.
<!-- SECTION:NOTES:END -->
