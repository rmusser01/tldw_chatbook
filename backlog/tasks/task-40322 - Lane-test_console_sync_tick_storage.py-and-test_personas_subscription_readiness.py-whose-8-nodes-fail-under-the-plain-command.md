---
id: TASK-40322
title: "Lane test_console_sync_tick_storage.py and test_personas_subscription_readiness.py, whose 8 nodes fail under the plain command"
status: Done
assignee:
- '@claude'
created_date: 2026-11-12 19:36
updated_date: 2026-11-13 03:53
labels:
- follow-up
- tests
- console
- personas
- lanes
priority: medium
references:
- Tests/UI/test_console_sync_tick_storage.py
- Tests/UI/test_personas_subscription_readiness.py
- scripts/ui_pr_gate_census.txt
dependencies:
- TASK-38948
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (send stream, TASK-38948 reviews r13n-rev1 and r13n-rev4). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** Both files guard round-13 fixes and run on no PR lane, so a regression would land silently.
- `Tests/UI/test_personas_subscription_readiness.py::test_handoff_readiness_refreshes_expiry_and_stale_snapshot_without_edits` (`:207`) is the only guard against the Personas Start-chat gate flickering to "Checking Claude", which round 13 fixed. All 8 nodes of the file (5 functions, one parametrized 4 ways) fail under the plain command with RecoveryRequired locally, on base and head alike, so the file was not added to the census: TASK-38948 verified the re-pin only under a temporary `@private_profile_test` marker (green 3 of 3 at load 37-50).
- `Tests/UI/test_console_sync_tick_storage.py` (2 tests, module `bootstrap_profile`) guards the once-a-second Character-scope check; review r13n-rev4 fixed its `_tick` to wait out an owed whole-sync replay (f85a33f521; 0 of 48 fails against 17 of 48 unfixed).

Neither path appears in `scripts/ui_pr_gate_census.txt`, `.github/` or `scripts/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Personas file passes under its plain command at `-n 4` (a profile marker or a harness fix, with the RecoveryRequired cause recorded)
- [x] #2 Both files run on a required PR lane at the census tail, with lane times recorded
- [x] #3 Both pass 3 of 3 whole-file runs at `-n 4`
<!-- AC:END -->

## Implementation Plan

1. Re-run both files on base 26b2e677dd with the plain command.
2. Find the Personas file's RecoveryRequired read and mark the file the way its sibling mounted readiness files are; fix whatever is red once it runs.
3. Append both files at the census tail, measure their serial lane times, and run each whole file 3 times at `-n 4`.

## Implementation Notes

- **Base 26b2e677dd, plain command, `-n 4`:** Personas 8 of 8 failed, sync-tick 2 of 2 passed (load 52-60).
- **RecoveryRequired cause.** The mounted `AppFooterStatus.compose` builds the Tamagotchi (`_build_tamagotchi`), which calls `get_cli_setting`. Under the per-test config sandbox that read raises `RecoveryRequired: raw_source_selection_changed` (`Backup_Recovery/raw_participants.py:307`, through `config_participants`). The file is now `pytest.mark.bootstrap_profile`, like `test_console_sync_tick_storage.py`.
- **A product defect showed up once the file ran.** `test_completion_between_header_and_inspector_reads_refreshes_both_surfaces` failed 3 of 3 (TimeoutError at :203; the header's blocked chip never cleared).
  - The 0.25 s handoff poll returns early when its cheap gate equals the last poll's gate (fad849155b, task-34914, 2026-10-13).
  - A direct `_sync_title_and_console_actions()` can paint a read that gate never saw: a completion between the header read and the inspector read. 0077b872f1 handles that case, and the gate made it unreachable, so the header stayed "blocked" after the credential was ready.
  - Fix: `_sync_title_and_console_actions` clears `_console_handoff_readiness_gate`, so the next poll checks in full once, then goes back to the cheap path. The cost is one extra provider-readiness check after a direct sync.
  - Red 3 of 3 before, green after. The covering `test_personas_handoff_poll_gate.py` passes, and so do the 13 readiness and handoff nodes of `test_personas_workbench.py`. One of those failed once at `-n 4` (`OptionDoesNotExist 'character:1'`), then passed 3 of 3 serially and in a rerun at `-n 4`.
- **Whole-file runs at `-n 4`:** Personas 8 of 8 passed, 3 of 3 runs (load 52-65). Sync-tick 2 of 2 passed, 3 of 3 runs (load 45-55).
- **Lane.** Both files are at the census tail by node, 10 entries: the Personas file's 8 (its parametrized test per case) and the sync-tick file's 2. Each file is over the 30 s per-entry budget whole. `MINIMUM_FILES` is 2038 here; no re-deal.
- **Serial lane times** (`-p no:xdist --timeout=180 --durations=0`): Personas file 38.3 s at load 47 (about 10-14 s a node at `-n 4`); sync-tick file 33.7 s at load 45, almost all of it `test_sync_ticks_during_a_run_check_the_character_scope_once_a_second` (one node, which cannot be split further).
- **Size.** `personas_screen.py` is exactly at its 16,835-line row (the fix is one line). The module size ratchet passes.
- **Files.** `tldw_chatbook/UI/Screens/personas_screen.py`, `Tests/UI/test_personas_subscription_readiness.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.

**Review r14ti-rev1 (2026-11-13): per-entry serial times for the 10 census nodes.** Measured in one serial run of all 59 entries this branch added (`-p no:xdist --timeout=180 --durations=0 --durations-min=0`, setup + call + teardown per entry, 73 passed in 167 s, load 19-26, 2026-11-13). The first entry in a process carries the shared import and fixture start-up.
    - `test_handoff_readiness_keeps_ui_responsive_and_finishes_automatically[file-ready]` 1.2 s
    - `test_handoff_readiness_keeps_ui_responsive_and_finishes_automatically[keychain-ready]` 0.9 s
    - `test_handoff_readiness_keeps_ui_responsive_and_finishes_automatically[file-missing]` 1.2 s
    - `test_handoff_readiness_keeps_ui_responsive_and_finishes_automatically[file-error]` 0.9 s
    - `test_completion_between_header_and_inspector_reads_refreshes_both_surfaces` 1.1 s
    - `test_handoff_readiness_refreshes_expiry_and_stale_snapshot_without_edits` 1.4 s
    - `test_credential_completion_uses_current_provider_and_selection` 1.6 s
    - `test_unmount_stops_handoff_refresh_before_credential_completion` 1.2 s
    - `test_sync_ticks_during_a_run_check_the_character_scope_once_a_second` 12.5 s
    - `test_a_scope_invalidated_during_a_run_check_is_checked_at_once` 0.3 s
