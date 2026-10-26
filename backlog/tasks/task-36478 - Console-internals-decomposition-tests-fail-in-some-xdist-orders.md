---
id: TASK-36478
title: Console internals decomposition tests fail in some xdist orders
status: Done
assignee:
  - '@claude'
created_date: '2026-10-23 11:02'
updated_date: '2026-10-26 11:00'
labels:
  - tests
  - console
dependencies: []
references:
  - Tests/UI/test_console_internals_decomposition.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Exists on dev today (measured on dev 7700c63a68 during Roleplay frame B2a's gates, TASK-33910.3; B2a changes no Console code and does not fix it): test_console_empty_regions_do_not_stack_nested_terminal_frames failed in a single-process replay of one xdist worker's 180-test sequence (9 component_pattern_governance, 84 console_character_avatar, 87 console_internals_decomposition tests) on BOTH arms, twice (base 21 and 20 failed, head 23 and 22 failed), while passing alone. In the same file, test_console_composer_cursor_blink_toggles, test_console_composer_hides_cursor_when_blurred, test_console_composer_shows_cursor_when_focused, test_console_rag_query_validation_blocks_unsafe_markup and test_console_empty_transcript_promotes_setup_card_over_banner failed under xdist and passed every alone run on both arms, and three single-process runs of the whole file failed different sets on both arms (11-61 failures). The evidence (traces, replay id lists, logs) is in the B2a evidence directory, gates/ and draft/t10/.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The order dependence is found: the earlier test that leaves the state these tests read is named, with a minimal reproduction
- [x] #2 No test in the file fails because of test order: in three single-process runs each of the file and of the recorded worker sequence, every failure that depends on order on dev passes here; tests that fail alone on dev are tracked in TASK-36478.1 (reworded 2026-10-26, controller ruling; original text in the notes)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce on fu-base (dev 0225e77310) with the b1_bootstrap_all evidence plugin: single-process file runs, the recorded B2a worker sequence (f5-replay.ids), and a per-test config-cache trace (cfgleak_trace plugin) to see what state crosses test boundaries.
2. Bisect the recorded sequence down to a minimal (two-test) reproduction for test_console_empty_regions_do_not_stack_nested_terminal_frames and name the leaking test; find the within-file leakers the same way.
3. Write a failing regression test first: an outer test that runs a child pytest over two bootstrap-profile probes (a writer that persists config and mutates the cached settings dict, then a reader that asserts the profile config is pristine). Show it RED on the current conftest for the right reason.
4. Fix at the source of the shared state: Tests/conftest.py isolate_test_environment hands every bootstrap-profile test the same config.toml; snapshot it before each such test and restore it (and drop config.py's settings/config caches) after teardown, so no bootstrap-profile test sees another's config writes or in-place cache mutations.
5. Verify: regression test GREEN plus named mutations; the minimal repro RED on base and GREEN on head; three single-process runs of the file and of the recorded worker sequence on each arm; paired runs of the affected files under the flake protocol; ruff check/format; preflight.
6. Record tests that fail ALONE on dev (not order-dependent) separately with their reasons; file a follow-up task for them rather than editing their assertions blind.
7. Implementation Notes, ACs from evidence, report, PR body.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Every bootstrap-profile test in one pytest process shared ONE config.toml, and nothing reset it between tests. That exists on dev today and is fixed here.

**Root cause, with minimal reproductions.**
- test_console_character_avatar.py::test_character_section_composes_when_config_on persists a ready llama.cpp provider (test_console_native_chat_flow's `_configure_native_ready_console`). Run before it, test_console_empty_regions_do_not_stack_nested_terminal_frames boots ready, its composer takes focus and draws a thick left edge: red on dev 3/3, green here 3/3.
- Inside the file, test_console_inspector_live_work_sources_stay_near_top persists the open Inspector (`console.rail_state`), so the next test_console_workbench_weights_transcript_as_primary_region fails: red on dev 3/3, green here 3/3.
- `app.app_config["chat_defaults"] = {}` edited load_settings()'s cached dict in place, and the edit outlived its test.
- Found with a per-test trace of `config._SETTINGS_CACHE` and a bisect of the recorded 180-test B2a worker sequence.

**Fix.** Tests/conftest.py `isolate_test_environment` reads the bootstrap config bytes before each bootstrap-profile test (private-profile children excepted). After teardown it writes them back if they changed and drops config.py's caches. This generalises Tests/Wizards/conftest.py's TASK-34100.1 restore, which stays in place.

**Tests.** Tests/test_bootstrap_profile_config_reset.py runs writer/reader probe pairs in a child pytest, red before the fix. Mutations: M1 (no restore), M2 (no cache drop) and M3 (no byte write-back) are red. M5 (mode 0o644) survives, because config.py accepts a 0o644 file; the comment was corrected.

**AC#2 ruling (controller, 2026-10-26).** The original text was "The file passes in any order: three single-process runs of it and of the recorded worker sequence pass on dev". It is not met as written. In three paired single-process runs each of the file and of the recorded sequence, every order-dependent victim fails on dev and none fails here. The file still fails here, for reasons that exist on dev and do not depend on order:
- 13 tests fail alone;
- setup-modal, focus and caret-blink timing races;
- full-app timeouts under load.

AC#2 is reworded to the order-dependence scope, and the rest is TASK-36478.1. Two tests passed on dev only because of the leak and now fail every time, as they already did alone on dev: test_console_gate15_keeps_existing_chat_send_control_reachable and test_console_collapsed_paste_confirm_click_outside_token_resets_to_collapsed.

**Paired evidence.**
- Agent: an xdist run of three files with the plugin on gave dev 21 failed, branch 22 failed; the flake protocol cleared all 11 branch-only failures. Single-process paired runs agree.
- Controller, after the rebase onto dev 7a5e2513c5: a plugin-off xdist run of the 940 files the conftest change reaches gave dev 3092F/24874P/40E and head 3735F/23966P/305E, with 966 head-only and 57 base-only failures.
  - 915 of the head-only failures came from one worker that ran 14,671 tests. It ran out of file descriptors (`filedescriptor out of range in select()`), a Backup_Recovery test failed mid-pause, and the rest of that worker failed with `RecoveryRequired: storage_locally_paused`. The base arm's busiest worker ran 9,840 tests.
  - Descriptor growth over the same 3,282-test sequence is the same on both arms: 18 to 1,347 on dev, 18 to 1,352 here. The leak exists on dev today, is not caused by this change, and is filed as TASK-36478.2.
  - The other 51 head-only ids:
    - 13 fail alone on both arms;
    - 30 pass alone on both arms (xdist-only under load; base had 57 of its own);
    - 3 fail alone on dev only;
    - 6 failed alone on head in one pass, and dev failed each of them alone within the flake protocol.
  - No regression.
- Evidence: .worktrees/.fu-evidence/36478/ (aff-v1-*, fd-*.tsv, alone-aff-v1.tsv, flake-aff-v1.txt).

**Limitations.** Each exists on dev today and is not fixed here.
- test_console_collapsed_paste_backspace_deletes_whole_chunk failed here 2 of 3 in the recorded sequence order and on dev 0 of 3; on dev it failed 1 of 3 in file order, and it passes 10/10 alone on each arm. This is the composer-focus family in TASK-36478.1.
- The bootstrap profile's data directory is still shared between bootstrap-profile tests; no failure from it was seen.
- Tests/Wizards/conftest.py's own restore is now redundant for the config bytes.

Every first config read in a bootstrap-profile test now rebuilds the settings cache (20-50 ms in isolation), as a per-test sandbox's first read already does.

**Files.** Tests/conftest.py, Tests/test_bootstrap_profile_config_reset.py, backlog/docs/lessons-testing-evidence.md, and the task files TASK-36478, TASK-36478.1 and TASK-36478.2.
<!-- SECTION:NOTES:END -->
