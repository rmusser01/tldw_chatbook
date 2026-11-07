---
id: TASK-38979
title: "Retiring the portrait region editor before it mounts leaves it open, and the Personas performance controller's close never returns"
status: In Progress
assignee: ['@claude']
created_date: 2026-11-06 01:42
updated_date: '2026-11-07 02:04'
labels:
- follow-up
- bug
- personas
- tests
- flaky
priority: medium
references:
- tldw_chatbook/Widgets/Persona_Widgets/portrait_region_editor.py
- tldw_chatbook/UI/Persona_Modules/portrait_regions.py
- tldw_chatbook/UI/Persona_Modules/portrait_performance.py
- tldw_chatbook/UI/Screens/personas_screen.py
- Tests/UI/test_portrait_region_editor.py
dependencies:
- TASK-38747
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. TASK-38747 recorded that `Tests/UI/test_portrait_region_editor.py::test_dialog_validation_is_coalesced_owned_and_drained` fails, then hangs past `--timeout`, in 2 of 3 `-n 4` runs. It did not own the failure. No lane runs the file.

**Measured on dev:**
- `[close]` alone, in a serial probe at load 5.5-7.4, hung in 4 of 19 runs; passing runs took 3.3-4.1 s.
- Alone at `--timeout=300` (load 12.3), `[replacement]` passed. `[close]` hit the timeout and then hung in teardown until it was killed at 1586 s.
- The whole file at `-n 4 --timeout=120` gave 23 passed and 1 failed on the timeout, then hung for more than 20 minutes.

**Mechanism, from faulthandler and asyncio task dumps:**
1. `PortraitPerformanceController.close()` waits on `self._task` (`tldw_chatbook/UI/Persona_Modules/portrait_performance.py:803`).
2. That task is the `handle("regions")` worker, parked in `edit_regions` on `screen.app.push_screen_wait(modal)` (`portrait_regions.py:126`).
3. `discard()` (`portrait_performance.py:246-252`) calls `PortraitRegionEditor.retire()` (`tldw_chatbook/Widgets/Persona_Widgets/portrait_region_editor.py:463-472`).
4. In every hung run, `retire()` ran while the editor was already `app.screen` but `is_mounted` was still False. It set `_retired` and skipped `dismiss_safe_once_when_on_top`.
5. The editor then finished mounting and stayed on top while retired, with the stack `['Screen','PersonasScreen','PortraitRegionEditor']`. Nothing dismisses it, so `push_screen_wait` never resolves.
6. In passing runs, `retire()` ran after mount and the dismiss returned True.
7. After pytest-timeout fails the test, its `finally: await controller.close()` leaves the pytest-asyncio loop finalizer idle in `select` for good, so `--timeout` never ends the process.

**Production reach.** PersonasScreen awaits `controller.close()` on a runtime-backend change (`personas_screen.py:1917`) and on unmount (`:1971`). A discard or close that lands between push and mount would hang those awaits. The window is narrow.

The TASK-38747 branch adds only `PortraitRegionEditor.confirm_quit`; `retire`, `edit_regions` and `close` are unchanged there.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Discarding the draft or closing the controller after the region editor is pushed but before it mounts closes the editor and close() returns; a deterministic test that holds the mount pins this and is red on dev
- [x] #2 The [close] case passes 20 of 20 serial runs and the file passes 3 of 3 runs at -n 4, with the load recorded (dev: [close] hung in 4 of 19 serial runs)
- [x] #3 When a case in this file fails, the run still ends: no case holds pytest past --timeout in teardown
- [x] #4 A Personas runtime-backend switch and a screen unmount made while a region editor is opening both complete
- [ ] #5 Once green, the file joins the UI census with its measured serial time, and its shard stays at least 5 minutes under its cap
<!-- AC:END -->

## Implementation Plan

1. Reproduce deterministically: a mounted test holds `PortraitRegionEditor.on_mount` (class-level async stand-in that awaits an event, then runs the real one), so the editor is `app.screen` with `is_mounted` False. While held, retire it four ways: `controller.discard()`, `controller.close()`, `PersonasScreen.handle_runtime_backend_changed(...)` and `PersonasScreen.on_unmount()`. Release the mount, then wait at most 5 s (plain `asyncio.wait`, no Pilot) for the worker and the retiring call. Red on dev: the editor stays on top and the wait times out; the test's cleanup dismisses the editor so the case still ends.
2. Fix at the editor: `retire()` before mount records the retirement and queues itself (`call_later`) so it closes the editor once mounted, through the same `dismiss_safe_once_when_on_top` path (a covering screen still keeps the close for resume). (`PortraitCompositionReview.retire()` has the same gate; it is outside these criteria and goes to the report as a follow-up.)
3. Bound the file's cleanups: the `finally` blocks that await `controller.close()` first dismiss a still-open editor and bound the close, so a failing case cannot hold pytest in teardown.
4. Measure: `[close]` 20 serial runs, the whole file 3 runs at `-n 4`, with load; then add the file to the UI census tail with its measured serial time, check the shard against its cap.

## Implementation Notes

**Approach.** `PortraitRegionEditor.retire()` gated its dismissal on `is_mounted`. `push_screen` puts the editor on top at once, but its Compose and Mount run later on its own task, so a retire in that window (`discard()`, `close()`, and through them a runtime-backend switch or Personas unmount) set `_retired`, skipped the dismiss, and left the editor open; the regions worker stayed in `push_screen_wait` and `close()` waited on it forever. A retire before mount now queues itself (`self.call_later(self.retire)`); the Callback is handled after `_pre_process` sets `_is_mounted`, so the editor closes through the usual `dismiss_safe_once_when_on_top` path (a covering screen still keeps the close for its resume).

**Tests.** `test_retiring_the_editor_before_it_mounts_still_closes_it[discard|close|backend|unmount]` holds the real editor in the window with a class-level async `on_mount` stand-in, retires it four ways (`discard()`, `close()`, `handle_runtime_backend_changed("local")`, and `PersonasScreen.on_unmount()`, the unmount handler that awaits `close()`), and waits at most 5 s with plain `asyncio.wait`. The file's cleanups go through `close_bounded` (dismiss a leftover editor, give `close()` 10 s, cancel and name stragglers), and the drained test's body awaits use `settled(...)` (10 s, names the task).

**Red/green.** On the base source the new test was 4 of 4 red in about 5 s each ("retired before mount, the editor stayed open (stack ['Screen', 'PersonasScreen', 'PortraitRegionEditor'], retired=True)"), and the run ended (4 failed in 42 s). With the fix, 4 passed.
- AC#2: `[close]` 20 of 20 serial (3.3-5.1 s each, load 8.2-12.3); the file 3 of 3 at `-n 4` (29 passed, 25-30 s, load 8.0-9.9).
- AC#3: with the fix temporarily reverted, the whole file serially at `--timeout=60` ended in 132 s: 5 failed (the four new cases, and `[replacement]`, which hit the window by itself and failed at its 10 s bound) and 24 passed, with no hang.
- AC#5: whole-file serial on a quiet machine was 29 passed in 98.6 s at load about 9 (`--timeout=180`, the lane's flags). After review r11m-rev1 the file is 30 census entries, not one: one per test, and one per case for its five heaviest tests (`test_open_dialog_retires_after_another_editor_changes_saved_artwork`, `test_held_save_rejects_retired_authority_and_stale_revision`, `test_retiring_the_editor_before_it_mounts_still_closes_it`, `test_dialog_validation_is_coalesced_owned_and_drained`, `test_retired_modal_drops_held_preview`). The whole-file entry's margin on shard 22 (slowest recent runs 12.3 and 13.1 min) rested on an estimated CI time. Spread over consecutive shards, no shard takes more than a few seconds from this file. One case lands on shard 3, whose slowest run (16.9 min) was already over the line, and it adds a few seconds there. Per-entry serial times, measured at load about 70 (several times a quiet machine's): test_contain_mapping_refuses_padding_and_accounts_for_cell_aspect 1.2 s; test_keyboard_confirmation_save_cancel_and_reopen 28.3 s; test_changed_save_resets_only_draft_and_stops_exact_sample 20.6 s; test_stale_region_save_cannot_write_replacement_editor 20.7 s; test_unchanged_save_keeps_revision_candidate_and_sample 17.5 s; test_held_save_rejects_retired_authority_and_stale_revision[cancel] 14.9 s; test_held_save_rejects_retired_authority_and_stale_revision[profile] 19.2 s; test_held_save_rejects_retired_authority_and_stale_revision[source] 14.6 s; test_held_save_rejects_retired_authority_and_stale_revision[settings] 14.9 s; test_pointer_padding_resize_and_narrow_keyboard_geometry 25.0 s; test_mask_preview_uses_original_crop_and_compositor_mask 1.9 s; test_retired_modal_drops_held_preview[True] 18.3 s; test_retired_modal_drops_held_preview[False] 15.9 s; test_unknown_saved_algorithm_allows_confirmed_reselection 17.1 s; test_preview_budget_refusal_retains_foreign_reservation 1.7 s; test_open_dialog_retires_after_another_editor_changes_saved_artwork[review-regions-False] 18.4 s; test_open_dialog_retires_after_another_editor_changes_saved_artwork[review-source-False] 15.3 s; test_open_dialog_retires_after_another_editor_changes_saved_artwork[regions-regions-False] 19.8 s; test_open_dialog_retires_after_another_editor_changes_saved_artwork[regions-source-False] 20.1 s; test_open_dialog_retires_after_another_editor_changes_saved_artwork[review-regions-True] 24.3 s; test_dialog_validation_is_coalesced_owned_and_drained[replacement] 24.2 s; test_dialog_validation_is_coalesced_owned_and_drained[close] 18.1 s; test_dialog_validation_is_coalesced_owned_and_drained[read_error] 17.6 s; test_retiring_the_editor_before_it_mounts_still_closes_it[discard] 17.8 s; test_retiring_the_editor_before_it_mounts_still_closes_it[close] 16.8 s; test_retiring_the_editor_before_it_mounts_still_closes_it[backend] 17.0 s; test_retiring_the_editor_before_it_mounts_still_closes_it[unmount] 8.4 s; test_retiring_the_pair_review_before_it_mounts_still_closes_it 25.8 s; test_queued_dialog_validation_cannot_overwrite_discard_status 16.9 s; test_save_completion_drops_removed_validation_control 14.5 s. The CI time is still estimated, not measured, until a PR run executes these entries.

**Sibling site (review r11m-rev1).** `PortraitCompositionReview.retire()` had the same `is_mounted` gate, and its `close()` hung the same way. It gets the same fix (`self.call_later(self.retire)` before mount). New test `test_retiring_the_pair_review_before_it_mounts_still_closes_it[discard|close]` was red 2/2 before the fix ("retired before mount, the pair review stayed open (stack ['Screen', 'PersonasScreen', 'PortraitCompositionReview'])") and is green after. `close_bounded` now dismisses a leftover review too.

**Placement (review r11m-rev2).** The tail block (the Promote node, the 17 quit-modal entries and the 30 region-editor entries) is ordered by `$FU/census_reorder.py`. Hot shards are the ones whose slowest UI Fast Lane job over the last 12 PR runs is not at least 5 min under the 20-min cap: 3 (16.9), 5 (16.7), 8 (15.3), 13 (16.1), 15 (15.0) and 19 (15.0), from `$FU/ui_shard_ci_minutes.json`. The lightest entries take the hot slots, hottest shard first, and the heaviest entries go to the coolest shards. Dealing is the checker's own (`entries[index::32]`), as `$FU/census_shards.py` reproduces. **Shortfall:** the block has 7 hot slots but only 3 entries under 5 s (at load ~70), so 4 hot slots still hold heavier entries. These are shard 8, the Promote node (13.0 s at load ~31); shard 13, `test_retiring_the_editor_before_it_mounts_still_closes_it[unmount]` (8.4 s); shard 15, `test_ctrl_q_waits_for_a_goal_that_is_starting` (13.5 s); and shard 19, `test_save_completion_drops_removed_validation_control` (14.5 s). Raised for the owner. **AC#5 is left open, and the task is In Progress, until the owner rules on the shortfall.** The hot shards were already over the line before these entries, and some entries cannot avoid them. **The final placement is set by the coordinator after the round-11 restack** (re-run `census_reorder.py` on the restacked tail); the shards below are this base's.

| Entry | Shard on this base (slowest recent CI min) | Serial s (load ~70) |
|---|---|---|
| `test_dialog_validation_is_coalesced_owned_and_drained[replacement]` | 22 (13.1) | 24.2 |
| `test_unchanged_save_keeps_revision_candidate_and_sample` | 23 (14.2) | 17.5 |
| `test_pointer_padding_resize_and_narrow_keyboard_geometry` | 25 (12.6) | 25.0 |
| `test_held_save_rejects_retired_authority_and_stale_revision[settings]` | 26 (14.5) | 14.9 |
| `test_changed_save_resets_only_draft_and_stops_exact_sample` | 27 (13.8) | 20.6 |
| `test_retired_modal_drops_held_preview[True]` | 31 (14.0) | 18.3 |
| `test_keyboard_confirmation_save_cancel_and_reopen` | 32 (11.8) | 28.3 |
| `test_retired_modal_drops_held_preview[False]` | 1 (14.5) | 15.9 |
| `test_retiring_the_editor_before_it_mounts_still_closes_it[backend]` | 2 (14.3) | 17.0 |
| `test_contain_mapping_refuses_padding_and_accounts_for_cell_aspect` | 3 (16.9) | 1.2 |
| `test_mask_preview_uses_original_crop_and_compositor_mask` | 5 (16.7) | 1.9 |
| `test_open_dialog_retires_after_another_editor_changes_saved_artwork[review-regions-True]` | 6 (12.7) | 24.3 |
| `test_dialog_validation_is_coalesced_owned_and_drained[read_error]` | 7 (14.1) | 17.6 |
| `test_held_save_rejects_retired_authority_and_stale_revision[profile]` | 9 (13.9) | 19.2 |
| `test_retiring_the_editor_before_it_mounts_still_closes_it[close]` | 12 (14.4) | 16.8 |
| `test_retiring_the_editor_before_it_mounts_still_closes_it[unmount]` | 13 (16.1) | 8.4 |
| `test_held_save_rejects_retired_authority_and_stale_revision[source]` | 16 (14.9) | 14.6 |
| `test_open_dialog_retires_after_another_editor_changes_saved_artwork[review-regions-False]` | 18 (14.0) | 18.4 |
| `test_save_completion_drops_removed_validation_control` | 19 (15.0) | 14.5 |
| `test_unknown_saved_algorithm_allows_confirmed_reselection` | 20 (14.3) | 17.1 |
| `test_open_dialog_retires_after_another_editor_changes_saved_artwork[regions-source-False]` | 27 (13.8) | 20.1 |
| `test_open_dialog_retires_after_another_editor_changes_saved_artwork[review-source-False]` | 28 (14.5) | 15.3 |
| `test_open_dialog_retires_after_another_editor_changes_saved_artwork[regions-regions-False]` | 29 (13.8) | 19.8 |
| `test_stale_region_save_cannot_write_replacement_editor` | 30 (13.6) | 20.7 |
| `test_dialog_validation_is_coalesced_owned_and_drained[close]` | 31 (14.0) | 18.1 |
| `test_retiring_the_pair_review_before_it_mounts_still_closes_it` | 32 (11.8) | 25.8 |
| `test_held_save_rejects_retired_authority_and_stale_revision[cancel]` | 1 (14.5) | 14.9 |
| `test_queued_dialog_validation_cannot_overwrite_discard_status` | 2 (14.3) | 16.9 |
| `test_preview_budget_refusal_retains_foreign_reservation` | 3 (16.9) | 1.7 |
| `test_retiring_the_editor_before_it_mounts_still_closes_it[discard]` | 4 (14.1) | 17.8 |

**Verification after review r11m-rev1** (2026-11-07 UTC; the machine was shared, and the load is given for each run).
- **Head.** 19 modal and fleet files under the scratch boot plugin, `-n 4`, load 20 to 80: 272 passed, 2 failed. The two failures are the inspector-section summary tests, which also fail on base.
- **Base.** The same files: 234 passed and 17 failed, and then the run hung with idle workers. The watchdog killed it after 30 minutes, with about 20 tests unfinished. An earlier base attempt hung for more than 2 hours. Base still has the hangs these tasks fix.
- `Tests/CI -n 4`: 1215 passed, 1 skipped.
- Preflight passes.
- Architecture and size ratchets: 216 passed.
- Boot budget: 32 passed.
- **Stress.** The new and changed nodes are 22 selectors, 45 tests, run at `-n 6` under the plugin, 3 runs per round:
  - Round 1, load 65 to 104: 1 failed, 0 failed, 1 failed.
  - Round 2, load 53 to 77: 2 failed, 1 failed, 0 failed.
  - Each test-wait failure was fixed (bounded waits, a held-save precondition) or traced to the product re-sync above.
  - Round 3, on the final head, load 42 to 61: 1 failed, 0 failed, 0 failed. The one failure is the freshness node's known "Mix-in attempt could not be saved" race, an automatic turn's mix-in anchor, which is reported as a follow-up.

**Verification after review r11m-rev2** (2026-11-07 UTC, load about 25-50).
- **Head.** 19 modal and fleet files under the scratch boot plugin, `-n 4`: 272 passed, 2 failed. The two failures are the inspector-section summary tests, which also fail on base.
- **Receipt and read-state coverage.** `test_console_chat_controller.py`, `test_console_activity_receipts.py`, `test_console_receipt_bootstrap.py`, `test_buddy_hidden_results.py`, `test_console_refresh_read_batching.py` and `test_console_historical_read_cancellation_lifetime.py`: 36 failed and 376 passed on both head and base, with identical failure sets.
- `Tests/CI -n 4`: 1215 passed, 1 skipped.
- Preflight passes.
- Architecture and size ratchets: 216 passed.
- Boot budget: 32 passed.
- **Stress.** The new and changed nodes are 22 selectors, 45 tests, run at `-n 6`:
  - First round: 1 failed, 0 failed, 1 failed. Both failures were the view-all test at 120x35, now fixed: it waits for the toggle, tail and steps to be painted.
  - Final round: 45 passed in each of the 3 runs.
- **Tooling, all in `$FU`.** `collect_ui_shard_minutes.py` writes `ui_shard_ci_minutes.json`. `census_shards.py` shows the shard dealt to each census entry and flags heavy entries on hot shards. `census_reorder.py` permutes one block. The weights are in `modals_census_weights.json`.

**Files.** `tldw_chatbook/Widgets/Persona_Widgets/portrait_region_editor.py`, `tldw_chatbook/Widgets/Persona_Widgets/portrait_composition_review.py`, `Tests/UI/test_portrait_region_editor.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `backlog/docs/lessons-textual.md`.
