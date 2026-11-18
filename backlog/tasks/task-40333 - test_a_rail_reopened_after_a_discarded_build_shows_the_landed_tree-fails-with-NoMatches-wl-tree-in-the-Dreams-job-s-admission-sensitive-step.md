---
id: TASK-40333
title: "test_a_rail_reopened_after_a_discarded_build_shows_the_landed_tree fails with NoMatches '#wl-tree' in the Dreams job's admission-sensitive step"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-13 04:26'
created_date: 2026-11-12 19:58
labels:
- follow-up
- tests
- watchlists
- flaky
- ci
priority: medium
references:
- Tests/Watchlists/test_watchlists_bulk_source_authoring.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-13 closeout of the 2026-10-23 burn-down, while reading CI after-times for TASK-39049 (read-only `gh`). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** The test runs in "Run Dreams admission-sensitive contracts" (`.github/workflows/derived-artifacts.yml:1159-1166`), a step of the Dreams boundary job, so when it flakes, an unrelated PR goes red. It failed in 2 of the 38 PR runs after round 12 merged (2026-11-10 ~15:40Z to 2026-11-12 17:03Z), each time as the step's only failure (`1 failed, 109 passed`):
- PR run 43142596949 (branch codex/send-feedback-dev-regressions, 2026-11-12 02:36Z, job 130937785349);
- PR run 43160188546 (branch codex/task-34569-streaming-badge, 2026-11-12 05:15Z, job 131064791688).

Both failed with `FAILED Tests/Watchlists/test_watchlists_bulk_source_authoring.py::test_a_rail_reopened_after_a_discarded_build_shows_the_landed_tree - textual.css.query.NoMatches: No nodes match '#wl-tree'`.

**Cause, by reading.**
- The test polls `_until(pilot, lambda: shown() == [watchlist["id"]])` (`:1193`). `shown()` calls `screen.query_one("#wl-tree", WatchlistTree)` (`:1190`), so while the reopened rail's tree is not mounted yet, the predicate raises instead of returning False.
- `_until` (`:663-667`) does not catch the error, so a late mount fails the test instead of being waited out.
- 6ee80bd822 ("Fix Console CI follow-up and scope Library outcome reads", merged with #3480 at 03:50Z on 2026-11-12) fixed the sibling `test_a_load_landing_while_a_rail_is_recreated_reaches_it` the same way: its poll now checks `bool(screen.query(pane_type)) and shown() == expected`. It did not touch this test. The second failure's merge ref already included that commit.

No open task names this test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The test's poll returns False while the reopened rail's tree is not mounted, and then waits for the landed rows, so a late mount cannot raise; shown red first under a temporary delay of the tree's mount (or an equivalent forced late mount), then green
- [x] #2 Every other poll in the file that queries a widget the test has just caused to mount gets the same guard
- [x] #3 The file passes 20 of 20 runs of the admission-sensitive step's own command under load, with the load recorded
<!-- AC:END -->

## Implementation Plan

1. Force a late tree mount (temporary: the workbench mounts a rebuilt `#wl-tree` 3 s after removing the old one) and show the node red with CI's `NoMatches '#wl-tree'`.
2. Guard the poll like 6ee80bd822 guarded its sibling: `bool(screen.query("#wl-tree")) and shown() == ...`; show it green under the same forced late mount, then revert the forcing.
3. Audit every other poll in the file for an unguarded query of a widget the test just caused to mount.
4. Run the admission-sensitive step's own command 20 times under load and record the load.

## Implementation Notes

**The named node.** Its poll now reads `bool(screen.query("#wl-tree")) and shown() == [...]`, the guard 6ee80bd822 gave its sibling. The landed load makes the screen's surface drain rebuild the reopened rail's tree (remove, then mount), and a poll between the two halves found no tree.
- Red first (run.sh, base 26b2e677dd): with a temporary edit making `WatchlistsWorkbench._replace_region_content` mount a rebuilt `#wl-tree` 3 s after removing the old one, the node failed 3 of 3 at `:1193` -> `shown()` `:1190` with CI's `NoMatches: No nodes match '#wl-tree'` (a 1 s delay inside the landing hook did not, because `pilot.pause()` waits out message-pump work). Green 3 of 3 under the same forced late mount, then the edit was reverted.

**AC#2: the other polls.** Every other `_until` predicate in the file either uses `screen.query(...)` (which cannot raise), reads screen state, or is already guarded (the sibling). The 20-run sieve below found five more reads of controls the test had just caused to mount or focus, each behind a single `pilot.pause()`; they now wait for the control (a new `_laid_out(pilot, node, selector)` next to `_until`: mounted AND laid out) or for the end state:
- `test_source_authoring_controls_fit_production_layout` (the filter editor's controls; seen as `Region(0, 0, 0, 0)` at `:114`);
- `test_an_open_create_form_offers_a_watchlist_the_tree_load_adds`, `test_an_open_destination_list_keeps_its_highlight_when_choices_change` and `test_backend_switch_preserves_open_source_draft_widget_and_focus` (the create form; `NoMatches '#sources-create-watchlist'` at `:1036`);
- `test_bulk_modal_escape_restores_focus_to_its_launch_action` (the modal the screen pushes on request, and the return; `isinstance(host.screen, BulkSourcesModal)` False at `:141`);
- `test_server_mode_never_enables_local_only_source_actions` (focus must land before the cursor move, or the pane ignores the highlight; `'local:subscription:1' == 'local:subscription:2'` at `:567`).
- Red/green for each under a forced late version, then reverted: the create form and filter editor recomposing 1 s late (old file 7 of 7 red with `NoMatches`, new 7 of 7 green); the modal pushed 1 s late (old red at `:143`, new green); the table focused 1 s late (old red at `:564`, new green).

**AC#3: 20 of 20.** The step's own command (the five targets, `--timeout=300 --tb=short`, no xdist) through run.sh, four processes at a time, 5 rounds, on head f28f65caf9: 20 of 20 passed, 110 passed each, 345-406 s per run, 1-min load 37-48 at run starts and ends (18 CPUs, other sessions' work plus these four). Before the five extra waits the same command failed 4 of 4 runs (2-4 watchlists nodes each, load 63-70), then 1 of 20 and 1 of 8 as each remaining race showed itself.

**Review r14lt-rev1 (Minor).** `_until` returns silently on timeout, and a suspended screen keeps its own `focused`, so the escape test passed with `BulkSourcesModal.action_cancel` made a no-op. It now asserts `host.screen is screen` after the wait. Red under the no-op cancel (temporary, reverted): `assert BulkSourcesModal() is WatchlistsCollectionsScreen` at `:147`; green after.

**Files.** `Tests/Watchlists/test_watchlists_bulk_source_authoring.py`.
