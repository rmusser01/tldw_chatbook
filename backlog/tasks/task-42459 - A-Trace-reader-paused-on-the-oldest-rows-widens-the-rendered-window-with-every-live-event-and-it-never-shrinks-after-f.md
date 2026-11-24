---
id: TASK-42459
title: "A Trace reader paused on the oldest rows widens the rendered window with every live event, and it never shrinks after f"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 06:38
updated_date: 2026-11-24 00:08
labels:
- follow-up
- trace
- performance
priority: low
references:
- tldw_chatbook/UI/Screens/trajectory_screen.py
- Tests/UI/test_trajectory_live.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), in TASK-42414's second review. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** Past 500 records each new turn drops the oldest rows, so a paused reader on them used to drift. TASK-42414 fixed that by growing the window to keep the reader's anchor row, as it already did for the cursor row: `_apply_live_snapshot` sets `_visible_count = max(_visible_count, window)` from `_window_for_key` (`UI/Screens/trajectory_screen.py:845-868`). The cost is that while that reader stays paused, every arriving event widens `_visible_count`, and live rebuilds render synchronously every 0.5 s. Nothing shrinks the window again after `f` resumes follow: it is reset to `min(_total_records, PAGE_SIZE)` only on a fresh load (line 400). So on a long live run the paused render keeps growing, and stays large after the reader returns to the tail. No lag has been measured; this task decides whether a bound is needed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A measurement over a long live run with a reader paused on the oldest rows (window size and render time per rebuild, recorded in the notes) decides whether the paused window needs an upper bound; if it does, the window has one and the anchor row stays in view
- [x] #2 After `f` resumes follow, the rendered window returns to the page size at the tail
- [x] #3 A test covers the shrink after `f` (and the bound, if one is added); it fails on the current screen code
- [x] #4 The existing paused-reader and live Trace tests pass unedited
<!-- AC:END -->

## Implementation Plan

1. Measure a long live run with a reader paused on the oldest rows: window size and render time per rebuild.
2. Decide on a bound from the measurement; add one only if the numbers call for it.
3. On `f`, shrink the window back to the page size at the tail.
4. Test the shrink (and any bound), red on the current screen code; run the paused-reader and live tests unedited.

## Implementation Notes

**Measurement** (`_apply_live_snapshot` wall time, synchronous on the UI thread; a reader paused at row 0 of 260 turns; load ~22). The window grows with every live event:

| Window (records) | Rows | Time a rebuild |
| --- | --- | --- |
| 502 | 754 | 11 ms |
| 980 | 1471 | 23 ms |
| 1480 | 2221 | 29 ms |
| 1980 | 2971 | 118 ms |
| 2480 | 3721 | 147 ms |
| 3480 | 5221 | 175 ms |
| 3980 | 5971 | 232 ms |

After `f` the window stayed at 4000, and a following rebuild took 197 ms. These times leave out the DataTable's own `_update_dimensions` pass. Rebuilds run every 0.5 s while a run streams, so past about 2000 records the UI is blocked a quarter to half of the time. The screen already sends first renders over `WORKER_THRESHOLD` (1000) to a worker for the same reason.

**Decision: a bound is needed.**
- Once a paused window reaches `WORKER_THRESHOLD` records, `_apply_live_snapshot` holds new snapshots instead of rendering them. The reader's rows and anchor stay in view untouched, and the state line reads "PAUSED · new events waiting · f resume". The window can pass the bound by at most one rebuild's arrivals.
- `f` returns the window to the newest page (`min(total, PAGE_SIZE)`) and renders any held snapshot at the tail (`_install_live_snapshot`, split out so `f` does not re-read follow from the paused position). It also moves the cursor to the newest row if its row left the window; otherwise it would land on the "Earlier events" row.
- Scrolling back to the bottom of a held ledger drops the hold and starts a fresh rebuild on the next tick. Rendering it from the poll tick would make the render paths timer-reachable, which `test_timer_path_static_update_inventory` rejects.

**Tests** (`Tests/UI/test_trajectory_live.py`):
- `test_f_returns_a_grown_paused_window_to_the_newest_page`: red at dev with `506 == 500`.
- `test_a_paused_window_at_the_bound_holds_new_events_until_f`: red at dev with "the paused window was re-rendered".
- Both are green after. With the cursor move removed, the first is red on `cursor_row` (`0 == 750`).

**Existing tests.** The paused-reader and live tests pass unedited.

**Docs.** `Docs/User_Guide/console/chat-basics.md` describes the held events and what `f` does.

**Soak.** 32 of 32 runs green (157 tests each, load 49-89; one combined serial run of the four Trace census files with the lane's flags: 124 passed in 222.9 s at load ~74-85) (see TASK-42457).

**Bound kept.** The coordinator ruled on 2026-11-23 to keep the bound.

**Review fixes (commit 787d0059cc).**
- **Export dropped held events.** `action_export_trace` now exports the held snapshot when there is one. This is checked in `test_a_paused_window_at_the_bound_holds_new_events_until_f`: at 057ed23ecf, "export wrote 1520 of 1522 records".
- **Scrolling back to the tail kept the grown window.** That release path now runs `f`'s path (`action_resume_follow`) through `call_later`, so the render is not part of the poll tick. The window shrinks, the held events render at the tail, and the cursor moves to the newest row if its row is gone. `test_timer_path_static_update_inventory` still passes: 19 tests. New test `test_scrolling_back_to_the_tail_of_a_held_window_shows_the_newest_page` was red at 057ed23ecf (`1500 == 500`).
- **`f` on a static Trace threw away pages loaded with `e`.** `f` now shrinks only a live window (`_snapshot_builder is not None`). New test `test_f_on_a_static_trace_keeps_the_pages_loaded_with_e` was red at 057ed23ecf (`500 == 1000`).
- **Formatting.** The two `until(lambda: ...)` waits added this round now match ruff format. `ruff format --diff` reports no hunk in this round's lines. The file's other hunks were there at base and are left alone.
- **Docs.** `chat-basics.md` says an export includes the waiting events, and that scrolling back to the bottom shows them the way `f` does.
- **Soak after the review fixes.** 20 of 20 runs of the five Trace files at `-n 4` passed (168 tests each, load 40-80). `test_trajectory_live.py`, a census entry, now has 41 tests: 37.8 s serial with the lane's flags at load 42. Tests/CI at `-n 4`: 1823 passed, 2 skipped.

**Re-review fixes (commit b56b6f1148).**
- **An older held snapshot was drawn over a newer ledger (a regression from 787d0059cc).** A live update that arrived after the reader reached the bottom of a held ledger was drawn, but the hold was not cleared. The tick's deferred resume then drew the older held snapshot over it (the probe ended on 1522 of 1524 records). Once something is held, `_apply_live_snapshot` now keeps the newest snapshot as what is held. A reader who is following again gets it at once through `action_resume_follow`; a paused reader keeps holding it.
  - Test: `test_a_newer_update_at_the_tail_of_a_held_window_is_not_redrawn_over`, red at 5cc37995e3 ("the ledger shows 1522 records, not 1524").
  - Control: reverting the hold condition fails it again.
- **The deferred resume overrode a reader who scrolled up again.** The tick now defers `_resume_if_still_at_tail`. It re-reads follow first and resumes only if the reader is still at the tail with something held.
  - Test: `test_a_reader_who_scrolls_up_again_before_the_resume_stays_paused`, red at 5cc37995e3 ("the held events were drawn").
  - Control: deferring `action_resume_follow` directly fails it.
- **Soak after the re-review fixes.** 20 of 20 runs of the five Trace files at `-n 4` passed (173 tests each, load 45-86). `test_trajectory_live.py` now has 46 tests: 37.3 s serial with the lane's flags at load 45. The timer-path and guarded-modal Architecture tests passed (26). Tests/CI at `-n 4`: 1822 passed. Its one failure, `test_bootstrap_root_cleanup.py::test_owned_root_is_removed[fork-child-terminated]` (a child process did not end in time at load ~50-60), passed 3 of 3 alone. The branch does not touch Tests/CI.
