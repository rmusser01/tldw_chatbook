---
id: TASK-42414
title: "The Trace screen follow-suspend test checks follow before its scroll lands and fails UI Fast Lane 39"
status: Done
assignee:
- '@claude'
created_date: 2026-11-22 21:17
updated_date: 2026-11-23 03:34
labels:
- follow-up
- ci
- tests
- flaky
priority: high
references:
- Tests/UI/test_trajectory_live.py
- tldw_chatbook/UI/Screens/trajectory_screen.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22). Checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports); paths are under `tldw_chatbook/` unless they start with `Tests/` or `scripts/`.

**Why.** `Tests/UI/test_trajectory_live.py::test_scrolling_up_suspends_follow_until_f_resumes` is the most frequent red in the required UI Fast Lane. In a scan of the 140 Derived Artifacts PR runs created from 2026-11-20 00:00Z to about 21:50Z on 2026-11-22 (every attempt; times are job start times), "UI Fast Lane (39)" failed in 39 jobs, from 2026-11-20 09:47Z to 2026-11-22 20:49Z, and every one of them failed on this node alone (for example 44973100546, 44978286545, 45027696251, 45048643145). The file is a whole-file census entry, and the whole file (with the rest of the 205-test set checked) passes locally at the base.

**The failure.** `test_trajectory_live.py:263`, `assert screen._follow is False` → `assert True is False`. The test calls `table.scroll_to(y=0, animate=False)`, waits one `pilot.pause()`, then calls `screen._sync_follow_from_scroll()` (`UI/Screens/trajectory_screen.py:868-886`). That method returns early while `table.max_scroll_y <= 0`, and otherwise reads `table.scroll_y`. On a loaded runner one pause is not enough for the table's layout and the scroll to land, so the sync still sees the tail and follow stays on. The later `assert table.scroll_y == 0` and the `f` step have the same single-pause shape.

**Test drift or a product race.** It may not be only the test. When the screen re-renders while following, it queues `table.call_after_refresh(table.scroll_end, animate=False)` (`UI/Screens/trajectory_screen.py:858-866`). If such a pending scroll lands after the reader's own scroll up, the table snaps back to the tail: a real snap-back a user would see, not a test timing problem. The fix has to establish which it is.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The notes record whether the failure is test timing or a pending tail scroll that lands after the reader scrolled up, with the evidence for it
- [x] #2 A scroll up suspends follow and keeps the reading position on a loaded runner: the test passes without depending on how many refreshes run between its steps, and if the cause is the pending tail scroll, the product no longer snaps back
- [x] #3 The change is shown needed: with the table's layout or scroll delayed by a temporary edit (reverted), the old test fails at line 263 and the new one passes
- [x] #4 The test passes in repeated runs of its file under parallel load (at least 20 runs, or a soak recorded in the notes)
- [x] #5 The file's other tests pass unedited
<!-- AC:END -->

## Implementation Plan

1. Decide (a) test timing or (b) product race from the code, the CI log for 44973100546 (UI Fast Lane (39) job 137058700280) and forced-delay probes: (a) delay the reader's `scroll_to(y=0)` landing or the table's layout by a temporary edit; (b) hold a live rebuild in flight, scroll up, then let it land.
2. Product (if (b) reproduces): `_apply_live_snapshot` re-reads follow from the reader's scroll position before its re-render resets it, so a rebuild that started at the tail cannot snap a reader who has since scrolled up back down. Track queued tail scrolls until they land (one helper for live follow and `f`), and let the scroll-position read skip while one is in flight, so a re-render's reset is never read as a reader scroll. This replaces `f`'s 1 s wall-clock grace.
3. Test: rewrite the follow-suspend test to wait for each scroll on the table's own geometry with a deadline (overflow, tail, reader's scroll, rebuild row count, no tail scroll in flight, `f`'s tail), keeping every follow and position assertion; add a product test for the in-flight-rebuild snap-back in the same file (a whole-file census entry, no census edit).
4. Show the old test red at :263 and the new one green under each temporary delay; show the new product test red with the fix's re-read removed.
5. Siblings: grep the trajectory and trace tests for one-pause scroll reads and fix those with the same cause.
6. Soak the file 20 times at -n 4 and 5 times serially with the UI lane flags; run covering tests, preflight, ruff F and the size ratchets; notes, ACs, Done.
7. Added after review (coordinator decision): open a live Trace at the tail with follow engaged, and keep a paused reader's rows across a live rebuild, each with a test red at base; then repeat step 6.
8. Review rev1 (coordinator decision, fix every finding): navigation during loading must not open at the tail; keep the reader's row through same-frame re-renders and a sliding page; record the known gap; then repeat step 6.
9. Coordinator decision: a render that lands while the first worker build runs must end the LOADING state; then repeat step 6.

## Implementation Notes

**Verdict: both.** The CI red is test timing (a). A separate product race (b) also exists, and the old test could not see it. Both are fixed. On the coordinator's call, two more user-facing defects on the same screen are fixed in this branch (see "Follow-ups fixed here").

- **(a) Test timing.**
  - CI run 44973100546, job 137058700280 (UI Fast Lane (39)): `test_trajectory_live.py:263 assert True is False`, call time 0.24 s.
  - No live rebuild can have run by :263, because the revision has not changed since mount.
  - `Widget.scroll_end` is deferred (`call_after_refresh`). `DataTable.scroll_to` (a `ScrollView` override) applies at once.
  - So either the table had not laid out its rows (`max_scroll_y` 0, and the sync returns early), or the one pause after the test's own `scroll_end` was too short. In the second case the reader's `scroll_to(y=0)` applied first, and the stale tail scroll landed in the next pause, before the check. The log cannot tell which.
  - Controls (temporary edits, reverted):
    - Ledger's first render delayed 0.4 s: old test red at :263, 3 of 3.
    - Test's tail scroll held until the reader's scroll, then released: old test red at :263, 3 of 3 with both the dev and the final screen file. The new file passes 3 of 3 under the same hold.
    - A first control delayed the reader's `y=0` scroll instead. It gave the same red, but a DataTable cannot delay that scroll; I kept the hold control as the faithful one.
    - A plain 30 ms delay on every deferred table callback moved the landing past the check, and the old test passed 3 of 3.
- **(b) Product race.**
  - A probe at base held a live rebuild that a tick had started at the tail, scrolled the reader to y=0, then released it. The rebuild snapped the reader back to the tail (`scroll_y` 160 of 160, follow still on).
  - `_apply_live_snapshot` used the follow flag sampled by the tick before the worker ran, and then queued `scroll_end`.
  - The old test's "new records" step waited on `row_count > 123`, which the 180-row ledger already met, so it checked before the rebuild landed.
- **Product fix (`UI/Screens/trajectory_screen.py`).**
  - `_apply_live_snapshot` re-reads follow from the scroll position before its re-render resets the scroll.
  - Every scroll the screen queues goes through `_queue_ledger_scroll(tail=...)`: the tail for live follow, `f` and the open, or the reader's recorded row for the paused restore, resolved against the rows present when it lands. `_scrolls_pending` counts each one until its `on_complete` reports it landed. It also counts each re-render's own scroll reset for two refresh hops.
  - `_sync_follow_from_scroll` skips while anything is pending, so the top-of-table position a re-render leaves behind is never read as the reader scrolling.
  - The count replaces `f`'s 1 s wall-clock grace and the `time` import. A grace renewed on every rebuild would have made follow impossible to suspend while streaming.
- **Follow-ups fixed here (coordinator decision).**
  - **Follow engaged on open.** A live Trace on an overflowing conversation opened at the top while showing FOLLOWING. The first tick then paused it (probe: follow True and y 0 at open, follow False 1.2 s later).
    - Now its first ledger render puts the cursor on the newest row and queues the counted tail scroll. It opens where the label says, consistent with the User Guide paragraph ("follows new events while you are at the bottom").
    - The cursor moves too, so the first key press does not jump back to the top.
    - Test: red at plain dev, at the first commit and at the second (`assert False is True`, :269/:270). With only the cursor move removed it is red on `cursor_row` (0 vs 179).
  - **Paused reader keeps their rows.** `DataTable.clear()` reset the scroll and only the cursor row came back, so a paused reader who had scrolled away from the cursor was thrown back to it on every rebuild (probe: y=40 → 0).
    - The rebuild now scrolls back to the row the reader had at the top, by row key, because past 500 records the newest page slides and a raw offset shifts. A row that is gone keeps the old offset, which the table clamps.
    - Test, parametrized `appended` / `newest-page-slides`: red at plain dev and at the first commit (`assert 0 == 40`, shown with a temporary copy that read the pending count with `getattr`). Red 3 of 3 with the restore removed. With a raw-offset restore, `newest-page-slides` is red (`turn:t24` where `turn:t23` was).
- **Review fixes (rev1, head 4483cbb4db).**
  - **Navigation while loading.** A live Trace over 1000 records builds its ledger in a worker. A navigation key (`n`, `j`, `u`) or a timeline click before that lands renders at once with a row to restore. That render counted as the first one, so the Trace opened at the tail with the picked row off screen, still FOLLOWING. Opening at the tail now also needs `_pending_restore_key is None`.
    - `test_navigating_while_a_live_trace_loads_keeps_the_cursor_in_view`: red at 4483cbb4db (cursor row 1, view 1777..1801), green after, and red again with the guard removed.
  - **A reset read as the reader's place.** The restore read the reader's row from a scroll that a re-render resets to 0 at once. A second paused rebuild, or a search key, `t`, `e` or a filter, in the same frame sent the reader to row 0.
    - Each re-render now counts its reset until the table re-lays out. `_note_reading_row` records the reader's anchor only while nothing is pending; otherwise it keeps the last one. The restore resolves the anchor when it lands.
    - `test_a_second_rebuild_in_the_same_frame_keeps_the_readers_row[paused-rebuild|re-render]`: both red at 4483cbb4db (`turn:t23` became `__load_earlier__`), green after.
    - Red in both cases with the anchor re-read on every call; red in the `re-render` case with the re-render count removed.
  - **Drift on the oldest rows.** Past 500 records each new turn drops the oldest rows, so a paused reader on them drifted (u10 → u11 → u12 → u13). The window now grows to keep the reader's anchor row, as it already did for the cursor row (`_window_for_key`). With "Earlier events" at the top, the anchor is the row under it.
    - `test_a_paused_reader_on_the_oldest_rows_keeps_them[earlier-row-at-top|oldest-record-at-top]`, three rebuilds each: both red at 4483cbb4db (u10 → u11 on the first rebuild), green after.
    - Red in both cases with only the cursor row kept in the window; red in `earlier-row-at-top` with "Earlier events" used as the anchor.
  - **Known gap (not fixed).** A reader scroll that arrives while one of the screen's own deferred scrolls is in flight (one or two refresh hops after a live rebuild, an `f` or the open) is overridden when that scroll lands. The same window also delays the follow check for that scroll to the next poll tick. Closing it needs reader-input events from the DataTable (wheel, keys, scrollbar) rather than its scroll position. The User Guide no longer claims a scroll is kept "even when an update was already loading"; it says a scroll made at the very moment an update is drawn can be undone by it.
  - **LOADING stuck after a render during loading** (found here; fixed on the coordinator's call). Navigating or searching while a Trace over 1000 records is still building its ledger left the state line on "LOADING · Building trace ledger…" for good. The synchronous render never cleared `_loading`, and the worker's later build is dropped as stale. A probe showed this at plain dev and at 4483cbb4db.
    - Any render that lands now clears `_loading`, because it has built the ledger. The worker's older build stays dropped: the synchronous render already reflects the current snapshot, window and filters, so nothing in the stale build is still wanted.
    - A retry in flight still reads RETRYING, since that label comes first, so clearing the flag during a live retry changes nothing visible. I removed a guard that only made that case explicit.
    - The navigation test now waits for the worker to finish, then asserts no LOADING in the state line and `_loading` False. It was red 2 of 2 at 9393084fa5 (`LIVE · FOLLOWING · LOADING · Building trace ledger… · 1200 events`) and is green after.
    - Its wait ends on the worker rather than on `_loading` alone, because `_loading` now clears at the synchronous render. A wait on it would end before the late, dropped build could be shown harmless.
- **Tests (`Tests/UI/test_trajectory_live.py`, a whole-file census entry; no census edit).**
  - `test_scrolling_up_suspends_follow_until_f_resumes` waits on the table's geometry with a deadline (existing `wait_for_condition` fixture, 10 s / 10 ms) and keeps every follow and position assertion. Its own `scroll_end` is `immediate=True`.
    - A deferred one failed 3 of 25 soak runs, plus 2 of 12 in a hunt, at the "new records" check (`assert True is False`).
    - An instrumented probe showed the test's `scroll_end` landing 14 ms after the reader's `scroll_to(0)`. The rebuild then read the reader at the bottom.
    - The snap-back and back-to-back tests got the same change.
  - New tests:
    - `test_a_live_trace_opens_on_its_newest_events_and_keeps_following`
    - `test_a_reader_scroll_during_a_live_rebuild_is_not_snapped_back` (red 3 of 3 without the re-read, `assert 160 == 0`)
    - `test_back_to_back_live_rebuilds_keep_following_the_tail` (red 3 of 3 without the pending skip)
    - `test_a_paused_reader_keeps_their_place_across_a_live_rebuild[appended|newest-page-slides]`
    - The three rev1 tests above.
  - The file's other 12 tests are unedited.
- **Siblings.**
  - I scanned the 84 failed Derived Artifacts runs from 2026-11-20 03:35Z to 2026-11-22 18:09Z (156 failed UI jobs in 71 runs). This node failed in 29 of those jobs, and no other Trace or trajectory node failed.
  - `test_trace_responsive.py`'s inspector `end` checks have a one-pause shape, but they assert `max_scroll_y > 0` first and never failed, so I left them unedited.
- **Soak of the final file** (23 tests, with the LOADING fix; load is on 18 CPUs).
  - -n 4, 20 runs: 20/20, load 33-37.
  - Serial with `-p no:xdist --timeout=180`, 5 runs: 5/5, 10.8-16.7 s per file at load 34-39. The 600-turn navigation test and the 260-turn tests added several seconds; an earlier 18-test soak took 7.1-8.5 s at load ~30.
  - Rev1 tree, before the LOADING fix: 20/20 at -n 4 (load 30-43), and 5/5 serial (17-26 s at load 41-44).
  - Before rev1 (18 tests): 20/20 at -n 4 (load 29-37), two concurrent loops of 10 runs 20/20 (load 26-32), and 5/5 serial.
- **Checks.**
  - Covering Trace, trajectory, modal and Console-trace files, final tree: 467 passed, 12 failed. All 12 fail the same way at plain dev:
    - the 3 no-lane Trace nodes (`test_trace_filters_navigation::test_live_insertion_before_selection_preserves_event_id_filters_and_viewport`, and `test_trajectory_timeline_integration`'s `test_click_outside_active_brush_atomically_clears_time_and_selects` and `test_timeline_renders_bars_for_timed_snapshot`);
    - 9 nodes in `test_console_tab_switch_binds_composer.py`, which stubs `open_trajectory_view`. It is the same failing set with the dev screen file.
    - In the rev1 run a 13th node, `test_trace_responsive.py::test_long_inspector_is_focusable_scrolls_to_painted_bottom_and_full_pane`, failed once at load 47 (`'▼ more — scroll' in ''` at :225) and passed 4 of 4 serially. Repeated 32 times at -n 4: 1 and 0 failures with the dev screen file, 2 and 0 with the final one. It is a pre-existing flake in the inspector overflow cue, which code this task does not touch; I proposed it as a follow-up.
  - Architecture size, timer-path and modal ratchets, run serially: 267 passed.
  - `ruff --select F` on the touched files, `Tests/lint/ruff_gate.py` and preflight: all pass.
- **Docs.**
  - `Docs/User_Guide/console/chat-basics.md`: a paragraph on live Trace. It opens on the newest events, or on the row you picked while a long Trace was loading. It follows while you are at the bottom. When you scroll up it pauses and keeps your rows, including the oldest rows of a long Trace, and a scroll at the very moment an update is drawn can be undone. `f` resumes.
  - `backlog/docs/lessons-textual.md`: `scroll_to` is immediate on a DataTable while `scroll_end` is deferred.
