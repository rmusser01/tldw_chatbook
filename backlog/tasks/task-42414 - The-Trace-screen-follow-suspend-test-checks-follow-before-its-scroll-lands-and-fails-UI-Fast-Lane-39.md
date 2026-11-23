---
id: TASK-42414
title: "The Trace screen follow-suspend test checks follow before its scroll lands and fails UI Fast Lane 39"
status: Done
assignee:
- '@claude'
created_date: 2026-11-22 21:17
updated_date: 2026-11-23 01:42
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

## Implementation Notes

**Verdict: both.** The CI red is test timing (a). A separate product race (b) also exists, and the old test could not see it. Both are fixed.

- **(a) Test timing.** CI run 44973100546, job 137058700280 (UI Fast Lane (39)): `test_trajectory_live.py:263 assert True is False`, call time 0.24 s. No live rebuild can have run by line 263, because the revision has not changed since mount. The only queued `scroll_end` was the test's own, and it lands in FIFO order before the reader's `scroll_to(y=0)`. So one `pilot.pause()` returned before the reader's deferred scroll landed, or before the table had laid out its rows (`max_scroll_y` 0, so the sync returns early). The log cannot tell which of the two happened. Two temporary edits (both reverted) each turn the old test red at :263, 3 of 3 runs: one delays the reader's `y=0` scroll by 0.3 s in the table's `_scroll_to`; the other delays the ledger's first render by 0.4 s. The new test passes 3 of 3 under each. The old test is still red at :263 under the scroll delay with the product fix in place (2 of 2), so the test is a cause on its own.
- **(b) Product race.** A probe at base held a live rebuild that a tick had started at the tail, scrolled the reader to y=0, then released it. The rebuild snapped the reader back to the tail (`scroll_y` 160 of 160, follow still on). `_apply_live_snapshot` used the follow flag sampled by the tick before the worker ran, and then queued `scroll_end`. The old test's "new records" step waited on `row_count > 123`, which the 180-row ledger already met. It therefore checked the position before the rebuild landed.
- **Product fix (`UI/Screens/trajectory_screen.py`).**
  - `_apply_live_snapshot` re-reads follow from the scroll position before its re-render resets the scroll.
  - Tail scrolls now go through `_queue_tail_scroll`, used by both live follow and `f`. It counts each scroll until `scroll_end`'s `on_complete` reports that it landed.
  - `_sync_follow_from_scroll` skips while a tail scroll is pending, so the top-of-table position a re-render leaves behind is never read as the reader scrolling up.
  - This count replaces `f`'s 1 s wall-clock grace and the `time` import. A renewed grace would not work: during streaming, each rebuild would renew it, and follow could never be suspended.
- **Tests (`Tests/UI/test_trajectory_live.py`, already a whole-file census entry; no census edit).**
  - `test_scrolling_up_suspends_follow_until_f_resumes` now waits on the table's geometry with a deadline, using the existing `wait_for_condition` fixture at 10 s / 10 ms: overflow, tail, the reader's scroll, the rebuild's 183 rows with no tail scroll in flight, and `f`'s tail. Every follow and position assertion is kept.
  - New `test_a_reader_scroll_during_a_live_rebuild_is_not_snapped_back`: red 3 of 3 with the fix's re-read removed (`assert 160 == 0`), green with it.
  - New `test_back_to_back_live_rebuilds_keep_following_the_tail`: red 3 of 3 with the pending-scroll skip removed (`_follow` False at :367), green with it.
- **Siblings.** I scanned the 84 failed Derived Artifacts runs from 2026-11-20 03:35Z to 2026-11-22 18:09Z (156 failed UI jobs in 71 runs). This node failed in 29 of those jobs, and no other Trace or trajectory node failed. `test_trace_responsive.py`'s inspector `end` checks have the same one-pause shape, but they assert `max_scroll_y > 0` first and never failed, so I left them unedited. The file's other 12 tests are unedited.
- **Soak.** The whole file at -n 4, 20 runs: 20/20, 15 passed each, load 30-44 on 18 CPUs. Serial with `-p no:xdist --timeout=180`, 5 runs: 5/5, 6.3-8.7 s per run, load 33-40. The file's serial time under load ~33 is about 7.5 s, two tests more than before.
- **Checks.**
  - Covering Trace/trajectory/modal files: 404 passed, 3 failed. The same 3 fail identically with the base screen file. They are in no PR lane: `test_trace_filters_navigation::test_live_insertion_before_selection_preserves_event_id_filters_and_viewport`, and `test_trajectory_timeline_integration`'s `test_click_outside_active_brush_atomically_clears_time_and_selects` and `test_timeline_renders_bars_for_timed_snapshot`.
  - Architecture size, timer-path and modal ratchets (serial): 267 passed.
  - `ruff --select F` on the touched files, `Tests/lint/ruff_gate.py` and preflight: all pass.
- **Docs.** `Docs/User_Guide/console/chat-basics.md` gains a paragraph on live Trace follow, pause and `f`.
