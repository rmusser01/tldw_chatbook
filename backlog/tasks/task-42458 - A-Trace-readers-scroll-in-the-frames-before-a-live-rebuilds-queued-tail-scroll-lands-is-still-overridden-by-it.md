---
id: TASK-42458
title: "A Trace reader's scroll in the frames before a live rebuild's queued tail scroll lands is still overridden by it"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 06:38
updated_date: 2026-11-24 01:23
labels:
- follow-up
- trace
- ux
priority: low
references:
- tldw_chatbook/UI/Screens/trajectory_screen.py
- Tests/UI/test_trajectory_live.py
- Docs/User_Guide/console/chat-basics.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23). Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths are under `tldw_chatbook/` unless they start with `Tests/` or `Docs/`.

**Why.** TASK-42414 made a live Trace keep a reader who scrolled up where they are across live rebuilds. It recorded one gap it did not close (its notes, "Known gap (not fixed)"). The screen queues its own tail scrolls (after a live rebuild, after `f`, and on open) and skips its scroll-position read while one is in flight, so a re-render's reset is not read as a reader scroll. A reader scroll that arrives in that window, one or two refresh hops long, is overridden when the queued scroll lands. The same window also delays the follow check for that scroll until the next poll tick. The screen cannot tell the reader's scroll from its own there, because it reads only the table's scroll position. Closing the gap needs reader-input events from the DataTable (wheel, keys, scrollbar drag) that cancel the pending scroll.

`Docs/User_Guide/console/chat-basics.md` (line 1029) now tells users that a scroll made at the very moment an update is drawn can be undone by it. No user report or CI failure names this; it matters if the window is noticeable in use.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A reader's wheel, key or scrollbar scroll toward older rows made while a live rebuild's, `f`'s or the open's tail scroll is still queued cancels that pending scroll, and the reader stays where they scrolled (amended 2026-11-23: a scroll toward the tail keeps the queued tail scroll, which moves the same way; cancelling it would leave a follower above the newest rows and pause follow)
- [x] #2 Such a scroll suspends follow at once, not at the next poll tick
- [x] #3 A test holds a queued tail scroll (with a held refresh or a temporary delay) and shows the reader's scroll is kept; it fails on the current screen code
- [x] #4 Follow and `f` behave as before when the reader does not scroll: the existing live Trace tests pass unedited
- [x] #5 The User Guide no longer says such a scroll can be undone
<!-- AC:END -->

## Implementation Plan

1. Find the DataTable's reader-input paths (wheel, keys, scrollbar drag) and how the screen's queued tail scrolls (live rebuild, `f`, open) are counted.
2. Let a reader scroll cancel any queued tail scroll and suspend follow at once; the cancelled scroll must not land.
3. Test: hold a queued tail scroll, scroll as the reader (wheel, key, scrollbar), release, and show the reader's position and paused follow are kept; red on the current screen code.
4. Run the live Trace tests unedited; remove the User Guide sentence about a scroll being undone.

## Implementation Notes

**Cause.** While the screen's own scroll is queued (after a live rebuild, `f` or the open), the table's position says nothing about the reader. A scroll up in that window was overridden when the queued tail scroll landed, and follow stayed on until the next poll tick. A live re-render also reset the view to the top, so a wheel in that window moved from the top rather than from where the reader was.

**Fix (`UI/Screens/trajectory_screen.py`).**
- The ledger is `_TraceTable`, a `DataTable` subclass. Its wheel-up, key (up, page up, ctrl+home) and scrollbar (click above the thumb, drag up) paths tell the screen first, before the table's own handler scrolls.
- In that window (`_scrolls_pending`), `_reader_scrolled_up` pauses follow at once and bumps `_scroll_epoch`. It records the reader's row after the table has scrolled (`call_next`).
- `_queue_ledger_scroll` checks the epoch when the scroll lands. A tail scroll from an older epoch lands on the reader's row instead. The two refresh hops are kept, with the epoch checked at the second.
- A live re-render puts the view back where the reader had it until its queued scroll lands.
- AC #1 now says "toward older rows". A scroll toward the tail keeps the queued tail scroll: it moves the same way, and cancelling it would leave a follower above the newest rows.

**Test.** `test_a_reader_scroll_while_a_tail_scroll_is_queued_keeps_the_reader[live-rebuild-wheel|live-rebuild-key|live-rebuild-scrollbar|f-wheel|open-wheel]` in `Tests/UI/test_trajectory_live.py` (a whole-file census entry). It holds the table's `call_after_refresh` (class-level for the open case), lets the reader scroll, then releases. It checks three things: the scroll moved from the reader's view, follow is off at once, and the reader's row survives the queued scroll.

**Red/green.** All 5 cases are red at dev's screen file: two at `0 < 0` (the wheel moved from the reset top) and three at "follow waited for the next poll tick". All are green after. Controls on the final code:
- Without the epoch check, the 5 fail on the reader's row (the tail landed).
- Without the view hold, the two wheel and scrollbar live-rebuild cases fail at `0 < 0`. The key case moves the cursor, so it does not depend on the hold.
- Without the immediate pause, the 5 fail on follow.

**Existing tests.** The live Trace tests pass unedited (all 23 earlier tests in the file).

**Docs.** In `Docs/User_Guide/console/chat-basics.md`, the "can be undone by that update" sentence is gone. The guide now says a scroll up pauses at once, even while an update is drawn.

**Soak.** 32 of 32 runs green (157 tests each, load 49-89; one combined serial run of the four Trace census files with the lane's flags: 124 passed in 222.9 s at load ~74-85) (see TASK-42457).

**AC #1 ratified.** The coordinator ratified the "toward older rows" amendment on 2026-11-23.

**Review fixes (commit 787d0059cc).** The names above changed: `_reader_scrolled_up` became `_reader_scrolled(toward_older)`, and the test became `test_a_reader_scroll_while_a_ledger_scroll_is_queued_keeps_the_reader`.
- **Animated scrollbar scrolls were still undone.** A click above the thumb runs an animated page up. A drag may animate too. The reader's row was read from `scroll_y` before the animation moved, so the view came back. `_top_row`, the view held across a re-render, and the follow check now read `scroll_target_y`. `on_scroll_to` compares against it as well.
- **A paused reader's scroll toward newer rows was undone.** The ledger now reports both directions: wheel, `cursor_down`, `page_down`, `scroll_bottom`, `ScrollDown`, and a downward `ScrollTo`. While paused, any reader scroll in the window bumps the epoch and records the reader's row again. While following, a scroll toward the tail still changes nothing.
- **A ledger that cannot scroll paused for good.** `max_scroll_y <= 0` now returns early: there is nothing to scroll away from, and the follow check never resumes such a ledger.
- **"Pauses at once" was only true in the window.** With nothing queued, a reader scroll now runs the follow check right after the table applies it (`call_next`). Because the check reads `scroll_target_y`, an animated scroll pauses at once too.
- **Tests (`Tests/UI/test_trajectory_live.py`).** Six cases were added: two for the animated scrollbar (`ScrollTo(animate=True)`, `ScrollUp()`), and four for a paused rebuild in the downward direction (wheel, page down, `ScrollTo`, `ScrollDown`). The paused cases put the cursor in view first, so page down moves the view a page instead of moving the cursor from the newest row to the tail. Also added: `test_a_reader_scroll_up_pauses_follow_at_once[wheel|scrollbar-click]` and `test_a_scroll_on_a_ledger_that_cannot_scroll_keeps_following`.
- **Red/green.** All the new cases were red at 057ed23ecf: the reader's row was overridden in the animated and paused cases, follow waited in the at-once cases, and follow stayed `False` in the cannot-scroll case. All are green after. Controls, each run once on the final code:
  - `_top_row` reading `scroll_y` fails the 3 animated cases.
  - No downward hooks fails the 4 paused-down cases.
  - No early return fails the cannot-scroll test.
  - A follow check that reads `scroll_y` fails `at_once[scrollbar-click]`.
- **Docs.** `chat-basics.md` already said a scroll pauses at once and the reader's rows stay. Both statements are now true in every case above.
- **Soak after the review fixes.** 20 of 20 runs of the five Trace files at `-n 4` passed (168 tests each, load 40-80). `test_trajectory_live.py`, a census entry, now has 41 tests: 37.8 s serial with the lane's flags at load 42. Tests/CI at `-n 4`: 1823 passed, 2 skipped.

**Re-review fix (commit b56b6f1148).**
- **Navigation keys and timeline clicks were not reader scrolls.** `_navigate_records` (n/p, j/k and the other anomaly keys) and `_on_bar_selected` now call `_reader_moved_cursor(from, to)` after the cursor move, in the cursor's real direction (row order, after any re-render). It is not called inside `_move_cursor_to_key`, which `f` also uses.
- **Test:** `test_a_navigation_move_while_a_tail_scroll_is_queued_keeps_the_cursor_in_view[previous-match|timeline-click|timeline-click-pages-in]`. All three cases are red at 5cc37995e3 ("the move did not pause follow"). Removing either hook turns its cases red.
- **Considered and left out:** recording the cursor's row as the reader's anchor directly, instead of the top row after the move. It was not needed in any case I could build, including the paging click, so it is not in the code.
- **Not tested:** a navigation move that re-renders without paging (a cleared time brush, or a revealed collapsed turn) while a ledger scroll is queued. A probe of the brush case failed on the test data's missing timing, not on the screen. This is a proposed follow-up.
- **Soak after the re-review fixes.** 20 of 20 runs of the five Trace files at `-n 4` passed (173 tests each, load 45-86). `test_trajectory_live.py` now has 46 tests: 37.3 s serial with the lane's flags at load 45. The timer-path and guarded-modal Architecture tests passed (26). Tests/CI at `-n 4`: 1822 passed. Its one failure, `test_bootstrap_root_cleanup.py::test_owned_root_is_removed[fork-child-terminated]` (a child process did not end in time at load ~50-60), passed 3 of 3 alone. The branch does not touch Tests/CI.

**Third re-review fix.** `_reader_moved_cursor` took the direction from the cursor's move. A follower's cursor sits above the view as rows append, so "next" (`n`, `j`, `u`, `v`, `a`, or a timeline click on a later record) scrolled the view up but counted as "newer", and a queued tail scroll still landed.
- **Fix:** navigation and timeline moves now always count as reader moves toward older rows. With a ledger scroll queued, follow pauses, and the next tick resumes it if the view is still at the bottom. Otherwise the follow check reads the view at once. Unlike a choice by the view's bottom, this does not depend on where the table's scroll stands during the window, when its position is unreliable.
- **Test:** a new case, `test_a_navigation_move_while_a_tail_scroll_is_queued_keeps_the_cursor_in_view[next-match-cursor-above-view]`. It runs ten live updates, checks that the cursor is above the view, then presses `n` while a tail scroll is queued. At 9ec7e7d1cb it fails with "the move did not pause follow". With that assertion skipped, it fails with "cursor row 181 is off screen (190..213)", which matches probe P12. Green after.
- **Regression check:** the other cases and all 267 Trace UI tests pass, including the `f` tests.
