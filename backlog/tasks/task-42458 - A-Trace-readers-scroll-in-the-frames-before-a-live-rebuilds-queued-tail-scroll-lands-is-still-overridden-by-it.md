---
id: TASK-42458
title: "A Trace reader's scroll in the frames before a live rebuild's queued tail scroll lands is still overridden by it"
status: Done
assignee:
- '@claude'
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 21:31
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
