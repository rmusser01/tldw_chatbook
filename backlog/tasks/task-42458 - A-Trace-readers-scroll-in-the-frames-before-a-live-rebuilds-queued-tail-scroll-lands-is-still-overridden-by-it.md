---
id: TASK-42458
title: "A Trace reader's scroll in the frames before a live rebuild's queued tail scroll lands is still overridden by it"
status: To Do
assignee: []
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 07:09
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
- [ ] #1 A reader's wheel, key or scrollbar scroll made while a live rebuild's, `f`'s or the open's tail scroll is still queued cancels that pending scroll, and the reader stays where they scrolled
- [ ] #2 Such a scroll suspends follow at once, not at the next poll tick
- [ ] #3 A test holds a queued tail scroll (with a held refresh or a temporary delay) and shows the reader's scroll is kept; it fails on the current screen code
- [ ] #4 Follow and `f` behave as before when the reader does not scroll: the existing live Trace tests pass unedited
- [ ] #5 The User Guide no longer says such a scroll can be undone
<!-- AC:END -->
