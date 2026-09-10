---
id: TASK-32228
title: Library Conversations footer offers neither 'esc focus rail' nor a '/' hint
status: Done
assignee: []
created_date: '2026-09-10 14:56'
updated_date: '2026-09-10 21:31'
labels:
  - library
  - conversations
  - footer
  - critique-9
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Conversations canvas footer is nearly empty while every sibling list advertises Escape and `/`. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 27.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Conversations footer advertises the same list keys as its siblings, and they work
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure the Conversations footer against its siblings.
2. Give the branch the list keys it honours without weakening the pinned two-step 'focus Items'/'focus Library' Escape grammar.
3. Docs + live-verify.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fix round 1: AC#1 delivered per ruling R1.

Two halves. The first was live at 100x30 before that round: the canvas showed
its Filter box, "/" focused it, and the footer said only "F6 next pane",
because the "/" chip reads `reader_layout.items_open` and the footer is
registered from `compose_content`, before the shell resolves its panes. A
pane-visibility change now re-registers the footer (task-32225's seam).

The second is the reason the footer looked "nearly empty" in the first place,
and nobody had traced it: Conversations was the one browse list left out of
task-2856's entry focus, so arriving put focus outside the reader shell, the
Escape hop had nowhere to start from, and its chip was correctly withheld. The
keys were never missing -- the state that makes them live was. Registering
"library-conversation-row" in `_LIBRARY_LIST_ROW_CLASSES` and
`_LIBRARY_LIST_ROW_CLASS_BY_ROW_ID` (the row class already existed on the
buttons) plus the rail-row arm site fixes it, and gives the list the Up/Down
row traversal its siblings have.

Two seams had to be honest before that worked, both measured:
* the arm schedules ONE attempt and relies on `compose_content` re-requesting
  while armed -- which covers rows arriving on a SCREEN recompose but not on a
  canvas-level one. It now retries on a coarse poll, bounded on both axes by
  the arm's own settle window. Fix round 2 gave that poll a stored handle, so
  one chain runs per arm and a disarm stops it early (re-review N5).
* the empty-list fallback treated `set_focus` as success. A control that is
  mounted but not yet focusable leaves focus on None, which is exactly the
  state Conversations arrives in, so only a landing counts now.

Live evidence (235x52, seeded profile,
caps/32228-conversations-235-entry-focus.txt): the footer reads
"/ focus filter | F6 next pane | esc focus Library | F1 help · Ctrl+P palette ·
Ctrl+Q quit" -- three canvas keys, the same count as its siblings -- and Down
walks the focus bar from "Launch checklist" to "Transformer scaling laws".

The label is "focus Library", not the siblings' "focus rail". On this canvas
Escape steps back through the visible panes, the grammar it shares with the
Media Reader and that test_conversations_escape_moves_to_nearest_visible_prior_role
pins (re-run green). Parity is on the KEYS, which is what AC#1 asks for; the
alternative -- an unconditional "esc focus rail" -- would restore the dead-key
lie task-31272 removed.

Known ceiling, now a filed rider rather than only a comment: the entry-focus
arm is a fixed 2-second window, so a session's FIRST visit to Conversations
(cold DB) can still miss it and land nowhere. task-32260 measured Library at
12.6s to open, which makes the miss routine rather than rare; the other list
canvases share the ceiling (a cold Prompts visit lands in its filter rather
than row 0). Upgrade path: **task-32301**, re-arm from each destination's own
list-arrived seam instead of a fixed window.

Files: UI/Library_Modules/screen_constants.py, UI/Screens/library_screen.py,
Tests/UI/test_library_crit9_shell.py,
Docs/User_Guide/library/media-and-conversations.md.
<!-- SECTION:NOTES:END -->
