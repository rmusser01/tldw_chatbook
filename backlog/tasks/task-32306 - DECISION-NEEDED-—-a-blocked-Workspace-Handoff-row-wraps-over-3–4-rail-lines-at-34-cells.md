---
id: TASK-32306
title: >-
  DECISION NEEDED — a blocked Workspace Handoff row wraps over 3–4 rail lines at
  34 cells
status: Done
assignee: []
created_date: '2026-09-11 00:54'
updated_date: '2026-09-11 17:29'
labels:
  - library
  - ux
  - critique-9
  - decision-needed
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task-32230 (critique-9 rail branch, PR #2581) made the Details ▸ Handoff row actionable: it now carries the reason and the next step instead of '● 1 blocked'. At the rail's 34-cell width that prose wraps over three to four lines. Shortening it drops either the reason or the next step; moving the remedy into a tooltip re-introduces hover-only meaning, which the wave's copy rule forbids. This is a product call the implementer and reviewer both declined to make.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A decision is recorded (keep the wrap, shorten the copy, or move the remedy) and applied
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the Handoff row's build path and the rail's Details row painter.
2. Record the user's decision verbatim in the task file.
3. Implement the hanging indent once, in the shared row painter, checking the wrapping siblings (counts line, Diagnostics, Collections block).
4. Pin the painted lines at 34 cells in a Textual harness at 235x52 and 60x24; mutation-test the guard.
5. Live-verify at 235x52; guide stamp; Implementation Notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The indent is applied once, in a shared row painter: every
``.library-details-row`` is now a ``LibraryDetailsRow`` (``library_rail.py``),
which re-wraps its renderable through ``library_hang_details_row`` on resize
and on ``update()``. The eight construction sites -- the Status source/counts
rows, the DB-size rows and their in-place patcher, the fold cue, the Workspace
Active/Handoff rows, the Chunking Lab gloss -- all get it, because the
sibling check found the same wrap on the Status counts line ("Notes 7 · Media
11 · Conversations 6" breaks after "Media 11 ·" at 34 cells) and it can reach
any row carrying a sentence. A row that fits its width is handed back
UNCHANGED, so nothing moves for the rows that never wrapped.

**The first implementation was wrong in a way worth recording.** A custom Rich
renderable (``__rich_console__`` doing the wrap at render time) produced
correct lines in isolation but measured as ONE line through Textual 8's visual
protocol, so the row painted its hung lines and then clipped them -- live at
235x52 the Handoff row lost everything after its first line, and a
``.renderable`` probe would have called that a pass. Handing Textual a plain
``Text`` with real newlines measures correctly; that is why the wrap happens
in a helper and the widget re-runs it on resize.

No TCSS change was needed in the end (a speculative ``width: 1fr`` on
``.library-details-row`` was measured to change nothing and was reverted), so
the CSS bundle is untouched.

**Evidence.** ``test_library_details_hanging_indent.py`` pins the painted
strips through the product path at 235x52 (2 lines, continuation indented by
2) and at 60x24 (the rail takes the full width there, so the same row fits one
line -- the other half of the rule), plus the exact blocked-Handoff sentence at
34 cells as a function, because that app state is not reachable from the
mounted harness on this base (``test_post_release_workspaces_library_depth``
is red for that reason before this branch). Neutering the helper turns both
hang assertions red. Live at 235x52 on a seeded profile, the blocked row reads:

```
 Handoff · 24 items can't be
   used in Console yet · not in
   this workspace · Copy or link
   them into this workspace
```

**Files:** ``tldw_chatbook/Widgets/Library/library_rail.py``,
``tldw_chatbook/UI/Screens/library_screen.py``,
``Tests/UI/test_library_details_hanging_indent.py`` (new),
``Docs/User_Guide/library.md``.
<!-- SECTION:NOTES:END -->

## Decision

<!-- SECTION:DECISION:BEGIN -->
User ruling, 2026-09-11, verbatim:

> keep the wrap, and indent the continuation lines

> tab over subsequent lines so it's clear what they correspond to

The copy is not shortened and nothing moves into a tooltip. The first line
starts at the row's glyph column; every continuation line is indented under
the text.
<!-- SECTION:DECISION:END -->
