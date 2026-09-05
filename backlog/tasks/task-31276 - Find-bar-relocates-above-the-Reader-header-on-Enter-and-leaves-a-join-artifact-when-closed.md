---
id: TASK-31276
title: >-
  Find bar relocates above the Reader header on Enter and leaves a join artifact
  when closed
status: Done
assignee: []
created_date: '2026-09-04 13:54'
updated_date: '2026-09-04 20:48'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 P2: pressing Enter in the Find bar moved the whole bar from under the `Read` label to the top of the pane above `Local Media item`, pushing the header down six rows (B cap_20). After Escape closes Find, a five-cell `┐─────Local Media item` artifact appears at the pane join and persists across later interactions (14 captures; absent on a fresh open).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Find bar stays in one place (under the mode row) through open, typing, Enter and match navigation
- [x] #2 No `┐─────` artifact at the pane join after Find close, tab clicks or the More menu
- [x] #3 Live-verified at 235x52 and 100x30 with captures in the notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: bar y unchanged through open/submit/Next; identity-row paint has no ┐───── after Find close, tab click, More
2. GREEN: remove task-15774's active-search dock (keep ACTIVE_SEARCH_CLASS for status/Prev/Next styling); root-cause the artifact
3. Live 235x52 + 100x30
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The dock is gone: the bar stays under the mode row through open, submit and match navigation (test_library_shell's dock assertion rewritten to the new contract). The join artifact was neither brief suspect — it is the focused pane grip's outline-top/bottom accent end-caps (the grip is 5 cells wide × full shell height, so its top row lands on the Reader's identity line) and it reproduced in the harness, surviving a forced refresh. The end-caps are retired; the grip's focus cue is now bold + colour on the arrow (bold is the non-colour carrier), shared by every adaptive Library shell. Removing the dock exposed a pre-existing height:1fr balloon on #library-media-content-mode-strip that pushed the search chrome below the fold at 80x24 — fixed with height:auto. Why the end-caps kept appearing: focus falls to the grip after any Reader recompose (wave rider, own task). Deferred: the grip-polish QA README/geometry.json still describe end-caps; the CSS comment overstates 'the viewer stopped scrolling'; test-strength polish.
<!-- SECTION:NOTES:END -->
