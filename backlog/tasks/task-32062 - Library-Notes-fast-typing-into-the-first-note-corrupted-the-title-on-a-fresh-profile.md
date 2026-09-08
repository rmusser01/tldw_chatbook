---
id: TASK-32062
title: >-
  Library Notes: fast typing into the first note corrupted the title on a fresh
  profile
status: Done
assignee: []
created_date: '2026-09-08 18:24'
updated_date: '2026-09-08 21:26'
labels:
  - library
  - notes
  - bug
  - critique-8
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Typing 'My first note', Tab, and a body within ~0.4 s produced the stored title 'Mhello from jordan, testing the libraryy first note' with an empty body; not reproducible on a populated profile. The likely cause is the graduation recompose ('Library tools are now available.' plus pane collapse) firing mid-typing and resetting the focused Input. Data-affecting. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 13.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Typing into a focused editor is never interrupted by a recompose or focus reset
- [x] #2 A test types rapidly during the first-content graduation and asserts title and body land in the right fields
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce on a FRESH (STARTER) profile: gated evidence rounds EMPTY then HAS_USER_CONTENT, open the new-note editor, type into the title Input while the first-content graduation runs.
2. Trace which recompose/focus reset reaches the editor (screen recompose, entry reconcile, or _sync_library_rail_lifecycle_presentation).
3. Defer any notes-canvas recompose while the editor (title Input or body TextArea) holds focus, or prove none reaches it and pin that.
4. Pin: type rapidly across the graduation, assert title and body land in their own fields.
5. Live-verify on the fresh profile and read the stored note back from the DB.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause was not the graduation toast. Every Notes refresh routes through LibraryNotesCanvas.sync_state, which ended in an unconditional refresh(recompose=True): with the editor open that rebuilt the title Input and body TextArea and dropped focus onto the Notes list grip, so keystrokes still in flight landed in the wrong box (the reported 'Mhello from jordan, testing the libraryy first note'). Measured: with the title focused and 'My first note' typed, one sync_state call replaces the Input instance and leaves focus on #library-notes-items-grip.

Fix: sync_state defers the recompose when the surface is not changing mode AND this canvas's own #library-note-title / #library-note-body has focus. Every compose input is stored before the deferral, so the rebuild is postponed, not dropped -- on_descendant_blur runs it once focus leaves the editor. A mode change still paints immediately (that is navigation the reader asked for). Known ceiling, recorded in the code: a banner that recompose would have painted (a 'changed elsewhere' conflict) also waits for the field to lose focus.

Also measured and recorded: the STARTER -> GRADUATED transition itself reaches the editor with ZERO LibraryNotesCanvas recomposes -- the title Input keeps its identity and focus through it, on this branch and at base 232a59fdc4. AC#2's test types across that transition anyway (title, Tab, body) and pins that both fields land, and the sibling test pins the sync_state hazard that the live incident actually hit.

Files: tldw_chatbook/Widgets/Library/library_notes_canvas.py; Tests/UI/test_library_crit8_polish_shell.py (3 tests); Docs/User_Guide/library/notes.md.
<!-- SECTION:NOTES:END -->
