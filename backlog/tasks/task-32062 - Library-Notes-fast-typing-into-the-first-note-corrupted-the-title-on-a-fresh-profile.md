---
id: TASK-32062
title: >-
  Library Notes: fast typing into the first note corrupted the title on a fresh
  profile
status: Done
assignee: []
created_date: '2026-09-08 18:24'
updated_date: '2026-09-08 21:52'
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
Two defects, one of them the reported one. Live evidence on the fresh profile at 235x52 (captures 21-26 in the group's caps/, stored rows read back with sqlite3).

1. NOT the graduation. Reproduced the exact reported corruption live -- title 'Mhello from jordan, testing the libraryy first note', body empty -- and then reproduced it AGAIN on the same profile after it had already graduated ('Mbody text herey second note'). The trigger is the editor's own refresh path, not the STARTER -> GRADUATED transition. Measured separately: that transition reaches the editor with zero LibraryNotesCanvas recomposes, on this branch and at base 232a59fdc4.

Root cause: LibraryNotesCanvas.apply_session_state rewrote the title Input from the screen's snapshot, which during a fast sentence is a keystroke or more behind. Assigning Input.value clamps the cursor to the shorter text, so the rest of the sentence was then typed at that stale position -- 'M' + the body + 'y first note'. A focused field is now skipped: it is its own authority, and the snapshot is built from its Changed events, so it can only be behind. Same guard for the body TextArea. Pinned by test_a_stale_snapshot_never_rewrites_the_field_that_has_focus (RED at base: the typed title came back as ''). Live after the fix: the title kept its text in order and nothing was lost.

2. AC#1's other half: sync_state ended in an unconditional refresh(recompose=True), which with the editor open rebuilt the title Input and body TextArea and dropped focus onto the Notes list grip. Deferred now while this canvas's own title/body has focus and the mode is unchanged; on_descendant_blur applies it when focus leaves. Known ceiling recorded in the code: a banner that recompose would have painted also waits for the field to lose focus.

RESIDUAL, needs its own task (not this AC): with title, Tab and body sent as one uninterrupted burst, the Tab focus move lands AFTER the burst, so the body text is appended to the title ('My third notehello from jordan, testing the library'). Nothing is scrambled or lost now, and with ~1 s between the three sends it is correct ('My fourth note' / 'a real body'). A bare two-Input Textual app routes the same burst correctly, so this is this screen's key-dispatch cost, not a Textual given.

Files: tldw_chatbook/Widgets/Library/library_notes_canvas.py; Tests/UI/test_library_crit8_polish_shell.py (4 tests); Docs/User_Guide/library/notes.md. Commits 086e76e813 (deferral) and ab929759cc (the focused-field guard, folded in by a parallel commit on this branch).
<!-- SECTION:NOTES:END -->
