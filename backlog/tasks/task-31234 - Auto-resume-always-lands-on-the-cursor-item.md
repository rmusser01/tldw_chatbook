---
id: TASK-31234
title: Auto-resume always lands on the cursor item
status: Done
assignee: []
created_date: '2026-09-04 01:50'
updated_date: '2026-09-04 03:05'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 P1: re-entering Media with an active review set re-arms the banner + walk footer ("Reviewing: Read later — 1 of 2") over whatever document happens to be showing — including one not in the set — and the first ] is a silent sync that advances nothing. The status line disagrees with the visible document exactly when the feature's promise is continuity. Root cause verified: the once-per-set gate (_review_set_auto_resumed, library_screen.py:39576) returns without opening the cursor item on re-entry while the banner re-arms unconditionally. USER RULING (critique #3 close): always load the cursor item — re-entering Media with an active set lands in the Reader at the saved place, every time. This supersedes task-28245's once-per-set/show-the-list AC; document the supersession.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Entering the Media area with an active review set opens the set's cursor item in the Reader, on every entry (once-per-set gate removed)
- [x] #2 The banner, footer progress, and visible document always agree after entry
- [x] #3 The cold-start yank guard (abort when the user has already navigated away mid-resume) still holds
- [x] #4 task-28245's superseded AC is annotated in that task file or the walker test docstrings
<!-- AC:END -->

## Implementation Plan

1. RED: rewrite the once-per-set pinning test to the new every-entry contract
2. GREEN: remove the _review_set_auto_resumed gate from the worker
3. Annotate task-28245's superseded AC; live tmux verify

## Implementation Notes

Removed the once-per-set gate block from _auto_resume_review_set_worker (the set_id is no longer consumed) and rewrote its docstring; the cold-start yank guards (still-on-media-list + is_current + view=="list") are untouched and still pinned by test_auto_resume_aborts_when_the_user_moved_away. task-28245 got a "## Superseded (task-31234)" section documenting the reversal and the user ruling. Live-verified twice: enter Media with an active set → Reader at cursor with agreeing banner; leave to Notes, re-enter → same landing again (the old gate would have shown the bare list under a lying banner).
