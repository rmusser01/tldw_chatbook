---
id: TASK-32267
title: >-
  Library Notes: Escape from the Info pane needs two presses
status: Done
assignee: []
created_date: '2026-09-10 18:05'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - keyboard
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Escape from the Info pane does not return on the first press; a second press does. Observed on the first-timer edit journey, at a point where the rest of the flow is a model (the footer names the focused button, the receipt names the note, Undo restores it, the empty state is written).

Peer task-32233 covers the other half of the same complaint -- Escape inert in the filter box and on the plain list although the footer promises `esc focus rail` -- and does not cover the Info pane's press count, which is this task.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One Escape press from the Info pane returns to the previous surface
- [x] #2 Covered by a test
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live from every state the report's journey can reach Info in, wide and compact.
2. If it reproduces, trace the ladder; if not, record the states tried.
3. Ship the regression pin at both sizes.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
**NOT REPRODUCED at dev 4a14b3f36f.** No code change; the task ships the AC#2 pin, parametrised over both sizes the critique ran at.

One Escape returned from Info in every state tried, live on a seeded profile:

1. Info opened by clicking the Info button, focus on that button (`wave3-caps/editor-keys/14-info-escape1.txt`).
2. Info opened, then three Tabs into it so an Info button (`Export Markdown`) held focus (`26-info-tabbed-esc1.txt`).
3. Info opened, then a click into the Keywords `Input` so a text field held focus (`28-kw-esc1.txt`).
4. The report's own first-timer journey: `n` -> blank note -> Info -> Escape (`31-blank-info-esc1.txt`).
5. Compact, 100x30 (`38-compact-info-esc1.txt`).

The ladder step is a single branch in `action_library_notes_escape` (`if self._library_note_context: ... return`) with nothing between it and the key. Nothing in the range e6cb464239..4a14b3f36f touched it — `#2590`/task-32233 changed the LIST's Escape tail, not Info's — so the likeliest explanation is a repaint the reviewer read as a missed press. Recorded rather than guessed at.

Files: `Tests/UI/test_library_notes_wave_editor_keys.py`.
<!-- SECTION:NOTES:END -->
