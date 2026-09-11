---
id: TASK-32247
title: >-
  Library Notes editor: Ctrl+End does not reach the end of a long note, in any
  encoding
status: To Do
assignee: []
created_date: '2026-09-10 18:05'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - keyboard
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repro on the seeded profile with a 35 KB note: click into the body -- which does place the caret, since a marker typed after a click landed exactly where clicked -- then press `Ctrl+End` and type. The text landed at character **23** of 35,187 (`R/caps/15`). Retried with the raw xterm sequence `\x1b[1;5F`: character 28. D measured character 125 by the same route. Plain `End` is delivered, so this is not a general key-delivery failure.

The consequence is that the power user's stated task -- edit near the end of a long note -- cannot be done by keyboard at all, and no on-screen affordance offers an alternative.

Cause INFERRED: Textual's `TextArea` binds `ctrl+end` to `cursor_document_end`, so something upstream is swallowing it; not traced.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A document-end key moves the caret to the end of the note body from any starting position, verified on a note of about 35 KB
- [ ] #2 The key is advertised in the editor footer beside the other editor keys
- [ ] #3 Covered by a test that presses it on a multi-thousand-line body and asserts the resulting caret location
<!-- AC:END -->
