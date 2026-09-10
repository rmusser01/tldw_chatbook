---
id: TASK-32249
title: >-
  Library Notes Markdown preview is capped at 20 rows at 235x52 while the
  compact layout gets 1fr, and its paging keys are inert on arrival
status: To Do
assignee: []
created_date: '2026-09-10 18:05'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - css
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cause PROVEN. `css/components/_agentic_terminal.tcss` carries `#library-note-preview-region { height: auto; min-height: 12; max-height: 20; }`, and the **very next rule** gives `#library-shell-grid.library-notes-compact #library-note-preview-region { height: 1fr }` -- so a 100-column terminal reads a long note better than a 235-column one. The lines are byte-identical at `c4a7b1911f`: pre-existing, newly found. No test pins the cap.

Confirmed live: the preview box closes at screen row 33 with 14 blank rows beneath it, the footer promises `pgup/pgdn scroll`, and PageDown from the landed state changes nothing (`R/caps/03`, `04`); one click inside the region and the same key reveals the code block, the table, the callout and the wikilink (`R/caps/05`). Preview is the reading surface for the researcher persona, and a reader's reasonable conclusion from a 20-row window with blank space under it is that content is missing.

Two riders on the same surface, both P3: the preview leaks the Obsidian callout marker `> [!note]` instead of rendering the callout, and the status line still reads "Next: Keep editing; changes save automatically." while Preview is showing.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The preview uses the available height in the wide layout (`1fr`, as the compact layout already does); no fixed 20-row cap at 235x52
- [ ] #2 Activating Preview focuses the region, so the footer's `pgup/pgdn scroll` promise is true on arrival without a click
- [ ] #3 An Obsidian callout renders as a callout rather than leaking its `[!note]` marker into the text
- [ ] #4 The status line does not advertise editing behaviour while Preview is showing
- [ ] #5 Covered by a test pinning the preview height rule in the wide layout
<!-- AC:END -->
