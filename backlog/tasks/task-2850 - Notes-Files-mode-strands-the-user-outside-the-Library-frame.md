---
id: TASK-2850
title: Notes Files mode strands the user outside the Library frame
status: To Do
assignee: []
created_date: '2026-08-07 01:10'
labels:
  - library
  - notes
  - ux
  - uat-2026-08-06
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library UAT 2026-08-06 (LIB-01, dual-agent critique `.impeccable/critique/2026-08-07T01-01-42Z__tldw-chatbook-ui-screens-library-screen-py.md`, observed at dev `6ffa56516`).

Notes canvas → "Database | Files" strip → clicking "Files" replaces the ENTIRE Library screen —
rail, search, groups, canvas frame all vanish. What remains is "Choose a notes folder." top-left
and a "Choose folder…" button ~150 columns away top-right, over ~40 blank rows. Escape does
nothing; the only exits are the small "Database" text link or the folder picker. Reproduced 100%.

A first-time user reads this as the app breaking. It is total context loss on a surface whose
sibling states all keep the rail + canvas frame, and it violates the product principle that
recovery paths stay visible.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Entering Notes ▸ Files mode keeps the Library rail and canvas frame visible
- [ ] #2 The folder chooser renders as a normal canvas empty state: prompt text and its action button adjacent, not separated by blank columns
- [ ] #3 Escape (or an equally advertised key/control) returns from Files mode to the Notes Database view
- [ ] #4 Live TUI verification confirms the above at 170×50 and 100×30
<!-- AC:END -->
