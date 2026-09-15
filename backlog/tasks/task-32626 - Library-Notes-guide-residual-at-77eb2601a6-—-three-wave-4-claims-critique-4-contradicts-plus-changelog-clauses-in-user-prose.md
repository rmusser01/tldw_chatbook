---
id: TASK-32626
title: >-
  Library Notes: guide residual at 77eb2601a6 — three wave-4 claims critique 4
  contradicts, plus changelog clauses in user prose
status: To Do
assignee: []
created_date: '2026-09-15 06:45'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A section 10 and assessor B section 8, every persona. Wave 4's own sweep (task-32558, PR #2687) re-verified about 300 claims, so each item below is either NEW -- a claim the sweep itself added for a wave-4 fix -- or MISSED -- a claim checked only on its happy path.

NEW (added by wave 4, contradicted live):
- notes.md promises Receipts shows 'Wrote note to file' for a note you edited in Chatbook. No user-reachable path produces that receipt (the P0's docs half).
- 'The New note view no longer shares the canvas with an empty notes list … takes the stage': at 235 columns it shares with navigation (37) and the list (62), taking about 105 (A cap 03).
- 'While no note is open the list takes the width the empty work area would otherwise waste': the list widens to about 132 columns but a 54-column work pane remains, holding one sentence (A cap 08).
- The chooser 'bar below holds only ‹ Notes': nothing is rendered there (A cap 17).

MISSED (true on the swept path, false beside it):
- 'slash focuses the filter and selects whatever is already in it' -- true only from the navigator (A caps 41-43; B VERIFIED from the navigator).
- 'The footer names whichever editor control has focus' -- false in Info (A caps 11-13).
- 'the picker opens with that File name field already focused' -- false for the Folder-files door (A caps 27, 28).
- 'Saved appears once per view' -- true of Info, but printed twice in the editor header (A cap 06).
- 'Once a note has saved in this session the state names the time' -- on reopening, a bare 'Saved' (A cap 09).
- A warning callout 'becomes Warning' -- type and body run together (A cap 25).

DOC DEFECT: about 30 inline '(Was … — superseded by task-3xxxx)' clauses sit inside user-facing prose across notes.md. That is a commit log in a user guide, and the wave added more of them.

B's own conformance pass VERIFIED 17 further claims (Ctrl+N, Escape, footer strings, Import-once copy, Folder-files keys, Session Git end to end), so the page is mostly true -- these are the exceptions. Adjacent open rider 32589 proposes making the guide-claim string check a gated test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each contradicted claim is corrected or removed, and the verification stamp names this critique
- [ ] #2 Claims a fix introduces state the width or region they hold for, rather than generalising from the one case walked
- [ ] #3 The changelog clauses move out of user-facing prose
<!-- AC:END -->
