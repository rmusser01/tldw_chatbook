---
id: TASK-32615
title: >-
  Library Notes: the Session Git commit form's actions sit 30 rows below its
  fields and the trust dialog breaks the path being consented to
status: To Do
assignee: []
created_date: '2026-09-15 06:41'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor B D9 and the layout half of D2, personas Alex and Sam, Obsidian workflow.

What happened. Session Git's consent design is one of the best things on the screen -- the commit review states What / Where / Impact / Recovery, names the identity, the branch and parent, says hooks will not run and the commit will be unsigned, and promises no unrelated staged content will be committed (B cap 44, verified end to end against git log). Its geometry undoes some of that.
- Commit form: Subject and Body at rows 17-27, the only actions (Cancel commit, Review commit) at row 48, twenty blank rows between, no more-below cue (B caps 42, 43). Same for the commit review (B cap 44).
- Repository-trust modal: rendered inside the roughly 45-column right pane, so the repository path wraps as '/Users/.../crit4/B/power/vaul' plus 't' on the next line, and the status truncates to 'Status: TRUST REQU…' (B cap 39). A security consent dialog must never mangle the identifier being consented to.
- And after a successful commit the message 'Committed 1 session note as c1db79e…; unrelated changes untouched.' renders under the Danger heading at 100x30 and 60x24, with 'Commit review ready.' doing the same at 235x52 (B caps 44, 47, 48, D12).

Cause INFERRED for all three (not traced).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The commit form's actions sit within the reader's view of its fields, or the pane states that more follows
- [ ] #2 The repository-trust dialog renders the full path it asks consent for, without wrapping it mid-token, and shows its status in full
- [ ] #3 A success message never renders under a Danger heading
- [ ] #4 The three sizes are captured after the fix
<!-- AC:END -->
