---
id: TASK-32624
title: 'Library Notes: four wordings for go back survive the back-cue unification'
status: To Do
assignee: []
created_date: '2026-09-15 06:44'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A heuristic 4, every persona. Residual of task-32553 (PR #2685), which unified the back cue to one spelling.

What happened. A counted four back-control wordings still reachable inside the sub-screen: '‹ Notes', '‹ Back to list', 'Back', and Escape-only surfaces with no rendered control at all (A caps 17, 26 and the editor and sync panes). B sees the unified '‹ Notes' and '‹ Files' where the wave touched them (B caps 11, 48), so the unification landed where it was applied and stopped there.

The Escape ladder itself is correct and undocumented: Escape from the notes list focuses the rail, from the editor returns to the list, from Info returns to the editor -- three destinations, one key, with nothing on screen teaching the ladder (A section 11).

Cause PROVEN by capture.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One spelling of the back control across every pane of the sub-screen
- [ ] #2 Every pane that Escape leaves renders a back control, or the footer names where Escape goes from here
- [ ] #3 The pins from task-32553 are extended to the panes it did not cover
<!-- AC:END -->
