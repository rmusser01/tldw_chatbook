---
id: TASK-32323
title: >-
  Inspector rail per-section collapse
status: To Do
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review A4. The Inspector rail is one giant scroll unit with no per-section collapse (right_rail.py module docstring notes the whole rail is a single collapse/expand unit), unlike the left rail's seven disclosure sections. Introduce per-section headers for at least Sources / Run / Session Settings, persisted like the left rail's section flags.

Filed from the 2026-09-10 Console rail UX review (review item A4).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Inspector's section-collapse structure is audited against the residual gap (Sources tray, run-inspector block) and the outcome is recorded in Implementation Notes: either close (bounded sections + scroll hints + More disclosure are sufficient) or a follow-up is filed with rationale
- [ ] #2 No regression to the existing collapsible sections (Environment/Tasks/Agents), n/p navigation, or the pinned overflow-tail contract
- [ ] #3 User guide documents the Inspector's section navigation (n/p) and collapsible sections
<!-- AC:END -->
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

PREMISE LARGELY GONE on dev: Environment/Tasks/Agents are collapsible ConsoleInspectorSections (right_rail.py:1608-1667), n/p section navigation exists, overflow tail 'more sections - scroll' pinned (right_rail.py:1772-1783). Residual: Sources tray and run-inspector block are not collapsible, but run inspector has a 'More' disclosure and Sources is bounded with its own scroll hint.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
