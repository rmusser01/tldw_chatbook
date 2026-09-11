---
id: TASK-32335
title: >-
  Truncated inspector section rows expose full text via tooltip
status: To Do
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: low
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review C6. Inspector section rows render a truncated secondary line with no tooltip, while changed-files rows carry full un-elided paths in theirs -- inconsistent truncation recovery (console_inspector_section.py rows ~610-617). Add tooltips carrying the untruncated primary+secondary text.

Filed from the 2026-09-10 Console rail UX review (review item C6).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every rendered inspector section row has a tooltip containing its full primary and secondary text
- [ ] #2 Tooltips are markup-escaped (user content has raised MarkupError before -- PR-T1 I1)
- [ ] #3 No per-row layout change
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

VERIFIED (code): ConsoleInspectorSectionRow sets no tooltip (console_inspector_section.py:820-1003); only chevron and view-all tails have them.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
