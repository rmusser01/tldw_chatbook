---
id: TASK-32324
title: >-
  Stacked collapsed rail labels discoverable plus badge legend
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
UX review A5. Collapsed handles read 'Context->' / '<-Inspect' with abbreviated badges ('1 appr', 'art') in 13/11 columns. The vertical stacked-label mode exists (console.stack_collapsed_rail_labels, default false) but is buried; and no legend explains the badge abbreviations.

Filed from the 2026-09-10 Console rail UX review (review item A5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The F1 help panel documents what the handle badges mean (approvals, artifact)
- [ ] #2 The stacked-labels setting is documented in the user guide next to the rail presentation settings
- [ ] #3 No behavior change to default handle rendering
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

VERIFIED (code): labels 'Context >'/'<-Inspect', badges '1 appr'/'art' (console_rail_handle.py:120-135); F1 built from CONSOLE_WORKBENCH_SHORTCUT_GROUPS (chat_screen.py:1199-1293).
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
