---
id: TASK-32336
title: >-
  Chat Context viewer renders human role names
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
UX review C7. The Ctrl+Shift+P viewer displays internal enum forms like '[ConsoleMessageRole.USER] complete' (task-2704). Map roles to display names in the viewer; this is the power user's audit surface.

Filed from the 2026-09-10 Console rail UX review (review item C7).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The viewer renders 'User'/'Assistant'/etc. instead of enum reprs for message roles
- [ ] #2 Any other internal-form leaks in the viewer's rendered rows are mapped or filed
- [ ] #3 Viewer tests assert the display mapping
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

VERIFIED (code): console_conversation_inspector.py:1810-1822 renders f'[{msg.role}] {msg.status}' with enum-derived values; on py>=3.11 that prints '[ConsoleMessageRole.USER] complete'.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
