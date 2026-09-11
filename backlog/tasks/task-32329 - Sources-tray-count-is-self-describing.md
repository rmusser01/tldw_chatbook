---
id: TASK-32329
title: >-
  Sources tray count is self-describing
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
UX review B5. The staged-context tray header shows a bare digit (str(state.source_count), console_staged_context.py ~58-67) with no unit; zero renders as '0' beside the word Sources. Render a self-describing count with a real empty word.

Filed from the 2026-09-10 Console rail UX review (review item B5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The tray count renders 'none' (word form) when zero sources are staged, and the plain digit otherwise (title already names the noun)
- [ ] #2 Existing sync_state fingerprinting still updates the count in place
- [ ] #3 Widget tests assert the wording for 0, 1, and N sources
<!-- AC:END -->
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

VERIFIED LIVE: header now reads 'Sources - next send' + bare digit '0'. Rescope: zero should read 'none' (word form only where it clarifies); title already carries the noun.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
