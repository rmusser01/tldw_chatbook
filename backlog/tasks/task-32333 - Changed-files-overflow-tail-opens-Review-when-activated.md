---
id: TASK-32333
title: >-
  Changed-files overflow tail opens Review when activated
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
UX review C4. The '+N more - open Review' tail and the pruned-history line are plain Statics phrased as instructions (console_changed_files_section.py ~196-211) -- false affordance. Make the tail an activatable control that opens the Review screen; reword the pruned line as passive status.

Filed from the 2026-09-10 Console rail UX review (review item C4).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The overflow tail is keyboard-focusable and clickable, and activating it opens the Review screen
- [ ] #2 The pruned-history line reads as status, not an instruction
- [ ] #3 Row-click behavior and the 12-row cap are unchanged
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

RELOCATED on dev: '+N more - open Review' became '... N more - Review opens all' in Environment->Changes (console_environment_state.py:655-660) - still non-clickable while the sibling 'Review in Change Review' row IS clickable. Scope: make the overflow tail open Review too.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
