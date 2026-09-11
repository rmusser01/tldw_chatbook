---
id: TASK-32337
title: >-
  Sources empty state offers an in-rail next action
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
UX review D1. The empty state 'No sources attached. Stage sources from Library.' (console_staged_context.py ~107-111) is guidance without an action, while the sibling scope row offers 'Narrow...' in place. Add an activatable 'Open Library...' action and resolve the docstring drift (sync_state docstring still describes an Attach button compose no longer mounts).

Filed from the 2026-09-10 Console rail UX review (review item D1).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The dead '#console-staged-context-attach' button handler is removed from ChatScreen (chat_screen.py:19763-19766)
- [ ] #2 The stale sync_state docstring (describing an Attach button compose no longer mounts) is corrected
- [ ] #3 The empty state's next action is verifiable in-rail: the Library search card mounted directly beneath the tray is documented as the empty-state continuation (user guide), and the empty-state copy no longer implies a control that does not exist
- [ ] #4 The deliberate attach-button-omission contract tests are updated only if their premise changed
<!-- AC:END -->
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

VERIFIED LIVE + code: empty copy stands (console_staged_context.py:170-177) and is a tested contract incl. deliberate attach-button omission. MITIGATION EXISTS: Library search controls now sit directly beneath the tray (TASK-24611, right_rail.py:1693-1703). Rescope: remove the dead '#console-staged-context-attach' handler (chat_screen.py:19763-19766) and stale sync_state docstring; empty-state action AC is superseded by the beneath-tray search card.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
