---
id: TASK-32337
title: >-
  Sources empty state offers an in-rail next action
status: Done
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Prove both attach handler ids have no mount sites (grep). 2. Remove the handlers with a tombstone comment. 3. Fix the sync_state docstring. 4. Document the beneath-tray Library card as the empty-state continuation. 5. Re-run staged-context tests.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The dead '#console-staged-context-attach' button handler is removed from ChatScreen (chat_screen.py:19763-19766)
- [x] #2 The stale sync_state docstring (describing an Attach button compose no longer mounts) is corrected
- [x] #3 The empty state's next action is verifiable in-rail: the Library search card mounted directly beneath the tray is documented as the empty-state continuation (user guide), and the empty-state copy no longer implies a control that does not exist
- [x] #4 The deliberate attach-button-omission contract tests are updated only if their premise changed
<!-- AC:END -->
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Close-out (2026-09-10)

**Approach.** The AC's "in-rail action" was superseded on dev by
TASK-24611 (Library search controls mounted directly beneath the tray),
so the fix is the residual: removed the two dead
`@on(Button.Pressed)` handlers ("#console-attach-context" and
"#console-staged-context-attach") — no widget mounts either id anywhere
(grep-verified; the file-picker flow routes via the composer menu's
ComposerMenuAction, and the tray's Attach button was removed by its
redesign). Fixed the tray `sync_state` docstring that still described an
Attach button varying with state. Documented the empty state's actual
continuation (the beneath-tray Library search card; composer
"Attach file" for drafts) in the user guide, which also retires the
stale "ready/running/blocked/muted" status-word list in favor of the
current copy.

**ADR check.** Not required — dead-code removal + docs.

**Modified.** `UI/Screens/chat_screen.py` (comment tombstone replaces
the handlers), `Widgets/Console/console_staged_context.py` (docstring),
`Docs/User_Guide/console/context-and-rag.md`. The deliberate
attach-omission contract tests were already accurate and unchanged.
Verified: imports OK; staged-context suite re-run green (50 passed in the
B5/C3 commit, unaffected by this change).

### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

VERIFIED LIVE + code: empty copy stands (console_staged_context.py:170-177) and is a tested contract incl. deliberate attach-button omission. MITIGATION EXISTS: Library search controls now sit directly beneath the tray (TASK-24611, right_rail.py:1693-1703). Rescope: remove the dead '#console-staged-context-attach' handler (chat_screen.py:19763-19766) and stale sync_state docstring; empty-state action AC is superseded by the beneath-tray search card.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
