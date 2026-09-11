---
id: TASK-32326
title: >-
  Console rail terminology pass resolves Context naming collisions
status: To Do
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review B2. The left rail is branded 'Context' but no longer contains the context-staging UI (Sources tray moved to the Inspector in task-400); 'Chat Context' viewer (Ctrl+Shift+P) is an unrelated surface one word away; 'Sources' has four senses; docs say both 'Inspector' and 'run inspector'. Decide and apply one vocabulary: user-facing rail brand, staging concept, viewer name. Code ids (console-context-rail-*) must NOT be renamed -- this is a copy/docs pass, not a refactor.

Filed from the 2026-09-10 Console rail UX review (review item B2).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A single decision record (ADR or task note) fixes the vocabulary: what the left rail is called, what the staging tray is called, what the Ctrl+Shift+P viewer is called
- [ ] #2 User-facing copy no longer uses 'Context' for two unrelated concepts; the F1 help panel and user-guide docs use the agreed terms consistently
- [ ] #3 All shell ids and config keys remain unchanged (verified by grep: no test churn from renames)
- [ ] #4 Docs glossary or terminology section added covering rail, handle, staged sources, scope, Inspector
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

VERIFIED LIVE/Code: left header 'Context <', right '> Inspect', handle constants 'Context v'/'<- Inspector' (rail_state.py:79-80), Ctrl+Shift+P opens 'Conversation Inspector' modal whose tab header still says 'Chat Context' (console_conversation_inspector.py:1777-1781). Three names, one concept space.
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
