---
id: TASK-21501
title: 'Console: surface improvement Undo and before-after review'
status: To Do
assignee: []
created_date: '2026-08-24 04:46'
updated_date: '2026-08-24 04:46'
labels:
  - console
  - prompts
  - ux
  - recovery
dependencies: []
references:
  - .impeccable/critique/2026-08-24T04-39-32Z__chatbook-widgets-console-console-prompts-modal-py.md
  - Docs/superpowers/qa/console-prompt-improvement-2026-08/README.md
  - backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make an automatic prompt replacement immediately understandable and reversible from the composer. A user should not need to reopen the composer menu to discover that the previous draft can be restored or to inspect what changed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The automatic mode is labeled `Replace draft automatically` before execution so its effect is clear.
- [ ] #2 After a successful automatic replacement, the composer immediately shows a persistent `Draft improved` status with keyboard-reachable `Undo` and `Review changes` actions.
- [ ] #3 Undo restores the exact pre-improvement composer transaction, including draft text and attachment-related state, and remains safe when invoked repeatedly or after unrelated late provider results.
- [ ] #4 `Review changes` opens a before-and-after comparison without first reverting the improved draft; the improved version remains editable and the user can keep it or restore the original.
- [ ] #5 The visible improvement status is cleared only by a subsequent draft edit, send, explicit restoration, or session/context replacement, and stale actions cannot mutate newer composer state.
- [ ] #6 Success, failure, and stale-result status changes are textually exposed for keyboard/accessibility users without logging prompt bodies or changing the existing sensitive-provider boundary.
- [ ] #7 The composer-menu Undo remains available as a secondary recovery path, and both recovery surfaces behave consistently at supported terminal sizes.
<!-- AC:END -->

