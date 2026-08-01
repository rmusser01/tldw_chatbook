---
id: TASK-1773
title: Verify composer action unification for the Prompt Workbench
status: In Progress
assignee: []
created_date: '2026-08-01 23:26'
updated_date: '2026-08-01 23:30'
labels: []
dependencies: []
references:
  - TASK-1680
  - Docs/superpowers/plans/2026-08-01-console-prompt-improvement-workbench.md
  - Docs/superpowers/specs/2026-08-01-console-prompt-improvement-design.md
  - >-
    backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish the already-approved, width-bounded Console composer action contract before Prompt Workbench changes depend on it. This stage verifies or minimally ports TASK-1680 semantics on the current target branch so later Prompt UX does not duplicate or destabilize existing actions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The composer at-rest row remains width-bounded and exposes overflow, Send, and Mic without duplicate Attach or Save Chatbook controls.
- [ ] #2 Overflow actions use stable attach-context and save-chatbook IDs that invoke the existing screen handlers, with unavailable actions remaining visible and explaining why.
- [ ] #3 Focused composer menu, collapse, native chat-flow, and Workbench contract tests verify action parity on the target branch.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Approved plan Task 0 excerpt:
1. Rebase/inspect the current target and the TASK-1680 reference semantics; port only missing behavior.
2. Add or port focused action-surface tests, then demonstrate RED only for missing behavior.
3. Keep the composer row fixed-width; route overflow selections to existing ChatScreen handlers.
4. Regenerate CSS only if its source changes and run the focused geometry/action-parity suite.
5. Commit only a required port; record verification-only evidence otherwise.

ADR required: yes
ADR path: backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md
Reason: This prerequisite implements the action-row boundary on which the approved long-lived Prompt Workbench UX depends.
<!-- SECTION:PLAN:END -->
