---
id: TASK-21503
title: 'Console: clarify System prompt read and replace controls'
status: Done
assignee:
  - '@codex'
created_date: '2026-08-24 04:46'
updated_date: '2026-08-24 07:06'
labels:
  - console
  - prompts
  - ux-copy
  - safety
dependencies: []
references:
  - >-
    .impeccable/critique/2026-08-24T04-39-32Z__chatbook-widgets-console-console-prompts-modal-py.md
  - Docs/superpowers/qa/console-prompt-improvement-2026-08/README.md
  - >-
    backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give the two System-prompt decisions distinct language so users can tell the difference between allowing the improver to read session context and replacing the active session's System prompt.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The analysis control is labeled `Let the improver read the current System prompt` and explains that it is used only to improve the draft and will not change the session.
- [x] #2 The final application control is labeled `Replace this session's System prompt` and explains that it changes the active session only when the user applies the reviewed result.
- [x] #3 System replacement remains off by default, while User replacement retains its existing default and no System mutation occurs merely from analysis, Fill, review, save, or cancellation.
- [x] #4 When no current System prompt exists, the analysis control is unavailable with a truthful reason and improvement can proceed using only the unsent user message.
- [x] #5 The analysis choice persists through Recipe edits and Fill, while the replacement choice remains a separate review-time decision.
- [x] #6 Auto, Review, and Recipe tests cover System present/included, System present/excluded, and System absent paths and prove the exact provider request and session mutation boundaries.
- [x] #7 The revised labels and explanations remain fully painted and keyboard-associated at 140x40, 100x30, and 80x24.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Introduce shared, exact copy for System analysis permission, its no-System recovery, and the separate System replacement control.
2. Preserve the existing analysis-choice state across Improve and Recipe Fill while keeping replacement off by default and scoped to Apply.
3. Add focused request-boundary and session-mutation tests for System included, excluded, and absent flows across Auto, Review, and Recipe.
4. Extend production-stylesheet responsive assertions/captures to prove both labels and explanations paint fully and remain keyboard reachable.
5. Run the targeted Prompt Workbench/native transaction tests, responsive QA, lint, compilation, and rendered visual inspection.

ADR required: no
ADR path: N/A; ADR-040 remains applicable.
Reason: this is copy and interaction-state clarification within the existing safe improvement/apply transaction boundary.

## Implementation Notes

- Gave System analysis and replacement separate exact labels and disclosures, with replacement off by default and mutation confined to an explicit Apply in the active session.
- Disabled System analysis with a truthful explanation when the session has no System prompt while preserving unsent-message-only improvement.
- Added the nine Auto/Review/Recipe request-boundary cases for included, excluded, and absent System context, including Recipe choice persistence and no incidental session mutation.
- Added production-stylesheet assertions and visually inspected responsive captures at 140x40, 100x30, and 80x24; shortened the Apply explanation so it remains fully painted at the narrowest size.
- Verified the complete Console modal test module (120 passed), the focused shared-editor/native responsive tests (29 passed), responsive QA, Ruff, and compilation. ADR-040 continues to govern the transaction boundary; no new ADR was required.
<!-- SECTION:PLAN:END -->
