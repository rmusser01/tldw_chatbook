---
id: TASK-21503
title: 'Console: clarify System prompt read and replace controls'
status: To Do
assignee: []
created_date: '2026-08-24 04:46'
updated_date: '2026-08-24 04:46'
labels:
  - console
  - prompts
  - ux-copy
  - safety
dependencies: []
references:
  - .impeccable/critique/2026-08-24T04-39-32Z__chatbook-widgets-console-console-prompts-modal-py.md
  - Docs/superpowers/qa/console-prompt-improvement-2026-08/README.md
  - backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give the two System-prompt decisions distinct language so users can tell the difference between allowing the improver to read session context and replacing the active session's System prompt.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The analysis control is labeled `Let the improver read the current System prompt` and explains that it is used only to improve the draft and will not change the session.
- [ ] #2 The final application control is labeled `Replace this session's System prompt` and explains that it changes the active session only when the user applies the reviewed result.
- [ ] #3 System replacement remains off by default, while User replacement retains its existing default and no System mutation occurs merely from analysis, Fill, review, save, or cancellation.
- [ ] #4 When no current System prompt exists, the analysis control is unavailable with a truthful reason and improvement can proceed using only the unsent user message.
- [ ] #5 The analysis choice persists through Recipe edits and Fill, while the replacement choice remains a separate review-time decision.
- [ ] #6 Auto, Review, and Recipe tests cover System present/included, System present/excluded, and System absent paths and prove the exact provider request and session mutation boundaries.
- [ ] #7 The revised labels and explanations remain fully painted and keyboard-associated at 140x40, 100x30, and 80x24.
<!-- AC:END -->

