---
id: TASK-1777
title: Provide sensitive provider-boundary prompt improvement UX
status: To Do
assignee: []
created_date: '2026-08-01 23:30'
labels: []
dependencies:
  - TASK-1776
references:
  - Docs/superpowers/plans/2026-08-01-console-prompt-improvement-workbench.md
  - Docs/superpowers/specs/2026-08-01-console-prompt-improvement-design.md
  - >-
    backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete user-controlled Prompt improvement only after the library and composer transaction contracts are safe. This stage adds one-shot, privacy-preserving use of the active Console provider/model and exposes Auto, Review, and Recipe-fill outcomes without silent session mutation or content logging.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Auxiliary completion resolves the active Console provider and model through the gateway, makes at most one non-streaming sensitive request per user action, and blocks tools, history, RAG, attachments, staged sources, transcript writes, and prompt-content logging.
- [ ] #2 PromptImprovementService validates typed Auto, Review, and Recipe-fill outcomes; no-change, malformed, protected-token, stale, cancellation, and provider-failure paths fail closed with an actionable UI state.
- [ ] #3 The Console workbench supports user-controlled Apply, optional system-prompt persistence, exact Undo, Recipe fill, and cancellation while preserving provider/model honesty and all approved privacy invariants.
<!-- AC:END -->
