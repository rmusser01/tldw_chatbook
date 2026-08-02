---
id: TASK-1777
title: Provide sensitive provider-boundary prompt improvement UX
status: In Progress
assignee: []
created_date: '2026-08-01 23:30'
updated_date: '2026-08-02 10:25'
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
- [ ] #4 Every registered chat adapter reachable from the auxiliary dispatcher propagates the request-scoped sensitive policy through worker threads and logs only permitted metadata; registry-parity and provider canary tests cover cloud, custom OpenAI-compatible, and local handlers without changing unrelated summarization or embedding paths.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Task 10: add one typed non-streaming auxiliary completion through the captured current Console provider/model and a ContextVar sensitive policy propagated to every registered final chat adapter. The mandatory audit expanded owned log guards to all cloud, custom OpenAI-compatible, and local handlers in `API_CALL_HANDLERS`; unrelated summarization and embedding libraries remain excluded.
2. Task 11: implement typed PromptImprovementService outcomes, strict envelopes, preservation guards, context budgets, stale/cancellation behavior, and one-call fail-closed semantics.
3. Task 12: connect Auto, Review, and Recipe modes to immutable request snapshots, current provider/model disclosure, atomic composer/System apply coordination, cancellation, and temporary Undo.
4. Task 13: add prompt presets, eval fixtures, privacy canary audit, docs, and final cross-stage regression evidence.

ADR required: yes
ADR path: backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md
Reason: ADR-040 already governs the sensitive auxiliary provider boundary, strict validation, and user-controlled application; the audit-driven adapter-log scope expansion implements that decision without creating a new architecture.
<!-- SECTION:PLAN:END -->
