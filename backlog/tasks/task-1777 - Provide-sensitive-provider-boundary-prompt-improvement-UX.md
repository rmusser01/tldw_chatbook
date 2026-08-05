---
id: TASK-1777
title: Provide sensitive provider-boundary prompt improvement UX
status: Done
assignee: []
created_date: '2026-08-01 23:30'
updated_date: '2026-08-02 22:11'
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
- [x] #1 Auxiliary completion resolves the active Console provider and model through the gateway, makes at most one non-streaming sensitive request per user action, and blocks tools, history, RAG, attachments, staged sources, transcript writes, and prompt-content logging.
- [x] #2 PromptImprovementService validates typed Auto, Review, and Recipe-fill outcomes; no-change, malformed, protected-token, stale, cancellation, and provider-failure paths fail closed with an actionable UI state.
- [x] #3 The Console workbench supports user-controlled Apply, optional system-prompt persistence, exact Undo, Recipe fill, and cancellation while preserving provider/model honesty and all approved privacy invariants.
- [x] #4 Every registered chat adapter reachable from the auxiliary dispatcher propagates the request-scoped sensitive policy through worker threads and logs only permitted metadata; registry-parity and provider canary tests cover cloud, custom OpenAI-compatible, and local handlers without changing unrelated summarization or embedding paths.
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the sensitive one-shot Prompt improvement boundary and user-controlled Auto, Review, and Recipe flows across Tasks 10-13, including normalized source identity, canonical Additional-context handling, real artifact compatibility guards, protected inline-file veto, and stale-result review behavior. Client SHA: b856795415cb8f8f6abf9eafeb2f73a7a6bae908. Server compatibility SHA: a6e289031ddbe531fab4983d7e5671a6a85292ed. Fresh final verification: the exact client matrix passed 943 tests with 2 environment warnings; the exact server matrix passed 160 tests with 2 test-environment warnings; CSS build, app import, compileall, QA-runner Ruff/format/py_compile, both diff checks, and recursive Bandit passed. Recursive Bandit scanned all 4,696 lines with zero findings; the prescribed non-recursive command retains its documented directory-skip caveat. The isolated real-app QA run regenerated and visually inspected 25 SVG captures at 140x40, 100x30, and 80x24. At every size, painted-output and geometry assertions prove the stacked System/User footer shows both checkbox glyphs and full labels without clipping or overlap. Its single canary provider call produced zero log occurrences for System, User, block, inline-file body, inline-file label, opaque placeholder, and response content. Prompts remains exclusively the first normal composer-hamburger item; it is absent from the tab/control row and idle composer geometry is unchanged. External Server Browse was unavailable, so its real unavailable/Retry state is paired with automated old/modern-server compatibility evidence. The server worktree's two unrelated untracked watchlist templates were preserved. Configured whole-feature Ruff/format/mypy checks reproduce the documented repository baseline and were not mechanically rewritten. ADR required: yes for the stage; existing ADR-040 remains applicable and unchanged because no new storage, authority, provider-boundary, security, dependency, or cross-module decision was introduced. Full evidence and design sections 1-15 traceability: Docs/superpowers/qa/console-prompt-improvement-2026-08/README.md.
<!-- SECTION:NOTES:END -->
