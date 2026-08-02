---
id: TASK-1774
title: Build Prompt and Recipe artifact foundations
status: In Progress
assignee: []
created_date: '2026-08-01 23:27'
updated_date: '2026-08-02 02:33'
labels: []
dependencies:
  - TASK-1773
references:
  - Docs/superpowers/plans/2026-08-01-console-prompt-improvement-workbench.md
  - Docs/superpowers/specs/2026-08-01-console-prompt-improvement-design.md
  - >-
    backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the durable Prompt/Recipe artifact contract that lets Console and server-backed libraries distinguish reusable recipes from executable prompts while preserving legacy and schema-v1 behavior. This foundation is required before any editor or improvement flow can safely list, save, search, or exchange structured artifacts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Local Prompt storage, normalized services, and import/export preserve a first-class prompt-or-recipe discriminator and compiled System/User compatibility fields without changing legacy or schema-v1 semantics.
- [ ] #2 Console block schema v2 and server structured kinds are explicitly dispatched, validated, and compiled without ambiguity; malformed, mismatched, and foreign records remain safely distinguishable.
- [ ] #3 Library source capabilities expose supported artifact kinds, limits, conditional-update availability, and honest backend search behavior, with focused local/server compatibility tests passing.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Implement the Console block-v2 immutable artifact models, strict decode states, deterministic compilation, conservative legacy decomposition, and focused fixtures/tests (plan Task 1).
2. Add local Prompt persistence migration, typed storage/service handling, and transactional expected-version updates (plan Task 2).
3. Extend server adapter/service compatibility and add server-side parity contract coverage without changing legacy or schema-v1 behavior (plan Task 3).
4. Add source capability descriptors, honest search routing, and capability-aware validation for prompt artifacts (plan Task 4).
5. Extend prompt import/export and unified scope behavior for first-class Prompt/Recipe artifacts, then verify the Stage 1 acceptance criteria (plan Task 5).

ADR required: yes; ADR path: backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md; Reason: the stage implements the adopted cross-module versioning/storage/runtime contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 4 server parity implementation committed in tldw_server2 on branch `codex/server-console-block-artifacts` at commit `7fd772cdd50d75b505d7449111458833252add87`.
<!-- SECTION:NOTES:END -->
