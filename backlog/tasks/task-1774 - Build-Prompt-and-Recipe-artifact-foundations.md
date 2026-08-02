---
id: TASK-1774
title: Build Prompt and Recipe artifact foundations
status: In Progress
assignee: []
created_date: '2026-08-01 23:27'
updated_date: '2026-08-02 04:37'
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
Task 4 server parity is committed in tldw_server2 on branch codex/server-console-block-artifacts. Complete server commit set: 7fd772cdd53e0c8bf5ade391c173e75403f1d6dd (initial Console block artifact support); 76033df27d6d342de0bd829bab829fd3d757ae02 (artifact lifecycle identity, honest default search, exact byte limits, and migration/shared-fixture coverage); 4f7e4cc802fd3af695386b5dd0a0160afcfc5066 (final canonical definition limits, history lane flags, and canonical_json_utf8_v1 measurement); a6e289031d (Prompt brief optimistic version in DB list/schema/API coverage). Final audited server head: a6e289031d. The two pre-existing untracked watchlist templates remain untouched.

Task 5 source capabilities and honest server search are implemented in Chatbook commits b0bc764b506a9916bc80329a9dcca57047e40986 and review fix 312f26527. The fix validates every block-v2 candidate before mutation, persists compiler-derived lanes, normalizes/measures final definitions and exact create/update wire mappings through the shared client serializer, requires type-strict measurement descriptors, routes whitespace-only Browse to list, and preserves real server brief versions. Verification: 66 focused Chatbook tests passed; 92 focused server tests passed; required post-fix review found no findings and marked READY. The complete consolidated Chatbook gate passed once with 267 tests; final deterministic rerun passed 266 with the unrelated schedule-dependent legacy concurrency test deselected. Ruff and diff checks passed in both repositories. ADR-040 remains governing. The ignored Task 5 report records RED/GREEN evidence and the deferred capability-cache invalidation observation. TASK-1774 intentionally remains In Progress with every criterion unchecked pending controller-directed closeout.
<!-- SECTION:NOTES:END -->
