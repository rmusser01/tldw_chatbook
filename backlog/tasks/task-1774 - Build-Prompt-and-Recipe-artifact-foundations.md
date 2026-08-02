---
id: TASK-1774
title: Build Prompt and Recipe artifact foundations
status: In Progress
assignee: []
created_date: '2026-08-01 23:27'
updated_date: '2026-08-02 03:35'
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
Task 4 server parity is committed in tldw_server2 on branch codex/server-console-block-artifacts. Commit set: 7fd772cdd53e0c8bf5ade391c173e75403f1d6dd (initial Console block artifact support); 76033df27d6d342de0bd829bab829fd3d757ae02 (review fix preserving artifact lifecycle identity, honest default search, exact byte limits, and migration/shared-fixture coverage); 4f7e4cc802fd3af695386b5dd0a0160afcfc5066 (review fix enforcing final canonical definition limits, truthful history lane flags, and the canonical_json_utf8_v1 measurement contract). Final audited server head: 4f7e4cc802fd3af695386b5dd0a0160afcfc5066. TASK-1774 remains In Progress pending the remaining stage work.

Task 5 source capabilities and honest server search are implemented in Chatbook commit b0bc764b506a9916bc80329a9dcca57047e40986. Frozen local/server capability normalization preserves exact structured kinds and artifact types, applies conservative/smaller advertised limits with canonical_json_utf8_v1 measurement, keeps server conditional update disabled, routes empty Browse to paginated list and non-empty queries to server search, raises typed unavailable outcomes, and preserves brief/detail identity without hidden fetches. Final verification: 54 focused tests and 255 consolidated Stage-1 tests passed; focused Ruff and diff checks passed. ADR-040 remains governing. All acceptance criteria have implementation evidence in the Task 5 report, but TASK-1774 intentionally remains In Progress with every criterion unchecked pending independent review and controller-directed closeout.
<!-- SECTION:NOTES:END -->
