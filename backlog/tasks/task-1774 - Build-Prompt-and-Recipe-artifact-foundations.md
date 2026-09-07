---
id: TASK-1774
title: Build Prompt and Recipe artifact foundations
status: Done
assignee: []
created_date: '2026-08-01 23:27'
updated_date: '2026-08-02 04:44'
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
- [x] #1 Local Prompt storage, normalized services, and import/export preserve a first-class prompt-or-recipe discriminator and compiled System/User compatibility fields without changing legacy or schema-v1 semantics.
- [x] #2 Console block schema v2 and server structured kinds are explicitly dispatched, validated, and compiled without ambiguity; malformed, mismatched, and foreign records remain safely distinguishable.
- [x] #3 Library source capabilities expose supported artifact kinds, limits, conditional-update availability, and honest backend search behavior, with focused local/server compatibility tests passing.
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
Completed the Stage 1 Prompt/Recipe artifact foundation across five implementation tasks under ADR-040.

Task 1 (44267a254, 2ffc09904) added immutable block-v2 models, deterministic System/User compilation, strict decode states, conservative legacy decomposition, and shared accepted/rejected fixtures in Prompt_Management and Docs/fixtures. Legacy, schema-v1, malformed, mismatched, foreign, and future records remain explicitly distinguishable instead of being coerced.

Task 2 (f32cb4e32, d0accd551) added local schema-v3 artifact_type persistence, normalized local/server adapter propagation, optimistic expected_version updates, WAL conflict handling, and import/overwrite forwarding across DB/Prompts_DB.py and Prompt_Management services. Task 3 (eb24409f8, ff270068c, 4b6c7f744) added structured Markdown Prompt/Recipe round-trip, safe legacy fallback, collision-safe generated names, and serialized fallback creates without changing classic Markdown output.

Task 4 is isolated in tldw_server2 branch codex/server-console-block-artifacts. Complete audited server commit set: 7fd772cdd53e0c8bf5ade391c173e75403f1d6dd (block artifacts); 76033df27d6d342de0bd829bab829fd3d757ae02 (artifact lifecycle, honest search, exact limits, migration/shared fixtures); 4f7e4cc802fd3af695386b5dd0a0160afcfc5066 (canonical final-definition limits, history lane flags, canonical_json_utf8_v1); a6e289031ddbe531fab4983d7e5671a6a85292ed (brief optimistic version). Final audited server head: a6e289031d. Core server modules are Prompts_DB.py, structured_prompts, prompt schemas, and prompt endpoints. The two pre-existing untracked watchlist templates were untouched.

Task 5 (b0bc764b506a9916bc80329a9dcca57047e40986, 312f26527) added frozen local/server capability descriptors, fail-closed exact kind/type/measurement normalization, conservative smaller-wins limits, honest list/search routing, shared create/update wire serialization, compiler-derived compatibility lanes, final-model definition/request measurement, and real brief version preservation in prompt_source_capabilities.py, prompt_scope_service.py, prompt_chatbook_schemas.py, client.py, normalizers, and service tests. Server conditional update intentionally remains false because the authenticated client contract does not enforce expected_version; local expected_version remains local-only. Nothing is truncated. Whitespace-only Browse lists while genuinely nonempty query bytes are preserved.

Verification evidence: Task-focused RED/GREEN suites passed throughout; final Chatbook focused gate passed 66 tests and final server gate passed 92 tests. The complete Chatbook consolidated gate passed once with 267 tests. The final deterministic rerun passed 266 with one unrelated schedule-dependent legacy concurrency test deselected; this does not claim that flaky test passes deterministically. The broader audited server Prompt Management gate passed 160 tests before the final brief-version slice, and the final 92-test server gate covered DB/schema/API behavior. Focused Ruff and git diff checks passed in both repositories. Recursive Bandit over the server structured-prompts package exited 0; no dependencies or licence boundaries changed.

ADR required: yes. ADR path: backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md. Reason: ADR-040 governs the cross-module versioned storage, compatibility, and safe-improvement transaction contract; no new ADR was needed.

Independent review authorized closeout with APPROVE and no Critical, Important, or Minor findings. Unrelated future-UI governance commit 1af323f15 (moving the future Prompts entry to the composer hamburger menu) remained separate and was never staged with Task 5 implementation. Deferred observation: successful server capability normalization is cached for the PromptScopeService lifetime; source-switch/retry invalidation after transient malformed health or a live server upgrade remains a later Browse-stage follow-up and is outside this task.
<!-- SECTION:NOTES:END -->
