---
id: TASK-512
title: Update chatbook dependency test for citation-aware export
status: Done
assignee: []
created_date: '2026-07-24 18:12'
updated_date: '2026-07-24 18:15'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the chatbook character-dependency unit test focused after conversation export began composing citation-aware services, without weakening production identity loading or citation export behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The unit test supplies the citation-aware conversation-service dependency at the module composition seam
- [x] #2 The test continues to verify character dependency tracking without requiring a real citation database
- [x] #3 Production citation identity loading and chatbook export code are unchanged
- [x] #4 The focused and full Chatbooks tests pass
- [x] #5 Task documentation includes the ADR decision, verification evidence, and implementation notes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the branch-only Mock-not-subscriptable failure and passing merge-base result.
2. Update the narrow character-dependency test to patch the citation conversation-service composition seam with a service returning no context messages.
3. Run the exact test and full Chatbooks suite.
4. Run Ruff format/check and git diff --check; independently review before completion.

ADR required: no
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: ADR-024 already defines citation identity and export provenance; this task updates a unit-test dependency fixture without changing production boundaries or behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated only the character-dependency unit test to patch `build_local_citation_conversation_service` at the `chatbook_creator` composition seam. The supplied mock conversation service returns no context messages, so the real collector continues to exercise conversation export and character-dependency tracking without sending the intentionally bare database mock through citation identity loading. Production citation identity, repository, and chatbook export code remain unchanged. RED evidence: the exact test failed because `load_local_citation_identity_context` attempted to subscript a bare `Mock` row. GREEN verification: the exact regression passed; the full `Tests/Chatbooks` suite passed with 138 tests and 1 skipped; Ruff check and format check passed; `git diff --check` passed. ADR required: no. ADR path: `backlog/decisions/024-rag-citation-provenance-and-source-resolution.md`. Reason: ADR-024 already governs citation identity and export provenance; this is a test-only dependency-fixture correction.
<!-- SECTION:NOTES:END -->
