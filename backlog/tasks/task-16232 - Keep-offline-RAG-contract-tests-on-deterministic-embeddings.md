---
id: TASK-16232
title: Keep offline RAG contract tests on deterministic embeddings
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 09:21'
updated_date: '2026-08-14 09:22'
labels:
  - testing
  - rag
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent offline RAG compatibility and fingerprint tests from initializing downloadable Hugging Face models when their assertions concern configuration, collection identity, or service shape.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Offline RAG contract tests avoid network egress
- [x] #2 Compatibility and service-shape assertions retain their original profile and storage coverage
- [x] #3 Collection fingerprint tests still distinguish embedding model identities
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the exact Hugging Face egress failures as RED evidence.
2. Use deterministic mock model identifiers only where the assertion does not require a real embedding model.
3. Run the affected RAG modules and static checks.

ADR required: no
ADR path: N/A
Reason: This is test-fixture isolation with no production RAG boundary change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Kept configuration/profile, service-shape, and collection-fingerprint tests offline by using the built-in deterministic mock embedding backend. The model-fingerprint case uses a distinct mock-prefixed identity, so it still proves model changes fork collections without downloading weights. Verification: 42 passed and 2 intended slow tests skipped across the three affected modules; Ruff lint and py_compile passed; git diff --check passed. Ruff format remains the identical pre-task failure in test_index_isolation_integration.py.
<!-- SECTION:NOTES:END -->
