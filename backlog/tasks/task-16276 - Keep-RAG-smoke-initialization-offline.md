---
id: TASK-16276
title: Keep RAG smoke initialization offline
status: Done
assignee: []
created_date: '2026-08-14 21:34'
updated_date: '2026-08-14 21:36'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the RAG initialization smoke test use the repository's deterministic test configuration so optional embedding packages cannot trigger model downloads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 RAG smoke initialization uses the supported offline mock embedding backend.
- [x] #2 The network guard observes no Hugging Face egress.
- [x] #3 The smoke module and original 25-file sweep block pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the teardown network-guard failure and trace it to the default embedding model.
2. Reuse the existing testing configuration helper with its mock embedding backend.
3. Run the focused smoke test, smoke module, and original sweep block.

ADR required: no
ADR path: N/A
Reason: hermetic test-fixture correction using an existing supported seam; production behavior is unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the production-default `RAGConfig()` in the initialization smoke test with the existing `create_config_for_testing()` seam, which uses the deterministic mock embedding backend and in-memory vector store. RED evidence was a repeatable teardown network-guard error containing four attempted `huggingface.co:443` connections whenever embedding extras were installed. GREEN evidence: the focused node passed, the 16-test smoke module passed, and the exact 25-file sweep block passed with 168 tests plus one Windows-only skip. The two initial socket failures in that block were sandbox-only and the complete network-guard module passed 14 tests plus one Windows-only skip when granted local Unix/loopback bind permission. Ruff check and diff-check passed; Ruff format is inherited-red on `Tests/test_smoke.py`, confirmed against HEAD, so no unrelated whole-file formatting churn was introduced. ADR required: no; production behavior is unchanged.
<!-- SECTION:NOTES:END -->
