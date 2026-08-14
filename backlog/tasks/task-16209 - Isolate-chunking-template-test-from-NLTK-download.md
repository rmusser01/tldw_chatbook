---
id: TASK-16209
title: Isolate chunking template test from NLTK download
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 00:08'
updated_date: '2026-08-14 00:08'
labels:
  - test-health
  - chunking
  - network
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the academic-template unit test deterministic and offline instead of allowing an optional NLTK corpus download during semantic chunking.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The domain-template test exercises the built-in sentence-splitting path without network access.
- [x] #2 The test still proves academic-template chunking returns metadata-bearing results.
- [x] #3 Focused, containing-file/chunk, mutation, static, and diff evidence pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: this isolates an existing unit test from an optional runtime dependency without changing application behavior.

1. Preserve the network guard's two blocked NLTK corpus downloads as RED evidence.
2. Force the existing no-NLTK fallback only within the domain-template test.
3. Prove removing the isolation reintroduces blocked egress, then run focused/file/chunk/static/diff gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Forced the optional-NLTK fallback only inside the academic-template unit test, preserving its actual contract: template loading, chunk creation, and metadata. RED: the chunk-14 sweep recorded blocked attempts to download both `punkt` and `punkt_tab` from `raw.githubusercontent.com`; removing the availability override reproduced the same two attempts. GREEN: the complete template test file passed 14 tests without network access. The containing 25-file chunk, scoped Ruff, and diff checks passed. ADR required: no; production download and fallback policy are unchanged.
<!-- SECTION:NOTES:END -->
