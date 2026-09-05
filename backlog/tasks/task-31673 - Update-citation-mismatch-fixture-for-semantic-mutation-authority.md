---
id: TASK-31673
title: Update citation mismatch fixture for semantic mutation authority
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:09'
updated_date: '2026-09-05 18:20'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep active citation mismatch coverage truthful under the current authorized message mutation contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The mismatch lookup returns the expected inactive state while historical summary remains available
- [x] #2 The test fixture follows current semantic message mutation authority
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md (existing)
Reason: Test-only alignment with current security and persistence contracts, not a new boundary.
1. Reproduce the active lookup failure and trace the rejected mutation.
2. Read relevant authority guidance and use the smallest truthful test fixture.
3. Run the focused repository tests and record evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The read-side mismatch fixture now corrupts only the owner fingerprint, following existing repository corruption tests. It no longer attempts a message UPDATE rejected by semantic mutation authority; active-to-body_mismatch transition and historical-summary assertions are unchanged. ADR-024 applies; no new ADR. RED: one mismatch failure among six combined failures. GREEN: both affected citation files 223 passed in 126.62s; Ruff lint/format checks passed. Self-review confirmed canonical-edit/replay coverage remains separate.
<!-- SECTION:NOTES:END -->
