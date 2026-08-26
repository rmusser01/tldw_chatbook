---
id: TASK-16231
title: Use a contract-valid Console provider resolution in notes UAT
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 09:19'
updated_date: '2026-08-14 09:19'
labels:
  - testing
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the Console notes workspace UAT aligned with the provider gateway contract so it verifies the joined path instead of failing on an unshaped placeholder.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The notes workspace UAT passes a ConsoleProviderResolution to the real bridge
- [x] #2 The scripted gateway still drives the same find/load/read/edit workflow
- [x] #3 The focused QA UAT and static checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the exact UAT failure caused by `resolution=object()` as RED evidence.
2. Replace the placeholder with the smallest contract-valid `ConsoleProviderResolution`.
3. Run the focused UAT and static checks.

ADR required: no
ADR path: N/A
Reason: This corrects a stale test fixture without changing the Console provider boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the UAT's unshaped object() provider placeholder with the smallest real ConsoleProviderResolution. The scripted gateway and real find/load/read/edit filesystem-tool path are unchanged; the bridge can now execute its current usage-normalization contract instead of failing on a missing provider attribute. Verification: focused QA UAT passed; Ruff lint/format, py_compile, and git diff --check passed.
<!-- SECTION:NOTES:END -->
