---
id: TASK-31561
title: Load Library split CSS in Notes widget harnesses
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 01:47'
updated_date: '2026-09-05 01:48'
labels:
  - library
  - tests
  - css
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore compact Notes pager geometry coverage by making its direct widget harnesses load the same Library owner stylesheet as production.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Notes widget harnesses use the complete app stylesheet stack.
- [x] #2 Compact pager copy wraps without clipping and nested rows retain full-width indentation.
- [x] #3 The complete Notes canvas widget module passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce both pager geometry failures and compare the harness stylesheet with the production app stack.
2. Replace bundle-only CSS in the direct Notes harnesses with the shared complete app stylesheet tuple.
3. Run both exact regressions, the complete Notes canvas widget module, Ruff, and diff checks.

ADR required: no
ADR path: N/A
Reason: TASK-25812 already defines Library stylesheet ownership; this only aligns stale direct test harnesses.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Replaced both bundle-only direct Notes harness stylesheet declarations with the complete shared app stack, restoring the Library-owned pager rules moved by TASK-25812.
- Evidence: both compact pager regressions pass and the complete Notes canvas widget module passes 31/31.
- ADR required: no; test-only alignment with established stylesheet ownership.
<!-- SECTION:NOTES:END -->
