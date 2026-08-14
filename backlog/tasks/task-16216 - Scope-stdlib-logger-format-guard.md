---
id: TASK-16216
title: Scope stdlib logger format guard
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 00:40'
updated_date: '2026-08-14 00:46'
labels:
  - test-health
  - logging
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent the stdlib logging format guard from classifying modules as stdlib-logger owners solely because a different variable name ends in `logger`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Exact `logger = logging.getLogger(...)` assignments remain scanned.
- [x] #2 Names such as `root_logger` do not cause Loguru calls in the same module to be scanned as stdlib calls.
- [x] #3 Planted brace-style stdlib violations remain detectable.
- [x] #4 The complete guard file, diagnostic architecture, containing chunk, static, and diff gates pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: this corrects the scope oracle of an existing test-only guard without changing logging policy.

1. Preserve the false-positive failure against `app.py` and its valid Loguru brace calls.
2. Match the exact `logger` assignment target and add positive/negative source fixtures.
3. Run the complete guard, diagnostic architecture, containing chunk, static, and diff gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the substring module classifier with an anchored exact-target assignment check. Added a discrimination fixture proving a real stdlib `logger` remains included while `root_logger` beside a Loguru binding is excluded; the planted brace-style violation remains detected. The complete guard passed four tests, all diagnostic architecture nodes passed, and final chunk 21 passed 962 tests with one Windows-only skip.
<!-- SECTION:NOTES:END -->
