---
id: TASK-16215
title: Restore Loguru STT callback formatting
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 00:35'
updated_date: '2026-08-14 00:46'
labels:
  - bug
  - diagnostics
  - library
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore interpolation of the callback name in the Library local-STT marshal failure diagnostic.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Loguru call uses Loguru-compatible lazy formatting and includes the safe callback name.
- [x] #2 The failure path remains metadata-only and does not capture the exception.
- [x] #3 The focused file, containing chunk, diagnostic/static, and diff gates pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: this is a one-call diagnostic-format regression fix with no logging-policy change.

1. Preserve the exact failed mock assertion and verify the logger binding is Loguru.
2. Restore the single placeholder to Loguru's `{}` syntax.
3. Run the focused test/file, logging/privacy checks, containing chunk, static, and diff gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Restored the single callback diagnostic from the stdlib `%s` placeholder to Loguru's `{}` placeholder. The callback name remains bounded metadata and the caught exception is not logged or captured. Regenerated the governed diagnostic inventory and verified that only the `app.py` owner digest changed; the non-write inventory and diagnostic architecture checks passed. The focused Library files and final 962-test containing chunk passed.
<!-- SECTION:NOTES:END -->
