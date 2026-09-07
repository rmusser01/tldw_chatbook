---
id: TASK-16073
title: Repair stale Console agent bridge resolution fixtures
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 03:59'
updated_date: '2026-08-14 03:59'
labels:
  - testing
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the Console agent bridge test suite after current provider-usage code began reading the real provider-resolution contract, without weakening production validation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Bridge tests pass contract-valid provider resolutions and canonical continuation-owner rows instead of stale doubles
- [x] #2 The formerly failing no-tool, native-tool, continuation, and complete bridge module tests pass
- [x] #3 No production behavior changes
- [x] #4 Static and diff checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the reproduced RED from the stale bare-object resolution fixture.
2. Replace only the shared test default with a contract-valid resolution carrying provider and model metadata.
3. Re-run the named regression, complete bridge module, Ruff, and diff checks; mutation-restore the stale fixture to prove the regression fails.
4. Record implementation notes and completion evidence.

ADR required: no
ADR path: N/A
Reason: test-fixture repair only; no production boundary or behavior changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Replaced the shared bare-object resolution and the native-tool fake with real `ConsoleProviderResolution` fixtures carrying the provider/model fields production now reads.
- Updated resumed-continuation fixtures to emit canonical assistant tool-call rows and owner metadata, preserving the production one-owner-per-private-group validation.
- Verified the stale shared fixture still makes the named stream test fail, then restored it. The complete bridge module passes: 193 tests, with only the existing requests dependency warning.
- Ruff check/format and `git diff --check` pass. Only the bridge test and this task record changed; no ADR or production change was needed.
<!-- SECTION:NOTES:END -->
