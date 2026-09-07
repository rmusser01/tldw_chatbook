---
id: TASK-1804
title: >-
  Parakeet lazy-loader tests fail on dev: loader not reached during buffer model
  construction
status: Done
assignee: []
created_date: '2026-08-02 01:25'
updated_date: '2026-08-02 23:10'
labels:
  - stt
  - test-failure
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Local_Ingestion/test_transcription_service_lazy_mlx.py has 2 failures on dev, including test_parakeet_buffer_model_construction_uses_loader ('assert None is RuntimeError(parakeet loader reached)') -- the test expects construction to route through the lazy loader and it does not. Found during a full sweep for TASK-596; verified pre-existing by reproducing in a checkout containing none of that work. Adjacent to the managed Parakeet resolver added in TASK-1696, so worth understanding rather than re-baselining.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both failing tests pass, or are replaced by tests pinning the intended lazy-loading behavior
- [ ] #2 It is stated whether the loader genuinely stopped being used or the test's expectation is stale
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Merged as PR #1231 on 2026-08-02. Environmental: numpy is optional-only in pyproject; the code guards it and raises before the loader, so the sentinel could never fire. Tests now skip on the module's own NUMPY_AVAILABLE (NOT importorskip: numpy 2.x refuses re-import in a loaded process, so importorskip skipped even where numpy IS installed, masking both tests in CI). Verified both directions: 9 passed with numpy 2.4.4, 7+2 skipped without.
<!-- SECTION:NOTES:END -->
