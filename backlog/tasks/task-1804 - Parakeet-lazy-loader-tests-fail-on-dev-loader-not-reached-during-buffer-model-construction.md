---
id: TASK-1804
title: >-
  Parakeet lazy-loader tests fail on dev: loader not reached during buffer model
  construction
status: Done
assignee: []
created_date: '2026-08-02 01:25'
updated_date: '2026-08-02 20:21'
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
Resolved 2026-08-02: ENVIRONMENTAL, not a defect. numpy appears only under [project.optional-dependencies] and transcription_service guards it correctly -- _transcribe_buffer_with_parakeet_mlx raises 'NumPy is required for buffer transcription' BEFORE reaching _ensure_parakeet_mlx_import, so in a numpy-less install the loader sentinel cannot fire and the failure reads as 'loader bypassed' when the guard is working as designed. Both tests now skip on the module's own NUMPY_AVAILABLE. NOT pytest.importorskip: numpy 2.x refuses re-import in a process that already loaded it, so importorskip skipped even WHERE numpy is installed -- masking the tests everywhere including CI. Verified both ways: 9 passed under an interpreter with numpy 2.4.4, 7 passed + 2 skipped without. Commit 8306bb3c5 (branch fix/task-1772-1804-preexisting-failures, pending the TASK-1772 work in the same branch).
<!-- SECTION:NOTES:END -->
