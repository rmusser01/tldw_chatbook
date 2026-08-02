---
id: TASK-1804
title: >-
  Parakeet lazy-loader tests fail on dev: loader not reached during buffer model
  construction
status: To Do
assignee: []
created_date: '2026-08-02 01:25'
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
