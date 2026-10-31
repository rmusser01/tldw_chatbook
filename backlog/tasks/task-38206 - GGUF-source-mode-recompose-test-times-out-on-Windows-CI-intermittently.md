---
id: TASK-38206
title: GGUF source-mode recompose test times out on Windows CI intermittently
status: To Do
created_date: 2026-10-30 10:50
labels:
- ci
- flaky-test
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/UI/test_llm_gguf_source_modes.py::test_claim_authority_survives_screen_recompose_and_not_window_selection fails intermittently on the "GGUF source evidence - windows-latest" job with "screen recompose did not reach deferred mount": its bounded _settle_pilot_until wait expires before the recomposed LLMManagementWindow reaches _finish_deferred_mount. It failed on PR #3286 in runs 40798969907 (2026-10-29, together with test_supported_width_keyboard_reaches_each_provider_source_and_actions[llamacpp-llama-cpp], "llamacpp managed source controls did not set") and 41091204880 (2026-10-30). Neither head changed the LLM screen, and the same tests passed on the macOS job and on neighbouring heads. The wait looks too tight for a loaded Windows runner, or the recompose has a Windows-only slow path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause is identified as a wait budget too tight for loaded Windows runners or a Windows-only slow recompose path, with timing evidence from a Windows run.
- [ ] #2 The GGUF source evidence job passes 10 consecutive Windows runs without dropping the deferred-mount assertion.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
