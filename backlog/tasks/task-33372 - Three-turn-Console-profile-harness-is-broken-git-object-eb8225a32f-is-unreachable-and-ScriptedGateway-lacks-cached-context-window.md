---
id: TASK-33372
title: 'Three-turn Console profile harness is broken: git object eb8225a32f is unreachable
  and ScriptedGateway lacks cached_context_window'
status: To Do
created_date: 2026-09-28 20:12
labels:
- testing
- performance
- tech-debt
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Performance/run_console_three_turn_profile.py pins ORIGINAL_HARNESS_SHA = 'eb8225a32f88ea43c337aff99804d360384e7668', and test_console_three_turn_profile.py references it. That commit exists on no ref in the repository, so 14 tests fail:
- test_original_runner_*
- test_original_protocol_*
- test_current_*_mismatches_pinned_original
- test_burn_in_summary_is_byte_equivalent_when_only_excluded_metrics_change

Separately, test_scripted_mounted_sample_uses_real_composer_queue_and_fs_write fails because the three-turn ScriptedGateway fake has no cached_context_window attribute. The production gateway grew one. Found 2026-09-28 while verifying PERF-02 (PR #2887) and PERF-01 (PR #2888): the failures reproduce on the unchanged base commit 9cd9aad65f (dev 48019b1914 plus the audit docs commit), so they are pre-existing on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The original-protocol tests no longer depend on an unreachable commit (for example the original runner is vendored or a reachable ref is pinned), and the 14 tests run their real assertions
- [ ] #2 ScriptedGateway matches the production gateway surface the mounted sample uses, and that test passes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-09-29 (via PR #2888, Qodo review): `ScriptedGateway` now implements `cached_context_window` and `resolve_context_window`, using the production gateway's pure first-paint fallback, so the mounted sample no longer raises `AttributeError`. It still fails later: "mounted sample did not settle before timeout". MCP `compose_catalog` fails in the harness, so the scripted `load_tools` -> `fs_write` turn never completes. AC #2 therefore stays open, on that MCP harness drift, and AC #1 (the unreachable commit eb8225a32f) is untouched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
