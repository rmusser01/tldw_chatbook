---
id: TASK-31700
title: Pay down the ui-ready module census breach (976 over the 972 ratchet)
status: Done
assignee:
- '@codex'
created_date: 2026-09-05 11:15
labels:
- performance
- boot
- adr-097
dependencies: []
priority: high
updated_date: 2026-09-05 19:45
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Performance/test_ui_ready_module_census.py` is RED on pristine dev
(verified 2026-09-05 on `e49a7a16d3`): 976 of this repo's modules resident
at `_ui_ready` against the 972 ratchet, 27 NEW modules since the pinned
snapshot. The guard's own culprit list names the recent Console feature
wave: `Chat.console_endpoint_provenance`, `console_environment_state`,
`console_interrupt_rounds`, `console_semantic_revision`,
`console_session_endpoint_policy`, `console_trace_custom_pii`,
`console_trace_errors`, `console_trace_final_values`, and ~19 more (run
the test for the full list). This is the exact ADR-097 consumption
pattern: the ratchet joined per-PR CI in `perf-guard.yml` (task-24461)
and was re-breached within about a day; every open PR now shows a red
"UI latency guardrails" check it did not cause. Per ADR-097 the constant
must not rise -- defer the imports past `_ui_ready` (the established
recipe: function-scope imports at the first-use seam, or the deferred-
wiring `set_timer` pattern app.py already uses) or shed equivalent
modules elsewhere.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 `test_ui_ready_module_census_stays_at_the_pinned_size` passes on pristine dev without raising `MAX_TLDW_MODULES_AT_UI_READY` (any exception needs an owner ledger row per ADR-097)
- [x] #2 The paydown defers cost off the first-paint leg rather than moving the measurement
- [x] #3 The boot-import-weight ratchet stays green after the deferrals (headroom was 19 on the same run)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR.
ADR path: backlog/decisions/097-boot-budget-ratchets.md.
Reason: direct implementation of existing import-deferral ratchet, without changing runtime ownership or UI-ready timing.
1. Reproduce the actual UI-ready module census and trace avoidable first-paint import edges outside the concurrently edited console controller.
2. Add a focused boot-closure/first-use regression and defer the smallest safe unused import graph at its existing interaction seam.
3. Re-measure the unchanged 972-module UI-ready ratchet and app-import guard; run affected first-use tests and lint.
4. Record exact before/after counts and limitations; parent owns commit and PR verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Deferred vLLM setup imports to existing exact-target validation and Environment gatherers/scanner construction to first requested refresh. The scanner remains owned and constructed once before worker dispatch, then reused. No UI-ready timing, budget constant, snapshot, or console_chat_controller.py changes. Existing ADR-097 applies; no new ADR.

Measured baseline UI-ready 976/972; after deferral complete CI boot-guard target passes 18 tests: UI-ready 972/972, app import 639/660, CSS 784399/804000. New subprocess boot-closure regression failed before the patch, then passed with canonical target validation and retained scanner first-use assertions. Parent review found no production defect; applied its concrete gatherer types and explicit TLDW_CONFIG_PATH/configured scratch data directory/task-specific scratch-root test isolation suggestions. Follow-up closure and Environment controller/wiring: 44 passed. New test Ruff/format clean; two production files Bandit clean; git diff --check clean. Three modified existing files retain precisely their 10 baseline Ruff findings (3 handoff, 2 Environment, 5 controller tests), with no new findings; unrelated formatter debt unchanged.

Broader targeted run: 379 passed, 1 failed in 285.89s. test_navigation_to_fresh_models_screen_preserves_exact_ready_handoff leaves VLLM_CONSOLE pending at line 2551. An isolated subprocess loading original HEAD source for both changed production modules via a temporary import loader (no workspace edits) reproduced the same assertion. A current-source isolated retry instead encountered earlier VllmSetupView NoMatches timing. This is not claimed as an all-green workflow run. Logs: /private/tmp/task31700-targeted.log, /private/tmp/task31700-handoff-baseline-source.log, /private/tmp/task31700-handoff-isolated.log, /private/tmp/task31700-ci-guard.log, /private/tmp/task31700-review-adjustment.log. Parent accepted the unchanged-source comparison as sufficient to distinguish the preexisting handoff failure from this import deferral. Scoped census, unchanged-boundary, first-use, and static checks are satisfied; parent owns task Markdown normalization, commit, and PR validation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Restored unchanged UI-ready972-module ratchet by deferring vLLM setup and Environment gatherer/scanner imports to first use. CI boot guards18 passed (UI-ready972, app import639), focused first-use44 passed; broader379 passed/1 preexisting handoff failure reproduced with original source and explicitly documented. No budget/timing relaxation or new static/security findings. Existing ADR097 applies.
<!-- SECTION:FINAL_SUMMARY:END -->
