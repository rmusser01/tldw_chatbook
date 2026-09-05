---
id: TASK-31700
title: Pay down the ui-ready module census breach (976 over the 972 ratchet)
status: To Do
assignee: []
created_date: '2026-09-05 11:15'
labels:
  - performance
  - boot
  - adr-097
dependencies: []
priority: high
---

## Description (the why)

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

## Acceptance Criteria (the what)

- [ ] `test_ui_ready_module_census_stays_at_the_pinned_size` passes on pristine dev without raising `MAX_TLDW_MODULES_AT_UI_READY` (any exception needs an owner ledger row per ADR-097)
- [ ] The paydown defers cost off the first-paint leg rather than moving the measurement
- [ ] The boot-import-weight ratchet stays green after the deferrals (headroom was 19 on the same run)
