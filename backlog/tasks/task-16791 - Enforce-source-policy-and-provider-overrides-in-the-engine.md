---
id: TASK-16791
title: Enforce source policy and provider overrides in the engine
status: Done
assignee:
  - '@robert'
created_date: '2026-08-16 13:21'
updated_date: '2026-08-16 13:31'
labels:
  - research
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Runs record source_policy (balanced default) and provider_overrides_json but the engine ignores both: lanes run based on construction flags, not the run's own routing. Give runs server-parity lane routing and per-run overrides, exposed in the Research window.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 source_policy web_only skips the academic lane and academic_only skips the web engine entirely (no search spend),academic_first and web_first control evidence merge order (docs-budget truncation follows the preferred lane),provider_overrides from the run merge over engine params (engine, result_count) and filter academic providers per run,The Research window gains a policy selector and a providers input, persisted in state and sent on run creation,Tests cover lane gating, merge ordering, override application, and the window payload wiring
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Engine: `execute_run` normalizes the run's `source_policy` (web_only / academic_only / web_first / academic_first / balanced, default balanced) and merges `provider_overrides` over construction params (`engine`, `result_count`), stashing both for the phase machine (cleared in the finally). `_collect_round` gates the web loop (`academic_only` spends nothing on the web engine); the paper lane gates on `web_only`; merge order follows the preferred lane (`academic_first` puts papers first, which is what docs-budget truncation keeps). `academic_providers` overrides reach the paper callable as a `providers=` kwarg (plain call fallback for callables without it).
- Window: policy Select (five options) + providers Input in the create row; `source_policy` always rides the create payload, `provider_overrides.academic_providers` when providers are listed; both persist via save/restore state.
- Gotcha found while wiring: the paper lane's TypeError fallback originally swallowed a NameError (bare `academic_providers` in `_execute_phases`), silently skipping the lane -- fixed by reading the stashed value; the lane-failure catch now only sees genuine provider errors.
- Verified TDD: 4 engine tests (web_only skip, academic_only zero web spend, academic_first ordering, overrides reaching params and papers) + 2 window tests (payload wiring, state persistence); suites 227 passed; ruff clean.
<!-- SECTION:NOTES:END -->
