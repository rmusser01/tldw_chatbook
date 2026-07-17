---
id: TASK-248
title: Fix cpu_percent event-loop freeze in PerformanceMetricsWidget
status: Done
assignee: []
created_date: '2026-07-16 14:30'
updated_date: '2026-07-17 00:20'
labels: [performance, ui]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
performance_metrics.py:196 calls psutil cpu_percent(interval=0.1) inside a sync set_interval(2.0) callback — a guaranteed 100ms event-loop sleep every 2s while Embeddings Management is open (whole app hitches metronomically). Correct non-blocking form already at Metrics/metrics.py:200. Latent copies (no UI callers today): metrics_logger.py:153, RAG_Search/simplified/health_check.py:277. Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P0 A3).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No blocking interval= form remains in the widget path; metrics still update
- [x] #2 Latent copies fixed or documented as non-wired
<!-- AC:END -->

## Implementation Notes

Changed all three `cpu_percent(interval=0.1)` call sites to
`cpu_percent(interval=None)` (the non-blocking form, matching the
already-correct template at `Metrics/metrics.py:200`):

- `Widgets/performance_metrics.py:196` (`PerformanceMetricsWidget._update_metrics`,
  the live UI caller -- was sleeping the Textual event loop 100ms every 2s
  tick while Embeddings Management is open). `self.process` is a
  long-lived `psutil.Process()` created once in `__init__`, so after the
  harmless first-call 0.0, every subsequent tick is an accurate delta.
- `Metrics/metrics_logger.py:153` (`MetricsLogger.log_resource_usage`,
  confirmed to have zero callers anywhere in the app -- `app.py` imports
  `log_resource_usage` from the sibling `Metrics/metrics.py` module
  instead, which was already correct). Fixed for consistency and to avoid
  leaving a trap for whoever eventually wires this class up. Note: this
  method used to create a *fresh* `psutil.Process()` every call, which
  with `interval=None` would have permanently returned 0.0 (no call to
  diff against) -- added a small `self._resource_process` cache in
  `__init__` so repeated calls actually get real deltas.
- `RAG_Search/simplified/health_check.py:277`
  (`HealthCheckService.check_system_resources_health`, no test coverage
  and no callers found in `tldw_chatbook/` or `Tests/` -- also a latent/
  unwired copy). Uses the module-level `psutil.cpu_percent()`, which
  psutil caches internally at module scope regardless of caller, so no
  extra state was needed there.

New test `Tests/Widgets/test_performance_metrics_nonblocking.py`. RED
before the fix: `_update_metrics()` called `cpu_percent(interval=0.1)`
(assertion failure) and the lexical guard found blocking call sites in
all three files. GREEN after: 3 passed. Full regression check:
`Tests/Widgets/` + `Tests/Utils/test_metrics_logger.py` -- 320 passed, 2
skipped (pre-existing, unrelated).
