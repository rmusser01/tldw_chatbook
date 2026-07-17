---
id: TASK-248
title: Fix cpu_percent event-loop freeze in PerformanceMetricsWidget
status: To Do
assignee: []
created_date: '2026-07-16 14:30'
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
- [ ] #1 No blocking interval= form remains in the widget path; metrics still update
- [ ] #2 Latent copies fixed or documented as non-wired
<!-- AC:END -->
