---
id: TASK-21519
title: >-
  Evals DB/Service crash on NULL JSON columns (config_overrides,
  eval_models.config)
status: To Do
assignee: []
created_date: '2026-08-31 03:43'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live UAT of Home recents (PR #2251) seeded an eval run and exposed pre-existing fragility: LocalEvaluationsService.list_runs -> Evals_DB.get_model call json.loads() on NULL config_overrides (run rows) and NULL eval_models.config, raising TypeError. Any run/model row created without those columns breaks every list_runs consumer, including the Evals screen itself. Home's open-tasks provider degrades quietly by design, which masked it there.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 get_model and list_runs enrichment tolerate NULL JSON columns (parse to empty dict or default),Regression tests seed rows with NULL config columns and assert list_runs returns them,Evals screen lists such runs without crashing
<!-- AC:END -->
