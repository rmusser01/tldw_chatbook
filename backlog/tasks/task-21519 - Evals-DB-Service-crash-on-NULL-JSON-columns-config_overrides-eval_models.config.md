---
id: TASK-21519
title: >-
  Evals DB/Service crash on NULL JSON columns (config_overrides,
  eval_models.config)
status: Done
assignee:
  - '@zcode'
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
- [x] #1 get_model and list_runs enrichment tolerate NULL JSON columns (parse to empty dict or default),Regression tests seed rows with NULL config columns and assert list_runs returns them,Evals screen lists such runs without crashing
<!-- AC:END -->

## Implementation Plan

1. RED: seed eval_models/eval_tasks/eval_runs rows with NULL JSON columns via raw SQL (the API always writes valid JSON; the crash data comes from rows created without those columns) and assert get_model/list_models/list_runs/get_run return them.
2. Add a NULL-tolerant JSON parse helper and route the four model/run parse sites through it (default {}); task/dataset sites stay untouched -- not this task's named columns.

ADR required: no
ADR path: N/A
Reason: Defensive-parse fix inside one DB module; no schema or contract change.

## Implementation Notes

Added ``EvalsDB._loads_json_or_default`` (NULL -> default; also tolerates malformed JSON) and routed the four named parse sites through it: ``get_model`` and ``list_models`` (``eval_models.config``), ``get_run`` and ``list_runs`` (``eval_runs.config_overrides``). Rows created without those columns now read back with ``{}`` instead of raising ``json.loads(None)`` TypeError.

TDD evidence, kept honest: ``TestNullJsonColumnTolerance`` (Tests/Evals/test_evals_db.py) seeds raw-SQL rows with NULL config columns; the three tests fail on the unfixed DB with exactly the filed crash (``TypeError: the JSON object must be str, bytes or bytearray, not NoneType``) and pass with the fix (48 passed file-wide). The first RED iteration's failure was a seed-constraint mistake (eval_tasks NOT NULL columns), so the RED was re-verified by stashing only the production change against the corrected seed.

AC#3 scope note: ``LocalEvaluationsService.list_runs`` delegates directly to ``db.list_runs`` (local_evaluations_service.py:531), and the Evals screen lists through that service, so the DB-seam tolerance above is what unblocks the screen; a separate screen-mount test would re-pin the same seam and was not added. The task/dataset JSON columns (``config_data``/``metadata``) share the bug class but are not this task's named columns and remain as filed.

Modified: ``tldw_chatbook/DB/Evals_DB.py``, ``Tests/Evals/test_evals_db.py``.
