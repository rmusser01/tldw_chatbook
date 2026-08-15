---
id: TASK-16323
title: Enforce run budgets with a reserve-and-settle ledger
status: To Do
assignee:
  - '@robert'
created_date: '2026-08-15 05:15'
labels:
  - research
  - budget
dependencies:
  - TASK-16322
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The runs schema carries limits_json but nothing enforces it and the deep-search pipeline bounds only time and bytes: phase 1 sub-query generation plus phase 2 map-reduce can make many LLM calls with unbounded spend. Port tldw_server's pattern: reserve before each unit of work, settle after, hard-stop when exhausted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 limits_json keys (max_searches and max_fetched_docs and max_runtime_seconds at minimum) are enforced before each unit of work with structured research_limit_exceeded errors,LLM calls in the deep-search pipeline reserve tokens before execution and settle after with the ledger persisted per run,An exhausted budget stops the run cleanly at a phase boundary with a terminal event while preserving partial artifacts,Ledger state is inspectable per run,Tests pin the enforcement and clean-stop behavior
<!-- AC:END -->
