---
id: TASK-3796
title: Remove raw summarization input data from debug logs
status: To Do
assignee: []
created_date: '2026-08-09 14:49'
labels:
  - llm-calls
  - observability
  - security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The complete TASK-2118 LLM_Calls logging sweep verified ten debug diagnostics in Local_Summarization_Lib.py and Summarization_General_Lib.py that write full input data or its first 500 characters. These are not raw provider request-payload dictionaries or tool definitions, so they are outside TASK-2118 acceptance criterion 4, but they still violate ADR-029's metadata-only persistent-log boundary and need an atomic containment repair.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All ten verified full-data or 500-character summarization diagnostics emit metadata only and never input content
- [ ] #2 Sentinel tests capture the real logging paths and prove distinctive input strings never reach logs
- [ ] #3 Both summarization modules are swept for equivalent raw input-content diagnostics and every hit is fixed or justified
- [ ] #4 The production diagnostic inventory is reconciled without changing unrelated owners, reasons, counts, or sink topology
<!-- AC:END -->
