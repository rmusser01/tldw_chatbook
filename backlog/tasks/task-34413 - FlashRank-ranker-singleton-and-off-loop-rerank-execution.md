---
id: TASK-34413
title: FlashRank ranker singleton and off-loop rerank execution
status: To Do
created_date: 2026-10-07 02:40
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 1 / F1: the default-on rerank step rebuilds a torch Ranker per query and runs it synchronously on the Textual event loop freezing the UI on every RAG chat send
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Ranker constructed once per process,Rerank and model load never run on the event loop,Ranker cache dir moved off /tmp to app cache dir,Warm vs cold measurement recorded in notes,Targeted RAG pipeline tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 1 (T1)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
