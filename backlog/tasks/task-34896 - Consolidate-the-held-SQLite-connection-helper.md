---
id: TASK-34896
title: Consolidate the held-SQLite-connection helper
status: To Do
created_date: 2026-10-12 17:52
labels:
- simplification
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Eleven DB classes hand-copy the same held thread-local SQLite connection with liveness ping pragma setup and closed-handle revival (~800-1000 LOC) and drift has already shipped. Every store is the same connection; only pragma set isolation policy and exception type differ and all three are constructor parameters. Per ADR-177 consolidate by helper not base class.

Evidence (verified on dev df4985e): The 11 copies: ChaChaNotes_DB.py:2958-3120 (~210) Client_Media_DB_v2.py:1108-1243 (~135) Prompts_DB.py:402-520 (~120) Subscriptions_DB.py:636-685 (~100) AgentRuns_DB.py:375-477 (~100) Library_Collections_DB.py:498-585 (~90) Workspace_DB.py:353-423 (~85) Dreams_DB.py:238-287 (~65) RAG_Indexing_DB.py:183-283 (~100) Evals_DB.py:240-282 (~45) Library_Ingest_Jobs_DB.py:92-133 (~45). Comments self-identify as copies (Dreams_DB.py:235 AgentRuns_DB.py:422). _LIVENESS_PING_IDLE_SECONDS duplicated in 6 files. Blockers: task-22224 exceptions (Media Prompts Subscriptions Evals legacy isolation each with named follow-up) and ChaCha-only quiescence registry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One shared helper owns ping pragma and closed-handle revival,Per-store pragma set isolation policy and exception type are parameters,task-22224 legacy-isolation stores keep their documented policy,:memory: path handling and the Dreams_DB close drift unified,Per-DB open close ping tests pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
