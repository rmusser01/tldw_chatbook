---
id: TASK-34905
title: Consolidate transaction context managers
status: To Do
created_date: 2026-10-12 17:54
labels:
- simplification
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Four shapes of transaction context manager exist across 12+ copies: ChaChaNotes depth-counted manager Subscriptions inline depth tracking the Media and Prompts borrow twins and the flat store template copied nine times. A transaction manager is nesting policy times begin mode - two enum knobs.

Evidence (verified on dev df4985e): Flat copies at AgentRuns_DB.py:479-515 Library_Collections_DB.py:648-680 Workspace_DB.py:432-470 Dreams_DB.py:299-319 RAG_Indexing_DB.py:295-331 Library_Ingest_Jobs_DB.py:63-89 automatic_work.py:134 Workflows_DB.py:50. Twins: Client_Media_DB_v2.py:1359-1416 vs Prompts_DB.py:658-706. Blockers: callers of the flat template rely on its raise-on-nesting as a misuse guard (docstrings at Library_Collections_DB.py:611-616); ChaChaNotes manager is fused with quiescence machinery so absorbing it is the medium-confidence tail.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One manager parameterized by nesting policy (flat borrow or depth-counted) and begin mode,The flat-template nesting-raise misuse guard preserved,read_transaction write-detection guard preserved,Evals gains a real manager replacing ~20 implicit with-blocks,Per-DB transaction tests pass
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
