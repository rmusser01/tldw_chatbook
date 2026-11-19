---
id: TASK-41052
title: "Decide whether finite workers retire the four stores that opt out"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- architecture
- backup-recovery
- database
priority: low
references:
- tldw_chatbook/DB/AgentRuns_DB.py
- tldw_chatbook/DB/RAG_Indexing_DB.py
- tldw_chatbook/DB/Library_Ingest_Jobs_DB.py
- tldw_chatbook/Notes/file_notes_replica.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36948.1 made each store declare its own backup participation (`@declare_store`) and kept dev's behaviour exactly. That made a disagreement visible:
- **What the declarations say.** Four stores declare `finite_worker=False` (`DB/AgentRuns_DB.py`, `DB/RAG_Indexing_DB.py`, `DB/Library_Ingest_Jobs_DB.py` and `Notes/file_notes_replica.py`). So `run_finite_local_worker` does not close their per-thread caches, while the pause path does.
- **What the ADR says.** ADR-199 says a finite operation retires any cache it first opens.
- **Which omissions look accidental.** agent_runs and rag_indexing use the same per-thread kernel as stores that are closed, so their omission looks like drift. library_ingest_jobs and file_notes share one connection across threads, so theirs may be deliberate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the four stores keeps or changes its finite_worker value, with the reason recorded in the task notes and next to the declaration
- [ ] #2 Each changed store has a test showing that a finite worker retires its cache
<!-- AC:END -->
