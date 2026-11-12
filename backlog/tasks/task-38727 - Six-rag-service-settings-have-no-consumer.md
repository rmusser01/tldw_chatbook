---
id: TASK-38727
title: "Six [rag.service] settings have no consumer"
status: To Do
created_date: 2026-11-04 02:52
labels:
- follow-up
- architecture-review-2026-10-25
- rag
priority: low
references:
- tldw_chatbook/RAG_Search/simplified/rag_service.py
- tldw_chatbook/RAG_Search/README_enhanced_services.md
dependencies:
- TASK-36945.3
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36945.3 made rag_service read its `[rag.service]` table when used. Before that, the read used a key that never exists, so the whole table was dead. Six of the table's 20 entries are still read by nothing: CACHE_TIMEOUT_SECONDS, EMBEDDING_IDLE_TIMEOUT, MIN_SYSTEM_MEMORY_MB, MEMORY_PRESSURE_REDUCTION, EMBEDDING_PROGRESS_INTERVAL and DEFAULT_BATCH_SIZE. A user who sets them gets no effect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the six entries is either read by the code it is meant to tune, with a test showing the setting takes effect, or removed from the table and any docs that mention it
- [ ] #2 The rag_service tests show no new failures against dev
<!-- AC:END -->
