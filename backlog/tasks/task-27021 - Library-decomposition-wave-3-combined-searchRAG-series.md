---
id: TASK-27021
title: 'Library decomposition wave 3: combined search+RAG series'
status: To Do
assignee: []
created_date: '2026-09-03 05:55'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave-2 Task 8's entanglement gate fired: 8/14 search methods (57.1%) call or are called by RAG methods; search submit is structurally the RAG entry point (_start_library_rag_query). Per the wave-2 plan's contingency, search and RAG extract as ONE combined series. Census in the wave-2 SDD task-8 report, copied to the recipe per-subsystem table.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Combined cluster enumerated with the recipe ownership script (search 14 + rag ~39 at 2026-09-02 snapshot)
- [ ] #2 Series follows the recipe (state, RED-commit wiring, controller(s), cleanup) with both guards green throughout
- [ ] #3 Recipe per-subsystem table updated with actual numbers
<!-- AC:END -->
