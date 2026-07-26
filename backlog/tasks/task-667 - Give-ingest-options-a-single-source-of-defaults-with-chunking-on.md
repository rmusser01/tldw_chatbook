---
id: TASK-667
title: Give ingest options a single source of defaults with chunking on
status: To Do
assignee: []
created_date: '2026-07-26 03:26'
labels:
  - ingest
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ingest option defaults are declared in two places that disagree, and the two ingest surfaces ship opposite chunking defaults. Chunking is off in the Library canvas, so imported documents are not chunked for retrieval, which quietly undermines search and RAG for anyone who never opens the advanced panel.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Option defaults come from one declaration used by every ingest surface
- [ ] #2 Chunking defaults to on with a chunk size of 1000
- [ ] #3 Analysis defaults to off so ingest does not make LLM calls unless asked
- [ ] #4 A user who previously saved option values keeps them
<!-- AC:END -->
