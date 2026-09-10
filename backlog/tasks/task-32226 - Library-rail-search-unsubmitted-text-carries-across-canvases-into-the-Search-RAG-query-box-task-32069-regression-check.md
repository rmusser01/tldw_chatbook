---
id: TASK-32226
title: >-
  Library rail search: unsubmitted text carries across canvases into the
  Search/RAG query box (task-32069 regression check)
status: To Do
assignee: []
created_date: '2026-09-10 14:55'
labels:
  - library
  - rail
  - regression
  - critique-9
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task-32069 claimed the rail box is emptied on canvas change; assessor B saw unsubmitted text persist across canvases and land in the RAG query box. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 25.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Re-verify; if reproducible, the rail box is emptied on canvas change and never seeds the RAG query box
<!-- AC:END -->
