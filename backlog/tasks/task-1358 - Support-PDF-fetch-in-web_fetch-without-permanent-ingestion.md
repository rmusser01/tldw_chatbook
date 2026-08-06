---
id: TASK-1358
title: Support PDF fetch in web_fetch without permanent ingestion
status: To Do
assignee: []
created_date: '2026-08-05 06:04'
labels:
  - web-tools
dependencies:
  - TASK-1354
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
web_fetch v1 rejects PDFs and routes users to media ingestion. Users need one-off PDF reads (papers, manuals) that do not write to the media DB.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 PDFs fetched via egress-guarded path and size-capped,Text extracted ephemerally (no media DB writes),Result truncated like HTML; tests with fixture PDFs
<!-- AC:END -->
