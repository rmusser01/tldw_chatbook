---
id: TASK-672
title: Stop scaring new users with RAG indexing failures on every ingest
status: To Do
assignee: []
created_date: '2026-07-26 04:05'
labels:
  - ingest
  - ux
  - rag
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On a fresh install with no embedding model downloaded, every successful ingest raises a red 'RAG indexing failed ... All chunks failed embedding generation' notification. The import itself worked, so the first thing a new user sees after their first successful action is a failure they did not cause and cannot act on. Observed during the ingest UAT once folder ingest started completing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A fresh install with no embedding model does not report an ingest as failing when only indexing was skipped
- [ ] #2 The user is told what indexing gives them and how to enable it, rather than shown a raw failure
- [ ] #3 Genuine indexing failures on a configured install are still surfaced
<!-- AC:END -->
