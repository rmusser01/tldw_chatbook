---
id: TASK-213
title: 'Add URL-ingest edge-path tests (DNS-permanent, persist URL payload)'
status: To Do
assignee: []
created_date: '2026-07-12 20:58'
labels:
  - library
  - ingest
  - test
  - follow-up
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The whole-branch review verified two seams by hand that lack automated coverage: (1) a DNS-resolution failure (httpx transport error whose __cause__ is socket.gaierror) maps to PermanentIngestError; (2) a URL payload (media_type=article, url=canonical, file_path=URL string) is accepted end-to-end by persist_parsed_media / add_media_with_keywords without any filesystem access. Lock both with tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test asserts a socket.gaierror-caused httpx error yields PermanentIngestError (permanent),A test drives a URL payload through persist_parsed_media and asserts a media row with the canonical url and correct media_type,No filesystem access occurs for a URL payload
<!-- AC:END -->
