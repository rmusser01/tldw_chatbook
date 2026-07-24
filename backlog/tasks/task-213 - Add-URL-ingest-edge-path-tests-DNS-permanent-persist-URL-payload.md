---
id: TASK-213
title: 'Add URL-ingest edge-path tests (DNS-permanent, persist URL payload)'
status: Done
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
- [x] #1 A test asserts a socket.gaierror-caused httpx error yields PermanentIngestError (permanent),A test drives a URL payload through persist_parsed_media and asserts a media row with the canonical url and correct media_type,No filesystem access occurs for a URL payload
<!-- AC:END -->

## Implementation Notes

AC #1 (DNS→permanent) was already covered by `test_dns_failure_is_permanent` (shipped in #614). This task adds the missing URL-payload `persist_parsed_media` coverage test: `test_persist_url_payload_writes_article_row_no_filesystem` in `Tests/Local_Ingestion/test_ingest_parse_worker.py`. The test drives a URL payload with all required keys through `persist_parsed_media`, verifies the media row persists with correct canonical URL and `type='article'`, and confirms no filesystem access occurs (file_path is a URL string, never accessed).
