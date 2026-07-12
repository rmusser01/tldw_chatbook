---
id: TASK-211
title: Stream + early-abort the web article body-size guard
status: Done
assignee: []
created_date: '2026-07-12 20:58'
updated_date: '2026-07-12 22:47'
labels:
  - library
  - ingest
  - follow-up
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The web article extractor's 10 MB guard (web_article_ingestion.py) materializes the full response body via resp.text BEFORE checking its size, so a hostile/oversized response is fully downloaded before rejection — the guard cannot prevent the memory pressure it documents. The spec called for a streaming abort. The existing test mock already stubs iter_bytes, anticipating this design. Blast radius is bounded today (fetch runs in a spawned pool worker + 30s timeout), so this is a bounded hardening, not a live bug.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Extractor streams the response (httpx client.stream) and raises PermanentIngestError the moment the running byte total crosses the 10 MB cap, before the whole body is in memory,Content-type gate still rejects non-HTML before reading the body,A test asserts an oversized streamed body is rejected without fully materializing it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented in PR #614 (commit f6c58568): extract_article_for_ingest now uses httpx client.stream('GET', url) and accumulates iter_bytes() into a running total, raising PermanentIngestError the moment it crosses the 10 MB cap — the body is never fully buffered. test_oversized_streamed_body_is_permanent_before_full_buffer asserts the generator is drained only up to the abort (2 chunks, not 1000). Two reviewer bots (Gemini, Qodo) independently flagged the buffer-then-check original, confirming the design.
<!-- SECTION:NOTES:END -->
