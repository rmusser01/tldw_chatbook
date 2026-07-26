---
id: TASK-664
title: Fail ingest jobs that extract no content instead of reporting success
status: Done
assignee: []
created_date: '2026-07-26 03:26'
updated_date: '2026-07-26 04:01'
labels:
  - ingest
  - bug
  - p1
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A PDF ingested without usable extraction is recorded as done, counted in the library media total, and given a media row whose content is empty. The user has no signal that the import produced nothing, and the entry silently returns no results from search and RAG.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An ingest that extracts no content is reported as failed, not done
- [x] #2 The failure names the file and explains that no content could be extracted
- [x] #3 No empty-content media row is left behind in the library
- [x] #4 Files that legitimately contain no text are distinguished from extraction failures in the message
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing tests at the single write seam for empty extraction, empty source and URL payloads
2. Reject a contentless payload before it reaches the database
3. Distinguish an empty source from a failed extraction in the message
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
persist_parsed_media is the only place the ingest pipeline writes to media_db, so guarding there covers the queue runner, batch_ingest_files, quick_ingest and the server ingest path at once.

A payload whose content is blank now raises FileIngestionError before the write, so the queue marks the job failed instead of done and no empty media row is created. An empty source file is reported as 'X is empty; there was nothing to ingest', while a non-empty source that yielded nothing says the content could not be extracted and points at scanned images or missing optional tooling -- the two are different problems with different fixes. URL payloads are never stat'd, matching the existing no-filesystem contract for that path.

Observed during UAT as a PDF reported as done, counted in the library total, with zero bytes of content -- an entry that looks imported but silently returns nothing from search and RAG.

Changed: tldw_chatbook/Local_Ingestion/local_file_ingestion.py, Tests/Local_Ingestion/test_ingest_parse_worker.py
<!-- SECTION:NOTES:END -->
