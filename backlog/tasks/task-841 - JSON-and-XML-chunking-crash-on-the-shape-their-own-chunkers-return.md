---
id: TASK-841
title: JSON and XML chunking crash on the shape their own chunkers return
status: Done
assignee: []
created_date: '2026-07-27 02:02'
updated_date: '2026-07-27 02:02'
labels:
  - chunking
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every request to chunk content by the JSON or XML method fails with a type error. The chunking service post-processes results assuming each chunk is a plain string, but the structure-aware chunkers return records carrying their text alongside metadata, so a string operation is attempted on a record. The text-oriented methods are unaffected because they do return strings. Found by exercising each supported method in turn rather than by reading the dispatch, which looks uniform.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chunking by JSON returns usable text chunks
- [x] #2 Chunking by XML returns usable text chunks
- [x] #3 A structured chunk carrying no text field is preserved rather than dropped
- [x] #4 The text-oriented methods are unchanged
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed. Every supported chunking method now returns records whose text is a string.

Chunker.chunk_text is not uniform: words, sentences, paragraphs, tokens and semantic return plain strings, while json, xml and ebook_chapters return records carrying their text alongside metadata. ChunkingService post-processed results assuming strings and called .split() on them, so every json and xml request died with "'dict' object has no attribute 'split'".

A _chunk_to_text normaliser now handles both shapes. A structured record with no text field is serialised rather than dropped, so its content still reaches the index.

Nearly misdiagnosed as two separate issues. JSON first failed with 'Chunkable key data not found', which reads as a deliberate constraint. Retesting with a data key produced the same .split() error as XML, showing one bug rather than two -- the first message was the JSON chunker's own input validation firing before the real defect could.

Third instance in one session of a caller assuming a shape its callee does not produce, after the remote-ingest pagination defect and the audio chunking call in task-868. All three are the same lesson, now recorded in backlog/docs/lessons-testing-evidence.md.

Verified by exercising all seven methods through the real service; regression test parametrises over every one. Tests/RAG + Tests/Local_Ingestion: 792 passed, 9 skipped.

Files: RAG_Search/chunking_service.py, Tests/RAG/test_chunking_service.py.
<!-- SECTION:NOTES:END -->
