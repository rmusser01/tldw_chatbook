---
id: TASK-679
title: Rewrite ingest option and warning copy for end users
status: Done
assignee: []
created_date: '2026-07-26 03:26'
updated_date: '2026-07-26 04:25'
labels:
  - ingest
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The per-type options panel titles the collapsible with a raw dump of internal field names and values, the missing-tooling hints read as truncated sentences, and a path that cannot be found offers a Retry action when the only fix is to correct the path. A first-time user cannot tell what any of it means.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The options panel title describes the settings in plain language rather than internal field names
- [x] #2 Missing-tooling hints read as complete sentences and name the install command
- [x] #3 A not-found path offers a correction affordance rather than Retry
- [x] #4 Retry remains available for failures that are genuinely retryable
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Three copy defects. The collapsed options title was a dump of internal field names and repr'd values (analyze=False, chunk_size=500); it now reads 'PDF engine: pymupdf4llm, Enable OCR: off'. The scope line said 'all PDF documents files'. And missing-tooling hints were built by pasting a label in front of a hint that already contained it, giving 'PDF processing: PDF processing is unavailable: PDF ingestion.'

The two metadata fields behind that hint are used inconsistently across extras -- for PDF, label/what are 'PDF processing'/'PDF ingestion'; for audio they are effectively swapped -- so any template combining both reads wrong for one of them. _install_hint now returns the capability at stake as a plain noun phrase and build_warning_lines composes the sentence, dropping the 'needed for' clause when it would merely repeat the label.

Retry was also the wrong verb for the most common pre-flight error: re-running the same analysis on the same missing path fails identically. Path-shaped errors are now marked at the source (PreflightResult.path_invalid) rather than string-matched in the UI, and the canvas offers 'Choose a file…' for them while keeping Retry for genuinely retryable failures such as an unreachable URL.

Changed: tldw_chatbook/Library/ingest_capabilities.py, tldw_chatbook/Library/ingest_types.py, tldw_chatbook/Library/ingest_preflight.py, tldw_chatbook/Library/library_ingest_state.py, tldw_chatbook/Widgets/Library/library_ingest_canvas.py, tldw_chatbook/UI/Screens/library_screen.py, and their tests
<!-- SECTION:NOTES:END -->
