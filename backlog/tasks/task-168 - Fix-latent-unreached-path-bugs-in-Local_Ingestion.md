---
id: TASK-168
title: Fix latent unreached-path bugs in Local_Ingestion
status: Done
assignee: []
created_date: '2026-07-11 22:03'
updated_date: '2026-07-11 23:52'
labels:
  - follow-up
  - ingest
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three pre-existing latent bugs on currently-unreached code paths, found during the F3 import-slimming audit: PDF_Processing_Lib calls undefined analyze_media_content (~:666); OCR_Backends DocextOCRBackend.cleanup references an unimported torch; process_pdf's result-dict check tests 'error' key membership instead of truthiness (the key is always present with a None default). Fix before any of these paths becomes reachable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 analyze_media_content reference resolved or removed
- [ ] #2 DocextOCRBackend.cleanup imports torch (or guards its absence)
- [ ] #3 process_pdf checks error truthiness not membership
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in the quick-wins batch (branch claude/followups-quickwins). See Docs/superpowers/plans/2026-07-11-followups-quickwins.md.
<!-- SECTION:NOTES:END -->
