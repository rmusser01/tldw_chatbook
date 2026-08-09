---
id: TASK-3307
title: >-
  Decide and wire image ingestion reachability — backend exists, unreachable from the Library surface
status: To Do
assignee: []
created_date: '2026-08-07 19:30'
labels:
  - library
  - ingest
  - parity
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the 2026-08-07 parity audit: `Local_Ingestion/Image_Processing_Lib.py` implements `process_image` (OCR backend/language, visual features, analysis) but no image extension is mapped in `detect_file_type`, no caller exists in `local_file_ingestion.py` or `Library/`, and the ingest surface's supported list omits images. Either wire the media type (extension mapping → type group → panel → `_ingest_job_options` branch → processor) or record the decision that images stay unshipped and ensure preflight names them as unsupported honestly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 An owner decision (ship or defer) is recorded; if ship, image files ingest end-to-end from the Library surface with their options panel
- [ ] #2 If deferred, image files are classified as unsupported with honest copy (not silently generic)
<!-- AC:END -->
