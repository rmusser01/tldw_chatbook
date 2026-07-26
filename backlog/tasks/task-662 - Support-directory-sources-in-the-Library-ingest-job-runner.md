---
id: TASK-662
title: Support directory sources in the Library ingest job runner
status: Done
assignee: []
created_date: '2026-07-26 03:26'
updated_date: '2026-07-26 03:44'
labels:
  - ingest
  - bug
  - p1
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pre-flight tells the user a folder contains N ingestible files and lets them start the job, but the runner then fails the whole thing because it treats the directory as a single file. Batch ingestion from a folder is advertised throughout the UI and does not work at all.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Submitting a folder ingests each supported file it contains
- [x] #2 The queue reflects per-file outcomes rather than one failure for the folder
- [x] #3 Unsupported files inside the folder fail individually without failing the supported ones
- [x] #4 The directory scan honours the configured scan limit
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing runner tests for a folder, a mixed folder and the scan limit
2. Expose the pre-flight directory walk as a public seam
3. Expand a directory into per-file jobs at the submit seam
4. Live-verify the folder that failed during UAT
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Pre-flight already walked directories and told the user how many files it found, but the submit seam created a single job for the directory path, which the parser then rejected with 'Unsupported file type: .'. Batch import was advertised everywhere and worked nowhere.

submit_library_ingest_job now expands a directory into one job per contained file, reusing the same walk pre-flight uses (via a new collect_directory_files seam) so the promised count and the queued count cannot drift. Each file gets its own queue row, outcome and retry, so one unsupported file no longer fails its siblings. An empty folder fails once with 'No files to ingest were found in this folder.' rather than the parser's confusing extension error, and the scan limit is honoured and logged when it truncates.

Title is deliberately not carried onto expanded files -- it is per-file metadata (the form already clears it on submit for that reason), so each file takes its filename-derived title while author and keywords carry across as batch metadata.

Live-verified on the folder that failed during UAT: 4 files, 4 done rows, 4 media rows with real content and distinct titles.

Changed: tldw_chatbook/app.py, tldw_chatbook/Library/ingest_preflight.py, Tests/Library/test_library_ingest_runner.py
<!-- SECTION:NOTES:END -->
