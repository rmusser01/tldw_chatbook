---
id: TASK-3308
title: >-
  Decide and wire XML ingestion reachability — backend exists, parse path raises
  "not yet implemented"
status: Done
assignee: []
created_date: '2026-08-07 19:30'
updated_date: '2026-08-09 03:58'
labels:
  - library
  - ingest
  - parity
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the 2026-08-07 parity audit: `Local_Ingestion/XML_Ingestion.py` exists, but `.xml` is not in `detect_file_type` and `parse_local_file_for_ingest` raises "XML file processing is not yet implemented". Either wire XML through (extension → group → parse) or record the deferral and keep the unsupported classification honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An owner decision (ship or defer) is recorded; if ship, `.xml` ingests end-to-end; if defer, the raise is unreachable from the queue (preflight classifies `.xml` unsupported with honest copy)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record owner ruling (defer, honest copy) from task-3310.
2. Verify .xml unmapped in detect_file_type -> get_type_group returns UNSUPPORTED_GROUP.
3. Pin with tests: .xml lands in unsupported bucket, will-skip line renders, Start gating matches other unsupported types, queue path never reaches the XML raise.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Owner ruling (recorded in task-3310's notes, applied here): DEFER XML ingestion; keep the unsupported classification honest. Verified the deferral is already structurally sound and pinned it so it cannot regress silently:

- .xml is unmapped in detect_file_type -> get_type_group returns 'unsupported' (pinned: Tests/Library/test_ingest_capabilities.py::test_get_type_group_xml_is_unsupported_task_3308, case-insensitive).
- Pre-flight buckets an .xml file into the unsupported group, never an error (Tests/Library/test_ingest_preflight.py::test_xml_file_lands_in_the_unsupported_bucket_task_3308).
- State: an .xml-only staging closes the Start gate ('Nothing in this selection can be imported -- 1 unsupported file.') and names the file; in a mixed staging the will-skip line renders ('1 unsupported file will be skipped: feed.xml.') and the commit forecast counts '1 will skip' (Tests/Library/test_library_ingest_state.py, two task-3308 tests).
- The 'XML file processing is not yet implemented' raise in parse_local_file_for_ingest is unreachable from the queue: detect_file_type (the only producer of file_type for local files) refuses .xml first with the honest 'Unsupported file type' error, and URLs classify only article/audio/video. Pinned: Tests/Local_Ingestion/test_ingest_parse_worker.py::test_parse_xml_raises_unsupported_not_not_yet_implemented (asserts the placeholder text never surfaces).

If XML is later wired through, the capability pin goes red on purpose -- retire the pins together with the deferral. No product code changes were needed; Docs/User_Guide/library/import-and-export.md stamp updated.
<!-- SECTION:NOTES:END -->
