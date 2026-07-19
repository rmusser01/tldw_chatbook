---
id: TASK-160
title: 'Ingest parallelism: heavy-lane cap for concurrent transcriptions'
status: Done
assignee: []
created_date: '2026-07-11 22:02'
updated_date: '2026-07-12 16:34'
labels:
  - follow-up
  - ingest
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
F3's parallel parse pool runs all media types through one pool (default min(3, cpu-1)). Two concurrent audio/video transcriptions can be RAM/CPU heavy. Add a dedicated heavy lane capping concurrent transcription parses to 1 while documents fan out wide. Revisit only if RAM pressure shows up in practice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Concurrent transcription parses are capped independently of document parses
- [x] #2 Config controls the heavy-lane cap
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Split across three tasks. Task 1 added detected_type to LibraryIngestJob/the registry's submit(), plus next_queued(skip_types=...) and parsing_count_for_types(...) for type-filtered queue selection. Task 2 added LibraryIngestQueueMixin._ingest_heavy_lane_max_workers(), reading library.ingest_heavy_lane_max_workers (default/invalid/non-positive -> 1) via the same dotted get_cli_setting form as _ingest_parse_worker_count(), plus documented the config key in the [library] template docs.

Task 3 (this pass) wired it together in app.py: submit_library_ingest_job now classifies detect_file_type() once at enqueue and stores it on the job; _top_up_ingest_parse_pool no longer recomputes the type at dispatch, and its claim uses the stored job.detected_type. The dispatch loop reads heavy_cap once per top-up pass, and on each iteration asks next_queued(skip_types=_INGEST_HEAVY_TYPES) whenever parsing_count_for_types(_INGEST_HEAVY_TYPES) >= heavy_cap, letting a queued document fill the slot instead of a second transcription. _INGEST_HEAVY_TYPES = frozenset({"audio", "video"}) is a module-level constant in app.py. Everything from mark_parsing onward (break guard, options build, pool ensure, apply_async) is unchanged -- pure dispatch selection.

Added test_heavy_lane_caps_transcriptions_while_documents_fill_pool to Tests/Library/test_library_ingest_runner.py (pool=3, heavy_lane=1, jobs [a1.mp3, a2.mp3, d1.txt, d2.txt, d3.txt] -> in-flight {a1, d1, d2}, a2 and d3 queued; completing a1 promotes a2), plus a heavy_lane override on _IngestRunnerHarness mirroring the existing worker_count override. Full Tests/Library + Tests/Local_Ingestion suites (117 tests) pass with no pre-existing test needing an expectation change -- other runner tests use single/light-type jobs that never exercise the heavy lane.
<!-- SECTION:NOTES:END -->
