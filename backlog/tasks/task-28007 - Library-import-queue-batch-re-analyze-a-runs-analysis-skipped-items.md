---
id: TASK-28007
title: Library import queue - batch re-analyze a run's analysis-skipped items
status: Done
assignee:
  - '@claude'
created_date: '2026-09-02 04:10'
updated_date: '2026-09-05 03:19'
labels:
  - library
  - media-ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When Analyze-after-import is on but the provider is unready, every imported row records analysis skipped with a reason (honest receipts) but there is no batch remediation - Retry re-imports rather than re-analyzing, so a 40-item run means 40 manual regenerations or a full re-import. Add a run-level Analyze-these-N action on completed runs that contain analysis-skipped rows. Builds on the viewer Generate seam (same provider resolution and persistence).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A completed import run with analysis-skipped items offers a single action that generates analyses for all of them
- [x] #2 Per-item success and failure are reported in the same receipt style as import rows
- [x] #3 Items that already have an analysis are not overwritten without an explicit choice
- [x] #4 Select mode offers a bulk Analyze action that runs the per-item generator over the selection in one worker group with an in-list receipt (Analyzing N of M · K failed · Retry) — critique #4 P1
- [x] #5 Generate in the Reader is disabled with the resolver's reason in its label when no provider is configured, instead of a post-click toast
- [x] #6 The collapsed Import behavior header summarises its analysis state (e.g. Import behavior · analysis off)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Task 1: pure analysis_unavailable_reason(resolution) + screen helper _library_media_analysis_provider_reason() (empty = ready), viewer arg analysis_provider_reason rendered as ○ Generate/○ Regenerate + tooltip (mirrors PR A Find grammar; compare+assign in _sync_library_media_viewer_state), computed only on the Analysis tab; Import-behavior header title from the Analyze-after-import toggle.
2. Task 2: LibraryMediaCanvasState analyze_receipt_* fields; Analyze on its own select-mode row (bulk row is 33/36 cells); ONE worker group library_media_analyze_selected over the selection in browse order; already-analysed partition arms Skip them / Overwrite; receipt Analyzing N of M · K failed → ✓/✗ analyzed · … with Retry failed / Dismiss; per-item generator gains viewer_owned=False; unmount cancels the screen-owned worker with an honest progress notify.
3. Task 3: import-run summary action Analyze N skipped gated on skipped rows AND a ready provider, reusing _start_library_media_analyze(ids, overwrite=False) with a per-item outcome hook; per-row ✓ analyzed / ✗ analysis failed · reason; Import-origin runs auto-skip analysed ids and never arm the Media card; origin-aware unmount notice.
4. SDD: task review + re-review per task, final whole-branch review + one fix round (choice row fits the 36-cell pane, Import runs paint no Media receipt, docs cover all three surfaces, en-US copy).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped a set-level analysis path for Library ▸ Media on branch fix/media-wave4-d (8 task commits + 2 final-fix commits, merged with dev). Reader: Generate/Regenerate render ○ + tooltip reason when no provider is ready (AC#5), resolved only on the Analysis tab because the Anthropic subscription readiness check reads a credentials file and shells out. Import: collapsed header reads 'Import behavior · analysis on|off' (AC#6). Select mode: bulk Analyze on its own row, one exclusive worker group, in-list receipt in PR A's two-row grammar with Retry failed/Dismiss (AC#4); analysed items need an explicit Skip them/Overwrite choice (AC#3). Import run summary: 'Analyze N skipped' runs exactly the skipped ids through the same seam and reports per row in the import receipt style (AC#1/#2). Trade-offs: the worker is screen-owned and cancelled on leaving Library (honest unmount notify with an origin-aware resume hint; app-owned worker filed as follow-up); Analyze could not share the Clear/Export/Review row (33 of 36 cells); the Import summary resolves the provider once per render (TTL-bounded). Verification: per-file suites green vs base in separate processes, preflight green, live tmux checks of the disabled states against the real no-provider config; receipt/choice/Import flows verified in real-screen app-tests with stubbed resolver+generator because the host hit POSIX semaphore exhaustion (mp.Lock → OSError 28) from ~40 concurrent pytest processes of other sessions. Files: tldw_chatbook/Library/ingest_analysis.py, ingest_capabilities.py, library_media_state.py, library_media_viewer_state.py; tldw_chatbook/UI/Screens/library_screen.py; tldw_chatbook/Widgets/Library/library_media_viewer.py, library_media_canvas.py, library_ingest_canvas.py; Docs/User_Guide/library/*.md; tests under Tests/Library and Tests/UI (test_library_ingest_analyze_skipped.py new).
<!-- SECTION:NOTES:END -->
