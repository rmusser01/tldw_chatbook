---
id: TASK-33280
title: 'PERF-21: Library screen - targeted updates, ingest O(N^2), Folder Files poll'
status: To Do
created_date: 2026-09-28 18:03
dependencies:
- TASK-33265
- TASK-33268
labels:
- performance
- library
- ui
- perf-audit-2026-09
priority: medium
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library still has several whole-screen and quadratic costs:
- most destination switches whole-screen await recompose() (TASK-281)
- the ingest registry listener does O(queue) work, deep-copies the queue and remounts the whole queue panel per notification, so folder submits are O(N^2)
- the workspace-depth build issues about 170 admission-wrapped full-scan registry reads on the loop
- Folder Files re-walks the whole notes folder every 1.5 s with pathlib (175-500 ms at 5k files), even while hidden
- Search/RAG and Notes-editor keystrokes run whole-screen DOM scans
- Notes canvas sync_state always recomposes
- the conversation reader's progressive load is O(N^2)
- Collections run two whole-screen recomposes per interaction Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-21; every issue with file:line is listed under PERF-21 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Destination switches and canvas interactions update targeted canvases instead of recomposing the screen
- [ ] #2 Folder submit of N files is O(N) on the UI thread
- [ ] #3 Folder Files polling is change-detected, uses scandir and stops while hidden
- [ ] #4 Search/RAG and Notes-editor keystrokes do no whole-screen DOM scans
- [ ] #5 The workspace-depth build runs off the loop with batched registry reads
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
