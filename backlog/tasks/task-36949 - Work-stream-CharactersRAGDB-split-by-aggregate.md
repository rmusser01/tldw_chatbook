---
id: TASK-36949
title: 'Work stream: CharactersRAGDB split by aggregate'
status: Done
created_date: 2026-10-26 01:11
updated_date: '2026-11-20 14:55'
dependencies:
- TASK-36942.1
labels:
- architecture-review-2026-10-25
- db
- architecture
priority: medium
documentation:
- Docs/QA/architecture-review-2026-10-25/report.md
- Docs/QA/architecture-review-2026-10-25/verification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`CharactersRAGDB` is one 24,237-line class with 437 methods fusing schema, messages, learning, conversations, characters, keywords, notes, sync and kept artifacts. It also calls domain code from inside write paths (semantic revisions, Sync v2 hashing). Repositories that join its transaction already exist (canvas, note folders, world books, traces, citations); this stream continues that pattern. TASK-34906 owns the migrations cluster.

Source: architecture review 2026-10-25, `Docs/QA/architecture-review-2026-10-25/report.md` §F8; claim-by-claim verification in `Docs/QA/architecture-review-2026-10-25/verification.md` (origin/dev 7a5e2513c5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every subtask of this stream is Done
- [x] #2 ChaChaNotes_DB.py has a size ratchet row that this stream lowers
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Closed 2026-11-20 on dev `d85e90bc9e`. AC #1: all six subtasks (TASK-36949.1 to .6) are Done. AC #2: TASK-36942.1 (`2e0adfee71`) added `ChaChaNotes_DB.py`'s size-ratchet row at 25,693. TASK-36949.2 (`9feb6e202b`) lowered it from 25,870 to 23,519. It is 22,544 now. The guard files `test_layer_direction.py`, `test_module_size_ratchet.py`, `test_app_string_lookup_census.py`, `test_agent_runtime_import_footprint.py` and `test_screen_size_ratchet.py` passed serially on dev `d85e90bc9e` through the scratch wrapper: 314 passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
