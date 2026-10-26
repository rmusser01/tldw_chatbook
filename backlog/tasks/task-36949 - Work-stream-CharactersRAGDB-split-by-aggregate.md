---
id: TASK-36949
title: 'Work stream: CharactersRAGDB split by aggregate'
status: To Do
created_date: 2026-10-26 01:11
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
- [ ] #1 Every subtask of this stream is Done
- [ ] #2 ChaChaNotes_DB.py has a size ratchet row that this stream lowers
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
