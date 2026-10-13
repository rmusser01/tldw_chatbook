---
id: TASK-34818
title: Enable durable server-readable Notes sync in Chatbook
status: To Do
created_date: 2026-10-12 14:47
dependencies:
- TASK-1602
labels:
- notes
- sync
- structured-editing
priority: high
references:
- backlog/tasks/task-1602 - Define-production-local-apply-store-ownership-for-manual-Sync-v2.md
- backlog/tasks/task-34658 - Qualify-structured-Notes-across-editors-sync-files-and-recovery.md
documentation:
- backlog/decisions/194-structured-note-language-and-local-editor-validation.md
- backlog/decisions/036-application-service-composition-lifecycle.md
- Docs/superpowers/reviews/2026-09-27-structured-document-release-evidence.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow users to synchronize local Chatbook Notes with tldw_server and receive server-side edits without losing exact source or language metadata. Structured-editor qualification found that the shipped app has no durable receiving-store or clear Notes mirror binding, so the existing workflow refuses rather than completing a round trip. This is a separate Notes-only follow-up to the linter/formatter work; TASK-1602 owns the prerequisite production data-ownership design.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The shipped Chatbook application can publish a local YAML, JSON or JSONL note to tldw_server, receive an edit made through the real server web Notes editor, and retain the exact source and content_language after reopening Chatbook and its durable local database.
- [ ] #2 Production Notes sync uses the canonical durable Notes owner and correctly scoped mirror, cursor and pending state; no in-memory or test-only receiving store is used as application wiring.
- [ ] #3 Negotiated complete snapshots preserve language-only updates, omitted versus explicit-null language, supported future language values and exact source; conflicts, failed or incomplete acknowledgements do not clear pending intent or advance accepted state.
- [ ] #4 Server, principal and workspace changes cannot apply another authority's note, mirror state or acknowledgement; conflicts and deletion or tombstone handling follow the approved canonical ownership rules.
- [ ] #5 Unavailable prerequisites, unsupported peers and enrollment refused for missing verified at-rest encryption produce actionable unavailable or pending status while retaining local drafts and pending metadata; no security attestation is fabricated and no false synchronized or empty-pending result is shown.
- [ ] #6 Targeted real-application and authenticated API qualification records actual versions, commands and a same-note publish, web-edit, pull and restart trace using owned real stores and legitimately verified deployment security; applicable canonical ADRs, user guidance and static checks are recorded before closure.
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
