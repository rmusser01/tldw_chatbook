---
id: TASK-34652
title: Recover exact Notes drafts with language across restart
status: Done
created_date: 2026-10-07 03:08
dependencies:
- TASK-34571
- TASK-34651
assignee:
- '@codex'
labels:
- notes
- structured-editing
documentation:
- /Users/macbook-dev/.codex/worktrees/structured-document-editing/tldw_chatbook/Docs/superpowers/plans/2026-09-27-structured-document-editing.md
updated_date: 2026-10-07 08:33
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep unsaved edits recoverable across navigation, rejected saves and process restart.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Body and source language share one canonical draft revision, including language-only mutation, conflicts and explicit null reset.
- [x] #2 An empty rejected draft recovers exactly after navigation and restart without claiming a successful save.
- [x] #3 Recovery is scoped to authority/workspace/note and removed only after its matching complete save; older acknowledgements preserve newer edits.
- [x] #4 Storage/quota failures prevent discard-producing navigation and show a truthful error; existing storage ownership is reused without diagnostics persistence.
- [x] #5 Targeted recovery/session tests cover profile isolation, logout, restart and actual copy/history owners touched by this change.
- [x] #6 Relevant governing ADR is assessed/linked and repo-local completion notes name actual evidence and remaining limitations.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/194-structured-note-language-and-local-editor-validation.md (C); Docs/ADR/070-portable-note-language-and-negotiated-structured-editing.md (S), existing programme decisions
Reason: implement approved scoped durable-draft recovery and serialized-language ownership; extend the existing decision before storage changes, without a second authority/store.
1. Read task5 brief and actual repo-local task/instructions/lessons/ADRs; trace canonical state, real note_session_port and LibraryScreen wiring, device-state migration/connection owner, and browser offline queue.
2. Add meaningful failing language-only, exact empty-draft restart, older-ack/newer-revision, scope/logout and storage-failure controls before production changes.
3. Carry source/language together through captured draft mutation, save and conflict reload/overwrite; distinguish omission from explicit null and preserve unknown future values.
4. Reuse the existing device-state DB with a dedicated scoped draft table and exact matching-revision cleanup; integrate actual session/navigation owner so storage failure cannot discard a draft. Qualify old schema upgrade and relevant existing consumers; no canonical Notes migration or diagnostics persistence.
5. Extend the existing browser offline queue version/snapshot with language/revision, legacy unknown-mode gating and quota/authority controls. No second storage mechanism or editor adoption.
6. Run named affected C state/session/recovery/device-store and real-port controls plus U structured-draft/authority/offline selections and whole W typecheck; scoped static/security/format comparisons against captured bases, no full sweep.
7. Document actual copy/history ownership, evidence and limitations, self-review and commit explicit C/S unit files separately. Keep In Progress pending independent review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented coherent source/language drafts and exact private recovery through the existing runtime-owned device-state store (schema3). Captured local database/principal/global scope, matching-revision acknowledgement cleanup and navigation fences preserve newer or rejected edits. Deferred save/conflict awaits revalidate session and revision; actual Library port/wiring, duplication and storage lifecycle are covered. Governing ADR194 was amended before storage changes. Evidence: broader affected310 and real-port8, then final amended session57 after formatting; no introduced Ruff/Bandit findings and changed Python regions formatted. Initial task review and two scoped rounds are clean. No external live-server/OS-kill or downstream formatting UI claim; those remain later qualification gates.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
F5 complete after independent task review and two scoped fixes. Exact local drafts survive navigation/reopen and late acknowledgements retain newer edits. Final amended session57 passed; broader owner and real-port evidence is recorded in task-5-report.md. ADR194 applies; live integration/OS-kill and formatter adoption remain later units.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
