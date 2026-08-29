---
id: TASK-24308
title: Extend Notes tools with portable organization transactions
status: To Do
assignee: []
created_date: '2026-08-29'
labels:
  - notes
  - agents
  - tools
priority: high
dependencies:
  - TASK-24307
documentation:
  - Docs/superpowers/specs/2026-08-29-agent-lessons-notes-organization-sync-design.md
  - Docs/superpowers/plans/2026-08-29-notes-organization-agent-tool-transactions.md
  - backlog/decisions/102-portable-notes-organization-and-agent-lessons.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give permitted agents exact folder and keyword discovery plus conflict-safe, additive organization-aware note saves, including durable local pending states when portable organization cannot yet publish.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 `library_search_notes` supports spelling-exact keyword and unambiguous exact folder filters while retaining bounded lexical search and pagination
- [ ] #2 Search and get responses expose bounded folder and keyword metadata plus stable public identities and the current `organization_version`
- [ ] #3 `library_save_note` can add requested keywords without removing user keywords and rejects stale note or organization state without overwriting concurrent user changes
- [ ] #4 Note content, requested organization, and immutable local synchronization intents commit or roll back together inside the owning Notes database
- [ ] #5 A permitted lesson save made before organization readiness remains locally discoverable and excluded from every normal dispatcher until atomic finalization
- [ ] #6 Folder-only collisions survive restart as non-blocking placement review, while deletion or permission denial leaves no orphaned receipt or hidden write
- [ ] #7 Targeted tool-parity, transaction-failure, restart, permission, pagination, and concurrency tests pass across Console and in-app MCP surfaces
<!-- AC:END -->
