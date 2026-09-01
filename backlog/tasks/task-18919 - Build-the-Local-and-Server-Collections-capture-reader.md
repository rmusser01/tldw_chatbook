---
id: TASK-18919
title: Build the Local and Server Collections capture reader
status: To Do
assignee: []
created_date: '2026-08-15 02:52'
updated_date: '2026-09-01 06:30'
labels:
  - library
  - collections
  - reading-list
  - captures
  - reader
  - pagination
  - server-parity
dependencies:
  - TASK-18912
  - TASK-18913
  - TASK-18914
  - TASK-18915
  - TASK-18916
references:
  - >-
    https://github.com/rmusser01/tldw_server/blob/dev/Docs/Product/Completed/Content_Collections_PRD.md
  - >-
    https://github.com/rmusser01/tldw_server/blob/dev/Docs/Product/Completed/Reading_List_PRD.md
  - >-
    https://github.com/rmusser01/tldw_server/blob/dev/Docs/API-related/Reading_List_API.md
  - >-
    Docs/superpowers/specs/2026-08-14-library-top-level-source-pagination-design.md
  - >-
    Docs/superpowers/specs/2026-08-24-library-destinations-adaptive-reader-design.md
  - >-
    Docs/superpowers/specs/2026-08-31-library-collections-capture-reader-design.md
  - backlog/decisions/067-library-top-level-pagination-contracts.md
  - backlog/decisions/107-collections-capture-authority-and-legacy-boundary.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace Chatbook's stale generic-container interpretation of Collections with the authoritative Pocket/Instapaper-style capture and reading domain, so users can save, find, read, annotate, and manage captures under one explicitly selected Local or Server authority without treating Collections as a Library item or conflating captures with Media.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Collections is presented as the capture and reading feature area—not as an arbitrary folder, cross-source membership container, or Library item type—and capture-to-Media links remain optional provenance rather than identity.
- [ ] #2 One capture-specific scope service selects either a dedicated local capture repository or the authenticated tldw_server Reading List API; switching authority replaces the dataset, identities are qualified by Local profile/database or Server profile/principal, local workspace changes do not repartition Server captures, and Local and Server records are never merged.
- [ ] #3 The Collections destination uses the shared Library, Items, and Work reader topology: capture scopes and saved searches live in Library, exact bounded capture rows live in Items, and the selected capture's clean reader lives in Work; Library and Items are independently collapsible and reclaimed width expands Items toward its comfort cap before flowing to Work.
- [ ] #4 Search, status, favorite, tag, domain, date, and supported sort scope are applied before deterministic 20-row paging; Local count and rows share one snapshot, Server browse requires the existing server operation to provide the same snapshot and advertise exact hasReadingSnapshotPagesV1=true through docs-info, the active scope has an exact coherent total, and tags or domains are not presented as complete facets without aggregate support.
- [ ] #5 The Work reader renders trustworthy capture provenance and readable text or sanitized HTML through Read, Highlights, Notes, and Info modes, while separating the capture's freeform note from linked Notes records.
- [ ] #6 Local Quick Capture durably commits before background extraction and preserves omitted state on retry; current Server Quick Capture truthfully waits for its authoritative synchronous response, never auto-retries an unknown outcome, warns that explicit retry may reapply defaults, and no confirmed save is reclassified as failed by a follow-up read.
- [ ] #7 An approved ADR defines authority-qualified identity, additive schema-v2 local storage with future-version refusal, canonical-URL upsert, optimistic revisions, safe migration, cross-database references, transactionally quota-reserved and restart-reconciled private offline files, and a mandatory reachable coherent-snapshot JSON recovery export for untouched legacy generic Collections records.
- [ ] #8 Server capabilities are unknown, supported, or unsupported per profile, principal, and advertised capability snapshot; only positively established status, favorite, tags, notes, highlights, summarize, listen, archive, offline-copy, delete, and recovery actions are enabled, unavailable actions carry explicit reasons, and destructive actions follow ADR-055.
- [ ] #9 Loading, empty, extraction, interrupted, stale, conflict, Retry, detail/back, focus, collapse/restore, exact pure-resolver geometry, and measured-shell 160x50, 120x35, 100x30, and 80x24 walkthroughs match the shared adaptive Library reader conventions without horizontal overflow.
- [ ] #10 Shared Local/Server contract tests, the server count/page concurrent-writer fix plus docs-info attestation, service and mounted Textual regressions, migration, managed-file and security tests, production-shaped cross-reader suites, and isolated Local plus enabled-Server live walkthroughs with more than 40 captures verify the complete design.
<!-- AC:END -->
