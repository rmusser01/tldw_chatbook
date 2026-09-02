---
id: TASK-18919
title: Build the Local and Server Collections capture reader
status: Done
assignee:
  - '@codex'
created_date: '2026-08-15 02:52'
updated_date: '2026-09-02 00:12'
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
- [x] #1 Collections is presented as the capture and reading feature area—not as an arbitrary folder, cross-source membership container, or Library item type—and capture-to-Media links remain optional provenance rather than identity.
- [x] #2 One capture-specific scope service selects either a dedicated local capture repository or the authenticated tldw_server Reading List API; switching authority replaces the dataset, identities are qualified by Local profile/database or Server profile/principal, local workspace changes do not repartition Server captures, and Local and Server records are never merged.
- [x] #3 The Collections destination uses the shared Library, Items, and Work reader topology: capture scopes and saved searches live in Library, exact bounded capture rows live in Items, and the selected capture's clean reader lives in Work; Library and Items are independently collapsible and reclaimed width expands Items toward its comfort cap before flowing to Work.
- [x] #4 Search, status, favorite, tag, domain, date, and supported sort scope are applied before deterministic 20-row paging; Local count and rows share one snapshot, Server browse requires the existing server operation to provide the same snapshot and advertise exact hasReadingSnapshotPagesV1=true through docs-info, the active scope has an exact coherent total, and tags or domains are not presented as complete facets without aggregate support.
- [x] #5 The Work reader renders trustworthy capture provenance and readable text or sanitized HTML through Read, Highlights, Notes, and Info modes, while separating the capture's freeform note from linked Notes records.
- [x] #6 Local Quick Capture durably commits before background extraction and preserves omitted state on retry; current Server Quick Capture truthfully waits for its authoritative synchronous response, never auto-retries an unknown outcome, warns that explicit retry may reapply defaults, and no confirmed save is reclassified as failed by a follow-up read.
- [x] #7 An approved ADR defines authority-qualified identity, additive schema-v3 local storage with future-version refusal and process-safe extraction leases, canonical-URL upsert, optimistic revisions, safe migration, cross-database references, transactionally quota-reserved and restart-reconciled private offline files, and a mandatory reachable coherent-snapshot JSON recovery export for untouched legacy generic Collections records.
- [x] #8 Server capabilities are unknown, supported, or unsupported per profile, principal, and advertised capability snapshot; only positively established status, favorite, tags, notes, highlights, summarize, listen, archive, offline-copy, delete, and recovery actions are enabled, unavailable actions carry explicit reasons, and destructive actions follow ADR-055.
- [x] #9 Loading, empty, extraction, interrupted, stale, conflict, Retry, detail/back, focus, collapse/restore, exact pure-resolver geometry, and measured-shell 160x50, 120x35, 100x30, and 80x24 walkthroughs match the shared adaptive Library reader conventions without horizontal overflow.
- [x] #10 Shared Local/Server contract tests, the server count/page concurrent-writer fix plus docs-info attestation, service and mounted Textual regressions, migration, managed-file and security tests, production-shaped cross-reader suites, and isolated Local plus enabled-Server live walkthroughs with more than 40 captures verify the complete design.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Land the independent `tldw_server` prerequisite that evaluates Reading List count, rows, and tag hydration under one database snapshot and advertises exact `hasReadingSnapshotPagesV1=true` through docs-info.
2. Add Chatbook's atomic schema-v2 capture tables plus schema-v3 extraction leases, future-version refusal, capture-owned contracts, exact Local paging, canonical-URL upsert, revisions, extraction state, saved searches, highlights, and linked-Note references beside untouched v1 tables.
3. Add transactionally quota-reserved, authority-rooted private offline files with two-phase publication, restart reconciliation, purge tombstones, and bounded resumable scavenging by composing existing private-path primitives.
4. Cut generic Collections over to `legacy_read_only` while keeping bounded v1 inspection and complete coherent-snapshot atomic JSON export reachable whenever compatible legacy rows exist.
5. Normalize the Local repository and authenticated Reading API through one capture-specific authority/scope service with exact Server paging attestation, tri-state capabilities, authority-qualified identity, no Local/Server merge, and no workspace partitioning of Server captures.
6. Build the generation-fenced Collections controller and mount contextual Library scopes, paged Items/Quick Capture, and permanent Read/Highlights/Notes/Info Work content in the existing adaptive reader shell.
7. Wire destination preferences and lifecycle, retire old generic-container inventories without redirecting them, run focused security/service/Textual tests, production-shaped cross-reader suites, and isolated 160x50/120x35/100x30/80x24 Local plus enabled-Server walkthroughs.

Detailed executable plan: `Docs/superpowers/plans/2026-08-31-library-collections-capture-reader.md`

ADR required: yes

ADR path: `backlog/decisions/107-collections-capture-authority-and-legacy-boundary.md`

Reason: TASK-18919 changes durable Collections storage, source authority, migration, service, and legacy-data boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced generic Collections with an authority-qualified capture domain backed by additive Local
schema-v3 storage and the authenticated Server Reading List API. The shared adaptive reader now
owns exact scopes and paging, Quick Capture, trustworthy Read/Highlights/Notes/Info content,
capability-gated actions, archive/Undo, private offline files, and complete legacy recovery without
conflating captures with Media or local workspaces.

The approved ADR records the durable authority, identity, storage, migration, and legacy boundary.
Local and enabled-Server production-shaped walkthroughs cover source replacement, exact pages,
all collapse postures, reclaimed Items width, F6 focus, all four required terminal sizes, confirmed
and controlled-unknown saves, archive/Undo, and return to Local. Integration testing additionally
corrected the real docs-info response contract, retained Quick Capture drafts across recomposition,
and found a tldw_server SQLite schema-memo defect now covered in merged prerequisite PR #2851. Detailed
evidence is in `Docs/superpowers/reviews/2026-08-31-library-collections-live-verification.md`.

ADR required: yes

ADR path: `backlog/decisions/107-collections-capture-authority-and-legacy-boundary.md`

Reason: TASK-18919 changes durable Collections storage, source authority, migration, service, and
legacy-data boundaries.

The cross-repository prerequisite landed in tldw_server PR #2851 at merge commit
`8140c679f3ea0334cea2dc1be32feb5b80e22ebe`. Final verification also hardened the
mounted action test to wait for the Notes mode to remount after highlight-save recomposition;
the focused client gate then passed 58 tests. PR verification subsequently exposed an ADR-097
first-paint census breach caused by eagerly importing six capture-runtime modules. Capture package
exports, controller annotations, and app wiring now load that runtime lazily after first paint or on
first use while preserving the public service seam. Import-closure regressions passed 17 tests, the
focused reader gate passed 44 tests with one enabled-Server skip, and the complete 23-test UI latency
and boot-budget workflow passed locally. TASK-18919 is complete.
<!-- SECTION:NOTES:END -->
