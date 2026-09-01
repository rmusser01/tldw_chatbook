---
id: TASK-18919
title: Build the server-backed Collections capture reader
status: To Do
assignee: []
created_date: '2026-08-15 02:52'
updated_date: '2026-09-01 00:30'
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
  - backlog/decisions/067-library-top-level-pagination-contracts.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace Chatbook's stale generic-container interpretation of Collections with the authoritative tldw_server Pocket/Instapaper-style capture and reading domain, so users can save, find, read, annotate, and manage captured content without treating Collections as a Library item or conflating captures with Media.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Collections is presented as the tldw_server capture/reading feature area—not as an arbitrary local folder, cross-source membership container, or Library item type—and capture-to-Media links remain optional provenance rather than identity.
- [ ] #2 A bounded, validated Chatbook service seam consumes the authenticated tldw_server Reading List contracts for save, exact-total list/search/filter/sort, detail, update, archive/delete, and supported reader actions; unavailable capabilities and malformed responses fail closed with actionable recovery.
- [ ] #3 The Collections destination uses the shared Library/Items/Work reader topology: capture scopes and saved views live in Library, exact bounded capture rows live in Items, and the selected capture's clean reader lives in Work; Library and Items are independently collapsible and the remaining columns reclaim their space.
- [ ] #4 Reading List queries apply server-side search, status, favorite, tag, domain, date, and supported sort scope before pagination; page totals, ordering, selection, return focus, and concurrent shrink handling remain deterministic.
- [ ] #5 The Work reader renders trustworthy capture provenance and readable text or sanitized HTML, preserves the source URL, and exposes only server-supported status, favorite, tags, notes, highlights, summarize, listen, archive, and recovery actions with explicit disabled reasons.
- [ ] #6 Quick capture accepts a URL plus supported title, tags, and notes without leaving the reader; a committed save is reported truthfully even when the follow-up placement or detail read fails, with stale state and Retry replacing unsafe actions.
- [ ] #7 An approved ADR defines tldw_server as Collections authority and documents removal, migration, or compatibility treatment for the existing local library_collections and library_collection_items model; local generic-container records are never silently merged into server captures.
- [ ] #8 Loading, empty, blocked, stale, failure, Retry, detail/back, focus, collapse/restore, and 160x50, 120x35, 100x30, and 80x24 geometry match the established adaptive Library reader conventions without horizontal overflow.
- [ ] #9 Request generations, unmount fencing, malformed envelopes, late responses, concurrent shrink, mutation follow-up failure, and collapse/restore behavior have service/state and mounted Textual regression coverage plus isolated live verification against a production-shaped server fixture containing more than 40 captures.
<!-- AC:END -->
