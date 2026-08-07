---
id: TASK-3072
title: 'Watchlists reader-first re-IA, phase 2: list and reader quality'
status: In Progress
assignee: []
created_date: '2026-08-07 17:50'
updated_date: '2026-08-07 18:11'
labels:
  - watchlists
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement phase 2 of the reader-first design (Docs/superpowers/specs/2026-08-05-watchlists-reader-first-design.md, ADR-042): reader rows with snippet/relative-date/date-groups/unread-bold/star/ingested+queued markers, s key + Starred smart feed, Subscriptions/content_render.py, o open-in-browser, reading-pane action row via shared helpers, position footer, Subscriptions/item_dates.py date helper.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Article list rows show source+relative time, bold-when-unread title, 1-2 line snippet, unread dot, star/queued/ingested markers, and Today/Yesterday/locale-date group headers,s toggles star and a Starred smart feed with count appears in the rail; flags persist across re-fetch,Reading-pane action row (Star/Mark unread/Open in browser/Ingest/Queue) calls the same shared helpers as the inspector,o opens the item in the browser,Item body renders via content_render.py with stdlib fallback; hostile HTML renders safely as text and failure never blanks the pane,Position footer shows N of M within the displayed list plus a Next Unread control,Tests/Watchlists green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-08-07-watchlists-reader-first-phase-2-list-reader-quality.md

ADR required: no (already exists)
ADR path: backlog/decisions/042-watchlists-reader-first-ia.md
Reason: ADR-042 covers the re-IA; phase 2 is a direct implementation of it.
<!-- SECTION:PLAN:END -->
