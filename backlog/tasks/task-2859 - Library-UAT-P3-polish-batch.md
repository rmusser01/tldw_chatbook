---
id: TASK-2859
title: Library UAT P3 polish batch
status: To Do
assignee: []
created_date: '2026-08-07 01:10'
labels:
  - library
  - polish
  - uat-2026-08-06
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library UAT 2026-08-06 P3 findings, critique snapshot
`.impeccable/critique/2026-08-07T01-01-42Z__tldw-chatbook-ui-screens-library-screen-py.md`,
observed at dev `6ffa56516`. One polish pass; none block tasks individually.

1. Conversations canvas has no title header (top row is "Export… / Select"; siblings show
   "Name (n)"); its filter input renders below the empty-state text instead of above.
2. Rail gloss "Prompts (0) — AI asks" is cryptic; a freshly created prompt is stamped "legacy".
3. Export quality caption describes the non-selected option ("quality: thumbnail ▸" captioned
   with the "original copies full media files…" explanation).
4. Ingest queue summary "3 done — in queue" self-contradicts (done vs in queue).
5. Details disclosure: clicking the "Details" label does nothing (only the ▸ chip toggles);
   content wraps mid-unit ("Prompts 144.0 / KB"); "Status" renders as a bare heading with no
   value; DB sizes exclude -wal files (reported 4.0 KB while the WAL held ~4 MB).
6. File/folder pickers are dev-flavored: raw sizes ("30624"), second-precision timestamps,
   $HOME default with no recent/suggested locations.
7. Canvas title grammar drifts: "Media (3)" vs "Library Collections"/"Library Search/RAG"; rail
   says "Search / RAG", canvas says "Search/RAG".
8. sort: defaults differ across siblings (Notes/Prompts "Newest", Skills "Name") with no system.
9. Skill editor copy "Not applied in v1 — shown for SKILL.md round-tripping only" is
   internal-version talk.
10. Search results lack a "N results for 'query'" headline; media evidence snippet text sits
    flush against the card border (missing left pad).
11. The toast panel overlaps the ingest queue area.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each numbered item is fixed, or declined with a one-line reason in the notes
- [ ] #2 Copy changes keep the DESIGN.md voice (plain language, labels carry meaning)
- [ ] #3 Touched surfaces are re-verified live at 170×50
<!-- AC:END -->
