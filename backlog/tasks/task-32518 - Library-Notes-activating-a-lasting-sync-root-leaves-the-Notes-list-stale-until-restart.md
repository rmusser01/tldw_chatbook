---
id: TASK-32518
title: "Library Notes: activating a lasting-sync root leaves the Notes list stale until restart"
status: To Do
assignee: []
created_date: '2026-09-13 00:30'
labels:
  - library
  - notes
  - rider
dependencies:
  - TASK-32269
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On a profile that already holds notes, **Activate reviewed root** writes the
60 synced notes and their managed folder to the database ("Sync root
activated. 60 applied · durable receipt recorded" — the row count in
`notes` goes 69 → 129 and `note_folders` gains the display-name folder), but
the Notes list beside it does not pick any of it up: the count stays
"Notes (69)", no **⇄ Sync managed** folder row appears, and neither a manual
**Check changes**, re-selecting the rail's Notes row, nor a Folder files →
Library notes round trip refreshes it. Restarting the app shows
"Notes (129)" and the "t12 sync ⇄ Sync managed" folder at once. The wave-3
sync walk (task-32269) saw the folder appear because it ran on a fresh
profile, where the empty state recomposes; the seeded case was never walked.

Evidence: wave-3 docs sweep on dev 7159fc0b99, 2026-09-12 —
`wave3-caps/docs-sweep/48-root-activated-235x52.txt` (activation receipt),
`49-list-after-activate`, `52-list-after-check`, `53-list-after-reload`
(all still 69, no folder), `54-list-after-restart` (129, folder present).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After Activate reviewed root, returning to the Notes list shows the managed folder row and a count that includes the synced notes, without restarting the app
- [ ] #2 A manual Check changes that applies changes refreshes the list the same way
- [ ] #3 A test on a seeded profile (existing notes and folders) pins the refresh; the fresh-profile path keeps working
<!-- AC:END -->
