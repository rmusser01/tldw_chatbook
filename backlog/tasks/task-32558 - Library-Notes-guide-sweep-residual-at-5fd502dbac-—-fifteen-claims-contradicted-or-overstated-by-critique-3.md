---
id: TASK-32558
title: >-
  Library Notes: guide sweep residual at 5fd502dbac — fifteen claims
  contradicted or overstated by critique #3
status: To Do
assignee: []
created_date: '2026-09-13 06:48'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), both assessors' docs tables merged (B: 52 VERIFIED, 7 CONTRADICTED, 10 PARTIAL; A §10). Same shape as task-32271. Each row either gets corrected copy or a citation to the task that fixes the behaviour.

| Claim (notes.md / file-notes.md) | Live |
|---|---|
| Getting there: click Notes in the rail's Browse section | Empty profile: rail = Import… / New note / Explore all tools; Browse appears once a note exists (B 06 vs 10) |
| "Nothing is ever painted as half a word." | "Remove pl" at 235x52 with a note open (A 05/09); "Add from" at 60x24 (B 52) |
| The footer names the focused editor control so focus is never unaccounted for | Not in Preview (A 13; B 13) |
| Sync review shows safe actions, attention items, skips, filesystem effects, deletion-like effects | "Safe item N / Create a Library note", no path (A 62; B 43) |
| Obsidian section under Import once | Keep a folder synced imports .trash/Templates and keeps frontmatter; the difference is unstated (A 62 + sqlite; B 45/46) |
| A root with nothing changed returns to ✓ Up to date; edits surface as ◌ / ⚠ | "Manual check failed" with the row still ✓ Up to date (A 71, 73) |
| Status line shows "Saved" | "Saved 05:48" in UTC (A 07/18; B 09) |
| Picker File name field can be typed into directly | Tree focused on open (A 28; B 27) |
| Folder files header "Linked · Local folder: <folder>" | Also "· Git · N change(s)" on a non-git folder (A 57; B 40/50) |
| Sync setup: choose a display name | Row reads "(name unavailable before cutover)" (A 69; task-32451) |
| Chrome strip "N words · L:C" | "404 words" on a 5,407-word note (B 31/33) |
| Use in Console hands the note to Console | Fails with two messages on a no-provider profile (B 14; A 56) |
| Unchanged repeat rows | notes.csv is New again (B 36; A 61) |
| Filter box Escape hands focus to the first control on the canvas | Footer unchanged, no visible focus row (B 36) |
| Blocked save → "Can't leave yet — fix the title or press Discard new note." | Whitespace-only blank note is discarded silently (B 55) |
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each contradicted claim is corrected in notes.md / file-notes.md, or the fixing task is cited beside it until it lands
- [ ] #2 Getting there gives the empty-profile route (Ctrl+N or the rail's New note row) as well as the Browse row
- [ ] #3 A Verified-against stamp records the walk
<!-- AC:END -->
