---
id: TASK-32558
title: >-
  Library Notes: guide sweep residual at 5fd502dbac — fifteen claims
  contradicted or overstated by critique #3
status: Done
assignee: []
created_date: '2026-09-13 06:48'
updated_date: '2026-09-14 22:54'
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
- [x] #1 Each contradicted claim is corrected in notes.md / file-notes.md, or the fixing task is cited beside it until it lands
- [x] #2 Getting there gives the empty-profile route (Ctrl+N or the rail's New note row) as well as the Browse row
- [x] #3 A Verified-against stamp records the walk
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-walk every guide claim wave 4 touched against current dev (fd30614dcd) — verify each factual Notes claim against code or a capture, not against memory.
2. Correct or qualify every claim that fails verification, with the supersession treatment; never delete a sentence.
3. AC#2: Getting there gains the empty-profile route (rail STARTER = Import… / New note / Explore all tools; the Browse ▸ Notes row appears once content exists).
4. Consolidate duplicate Verified-against stamps, blank line before each; add this wave's stamp.
5. Mint the wave's riders; add the wave's lessons; preflight + Tests/Docs green.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reframed from a tidy-up into a verification pass, because wave 4 found eight authoritative-sounding false guide sentences. Method: extract every quoted UI string from the five pages, grep each against tldw_chatbook/ (composed lines probed by their longest literal run), read the misses, and drive the live app for anything a grep cannot settle. Two scratch profiles (empty and 10-note seeded) at 235x52 and 100x30; TLDW_CONFIG_PATH and its data_dir printed and confirmed before each launch; nine captures under wave4-caps/docs-sweep; profile logs hold zero unhandled_exception and one app_stopping each, the INFO record of a deliberate Ctrl+Q.

Roughly 300 factual claims checked across notes.md, file-notes.md, library.md, import-and-export.md and console.md; **nine were wrong and were corrected**, none deleted:

1. notes.md 'Getting there' named the rail's Browse ▸ Notes row as the only route. An empty profile has no Browse section — LibraryLifecycle.STARTER composes exactly Import… / New note / Explore all tools (library_rail.py:1076-1079). Walked: docs-01, and docs-02 shows Explore all tools growing the full Browse section with a Notes (0) row on an empty Library. AC#2.
2. The duplicate-title tie-break was claimed 'in Edit, Preview and Info alike' with no width. True at 235x52 ('Reading list · #d3d7', docs-05); at 100x30 the heading strip carries the back cue and source name either side and the title ellipsizes first — '‹ Back to list    Reading … Library notes' (docs-08, docs-09). Rider 32575.
3. The toolbar Tab counts were said to turn on the filter alone; select_disabled = rendered_count == 0 or running (library_notes_canvas.py:1747), and '○ Select' is visible live on an empty list (docs-03).
4. Retarget/Disconnect's line under the root list is one combined sentence naming both, not 'the same line repeated'.
5. Preview's Escape chip is 'esc back to notes' wide, 'esc notes' compact (LIBRARY_NOTES_PREVIEW_SHORTCUTS(_COMPACT)).
6-8. file-notes.md's Session Git heading, scope line and keyboard guide were all three rewritten by task-15122 on 2026-08-11 and never updated — five 'Verified against' stamps were added to that page in between, two of them live walks of that very panel.
9. file-notes.md's Chunking Lab quirk described a header strip task-32064 (Done) removed, and cited that task as still tracking it.
Plus console.md's 'Both actions' on the Get started card, contradicted by its own task-32555 stamp four paragraphs below (a detected loopback server adds a third).

Also added to the body rather than left in a stamp: every Manage sync folders row is titled 'Sync folder (name unavailable before cutover)' and carries no path, so two roots are distinguishable only by status line and order (task-32451, open).

Stamps: one genuine duplicate consolidated (the trailing fix/library-notes-list stamp repeated its task-32123/32124 clauses verbatim; kept beside the copy they verify, pointer left behind). The other same-branch stamp groups name different tasks and different evidence — merging those would delete evidence, not duplication, so they were left alone and that judgement is recorded here. The new stamps record what was CHECKED including the claims that held, and say which corrections came from a capture and which from source.

Riders minted: 32568-32588 (21), ids swept against every worktree's backlog/tasks, drafts and archive and every origin/* and local ref before the first mint and after the last; global max was 32567. Lessons added to all three files, each with its incident.

Verification: preflight exit 0 (all seven checks). Tests/Docs 22 passed / 2 failed, the identical two README.md names failing on a detached origin/dev worktree. Tests/CI/test_backlog_task_id_uniqueness.py 3 passed. Sibling pins 65 + 15 passed. The diff touches Docs/ and backlog/ only — no production code, no test files.

Files: Docs/User_Guide/library/notes.md, Docs/User_Guide/library/file-notes.md, Docs/User_Guide/console.md, backlog/docs/lessons-testing-evidence.md, backlog/docs/lessons-live-verification.md, backlog/docs/lessons-backlog-hygiene.md, 21 new backlog/tasks files.


**Fix round 1.** The review found the sweep's own failure mode in the sweep's own prose: two counted claims that do not survive the check everything else got.

- "Five 'Verified against' stamps" was wrong. Re-derived from the file's history rather than restated: the pre-sweep page carries 17 stamps, two of them dated 2026-08-07 and therefore before the 2026-08-11 rewrite, so FIFTEEN fall in the window, across six dates. THREE named the Session Git panel — `w3-pickers-git` and `wave3-docs` (both live walks that drove it end to end) plus `w4-import-kbd`, which captured that panel's own header control with the false heading three lines above it. The review said four, counting `w4-file-notes`; that stamp is about the authority line "Git · 1 change", a different surface, so the derived figure is three. Command recorded in the lesson so the next reader can re-run it. Fixed in file-notes.md and lessons-live-verification.md; the finding is stronger with the true numbers.
- "Four paragraphs below" was wrong by ~455 lines — the task-32555 stamp is in console.md's own "Verified against" section. Cited by section now: a positional pointer decays on the next insertion, which is the class of defect this task exists to remove.
- Minor 4: library.md and import-and-export.md were checked and cleared with nothing in the repo saying so — exactly the gap this sweep's own new lesson names. Each now carries a stamp listing what was checked and that nothing needed correcting.
- Minors 3, 5, 6, 7: task-32574's description records the one pin (Tests/UI/test_library_notes_w4_editor.py:551-570) that patches the `list_deleted_notes` seam in place; task-32581 AC#1 reshaped from an implementation step to an outcome; console.md's ragged reflow joined; the lesson's ambiguous "eight" disambiguated (eight before this sweep, nine in it, seventeen in the wave).

**Controller ruling, applied:** the falsifiability script is committed as `scripts/check_guide_claim_strings.py` rather than deleted, with a `--self-test` that asserts its parsing, normalisation and both grep paths against real source. Verified against the pre-sweep file it flags all four of file-notes.md's false claims by line number; against the corrected page only the deliberate historical "(Was …)" clause remains. It exits 0 and prints a read-list on purpose — the judgement half is the reviewed-exception allowlist, filed as **task-32589** (id swept immediately before minting; global max was 32588). The lesson now points at both.

Re-verified after the fix round: preflight exit 0; Tests/Docs 22 passed / 2 failed (the same two README.md names that fail on detached origin/dev); id-uniqueness 3 passed.
<!-- SECTION:NOTES:END -->
