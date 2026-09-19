---
id: TASK-32627
title: 'Library Notes: critique 4 improvement ideas (decide before wave 5)'
status: Done
assignee: []
created_date: '2026-09-15 06:45'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A section 13. Ten ideas beyond the defect list, filed as one decision rather than ten tasks. Each needs a yes/no before wave 5 plans; none should be built without one. Sizes are A's.

1. A 'where does this live?' line in the editor header -- one row under the title reading the note's world and, for a synced note, its file path and when it was written. Makes the three worlds answerable from inside the note. Jordan plus solo-operator. S.
2. Fold the three-worlds decision into one screen with three outcomes -- copy in (keeps your folders), keep in step (one managed folder), edit where it is (nothing copied) -- each with a one-line consequence and the same folder picker. Replaces the chooser, the Folder-files empty state and the sync setup entry. Jordan. M.
3. Obsidian-aware import intelligence surfaced BEFORE the review: detect a vault on folder selection and say so on the confirmation line, with counts and what will be skipped. Turns the toggle from a checkbox into a recognised handshake. Jordan plus Alex. S.
4. Backlinks in Edit, not buried in Info -- a collapsed strip under the body expanding to the titles. The data already exists. Researcher/student. M.
5. Capture from Console into Notes -- today only Notes to Console exists, one note at a time. Add Save to Notes on a Console message and Send selection to Console on the Notes list. Researcher/student. M.
6. A pending-writes strip on any active sync root -- the honest version of the P0 fix, making the sync relationship legible during ordinary editing rather than only in Manage. Solo-operator. M.
7. One dense review component shared by Import once and lasting sync -- would close the density and granularity split in one move. Alex. M.
8. Editor chrome that earns its rows -- merge the duplicated save state, put keywords inline under the title, give Info's 25 blank rows a two-column Properties layout. Jordan plus Alex. M.
9. Make the folder picker vault-aware -- folders-first name-ascending, a note-count badge per folder, a recent-roots list. All personas. S.
10. A keyboard map on the list footer that is true, then extended -- n new, g go to folder, e export selected. The accelerator layer is one bug away from being a selling point. Alex plus Sam. S.

Idea 6 overlaps the P0's fix and should be decided with it. Idea 7 overlaps the sync-review density task. Idea 2 overlaps the chooser task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the ten ideas has an explicit accept, defer or decline recorded on this task
- [x] #2 Accepted ideas become their own tasks with acceptance criteria before wave 5 starts
- [x] #3 Declined ideas record why, so critique 5 does not re-raise them as findings
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Decided by the wave-5 controller, 2026-09-15. Rulings, one per idea. Nothing
here is a new surface invented on top of the critique; every accept either has
its own task now or is folded into a defect task that already owned the ground.

| # | idea | ruling | where it went |
|---|---|---|---|
| 1 | "where does this live?" line in the editor header | **ACCEPT** | task-32640 |
| 2 | fold the three-worlds decision into one screen | **DEFER** | — |
| 3 | Obsidian-aware import intelligence before the review | **ACCEPT** | task-32641 |
| 4 | backlinks in Edit rather than Info | **DEFER** | — |
| 5 | capture from Console into Notes | **ALREADY DONE** | task-32146 |
| 6 | pending-writes strip on an active sync root | **ACCEPT, folded** | task-32633 |
| 7 | one dense review component shared by both importers | **ACCEPT, folded** | task-32625 |
| 8 | editor chrome that earns its rows | **HALF DONE, rest accepted** | task-32143 done; task-32642 |
| 9 | vault-aware folder picker | **ACCEPT, folded** | task-32643, with the picker work |
| 10 | a true keyboard map on the list footer, then extended | **ACCEPT, folded** | task-32607 |

**Why the two defers.**

*Idea 2 (one screen, three outcomes).* It is an M-sized IA replacement of a
surface task-32612 already owns — 32612's job is to make the existing chooser
name the structural difference between the three worlds. Building the
replacement screen first would mean 32612's finding is fixed by deleting the
thing it was filed against, and we would learn nothing about whether naming the
difference was sufficient. **Ruling: ship 32612, then re-raise the unified
screen at critique #5 if the chooser still confuses an assessor.** Cost if
wrong: one more critique cycle on a screen we would then rebuild anyway.

*Idea 4 (backlinks in Edit).* Not a defect — no assessor could not do
something because of it. task-32186 has only just made backlink lookup cheap
enough to consider putting on the editor's hot path, and putting an M-sized
strip there in the same wave that fixed the lookup is how a performance fix
gets spent before it is measured. **Ruling: defer past wave 5**; revisit when
there is a measurement of the editor open path with 32186's change in it.

**Why three accepts were folded rather than filed.**

Ideas 6, 7 and 10 are each the *implementation* of a defect task that already
exists — the pending-writes strip is what task-32633 means by surfacing a
refusal; the shared review component is how task-32625 closes its granularity
split; the true-then-extended keyboard map is task-32607's honesty fix plus one
small addition. Filing them separately would have produced two tasks racing for
the same file, which is the shape that has cost this programme two rounds
already. Idea 9 is folded to the picker work for the same reason: it is the
same dialog task-32606 and task-32611 are in.

**Idea 5 verified, not assumed:** task-32146 (Console answer → Library note
with the conversation as provenance) is Done. The critique's own words were
"today only Notes to Console exists" — that direction already shipped, and
32146 built the other one. Nothing left.
<!-- SECTION:NOTES:END -->
