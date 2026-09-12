---
id: TASK-32254
title: >-
  Library Notes duplicate titles: folder-then-age cannot disambiguate two
  unfiled notes of the same age -- design decision to revisit
status: Done
assignee:
  - '@claude'
created_date: '2026-09-10 18:05'
updated_date: '2026-09-11 15:50'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - design-decision
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed as a design decision to revisit, **not** as a defect. Task-32137 shipped exactly as specified and `test_duplicate_titles_render_folder_and_age_suffixes` pins the behaviour with rows aged 2h and 5d.

The spec hole: two notes titled `Reading list`, both unfiled and both the same age, render byte-identically as `Reading list . Unfiled . 2m` (`R/caps/01`). Unfiled-and-recent is the default state for precisely the notes a user is most likely to have just created twice, so the discriminator is a no-op in the case it is most needed. The guide's claim ("each row also names its folder -- 'Reading list . Unfiled . 2h'") is technically true and functionally false.

Proposed discriminator: fall back to the first line of the body, or to an absolute timestamp, when folder and age both match. This changes a pinned test, so it needs a product decision first.

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A user decision is recorded on whether folder-then-age is sufficient
- [x] #2 If revised: two rows sharing title, folder and age are distinguishable at a glance, and `test_duplicate_titles_render_folder_and_age_suffixes` is extended to cover that case rather than only the 2h/5d one
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live at 235x52 on a seeded profile (two "Reading list" notes, both unfiled, both minutes old).
2. Get the product decision (see below), then add a THIRD key that only fires when folder and age tie.
3. Keep `test_duplicate_titles_render_folder_and_age_suffixes` and extend the coverage rather than replace it.
4. Verify live at 235x52 and 100x30; update the guide.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Decision (recorded per AC#1).** The wave-3 controller ruled: add a third
disambiguator ONLY when folder and age both tie. Preferred key is the
modified time of day; a stable short id suffix is the fallback. Not the
first line of the body (unbounded width, and an empty note has none), and
not a full absolute timestamp (it repeats the age the row already carries).
`test_duplicate_titles_render_folder_and_age_suffixes` was updated to the
new truth by EXTENSION, not deletion: it still pins the 2h/5d case
unchanged, since those rows do not tie and must keep the 32137 label.

**Reproduced** live at 235x52 on the seeded profile
(`wave3-caps/list-tree/10-duplicate-rows-before.txt`): two rows reading
`Reading list · Unfiled · 8m`, byte-identical. One line on dev 4a14b3f36f
shows the same thing: `compose_note_row_label` called twice with the same
arguments returns the same string.

**Approach.** `LibraryNotesTreeRow` gains `clock_label` (local `HH:MM`,
from the same timestamp `age_label` already reads, through the same
`parse_browser_timestamp` seam Info's absolute label uses). A new pure
`note_row_tiebreak_labels()` groups note rows by (folder, title, age) and
hands a third part only to groups of two or more: the clock when it is
distinct across the whole group, otherwise `#` plus the first four
characters of the note id. Rows that do not collide are untouched.

**Verified live** after the fix at 235x52
(`20-duplicate-rows-after-wide.txt`) and 100x30
(`23-duplicate-rows-after-compact.txt`): `Reading list · Unfiled · 34m ·
#0874` against `· #cd50` — the seeded pair shares the minute, so the id
fallback is what the live run exercises; the clock branch is pinned by
test.

**Files.** `tldw_chatbook/Library/library_notes_tree_state.py`,
`tldw_chatbook/Widgets/Library/library_notes_canvas.py`,
`Tests/UI/test_library_notes_wave_list.py`,
`Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
