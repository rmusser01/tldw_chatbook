---
id: TASK-32622
title: >-
  Library Notes: import receipt and picker nits — no way forward, an ambiguous
  link count, a wrong entry count, a front-truncated row
status: Done
assignee: []
created_date: '2026-09-15 06:43'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor B D16, D13, D14, personas Jordan and Alex, Obsidian workflow. Four small things on the strongest screen in the product.

1. The receipt after a 54-note import offers only a collapsible Skipped disclosure and 'esc back to notes' -- no 'View imported notes' action after the biggest thing the user has done all session (B cap 25).
2. '54 links resolved' is ambiguous: the review itself counted about 9 wikilinks across 9 notes, so the number is either counting something else or counting it twice (B cap 25). A independently checked the vault and found 57 bracketed links, so the number may be honest and is certainly unexplained (A cap 22).
3. The Import-once picker reports 'Loaded · 17 entries' while displaying 14 entries plus the parent row: the three hidden dot-entries are counted but not shown (B cap 20).
4. A long file name is FRONT-truncated in the review table, so the row reads as a tail fragment ending in .md and its 'vault/Inbox/' prefix is gone -- the one thing that would let the reader find it in the tree, while every other row in the same table carries its folder (B cap 23).

Cause PROVEN by capture for all four. Wave 4's import copy work (task-32554, PR #2685) covered a different set of nits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The import receipt offers a way to the notes it created
- [x] #2 The links figure says what it counted, or is removed
- [x] #3 The picker's entry count matches what it displays, or says what it excludes
- [x] #4 Long row labels keep their folder prefix and truncate where the information is least
<!-- AC:END -->

## Implementation Plan

1. A named way forward on the receipt.
2. Make the links figure say what it counted.
3. Make the picker's count match its list.
4. Stop front-truncating a review row.

## Implementation Notes

**AC#1 -- a way forward.** The receipt's primary-action slot gains "View N
imported notes" whenever the run wrote any (`notes_written` on the snapshot =
imported + updated), posting `ViewImportedNotesRequested`. The handler is the
one "‹ Notes" already uses, and the import controller already reloads Notes
when execution settles, so the rows are fresh on arrival. *Ceiling:* it lands
on the Notes list, not scrolled to the destination folder -- revealing one
folder needs a tree-reveal seam the list does not have. Marked `ponytail:` at
the handler.

**AC#2 -- the ambiguous figure.** "54 links resolved" counted link
OCCURRENCES that reach a note the same batch creates, across every note --
which is why it can exceed the number of notes carrying one, and why a review
showing about nine wikilinks read it as a different number or a double count.
It now reads "54 links to imported notes rewritten". That needed a real plural
per row rather than the `noun + "s"` rule, so the receipt's count table now
carries (singular, plural).

**AC#3 -- the picker's count.** `listing_status` counted `_records`
(everything the scan found) while the list renders `_display_records`. The
settled line now says "Loaded · 4 of 6 entries shown" when they differ and
"Loaded · 6 entries" when they do not. The Scanning… line is left alone: it
says "found", which is what it means.

**AC#4 -- the front-truncated row.** `elide_path_middle` keeps the basename
whole and truncates the head, which is right until the basename ALONE
overflows -- then it falls back to the basename's TAIL and the row loses the
folder that makes it findable. Fixed in `bounded_row_name` (the review's own
helper) rather than in the shared util, because the util's fallback is correct
for its other callers: keep up to 20 cells of folder prefix and elide inside
the NAME, where the start and the extension both survive.

**Not touched, on purpose.** Import once on an already-synced folder still
duplicates it -- that is task-32637, with its own both-orders pin. Nothing
here goes near that path.

**Files.** `Widgets/Library/library_note_import_canvas.py`,
`Library/library_note_import_state.py`,
`UI/Library_Modules/library_notes_controller.py`,
`UI/Screens/library_screen.py`,
`Third_Party/textual_fspicker/parts/progressive_directory_navigation.py`,
`Tests/UI/test_library_notes_w5_import_preview.py`,
`Docs/User_Guide/library/notes.md`, `Docs/User_Guide/file-picker.md`.
