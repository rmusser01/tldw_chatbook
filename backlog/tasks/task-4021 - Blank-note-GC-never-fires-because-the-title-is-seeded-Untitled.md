---
id: TASK-4021
title: Blank-note GC never fires because the title is seeded "Untitled"
status: To Do
assignee: []
created_date: '2026-08-09 20:30'
labels:
  - library
  - notes
  - recritique-2026-08-09
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library re-critique 2026-08-09 (RC-03), confirmed by both critique arms AND by the P0 fix agent
reading the code path.

Opening Library ▸ Notes ▸ New ▸ Blank note bumps the rail count (`Notes (2)` → `(3)`) and shows
`Saved` before any keystroke; exiting by ANY path retains the row. Typing text and then deleting
all of it also retains it. Four indistinguishable `Untitled` rows accumulate from merely opening
the editor, and they propagate outward — the Study staging canvas reads
`Carries forward: Untitled, Untitled, Untitled and 1 more.`

**Root cause (confirmed, not hypothesised):** the session-blank GC added by the P2 batch is present
and `_flush_library_note_save` is wired to ~7 exit paths, but its emptiness test reads the
coordinator snapshot's `title`, which `handle_library_notes_create_blank` seeds with the **literal
string `"Untitled"`** rather than leaving it blank with a placeholder. So
`any(value.strip() for value in (title, content, keywords))` is always truthy and the GC branch is
unreachable.

Pre-existing on `origin/dev` (byte-identical source; the three GC tests fail identically there).
Not Escape-specific — reproduces via the Back button too.

**Prior art:** an unmerged sibling branch carries commit `f8bd6e8ac` (task-3315) fixing this in
tandem with a coupled save-seam change. Read it before implementing; the coupling is the reason it
was not a one-liner.

Contrast worth preserving: empty **prompt** and **skill** drafts discard correctly on exit — the
right behaviour already exists twice in the same screen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening the blank-note editor and leaving without typing persists no row, by every exit path (Escape, Back, rail switch, screen leave)
- [ ] #2 Typing into a session blank and then deleting everything also persists no row; a pre-existing note emptied out still saves
- [ ] #3 The title is a placeholder rather than a literal seeded value, or the emptiness predicate reads a field that is genuinely empty
- [ ] #4 The three currently-failing GC tests pass, and the fix is reconciled with task-3315's prior art rather than duplicating it
<!-- AC:END -->
