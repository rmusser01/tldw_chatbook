---
id: TASK-4021
title: Blank-note GC never fires because the title is seeded "Untitled"
status: Done
assignee:
  - '@claude'
created_date: '2026-08-09 20:30'
updated_date: '2026-08-09 22:11'
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
- [x] #1 Opening the blank-note editor and leaving without typing persists no row, by every exit path (Escape, Back, rail switch, screen leave)
- [x] #2 Typing into a session blank and then deleting everything also persists no row; a pre-existing note emptied out still saves
- [x] #3 The title is a placeholder rather than a literal seeded value, or the emptiness predicate reads a field that is genuinely empty
- [x] #4 The three currently-failing GC tests pass, and the fix is reconciled with task-3315's prior art rather than duplicating it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce at HEAD with a real file-backed DB (via the existing 7-test cluster in Tests/UI/test_library_shell.py around test_library_shell_blank_note_*); capture the RED failures as evidence (expect 4 currently-failing, not 3 -- reconcile the count against the AC in Implementation Notes).
2. Read prior art commit f8bd6e8ac (task-3315, unmerged branch feat/media-ingest-followups) in full via `git show`; identify which of its hunks are in-scope for task-4021 (the LIBRARY_NOTE_BLANK_SEED_TITLE constant, the save-seam Untitled fallback in _LibraryDatabaseNoteSessionPort.save_note, the title_blank broadened check + destructive_running/destructive_admission guard in _flush_library_note_save) vs out-of-scope (the Esc-crash fix -- already independently fixed in this tree via PR #1464's _exit_library_note_editor_guarded; the "/" rail-search-grab fix; the app.py ingest-runtime-state refactor -- all unrelated regressions bundled into the same commit).
3. Decide adopt-vs-adapt: the existing (already-committed, not authored by me) test test_library_shell_pre_existing_note_emptied_out_still_saves_in_real_db asserts detail["title"] == "Untitled" after emptying a pre-existing note's title, which pins AC#3's "b" option (emptiness predicate reads a field that is genuinely blank via a broadened check) over its "a" option (never seed a literal value) -- adopt the prior art's approach for that reason, adapting hunk boundaries to this tree's diverged state rather than blind cherry-pick.
4. Check whether the notes path can share the prompt/skill "empty draft discards" seam per the guard rail -- read _enter_library_prompt_create_editor/_enter_library_skill_create_editor to see whether they commit a DB row on click the way handle_library_notes_create_blank does; if notes commit-on-click for a documented reason (task-2858 AC#5's rejected create-on-first-edit alternative), record that the seams are structurally different and cannot be merged without redoing that decision, out of scope here.
5. Implement in tldw_chatbook/UI/Screens/library_screen.py: add the LIBRARY_NOTE_BLANK_SEED_TITLE constant, restore the save-seam fallback in _LibraryDatabaseNoteSessionPort.save_note, broaden the emptiness check in _flush_library_note_save to treat the literal seed as blank, and add the destructive_running/destructive_admission guard so GC never races an in-flight delete.
6. Run the full blank-note-GC test cluster (7 tests) plus a --collect-only sanity sweep of Tests/UI/test_library_shell.py and Tests/Library/ to confirm no collection regressions; capture GREEN evidence.
7. Live-verify in tmux (unique socket/scratch per Global Constraints): create blank -> leave -> rail count unchanged; type -> delete all -> leave -> count unchanged; edit a real pre-existing note -> still saved.
8. Backlog hygiene: tick ACs, write Implementation Notes (including the prior-art reconciliation decision and the 3-vs-4-failing-tests discrepancy), mark Done.
9. Commit with specific paths; self-review the diff before reporting.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced at HEAD with a real file-backed DB: 4 pre-existing GC tests fail identically (not 3 -- the task/AC text undercounts; test_library_shell_blank_note_autosaved_then_emptied_still_gcs_on_back is a 4th, same root cause). Root cause confirmed exactly as briefed: handle_library_notes_create_blank seeds the DB row's title with the literal "Untitled", _read_library_note_editor_fields projects the coordinator SNAPSHOT (not the placeholder-only widget rendering), so _flush_library_note_save's `any(value.strip() for value in (title, ...))` is always truthy and the GC branch is unreachable.

Prior art (f8bd6e8ac, task-3315, unmerged branch feat/media-ingest-followups): ADOPTED the notes-GC-specific hunks (LIBRARY_NOTE_BLANK_SEED_TITLE constant; the save-seam "Untitled" fallback in _LibraryDatabaseNoteSessionPort.save_note; the broadened title_blank check plus a destructive_running/destructive_admission guard in _flush_library_note_save), adapted to this tree's diverged state rather than a blind cherry-pick -- the same commit also carried three unrelated regressions already fixed differently here (the Esc-crash fix already lives at _exit_library_note_editor_guarded via PR #1464) or out of scope (a "/" rail-search-grab fix; an app.py ingest-runtime-state refactor). The already-existing (not authored by me) test test_library_shell_pre_existing_note_emptied_out_still_saves_in_real_db asserts detail["title"] == "Untitled" after emptying a pre-existing note, which settles AC#3's either/or in favor of "broaden the emptiness predicate" over "never seed a literal value" -- adopted for that reason, not preference.

Guard-rail check (prompt/skill seam sharing): _enter_library_prompt_create_editor/_enter_library_skill_create_editor never call a service/DB write on entry (pure client-side draft, _selected_prompt_id stays None / _selected_skill_name stays ""); "Blank note" commits its DB row immediately on click by a previously reviewed, documented decision (task-2858 AC#5 rejected create-on-first-edit as the larger-diff option). The two seams are structurally different by design; sharing them would mean re-litigating that decision, out of scope here.

Fix: added LIBRARY_NOTE_BLANK_SEED_TITLE; _LibraryDatabaseNoteSessionPort.save_note falls back an emptied title to that constant; _flush_library_note_save now treats a snapshot title equal to the literal seed as blank too (title_blank check) and skips the GC branch entirely while a destructive op is running/admitted for this session.

Tests: the 4 pre-existing failing tests now pass. Added 2 new real-DB tests for the rail-switch and screen-leave exit paths (AC#1's full enumeration only had Back/Escape coverage before); RED-verified by reverting the library_screen.py diff and re-running (both failed as expected), then GREEN after reapplying. Updated test_library_shell_blank_note_escape_key_returns_to_list_without_crash, whose docstring/assertions had pinned "today's real end state = row survives" as a deliberately out-of-scope P0 boundary; now that the shared root cause is fixed, Escape GCs the row exactly like Back, so the pin was updated from "row survives" to "row GC'd" (with a polling wait, matching the other real-DB GC tests) -- documented as a stale assumption that did not survive contact with the fix, mirroring Task 1's caution.

Full cluster: 9/9 pass (Tests/UI/test_library_shell.py -k "blank_note or pre_existing_note_emptied"). --collect-only sanity: 2096 tests collected clean across test_library_shell.py + Tests/Library/. A/B verified two unrelated pre-existing failures (test_library_note_footer_covers_navigator_create_sync_and_exit, test_library_note_local_shortcuts_are_region_scoped_and_flush_guarded) fail identically with and without this fix -- part of the documented ~52 known-ambient 60x20-geometry-family failures in this file, not a regression.

Live-verified in tmux (scratch profile sdd_rct2): (1) create blank note -> rail bumps to Notes (1) -> Back with no typing -> Notes (0), row gone; (2) create blank -> type into title -> delete all back to empty -> Back -> Notes (0), row gone; (3) create+save a real note (title "This is a real, permanent note...") -> Notes (1) -> reopen -> clear the title field completely -> Back -> Notes (1) unchanged, row now reads "Untitled" (save-seam fallback confirmed live, not just in tests).

Docs: Docs/User_Guide/library/notes.md already described the correct (target) behavior from task-2858's original LIB-14 work (stamped "Verified against dev @ 6b38a13b8") -- it had silently regressed in code without the doc being wrong, so no prose change was needed; added a "Re-verified against fix/library-recritique-p1s" stamp documenting the regression-and-restore and that the fix now covers all 4 exit seams, not just the 2 the existing prose called out.

Files changed: tldw_chatbook/UI/Screens/library_screen.py (constant + save fallback + emptiness check + destructive guard), Tests/UI/test_library_shell.py (2 new tests + 1 updated test's stale pin), Docs/User_Guide/library/notes.md (re-verification stamp).
<!-- SECTION:NOTES:END -->
