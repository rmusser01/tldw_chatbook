---
id: TASK-32610
title: >-
  Library Notes: the lasting-sync setup pane ships a garbled Obsidian sentence,
  a false scroll hint and a stale Resolution history line
status: Done
assignee:
  - '@robert'
created_date: '2026-09-15 06:39'
updated_date: '2026-09-15 17:40'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A P2 and assessor B D4, personas Jordan and Alex, Obsidian workflow. Heuristic 10 setter -- the one thing that kept Help and documentation from rising.

What happened, three strings on the Keep-a-folder-synced panes.
1. The toggle that decides how a whole vault is read explains itself as, verbatim: 'Turning it off lasts until you quit Chatbook: a vault is offered it again, on, on the next start.' (A cap 48, B cap 31).
2. 'More below — scroll.' is printed above 21 blank rows on the same pane (A cap 48 tail).
3. After activation the pane still reads 'Resolution history unavailable — it starts after this root is activated' with the control disabled (A cap 50); in the sync review the same control is a bare circle glyph with no reason at all (B cap 32, D19) -- the pattern the screen follows correctly for Sort, Server notes and Commit.

Attribution: string 1 is a WAVE-4 REGRESSION, PROVEN. git show 5fd502dbac of Widgets/Library/library_notes_add_from_files_canvas.py has zero occurrences of the sentence and no notes-sync-obsidian checkbox at all: the toggle is new in commit 29cd22159e (task-32535, PR #2679), source lines 425-426. String 2 is INFERRED as consequential -- the pane grew with that toggle. String 3 is PROVEN from the capture; task-32549 gave the disabled controls their reasons but not this recomputation.

Related open task: 32587 owns the behaviour half (the toggle choice does not survive a quit because it has nowhere durable to live). This task is the copy and the state, not the persistence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Obsidian toggle's help text is a sentence a reader can parse, and it states the same scope the behaviour has
- [x] #2 The scroll hint renders only when the scroll container actually overflows
- [x] #3 The Resolution history line is recomputed on activation, and wherever the control is disabled it carries its reason on screen rather than a bare glyph
- [x] #4 Every user-visible string added by wave 4 to these panes is re-read against the shipped render, and the guide's matching paragraphs agree
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify string 1 (Obsidian toggle sentence) against the live tree -- PR #2689 already repaired it on dev; confirm and cite the current string.
2. Trace strings 2 and 3 to their compose()/sync_state() call sites in library_notes_add_from_files_canvas.py; determine root cause of each.
3. Fix the fold hint by mirroring LibraryIngestCanvas's established task-3304/MI-08 convention: always-mounted Static, display managed by a post-layout sync_fold_hint() comparing the scroll body's virtual_size vs container_size, wired from on_mount/on_resize and the two in-place sync_state() fast paths.
4. Fix the Resolution history reason line the same way: always-mounted (once a root exists) in review/receipt phases, recomputed and display-toggled by _sync_review() so activation via the in-place fast path no longer leaves it stale or missing.
5. Update/add unit tests; prove each new pin red against a reverted scratch copy before restoring the fix.
6. Re-check the guide's matching paragraphs for these three strings; no correction needed since none of the literal strings changed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
String 1 (Obsidian toggle sentence): already repaired on dev by PR #2689
(commit be6d7e88fb, task-32535) before this worktree branched. Verified the
shipped sentence in LibraryNotesAddFromFilesCanvas.compose() (configure
phase, Obsidian-vault branch) reads "Skips .obsidian/, .trash/ and
Templates/, and empty files. Note properties become the title and keywords
and stay in the note. Turning it off lasts until you quit Chatbook — the
next start offers a vault this choice again, switched on." -- parseable,
and its scope claim (in-session only, not persisted) matches today's
behaviour (open task-32587 owns making it durable). No further change made.

String 2 (false scroll hint): root cause was `_expects_body_overflow()`, a
compose-time heuristic that unconditionally returned True for the whole
"configure" phase regardless of terminal size or content, printing "More
below — scroll." above blank rows whenever the form actually fit. Replaced
with the same always-mounted / post-layout `sync_fold_hint()` pattern
`LibraryIngestCanvas` already uses (task-3304/MI-08): the Static is always
composed with `display=False`, and a method compares
`#notes-sync-body`'s `virtual_size.height` to its `container_size.height`
after layout (`on_mount` -> `call_after_refresh`, `on_resize`, and both
`sync_state()` in-place-update fast paths). `_expects_body_overflow()` and
its heuristic were deleted. Verified under real (non-mocked-config) pytest
runs that the "configure" phase does NOT actually overflow a 60x20 terminal
with the default snapshot's content (virtual==container height), contrary
to the old heuristic's unconditional claim -- a smaller viewport (60x12)
does overflow, and a much larger one (100x60) does not, both now reported
correctly.

String 3 (stale/missing Resolution history reason): `_history_disabled_reason()`
was only ever evaluated once, at full `compose()` time. `sync_state()`'s
review-phase fast path (`_sync_review()`, used when only in-review fields
change without root_id/token/stale/page/rows/receipts changing -- exactly
what an "Activate reviewed root" click does) already recomputed the
button's own disabled/label/tooltip but never touched the separate visible
`.library-disabled-reason` line, so it either stuck on its pre-activation
text (A cap 50) or never appeared in the first place if the root_id first
became present through that same fast path (B cap 32/D19's bare glyph).
Root-caused to one shared bug: the reason Static's compose-time-only
`if history_reason: yield ...` conditional. Fixed the same way as the fold
hint -- always-mounted once a root exists in review/receipt phases,
display managed by a value recomputed both at compose time and inside
`_sync_review()`.

Tests: extended
Tests/Widgets/Library/test_library_notes_add_from_files_canvas.py (fold-hint
hide/show pins at real measured sizes, replacing a stale always-true 60x20
assertion) and Tests/UI/test_library_notes_w4_layout.py (activation-recompute
pin). Each new pin was proven red by patching a scratch backup copy of the
target function to the old/broken behaviour, confirming the test failed,
then restoring the real fix -- see the three REVERT-A/B/C notes in this
session's history (not committed). One existing test
(test_deferred_comparison_focus_rechecks_origin_before_moving) needed its
`call_after_refresh` count updated from 1 to 2, since `_sync_review` now
also defers a `sync_fold_hint` refresh.

Full suite: Tests/Widgets/Library/test_library_notes_add_from_files_canvas.py
+ test_library_notes_canvas.py + Tests/UI/test_library_notes_w4_layout.py =
99 passed, 0 failed (was 96 before this task's new tests).

Guide: no changes needed. None of the three literal strings changed (the
Obsidian sentence was already fixed by #2689; "More below — scroll." and the
Resolution-history reason text are unchanged, only their visibility/timing
changed), and Docs/User_Guide/library/notes.md does not make any claim about
when the scroll cue or resolution-history reason appear that these fixes
contradict.

Modified: tldw_chatbook/Widgets/Library/library_notes_add_from_files_canvas.py,
Tests/Widgets/Library/test_library_notes_add_from_files_canvas.py,
Tests/UI/test_library_notes_w4_layout.py.
<!-- SECTION:NOTES:END -->
