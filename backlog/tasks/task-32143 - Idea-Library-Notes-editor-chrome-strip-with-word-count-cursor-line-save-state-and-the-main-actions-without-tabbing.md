---
id: TASK-32143
title: >-
  Idea: Library Notes editor chrome strip with word count, cursor line, save
  state and the main actions without tabbing
status: Done
assignee:
  - '@claude'
created_date: '2026-09-08 21:39'
updated_date: '2026-09-11 17:24'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - idea
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Improvement pitched by the design assessor for the power user: a status strip at the bottom of the body (where terminal users look) carrying words, cursor line and save state, plus Save / Preview / Keywords / Delete reachable without Tab. Today 'Saved' floats above the mode tabs and the word count lives under Info. Size S. Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design agreed with the user before implementation
- [x] #2 The strip replaces the dead meta line; the floating save state stays where the compact band pins it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Live-reproduce the editor chrome at 235x52: confirm which line 'floats' (AC#2's target).
2. Read every pin on the candidate widgets before moving anything.
3. RED test on the real editor route: the strip's facts cell is absent.
4. Add the facts cell to the existing save-state row (the floating line becomes the strip), fed by the word count the controller already computes plus the body TextArea's cursor.
5. Hide it below 80 columns; never focusable.
6. GREEN; live captures at 235x52 and 100x30 and below 80; guide + stamp; CSS bundle sync + boot-CSS budget.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Ships the editor chrome strip, plus the dead meta line's removal that AC#2 (as amended) names. Coordinator rulings of 2026-09-11 applied.

**What shipped.** One right-aligned Static (`#library-note-chrome-facts`) directly under the note body, reading `N words · L:C` with a one-based caret. Shown only while Edit is the open view on a terminal >= 80 columns; Preview and Info have no caret, and a narrower terminal gives the row back to the body. Never focusable, no new colours (muted to the meta line's tier).

**State feed, at no new cost.** `_library_note_presentation_state()` already counts the body's words for `metadata_line` on every change, so the count rides the presentation state as a number (`LibraryNotePresentationState.word_count`) -- no scan of its own, no database read. The caret comes off the mounted TextArea through `@on(TextArea.SelectionChanged, '#library-note-body')`, because arrow keys never reach the presentation state; `@on(Resize)` re-decides the 80-column gate (the compact flag only flips at 120). No debounce needed: the per-keystroke work is one string format.

**AC#1.** Agreed by the user's wave-3 instruction, 'wave 3 all of them' -- the approval for this idea and the rest of the wave.

**AC#2 as amended: the dead meta line is gone.** `#library-note-meta` was composed inside `#library-note-wide-utilities`, which `apply_session_state` sets to `display = False` unconditionally, and was re-rendered on every state apply for nobody. Removed, along with its half of the two-selector meta update loop. Its muted-colour rule moved to `#library-note-context-meta` -- the meta line that actually renders in Info, which had no colour rule of its own, so the muting had been spent on an invisible widget. Four `test_library_shell.py` references became absence pins rather than deletions (two `renderable` reads retargeted to Info's line with `assert not screen.query('#library-note-meta')` beside them; the CSS-block test now asserts the old selector is absent and the new one is muted). The floating save state stays where the compact band pins it -- rider TASK-32513, with both blockers measured.

**No accelerator on the strip, and that is correct.** Library Notes deliberately has NO save accelerator: `action_library_notes_save` exists but is bound to no key, and two tests pin the absence -- `Tests/UI/test_library_honesty_accessibility.py::test_notes_ctrl_s_is_absent_from_binding_footer_and_f1_while_skill_keeps_it` and `test_library_shell.py::test_library_note_ctrl_s_is_unavailable_while_save_remains_explicit`. The one real editor binding (`esc` -> back to notes) is already in the footer one row below the strip, so repeating it would be noise.

**Rejected first attempt, kept as evidence.** Putting the facts on the existing save-state row instead of a new one was built, measured live and reverted: that row has no slack, because `#library-note-primary-actions` resolves ~93 cells wide whatever the pane is, and the save state is the row's `1fr`. At 120x40, 160x45 and 190x45 it was crushed to one character -- 'S5,408 words · 1:36'. Capture: wave3-caps/chrome-strip/93-evidence-status-crushed-at-120x40.txt.

**Second rider.** TASK-32514: the editor states its save state twice in two vocabularies -- the authority line's 'Saved 16:47' (from `status_line`) versus `#library-note-status`'s plain 'Saved' (from `resolve_database_note_status_channels`, which has no timestamped form).

**Test changes in `test_library_shell.py`.** (a) `test_library_note_compact_surplus_allocation_expands_only_named_owner[editor]`: the strip is a fixed row at >= 80 columns, so the 1fr body owner is one shorter at both sizes (10 -> 9, 16 -> 15); the strip joins the fixed-selector list at height 1, so the test still pins exact heights and still asserts only the named owner grows, by the same 6. Below 80 the strip is hidden and the four 60x20 allocation tests are untouched. (b) `test_library_note_css_bounds_editor_body_and_mutes_meta` was red on dev for two stale reasons and is now green: it asserted the auto/12/20 body ceiling task-32217 retired (now 1fr/6, no ceiling), and it read `tldw_cli_modular.tcss` for `#library-*` rules that task-25812/24459 split into `screen_agentic_library.tcss`. Repaired to the shipped truth, which is also what lets its meta half run at all.

**Live verification** (seeded profile, real app under tmux, captures in wave3-caps/chrome-strip/): on open `6 words · 1:1`; after typing four words `10 words · 1:28`; after Down + Right x2 `10 words · 3:2` (caret only); 100x30 one row under the body; 79 columns hidden with the body taking the row back; Info shows no strip while keeping the save state and Info's own meta line; 100 -> 79 -> 100 restores it.

**Files.** `tldw_chatbook/Widgets/Library/library_notes_canvas.py`, `tldw_chatbook/UI/Library_Modules/library_notes_controller.py`, `tldw_chatbook/css/components/_agentic_terminal.tcss` (+ generated `screen_agentic_library.tcss`; boot bundle unchanged), `Tests/UI/test_library_notes_wave_chrome_strip.py` (new, 6 tests), `Tests/UI/test_library_shell.py`, `Docs/User_Guide/library/notes.md`, riders `backlog/tasks/task-32513*` and `task-32514*`.

**AC#2's original wording, for the record.** "The strip replaces the floating meta line rather than adding to it" -- amended by the coordinator on 2026-09-11, after the two blockers above were measured, to "The strip replaces the dead meta line; the floating save state stays where the compact band pins it". Rider TASK-32513 carries the save-state move.

**Fix round 1 (review findings F1-F6, all accepted).** F1: the `@on(Resize)` handler was unpinned -- the suite's only resize crossed the 120-column compact breakpoint, which re-runs `apply_session_state` and re-decides the gate for free, so neutering `_note_chrome_follows_width` still passed. The resize walk is now 100 -> 79 -> 100, entirely below 120, and the handler is load-bearing (RED with it neutered, GREEN restored). F2: the colour move had carried the old block's `width: 100%` and `margin: 0 0 1 0` across, and that margin pushed Info's backlinks title and its Reuse & Export / Danger headers down a row at 170x48 (20->21, 21->22, 25->26) where no compact test could see it -- the rule is colour-only now, the rows are back at 20/21 and [14, 21, 25], and `test_info_meta_line_is_muted_without_pushing_the_rows_below_it` pins it at wide size. F3: the repaint was not "only a string format" -- truth-testing `self.query("#id")` walks the whole subtree, so the old four-lookup sequence measured 263.6 us/call on a 35,310-byte note against 80.4 us for the `len(text.split())` the design banned; the caret handler now passes the `TextArea` its event carries and the rest share one `try/except NoMatches`, measuring 0.7 us, and the docstring states the numbers. F4: the guide's Status line sentence "It does not carry a word count" was and is true, so it is restored; the Chrome strip row beside it and the stamp already record what changed. F5: this amendment record moved inside the managed section. F6: PARTIALLY DISPUTED -- see below.

**F6, disputed on the facts.** The finding says `_authority_prefix` "never reads `self.app` and cannot raise `NoActiveAppError`; it branches on `self.compact`". It does read it: `narrow_stage = self.compact and self.app.size.width < _AUTHORITY_PREFIX_MIN_WIDTH`, inside its own `try`, and its docstring states the same fallback rule ("A widget with no live app keeps the prefix: that is the answer that loses nothing"). So the precedent is real and the citation stays -- made exact instead, naming the line and the one true difference: that method catches `Exception` where this one catches the single error `self.app` actually raises.
<!-- SECTION:NOTES:END -->
