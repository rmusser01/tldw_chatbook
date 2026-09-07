---
id: TASK-17611
title: Turn-file-card polish follow-ups from V1.5 final review
status: Done
assignee:
  - '@claude'
created_date: '2026-08-17 17:19'
updated_date: '2026-08-18 02:24'
labels:
  - console
  - ux-polish
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Five parked minors from the turn-file-card annotate feature's V1.5 final review
(`feat/console-turn-file-annotate`), each a real but low-priority polish item deliberately not
folded into the final-review fix wave (which scoped itself to correctness/safety fixes and doc
honesty). Filed together as one follow-up task since each is small and independent; none blocks
the others.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `ConsoleTurnFileCard`'s hunk blocks have an honest ceiling: a file with an unusually large number of hunks stops mounting one block per hunk past a cap and instead shows a "… N more hunks" tail, rather than mounting an unbounded number of blocks.
- [x] #2 The header's expand/collapse-all toggle button label derives from the live DOM state at every render (not just at toggle-press time), so it never shows a stale "expand all" chevron/tooltip after the user has manually expanded some rows individually.
- [x] #3 The card's ✎ (note) and ✕ (delete) glyphs are routed through `resolve_glyph` for terminal-fallback safety, matching the card's existing chevron glyphs. 📝 (the diff-feedback delivery disclosure, `format_diff_feedback_disclosure` in `console_display_state.py`) deliberately stays a raw literal: it renders as a plain TOOL transcript marker with no card-side render hook, shared verbatim by live emission and resume re-derivation, and every other transcript marker in this codebase stays raw for the same live/resume byte-identity contract (see `format_agent_step_marker`'s docstring) -- resolving it at format time would make a persisted marker's glyph depend on whatever ASCII-mode setting happened to be active when it was (re)rendered.
- [x] #4 The bundle-attach and diff-feedback-attach loops in `run_reply` (`Chat/console_agent_bridge.py` or wherever `run_reply` lives) share one "append to the last user message" helper instead of two near-duplicate inline loops.
- [x] #5 `middle_elide_path` budgets in terminal display CELLS (accounting for double-width characters) rather than raw `len()` characters, so a path containing wide characters elides to the correct visual width instead of overflowing or under-filling the row.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. TDD each of the five polish ACs (ceiling, live toggle label, glyphs, attach helper, cell-width elision)\n2. Suites: card/notes/factory + hunks + delivery + bridge\n3. PR
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All five items landed in `tldw_chatbook/Widgets/Console/console_turn_file_card.py`,
`tldw_chatbook/Chat/console_display_state.py`, and `tldw_chatbook/Chat/console_agent_bridge.py`,
each with a red-first regression test (behavior-preserving items #3/#4 rely on the existing
suites plus one new targeted test each).

- **#1 (ceiling)**: new `MAX_MOUNTED_HUNKS = 50` constant. `_mount_hunk_blocks` now mounts only
  `hunks[:MAX_MOUNTED_HUNKS]` and, when hunks were elided, appends one trailing
  `.console-turn-file-hunk-tail` `Static` reading "… N more hunks (open Review for the full
  diff)". The affordance-honesty requirement is satisfied cheaply: a bounded scan over the
  file's already-in-memory `_notes_by_key` entries (same matching rule as the mounted-hunk note
  filter, incl. the snapshot-id tiebreaker) appends "— N carrying note(s)" when at least one
  elided hunk has an existing note. Two new tests (`test_expand_past_hunk_ceiling_mounts_capped_
  blocks_and_honest_tail`, `test_hunk_ceiling_tail_flags_when_an_elided_hunk_carries_a_note`) in
  `Tests/UI/test_console_turn_file_card.py`, using a new `_NotesCapableMultiHunkProvider` fake.
- **#2 (toggle-all label)**: new `_refresh_toggle_all_button` method, re-deriving the header
  chevron/tooltip from live `.console-turn-file-diff` `display` state (the same "all bodies
  visible" rule `_toggle_all` itself reads). Called from `on_button_pressed` after every
  single-row expand/collapse; the header-driven paths (`_expand_all`/`_collapse_all`) already
  set it correctly and are unchanged. Verified RED without the two new call sites before adding
  them back. New test: `test_toggle_all_label_reflects_manually_expanded_rows_not_just_toggle_
  state`.
- **#3 (glyphs)**: `✎`/`✕` now route through `resolve_glyph` (new `_GLYPH_NOTE`/`_GLYPH_DELETE`
  module constants). Found and fixed a real latent bug while wiring this up: `Button.label`
  markup-parses a bare `str`, and the ASCII fallback for `✎` is `"[N]"` -- passed as a plain
  f-string, Textual reads `[N]` as an unclosed style tag and the bracket text itself vanishes
  from the rendered button (`"[N] note"` rendered as `" note"`). Both buttons now build their
  label as a `rich.text.Text(...)` (never markup-parsed) instead of a raw string. 📝 was
  evaluated and deliberately left raw -- see AC#3's text above for the full reasoning; this is a
  documented deviation from the AC's original literal wording ("all routed through
  resolve_glyph"), not an oversight. New test: `test_note_and_delete_buttons_route_through_
  resolve_glyph` in `Tests/UI/test_console_turn_file_card_notes.py`, using the existing
  `Tests/Chat/test_console_glyphs.py` `ascii_mode` fixture pattern.
- **#4 (shared attach helper)**: new module-level `_append_to_last_user_message(messages,
  block) -> tuple[list, bool]` in `console_agent_bridge.py`, replacing both inline backward-scan
  loops in `run_reply`. Copy-on-write semantics preserved exactly (never mutates the caller's
  list/dicts; a falsy block or no eligible carrier both return the input list unchanged, same
  object identity). One behavioral simplification, confirmed non-observable: the old code
  special-cased "already copied by the bundle-block loop" to mutate that private copy in place
  for the second (diff-feedback) attach rather than copying again; the shared helper always
  copies-on-attach instead, so the two-block-stacking case now does one extra (harmless) list
  copy. Full existing bridge (`Tests/Chat/test_console_agent_bridge.py`, incl.
  `test_run_reply_appends_bundle_block_copy_safely`) and delivery
  (`Tests/Chat/test_console_diff_feedback_delivery.py`) suites pass unchanged -- no new test
  needed, per the task's own guidance for behavior-preserving items.
- **#5 (cell-width elision)**: `middle_elide_path` now budgets via `rich.cells.cell_len` instead
  of `len()`. `rich.cells` is already importable in this repo's pinned rich. ASCII paths are
  unaffected (`cell_len(text) == len(text)` for narrow-only text), confirmed by the full existing
  `test_middle_elide_path_*` suite passing unchanged. New red-then-green test
  `test_middle_elide_path_budgets_by_terminal_cell_width_not_char_count` in
  `Tests/Chat/test_console_diff_hunks.py`, using a 3-component CJK path that is exactly 10
  *characters* but 15 *cells* wide -- confirmed RED against the old `len()`-based
  implementation (returns the path unchanged, silently overflowing by 5 cells) before applying
  the fix.

Modified files: `tldw_chatbook/Widgets/Console/console_turn_file_card.py`,
`tldw_chatbook/Chat/console_display_state.py`, `tldw_chatbook/Chat/console_agent_bridge.py`,
`Tests/UI/test_console_turn_file_card.py`, `Tests/UI/test_console_turn_file_card_notes.py`,
`Tests/Chat/test_console_diff_hunks.py`.

Full required suite (card/notes/factory + hunks + delivery + bridge): 296 passed (291 baseline +
5 new tests), 0 failed.
<!-- SECTION:NOTES:END -->
