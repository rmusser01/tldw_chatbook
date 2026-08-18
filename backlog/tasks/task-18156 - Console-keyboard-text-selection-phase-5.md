# task-18156 - Console-keyboard-text-selection-phase-5

## Description

Implement keyboard-driven text selection in the Console transcript view. Phase 5 delivers pure motion helper functions for navigating and selecting text in messages — word boundaries, line boundaries, and line up/down movement — as defined in the architectural spec for console selection. These helpers form the foundation for Tasks 2-5 which wire them into the UI.

## Acceptance Criteria

- [x] `s` enters selection mode on eligible selected rows only
- [x] motions per row kind incl. `o` (open/select)
- [x] Enter opens the identical menu incl. feedback gating
- [x] Esc layering
- [x] hint truthful per row kind
- [x] release-click token drained on keyboard finish
- [x] tests green
- [x] docs updated

## Implementation Plan

1. Sweep task IDs to find next available ID
2. Create backlog task file for phase-5 phase
3. Write failing tests for six motion helpers (word_forward, word_back, line_start, line_end, next_line, prev_line)
4. Implement pure motion helpers in console_selection.py
5. Verify tests pass
6. Ruff both files for linting
7. Commit with "feat(console): pure keyboard-motion helpers for selection phase 5"

## Implementation Notes

Implemented six pure motion helpers for keyboard text selection in console_selection.py:

- `word_forward_offset()` — Vim-w: jump to start of next word (whitespace-delimited)
- `word_back_offset()` — Vim-b: jump to start of previous word
- `line_start_offset()` — Vim-0: start of current line
- `line_end_offset()` — Vim-$: end of current line (before newline)
- `next_line_offset()` — Move down one line, preserving column where possible
- `prev_line_offset()` — Move up one line, preserving column where possible

All functions:
- Take `text: str` and `offset: int`, return clamped offset in `[0, len(text)]`
- Are pure functions (no I/O, no state)
- Handle empty text gracefully (return 0)
- Follow Vim semantics for motion direction and scope

Tests added to `test_console_selection_core.py` validate each motion's behavior on the text: "alpha beta\ngamma  delta\n\nepsilon"

All tests pass; code linted clean.

Modified files:
- `tldw_chatbook/Widgets/Console/console_selection.py` — added 7 functions (_clamp + 6 motions)
- `Tests/UI/test_console_selection_core.py` — added 5 test functions covering all motions and edge cases
