---
id: TASK-37870
title: 'Console: a paste that starts with a line break can be expanded only by a click'
status: Done
assignee:
  - '@claude'
created_date: '2026-10-30 01:35'
updated_date: '2026-10-31 01:27'
labels:
  - console
  - composer
  - ux-review-2026-09-29
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-37800 (owner ruling 2026-10-29, paste then Enter sends) made Enter expand a collapsed token only with the caret right before it. A paste whose text starts with a line break draws that break outside its chip, so the caret can never sit right before the chip, and the only way to expand such a paste is a mouse click. Before TASK-37800, Enter could reach it (dev stepped the first token; the first TASK-33622.1 reading also counted the right edge). In a keyboard-first TUI a token only the mouse can expand is a gap. The User Guide states the limit today. Options for the owner: let the left edge count across a leading break (the caret before the break), or add a dedicated expand key.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner has chosen how a keyboard user expands a paste that starts with a line break, and the choice is recorded in this task
- [x] #2 With the composer focused and no mouse, a paste that starts with a line break can be expanded to its full text, pinned by a Pilot test on the real ChatScreen that fails on the pre-change code
- [x] #3 Paste then Enter still sends on one Enter, and the TASK-37800 tests stay green
- [x] #4 Docs/User_Guide/console/chat-basics.md 'Large pastes' no longer says such a paste must be clicked
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Owner decision 2026-10-30: Enter always sends; a dedicated key named in each token's hint expands the token beside the caret; a click still expands. Pick the key from the Console's free bindings and ADR-031 (chosen: Ctrl+E).
2. Tests first on the real ChatScreen: Ctrl+E expands every kind of token (paste, file, /prompt, a paste starting with a line break) from either side; one Enter sends every flow, a leading-line-break paste included (red on the pre-change head and on dev).
3. Composer: Ctrl+E handling in handle_console_key, the token beside the caret at either edge, else the nearest; chips and the click confirm name Ctrl+E. ChatScreen: Enter no longer steps tokens; F1 Composer group lists Ctrl+E.
4. User Guide chat-basics Large pastes + key table, context-and-rag /prompt line.
5. Rebase onto newest dev, gates, live verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Owner decision (2026-10-30), recorded for AC#1.** ENTER ALWAYS SENDS: Enter never expands a collapsed paste, file or prompt token. A dedicated key, named in the token's own hint, expands the token next to (or under) the caret; clicking a token still expands it. The key is **Ctrl+E**, picked from what is free: bare letters type into the draft; Ctrl+O/G/K/T/Y/Z/U/W/J/H/A/C are Console or composer keys; Ctrl+C/V/X/S/D/Z/A/R/W are ADR-031 terminal-convention keys and Ctrl+Q/P, F1, F6 app globals; F2-F5, F7 and Ctrl+digits are destination keys; Alt chords type characters on macOS terminals; Ctrl+Shift+<letter> arrives as plain Ctrl+<letter> without the Kitty protocol. Ctrl+E is bound nowhere in the Console, arrives as 0x05 in every terminal and in tmux, and its E is the chip's 'expand'. Trade-off: Textual Inputs and readline use Ctrl+E for end-of-line; the composer never did (End is its key, and it already departs from readline for Ctrl+A/Ctrl+K), and with no token in the draft Ctrl+E changes nothing.

**Implementation.** `ConsoleComposerBar`: `EXPAND_TOKEN_KEY`/`EXPAND_TOKEN_KEY_LABEL`; `handle_console_key` routes Ctrl+E to `expand_token_at_caret`, which expands in one press the token beside the caret at either canonical edge (a pending click confirm first, then the token after the caret, then the one before; a paste's edge line break no longer parts it), else the nearest collapsed token, so a chip's hint still holds after typing. Chips read 'Pasted text | N characters | Ctrl+E expand' / '<file label> | Ctrl+E expand'; a click still asks 'Expand pasted text (N characters)? Ctrl+E: yes · Esc: no', answered by a second click or Ctrl+E. `activate_focused_paste_token` is gone and ChatScreen's Enter no longer steps a token. F1's Composer group lists Ctrl+E.

**Tests (test-first).** New Tests/UI/test_console_composer_token_expand_key.py (18 cases, real ChatScreen/composer, real keys): Ctrl+E expands a paste, a file inserted through the real attachment path, a real /prompt insert and a paste starting with a line break, from either side, with the draft, caret and focus unchanged and nothing sent; the token after the caret, then the one before; the nearest one; nothing without a token (mutation: Ctrl+E as End turns it red, 11 == 0); chip copy; a click still expands (second click or Ctrl+E); Esc; file confirm copy; caret leaving a confirm; F1 lists Ctrl+E. 17 of 18 red on the pre-change head 2f85b25daf (e.g. "Ctrl+E did not expand the token; the draft shows 'Summarize this log: \nPasted text | 200 characters | Expand tail'") and on dev 1eaa6a9429; the 18th is the no-token negative control. AC#3: test_console_composer_token_enter.py (14 cases) sends on one Enter in every flow, a leading-line-break paste included. TASK-37800's two tests changed with the decision (see its note). Census: two TASK-37800 node ids swapped in place for test_one_enter_sends_wherever_the_caret_is[question_in_front_of_paste] and test_the_expand_key_expands_each_kind_of_token_from_either_side[leading_break_paste-before]; floor unchanged.

**Live** (real app, Anthropic claude-haiku-4-5 via the review proxy, isolated profile, null keyring, 160x45). Pre-change head: a leading-break paste could not be expanded by keys; Left then Enter sent it (1 request). Fix: 'Summarize: ' + a leading-break paste, Left, Ctrl+E expanded it in place (0 requests), then one Enter sent it (1 request, model replied); a pasted .md path became '📄 photosynthesis.md · 116 B | Ctrl+E expand' and Ctrl+E expanded it from after and from before; a click asked 'Expand pasted text (156 characters)? Ctrl+E: yes · Esc: no' and Ctrl+E expanded it; F1 lists 'Ctrl+E: expand the paste, file or prompt token beside the caret'. At 80x24 the longer chip wraps over three rows of the 26-column draft area. /prompt was not driven live (no saved prompt in the scratch profile); its real command is covered by the tests. Evidence: tools/evidence/bd5/U11/B0-BK*.

**Docs.** User Guide console/chat-basics.md 'Large pastes' (no more 'must be clicked'), its key table (Enter never expands; new Ctrl+E row), and console/context-and-rag.md's /prompt line. Size rows: composer 7155 -> 7174 (owner rule, dated); ChatScreen 24332 -> 24330.
<!-- SECTION:NOTES:END -->
