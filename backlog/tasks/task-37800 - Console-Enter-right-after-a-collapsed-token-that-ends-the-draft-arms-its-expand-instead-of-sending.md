---
id: TASK-37800
title: >-
  Console: Enter right after a collapsed token that ends the draft arms its
  expand instead of sending
status: Done
assignee:
  - '@claude'
created_date: '2026-10-29 17:13'
updated_date: '2026-10-30 05:15'
labels:
  - console
  - composer
  - ux-review-2026-09-29
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33622.1 implemented the owner decision of 2026-10-27 (Enter expands a token only when the caret is on it; otherwise Enter sends) as: the caret at EITHER edge of a collapsed token is on it. The caret never sits inside a token, so some reading of 'on' was needed; counting the right edge was the implementer's reading, not the owner's words. Its consequence: whenever the draft ends in a token, Enter still does not send. Paste then Enter, question then paste then Enter, a /prompt insert then Enter, and a file insert then Enter each still take three Enters (confirm, splice open, send), which is the original P1 symptom in token-last order. Review of the TASK-33622.1 branch reproduced it: 'Summarize this log: ' + paste + Enter showed the expand confirm, and a second Enter spliced the paste open. One alternative is block-cursor semantics: only a token's left edge counts as on it, so paste-then-Enter sends while Left/Home still reach a token to expand it. This needs an owner ruling before any change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner has ruled which caret positions count as on a collapsed token, and the ruling is recorded in this task
- [x] #2 If the ruling changes the behaviour: with the caret right after a collapsed token that ends the draft, a single Enter dispatches exactly one send carrying the full text, pinned by a Pilot test on the real ChatScreen that fails on the pre-change code
- [x] #3 Docs/User_Guide/console/chat-basics.md 'Large pastes' states the ruled caret rule
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin the owner ruling (2026-10-29): only a caret on a token or immediately before it (its left edge) expands it; right after it sends.
2. Tests first on the real ChatScreen: paste then Enter, question then paste then Enter, /prompt insert then Enter, file insert then Enter, and a caret right after a mid-draft token each send once with the full text (red on the pre-change head and on dev); Left onto a trailing token then Enter still expands.
3. Composer: _caret_token counts a collapsed token only at its left edge (not across a paste's leading line break); a pending confirm still owns Enter at either edge.
4. Update tests that pressed Enter right after a token to reach its left edge first; re-word TASK-33622.1 ACs; User Guide chat-basics Large pastes + key table, context-and-rag /prompt line.
5. Rebase onto newest dev, gates, live verification (paste then Enter sends once).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Owner ruling (2026-10-29), recorded for AC#1.** PASTE THEN ENTER SENDS: Enter expands a collapsed token only when the caret is ON the token or immediately BEFORE it; a caret right AFTER a trailing token (paste then Enter, a question then a paste then Enter, a /prompt insert then Enter, a file insert then Enter) sends.

**Implementation.** `ConsoleComposerBar._caret_token` counts a collapsed token only at its left edge (the caret never sits inside a token), and not when a paste's leading line break, drawn outside its chip, parts the two. A pending confirm still owns Enter at either edge, so the Enter after a click (whose caret lands after the token) answers its confirm. A caret right after any token, mid-draft included, is not on it. The composer module shrank by one line (7142, under its 7143 row); no new modules or CSS.

**Tests (test-first).** `Tests/UI/test_console_composer_token_enter.py::test_one_enter_sends_with_the_caret_right_after_a_token` drives the real ChatScreen with real keys for [paste, question_then_paste, prompt_insert (the real `/prompt <name>` command over a real PromptsDatabase), file_insert (the real `_process_console_attachment` on a .md file), after_mid_token]: one Enter, one provider stream call, the full text. All five were red on the pre-change head 4bee0aff92 with the confirm shown (e.g. "one Enter did not send; the draft shows 'Summarize this log: Expand pasted text (199 characters)? Enter: yes · Esc: no'") and red on dev ed7481a30c with a bare "Expand?"; green after. `test_left_onto_a_trailing_token_then_enter_still_expands_it` pins the other half (Left, Enter asks, Enter expands, nothing sends; then the next Enter sends); red on dev ("Summarize this log: Expand?"). Cases that pressed Enter right after a token now press Left first: three in the same file, keymap's confirm-copy case, and decomposition's Enter-matches-click case (its caret cell now leads the row).

**Comparison by test id** (rebased head vs pre-change head 4bee0aff92 and dev, the env-red files re-collected under `bootstrap_profile` by throwaway probes, deleted): token_enter 17/17 pass (dev: 17 red, the original P1); raw_cli_arm_in_place 4/4; keymap, internals_decomposition, composer_collapse, send_draft_snapshot, composer_undo, caret_nav, cursor: no fix-only red that survives a lone rerun (every one passed alone; the setup-modal, focus and click-race cases flip run to run on both trees). Module-size ratchet: the same 8 rows red on dev; screen-size ratchet and the UI PR gate census checker pass.

**Live** (real app, Anthropic claude-haiku-4-5 via the review proxy, isolated profile, null keyring, 160x45). Pre-change build: paste then Enter showed "Expand pasted text (125 characters)? Enter: yes · Esc: no" and sent nothing (0 proxy requests). Rebased fix: a four-line paste then one Enter sent at once (1 request) and the model described all four lines; "Summarize this log in 5 words: " + paste + Enter sent once (2nd request) and the model replied "Workers crashed, recovered, stable."; Left put the caret before the trailing chip, Enter showed the named confirm and a second Enter expanded it in place, with no request. Evidence: `tools/evidence/bd5/U11/70-80-v2*`.

**Docs.** User Guide `console/chat-basics.md` 'Large pastes' states the ruled caret rule and the key table's Enter row follows; `console/context-and-rag.md`'s /prompt sentence says Enter sends and Left then Enter expands. TASK-33622.1 ACs #1-#3 re-worded to cite this ruling.

**Known limit.** A paste that starts with a line break can only be expanded by a click: the caret can never sit right before its chip. The User Guide says so.

**Merge evidence (2026-10-29, rebased onto dev ff815f8559).** Review of 6e6447f19a: ship, no blocking findings. Follow-ups applied: both TASK-37800 tests joined the UI PR gate census (floor 1152 + 2 = 1154); a new real-ChatScreen case pins that a pending expand question still answers Enter at its token's right edge (live: Home, Enter, End left the question up and the next Enter expanded it), and the User Guide now says so; the composer's size row was lowered to its measured 7142 before TASK-37873 raised it. The raw-CLI hidden-paste finding is fixed here as TASK-37873; owner questions filed as TASK-37870, TASK-37871 and TASK-37872. Under the plain command test_console_composer_token_enter.py 18 passed, and all 18 fail on origin/dev (a bare 'Expand?', no send, or a chip without '| Expand'). Composer test files compared by test id with a detached origin/dev worktree: no new failures. ./scripts/preflight.sh passes.
<!-- SECTION:NOTES:END -->
