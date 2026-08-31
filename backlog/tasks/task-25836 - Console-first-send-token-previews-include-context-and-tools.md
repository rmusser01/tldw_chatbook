---
id: TASK-25836
title: Console first-send token previews include context and tools
status: Done
assignee:
  - '@Robert'
created_date: '2026-08-31 02:42'
updated_date: '2026-08-31 06:26'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When composing the first message of a Console conversation, the two token readouts a user checks -- the context viewer's '~N tokens' header and the cost chip's token label -- count only the draft (chip) or nothing at all (modal header is draft-only, chip ignores system prompt/tools). The first send actually ships the system prompt, project-instruction bodies, tool schemas, and staged evidence, so both readouts massively understate the request. Make both count the full next-send request.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Context viewer: the "~N tokens" header estimate is computed from the loaded next-send payload (messages incl. system row and draft turn, tool schemas, staged evidence text), not the draft alone
- [x] #2 No double counting: the payload's duplicated `system` field is not counted again when the leading system row is already in `messages`
- [x] #3 Cost chip: while the active session has no assistant rows and no usage rows (first send pending), the chip folds the session system prompt, tool schemas, and current draft in as estimated rows alongside the existing staged-evidence row
- [x] #4 Cost chip: once the session has an assistant/usage row, the running total is unchanged from today's behavior (no system/tools/draft pseudo-rows)
- [x] #5 Both readouts stay estimates ("~" semantics preserved) and degrade safely (snapshot error payload, empty draft, no tools)
- [x] #6 Targeted tests pass (context-modal tests + new estimator/chip tests)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: estimate-only change inside two existing UI surfaces; no schema, sync, provider-boundary, or interface-contract decision.

1. Add a pure shared estimator (`Chat/console_display_state.py`) that counts a next-send request: payload messages (role-aware, via `_estimate_tokens_locally`), tool-schema JSON, and extra texts (staged evidence), with a system-dedupe guard.
2. Context viewer: `ConsoleContextModal` gains an optional `payload_estimate` callable used after `_load_snapshot`; `ChatScreen.action_view_chat_context` wires it (draft-only value stays as the pre-load header).
3. Cost chip: in `ChatScreen._build_console_cost_state`, when no assistant/usage rows exist, append estimated pseudo-rows (system prompt, tools JSON, draft) next to the existing staged-evidence pseudo-row.
4. Tests first for each piece; targeted runs of `Tests/UI/test_chat_screen_context_modal.py` plus the new tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Summary.** Both Console "tokens" readouts now count what a send actually
ships. The context viewer's header estimate (`Ctrl+Shift+P`) is computed from
the loaded `next_send_payload` (system row + messages incl. the draft turn +
`native_schemas` JSON + staged evidence text) instead of the bare draft; the
cost chip folds the session system prompt, tool schemas, and the composer
draft in as estimated pseudo-rows while a conversation has no assistant/usage
rows, flipping the existing `~`/"includes estimated" markers.

**Approach & decisions.**
- One shared pure estimator, `estimate_console_next_send_tokens` in
  `Chat/console_display_state.py` (next to `console_prompted_evidence_text`),
  routing through `count_tokens_messages` so the number keeps the same
  semantics as the settings estimate and the chip. Guards: the payload's
  by-design duplicated `system` field is skipped when `messages` already
  carries a system row; prose `tools_info` notes contribute nothing; blank
  extra texts are dropped; all-empty input returns `None` (no count rather
  than a misleading zero).
- `ConsoleContextModal` gained an optional `payload_estimate(snapshot)`
  callable; `_load_snapshot` prefers it post-load and falls back to the
  draft-only `estimate_factory`. Existing callers/tests that pass only
  `estimate_factory` behave exactly as before. Also added
  `watch_token_estimate` — previously the post-load estimate NEVER repainted
  the header (latent bug found by a TDD regression test; the pre-load value
  was the only one users ever saw after a refresh).
- Chip fold-in lives in `ChatScreen._console_first_send_pseudo_rows()` and is
  gated: blank draft returns `[]` before any state is read (idle tick stays
  free), and the fold-in stops the moment the session has any assistant row
  or recorded usage — after a reply the chip is a running total of real
  spend, and re-adding system/tools would corrupt it. Pseudo-rows reuse the
  staged-evidence row contract (`usage=None`), so pricing/`~`-marker
  behavior in `build_cost_snapshot` is untouched.

**Verification.** TDD throughout (every new test watched failing first).
Targeted: 93 passed across the four directly-touched test files + 56 in the
adjacent cost/chip files; ruff clean. Live web-UI check via `tldw-serve`
confirmed the served app runs the new code, the viewer opens through the real
command-palette path with the payload-based header ("Chat Context (~33
tokens)" for a context-free session — truthful there: no system prompt, no
tools), and the chip reads "0 tok" at rest with an empty draft. The
system-prompt/tools leg could not be staged live (synthetic clicks/modifier
keys don't reach textual-web's xterm layer; see
`backlog/docs/lessons-live-verification.md` entry) and is covered by the
mounted-app tests instead.

**Files changed.** `tldw_chatbook/Chat/console_display_state.py` (estimator),
`tldw_chatbook/Widgets/Console/console_context_modal.py` (payload_estimate +
watcher), `tldw_chatbook/UI/Screens/chat_screen.py` (wiring + chip fold-in);
tests in `Tests/Chat/test_console_display_state.py`,
`Tests/UI/test_console_context_modal.py`,
`Tests/UI/test_chat_screen_context_modal.py`,
`Tests/UI/test_console_cost_chip_screen.py` (also dropped one pre-existing
unused import the file already had at HEAD).
<!-- SECTION:NOTES:END -->
