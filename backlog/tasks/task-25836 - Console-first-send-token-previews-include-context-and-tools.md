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
ships. The Next Send tab's header estimate (`Ctrl+Shift+P`) is computed from
the loaded `next_send_payload` (system row + messages incl. the draft turn +
`native_schemas` JSON + staged evidence text) instead of the bare draft; the
cost chip folds the session system prompt, tool schemas, effective-memory
rows, response prefill, and the composer draft in as estimated pseudo-rows
while a conversation has no assistant/usage rows, flipping the existing
`~`/"includes estimated" markers.

**Approach & decisions.**
- One shared pure estimator, `estimate_console_next_send_tokens` in
  `Chat/console_display_state.py` (next to `console_prompted_evidence_text`),
  routing through `count_tokens_messages` so the number keeps the same
  semantics as the settings estimate and the chip. Guards: the payload's
  by-design duplicated `system` field is skipped when `messages` already
  carries a system row; prose `tools_info` notes contribute nothing; blank
  extra texts are dropped; schemas that fail to serialize degrade to "no
  tools row" rather than raising; all-empty input returns `None` (no count
  rather than a misleading zero).
- `ConsoleConversationInspector` (dev's unified Costs/Exchange/Next Send
  modal) gained an optional `payload_estimate(snapshot)` kwarg threaded
  through `_push_console_inspector`; `_load_snapshot` prefers it post-load
  and falls back to the draft-only `estimate_factory`. Existing
  callers/tests that pass only `estimate_factory` behave exactly as before.
  (On the feature branch the standalone `ConsoleContextModal` got the
  equivalent change plus a `watch_token_estimate` repaint fix for a latent
  staleness bug dev's port had already solved by ordering.)
- Chip fold-in lives in `ChatScreen._console_first_send_pseudo_rows()` and
  is gated: blank draft returns `[]` before any state is read (idle tick
  stays free), and the fold-in stops the moment the session has any
  assistant row or recorded usage — after a reply the chip is a running
  total of real spend, and re-adding system/tools would corrupt it.
  Pseudo-rows reuse the staged-evidence row contract (`usage=None`), so
  pricing/`~`-marker behavior in `build_cost_snapshot` is untouched.
- Qodo review round (PR #2261), all four findings addressed:
  1. Undercount — the fold-in now resolves effective memory through the
     controller's own `_project_session_effective_memory` and the response
     prefill through `_resolve_submit_prefill` (both synchronous,
     read-only; the preview path uses the same seams) instead of
     re-deriving partial context.
  2. High: mouse-send race double-count — when the trailing transcript row
     is a user message with the draft's exact text (send in flight, draft
     not yet cleared), the draft pseudo-row is dropped; the persisted row
     is what the chip prices.
  3. Unguarded `json.dumps` — serialization failures in the estimator skip
     the tools row instead of propagating.
  4. Session-scoped staged evidence — `_console_next_send_token_estimate`
     gates the screen-global pending launch on the captured session still
     being active, mirroring `_console_settings_context_estimate_for_session`.
- Task ID renumbered 21513 → 25836: dev's Daily-Reports task had taken 21513
  between branch creation and the PR (the exact collision class the
  `docs/lesson-adr-number-collisions` work tracks).

**Verification.** TDD throughout (every new test watched failing first).
On dev's base: 112 passed across the touched test files (estimator
counting/guards incl. serialization degradation, inspector header
payload-preference + fallback, chip fold-in content/gating/stop-after-reply,
race skip, prefill/memory inclusion, staged-evidence session gate) plus the
backlog task-ID uniqueness guard; ruff clean. Two failures in
`test_console_inspector_navigation.py` reproduce on clean dev — pre-existing.
Live web-UI check via `tldw-serve` (on the branch build) confirmed the served
app runs the new code end-to-end; the system-prompt/tools leg is covered by
the mounted-app tests (browser automation can't deliver synthetic
clicks/modifier keys to textual-web's xterm layer — see
`backlog/docs/lessons-live-verification.md`).

**Files changed (dev port, PR #2261).**
`tldw_chatbook/Chat/console_display_state.py` (estimator + json guard),
`tldw_chatbook/Widgets/Console/console_conversation_inspector.py`
(payload_estimate), `tldw_chatbook/UI/Screens/chat_screen.py` (wiring,
session-gated staged evidence, chip fold-in with memory/prefill/race
handling); tests in `Tests/Chat/test_console_display_state.py`,
`Tests/UI/test_console_conversation_inspector.py`,
`Tests/UI/test_console_cost_chip_screen.py`.
<!-- SECTION:NOTES:END -->
