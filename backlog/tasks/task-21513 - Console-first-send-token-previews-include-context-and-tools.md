---
id: TASK-21513
title: Console first-send token previews include context and tools
status: In Progress
assignee:
  - '@Robert'
created_date: '2026-08-31 02:42'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When composing the first message of a Console conversation, the two token readouts a user checks -- the context viewer's '~N tokens' header and the cost chip's token label -- count only the draft (chip) or nothing at all (modal header is draft-only, chip ignores system prompt/tools). The first send actually ships the system prompt, project-instruction bodies, tool schemas, and staged evidence, so both readouts massively understate the request. Make both count the full next-send request.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] Context viewer: the "~N tokens" header estimate is computed from the loaded next-send payload (messages incl. system row and draft turn, tool schemas, staged evidence text), not the draft alone
- [ ] No double counting: the payload's duplicated `system` field is not counted again when the leading system row is already in `messages`
- [ ] Cost chip: while the active session has no assistant rows and no usage rows (first send pending), the chip folds the session system prompt, tool schemas, and current draft in as estimated rows alongside the existing staged-evidence row
- [ ] Cost chip: once the session has an assistant/usage row, the running total is unchanged from today's behavior (no system/tools/draft pseudo-rows)
- [ ] Both readouts stay estimates ("~" semantics preserved) and degrade safely (snapshot error payload, empty draft, no tools)
- [ ] Targeted tests pass (context-modal tests + new estimator/chip tests)
<!-- AC:END -->

## Implementation Plan

ADR required: no
ADR path: N/A
Reason: estimate-only change inside two existing UI surfaces; no schema, sync, provider-boundary, or interface-contract decision.

1. Add a pure shared estimator (`Chat/console_display_state.py`) that counts a next-send request: payload messages (role-aware, via `_estimate_tokens_locally`), tool-schema JSON, and extra texts (staged evidence), with a system-dedupe guard.
2. Context viewer: `ConsoleContextModal` gains an optional `payload_estimate` callable used after `_load_snapshot`; `ChatScreen.action_view_chat_context` wires it (draft-only value stays as the pre-load header).
3. Cost chip: in `ChatScreen._build_console_cost_state`, when no assistant/usage rows exist, append estimated pseudo-rows (system prompt, tools JSON, draft) next to the existing staged-evidence pseudo-row.
4. Tests first for each piece; targeted runs of `Tests/UI/test_chat_screen_context_modal.py` plus the new tests.
