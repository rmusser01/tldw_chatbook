---
id: TASK-322
title: Bound Console conversation history by tokens before dispatch
status: Done
assignee: []
created_date: '2026-07-20 18:45'
updated_date: '2026-07-22 14:42'
labels:
  - console
  - llm
dependencies:
  - task-320
  - task-321
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The native Console send path sends the entire conversation to the provider on every turn with no token or message cap. `console_chat_controller.py:1784` (`_provider_messages_for_session`) collects all session messages; the only budget applied in `_provider_message_payloads` (`console_chat_controller.py:1820`) is `max_history_images` — it counts images, not text tokens. Once a conversation exceeds the model window, cloud providers reject the request with a 400 (the conversation becomes un-continuable with no graceful degradation) and local servers silently front-truncate — dropping the prepended system prompt first — while token cost grows without bound. (The deprecated enhanced/legacy chat path shares this defect but is out of scope as it is being replaced by the Console.)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console history is bounded by real tokens against the model input window (task-320/task-321) before dispatch
- [x] #2 The system prompt and the latest user turn are always preserved; oldest turns are dropped in whole request-valid units (never splitting a tool_call from its tool_result)
- [x] #3 The user is notified when earlier history was trimmed
- [x] #4 A conversation that exceeds the model window remains continuable (no overflow-driven provider 400)
- [x] #5 Tests cover the trim boundary and system-prompt preservation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-22-console-history-token-budget.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Bounded the native Console conversation history by real tokens against the model window before dispatch.

Approach: a NEW pure module console_history_budget.py does multimodal-aware token counting + whole-turn trimming (BoundResult, bound_messages_to_window, count_console_messages_tokens, DEFAULT_RESPONSE_RESERVATION); ConsoleChatController wires it at the SINGLE dispatch choke point (top of _stream_assistant_response, before the agent-vs-direct branch — _run_agent_reply copies via list(provider_messages), so both paths are bounded) and appends a display-only SYSTEM trim note when history dropped.

Details: always preserves the leading system prefix + the current turn (last user to end); drops oldest WHOLE turns (never splits a user/assistant pair); budget = window - response_reservation - max(512, window//50); response_reservation = self.max_tokens or 1024. Built on the existing token_counter seam (get_model_token_limit + count_tokens_messages) — tasks 320/321 sharpen the numbers behind it transparently. The counting ADAPTER exists because count_tokens_messages crashes on the Console's multimodal list content (does content.split() on a list); it flattens text + adds a flat per-image estimate (1024). The trim-note session lookup is guarded (try/except KeyError, mirroring _append_failure_system_row). SYSTEM notes are filtered from _provider_message_payloads so they are never resent or re-counted. tiktoken is absent in the venv (word-split fallback) so trim-logic tests inject a deterministic count_fn.

Known limitations (documented): unknown local models get a conservative 4096 default window (over-trims until task-320 refreshes the table / task-325 wires the window override — the bound_messages_to_window(window=...) seam is ready); intra-agent-loop growth is task-326's scope, not this; per-send notes can repeat on retries (follow-up).

Verification: 8 unit tests (boundary/preservation/degenerate/multimodal/window-override, hand-traced) + 3 controller tests (trim+note, no-trim, session-close-race) = 114 passed; full Tests/Chat sweep zero novel regressions vs origin/dev; opus whole-branch review Ready-to-merge (0 Crit/0 Imp).

Files: tldw_chatbook/Chat/console_history_budget.py (new), tldw_chatbook/Chat/console_chat_controller.py, Tests/Chat/test_console_history_budget.py (new), Tests/Chat/test_console_chat_controller.py.
<!-- SECTION:NOTES:END -->
