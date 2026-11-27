---
id: TASK-42584
title: "Trace the actions of a chat whose provider replays its own continuations"
status: To Do
assignee: []
created_date: 2026-11-26 18:48
labels:
- follow-up
- console
- trace
priority: medium
references:
- tldw_chatbook/Chat/console_action_capture.py
- tldw_chatbook/Chat/console_trace_action_calls.py
- tldw_chatbook/Chat/console_agent_bridge.py
- Tests/UI/test_console_moonshot_reasoning_reply.py
- backlog/decisions/337-console-reference-backed-semantic-trace-ledger.md
dependencies:
- TASK-42560
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while doing TASK-42560 (round 27, 2026-11-26).

**Why.** TASK-42560 traces Edit & resend, Regenerate and Continue as calls of their turn. In a chat whose provider replays its own continuation checkpoints (Kimi reasoning on moonshot), the agent's requests carry the saved replies as provider artifacts (`active_request`), not as saved revisions. The proofs that admit an action's call (it re-records its turn, and a Continue's turn is found behind its saved replies) cannot name those rows. Traced, a Kimi Continue was refused (`trace_turn_unavailable`, `test_continue_and_regenerate_keep_one_row_per_reply[moonshot-*]`), so TASK-42560 keeps such chats' actions Capture Off, as before (ADR-337 amendment 2026-11-26, point 1).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With Capture On in a Kimi chat, Edit & resend, Regenerate and Continue are traced as calls of their turn, and the message after each is sent and traced, shown by a test through the real controller with only the provider faked, red before and green after
- [ ] #2 The rule that admits them names the saved replies the agent request carries, or the reason it cannot is recorded with an owner decision
- [ ] #3 Any new admission rule is recorded in ADR-337 before the code
<!-- AC:END -->
