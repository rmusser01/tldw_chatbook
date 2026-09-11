---
id: TASK-32279
title: Transcript labels a user Deny as blocked and shows model-facing refusal text
status: To Do
assignee: []
created_date: '2026-09-10 19:12'
labels:
  - console
  - approvals
  - ux-copy
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After Deny, the tool marker reads 'blocked' and expanding it shows 'tool call denied by the user: ... Do not retry this call ...' under 'Full output', which is the instruction meant for the model. The card says Deny, the matrix says Off, the audit says denied, and the transcript says blocked. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A user denial renders a user-facing status ('denied by you') distinct from a policy block.
- [x] #2 The model-facing instruction stays in the collapsed detail under a label that says it was sent to the model.
- [x] #3 One vocabulary table for allow, ask, off and deny states is used by the card, matrix, inspector, transcript and audit.
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
The transcript now derives *who refused* from the refusal strings themselves
instead of collapsing every refusal to the word "blocked".
`classify_activity_status` gained one table (`_refusal_statuses` +
`_REFUSAL_STATUS_PREFIXES` in `console_agent_bridge.py`) keyed by the shared
refusal constants (`mcp_tool_provider`, `builtin_tool_gate`,
`local_tool_provider`, `console_chat_controller`), mapping a user Deny to the
new `denied` status, a configured Off to `blocked_off`, a kill switch to
`blocked_kill_switch`, and everything else (unresolved, timeout, resolver
failure) to the generic `blocked`. The live defect was an ordering bug as much
as a vocabulary one: a structured `tool_outcome == "blocked"` returned early
and the refusal text was never read, so the hand-made Deny rendered
identically to a policy block; the outcome is now a floor, not a short circuit.

Display copy lives in one place, `console_chat_models.console_activity_status_word`
(`denied by you` / `blocked (Off)` / `blocked (kill switch)`), used by both
surfaces that render a status word: the activity header row and the plain-text
transcript export. The three new statuses were added to the
`ConsoleActivityStatus` literal, its validation frozenset, the header's CSS
class loop, and the stylesheet (sharing the blocked tint, widening the
nine-cell column the way the existing `unavailable` state does -- state is
carried by the word, never by colour). `CONSOLE_ACTIVITY_REFUSAL_STATUSES`
names the whole family once; `ConsoleMessageActionService` uses it to relabel a
refused marker's disclosure from "Full output" to "Sent to the model", because
what it hides is the instruction the model received, not tool output.

Scope note on AC#3: the audit tokens are lane C's and were deliberately not
edited; the two vocabularies stay consistent because both classify from the
same refusal constants rather than from each other's formatted text.

Files: `tldw_chatbook/Chat/console_agent_bridge.py`,
`tldw_chatbook/Chat/console_chat_models.py`,
`tldw_chatbook/Chat/console_message_actions.py`,
`tldw_chatbook/Widgets/Console/console_assistant_turn.py`,
`tldw_chatbook/Widgets/Console/console_transcript.py`,
`tldw_chatbook/css/components/_agentic_terminal.tcss` (+ regenerated
`screen_agentic_console.tcss`), `Docs/User_Guide/console/agent-runs-and-tools.md`,
and the four test files pinning the old copy.
<!-- SECTION:NOTES:END -->
