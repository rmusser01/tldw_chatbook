---
id: TASK-42560
title: "With project instructions, the captured send after a capture-off reply is refused"
status: To Do
assignee: []
created_date: 2026-11-26 08:07
labels:
- follow-up
- console
- trace
- needs-owner-decision
priority: medium
references:
- tldw_chatbook/Chat/console_trace_rewritten_tail.py
- tldw_chatbook/Chat/console_trace_continuation.py
- tldw_chatbook/Chat/console_trace_service.py
- backlog/decisions/337-console-reference-backed-semantic-trace-ledger.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while doing TASK-42550 (round 27, 2026-11-26).

**Why.** In a captured chat with project instructions (AGENTS.md), several actions send their reply with capture off: Edit & resend, Continue, Regenerate, and Retry of a failed reply. After any of them, the next ordinary message stops with "Trace capture could not start", so every later captured send in the chat costs a **Send without capture** click. This happens on any provider, including chats that never switch. Without project instructions the same next message is sent.

**Evidence.** Probes through the real controller, bridge, gateway and trace factory, with only provider HTTP faked (llama.cpp; scratch, not committed). Each probe sent "Hello" and "And now?", ran the action, then sent one more message. The results were identical on dev 6e41919880 and on fu/r27-slotdrop.

| Action | Without AGENTS.md | With AGENTS.md |
|---|---|---|
| none (control) | sent | sent |
| Edit & resend | sent | refused: `unsupported_surface_change` (`count_or_span`) |
| Continue | sent | refused: `unsupported_surface_change` |
| Regenerate | sent | refused: `completed_tool_turn_revision` |
| Retry of a failed reply | sent | refused: `completed_tool_turn_unavailable` (also found by the retry stream) |
| Impersonate | sent | sent (it never appends to the transcript) |
| Citation repair | not probed | not probed |

Citation repair needs a RAG citation contract to drive through the controller. It replaces the saved reply after the captured call, so it is expected to behave like Regenerate.

**Cause.** The next send must replace the recorded project-context row with saved rows the trace never saw, plus the new user message and the turn's new project context.
- **Edit & resend and Continue:** the history changed. The rewritten-tail rule (ADR-337, 2026-10-27) would prove it, but `build_rewritten_tail_witness` requires the last incoming row to be the current user turn, and here it is the project-context artifact. The amendment says "tool or project artifacts in the replacement ... stay refused". Without AGENTS.md there is no trailing project row, so the rewritten tail admits the send.
- **Regenerate:** the counts match the completed-turn transition (2026-09-05/09-07), so the planner selects it. Its validation then refuses, because the saved assistant is the regenerated sibling and not the captured call's response revision. The ADR forbids retrying another proof after a validation error. Without AGENTS.md the send is a plain append.
- **Retry of a failed reply:** the uncaptured reply follows a failed captured call, and the completed-turn witness's chain and terminal checks refuse (`console_trace_continuation.py`, `validate_completed_tool_turn`). TASK-42555 (a captured Retry, blocked on PR #3717) changes this shape.

**Decision needed.** Admitting these sends needs a rule that ADR-337 does not grant, so this is an owner decision. The question:

> After a reply Chatbook sends with capture off (Edit & resend, Continue, Regenerate, Retry, probably citation repair), should the next captured message in a chat with project instructions be admitted by the rewritten-tail rule extended to project context? The replaced tail would also drop the earlier turns' recorded project-context rows, and the current turn's project context would be recorded after the new user message. The history rows stay proven by the existing saved-branch checks.

Recommendation: allow it, under the rewritten tail's existing checks plus these:
- project-context rows only after the current user row;
- only `project_instruction` artifacts under the call's frozen policy;
- counted in the 256-sequence span bound and the chain.
- the completed-turn transition is still chosen when its response link holds. Otherwise, as after a Regenerate, the rewritten tail is chosen at planning, never as a retry after a validation error.

The alternatives:
- capture the actions themselves. The 2026-10-27 ruling kept Edit & resend and Regenerate uncaptured; TASK-42555 does this for Retry.
- keep refusing; **Send without capture** still sends.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's decision is recorded in the task and, if granted, in ADR-337 before any code
- [ ] #2 If granted: with Capture On and AGENTS.md, the message after each of Edit & resend, Continue, Regenerate and citation repair is sent and traced as sent, in a chat on one provider
- [ ] #3 One test per action drives it through the real controller; each is red before the change and green after
- [ ] #4 Retry of a failed reply is covered once TASK-42555 lands, or its refusal is recorded as that task's
<!-- AC:END -->
