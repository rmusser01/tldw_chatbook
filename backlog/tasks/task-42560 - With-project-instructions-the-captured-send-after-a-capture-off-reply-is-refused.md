---
id: TASK-42560
title: "With project instructions, the captured send after a capture-off reply is refused"
status: In Progress
assignee:
- '@claude'
created_date: 2026-11-26 08:07
updated_date: 2026-11-26 16:08
labels:
- follow-up
- console
- trace
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

**Owner decision (2026-11-26).** The owner was asked: "TASK-42560: in a captured chat with AGENTS.md, after Edit & resend, Continue or Regenerate (and Retry, the same family), the next message is refused with 'Trace capture could not start'. It happens even when the provider never changes. Those replies go out with capture off (per the 2026-10-27 ruling), and the rule that admits the next send after them (the rewritten-tail rule) refuses project-instruction rows. Without AGENTS.md the next send works. What should happen?"

The owner chose **"Capture those actions"**: "Reverse the 2026-10-27 ruling: Edit & resend, Continue and Regenerate are traced themselves, as Retry will be (TASK-42555). This is larger work, and it overlaps #3717."

The investigation had recommended the other option, extending the rewritten-tail rule to project context for the send after a capture-off reply; the owner did not choose it. Retry stays TASK-42555's. Impersonate is unaffected (it never adds to the transcript), and citation repair is left for a follow-up.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's decision is recorded verbatim in this task, and an ADR-337 amendment (2026-11-26, owner-approved) that defines how each action's call is admitted is written before any code
- [ ] #2 With Capture On, Edit & resend, Continue and Regenerate each dispatch a traced call of their turn, with and without AGENTS.md and with prompt mix-ins, and the message after each is sent and traced
- [ ] #3 One test per action drives it through the real controller, with only the provider handlers faked; each is red before the change and green after
- [ ] #4 Retry of a failed reply is covered by TASK-42555 (which waits on PR #3717); this task leaves Retry unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the owner's decision and the new ACs; write the ADR-337 amendment (2026-11-26) and commit it before any code.
2. Controller: a new `Chat/console_action_capture.py` freezes each action's Capture-On request the way a normal send does (the chat's effective capture policy, a policy-carrier preparation, the saved rows read before any transform, the mix-ins composed into the request), hooked into the three action bodies with a few lines each. A temporary chat with Capture On refuses the action; a trace refusal on the direct path fails the reply with the agent path's copy.
3. Trace: the run's route descriptor names the action (from the request the action froze); the runtime finds a Continue's turn behind its continuation instruction; the planner re-records the action's turn from its user row, proven as an extended rewritten tail (Continue reply and instruction, project context) and rechecked at binding; the send after a captured Continue replaces its instruction; a direct-path action call counts as its turn's first call (slot rules, completed-turn origin). New logic in new modules; minimal diffs in the files PR #3717 touches.
4. Tests through `_project_console` (real controller, bridge, gateway, trace factory; provider HTTP or the hosted handler faked): each action on the agent and direct paths, llama.cpp and hosted, with and without AGENTS.md, with mix-ins, warm and after a trace-runtime restart, then the next message; hard negatives for each new rule with mutations. Lane the file where there is headroom.
5. User Guide, size rows, lane timing, the brief's checks; list the census/test/guide rows to flip after the restack onto reds3.
<!-- SECTION:PLAN:END -->
