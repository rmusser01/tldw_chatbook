---
id: TASK-42567
title: "A captured sub-agent's preset fallback after a partial reply fails with the generic trace-capture copy"
status: To Do
assignee: []
created_date: 2026-11-26 15:59
labels:
- follow-up
- console
- agents
- trace
priority: low
references:
- tldw_chatbook/Chat/console_trace_child_fallback.py
- tldw_chatbook/Agents/agent_runtime.py
- backlog/decisions/337-console-reference-backed-semantic-trace-ledger.md
dependencies:
- TASK-42564
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Raised by the round-27 re-review of TASK-42564 (2026-11-26).

**What happens.** With Capture On, a spawned child's call streams part of a reply, then fails with a transient error, and its in-place retries fail too. ADR-200 then switches to the preset's fallback, because no tool batch was proposed. The ADR-337 amendment of 2026-11-26 admits a child's fallback chain only when no earlier call of the child started a response, so the trace refuses the fallback's first call in the reservation transaction. The refusal rolls back and nothing is reserved or sent, but the child ends on the generic "trace capture could not be saved, so this request was not sent". That is the copy shape TASK-42556 replaced with a visible skip.

**A second consequence.** By the time the trace refuses, the runtime has already selected the alternate. ADR-200 persists the selection before dispatch, so the child run's durable `active_fallback_index` names the alternate, and "Continuing a finished sub-agent" resumes on a model that was never asked.

**Why it is not a one-line fix.** The fallback runtime's `refuse` hook (TASK-42564) decides before the call, but it does not see that a failed call started a response. The streaming adapter would have to record it per child thread, and the Console's refusal would become a callable. A real-controller test needs a transient mid-stream failure fixture that `_spawn_turn` does not have.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With Capture On, a child's preset fallback after a partial reply is a visible skip in the child's steps, with a reason that says a partial reply came first, before any reservation or provider call
- [ ] #2 When that fallback is refused, the child run's selection is not persisted: `active_fallback_index` and the resolved target still name the target the child actually used, so continuing it resumes there
- [ ] #3 Capture Off is unchanged: the child still falls back
- [ ] #4 A test through the real controller, red before and green after, on a PR lane
<!-- AC:END -->
