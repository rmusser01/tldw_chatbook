---
id: TASK-41652
title: "No test covers prompt mix-in binding for a spawned child run"
status: Done
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- agents
- tests
priority: low
references:
- Tests/Chat/test_console_spawn_subagent_trace_capture.py
dependencies:
- TASK-41310
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** ADR-286 binds operational prompt mix-ins to the conversation (schema v86). TASK-41310's child segments exercise the parent and child routes, but no test says whether a spawned child gets the parent conversation's mix-ins.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The intended behaviour (child inherits or does not) is stated in ADR-286 or this task's notes
- [x] #2 A test spawns a child from a conversation with a mix-in and asserts that behaviour
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new decision; ADR-286 is silent on sub-agents, so its answer (from rules 5 and 6) is recorded as a dated clarification amendment.

1. Read ADR-286 for what a spawned child should get.
2. Add a mix-in to the TASK-41310 spawn harness (the prompt mix-in journeys' local source and `ConsolePromptMixinService`) and assert, Capture On and Off, that the parent's requests carry the composed message and the child's carry no body.
3. If the code disagrees with the ADR, fix red-first; otherwise pin the current behaviour.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Intended behaviour: the child does not inherit.** ADR-286 does not name sub-agents, but rule 6 composes a mix-in into the user's current message only (reused through the parent's own tool continuations), and rule 5 says machine-originated work does not newly resolve human-selected modifiers. A child's request is its own system prompt plus the task the parent model wrote, so it gets no mix-in body and binds no mix-in attempt. The answer the user reads is the parent's, still composed under the mix-ins. Recorded as a dated clarification in ADR-286 ("Amendment 2026-11-18"). Not an owner product choice: the ADR's rules decide it.

**Current code already does this; no product change.** The controller composes the mix-in into the parent's provider messages only; the child's seed is built from its task; the trace runtime skips the mix-in attempt binding for a child route (TASK-41310 B).

**Test:** `Tests/Chat/test_console_spawn_subagent_trace_capture.py::test_a_spawned_child_never_gets_the_chats_mix_ins[capture_on|capture_off-native|fenced]`. One enabled Before mix-in on the chat; asserts both parent requests' first user row is the body, a blank line, then the draft (no other row carries it), the child's requests contain no body, the child run is `done`, no boundary errors or refusals.
- Passes on dev d0486978dd and here (characterization, 4/4).
- Mutation: dropping the child guard on the trace runtime's `bind_attempt_trace_call` (`console_trace_runtime.py`, `child_turn_revision_id is None and`) fails `[capture_on-fenced]` with `TraceReservationRefused('mixin_trace_owner_unavailable')`, child run `error`. The native spawn cannot see it (the parent's dispatch checkpoint is handed to the continuation before the child exists), which is why TASK-41310 B listed this guard as uncovered; the fenced variant closes that gap.

**Files:** `Tests/Chat/test_console_spawn_subagent_trace_capture.py` (harness `mixin`/`fenced_spawn` options, `_enable_mixin`), `backlog/decisions/286-console-prompt-mix-ins.md`.
<!-- SECTION:NOTES:END -->
