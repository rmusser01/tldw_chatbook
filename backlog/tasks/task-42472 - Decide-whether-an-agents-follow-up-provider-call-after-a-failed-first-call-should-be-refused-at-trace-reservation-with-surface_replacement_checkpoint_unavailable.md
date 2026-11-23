---
id: TASK-42472
title: "Decide whether an agent's follow-up provider call after a failed first call should be refused at trace reservation with surface_replacement_checkpoint_unavailable"
status: To Do
assignee: []
created_date: 2026-11-23 06:38
updated_date: 2026-11-23 07:09
labels:
- follow-up
- console
- agents
- trace
- investigation
priority: low
references:
- tldw_chatbook/Chat/console_trace_service.py
- tldw_chatbook/Chat/console_trace_runtime.py
- tldw_chatbook/Agents/activation.py
- Tests/Performance/test_console_managed_received_acceptance.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 22 of the follow-up burn-down (2026-11-23), by TASK-42412. Checked on the round-22 stack 336d15338b (dev 2f7790873a plus round 22's stacked branches); paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** In TASK-42412's local reproductions (a natural one in 1 of 18 standalone runs, and a forced one), the agent's first provider call (AGENT_FIRST) reserved its trace, and the TOOL_LOOP call that followed (`first_call=False`) was refused at reservation. The refusal was `ValueError("surface_replacement_checkpoint_unavailable")` from `prepare_current_surface_delta` (`Chat/console_trace_service.py:2927`, the branch `if replacement_range is not None and bootstrap`). It was reached through `ConsoleTraceBoundaryFactory.__call__` (`Chat/console_trace_runtime.py:816`), re-raised as `TraceReservationRefused` by `_rolled_back_refusal` (line 84), and left the reservation `not_established`. There the first call had failed because the test's own recorder (its stub provider path) raised, not a real provider. Whether a real provider error on an agent's first call leads to the same refusal on its next call, so that a user's agent run stops at trace reservation instead of reporting the provider error, is not known.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The notes record whether a real provider error on an agent's first call (not a test double) can be followed by a TOOL_LOOP call, and if so whether that call is refused with `surface_replacement_checkpoint_unavailable`
- [ ] #2 If it can, the intended behaviour is recorded (refusal with a reason the user sees, or a reservation against the failed call's checkpoint) and the code matches it, with a test shown red first for any change
- [ ] #3 If it cannot, the task closes with the code path that rules it out
<!-- AC:END -->
