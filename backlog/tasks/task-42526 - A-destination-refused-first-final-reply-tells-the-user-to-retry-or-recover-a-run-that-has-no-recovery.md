---
id: TASK-42526
title: A destination-refused first final reply tells the user to retry or recover a run that has no recovery
status: In Progress
assignee:
  - '@claude'
created_date: '2026-11-24 00:19'
updated_date: '2026-11-24 21:28'
labels:
  - console
  - providers
  - copy
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in TASK-34100.3's pre-merge review (2026-11-13) and asked of TASK-35582's owners then. It was left open when the work was ported (round 24, 2026-11-23).

A provider can return reasoning that names a different destination from the one the send was accepted for, for example a different endpoint. When that happens on a tool-free first final reply, TASK-35582's durable continuation binding check refuses the reasoning checkpoint (`destination_mismatch` and the other binding refusals). The run then fails closed. The reply stays on screen but is saved `failed`, without its reasoning. The run is FAILED, no response-recovery card appears, no dispatch checkpoint is left behind, and the next Send reaches the provider without the failed reply in its history.

The row the user sees still says otherwise. All continuation-error exits share one summary, written in `_continuation_error` in `tldw_chatbook/Agents/agent_runtime.py`: "Provider continuation could not be persisted; retry or recover the interrupted run." In this case nothing can be retried or recovered, so the row gives an instruction that has nothing behind it.

Evidence:
- the shared summary copy in `_continuation_error` (`tldw_chatbook/Agents/agent_runtime.py`);
- `Tests/UI/test_console_moonshot_reasoning_reply.py::test_destination_refused_reasoning_checkpoint_fails_the_reply`, which pins the current behaviour through the mounted Console: codes `final_persist_failed` + `destination_mismatch`, the reply saved `failed`, the run FAILED, no recovery card and 0 dispatch checkpoints;
- the Moonshot Kimi bullet in `Docs/User_Guide/console.md`, which quotes that row and says "No response-recovery card appears and there is no run to recover".

Options:
1. Offer recovery for this case: a recovery action that re-sends the turn. The cost is a second billed provider call for a reply that already arrived, plus a recovery path for a custody refusal that TASK-35582 built to fail closed.
2. Change the copy for this case only. The runtime already tells the exits apart by reason and store code, so an exit with no dispatch recovery can say what happened (the reply was kept as failed, without its reasoning, because it named a different destination) and what to do next (send again). Every other exit keeps the current copy.

Recommendation: option 2. The binding refusal is a deliberate fail-closed custody check, the reply is already on screen, and Send already works. Telling the user so is accurate, costs no extra provider call and needs no new recovery state. The question is open to TASK-35582's owners; TASK-34100.3's notes record it.

Owner decision 2026-11-24: change the copy (option 2), not build a recovery. Recorded in the round-26 session as "4 ok", approving this task's recommendation. TASK-35582 is the related task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 TASK-35582's owners choose between offering recovery and changing the copy for a destination-refused first final reply, and the choice is recorded in this task
- [ ] #2 A destination-refused first final reply's row tells the user only what Console can actually do: either the chosen recovery works from that row, or the copy names what happened and that sending again continues the chat
- [ ] #3 Every other continuation-error exit keeps its current behaviour and copy
- [ ] #4 A mounted Console test pins the chosen behaviour, red before the change and green after
- [ ] #5 The Moonshot Kimi bullet in Docs/User_Guide/console.md describes the chosen behaviour
<!-- AC:END -->

## Implementation Plan

1. Scope the case as the runtime already classifies it: a `final_persist_failed` exit on a tool-free first final (no active checkpoint, the same test `_keep_reply_without_replay` uses) whose store failure is one of TASK-35582's binding refusals (`host_target_unavailable`, `destination_unavailable`, `destination_mismatch`, `binding_erased`, `managed_finalizer_unavailable`, the set `test_destination_binding_refusals_still_fail_closed` already names).
2. Put that set and the summary choice beside the other code sets in `Chat/provider_continuation_diagnostics.py`; `_continuation_error` asks it for the summary. The behaviour (RUN_ERROR, reply saved `failed`, no recovery) does not change; only the summary, and so the row, does.
3. New copy, plain: what happened (the reply's reasoning could not be matched to the endpoint the message was sent to, so the reply was kept as failed without it) and what to do (send again to continue the chat).
4. Tests: change the mounted Moonshot test to pin the new row (red at base first); in `Tests/Agents/test_continuation_error_reasons.py` pin the old summary on every other exit (each exit test already there, plus the same binding refusals after a tool run and the managed-seal refusals) and the new summary for each binding refusal on a first final.
5. Update the Moonshot Kimi bullet in `Docs/User_Guide/console.md`.
