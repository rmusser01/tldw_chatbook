---
id: TASK-26001
title: 'Agent loop: graceful wrap-up before budget exhaustion'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:43'
updated_date: '2026-09-01 01:11'
labels:
  - agents
  - reliability
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A run that exhausts its budget dies with a bare error and no answer. Verified on origin/dev: Agents/agent_runtime.py:1169-1179 checks four budgets at the loop top and on breach adds STEP_ERROR "step budget exhausted" then returns RUN_STUCK - the user gets nothing usable from work already done. Hermes appends a one-time wrap-up notice to the newest tool message at roughly 80 percent of the wall budget (cache-safe, no synthetic user turn) and on exhaustion makes one tools-stripped call so the user gets a summary instead of a dead run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When a run passes a configurable fraction of its wall or step budget, the model is told once, without inserting a synthetic user turn
- [x] #2 The notice is attached so it does not break the prompt-cache stable prefix - verified and recorded in the notes
- [x] #3 On budget exhaustion the loop makes one final model call with tools removed, and its output is presented as the run result
- [x] #4 The final wrap-up call is itself bounded and cannot loop or spawn tools
- [x] #5 If the wrap-up call fails, the run still terminates honestly with the existing exhaustion message rather than hanging
- [x] #6 Budget-exhausted runs remain distinguishable from successful completion in the run record
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. Additions at the loop's existing terminal branches; no new seam.

1. Warning rides the NEWEST tool-result message (native role:"tool" or the fence-protocol user message) -- appending to the last message keeps every earlier byte identical, so the provider prompt-cache prefix survives; a synthetic user turn would break it AND put words in the user's mouth. If the newest message is not a tool result, delivery waits for the next iteration.
2. One wrap-up call at each of the four budget-exhaustion branches, with EMPTY tool schemas and the response's tool calls ignored, over the COHERENT prefix -- not raw messages, which can end inside a half-answered batch at a mid-batch step exhaustion.
3. Record the honest exhaustion step FIRST, keep the status RUN_STUCK, and put the summary only in final_text -- an exhausted run must never read as success.
4. Skip the wrap-up mid-continuation and on cancel; a failure inside it costs only the summary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
A run that hits its budget now ends with a summary instead of a dead stop, and the model is warned once as the budget approaches.

**The warning (AC#1/#2).** Delivered at a configurable fraction (`RunBudget.budget_warning_fraction`, default 0.8) of whichever budget dimension is furthest along, by APPENDING to the newest tool-result message -- native `role:"tool"` or the fence-protocol user-role result. Appending to the last message leaves every earlier byte identical, so the provider-side prompt-cache prefix survives; that property is pinned by a test asserting `first_msgs[:len(prev_msgs)] == prev_msgs` across the delivery boundary. No synthetic user turn is ever inserted. When the newest message is not a tool result, delivery simply waits for the next iteration, and it happens at most once per run.

**The wrap-up (AC#3-#6).** Each of the four exhaustion branches (step, model-turn, wall-clock, token) now makes one final model call with EMPTY tool schemas over the coherent prefix plus a single instruction message. Any tool call in the response is ignored -- it cannot loop or spawn (AC#4). The honest exhaustion step is recorded BEFORE the attempt and the status stays RUN_STUCK; only `final_text` carries the summary, so an exhausted run remains distinguishable from success (AC#6). A wrap-up failure is traced and costs only the summary (AC#5). Skipped mid-continuation (a wrap-up without the in-flight checkpoint would trip provider-continuation validation) and on cancel.

**Two things the tests caught during the work, recorded because both generalize:**

1. My first warning tests used three identical `calculator({})` fence calls and got RUN_STUCK -- from the (correct) cycle detector, not from anything under test. Scripted tool calls in loop tests need varying arguments or they collide with the loop-detection threshold.

2. The fleet-continuation coherence property (`test_property_final_messages_end_at_a_coherent_boundary`) failed against the first implementation, and it was RIGHT: the wrap-up sent raw `messages`, which at a mid-batch step exhaustion ends inside a half-answered native `tool_calls` pair -- exactly the shape the property exists to keep away from providers. The wrap-up now uses `messages[:coherent_len]`. The property test gained a narrow carve-out describing the new contract: on RUN_STUCK, the model's last view may be the coherent prefix plus the wrap-up instruction, and the instruction must never be retained in the transcript.

Verified non-vacuous by mutation: removing the wrap-up call fails 3 of the 8 tests.

`Tests/Agents/` holds at the same 15 baseline failures (2296 passing); `Tests/App/`, `Tests/MCP/`, `Tests/Metrics/` unchanged at the 2 known MCP baselines.

**Files:** `tldw_chatbook/Agents/agent_runtime.py`, `tldw_chatbook/Agents/agent_models.py`, `Tests/Agents/test_budget_wrapup.py` (new), `Tests/Agents/test_fleet_continuation.py` (property updated to the new contract).
<!-- SECTION:NOTES:END -->
