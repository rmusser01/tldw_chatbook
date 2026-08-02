---
id: TASK-1861
title: 'Approval: refusing one call is overwritten by approving another of the same tool'
status: Done
assignee: []
created_date: '2026-08-01 21:30'
labels:
  - console
  - agents
  - security
  - regression
dependencies:
  - TASK-1845
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**Fails open.** The approval card offers a decision per call, but the pipeline could not honour it: a refusal and an approval of the same tool name resolved last-write-wins on a single name-keyed stamp. Refuse `secrets.md`, approve `spec.md`, and the surviving stamp is the approval — the file the user explicitly refused is read.

Introduced by TASK-1845's per-call re-key. Before it, `_collapse_pending_calls` grouped by name and rendered ONE row, so two verdicts could never disagree. Splitting the rows created decisions the enforcement layers could not express.

Both consumers are name-keyed by contract, and correctly so: `builtin_gate.stamp` records a grant against a tool NAME because a session/always grant belongs to a tool, and `MCPToolProvider.apply_batch_decisions` takes llm_names. The gap was that `build_tool_review_hook` returned a flat `{name: "proceed"}` to the runtime, so **no** layer carried the per-call refusal — even though the runtime already resolves `call_id` before name and turns a non-"proceed" verdict into the call's result without dispatching it.

Observed directly:

```
STAMPED ON THE GATE: [('read_file', 'deny'), ('read_file', 'approve_once')]
VERDICTS RETURNED TO RUNTIME: {'read_file': 'proceed'}
```

A shipped comment asserted the opposite ("the per-call refusals are still enforced by the runtime, which reads the call-id keys directly"). That was never true.

Second half of the same defect: MCP rows never carried a `call_id` at all. `_collect_mcp_pending` called `provider.pending_gate_for(call.name, call.args)`, dropping the id at that boundary, so every MCP row collapsed by name into one `xN` row with one verdict — the exact defect the re-key fixed for built-in tools and left standing for MCP.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Refusing one call and approving another of the same tool refuses only the refused target, in either decision order
- [x] #2 A refusal is never stamped against the tool name while a sibling call of that name is approved
- [x] #3 A call the runtime cannot address individually (no `call_id`) refuses every same-name call in the batch rather than none
- [x] #4 MCP pending rows carry their `call_id`, so MCP calls are one decision per target
- [x] #5 The refusal the model receives names the user as the actor and the tool refused
- [x] #6 Tests cover both decision orderings, and are mutation-verified to fail when either half is reverted
- [x] #7 Per-call rows approved at different SCOPES keep the broadest scope the user chose
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Refusals are enforced at the review hook, keyed per `call_id`; stamps carry only what was APPROVED. Stamping an approval while a sibling was refused is safe because the refused call is stopped before dispatch and never reaches `invoke()`. When every call of a name was refused there is no approval to preserve, so `deny` is still stamped as defense in depth.

Rows with no `call_id` fall back to the name key, which stops every same-name call — fail-closed, and the only honest option when the runtime cannot tell them apart.

**AC #6 exists because the first version of the ordering test was vacuous.** It asserted `("read_file", "deny") not in stamped` with the refusal decided FIRST, so a later approval overwrote it and a mutation that stamped every verdict regardless still passed. The deny-decided-LAST case is a separate test; both are mutation-verified.

Also corrects `_fence_call` -> `parse_tool_call` in three comments (two of them shipped): the referenced function never existed.

## Review findings

**A stale test double, and a sweep that missed the obvious file.** Threading `call_id` changed `pending_gate_for`'s signature, and `Tests/Chat/test_console_chat_controller.py`'s `_FakeReviewProvider` still had the 2-arg form, so every MCP review-hook test raised TypeError. My regression covered the approval, agents, bridge and MCP suites but not the controller's OWN test file. The double now mirrors the real provider; no compatibility shim, because the protocol is in-repo only and a shim would hide exactly this drift.

**Scope, not just allow/refuse.** Per-call rows can be approved at different scopes while only ONE scope per name can be stamped (a session/always grant belongs to a tool). Last-write-wins silently downgraded "Approve for session" to "approve once" whenever a later row of the same tool was approved once -- dropping the grant and re-prompting next call. The broadest chosen scope now wins (`always_allow` > `approve_session` > `approve_once`): choosing "for session" on any call of a tool IS choosing to grant that tool for the session, which is what the control means and what its label says.
<!-- SECTION:NOTES:END -->
