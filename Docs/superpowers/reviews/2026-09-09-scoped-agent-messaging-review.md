# Scoped messaging: pre-implementation review

Date: 2026-09-09
Task: [TASK-32022](../../../backlog/tasks/task-32022%20-%20Design-scoped-bidirectional-agent-messaging.md)
Decision: [ADR-136](../../../backlog/decisions/136-scoped-child-progress-and-supervisor-relay.md)
Spec: [Scoped messaging design](../specs/2026-09-08-scoped-agent-messaging-design.md)
Scope: Review the approved relay-first direction against the current working
tree before implementing new runtime behavior. The checkout contains unrelated
and earlier orchestration changes; this pass changes design/review documents only.

## Findings and disposition

These findings concern missing or conflicting requirements in the original
draft. They are not vulnerabilities in an existing child-progress implementation;
that implementation does not exist yet. All seven are addressed in the revised
ADR/spec and their future acceptance matrix.

| ID | Priority | Finding and evidence | Design correction |
| --- | --- | --- | --- |
| M1 | P2 | **Serialized expansion can create an unreadable queue head.** The draft accepts 2,000 valid Unicode characters but caps collection at 8,000 serialized characters. A body of 2,000 NUL characters produces 12,014 characters even in the minimal compact JSON envelope. Whole-message FIFO then blocks every later eligible report. | Check the exact serialized envelope before admission, cap it at 4,000 characters, reject terminal-control input, and keep the existing body bound. Smaller configured result caps may still require explicit user recovery, never silent truncation. |
| M2 | P2 | **Successful draining can trigger loop detection.** The runtime records `(tool name, argument JSON)` before dispatch and stops on the third identical call. `read_agent_messages()` always has the same arguments, so more than two batches cannot reliably drain despite removing messages. The actual `_detect_cycle` returns `(1, 3)` for three reader calls. [Runtime source](../../../tldw_chatbook/Agents/agent_runtime.py). | Defer cycle evaluation only for this bounded reader until its trusted collection outcome. Productive collection resets no-progress history; empty/refused reads and other tools retain loop protection. Keep all run/model budgets. Test productive, empty, and mixed cycles. |
| M3 | P2 | **Bounded queues have no usable recovery from stale entries.** Old-chain reports cannot be collected by a new automatic chain; a low result cap can block even a manual reader. The draft makes the view read-only and retains messages until session disposal. Eight full idle conversations can occupy all 256 runtime entries indefinitely. | Add explicit user discard of selected queued IDs, with snapshot identity checks and exact allowance release. Keep pending views/counts reachable after handle pruning or agent-mode changes and through conversation navigation. Close inboxes before cancelling workers; define coordinator-before-progress lock order. No automatic eviction or model discard tool. |
| M4 | P2 | **Session-only wording omits an independent durable copy.** The initial text only identifies opt-in full run capture. Supported persistent-primary continuations commit the collected tool result with capture disabled and may replay it as historical context; ADR-063 also governs sync/export. [Runtime](../../../tldw_chatbook/Agents/agent_runtime.py), [store](../../../tldw_chatbook/Chat/console_chat_store.py), [ADR-063](../../../backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md). | Distinguish ephemeral queue availability from private continuation history and optional full capture. Keep primary executing/result barriers and truthful history/discard wording. Do not impose a persistent-primary checkpoint on child reporting; the store rejects persistent non-primary events. |
| M5 | P2 | **Generic step processing violates the promised body-free metadata.** Runtime tool arguments/results enter `AgentStep`; the service persists steps, and the bridge displays result prefixes in live/resumed markers and rail summaries. Full-capture settings do not remove this path. [Runtime](../../../tldw_chatbook/Agents/agent_runtime.py), [service](../../../tldw_chatbook/Agents/agent_service.py), [bridge](../../../tldw_chatbook/Chat/console_agent_bridge.py). | Use messaging-specific body-free argument/result/summary projections before callbacks or persistence. Preserve the complete bounded provider result and permitted private history separately. Test DB steps, live/resumed markers, and rail with capture disabled. |
| M6 | P2 | **An old pending read can acquire replacement queue authority on Resume.** Restoration rebuilds pending calls from stored name/arguments and dispatches via the resumed run's current callbacks. The draft checks the new capability but carries no original inbox capability in the checkpoint. [Runtime restore and dispatch](../../../tldw_chatbook/Agents/agent_runtime.py). | Refuse all restored pending messaging calls in v1 with a bounded recorded refusal before invocation. Completed results replay without queue dispatch; executing remains ambiguous. A subsequent fresh model-issued read can use current authority. No extra durable capability or receipt registry. |
| M7 | P3 | **The pull-only channel has weak discovery guidance.** A primary blocked in `wait_agents` cannot collect reports, and merely publishing another tool schema does not establish when it should read. [Current wait/check behavior](../../../tldw_chatbook/Agents/agent_service.py). | Add capability-gated instructions to collect before waiting/finalizing, include eligible counts in status/reader results, and stop polling on empty. Children continue independent work or finish with a blocked result; essential findings still belong in the final result. Keep the explicit no-interruption/no-progress-wake limitation. |

The original draft already specifies role-bound identity, conversation scope,
chain-restricted automatic reads, finite admission, no implicit approvals, and
cooperative lifecycle boundaries. Review found no reason to add direct peers,
message-triggered wakes, or another persistence subsystem to this version.

## Evidence and review method

The source review inspected actual runtime dispatch and continuation state
transitions, durable step persistence, bridge marker construction, coordinator
ownership/pruning, and controller close paths. Relevant source locations at the
reviewed working tree include:

- `agent_runtime.py`: cycle detection 751–774 and 1276–1277; restored pending
  calls 995–1031; execution barrier 1379; result checkpoint/step projection
  1639–1696.
- `agent_service.py`: `_persist` 2209–2218; child continuation kind 2752;
  primary fleet disclosure 2419–2429; `wait_agents` 3319 and `check_agents` 3441.
- `console_agent_bridge.py`: marker formatting 883–892; live step projection
  3956–3973; coordinator reuse/pruning 5048–5067; rail summary 6073.
- `console_chat_store.py`: primary-only durable checkpoint admission 6450–6451
  and checkpoint persistence 6507–6531.
- `console_chat_controller.py`: `close_session` 4130 onward. Existing worker
  cancellation alone does not establish the proposed inbox's disposal semantics.

These locations describe existing constraints, not implemented messaging code.
The new tools need dedicated integration tests after implementation.

Two small executable probes confirmed M1 and M2:

```text
len(json.dumps({"message": "\x00" * 2000}, ensure_ascii=False,
               separators=(",", ":"))) == 12014
_detect_cycle(deque([("read_agent_messages", "{}")] * 3)) == (1, 3)
```

Fresh targeted existing-runtime verification:

```text
.venv/bin/python -m pytest Tests/Agents/test_provider_continuation_runtime.py Tests/Agents/test_agent_runtime.py -q --tb=short
106 passed, 1 warning in 0.78s
Output: /tmp/task32022-design-runtime-evidence.txt
```

The warning is the environment's existing requests dependency-version warning.
No provider network request or full test suite was run. The six macOS semaphore
failures from the earlier task-store baseline were not rerun; they remain the
separate verification limitation recorded in TASK-32022.

An independent reviewer checked continuation, authority, and projections. It
confirmed M4/M5, then confirmed M6 and the primary/child checkpoint distinction.
After revision it found no remaining blocker in those sections or matching
acceptance rows. Parent review covered M1/M2/M3/M7 and overall contract coherence.

## Result and remaining limits

The reviewed relay-first design is ready for implementation planning under the
user's existing approval. No new messaging runtime or schema has been added.
The acceptance matrix now covers these findings; none is called a verified
runtime fix.

Remaining deliberate limits: no response while a parent is blocked or idle;
no durable inbox; possible loss between collection and provider consumption;
no model-understanding/side-effect acknowledgment; no direct peer addressing.
Existing private history may retain collected content. User discard recovers
queue capacity but does not erase history copies. These limits are explicit in
the tool/UI contract and do not require additional infrastructure for v1.
