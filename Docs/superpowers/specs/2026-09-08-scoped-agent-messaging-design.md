# Scoped agent messaging: child progress and supervisor relay

Date: 2026-09-08
Status: Draft for review; no new runtime behavior implemented
Task: [TASK-32022](../../../backlog/tasks/task-32022%20-%20Design-scoped-bidirectional-agent-messaging.md)
ADR required: yes
ADR path: [ADR-136](../../../backlog/decisions/136-scoped-child-progress-and-supervisor-relay.md)
Reason: A new child-to-supervisor API changes cross-agent communication and authority boundaries.

## Decision to review

Add an explicit, bounded progress channel from a live threaded child to its
conversation's supervisor. Recommend supervisor relay first: `report_to_supervisor`
posts and `read_agent_messages` collects; existing `send_to_agent` supplies the
response or relay. An idle supervisor is not woken. Direct peer addressing and
durable progress delivery are outside this version. This recommendation awaits
user approval; the conservative admission and wake defaults are already approved.

The ADR is the canonical contract for identity, bounds, receipts, lifecycle,
privacy, and alternatives. This document connects it to the observed code and
observable acceptance requirements. It is not an implementation plan.

## Current seam map

| Existing code | Observed behavior and proposed integration |
| --- | --- |
| `Agents/fleet_coordinator.py`: `post_steering`, `drain_steering`, `finish`, `prune_terminal` | Locked inbound child queues and retention exist. Add a separate progress inbox owned for the same conversation lifetime; do not conflate steering, completion events, and progress. |
| `Agents/agent_service.py`: `_run_one`, `_launch_fleet_child`, `check_agents`, `send_to_agent` | Primary-only fleet callbacks enforce current isolation. Bind a report-only closure to an exact child and a reader-only closure to an active primary. Current `fleet_active` also tests spawn allowance: do not reuse it to make pending reports unreadable when allowance is zero. |
| `Agents/agent_models.py`, `Agents/tool_catalog.py`, `Agents/agent_runtime.py` | Runtime tool names, schemas, callbacks, dispatch branches, and normal tool-result appends form the existing interface. Reserve both new names against catalog collisions and validate at the service boundary as well as the schema. |
| `Agents/session_todo_store.py`, `Agents/local_tool_provider.py` | Session task CRUD is bounded, stable-ID, and CAS-based. Keep task mutations separate from message delivery and preserve their local-tool approvals. |
| `Chat/console_agent_bridge.py`, `Chat/console_runtime.py` | Find the existing conversation/coordinator and native-runtime owners when wiring inbox lifetime and the shared pending allowance. Do not create a new inbox on every reply or screen mount. |
| `Agents/automatic_work_runtime.py`, `Chat/console_fleet_wake.py` | Validate current automatic ownership and filter reader eligibility to one chain. Do not create a progress wake, result claim, generation reservation, or fresh chain. |
| `UI/Console_Modules/fleet.py`, `UI/Console_Modules/agent.py` | Extend existing fleet detail composition with pending counts and an explicit progress view. Keep screen lifecycle work in the existing modules and avoid enlarging the Console screen class. |

Paths in this table are relative to `tldw_chatbook/`. They were inspected during
design discovery; detailed signatures and exact edits belong to the subsequent
implementation plan after approval.

## Example interaction and limits

1. A manual primary starts a reader and an implementation child. Both retain
   their service-generated identities and current permissions.
2. The reader calls `report_to_supervisor(message="The response uses cursor pagination; the implementation assumes page numbers.")`.
3. The tool returns a generated ID with a queued receipt. The reader continues
   independent work or includes any unresolved question in its final result.
4. While doing its own work, the primary calls `read_agent_messages()`. The result
   contains the source identity, complete report body, and remaining eligible
   count. It can collect at most four reports within the serialized result cap.
5. The primary chooses whether to call existing `send_to_agent` for the
   implementation child. If that child has finished, existing explicit
   continuation and resource admission apply. Nothing runs merely because the
   report mentioned a child ID or a suggested action.

If the primary is blocked in `wait_agents`, it cannot collect mid-wait. If it
has finished, the report waits until a later eligible primary or explicit user
inspection; a normal completion may independently cause its existing budgeted
wake. This first version does not solve urgent preemption or agent-to-agent RPC.

Automatic readers cannot consume reports from another causal chain. A manual
reader may collect prior-turn reports in the same conversation, but their bodies
remain data and do not replenish the sending chain's budget or grant tool access.

## Acceptance matrix for implementation

These are future requirements, not passing-test claims. Use actual service/loop
dispatch and controlled provider payload capture where behavior crosses layers.

| Case | Required result |
| --- | --- |
| Live threaded child, primary, inline child, absent coordinator | Only the threaded child gets the reporting capability; only the bound primary gets the reader. Inline children retain completion-only behavior. |
| Hallucinated runtime call or colliding catalog tool name | No role bypass through generic invocation, dynamic loading, or registry collision. Child cannot read the inbox, steer, wait on, or discover siblings through these runtime callbacks. |
| Forged sender/target/conversation/chain arguments | Reject unexpected arguments; real source identity comes from the captured run capability. Other conversation queues are unchanged. |
| Child from turn A survives service setup for turn B | Reports keep A's child, parent, and chain identity. They cannot be relabeled using mutable latest-turn fields. |
| Automatic reader, manual reader, unknown/foreign chain | Automatic reader collects only its chain in FIFO order. Manual reader can collect across prior turns of the same conversation. Ineligible reports stay queued. |
| Primary with zero spawn allowance or current spawning disabled | Existing coordinator reports remain readable; no child slot is spent. |
| Concurrent last-slot sends, full conversation, full runtime | Exactly the permitted reports enter; refused posts preserve accepted bodies and all counts. One busy child cannot occupy the full conversation quota. |
| Repeated send/drain from one live child | Lifetime send allowance remains finite despite pending-space reuse; terminal metadata does not leak afterward. |
| Invalid/blank/oversized text, multibyte text, unexpected arguments | Atomic, bounded refusals; no partial body, implicit coercion, or quota residue. |
| Reader formatting failure or first report exceeding effective tool-result cap | Reports remain queued; a complete returned report is never later truncated by the normal tool-result path. The UI can still show the pending body. |
| Two readers or view and reader race | Only the authoritative primary may consume; each selected envelope is collected at most once in memory. User inspection does not consume. |
| Native tool batch, fence batch, restored native continuation | Reader results follow the existing complete tool-result path. No asynchronous injection or replay of an already restored tool result removes another batch. |
| Child finish races with post; finish/prune races with read | Admission either precedes terminalization and keeps the report, or refuses. Pruning handles does not erase admitted pending reports. |
| Child cancellation/abandonment; primary cancellation; runtime replacement | Stale callbacks cannot send or collect. A view reattach alone preserves valid capabilities and pending messages. |
| Report contains approval language, tool syntax, or another agent ID | No permission-store writes, approval resolution, tool dispatch, child continuation, or target lookup from the body. A later explicit relay still uses ordinary permission checks. |
| Report while primary idle, waiting for approval, or in `wait_agents` | No new wake/provider call, interrupted wait, completion stamp, or changed drain accounting. |
| Automatic primary reads and relays | Provider/tool/continuation work remains in the same accepted chain with its existing finite reservations; receipt collection grants no dispatch authority. |
| Actual session disposal, shutdown, repeated open/close | Pending runtime allowance returns to its previous value; no retained message/receipt registry grows outside the documented bounds. |
| Restart after enqueue or collection; collection then provider failure | No inbox reconstruction or automatic replay; do not claim progress survived or was consumed by the provider. Durable completion recovery remains separate. |
| Rendered pending view and collection update | Correct count and source labels; explicit bodies escaped as untrusted text; view does not consume or mark completion seen; count updates after model collection. |
| Diagnostics and existing full run capture | Operational metadata omits bodies. Existing opt-in capture accurately records tool data without claiming durable inbox/receipt state. |

Run targeted fleet/service/runtime, automatic-work, continuation, and rendered
UI tests appropriate to the eventual patch. No full-suite run is implied. Design
review can validate contract coherence and source references, but cannot certify
these runtime outcomes.

## Review and delivery status

Design discovery verified that explicit collection avoids a new asynchronous
history-injection path, but found two existing constraints the implementation
must handle: fleet disclosure is coupled to spawn allowance, and ordinary tool
results can be truncated after dispatch. The contract separates read eligibility
from spawning and requires full-result sizing before removing queue entries.

Self-review also covered blocked-parent deadlock, late child callbacks, global
memory growth, old-chain consumption, misleading read receipts, terminal handle
pruning, navigation lifetime, and body leakage through metadata. The corresponding
rows above remain acceptance requirements until exercised against an implementation.

No runtime files, database schema, or provider behavior changed for this proposal.
After design approval, write the implementation plan and atomic Backlog tasks
before changing code. ADR-136's number was checked against 454 local branch/remote
refs and available worktree files; recheck remote/PR claims before integration.
