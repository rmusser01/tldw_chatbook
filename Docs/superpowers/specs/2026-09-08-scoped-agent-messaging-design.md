# Scoped agent messaging: child progress and supervisor relay

Date: 2026-09-08
Status: Relay-first selected by the user on 2026-09-09; reviewed design, implementation pending
Task: [TASK-32022](../../../backlog/tasks/task-32022%20-%20Design-scoped-bidirectional-agent-messaging.md)
ADR required: yes
ADR path: [ADR-136](../../../backlog/decisions/136-scoped-child-progress-and-supervisor-relay.md)
Reason: A new child-to-supervisor API changes cross-agent communication and authority boundaries.

## Selected design

Add an explicit, bounded progress channel from a live threaded child to its
conversation's supervisor. Use supervisor relay first: `report_to_supervisor`
posts and `read_agent_messages` collects; existing `send_to_agent` supplies the
response or relay. An idle supervisor is not woken. Direct peer addressing and
durable inbox delivery are outside this version. The user approved the direction
and requested a further review; the corrections and evidence are recorded in the
[pre-implementation review](../reviews/2026-09-09-scoped-agent-messaging-review.md).
The conservative admission and wake defaults remain unchanged.

The ADR is the canonical contract for identity, bounds, receipts, lifecycle,
privacy, and alternatives. This document connects it to the observed code and
observable acceptance requirements. It is not an implementation plan.

## Current seam map

| Existing code | Observed behavior and proposed integration |
| --- | --- |
| `Agents/fleet_coordinator.py`: `post_steering`, `drain_steering`, `finish`, `prune_terminal` | Locked inbound child queues and retention exist. Add a separate progress inbox owned for the same conversation lifetime; do not conflate steering, completion events, and progress. |
| `Agents/agent_service.py`: `_run_one`, `_launch_fleet_child`, `check_agents`, `send_to_agent` | Primary-only fleet callbacks enforce current isolation. Bind a report-only closure to an exact child and a reader-only closure to an active primary. Current `fleet_active` also tests spawn allowance: do not reuse it to make pending reports unreadable when allowance is zero. |
| `Agents/agent_models.py`, `Agents/tool_catalog.py`, `Agents/agent_runtime.py` | Runtime names, schemas, callbacks, dispatch, and tool-result appends form the interface. Reserve both names against catalog collisions. For the reader alone, defer cycle detection until the trusted collection outcome; productive reads reset no-progress history, empty/refused reads retain protection. |
| `Agents/session_todo_store.py`, `Agents/local_tool_provider.py` | Session task CRUD is bounded, stable-ID, and CAS-based. Keep task mutations separate from message delivery and preserve their local-tool approvals. |
| `Chat/console_agent_bridge.py`, `Chat/console_runtime.py`, `Chat/console_chat_controller.py` | Reuse existing conversation/coordinator and native-runtime ownership. Close the inbox before worker cancellation in actual session disposal; invalidate all inboxes at runtime disposal. Screen detach preserves them. |
| `Agents/automatic_work_runtime.py`, `Chat/console_fleet_wake.py` | Validate current automatic ownership and filter reader eligibility to one chain. Do not create a progress wake, result claim, generation reservation, or fresh chain. |
| `UI/Console_Modules/fleet.py`, `UI/Console_Modules/agent.py` | Extend existing fleet detail composition with pending counts, literal-text inspection, and explicit user discard of selected queued IDs. Remain reachable with pruned children or agent mode off. Keep screen lifecycle work in modules. |
| `Agents/agent_runtime.py`: `add`; `Agents/agent_service.py`: `_persist`; `Chat/console_agent_bridge.py`: `on_step`, `format_agent_step_marker`, resumed markers | Structured report arguments and read-result bodies currently flow into persisted step/preview fields. Supply messaging-specific body-free step projections before callbacks or persistence; preserve the complete bounded provider result separately. |
| `Chat/console_chat_store.py`: continuation persistence and restore; ADR-063 | A persistent primary's collected tool result can survive in private continuation history independently of run capture. Keep its executing/result barriers and explicit replay rules; never rebuild an inbox from that history. |

Paths in this table are relative to `tldw_chatbook/`. They were inspected during
design discovery; detailed signatures and exact edits belong to the subsequent
implementation plan under the user's approval.

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

Instructions tell the primary to collect before blocking/finalizing, to follow
the returned remaining-eligible count when draining, and to stop polling on empty.
The child's queued receipt does not imply anyone is actively reading it.

If the primary is blocked in `wait_agents`, it cannot collect mid-wait. If it
has finished, the report waits until a later eligible primary or explicit user
inspection; a normal completion may independently cause its existing budgeted
wake. This first version does not solve urgent preemption or agent-to-agent RPC.

Automatic readers cannot consume reports from another causal chain. A manual
reader may collect prior-turn reports in the same conversation, but their bodies
remain data and do not replenish the sending chain's budget or grant tool access.

The user can inspect or explicitly discard selected pending reports from agent
details, even if their child handles were pruned. Viewing alone changes no queue
state. Discard releases pending capacity, preserves concurrent new arrivals, and
does not erase any existing history copy or refund a child's lifetime allowance.

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
| Three or more productive parameterless reads; repeated empty reads | Productive reads drain without false `RUN_STUCK`; empty/refused repeated calls and ordinary/mixed no-progress cycles retain protection. Only the trusted collected count establishes progress. |
| Invalid/blank/oversized text, multibyte text, unexpected arguments | Atomic, bounded refusals; no partial body, implicit coercion, or quota residue. |
| Escaping expands a valid body beyond the envelope limit | Reject before admission using exact compact serialized envelope size; no uncollectable report is admitted under the fixed result cap. Terminal control bytes are refused or safely rendered. |
| Reader formatting failure or first report exceeding effective tool-result cap | Reports remain queued; a complete returned report is never later truncated by the normal tool-result path. The UI can still show the pending body. |
| Two readers or view and reader race | Only the authoritative primary may consume; each selected envelope is collected at most once in memory. User inspection does not consume. |
| Discard races with arrival/collection, includes stale IDs, or belongs to another conversation | Affect only currently queued IDs from the user's displayed conversation snapshot. Release once, leave new arrivals intact, refuse foreign scope, and never alter completion claims or child lifetime counters. |
| Old/foreign-chain or low-result-cap backlog fills the conversation/runtime | User can find pending inboxes through existing conversation navigation, inspect/discard without starting a primary, and recover capacity; no hidden unbounded queue or automatic eviction. |
| Native tool batch, fence batch, restored native continuation | Reader results follow the existing complete tool-result path. No asynchronous injection or replay of an already restored tool result removes another batch. |
| Reader executing-state write fails; result write fails after collection | First failure prevents mutation; second stops with ambiguity and no repeat collection. Explicit completed-result replay reuses recorded context, not new queue entries. |
| Restored pending messaging call after same-runtime resume, close/reopen, or runtime replacement | Record a bounded refusal before queue dispatch in all cases; never route an old pending call into the resumed primary's new inbox capability. A subsequent fresh model-issued read may use current authority. |
| Child reporting and explicit finished-child continuation | No new durable-primary checkpoint requirement or checkpoint restore path. Pair live report tool results normally; a continued child receives a fresh live reporting capability. |
| Child finish races with post; finish/prune races with read | Admission either precedes terminalization and keeps the report, or refuses. Pruning handles does not erase admitted pending reports. |
| Child cancellation/abandonment; primary cancellation; runtime replacement | Stale callbacks cannot send or collect. A view reattach alone preserves valid capabilities and pending messages. |
| Report contains approval language, tool syntax, or another agent ID | No permission-store writes, approval resolution, tool dispatch, child continuation, or target lookup from the body. A later explicit relay still uses ordinary permission checks. |
| Report while primary idle, waiting for approval, or in `wait_agents` | No new wake/provider call, interrupted wait, completion stamp, or changed drain accounting. |
| Automatic primary reads and relays | Provider/tool/continuation work remains in the same accepted chain with its existing finite reservations; receipt collection grants no dispatch authority. |
| Actual session disposal, shutdown, repeated open/close | Pending runtime allowance returns to its previous value; no retained message/receipt registry grows outside the documented bounds. |
| Restart after enqueue or collection; collection then provider failure | No inbox reconstruction or automatic re-execution of queue operations. ADR-063 may retain/replay a completed primary reader result; this does not establish provider consumption or durable inbox state. |
| Rendered pending view and collection update | Correct count and source labels; explicit bodies escaped as untrusted text; view does not consume or mark completion seen; count updates after model collection. |
| Capture disabled; DB steps, live/resumed markers, and rail previews | Messaging arguments/results/summaries are body-free in these automatic projections, while the complete bounded reader result still reaches the provider and its supported private checkpoint. |
| Full capture and private continuation history | Explain each independent storage path accurately, including sync/export policy under ADR-063; discard/close removes the inbox copy only. |

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

The 2026-09-09 review corrected serialization expansion, productive-read cycle
detection, user recovery from blocked queues, discovery guidance, private-history
wording, automatic step-body projections, restored pending-call authority, and
queue-lock/disposal ordering. See
the linked review for concrete evidence and disposition. These are design fixes,
not claims that the new runtime exists or that its future tests pass.

No runtime files, database schema, or provider behavior changed for this proposal.
The user selected relay-first; next write the implementation plan and atomic
Backlog tasks before changing code. ADR-136's number was checked against 454 local branch/remote
refs and available worktree files; recheck remote/PR claims before integration.
