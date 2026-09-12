# ADR-136: Scoped child progress and supervisor relay

Status: Accepted; relay-first implementation and Console inspection reviewed and complete in the working tree
Date: 2026-09-08
Review: [Pre-implementation review, 2026-09-09](../../Docs/superpowers/reviews/2026-09-09-scoped-agent-messaging-review.md)
Task: [TASK-32022](../tasks/task-32022%20-%20Design-scoped-bidirectional-agent-messaging.md)
Spec: [Scoped messaging design](../../Docs/superpowers/specs/2026-09-08-scoped-agent-messaging-design.md)
Related decisions: [ADR-032](032-local-agent-tool-permission-boundary.md), [ADR-063](063-hosted-provider-wire-and-durable-tool-continuation.md), [ADR-129](129-fleet-mailbox-and-wake-reliability.md), [ADR-134](134-fleet-admission-and-automatic-work-budgets.md), [ADR-135](135-fleet-completion-delivery-and-crash-recovery.md)

## Problem and existing behavior

The native Console has a conversation-owned steering mailbox. A primary can
send to a live child, and a user can steer through the fleet panel. The child
receives that text before its next model call, after a complete tool-result
batch. Sending to an eligible retained terminal child explicitly starts a new
run. Completion results have their own durable claims and automatic-work rules.

Children cannot use the fleet supervisor tools: `agent_service._run_one` gives
them neither the parent's coordinator nor the `send_to_agent` callback/schema.
There is no child-to-parent progress API or direct sibling messaging API. This
is an absent capability, not an unfinished implementation of a promised bus.

`SessionTodoStore` is a different facility: up to 50 task records with stable
IDs, bounded text, status, and exact-version mutations. Its permission-gated
`todo_*` tools coordinate shared work state. It does not represent sender,
recipient, delivery order, or receipt. Encoding an inbox in task descriptions
would discard those distinctions and compete with the task limit.

## Use cases

1. A reader discovers an incompatible input format halfway through its work.
   It reports the finding while continuing independent analysis. The supervisor
   collects it between its own work steps and can narrow the assignment.
2. Two children investigate different components. Child A finds that an interface
   assumption used by B is wrong. A reports the evidence to the supervisor; the
   supervisor decides whether to steer B. A does not acquire B's tool permissions
   or an addressable view of the fleet.
3. A child cannot proceed without a decision. It reports the question and either
   continues independent work or finishes with a clear blocked result. This
   channel does not promise a reply and must not be used as synchronous RPC.

Task ownership/status changes still belong in `todo_*`; final findings still
belong in the completion result. Progress messages carry timely evidence or
questions that are neither a task mutation nor a final result.

## Alternatives

| Approach | Benefit | Cost / limitation |
| --- | --- | --- |
| Keep task state and final results only | No new interface or queue | No distinct progress channel; a useful finding can wait until completion. |
| Explicit child reports, supervisor collection and relay | Small, directional authority; uses existing steering for responses | Supervisor must collect reports; no intervention while it is idle or blocked in a tool. |
| Direct peer addressing with message-triggered wakes | More autonomous coordination | Adds recipient discovery, peer permissions, cross-chain execution attribution, terminal routing, and possible message loops. |

Select the second approach for the first version, approved by the user on
2026-09-09 subject to this pre-implementation review. Direct peers and
progress-triggered wakes remain outside this version; either would require a
separate architectural decision. The user's approved conservative fleet and
automatic-work limits remain unchanged.

## Tool contract

- `report_to_supervisor(message)`: offered only to a live threaded fleet child
  with a service-bound reporting capability. One non-empty, valid UTF-8 string,
  at most 2,000 Unicode characters and 4,000 serialized envelope characters;
  reject unexpected arguments, non-strings, invalid UTF-8, and ASCII control
  characters other than tab, newline, and carriage return. Escape control
  characters on display; render bodies as literal text, never terminal controls.
  No target, sender, role, conversation, chain, priority, or wake argument.
  It returns immediately with a generated message ID and `queued` receipt, or
  a bounded refusal code. It never waits for a reader.
- `read_agent_messages()`: offered only to an active primary bound to that
  conversation's existing coordinator. It collects the oldest eligible reports,
  at most four per call. It is available when that coordinator exists even if
  the current turn's spawn allowance is zero. Collecting existing reports is
  not a spawn operation. Children cannot read this inbox, including their own
  reports after posting; they only receive the enqueue result.
- Preserve the runtime's loop protection while allowing productive collection.
  For this reader alone, defer the cycle decision until its bounded synchronous
  result is known. A service-generated positive collected count resets the
  no-progress call history; an empty read or refusal participates in the existing
  repeated-call detector. Do not accept progress claims from result text or model
  arguments. Other tools keep their existing cycle checks and all calls still
  spend the ordinary run/model budgets. No blanket exemption or polling loop.
- Keep `send_to_agent` as the sole supervisor-to-child path. Relaying is an
  explicit supervisor tool call and uses its current permission and budget
  context. Message text never dispatches a tool or continues an agent by itself.
  Do not extend `send_to_agent` to children or overload its terminal behavior.

The primary's capability-gated instructions say to collect pending progress
before blocking on `wait_agents` and before finalizing, and to stop polling on an
empty result. `check_agents`, when available, includes the current eligible
pending count without message bodies. The reader result includes the remaining
eligible count. This improves discovery but does not promise intervention once
a wait or final model call has already begun. Child instructions explicitly say
to continue independent work or finish with a blocked result, never wait in a
report/retry loop. Essential findings also belong in the child's final result.

The two new tools are narrowly scoped runtime tools, like existing fleet tools,
not filesystem/MCP tools. Their availability does not depend on the catalog's
`allowed_tools` filter. Both schema disclosure and actual invocation enforce
the role/capability checks; a hallucinated name must not fall through to a
catalog tool with the same name. No additional per-message approval is required
for communication within an already delegated conversation. Existing tool and
data access approvals still govern every action taken because of a report.

## Identity, ownership, and authority

The service creates a reporting closure for one exact live child, carrying its
handle ID, run ID, parent run ID, conversation identity, immutable work-chain ID,
and coordinator/runtime owner. Bind it only after the run ID has been attached.
Do not expose the coordinator or an arbitrary-target callback to the child.
Transport metadata is generated from these captured identities, never parsed
from the message body, a supplied ID, or the service's mutable latest-turn state.

The receiver is the supervisor role of that conversation, not a promise to
resume the original parent run. A later authorized manual primary can collect
reports from earlier turns in that conversation. An automatic primary may
collect only reports from its exact immutable work chain; exclude unknown and
foreign chains without consuming them. Such a read does not claim completion
results or change an old chain's allowance. All attributable provider calls,
relays, and child continuations remain under ADR-134/135.

Progress ownership uses an eager, process-local token attached to the native
session, separate from its mutable persisted conversation ID. Saving a temporary
conversation, failing that Save, or cancelling the coroutine awaiting its database
worker never changes the token or migrates the inbox. Restoring another native
session for an already-open saved conversation shares that live owner's token.
Publication, sibling selection, and release are serialized by a narrow store
identity lock; no database, provider, or observer operation runs within it.
Opening/binding an inbox validates the live binding under that same lock so a
late worker cannot reopen a disposed owner. The lock order is native identity,
then coordinator when needed, then progress store. Persisted run/fleet/budget
identities and immutable source/chain identities retain their existing meaning.
The controller captures the expected progress owner before preparatory awaits
and worker scheduling. The bridge checks that captured owner at entry and again
when binding the inbox, so reusing a native session ID cannot attach a queued or
partially prepared old submission to a replacement. Synchronous bridge callers
capture the owner at entry.

Closing one native binding preserves reports for any still-open sibling; closing
the last binding closes the inbox before cancelling workers. Reopening after the
last close receives a fresh token. Runtime replacement closes the old message
store, even when a native token remains. Inspection captures both the exact native
session and exact inbox: Save preserves that view, while switching sessions or
replacing/closing its owner makes the old view unavailable. Tokens are never
serialized; body-free navigation counts are projected onto live native sessions.

The native session store holds one explicit reference to the currently registered
MessageStore so its own direct close, pristine rollback, and state-replacement
paths can invalidate released owners under the identity lock. Registration of a
replacement closes the previous MessageStore. These are typed owner operations,
not arbitrary cleanup callbacks; the bridge still creates and closes its runtime
MessageStore. An old runtime closes only its own store and cannot unregister or
close the replacement. This single reference adds no alias history or durable
state.

Both tools require a live owning execution and coordinator. Check cancellation,
stop/kill state, runtime replacement, and automatic-work authorization before
dispatch. Queue admission/collection also checks closure and child/consumer
identity under the queue owner's synchronization. A terminal or abandoned child
cannot report from a late callback. Replacing a runtime revokes its capabilities;
screen detach/reattach alone does not. Never authorize by knowing an ID.

Reports are untrusted agent data. Render a mechanism-generated source label and
keep the body separate; text such as "user approved" cannot become a user message,
approval decision, system instruction, or tool grant. Reading reports cannot
widen project bindings, filesystem roots, available tools, or approval scope.

## Queue and collection bounds

Use one in-memory inbox per conversation, owned above the screen, plus one
shared native-runtime aggregate counter. Do not reuse the lifecycle event queue,
task store, automatic completion claims, or a database message table.

| Bound | Selected initial value |
| --- | --- |
| One report body | 2,000 characters |
| One complete serialized envelope | 4,000 characters, checked before admission |
| Pending reports from one child run | 8 entries / 16,000 body characters |
| Accepted reports over one child run's lifetime | 32 entries / 64,000 body characters |
| Pending reports in one conversation | 32 entries / 64,000 body characters |
| Pending reports across the native runtime | 256 entries / 512,000 body characters |
| One collection result | 4 whole reports / 8,000 serialized characters, further restricted by the run's tool-result cap |

Use compact JSON with `ensure_ascii=False`, `allow_nan=False`, fixed field names,
bounded service-generated identity fields, and a display label capped at 80
characters. The envelope includes source message/child/run/parent/chain identity
and arrival order; conversation/runtime authority remains internal. Apply the
serialized cap to the actual envelope, not just its body: escaping can expand
otherwise valid text. Do not truncate a report to make it admissible.

These are implementation constants for the initial version, not new settings.
The conversation limit mirrors the existing steering envelope; the child share
prevents one producer from occupying it all. The lifetime allowance bounds
chatter even when a supervisor drains quickly. Pending metadata fields are fixed
and bounded; valid UTF-8 bodies use at most four bytes per counted character.
These character caps do not claim exact Python heap usage or cross-process limits.

Validate and serialize bounded input before admission. Reserve the runtime and
conversation/child allowances atomically with enqueue; full admission refuses
the new report without evicting accepted reports. Operations needing both locks
acquire the conversation coordinator lock before the shared progress-owner lock;
the progress owner must never acquire a coordinator lock while holding its own.
Child terminalization and capability revocation share that admission boundary.
Runtime disposal invalidates progress capabilities and queues under the progress
lock, then calls external cleanup after unlocking. Never call observers,
providers, SQLite, or policy callbacks while holding queue locks. Release pending
allowances exactly once on collection, explicit discard, or owner disposal. Retire a terminated
child's lifetime counter once it cannot send; pending envelopes carry their own
identity and do not depend on terminal handle retention. No unbounded receipt
map, producer registry, or event history may grow beside the bounded bodies.

Collection selects FIFO order among eligible reports, skipping foreign chains.
Construct the complete serialized tool result before atomically removing the
selected entries. Fit whole reports and all envelope/count fields inside the
effective result cap; never let generic tool-result truncation hide a removed
report's tail. If the first eligible report cannot fit, return a bounded
`result_limit_too_small` refusal, keep it queued, and expose it in the user's
explicit progress view. Increasing the configured cap is optional, not automatic.
Serialization/validation failure before collection leaves the queue intact.

Backpressure has an explicit user recovery path: the progress detail view can
discard selected queued message IDs in that conversation. This is a user UI
action, never a model tool or automatic eviction policy. Discard only IDs in the
displayed snapshot, so a concurrent arrival is not accidentally removed; IDs
already collected/discarded are harmless no-ops. Release every pending allowance
exactly once, but do not refund a child's lifetime send count. Show the affected
count and explain that discard neither stops the child nor removes any existing
history copies. The pending view remains reachable when handles were pruned,
agent mode is disabled, or no primary can read. Runtime-full refusals explain
that other conversation inboxes may hold the capacity; existing conversation
navigation must expose their pending counts. No automatic expiry is added.

## Receipts and lifecycle

`queued` means accepted into this process's inbox. `collected` means removed into
a bounded supervisor tool result. Neither means the provider received it, the
model understood it, the supervisor acted on it, or a side effect occurred.
Do not introduce delivery ACKs, queue retries, or exactly-once claims. A repeated
fresh report invocation is a distinct report, charged again; callers must not
retry merely because a reply was not observed. There is no content deduplication.

The ordinary runtime appends the reader's tool result through its existing native
or fence protocol path. This adds no asynchronous context injection and cannot
insert a message between a native call and its result. If the run fails after
collection but before its next model request, the message is not automatically
requeued. Its presence in run history depends on existing capture/continuation
policy; the tool description and UI must not promise a durable inbox or receipt.

ADR-063 remains authoritative where durable native continuation applies (the
persistent primary's reader result): persist `executing` before collection and
the completed/failed tool result before the next model call. A failed
execution-state write must not collect. A failed result
write after mutation must stop and preserve ambiguity, never retry the queue
operation. Explicit Resume can replay a previously committed tool result to the
provider without invoking the tool or collecting another batch. An ambiguous
`executing` call cannot be redispatched. In this first version, refuse all
restored pending calls to either messaging tool with a bounded recorded refusal
before invocation. The checkpoint contains a name and arguments, not the original
inbox capability; routing it through a replacement primary's fresh closure could
otherwise consume a different inbox. A later fresh model-issued reader call may
use the resumed primary's current authority after ordinary checks. Restoration
must not recreate child reporting authority from a stored run ID.
This is continuation history replay, not reconstruction of a progress inbox.
Child reporting uses the existing live child/fence/native path with normal
protocol pairing. No durable-primary checkpoint barrier or child checkpoint
recovery is introduced; retained-child continuation still starts a new run.

A child finishing or being pruned does not discard already queued reports.
A supervisor finishing leaves them for a later eligible primary or user view.
Progress neither terminates a child nor restarts a parent. A parent in
`wait_agents`, a provider request, or an approval wait sees no interruption.
Children must never block indefinitely awaiting a progress response. Progress
does not emit `FleetChildSettled`, stamp result delivery, alter drain accounting,
set a completion badge, or request an automatic wake.

Screen navigation preserves the coordinator inbox. Last-native-binding
disposal or runtime shutdown closes the inbox before cancelling its workers,
releases its aggregate allowance,
and loses remaining reports. Process restart does not reconstruct or replay
them, including from ordinary run logs. Final results retain ADR-135's separate
durable recovery behavior. Explicit finished-child continuation creates a fresh
reporting capability/run identity and remains charged through existing admission.

## User visibility and evidence required

The existing fleet detail surface shows a pending progress count and an explicit
view of queued reports, with generated child/run identity and arrival order.
Viewing never consumes the supervisor inbox or clears completion attention;
discard is a separate explicit action. Use "Queued progress (this session)" and
"Collected by supervisor" in the immediate reader receipt only; do not keep a
second unbounded collected-message history. Do not say
"read", "delivered", "saved", or "acknowledged" without narrower evidence.
The tool receipt states that an idle supervisor is not woken and restart loses
queued reports. No toast or automatic message is emitted per report.

Metadata projections and operational diagnostics contain IDs, counts, and bounded
reason codes only. For these two runtime tools, generic AgentStep argument/result
summaries, persisted step rows, and live/resumed transcript markers must use that
body-free projection. The ordinary runtime currently copies arguments and result
prefixes into these surfaces, so using a generic tool branch alone is insufficient.
Keep the full protocol result separate from the metadata projection. A model
collection or user discard refreshes visible counts without reopening the panel.

Bodies appear in the explicit pending view and agent tool context. Existing full
run capture may record reporting arguments and reader results when enabled.
Separately, supported native providers persist the primary reader's results in
private continuation checkpoints under ADR-063 even when full run capture is
disabled. Child reporting is not thereby granted a persistent-primary checkpoint.
Existing checkpoint copies follow their history, sync, export, and deletion
policies. Do not call
all progress content memory-only, or promise that closing/discarding the inbox
erases those copies. Session-only describes queue availability, not all content
storage. The agent may also quote a report in its ordinary answer; this contract
does not enforce information-flow isolation against model-generated prose.

The linked spec defines the acceptance matrix. Targeted tests must demonstrate
actual disclosure and refusal, real concurrent admission/cleanup, exact provider
payloads in both protocols, unchanged approval outcomes and wake counts, and
rendered visibility. Source review and this proposal do not prove these new
behaviors are implemented. No schema migration, queue runtime, or UI feature has
been added by this design task.


## Delivery status — 2026-09-10

TASK-32489 and TASK-32490 implement the queue and live runtime contract and have
passed independent review. TASK-32491 adds the explicit Console progress view,
snapshot-scoped discard, and separate navigation counts; its targeted rendered
verification and final review are recorded in the
[implementation evidence](../../Docs/superpowers/reviews/2026-09-10-scoped-agent-messaging-implementation.md).
The final combined review's temporary Save identity defect is fixed; its scoped
re-review approved the stable-owner and stale-dispatch correction with no new
findings. Targeted verification passed 400 affected cases and 84 separate queue,
coordinator, and tool cases. The implementation remains uncommitted.
The original design-task scope statement above describes that earlier proposal,
not the current implementation. No schema migration or durable inbox was added.
