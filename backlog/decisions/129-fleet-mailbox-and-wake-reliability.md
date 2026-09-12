# ADR-129: Fleet mailbox and wake reliability

Status: Accepted
Delivery-policy amendment: [ADR-135](135-fleet-completion-delivery-and-crash-recovery.md)
selects incremental notifications and two bounded wake slots, implemented in
TASK-32036/32037.
Date: 2026-09-07
Related review: [Agent orchestration review](../docs/agent-orchestration-review-2026-09-07.md)
Related design: [Supervisor fleet](../../Docs/superpowers/specs/2026-08-08-supervisor-agent-fleet-design.md)

## Decision

Preserve the current conversation-owned supervisor/worker architecture,
primary-only steering, explicit finished-agent continuation, existing approval
authority, and in-memory retention. Repair resource and delivery-state gaps:

- A child mailbox admits at most 32 pending entries and 64,000 text characters.
  Admission checks and enqueue happen under the coordinator lock. A full queue
  refuses the new entry without dropping accepted entries. Producers explain
  the refusal; the Console keeps the draft. The 4,000-character per-message
  limit remains unchanged.
- Retention measures the serialized transcript and unread steering together.
  Oversized history is refused whole; native tool pairs are never truncated.
  Retained history is deeply copied both on admission and on read.
- A terminal handle reports unread steering separately from the active queue,
  with whether its retained transcript is currently available. A final answer
  is not silently restarted to consume late steering. The UI states when a
  supervisor continuation can recover unread messages and when it cannot.
- Lifecycle events are ephemeral alongside handles. Pruning a terminal handle
  also removes its undrained events; events for surviving handles remain FIFO.
  Durable run rows and the existing bridge fanout remain the production truth.
- TASK-18312 follow-up: pruning keeps at most 256 terminal identity records
  (handle ID, optional run ID, terminal status) per conversation coordinator,
  evicting oldest first. These contain no task, result, error, or transcript
  text and grant no continuation. `send_to_agent` checks this record after
  active/retained/unpruned terminal resolution and before its scoped DB run-ID
  fallback. Known pruned children receive a not-retained refusal. DB-only
  terminal rows do not imply an earlier session; unknown-ID copy explains that
  old handle IDs may expire and suggests the durable run ID. Identity records
  remain process-local and do not change run history storage.
- A refused or raised automatic wake waits at least one second before its
  next automatic attempt. A delayed retry is maintained while work is pending.
  Other ready conversations can proceed during that delay. Manual work retains
  priority. ADR-135 replaces this decision's original global serialization with
  two bounded wake slots and a manual reserve. No new tool grants, automatic
  approval, or per-conversation parallel wake is added.

## Alternatives and consequences

An unbounded mailbox and immediate recursive retry are simple but permit
unbounded memory, provider-readiness churn, and starvation. Dropping old messages
would hide failed delivery, so new admission is refused instead. Persisting
mailboxes or automatically continuing a finished child would introduce new
replay and execution semantics; those belong to a separate design.

The queue sizes allow several substantial corrections without allowing a
stalled child to accumulate unlimited context. Fixed retry delay is sufficient
for the local readiness failure and avoids a new configurable backoff system.
The delay does not promise immediate recovery after provider settings change.
The resource limits are implementation constants; the existing configurable
retention ceiling still controls the full retained payload.

This ADR does not claim exactly-once tool execution across crashes, introduce
global fleet budgets, or settle direct peer messaging. TASK-32019 through
TASK-32022 cover those proposals independently.
