# ADR-184: Server-executed `agent_task` automations — message references and approval escalation

- **Status:** Proposed (awaiting owner ruling on decisions 1 and 2)
- **Date:** 2026-09-12
- **Amends:** [ADR-077 — Server-offloaded scheduled agent tasks](077-server-offloaded-scheduled-agent-tasks.md) (phase-2 scope); relates to tldw_server issue #2805
- **Task:** TASK-18940 AC#2 (rest)

## Context

ADR-077 phase 1 delivered the server-execution seam: definitions are authored
through the control plane, `recurring_question` executes end-to-end with
durable runs, audit, notification pass-back, run-now, and the authoring-time
provider bound (tldw_server TASK-13234). Two deliberate phase-1 refusals keep
`agent_task` (tool-using agent automations) from executing:

1. **The message is redacted at rest.** The automation service replaces
   `input.message` with `message_redacted`/`message_ref`/`message_preview`
   metadata (`metadata_only` policy) before the definition row is persisted.
   The consumer therefore finds no usable prompt and records an honest failed
   run. This was a privacy decision, honored end-to-end, and explicitly not
   reversed during the phase-1 reviews.
2. **Server-side tool use has no approval enforcement.** `approval_policy` is
   validated at authoring but enforced nowhere; the phase-1 executor boundary
   refuses tool calls outright. Unattended scheduled execution with
   side-effecting tools without an escalation design would violate the
   approval stance ADR-077 was built on.

tldw_server issue #2805 tracks both; it requires a design/ADR before code.

## Decision

### 1. Message durability — side-store message references *(recommended; judgment call)*

Persist the raw message in a **separate, access-controlled message store**
keyed by the definition's `message_ref` (issue #2805's "option 2"), keeping
the `metadata_only` redaction in the scheduled-tasks DB untouched:

- The store is owner-scoped (`user_id` + `message_ref`), encrypted at rest
  with the same envelope as other server-side secret material, with its own
  retention window (default: delete when no non-archived definition
  references the ref; archived definitions retain the ref for audit but the
  payload may be purged on a TTL).
- Write path: definition create/update writes the raw message to the store
  in the same transaction boundary as the definition row and keeps only the
  ref in the row. Delete/archive flows leave the store's TTL to reclaim.
- Read path: ONLY the `agent_task` executor resolves the ref at dispatch
  time, in-memory; logs, API projections, previews, audit events, and run
  rows never carry the raw message (they already carry the ref).
- The consumer's "no usable persisted prompt" failure stays as the honest
  outcome when a ref cannot be resolved (purged TTL, cross-instance store
  miss): a failed run with a precise reason, never a silent skip.

**Why not the alternative (move redaction to API-response boundaries):** it
simplifies the executor but silently changes what the scheduled-tasks DB,
its backups, exports, and any replication contain — raw user prompts in the
same tables as schedules and audit history. That is exactly the posture
phase 1 refused; reversing it should be an explicit owner choice, not an
implementation convenience. If the owner prefers this option, the ADR flips
to it and the store design above is dropped.

### 2. Approval escalation — side-effect floor first, queued escalation second *(recommended; judgment call)*

Phase 2 ships in two steps:

**Step 1 (lands with this ADR):** `agent_task` runs execute with a
**read-only tool envelope** — tools whose capability declarations are
side-effect-free (read/search/list style). Any side-effecting tool call
**fails the run with a new terminal outcome `approval_required`**: the run
row records the attempted call and the result notification carries the
pending request. Nothing escalates implicitly; the floor from phase 1 is
preserved and made precise.

**Step 2 (follow-up, its own task):** per-definition `approval_policy`
`escalate` mode — a side-effecting call suspends the run, emits an
approval-request notification through the existing pass-back channel, and
the client's approval surface (Console approvals / notification actions)
approves or denies; an approved run resumes within a bounded window
(approval TTL), an expired/denied run terminates as `approval_required`
denied/expired. Runs never hold credentials while suspended.

The judgment call: whether step 1's floor is acceptable to ship at all, or
whether `agent_task` execution stays fully refused until step 2's complete
escalation loop exists. Recommendation: ship step 1 — it converts an
absolute refusal into an honest, bounded capability (monitoring/analysis
automations work; anything touching state reports precisely why it didn't
run) without weakening the approval stance.

### 3. Executor registration and capabilities truth

With 1 (and step 1 of 2) landed: register the `agent_task` executor, lift
the consumer's `family_not_wired_for_execution:agent_task` skip, and flip
the execution-certification capability matrix so `agent_task` reports
`execute: available` only when both the message store and the read-only
envelope are operational — the same capability-truth discipline phase 1
applied to the scheduler feed and run-now.

## Consequences

- The scheduled-tasks DB keeps its at-rest privacy posture; raw prompts live
  only in the encrypted, owner-scoped, TTL-bounded message store.
- `agent_task` automations become executable in a bounded envelope; the
  "permanent execution_unavailable" era for agent tasks ends only when the
  certification matrix says so.
- New failure modes are explicit: unresolvable message ref → failed run
  with reason; side-effecting call → `approval_required` outcome + queued
  notification.
- Step 2 of decision 2 (full queued escalation with client approval
  surface) remains future work with its own task; this ADR does not
  promise it.
- ADR-077's phase-1 vocabulary (single-owner execution, run-slot dedupe,
  timed-out, notification pass-back, no-double-count) carries over
  unchanged.

## Owner rulings requested

1. **Decision 1:** side-store message references (recommended) vs.
   redaction-at-API-boundaries-only.
2. **Decision 2:** ship step 1's read-only envelope with
   `approval_required` as a terminal outcome (recommended) vs. keep
   `agent_task` fully refused until the complete escalation loop exists.
