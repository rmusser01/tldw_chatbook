# ADR-135: Fleet completion delivery and crash recovery

Status: Accepted; durable ledger and runtime integration implemented in TASK-32036/32037
Date: 2026-09-08
Tasks: TASK-32020, TASK-32021; implementation in TASK-32036/32037
Amends: [ADR-129](129-fleet-mailbox-and-wake-reliability.md), [ADR-134](134-fleet-admission-and-automatic-work-budgets.md)
Evidence: [Delivery baseline](../docs/agent-fleet-delivery-baseline-2026-09-08.json)
Plan: [Delivery and recovery](../../Docs/superpowers/plans/2026-09-08-fleet-delivery-and-recovery.md)

## Problem and scope

Before TASK-32037, the bridge emitted completion events only after every child in
a conversation had settled. A ready result could wait indefinitely for a sibling.
The wake coordinator then held one global delivery slot until the whole turn
returned, including human approval waits. A ready second conversation therefore
waited even when the controller had unused primary capacity.

The original `wake_delivered_at` column deduplicated notifications after a
successful stamp. It could not prove exactly-once tool execution: a provider or
tool could finish before the stamp, and a stamp failure was swallowed. An unseen
conversation could replay that work on a later claim. Viewing could instead clear
the staging mark, so that recovery policy provided neither reliable exactly-once
delivery nor an unconditional at-least-once guarantee.

This decision governs native Console automatic completion work, including the
plain-provider fallback. It adds no mailbox persistence, remote-process quota,
new tool authority, or automatic approval. ADR-134's finite causal-chain budgets
and the user's conservative default preference remain authoritative.

## Completion events and scheduling

1. Emit a new typed individual-settlement event after the child's terminal
   run row and result have been persisted, on both normal and exceptional exits.
   Carry run/conversation/session/spawning-turn identity, terminal status, and
   the existing classification made at settlement of whether it outlived its
   turn. Do not infer that classification later from a shared counter.
2. Emit outside the bridge lock through named bridge-lifetime consumers.
   Notification failure must not suppress final settlement or other consumers.
   A child without a durable run ID produces no automatic wake. Duplicate run
   IDs are harmless intake; the database claim is the admission authority.
3. Keep `FleetDrained` and its last-child-settled counter intact. Final usage
   reconciliation still runs at the drain. Do not reuse the earlier scope-exit
   event: exceptional children may not yet have a terminal row there. Individual
   notification may refresh completion attention but must not prematurely fold
   final child usage or close the change-review window.
4. Start a fixed 250 ms coalescing window at the first eligible completion.
   Later completions do not reset it. When admission becomes available, batch
   the oldest eligible causal chain in that conversation. A slow sibling does
   not hold this window open. A result settling during the wake stays pending
   for a subsequent attempt. Within-turn results remain on their existing path.
5. Replace the global one-wake rule with at most two simultaneous automatic
   primary turns, at most one per conversation. All accepted/manual/queued and
   pre-acceptance reservations count toward primary occupancy. An automatic
   reservation also requires total occupancy below `max_parallel_runs - 1`.
   Thus the default primary cap of three admits two wakes and leaves one slot
   for manual work; a cap of one admits no automatic wake. Lowering a cap stops
   new admissions and does not cancel accepted work.
6. Round-robin eligible conversations after each accepted wake. Within one
   conversation, select the oldest eligible chain by first pending settlement,
   with a stable ID tie-break. Skip closed, busy, cooling-down, exhausted, or
   review-required candidates without dropping them. A chain that has finished
   one wake joins the end of the eligible queue so a recurring producer cannot
   monopolize dispatch. A refused candidate keeps the existing one-second delay.
7. Manual drafts and queued work win each admission check. Recheck ownership
   immediately before acceptance. An approval wait keeps its primary ownership
   and cannot be preempted or interpreted as approval. Another conversation may
   use the second wake slot if the total/manual reserve permits it. If both
   slots are waiting, there is no promise of bounded latency: the UI reports
   waiting, and budgets/cancellation still apply.

The 250 ms window trades a small initial delay for batching fast siblings. Three
generations remain a hard chain limit; a long sequence of staggered completions
can exhaust it before every result has been consumed automatically. Remaining
results stay saved for explicit manual work. No per-result fresh allowance or
cross-chain batch may bypass the limit.

## Durable attempt and acceptance contract

AgentRunsDB owns the chain, reservations, attempts, and result claims in one
SQLite transaction boundary. The chat transcript and conversation marks live
in a different database and cannot participate in that transaction. Neither is
execution authority. Do not attempt to synthesize a cross-database transaction
or use a transcript search as an acceptance proof.

The current AgentRunsDB connection uses WAL with `synchronous=NORMAL`, which
can lose recent commits after an OS/power failure. That is insufficient for an
execution fence. Automatic-work ledger mutations use a dedicated transaction
context that sets `synchronous=FULL` before `BEGIN IMMEDIATE` and restores the
connection's previous setting after commit/rollback. Connections are thread-local;
do not change the pragma inside an active transaction. Keep ordinary per-step
run writes on their existing policy. FULL commit must complete before granting
dispatch authority; the guarantee remains subject to SQLite/filesystem durability.

An attempt has an opaque unique ID, chain ID, conversation and session identity,
generation reservation ID, owner-runtime ID, state, timestamps, and bounded
reason code. A separate result-claim table uniquely maps each source run ID to
one attempt. Validate terminal survivor status, immutable chain membership,
conversation scope, and undelivered/unclaimed state in the claim transaction.
Persist no result bodies, prompts, tool arguments, or approval decisions here.

| State | Meaning and permitted next action |
| --- | --- |
| `prepared` | Batch claimed and generation reserved before scheduling. The original owner may accept or prove refusal and abort. No execution authorization yet. |
| `accepted` | Acceptance and generation consumption committed durably before any attributable model generation or tool work. Repeat acceptance is idempotent; a second caller never receives new dispatch authority. |
| `completed` | The accepted turn has returned and the terminal bookkeeping transaction committed. Success of the user's objective is separate; errors and accepted cancellation also consume the attempt. |
| `aborted` | Original owner proved it never accepted or dispatched work. Release only its uncommitted generation reservation and result claims, atomically. Repeated abort is harmless. |
| `review_required` | Ownership or execution outcome is uncertain. Retain claims and reserved/consumed allowance, preserve results, and block automatic replay and further automatic chain work. |

The controller calls a required, typed acceptance operation at its authoritative
dispatch boundary. This is not a best-effort UI accepted-hook. Failure to commit
must prevent dispatch. The coordinator's private authorization is bound to the
attempt, chain, session, conversation, and owner; payload text cannot supply it.
Agent and plain paths use the same fence. Wakes must not invoke a billable
preparation helper (query rewrite, summarization, or another generation) outside
this fence and the call/token reservations. If preparation itself performs
generation, acceptance occurs before that first work and its later failure is
an accepted failed attempt, even if no main reply was generated. Readiness and
pure validation alone do not constitute generation.

After acceptance, provider calls and automatic child launches reserve their own
finite resources before dispatch. Accepted cancellation spends the generation;
uncertain usage retains its estimate as ADR-134 requires. A claimed attempt
cannot be replayed merely because `submit_draft` raises or returns no result.
Only an explicit typed pre-dispatch refusal permits abort/refund.

When the turn returns, commit completion state and legacy delivery stamps for
that exact batch in one AgentRunsDB transaction. Only then update in-memory
pending state and project unseen marks. If completion persistence fails, retain
the durable accepted claim, pause that chain, and preserve its result indicator.
It is safe to retry bookkeeping by attempt ID; it is never safe to rerun the
turn just to obtain a stamp.

## Recovery and visible states

Runtime/view attachment is not restart. Reattaching a screen or opening a second
DB handle must not invalidate live owners. Recovery is an explicit application
startup operation on the selected native runtime's durable state. A previous
owner's `prepared` or `accepted` attempts, interrupted call reservations, and
clock uncertainty become `review_required`; even a prepared reservation is not
automatically refunded on restart. This deliberately sacrifices unused budget
to avoid inferring that another owner performed no work. A process-local stale
callback cannot resume an attempt owned by a replacement runtime.

Atomic claims protect the same source results across DB connections. This does
not establish cross-process primary/worker admission or a lease permitting one
process to reclaim another's live work; ADR-134's resource scope stays local.

Accepted/completed/review-required claims exclude results from new automatic
attempts even if the legacy stamp is missing. Legacy chainless rows remain
inspectable and receive no fresh automatic allowance. Recovery reconstructs
pending/review state from attempt and run rows independently of marks; a badge
is a projection, never the only discovery index. Repairs to marks are idempotent.

The existing fleet surface receives a body-free typed projection with pending
result count, active attempt, chain usage/limits, and pause reason. It distinguishes
waiting for capacity, waiting for approval, budget exhausted, and interrupted
delivery requiring review. Suggested copy: "Automatic follow-up paused. Results
are saved; a previous attempt may already have run. Review its run and tools
before continuing manually." Viewing clears unseen attention only; it does not
clear the execution pause or mint a new allowance. Explicit new user work may
start a new chain, while old survivors keep their original one.

## Crash and contention acceptance matrix

These are implementation requirements. TASK-32036/32037 exercise them with real
SQLite reopen/rollback and deterministic controller/tool gates; this is not a
killed-process, power-loss, or external-provider certification. See the
implementation record and task notes for the scope of automated evidence.

| Interruption or race | Required observable outcome |
| --- | --- |
| Before claim transaction commits | No attempt, claim, or reservation residue; result remains available. |
| Two claimers select the same result | Exactly one gets dispatch authority; no doubled generation charge or mixed-chain batch. |
| After `prepared`, before scheduling or while preflight awaits | Same live owner can explicitly abort a proven refusal; restarted owner pauses with allowance retained. |
| Acceptance write fails or rolls back | No model generation or tool dispatch; no UI callback substitutes for the missing fence. |
| Fence write uses a reopened or alternate-thread DB connection | Verify FULL synchronization applies to that transaction too; ordinary bookkeeping policy is restored afterward. |
| After acceptance commit, before transcript/model dispatch | Restart pauses, consumes the generation, preserves result; absence of a chat row does not authorize replay. |
| During helper generation, main generation, approval wait, or tool execution | No automatic replay; retain reservations with unknown outcomes and present review state. Approval remains unresolved unless the user resolved it. |
| After a tool side effect, before provider/turn return | Same conservative pause; never claim exactly-once external side effects. |
| After turn return, before completion stamp transaction | Accepted claim blocks replay; bookkeeping may be retried without another model/tool call. |
| Completion transaction rolls back or is retried | Attempt state and source stamps remain atomic; repeat commit has no additional resource consumption. |
| After completion commit, before memory or mark update | Reopen skips delivery and repairs attention from durable rows without executing work. |
| Another child settles during completion/mark clearing | Clear only the completed batch; the new result and its unseen state remain pending. |
| View clears badge on an interrupted chain | Pause remains; restart still discovers the interrupted attempt without that badge. |
| DB handle reopen or screen remount during live work | No interruption classification, refund, or second dispatch. |
| Manual submission accepted while old survivors exist | New manual chain; old results retain their previous chain and exhausted/review state. |
| Concurrent model reservations, late usage, cancellation, backwards clock | Atomic conservative accounting per ADR-134; no negative balance or renewed allowance. |

## Alternatives

- Keeping drain-only delivery and one global wake is simpler but leaves the
  measured delays even with idle capacity. Two wakes are a bounded change and
  fit the existing default of three primary slots plus a manual reserve.
- Unlimited per-session wakes undermine the conservative aggregate limits.
  Exempting approval waits from occupancy misrepresents work still owned and
  can accumulate paused turns and later dispatch bursts.
- Stamping at scheduling loses results when a task is refused. Stamping only
  after completion permits replay of already performed work. Durable claims
  plus a required acceptance fence separate these cases.
- Retrying every interrupted attempt favors liveness over safety. Exactly-once
  external work would require end-to-end idempotency contracts for every tool
  and provider, which this system does not possess. Conservative manual review
  is the chosen recovery policy.

## Delivery status

TASK-32020/32021 deliver this decision and its measurements. TASK-32036 provides
the durable chain/attempt/reservation state. TASK-32037 wires the incremental
event, fair bounded scheduler, acceptance fence, both provider paths, and visible
pause states. These changes are implemented in the working tree with targeted
verification recorded in those tasks. The drain barrier, one global wake, and
completion-only stamping describe the prior behavior, not the current scheduler.

## Implementation record (TASK-32036, 2026-09-08)

AgentRunsDB schema v17 owns the chain, reservation, attempt, and unique source
claim tables. The `AutomaticWorkLedger` at `db.automatic_work` uses the same
thread-local DB connections and restores their prior synchronization policy
after each FULL transaction. Recovery remains an explicit runtime operation,
separate from opening a database or a view.

The deadline is fixed at first automatic acceptance. A persisted process ID and
monotonic anchor keep elapsed accounting consistent across handles in that
process. An unambiguous chain entering a replacement process establishes an
anchor that carries forward elapsed wall time, without extending its deadline.
Detected clock reversal or elapsed-time exhaustion remains durable. This is an
implementation of ADR-134's elapsed-time contract, not a power-loss test result.

TASK-32037 integrates the required acceptance fence, physical provider/helper
reservations, shared child launches, individual settlement notifications, fair
wake scheduling, startup recovery independent of badges, and visible saved-result
pauses. Cancellation during preparation also terminalizes the owning Console run.
Provider workers recheck authority after executor/client waits, and tools recheck
after approval before side effects.

Schema v18 adds a singleton runtime owner fence, updated only by explicit FULL
startup recovery. It revokes stale completed-parent contexts without changing
completed attempt history or pausing a healthy replacement chain. Admission and
resource commitment check ownership transactionally; late usage settlement may
still improve accounting without granting dispatch authority. This also protects
children that outlive their accepted parent and were waiting for approval when
the native runtime was replaced.

The automated evidence uses real SQLite, controlled provider/tool boundaries,
and rendered Textual tests. No killed-process, filesystem power-loss, or external
provider certification is claimed; already-dispatched external effects and live
Python threads retain their existing cooperative cancellation limits.

## PR 2631 integration amendment (2026-09-11)

Automatic work moves to v17 and the runtime owner fence to v18 after the budget column at v16. Upstream v13-v15 remain reserved for run steps, spawn-event identity and activity receipts. Runtime migrations remain guarded by actual schema presence; opening either an upstream v15 database or a historical draft database preserves existing records and does not perform recovery. Standalone SQL references apply once in version order.

Index verification captures the actual reservation lookup, active-wake check, and claim-release statements and checks their plans without sqlite_stat1. The unused conversation index on chains is omitted: production reads chains by primary id or unique root-submission identity.
