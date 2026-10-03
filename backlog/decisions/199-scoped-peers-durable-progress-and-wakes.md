# ADR-199: Scoped peers, durable progress and progress wakes

Status: Accepted for the user-authorized orchestration burn-down
Date: 2026-09-29
Tasks: TASK-33430, TASK-33431, TASK-33432
Amends: ADR-136 (relay-only, process-local reports), ADR-134/135 (completion-only automatic wake source)

## Decision

Extend the existing coordinator, message inbox and automatic-work scheduler. Do not add a bus or second scheduler. Bodies are untrusted provider context and never approval, permission or instructions from the human user.

- **Peers:** child-only `list_peer_agents` and `send_to_peer` capabilities bind after exact run attachment. A peer shares the same live coordinator, parent and immutable chain. Discovery exposes bounded handle IDs, labels and status; send accepts a handle ID and bounded text. Self, unknown, foreign, terminal, pruned and revoked owners refuse. Reuse steering queues and coherent drain boundaries. Reports and peer sends share the existing sender lifetime allowance; recipient steering limits still apply. Generated peer source metadata is recorded in steps/logs; body text is omitted. Peer delivery never starts or resumes another run.
- **Durable progress:** SQLite is the pending queue authority for saved chats, under the existing ChaChaNotes chat-data owner. Reuse the existing Console transaction contribution so explicit Save commits the chat and its pending reports in one transaction. A `fleet_progress_messages` table stores stable message ID, FIFO sequence, saved conversation and source identities, body and creation time. No capability token is serialized. Enqueue and whole-report collection commit before receipts. Preserve all existing bounded sender/inbox/runtime limits, no eviction, chain-filtered automatic reads, explicit discard and body-free inspection metadata. A fresh authorized primary can collect saved data after reopen; old senders remain revoked. Runtime replacement and last binding closure revoke live authority without discarding saved reports.
- **Temporary chats and Save:** keep temporary bodies in memory. Save copies pending reports by stable message ID through the existing owned promotion operation. Failed or cancelled promotion retains the memory queue; repeated promotion is idempotent. Bind durable inboxes only after the saved identity is committed. Database operations follow existing worker ownership and connection lifetimes. The queue lock may admit a synchronous owned SQLite leaf transaction; that leaf never calls native identity, coordinator or observers. Initial durable loading is prepared outside native identity locks and bound only after exact-owner revalidation. Do not run external observers under queue/identity locks or revive a revoked owner from a late callback. An uncertain collection checkpoint stops execution without requeue or redispatch.
- **Progress wakes:** committed enqueue publishes IDs/counts only to the existing coordinator. Progress claims are separate from terminal survivor claims: `automatic_progress_wake_claims` keys each message to one attempt, and wake attempts carry a cause plus bounded message IDs. Claiming does not collect. The notice requests a fresh `read_agent_messages` call. Pending reports can remain manually readable after their one wake. Reuse 250 ms coalescing, two automatic primary slots, manual reserve/priority, chain fairness, deadlines, physical ownership, cancellation and the existing three-generation ceiling. Completion and progress spend the same chain allowance. Busy/waiting primaries are not interrupted. Preacceptance abort releases only its claims; accepted/uncertain attempts never replay automatically, including after restart. Temporary-chat wake claims remain process-local without persisting their bodies.

Each schema change increments the installed schema and updates the independently frozen backup/recovery schema and exact migration authorizer. Chat schema 74 ships auxiliary failure reasons under ADR-052, and schema 75 ships hook-continuation receipts under ADR-163. Preserve both shipped 73→74 and 74→75 migrations unchanged and add this unmerged durable progress in 75→76. Current Chat and shared Subscriptions recovery catalogs and stamp gates use 76; historical AgentRuns catalogs are unchanged. Durable data never grants live execution authority. Existing restored-pending message tool calls remain refused; all message tools remain catalog-reserved and private in diagnostics.

## Alternatives

A message bus duplicates existing bounds and owners. Treating progress as a terminal survivor would falsify settlement and claim accounting. Restoring serialized sender tokens would resurrect revoked authority. Persisting temporary bodies implicitly would violate temporary-chat privacy. Polling or automatically replaying uncertain accepted attempts would spend unapproved work.

## Verification

Use focused coordinator/runtime tests for peer authorization, limits and private coherent drain. Use real SQLite reopen/rollback/promotion and replacement-owner tests for durable inboxes. Exercise duplicate and mixed-source wakes, generation exhaustion, manual priority, collection racing admission, acceptance rollback and restart review while retaining completion regressions. Mounted inspection must stay count-only outside explicit body views.

## Storage choice clarification

Progress bodies live with their conversation in ChaChaNotes rather than AgentRunsDB. The existing Console promotion writer accepts parameterized sidecar inserts in the chat transaction; AgentRunsDB would require a cross-database promotion protocol with an uncertain handoff. This choice keeps temporary Save rollback atomic and conversation deletion ownership local. AgentRunsDB continues to own metadata-only wake claims and automatic budgets. Progress wake intake is an idempotent hint; admission rechecks pending report IDs without treating that hint as body receipt or execution authority.

For temporary chats, the existing generation reservation and attempt lifecycle
remain durable budget metadata. Exact report/source ID claims stay in the
running ledger only; temporary attempt rows persist an empty message-ID list.
Native Save retains that ledger's claims through promotion, preventing a
second wake for an already admitted report. Restart cannot reconstruct or
replay those temporary claims. Recovery fences an old completed progress or
mixed attempt's chain for review when its empty stored message-ID list has
lost the process-local claim set, including after Save. The saved reports
remain manually readable. Prepared and accepted ambiguous attempts retain
the existing review fence; Save adds no cross-database claim transaction.

## Frozen wake source and native Save identity

A child's immutable causal chain can retain the temporary conversation ID after
explicit Save while a new manual chain uses the saved conversation ID. Progress
intake resolves the source conversation from existing run/chain metadata; it does
not infer that conversation from the inbox owner. Both source buckets retain
their own claims and budgets, and share one automatic delivery per live native
session. An authenticated, accepted `AGENT_WAKE` authorization binds the exact
native session and chain before carrying its frozen source conversation into
ledger and agent-run metadata. Ordinary submissions cannot supply this alias.
Chat storage, Canvas scope, policy, scratch, project roots and approvals continue
to use the current native session and committed saved conversation identity.
Save does not change a surviving child's chain ownership or grant new authority.


## Native progress close and physical SQL ownership

Native close revokes the exact inbox generation before waiting for an admitted
SQLite leaf. A standard-library event only denies authority; sending, reading,
discard, ownership and budget checks still take the existing writer lock.
Published UI observations cannot grant or restore any capability. The runtime
adds exact-inbox cleanup to its existing bounded close/dispose drains and keeps
physical cleanup shielded through caller cancellation or a grace timeout.
Already-admitted commits may finish; saved pending rows remain authoritative.
Exact-object cleanup cannot remove a replacement inbox.

Replacement revokes the old store under native identity, then drains outside
that lock before loading fresh committed rows. This ordering keeps UI metadata
reads responsive and prevents an unfinished discard from reappearing in the
replacement cache. Constructor/CLI calls without an event loop retain their
synchronous compatibility cleanup.

After Save, close counts, fences, cancels and drains both known causal IDs: the
native session ID and its current saved conversation ID. No alias registry is
introduced. A promoted session's old native ID is permanently retired and its
fence remains latched. Only the saved ID can release after a proven drain for a
fresh authorized reopen; late old-source completion cannot start another wake.

Immediate whole-store revocation also avoids the bridge initialization lock. A
replacement can be physically draining the old store while it holds that lock;
disposal must remain deny-only at that boundary. Positive store creation and
return revalidate closure under the original lock. The asynchronous physical
cleanup owns completion of initialization before closing the exact resulting
store, so delayed registration cannot reopen authority after disposal.


## Writer contention at native and coordinator boundaries

A durable SQL leaf still owns the original global queue lock through commit.
Native binding attempts that lock without waiting while identity is held; a busy
attempt releases identity, waits outside it, then revalidates the exact session,
store and inbox owner before retrying. Fleet construction receives its prepared
inbox, and child sender preparation waits outside coordinator admission. Sender
revocation sets an exact-token deny event immediately; positive membership and
allowance checks continue under the writer lock. Already-admitted reports may
commit after terminal revocation. Peer discovery and steering retry a busy queue
outside coordinator ownership, then repeat all sibling and allowance checks;
steering admission and the shared sender allowance remain one atomic operation.
No observational cache participates in these decisions.

Shipping saved-chat hydration and chat creation keep native changes on the UI
loop and prepare bounded progress data in the existing finite Chat worker.
Rollback revokes its exact owner immediately and submits physical release to
that same worker, so cleanup cannot retain native identity during SQL contention.
A concurrent worker receipt owns preparation outside asyncio's task registry.
Cancelled callers and cancelled asyncio observers wait for an already-started
leaf to settle before propagating cancellation; a terminal cancelled receipt
settles immediately instead of retrying shield indefinitely. Revalidation keeps
late preparation or cleanup from publishing a revoked or replacement owner.


The new saved-preparation and modal prepare/discard worker entries reuse
`run_finite_local_worker` around their existing SQL callbacks. A finite operation
owns retirement of any installed file-backed database cache it first opens;
closing the UI-thread handle or shutting down the thread pool alone does not
retire that registered worker connection. Existing worker caches, borrowed raw
transactions, custom repositories and memory handles remain caller-owned. The
physical receipt includes fresh-cache retirement before UI completion, without
shortening an admitted leaf or changing committed report receipts.


The child's runtime report callback is the fourth finite-cache boundary. Fleet
run teardown already owns AgentRuns and workspace handles, and does not own the
new Chat database cache opened by a saved report. The authorized report callback
therefore uses the same finite local worker wrapper. This retires only a Chat
cache first acquired by that report; a prior caller's worker cache remains
borrowed. The original synchronous report API and caller-owned main-thread
handle are preserved, and commit still precedes a queued receipt.
