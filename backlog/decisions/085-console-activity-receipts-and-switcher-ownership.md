# ADR-085: Console activity receipts and session-switcher ownership

Status: Accepted
Amended by: [ADR-210](210-console-region-ownership.md) (accepted 2026-10-01)
Date: 2026-08-23
Related Task: [TASK-21351](../tasks/task-21351%20-%20Add-activity-views-to-CtrlK-session-switcher.md)
Related Spec: [Console session-switcher activity views design](../../Docs/superpowers/specs/2026-08-23-console-session-switcher-activity-views-design.md)
Preserves: ADR-010, ADR-031, ADR-083

## Decision

Console `Ctrl+K` remains a conversation-scoped switcher. Every selectable
subject is either an unbound native Console session or a persisted local
conversation. Open sessions for one persisted conversation merge under that
conversation's profile-local identity; titles and workspace labels never
establish identity. Activity contributions determine the row's state and one
explicit immutable activation target. Server-only and uncorrelated workflow
runs do not appear.

The first release is local and independently releasable. Active membership is
the ordered union of action-required work, running work, unseen terminal
outcomes, the current conversation, and other open sessions. History remains a
separate bounded projection over every persisted local conversation. Console
Context and Inspector rails remain consumers of their existing projections and
do not read the switcher's normalized model.

The profile-local `AgentRunsDB` owns additive v15
`console_activity_receipts`. A receipt stores only safe identity, terminal
status, timestamps, and destination fields. Stable logical-outcome identity
plus a monotonic transition revision makes duplicate publication idempotent and
effective correction explicit. Supersession removes an obsolete revision from
Active without claiming the user saw it. Existing terminal history is not
backfilled.

One app-lifetime activity-receipt service coordinates receipt publication,
acknowledgement, and the existing `FLEET_UNSEEN` compatibility mark under one
process-wide re-entrant lock. Receipts are authoritative for switcher
membership. The mark remains a derived coarse badge for incumbent Console/wake
surfaces and is cleared only when no unseen FLEET survivor receipt remains. If
receipt publication fails after settlement, a separate local-only
`FLEET_RECEIPT_FALLBACK` companion mark makes that coarse evidence durable
until the user visits the conversation or a complete event replay creates the
receipt. Starred marks remain unrelated.

Direct-run, queue-chain, and live `FleetDrained` publication is non-throwing
presentation bookkeeping: failure preserves execution settlement and exposes a
content-free degraded-state signal. Startup orphan repair is different because
it changes durable execution state; its `running → error` update and failed
receipt insert commit in the same AgentRunsDB transaction or both roll back.
Receipt DDL/read failure disables only the optional receipt capability: core
AgentRunsDB construction, agent execution, open-session switching, and History
remain available. Core schema failures are not downgraded. AgentRunsDB is never
automatically deleted, quarantined, or rebuilt.

Acknowledgement is consequence-aware and evidence-specific. A successful
outcome is acknowledged only after the exact destination and receipt-keyed
notice visibly paint. Failed, stuck, stopped, and cancelled outcomes require
the notice's explicit `Mark seen` action. Activation captures exact receipt
IDs and statuses as immutable evidence, so mutable current state cannot change
consequence policy and a newer outcome cannot be cleared accidentally. The
post-refresh acknowledgement callback is additionally fenced by destination
identity and notice presentation generation. If an unbound ephemeral session
has disappeared, its receipt remains unseen and Active shows a receipt-keyed
`Session unavailable` notice; only that notice's explicit `Mark seen` action may
clear it, and no alternate destination is inferred. This manual unavailable
notice is the sole exception to successful outcomes' destination-paint rule.
Unavailable-session notices are frozen, receipt-keyed non-target records. They
aggregate by profile/session identity and share Active grouping, search, count,
ordering, and the bounded result page with conversation subjects.
The incumbent persisted-conversation star property participates only in Active
ordering after group priority; it never creates membership. Unbound sessions
and unavailable notices are unstarred.

The app-lifetime receipt service also owns restart hydration. Construction
leaves it cold and performs no receipt read or badge reconciliation. Ctrl+K paints
open/live rows from memory immediately, while one runtime-owned off-loop
hydration call serializes durable read/merge with publication and
acknowledgement under the service lock. Hydration failure preserves the last
valid cache, exposes degraded state, and can be retried without blocking
History.

The switcher owns bounded asynchronous History paging and validates modal,
profile, mode, query, and generation before committing a result. Its complete
geometry is capped at 35 terminal rows, result labels are exactly two rows, and
Cancel is always pointer-accessible. F3 is a deliberate modal-scoped exception
to ADR-031's general single-letter screen-action rule: it toggles the two
switcher modes without inserting text into the focused search input, matches
the incumbent F2 modal action vocabulary, and is advertised only while it is
implemented and active. ADR-031's reserved globals, terminal-convention keys,
truthful hints, and safe modal dismissal remain unchanged.

Blank-query activation is a deliberate last-tab command: when another open
native tab exists, the explicit candidate is the process-local MRU other tab;
after restore with no navigation history, the most recently updated other open
tab is the deterministic fallback. Explicit row navigation overrides that
candidate, and nonblank search activates only the committed top result for the
exact query generation. The current tab remains visibly labeled rather than
silently receiving a no-op Enter.

Switcher search is domain-semantic over safe normalized presentation metadata.
Plain operational aliases and explicit `is:`/`workspace:` filters resolve to
deterministic state, destination, and workspace predicates. It does not read
transcript content, invoke an embedding model, add a vector index, or perform a
network request. Onboarding is inline through mode copy, placeholder, empty and
zero-match recovery states, and truthful key hints; no tutorial modal,
persistent onboarding flag, or telemetry owner is introduced.

Correlated server workflow activity, authority/version caches, exact Workflows
handoff, and standalone workflow browsing are not owned by this ADR. They need
a separate server contract, task, and ADR after the local release.

## Amendment (2026-09-03, TASK-31241 — Character chats mode and activation)

[ADR-120](120-character-conversation-navigation-and-local-semantic-search.md)
adds `Character chats` as a third `Ctrl+K` mode beside Active and History while
preserving the switcher's operational ownership. Every ordinary open still
starts in Active, and blank Active Enter retains MRU-other-tab behavior. F3
remains the sole modal-local mode key and cycles Active → History → Character
chats → Active under ADR-031. Character chats owns a separate per-visit query
and searches only eligible local character conversations in the active Data
Profile; it does not auto-widen into another corpus or admit Personas, server,
or cached-server rows.

The modal remains mounted around an immutable highlighted target through
`IDLE`, `OPENING_CANCELLABLE`, `COMMITTING`, and `FAILURE_VISIBLE`. It delegates
to the Console-owned opener and waits for exactly `OPENED`,
`CANCELLED_PRECOMMIT`, `NOT_FOUND`, `DATA_PROFILE_CHANGED`,
`CHARACTER_UNAVAILABLE`, or `FAILED`. Escape can cancel only before the opener's
atomic `commit_started` acknowledgement; later Escape is ignored while commit
finishes or rolls back. Duplicate Enter, mode/query changes, and result movement
remain disabled during activation, and only `OPENED` dismisses the modal after
the exact Console destination is current and visible. This amendment is owned
by [TASK-31241](../tasks/task-31241%20-%20Align-character-conversation-navigation-decisions.md).

## Amendment (2026-10-04, TASK-33620.9 — explicit title publication)

An explicit rename of a saved local conversation commits its optimistic-lock
database write before publishing the new title to any matching open Console
runtime. Rail/tree and bound tab/F2/palette actions share the existing Console
workspace rename worker. The store publishes a committed title by exact
persisted-conversation identity to every matching runtime; the captured database
and store must still belong to the same Data Profile. An unsuccessful durable
write leaves both saved and live titles unchanged. Unbound scratch-session
renaming and automatic first-message titles retain their existing behavior.

Before the durable write, the store reserves existing fork-source admissions
for every exact alias and rejects new hydration or rebinding into that title
transition. Publication verifies both the captured runtime instance and its
persisted binding; rebinding away is not renamed. The publication callback is
valid only inside its admission scope. Same-binding preparation cleanup remains
a no-op rather than being refused. A cancelled worker drains its retained
SQLite write and publishes any committed title into the captured store before
releasing admission, but never paints or toasts a retired or changed-profile
view. Complete renames serialize through the existing workspace owner.

The worker invalidates persisted-row projections and awaits the existing
tab/header and conversation-browser publication before success feedback.
Requesting a coalesced broad refresh alone does not prove publication. No new
event bus, persistence owner, Library title editor, or cross-profile mutation is
introduced. Library currently has no conversation-title editing control; a
future editor must use this same durable-before-publication contract. Renames
never activate a tab or infer identity from the title. Publishing only the
active runtime was rejected because inactive and intentionally duplicated
runtimes can refer to the same saved conversation.

For an unchanged saved binding, an older send's title snapshot is not title
authority: preparation cancellation, optimistic-send rollback, and delayed
successful-send publication preserve the current committed title. Scratch or
changed-binding rollback still restores its prior identity, and first
persistence still publishes its staged title. Saved rename input is checked
again after sanitization, before admitting any durable write.

## Context

The incumbent switcher eagerly loads a mixed local tuple, mounts at most twenty
results, sorts the selected row before recency, identifies widgets by position,
and may resume any row carrying a conversation ID. It cannot distinguish open
state, live work, unseen success, action-required failure, or historical
recency. Ordinary terminal outcomes live only in controller memory, while
post-turn FLEET attention has only a conversation-level mark. Neither is a
restart-safe per-outcome acknowledgement model.

`AgentRunsDB` already owns local agent-run identity and is profile-local, but it
also stores durable user-authored agent definitions and change notes. That
makes it the correct additive owner for activity receipts and makes destructive
recovery unacceptable. The existing `FleetDrained` event carries survivor
identity and settlement timing, and the controller owns both direct and queue
terminal seams, so no new event bus or dependency is needed.

The user approved a local-first release because it improves the common Console
path without waiting for cross-repository workflow schema, authorization,
sequence paging, or exact-run navigation. A universal inbox would also make
the `Switch Session` title dishonest before Workflows can open a useful exact
run.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Treat open, running, or recent as the sole meaning of Active | Each omits important work and conflates lifecycle with execution; the ranked union is explicit and deterministic. |
| Reuse only `_unvisited_outcomes` | It is process memory, has no per-outcome identity, and cannot survive restart or safe concurrent acknowledgement. |
| Extend only `FLEET_UNSEEN` | A conversation-level bit cannot identify status, revision, producer, or the exact evidence being acknowledged. |
| Store receipts in a new database | Adds another lifecycle, migration, and failure boundary although AgentRunsDB already owns the durable run identities. |
| Rebuild AgentRunsDB when receipts are corrupt | Risks deleting user-authored definitions and change notes to repair optional presentation state. |
| Acknowledge every terminal result on activation | Merely painting a failed or cancelled result does not prove it was understood; non-success requires an explicit action. |
| Load all History before opening | Makes Ctrl+K latency scale with the full corpus and repeats the incumbent eager-load problem. |
| Add a virtualization or cursor dependency | Bounded native paging and at most fifty mounted rows satisfy the approved scale with less code and risk. |
| Put standalone workflow runs in Ctrl+K | Violates conversation scope and cannot provide an honest session-switch destination. |
| Use a printable letter for mode switching | Search owns printable keys while focused; F3 is unambiguous and local to this modal. |

## Consequences

- AgentRunsDB advances to v15 through guarded additive DDL and gains an optional
  receipt capability; genuine v14, fresh-v15, and receipt-DDL degradation paths
  require coverage.
- One small service becomes the only writer that coordinates receipts with the
  FLEET compatibility badge. Multiple app processes sharing one profile remain
  unsupported.
- Ordinary outcome producers must carry stable turn/queue-chain identity;
  queue-chain identity comes from the final accepted durable dispatch
  checkpoint rather than a process-local context epoch. `FleetDrained` gains
  stable drain identity for null-run survivors.
- The modal opens from cached local Active state and loads History lazily;
  History may shift between pages under concurrent mutation, but immutable
  activation identity prevents wrong-target activation.
- TASK-28125's exact-query, strict-F2, scroll ownership, textual state, and
  MRU-other trust repairs remain required behavior inside the replacement modal.
- The destination surface gains a compact receipt-keyed outcome notice with a
  visible `Mark seen` action for non-success.
- Production verification must load the real stylesheet hierarchy, inspect the
  painted compositor, exercise the real Ctrl+K-to-destination route, and compare
  equal row/column dimensions in iTerm2 and Windows Terminal.
- Phase 2 server integration cannot silently enter this task; it needs a new
  authority/version/sequence decision and independently releasable work.

## Links

- [Approved design](../../Docs/superpowers/specs/2026-08-23-console-session-switcher-activity-views-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-08-23-task-21351-console-session-switcher-activity-views.md)
- [ADR-010: Console conversation-local marks](010-console-conversation-local-marks.md)
- [ADR-031: TUI keybinding and footer-hint conventions](031-tui-keybinding-and-footer-hint-conventions.md)
- [ADR-083: Console edge rails and workspace-owned conversation Tree](083-console-edge-rails-and-workspace-tree-ownership.md)


## Finite initial Console receipt preparation (2026-10-05)

Reason: explicitly define the original receipt initializer's asynchronous startup and same-App/database publication contract. No persistent preparation service, readiness capability, storage/schema change, permission cache or new authority is introduced.

Proposal before any managed production edit:
- The original resolved initial Chat route prepares only its durable receipt store before constructing its Console screen. Other destinations and original synchronous/custom/headless Runtime APIs keep their preceding routes.
- Only exact stock Runtime/helper/reader defining metadata qualifies the optional worker route. Preinstalled custom functions/instance shadows, subclasses and memory/custom database receivers retain their original route. After selection, queued/body/default/class drift refuses before replacement invocation; there is no custom fallback inside the selected interval.
- Source qualification for publication is scoped only to the selected finite worker through a thread-local proof restored in its original finally; preinstalled class/instance/subclass wrappers may continue delegating to the original synchronous reader. The proof contains the exact Runtime and its captured source-current check, and is not a service, cache, authority or reusable task.
- The original initializer captures the actual App, ChaChaNotes owner/path and marks-service owner before construction. Publication must still belong to that same owner after native construction, and every newly created refused initialization connection retires on its original source thread. Existing service/native borrowers are not adopted or closed.
- The initial startup task holds one finite child task; repeated cancellation drains the same actual callback before propagating cancellation or releasing startup custody. Normal exception stays visible, and cancellation does not publish a screen. Runtime disposal closes admission first and serializes with the original initializer lock; no disposed Runtime publishes storage.
- After actual callback retirement, the initial route rechecks exact App/Runtime, startup task/loop/thread, profile/source, screen stack, current tab, and shutdown/initial-push state before the original screen construction and push. A newer destination is never overwritten by a stale initial push.
- Live first Send retains every original bridge, storage, permission, provider, capture and readiness guard. Receipt preparation makes no claim of permission readiness. All original startup, Send, heartbeat, helper and native-open limits remain unchanged.

Native qualification plan: retain the original cold held-query causal RED first; then real queued reader replacement and same-function code/default drift, in-body App/DB/marks/generation drift, repeated cancellation while actual native SQL is held, disposal during the same held callback, warm borrowed original connection, preinstalled custom/instance/subclass/memory fallbacks, and normal first composition/key/Send/disposal controls. Controls observe original code/native handles; no native calls, guards, waits or budgets are replaced.

Evidence-only candidate is not installed or Native-qualified. Root owns task/plan/ADR registration and serialized actual runs.

Actual original-effect qualification: original-cold-receipts-native-red-6 completes31.375s with current production/test sources. The original Inspector/bridge/receipt/schema chain runs on the main Thread and actual shared loop. Holding its positively identified admitted native SQLite connection prevents the original loop callback; after release, normal Runtime disposal physically closes the exact handle and retires its original StorageLease/participant registration. Source/currentness, original five-cell close closure, global monitoring zero, no invalid observation and monitoring retirement all qualify before the responsiveness assertion. Attempts1-5 fail observer prerequisites and are excluded. Artificial hold and inclusive native spans establish this synchronous blocking site, not normal latency or whole-budget savings.

Rejected alternatives: precreating a persistent store outside startup; a detached readiness service/task; moving UI bridge construction to a worker; weakening admission or changing synchronous custom/public ABI; increasing original budgets; or accepting a captured result after navigation/owner/source drift. Final acceptance requires actual native normal, drift, borrowed, custom, cancellation and disposal controls, then unchanged original whole/platform and live first-Send evidence.


### TASK-34406: finite Character view work retires before creator close

The actual shipping resume worker and supported Console host must retain their
issued finite Character callback until native resources have physically
retired. App or host exit cannot establish healthy creator cleanup from a
cancelled or terminal logical Worker alone. Two original native routes reached
exit with the exact callback, operation and lease still live; their subsequent
settled-close correctly refused, and all final native/source cleanup controls
passed after release.

For the source-qualified standard UI resume, use the existing owned Character
presentation facade with explicit force-fresh entry. This bypasses only its
display memo; retain the original initial metadata pair, inner refresh, final
owner publication checks and repeated-cancel physical callback drain. Direct
actions and custom/subclass/memory/overridden callbacks retain their original
ABI, TTL and fresh reads.

App shutdown and the supported Console test host share only a bounded finite
view-worker drain. Capture the exact manager, owning nodes, same-loop Tasks and
issued workers before cancellation; include Character resume refresh alongside
existing sync and navigation work. Close host intake through original Textual
shutdown and drain the captured callbacks before host return or creator close.
Do not infer physical retirement from Worker state or adopt an unrelated
Runtime/database/worker. Runtime disposal and creator acceptance keep their
existing owners and ordering, including borrower and foreign-source refusal.

Original source, actor, actual Future/native handle/operation/lease retirement,
error priority, repeated cancellation and timing limits remain mandatory.
Explicit captured work references last until actual retirement; disposal
refuses late publication. This refines existing ADR-085 view/App ownership and
ADR-126 finite custody; no permission cache or new storage authority is added.
