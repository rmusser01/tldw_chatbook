# ADR-085: Console activity receipts and session-switcher ownership

Status: Accepted
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
