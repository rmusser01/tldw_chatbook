# Console Session Switcher Activity Views Design

**Date:** 2026-08-23
**Status:** Approved by independent specification review and user; implementation gated by TASK-20937
**Task:** TASK-21351
**Surface:** Console `Ctrl+K` session switcher

## Goal

Make `Ctrl+K` a fast, conversation-scoped way to find work that is happening
now, needs attention, or was recently completed while preserving complete
historical conversation search.

The switcher distinguishes conversation lifecycle, execution activity, open
state, and recency. A resolved, backlog, or non-viable conversation remains in
`Active` while associated work is running or requires action. The first release
ships from local state and is not blocked by a new server activity subsystem.

## Approved product decisions

### Ctrl+K remains conversation-scoped

Every selectable subject is a Console session or a persisted conversation.
Correlated workflow status may decorate a conversation in a later phase, and a
correlated workflow may become that conversation row's explicit activation
destination. Standalone workflow runs do not appear in this switcher.

`Switch Session` remains honest: Enter never selects an uncorrelated server run.
A future universal work inbox or `Jump to Work` surface is separate work.

### Delivery is phased

1. **Local Active/History** ships first using open sessions, controller state,
   `AgentRunsDB`, and one durable local activity-receipt seam.
2. **Correlated server activity** follows as an independently deployable
   integration after Workflows can open a useful exact correlated run.
3. **Standalone workflows and a lossless global server activity feed** are
   deferred to separate future design/task work and are not acceptance criteria
   for TASK-21351.

The local phase must be useful and releasable without the server phase.

### Acknowledgement depends on consequence

- Successful `done`/`succeeded` results are acknowledged automatically only
  after the exact destination has visibly loaded the selected result.
- Failed, error, stuck, stopped, or cancelled results require an explicit
  `Mark seen` action at the destination.
- Approval, human-input, paused, queued, and running states are not terminal and
  are never acknowledged by opening them.
- A newer result arriving during navigation remains unseen.

This avoids silently dismissing a problem merely because a detail card painted.

## Scope

### Phase 1 — TASK-21351 local release

This specification requires:

- `Active` and `History` modes in the Console `Ctrl+K` modal;
- conversation-scoped normalized rows and explicit activation targets;
- local open/current/running/approval/paused/terminal-unseen state;
- durable local receipts for ordinary inactive-session and FLEET survivor
  outcomes;
- automatic acknowledgement of successful outcomes after destination paint,
  with one manual exception: a vanished session-only destination clears only
  through its receipt-keyed `Session unavailable` / `Mark seen` action;
- complete persisted-conversation History search;
- deterministic ordering, bounded paging, safe async search, and stable focus;
- a 35-total-row modal ceiling; and
- iTerm2/Windows Terminal parity evidence inherited from TASK-20937.6.

### Phase 2 — correlated server integration

The later integration adds only:

- explicit workflow-run `conversation_id` correlation;
- cached background synchronization of correlated workflow activity;
- per-run activity versions and per-device acknowledgements;
- an exact correlated-run Workflows destination with useful status, recovery,
  and `Mark seen`; and
- local-only degradation when the server capability is absent.

Phase 2 does not add general remote conversation browsing or standalone workflow
rows to Ctrl+K.

### Out of scope

- a general notification center or universal work inbox;
- standalone or uncorrelated workflow runs in Ctrl+K;
- a lossless global workflow event feed, signed cursor, retention-gap recovery,
  or deletion-event history;
- cross-device acknowledgement synchronization;
- remote conversation editing;
- workflow inputs, outputs, artifacts, prompts, messages, or tool details in the
  switcher projection;
- approval/retry/pause implementation beyond the destination's existing or
  separately scoped controls; and
- changes to the Console Context rail ownership model.

## Evidence and incumbent seams

The current switcher:

- mounts at most 20 two-row buttons;
- sorts the selected row before recency;
- receives an eagerly assembled mixed tuple;
- waits for persisted rows before opening;
- identifies result widgets positionally;
- uses offset-based conversation pagination;
- has no mode, authority, activity, or typed activation target; and
- lets F2 fall back to an unrelated first native session.

Local activity signals already exist but are fragmented:

- open native sessions and unsaved drafts;
- controller run and approval state;
- `AgentRunsDB` statuses;
- in-memory `_unvisited_outcomes` for ordinary inactive-session completion;
- durable `FLEET_UNSEEN` conversation marks; and
- `FleetDrained.children`, which carries exact survivor run, session, message,
  and terminal-status identity.

The local design consolidates terminal presentation state without replacing
execution storage. `wake_delivered_at` remains supervisor-delivery state and is
never reused as user-seen state.

## Active membership and ordering

Conversation lifecycle (`in-progress`, `resolved`, `backlog`, `non-viable`) is
independent from execution activity. Lifecycle never suppresses an otherwise
eligible Active row.

Active groups are ordered:

1. **Waiting for you**
2. **Working**
3. **New results**
4. **Current**
5. **Other open**

Only nonempty headings render.

| Effective state | Group | Notes |
| --- | --- | --- |
| approval or human input required | Waiting for you | Destination must name the action or honest recovery path. |
| failed / error / stuck / stopped, unseen | Waiting for you | Requires explicit `Mark seen`. |
| paused | Waiting for you | Opening does not clear it. |
| running | Working | Local first; correlated server in Phase 2. |
| queued | Working | Correlated server in Phase 2. |
| done / succeeded, unseen | New results | Auto-acknowledged only after exact visible load. |
| cancelled, unseen | New results | Copy says cancelled; requires explicit `Mark seen`. |
| superseded | Excluded | Retracted/replaced work is not an outcome. |
| current conversation | Current | Only when no higher state applies. |
| other open idle session | Other open | Only when no higher state applies. |
| terminal acknowledged | Excluded | Conversation remains available in History. |

Rows sort by:

1. group priority;
2. existing conversation starred state, starred first;
3. latest relevant activity time, descending;
4. case-folded title; and
5. stable subject key.

Persisted conversation subjects read the incumbent conversation-star property;
unbound sessions and unavailable-session notices are unstarred. Star affects
ordering only and never creates Active membership.

Malformed/missing timestamps sort after valid timestamps inside their group and
still use deterministic title/key tie-breakers.

## History behavior

History contains every conversation in the existing Chatbook persisted source,
including conversations also present in Active. Unsaved drafts are Active-only.
General server conversation browsing is not added.

History headings are:

- `Today`
- `Yesterday`
- `Previous 7 days`
- `Older`

Aware timestamps convert to the application's local display timezone before
calendar grouping. `zoneinfo` supplies DST behavior. Invalid/missing timestamps
fall under `Older`; future timestamps group under `Today` and may carry a safe
clock-skew diagnostic outside user content logs.

### Search widening

Every modal opens in Active. A blank query shows only Active rows.

For a nonblank query:

1. Active search completes first.
2. When it has one or more matches, only those matches render.
3. When it has zero matches, the same query automatically searches History and
   renders `No active matches — showing History` followed by History matches.
4. F3 still toggles full History browsing and retains the query for the life of
   the modal.

Closing the modal discards its query and mode. This preserves activity-first
browsing without adding an F3 tax to ordinary historical lookup.

## Normalized local projection

The modal consumes presentation records; it does not read controller/database
schemas directly. Each selectable conversation subject row contains:

- stable subject key and local profile authority;
- title and optional workspace label;
- existing conversation starred state (false for unbound sessions);
- native session ID and/or persisted conversation ID;
- explicit activation target carrying the stable profile authority and opaque
  runtime authority token captured when the row was built;
- effective activity state and group;
- lifecycle state when known;
- latest relevant timestamp;
- current/open flags;
- zero or more immutable local activity receipt ID/status pairs;
- run multiplicity; and
- literal-safe display tokens.

The projection stores no prompt, message, response, tool arguments/results,
hidden reasoning, artifact content, model output, or credentials.

An unresolvable session-only receipt uses a separate frozen
`UnavailableSessionNotice` record, not a conversation subject or activation
target. It contains only:

- stable key `unavailable-session:<profile-authority>:<native-session-id>`;
- profile authority and the unavailable native session ID;
- effective group/state and latest receipt timestamp;
- the frozen effective receipt ID/status pairs; and
- literal-safe `Session unavailable` / `Mark seen` copy.

All effective receipts for the same unavailable session aggregate into one
notice. `done` and `cancelled` notices rank in New results; failed, stuck, and
stopped notices rank in Waiting for you. They are unstarred and use the same
group/star/time/title/key comparator as conversation subjects. For mixed
receipts, the highest-priority group wins. Within that group, primary display
status uses latest receipt time, then `stuck > failed > stopped > cancelled >
done`, then activity ID. The second line renders the primary status plus `+N`,
and search indexes every unique safe status. They also match `session
unavailable` or exact session ID. The notice's `Mark seen` control
is its only selectable action and acknowledges only its frozen receipt IDs.
Unavailable notices participate in the Active count and the same bounded paging
as conversation subjects, so any number remains scroll/search reachable without
exceeding the result-page mount bound.

### Canonical subject aggregation

Aggregation is conversation-first within one local profile authority:

- when a persisted conversation ID exists, the subject key is
  `conversation:<profile-authority>:<conversation-id>`;
- every open native session bound to that conversation merges into that one
  subject; and
- an unsaved/unbound draft uses
  `session:<profile-authority>:<native-session-id>` and never merges by title or
  workspace label.

When multiple native sessions represent one persisted conversation, the
deterministic local session is the current session when present, otherwise the
session with the latest valid activity time, then lexicographically smallest
session ID. A receipt naming a still-open exact session retains that destination;
otherwise it resumes the persisted conversation.

Each raw contribution has a stable source key: `receipt:<activity-id>` for a
receipt, `controller:<target-key>:<state>` for the current controller signal,
and `shell:<session-id>` for current/open shell state. Duplicate source keys
collapse.

Raw contributions sharing the same explicit activation target reduce to one
target contribution as follows:

1. effective group is the highest-priority nonempty group represented;
2. only raw contributions in that group compete for primary display state;
3. primary display state uses latest valid timestamp, then the fixed state rank
   `human-input/approval > stuck > failed/error > stopped > paused > running >
   queued > cancelled > done/succeeded > current > other-open`, then source key;
4. latest relevant timestamp is the winning primary contribution's timestamp;
5. captured receipt IDs include every effective receipt for that exact target,
   deduplicated and sorted by activity ID; and
6. target multiplicity is the number of distinct raw source keys.

Each merged subject then ranks these reduced target contributions by approved
group, latest relevant timestamp descending, destination precedence, and a final
stable target contribution key:

1. an actionable or executing native Console session;
2. a correlated workflow run;
3. a local terminal receipt; and
4. the current/open idle Console shell.

Target contribution keys are explicit identity, never title:
`native:<session-id>` for native targets and
`conversation:<conversation-id>` for persisted targets. An idle shell for a
native target reduces under that target's `native:` key; `shell:` is only its
raw source key. A future Phase 2 contract may add an authority-qualified
workflow run/version target key. Lexicographically smallest target key wins an
otherwise exact tie.

The highest-ranked target contribution supplies the row's primary state and
explicit activation target. `+N` is total distinct raw source keys across the
subject minus the one primary display contribution, so same-target aggregation
retains deterministic multiplicity. Enter acts only on
that captured primary target: a local destination captures only its represented
receipt IDs, while a Workflows destination captures only its run ID and visible
activity version. Other contributions remain unseen/active.

Phase 2 activity may join only a persisted local conversation subject whose
canonical conversation ID and configured server authority match the correlated
run. A correlated run with no such local subject is excluded from Ctrl+K; the
switcher never manufactures a server-only conversation row.

### Explicit activation targets

Phase 1 target kinds are:

- `console_native_session`; and
- `console_persisted_conversation` through the explicitly named local adapter.

Phase 2 may add `correlated_workflow_run`, always carrying conversation ID,
server authority, run ID, and captured activity version. Nullable-field inference
in the button handler is forbidden.

## Unified local activity receipts

The profile-local `AgentRunsDB` owns one additive `console_activity_receipts`
table for both ordinary inactive-session outcomes and FLEET survivor outcomes.

Each receipt has immutable identity/payload fields and mutable lifecycle
timestamps:

- `activity_id` primary key;
- origin: `ordinary` or `fleet_survivor`;
- stable logical-outcome identity and positive transition revision;
- nullable Console session ID;
- nullable persisted conversation ID;
- nullable AgentRunsDB run ID;
- nullable assistant-message ID when the producing turn has one;
- safe terminal status (`done`, `failed`, `stuck`, `stopped`, `cancelled`);
- created timestamp;
- nullable acknowledged timestamp; and
- nullable superseded timestamp.

Storage enforces that at least one destination identity—Console session ID or
persisted conversation ID—is present. A receipt may therefore survive a restart
even when only the durable conversation identity remains.

There is no separate ordinary-outcome version ledger and no FLEET causal-epoch
table.

### Stable receipt identity

- Every producer supplies a stable logical-outcome identity. Ordinary direct
  turns namespace the durable assistant-message/turn identity; queue chains
  namespace the most recent accepted durable dispatch checkpoint's
  `preparation_id`. Each accepted queued turn replaces that identity, so the
  terminal receipt is keyed by the final accepted turn; recovery reconstructs
  the same identity from that checkpoint. Process-local context epochs are
  never receipt identity. FLEET survivors use the identities below.
- `activity_id` is deterministic from origin, logical-outcome identity,
  transition revision, and safe terminal status. The publication transaction
  reads the latest revision: restamping the same effective status is a no-op;
  changing status supersedes that receipt and inserts revision `N+1` atomically.
  A uniqueness constraint over `(origin, logical_outcome_id, revision)` makes
  retries idempotent at the storage boundary rather than relying on a toast
  guard or an in-memory cache.
- A survivor with a run ID uses a namespaced run identity.
- `FleetDrained` gains one stable `drain_id` generated when the bridge constructs
  the event. A survivor without a run ID uses a deterministic namespaced identity
  from that drain ID and its stable child ordinal.
- Duplicate insertion is idempotent by `activity_id`.
- A later effective correction is a new revision. Superseding the prior receipt
  is not acknowledgement, but removes that obsolete revision from Active.
  Acknowledging an old ID cannot acknowledge the correction. A sequence such as
  `done → failed → done` therefore produces revisions 1, 2, and 3 rather than
  reusing the first `done` identity.

The terminal transition seam writes the receipt before updating its in-memory
compatibility cache. On restart, unseen rows repopulate the projection.
Pre-migration terminal AgentRunsDB history is not imported, preventing an
upgrade flood.

### Ordinary producer contract

One controller helper owns ordinary outcome publication and is called from both
the direct run-state transition path and the queue-chain terminal callback. The
helper takes the stable logical-outcome identity, destination identity, and the
effective source transition; it performs the idempotent receipt insert before
refreshing `_unvisited_outcomes` compatibility state.

The exact mapping is:

| Source transition | Receipt status | Rule |
| --- | --- | --- |
| `COMPLETED` | `done` | Publish only for an inactive, terminal-notification-eligible turn/chain. |
| `FAILED` | `failed` | Publish only for an inactive, terminal-notification-eligible turn/chain. |
| `STOPPED` | `stopped` | Publish only when an inactive turn/chain produces a genuine terminal stop outcome. |
| `BLOCKED` | none | Remains live `Waiting for you`; it is not a terminal receipt. |
| all other ordinary states | none | Execution/open-state projection remains authoritative. |

Queue chains currently publish only `COMPLETED` and `FAILED`; adding a stopped
chain transition must use the same helper. An effective correction to a
different terminal status creates a new receipt revision, while restamping the
same logical outcome and status does not.

### Additive migration and crash reconciliation

`AgentRunsDB` advances from schema v14 to v15 with an additive, guarded
`console_activity_receipts` migration and the equivalent fresh-schema DDL.
Migration failure fails closed for local activity; it never deletes, replaces,
quarantines, or rebuilds `AgentRunsDB`, because that database also owns run
definitions and change notes.

The startup orphan reconciliation that changes an existing FLEET run from
`running` to `error` also inserts its `failed` receipt in the same transaction.
Its logical identity is namespaced from the run ID, its persisted conversation
ID comes from `agent_runs`, and its Console session ID may be null. Only rows
changed by post-v15 reconciliation produce these receipts. Existing terminal
history is not backfilled.

### FLEET producer contract

`FleetDrained` is the sole live FLEET receipt producer. It considers only child
records with `settled_after_turn=True`; children settled in-turn remain ordinary
turn outcomes and must not be duplicated as survivor receipts. Safe mapping is:

| `FleetDrained` child status | Receipt status |
| --- | --- |
| `done` | `done` |
| `error` | `failed` |
| `cancelled` | `cancelled` |

An unknown status fails closed: it creates no receipt, emits only a content-free
diagnostic/degraded-state signal, and cannot interrupt event fanout.

### FLEET mark compatibility

`console_activity_receipts` is the source of truth for switcher membership and
acknowledgement. `FLEET_UNSEEN` remains a coarse compatibility badge for existing
Console/wake surfaces; it does not provide per-run status or control Active.

All receipt/mark writes route through one service that owns a new process-wide
re-entrant lock. The lock spans the database receipt insert/acknowledgement,
the unseen-survivor count, and the compatibility-mark set/clear/reconciliation
as one ordered critical section:

1. persist a survivor receipt;
2. best-effort set the coarse FLEET mark;
3. on acknowledgement, acknowledge exact receipt IDs first;
4. clear the coarse mark only when no unseen survivor receipt remains; and
5. reconcile marks from unseen survivor receipts on startup and after a failed
   mark write.

A crash can temporarily leave a false-positive coarse badge, but cannot hide an
unseen receipt. Star/unstar storage and semantics are unchanged. Multiple app
processes concurrently writing the same local profile remain outside the
supported runtime model.

### Publication failure isolation

Receipt publication is best-effort presentation bookkeeping on the direct
run-state, queue-chain, and live FLEET event paths. Each seam catches storage or
mapping failures, preserves incumbent terminal state, cleanup, toast, and event
fanout behavior, sets one content-free recoverable local-activity degraded-state
signal, and never raises into execution settlement. A later successful refresh
may clear that signal.

Startup orphan reconciliation is the deliberate exception because status and
receipt share one durable repair: a receipt failure rolls back the orphan status
transition so the complete reconciliation can retry on the next startup.

### Acknowledgement coordinator

Activation captures exact receipt IDs.

- After the exact Console session/conversation is selected, it paints a compact
  outcome notice keyed to the captured receipt IDs/statuses. Only after that
  notice is visible does the coordinator mark the captured `done` receipt IDs
  acknowledged.
- For `failed`, `stuck`, `stopped`, or `cancelled`, activation opens the exact
  destination and presents a compact outcome notice with status, safe recovery
  copy, and a focusable `Mark seen` action. Only that action acknowledges the
  captured IDs.
- An aggregate containing both successful and unsuccessful receipts auto-clears
  only its captured successful subset; the unsuccessful subset remains behind
  the explicit notice.
- A newer receipt created during navigation remains unseen because its ID was
  not captured.
- Navigation failure, profile change, missing destination, or dismissal does
  not acknowledge anything.
- A terminal receipt whose unbound ephemeral session no longer exists remains
  unseen. Active renders a receipt-keyed `Session unavailable` notice with a
  separate `Mark seen` action; the unavailable notice is not a selectable
  conversation subject and never falls back to another destination. This is
  the only cleanup path for an unresolvable session-only receipt.

Visible-paint acknowledgement is generation-fenced. Showing, replacing,
hiding, or unmounting the outcome notice increments its presentation generation.
The post-refresh callback captures the destination identity, exact receipt IDs,
and presentation generation, then revalidates that the same destination is
still selected and the same receipt-keyed notice is mounted, displayed, and
current. Switching away or remounting before the callback therefore leaves the
receipts unseen.

The outcome notice uses button/Tab/Enter interaction and introduces no new
protected or terminal-convention binding.

## Modal interaction and layout

### Structure

The modal contains:

1. `Switch Session` heading;
2. one-row `Active (N) | History` mode controls;
3. search input;
4. grouped results/status;
5. accurate key hints; and
6. the existing pointer-accessible Cancel action.

`N` is the number of aggregated Active conversation/session subjects plus
unavailable-session notices before query filtering. It is not a run count.

The entire modal—including border, padding, heading, modes, input, results,
hints, and Cancel—uses at most **35 terminal rows**. Smaller terminals clamp to
the safe viewport. Results own the remaining vertical scroll area.

The modal also defines a width contract:

- preferred width: incumbent width or wider when available;
- minimum supported width: 52 cells inside the terminal-safe viewport;
- below that width, lifecycle/workspace/recency tokens omit in that order;
- title ellipsizes after required state/destination tokens receive space; and
- no row horizontally scrolls.

### Exact two-row grammar

Each selectable result is exactly two terminal rows:

1. literal-safe title;
2. one non-wrapping token line.

Token order is:

`STATE · SOURCE/DESTINATION · WORKSPACE · LIFECYCLE · RECENCY · +N`

Never truncate:

- state/attention;
- ambiguous source or authority;
- activation destination when it is not Console; and
- stale/error/unavailable state.

First omissions are lifecycle, workspace, then recency. Multiplicity collapses
to `+N`; the title receives remaining width. User-controlled text uses Rich
`Text` or escaping and cannot introduce markup.

Current/open are group headings in the approved v1 model, not extra subtitle
badges. A later simplification may merge them only through a separately approved
UX change.

Mode controls use the literal grammar `Active (N) — selected | History` or
`Active (N) | History — selected`, in addition to focus/background styling.
State never relies on color alone. A persistent one-row
`#console-switcher-status` line exposes `Selection: <ordinal> of <count> —
<literal-safe title>` or a mode/loading/error message. When live reconciliation
moves focus because a subject disappeared, the modal updates that line and
calls the existing app notification channel with the same text. Compositor and
notification-spy tests assert the exact copy; no unsupported terminal screen
reader API is assumed.

### Keyboard and pointer contract

- `Ctrl+K`: open a new modal in Active with search focused.
- `F3`: toggle Active/History and retain the query within the modal.
- `Up`/`Down`: move through selectable rows without wrapping; Up from the first
  result returns to search.
- `Enter`: activate the focused conversation row. From search, Enter activates
  the top current-query conversation result; when the top result is an
  unavailable-session notice, it moves focus to that notice's explicit
  `Mark seen` action and updates the status line without acknowledging. Only a
  subsequent Enter on the focused `Mark seen` action (or its pointer click)
  acknowledges the frozen IDs.
- `F2`: rename only the focused renameable native session; no fallback.
- `Tab`/`Shift+Tab`: follow visual order through modes, search, results/page
  actions, and Cancel.
- `Esc`: close without acknowledgement.

Pointer activation matches keyboard activation. Group/status rows are not
focusable. Modal-scoped F3 must be documented as an explicit ADR-031 exception
or the ADR wording updated before implementation; footer hints advertise only
implemented actions.

### Safe asynchronous search and live updates

Every search attempt captures:

- modal instance ID;
- mode;
- exact query and query generation;
- local profile/authority generation; and
- Active projection generation when applicable.

A result commits only when all captured values still match.

Active results are the closed union of conversation `ConsoleSwitcherEntry` and
`UnavailableSessionNotice`. Both expose `stable_result_key`; subject entries map
their subject key to it, and unavailable notices use their documented
unavailable-session key. Each result widget owns its immutable union payload
keyed by that stable result key.
Activation reads that payload, never a positional index or a freshly looked-up
list entry.

Enter from search acts only on a result committed for the current query and
generation. A conversation result activates; an unavailable-session notice only
receives focus as specified above. While the current query is pending, Enter
waits for that attempt or shows non-destructive `Searching…`; it never activates
an older rendered match or acknowledges an unavailable notice.

Live Active reconciliation preserves focus by stable result key. If the focused
result disappears, focus moves to the next row, then previous row, then search,
and the change is reported through `#console-switcher-status` plus the existing
app notification channel.
Reordering never changes the target owned by an already focused button.

The existing 200 ms debounce may remain. Mode changes supersede pending work.
No network request occurs on modal open or F3.

### Paging

Both modes use pages of at most 50 result records. In Active that bound includes
conversation subjects and unavailable-session notices. Page actions are bounded
modal controls outside that result-record count and are explicit rows such as:

- `Previous · 1–50 of 243`
- `Next · 101–150 of 243`

They are keyboard/pointer activatable and never accept F2.

Active pages come from the already cached projection. When that projection
changes materially, the modal recomputes the page containing the focused stable
result key; it returns to page one only when that result no longer exists.

History reuses the repository's bounded storage query with deterministic order
and immutable conversation IDs. Offset pagination is acceptable for this
ephemeral browsing surface: concurrent mutation may move an item between pages,
but it cannot change an entry's activation target. The UI does not promise a
frozen historical snapshot, and it does not add a repository-wide mutation
generation solely for this modal.

Changing query or mode resets to page one. Late pages cannot replace a newer
query, mode, profile, or modal instance.

## Nonbinding Phase 2 research notes

This section records research constraints and one plausible future shape only.
It is not an approved server contract, does not add TASK-21351 acceptance
criteria, and grants no implementation authority. A separately created server
task and ADR must revalidate or replace every schema, endpoint, authorization,
polling, cache, and Workflows-handoff detail below before Phase 2 implementation.
Nothing in this section may delay or enter the Phase 1 diff.

### Server correlation and snapshot sequence

`workflow_runs` adds:

- nullable immutable `conversation_id` using the Chat API's canonical string;
- monotonically increasing per-run `activity_version`;
- server-managed `state_changed_at`; and
- global monotonically increasing `activity_sequence` assigned on each
  activity-relevant change.

`session_id` keeps its existing workflow/ACP meaning. Existing runs are never
heuristically correlated.

One server database method owns effective status/reason transitions and, in one
transaction, updates status/reason, increments activity version, assigns the
next activity sequence, and writes state-change time. Run creation performs the
same sequence assignment. Existing workflow-event logging may remain separate;
the lossless event-feed refactor is deferred.

Sequence allocation uses a durable one-row counter serialized inside that same
transaction. SQLite relies on its serialized write transaction. Backends that
permit concurrent writers lock the counter row until the run mutation commits
or rolls back. The endpoint captures its upper bound by reading the committed
counter through the same serialization boundary. Consequently, once it returns
that bound, no transaction can later commit an activity sequence at or below
it.

### Correlated activity endpoint

A user-scoped endpoint returns only workflow runs with an explicit visible
conversation correlation. It:

- enforces tenant/user authorization;
- returns immutable `(server_instance_id, tenant_id, user_id)` authority;
- orders/pages by `activity_sequence` then run ID;
- accepts a validated nonnegative `after_sequence` and a fixed
  `snapshot_upper_bound` after the first page;
- caps pages at 200;
- returns `snapshot_upper_bound`, `next_after_sequence`, and `has_more` on every
  page;
- selects only `after_sequence < activity_sequence <= snapshot_upper_bound`;
- supports a complete current-nonterminal listing;
- supports bounded reconciliation of cached run IDs;
- returns only safe list metadata; and
- never returns prompts, messages, inputs, outputs, errors bodies, artifacts, or
  hidden reasoning.

The sequence is continuation state, not an authorization token; endpoint scope
is enforced independently. Arbitrary sequence values may skip or replay the
caller's own results but cannot cross authority.

First sync captures a committed upper bound, baselines old terminal runs,
imports the complete current nonterminal set, then consumes later sequences.
The client holds the upper bound fixed across all pages and advances its durable
authority watermark only after `has_more=false`. A crash before completion
replays the snapshot idempotently from the prior durable watermark. Run rows
retain their latest activity sequence. Hard-deletion-before-sync and a lossless
deletion/event history are explicitly deferred; known cached IDs are reconciled
so revocation/deletion does not leak stale metadata.

### Server authority and per-device acknowledgement

The server provisions `server_instance_id` once during database/configuration
initialization, stores it durably, and keeps it stable across restarts and URL
changes. It is never derived from a URL or credential. Docs-info, correlated
activity pages, and exact-run responses return the same identifier.

An additive local `correlated_workflow_activity_cache` table keys every server
row by configured target scope plus
`(server_instance_id, tenant_id, user_id, run_id)`. It stores the latest safe
snapshot metadata, `activity_version`, `activity_sequence`, and the highest
locally acknowledged activity version. Acknowledgement is per device and
local-only: successful post-paint confirmation or explicit `Mark seen` writes
the exact version that was visible. If a newer version has already arrived,
that newer version remains unseen. No server acknowledgement API is introduced.

### Background cache and capability

Ctrl+K reads only local/cached state. A single background sync per authority:

- polls approximately every five seconds while correlated nonterminal work
  exists and every thirty seconds otherwise;
- uses jittered exponential backoff up to five minutes;
- validates authority generation before commit; and
- applies each page idempotently and advances the durable watermark atomically
  only with the final page of a fully drained snapshot.

`/api/v1/config/docs-info` advertises
`hasCorrelatedWorkflowActivityV1=true` only when correlation columns, sequence
assignment, the correlated endpoint, authority envelope, reconciliation, and
exact-run fetch are all ready. Old/partial servers remain local-only without
repeated endpoint probing.

### Exact Workflows destination

A correlated server-primary row carries an explicit Workflows handoff with
authority, conversation ID, run ID, and captured activity version.

Workflows:

1. claims the handoff under the same authority generation;
2. fetches the exact run;
3. rejects authority/run mismatch or version regression;
4. renders a selected-run card containing safe workflow label, status,
   attention/recovery reason, state-change time, and activity version;
5. provides the concrete existing action when available, otherwise honest
   `Read-only here` recovery instructions naming where the action must occur;
6. provides `Mark seen` for failed/stuck/cancelled terminal outcomes; and
7. reports visible-load success after Textual's post-refresh confirmation.

Successful terminal results auto-acknowledge the exact visible fetched version.
Unsuccessful terminal results acknowledge only through `Mark seen`. Waiting,
paused, queued, and running items remain Active after opening.

## Failure and degradation

- Local receipt storage failure does not prevent opening/switching; the modal
  shows one recoverable local-activity status.
- Receipt migration/read failure disables only local activity membership for
  that profile, reports a recoverable status, and leaves open-session switching
  and History available. The application never automatically deletes or
  rebuilds `AgentRunsDB`.
- Missing local conversations never redirect to another session.
- Search/page failures preserve query and current safe results.
- Profile change dismisses the modal and invalidates pending activation.
- Server absence, old capability, auth loss, or offline state leaves Phase 1
  fully operational.
- Cached server rows show stale age; authority changes never reuse cached data.
- Missing/revoked correlated runs remain unacknowledged until authoritative
  reconciliation removes their cached metadata.

## Security and privacy

- Local receipts contain IDs/status/timestamps only.
- Server endpoints enforce user/tenant scope independently from continuation
  values.
- Correlation is explicit and server-validated.
- Source authority is part of every server key and activation target.
- User titles/workspace labels are never interpreted as markup.
- Activity diagnostics do not add conversation content, model output, workflow
  payloads, or credentials.
- Parameterized queries and existing validation helpers are required at every
  storage/API boundary.

## Performance requirements

- Ctrl+K opens immediately from local/cached Active state; History loads lazily.
- Modal open/F3 perform no network I/O.
- Database/search work over 100 ms runs outside Textual's event loop.
- At most 50 selectable result widgets plus bounded headings/page actions mount.
- No transcript reconstruction or workflow detail fetch occurs while listing.
- Poll attempts never overlap.
- No new UI, paging, cursor, or virtualization dependency is added.

Production-shaped benchmarks record modal open, Active filtering, automatic
History widening, F3 toggle, History paging, and one background cache commit at
representative/stress sizes before any speed claim.

## Verification

### Pure model tests

- every local status maps to the approved group;
- lifecycle never suppresses active execution/attention;
- acknowledged terminal rows leave Active but remain in History;
- ties use group/star/time/title/key deterministically;
- superseded runs are excluded;
- multiple receipts aggregate into one conversation row;
- primary state and `+N` are deterministic;
- exact same-group/time/destination contribution ties resolve by stable
  contribution key;
- a same-target done+failed+shell aggregate reduces to Waiting for you, chooses
  the deterministic failure display state/time, captures both receipt IDs for
  that target, and reports the exact raw-signal `+N`;
- mixed unavailable done/failed receipts choose Waiting for you, render the
  deterministic primary status plus `+N`, and remain searchable by both states;
- malformed timestamps/metadata remain safe; and
- literal markup, emoji, CJK, RTL, and mixed-width titles remain literal.

### Local receipt tests

- fresh v15 creation and guarded v14-to-v15 migration preserve existing run
  definitions/change notes;
- receipt migration/read failure fails closed without deleting or rebuilding
  `AgentRunsDB`;
- ordinary inactive completion/failure persists across restart;
- identical terminal restamp creates no duplicate receipt;
- `failed → done` and `done → failed → done` create monotonic revisions,
  supersede obsolete receipts without acknowledging them, and expose only the
  latest effective revision;
- direct and queue-chain terminal seams publish through the same idempotent
  helper with distinct stable logical-outcome identities;
- post-v15 orphan reconciliation changes status and inserts its receipt in one
  transaction, while pre-v15 terminal history is not backfilled;
- later correction has a different immutable ID;
- FLEET duplicate delivery is idempotent;
- `FleetDrained` excludes in-turn children, maps survivor
  `done/error/cancelled` to `done/failed/cancelled`, and fails closed for an
  unknown status;
- null-run survivor identities remain distinct within/across drain events;
- cold restart hydration restores unseen receipts, concurrent calls coalesce,
  degraded hydration can retry, and runtime disposal rejects a late commit;
- success activation acknowledges only captured success IDs;
- failed/stuck/stopped/cancelled remain until explicit `Mark seen`;
- a receipt created during navigation remains unseen;
- switch-away, notice replacement, hide, or remount before post-refresh leaves
  every captured receipt unseen;
- an unavailable ephemeral destination never falls back and clears only through
  its receipt-keyed `Mark seen` action;
- mark-write failure cannot hide an unseen survivor receipt;
- a forced interleaving of receipt publication, acknowledgement, survivor
  counting, and mark reconciliation cannot clear a badge while an unseen
  survivor exists;
- injected receipt-write failure cannot escape the direct, queue-chain, or live
  FLEET settlement seam and preserves incumbent cleanup/toast/fanout behavior;
- injected orphan-reconciliation receipt failure rolls back its status change
  for retry;
- startup reconciliation repairs false-positive/false-negative FLEET badges;
- star/unstar behavior is unchanged; and
- historical AgentRunsDB rows do not create upgrade floods.

### Modal tests

- every invocation starts Active with search focused;
- open sessions for one persisted conversation merge under its canonical
  conversation key; duplicate session selection is deterministic;
- an unbound draft remains a distinct session subject;
- blank Active query never loads History;
- zero Active matches automatically show History matches for the same query;
- F3 retains the query only for the modal lifetime;
- Enter while search is pending cannot activate an old result;
- a late query/page cannot replace a newer generation;
- result activation uses immutable payload, not positional index;
- live reorder preserves subject focus and activation identity;
- disappearing focus follows/announces the deterministic fallback;
- F2 never falls back to another session;
- exact row grammar/truncation preserves required tokens at supported widths;
- mode/state/focus and forced-focus-fallback status remain readable in
  monochrome/ASCII fallback and use the exact literal/notification grammar;
- total geometry never exceeds 35 rows and clamps on small terminals;
- page rows show useful ranges and remain reachable; and
- no network request occurs on modal open or F3.

Production-shaped Textual tests load the bundled stylesheet and assert
compositor text, containment, focus, hit testing, row height, truncation, and
small-terminal scrolling. Equal-cell manual evidence is recorded in iTerm2 and
Windows Terminal; TASK-20937.6 remains the owner of its prerequisite parity
evidence.

### Nonbinding future Phase 2 verification ideas

These ideas are not TASK-21351 gates. A future server task and ADR must select,
revise, and own them together with the corresponding contract.

- fresh/historical SQLite and backend migrations;
- durable `server_instance_id` survives restart and URL changes and matches
  docs-info, activity-page, and exact-run responses;
- exact schema target verification and fail-closed capability;
- explicit correlation authorization and immutability;
- sequence/version assignment is atomic with effective run-state changes;
- interleaved writer transactions cannot commit a sequence at or below an
  already returned snapshot upper bound;
- unchanged writes do not advance sequence/version;
- multi-page snapshots hold a fixed upper bound while concurrent later updates
  wait for the next sync cycle;
- a crash during paging replays idempotently and advances the durable watermark
  only after the snapshot is fully drained;
- correlated endpoint excludes standalone runs and other principals;
- first-sync baseline does not flood old terminal history;
- complete nonterminal and cached-ID reconciliation;
- authority generation rejects late responses;
- old/partial/offline servers remain local-only;
- exact Workflows fetch rejects mismatches/regressions;
- per-device acknowledgement records the exact visible activity version and
  does not hide a newer cached version;
- successful result auto-acks only after visible confirmation;
- failed/stuck/cancelled result requires `Mark seen`; and
- destination recovery copy is actionable or explicitly read-only.

## Rollout and task boundaries

1. Approve this revised written specification.
2. Update TASK-21351 acceptance criteria to make the local release independently
   complete and releasable.
3. Complete TASK-20937 and its terminal-parity dependency.
4. Keep TASK-21351 In Progress as the design/umbrella owner. Create one atomic
   Chatbook child for the local projection, receipts, modal, and acknowledgement
   notice; copy the Phase 1 implementation criteria, link the spec/plan/ADR, and
   put the child In Progress only when adding its plan.
5. Implement, verify, and merge the local child before starting server work.
6. File a separate correlated-server companion task/ADR and Chatbook integration
   child for Phase 2.
7. Treat standalone workflows/lossless global activity feed as separate future
   design work, not an implicit extension of TASK-21351.

No uncreated future task ID is referenced. Task files receive concrete
dependencies only after the companion tasks exist.

## ADR decision

ADR required: **yes**.

- **Phase 1 Chatbook ADR:**
  `backlog/decisions/085-console-activity-receipts-and-switcher-ownership.md`
  records durable local activity-receipt ownership, canonical switcher targets,
  and the relationship between receipts and the compatibility FLEET mark.
- **Phase 2 server ADR:** workflow conversation correlation, activity sequence,
  authority, and correlated snapshot contract.

Phase 1 does not wait for the Phase 2 ADR. The standalone/lossless feed requires
its own later design decision if pursued.

## Rejected alternatives

### Universal workflow rows in Ctrl+K

Rejected for TASK-21351 because `Switch Session` must remain predictable and the
current Workflows surface cannot provide a complete universal-work experience.

### Server-first delivery

Rejected because local state already delivers the core user value and should not
wait for cross-repository schema/feed work.

### Automatic acknowledgement of every terminal result

Rejected because painting a failed/stuck/cancelled card does not prove the user
understood or acted on the problem.

### Three separate local acknowledgement systems

Rejected in favor of one immutable activity-receipt table. FLEET marks remain a
derived compatibility badge, and `wake_delivered_at` remains supervisor state.

### Repository-wide History mutation generation

Rejected for v1. Immutable activation payloads prevent wrong-target activation;
an ephemeral History page need not promise a frozen snapshot.

### Importing old terminal AgentRunsDB history as unseen

Rejected because it would create an upgrade notification flood unrelated to the
user's acknowledgement baseline.
