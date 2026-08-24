# TASK-21351 Console Session-Switcher Activity Views Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the local Phase 1 `Ctrl+K` Active/History switcher with durable per-outcome acknowledgement, exact destination activation, and a production-verified 35-row modal.

**Architecture:** `AgentRunsDB` v15 owns safe local activity receipts; one app-lifetime service coordinates those receipts with the legacy FLEET badge. The switcher consumes a pure canonical Active projection immediately and loads bounded persisted History pages asynchronously. Activation carries an immutable target and frozen receipt ID/status evidence into an always-mounted destination notice, which acknowledges only after visible paint or explicit `Mark seen`.

**Tech Stack:** Python 3.11+, SQLite, Textual 8.x, Rich `Text`, pytest/pytest-asyncio, existing Console runtime and conversation services. No new dependency.

---

## Scope and dependency gate

This plan implements only the approved local Phase 1 in
`Docs/superpowers/specs/2026-08-23-console-session-switcher-activity-views-design.md`.
Do not add server workflow correlation, polling, authority caches, or a universal
work inbox. TASK-20937.6 must be complete before final terminal-parity closeout;
implementation may begin only from a branch containing the completed TASK-20937
changes. TASK-21351 remains In Progress as the approved design/umbrella owner.
At that gate, create one atomic Phase 1 child through Backlog.md with parent
TASK-21351 and dependency TASK-20937, copy all Phase 1 implementation acceptance
criteria into it, and link this plan, the approved spec, and ADR-085. Put that
child In Progress while the parent remains In Progress, and execute these tasks
under the child. Complete the child's criteria/notes/Done transition first;
complete the parent's criteria/notes/Done transition only after the child and
all parent-level evidence are complete. Do not mint the child early: task IDs
are provisional until re-swept against current remote refs and worktrees.

ADR required: yes
ADR path: `backlog/decisions/085-console-activity-receipts-and-switcher-ownership.md`
Reason: ADR-085 owns durable local acknowledgement, the derived FLEET badge,
canonical switcher identity/targets, and the modal-scoped F3 exception to ADR-031.

## File map

| File | Responsibility |
| --- | --- |
| `tldw_chatbook/DB/AgentRuns_DB.py` | v15 optional receipt capability, idempotent revision/supersession, exact acknowledgement, unseen reads/counts, atomic orphan repair. |
| `tldw_chatbook/Chat/console_activity_receipts.py` | App-lifetime receipt/mark coordinator, safe status mapping, degraded-state signal, in-memory unseen snapshot. |
| `tldw_chatbook/Chat/console_runtime.py` | Construct and retain exactly one receipt service next to the one AgentRunsDB instance; own stable profile authority and runtime authority token. |
| `tldw_chatbook/Chat/console_agent_bridge.py` | Add stable `FleetDrained.drain_id` for null-run survivor identity. |
| `tldw_chatbook/Chat/console_fleet_attention.py` | Publish only post-turn survivors through the receipt service; keep toast/handoff behavior. |
| `tldw_chatbook/Chat/console_fleet_wake.py` | Route wake-delivery badge reconciliation through the receipt coordinator. |
| `tldw_chatbook/Chat/console_launch_wake.py` | Route unresolvable launch-wake cleanup through the receipt coordinator. |
| `tldw_chatbook/Chat/console_prompt_queue_coordinator.py` | Give each queue chain one stable logical-outcome ID and pass it to the terminal callback. |
| `tldw_chatbook/Chat/console_chat_controller.py` | Route direct/queue outcomes through one non-throwing helper while retaining compatibility markers/toasts. |
| `tldw_chatbook/Chat/console_switcher_state.py` | Pure subject aggregation, activity grouping, sorting, display tokens, History calendar grouping, immutable activation payloads. |
| `tldw_chatbook/UI/Console_Modules/workspace.py` | Build immediate Active snapshot and bounded local History loader without altering rail state. |
| `tldw_chatbook/Widgets/Console/console_session_switcher_modal.py` | 35-row Active/History modal, async generation guards, paging, stable focus, keyboard/pointer contract. |
| `tldw_chatbook/Widgets/Console/console_activity_outcome_notice.py` | Compact receipt-keyed success/failure notice and explicit `Mark seen` action. |
| `tldw_chatbook/Widgets/Console/console_session_surface.py` | Always mount and display-manage the outcome notice at the destination. |
| `tldw_chatbook/UI/Screens/chat_screen.py` | Open Ctrl+K immediately from Active snapshot and pass the History loader. |
| `tldw_chatbook/UI/Console_Modules/session.py` | Dispatch explicit targets and coordinate post-paint acknowledgement. |
| `tldw_chatbook/UI/Console_Modules/wiring.py` | Inject coordinator-owned FLEET read/clear seams instead of legacy direct helpers. |
| `Docs/User_Guide/console/sessions-tabs-workspaces.md` | Explain Active/History, acknowledgement, keys, and degradation. |

Do not create a generic notification framework, repository-wide pagination
abstraction, or new CSS dependency. Keep component CSS with the two Console
widgets unless a production-bundle rule demonstrably overrides it; if bundled
TCSS changes become necessary, regenerate it with the repository CSS builder.

### Task 1: Add AgentRunsDB v15 activity receipts

**Files:**
- Modify: `tldw_chatbook/DB/AgentRuns_DB.py`
- Modify: `Tests/DB/test_agent_runs_db.py`

- [ ] **Step 1: Write fresh-v15 and genuine v14 migration tests**

Add tests that construct the current inline pre-v14 fixture/v14 shape, assert
`console_activity_receipts` is absent, open `AgentRunsDB`, and assert the full
table/index/check-constraint shape plus `MAX(schema_version) == 15`. Reopen the
same file and verify existing agent definitions/change notes remain unchanged.

Also inject failure at only the v15 receipt DDL boundary and prove core
`AgentRunsDB` construction and existing run/definition/note reads still succeed
with `receipt_capability_available is False`. Core schema/migration failures
must still raise.

- [ ] **Step 2: Run the migration tests and confirm RED**

Run:

```bash
pytest Tests/DB/test_agent_runs_db.py -k 'activity_receipt or pre_v15 or receipt_capability' -q
```

Expected: FAIL because schema v15 and receipt operations do not exist.

- [ ] **Step 3: Add the guarded additive schema**

Use the incumbent idempotent `CREATE TABLE IF NOT EXISTS` mechanism, but isolate
only the optional v15 receipt DDL/version-row work in a named savepoint. On a
receipt-specific DDL failure, roll back that savepoint, leave versions 1-14 and
all core tables usable, set an instance capability flag false, and continue.
Never catch or downgrade failures from the core schema/versions 1-14. On a
successful receipt savepoint, bump the constant/version row together:

```sql
CREATE TABLE IF NOT EXISTS console_activity_receipts (
    activity_id TEXT PRIMARY KEY,
    origin TEXT NOT NULL CHECK(origin IN ('ordinary', 'fleet_survivor')),
    logical_outcome_id TEXT NOT NULL,
    transition_revision INTEGER NOT NULL CHECK(transition_revision > 0),
    session_id TEXT,
    conversation_id TEXT,
    run_id TEXT,
    assistant_message_id TEXT,
    status TEXT NOT NULL CHECK(status IN
        ('done', 'failed', 'stuck', 'stopped', 'cancelled')),
    created_at TEXT NOT NULL,
    acknowledged_at TEXT,
    superseded_at TEXT,
    CHECK(session_id IS NOT NULL OR conversation_id IS NOT NULL),
    UNIQUE(origin, logical_outcome_id, transition_revision)
);
CREATE INDEX IF NOT EXISTS idx_console_activity_receipts_unseen
    ON console_activity_receipts(conversation_id, created_at)
    WHERE acknowledged_at IS NULL AND superseded_at IS NULL;
```

Set `_CURRENT_SCHEMA_VERSION = 15` and append version 15 only after receipt DDL
succeeds. Receipt operations check the capability and raise one focused
`ConsoleActivityReceiptsUnavailable` error; callers can degrade without
misreporting the core database as corrupt. Never add quarantine/delete/rebuild
behavior.

- [ ] **Step 4: Write RED repository-operation tests**

Cover:

- identical `(origin, logical_outcome_id, status)` restamp returns the same ID;
- `failed → done` and `done → failed → done` create revisions 1/2 and 1/2/3;
- only the latest unsuperseded revision appears unseen;
- acknowledging exact IDs cannot acknowledge a newer revision;
- mixed-success/failure acknowledgement updates only supplied IDs;
- unseen survivor count excludes ordinary, acknowledged, and superseded rows.

- [ ] **Step 5: Implement the minimum transaction-local operations**

Add one private insert/update helper that accepts an existing SQLite connection
so orphan reconciliation can share its transaction, plus public wrappers:

```python
def publish_console_activity(self, *, origin: str, logical_outcome_id: str,
                             status: str, session_id: str | None,
                             conversation_id: str | None,
                             run_id: str | None = None,
                             assistant_message_id: str | None = None) -> tuple[str, bool]: ...

def list_unseen_console_activity(self) -> tuple[dict, ...]: ...
def acknowledge_console_activity(self, activity_ids: Sequence[str]) -> int: ...
def count_unseen_fleet_activity(self, conversation_id: str) -> int: ...
```

Generate deterministic IDs with stdlib `uuid.uuid5` from origin, logical ID,
revision, and status. Parameterize every query and validate enum/destination
fields before entering SQLite.

- [ ] **Step 6: Make orphan repair atomic with receipt publication**

Inside `reconcile_orphaned_runs`, select `conversation_id` with each orphan and,
in the existing transaction, insert `fleet-run:<run_id>` / `failed` before changing
`running → error`. Let any receipt failure roll back the entire repair; register
`_swept_paths` only after commit. If the optional receipt capability is absent,
the constructor's existing reconciliation guard logs the exception and leaves
the orphan `running` for a later retry; core DB construction and bridge use still
succeed. Do not backfill already-terminal rows.

- [ ] **Step 7: Run the complete AgentRunsDB suite**

Run:

```bash
pytest Tests/DB/test_agent_runs_db.py -q
```

Expected: PASS, including existing definitions/change-note and schema-version
tests.

- [ ] **Step 8: Commit the database boundary**

```bash
git add tldw_chatbook/DB/AgentRuns_DB.py Tests/DB/test_agent_runs_db.py
git commit -m "feat(console): persist activity receipts"
```

### Task 2: Coordinate receipts, FLEET marks, and producer identities

**Files:**
- Create: `tldw_chatbook/Chat/console_activity_receipts.py`
- Create: `Tests/Chat/test_console_activity_receipts.py`
- Modify: `tldw_chatbook/Chat/console_runtime.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/Chat/console_fleet_attention.py`
- Modify: `tldw_chatbook/Chat/console_fleet_wake.py`
- Modify: `tldw_chatbook/Chat/console_launch_wake.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `Tests/Chat/test_fleet_attention.py`
- Modify: `Tests/Chat/test_console_fleet_wake.py`
- Modify: `Tests/UI/test_console_launch_wake.py`
- Modify: `Tests/UI/test_console_runtime_ownership.py`

- [ ] **Step 1: Write RED service and forced-interleaving tests**

Use a real temporary AgentRunsDB and a recording marks service. Cover exact
status mapping, duplicate drain delivery, null-run children in two drain events,
unknown-status degradation, mark-write failure, and an event-gated interleaving
where acknowledgement attempts to clear the badge while a new survivor publishes.
The final invariant is: an unseen survivor receipt implies the coarse badge is
set after reconciliation. Cover optional receipt capability unavailable and a
post-migration read error: both expose a content-free degraded state and empty
receipt cache without breaking bridge construction, open-session switching, or
History. Cover cold hydration, concurrent-call coalescing, publication and
acknowledgement serialization during hydration, successful degraded-state retry,
and runtime disposal before the hydration callback. The runtime-ownership test
must exercise the real
`ConsoleRuntime.ensure_agent_bridge` construction seam. Task 4's production-path
UI test must later prove Active still shows open sessions and History still
returns bounded rows while the receipt service is degraded.

- [ ] **Step 2: Add the focused app-lifetime service**

Implement one class with no generic repository abstraction:

```python
class ConsoleActivityReceiptService:
    def __init__(self, runs_db: AgentRunsDB, marks: Any | None) -> None:
        self._db = runs_db
        self._marks = marks
        self._lock = threading.RLock()
        self._degraded = False

    def publish_ordinary(...): ...
    def publish_fleet_drain(self, event: FleetDrained) -> tuple[str, ...]: ...
    def acknowledge(self, activity_ids: Sequence[str]) -> int: ...
    def unseen_snapshot(self) -> tuple[ConsoleActivityReceipt, ...]: ...
    def hydrate_from_storage(self) -> int: ...
    def hydration_state(self) -> Literal["cold", "loading", "ready", "degraded"]: ...
    def reconcile_fleet_marks(self) -> None: ...
```

The lock spans receipt write/ack, unseen survivor count, and badge set/clear.
Catch and log only content-free exception type/context on live publication;
set `degraded=True` and never raise into execution. Keep database exceptions
observable from the explicit orphan-reconciliation path. Maintain an immutable
in-memory unseen snapshot and monotonically increasing projection generation;
the switcher read path never queries SQLite synchronously. When AgentRunsDB
reports the optional receipt capability unavailable, initialize directly in
degraded mode with an empty cache; do not suppress unrelated core DB errors.

`hydrate_from_storage` is the single restart-hydration owner. The caller runs it
off the event loop; inside the service it acquires the same lock as publication
and acknowledgement, transitions `cold/degraded → loading`, reads the durable
unseen rows, replaces the immutable cache, reconciles FLEET marks, increments
the projection generation, and commits `ready`. A read failure atomically
publishes `degraded` without clearing the last valid cache. A later invocation
is an explicit retry; concurrent calls coalesce behind the service state/lock.

- [ ] **Step 3: Give every FLEET drain stable identity**

Add `drain_id: str = field(default_factory=lambda: str(uuid.uuid4()))` to
`FleetDrained`. Preserve explicit construction in tests. For survivors, use
`fleet-run:<run_id>` when present and `fleet-drain:<drain_id>:<ordinal>`
otherwise. Filter `settled_after_turn is True`, map `done/error/cancelled` to
`done/failed/cancelled`, and fail closed on anything else.

- [ ] **Step 4: Construct the service once in ConsoleRuntime**

Create it immediately after the one `AgentRunsDB` instance, store it cold on
`ConsoleRuntime`, and pass it to bridge/controller consumers. Construction does
no receipt read or badge reconciliation and does not construct a second
AgentRunsDB against the same path. Add
`ConsoleRuntime.ensure_activity_hydration()`: it owns the one in-flight off-loop
hydration call, coalesces concurrent callers, allows a later retry only from
`degraded`, and invalidates completion after runtime disposal. Hydration is the
sole initial receipt-load and FLEET-badge reconciliation path.

Define the stable profile authority as the normalized resolved string form of
the durable `chachanotes_db.db_path`. Add a fresh opaque
`ConsoleRuntime.authority_token` at runtime construction. Subject keys use only
the stable profile authority; async jobs capture both the authority and token,
then compare them with the current runtime before commit. Do not reuse the
existing dispose counter `ConsoleRuntime.generation` as a profile token.

- [ ] **Step 5: Route FLEET attention through the service**

Replace direct mark writes in `ConsoleFleetAttentionConsumer` with
`publish_fleet_drain(event)`. Keep the incumbent app-loop toast and deep-link
fanout, using only successfully mapped survivor statuses. Inject a DB failure
and assert the consumer still announces/returns without raising.

- [ ] **Step 6: Route every legacy FLEET mark mutation through the coordinator**

Replace the direct set/clear helpers reached from `console_fleet_wake.py`,
`console_launch_wake.py`, and `UI/Console_Modules/wiring.py` with coordinator
methods. Wake delivery and visible-session acknowledgement may request
reconciliation, but the coordinator clears `FLEET_UNSEEN` only after its locked
unseen-survivor count is zero. Unresolvable ephemeral launch cleanup explicitly
does not acknowledge receipts: it preserves the receipt and coarse mark, and the
Active projection exposes a receipt-keyed `Session unavailable` notice with an
explicit `Mark seen` action. That action acknowledges only its frozen receipt
IDs. When receipt storage is degraded, keep the coarse mark rather than risk
hiding unseen work.
Retain the exported legacy helpers as thin compatibility delegates only if tests
or out-of-cluster callers still require them—no production path may mutate the
mark outside the coordinator. Add forced-interleaving tests for all three paths.

- [ ] **Step 7: Run focused service/FLEET tests**

Run:

```bash
pytest Tests/Chat/test_console_activity_receipts.py Tests/Chat/test_fleet_attention.py Tests/Chat/test_console_fleet_wake.py Tests/UI/test_console_launch_wake.py Tests/UI/test_console_runtime_ownership.py -q
```

Expected: PASS; the forced interleaving must fail if the shared lock is removed.

- [ ] **Step 8: Commit the service boundary**

```bash
git add tldw_chatbook/Chat/console_activity_receipts.py tldw_chatbook/Chat/console_runtime.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_fleet_attention.py tldw_chatbook/Chat/console_fleet_wake.py tldw_chatbook/Chat/console_launch_wake.py tldw_chatbook/UI/Console_Modules/wiring.py Tests/Chat/test_console_activity_receipts.py Tests/Chat/test_fleet_attention.py Tests/Chat/test_console_fleet_wake.py Tests/UI/test_console_launch_wake.py Tests/UI/test_console_runtime_ownership.py
git commit -m "feat(console): coordinate activity receipts"
```

### Task 3: Publish ordinary direct and queue-chain outcomes

**Files:**
- Modify: `tldw_chatbook/Chat/console_prompt_queue_coordinator.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `Tests/Chat/test_console_prompt_queue_coordinator.py`
- Modify: `Tests/Chat/test_console_chat_controller.py`
- Modify: `Tests/Chat/test_console_activity_receipts.py`

- [ ] **Step 1: Write RED logical-identity and failure-isolation tests**

Cover one direct completed turn, one direct failed turn, a defensive same-status
restamp, a corrected terminal status, one multi-entry queue chain, and injected
receipt-write failure on direct and queue terminal seams. Assert incumbent run
state, cleanup, marker, toast, and wake retry still occur exactly as before.

- [ ] **Step 2: Add one restart-stable queue-chain ID**

Do not use `conversation_context_epoch` or `PromptQueueSnapshot.expected_context_epoch`:
both are process-local fences. Give `_PromptChain` a
`logical_outcome_id: str | None` derived from the most recently accepted durable
dispatch checkpoint's `preparation_id`, for example
`queue-chain:<preparation-id>`. Thread `preparation_id` into the manual
`turn_accepted` call, reuse the already-present argument in
`acknowledge_durable_acceptance`, and set the same value from
`hydrate_dispatch_recovery`. Each later accepted queued turn replaces the chain
value, so the one terminal receipt is keyed by the durable final accepted turn;
a crash replay of that checkpoint reconstructs the identical ID. Change
`on_chain_terminal` to receive `(session_id, status, logical_outcome_id)`.

Fail closed when a terminal chain has no durable preparation identity: preserve
the incumbent terminal behavior but publish no receipt and expose degradation.
Add tests proving process-local epoch reuse cannot collide, the same recovered
checkpoint produces the same ID after coordinator reconstruction, and two later
accepted checkpoints produce different terminal IDs.

- [ ] **Step 3: Route terminal publication through one controller helper**

Add a non-throwing helper called by both `_set_run_state` and
`_publish_queue_chain_terminal`:

```python
def _publish_inactive_outcome(
    self, *, session_id: str, status: ConsoleRunStatus,
    logical_outcome_id: str, assistant_message_id: str | None = None,
) -> None: ...
```

Map `COMPLETED → done`, `FAILED → failed`, and a genuine inactive terminal
`STOPPED → stopped`; publish only when `terminal_notification_eligible` and the
session is not active. `BLOCKED` remains live Waiting for you and creates no
receipt. Thread the owning assistant message/turn ID through terminal call sites;
do not infer identity from title, timestamp, or current positional state.

- [ ] **Step 4: Keep compatibility state derived but behaviorally intact**

After successful publication, update `_unvisited_outcomes` for incumbent tab/rail
badges. On publication failure, preserve its current marker/toast behavior and
surface the receipt service's degraded flag. `mark_session_visited` may clear the
in-memory compatibility marker but must not acknowledge durable receipts without
captured receipt IDs.

- [ ] **Step 5: Run controller and queue tests**

Run:

```bash
pytest Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_activity_receipts.py -q
```

Expected: PASS; the multi-entry chain produces one logical outcome receipt.

- [ ] **Step 6: Commit producer wiring**

```bash
git add tldw_chatbook/Chat/console_prompt_queue_coordinator.py tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_activity_receipts.py
git commit -m "feat(console): publish inactive outcomes"
```

### Task 4: Build canonical Active and bounded History projections

**Files:**
- Modify: `tldw_chatbook/Chat/console_switcher_state.py`
- Modify: `tldw_chatbook/UI/Console_Modules/workspace.py`
- Modify: `Tests/Chat/test_console_switcher_state.py`
- Create: `Tests/UI/test_console_activity_switcher.py`

- [ ] **Step 1: Write RED pure aggregation tests**

Test canonical keys, duplicate native sessions, current-session preference,
activity-time/session-ID tie breaks, unbound drafts, lifecycle independence,
all five groups, existing-star ordering without star-created membership, mixed
local contributions, `+N`, unavailable-session
aggregation/group/search/order, mixed done+failed notice precedence, more than
50 unavailable notices across pages, exact contribution ties resolved by stable
contribution key, and a same-target done+failed+shell reduction with exact
group/state/time/captured IDs/`+N`. Cover malformed timestamps, literal
markup/emoji/CJK/RTL, and
group/star/time/title/key ordering. Add pure History bucket
tests for `Today`, `Yesterday`, `Previous 7 days`, and `Older` using an injected
local timezone and clock. Pin midnight boundaries, a DST transition, future
timestamps (`Today`), and missing/invalid timestamps (`Older`).

- [ ] **Step 2: Replace nullable activation inference with explicit payloads**

Use small frozen types in `console_switcher_state.py`:

```python
class SwitcherMode(Enum):
    ACTIVE = "active"
    HISTORY = "history"

class ActivityGroup(Enum):
    WAITING_FOR_YOU = "waiting_for_you"
    WORKING = "working"
    NEW_RESULTS = "new_results"
    CURRENT = "current"
    OTHER_OPEN = "other_open"

class SwitcherTargetKind(Enum):
    NATIVE_SESSION = "console_native_session"
    PERSISTED_CONVERSATION = "console_persisted_conversation"

@dataclass(frozen=True)
class CapturedReceipt:
    activity_id: str
    status: str

@dataclass(frozen=True)
class ConsoleSwitcherTarget:
    kind: SwitcherTargetKind
    profile_authority: str
    authority_token: str
    session_id: str | None
    conversation_id: str | None
    scope_type: str | None
    workspace_id: str | None
    receipts: tuple[CapturedReceipt, ...] = ()

@dataclass(frozen=True)
class UnavailableSessionNotice:
    stable_result_key: str
    profile_authority: str
    session_id: str
    group: ActivityGroup
    latest_at: datetime | None
    receipts: tuple[CapturedReceipt, ...]

ConsoleSwitcherActiveResult = ConsoleSwitcherEntry | UnavailableSessionNotice
```

`ConsoleSwitcherEntry` exposes `stable_result_key` equal to its canonical
subject key; `UnavailableSessionNotice` exposes the documented unavailable key.
Widget maps, focus retention, page relocation, and immutable payload lookup use
that common result key for both union variants.

Validate the required fields in `__post_init__`. Buttons and callbacks carry
this target directly; no handler decides a kind from whichever field is non-null.
The target's profile authority and runtime token are the immutable choice
evidence Task 6 revalidates. Validate captured receipt statuses against the local
receipt enum so activation never needs to consult mutable current receipt state
to decide acknowledgement.

`UnavailableSessionNotice` is not a `ConsoleSwitcherTarget`. Aggregate effective
session-only receipts by `unavailable-session:<profile>:<session_id>`; map
`done` and `cancelled` to New results and failed/stuck/stopped to Waiting for
you; for mixed receipts choose the highest-priority group, then primary status
by latest receipt time, `stuck > failed > stopped > cancelled > done`, and
activity ID. Render primary status plus `+N`; index every unique safe status for
search. Mark notices unstarred and sort all records by
group/star/time/title/key; also search literal `Session unavailable` and exact
session ID. Its sole
selectable control is receipt-keyed `Mark seen` for the frozen IDs. Count and
page these notices with subject results, keeping every result page at no more
than 50 mounted result rows.

- [ ] **Step 3: Implement the pure Active projection**

Merge subjects by `conversation:<profile>:<conversation_id>` or
`session:<profile>:<session_id>`. Deduplicate raw sources by
`receipt:<activity-id>`, `controller:<target-key>:<state>`, or
`shell:<session-id>`. Reduce each exact activation target by highest-priority
group, then latest valid time, the spec's fixed state rank, and source key.
Carry the winning time/state, every sorted effective receipt ID for that target,
and distinct raw-source multiplicity. Rank reduced targets by group/time,
local actionable/executing then receipt then shell precedence, and finally
`native:<session-id>` or `conversation:<conversation-id>` target key; an idle
shell reduces under its native target key. `+N` is total distinct raw sources in
the subject minus one. Capture only receipt ID/status pairs represented by the chosen local
destination. Emit the separately normalized unavailable-session notices after
subject aggregation, using the same deterministic group/page ordering and no
activation fallback.

- [ ] **Step 4: Split immediate Active from lazy History in workspace.py**

Add the new APIs:

```python
def console_session_switcher_active_entries(self) -> tuple[ConsoleSwitcherActiveResult, ...]: ...

async def load_console_session_switcher_history(
    self, *, query: str, offset: int, limit: int,
) -> ConsoleSwitcherHistoryPage: ...
```

Retain `console_session_switcher_rows()` as a compatibility adapter with its
incumbent return contract until Task 5 migrates `ChatScreen` and the modal in the
same commit. Task 4 must therefore remain independently runnable and must not
change current Ctrl+K behavior.

The Active call reads open sessions, controller state, and the receipt service's
in-memory snapshot only. The History loader runs existing flat-conversation and
existing `LocalChatConversationService.list_conversations(scope_type="all")`
call off the event loop and returns at most 50 entries plus total/range metadata.
Resolve workspace labels from the existing registry cache after the bounded
page returns. It must not perform the incumbent full membership scan and must
not mutate or reuse the Context rail's search/page lanes.

Apply the pure calendar bucketer after each bounded page returns. Parse stored
timestamps as aware instants, convert to the captured local `ZoneInfo`, compare
local dates against the captured clock, and preserve the fixed section order.
Future instants are `Today`; missing or invalid values are `Older`.

- [ ] **Step 5: Prove immediate projection and rail isolation**

In the workspace/projection test, block the History loader with an event, call
`console_session_switcher_active_entries()`, and assert it returns and filters
open/live rows before History is released. Do not press Ctrl+K or assert the new
modal before Task 5 migrates its caller. Snapshot Context/Inspector projection
owners before/after and assert they are unchanged. When the receipt cache has
not warmed yet, call
`ConsoleRuntime.ensure_activity_hydration()` directly without awaiting its
off-loop storage work; completion verifies the current runtime authority token
and requests Active reconciliation through the projection-generation guard.
Concurrent calls coalesce, a later call retries only `degraded`, and runtime
disposal invalidates the callback. Inject a post-migration receipt read
failure and assert open-session Active rows and bounded History search still
work while the local-activity-unavailable status is visible.

- [ ] **Step 6: Run projection/service tests**

Run:

```bash
pytest Tests/Chat/test_console_switcher_state.py Tests/UI/test_console_activity_switcher.py -q
```

Expected: PASS with immediate Active projection, bounded off-loop History, and
no Context/Inspector projection mutation.

- [ ] **Step 7: Commit projection and paging**

```bash
git add tldw_chatbook/Chat/console_switcher_state.py tldw_chatbook/UI/Console_Modules/workspace.py Tests/Chat/test_console_switcher_state.py Tests/UI/test_console_activity_switcher.py
git commit -m "feat(console): project active switcher rows"
```

### Task 5: Replace the eager modal with Active/History interaction

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_session_switcher_modal.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Console_Modules/workspace.py`
- Modify: `Tests/UI/test_console_activity_switcher.py`
- Modify: `Tests/UI/test_console_native_chat_flow.py`

- [ ] **Step 1: Write RED modal interaction and geometry tests**

Under `Tests.UI.consolidated_css.ConsolidatedCSSApp`, cover Active-on-open,
search focus, automatic zero-match History widening, F3 query retention, F2
focused-native-only behavior, immutable button payloads, Cancel/Esc, pointer
parity, page actions, stable focus across live reorder, and deterministic focus
fallback when any result variant disappears. In the production-path case, block History,
press Ctrl+K, and prove Active paints and filters before History is released.
When the top search result is unavailable, assert the first Enter only focuses
its `Mark seen` action and a second Enter performs the explicit acknowledgement.

Add async barriers proving:

- Enter while a query is pending cannot activate an old row;
- a late query/page/profile/modal generation cannot replace current state; and
- closing during a load produces no remount or callback.

- [ ] **Step 2: Implement the bounded modal structure**

Compose heading, literal selected-mode controls (`Active (N) — selected |
History` or `Active (N) | History — selected`), search, grouped result scroll,
the persistent one-row `#console-switcher-status`, status/page actions, truthful
hints, and visible Cancel. The status line renders exact selection ordinal/title
or mode/loading/error copy; forced live-reconciliation focus fallback also uses
the existing app notification channel with identical text. Cap the complete
modal at 35 rows and mount at most 50 result rows total. Conversation subjects
use selectable two-row buttons; unavailable notices use two-row receipt-keyed
`Mark seen` actions. Store a mapping from stable widget ID to the frozen union
payload; conversation variants carry a target and notice variants carry the
explicit receipt action. Never derive behavior from index. Bounded
mode/page/Cancel controls sit
outside the 50-row result count.
Session-only receipts whose ephemeral target vanished render as non-subject
`Session unavailable` notices with receipt-keyed `Mark seen` actions; they never
activate or fall back to another row.

- [ ] **Step 3: Implement generation-gated local async work**

Capture modal instance, mode, query generation, stable profile authority,
`ConsoleRuntime.authority_token`, Active projection generation, and page request.
Commit only if every captured value still matches the current runtime and the
modal is mounted. Changing query/mode resets page one; blank Active never calls
History. No mode action performs network I/O.

Migrate `ChatScreen.action_open_console_session_switcher` in this same task: it
opens from the immediate Active snapshot, calls
`ConsoleRuntime.ensure_activity_hydration()` without awaiting it, passes the
lazy History loader, and stops awaiting the compatibility
`console_session_switcher_rows()` adapter. Remove that compatibility adapter in
this task only after the caller and modal migration are covered together.

- [ ] **Step 4: Implement exact keyboard/pointer behavior**

Add modal-scoped F3, Up/Down without wrap, Up-first-to-search, Enter on a focused
conversation row or current-query top conversation result, F2 only on a focused renameable native row,
Tab/Shift+Tab visual order, Esc safe dismissal, and always-visible Cancel.
If search submission's top result is an unavailable notice, move focus to its
`Mark seen` action and update status without acknowledging; only a subsequent
Enter on that focused action or a pointer click acknowledges.
Keep footer/hints exactly aligned with implemented actions.

- [ ] **Step 5: Assert painted rows, not only styles**

At widths 52, 72, and 120 and heights 20/35/50, inspect compositor text and
visible-widget containment. Assert state/destination tokens survive, omission
order is lifecycle/workspace/recency, labels never wrap past two rows, Cancel is
visible/reachable, selected mode and selection status are literal text, forced
focus fallback emits the same notification copy, and total modal height is
`<= 35`.

- [ ] **Step 6: Run modal and incumbent draft-integrity suites**

Run:

```bash
pytest Tests/UI/test_console_activity_switcher.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_switch_draft_integrity.py -q
```

Expected: PASS; update the old F2-fallback test to assert the approved no-op.

- [ ] **Step 7: Commit the modal**

```bash
git add tldw_chatbook/Widgets/Console/console_session_switcher_modal.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Console_Modules/workspace.py Tests/UI/test_console_activity_switcher.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_switch_draft_integrity.py
git commit -m "feat(console): add active and history switcher modes"
```

### Task 6: Add exact destination notices and acknowledgement

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_activity_outcome_notice.py`
- Create: `Tests/UI/test_console_activity_outcome_notice.py`
- Modify: `tldw_chatbook/Widgets/Console/console_session_surface.py`
- Modify: `tldw_chatbook/UI/Console_Modules/session.py`
- Modify: `Tests/UI/test_console_activity_switcher.py`
- Modify: `Tests/UI/test_console_native_chat_flow.py`

- [ ] **Step 1: Write RED notice and activation tests**

Drive the real `_apply_console_switcher_choice` route for native session and
persisted conversation targets. Cover success, mixed success/failure, failure,
new receipt during navigation, missing destination, profile change, dismissal,
receipt-service failure, switch-away before post-refresh, notice replacement,
notice hide, and notice remount. Assert no unrelated session fallback and no
acknowledgement from a stale presentation callback.

- [ ] **Step 2: Always mount a compact destination notice**

Add `ConsoleActivityOutcomeNotice` between task cards and transcript. Keep it
mounted with `display: none` when inactive so hot updates never require a
structural recompose. Its state contains literal-safe status/copy and the frozen
`CapturedReceipt` evidence selected in the modal, including whether `Mark seen`
is required, plus a monotonically increasing presentation generation. Showing,
replacing, hiding, or unmounting the notice increments that generation. Give the
button ordinary Tab/Enter and pointer behavior; add no global binding.

- [ ] **Step 3: Dispatch only explicit activation targets**

In `_apply_console_switcher_choice`, match `target.kind` and call only the named
native-session or persisted-conversation path. Revalidate captured stable profile
authority, runtime authority token, and destination identity. Navigation failure
leaves every receipt unacknowledged and shows existing honest recovery copy.

- [ ] **Step 4: Acknowledge only after visible paint**

After activation and `_sync_native_console_chat_ui()`, show the receipt-keyed
notice, capture destination identity, exact receipt IDs, and notice presentation
generation, then use the mounted notice's `call_after_refresh` confirmation. On
that confirmation, first revalidate the same destination is still selected and
the same notice is mounted, displayed, and current at the captured generation;
otherwise return without acknowledgement. A valid callback acknowledges only
IDs whose frozen captured status is `done`. For
`failed/stuck/stopped/cancelled`, acknowledge only from the notice button. If a
captured ID is superseded or a newer revision exists, the service's exact-ID
update cannot hide the newer row. Never re-read current receipt status to decide
which captured outcomes auto-acknowledge.

- [ ] **Step 5: Prove the paint boundary and focus safety**

Hold `call_after_refresh`, assert successful receipts remain unseen, release it,
assert the notice text is in the compositor, then assert acknowledgement. Verify
switch-away, replacement, hide, and remount each invalidate the captured
presentation generation and leave receipts unseen. Verify
the activation settle does not steal focus or reorder text typed immediately in
the composer; retain the existing draft-integrity test as a reachable guard.

- [ ] **Step 6: Run outcome and production-path tests**

Run:

```bash
pytest Tests/UI/test_console_activity_outcome_notice.py Tests/UI/test_console_activity_switcher.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_switch_draft_integrity.py -q
```

Expected: PASS, including mixed receipts and new-arrival race.

- [ ] **Step 7: Commit acknowledgement UX**

```bash
git add tldw_chatbook/Widgets/Console/console_activity_outcome_notice.py tldw_chatbook/Widgets/Console/console_session_surface.py tldw_chatbook/UI/Console_Modules/session.py Tests/UI/test_console_activity_outcome_notice.py Tests/UI/test_console_activity_switcher.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_switch_draft_integrity.py
git commit -m "feat(console): acknowledge visible activity outcomes"
```

### Task 7: Documentation, performance, and final evidence

**Files:**
- Modify: `Docs/User_Guide/console/sessions-tabs-workspaces.md`
- Create: `Docs/superpowers/qa/task-21351-console-switcher-activity/README.md`
- Modify: `backlog/tasks/task-21351 - Add-activity-views-to-CtrlK-session-switcher.md`
- Modify: the resolved Phase 1 child task file created at the dependency gate

When the child is created, replace this descriptive file-list entry and the
Step 9 placeholder with its exact `backlog/tasks/...md` path before beginning
implementation; do not leave an unresolved child path in the executable plan.

- [ ] **Step 1: Update the user guide**

Document Active groups, History and automatic widening, F3/F2/arrow/Enter/Esc,
successful auto-ack versus `Mark seen`, the 35-row/scroll behavior, and the
local-activity-unavailable degradation that leaves History usable.

- [ ] **Step 2: Run focused lint and format checks**

Run Ruff against only changed Python files, then the repository formatter/check
appropriate to the branch. Do not mass-format unrelated code.

- [ ] **Step 3: Run all reachable automated suites**

At minimum:

```bash
pytest Tests/DB/test_agent_runs_db.py Tests/Chat/test_console_activity_receipts.py Tests/Chat/test_fleet_attention.py Tests/Chat/test_console_fleet_wake.py Tests/Chat/test_console_prompt_queue_coordinator.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_switcher_state.py Tests/UI/test_console_launch_wake.py Tests/UI/test_console_runtime_ownership.py Tests/UI/test_console_activity_switcher.py Tests/UI/test_console_activity_outcome_notice.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_switch_draft_integrity.py -q
```

Then run the full test suite before PR closeout. Compare any failure set against
the identical command on untouched current `dev`; counts alone are insufficient.

- [ ] **Step 4: Measure bounded work**

Record median/p95 modal open, Active filter, zero-match widening, F3 toggle,
History page, and one receipt-cache refresh at small/representative/stress data
sizes. Record mounted selectable widgets and database rows materialized. Do not
claim a speed improvement without measured evidence.

- [ ] **Step 5: Verify the real user path**

Launch with an isolated profile and ensure this worktree wins import ordering.
Drive: background success → Ctrl+K → Enter → visible notice → auto-ack; background
failure → Ctrl+K → Enter → visible notice → `Mark seen`; History search and page;
F2 no-fallback; Cancel/Esc; immediate typing after activation. Capture compositor
or SVG evidence, not style values alone.

- [ ] **Step 6: Record equal-cell terminal parity**

After TASK-20937.6 is complete, capture the same rows/columns in iTerm2 and
Windows Terminal. Confirm complete modal `<= 35` rows, two-row results, visible
Cancel/page actions, identical search/mode behavior, and no rail ownership
regression. Missing Windows Terminal access blocks closeout.

- [ ] **Step 7: Self-review and request code review**

Review the diff against every TASK-21351 criterion, ADR-085, ADR-031, and the
approved spec. Use `superpowers:requesting-code-review`; resolve correctness
findings before final verification.

- [ ] **Step 8: Complete task hygiene only after every gate passes**

Check every child acceptance criterion, add concise child Implementation Notes
naming approach/tradeoffs/files/evidence, state whether a lessons entry was
warranted, and set the child Done through Backlog CLI. Then check the umbrella
TASK-21351 criteria, add its roll-up Implementation Notes, and set the parent
Done only after the child and all parent-level evidence are complete. Re-read
both tasks afterward because the CLI can replace free-form sections.

- [ ] **Step 9: Commit documentation and closeout evidence**

```bash
git add Docs/User_Guide/console/sessions-tabs-workspaces.md Docs/superpowers/qa/task-21351-console-switcher-activity/README.md 'backlog/tasks/task-21351 - Add-activity-views-to-CtrlK-session-switcher.md' '<resolved Phase 1 child task path>'
git commit -m "docs(console): verify activity switcher"
```

## Final verification checklist

- [ ] TASK-20937 and TASK-20937.6 are Done on the branch baseline.
- [ ] ADR-085 remains Accepted and matches the implementation.
- [ ] No Phase 2 server integration entered the diff.
- [ ] `git diff --check` is clean.
- [ ] Focused DB/Chat/Workspace/UI suites pass.
- [ ] Full pytest and scoped static checks pass or have an identical documented
  current-dev failure set.
- [ ] Production-stylesheet compositor evidence proves geometry and visible copy.
- [ ] Real Ctrl+K activation proves the receipt-to-visible-notice boundary.
- [ ] Equal-cell iTerm2/Windows Terminal evidence is recorded.
- [ ] The implementation child is complete with criteria/notes/Done before the
  umbrella TASK-21351 criteria/notes/Done transition.
- [ ] ADR-085 is re-swept against `origin/dev` and open PRs; any collision is
  renumbered across the file, header, task, spec, and plan before merge.
