# Console Watchlists Commands and Operations Implementation Plan

> Execution: use `superpowers:test-driven-development` per backlog task and
> `superpowers:verification-before-completion` at every commit boundary.

**Goal:** Let the Console agent safely author Watchlists, accept durable source
checks and briefing generation, follow their exact receipts, and create an
honestly observable every-24-hours schedule.

**Architecture:** A synchronous `WatchlistsCommandService` validates and shapes
Console tool calls but delegates all state changes to application-owned domain
services. Short database-local mutations execute directly on the Console tool
worker through application-owned synchronous domain seams and return only
after commit or rollback. Long work is accepted durably, then owned by one
app-lifetime `WatchlistsOperationCoordinator`; navigation and model timeouts
cannot cancel it. `SchedulerLoop` owns queue reload acknowledgement and wake-up.

**Tech stack:** Python worker execution plus app-loop ownership for accepted
long work, SQLite
transactions/partial constraints from TASK-22860, Textual state/cards, pytest
and pytest-asyncio.

**Backlog tasks:** TASK-22862 → TASK-22863 → TASK-22864.

**ADR required:** yes

**ADR path:** `backlog/decisions/032-local-agent-tool-permission-boundary.md`
for Console-only commands; existing `backlog/decisions/019-watchlist-scheduler-migration.md`
governs the unified scheduler ownership.

**Reason:** The approved ADR-032 addendum defines the command/exposure boundary.
The scheduling task extends the existing SchedulerLoop contract without adding
another scheduler or calendar model.

## TASK-22862 — Transactional Console authoring commands

### Files

- Create: `tldw_chatbook/Tools/watchlists_command_service.py`
- Create: `Tests/Tools/test_watchlists_command_service.py`
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py`
- Modify: `tldw_chatbook/Subscriptions/local_watchlists_service.py`
- Modify: `tldw_chatbook/Subscriptions/watchlist_bundle_service.py`
- Modify: `tldw_chatbook/Subscriptions/watchlist_opml_service.py`
- Modify: `tldw_chatbook/Agents/local_tool_provider.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `Tests/DB/test_subscriptions_db_watchlists.py`
- Modify: `Tests/Subscriptions/test_local_watchlists_service.py`
- Modify: `Tests/Subscriptions/test_watchlist_bundle_service.py`
- Modify: `Tests/Subscriptions/test_watchlist_opml_service.py`
- Modify: `Tests/Agents/test_local_tool_provider.py`
- Modify: `Tests/Chat/test_console_local_review_hook.py`

### Step 1: Pin exact source identity under concurrent writers

Add real-file SQLite tests with two service/DB instances racing the same source
string. Pin identity as `source.strip()` only: do not lowercase paths, reorder
queries, or remove query parameters before comparison. Test within-request
duplicates, pre-existing rows, URL userinfo rejection, booleans in integer
fields, query redaction, and the 50-row ceiling.

Add `SubscriptionsDB.create_sources_exact_batch(rows: Sequence[Mapping[str,
Any]]) -> list[dict[str, Any]]` as the exact DB-owner API with explicit write
intent. The command facade supplies already validated mappings; each returned
mapping carries `input_index`, `outcome`, and canonical `source_id` when one
exists.

The method enters `BEGIN IMMEDIATE` before lookup/insert and preserves input
order. Add synchronous and async/off-loop `LocalWatchlistsService` wrappers
around the same owner operation. The command facade uses the synchronous seam
on its existing Console tool worker; direct async UI and OPML paths retain the
async wrapper. Route existing `create_source()` through the same DB-owner seam
so no caller can bypass serialization.

Run:

```bash
pytest -q Tests/DB/test_subscriptions_db_watchlists.py Tests/Subscriptions/test_local_watchlists_service.py Tests/Subscriptions/test_watchlist_opml_service.py -k "source and (batch or duplicate or concurrent or identity)"
```

### Step 2: Pin atomic collection collision and membership behavior

Add failing `WatchlistBundleService` tests for `conflict`, `return_existing`,
and `auto_suffix`; up to 100 memberships; missing source rollback; and a
membership insert failure after collection insertion.

Add these owner methods:

```python
def create_with_sources(
    self,
    name: str,
    *,
    description: str | None,
    tags: Sequence[str] | None,
    source_ids: Sequence[int],
    if_exists: CollisionPolicy,
) -> dict[str, Any]:
    """Create or resolve one collection under an explicit collision policy."""

def update_sources(
    self,
    watchlist_id: int,
    *,
    add_ids: Sequence[int],
    remove_ids: Sequence[int],
) -> dict[str, Any]:
    """Apply one all-or-nothing membership update."""
```

Validate every referenced row before mutation. Returning an existing
collection must not change its settings or membership. Creation plus membership
and add/remove updates each run in one transaction.

Run:

```bash
pytest -q Tests/Subscriptions/test_watchlist_bundle_service.py -k "collision or membership or atomic or rollback"
```

### Step 3: Build the synchronous command facade

Write failing service tests for exact schemas/outcomes, partial-success stop,
server-mode refusal before DB resolution, unexpected-error scrubbing, and the
absence of implicit collection/check/generate/schedule side effects.

Implement `WatchlistsCommandService` with injected callables rather than app or
widget imports:

```python
class WatchlistsCommandService:
    def create_sources(self, arguments: object) -> str:
        """Validate and create one bounded source batch."""

    def create_collection(self, arguments: object) -> str:
        """Validate and atomically create or resolve one collection."""

    def update_collection_sources(self, arguments: object) -> str:
        """Validate and atomically replace the requested memberships."""
```

Reuse canonical ID/redaction/bounding utilities extracted from
`watchlists_tool_service.py` into a small shared helper only when two real
callers exist. Invoke the injected synchronous mutation seams directly on the
existing Console tool worker. Do not submit short SQLite mutations back to the
app loop, wrap them in a non-cancellable timeout, or call `asyncio.run()`; a
tool result is emitted only after the owner has committed or rolled back.

Run:

```bash
pytest -q Tests/Tools/test_watchlists_command_service.py
```

### Step 4: Register Console-only mutation descriptors

Inject the command service in `ConsoleChatController._compose_local_provider`
from the app's shared `subscriptions_db`, `local_watchlists_service`, and
`watchlist_bundle_service`. Register these exact Console-only tools:

- `watchlists_create_sources`
- `watchlists_create_collection`
- `watchlists_update_collection_sources`

Each carries `("mutates",)` plus `MUTATES_LOCAL`; source creation also reports
the destinations' sanitized hosts in approval presentation. Read-only project
bindings must omit all three.

Run:

```bash
pytest -q Tests/Agents/test_local_tool_provider.py Tests/Chat/test_console_local_review_hook.py Tests/MCP/test_local_server_tools.py -k "watchlists and (create or update or external or read_only)"
```

### Step 5: Verify and commit TASK-22862

```bash
pytest -q Tests/Tools/test_watchlists_command_service.py Tests/Subscriptions/test_local_watchlists_service.py Tests/Subscriptions/test_watchlist_bundle_service.py Tests/Subscriptions/test_watchlist_opml_service.py
ruff check tldw_chatbook/Tools/watchlists_command_service.py tldw_chatbook/DB/Subscriptions_DB.py tldw_chatbook/Subscriptions/local_watchlists_service.py tldw_chatbook/Subscriptions/watchlist_bundle_service.py tldw_chatbook/Subscriptions/watchlist_opml_service.py tldw_chatbook/Agents/local_tool_provider.py tldw_chatbook/Chat/console_chat_controller.py
git diff --check
```

Commit boundary:

```bash
git add tldw_chatbook/Tools/watchlists_command_service.py tldw_chatbook/DB/Subscriptions_DB.py tldw_chatbook/Subscriptions/local_watchlists_service.py tldw_chatbook/Subscriptions/watchlist_bundle_service.py tldw_chatbook/Subscriptions/watchlist_opml_service.py tldw_chatbook/Agents/local_tool_provider.py tldw_chatbook/Chat/console_chat_controller.py Tests/Tools/test_watchlists_command_service.py Tests/DB/test_subscriptions_db_watchlists.py Tests/Subscriptions/test_local_watchlists_service.py Tests/Subscriptions/test_watchlist_bundle_service.py Tests/Subscriptions/test_watchlist_opml_service.py Tests/Agents/test_local_tool_provider.py Tests/Chat/test_console_local_review_hook.py backlog/tasks/task-22862\ -\ Add-transactional-Console-Watchlists-authoring-commands.md
git commit -m "feat: add transactional Console Watchlists commands"
```

## TASK-22863 — Durable source-check and briefing coordination

### Files

- Create: `tldw_chatbook/Subscriptions/watchlists_operation_coordinator.py`
- Create: `Tests/Subscriptions/test_watchlists_operation_coordinator.py`
- Create: `tldw_chatbook/Widgets/Chat_Widgets/watchlists_operation_card.py`
- Create: `Tests/Widgets/test_watchlists_operation_card.py`
- Modify: `tldw_chatbook/css/features/_chat.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `tldw_chatbook/Subscriptions/local_watchlists_service.py`
- Modify: `tldw_chatbook/Subscriptions/briefing_service.py`
- Modify: `tldw_chatbook/Tools/watchlists_command_service.py`
- Modify: `tldw_chatbook/Agents/local_tool_provider.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen_state.py`
- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/Tools/test_watchlists_command_service.py`
- Modify: `Tests/Subscriptions/test_local_watchlists_service.py`
- Modify: `Tests/Subscriptions/test_briefing_service.py`
- Modify: `Tests/Chat/test_console_local_review_hook.py`
- Modify: `Tests/UI/test_watchlists_check_now_progress.py`
- Modify: `Tests/Watchlists/test_watchlists_artifacts_pane.py`

### Step 1: Split durable acceptance from execution

Add failing tests for these public domain contracts:

```python
async def accept_source_checks(
    source_ids: Sequence[int],
) -> list[RunReceipt]:
    """Commit or resolve one receipt for each validated source."""

async def execute_accepted_run(run_id: int) -> dict[str, Any]:
    """Execute exactly the already accepted run row."""

async def accept_briefing(
    watchlist_id: int, preset_id: int | None
) -> BriefingReceipt:
    """Commit or resolve one generating receipt for a collection."""

async def execute_accepted_briefing(briefing_id: int) -> dict[str, Any]:
    """Generate exactly the already accepted briefing row."""
```

Acceptance validates all entities, uses TASK-22860's DB constraint, and returns
the winning existing active receipt after a race. Execution transitions that
same row; it never inserts a second receipt. Refactor `generate_briefing`
internals behind these public seams without exposing `_start_generation` or
other private helpers.

Run:

```bash
pytest -q Tests/Subscriptions/test_local_watchlists_service.py Tests/Subscriptions/test_briefing_service.py -k "accept or accepted or active_receipt or race"
```

### Step 2: Implement the app-lifetime coordinator

Add coordinator tests for a four-check semaphore, strong task retention,
duplicate submissions, task exceptions, model/tool timeout independence,
navigation independence, stop-accepting, bounded shutdown, and startup
reconciliation.

Implement one coordinator owned by `TldwCli`:

```python
class WatchlistsOperationCoordinator:
    async def accept_checks(
        self, source_ids: Sequence[int]
    ) -> list[RunReceipt]:
        """Accept validated checks and schedule their durable execution."""

    async def accept_briefing(
        self, watchlist_id: int, preset_id: int | None
    ) -> BriefingReceipt:
        """Accept one briefing and schedule its durable execution."""

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Stop acceptance and reconcile coordinator-owned tasks."""
```

The coordinator uses `asyncio.create_task` only on the app loop, stores tasks
by canonical receipt ID, consumes every terminal exception, and removes a task
only after durable terminalization. Startup reconciliation handles receipts
left active by process loss. Do not make the in-memory map authoritative.

Run:

```bash
pytest -q Tests/Subscriptions/test_watchlists_operation_coordinator.py Tests/Watchlists/test_startup_reconcile_scheduler_race.py
```

### Step 3: Add the two long-running command tools

Extend `WatchlistsCommandService` and `_default_specs` with Console-only:

- `watchlists_check_sources`
- `watchlists_generate_briefing`

Validate the 50-source/one-collection boundary before acceptance. Return
`accepted` content immediately with canonical receipt IDs and exact:

```json
{
  "poll_tool": "watchlists_get_operation_status",
  "poll_arguments": {"operation_id": "local:briefing:42"},
  "suggested_poll_seconds": 2,
  "terminal_states": ["complete", "empty", "failed", "cancelled"]
}
```

Source checks carry `MUTATES_LOCAL` and `NETWORK`; briefing generation carries
`MUTATES_LOCAL` and `LLM_SPEND`.

Run:

```bash
pytest -q Tests/Tools/test_watchlists_command_service.py Tests/Agents/test_local_tool_provider.py -k "check_sources or generate_briefing or poll"
```

### Step 4: Route direct UI actions through the same ownership boundary

Replace screen-owned long workers for Check Now and Generate with coordinator
acceptance/following. The Watchlists screen may follow/reload receipts but must
not own accepted execution. Preserve the existing direct UI user flow and its
failure notifications.

Run:

```bash
pytest -q Tests/UI/test_watchlists_check_now_progress.py Tests/UI/test_watchlists_check_now_failure.py Tests/Watchlists/test_watchlists_artifacts_pane.py -k "generate or check or navigation or receipt"
```

### Step 5: Render durable Console receipt state

Create one small operation card for queued/running/complete/empty/failed and
the exact Runs/Artifacts destination. Store only canonical receipt identity in
Console screen state; refresh content from `watchlists_get_operation_status` or
the shared query service. “Stop following” must not cancel domain work. Show
Retry/Cancel only when the receipt reports support.

Use production Textual hierarchy and put all new rules in `_chat.tcss`; do not
add class-level `DEFAULT_CSS` or duplicate those rules in `BUNDLED_CSS`.
Regenerate the checked-in app bundle. Pin state-word plus non-color indicators,
focus, navigation survival, and bounded error copy. `ChatScreen` polls every
two seconds only while at least one followed receipt is nonterminal and the
screen is mounted; unmount stops following but never cancels the domain
operation.

Run:

```bash
pytest -q Tests/Widgets/test_watchlists_operation_card.py Tests/UI/test_watchlists_check_now_progress.py Tests/Chat/test_console_local_review_hook.py
python -m tldw_chatbook.css.build_css
python tldw_chatbook/css/check_bundle_sync.py
```

### Step 6: Verify and commit TASK-22863

```bash
pytest -q Tests/Subscriptions/test_watchlists_operation_coordinator.py Tests/Subscriptions/test_local_watchlists_service.py Tests/Subscriptions/test_briefing_service.py Tests/Tools/test_watchlists_command_service.py
pytest -q Tests/Widgets/test_watchlists_operation_card.py Tests/UI/test_watchlists_check_now_progress.py Tests/UI/test_watchlists_check_now_failure.py Tests/Watchlists/test_watchlists_artifacts_pane.py
ruff check tldw_chatbook/Subscriptions/watchlists_operation_coordinator.py tldw_chatbook/Subscriptions/local_watchlists_service.py tldw_chatbook/Subscriptions/briefing_service.py tldw_chatbook/Tools/watchlists_command_service.py tldw_chatbook/Agents/local_tool_provider.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Widgets/Chat_Widgets/watchlists_operation_card.py tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py tldw_chatbook/UI/Screens/chat_screen_state.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Screens/watchlists_collections_screen.py tldw_chatbook/app.py
python tldw_chatbook/css/check_bundle_sync.py
git diff --check
```

Commit boundary:

```bash
git add tldw_chatbook/Subscriptions/watchlists_operation_coordinator.py tldw_chatbook/Subscriptions/local_watchlists_service.py tldw_chatbook/Subscriptions/briefing_service.py tldw_chatbook/Tools/watchlists_command_service.py tldw_chatbook/Agents/local_tool_provider.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Widgets/Chat_Widgets/watchlists_operation_card.py tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py tldw_chatbook/UI/Screens/chat_screen_state.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Screens/watchlists_collections_screen.py tldw_chatbook/app.py tldw_chatbook/css/features/_chat.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/Subscriptions/test_watchlists_operation_coordinator.py Tests/Subscriptions/test_local_watchlists_service.py Tests/Subscriptions/test_briefing_service.py Tests/Tools/test_watchlists_command_service.py Tests/Widgets/test_watchlists_operation_card.py Tests/Chat/test_console_local_review_hook.py Tests/UI/test_watchlists_check_now_progress.py Tests/UI/test_watchlists_check_now_failure.py Tests/Watchlists/test_watchlists_artifacts_pane.py backlog/tasks/task-22863\ -\ Coordinate-durable-Watchlists-check-and-briefing-operations.md
git commit -m "feat: coordinate durable Watchlists operations"
```

## TASK-22864 — Every-24-hours schedule observability

### Files

- Modify: `tldw_chatbook/Scheduling/scheduler/loop.py`
- Modify: `tldw_chatbook/Scheduling/services/briefing_projection.py`
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py`
- Modify: `tldw_chatbook/Tools/watchlists_command_service.py`
- Modify: `tldw_chatbook/Agents/local_tool_provider.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py`
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/Scheduling/test_scheduler_loop.py`
- Modify: `Tests/Scheduling/test_briefing_projection.py`
- Modify: `Tests/Subscriptions/test_briefing_cadence_db.py`
- Modify: `Tests/Tools/test_watchlists_command_service.py`
- Modify: `Tests/Watchlists/test_watchlists_artifacts_pane.py`

### Step 1: Pin the approved interval semantics

Add failing projection/DB tests for:

- never attempted ⇒ eligible now;
- latest attempt (including failure) + interval;
- overdue ⇒ eligible when scheduler resumes;
- `every_24_hours` ⇒ exactly 86,400 seconds;
- `off` preserves receipts/preset/selection;
- optional cadence/preset/selection commit is atomic;
- omitted preset/model input reuses the collection's stored briefing preset,
  then the briefing pipeline's existing app/provider defaults, and never the
  current Console conversation model.

Keep UTC storage/comparison; pass an injected clock and timezone only to
display formatting.

Run:

```bash
pytest -q Tests/Scheduling/test_briefing_projection.py Tests/Subscriptions/test_briefing_cadence_db.py
```

### Step 2: Replace the bool reload flag with a token/ack contract

Write scheduler-loop tests before implementation. The public shape is:

```python
def request_reload(self) -> QueueReloadToken:
    """Wake the loop and return a monotonic reload request token."""

async def wait_for_reload(
    self, token: QueueReloadToken, timeout: float
) -> bool:
    """Return true only after a successful load acknowledges this token."""
```

`request_reload` is thread-safe, monotonically identifies the request, and
wakes the sleeping loop through `call_soon_threadsafe`. The run loop waits for
either the normal poll deadline or a reload event. It acknowledges the token
only after `queue.load` succeeds. A stopped loop, timeout, or load failure
never reports acknowledgement. Coalesce requests while acknowledging every
token covered by the successful load.

Run:

```bash
pytest -q Tests/Scheduling/test_scheduler_loop.py -k "reload or wake or acknowledge"
```

### Step 3: Add and wire `watchlists_set_briefing_schedule`

Add command-service RED tests for vocabulary, advanced range, booleans,
canonical IDs, server mode, persistence failure, gate disabled, loop stopped,
reload timeout, and successful acknowledgement.

Implement the Console-only command with `MUTATES_LOCAL`. Its receipt reports
stored cadence, global gate, loop state, requested token, acknowledged bool,
next eligible time, last attempt/success, the effective briefing-preset/model
resolution source, and fixed recovery copy. Do not claim acknowledgement from
the persisted write or request alone.

Run:

```bash
pytest -q Tests/Tools/test_watchlists_command_service.py Tests/Agents/test_local_tool_provider.py -k "schedule or cadence or reload"
```

### Step 4: Share receipt semantics with Artifacts/Settings

Replace “Daily” with “Every 24 hours.” Refresh the Artifacts automation receipt
after save, including stored-but-disabled and stopped-loop states. Keep Settings
as the app-level gate owner and Artifacts as per-collection cadence owner.

Run production-shaped Textual tests at the existing pressure-point size and a
normal size.

```bash
pytest -q Tests/Watchlists/test_watchlists_artifacts_pane.py Tests/UI/test_schedules_workbench.py -k "cadence or schedule or reload or disabled"
```

### Step 5: Verify and commit TASK-22864

```bash
pytest -q Tests/Scheduling/test_scheduler_loop.py Tests/Scheduling/test_briefing_projection.py Tests/Scheduling/test_briefing_handler.py Tests/Subscriptions/test_briefing_cadence_db.py Tests/Tools/test_watchlists_command_service.py Tests/Watchlists/test_watchlists_artifacts_pane.py
ruff check tldw_chatbook/Scheduling/scheduler/loop.py tldw_chatbook/Scheduling/services/briefing_projection.py tldw_chatbook/DB/Subscriptions_DB.py tldw_chatbook/Tools/watchlists_command_service.py tldw_chatbook/Agents/local_tool_provider.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py tldw_chatbook/UI/Screens/watchlists_collections_screen.py tldw_chatbook/app.py
git diff --check
```

Commit boundary:

```bash
git add tldw_chatbook/Scheduling/scheduler/loop.py tldw_chatbook/Scheduling/services/briefing_projection.py tldw_chatbook/DB/Subscriptions_DB.py tldw_chatbook/Tools/watchlists_command_service.py tldw_chatbook/Agents/local_tool_provider.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py tldw_chatbook/UI/Screens/watchlists_collections_screen.py tldw_chatbook/app.py Tests/Scheduling/test_scheduler_loop.py Tests/Scheduling/test_briefing_projection.py Tests/Subscriptions/test_briefing_cadence_db.py Tests/Tools/test_watchlists_command_service.py Tests/Watchlists/test_watchlists_artifacts_pane.py backlog/tasks/task-22864\ -\ Make-every-24-hours-briefing-schedules-immediately-observable.md
git commit -m "feat: acknowledge briefing schedule reloads"
```

## Plan-level self-review gate

- No command is published to external MCP.
- No tool drives a Textual widget or waits for network/LLM completion.
- A returned accepted receipt always exists durably first.
- Duplicate active work resolves to the winning receipt across processes.
- App shutdown reaches every coordinator-owned task.
- “Requested” and “acknowledged” are never conflated.
- No operation or schedule receipt includes raw exception text, signed queries,
  credentials, headers, or database paths.
