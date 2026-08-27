# Watchlists Agent Boundary and Provenance Implementation Plan

> Execution: use `superpowers:test-driven-development` for each task, and use
> `superpowers:verification-before-completion` before any completion claim.

**Goal:** Establish a fail-closed Console/external-MCP tool boundary, migrate
ordered durable briefing provenance and atomic operation claims, and expose the
bounded query surface that lets the user's Console agent consume briefings.

**Architecture:** `LocalToolSpec` is the single descriptor owner for exposure,
approval effects, schema, and handler. External MCP derives publication from
that descriptor; it does not maintain a tool-name denylist. `SubscriptionsDB`
owns the v1→v2 migration, active-claim constraints, and atomic briefing
completion. `WatchlistsToolService` remains the synchronous bounded shaping
facade used by both Console and read-only external MCP.

**Tech stack:** Python 3.11+, dataclasses/typing, SQLite partial indexes and
transactions, Textual approval cards, pytest/pytest-asyncio/jsonschema.

**Backlog tasks:** TASK-22859 → TASK-22860 → TASK-22861.

**ADR required:** yes

**ADR path:** `backlog/decisions/032-local-agent-tool-permission-boundary.md`

**Reason:** ADR-032 owns the synthetic local principal, approval semantics,
external MCP publication, and existing Watchlists reads. The addendum records
the new exposure/effect contract and Console-only private/mutating boundary.

## Task 1 — TASK-22859: Define Watchlists Console tool exposure and approval effects

### Files

- Modify: `tldw_chatbook/Agents/local_tool_provider.py`
- Modify: `tldw_chatbook/Agents/mcp_tool_provider.py`
- Modify: `tldw_chatbook/MCP/local_server_tools.py`
- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `Tests/Agents/test_local_tool_provider.py`
- Modify: `Tests/MCP/test_local_server_tools.py`
- Modify: `Tests/Chat/test_console_local_review_hook.py`
- Modify: `Tests/UI/test_chat_approval_card.py`
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md`
- Modify: `backlog/decisions/032-local-agent-tool-permission-boundary.md`

### Step 1: Pin fail-closed descriptor construction

Add failing tests that instantiate a `LocalToolSpec` without exposure, with an
unknown exposure, and with an unknown approval effect. Pin the public values:

```python
class LocalToolExposure(StrEnum):
    CONSOLE_AND_EXTERNAL_MCP = "console_and_external_mcp"
    CONSOLE_ONLY = "console_only"

class LocalApprovalEffect(StrEnum):
    PRIVATE_READ = "private_read"
    MUTATES_LOCAL = "mutates_local"
    NETWORK = "network"
    LLM_SPEND = "llm_spend"
```

Require every default descriptor to declare `exposure=` and
`approval_effects=`. Do not add permissive defaults.

Run:

```bash
pytest -q Tests/Agents/test_local_tool_provider.py -k "exposure or effect or catalog"
```

Expected RED: current descriptors have no exposure/effect contract.

### Step 2: Implement the descriptor contract and explicit inventory

Add the two enums and required frozen fields to `LocalToolSpec`. Classify every
incumbent descriptor explicitly. Keep permission-enforced `tags` separate from
human-facing effects; mutation descriptors still carry `("mutates",)`.
Classify `watchlists_search_items`, `watchlists_get_item`, and
`watchlists_get_briefing` as `CONSOLE_ONLY`; only bounded source, collection,
operation, and briefing-receipt metadata may be
`CONSOLE_AND_EXTERNAL_MCP`.

Add these provider helpers:

```python
def specs_for_exposure(
    self, exposure: LocalToolExposure
) -> tuple[LocalToolSpec, ...]:
    """Return descriptors carrying exactly the requested exposure."""

def approval_effects_for(
    self, tool_id: str
) -> tuple[LocalApprovalEffect, ...]:
    """Return the code-owned effects for one registered local tool."""
```

Change `allow_write=False` filtering to use `MUTATES_LOCAL`, so it also omits
future Watchlists mutations instead of naming only `fs_write`, `fs_edit`, and
`fs_patch`.

Run the Step 1 selection and make it GREEN.

### Step 3: Derive external MCP publication from exposure

Add a failing `Tests/MCP/test_local_server_tools.py` case with one externally
eligible descriptor and one Console-only descriptor whose permission is
persisted Allow. Assert the latter never appears in `_local_agent_tool_registrations`.

Update `_local_agent_tool_registrations()` to iterate only specs/descriptors
marked `CONSOLE_AND_EXTERNAL_MCP`. Do not introduce a second list of excluded
names. Preserve the lazy read-only Watchlists DB resolver for exposed reads.

Run:

```bash
pytest -q Tests/MCP/test_local_server_tools.py Tests/MCP/test_gateway_runtime_tools.py -k "local or watchlists"
```

### Step 4: Carry code-owned effects onto approval rows

Extend `MCPPendingCall` with a backward-compatible empty `effects` tuple, then
populate it from `LocalToolProvider.pending_gate_for()`. Add approval-card tests
that render plain-language labels for private read, mutation, network, and LLM
spend while preserving redacted arguments, per-call decisions, and the existing
reason/options behavior.

The card must never derive effects from raw arguments. Existing MCP/builtin
callers may leave `effects=()`.

Run:

```bash
pytest -q Tests/Chat/test_console_local_review_hook.py Tests/UI/test_chat_approval_card.py -k "approval or effect or local"
```

### Step 5: Record the decision and documentation

Add the approved addendum to ADR-032 with TASK-22859/TASK-22860/TASK-22861
links. Update the Console tool guide to distinguish catalog exposure,
authorization, risk tags, and approval effects.

Run:

```bash
pytest -q Tests/MCP/test_mcp_documentation_contract.py -k "local or watchlists"
ruff check tldw_chatbook/Agents/local_tool_provider.py tldw_chatbook/Agents/mcp_tool_provider.py tldw_chatbook/MCP/local_server_tools.py tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py tldw_chatbook/Chat/console_chat_controller.py
git diff --check
```

Commit boundary:

```bash
git add backlog/decisions/032-local-agent-tool-permission-boundary.md backlog/tasks/task-22859\ -\ Define-Watchlists-Console-tool-exposure-and-approval-effects.md Docs/User_Guide/console/agent-runs-and-tools.md tldw_chatbook/Agents/local_tool_provider.py tldw_chatbook/Agents/mcp_tool_provider.py tldw_chatbook/MCP/local_server_tools.py tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py tldw_chatbook/Chat/console_chat_controller.py Tests/Agents/test_local_tool_provider.py Tests/MCP/test_local_server_tools.py Tests/Chat/test_console_local_review_hook.py Tests/UI/test_chat_approval_card.py
git commit -m "feat: define Watchlists tool exposure boundary"
```

## Task 2 — TASK-22860: Migrate durable provenance and atomic claims

### Files

- Create: `Tests/DB/test_subscriptions_db_briefing_provenance_migration.py`
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py`
- Modify: `tldw_chatbook/Subscriptions/briefing_service.py`
- Modify: `tldw_chatbook/Subscriptions/local_watchlists_service.py`
- Modify: `Tests/Subscriptions/test_briefing_service.py`
- Modify: `Tests/DB/test_subscriptions_db_agent_read_only.py`

### Step 1: Write the migration RED tests

Build a real temporary v1 database from an explicit historical v1 schema
fixture; do not stamp a partial/current schema with version 1. Assert the
fixture has the old `briefing_items` columns and lacks both partial unique
indexes before seeding:

- a completed briefing and legacy `briefing_items` rows;
- source/item rows that can later be edited/deleted;
- duplicate queued/running runs for one source;
- duplicate generating briefings for one collection.

Assert a normal open migrates to v2 atomically; a failure injected between
table rebuild and version update rolls back; reopening is idempotent; and a
read-only old-schema open returns feature unavailable rather than migrating.

Run:

```bash
pytest -q Tests/DB/test_subscriptions_db_briefing_provenance_migration.py
```

Expected RED: no v2 schema or constraints exist.

### Step 2: Implement the focused v1→v2 migration

Follow the Subscriptions convention recorded in
`tldw_chatbook/DB/migrations/README.md`: introduce module-level
`_CURRENT_SCHEMA_VERSION = 2`, `SUBSCRIPTIONS_V1_TO_V2_SQL`, and
`_migrate_from_v1_to_v2()` in `Subscriptions_DB.py` rather than adding a split
`.sql` asset. Refactor the version bootstrap so `schema_version` contains
exactly one row: a fresh database is created directly at v2; a historical v1
database is migrated and updates that row to 2; reopening v2 cannot reinsert a
v1 row.

The migration runs under one explicit `BEGIN IMMEDIATE` and rebuilds
`briefing_items` so:

- `item_id` remains the immutable original numeric identity;
- `live_item_id` is nullable and `ON DELETE SET NULL`;
- selection/citation positions are nullable for legacy rows;
- featured/cited flags and item/source snapshot fields are stored;
- `provenance_version=1` means legacy best effort and `2` means ordered snapshot.

Copy legacy rows through a Python owner loop inside that same transaction so
the existing URL sanitizer can strip userinfo, query, and fragment before each
snapshot insert; the SQL constant never copies auth/header/raw-body fields.
Migration code receives no caller-supplied data. Reconciliation keeps the
newest active receipt by `(created_at, id)`, changes older source runs to
`failed` with `INTERRUPTED_RUN_ERROR`, and changes older generating briefings
to `failed` with `INTERRUPTED_ERROR` before creating:

```sql
CREATE UNIQUE INDEX uq_local_watchlist_runs_active_source
ON local_watchlist_runs(source_id)
WHERE status IN ('queued', 'running');
CREATE UNIQUE INDEX uq_briefings_generating_watchlist
ON briefings(watchlist_id)
WHERE status = 'generating';
```

Run the Step 1 tests to GREEN.

### Step 3: Make briefing publication one transaction

Add a failing test that injects failure after provenance insertion but before
the briefing status update and asserts neither partial provenance nor a false
`complete` state survives.

Move `_write_junction` plus success publication behind this DB-owner method:

```python
def complete_briefing(
    self,
    briefing_id: int,
    *,
    body_markdown: str,
    model_used: str,
    covers_through_item_id: int | None,
    covers_from_ts: str | None,
    selection_mode: str,
    preset_id: int | None,
    overflow_count: int,
    provenance: Sequence[BriefingProvenanceRow],
) -> dict[str, Any]:
    """Atomically snapshot provenance and publish one completed briefing."""
```

Parse citation IDs before that call, preserve first-seen citation order, write
selection order, and publish `status='complete'` last inside the same
transaction.

Run:

```bash
pytest -q Tests/Subscriptions/test_briefing_service.py -k "junction or citation or transaction or provenance"
```

### Step 4: Add database-owned accept/transition primitives

Write concurrency tests using two `SubscriptionsDB` instances against one
temporary file. Add owner methods that insert an active source-run or briefing
receipt under the partial unique index and, on `IntegrityError`, read and
return the winning active row. Add guarded terminal transitions that release
the partial-index claim.

Do not replace durable uniqueness with the existing in-process sets; those
sets may remain execution optimizations only.

Run:

```bash
pytest -q Tests/DB/test_subscriptions_db_briefing_provenance_migration.py Tests/Subscriptions/test_briefing_service.py -k "claim or concurrent or provenance"
```

### Step 5: Run the complete Subscriptions migration surface

Because this is a schema bump, run every Subscriptions DB migration/readiness
test, not only the new file:

```bash
pytest -q Tests/DB/test_subscriptions_db.py Tests/DB/test_subscriptions_db_watchlists.py Tests/DB/test_subscriptions_db_agent_read_only.py Tests/DB/test_subscriptions_db_briefing_provenance_migration.py
pytest -q Tests/Subscriptions/test_briefing_selection.py Tests/Subscriptions/test_briefing_service.py
ruff check tldw_chatbook/DB/Subscriptions_DB.py tldw_chatbook/Subscriptions/briefing_service.py tldw_chatbook/Subscriptions/local_watchlists_service.py Tests/DB/test_subscriptions_db_briefing_provenance_migration.py
git diff --check
```

Use only `tmp_path`/in-memory databases. Do not launch the app against the
user's shared profile while the schema bump is isolated on this branch.

Commit boundary:

```bash
git add tldw_chatbook/DB/Subscriptions_DB.py tldw_chatbook/Subscriptions/briefing_service.py tldw_chatbook/Subscriptions/local_watchlists_service.py Tests/DB/test_subscriptions_db_briefing_provenance_migration.py Tests/DB/test_subscriptions_db_agent_read_only.py Tests/Subscriptions/test_briefing_service.py backlog/tasks/task-22860\ -\ Migrate-durable-briefing-provenance-and-atomic-Watchlists-claims.md
git commit -m "feat: migrate durable briefing provenance"
```

## Task 3 — TASK-22861: Expose bounded receipt and briefing queries

### Files

- Modify: `tldw_chatbook/Tools/watchlists_tool_service.py`
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py`
- Modify: `tldw_chatbook/Agents/local_tool_provider.py`
- Modify: `tldw_chatbook/MCP/local_server_tools.py`
- Modify: `Tests/Tools/test_watchlists_tool_service.py`
- Modify: `Tests/DB/test_subscriptions_db_watchlists_agent_search.py`
- Modify: `Tests/Agents/test_local_tool_provider.py`
- Modify: `Tests/MCP/test_local_server_tools.py`
- Modify: `Tests/MCP/test_gateway_runtime_tools.py`

### Step 1: Pin DB query ordering and stable cursors

Add failing DB tests for source, collection, briefing-receipt, operation
overview, exact operation receipt, and briefing-provenance queries. Pin each
ordering/cursor from the approved spec, including newest completed briefing
selection when a newer failed/generating row exists.

Implement narrow, parameterized DB readers. Keep external read-only readiness
checks explicit for every required v2 column/index.

Run:

```bash
pytest -q Tests/DB/test_subscriptions_db_watchlists_agent_search.py -k "source or collection or briefing or operation"
```

### Step 2: Add bounded service methods

In `WatchlistsToolService`, add:

```python
list_sources(arguments) -> str
list_collections(arguments) -> str
list_briefings(arguments) -> str
get_briefing(arguments) -> str
get_operations_status(arguments) -> str
get_operation_status(arguments) -> str
```

Reuse the existing exact-argument, canonical-ID, URL sanitation, structured
outcome, and 30 KiB finalization helpers. Reserve a fixed body budget in
`get_briefing`; pack bounded provenance separately so metadata cannot consume
all readable Markdown. Always return valid JSON after Unicode-safe truncation.

Run:

```bash
pytest -q Tests/Tools/test_watchlists_tool_service.py -k "list_sources or list_collections or briefing or operation"
```

### Step 3: Register and partition the descriptors

Add exact JSON schemas in `_default_specs`:

- shared: `watchlists_list_sources`, `watchlists_list_collections`,
  `watchlists_list_briefings`, `watchlists_get_operations_status`,
  `watchlists_get_operation_status`;
- Console-only: `watchlists_search_items`, `watchlists_get_item`, and
  `watchlists_get_briefing`.

Give all private reads the `PRIVATE_READ` approval effect. Verify external MCP
publishes receipt/metadata descriptors only and never resolves article-search,
article-body, or full-briefing handlers, even with persisted Allow.

Run:

```bash
pytest -q Tests/Agents/test_local_tool_provider.py Tests/MCP/test_local_server_tools.py Tests/MCP/test_gateway_runtime_tools.py -k "watchlists"
```

### Step 4: Verify the complete query boundary

Run:

```bash
pytest -q Tests/Tools/test_watchlists_tool_service.py Tests/DB/test_subscriptions_db_watchlists_agent_search.py Tests/DB/test_subscriptions_db_agent_read_only.py
pytest -q Tests/Agents/test_local_tool_provider.py Tests/MCP/test_local_server_tools.py Tests/MCP/test_gateway_runtime_tools.py -k "watchlists or local_tool"
ruff check tldw_chatbook/Tools/watchlists_tool_service.py tldw_chatbook/DB/Subscriptions_DB.py tldw_chatbook/Agents/local_tool_provider.py tldw_chatbook/MCP/local_server_tools.py
git diff --check
```

Commit boundary:

```bash
git add tldw_chatbook/Tools/watchlists_tool_service.py tldw_chatbook/DB/Subscriptions_DB.py tldw_chatbook/Agents/local_tool_provider.py tldw_chatbook/MCP/local_server_tools.py Tests/Tools/test_watchlists_tool_service.py Tests/DB/test_subscriptions_db_watchlists_agent_search.py Tests/Agents/test_local_tool_provider.py Tests/MCP/test_local_server_tools.py Tests/MCP/test_gateway_runtime_tools.py backlog/tasks/task-22861\ -\ Expose-bounded-Watchlists-receipts-and-briefing-query-tools.md
git commit -m "feat: expose bounded Watchlists briefing queries"
```

## Plan-level self-review gate

- Confirm no external registration code names Console-only tools.
- Confirm no approval effect changes permission resolution semantics.
- Confirm no migration or verification touches the live user database.
- Confirm every SQL identifier is static and every value is parameterized.
- Confirm completed briefing reads remain possible after live source/item deletion.
- Confirm all tool results stay below the internal 30 KiB boundary and contain
  no auth config, headers, queries/fragments, raw exception text, or DB paths.
