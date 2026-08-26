# Watchlists Agent Search Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to execute this plan task-by-task with review checkpoints.

**Goal:** Add permission-gated, read-only Watchlists search and item-detail tools to the Console agent and approved external MCP clients, returning bounded dated source-linked local evidence and an explicit unsupported outcome in server mode.

**Architecture:** Extend the existing synchronous `SubscriptionsDB` read path with narrow search/detail/resolution seams, then put all argument validation, cursor handling, output allowlisting, URL sanitization, untrusted-evidence labeling, and byte packing in one `WatchlistsToolService`. Register two optional-dependency `LocalToolSpec`s in the existing `LocalToolProvider`; Console injects its long-lived database owner, while external MCP lazily opens a separately registered read-only view only after the runtime-source check.

**Tech Stack:** Python 3.11+, SQLite/FTS5, Textual Console composition, existing local-tool/MCP gateway, pytest, Ruff.

**References:**

- Design: [`Docs/superpowers/specs/2026-08-14-watchlists-agent-search-tools-design.md`](../specs/2026-08-14-watchlists-agent-search-tools-design.md)
- Backlog: [`backlog/tasks/task-16222 - Expose local Watchlists search tools to Console and MCP.md`](../../../backlog/tasks/task-16222%20-%20Expose-local-Watchlists-search-tools-to-Console-and-MCP.md)
- Domain-tool precedent: [`backlog/decisions/030-local-library-agent-tool-boundary.md`](../../../backlog/decisions/030-local-library-agent-tool-boundary.md)
- Permission/data boundary: [`backlog/decisions/032-local-agent-tool-permission-boundary.md`](../../../backlog/decisions/032-local-agent-tool-permission-boundary.md)

ADR required: yes

ADR path: `backlog/decisions/032-local-agent-tool-permission-boundary.md`

Reason: Watchlists data expands the existing synthetic local-tool principal into private feed/article evidence and external MCP exposure. The approved ADR-032 addendum records that boundary, the shared-principal trade-off, and the permission/help-copy requirements; no second ADR is needed.

---

## Task 1: Add a non-mutating SubscriptionsDB construction path

**Files:**

- Modify: `tldw_chatbook/DB/base_db.py`
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py`
- Modify: `tldw_chatbook/DB/private_sqlite.py`
- Modify: `Tests/DB/test_private_sqlite_inventory.py`
- Create: `Tests/DB/test_subscriptions_db_agent_read_only.py`

- [ ] **Step 1: Write failing construction and owner-policy tests**

  Add tests proving that a keyword-only read-only `SubscriptionsDB` view:

  - rejects `:memory:` and missing files;
  - skips `_initialize_schema` and every migration/write probe;
  - uses `connect_private_sqlite(..., read_only=True, must_exist=True)` under a dedicated `db.subscriptions.agent_read` owner;
  - sets `row_factory` and safe connection-local behavior but never executes `journal_mode=WAL` or another write-oriented PRAGMA;
  - cannot insert, update, create a table, or change the database file/schema/row counts;
  - remains closeable after a failed readiness probe;
  - leaves every existing constructor call on the current initialize-on behavior.

- [ ] **Step 2: Run the tests to verify RED**

  Run:

  ```bash
  ../../.venv/bin/python -m pytest -q \
    Tests/DB/test_subscriptions_db_agent_read_only.py \
    Tests/DB/test_private_sqlite_inventory.py
  ```

  Expected: failures for the missing keyword-only skip-init/read-only seams and unregistered SQLite owner.

- [ ] **Step 3: Implement the minimal read-only seam**

  Add a keyword-only `initialize_schema: bool = True` (or equivalently named positive default) to `BaseDB.__init__`, and a keyword-only `read_only: bool = False` to `SubscriptionsDB.__init__`. The read-only branch must skip schema initialization and startup integrity writes, use the dedicated private-SQLite owner with `read_only=True, must_exist=True`, omit WAL/synchronous write tuning, and keep normal app/test behavior unchanged. Register the owner policy as read-only URI only with source-file mode preservation.

  This is a logical database read-only guarantee, not a byte-stable-sidecar guarantee. A normal SQLite `mode=ro` reader must participate in WAL coordination and may create or update SQLite-managed `-wal`/`-shm` sidecars while reading the current database. Do not use `immutable=1` to suppress that coordination: an immutable view can ignore uncheckpointed WAL and return stale or unavailable Watchlists data. A zero-sidecar-mutation guarantee would require a separately designed snapshot boundary and is outside this task.

- [ ] **Step 4: Add the exact readiness probe**

  On the read-only path, verify only the tables/columns required by the two tools. Raise a fixed internal availability exception without embedding SQL, paths, or stored values. Ensure `close()` can close the current thread-local connection after either success or failure.

- [ ] **Step 5: Run focused GREEN and compatibility tests**

  Run:

  ```bash
  ../../.venv/bin/python -m pytest -q \
    Tests/DB/test_subscriptions_db_agent_read_only.py \
    Tests/DB/test_private_sqlite.py \
    Tests/DB/test_private_sqlite_inventory.py \
    Tests/DB/test_subscriptions_db_watchlists.py
  ```

  Expected: all pass; existing mutable/in-memory construction remains unchanged.

- [ ] **Step 6: Commit**

  ```bash
  git add tldw_chatbook/DB/base_db.py tldw_chatbook/DB/Subscriptions_DB.py \
    tldw_chatbook/DB/private_sqlite.py Tests/DB/test_private_sqlite_inventory.py \
    Tests/DB/test_subscriptions_db_agent_read_only.py
  git commit -m "feat(watchlists): add read-only agent database view"
  ```

## Task 2: Add authoritative Watchlists evidence-query seams

**Files:**

- Modify: `tldw_chatbook/DB/Subscriptions_DB.py`
- Create: `Tests/DB/test_subscriptions_db_watchlists_agent_search.py`

- [ ] **Step 1: Write failing database-query tests**

  Cover these outcomes against a real temporary SQLite database:

  - blank search returns all statuses newest-first;
  - literal AND terms match title, author, and deep body text;
  - hostile FTS operators remain literals;
  - absent FTS uses escaped LIKE with `%`, `_`, and `\\` treated literally;
  - an operational failure during an otherwise available FTS query falls back to the same literal LIKE result in that call;
  - an item-ID anti-join against `subscription_items_fts_docsize` detects partial coverage and equal-cardinality/wrong-membership coverage;
  - incomplete coverage uses LIKE, rechecks later, and switches to FTS only after the same owner becomes complete;
  - source/collection/status/since/snapshot-high-water/keyset filters compose;
  - equal effective dates, null-date sink, deletion between pages, pre-existing future dates, and later inserts produce stable keyset traversal;
  - one lookahead drives `has_more` without being returned;
  - exact/partial scope resolvers are bounded and deterministic;
  - item detail distinguishes missing rows from present rows with null content;
  - source-to-collection enrichment is one bounded query, not N+1.

- [ ] **Step 2: Run the new module to verify RED**

  Run:

  ```bash
  ../../.venv/bin/python -m pytest -q Tests/DB/test_subscriptions_db_watchlists_agent_search.py
  ```

  Expected: failures for missing query, resolution, detail, membership, and FTS-coverage methods.

- [ ] **Step 3: Implement additive synchronous read methods**

  Extend the existing `_search_items_rows`/`get_new_items` machinery instead of creating parallel SQL. Add the minimum projections and predicates for `effective_date`, `snapshot_max_item_id`, nullable-date-aware keyset continuation, one-row lookahead, match-context inputs, joined detail, bounded source/collection candidates, and batched memberships. Preserve the existing same-call `sqlite3.OperationalError` fallback from FTS to literal LIKE. Keep all SQL parameters bound.

- [ ] **Step 4: Implement conservative FTS completeness state**

  Cache only the monotonic complete state. Until an ID anti-join proves every current item exists in FTS docsize, retry the probe on later searches and force the existing LIKE semantics. Do not infer completeness from table existence or count equality.

- [ ] **Step 5: Run focused and legacy GREEN tests**

  Run:

  ```bash
  ../../.venv/bin/python -m pytest -q \
    Tests/DB/test_subscriptions_db_watchlists_agent_search.py \
    Tests/DB/test_subscriptions_db_watchlists.py
  ```

  Expected: all pass, including legacy Read-tab search behavior.

- [ ] **Step 6: Commit**

  ```bash
  git add tldw_chatbook/DB/Subscriptions_DB.py \
    Tests/DB/test_subscriptions_db_watchlists_agent_search.py
  git commit -m "feat(watchlists): add bounded agent evidence queries"
  ```

## Task 3: Build the shared Watchlists tool service and contracts

**Files:**

- Create: `tldw_chatbook/Tools/watchlists_tool_service.py`
- Create: `Tests/Tools/test_watchlists_tool_service.py`

- [ ] **Step 1: Write failing validation and scope tests**

  Pin the public contract before implementation:

  - search accepts only `query`, `collection`, `source`, `statuses`, `since`, `limit`, and `cursor`;
  - query is at most 512 characters/32 whitespace terms; blank means browse;
  - source/collection accept bare JSON positive integers or canonical local IDs, reject bools/foreign IDs, and treat numeric strings as names;
  - statuses are unique, non-empty when supplied, and from the five-value allowlist;
  - `since` accepts `YYYY-MM-DD` or RFC 3339 and normalizes to UTC;
  - limit defaults to 10 and accepts only integer 1..50;
  - detail accepts only canonical `local:watchlist_item:<positive integer>`;
  - collection text is bounded to 256 characters and source text to 2,048 characters in core validation, independent of JSON Schema;
  - surrounding whitespace is stripped before exact case-insensitive name, exact raw configured URL, or unique-partial resolution;
  - exact case-insensitive name, exact raw configured URL, unique partial, ambiguous candidates, and missing scopes return the approved structured outcomes;
  - every disambiguation candidate canonical ID round-trips through the same source/collection parameter;
  - collection+source is an intersection, never a widening;
  - `since` is inclusive at the normalized boundary;
  - no matches is `status: "ok"` with `items: []`, while invalid/missing outcomes set `retryable: false`;
  - permanent missing/pre-migration dependencies are non-retryable with operator guidance, while a failed lazy candidate that may succeed on a later call is marked retryable;
  - validation happens before database item queries.

  Also pin the exact successful search/detail envelopes: `query_mode`, `ordering`, traversal `as_of`, exposed `snapshot_max_item_id`, `returned_count`, `has_more`, `next_cursor`, resolved `scope.collection`/`scope.source` canonical IDs and names, source type/link metadata, per-item collection memberships, distinct item date fields, selected-source `created_at`/`updated_at`/`last_checked`/`last_successful_check`, source active/paused state, and search/detail metadata parity. Assert there is no invented aggregate `last_updated` field.

- [ ] **Step 2: Run the new tests to verify RED**

  Run:

  ```bash
  ../../.venv/bin/python -m pytest -q Tests/Tools/test_watchlists_tool_service.py
  ```

  Expected: import/missing-service failures.

- [ ] **Step 3: Implement the smallest synchronous service**

  Create `WatchlistsToolService` with injected database resolver and runtime-source loader. Keep validation, canonical-ID parsing, scope resolution, and orchestration in this module; keep persistence SQL in `SubscriptionsDB`. Expected domain outcomes must serialize as successful JSON objects with `status` in `ok`, `invalid_argument`, `not_found`, `needs_disambiguation`, `unsupported`, or `feature_unavailable`.

- [ ] **Step 4: Enforce dependency order and server behavior**

  For each handler: validate arguments, read current runtime source, return the exact non-retryable message `server Watchlists search is not supported; switch Watchlists to Local before retrying`, then—and only then—resolve the local database. Add spies proving server mode never touches the database resolver and malformed/absent runtime-policy state uses the existing local default.

- [ ] **Step 5: Run focused GREEN tests**

  Run:

  ```bash
  ../../.venv/bin/python -m pytest -q \
    Tests/Tools/test_watchlists_tool_service.py \
    Tests/DB/test_subscriptions_db_watchlists_agent_search.py
  ```

  Expected: all pass.

- [ ] **Step 6: Commit**

  ```bash
  git add tldw_chatbook/Tools/watchlists_tool_service.py \
    Tests/Tools/test_watchlists_tool_service.py
  git commit -m "feat(watchlists): add shared agent search service"
  ```

## Task 4: Harden cursor, evidence, URL, and byte-boundary output

**Files:**

- Modify: `tldw_chatbook/Tools/watchlists_tool_service.py`
- Modify: `Tests/Tools/test_watchlists_tool_service.py`

- [ ] **Step 1: Add failing cursor and stable-traversal tests**

  Test a versioned URL-safe cursor containing only traversal `as_of`, `snapshot_max_item_id`, nullable last effective date, last item ID, and SHA-256 filter fingerprint. Assert round-trip; equivalent status-order, collapsed-query-whitespace, resolved-numeric-scope, and UTC-date-floor normalization; inclusion of the ordering contract in the fingerprint; malformed/unknown-version/key rejection; filter mismatch before item query; later pages preserving the original `as_of` and snapshot boundary; and decoded payload absence of raw query, names, URLs, body text, or paths.

- [ ] **Step 2: Add failing output-safety tests**

  Test that:

  - all result/detail JSON is strict (`allow_nan=False`) and below 30 KiB;
  - packing stops on a complete item and returns a cursor rather than slicing JSON;
  - oversized Unicode fields and detail bodies truncate on character boundaries with explicit flags;
  - match excerpts center on title/author/body terms and blank browse uses a leading-body preview;
  - detail uses `readable_body_text`, labels normalization/untrusted content, and preserves content format/kind;
  - change-only evidence remains named change evidence;
  - prompt-injection/control-shaped content remains JSON-escaped and labeled untrusted;
  - non-finite numbers become `null` plus an invalid marker;
  - only the contract allowlist appears—never auth config, custom headers, rate limits, extracted data, raw processing/source errors, DB paths, or raw exceptions;
  - every emitted URL is absolute HTTP(S) with host, strips userinfo/query/fragment, preserves its path, and sets `url_redacted`; malformed, hostless, `file:`, and `javascript:` URLs become null/redacted in items, sources, scope, and candidates.

  Capture logs on unexpected failures and assert raw configured URLs, stored canaries, database paths, SQL fragments, and exception messages appear in neither returned content nor logs.

- [ ] **Step 3: Run the selected tests to verify RED**

  Run:

  ```bash
  ../../.venv/bin/python -m pytest -q Tests/Tools/test_watchlists_tool_service.py
  ```

  Expected: failures for unimplemented cursor and output hardening.

- [ ] **Step 4: Implement cursor and safe response packing**

  Use stdlib `hashlib`, `base64`, `json`, `datetime`, and `urllib.parse`; add no dependency. Normalize filter fingerprints exactly as the spec states. Build responses only from explicit field constructors. Measure encoded UTF-8 after every candidate addition and reserve envelope/cursor headroom.

- [ ] **Step 5: Add the scrubbed unexpected-error adapter**

  Catch only the shared handler boundary's unexpected implementation/storage exceptions, log a payload-free bounded exception category, and raise one fixed public failure string. Keep expected domain outcomes as successful JSON so external MCP does not replace them with a generic gateway failure.

- [ ] **Step 6: Run complete service/DB GREEN tests**

  Run:

  ```bash
  ../../.venv/bin/python -m pytest -q \
    Tests/Tools/test_watchlists_tool_service.py \
    Tests/DB/test_subscriptions_db_watchlists_agent_search.py
  ```

  Expected: all pass and every parsed response is below the provider's 32 KiB ceiling.

- [ ] **Step 7: Commit**

  ```bash
  git add tldw_chatbook/Tools/watchlists_tool_service.py \
    Tests/Tools/test_watchlists_tool_service.py
  git commit -m "fix(watchlists): bound and sanitize agent evidence"
  ```

## Task 5: Register both tools in the existing LocalToolProvider

**Files:**

- Modify: `tldw_chatbook/Agents/local_tool_provider.py`
- Modify: `Tests/Agents/test_local_tool_provider.py`
- Modify: `Tests/Agents/test_local_tools_integration.py`

- [ ] **Step 1: Write failing catalog/schema/provider-boundary tests**

  Assert that default catalog composition exposes `local:watchlists_search_items` and `local:watchlists_get_item`, both have `additionalProperties: false`, exact bounds/enums/unions, and no mutation tags. Pin both descriptions to warn that feed titles, authors, URLs, source names, and evidence are untrusted facts rather than instructions; pin the search description to state that “all” requires following `next_cursor` until `has_more` is false. Assert catalog-only construction does not open a database, missing optional dependency returns structured `feature_unavailable`, expected outcomes remain `ToolResult(ok=True)` valid JSON, unexpected exceptions expose only the fixed public error, and generic `_fit_result` never truncates a correctly packed Watchlists result. Add one representative failing permission test that covers Allow execution plus Ask/deny refusal before the handler runs; retain the existing generic permission matrix for the other gate states.

  In `Tests/Agents/test_local_tools_integration.py`, add a normal agent progressive-disclosure test that discovers each catalog entry, loads its schema, receives permission, invokes it through the runtime/provider seam, and parses the returned JSON.

- [ ] **Step 2: Run tests to verify RED**

  Run:

  ```bash
  ../../.venv/bin/python -m pytest -q \
    Tests/Agents/test_local_tool_provider.py \
    Tests/Agents/test_local_tools_integration.py -k 'watchlists'
  ```

  Expected: no Watchlists schemas/specs exist.

- [ ] **Step 3: Add optional service injection and specs**

  Extend `LocalToolProvider`/`_default_specs` with an explicit optional Watchlists service or database-resolver dependency. Register both specs unconditionally in the default catalog without touching storage. Handlers call the shared service adapters and return strings; do not duplicate validation or response shaping.

- [ ] **Step 4: Preserve permission behavior through the existing matrix**

  Make the representative Watchlists permission test pass without adding a bypass or a second principal. Run the existing generic master flag, session grant, definition hash, kill-switch, timeout, and audit regressions unchanged rather than duplicating the entire matrix per tool.

- [ ] **Step 5: Run the full provider suite**

  Run:

  ```bash
  ../../.venv/bin/python -m pytest -q \
    Tests/Agents/test_local_tool_provider.py \
    Tests/Agents/test_local_tools_integration.py
  ```

  Expected: all pass.

- [ ] **Step 6: Commit**

  ```bash
  git add tldw_chatbook/Agents/local_tool_provider.py \
    Tests/Agents/test_local_tool_provider.py \
    Tests/Agents/test_local_tools_integration.py
  git commit -m "feat(watchlists): register local agent tools"
  ```

## Task 6: Wire Console and external MCP ownership correctly

**Files:**

- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/MCP/local_server_tools.py`
- Modify: `Tests/Chat/test_console_local_review_hook.py`
- Modify: `Tests/MCP/test_local_server_tools.py`
- Modify: `Tests/MCP/test_gateway_runtime_tools.py`

- [ ] **Step 1: Write failing Console composition tests**

  Assert `_compose_local_provider` injects exactly `app.subscriptions_db`, supplies a fresh runtime-source loader, exposes both tool IDs only when `[console] local_tools_enabled` and existing permission gates allow composition, and does not construct a second database or run async code in the sync handler. Snapshot items, statuses, flags, sources, collections, schema version, and runtime-policy state before real search/detail calls through this mutable app owner and prove they are byte-for-byte/logically unchanged afterward.

- [ ] **Step 2: Write failing external-MCP lazy-owner tests**

  Assert `build_server_local_provider`:

  - remains hidden at the real server registration surface when `[mcp] expose_local_tools` is false, and publishes both exact schemas only when the flag is true;
  - registers both tools without opening the Watchlists database;
  - checks runtime source per call;
  - returns server unsupported without path resolution/opening;
  - lazily resolves the configured subscriptions path only for a local Watchlists call;
  - uses a lock plus double-check so concurrent first calls retain one successful read-only instance;
  - closes every failed candidate and permits a later retry;
  - maps missing/pre-migration storage to bounded `feature_unavailable` while unrelated fs/git/web tools remain registered;
  - preserves permission-denial behavior and external Ask failure before service execution.

- [ ] **Step 3: Add gateway integration tests**

  Register the real provider with `ChatbookGatewayRuntime`. Prove successful structured `invalid_argument`, `not_found`, `unsupported`, and `feature_unavailable` JSON passes through unchanged; permission failures retain the existing allowlisted public errors; unexpected failures are generic and payload-free; blocking database work stays off the event loop.

- [ ] **Step 4: Run tests to verify RED**

  Run:

  ```bash
  ../../.venv/bin/python -m pytest -q \
    Tests/Chat/test_console_local_review_hook.py \
    Tests/MCP/test_local_server_tools.py \
    Tests/MCP/test_gateway_runtime_tools.py -k 'watchlists or compose_local_provider'
  ```

  Expected: failures for absent injection/lazy resolver/integration behavior.

- [ ] **Step 5: Implement Console injection**

  Reuse the app's long-lived `subscriptions_db`. Supply a runtime loader backed by the active profile's `default_runtime_policy_path()` and `RuntimeSourceStateStore.load()`. Preserve all existing local-provider composition and review-hook code.

- [ ] **Step 6: Implement external MCP lazy resolution**

  Add a Watchlists-only resolver object/function in `MCP/local_server_tools.py`, using `get_subscriptions_db_path()`, the new read-only constructor, a `threading.Lock`, double-check caching, readiness validation, and close-on-failure. Do not create the DB during provider/catalog registration. Do not alter the gateway/server registration architecture.

- [ ] **Step 7: Run complete integration GREEN tests**

  Run:

  ```bash
  ../../.venv/bin/python -m pytest -q \
    Tests/Chat/test_console_local_review_hook.py \
    Tests/Agents/test_local_tool_provider.py \
    Tests/MCP/test_local_server_tools.py \
    Tests/MCP/test_gateway_runtime_tools.py
  ```

  Expected: all pass.

- [ ] **Step 8: Commit**

  ```bash
  git add tldw_chatbook/Chat/console_chat_controller.py \
    tldw_chatbook/MCP/local_server_tools.py \
    Tests/Chat/test_console_local_review_hook.py \
    Tests/MCP/test_local_server_tools.py Tests/MCP/test_gateway_runtime_tools.py
  git commit -m "feat(watchlists): expose search through Console and MCP"
  ```

## Task 7: Correct permission, settings, and operator documentation

**Files:**

- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_tools_mode.py`
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py`
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md`
- Modify: `Docs/User_Guide/mcp.md`
- Modify: `Docs/Examples/skills/README.md`
- Modify: `Tests/MCP/test_mcp_documentation_contract.py`

- [ ] **Step 1: Write failing copy/inventory contract tests**

  Update the documentation contract to require both exact tool names, every public parameter and bound, the literal/full-text and cursor semantics, the stable-against-later-inserts but not-snapshot-isolation limitation, the server unsupported behavior, external permission/egress warning, and group wording that explicitly includes Watchlists data. Require links to TASK-16222, ADR-030, and amended ADR-032. Reject the stale “Local workspace + web tools” label where it describes the expanded principal.

- [ ] **Step 2: Run documentation contract tests to verify RED**

  Run:

  ```bash
  ../../.venv/bin/python -m pytest -q Tests/MCP/test_mcp_documentation_contract.py
  ```

  Expected: failures on stale inventories and privacy copy.

- [ ] **Step 3: Update all user-facing surfaces**

  Rename the shared group/master-switch copy to include “workspace, web, and Watchlists” consistently. Document every search/detail parameter, that results are local, literal (not semantic), bounded, untrusted, date-explicit, and require cursor continuation for “all.” State continuation excludes later inserts but is not snapshot isolation. Explain that source/item URL paths are authorized Watchlists metadata, while userinfo/query/fragment are removed. State external MCP requires `[mcp] expose_local_tools`, per-tool permission Allow, and can send approved evidence to its client/model. Link TASK-16222, ADR-030, and amended ADR-032 from the developer-facing inventory/design notes.

- [ ] **Step 4: Run docs, provider inventory, and UI-copy tests**

  Run:

  ```bash
  ../../.venv/bin/python -m pytest -q \
    Tests/MCP/test_mcp_documentation_contract.py \
    Tests/Agents/test_local_tool_provider.py \
    Tests/MCP/test_local_server_tools.py
  ```

  Expected: all pass.

- [ ] **Step 5: Commit**

  ```bash
  git add tldw_chatbook/config.py \
    tldw_chatbook/UI/MCP_Modules/mcp_tools_mode.py \
    tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py \
    Docs/User_Guide/console/agent-runs-and-tools.md Docs/User_Guide/mcp.md \
    Docs/Examples/skills/README.md Tests/MCP/test_mcp_documentation_contract.py
  git commit -m "docs(watchlists): explain agent evidence permissions"
  ```

## Task 8: Verify the product path and close TASK-16222

**Files:**

- Create temporarily, then delete: `Tests/Watchlists/test__tmp_watchlists_agent_tools_live_qa.py`
- Modify: `backlog/tasks/task-16222 - Expose-local-Watchlists-search-tools-to-Console-and-MCP.md`
- Modify only if this task produced a genuinely reusable incident: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md`

- [ ] **Step 1: Run the focused automated suite from a clean process**

  Run:

  ```bash
  ../../.venv/bin/python -m pytest -q \
    Tests/DB/test_subscriptions_db_agent_read_only.py \
    Tests/DB/test_subscriptions_db_watchlists_agent_search.py \
    Tests/Tools/test_watchlists_tool_service.py \
    Tests/Agents/test_local_tool_provider.py \
    Tests/Chat/test_console_local_review_hook.py \
    Tests/MCP/test_local_server_tools.py \
    Tests/MCP/test_gateway_runtime_tools.py \
    Tests/MCP/test_mcp_documentation_contract.py
  ```

  Expected: all pass.

- [ ] **Step 2: Run the relevant regression suite**

  Run:

  ```bash
  ../../.venv/bin/python -m pytest -q \
    Tests/DB/test_subscriptions_db_watchlists.py \
    Tests/DB/test_private_sqlite.py \
    Tests/DB/test_private_sqlite_inventory.py \
    Tests/Agents/test_local_tools_integration.py
  ```

  Expected: all pass.

- [ ] **Step 3: Perform isolated Console/MCP live QA**

  Create a temporary pytest-backed live harness only after automated collection is stable. The parent test must launch a fresh subprocess whose command-level environment sets `TLDW_CONFIG_PATH`, `XDG_CONFIG_HOME`, and `XDG_DATA_HOME` beneath one scratch root **before the child imports any `tldw_chatbook` module**. Write the scratch TOML with an explicit subscriptions database path before launch. In the child, assert every resolved config/runtime-policy/data/subscriptions path is beneath the scratch root, then seed at least two partially colliding source/collection names for disambiguation plus dated items, deep-body matches, visibly distinct source/item timestamps, a hostile-looking evidence string, and a redacted URL. Exercise the real Console-composed provider and real external gateway registration under explicit Allow. Verify search, continuation, detail, scope disambiguation, current-source switching, server unsupported/no local open, and distinct timestamp meanings. Compare parsed Console and external-MCP responses for semantic equivalence after independently validating and then excluding traversal-specific `as_of`/`next_cursor` values; do not require sequential calls to produce identical clocks or cursor bytes. The parent must record existence plus hashes/metadata of the real profile's runtime-policy and subscriptions files before launch and prove they are unchanged afterward. Use the repository interpreter—never a bare system interpreter.

  Run:

  ```bash
  ../../.venv/bin/python -m pytest -q Tests/Watchlists/test__tmp_watchlists_agent_tools_live_qa.py -s
  ```

  Expected: all live probes pass with no writes outside the isolated profile.

- [ ] **Step 4: Remove the temporary live-QA file**

  Delete only `Tests/Watchlists/test__tmp_watchlists_agent_tools_live_qa.py` with `apply_patch`, then verify it is absent from `git status`.

- [ ] **Step 5: Run static and repository checks**

  Run:

  ```bash
  ../../.venv/bin/python -m ruff check \
    tldw_chatbook/DB/base_db.py \
    tldw_chatbook/DB/Subscriptions_DB.py \
    tldw_chatbook/DB/private_sqlite.py \
    tldw_chatbook/Tools/watchlists_tool_service.py \
    tldw_chatbook/Agents/local_tool_provider.py \
    tldw_chatbook/Chat/console_chat_controller.py \
    tldw_chatbook/MCP/local_server_tools.py \
    tldw_chatbook/config.py \
    tldw_chatbook/UI/MCP_Modules/mcp_tools_mode.py \
    tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py \
    Tests/DB/test_subscriptions_db_agent_read_only.py \
    Tests/DB/test_subscriptions_db_watchlists_agent_search.py \
    Tests/Tools/test_watchlists_tool_service.py \
    Tests/Agents/test_local_tool_provider.py \
    Tests/Agents/test_local_tools_integration.py \
    Tests/Chat/test_console_local_review_hook.py \
    Tests/MCP/test_local_server_tools.py \
    Tests/MCP/test_gateway_runtime_tools.py \
    Tests/MCP/test_mcp_documentation_contract.py
  ../../.venv/bin/python -m ruff format --check \
    tldw_chatbook/DB/base_db.py \
    tldw_chatbook/DB/Subscriptions_DB.py \
    tldw_chatbook/DB/private_sqlite.py \
    tldw_chatbook/Tools/watchlists_tool_service.py \
    tldw_chatbook/Agents/local_tool_provider.py \
    tldw_chatbook/Chat/console_chat_controller.py \
    tldw_chatbook/MCP/local_server_tools.py \
    tldw_chatbook/config.py \
    tldw_chatbook/UI/MCP_Modules/mcp_tools_mode.py \
    tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py \
    Tests/DB/test_subscriptions_db_agent_read_only.py \
    Tests/DB/test_subscriptions_db_watchlists_agent_search.py \
    Tests/Tools/test_watchlists_tool_service.py \
    Tests/Agents/test_local_tool_provider.py \
    Tests/Agents/test_local_tools_integration.py \
    Tests/Chat/test_console_local_review_hook.py \
    Tests/MCP/test_local_server_tools.py \
    Tests/MCP/test_gateway_runtime_tools.py \
    Tests/MCP/test_mcp_documentation_contract.py
  git diff --check
  ```

  Expected: Ruff and diff check pass.

- [ ] **Step 6: Run the full test suite**

  Run:

  ```bash
  ../../.venv/bin/python -m pytest -q
  ```

  Expected: all tests pass. If an unrelated environmental/baseline failure appears, reproduce it isolated and document exact evidence; do not weaken this feature's contracts to hide it.

- [ ] **Step 7: Complete Backlog and documentation hygiene, except status**

  Check all acceptance criteria, add concise implementation notes with test/live evidence and links to TASK-16222, ADR-030, and amended ADR-032, document any plan deviation, and add a lesson only if a concrete reusable incident occurred. Keep the task In Progress until the final review and closeout commit succeed.

- [ ] **Step 8: Final self-review and commit**

  Review the branch diff for privacy leakage, permission bypasses, mutation paths, N+1 queries, invalid JSON truncation, cursor drift, stale runtime-source capture, and unnecessary abstraction. Run a focused security/privacy review of the diff. If the review is clean and every DoD item is satisfied, set the task Done and commit the task closeout files:

  ```bash
  backlog task edit 16222 -s Done
  git add "backlog/tasks/task-16222 - Expose-local-Watchlists-search-tools-to-Console-and-MCP.md" \
    backlog/docs/lessons-testing-evidence.md backlog/docs/lessons-live-verification.md
  git commit -m "chore(watchlists): complete agent search task"
  ```

  Omit unchanged lesson files from `git add`.
