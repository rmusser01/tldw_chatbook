# Token-Budgeted Tool Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed 16/24 tool-count gates with deterministic discovery, provider-aware schema budgeting, and a replaceable catalog working set that preserves permission correctness.

**Architecture:** A pure catalog layer ranks metadata and probes whether the complete allowed schema set stays under the automatic-disclosure threshold. The service owns provider-visible token estimation and projected request-fit decisions; the runtime owns the single-threaded commit that replaces both the model-visible catalog set and permission-name set between dispatch batches.

**Tech Stack:** Python 3.11+, dataclasses, Textual application service layer, pytest.

**Spec:** `Docs/superpowers/specs/2026-08-30-token-budgeted-tool-discovery-design.md`

## Global Constraints

- Complete-catalog automatic disclosure requires schema share at or below 10% of the selected model context and a projected first request that fits after response reserve.
- `find_tools` returns at most eight allow-listed results with deterministic exact-name, prefix, name-substring, then description-substring ordering.
- Deferred loads are limited by projected next-request headroom, not by 10% and never by a cumulative tool count.
- `load_tools` replaces the catalog working set, must be the only call in its model-produced batch, and preserves the old set on every failed or empty selection.
- Permission authority never follows discovery automatically; it mirrors only the committed model-visible catalog set.
- No database migration, Settings control, LRU, persistence, new dependency, or historical-document rewrite.
- Use targeted tests throughout; do not run the full repository suite without renewed user approval.

---

### Task 1: Catalog policy primitives and deterministic search

**Files:**
- Modify: `tldw_chatbook/Agents/agent_models.py`
- Modify: `tldw_chatbook/Agents/tool_catalog.py`
- Test: `Tests/Agents/test_tool_catalog.py`
- Test: `Tests/Agents/test_agent_models.py`

**Interfaces:**
- Produces: `DIRECT_DISCLOSURE_CONTEXT_FRACTION: float = 0.10`
- Produces: `FIND_TOOLS_RESULT_LIMIT: int = 8`
- Produces: immutable `ToolLoadSelection(accepted, omitted_for_budget, invalid_inputs)`
- Produces: `ToolCatalogRegistry.find(query, *, allowed_names=None, limit=FIND_TOOLS_RESULT_LIMIT)`
- Produces: `probe_initial_catalog(registry, allowed_names, max_schema_tokens, measure_schema_set)` returning all allowed schemas or `None` to defer.

- [ ] **Step 1: Write failing catalog-policy tests**

Add literal fixtures proving that 25 compact schemas can be returned by the probe when the injected complete-set cost is 99 against a 100-token threshold, while 5 large schemas defer at cost 101. Add failure cases for a non-positive/raising cost callback and verify disallowed schemas are excluded before cost measurement.

```python
def test_probe_uses_schema_cost_not_catalog_count():
    direct = probe_initial_catalog(reg, allowed, 100, lambda schemas: 99)
    assert len(direct) == 25

def test_probe_defers_when_complete_set_exceeds_threshold():
    assert probe_initial_catalog(reg, allowed, 100, lambda schemas: 101) is None
```

- [ ] **Step 2: Write failing search-ranking tests**

Register entries in reverse relevance order and assert literal result names for exact-name, prefix, name substring, and description substring matches. Put disallowed exact matches ahead of allowed matches and prove filtering occurs before the eight-result slice.

```python
assert [entry.name for entry in reg.find("clock", allowed_names=allowed)] == [
    "clock", "clock_sync", "wall_clock", "timezone_lookup"
]
assert len(reg.find("tool", allowed_names=allowed)) == 8
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run: `../../.venv/bin/python -m pytest Tests/Agents/test_tool_catalog.py Tests/Agents/test_agent_models.py -q`

Expected: failures for missing constants, `ToolLoadSelection`, new `find` keywords, and `probe_initial_catalog`; existing count assertions also identify the obsolete contract to rewrite.

- [ ] **Step 4: Implement the minimal pure primitives**

Remove `DIRECT_DISCLOSE_THRESHOLD` and `RunBudget.max_active_tools`, including child-budget propagation. Add the frozen selection dataclass and pure ranking/probe behavior. Normalize comparisons with `casefold()` and sort ties by `(normalized_name, id)` after relevance rank. Never slice before applying `allowed_names`.

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run: `../../.venv/bin/python -m pytest Tests/Agents/test_tool_catalog.py Tests/Agents/test_agent_models.py -q`

- [ ] **Step 6: Commit the catalog policy slice**

```bash
git add tldw_chatbook/Agents/agent_models.py tldw_chatbook/Agents/tool_catalog.py Tests/Agents/test_tool_catalog.py Tests/Agents/test_agent_models.py
git commit -m "refactor(agents): replace tool counts with catalog policy primitives"
```

### Task 2: Provider-visible schema measurement and first-request planning

**Files:**
- Modify: `tldw_chatbook/Agents/agent_service.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Test: `Tests/Agents/test_agent_service.py`
- Test: `Tests/Chat/test_console_agent_bridge.py`
- Test: `Tests/UI/test_console_project_instructions.py`

**Interfaces:**
- Consumes: `probe_initial_catalog(...)` from Task 1.
- Produces: `catalog_schema_tokens(schemas, *, model, api_endpoint, native_tools) -> int` measuring one complete native JSON array or fence protocol string.
- Produces: a shared first-request planner that receives allowed names, resolved model/provider, messages, response reserve, runtime feature gates, and direct/discovery prompt variants.
- Produces: a `FirstRequestSchemaPlan` reused by Console preview and passed into live primary execution so the first payload cannot be recomputed differently.

- [ ] **Step 1: Write failing set-level measurement tests**

Monkeypatch only `estimate_tokens` to capture its real input. For native mode assert it receives one compact JSON rendering of `schemas_to_openai_tools(all_schemas)`; for fence mode assert it receives one `render_tool_protocol(all_schemas)` string. Include two schemas so per-schema summation cannot satisfy the assertion.

- [ ] **Step 2: Write failing first-request decision tests**

Use real `AgentConfig`, registry, and request building with injected model limit/token estimates. Prove: 25 compact schemas direct-disclose; 5 large schemas defer; a catalog at 9% still defers when messages plus response reserve make the direct request exceed context; estimator/model-limit failures defer.

- [ ] **Step 3: Run focused tests and verify RED**

Run: `../../.venv/bin/python -m pytest Tests/Agents/test_agent_service.py Tests/Chat/test_console_agent_bridge.py Tests/UI/test_console_project_instructions.py -q`

Expected: failures because first-request planning still accepts `RunBudget`, uses counts, and lacks provider/message inputs.

- [ ] **Step 4: Implement shared provider-aware planning**

Use the existing `schemas_to_openai_tools`, `render_tool_protocol`, `estimate_tokens`, `get_model_token_limit`, `_count_model_messages`, and response reserve. Build direct and discovery candidates with the same runtime gates used live. Return discovery on every exception, invalid/non-positive estimate, schema-load failure, threshold miss, or projected request-fit miss.

- [ ] **Step 5: Wire Console preflight and live primary reuse**

Make `_build_console_first_request_plan` produce the final prompt/config and schema plan together. Pass that immutable plan into `AgentService.run_turn` for the primary run; keep service-side planning for generic callers and child runs. Preserve the existing disposable project-instruction preview behavior.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run: `../../.venv/bin/python -m pytest Tests/Agents/test_agent_service.py Tests/Chat/test_console_agent_bridge.py Tests/UI/test_console_project_instructions.py -q`

- [ ] **Step 7: Commit first-request budgeting**

```bash
git add tldw_chatbook/Agents/agent_service.py tldw_chatbook/Chat/console_agent_bridge.py Tests/Agents/test_agent_service.py Tests/Chat/test_console_agent_bridge.py Tests/UI/test_console_project_instructions.py
git commit -m "feat(agents): budget initial tools by provider request cost"
```

### Task 3: Exclusive runtime working-set replacement

**Files:**
- Modify: `tldw_chatbook/Agents/agent_runtime.py`
- Modify: `tldw_chatbook/Agents/tool_catalog.py`
- Test: `Tests/Agents/test_agent_loop_load_dedupe.py`
- Test: `Tests/Agents/test_agent_runtime.py`

**Interfaces:**
- Consumes: `ToolLoadSelection` from Task 1.
- Changes: `LoopDeps.load_schemas(ids: list[str], messages: list[dict], call: ToolCall) -> ToolLoadSelection` replaces the list-only callback so request-fit planning receives the current history and native call identity.
- Produces: `LoopDeps.replace_disclosed_names: Callable[[frozenset[str]], None]` with a non-throwing default.
- Produces: pure load-result formatting shared by projected-fit planning and actual runtime output.

- [ ] **Step 1: Replace count-era tests with failing replacement tests**

Script two load-only turns: first selects `foo`, second selects `bar`. Assert model-call active names are `[(), ("foo",), ("bar",)]`, the commit callback sees `{"foo"}` then `{"bar"}`, and no result contains `no room`. Add valid-but-omitted and invalid-only selections proving the old set is unchanged and messages differ.

- [ ] **Step 2: Add failing mixed-batch tests**

Use a native `ModelTurn` containing `load_tools` plus an ordinary already-disclosed call in both orders. Assert the ordinary call executes once under the old set, the load returns a retry-alone error, and neither active schemas nor permission names change. Repeat with two load calls.

- [ ] **Step 3: Run focused runtime tests and verify RED**

Run: `../../.venv/bin/python -m pytest Tests/Agents/test_agent_loop_load_dedupe.py Tests/Agents/test_agent_runtime.py -q`

- [ ] **Step 4: Implement minimal exclusive replacement**

Detect non-exclusive load batches before per-call dispatch. For a load-only call, format the structured selection; commit only when `accepted` is non-empty by replacing permission names and then assigning the runtime `active` list before the next dispatch boundary. Leave the old set untouched for invalid-only, fully omitted, mixed, or repeated loads.

- [ ] **Step 5: Update the model-facing load contract**

Change `LOAD_TOOLS_SCHEMA.description` to state: call alone; the accepted IDs replace the current catalog tool set; include every tool to retain. Keep runtime tools outside the catalog working set.

- [ ] **Step 6: Run focused runtime tests and verify GREEN**

Run: `../../.venv/bin/python -m pytest Tests/Agents/test_agent_loop_load_dedupe.py Tests/Agents/test_agent_runtime.py Tests/Agents/test_tool_catalog.py -q`

- [ ] **Step 7: Commit runtime replacement**

```bash
git add tldw_chatbook/Agents/agent_runtime.py tldw_chatbook/Agents/tool_catalog.py Tests/Agents/test_agent_loop_load_dedupe.py Tests/Agents/test_agent_runtime.py Tests/Agents/test_tool_catalog.py
git commit -m "feat(agents): replace loaded tool working sets atomically"
```

### Task 4: Deferred-load request-fit selection and permission lockstep

**Files:**
- Modify: `tldw_chatbook/Agents/agent_service.py`
- Modify: `tldw_chatbook/Agents/agent_runtime.py`
- Test: `Tests/Agents/test_agent_service.py`
- Test: `Tests/Agents/test_fleet_runtime.py`
- Test: `Tests/Chat/test_provider_continuation_crash_recovery.py`

**Interfaces:**
- Consumes: structured runtime load interface from Task 3.
- Produces: a side-effect-free `load_schemas(ids, messages, call) -> ToolLoadSelection` service selector that resolves IDs/names, filters authority, deduplicates names, and greedily evaluates requested schemas against the projected next request in request order.
- Produces: a non-throwing disclosed-name replacement closure bound to the same mutable set used by `_make_invoke_tool`.

- [ ] **Step 1: Write failing service selection tests**

Prove an individual schema estimated above the 10% automatic threshold is accepted when the complete next request fits. Prove the same schema is omitted under history pressure, and a later smaller requested schema can still be accepted. Assert invalid IDs, disallowed names, and name/ID aliases produce literal structured categories without leaking disallowed schemas into `accepted`.

- [ ] **Step 2: Write failing permission lockstep test**

Run `load foo → invoke foo → load bar → invoke foo → invoke bar`. Use the real registry and permission-gate closure. Assert the first `foo` call succeeds, the replaced `foo` call is blocked, and `bar` succeeds immediately after its selection commits.

- [ ] **Step 3: Run focused service tests and verify RED**

Run: `../../.venv/bin/python -m pytest Tests/Agents/test_agent_service.py Tests/Agents/test_fleet_runtime.py Tests/Chat/test_provider_continuation_crash_recovery.py -q`

- [ ] **Step 4: Implement projected request-fit selection**

Reuse the live effective request builder, including current messages, provider continuation metadata, runtime schemas, run-log availability, response reserve, and deterministic load result. Recompute the complete candidate request for each requested schema so a rejected large schema does not block a later small schema. Return categories only; do not mutate `disclosed_names` during selection.

- [ ] **Step 5: Implement lockstep commit and cache invalidation**

Bind `replace_disclosed_names` to `clear()` plus `update()` on the exact set passed to `_make_invoke_tool`. Ensure the existing fence protocol cache re-renders on replacement, including same-sized and shrinking sets, by keying the complete immutable schema representation rather than monotonic size assumptions.

- [ ] **Step 6: Run focused service tests and verify GREEN**

Run: `../../.venv/bin/python -m pytest Tests/Agents/test_agent_service.py Tests/Agents/test_fleet_runtime.py Tests/Chat/test_provider_continuation_crash_recovery.py -q`

- [ ] **Step 7: Commit deferred load selection**

```bash
git add tldw_chatbook/Agents/agent_service.py tldw_chatbook/Agents/agent_runtime.py Tests/Agents/test_agent_service.py Tests/Agents/test_fleet_runtime.py Tests/Chat/test_provider_continuation_crash_recovery.py
git commit -m "feat(agents): fit deferred tools to live request headroom"
```

### Task 5: Production-shaped reachability, obsolete-contract cleanup, and closeout

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: count-era tests returned by `rg "DIRECT_DISCLOSE_THRESHOLD|max_active_tools|initial_disclosure\\(" Tests tldw_chatbook`
- Modify: normative comments returned by the same search under `tldw_chatbook/`
- Test: `Tests/Agents/test_local_tools_integration.py`
- Test: `Tests/Chat/test_console_agent_bridge.py`
- Modify: `backlog/tasks/task-15261 - Replace-the-fixed-active-tool-cap-with-token-budgeted-discovery.md`
- Modify: `backlog/decisions/104-token-budgeted-agent-tool-disclosure.md`

**Interfaces:**
- Consumes: all earlier task interfaces.
- Produces: production-shaped MCP-last find → load → approve → execute evidence.

- [ ] **Step 1: Write the failing production-shaped regression**

Build the shipped builtin/local/Library-sized registry, register a real in-process fake MCP provider last, force discovery by schema cost rather than entry count, and script exact-name search, load-only selection, approval, and invocation. Assert the MCP result and approval path are observed; do not assert provider registration indexes.

- [ ] **Step 2: Run the integration test and verify RED**

Run: `../../.venv/bin/python -m pytest Tests/Agents/test_local_tools_integration.py Tests/Chat/test_console_agent_bridge.py -q`

- [ ] **Step 3: Rewrite obsolete count pins and normative comments**

Replace tests that manufacture `DIRECT_DISCLOSE_THRESHOLD + 1` entries with injected large schema costs or deliberately constrained request limits. Delete `max_active_tools` constructor arguments and propagation assertions. Preserve historical specs, reviews, and QA captures unchanged as provenance.

- [ ] **Step 4: Run the complete focused regression shard**

Run: `../../.venv/bin/python -m pytest Tests/Agents/test_agent_models.py Tests/Agents/test_tool_catalog.py Tests/Agents/test_agent_loop_load_dedupe.py Tests/Agents/test_agent_runtime.py Tests/Agents/test_agent_service.py Tests/Agents/test_local_tools_integration.py Tests/Agents/test_skill_tool_spawn.py Tests/Agents/test_fleet_runtime.py Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_console_agent_swap.py Tests/Chat/test_console_fleet_wake_safety.py Tests/Chat/test_console_agent_run_budget.py Tests/Chat/test_provider_continuation_crash_recovery.py Tests/UI/test_console_project_instructions.py -q`

- [ ] **Step 5: Run static and diff verification**

```bash
../../.venv/bin/python -m py_compile tldw_chatbook/Agents/agent_models.py tldw_chatbook/Agents/tool_catalog.py tldw_chatbook/Agents/agent_runtime.py tldw_chatbook/Agents/agent_service.py tldw_chatbook/Chat/console_agent_bridge.py
git diff --check
rg -n "DIRECT_DISCLOSE_THRESHOLD|max_active_tools" tldw_chatbook Tests
```

Expected: compilation and diff check succeed; the final search returns no live runtime/test references.

- [ ] **Step 6: Self-review against the specification**

Check every acceptance criterion, permission transition, provider representation, error outcome, mixed batch, first-request parity, child run, continuation restore, and cache invalidation path. Add a failing regression before correcting any discovered behavior bug.

- [ ] **Step 7: Complete task documentation**

Check every acceptance criterion, add concise implementation notes listing approach/trade-offs/files/tests, keep ADR-104 `Accepted`, and set TASK-15261 to `Done` only after all verification succeeds.

- [ ] **Step 8: Commit closeout**

```bash
git add tldw_chatbook Tests backlog Docs/superpowers/plans/2026-08-30-token-budgeted-tool-discovery.md
git commit -m "test(agents): prove unbounded catalog tool reachability"
```
