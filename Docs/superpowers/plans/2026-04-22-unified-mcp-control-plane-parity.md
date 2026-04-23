# Unified MCP Control Plane Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Unified MCP control plane inside Chatbook so `Tools & Settings` can host explicit `Local` and `Server` MCP panes with configured server targets, destination-local context persistence, canonical runtime-policy enforcement, and staged server-side administration parity.

**Architecture:** Land this in four milestones. Milestone A establishes the shared seams: MCP API client/contracts, configured server targets, Unified MCP destination context, canonical MCP runtime-policy ids, local/server control services, and a first host inside `Tools & Settings` for `Overview` and `Inventory`. Milestones B-D extend the same seam with local external profiles plus server catalogs/external servers, then server governance, then advanced MCP Hub administration, while keeping server authority explicit and cache behavior section-sensitive.

**Tech Stack:** Python 3.11+, Textual, `httpx`, SQLite/JSON-backed local stores, existing `tldw_chatbook/MCP/*`, existing `tldw_api` client stack, pytest

---

## Scope Check

This remains one plan because the implementation is coupled at one seam:

- configured server targets
- destination-local Unified MCP context
- canonical MCP runtime-policy action ids
- local/server Unified MCP services
- one `Tools & Settings` host shell

Execution is still staged by the approved slices:

- Tasks 1-6: Slice 1 foundation and first browse host
- Task 7: Slice 2 local external profiles plus server catalogs/external servers
- Task 8: Slice 3 governance core
- Task 9: Slice 4 advanced administration, docs, and closing verification

This plan assumes the `tldw_chatbook/runtime_policy/*` package already exists on the execution branch. If execution happens on another branch that does not yet have that package, bring that foundation in before starting Task 3.

Implementation note for agentic workers:

- Use `@superpowers:test-driven-development` before each task.
- Use `@superpowers:verification-before-completion` before calling any task complete.
- Do not mutate app-global runtime source state when implementing destination-local Unified MCP source switching.

## File Map

- Create: `tldw_chatbook/tldw_api/mcp_unified_schemas.py`
  Responsibility: Pydantic request/response models for Unified MCP runtime, catalogs, external servers, governance, advanced admin, and access-context negotiation payloads.
- Create: `tldw_chatbook/tldw_api/mcp_unified_client.py`
  Responsibility: Thin endpoint wrapper around `TLDWAPIClient` for `/api/v1/mcp/*`, scoped catalog endpoints, `/api/v1/mcp/hub/*`, and any access-context bootstrap helpers needed to normalize manageable server scopes.
- Modify: `tldw_chatbook/tldw_api/__init__.py`
  Responsibility: Export the Unified MCP client and schemas.
- Create: `Tests/tldw_api/test_mcp_unified_client.py`
  Responsibility: Verify endpoint wiring, payload serialization, capability/bootstrap helpers, and scoped route selection.

- Create: `tldw_chatbook/MCP/unified_control_models.py`
  Responsibility: Shared dataclasses/TypedDict-style records for `ConfiguredServerTarget`, `UnifiedMCPContext`, `ServerAccessContext`, target-status metadata, section capability flags, and normalized panel records.
- Create: `tldw_chatbook/MCP/server_target_store.py`
  Responsibility: Dedicated local store for configured server targets, legacy single-target import, target selection, and target metadata updates.
- Create: `tldw_chatbook/MCP/unified_context_store.py`
  Responsibility: Dedicated local store for destination-local Unified MCP state, including per-server restored section/scope selections.
- Create: `Tests/MCP/test_server_target_store.py`
  Responsibility: Verify one-time legacy import, post-import registry precedence, target update semantics, and active-target resolution.
- Create: `Tests/MCP/test_unified_context_store.py`
  Responsibility: Verify Unified MCP context persistence, per-server state partitioning, and restore behavior independent of app-global runtime state.

- Modify: `tldw_chatbook/runtime_policy/registry.py`
  Responsibility: Register the canonical Unified MCP action ids in existing `<resource>.<action>.<source>` format and maintain compatibility aliases for the coarse old MCP buckets during rollout.
- Modify: `tldw_chatbook/runtime_policy/types.py`
  Responsibility: Extend runtime-policy types only where needed for explicit destination-local source evaluation and clearer MCP-specific decisions.
- Modify: `tldw_chatbook/runtime_policy/engine.py`
  Responsibility: Support evaluation against explicit runtime state overrides for Unified MCP destination actions without requiring the app-global authoritative source to change.
- Modify: `tldw_chatbook/runtime_policy/enforcement.py`
  Responsibility: Expose service-level helpers that can enforce canonical MCP action ids using destination-local runtime state.
- Modify: `tldw_chatbook/runtime_policy/bootstrap.py`
  Responsibility: Keep app-global runtime source bootstrap authoritative for the rest of the app while avoiding accidental coupling to Unified MCP destination context.
- Modify: `tldw_chatbook/runtime_policy/source_state.py`
  Responsibility: Keep app-global snapshot semantics intact while remaining compatible with the new MCP-local context store.
- Modify: `Tests/RuntimePolicy/test_runtime_policy_core.py`
  Responsibility: Verify canonical MCP action registration, alias compatibility, and denial of unknown action ids.
- Modify: `Tests/RuntimePolicy/test_runtime_policy_bootstrap.py`
  Responsibility: Verify app-global runtime policy state remains separate from Unified MCP target/context stores.
- Modify: `Tests/RuntimePolicy/test_boundary_guards.py`
  Responsibility: Verify destination-local MCP actions cannot silently cross source/server boundaries.

- Create: `tldw_chatbook/MCP/local_store.py`
  Responsibility: Persist local external MCP profiles, discovery snapshots, and local governance rules.
- Create: `tldw_chatbook/MCP/local_control_service.py`
  Responsibility: Expose local `Overview`, `Inventory`, `External Servers`, and `Governance` operations on top of `tldw_chatbook/MCP/*` and the new local store.
- Modify: `tldw_chatbook/MCP/server.py`
  Responsibility: Expose stable local capability manifest metadata for `Overview` and `Inventory` without relying on loopback self-connection.
- Modify: `tldw_chatbook/MCP/client.py`
  Responsibility: Add helpers for profile-backed connect/disconnect/test/discovery refresh that the local control service can call.
- Create: `Tests/MCP/test_local_store.py`
  Responsibility: Verify local profile CRUD, discovery snapshot persistence, and governance record updates.
- Create: `Tests/MCP/test_local_control_service.py`
  Responsibility: Verify local manifest inventory, profile lifecycle actions, and local governance behavior.

- Create: `tldw_chatbook/MCP/server_unified_service.py`
  Responsibility: Resolve `ServerAccessContext`, negotiate section/endpoint capabilities, enforce section-sensitive cache policy, and normalize server MCP records for the UI.
- Create: `tldw_chatbook/MCP/unified_control_plane_service.py`
  Responsibility: Orchestrate local/server operations, destination-local source/scope changes, and panel-facing view models across all slices.
- Create: `Tests/MCP/test_server_unified_service.py`
  Responsibility: Verify capability negotiation, scope reset behavior, cache invalidation, and server mutation refresh rules.
- Create: `Tests/MCP/test_unified_control_plane_service.py`
  Responsibility: Verify orchestration, source/scope selection, per-server context restore, and section routing.

- Create: `tldw_chatbook/UI/MCP_Modules/__init__.py`
  Responsibility: Package marker for MCP-specific UI modules.
- Create: `tldw_chatbook/UI/MCP_Modules/unified_mcp_panel.py`
  Responsibility: Main Unified MCP host widget mounted inside `Tools & Settings`.
- Create: `tldw_chatbook/UI/MCP_Modules/unified_mcp_sections.py`
  Responsibility: Section-specific render helpers/widgets for `Overview`, `Inventory`, `Catalogs`, `External Servers`, `Governance`, and `Advanced`.
- Modify: `tldw_chatbook/UI/Tools_Settings_Window.py`
  Responsibility: Add the Unified MCP navigation entry and mount the MCP panel in a new `ts-view-unified-mcp` area.
- Modify: `tldw_chatbook/UI/Screens/tools_settings_screen.py`
  Responsibility: Preserve and restore Unified MCP view state cleanly through the existing screen wrapper.
- Modify: `tldw_chatbook/app.py`
  Responsibility: Bootstrap target/context stores and Unified MCP services, then expose them to the Tools & Settings shell.
- Modify: `Tests/UI/test_tools_settings_window.py`
  Responsibility: Verify navigation, mounting, and state restore for the new Unified MCP host view.
- Create: `Tests/UI/test_unified_mcp_panel.py`
  Responsibility: Verify source switching, server target selection, scope switching, browse behavior, stale-cache labels, and write-surface gating.

- Modify: `Docs/Parity/2026-04-21-capability-matrix.md`
  Responsibility: Mark Unified MCP parity state based on verified shipped behavior.
- Modify: `Docs/Parity/2026-04-21-gap-ledger.md`
  Responsibility: Move Unified MCP gaps from open to landed or narrowed follow-on state.
- Modify: `Docs/Parity/2026-04-21-execution-roadmap.md`
  Responsibility: Reflect rollout status after verification.

## Task 1: Add Unified MCP API Schemas And Client Foundation

**Files:**
- Create: `tldw_chatbook/tldw_api/mcp_unified_schemas.py`
- Create: `tldw_chatbook/tldw_api/mcp_unified_client.py`
- Modify: `tldw_chatbook/tldw_api/__init__.py`
- Test: `Tests/tldw_api/test_mcp_unified_client.py`

- [ ] **Step 1: Write the failing MCP client tests**

```python
@pytest.mark.asyncio
async def test_get_mcp_status_hits_unified_status_endpoint(monkeypatch):
    root = TLDWAPIClient("https://example.com")
    client = MCPUnifiedClient(root)
    mocked = AsyncMock(return_value={"status": "ok"})
    monkeypatch.setattr(root, "_request", mocked)

    await client.get_status()

    args, kwargs = mocked.await_args
    assert args[:2] == ("GET", "/api/v1/mcp/status")
    assert kwargs["params"] is None


@pytest.mark.asyncio
async def test_list_visible_catalogs_hits_unified_catalog_endpoint(monkeypatch):
    root = TLDWAPIClient("https://example.com")
    client = MCPUnifiedClient(root)
    mocked = AsyncMock(return_value={"catalogs": []})
    monkeypatch.setattr(root, "_request", mocked)

    await client.list_visible_tool_catalogs()

    args, _kwargs = mocked.await_args
    assert args[:2] == ("GET", "/api/v1/mcp/tool_catalogs")
```

- [ ] **Step 2: Run the focused MCP client tests to verify they fail**

Run: `python3 -m pytest Tests/tldw_api/test_mcp_unified_client.py -q`
Expected: FAIL with missing Unified MCP client/schemas.

- [ ] **Step 3: Implement the minimal client and schema layer**

```python
class MCPUnifiedClient:
    def __init__(self, root_client: TLDWAPIClient):
        self.root_client = root_client

    async def get_status(self) -> dict[str, Any]:
        return await self.root_client._request("GET", "/api/v1/mcp/status")

    async def list_modules(self) -> dict[str, Any]:
        return await self.root_client._request("GET", "/api/v1/mcp/modules")

    async def list_visible_tool_catalogs(self) -> dict[str, Any]:
        return await self.root_client._request("GET", "/api/v1/mcp/tool_catalogs")

    async def test_catalog_connection(self, payload: CatalogConnectionTestRequest) -> dict[str, Any]:
        return await self.root_client._request(
            "POST",
            "/api/v1/mcp/catalog/test-connection",
            json_data=payload.model_dump(exclude_none=True),
        )
```

Required methods in this task:

- `get_status`
- `get_health`
- `get_metrics`
- `list_modules`
- `get_module_health`
- `list_tools`
- `list_resources`
- `list_prompts`
- `execute_tool`
- `list_visible_tool_catalogs`
- `test_catalog_connection`

Also add the minimal access-context bootstrap helper(s) that Task 5 will need to normalize current principal/team/org scope options, even if those helpers compose multiple underlying endpoints instead of hitting one dedicated route.

Do not add governance or external-server mutation methods in this task. Those belong to later slices.

- [ ] **Step 4: Run the MCP client tests again**

Run: `python3 -m pytest Tests/tldw_api/test_mcp_unified_client.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/tldw_api/mcp_unified_schemas.py tldw_chatbook/tldw_api/mcp_unified_client.py tldw_chatbook/tldw_api/__init__.py Tests/tldw_api/test_mcp_unified_client.py
git commit -m "feat: add unified mcp api client foundation"
```

## Task 2: Add Configured Server Target And Unified MCP Context Stores

**Files:**
- Create: `tldw_chatbook/MCP/unified_control_models.py`
- Create: `tldw_chatbook/MCP/server_target_store.py`
- Create: `tldw_chatbook/MCP/unified_context_store.py`
- Test: `Tests/MCP/test_server_target_store.py`
- Test: `Tests/MCP/test_unified_context_store.py`

- [ ] **Step 1: Write the failing store tests**

```python
def test_bootstrap_from_legacy_config_only_when_registry_is_empty(tmp_path):
    store = ConfiguredServerTargetStore(tmp_path / "servers.json")
    imported = store.bootstrap_from_legacy_config(
        {"tldw_api": {"base_url": "https://example.com/api/", "api_key": "secret"}}
    )

    assert imported is True
    targets = store.list_targets()
    assert len(targets) == 1
    assert targets[0].base_url == "https://example.com/api"


def test_legacy_config_does_not_overwrite_existing_target_registry(tmp_path):
    store = ConfiguredServerTargetStore(tmp_path / "servers.json")
    store.save_targets([ConfiguredServerTarget(server_id="saved", label="Saved", base_url="https://saved.example/api")])

    imported = store.bootstrap_from_legacy_config(
        {"tldw_api": {"base_url": "https://other.example/api/", "api_key": "secret"}}
    )

    assert imported is False
    assert store.list_targets()[0].server_id == "saved"


def test_unified_mcp_context_partitions_per_server_state(tmp_path):
    store = UnifiedMCPContextStore(tmp_path / "mcp_context.json")
    context = UnifiedMCPContext(
        selected_source="server",
        selected_active_server_id="server-a",
        per_server_state={"server-a": {"selected_section": "inventory"}},
    )

    store.save(context)
    restored = store.load()
    assert restored.per_server_state["server-a"]["selected_section"] == "inventory"
```

- [ ] **Step 2: Run the focused store tests to verify they fail**

Run: `python3 -m pytest Tests/MCP/test_server_target_store.py Tests/MCP/test_unified_context_store.py -q`
Expected: FAIL with missing store/model modules.

- [ ] **Step 3: Implement the minimal stores and shared records**

```python
@dataclass(frozen=True, slots=True)
class ConfiguredServerTarget:
    server_id: str
    label: str
    base_url: str
    auth_mode: str = "api_key"
    auth_reference: str | None = None
    is_default: bool = False
    last_known_server_label: str | None = None
    last_known_reachability: str | None = None
    last_known_auth_state: str | None = None
    last_connected_at: str | None = None
    updated_at: str | None = None


class ConfiguredServerTargetStore:
    def bootstrap_from_legacy_config(self, app_config: Mapping[str, Any]) -> bool:
        if self.list_targets():
            return False
        api_config = dict(app_config.get("tldw_api", {}) or {})
        base_url = str(api_config.get("base_url") or api_config.get("api_url") or "").strip()
        if not base_url:
            return False
        target = ConfiguredServerTarget(
            server_id=_normalize_server_identity(base_url),
            label=urlsplit(base_url).netloc or base_url,
            base_url=base_url.rstrip("/"),
            auth_mode="bearer" if api_config.get("bearer_token") else "api_key",
            auth_reference="legacy:tldw_api",
            is_default=True,
        )
        self.save_targets([target])
        return True
```

Required behavior in this task:

- one-time import from legacy `tldw_api` config only when the target registry is empty
- registry becomes authoritative after import
- no automatic overwrite when legacy config changes later
- configured targets persist the minimum metadata required by the approved spec: `last_known_server_label`, `last_known_reachability`, `last_known_auth_state`, `last_connected_at`, and `updated_at`
- Unified MCP context stores `selected_source`, `selected_active_server_id`, selected scope, selected section, and `per_server_state`
- no raw secrets duplicated into either store

- [ ] **Step 4: Run the store tests again**

Run: `python3 -m pytest Tests/MCP/test_server_target_store.py Tests/MCP/test_unified_context_store.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/MCP/unified_control_models.py tldw_chatbook/MCP/server_target_store.py tldw_chatbook/MCP/unified_context_store.py Tests/MCP/test_server_target_store.py Tests/MCP/test_unified_context_store.py
git commit -m "feat: add unified mcp target and context stores"
```

## Task 3: Extend Runtime Policy For Canonical Unified MCP Actions

**Files:**
- Modify: `tldw_chatbook/runtime_policy/registry.py`
- Modify: `tldw_chatbook/runtime_policy/types.py`
- Modify: `tldw_chatbook/runtime_policy/engine.py`
- Modify: `tldw_chatbook/runtime_policy/enforcement.py`
- Modify: `tldw_chatbook/runtime_policy/bootstrap.py`
- Modify: `tldw_chatbook/runtime_policy/source_state.py`
- Test: `Tests/RuntimePolicy/test_runtime_policy_core.py`
- Test: `Tests/RuntimePolicy/test_runtime_policy_bootstrap.py`
- Test: `Tests/RuntimePolicy/test_boundary_guards.py`

- [ ] **Step 1: Write the failing runtime-policy tests**

```python
def test_runtime_policy_registers_canonical_unified_mcp_action_ids():
    assert "mcp.runtime.list.local" in CAPABILITY_REGISTRY
    assert "mcp.inventory.list.local" in CAPABILITY_REGISTRY
    assert "mcp.catalogs.configure.server" in CAPABILITY_REGISTRY
    assert "mcp.external_servers.observe.server" in CAPABILITY_REGISTRY


def test_policy_engine_can_evaluate_against_explicit_runtime_state():
    engine = PolicyEngine(CAPABILITY_REGISTRY)
    state = RuntimeSourceState(active_source="local")

    decision = engine.evaluate(
        action_id="mcp.inventory.list.server",
        runtime_state_override=state,
    )

    assert decision.allowed is False
    assert decision.effective_source == "local"
```

- [ ] **Step 2: Run the runtime-policy tests to verify they fail**

Run: `python3 -m pytest Tests/RuntimePolicy/test_runtime_policy_core.py Tests/RuntimePolicy/test_runtime_policy_bootstrap.py Tests/RuntimePolicy/test_boundary_guards.py -q`
Expected: FAIL with missing canonical MCP ids and missing explicit-state evaluation support.

- [ ] **Step 3: Implement the minimal MCP runtime-policy expansion**

```python
_capability(
    "remote_mcp_control_plane_governance",
    "Remote MCP Control Plane / Governance",
    "mcp_governance",
    sources=REMOTE_ONLY_SOURCES,
    resources=(
        _resource("mcp.inventory", actions=(LIST, OBSERVE), sources=SEPARATED_SOURCES),
        _resource("mcp.catalogs", actions=(LIST, CONFIGURE, TRIGGER, OBSERVE), sources=REMOTE_ONLY_SOURCES),
        _resource("mcp.external_servers", actions=(LIST, CONFIGURE, TRIGGER, OBSERVE), sources=REMOTE_ONLY_SOURCES),
        _resource("mcp.credentials", actions=(LIST, CONFIGURE, OBSERVE), sources=REMOTE_ONLY_SOURCES),
        _resource("mcp.advanced", actions=(LIST, CONFIGURE, TRIGGER, OBSERVE), sources=REMOTE_ONLY_SOURCES),
    ),
)

def evaluate(self, *, action_id: str, runtime_state_override: RuntimeSourceState | None = None) -> PolicyDecision:
    runtime_state = runtime_state_override or self.runtime_state
    ...
```

Required behavior in this task:

- register the canonical action ids from the spec in existing `<resource>.<action>.<source>` format
- extend, rather than silently replace, the already-landed coarse MCP buckets such as `mcp.runtime.*` and `mcp.governance.*`
- preserve compatibility aliases from the old coarse MCP buckets where still needed during rollout
- add explicit runtime-state override support so Unified MCP destination actions can be checked without mutating the app-global authoritative source
- keep `runtime_policy_snapshot` semantics app-global; do not repurpose it for Unified MCP destination state

- [ ] **Step 4: Run the runtime-policy tests again**

Run: `python3 -m pytest Tests/RuntimePolicy/test_runtime_policy_core.py Tests/RuntimePolicy/test_runtime_policy_bootstrap.py Tests/RuntimePolicy/test_boundary_guards.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/runtime_policy/registry.py tldw_chatbook/runtime_policy/types.py tldw_chatbook/runtime_policy/engine.py tldw_chatbook/runtime_policy/enforcement.py tldw_chatbook/runtime_policy/bootstrap.py tldw_chatbook/runtime_policy/source_state.py Tests/RuntimePolicy/test_runtime_policy_core.py Tests/RuntimePolicy/test_runtime_policy_bootstrap.py Tests/RuntimePolicy/test_boundary_guards.py
git commit -m "feat: add runtime policy support for unified mcp"
```

## Task 4: Build The Local Unified MCP Store And Control Service

**Files:**
- Create: `tldw_chatbook/MCP/local_store.py`
- Create: `tldw_chatbook/MCP/local_control_service.py`
- Modify: `tldw_chatbook/MCP/server.py`
- Modify: `tldw_chatbook/MCP/client.py`
- Test: `Tests/MCP/test_local_store.py`
- Test: `Tests/MCP/test_local_control_service.py`

- [ ] **Step 1: Write the failing local MCP tests**

```python
def test_local_control_service_builds_inventory_from_local_manifest_without_loopback():
    service = LocalMCPControlService(
        store=FakeLocalStore(),
        manifest_provider=lambda: {
            "tools": [{"name": "search_notes"}],
            "resources": [{"uri": "note://1"}],
            "prompts": [{"name": "summarize_note"}],
        },
    )

    inventory = service.get_inventory()
    assert inventory["tools"][0]["name"] == "search_notes"


@pytest.mark.asyncio
async def test_local_control_service_connects_profile_and_persists_discovery_snapshot():
    store = FakeLocalStore()
    client = FakeMCPClient()
    service = LocalMCPControlService(store=store, client=client, manifest_provider=lambda: {})

    await service.connect_profile("profile-a")

    assert store.discovery_snapshots["profile-a"]["tools"][0]["name"] == "remote_tool"
```

- [ ] **Step 2: Run the local MCP tests to verify they fail**

Run: `python3 -m pytest Tests/MCP/test_local_store.py Tests/MCP/test_local_control_service.py -q`
Expected: FAIL with missing local store/service and missing manifest helpers.

- [ ] **Step 3: Implement the minimal local control seam**

```python
class LocalMCPControlService:
    def __init__(self, *, store, client=None, manifest_provider=None):
        self.store = store
        self.client = client or MCPClient()
        self.manifest_provider = manifest_provider or describe_local_mcp_capabilities

    def get_inventory(self) -> dict[str, Any]:
        return self.manifest_provider()

    async def connect_profile(self, profile_id: str) -> dict[str, Any]:
        profile = self.store.get_profile(profile_id)
        await self.client.connect_to_server(
            profile_id,
            profile.command,
            args=profile.args,
            env=profile.env,
        )
        snapshot = await self.client.describe_server(profile_id)
        self.store.save_discovery_snapshot(profile_id, snapshot)
        return snapshot
```

Required behavior in this task:

- local inventory must come from a manifest/helper built over `MCP/tools.py`, `MCP/resources.py`, and `MCP/prompts.py`, not from loopback self-connection
- local external profile registry persists command/args/env placeholders, discovery snapshots, and local governance rules
- local secret handling stores only non-secret literals or env placeholders, never raw secrets
- local service exposes `Overview`, `Inventory`, `External Servers`, and `Governance` primitives

- [ ] **Step 4: Run the local MCP tests again**

Run: `python3 -m pytest Tests/MCP/test_local_store.py Tests/MCP/test_local_control_service.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/MCP/local_store.py tldw_chatbook/MCP/local_control_service.py tldw_chatbook/MCP/server.py tldw_chatbook/MCP/client.py Tests/MCP/test_local_store.py Tests/MCP/test_local_control_service.py
git commit -m "feat: add local unified mcp control service"
```

## Task 5: Build The Server Unified MCP Service And Control-Plane Orchestrator

**Files:**
- Modify: `tldw_chatbook/tldw_api/mcp_unified_client.py`
- Create: `tldw_chatbook/MCP/server_unified_service.py`
- Create: `tldw_chatbook/MCP/unified_control_plane_service.py`
- Test: `Tests/tldw_api/test_mcp_unified_client.py`
- Test: `Tests/MCP/test_server_unified_service.py`
- Test: `Tests/MCP/test_unified_control_plane_service.py`

- [ ] **Step 1: Write the failing server-service/orchestrator tests**

```python
@pytest.mark.asyncio
async def test_server_service_negotiates_section_capabilities_from_available_endpoints():
    client = FakeMCPUnifiedClient(status={"status": "ok"}, tools={"tools": []}, governance_forbidden=True)
    service = ServerUnifiedMCPService(client=client)

    context = await service.resolve_access_context(target=ConfiguredServerTarget(server_id="srv", label="Srv", base_url="https://srv"))

    assert context.section_capabilities["overview"] is True
    assert context.section_capabilities["inventory"] is True
    assert context.section_capabilities["governance"] is False


@pytest.mark.asyncio
async def test_control_plane_service_restores_per_server_context_without_touching_global_runtime_source():
    orchestrator = UnifiedMCPControlPlaneService(...)
    context = await orchestrator.select_server_target("server-a")

    assert context.selected_active_server_id == "server-a"
    assert orchestrator.selected_source == "server"
```

- [ ] **Step 2: Run the server-service/orchestrator tests to verify they fail**

Run: `python3 -m pytest Tests/MCP/test_server_unified_service.py Tests/MCP/test_unified_control_plane_service.py Tests/tldw_api/test_mcp_unified_client.py -q`
Expected: FAIL with missing server service/orchestrator and missing capability-negotiation helpers.

- [ ] **Step 3: Implement the minimal server control seam**

```python
class ServerUnifiedMCPService:
    async def resolve_access_context(self, target: ConfiguredServerTarget) -> ServerAccessContext:
        status = await self.client.get_status()
        section_capabilities = {
            "overview": True,
            "inventory": await self._probe_inventory_capability(),
            "catalogs": await self._probe_catalog_capability(),
            "external_servers": await self._probe_external_server_capability(),
            "governance": await self._probe_governance_capability(),
            "advanced": await self._probe_advanced_capability(),
        }
        return ServerAccessContext(
            server_id=target.server_id,
            can_use_personal_scope=True,
            manageable_team_ids=[],
            manageable_org_ids=[],
            can_use_system_admin_scope=False,
            section_capabilities=section_capabilities,
            endpoint_capabilities={},
        )


class UnifiedMCPControlPlaneService:
    async def select_source(self, source: Literal["local", "server"]) -> UnifiedMCPContext:
        self.context = self.context_store.update_source(source)
        return self.context
```

Required behavior in this task:

- server access context resolves negotiated `section_capabilities` and `endpoint_capabilities`
- `resolve_access_context` may use a dedicated bootstrap endpoint or a composed helper over current-principal and team/org membership endpoints, but the normalization seam must live in `ServerUnifiedMCPService` and be covered by tests
- browse-oriented cache is partitioned by `server_id`, scope, and section
- no optimistic local mutation for server-owned records
- post-mutation refresh is required before server state is presented as authoritative
- orchestrator routes `Overview` and `Inventory` through local or server services using destination-local context only

- [ ] **Step 4: Run the server-service/orchestrator tests again**

Run: `python3 -m pytest Tests/MCP/test_server_unified_service.py Tests/MCP/test_unified_control_plane_service.py Tests/tldw_api/test_mcp_unified_client.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/tldw_api/mcp_unified_client.py tldw_chatbook/MCP/server_unified_service.py tldw_chatbook/MCP/unified_control_plane_service.py Tests/tldw_api/test_mcp_unified_client.py Tests/MCP/test_server_unified_service.py Tests/MCP/test_unified_control_plane_service.py
git commit -m "feat: add unified mcp server service foundation"
```

## Task 6: Mount Unified MCP Inside Tools & Settings For Slice 1 Browse

**Files:**
- Create: `tldw_chatbook/UI/MCP_Modules/__init__.py`
- Create: `tldw_chatbook/UI/MCP_Modules/unified_mcp_panel.py`
- Create: `tldw_chatbook/UI/MCP_Modules/unified_mcp_sections.py`
- Modify: `tldw_chatbook/UI/Tools_Settings_Window.py`
- Modify: `tldw_chatbook/UI/Screens/tools_settings_screen.py`
- Modify: `tldw_chatbook/app.py`
- Test: `Tests/UI/test_tools_settings_window.py`
- Test: `Tests/UI/test_unified_mcp_panel.py`

- [ ] **Step 1: Write the failing UI/app wiring tests**

```python
@pytest.mark.asyncio
async def test_tools_settings_window_exposes_unified_mcp_view(settings_window):
    nav_button = settings_window.query_one("#ts-nav-unified-mcp", Button)
    assert nav_button.label.plain == "Unified MCP"

    await settings_window.on_button_pressed(Button.Pressed(nav_button))
    assert settings_window.query_one("#tools-settings-content-pane").current == "ts-view-unified-mcp"


@pytest.mark.asyncio
async def test_unified_mcp_panel_switches_between_local_and_server_views():
    panel = UnifiedMCPPanel(app_instance=FakeAppWithMCPService())
    await panel.load_context()
    await panel.select_source("server")
    assert panel.context.selected_source == "server"
```

- [ ] **Step 2: Run the UI/app wiring tests to verify they fail**

Run: `python3 -m pytest Tests/UI/test_tools_settings_window.py Tests/UI/test_unified_mcp_panel.py -q`
Expected: FAIL with missing nav/view/panel wiring.

- [ ] **Step 3: Implement the minimal Slice 1 host**

```python
# Tools_Settings_Window.py
yield Button("Unified MCP", id="ts-nav-unified-mcp", classes="ts-nav-button")

with Container(id="ts-view-unified-mcp", classes="ts-view-area"):
    yield UnifiedMCPPanel(self.app_instance, id="unified-mcp-panel")


# app.py
self.mcp_server_target_store = ConfiguredServerTargetStore(...)
self.mcp_context_store = UnifiedMCPContextStore(...)
self.local_mcp_control_service = LocalMCPControlService(...)
self.server_mcp_unified_service = ServerUnifiedMCPService(...)
self.unified_mcp_service = UnifiedMCPControlPlaneService(...)
```

Required behavior in this task:

- initial host is inside `Tools & Settings`, not a new top-level destination
- panel exposes source switch, configured-server selector, server scope switch, and section switch
- Slice 1 UI only needs working `Overview` and `Inventory` browse behavior
- server scope options and any team/org entity pickers are derived from the active `ServerAccessContext`; invalid restored selections are evicted before browse requests run
- destination-local state restore must use `UnifiedMCPContextStore`, not `runtime_policy_snapshot`

- [ ] **Step 4: Run the UI/app wiring tests again**

Run: `python3 -m pytest Tests/UI/test_tools_settings_window.py Tests/UI/test_unified_mcp_panel.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/MCP_Modules/__init__.py tldw_chatbook/UI/MCP_Modules/unified_mcp_panel.py tldw_chatbook/UI/MCP_Modules/unified_mcp_sections.py tldw_chatbook/UI/Tools_Settings_Window.py tldw_chatbook/UI/Screens/tools_settings_screen.py tldw_chatbook/app.py Tests/UI/test_tools_settings_window.py Tests/UI/test_unified_mcp_panel.py
git commit -m "feat: add unified mcp slice 1 ui host"
```

## Task 7: Land Slice 2 Local External Profiles And Server Catalogs/External Servers

**Files:**
- Modify: `tldw_chatbook/tldw_api/mcp_unified_client.py`
- Modify: `tldw_chatbook/tldw_api/mcp_unified_schemas.py`
- Modify: `tldw_chatbook/MCP/local_store.py`
- Modify: `tldw_chatbook/MCP/local_control_service.py`
- Modify: `tldw_chatbook/MCP/server_unified_service.py`
- Modify: `tldw_chatbook/MCP/unified_control_plane_service.py`
- Modify: `tldw_chatbook/UI/MCP_Modules/unified_mcp_panel.py`
- Modify: `tldw_chatbook/UI/MCP_Modules/unified_mcp_sections.py`
- Test: `Tests/tldw_api/test_mcp_unified_client.py`
- Test: `Tests/MCP/test_local_control_service.py`
- Test: `Tests/MCP/test_server_unified_service.py`
- Test: `Tests/UI/test_unified_mcp_panel.py`

- [ ] **Step 1: Write the failing Slice 2 tests**

```python
@pytest.mark.asyncio
async def test_local_control_service_creates_and_tests_external_profile():
    service = LocalMCPControlService(...)
    profile = await service.save_external_profile({"profile_id": "p1", "command": "python", "args": ["-m", "demo"]})
    result = await service.test_external_profile("p1")
    assert profile["profile_id"] == "p1"
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_server_service_creates_external_server_and_refreshes_live_state():
    service = ServerUnifiedMCPService(client=FakeMCPUnifiedClient())
    created = await service.create_external_server({"name": "Docs", "transport": "http"})
    assert created["name"] == "Docs"
    assert service.cache_for("external_servers") is None
```

- [ ] **Step 2: Run the Slice 2 tests to verify they fail**

Run: `python3 -m pytest Tests/tldw_api/test_mcp_unified_client.py Tests/MCP/test_local_control_service.py Tests/MCP/test_server_unified_service.py Tests/UI/test_unified_mcp_panel.py -q`
Expected: FAIL with missing profile/catalog/external-server methods and UI actions.

- [ ] **Step 3: Implement the minimal Slice 2 behavior**

```python
async def create_external_server(self, payload: ExternalServerCreateRequest) -> dict[str, Any]:
    created = await self.client.create_external_server(payload)
    self.invalidate_cache(section="external_servers")
    return await self.get_external_server(created["id"])

async def save_external_profile(self, payload: Mapping[str, Any]) -> dict[str, Any]:
    profile = self.store.upsert_profile(payload)
    return profile
```

Required behavior in this task:

- local external profiles support create/update/delete, connect/disconnect/test, and discovery refresh
- server `Catalogs` section supports visible-catalog browse and org/team scoped catalog CRUD where permitted
- server `External Servers` section supports registry CRUD, import-from-catalog, auth templates, credential slots, credential bindings, and write-only secret set/clear flows
- `Overview`, `Inventory`, and `Catalogs` may show stale read-only cache when labeled
- `External Servers` and credential mutation flows must use live authoritative fetches
- write affordances for local profiles and server catalog/external-server actions only appear when both section capability checks and action-level runtime-policy checks allow them
- server scope/entity selection must be revalidated before any catalog or external-server mutation so stale team/org selections cannot leak across scopes

- [ ] **Step 4: Run the Slice 2 tests again**

Run: `python3 -m pytest Tests/tldw_api/test_mcp_unified_client.py Tests/MCP/test_local_control_service.py Tests/MCP/test_server_unified_service.py Tests/UI/test_unified_mcp_panel.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/tldw_api/mcp_unified_client.py tldw_chatbook/tldw_api/mcp_unified_schemas.py tldw_chatbook/MCP/local_store.py tldw_chatbook/MCP/local_control_service.py tldw_chatbook/MCP/server_unified_service.py tldw_chatbook/MCP/unified_control_plane_service.py tldw_chatbook/UI/MCP_Modules/unified_mcp_panel.py tldw_chatbook/UI/MCP_Modules/unified_mcp_sections.py Tests/tldw_api/test_mcp_unified_client.py Tests/MCP/test_local_control_service.py Tests/MCP/test_server_unified_service.py Tests/UI/test_unified_mcp_panel.py
git commit -m "feat: add unified mcp external profile and catalog controls"
```

## Task 8: Land Slice 3 Governance Core

**Files:**
- Modify: `tldw_chatbook/tldw_api/mcp_unified_client.py`
- Modify: `tldw_chatbook/tldw_api/mcp_unified_schemas.py`
- Modify: `tldw_chatbook/MCP/server_unified_service.py`
- Modify: `tldw_chatbook/MCP/unified_control_plane_service.py`
- Modify: `tldw_chatbook/UI/MCP_Modules/unified_mcp_panel.py`
- Modify: `tldw_chatbook/UI/MCP_Modules/unified_mcp_sections.py`
- Test: `Tests/tldw_api/test_mcp_unified_client.py`
- Test: `Tests/MCP/test_server_unified_service.py`
- Test: `Tests/MCP/test_unified_control_plane_service.py`
- Test: `Tests/UI/test_unified_mcp_panel.py`

- [ ] **Step 1: Write the failing governance tests**

```python
@pytest.mark.asyncio
async def test_server_service_lists_permission_profiles_without_using_stale_cache_for_mutation_paths():
    service = ServerUnifiedMCPService(client=FakeMCPUnifiedClient())
    profiles = await service.list_permission_profiles()
    assert profiles[0]["name"] == "Default"


@pytest.mark.asyncio
async def test_governance_mutation_refreshes_live_state_before_return():
    service = ServerUnifiedMCPService(client=FakeMCPUnifiedClient())
    updated = await service.update_approval_policy("policy-1", {"name": "Updated"})
    assert updated["name"] == "Updated"
    assert service.last_refresh["governance"] == "live"
```

- [ ] **Step 2: Run the governance tests to verify they fail**

Run: `python3 -m pytest Tests/tldw_api/test_mcp_unified_client.py Tests/MCP/test_server_unified_service.py Tests/MCP/test_unified_control_plane_service.py Tests/UI/test_unified_mcp_panel.py -q`
Expected: FAIL with missing governance client/service/UI methods.

- [ ] **Step 3: Implement the minimal governance core**

```python
async def update_permission_profile(self, profile_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    await self.client.update_permission_profile(profile_id, payload)
    return await self.client.get_permission_profile(profile_id)

async def list_effective_access(self, *, scope_kind: str, scope_ref: str | None) -> dict[str, Any]:
    return await self.client.get_effective_external_access(scope_kind=scope_kind, scope_ref=scope_ref)
```

Required behavior in this task:

- server governance covers permission profiles, policy assignments, approval policies, approval decisions, effective policy, effective external access, and policy override views/mutations where exposed
- governance surfaces must default to live fetches and post-mutation refresh
- write affordances only appear when `section_capabilities["governance"]` and action-level policy checks both allow them

- [ ] **Step 4: Run the governance tests again**

Run: `python3 -m pytest Tests/tldw_api/test_mcp_unified_client.py Tests/MCP/test_server_unified_service.py Tests/MCP/test_unified_control_plane_service.py Tests/UI/test_unified_mcp_panel.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/tldw_api/mcp_unified_client.py tldw_chatbook/tldw_api/mcp_unified_schemas.py tldw_chatbook/MCP/server_unified_service.py tldw_chatbook/MCP/unified_control_plane_service.py tldw_chatbook/UI/MCP_Modules/unified_mcp_panel.py tldw_chatbook/UI/MCP_Modules/unified_mcp_sections.py Tests/tldw_api/test_mcp_unified_client.py Tests/MCP/test_server_unified_service.py Tests/MCP/test_unified_control_plane_service.py Tests/UI/test_unified_mcp_panel.py
git commit -m "feat: add unified mcp governance controls"
```

## Task 9: Land Slice 4 Advanced Administration, Update Docs, And Verify End-To-End

**Files:**
- Modify: `tldw_chatbook/tldw_api/mcp_unified_client.py`
- Modify: `tldw_chatbook/tldw_api/mcp_unified_schemas.py`
- Modify: `tldw_chatbook/MCP/server_unified_service.py`
- Modify: `tldw_chatbook/MCP/unified_control_plane_service.py`
- Modify: `tldw_chatbook/UI/MCP_Modules/unified_mcp_panel.py`
- Modify: `tldw_chatbook/UI/MCP_Modules/unified_mcp_sections.py`
- Modify: `Docs/Parity/2026-04-21-capability-matrix.md`
- Modify: `Docs/Parity/2026-04-21-gap-ledger.md`
- Modify: `Docs/Parity/2026-04-21-execution-roadmap.md`
- Test: `Tests/tldw_api/test_mcp_unified_client.py`
- Test: `Tests/MCP/test_server_unified_service.py`
- Test: `Tests/MCP/test_unified_control_plane_service.py`
- Test: `Tests/UI/test_unified_mcp_panel.py`
- Test: `Tests/UI/test_tools_settings_window.py`

- [ ] **Step 1: Write the failing advanced-admin and docs tests**

```python
@pytest.mark.asyncio
async def test_server_service_exposes_governance_pack_summary_and_tool_registry_browse():
    service = ServerUnifiedMCPService(client=FakeMCPUnifiedClient())
    packs = await service.list_governance_packs()
    registry = await service.list_tool_registry()
    assert packs[0]["name"] == "Baseline"
    assert registry["modules"][0]["module_id"] == "search"


def test_capability_matrix_mentions_unified_mcp_after_vertical_lands():
    matrix = Path("Docs/Parity/2026-04-21-capability-matrix.md").read_text(encoding="utf-8")
    assert "Unified MCP" in matrix
```

- [ ] **Step 2: Run the advanced-admin/docs tests to verify they fail**

Run: `python3 -m pytest Tests/tldw_api/test_mcp_unified_client.py Tests/MCP/test_server_unified_service.py Tests/MCP/test_unified_control_plane_service.py Tests/UI/test_unified_mcp_panel.py Tests/UI/test_tools_settings_window.py -q`
Expected: FAIL with missing advanced client/service methods and parity-doc updates.

- [ ] **Step 3: Implement the minimal advanced/admin completion work**

```python
async def list_governance_packs(self) -> list[dict[str, Any]]:
    return (await self.client.list_governance_packs())["items"]

async def list_tool_registry(self) -> dict[str, Any]:
    return await self.client.get_tool_registry_summary()
```

Required behavior in this task:

- advanced section covers tool registry, capability mappings, governance packs plus trust-policy and upgrade flows, ACP profiles, path-scope objects, workspace-set objects/members, shared workspaces, and governance audit findings
- browse-only advanced sub-surfaces may use labeled stale read-only cache
- any advanced mutation path still uses live fetch + post-mutation refresh
- advanced mutation affordances only appear when the active server scope is allowed and the corresponding action-level runtime-policy check passes
- parity docs must reflect shipped behavior only, not intent

- [ ] **Step 4: Run the focused advanced suite, then the full Unified MCP suite**

Run: `python3 -m pytest Tests/tldw_api/test_mcp_unified_client.py Tests/MCP/test_server_target_store.py Tests/MCP/test_unified_context_store.py Tests/MCP/test_local_store.py Tests/MCP/test_local_control_service.py Tests/MCP/test_server_unified_service.py Tests/MCP/test_unified_control_plane_service.py Tests/RuntimePolicy/test_runtime_policy_core.py Tests/RuntimePolicy/test_runtime_policy_bootstrap.py Tests/RuntimePolicy/test_boundary_guards.py Tests/UI/test_tools_settings_window.py Tests/UI/test_unified_mcp_panel.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/tldw_api/mcp_unified_client.py tldw_chatbook/tldw_api/mcp_unified_schemas.py tldw_chatbook/MCP/server_unified_service.py tldw_chatbook/MCP/unified_control_plane_service.py tldw_chatbook/UI/MCP_Modules/unified_mcp_panel.py tldw_chatbook/UI/MCP_Modules/unified_mcp_sections.py Docs/Parity/2026-04-21-capability-matrix.md Docs/Parity/2026-04-21-gap-ledger.md Docs/Parity/2026-04-21-execution-roadmap.md Tests/tldw_api/test_mcp_unified_client.py Tests/MCP/test_server_unified_service.py Tests/MCP/test_unified_control_plane_service.py Tests/UI/test_unified_mcp_panel.py Tests/UI/test_tools_settings_window.py
git commit -m "feat: complete unified mcp control plane parity"
```
