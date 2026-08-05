# Application Sync Runtime Provider Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind the application-composed Sync graph to `TldwCli`'s runtime server provider and exact repository/key-cache owners while preserving offline installed behavior and public compatibility APIs.

**Architecture:** `TldwCli` remains the sole composition root and passes its existing `RuntimeServerContextProvider` into `ServerSyncService`; the service resolves clients lazily and never owns a cached client. The existing Sync scope, Local-first, and manual-control services retain exact application-owned repository and memory-only key-cache identities. The shared provider-migration audit is updated by TASK-1601 as this tranche's single audit-integration owner.

**Tech Stack:** Python 3.11+, Textual production `TldwCli`, pytest/pytest-asyncio, Python AST, Ruff, Backlog.md, setuptools wheel/sdist probe.

## Global Constraints

- Work only in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/sync-runtime-provider` on `codex/sync-runtime-provider`.
- Rebase onto the latest `origin/dev` before implementation and again before closeout; re-derive the provider inventory after each rebase.
- Do not define, import, instantiate, or use a test, surrogate, simplified, or locally redefined Textual application. Application integration tests must construct the production `TldwCli`; app-independent behavior must test the production function or class directly.
- Keep `ServerSyncService.from_config(...)` importable and behavior-compatible for non-application consumers.
- Keep `server_sync_service.client is None`; client construction remains lazy through the supplied provider.
- Keep dataset keys process-memory-only. Do not persist, serialize, log, snapshot, or add them to diagnostics.
- Keep the production local apply store unavailable (`None`) in this tranche. Do not substitute `InMemoryNotesStore` or another verification store.
- Do not migrate any non-Sync app-level `Server*Service.from_config(...)` call.
- Do not add a service container, service locator, generic lifecycle manager, reentrant wiring API, retry loop, or background Sync behavior.
- Retain the existing post-construction Sync reconciliation loop unchanged.
- TASK-1601 is the single owner of `Docs/Development/server-client-provider-migration-audit.md` and `Tests/RuntimePolicy/test_server_client_provider_migration_audit.py` for this tranche.
- ADR required: yes.
- ADR path: `backlog/decisions/036-application-service-composition-lifecycle.md`.
- ADR reason: this changes application runtime-provider ownership, a public service factory contract, shared in-memory key ownership, and shutdown ownership under the existing ADR-036 composition policy.

## File Responsibility Map

- `backlog/decisions/036-application-service-composition-lifecycle.md`: canonical decision for the extended Sync provider/repository/key-cache lifecycle.
- `Tests/RuntimePolicy/test_server_client_provider_migration_audit.py`: numeric-safe semantic audit guard and regression for service class names containing digits.
- `Docs/Development/server-client-provider-migration-audit.md`: authoritative semantic inventory, Sync migration delta, numeric-safe scan command, and rebase-derived residual count.
- `tldw_chatbook/Sync_Interop/server_sync_service.py`: provider-aware Sync factory that forwards the optional state repository.
- `Tests/Sync_Interop/test_server_sync_service.py`: direct factory laziness, repository identity, repeated provider resolution, and compatibility-provider cleanup.
- `tldw_chatbook/Sync_Interop/local_first_sync_service.py`: retain an explicitly supplied empty dataset-key cache.
- `Tests/Sync_Interop/test_local_first_sync_service.py`: late key mutation behavior through the retained application cache.
- `tldw_chatbook/app.py`: compose only `ServerSyncService` through the app-owned runtime provider.
- `Tests/ProductionApp/test_service_composition_lifecycle.py`: narrow source sentinel plus real `TldwCli` graph and provider-close lifecycle proof.
- `Tests/Packaging/test_installed_distribution.py`: offline installed-wheel proof of the same production application graph.
- `Docs/superpowers/specs/2026-07-31-application-sync-runtime-provider-design.md`: final implementation status and verified deviations.
- `backlog/tasks/task-1601 - Bind-application-Sync-graph-to-the-runtime-server-context-provider.md`: implementation plan, acceptance status, verification evidence, and closeout notes.

---

### Task 1: Refresh the base and amend ADR-036 before implementation

**Files:**
- Modify: `backlog/decisions/036-application-service-composition-lifecycle.md:1-122`

**Interfaces:**
- Consumes: accepted TASK-1601 design and existing ADR-036 composition-root policy.
- Produces: canonical decision text governing the provider, repository, key-cache, error, and shutdown behavior implemented by later tasks.

- [ ] **Step 1: Rebase the design commits onto current `dev`**

Run:

```bash
git fetch origin dev
git rebase origin/dev
git status --short
```

Expected: rebase succeeds and `git status --short` prints nothing. If `dev` changed any TASK-1601-owned file or Sync call, reconcile the design/task first and keep this plan's stated invariants intact.

- [ ] **Step 2: Re-derive the executable provider inventory and focused baseline**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import ast,pathlib; tree=ast.parse(pathlib.Path('tldw_chatbook/app.py').read_text()); calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='from_config' and isinstance(node.func.value,ast.Name) and node.func.value.id.startswith('Server') and node.func.value.id.endswith('Service')]; sync=sum(node.func.value.id=='ServerSyncService' for node in calls); print({'total':len(calls),'sync':sync,'residual_after_sync':len(calls)-sync}); assert sync==1"
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Sync_Interop/test_server_sync_service.py Tests/Sync_Interop/test_local_first_sync_service.py Tests/ProductionApp/test_service_composition_lifecycle.py Tests/Packaging/test_installed_distribution.py --tb=short
```

Expected on planning base `70f08e5ba`: `{'total': 32, 'sync': 1, 'residual_after_sync': 31}` and `72 passed`. Record a changed total honestly after a later rebase; do not force 32/31 if unrelated migrations have landed.

- [ ] **Step 3: Amend ADR-036 with the approved Sync decision**

Add TASK-1601 to `Related Tasks` and add this decision text after the Writing-provider paragraph:

```markdown
The app-composed `ServerSyncService` also resolves through the application's
long-lived `server_context_provider` while retaining the exact app-owned
`sync_state_repository`. The public `ServerSyncService.from_config(...)`
compatibility constructor remains available outside application composition.
Sync owns no client or close hook; the application provider remains the sole
client-cache invalidation and shutdown owner.

`LocalFirstSyncService` and `ManualSyncControlService` retain the same
process-memory-only dataset-key mapping even when it is initially empty. Keys
are not persisted, serialized, logged, or projected into diagnostics. The
production local apply store remains unavailable until TASK-1602 defines its
data-owner, transaction, tombstone, conflict, and privacy contracts.
```

Extend Context, Benefits, Accepted Trade-offs, and Links with the verified private Sync provider, detached empty cache, residual inventory, TASK-1601 design, and TASK-1602 follow-up. Preserve the existing container trigger and reentrancy exclusions.

Use this Context text:

```markdown
After TASK-1538, application Sync still used
`ServerSyncService.from_config(...)`, creating a private compatibility
provider outside the application's runtime rebind and shutdown boundary. The
provider-aware factory did not forward `sync_state_repository`. In addition,
`LocalFirstSyncService` replaced an explicitly supplied empty dataset-key
mapping, so later key loads could become visible to manual preview but not to
execution.
```

Add these Benefits and Accepted Trade-offs bullets:

```markdown
- App-composed Sync follows the active runtime server and the application's
  existing cached-client shutdown.
- Sync transport, scope, Local-first, and manual control retain one repository
  and one memory-only key-cache owner.

- Manual Sync remains explicitly blocked because no production local apply
  store exists; TASK-1602 owns that design.
- The other AST-verified app-level compatibility-provider calls remain for
  separate semantic migrations, and their count is re-derived after rebases.
```

Add these Links:

```markdown
- [TASK-1601 design](../../Docs/superpowers/specs/2026-07-31-application-sync-runtime-provider-design.md)
- [TASK-1602: Define production local apply-store ownership](../tasks/task-1602%20-%20Define-production-local-apply-store-ownership-for-manual-Sync-v2.md)
```

- [ ] **Step 4: Verify the ADR-only change**

Run:

```bash
git diff --check
git diff -- backlog/decisions/036-application-service-composition-lifecycle.md
```

Expected: no whitespace errors; the diff changes only ADR-036 and contains no claim that manual Sync is operational.

- [ ] **Step 5: Commit the canonical decision**

```bash
git add backlog/decisions/036-application-service-composition-lifecycle.md
git commit -m "docs: extend ADR-036 to the Sync graph"
```

---

### Task 2: Make the provider-migration audit numeric-safe

**Files:**
- Modify: `Tests/RuntimePolicy/test_server_client_provider_migration_audit.py:17-23,233-256`
- Modify: `Docs/Development/server-client-provider-migration-audit.md:18-88`

**Interfaces:**
- Consumes: `_audit_drift(audit_path, source_root, repo_root) -> list[str]` and semantic audit rows.
- Produces: `INDIRECT_BUILDER_RE` that detects `ServerText2SQLService.from_config(...)` and a reconciled current pre-migration audit.

- [ ] **Step 1: Add the failing numeric-service regression**

Add beside the existing unlisted-service test:

```python
def test_audit_guard_rejects_new_unlisted_numeric_server_service_from_config(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    source_root = repo_root / "tldw_chatbook"
    source_root.mkdir(parents=True)
    audit_path = (
        repo_root / "Docs/Development/server-client-provider-migration-audit.md"
    )
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(
        "| Module | Audit lines | Notes |\n| --- | ---: | --- |\n",
        encoding="utf-8",
    )
    source_line = "ServerText2SQLService.from_config(app_config)"
    (source_root / "example.py").write_text(
        f"{source_line}\n",
        encoding="utf-8",
    )

    drift = _audit_drift(
        audit_path=audit_path,
        source_root=source_root,
        repo_root=repo_root,
    )

    assert drift
    assert source_line in drift[0]
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/RuntimePolicy/test_server_client_provider_migration_audit.py::test_audit_guard_rejects_new_unlisted_numeric_server_service_from_config --tb=short
```

Expected: FAIL because `Server[A-Za-z]+Service` does not match `ServerText2SQLService`, so `drift` is empty.

- [ ] **Step 3: Broaden only the service-name matcher**

Change:

```python
r"Server[A-Za-z]+Service\.from_config"
```

to:

```python
r"Server[A-Za-z0-9]+Service\.from_config"
```

Do not broaden the direct builder patterns or accept arbitrary attribute expressions.

- [ ] **Step 4: Reconcile the pre-migration audit**

In the documented direct scan, make the same `[A-Za-z0-9]+` correction. In the app compatibility row, retain the existing Sync semantic match and add the previously invisible current match:

```text
self.server_text2sql_service = ServerText2SQLService.from_config(
```

Run this command and use its reported line numbers only as informational hints:

```bash
rg -n "Server(?:[A-Za-z0-9]+)Service\.from_config" tldw_chatbook/app.py
```

The row must contain all four current semantic matches for Text2SQL, Sync, LLM Provider Catalog, and Audio Services. Do not change unrelated rows.

- [ ] **Step 5: Run the focused and complete audit guards**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/RuntimePolicy/test_server_client_provider_migration_audit.py --tb=short
```

Expected: all audit tests pass, including the numeric-name regression and the live repository drift check.

- [ ] **Step 6: Commit the audit hardening**

```bash
git add Tests/RuntimePolicy/test_server_client_provider_migration_audit.py Docs/Development/server-client-provider-migration-audit.md
git commit -m "test: make provider audit numeric-service safe"
```

---

### Task 3: Preserve repository ownership in the provider-aware Sync factory

**Files:**
- Modify: `Tests/Sync_Interop/test_server_sync_service.py:478-520`
- Modify: `tldw_chatbook/Sync_Interop/server_sync_service.py:53-64`

**Interfaces:**
- Consumes: `ServerSyncService.from_server_context_provider(provider, *, policy_enforcer=None)`.
- Produces: `ServerSyncService.from_server_context_provider(provider, *, policy_enforcer=None, state_repository=None) -> ServerSyncService` with lazy provider and exact repository retention.

- [ ] **Step 1: Extend the lazy-factory test with repository identity**

Change the test body to:

```python
def test_server_sync_service_from_server_context_provider_is_lazy():
    client = object()
    provider = FakeClientProvider(client)
    state_repository = object()
    service = ServerSyncService.from_server_context_provider(
        provider,
        state_repository=state_repository,
    )

    assert isinstance(service, ServerSyncService)
    assert service.client is None
    assert service.client_provider is provider
    assert service.state_repository is state_repository
    assert provider.build_calls == 0
    assert service._require_client() is client
    assert service.client is None
    assert provider.build_calls == 1
```

- [ ] **Step 2: Run the factory test and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Sync_Interop/test_server_sync_service.py::test_server_sync_service_from_server_context_provider_is_lazy --tb=short
```

Expected: FAIL with `TypeError` because `state_repository` is not accepted.

- [ ] **Step 3: Implement the minimal factory contract**

Use this implementation:

```python
@classmethod
def from_server_context_provider(
    cls,
    provider: Any,
    *,
    policy_enforcer: Any | None = None,
    state_repository: Any | None = None,
) -> "ServerSyncService":
    """Build a lazy Sync service over an application server provider.

    Args:
        provider: Runtime provider used to resolve the current server client.
        policy_enforcer: Optional policy boundary for Sync operations.
        state_repository: Optional repository for Sync v2 profile and cursor state.

    Returns:
        A Sync service that retains the provider and repository without building
        or caching a client locally.
    """
    return cls(
        client=None,
        client_provider=provider,
        policy_enforcer=policy_enforcer,
        state_repository=state_repository,
    )
```

- [ ] **Step 4: Make the compatibility test close its provider in `finally`**

Convert the existing compatibility test to async and retain every assertion:

```python
@pytest.mark.asyncio
async def test_server_sync_service_from_config_returns_provider_backed_service():
    service = ServerSyncService.from_config(
        {"tldw_api": {"base_url": "https://example.com", "api_key": "test-key"}}
    )

    try:
        assert isinstance(service, ServerSyncService)
        assert service.client is None
        assert service.client_provider is not None

        client = service.client_provider.build_client()

        assert service.client is None
        assert client.base_url == "https://example.com"
        assert service.client_provider.build_client() is client
    finally:
        await service.client_provider.close_cached_client()
```

- [ ] **Step 5: Run the direct Sync service suite**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Sync_Interop/test_server_sync_service.py --tb=short
```

Expected: all server Sync tests pass; the existing fresh-client test still proves two provider resolutions and no service-local cache.

- [ ] **Step 6: Commit the provider factory contract**

```bash
git add tldw_chatbook/Sync_Interop/server_sync_service.py Tests/Sync_Interop/test_server_sync_service.py
git commit -m "fix(sync): preserve repository in runtime provider factory"
```

---

### Task 4: Retain an initially empty dataset-key cache

**Files:**
- Modify: `Tests/Sync_Interop/test_local_first_sync_service.py:122-220`
- Modify: `tldw_chatbook/Sync_Interop/local_first_sync_service.py:25-36`

**Interfaces:**
- Consumes: `LocalFirstSyncService(..., dataset_keys: dict[str, bytes] | None)`.
- Produces: the exact supplied dictionary identity, including when empty, so later key mutations are visible to `sync_once()`.

- [ ] **Step 1: Add a late-key-mutation behavior test**

Add after `_repo_with_profile(...)`:

```python
async def test_local_first_sync_service_observes_key_added_to_empty_shared_cache(
    tmp_path,
):
    dataset_key = generate_dataset_key()
    repo = _repo_with_profile(tmp_path)
    server = FakeLocalFirstServer()
    shared_dataset_keys: dict[str, bytes] = {}
    service = LocalFirstSyncService(
        server_service=server,
        state_repository=repo,
        local_store=RecordingLocalStore(),
        dataset_keys=shared_dataset_keys,
    )

    assert service.dataset_keys is shared_dataset_keys
    shared_dataset_keys["dataset-1"] = dataset_key

    result = await service.sync_once(
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        domains=["notes"],
    )

    assert result["pulled_envelopes"] == 0
    assert server.calls[0][0] == "pull"
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Sync_Interop/test_local_first_sync_service.py::test_local_first_sync_service_observes_key_added_to_empty_shared_cache --tb=short
```

Expected: FAIL at the identity assertion because the empty dictionary was replaced.

- [ ] **Step 3: Retain only `None` as the default case**

Change:

```python
self.dataset_keys = dataset_keys or {}
```

to:

```python
self.dataset_keys = dataset_keys if dataset_keys is not None else {}
```

Do not modify the uncomposed recovery or restore services in this tranche.

- [ ] **Step 4: Run the Local-first Sync suite**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Sync_Interop/test_local_first_sync_service.py --tb=short
```

Expected: all Local-first tests pass, including the late mutation behavior.

- [ ] **Step 5: Commit the shared key-cache repair**

```bash
git add tldw_chatbook/Sync_Interop/local_first_sync_service.py Tests/Sync_Interop/test_local_first_sync_service.py
git commit -m "fix(sync): retain the application dataset key cache"
```

---

### Task 5: Bind and verify the real application Sync graph

**Files:**
- Modify: `Tests/ProductionApp/test_service_composition_lifecycle.py:23-200`
- Modify: `Tests/Packaging/test_installed_distribution.py:402-557`
- Modify: `tldw_chatbook/app.py:5147-5158`
- Modify: `Docs/Development/server-client-provider-migration-audit.md:52-88`

**Interfaces:**
- Consumes: `ServerSyncService.from_server_context_provider(provider, *, policy_enforcer, state_repository)` from Task 3 and exact cache retention from Task 4.
- Produces: one application Sync graph rooted at `TldwCli.server_context_provider`, plus source/production/installed sentinels and a reconciled residual inventory.

- [ ] **Step 1: Add the narrow source sentinel**

Add this helper and test to the ProductionApp lifecycle module:

```python
def _server_sync_config_factory_calls() -> list[int]:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))
    app_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TldwCli"
    )
    return [
        node.lineno
        for node in ast.walk(app_class)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_config"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ServerSyncService"
        )
    ]


def test_app_composition_does_not_use_sync_config_factory() -> None:
    assert _server_sync_config_factory_calls() == []
```

- [ ] **Step 2: Extend the real production graph and lifecycle assertions**

Add these names to `SERVICE_ATTRIBUTES`:

```python
"server_sync_service",
"local_first_sync_service",
"manual_sync_control_service",
"sync_v2_dataset_keys",
"sync_state_repository",
```

Add these assertions to `_assert_service_graph(app)`:

```python
assert app.server_sync_service.client is None
assert app.server_sync_service.client_provider is app.server_context_provider
assert app.server_sync_service.state_repository is app.sync_state_repository
assert app.sync_scope_service.server_service is app.server_sync_service
assert app.sync_scope_service.state_repository is app.sync_state_repository
assert app.local_first_sync_service.server_service is app.server_sync_service
assert app.local_first_sync_service.state_repository is app.sync_state_repository
assert app.local_first_sync_service.local_store is None
assert app.local_first_sync_service.dataset_keys is app.sync_v2_dataset_keys
assert (
    app.manual_sync_control_service.local_first_sync_service
    is app.local_first_sync_service
)
assert app.manual_sync_control_service.state_repository is app.sync_state_repository
assert app.manual_sync_control_service.dataset_keys is app.sync_v2_dataset_keys
```

Immediately after constructing `app`, instrument the real provider while retaining its implementation:

```python
provider_close_calls = 0
original_close_cached_client = app.server_context_provider.close_cached_client

async def counted_close_cached_client() -> None:
    nonlocal provider_close_calls
    provider_close_calls += 1
    await original_close_cached_client()

monkeypatch.setattr(
    app.server_context_provider,
    "close_cached_client",
    counted_close_cached_client,
)
```

After the `async with app.run_test(...)` block and before best-effort final cleanup, assert:

```python
assert provider_close_calls == 1
```

- [ ] **Step 3: Extend the offline installed-wheel graph**

Add these names to the `INSTALLED_PROBE` `service_identities(app)` tuple:

```python
"server_sync_service",
"local_first_sync_service",
"manual_sync_control_service",
"sync_v2_dataset_keys",
"sync_state_repository",
```

Add these exact installed graph assertions:

```python
assert app.server_sync_service.client is None
assert app.server_sync_service.client_provider is app.server_context_provider
assert app.server_sync_service.state_repository is app.sync_state_repository
assert app.sync_scope_service.server_service is app.server_sync_service
assert app.sync_scope_service.state_repository is app.sync_state_repository
assert app.local_first_sync_service.server_service is app.server_sync_service
assert app.local_first_sync_service.state_repository is app.sync_state_repository
assert app.local_first_sync_service.local_store is None
assert app.local_first_sync_service.dataset_keys is app.sync_v2_dataset_keys
assert (
    app.manual_sync_control_service.local_first_sync_service
    is app.local_first_sync_service
)
assert app.manual_sync_control_service.state_repository is app.sync_state_repository
assert app.manual_sync_control_service.dataset_keys is app.sync_v2_dataset_keys
```

Do not instrument client construction or access credentials in the child process. Keep `get_app()` and the Home-to-Chat flow unchanged.

- [ ] **Step 4: Run source/production/installed tests and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/ProductionApp/test_service_composition_lifecycle.py::test_app_composition_does_not_use_sync_config_factory Tests/ProductionApp/test_service_composition_lifecycle.py::test_production_app_composes_one_stable_dependency_graph --tb=short
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Packaging/test_installed_distribution.py::test_installed_wheel_loaders_entry_points_and_assets_are_immutable --tb=short
```

Expected: FAIL because the app still calls `ServerSyncService.from_config(...)`, retains a private provider, and detaches the initially empty key cache. The installed probe must fail offline on the same identity contract, not on network access.

- [ ] **Step 5: Replace only the app-composed Sync factory block**

Replace the Sync `try/except ValueError` block with:

```python
self.server_sync_service = ServerSyncService.from_server_context_provider(
    self.server_context_provider,
    policy_enforcer=self.service_policy_enforcer,
    state_repository=self.sync_state_repository,
)
```

Leave the scope, Local-first, manual-control, domain reconciliation, runtime service, and every non-Sync compatibility call unchanged.

- [ ] **Step 6: Reconcile the shared audit after the app change**

Remove only the `ServerSyncService.from_config(` semantic match from the app compatibility row. Retain the Text2SQL, LLM Provider Catalog, and Audio Services matches added in Task 2. Refresh their informational line hints from:

```bash
rg -n "Server(?:[A-Za-z0-9]+)Service\.from_config" tldw_chatbook/app.py
```

Add a short application composition inventory note recording the AST-derived residual count. Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import ast,pathlib; tree=ast.parse(pathlib.Path('tldw_chatbook/app.py').read_text()); calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='from_config' and isinstance(node.func.value,ast.Name) and node.func.value.id.startswith('Server') and node.func.value.id.endswith('Service')]; sync=sum(node.func.value.id=='ServerSyncService' for node in calls); print({'total':len(calls),'sync':sync}); assert sync==0"
```

Expected on the planning base: `{'total': 31, 'sync': 0}`. Record a different rebase-derived total honestly if unrelated migrations landed.

On an unchanged residual inventory, add this exact audit statement:

```markdown
After TASK-1601, the AST inventory finds 31 executable app-level
`Server*Service.from_config(...)` calls and no `ServerSyncService.from_config(...)`
call. The count is re-derived after rebases and is not a global constant.
```

If the printed total differs, replace only `31` with the printed total and
record the independently landed migration that changed it.

- [ ] **Step 7: Run the focused integration matrix**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Sync_Interop/test_server_sync_service.py Tests/Sync_Interop/test_local_first_sync_service.py Tests/ProductionApp/test_service_composition_lifecycle.py Tests/Packaging/test_installed_distribution.py Tests/RuntimePolicy/test_server_client_provider_migration_audit.py --tb=short
```

Expected: all focused tests pass, including the installed wheel, numeric audit guard, provider-close observation, exact repository/key identities, and blocked local-store assertion.

- [ ] **Step 8: Commit the application graph**

```bash
git add tldw_chatbook/app.py Tests/ProductionApp/test_service_composition_lifecycle.py Tests/Packaging/test_installed_distribution.py Docs/Development/server-client-provider-migration-audit.md
git commit -m "fix(sync): bind the application graph to runtime context"
```

---

### Task 6: Verify, rebase, document, and close TASK-1601

**Files:**
- Modify: `Docs/superpowers/specs/2026-07-31-application-sync-runtime-provider-design.md`
- Modify: `backlog/tasks/task-1601 - Bind-application-Sync-graph-to-the-runtime-server-context-provider.md`

**Interfaces:**
- Consumes: complete Tasks 1-5 and every TASK-1601 acceptance criterion.
- Produces: rebase-current verified branch, exact implementation evidence, completed Backlog task, and a clean reviewable diff.

- [ ] **Step 1: Run focused runtime and application suites**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Sync_Interop/test_server_sync_service.py Tests/Sync_Interop/test_local_first_sync_service.py Tests/Sync_Interop/test_manual_sync_control.py --tb=short
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/RuntimePolicy/test_server_context_provider.py Tests/RuntimePolicy/test_runtime_policy_full_app.py Tests/RuntimePolicy/test_server_client_provider_migration_audit.py --tb=short
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/ProductionApp Tests/Packaging --tb=short
```

Expected: all focused suites pass with no provider, lifecycle, surrogate-app, or installed-distribution failure.

- [ ] **Step 2: Run static, formatting, and privacy/test-contract guards**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q tldw_chatbook/Sync_Interop/server_sync_service.py tldw_chatbook/Sync_Interop/local_first_sync_service.py tldw_chatbook/app.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/Sync_Interop/server_sync_service.py tldw_chatbook/Sync_Interop/local_first_sync_service.py tldw_chatbook/app.py Tests/Sync_Interop/test_server_sync_service.py Tests/Sync_Interop/test_local_first_sync_service.py Tests/ProductionApp/test_service_composition_lifecycle.py Tests/RuntimePolicy/test_server_client_provider_migration_audit.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check tldw_chatbook/Sync_Interop/server_sync_service.py tldw_chatbook/Sync_Interop/local_first_sync_service.py tldw_chatbook/app.py Tests/Sync_Interop/test_server_sync_service.py Tests/Sync_Interop/test_local_first_sync_service.py Tests/ProductionApp/test_service_composition_lifecycle.py Tests/RuntimePolicy/test_server_client_provider_migration_audit.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/ProductionApp/test_reactive_ownership_maturity.py::test_production_app_tests_contain_no_surrogate_application_patterns --tb=short
git diff --check origin/dev...HEAD
```

Expected: compile, Ruff, surrogate guard, and diff hygiene all pass. No dataset-key value appears in logs, snapshots, diagnostics, docs examples, or test output.

- [ ] **Step 3: Run the full repository suite**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q --tb=short
```

Expected: the complete repository suite passes. Record exact pass/skip counts from this run.

- [ ] **Step 4: Rebase onto the latest `dev` and re-run affected gates**

Run:

```bash
git fetch origin dev
git rebase origin/dev
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import ast,pathlib; tree=ast.parse(pathlib.Path('tldw_chatbook/app.py').read_text()); calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='from_config' and isinstance(node.func.value,ast.Name) and node.func.value.id.startswith('Server') and node.func.value.id.endswith('Service')]; sync=sum(node.func.value.id=='ServerSyncService' for node in calls); print({'total':len(calls),'sync':sync}); assert sync==0"
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Sync_Interop/test_server_sync_service.py Tests/Sync_Interop/test_local_first_sync_service.py Tests/RuntimePolicy/test_server_client_provider_migration_audit.py Tests/ProductionApp Tests/Packaging --tb=short
git diff --check origin/dev...HEAD
```

Expected: rebase succeeds, the Sync count remains zero, the audit doc is updated if the unrelated residual total changed, and every affected test passes on latest `dev`.

- [ ] **Step 5: Self-review the complete branch**

Run:

```bash
git diff --stat origin/dev...HEAD
git diff --check origin/dev...HEAD
git diff --name-only origin/dev...HEAD
git status --short
```

Review every changed hunk against TASK-1601 AC1-AC7. Confirm TASK-1602 remains To Do, no non-Sync service was migrated, no local apply store was installed, no dataset key was persisted/logged, no compatibility API was removed, and `git status --short` is clean before documentation edits.

- [ ] **Step 6: Record exact implementation and verification evidence**

Update the design status to implemented and verified, including the final rebase hash and the actual AST residual count. Use Backlog CLI to set concise Implementation Notes containing:

- the provider-aware factory and app composition changes;
- the exact repository and key-cache identity repair;
- numeric-safe audit ownership and final residual count;
- the explicit unchanged blocked local-store behavior and TASK-1602 follow-up;
- exact focused, ProductionApp/Packaging, RuntimePolicy, full-suite, Ruff, compile, surrogate-guard, installed-wheel, and diff-check results;
- ADR-036 as the governing decision and any plan deviation.

Then check AC1-AC7 and DoD1-DoD8 with `backlog task edit 1601 --check-ac ... --check-dod ...`. Do not mark the task Done until every recorded command has passed.

- [ ] **Step 7: Mark Done and commit closeout documentation**

After AC1-AC7 and DoD1-DoD8 are checked, run:

```bash
backlog task edit 1601 -s Done --check-dod 9 --plain
git add Docs/superpowers/specs/2026-07-31-application-sync-runtime-provider-design.md 'backlog/tasks/task-1601 - Bind-application-Sync-graph-to-the-runtime-server-context-provider.md'
git commit -m "docs: close TASK-1601 Sync provider ownership"
git status --short
```

Expected: TASK-1601 is Done with all AC and DoD items checked, the closeout commit succeeds, and the worktree is clean.

- [ ] **Step 8: Run the final post-commit smoke gate**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/ProductionApp/test_service_composition_lifecycle.py Tests/Packaging/test_installed_distribution.py Tests/RuntimePolicy/test_server_client_provider_migration_audit.py --tb=short
git diff --check origin/dev...HEAD
git status --short
```

Expected: final production/installed/audit smoke passes, diff hygiene passes, and the worktree remains clean.
