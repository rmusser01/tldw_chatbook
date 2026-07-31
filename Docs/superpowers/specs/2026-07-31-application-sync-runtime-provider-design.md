# Application Sync Runtime Provider Ownership Design

**Date:** 2026-07-31
**Status:** Approved for implementation planning
**Task:** TASK-1601
**Decision:** [ADR-036](../../../backlog/decisions/036-application-service-composition-lifecycle.md)
**Related follow-up:** [TASK-1602](../../../backlog/tasks/task-1602%20-%20Define-production-local-apply-store-ownership-for-manual-Sync-v2.md)
**Reviewed baseline:** `origin/dev` at `ff435772c`

## Purpose

Move only the application-composed Sync graph from a private compatibility
provider to `TldwCli`'s existing `RuntimeServerContextProvider`. Preserve the
exact Sync repository and memory-only dataset-key owners throughout that graph,
without introducing a service container, enabling incomplete local-apply
behavior, or changing the public `from_config(...)` compatibility constructor.

## Verified Current State

### Application Sync owns a private compatibility provider

`TldwCli` creates `server_context_provider` before service composition and
closes its cached client during application unmount. Later,
`_wire_watchlists_and_notifications_services()` constructs Sync through:

```python
ServerSyncService.from_config(
    self.app_config,
    policy_enforcer=self.service_policy_enforcer,
    state_repository=self.sync_state_repository,
)
```

`from_config(...)` creates a `LegacyConfigServerClientProvider`. That provider
retains the startup configuration mapping and may cache a client, but it is not
the provider closed by `TldwCli`. App-composed Sync can therefore follow stale
server configuration and retain a cached client outside the application
shutdown boundary.

The constructor is wrapped in `try/except ValueError`, but
`build_runtime_api_client_provider_from_config(...)` only constructs a lazy
provider. It does not validate or build a client at composition time. The
fallback is not a valid provider-availability contract and would produce a
providerless Sync service if a future factory defect raised `ValueError`.

### The provider-aware Sync factory drops repository ownership

`ServerSyncService.from_server_context_provider(...)` already exists and
retains the supplied provider lazily, but it has no `state_repository`
parameter. Using it as written would silently remove the repository required
by Sync v2 dry-run and transport state operations.

The public `from_config(...)` compatibility factory already accepts and
forwards `state_repository`. The provider-aware factory must offer the same
repository-preservation contract before app composition can move to it.

### An initially empty dataset-key owner is detached

`TldwCli` creates one memory-only `sync_v2_dataset_keys` dictionary and passes
it to both `LocalFirstSyncService` and `ManualSyncControlService`.
`ManualSyncControlService` correctly retains an empty supplied mapping.
`LocalFirstSyncService` instead uses:

```python
self.dataset_keys = dataset_keys or {}
```

An empty application dictionary is falsey, so Local-first Sync replaces it
with a private dictionary. A key loaded later into the application-owned cache
can make manual preview ready while remaining invisible to `sync_once()`.
This is an executable behavior defect, not merely an identity preference.

`SyncKeyRecoveryService` and `SyncRestoreService` contain the same expression,
but neither is composed by `TldwCli`. They are outside this application-graph
tranche.

### Production local apply remains intentionally unavailable

`TldwCli` passes `getattr(self, "sync_v2_local_store", None)` into
`LocalFirstSyncService`, and no production code defines that attribute.
`notes_local_store.py` explicitly describes its in-memory implementation as a
round-trip verification store and defers a ChaChaNotes-backed integration.
Manual Sync therefore remains blocked by its local-apply-store preflight.

Substituting the in-memory store would make the UI appear runnable while
discarding production data ownership, transaction, tombstone, and conflict
requirements. TASK-1602 records the required design work separately.

### Verified inventory and baseline

The reviewed `origin/dev` contains 32 executable
`Server*Service.from_config(...)` calls in `tldw_chatbook/app.py`. The Sync
migration will leave 31. An AST inventory includes `ServerText2SQLService`;
text patterns restricted to alphabetic class names incorrectly report 31.

Before design changes, the focused existing baseline passed:

```text
Tests/Sync_Interop/test_server_sync_service.py
Tests/Sync_Interop/test_local_first_sync_service.py
Tests/ProductionApp/test_service_composition_lifecycle.py
Tests/Packaging/test_installed_distribution.py

72 passed, 1 environment dependency warning
```

## Goals

- Bind app-composed `ServerSyncService` to the exact application-owned
  `server_context_provider`.
- Preserve the exact `sync_state_repository` throughout the app Sync graph.
- Preserve one initially empty, memory-only dataset-key cache throughout the
  app Sync graph so later key mutations are visible to execution.
- Keep Sync client resolution lazy and under the provider's existing runtime
  switch, cache invalidation, and shutdown behavior.
- Keep public `ServerSyncService.from_config(...)` importable and behavior
  compatible.
- Prove the ownership graph in source, the real production `TldwCli`, and an
  offline installed wheel.
- Keep the change small enough for one independently reviewable PR.

## Non-Goals

- Migrating the other 31 app-level compatibility-provider calls.
- Adding a dependency-injection container, service locator, service registry,
  generic close protocol, or new provider-rebinding API.
- Making `_wire_server_context_provider()` reentrant after full application
  construction.
- Enabling manual Sync with an in-memory or otherwise unverified local store.
- Designing the production local apply store, its database transactions, its
  tombstone rules, or its conflict policy; TASK-1602 owns that work.
- Changing uncomposed `SyncKeyRecoveryService` or `SyncRestoreService`.
- Persisting, serializing, logging, or diagnosing dataset keys.
- Changing Sync scheduling, automatic retries, background mutation, or user
  workflow behavior.
- Using a test, surrogate, simplified, or locally redefined Textual
  application in integration coverage.

## Design

### 1. Provider-aware Sync construction

Extend the existing factory without changing its current caller contract:

```python
@classmethod
def from_server_context_provider(
    cls,
    provider: Any,
    *,
    policy_enforcer: Any | None = None,
    state_repository: Any | None = None,
) -> "ServerSyncService":
    return cls(
        client=None,
        client_provider=provider,
        policy_enforcer=policy_enforcer,
        state_repository=state_repository,
    )
```

The changed public factory receives an `Args`/`Returns` docstring. The
repository remains optional because transport-only callers do not require it.
Existing calls that pass only a provider or policy enforcer remain valid.

Compose the application service through:

```python
self.server_sync_service = ServerSyncService.from_server_context_provider(
    self.server_context_provider,
    policy_enforcer=self.service_policy_enforcer,
    state_repository=self.sync_state_repository,
)
```

Remove only the surrounding Sync `try/except ValueError` fallback. The app has
already established both provider and repository owners. Construction remains
lazy and performs no credential lookup or network I/O.

The public `from_config(...)` factory stays unchanged as a compatibility API
for non-application consumers.

### 2. Exact application Sync graph

The production graph must satisfy all of these identities:

```text
app.server_sync_service.client is None
app.server_sync_service.client_provider
    is app.server_context_provider
app.server_sync_service.state_repository
    is app.sync_state_repository

app.sync_scope_service.server_service
    is app.server_sync_service
app.sync_scope_service.state_repository
    is app.sync_state_repository

app.local_first_sync_service.server_service
    is app.server_sync_service
app.local_first_sync_service.state_repository
    is app.sync_state_repository

app.manual_sync_control_service.local_first_sync_service
    is app.local_first_sync_service
app.manual_sync_control_service.state_repository
    is app.sync_state_repository
```

The existing `sync_scope_service`, `local_first_sync_service`, and
`manual_sync_control_service` construction order stays unchanged. The existing
post-construction domain-scope reconciliation loop also stays unchanged; this
task does not declare any wiring helper reentrant.

### 3. Memory-only dataset-key cache identity

Change only `LocalFirstSyncService`'s falsey-default expression:

```python
self.dataset_keys = dataset_keys if dataset_keys is not None else {}
```

This matches the already-documented `ManualSyncControlService` contract and
makes the following identities true even while the cache is empty:

```text
app.local_first_sync_service.dataset_keys
    is app.sync_v2_dataset_keys
app.manual_sync_control_service.dataset_keys
    is app.sync_v2_dataset_keys
```

The cache remains process-memory-only. The task adds no persistence, logging,
diagnostic projection, key generation, or key loading behavior.

### 4. Runtime, failure, and shutdown behavior

`ServerSyncService` continues to keep `client=None`. Each operation calls
`client_provider.build_client()`. The shared provider remains responsible for:

- resolving the current runtime server and credentials;
- returning its current cached client;
- invalidating and closing that client after a server/configuration change;
- closing its final cached client during application unmount.

Sync acquires no close method or resource ownership. Provider, credential,
policy, validation, and transport exceptions continue through their existing
operation paths. No constructor fallback, retry, or exception swallowing is
introduced.

Repository initialization retains its existing production-to-private
in-memory fallback inside `_wire_server_parity_state_repositories()`. This task
does not weaken or duplicate that boundary.

The missing production local apply store remains `None`, so manual Sync keeps
returning its existing explicit blocked result. This provider-ownership change
must not make incomplete local mutation appear available.

### 5. Verification

#### Direct production functions

Update `Tests/Sync_Interop/test_server_sync_service.py` to prove the
provider-aware factory:

- retains the exact provider and repository;
- stays lazy at construction;
- asks the provider for a client on each service resolution;
- keeps no service-local client reference.

Keep the public `from_config(...)` compatibility test and explicitly close the
compatibility provider's cached client after its assertions.

Update `Tests/Sync_Interop/test_local_first_sync_service.py` with an initially
empty shared cache. Add a key after construction and prove the service observes
that mutation through the retained reference. This test exercises the
production class directly; it does not use an application.

#### Source and production application

Extend `Tests/ProductionApp/test_service_composition_lifecycle.py` rather than
adding another broad application lifecycle module. Its AST scan will narrowly
assert that `TldwCli` no longer calls `ServerSyncService.from_config(...)`.
It will not ban other service compatibility calls.

The existing test will continue constructing and mounting the real
`TldwCli`. Its graph assertion will prove every provider, repository, service,
and dataset-key identity listed above before, during, and after the mounted
production-app lifecycle. It will also prove the local apply store remains
unavailable rather than installing a verification store.

Runtime client re-resolution is covered compositionally: the production-app
test proves Sync holds the exact app provider, the direct Sync test proves the
service resolves through its provider without caching, and the existing
runtime-provider suites prove rebind/switch invalidation. No live server or
credential lookup is required.

#### Installed distribution

Extend the existing self-contained `INSTALLED_PROBE` in
`Tests/Packaging/test_installed_distribution.py`. The child process will
construct and exercise the installed production `TldwCli` and assert the same
provider, repository, service, key-cache, and blocked-local-store identities.

The probe remains offline. It will not build a Sync client, access real
credentials, call a server, import checkout helpers, or define another App.
All existing installed-root, asset, immutability, entry-point, and loaded-module
assertions remain intact.

#### Regression matrix

Implementation verification includes:

- the changed direct Sync tests;
- the focused production-app lifecycle module;
- runtime provider and full-app runtime-policy tests;
- all `Tests/ProductionApp/` and `Tests/Packaging/` tests;
- the full repository suite;
- compile, Ruff lint/format, static privacy/surrogate guards, and diff hygiene;
- the installed sdist/wheel probe from a clean temporary installation.

### 6. Documentation and inventory

Amend ADR-036 because it is the existing canonical application service
composition and provider-ownership decision. Add TASK-1601 to its related
tasks and record the Sync provider, repository, key-cache, error, and shutdown
boundaries. Do not create a duplicate ADR.

Update `Docs/Development/server-client-provider-migration-audit.md` to remove
the app-composed Sync compatibility call and explicitly record 31 remaining
executable app-level `Server*Service.from_config(...)` calls after the change.
The public compatibility factory remains an intentional provider-backed API,
not a removal target.

TASK-1602 separately records the verified production local-apply-store design
gap. TASK-1601 must not claim that manual Sync is operational merely because
its provider graph is correct.

## Alternatives Considered

### Construct `ServerSyncService` directly in `TldwCli`

This would allow the app to pass both provider and repository without changing
the factory. It was rejected because the named provider-aware factory already
expresses the project's composition boundary; leaving it unable to preserve a
required Sync dependency would keep a misleading public contract.

### Add a provider argument to `from_config(...)`

This would retain the app's current call shape but give one factory two
competing provider authorities. It was rejected because application
composition should state provider ownership directly while the compatibility
factory remains unambiguous.

### Migrate all remaining app service calls

The services do not share one verified dependency, error, or lifecycle
contract. A 32-call migration would be too broad for one independently
reviewable change and would obscure Sync's repository and key-cache defects.

### Introduce an application service container

ADR-036 already rejects this until the codebase has multiple real composition
roots, coordinated hot replacement, or enough resource owners to justify a
common teardown protocol. This task satisfies none of those triggers.

### Enable manual Sync with `InMemoryNotesStore`

This would make a visible workflow pass preflight while storing mutations only
in process memory and without defining Chat, Media, Workspace, source-cache,
tombstone, or conflict ownership. It was rejected as data-loss-prone and
architecturally false.

### Repair every falsey dataset-key default

The same expression exists in uncomposed recovery and restore services. They
are excluded because this task repairs the verified application graph. Their
contracts should be evaluated when those services receive a production
composition design.

## ADR Check

ADR required: yes

ADR path: `backlog/decisions/036-application-service-composition-lifecycle.md`

Reason: This task changes an application runtime-provider boundary, a public
service factory contract, shared in-memory secret ownership, and shutdown
ownership. ADR-036 already governs the exact composition-root and provider
policy, so it will be amended instead of duplicating the decision.

## Acceptance Mapping

- TASK-1601 AC1: Design sections 1 and 2.
- TASK-1601 AC2: Design sections 2 and 3.
- TASK-1601 AC3: Design sections 1 and 4.
- TASK-1601 AC4: Verified Current State and Design section 4.
- TASK-1601 AC5: Design section 5.
- TASK-1601 AC6: Design section 6 and ADR Check.
- TASK-1601 AC7: Design section 5 regression matrix.
