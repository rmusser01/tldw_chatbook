# Application Service Composition Lifecycle Design

**Date:** 2026-07-28
**Status:** Approved; implementation plan written
**Task:** TASK-1214
**Decision:** [ADR-036](../../../backlog/decisions/036-application-service-composition-lifecycle.md)
**Reviewed baseline:** `origin/dev` at `61960f436`

## Purpose

Correct the verified `TldwCli` service-composition defects without introducing
a service container or claiming to solve application-wide shutdown ownership.
The result must keep the affected service graphs stable for one application
lifetime, attach them to the existing runtime owners, and prove the same
contract from a clean installed wheel.

## Verified Current State

### Duplicate composition

`TldwCli.__init__` contains two direct calls each to:

- `_wire_writing_services()`
- `_wire_chat_conversation_services()`

The commit history shows that each second call arrived through overlapping
feature work. No caller observes or depends on a deliberate retry. The first
Writing call already runs after configuration, policy, and server-context
construction. The first Chat conversation call already runs after the
ChaChaNotes database is resolved.

Writing constructs a new local service, server service, and scope service on
both calls. Chat preserves some citation-local objects on its second call but
still replaces the conversation marks, server, and scope service identities.

### Writing owns the wrong server provider

`_wire_writing_services()` uses
`ServerWritingService.from_config(self.app_config)`. That compatibility path
creates a `LegacyConfigServerClientProvider` with its own cached client and a
startup configuration reference.

The application already owns `server_context_provider`. It follows the active
runtime server context and has the app's only explicit cached-client shutdown
call. Writing exposes `from_server_context_provider(...)`, so no new adapter is
required.

### Initial Sync propagation misses Chat and Media

`_wire_watchlists_and_notifications_services()` creates
`sync_scope_service`. Media's local, server, and scope services are currently
constructed before that helper, so its scope receives no Sync dependency and
is mutated later by the helper's reconciliation loop. Chat is constructed
after the helper but also omits the available dependency, so it remains
unbound. Notes and Research are constructed later and receive Sync.

Final object identity is insufficient evidence for Media's construction
contract because the later loop makes the final identity look correct. The
production and installed sentinels must therefore capture the actual
constructor argument while still constructing the real `TldwCli`.

A focused test separately reinvokes the private server-context provider helper
after application construction to exercise unavailable credential storage.
Production does not do so, and that test does not prove the wider provider
graph is safe to replace. This tranche retains the existing Sync loop and does
not declare provider wiring a supported rewire API.

### Baseline verification

Before design-file changes, the focused existing baseline passed:

```text
pytest
  Tests/ProductionApp/test_reactive_ownership_maturity.py::
    test_production_app_tests_contain_no_surrogate_application_patterns
  Tests/RuntimePolicy/test_runtime_policy_full_app.py::
    test_full_app_wiring_uses_unavailable_store_when_secure_store_is_missing
  Tests/Packaging/test_installed_distribution.py

8 passed, 1 environment dependency warning
```

## Goals

- Compose the Writing and Chat conversation graphs once per `TldwCli`
  construction.
- Preserve one stable local, server, and scope identity for each affected
  graph.
- Put app-composed Writing under the long-lived server-context provider.
- Give Chat and Media the app-owned Sync scope during initial construction.
- Leave the existing post-construction Sync reassignment behavior unchanged.
- Prove source and installed-distribution behavior with the production
  `TldwCli`.
- Ensure the exact ChaChaNotes citation-provenance migration required by that
  installed production path is explicit sdist and wheel runtime data.
- Keep the change small enough for one independently reviewable PR.

## Non-Goals

- Adding a dependency-injection container, service locator, or mutable service
  registry.
- Centralizing every service call in one new mega-composition function.
- Defining a generic close protocol for every service.
- Migrating every remaining `Server*Service.from_config(...)` app call site.
- Removing or redesigning `_wire_server_context_provider()`.
- Making `_wire_server_context_provider()` reentrant or graph-safe after full
  application construction.
- Removing the `TldwCli.query_one()` active-screen fallback.
- Retrying failed service construction.
- Changing public `from_config(...)` compatibility APIs.
- Using a surrogate App, simplified App, unbound `TldwCli` call, or
  `SimpleNamespace` as application integration coverage.

## Design

### 1. Single-pass call graph

Keep the first dependency-ready call for each affected graph:

1. Compose Writing after the server context, local media, Library collection,
   workspace, prompt/Chatbook, and watchlist services have been established.
2. Resolve `chachanotes_db`.
3. Compose Chat conversation services immediately after that database
   resolution.

Delete the later Writing and Chat conversation calls from the subsequent
domain-service sequence. Do not add flags or "already wired" branches inside
either helper.

The narrow source sentinel locates `TldwCli.__init__` in the AST and asserts
exactly one `self._wire_writing_services()` call and one
`self._wire_chat_conversation_services()` call. It does not impose a generic
rule on every `_wire_*` method.

### 2. Runtime provider ownership

Compose `ServerWritingService` with:

```text
ServerWritingService.from_server_context_provider(
    self.server_context_provider,
    policy_enforcer=self.service_policy_enforcer,
)
```

The application does not build a client during startup. Server calls continue
to resolve lazily. The public `from_config(...)` path remains available for
compatibility consumers, but the app no longer uses it for Writing.
The app wiring uses `from_server_context_provider(...)` directly; it does not
retain the unreachable `ValueError` fallback around the compatibility
constructor.

This makes the following identities part of the production contract:

```text
app.server_writing_service.client_provider
    is app.server_context_provider
app.writing_scope_service.server_service
    is app.server_writing_service
app.writing_scope_service.local_service
    is app.local_writing_service
```

### 3. Sync dependency ownership

Keep Media's local and server services in their existing early positions
because watchlist/notification composition configures the local service.
Defer only `MediaReadingScopeService` construction until immediately after
`_wire_watchlists_and_notifications_services()` has created Sync.

Pass the available `self.sync_scope_service` when constructing:

- `MediaReadingScopeService`
- `ChatConversationScopeService`

Initial production identities must satisfy:

```text
app.media_reading_scope_service.sync_scope_service
    is app.sync_scope_service
app.chat_conversation_scope_service.sync_scope_service
    is app.sync_scope_service
```

The existing reassignment loop in
`_wire_watchlists_and_notifications_services()` stays unchanged. It is not the
mechanism used to make initial composition correct. TASK-1214 also does not
claim that reinvoking the separate server-context provider helper preserves
every service-provider identity.

### 4. Failure and shutdown behavior

Local Writing construction keeps its existing guarded failure behavior:
failure logs the existing warning and exposes an unavailable local backend.
Removing the duplicate also removes the accidental second construction
attempt. No automatic retry is introduced.

Chat conversation construction keeps its existing failure behavior. Citation
repository and coordinator ownership are unchanged.

Writing shares `server_context_provider`, whose cached client is already
closed from `TldwCli.on_unmount()`. No new shutdown hook or independently
owned client is added.

### 5. Verification

#### Source and production application

Add a focused module under `Tests/ProductionApp/` that:

- statically asserts the two exact constructor call counts;
- instruments the two real `TldwCli` helper methods while still invoking their
  original implementations;
- instruments the real Chat and Media scope constructors while still invoking
  their original implementations, recording the initial Sync argument;
- constructs and mounts the production `TldwCli`;
- runs through normal unmount;
- proves each helper ran once;
- proves the Writing and Chat local/server/scope identity graph;
- proves Writing uses `server_context_provider`;
- proves Chat and Media use `sync_scope_service`.

The existing production-test surrogate guard must accept the new module
without modification or allowlist growth.

#### Installed distribution

Extend the existing self-contained `INSTALLED_PROBE` rather than creating a
second package probe. Before it calls `get_app()`, the child process
instruments the installed `TldwCli` helpers while invoking the original
methods. After the already-existing installed Home-to-Chat run, it checks:

- one Writing and one Chat composition call;
- the same Writing provider and scope identities;
- the same Chat and Media Sync identities;
- all existing installed-root, resource, immutability, and loaded-module
  assertions.

The child remains outside both the checkout and copied build source and does
not import test helpers.

The first RED execution of this probe exposed an additional prerequisite: the
installed app reached the v26-to-v27 ChaChaNotes migration, but the wheel did
not contain
`tldw_chatbook/DB/migrations/chachanotes_v26_to_v27_citation_provenance.sql`.
That file is read at runtime by `ChaChaNotes_DB.py`; the earlier packaging
contract predated the migration and therefore did not require it. TASK-1214
will add this exact file—not a recursive data catch-all—to the root manifest,
setuptools package-data table, release checker, artifact tests, and packaging
checklist under existing ADR-032.

#### Regression matrix

The implementation plan must include at least:

- the new focused ProductionApp module;
- the ProductionApp no-surrogate guard;
- affected runtime-policy provider tests;
- affected Writing and Chat service tests;
- citation repository, migration, and coordinator identities in the new
  production-app test;
- explicit sdist/wheel and release-checker coverage for the runtime
  citation-provenance migration;
- the installed-distribution suite;
- compile, scoped Ruff lint/format, and `git diff --check`.

No `Tests/UI` surrogate application collection is used as evidence.

## Scope Boundary and Follow-up Ledger

### Remaining legacy config providers

The reviewed `app.py` AST contains 33 executable
`Server*Service.from_config(...)` calls before TASK-1214. Writing accounts for
one, so the expected post-change executable inventory is 32. A separate
docstring also names the RAG-admin compatibility constructor but is not an
executable call. The remaining inventory requires a separate semantic audit
because each service can expose different provider and shutdown behavior.
TASK-1214 will record the post-change count but will not claim those providers
are closed, rebound, or migrated.

### Broader service lifecycle

Generic construction ordering, provider rewiring, resource teardown, and a
possible typed `ApplicationServices` extraction remain separate. The trigger
for such an extraction is concrete evidence of multiple composition roots,
coordinated hot replacement, or a common close protocol—not the size of
`app.py` alone.

### Provider-wiring reentrancy

Production invokes `_wire_server_context_provider()` once during construction.
Its focused post-construction test replaces the provider graph to validate an
unavailable credential-store branch, but it does not exercise every retained
domain service afterward. That helper is separate from the broad
watchlist/notification composition helper that owns Sync. Determining whether
provider wiring should be made reentrant, split into a pure builder, or
replaced by narrow rebinding is separate from TASK-1214.

### Query fallback

The `TldwCli.query_one()` fallback into the active screen remains excluded.
It is a handler/navigation coupling seam and requires its own consumer
inventory and lifecycle design.

## Rollback

No schema, storage, or persistent-data migration is involved. Reverting the
implementation restores the prior composition order. The source and installed
sentinels will then fail on the duplicate calls, making the architectural
regression explicit.

## Acceptance Mapping

| TASK-1214 criterion | Design evidence |
| --- | --- |
| Single-pass stable Writing and Chat graphs | Sections 1 and 5 |
| Writing uses the long-lived provider | Sections 2 and 4 |
| Chat and Media receive Sync initially | Sections 3 and 5 |
| No surrogate app tests | Section 5 and Non-Goals |
| Installed-wheel proof | Section 5 |
| Checks and honest remaining inventory | Regression matrix and follow-up ledger |

## ADR Check

ADR required: yes
ADR path:
`backlog/decisions/036-application-service-composition-lifecycle.md`
Reason: The task changes application service construction, provider ownership,
Sync dependency binding, and the rejected future container boundary.
