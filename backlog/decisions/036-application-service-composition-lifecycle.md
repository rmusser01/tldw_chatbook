# ADR-036: Application Service Composition Lifecycle

Status: Accepted
Date: 2026-07-28
Related Tasks:
[TASK-1538](../tasks/task-1538%20-%20Enforce-single-pass-service-composition-and-runtime-dependency-binding.md)
[TASK-1601](../tasks/task-1601%20-%20Bind-application-Sync-graph-to-the-runtime-server-context-provider.md)

## Decision

`TldwCli` remains the application composition root. The Writing and Chat
conversation service graphs are composed exactly once during `TldwCli`
construction, at the first call site where each graph's dependencies are
available. Their `_wire_*` helpers are construction helpers, not implicit
retry or runtime-rebinding APIs.

The app-composed `ServerWritingService` resolves clients through the
application's long-lived `server_context_provider`. It does not own a
`LegacyConfigServerClientProvider`. This keeps Writing on the same runtime
server selection, credential, client-cache, invalidation, and shutdown owner
as the other provider-backed application services. The public
`ServerWritingService.from_config(...)` compatibility constructor remains
available outside application composition.

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

Scope services receive dependencies that already exist at their construction
boundary. `ChatConversationScopeService` and `MediaReadingScopeService`
therefore receive the current app-owned `sync_scope_service` directly when
they are composed. Media's local and server services remain early because
watchlist/notification composition configures the local service, while only
the Media scope construction moves after that helper has established Sync.
The existing reconciliation loop inside
`_wire_watchlists_and_notifications_services()` is retained unchanged.
Initial composition does not rely on that loop.

This decision does not introduce a dependency-injection container, mutable
service registry, or generic lifecycle manager. A future extraction may use a
typed immutable `ApplicationServices` bundle only after the codebase has more
than one real composition root, requires coordinated hot replacement, or has
enough resource-owning services to justify a shared teardown protocol.

The single-pass regression guard is intentionally narrow. Source and runtime
sentinels cover the two verified duplicate helpers rather than forbidding
every future syntactic use of a `_wire_*` name.

## Context

`TldwCli.__init__` currently calls `_wire_writing_services()` twice and
`_wire_chat_conversation_services()` twice. The call history shows that these
duplicates were introduced by overlapping feature changes, not by a designed
retry contract. The later calls replace scope, server, and auxiliary service
identities. Writing also repeats local schema initialization.

Writing is currently composed through `ServerWritingService.from_config(...)`.
That path creates a private `LegacyConfigServerClientProvider`, which can cache
an HTTP client. Application shutdown closes only
`TldwCli.server_context_provider`, so a Writing client obtained from the
private provider is outside the app's established shutdown boundary. The
private provider also retains the configuration mapping captured during
startup instead of following the active runtime server context.

After TASK-1538, application Sync still used
`ServerSyncService.from_config(...)`, creating a private compatibility
provider outside the application's runtime rebind and shutdown boundary. The
provider-aware factory did not forward `sync_state_repository`. In addition,
`LocalFirstSyncService` replaced an explicitly supplied empty dataset-key
mapping, so later key loads could become visible to manual preview but not to
execution.

`_wire_watchlists_and_notifications_services()` creates `sync_scope_service`.
Before this decision, the Media scope is constructed first without Sync; the
helper's later reconciliation loop mutates Media to attach the newly created
scope. Chat is constructed after the helper but also omits Sync, so it remains
unbound. Notes and Research are constructed later and receive the available
Sync scope. Both Chat and Media expose sync operations that reject a missing
scope, and final identity alone cannot prove initial injection because the
legacy loop repairs Media after construction.

A focused test separately invokes the private server-context provider helper
after construction to exercise an unavailable credential-store branch. That
second invocation is not a production call path and does not prove the wider
provider graph is safe to replace. TASK-1538 does not define or repair a
reentrant provider-wiring contract.

There are additional app call sites that use `Server*Service.from_config(...)`.
TASK-1538 records that inventory as separate follow-up work. Correcting every
provider and shutdown contract in the same change would turn a verified
single-pass repair into an application-wide lifecycle migration.

The executable AST inventory contains 32 application-level
`Server*Service.from_config(...)` calls: one Sync call and 31 residual calls
after this narrow Sync migration. The residual inventory remains subject to
separate semantic migrations.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep both calls and make each helper idempotent | It hides a defective call graph, complicates partial-failure behavior, and risks retaining stale dependencies. |
| Add a mutable central service container | `TldwCli` already supplies the one current composition root. A service locator would add indirection without a second composition root or a uniform lifecycle protocol. |
| Consolidate every service call behind one new mega-helper | It would move and retest unrelated service graphs without being necessary to correct the verified defects. |
| Keep Writing on `from_config(...)` | It preserves a separate cached-client and configuration owner outside application rebind and shutdown ownership. |
| Rely only on late Sync reconciliation | It mutates Media after construction and never reaches Chat during normal startup, so the initial-injection contract remains broken. |
| Delete the reconciliation loop | A focused test observes it after explicit private-helper reinvocation. Deleting it would change behavior before the broader reentrancy contract is audited. |
| Enforce one syntactic call for every `_wire_*` helper | It is broader than the verified defect and could reject valid mutually exclusive construction code. |

## Consequences

### Benefits

- Writing and Chat conversation service identities remain stable after
  construction.
- Writing follows the active server-context provider and its existing cached
  client shutdown.
- App-composed Sync follows the active runtime server and the application's
  existing cached-client shutdown.
- Chat and Media Sync operations start with the real application Sync owner.
- Sync transport, scope, Local-first, and manual control retain one repository
  and one memory-only key-cache owner.
- Regression coverage remains focused on the verified call sites and the real
  production app.
- The repair does not add a service locator or speculative lifecycle
  framework.

### Accepted Trade-offs

- `TldwCli` remains a large composition root.
- Existing post-construction Sync reassignment remains in place, although
  correct initial composition no longer depends on it.
- Reinvoking `_wire_server_context_provider()` after full construction is not
  declared graph-safe by this decision.
- Other `Server*Service.from_config(...)` application call sites remain for a
  separate audited migration.
- Manual Sync remains explicitly blocked because no production local apply
  store exists; TASK-1602 owns that design.
- The other AST-verified app-level compatibility-provider calls remain for
  separate semantic migrations, and their count is re-derived after rebases.
- Writing local-service failure remains represented by an unavailable local
  backend; the accidental second construction attempt is removed.

## Links

- [TASK-1538 design](../../Docs/superpowers/specs/2026-07-28-application-service-composition-lifecycle-design.md)
- [TASK-1601 design](../../Docs/superpowers/specs/2026-07-31-application-sync-runtime-provider-design.md)
- [TASK-1602: Define production local apply-store ownership](../tasks/task-1602%20-%20Define-production-local-apply-store-ownership-for-manual-Sync-v2.md)
- [ADR-033: Application Session State Ownership](033-application-session-state-ownership.md)
- [ADR-032: Immutable Installed Distribution Assets](032-immutable-installed-distribution-assets.md)
