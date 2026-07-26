# TASK-643 Structural Ownership Enforcement Design

**Status:** Approved direction; written-spec review corrections in progress

**Task:** [TASK-643](../../../backlog/tasks/task-643%20-%20Make-runtime-policy-the-sole-application-runtime-source-authority.md)

**Parent design:** [Application Session State Ownership Design](2026-07-26-application-session-state-ownership-design.md)

**Decision:** [ADR-026](../../../backlog/decisions/026-application-session-state-ownership.md)

**Supersedes:** The `_store` and projection-callback naming/enforcement shape in
Task 2, Step 3; the projection/store-enforcement shape in Task 4, Steps 1 and
3; and the Settings runtime-policy reload omitted by the original TASK-643
implementation plan. It also replaces the alias-flow implementation inside
`Tests/test_application_state_ownership.py`. It does not replace the runtime
authority, persistence ordering, or compatibility decisions already made for
TASK-643. The implementation plan must be amended to this shape before
implementation resumes.

**ADR required:** yes

**ADR path:** `backlog/decisions/026-application-session-state-ownership.md`

**Reason:** Existing ADR-026 remains the applicable decision and no new ADR is
needed. Its projection wording is amended to clarify that the legacy app
attributes are getter-only properties backed by the one private publisher.

## Problem

TASK-643 initially enforced ownership with an AST visitor that inferred
whether arbitrary local variables might refer to the app, runtime-policy
context, or runtime store. Review-driven additions expanded that visitor into
a partial Python flow analyzer covering aliases, branches, loops, exception
paths, comprehensions, assignment expressions, and lexical scopes.

The resulting test is disproportionate and still unsound. Standard language
features such as tuple destructuring, comprehension assignment expressions,
and definition bindings can produce both false negatives and false positives.
Adding more Python interpretation would make the test suite itself a new
architectural risk.

The ownership boundary should instead be enforced by the runtime objects'
public interfaces. Static tests should verify small, uniquely named structural
facts rather than attempt to infer object identity through arbitrary Python
control flow.

## Constraints

- `RuntimePolicyContext` remains the only mutable runtime-source authority.
- One `RuntimePolicyContext` identity is installed for the `TldwCli` lifetime;
  configuration changes rebind that context instead of replacing it.
- Runtime-source commits remain owner-thread-affine, revisioned, and
  persist-before-publish.
- `current_runtime_backend`, `runtime_backend`, and `active_server_id` remain
  readable compatibility attributes on `TldwCli`.
- Those compatibility attributes are projections, never independent
  authorities.
- Existing lightweight bootstrap test doubles may continue receiving the
  three projected attributes.
- `AppState` and the other legacy state values remain importable serialization
  containers; `TldwCli` remains independent of them.
- No new root application-state object, persisted format, migration, or
  dependency is introduced.
- Runtime-policy storage continues using the effective config path and the
  ADR-022 private-file boundary.

## Considered Approaches

### 1. Runtime-enforced read-only surfaces with exact structural guards

Make the public projection attributes read-only properties on `TldwCli`, back
them with one private immutable tuple, and update that tuple through one
private publisher method. Give the context store and projection callback
unique name-mangled attributes. Make runtime-policy installation one-shot.
Replace alias analysis with exact AST and descriptor checks.

This is the selected approach. It makes aliases irrelevant: an alias to the
real app or context still reaches the same read-only descriptor.

### 2. Continue extending the custom AST flow analyzer

This preserves the current production surface but requires increasingly
complete modeling of Python binding and control-flow semantics. Review has
already demonstrated repeated bypasses and false positives. This approach is
rejected as a maintenance and correctness risk.

### 3. Keep only a shallow current-tree inventory check

A small AST allowlist could confirm that today's tree has no extra writers,
but future alias-based writes could pass while the test remained green. This
would knowingly weaken the verified ownership claim and is rejected.

## Selected Design

### Read-only TldwCli projections

`TldwCli` exposes three getter-only properties:

- `current_runtime_backend`
- `runtime_backend`
- `active_server_id`

All three properties read one
`_runtime_policy_projection_snapshot: tuple[str, str | None]`. Both backend
properties return item `0`; the server property returns item `1`. There are no
public setters.

One private method, `_publish_runtime_policy_projection()`, receives an
immutable `RuntimeSourceState` and atomically assigns one derived
`(active_source, active_server_id)` tuple. The tuple is compatibility data, not
a second state authority or a new root state object, and does not accept
independent mutation.

The getters return safe startup defaults from `("local", None)` if publication
has not happened. Normal application construction installs and publishes
`RuntimePolicyContext` before consumers need the projections. A single tuple
assignment prevents readers from observing a new source paired with an old
server ID and prevents a partially updated projection if publication fails.

Writing any of the three public attributes on a real `TldwCli`, through any
alias, raises `AttributeError` because the properties are data descriptors
without setters.

### Bootstrap publication

`_apply_runtime_policy_to_app()` remains the sole projection boundary.

For a real `TldwCli`, it invokes the private projection publisher. For a
lightweight app double that does not expose the publisher, it retains the
existing three `setattr()` operations. This preserves focused bootstrap tests
and helper compatibility without weakening the production app surface.

The fallback is selected only when the publisher attribute is statically
absent. `_apply_runtime_policy_to_app()` uses `inspect.getattr_static()` with a
private sentinel for the presence check, then ordinary `getattr()` to bind a
present publisher. A descriptor binding error propagates; it is not
misclassified as absence. A present but non-callable publisher is an error.
For commit-triggered publication, a publisher exception reaches
`RuntimePolicyContext.commit_state()`'s existing projection-failure
containment; bootstrap must not retry through public `setattr()`. Logs include
only bounded exception-category metadata.

When an already-loaded state needs no synchronization commit,
`load_runtime_policy_for_app()` performs the one permitted initial direct
projection. An exception from that initial publication propagates and aborts
bootstrap; it is not contained as a successful commit and does not trigger the
lightweight-double fallback.

The symbol `_publish_runtime_policy_projection` may occur in production only
as its `TldwCli` method definition and as the direct or constant-string lookup
and invocation inside `_apply_runtime_policy_to_app()`. Static checks reject
direct calls, captured callbacks, and constant-string dynamic references
elsewhere. Publication therefore occurs only after a durable context commit or
from the one initial projection of an already-loaded authoritative state.

The symbol `_apply_runtime_policy_to_app` may occur in production only as:

- its definition;
- the contained projection callback registered by
  `load_runtime_policy_for_app()`; and
- the single initial projection of an already-loaded authoritative state in
  that same loader.

Exact direct, captured, qualified, and constant-string dynamic references are
rejected elsewhere.

### One-time context installation and Settings rebind

`load_runtime_policy_for_app()` is a one-time installation boundary. Before
loading a store or publishing a projection, it checks for an installed
`RuntimePolicyContext` and raises a bounded `RuntimeError` if one exists.
`ensure_runtime_policy_for_app()` remains the idempotent accessor: it returns
the installed context or invokes the loader only when none exists.

The loader constructs the context locally and installs it on
`app.runtime_policy` only after any required synchronization commit and initial
projection have succeeded. A store/load/synchronization failure, or a failure
from the direct initial projection path, leaves no context installed and
permits a clean retry. A projection-callback exception after a successful
synchronization commit remains contained by `commit_state()`; the durable
context is installed because that commit succeeded. No await or externally
reentrant step separates successful preparation from installation.
`TldwCli.__init__` invokes the installing loader as a standalone call rather
than assigning its return value, so the prepared context is attached exactly
once.

Production calls to the loader are confined to the `TldwCli` constructor and
the internal `ensure_runtime_policy_for_app()` fallback. Loader-symbol
references are confined to its definition, the internal fallback call, the
`app.py` import, and the constructor call. Exact structural checks reject
direct, captured, qualified, imported, and constant-string dynamic references
elsewhere.

Settings no longer calls the loader after saving a new server configuration.
It writes `base_url` and `auth_token` through one
`save_settings_to_cli_config()` batch and checks the result, avoiding the
current partial two-write update. If that save fails, it does not reload,
rebind, or report success. After a successful save, it reloads the saved
configuration into a local `refreshed_config` value and passes that value to
`handle_runtime_backend_changed()` as an explicit configuration override. It
does not assign `app.app_config` first.

`handle_runtime_backend_changed()` is the single app-level rebind
coordinator. Before mutating the app, provider, target store, or client cache,
it asks `set_authoritative_runtime_source()` to derive and validate the
candidate binding from the explicit configuration override and durably commit
that candidate through the already-installed context. The helper accepts the
override only for candidate derivation; it does not install the mapping on the
app or provider. A `False` compare-and-swap result is a failed commit, not a
successful return of whichever newer snapshot happens to exist.

If derivation, compare-and-swap, or persistence fails, the coordinator returns
`False`. It leaves the context state and revision, projection tuple,
`app.app_config`, provider configuration, configured-target store/default,
cached client, and active screen unchanged. Settings emits no success notice
and performs no Sync v2 preparation on that path. The already-successful
Settings file update remains on disk for an explicit retry or the next
startup; the coordinator does not claim cross-file transactional rollback.

After a successful commit, and before any awaited screen callback, the
coordinator installs `refreshed_config` on `app.app_config` and calls a focused
`RuntimeServerContextProvider.rebind_app_config()` operation. The provider
first replaces its config mapping and detaches its cached client/key, with
client-close failures contained and logged only by exception category. Using
the coordinator's captured previous server ID and committed next server ID, it
also invokes the existing event/sync server-switch invalidation hooks exactly
once when the identity changed. It then best-effort upserts the derived
legacy-config target. Target-store failure is contained: the newly committed
authority remains usable through the provider's existing config-derived
fallback, and a subsequent successful rebind may repair the materialized
target/default. The post-commit provider rebind is therefore non-throwing for
these cleanup/materialization failures.

`handle_runtime_backend_changed()` only notifies the active screen after a
successful coordinated commit/rebind. That callback is a post-commit observer:
its exception is contained with category-only diagnostics and cannot be
reported as an activation rollback. The coordinator returns `True` because the
authority and provider binding are already committed. The local source path
uses the same Boolean success contract without a configuration override.

The app context identity does not change. Long-lived consumers—including
`ServicePolicyEnforcer`, `RuntimeServerContextProvider`,
`ActiveServerCapabilityService`, and Home—continue observing the same
authority object and see the committed rebind.

### Uniquely private context store and projection callback

`RuntimePolicyContext._store` becomes
`RuntimePolicyContext.__runtime_policy_state_store` and remains in
`__slots__`. Python name mangling is not treated as a security boundary; the
unique name exists so structural tests can identify every direct or dynamic
reference precisely.

The attribute is:

- assigned once in `RuntimePolicyContext.__init__`;
- read only as the receiver of the single immediate
  `save(candidate)` statement in `commit_state()`.

No property, accessor, alias, standalone persistence method, or callback
exposes the store. Existing tests retain injected recording stores separately.

The exact structural allowlist is:

- raw string `__runtime_policy_state_store` once in
  `RuntimePolicyContext.__slots__`;
- one raw attribute assignment in `RuntimePolicyContext.__init__`;
- one raw attribute load as the receiver of the direct, immediate
  `save(candidate)` expression in synchronous, non-generator `commit_state()`.

The mangled spelling
`_RuntimePolicyContext__runtime_policy_state_store` must not occur anywhere in
production source. Raw or mangled constant-string dynamic access is forbidden
outside the allowed `__slots__` declaration. Tests inspect source structure;
they do not claim that Python name mangling makes exact mangled runtime access
impossible.

`RuntimePolicyContext._publish` likewise becomes
`RuntimePolicyContext.__runtime_policy_projection_callback`. Its exact
structural allowlist is:

- raw string `__runtime_policy_projection_callback` once in `__slots__`;
- one raw attribute assignment in `__init__`;
- direct raw attribute loads only for the contained callback invocation in
  `commit_state()`.

The mangled spelling
`_RuntimePolicyContext__runtime_policy_projection_callback` must not occur in
production source. Raw or mangled constant-string dynamic access is forbidden.
This prevents callers from publishing app projections without a context
commit.

### Small structural ownership checks

Delete the custom alias, lexical-scope, and control-flow visitors rather than
retaining them beside the replacement. The new checks use small stateless AST
collectors and direct shape assertions. They inspect only name-bearing syntax:
`Name`, `Attribute`, import aliases, and constant names passed to
`getattr`/`setattr`/`delattr` or equivalent subscript lookup. They do not scan
arbitrary strings or docstrings.

Replacement tests verify:

1. `TldwCli` defines all three public projections as properties whose
   `fset` is `None`.
2. Assigning a projection on a minimally constructed real `TldwCli` fails,
   including through a local alias.
3. `_runtime_policy_projection_snapshot` is read only by the three getters and
   assigned once per publication only by
   `_publish_runtime_policy_projection()`. Direct, captured, qualified,
   imported, and constant-string dynamic references elsewhere are rejected.
4. `_apply_runtime_policy_to_app` occurs only at its definition, the contained
   callback registration, and the one initial loaded-state projection in
   `load_runtime_policy_for_app()`. Direct, captured, qualified, and
   constant-string dynamic references elsewhere are rejected.
5. `_publish_runtime_policy_projection` occurs only at its method definition
   and at the publisher lookup/invocation inside
   `_apply_runtime_policy_to_app()`. Direct and constant-string dynamic
   references elsewhere are rejected.
6. The bootstrap boundary uses the three public `setattr()` calls only in its
   explicit statically publisher-absent lightweight-double fallback.
   Descriptor binding failures, non-callable publishers, and publisher errors
   never enter the fallback.
7. `RuntimePolicyContext.state` has no setter; the context has no public
   `persist` or `store` surface; direct assignment fails through an alias.
8. The raw context-store name occurs only in the exact `__slots__`,
   constructor-assignment, and immediate-commit structures above. The mangled
   name occurs nowhere. Raw or mangled constant-string dynamic access is
   rejected.
9. `RuntimeSourceStateStore` definition, import, construction, and qualified or
   constant-`getattr` references remain confined to `source_state.py` and
   `bootstrap.py`.
10. The production tree contains exactly one direct private-store save
    statement, in synchronous, non-generator `commit_state()`.
11. The raw projection-callback name occurs only in its exact `__slots__`,
    constructor-assignment, and contained `commit_state()` structures. The
    mangled name occurs nowhere, and raw or mangled constant-string dynamic
    access is rejected.
12. `load_runtime_policy_for_app` production references are exactly its
    definition, the internal ensure fallback call, the `app.py` import, and the
    `TldwCli` constructor call. Direct, captured, qualified, imported, and
    constant-string dynamic references elsewhere are rejected.
13. The old alias/scope/control-flow analyzer classes are absent.

These checks depend on unique symbols and descriptor behavior, not variable
spelling, alias propagation, or control-flow interpretation.

## Data Flow

1. `load_runtime_policy_for_app()` loads and verifies the durable state.
2. It creates `RuntimePolicyContext` with the private store and a contained
   publication callback.
3. It durably synchronizes and initially projects that context when required,
   then installs the successfully prepared context on the app.
4. A higher-level runtime operation captures `(state, revision)` and derives a
   candidate.
5. `commit_state()` verifies the owner thread and revision.
6. It persists through the unique private store reference.
7. It publishes the new immutable snapshot and advances the revision before
   invoking the projection callback, so callback readers observe the committed
   `(state, revision)`.
8. The callback reaches `_apply_runtime_policy_to_app()`.
9. A real `TldwCli` atomically replaces its private projection tuple through
   the one private publisher; the three public properties immediately reflect
   one coherent pair.

When Settings changes the configured server, it loads a candidate
configuration without publishing it, derives and durably commits the new
binding through the already-installed context, then installs the same
configuration on the app and provider before notifying the active screen. It
does not repeat steps 1–2.

Persistence failure leaves the context snapshot, revision, private projection
tuple, app and provider configuration, configured target/default, cached
client, active screen, and target-status side effects unchanged.

## Compatibility

- Reads of all three app projection names remain unchanged.
- External or internal writes to those names now fail intentionally; ADR-026
  promises read compatibility, not write compatibility.
- Lightweight objects passed directly to bootstrap helpers retain their
  projected attributes through the explicit fallback.
- Repeated loader calls do not replace a live context; callers use
  `ensure_runtime_policy_for_app()` when idempotent access is required.
- Legacy state imports and `to_dict()`/`from_dict()` behavior remain unchanged.
- Runtime-policy JSON path, schema, privacy posture, and migration behavior are
  unchanged.

## Verification

The implementation must demonstrate red-to-green tests for:

- a minimally constructed real `TldwCli` reading
  `("local", "local", None)` before publication;
- projection assignment through a `TldwCli` alias;
- real publication mapping one source to both backend getters and mapping the
  active server ID through one tuple assignment;
- the projection callback observing the newly committed state and revision;
- context-state assignment through a context alias;
- structural raw and mangled private-store references, including
  constant-string dynamic access, without claiming mangled runtime access is
  inaccessible;
- store construction outside its two owner modules;
- the real bootstrap publication path and publisher-absent lightweight-double
  fallback;
- a publisher descriptor whose binding raises `AttributeError` being treated
  as present and failing without fallback;
- a present non-callable publisher failing without fallback;
- a throwing publisher being contained after durable commit without public
  `setattr()` fallback;
- a throwing publisher during the direct initial loaded-state projection
  propagating and aborting bootstrap without public `setattr()` fallback;
- a failed initial load, synchronization persistence, or direct projection
  leaving no context installed and allowing a clean retry;
- a synchronized durable commit whose contained projection callback fails
  still installing the committed context exactly once;
- `TldwCli` construction invoking the installing loader without a duplicate
  assignment of its return value;
- persistence failure leaving a previously published real projection
  unchanged;
- a second loader call rejecting before store I/O or publication and retaining
  the original context identity;
- a Settings server rebind preserving context identity while the enforcer,
  server-context provider, capability service, and Home observe the committed
  state;
- a Settings rebind whose runtime-store failure retains the old app/provider
  config, cache, target/default, context, projection, and active screen;
- a forced compare-and-swap rejection following the same no-side-effect
  failure path;
- a failed batched Settings save causing no reload, rebind, partial
  base-URL/token activation, or success notification;
- the provider rebind refreshing its config mapping/default legacy target and
  invalidating its cached client and changed-server hooks only after the
  authoritative commit, without duplicate hook invocation;
- a legacy-target upsert failure leaving the new committed authority usable
  through the provider's refreshed-config fallback;
- a throwing active-screen callback occurring only after commit, remaining
  bounded, and not converting a committed activation into a `False` result;
- direct, mangled, and constant-string access to the private projection
  callback failing structural checks;
- deletion of the old alias/scope/control-flow visitors.

It must then pass:

- `Tests/RuntimePolicy`;
- the simplified application-state ownership suite;
- Media Ingest and Study runtime-change tests;
- Settings runtime-source switch and server-context-provider rebind tests;
- affected app/bootstrap caller tests;
- ADR-022 private-writer/store tests;
- scoped Ruff, formatter, compilation, and diff checks.

TASK-643 remains In Progress with unchecked acceptance criteria until TASK-646
runs the shared installed-wheel, product-maturity, static, and full-suite
release gates.
