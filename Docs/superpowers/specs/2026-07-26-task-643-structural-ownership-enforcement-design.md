# TASK-643 Structural Ownership Enforcement Design

**Status:** Approved direction; written-spec review corrections in progress

**Task:** [TASK-643](../../../backlog/tasks/task-643%20-%20Make-runtime-policy-the-sole-application-runtime-source-authority.md)

**Parent design:** [Application Session State Ownership Design](2026-07-26-application-session-state-ownership-design.md)

**Decision:** [ADR-026](../../../backlog/decisions/026-application-session-state-ownership.md)

**Supersedes:** The projection/store-enforcement shape in Task 4, Steps 1 and
3 of the original TASK-643 implementation plan, together with the alias-flow
implementation inside `Tests/test_application_state_ownership.py`. It does not
replace the runtime authority, persistence ordering, or compatibility
decisions already made for TASK-643. The implementation plan must be amended
to this shape before implementation resumes.

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

Make the public projection attributes read-only properties on `TldwCli`, give
their backing fields unique private names, and update them through one private
publisher method. Give the context store one unique name-mangled attribute.
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

Both backend properties read `_runtime_policy_projection_source`. The server
property reads `_runtime_policy_projection_active_server_id`. There are no
public setters.

One private method, `_publish_runtime_policy_projection()`, receives an
immutable `RuntimeSourceState` and assigns exactly those two private fields.
The fields are derived compatibility data; they are not a second state
container and do not accept independent mutation.

The getters return safe startup defaults if publication has not happened:
`"local"` for the source and `None` for the server ID. Normal application
construction installs and publishes `RuntimePolicyContext` before consumers
need the projections.

Writing any of the three public attributes on a real `TldwCli`, through any
alias, raises `AttributeError` because the properties are data descriptors
without setters.

### Bootstrap publication

`_apply_runtime_policy_to_app()` remains the sole projection boundary.

For a real `TldwCli`, it invokes the private projection publisher. For a
lightweight app double that does not expose the publisher, it retains the
existing three `setattr()` operations. This preserves focused bootstrap tests
and helper compatibility without weakening the production app surface.

The fallback is selected only when the publisher attribute is absent. A
present but non-callable publisher is an error. If a publisher call raises,
the exception propagates to `RuntimePolicyContext.commit_state()`'s existing
projection-failure containment; bootstrap must not retry through public
`setattr()`. Logs include only bounded exception-category metadata.

The symbol `_publish_runtime_policy_projection` may occur in production only
as its `TldwCli` method definition and as the direct or constant-string lookup
and invocation inside `_apply_runtime_policy_to_app()`. Static checks reject
direct calls, captured callbacks, and constant-string dynamic references
elsewhere, so no production caller can publish projections without a durable
context commit.

### Uniquely private context store

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

### Small structural ownership checks

The custom alias and control-flow engine is removed. Replacement tests verify:

1. `TldwCli` defines all three public projections as properties whose
   `fset` is `None`.
2. Assigning a projection on a minimally constructed real `TldwCli` fails,
   including through a local alias.
3. `_runtime_policy_projection_source` is read only by the two backend
   getters, `_runtime_policy_projection_active_server_id` is read only by its
   getter, and both are assigned only by
   `_publish_runtime_policy_projection()`. Direct and constant-string
   `getattr`/`setattr`/`delattr` references elsewhere are rejected.
4. `_publish_runtime_policy_projection` occurs only at its method definition
   and at the publisher lookup/invocation inside
   `_apply_runtime_policy_to_app()`. Direct and constant-string dynamic
   references elsewhere are rejected.
5. The bootstrap boundary uses the three public `setattr()` calls only in its
   explicit publisher-absent lightweight-double fallback. Publisher errors
   propagate to context containment and never enter the fallback.
6. `RuntimePolicyContext.state` has no setter; the context has no public
   `persist` or `store` surface; direct assignment fails through an alias.
7. The raw context-store name occurs only in the exact `__slots__`,
   constructor-assignment, and immediate-commit structures above. The mangled
   name occurs nowhere. Raw or mangled constant-string dynamic access is
   rejected.
8. `RuntimeSourceStateStore` definition, import, construction, and qualified or
   constant-`getattr` references remain confined to `source_state.py` and
   `bootstrap.py`.
9. The production tree contains exactly one direct private-store save
   statement, in synchronous, non-generator `commit_state()`.

These checks depend on unique symbols and descriptor behavior, not variable
spelling, alias propagation, or control-flow interpretation.

## Data Flow

1. `load_runtime_policy_for_app()` loads and verifies the durable state.
2. It creates `RuntimePolicyContext` with the private store and a contained
   publication callback.
3. A higher-level runtime operation captures `(state, revision)` and derives a
   candidate.
4. `commit_state()` verifies the owner thread and revision.
5. It persists through the unique private store reference.
6. It publishes the new immutable snapshot and advances the revision before
   invoking the projection callback, so callback readers observe the committed
   `(state, revision)`.
7. The callback reaches `_apply_runtime_policy_to_app()`.
8. A real `TldwCli` updates its two private projection fields through the one
   private publisher; the three public properties immediately reflect them.

Persistence failure leaves the context snapshot, revision, private projection
fields, active screen, and target-status side effects unchanged.

## Compatibility

- Reads of all three app projection names remain unchanged.
- External or internal writes to those names now fail intentionally; ADR-026
  promises read compatibility, not write compatibility.
- Lightweight objects passed directly to bootstrap helpers retain their
  projected attributes through the explicit fallback.
- Legacy state imports and `to_dict()`/`from_dict()` behavior remain unchanged.
- Runtime-policy JSON path, schema, privacy posture, and migration behavior are
  unchanged.

## Verification

The implementation must demonstrate red-to-green tests for:

- a minimally constructed real `TldwCli` reading
  `("local", "local", None)` before publication;
- projection assignment through a `TldwCli` alias;
- real publication mapping one source to both backend getters and mapping the
  active server ID;
- the projection callback observing the newly committed state and revision;
- context-state assignment through a context alias;
- structural raw and mangled private-store references, including
  constant-string dynamic access, without claiming mangled runtime access is
  inaccessible;
- store construction outside its two owner modules;
- the real bootstrap publication path and publisher-absent lightweight-double
  fallback;
- a throwing publisher being contained after durable commit without public
  `setattr()` fallback;
- persistence failure leaving a previously published real projection
  unchanged.

It must then pass:

- `Tests/RuntimePolicy`;
- the simplified application-state ownership suite;
- Media Ingest and Study runtime-change tests;
- affected app/bootstrap caller tests;
- ADR-022 private-writer/store tests;
- scoped Ruff, formatter, compilation, and diff checks.

TASK-643 remains In Progress with unchecked acceptance criteria until TASK-646
runs the shared installed-wheel, product-maturity, static, and full-suite
release gates.
