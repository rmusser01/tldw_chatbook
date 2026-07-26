# TASK-643 Structural Ownership Enforcement Design

**Status:** Approved direction; pending written-spec review

**Task:** [TASK-643](../../../backlog/tasks/task-643%20-%20Make-runtime-policy-the-sole-application-runtime-source-authority.md)

**Parent design:** [Application Session State Ownership Design](2026-07-26-application-session-state-ownership-design.md)

**Decision:** [ADR-026](../../../backlog/decisions/026-application-session-state-ownership.md)

**Replaces:** The alias-flow implementation inside
`Tests/test_application_state_ownership.py`; it does not replace the runtime
authority, persistence, projection, or compatibility decisions already made
for TASK-643.

**ADR required:** no new ADR

**ADR path:** `backlog/decisions/026-application-session-state-ownership.md`

**Reason:** ADR-026 already requires read-only runtime state, a private backing
store, and read-only app compatibility projections with no independent
writers. This design changes how those existing requirements are enforced.

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

Both backend properties read the same uniquely named private source field.
The server property reads a separate uniquely named private server-ID field.
There are no public setters.

One private method, `_publish_runtime_policy_projection()`, receives an
immutable `RuntimeSourceState` and writes exactly those two private fields.
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

The context continues containing projection callback failures after durable
commit. Logs include only bounded exception-category metadata.

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

### Small structural ownership checks

The custom alias and control-flow engine is removed. Replacement tests verify:

1. `TldwCli` defines all three public projections as properties whose
   `fset` is `None`.
2. Assigning a projection on a minimally constructed real `TldwCli` fails,
   including through a local alias.
3. Only `_publish_runtime_policy_projection()` writes the two unique private
   projection fields.
4. The bootstrap boundary invokes the private publisher when present and uses
   the three public `setattr()` calls only in its explicit lightweight-double
   fallback.
5. `RuntimePolicyContext.state` has no setter; the context has no public
   `persist` or `store` surface; direct assignment fails through an alias.
6. The unique raw and name-mangled context-store names occur only in the
   allowed constructor/commit structure. Constant-string dynamic access
   outside that structure fails the scan.
7. `RuntimeSourceStateStore` definition, import, construction, and qualified or
   constant-`getattr` references remain confined to `source_state.py` and
   `bootstrap.py`.
8. The production tree contains exactly one direct private-store save
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
6. It publishes the new immutable snapshot and advances the revision.
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

- projection assignment through a `TldwCli` alias;
- context-state assignment through a context alias;
- attempted raw and mangled private-store access;
- attempted dynamic private-store access;
- store construction outside its two owner modules;
- the real bootstrap publication path and lightweight-double fallback.

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
