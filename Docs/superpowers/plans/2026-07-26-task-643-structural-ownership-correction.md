# TASK-643 Structural Ownership Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish TASK-643 with runtime-enforced ownership, failure-atomic bootstrap and Settings rebinds, and verified full-application integration—without partial or simplified application fixtures.

**Architecture:** `RuntimePolicyContext` remains the sole live runtime-source authority. App-independent preparation, context, provider, and derivation behavior is tested directly; application behavior is tested only with a fully constructed `TldwCli`, using its real Textual lifecycle when a screen is involved. `TldwCli` publishes one immutable compatibility tuple through getter-only properties, and one coordinator commits a Settings candidate before updating the app/provider.

**Tech Stack:** Python 3.11+, Textual, frozen dataclasses, `threading.get_ident`, Loguru, pytest/pytest-asyncio, Ruff.

**Backlog:** [TASK-643](../../../backlog/tasks/task-643%20-%20Make-runtime-policy-the-sole-application-runtime-source-authority.md)

**Specification:** [TASK-643 Structural Ownership Enforcement Design](../specs/2026-07-26-task-643-structural-ownership-enforcement-design.md)

**Parent specification:** [Application Session State Ownership Design](../specs/2026-07-26-application-session-state-ownership-design.md)

**Original plan:** [Runtime Policy Authority Implementation Plan](2026-07-26-task-643-runtime-policy-authority.md)

**ADR required:** yes

**ADR path:** `backlog/decisions/033-application-session-state-ownership.md`

**Reason:** ADR-033 already defines the single runtime authority, projection, persistence, thread-affinity, bootstrap, and coordinated configuration-rebind boundaries. This correction implements the amended decision without a new architectural choice.

---

## Non-Negotiable Test Boundary

Application behavior must use the real application:

```python
app = TldwCli()
async with app.run_test() as pilot:
    ...
```

Direct unit coverage may call an app-independent function or construct the real
non-application class under test, such as `RuntimePolicyContext`,
`RuntimeSourceStateStore`, `ConfiguredServerTargetStore`, or
`RuntimeServerContextProvider`.

Do not use any of the following as an application:

- `SimpleNamespace`, `MagicMock`, a hand-built object, or an application-shaped
  protocol stub;
- `object.__new__(TldwCli)`;
- a `TldwCli`, Textual `App`, or SettingsScreen test subclass;
- a partial object passed to an unbound `TldwCli` method.

Mocks/fakes remain permitted for narrow external collaborators such as a
durable store, client close, target-store failure, or a patched function
return when the unit under test is not the application itself. Such
collaborators must not impersonate `TldwCli`.

Known legacy harness modules are not valid evidence and are not executed by
this plan. Any affected surrogate-app test encountered by TASK-643 is deleted
or migrated to a full `TldwCli`; it is never preserved as a compatibility
test. Unrelated pre-existing harness debt does not authorize creating,
copying, importing, or relying on another test application.

## Execution Environment

Run commands from:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/privacy-lifecycle-eval-wheel-hardening
```

Activate the repository environment:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/activate
python -c "import pathlib, tldw_chatbook; print(pathlib.Path(tldw_chatbook.__file__).resolve())"
```

The printed package path must be inside this worktree. The verified toolchain is
Python 3.12.11, pytest 8.4.2, and Ruff 0.15.22.

The worktree contains unrelated `.superpowers/sdd/` changes and untracked
artifacts. Do not edit, stage, revert, or delete them. Every commit command
below names exact TASK-643 files.

## File Structure

- Modify `tldw_chatbook/app.py`: one private projection publisher, getter-only
  projections, standalone context installation, and the Boolean commit-first
  coordinator.
- Modify `tldw_chatbook/runtime_policy/bootstrap.py`: app-independent context
  preparation, private context internals, one-time attach-after-prepare
  installation, projection publication, explicit candidate configuration,
  and surfaced CAS rejection.
- Modify `tldw_chatbook/runtime_policy/server_context.py`: focused provider
  rebind, bounded close handling, changed-server hook invalidation, and
  best-effort legacy-target materialization.
- Modify `tldw_chatbook/UI/Screens/settings_screen.py`: checked batched
  URL/token persistence and existing-context rebind.
- Rewrite `Tests/test_application_state_ownership.py`: small exact structural
  collectors and compatibility assertions; remove the Python flow analyzer.
- Modify `Tests/RuntimePolicy/test_runtime_policy_bootstrap.py`: convert every
  application-shaped test to a direct app-independent unit test or move it to
  full-application coverage.
- Create `Tests/RuntimePolicy/test_runtime_policy_full_app.py`: real
  `TldwCli` construction/lifecycle, projection, installation, provider wiring,
  coordinator, and screen-observer tests.
- Modify `Tests/RuntimePolicy/test_runtime_policy_context.py`: direct context
  privacy, revision, callback, owner-thread, and persistence tests.
- Modify `Tests/RuntimePolicy/test_server_context_provider.py`: direct provider
  rebind/cache/hook/close/target coverage.
- Create `Tests/UI/test_settings_runtime_source_switch.py`: real
  `TldwCli.run_test()` plus the actual mounted `SettingsScreen`.

No schema, migration, dependency, persisted format, TASK-644 snapshot, or
TASK-645/646 handoff changes belong in this plan.

## Task 1: Extract App-Independent Preparation and Remove TASK-643 App Stand-Ins

**Files:**

- Modify: `tldw_chatbook/runtime_policy/bootstrap.py`
- Modify: `Tests/RuntimePolicy/test_runtime_policy_bootstrap.py`
- Create: `Tests/RuntimePolicy/test_runtime_policy_full_app.py`

- [ ] **Step 1: Classify every existing bootstrap test**

For every application-shaped `SimpleNamespace`, `_make_app_like`, direct
unbound `TldwCli` call, or nested screen/provider stand-in in
`test_runtime_policy_bootstrap.py`, choose exactly one:

1. test an app-independent function/class directly;
2. move the assertion to `test_runtime_policy_full_app.py` and instantiate
   `TldwCli`;
3. delete it if full-application or direct-unit coverage makes it redundant.

Specific routing:

- effective-path, load/synchronize, persisted-binding, and projection-callback
  ordering tests call the preparation function directly;
- AppState serialization stays a direct compatibility unit test;
- server-context wiring and app projection assertions move to full-app tests;
- `TldwCli.handle_runtime_backend_changed()` tests move to full-app tests;
- `set_authoritative_runtime_source()` tests use a full `TldwCli` until Task 5
  changes its signature/behavior.

Remove `_make_app_like`, the `SimpleNamespace` import if no non-app value needs
it, and every `app_like` application.

- [ ] **Step 2: Add direct preparation tests first**

Write tests for a private app-independent function with this contract:

```python
def _prepare_runtime_policy_context(
    *,
    app_config: Mapping[str, Any] | None,
    publish: Callable[[RuntimeSourceState], None],
    store: RuntimeSourceStateStore | None = None,
    path: str | Path | None = None,
) -> RuntimePolicyContext:
    ...
```

Directly verify:

- effective config-path and explicit-path selection;
- no default-path fallback/migration under `TLDW_CONFIG_PATH`;
- load and configured-binding synchronization;
- synchronization is durably committed as revision 1;
- unchanged loaded state is published once without a save;
- store load/save failure propagates without returning a context;
- a throwing direct initial callback propagates;
- a throwing post-commit callback is contained after the durable commit.

Use real `RuntimeSourceStateStore` where filesystem behavior matters and narrow
recording/raising stores only for deterministic call ordering/failure.

- [ ] **Step 3: Add full-application baseline tests**

In `test_runtime_policy_full_app.py`, reuse the repository's
`app_with_cleanup` fixture for non-screen integration. It yields an actual
`TldwCli`, not a replacement application. Add:

```python
@pytest.mark.asyncio
async def test_full_app_wires_one_runtime_context_to_long_lived_consumers(
    app_with_cleanup: TldwCli,
) -> None:
    app = app_with_cleanup

    assert app.server_context_provider.runtime_context is app.runtime_policy
    assert (
        app.active_server_capability_service.runtime_context
        is app.runtime_policy
    )
    assert app.home_active_work_adapter.runtime_policy is app.runtime_policy
    assert app.service_policy_enforcer.current_state() is app.runtime_policy.state
```

Use actual attribute names from the constructed app; do not fabricate missing
consumers. Also verify the real provider, credential store, and configured
target store are wired by normal construction.

For tests requiring a live screen, use `async with app.run_test()` and the
actual screen stack. Do not introduce a custom App class.

- [ ] **Step 4: Run the new tests to verify RED**

```bash
pytest -q Tests/RuntimePolicy/test_runtime_policy_bootstrap.py
pytest -q Tests/RuntimePolicy/test_runtime_policy_full_app.py
```

Expected: FAIL because the app-independent preparation function and the new
full-application wiring contracts do not yet exist.

- [ ] **Step 5: Extract preparation without adding an app compatibility seam**

Move store construction, load, configured-binding synchronization, context
construction, revision-zero synchronization commit, and direct initial
publication into `_prepare_runtime_policy_context()`.

`load_runtime_policy_for_app()` must:

1. reject an installed `RuntimePolicyContext`;
2. call `_prepare_runtime_policy_context()` with the real app's config and a
   callback to `_apply_runtime_policy_to_app(app, state)`;
3. attach only the returned successfully prepared context;
4. return that same context.

Do not add a second application protocol, optional publisher behavior, public
preparation API, or test-only branch.

- [ ] **Step 6: Complete the test migration and enforce the rule**

Run:

```bash
rg -n \
  'SimpleNamespace|MagicMock|object\\.__new__|_make_app_like|app_like|class .*App' \
  Tests/RuntimePolicy/test_runtime_policy_bootstrap.py \
  Tests/RuntimePolicy/test_runtime_policy_full_app.py
```

Expected: no matches representing an application or application subclass.
If `SimpleNamespace` remains for a non-app value, replace it with the actual
domain value/class so the result is unambiguous.

- [ ] **Step 7: Run direct and full-app tests to verify GREEN**

```bash
pytest -q \
  Tests/RuntimePolicy/test_runtime_policy_bootstrap.py \
  Tests/RuntimePolicy/test_runtime_policy_full_app.py
```

Expected: PASS.

- [ ] **Step 8: Commit preparation and test-boundary correction**

```bash
git add \
  tldw_chatbook/runtime_policy/bootstrap.py \
  Tests/RuntimePolicy/test_runtime_policy_bootstrap.py \
  Tests/RuntimePolicy/test_runtime_policy_full_app.py
git commit -m "test(runtime-policy): use direct units and full app (task-643)"
```

## Task 2: Enforce Read-Only Full-App Projections

**Files:**

- Modify: `Tests/test_application_state_ownership.py`
- Modify: `Tests/RuntimePolicy/test_runtime_policy_bootstrap.py`
- Modify: `Tests/RuntimePolicy/test_runtime_policy_full_app.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/runtime_policy/bootstrap.py`

- [ ] **Step 1: Replace the ownership interpreter with small collectors**

Delete `OwnScopeYieldVisitor`, `ScopedVisitor`, `ProjectionWriteVisitor`,
`ContextOwnershipVisitor`, and their synthetic alias/control-flow tests.
Preserve AppState import/serialization/documentation coverage.

Collectors may inspect only name-bearing AST fields:

```python
def _constant_dynamic_name(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"getattr", "setattr", "delattr"}
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        return node.args[1].value
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return node.slice.value
    return None
```

Inventory `Name`, `Attribute`, `FunctionDef`/`AsyncFunctionDef`/`ClassDef`
names, `arg`, import aliases, keyword names, and the contextual dynamic names
above. Inspect `__slots__` string constants only inside the exact class
assignment. Do not scan comments, docstrings, or arbitrary strings.

- [ ] **Step 2: Add full-application projection tests**

Construct `TldwCli()` normally and assert:

- both backend properties and the server-ID property read one coherent tuple;
- all three descriptors have `fset is None`;
- direct assignment and assignment through an alias to the full app raise
  `AttributeError`;
- calling `_apply_runtime_policy_to_app(full_app, state)` updates one tuple;
- monkeypatching the real class publisher to be missing, descriptor-failing,
  non-callable, or throwing causes the private boundary to fail—never public
  fallback writes.

Test direct initial callback and post-commit callback failure through
`_prepare_runtime_policy_context()` and `RuntimePolicyContext`, respectively,
without creating an application surrogate.

- [ ] **Step 3: Add exact projection structural assertions**

Assert:

- `_runtime_policy_projection_snapshot` has one class default, is read only by
  the three getters, and is assigned only by
  `_publish_runtime_policy_projection()`;
- `_publish_runtime_policy_projection` is defined once and invoked only by
  `_apply_runtime_policy_to_app()`;
- `_apply_runtime_policy_to_app` has exactly two production occurrences: its
  definition and the contained callback closure created by
  `load_runtime_policy_for_app()`; `_prepare_runtime_policy_context()` invokes
  only its callback parameter;
- `_apply_runtime_policy_to_app()` contains no public projection `setattr()`
  calls or missing-publisher fallback;
- no production code assigns any of the three public properties.

- [ ] **Step 4: Run projection tests to verify RED**

```bash
pytest -q \
  Tests/test_application_state_ownership.py \
  Tests/RuntimePolicy/test_runtime_policy_bootstrap.py \
  Tests/RuntimePolicy/test_runtime_policy_full_app.py \
  -k "projection or publisher or AppState or legacy_state"
```

Expected: FAIL because the full app has writable projected instance
attributes and bootstrap contains public fallback writes.

- [ ] **Step 5: Implement the single tuple and private publisher**

In `TldwCli`:

```python
_runtime_policy_projection_snapshot: tuple[str, str | None] = ("local", None)

@property
def current_runtime_backend(self) -> str:
    return self._runtime_policy_projection_snapshot[0]

@property
def runtime_backend(self) -> str:
    return self._runtime_policy_projection_snapshot[0]

@property
def active_server_id(self) -> str | None:
    return self._runtime_policy_projection_snapshot[1]

def _publish_runtime_policy_projection(self, state: RuntimeSourceState) -> None:
    self._runtime_policy_projection_snapshot = (
        state.active_source,
        state.active_server_id,
    )
```

In bootstrap:

```python
def _apply_runtime_policy_to_app(app: Any, state: RuntimeSourceState) -> None:
    publisher = getattr(app, "_publish_runtime_policy_projection")
    if not callable(publisher):
        raise TypeError("runtime policy projection publisher is not callable")
    publisher(state)
```

There is no publisher-absent/public-write branch.

- [ ] **Step 6: Run projection and full-app tests to verify GREEN**

```bash
pytest -q \
  Tests/test_application_state_ownership.py \
  Tests/RuntimePolicy/test_runtime_policy_bootstrap.py \
  Tests/RuntimePolicy/test_runtime_policy_full_app.py
```

Expected: PASS.

- [ ] **Step 7: Commit the projection boundary**

```bash
git add \
  tldw_chatbook/app.py \
  tldw_chatbook/runtime_policy/bootstrap.py \
  Tests/test_application_state_ownership.py \
  Tests/RuntimePolicy/test_runtime_policy_bootstrap.py \
  Tests/RuntimePolicy/test_runtime_policy_full_app.py
git commit -m "refactor(app): enforce one runtime projection (task-643)"
```

## Task 3: Make Context Internals Private and Install Once

**Files:**

- Modify: `Tests/test_application_state_ownership.py`
- Modify: `Tests/RuntimePolicy/test_runtime_policy_context.py`
- Modify: `Tests/RuntimePolicy/test_runtime_policy_bootstrap.py`
- Modify: `Tests/RuntimePolicy/test_runtime_policy_full_app.py`
- Modify: `tldw_chatbook/runtime_policy/bootstrap.py`
- Modify: `tldw_chatbook/app.py`

- [ ] **Step 1: Add direct context and preparation failure tests**

Without an application object, test:

- private store save is the only persistence route;
- state assignment through a context alias raises;
- owner-thread mismatch rejects before save/publication;
- revision mismatch returns `False` before save/publication;
- persistence failure retains snapshot/revision and emits no sentinel;
- callback observes the committed state/revision;
- initial load, synchronization save, and direct initial callback failures
  return no prepared context;
- synchronized durable commit plus contained callback failure returns the
  committed context at revision 1.

- [ ] **Step 2: Add full-app one-time installation tests**

Using `TldwCli()`:

- `app.runtime_policy` is the identity held by every long-lived consumer;
- a second `load_runtime_policy_for_app(app)` raises before any store
  construction/I/O/publication and retains that identity;
- normal construction invokes the installing loader as a standalone
  expression rather than assigning its return value twice.

The attach-after-prepare ordering itself is an exact AST assertion over
`load_runtime_policy_for_app()`, while preparation failures are tested directly
through `_prepare_runtime_policy_context()`. Do not construct a partial app to
observe failed construction.

- [ ] **Step 3: Add exact private structural assertions**

Verify:

- raw `__runtime_policy_state_store` occurs only in `__slots__`, its
  `__init__` assignment, and the immediate top-level `save(candidate)` in
  synchronous non-generator `commit_state()`;
- raw `__runtime_policy_projection_callback` occurs only in `__slots__`, its
  `__init__` assignment, and contained callback invocation;
- both mangled spellings and raw/mangled dynamic accesses are absent;
- `state` has no setter and `persist`/public `store` do not exist;
- production `RuntimeSourceStateStore` references remain confined to
  `source_state.py` and `bootstrap.py`;
- preparation completes before the single `app.runtime_policy` attachment.

- [ ] **Step 4: Run private/bootstrap tests to verify RED**

```bash
pytest -q \
  Tests/test_application_state_ownership.py \
  Tests/RuntimePolicy/test_runtime_policy_context.py \
  Tests/RuntimePolicy/test_runtime_policy_bootstrap.py \
  Tests/RuntimePolicy/test_runtime_policy_full_app.py \
  -k "private or store or callback or revision or one_time or installation"
```

Expected: FAIL on `_store`, `_publish`, repeated loader replacement, and the
constructor's duplicate assignment.

- [ ] **Step 5: Rename private fields and finalize attach-after-prepare**

Use:

```python
__slots__ = (
    "_owner_thread_id",
    "_snapshot",
    "__runtime_policy_projection_callback",
    "__runtime_policy_state_store",
)
```

Keep the save as the immediate top-level expression:

```python
self.__runtime_policy_state_store.save(candidate)
```

Do not add accessors, aliases returned to callers, standalone persistence, or
test escape hatches. Recording tests retain their injected store reference.

Reject an already installed context before preparation. Attach only the
successfully returned prepared context. In `TldwCli.__init__`, call:

```python
load_runtime_policy_for_app(self)
```

without assigning the return; subsequent construction reads
`self.runtime_policy`.

- [ ] **Step 6: Run context/bootstrap/full-app tests to verify GREEN**

```bash
pytest -q \
  Tests/test_application_state_ownership.py \
  Tests/RuntimePolicy/test_runtime_policy_context.py \
  Tests/RuntimePolicy/test_runtime_policy_bootstrap.py \
  Tests/RuntimePolicy/test_runtime_policy_full_app.py
```

Expected: PASS.

- [ ] **Step 7: Commit private one-time installation**

```bash
git add \
  tldw_chatbook/app.py \
  tldw_chatbook/runtime_policy/bootstrap.py \
  Tests/test_application_state_ownership.py \
  Tests/RuntimePolicy/test_runtime_policy_context.py \
  Tests/RuntimePolicy/test_runtime_policy_bootstrap.py \
  Tests/RuntimePolicy/test_runtime_policy_full_app.py
git commit -m "refactor(runtime-policy): install one private context (task-643)"
```

## Task 4: Add the Post-Commit Provider Rebind Primitive

**Files:**

- Modify: `Tests/RuntimePolicy/test_server_context_provider.py`
- Modify: `tldw_chatbook/runtime_policy/server_context.py`

- [ ] **Step 1: Add direct provider tests**

Construct the real `RuntimeServerContextProvider` with real
`RuntimePolicyContext`, real temporary `ConfiguredServerTargetStore`, and real
in-memory credential store. Use narrow failing collaborators only to force
target-write or client-close errors.

Prove `rebind_app_config()`:

- installs refreshed config;
- invalidates its cached client even when server ID is unchanged;
- invokes event/sync invalidation hooks once only when normalized IDs differ;
- upserts the refreshed legacy target/default after config/cache installation;
- contains upsert failure with exception-category-only diagnostics;
- resolves the committed binding through refreshed-config fallback when no
  matching target exists;
- preserves `_legacy_cleared_server_ids`;
- contains synchronous and scheduled close failures after detaching the cached
  client/key.

Use endpoint, token, path, and exception-message sentinels and assert they do
not enter warnings.

- [ ] **Step 2: Run provider tests to verify RED**

```bash
pytest -q Tests/RuntimePolicy/test_server_context_provider.py \
  -k "rebind_app_config or close_failure"
```

Expected: FAIL because no rebind exists and close failures are uncontained.

- [ ] **Step 3: Implement bounded provider rebind**

Add:

```python
def rebind_app_config(
    self,
    app_config: Mapping[str, Any] | None,
    *,
    previous_server_id: str | None,
    next_server_id: str | None,
) -> None:
    self.app_config = app_config or {}
    self._invalidate_cached_client()

    previous = self._normalize_optional_server_id(previous_server_id)
    next_id = self._normalize_optional_server_id(next_server_id)
    if previous != next_id:
        self._invalidate_event_handles_for_server_switch(previous, next_id)
        self._invalidate_sync_handles_for_server_switch(previous, next_id)

    try:
        self.target_store.upsert_legacy_config_target(self.app_config)
    except Exception as exc:
        logger.warning(
            "Legacy server target refresh failed after runtime commit "
            "(exception_category={}).",
            type(exc).__name__,
        )
```

Contain close exceptions inside the existing `_close()` coroutine so both
`asyncio.run()` and scheduled-task paths consume failures and log only category.
Config installation/cache detachment must precede fallible target
materialization.

- [ ] **Step 4: Run focused and full provider tests**

```bash
pytest -q Tests/RuntimePolicy/test_server_context_provider.py \
  -k "rebind_app_config or invalidate_for_server_switch or close"
pytest -q Tests/RuntimePolicy/test_server_context_provider.py
```

Expected: PASS.

- [ ] **Step 5: Commit the provider primitive**

```bash
git add \
  tldw_chatbook/runtime_policy/server_context.py \
  Tests/RuntimePolicy/test_server_context_provider.py
git commit -m "refactor(runtime-policy): add provider config rebind (task-643)"
```

## Task 5: Coordinate Commit-First Runtime Changes in the Full App

**Files:**

- Modify: `Tests/RuntimePolicy/test_runtime_policy_bootstrap.py`
- Modify: `Tests/RuntimePolicy/test_runtime_policy_full_app.py`
- Modify: `Tests/test_application_state_ownership.py`
- Modify: `Tests/UI/test_media_ingest_window_rebuilt.py`
- Modify: `Tests/UI/test_study_screen.py`
- Modify: `tldw_chatbook/runtime_policy/bootstrap.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py`

- [ ] **Step 1: Add direct helper tests**

Test configured-binding derivation and CAS through app-independent inputs.
Refactor `set_authoritative_runtime_source()` to accept the actual context and
configuration rather than an application-shaped object:

```python
def set_authoritative_runtime_source(
    context: RuntimePolicyContext,
    active_source: str,
    *,
    app_config: Mapping[str, Any] | None,
) -> RuntimeSourceState:
    ...
```

Direct tests prove:

- binding derives only from the supplied configuration;
- invalid source returns the unchanged state without save;
- persistence failure leaves the snapshot unchanged;
- `commit_state()` returning `False` raises bounded `RuntimeError` instead of
  returning an unrelated newer snapshot.

Before changing production, add a small AST assertion to
`Tests/test_application_state_ownership.py` that requires the Schedules call
to supply `self.app_instance.runtime_policy` and
`self.app_instance.app_config`, and that rejects any production call passing
an application object as the helper's first argument. Name it
`test_schedules_calls_authoritative_runtime_source_with_context_and_config`.

- [ ] **Step 2: Add full-app coordinator tests**

Construct `TldwCli()` normally. When a screen callback is required, enter
`app.run_test()` and use the actual active screen; patch its real method at the
class or instance boundary only to record/raise.

Prove:

1. a candidate config is not assigned to app/provider before store save;
2. store failure retains old app/provider config, cache, target/default,
   context, projection, and screen and returns `False`;
3. CAS rejection follows the same path;
4. success orders store save → projection → app config → provider rebind →
   actual screen callback and returns `True`;
5. screen-callback failure is contained after commit and still returns `True`;
6. local/no-candidate behavior preserves existing switch invalidation;
7. invalid input returns `False` without publication/callback.

Inject a recording/raising store at full-app construction by patching the store
factory before `TldwCli()` is instantiated. Do not replace the app or call its
method unbound.

Migrate the two legacy unbound-method tests into this full-app suite:

- delete
  `Tests/UI/test_study_screen.py::test_app_level_callback_without_policy_forwards_without_projection_write`;
  a normally constructed `TldwCli` always has a runtime policy, so its
  no-policy surrogate scenario is not a supported application contract;
- move the behavioral assertion from
  `Tests/UI/test_media_ingest_window_rebuilt.py::test_app_level_runtime_backend_change_refreshes_only_media_screen_state`
  into a test that enters `TldwCli.run_test()`, mounts the actual media-ingest
  screen, invokes the bound app coordinator, and observes the mounted screen's
  real state.

Do not retain wrappers or compatibility tests around either partial
application object. Where Study screen notification behavior still needs
coverage, exercise the mounted actual Study screen in a separate
`TldwCli.run_test()` case.

- [ ] **Step 3: Run helper/full-app tests to verify RED**

```bash
pytest -q \
  Tests/RuntimePolicy/test_runtime_policy_bootstrap.py \
  Tests/RuntimePolicy/test_runtime_policy_full_app.py \
  -k "authoritative or coordinator or config_candidate or cas or screen_callback"
pytest -q \
  Tests/test_application_state_ownership.py::test_schedules_calls_authoritative_runtime_source_with_context_and_config
```

Expected: FAIL on the app-coupled helper, CAS handling, provider ordering, and
Boolean coordinator contract, including the unchanged Schedules call shape.

- [ ] **Step 4: Implement the helper and Boolean coordinator**

In `app.py`, change:

```python
async def handle_runtime_backend_changed(
    self,
    runtime_backend: str,
    *,
    app_config_override: Mapping[str, Any] | None = None,
) -> bool:
```

Required sequence:

1. reject invalid source with `False`;
2. capture previous server ID;
3. call `set_authoritative_runtime_source(self.runtime_policy, ...,
   app_config=candidate_or_current)`;
4. on derivation/CAS/persistence failure, emit category-only warning, notify
   prior source remains active, return `False`;
5. after commit, assign candidate config when supplied;
6. call provider `rebind_app_config()` for a candidate, otherwise preserve the
   existing server-switch invalidation;
7. derive source from committed context;
8. await the actual active-screen callback with category-only containment;
9. return `True` and never report rollback after durable commit.

Update the production Schedules caller to pass
`self.app_instance.runtime_policy` and `self.app_instance.app_config`. This is
a mechanical caller adaptation after the structural test is RED; do not add a
compatibility overload that accepts an app.

- [ ] **Step 5: Run direct, structural, and full-app tests**

```bash
pytest -q \
  Tests/RuntimePolicy/test_runtime_policy_bootstrap.py \
  Tests/RuntimePolicy/test_runtime_policy_full_app.py \
  Tests/test_application_state_ownership.py
```

Do not execute `Tests/UI/test_schedules_workbench.py` as evidence: it defines
custom Textual test applications. The exact caller shape is covered by the
structural assertion above, and helper behavior is covered directly.

- [ ] **Step 6: Commit the coordinator**

```bash
git add \
  tldw_chatbook/app.py \
  tldw_chatbook/runtime_policy/bootstrap.py \
  tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py \
  Tests/RuntimePolicy/test_runtime_policy_bootstrap.py \
  Tests/RuntimePolicy/test_runtime_policy_full_app.py \
  Tests/test_application_state_ownership.py \
  Tests/UI/test_media_ingest_window_rebuilt.py \
  Tests/UI/test_study_screen.py
git commit -m "fix(app): coordinate runtime policy rebinds (task-643)"
```

## Task 6: Rebind the Actual Settings Screen Without Replacing Authority

**Files:**

- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `Tests/test_application_state_ownership.py`
- Create: `Tests/UI/test_settings_runtime_source_switch.py`

- [ ] **Step 1: Add full-application Settings tests**

Every test:

1. constructs `TldwCli()`;
2. enters `app.run_test()`;
3. pushes or navigates to the real `SettingsScreen`;
4. obtains that mounted screen from the real screen stack;
5. invokes the actual `_perform_runtime_source_switch()` or drives the modal
   result path.

Do not instantiate `SettingsScreen` alone, override its `app` property, subclass
it, or use a destination harness.

Patch narrow collaborators (`save_settings_to_cli_config`, `load_settings`,
Sync v2 prepare, or the already separately verified coordinator result) to
force each outcome. Prove:

- local failure has no Settings success notice/refresh;
- local success notifies and refreshes once;
- one batch contains both URL and token, including an empty token;
- failed batch save causes no reload, coordinator, partial activation, success
  notice, or refresh;
- reload failure retains old in-memory authority/config and reports bounded
  activation failure;
- successful reload passes the mapping only as `app_config_override`; Settings
  itself never assigns `app.app_config`;
- coordinator `False` causes no Sync v2 preparation/success/refresh while the
  saved file remains available for retry;
- success retains server-ID validation and Sync v2 preparation behavior.
- one end-to-end server activation uses the real batched config writer, real
  reload, real app coordinator, real context, and real provider; it retains the
  original context identity while all committed observers select the new
  server;
- one end-to-end runtime-store failure after a successful Settings-file save
  retains every in-memory observer and the actual mounted screen while leaving
  the saved file available for retry.

Capture path, URL, token, server-ID, and exception-message sentinels and assert
new failure diagnostics omit them.

- [ ] **Step 2: Add the final loader-reference allowlist**

Production references to `load_runtime_policy_for_app` must be exactly:

- definition in `bootstrap.py`;
- internal ensure fallback;
- import in `app.py`;
- standalone `TldwCli.__init__` call.

Count direct/qualified names, definition bindings, import aliases, and
contextual constant dynamic access. The Settings import/call makes this RED.

- [ ] **Step 3: Run Settings/ownership tests to verify RED**

```bash
pytest -q \
  Tests/UI/test_settings_runtime_source_switch.py \
  Tests/test_application_state_ownership.py
```

Expected: FAIL because Settings performs two unchecked saves, assigns app
config, reloads a second context, and ignores coordinator outcome.

- [ ] **Step 4: Implement checked batch save and coordinated rebind**

Local path:

```python
switched = await app.handle_runtime_backend_changed("local")
if not switched:
    return
self.app.notify("Runtime source set to local.", severity="information")
self._refresh_manual_sync_rows()
```

Server path:

```python
saved = save_settings_to_cli_config(
    {
        "tldw_api": {
            "base_url": base_url,
            "auth_token": auth_token,
        }
    }
)
if not saved:
    self.app.notify(
        "Server settings could not be saved; the previous source remains active.",
        severity="error",
    )
    return

try:
    refreshed_config = load_settings(force_reload=True)
except Exception as exc:
    logger.warning(
        "Saved server settings could not be loaded "
        "(exception_category={}).",
        type(exc).__name__,
    )
    self.app.notify(
        "Server settings were saved but could not be activated; "
        "the previous source remains active.",
        severity="error",
    )
    return

switched = await app.handle_runtime_backend_changed(
    "server",
    app_config_override=refreshed_config,
)
if not switched:
    return
```

Remove Settings' loader import/call and direct app-config assignment. Do not
roll back a successful Settings-file write after later policy failure.

- [ ] **Step 5: Run Settings, ownership, and full-app tests**

```bash
pytest -q \
  Tests/UI/test_settings_runtime_source_switch.py \
  Tests/test_application_state_ownership.py \
  Tests/RuntimePolicy/test_runtime_policy_full_app.py
```

Expected: PASS.

- [ ] **Step 6: Enforce the no-app-stand-in rule in new/rewritten tests**

```bash
rg -n \
  'SimpleNamespace|MagicMock|object\\.__new__|app_like|_make_app_like|class .*App' \
  Tests/RuntimePolicy/test_runtime_policy_bootstrap.py \
  Tests/RuntimePolicy/test_runtime_policy_full_app.py \
  Tests/UI/test_settings_runtime_source_switch.py
```

Expected: no application stand-ins or subclasses. Direct collaborator fakes
must have narrow store/client/target names and cannot expose app fields.

- [ ] **Step 7: Commit the Settings boundary**

```bash
git add \
  tldw_chatbook/UI/Screens/settings_screen.py \
  Tests/UI/test_settings_runtime_source_switch.py \
  Tests/test_application_state_ownership.py
git commit -m "fix(settings): rebind the existing runtime authority (task-643)"
```

## Task 7: Verify Corrected TASK-643 and Hold Reconciliation

**Files:**

- No planned production changes.
- Fix only TASK-643 regressions.
- Do not check acceptance criteria, add final Implementation Notes, or mark
  TASK-643 Done before TASK-646 shared release gates.

- [ ] **Step 1: Run authoritative direct/full-app suites**

```bash
pytest -q Tests/RuntimePolicy
pytest -q Tests/test_application_state_ownership.py
pytest -q Tests/UI/test_settings_runtime_source_switch.py
```

Expected: PASS. These are TASK-643 acceptance evidence.

- [ ] **Step 2: Exclude surrogate-app suites from claimed evidence**

Do not execute or cite the known legacy harness suites as TASK-643 regression
evidence. Several construct Textual test applications or application-shaped
`SimpleNamespace` objects, which violates this plan's test boundary. All
application-level regression claims in this tranche come from the full
`TldwCli` suites in Step 1; app-independent behavior comes from direct
function/class tests.

The two coordinator tests removed from
`Tests/UI/test_media_ingest_window_rebuilt.py` and
`Tests/UI/test_study_screen.py` have full-app replacements in
`Tests/RuntimePolicy/test_runtime_policy_full_app.py`; do not replace them with
another harness.

- [ ] **Step 3: Run scoped compilation and lint**

```bash
python -m compileall -q \
  tldw_chatbook/runtime_policy \
  tldw_chatbook/state \
  tldw_chatbook/app.py \
  tldw_chatbook/UI/Screens/settings_screen.py \
  tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py
python -m ruff check \
  tldw_chatbook/runtime_policy \
  tldw_chatbook/state \
  tldw_chatbook/app.py \
  tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py \
  Tests/RuntimePolicy \
  Tests/test_application_state_ownership.py \
  Tests/UI/test_settings_runtime_source_switch.py
python -m ruff check --ignore F841 tldw_chatbook/UI/Screens/settings_screen.py
python -m ruff check --ignore F841 tldw_chatbook/config.py
python -m ruff check --select F841 --output-format concise \
  tldw_chatbook/UI/Screens/settings_screen.py \
  tldw_chatbook/config.py
```

Expected: all commands except the final baseline inventory exit 0. The final
inventory exits 1 with exactly these four pre-tranche diagnostics and no
others:

- `settings_screen.py`: `config_path` in
  `_save_advanced_config_text()` and `_read_advanced_backup_preview()`;
- `config.py`: `file_validation_section` and
  `providers_section_from_toml`.

Any additional F841 is a tranche failure. No file-wide or repo-wide suppression
may be added.

- [ ] **Step 4: Run scoped formatting and diff checks**

```bash
python -m ruff format --check \
  tldw_chatbook/runtime_policy/bootstrap.py \
  tldw_chatbook/runtime_policy/server_context.py \
  tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py \
  tldw_chatbook/state/app_state.py \
  tldw_chatbook/state/__init__.py \
  Tests/RuntimePolicy \
  Tests/test_application_state_ownership.py \
  Tests/UI/test_settings_runtime_source_switch.py
git diff --check
```

Expected: exit 0. Whole-file formatting excludes pre-existing unformatted
`app.py`, `config.py`, and `settings_screen.py`; review their changed hunks
directly.

- [ ] **Step 5: Review lifecycle/privacy evidence**

Confirm:

1. no TASK-643 application acceptance test uses a partial/simplified/mock
   application, application subclass, or unbound app method;
2. persistence/CAS/preparation failures publish nothing;
3. failed preparation returns no context to install;
4. one full-app context identity reaches every long-lived consumer;
5. failed Settings activation changes no in-memory observer before commit;
6. provider target/close and screen-callback failures occur after commit and
   cannot masquerade as rollback;
7. path, endpoint, token, server-ID, label, serialized-state, exception-value,
   and object-representation sentinels are absent from new diagnostics;
8. alias writes to full-app projections raise at runtime;
9. old alias/scope/control-flow analyzer classes are absent.

- [ ] **Step 6: Record pending shared-gate state**

```bash
backlog task 643 --plain
git status --short
```

Expected:

- TASK-643 remains `In Progress`;
- acceptance criteria remain unchecked;
- no final Implementation Notes claim completion;
- unrelated `.superpowers/sdd/` artifacts remain untouched.

TASK-646 will run installed-wheel, product-maturity, static, and full-suite
release gates before TASK-643–646 are reconciled as Done.
