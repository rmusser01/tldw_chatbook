# Runtime Policy Authority Implementation Plan (TASK-643)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `RuntimePolicyContext` the only live runtime-source authority, with private durable storage, persist-before-publish revision commits, and stale capability-result rejection.

**Architecture:** Keep the existing immutable `RuntimeSourceState` value and runtime-policy package. Replace the mutable context seam with an owner-thread-affine compare-and-swap context whose only mutation path persists before publication and invokes a contained projection callback. Reuse ADR-022 private-path primitives for disk I/O, derive the default policy file from the effective config path at construction time, and leave the exported legacy state dataclasses as importable compatibility containers.

**Tech Stack:** Python 3.11+, frozen dataclasses, `threading.get_ident`, Textual application callbacks, Loguru, ADR-022 private-path helpers, pytest/pytest-asyncio.

**Backlog:** [TASK-643](../../../backlog/tasks/task-643%20-%20Make-runtime-policy-the-sole-application-runtime-source-authority.md)

**Specification:** [Application Session State Ownership Design](../specs/2026-07-26-application-session-state-ownership-design.md)

**ADR required:** yes

**ADR path:** `backlog/decisions/026-application-session-state-ownership.md`

**Reason:** ADR-026 already defines the runtime authority, mutation, projection, thread-affinity, and persistence boundary implemented by this task.

> **Quality-review amendment:** Task 2, Step 3's `_store` and callback
> naming/enforcement details; Task 4's public-projection and private-store
> enforcement steps; and the unplanned Settings runtime-policy reload are
> superseded by
> [TASK-643 Structural Ownership Enforcement Design](../specs/2026-07-26-task-643-structural-ownership-enforcement-design.md).
> Do not resume implementation from the old direct-public-write or alias-flow
> instructions. A corrective implementation plan must replace those steps
> after the amended design passes written-spec and user review.

---

## Execution Environment

This worktree has no `.venv`, and `/usr/bin/python3` is Python 3.9. Before
running any command in this plan:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/activate
python -c "import pathlib, tldw_chatbook; print(pathlib.Path(tldw_chatbook.__file__).resolve())"
```

The printed path must be inside
`.../.worktrees/privacy-lifecycle-eval-wheel-hardening/tldw_chatbook`, not the
main checkout or site-packages. The verified environment is Python 3.12.11,
pytest 8.4.2, and Ruff 0.15.22.

## File Structure

- Modify `tldw_chatbook/runtime_policy/source_state.py`: make runtime-policy JSON use the ADR-022 verified private reader and random-name atomic private writer.
- Modify `tldw_chatbook/runtime_policy/bootstrap.py`: implement revisioned `RuntimePolicyContext`, effective-path construction, contained projections, and persist-before-publish source changes.
- Modify `tldw_chatbook/runtime_policy/server_capabilities.py`: derive from a captured revision and discard stale probe results and side effects.
- Modify `tldw_chatbook/runtime_policy/server_context.py`: rebind refreshed app
  configuration without replacing the runtime-policy context.
- Modify `tldw_chatbook/config.py`: expose the existing application-owned config-directory decision as a reusable helper rather than duplicating override policy.
- Modify `tldw_chatbook/app.py`: remove `AppState`, install only the runtime projection callback, and contain runtime change persistence failures.
- Modify `tldw_chatbook/state/app_state.py` and `tldw_chatbook/state/__init__.py`: correct the false live-authority documentation while retaining imports and serialization.
- Modify `tldw_chatbook/UI/Screens/media_ingest_screen.py` and `tldw_chatbook/UI/Screens/study_screen.py`: remove independent writes to the three compatibility projections.
- Modify `tldw_chatbook/UI/Screens/settings_screen.py`: replace runtime-policy
  reload with existing-context/provider rebind.
- Create `Tests/RuntimePolicy/test_runtime_policy_private_store.py`: focused path, permission, symlink, posture, and redaction checks.
- Create `Tests/RuntimePolicy/test_runtime_policy_context.py`: revision, ordering, failure atomicity, projection, and owner-thread checks.
- Modify `Tests/RuntimePolicy/test_runtime_policy_bootstrap.py`: effective-path, app integration, failure recovery, and compatibility tests.
- Modify `Tests/RuntimePolicy/test_server_context_provider.py`: retain the
  injected recording store in the test instead of reaching through the
  context's now-private backing store.
- Modify `Tests/RuntimePolicy/test_active_server_capabilities.py`: revision-aware fake context and deterministic stale-probe tests.
- Create `Tests/test_application_state_ownership.py`: scoped AST ownership guard, extended by TASK-644–646.

## Task 1: Make Runtime-Policy Storage Private

**Files:**

- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/runtime_policy/source_state.py`
- Modify: `tldw_chatbook/runtime_policy/bootstrap.py`
- Create: `Tests/RuntimePolicy/test_runtime_policy_private_store.py`
- Modify: `Tests/RuntimePolicy/test_runtime_policy_bootstrap.py`

- [ ] **Step 1: Write failing effective-path and private-store tests**

Add tests that:

```python
def test_default_policy_path_follows_effective_config_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "custom" / "config.toml"
    config_path.parent.mkdir()
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    app = SimpleNamespace(app_config={})

    constructed: list[RuntimeSourceStateStore] = []
    real_store_type = RuntimeSourceStateStore

    def capture_store(path, **kwargs):
        store = real_store_type(path, **kwargs)
        constructed.append(store)
        return store

    monkeypatch.setattr(bootstrap, "RuntimeSourceStateStore", capture_store)
    load_runtime_policy_for_app(app)

    assert constructed[0].path == config_path.parent / "runtime_policy.json"


def test_store_rejects_symlink_before_parsing(tmp_path: Path) -> None:
    target = tmp_path / "real.json"
    target.write_text('{"active_source": "server"}', encoding="utf-8")
    link = tmp_path / "runtime_policy.json"
    link.symlink_to(target)

    with pytest.raises(PrivatePathError):
        RuntimeSourceStateStore(link).load()


def test_store_hardens_eligible_mode_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "runtime_policy.json"
    path.write_text('{"active_source": "local"}', encoding="utf-8")
    path.chmod(0o644)

    assert RuntimeSourceStateStore(path).load().active_source == "local"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
```

Also cover: missing file returns the safe default; malformed JSON returns the safe default only after verified opening; FIFO/non-regular and replaced targets fail closed; custom parents are neither created nor chmodded; default config parent is created/hardened to `0700`; override mode never reads the ordinary default file; atomic writes leave no predictable `.tmp`; Windows reports `UNVERIFIED_PLATFORM`; a unique path/state sentinel never enters captured logs.

- [ ] **Step 2: Run the focused tests and confirm the old I/O fails**

Run:

```bash
pytest Tests/RuntimePolicy/test_runtime_policy_private_store.py -q
pytest Tests/RuntimePolicy/test_runtime_policy_bootstrap.py -q -k "effective_config or override"
```

Expected: FAIL because `RuntimeSourceStateStore` uses ordinary `Path.open`, creates arbitrary parents, uses a predictable `.tmp`, and `DEFAULT_RUNTIME_POLICY_PATH` ignores the effective config override.

- [ ] **Step 3: Expose the config-directory ownership decision**

Rename `_application_owned_config_directory()` in `tldw_chatbook/config.py` to `application_owned_config_directory()` and keep its existing callers on the same helper:

```python
def application_owned_config_directory(config_path: Path) -> Path | None:
    """Return the app-owned default config parent, never a custom parent."""
    if os.environ.get("TLDW_CONFIG_PATH"):
        return None
    default_path = lexical_path(DEFAULT_CONFIG_PATH)
    return default_path.parent if lexical_path(config_path) == default_path else None
```

Do not broaden which directories the application may create or chmod.

- [ ] **Step 4: Implement verified load and atomic private save**

Give `RuntimeSourceStateStore` an optional `application_owned_directory` constructor argument. Serialize before opening any file, load with `open_private_binary()`, and save with `atomic_private_write_text()`:

```python
class RuntimeSourceStateStore:
    def __init__(
        self,
        path: str | Path,
        *,
        application_owned_directory: str | Path | None = None,
    ) -> None:
        self.path = lexical_path(path)
        self.application_owned_directory = (
            lexical_path(application_owned_directory)
            if application_owned_directory is not None
            else None
        )

    def load(self) -> RuntimeSourceState:
        try:
            with open_private_binary(self.path) as opened:
                _report_runtime_policy_posture(opened.result)
                data = json.load(opened.stream)
        except FileNotFoundError:
            return RuntimeSourceState()
        except (TypeError, ValueError, json.JSONDecodeError):
            return RuntimeSourceState()
        return (
            RuntimeSourceState.from_dict(data)
            if isinstance(data, dict)
            else RuntimeSourceState()
        )

    def save(self, state: RuntimeSourceState) -> None:
        payload = json.dumps(
            runtime_source_state_to_dict(state),
            indent=2,
            sort_keys=True,
        )
        result = atomic_private_write_text(
            self.path,
            payload,
            application_owned_directory=self.application_owned_directory,
        )
        _report_runtime_policy_posture(result)
```

Catch malformed content only. Let privacy/path `OSError` and `PrivatePathError` propagate. `_report_runtime_policy_posture()` may log only operation/posture categories, never the path, exception value, or serialized state.

- [ ] **Step 5: Derive the default path at context construction**

Delete the import-time `DEFAULT_RUNTIME_POLICY_PATH`. In `load_runtime_policy_for_app()`:

```python
effective_config_path = get_cli_config_path()
selected_path = lexical_path(path) if path is not None else (
    effective_config_path.parent / "runtime_policy.json"
)
runtime_store = store or RuntimeSourceStateStore(
    selected_path,
    application_owned_directory=(
        application_owned_config_directory(effective_config_path)
        if path is None
        else None
    ),
)
```

An injected store remains authoritative. An explicitly injected path is custom even when it resembles the default and must not gain directory-mutation authority.

- [ ] **Step 6: Run the storage tests**

Run:

```bash
pytest Tests/RuntimePolicy/test_runtime_policy_private_store.py Tests/RuntimePolicy/test_runtime_policy_bootstrap.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit the storage boundary**

```bash
git add tldw_chatbook/config.py tldw_chatbook/runtime_policy/source_state.py tldw_chatbook/runtime_policy/bootstrap.py Tests/RuntimePolicy/test_runtime_policy_private_store.py Tests/RuntimePolicy/test_runtime_policy_bootstrap.py
git commit -m "fix(runtime-policy): use private effective-path storage (task-643)"
```

## Task 2: Replace Mutable Context State with Revisioned Commits

**Files:**

- Modify: `tldw_chatbook/runtime_policy/bootstrap.py`
- Create: `Tests/RuntimePolicy/test_runtime_policy_context.py`
- Modify: `Tests/RuntimePolicy/test_runtime_policy_bootstrap.py`
- Modify: `Tests/RuntimePolicy/test_server_context_provider.py`

- [ ] **Step 1: Write failing context contract tests**

Cover successful ordering, stale revisions, persistence failure, projection failure containment, immutable snapshots, missing setter/persist, and foreign-thread rejection:

```python
def test_commit_persists_before_publish() -> None:
    events: list[tuple[str, RuntimeSourceState]] = []
    initial = RuntimeSourceState()
    candidate = replace(initial, active_source="server")
    store = RecordingStore(lambda: events.append(("persist", candidate)))
    context = RuntimePolicyContext(
        initial,
        store,
        publish=lambda state: events.append(("publish", state)),
    )

    assert context.commit_state(candidate, expected_revision=0) is True
    assert events == [("persist", candidate), ("publish", candidate)]
    assert context.snapshot() == (candidate, 1)


def test_persistence_failure_leaves_state_revision_and_projection_unchanged() -> None:
    projected: list[RuntimeSourceState] = []
    context = RuntimePolicyContext(
        RuntimeSourceState(),
        RaisingStore(),
        publish=projected.append,
    )

    with pytest.raises(OSError):
        context.commit_state(
            RuntimeSourceState(active_source="server"),
            expected_revision=0,
        )

    assert context.snapshot() == (RuntimeSourceState(), 0)
    assert projected == []
```

Use `ThreadPoolExecutor(max_workers=1)` for deterministic off-owner rejection.
Assert a stale commit never calls the injected store or callback. Assert
assigning `context.state` raises `AttributeError`, `persist` is absent, and no
public `store` attribute exposes a direct persistence path.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest Tests/RuntimePolicy/test_runtime_policy_context.py -q
```

Expected: FAIL because the context exposes mutable `state`, `persist()`, no revision, and no thread-affinity check.

- [ ] **Step 3: Implement the minimal owner-thread-affine context**

Use a normal slotted class so `state` can be a read-only property:

```python
class RuntimePolicyContext:
    __slots__ = (
        "_owner_thread_id",
        "__runtime_policy_projection_callback",
        "_snapshot",
        "__runtime_policy_state_store",
    )

    def __init__(
        self,
        state: RuntimeSourceState,
        store: RuntimeSourceStateStore,
        *,
        publish: Callable[[RuntimeSourceState], None] | None = None,
    ) -> None:
        self._snapshot = (state, 0)
        self._owner_thread_id = threading.get_ident()
        self.__runtime_policy_projection_callback = publish
        self.__runtime_policy_state_store = store

    @property
    def state(self) -> RuntimeSourceState:
        return self._snapshot[0]

    def snapshot(self) -> tuple[RuntimeSourceState, int]:
        return self._snapshot

    def commit_state(
        self,
        candidate: RuntimeSourceState,
        *,
        expected_revision: int,
    ) -> bool:
        self._assert_owner_thread()
        _, current_revision = self._snapshot
        if expected_revision != current_revision:
            return False
        self.__runtime_policy_state_store.save(candidate)
        self._snapshot = (candidate, current_revision + 1)
        if self.__runtime_policy_projection_callback is not None:
            try:
                self.__runtime_policy_projection_callback(candidate)
            except Exception as exc:
                logger.warning(
                    "Runtime policy projection failed after durable commit "
                    "(exception_category={})",
                    type(exc).__name__,
                )
        return True
```

Use an ordinary `except Exception as exc` if preferred; log only `type(exc).__name__`. `_assert_owner_thread()` raises `RuntimeError("runtime policy mutation requires the owner thread")` without identifiers.
Keeping state and revision in one tuple makes `snapshot()` a coherent atomic
read for ordinary foreign-thread readers while all writes remain owner-thread
affine; do not publish them as two independently assigned attributes.

In `test_server_context_provider.py`, let `_runtime_context()` accept an
optional injected `SavingRuntimeStore`. The authority test retains that store
in a local variable, passes it into the context, and asserts
`runtime_store.saved_states == []`; it must not reach through
`runtime_context.__runtime_policy_state_store`, its mangled spelling, or
require a public compatibility property.

- [ ] **Step 4: Route bootstrap synchronization and source changes through commits**

Construct the context around the loaded state, set it on the app, and commit a synchronized candidate only when it differs. Otherwise apply the initial projection once. `set_authoritative_runtime_source()` must capture `(state, revision)`, derive solely from that state, call `commit_state()`, and return the fresh authoritative state if the commit is stale:

```python
state, revision = context.snapshot()
candidate = replace(...)
if context.commit_state(candidate, expected_revision=revision):
    return candidate
return context.snapshot()[0]
```

Do not reintroduce direct storage or projection calls in this function.

- [ ] **Step 5: Run context and bootstrap tests**

Run:

```bash
pytest Tests/RuntimePolicy/test_runtime_policy_context.py Tests/RuntimePolicy/test_runtime_policy_bootstrap.py -q
pytest Tests/RuntimePolicy/test_server_context_provider.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit revisioned publication**

```bash
git add tldw_chatbook/runtime_policy/bootstrap.py Tests/RuntimePolicy/test_runtime_policy_context.py Tests/RuntimePolicy/test_runtime_policy_bootstrap.py Tests/RuntimePolicy/test_server_context_provider.py
git commit -m "refactor(runtime-policy): publish through revisioned commits (task-643)"
```

## Task 3: Reject Stale Capability Results

**Files:**

- Modify: `tldw_chatbook/runtime_policy/server_capabilities.py`
- Modify: `Tests/RuntimePolicy/test_active_server_capabilities.py`

- [ ] **Step 1: Replace mutable test doubles with the real commit contract**

Introduce a small fake with `snapshot()` and `commit_state()` or use `RuntimePolicyContext` with an in-memory recording store. Remove every test assignment to `.state` and every `persist` assertion.

- [ ] **Step 2: Write deterministic stale-probe tests**

Block `probe_health()` on an `asyncio.Event`, advance the context revision on the event-loop thread, release the probe, and assert:

```python
assert snapshot["errors"] == [{
    "reason_code": "capability_result_superseded",
    "message": "Capability refresh was superseded by a newer runtime selection.",
}]
assert snapshot["health"] == {}
assert snapshot["readiness"] == {}
assert snapshot["docs_info"] == {}
assert target_store.upsert_status.call_count == 0
assert context.state == newer_state
```

Add separate source-change and active-server-change cases. Add the same revision protocol assertion for the no-server-configured branch.

- [ ] **Step 3: Run the stale tests and verify failure**

Run:

```bash
pytest Tests/RuntimePolicy/test_active_server_capabilities.py -q -k "stale or superseded or server_is_cleared"
```

Expected: FAIL because refresh directly assigns and persists a candidate derived from old state.

- [ ] **Step 4: Commit capability state only against the captured revision**

At refresh start:

```python
state, revision = self.runtime_context.snapshot()
```

After deriving the candidate:

```python
committed = self.runtime_context.commit_state(
    updated_state,
    expected_revision=revision,
)
if not committed:
    fresh_state, _ = self.runtime_context.snapshot()
    return self._superseded_snapshot(fresh_state, now=now)
```

Call `_persist_target_status()` only after `committed is True`. Build `_superseded_snapshot()` with empty discovery payload mappings and only the stable superseded reason code. Convert exception messages in capability diagnostics to bounded, non-secret copy; never include `str(exc)` when it can contain endpoints, credentials, or response bodies.

- [ ] **Step 5: Run the capability suite**

Run:

```bash
pytest Tests/RuntimePolicy/test_active_server_capabilities.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit stale-result protection**

```bash
git add tldw_chatbook/runtime_policy/server_capabilities.py Tests/RuntimePolicy/test_active_server_capabilities.py
git commit -m "fix(runtime-policy): discard superseded capability probes (task-643)"
```

## Task 4: Remove AppState and Independent Projection Writers

**Files:**

- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/state/app_state.py`
- Modify: `tldw_chatbook/state/__init__.py`
- Modify: `tldw_chatbook/UI/Screens/media_ingest_screen.py`
- Modify: `tldw_chatbook/UI/Screens/study_screen.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/runtime_policy/server_context.py`
- Modify: `Tests/RuntimePolicy/test_runtime_policy_bootstrap.py`
- Modify: `Tests/RuntimePolicy/test_server_context_provider.py`
- Create: `Tests/test_application_state_ownership.py`

- [ ] **Step 1: Write failing integration and AST guard tests**

The guard must parse production Python, not grep comments. Assert:

- `tldw_chatbook/app.py` neither imports nor instantiates `AppState`;
- the runtime-policy projection boundary does not read, write, or dynamically
  access `app.app_state`;
- `RuntimePolicyContext.state` is a setter-free property and calls to its
  removed `persist()` are absent;
- `TldwCli` exposes getter-only public projections backed by the exact private
  tuple and publisher defined in the structural-enforcement design;
- the private publisher is invoked only by `_apply_runtime_policy_to_app`;
- the context's private projection callback is accessible only by the exact
  constructor/commit structures in the structural-enforcement design;
- `load_runtime_policy_for_app` is a one-time installer whose production
  references are confined to its definition, ensure fallback, app import, and
  `TldwCli` construction;
- production calls to `RuntimeSourceStateStore.save()` occur only from
  `RuntimePolicyContext.commit_state()`;
- the uniquely named context backing store has only the exact structural
  references allowed by the structural-enforcement design, and no public
  `context.store` compatibility property is introduced.

Delete the existing alias, lexical-scope, and control-flow visitor
implementation. Replacement checks must be stateless, direct AST-shape and
descriptor assertions over name-bearing syntax only.

Add an async bootstrap test whose store raises on save and assert `handle_runtime_backend_changed()` keeps the previous context snapshot and all three app projections, does not invoke the active screen callback, and emits one bounded warning that omits unique path/server sentinels.
Add a projection test using an app double with no `app_state` attribute and a
second double whose trap property fails if `app_state` is accessed; both must
receive only the three compatibility projection attributes.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest Tests/test_application_state_ownership.py Tests/RuntimePolicy/test_runtime_policy_bootstrap.py -q -k "AppState or projection or persistence_failure or ownership"
```

Expected: FAIL on the live `AppState`, fallback projection branch, and screen-owned writes.

- [ ] **Step 3: Remove the live AppState dependency and correct its claim**

Delete the import and `self.app_state = AppState()` from `TldwCli`. Delete the
`getattr(app, "app_state", ...)` branch and `app_state.runtime_source` write
from `_apply_runtime_policy_to_app()`. Implement the getter-only projections,
single private projection tuple, private publisher, uniquely named context
store and callback, one-time installation, Settings/provider rebind, static
fallback detection, and exact structural checks from the
structural-enforcement design. Keep all
compatibility exports. Change documentation to state that these classes are
caller-owned serializable compatibility containers and are not the
application's live authority. Preserve `to_dict()`/`from_dict()` behavior
byte-for-byte.

- [ ] **Step 4: Contain runtime persistence failure at the app boundary**

Wrap `set_authoritative_runtime_source()` in `handle_runtime_backend_changed()`:

```python
try:
    updated_state = set_authoritative_runtime_source(self, normalized_backend)
except Exception as exc:
    logger.warning(
        "Runtime source change was not committed "
        "(exception_category={})",
        type(exc).__name__,
    )
    self.notify(
        "Runtime source could not be changed; the previous source remains active.",
        severity="warning",
    )
    return
```

Do not include `exc`, paths, endpoints, labels, or state in logs. Remove the no-policy projection fallback: production startup always installs the context. In Media Ingest and Study, update only screen-owned view state and let the app projection callback remain the sole writer.

- [ ] **Step 5: Run compatibility and ownership tests**

Run:

```bash
pytest Tests/test_application_state_ownership.py Tests/RuntimePolicy -q
pytest Tests/UI/test_media_ingest_window_rebuilt.py Tests/UI/test_study_screen.py -q
```

Expected: PASS, including legacy state serialization.

- [ ] **Step 6: Commit application detachment**

```bash
git add tldw_chatbook/app.py tldw_chatbook/state tldw_chatbook/UI/Screens/media_ingest_screen.py tldw_chatbook/UI/Screens/study_screen.py Tests/test_application_state_ownership.py Tests/RuntimePolicy/test_runtime_policy_bootstrap.py Tests/UI/test_media_ingest_window_rebuilt.py Tests/UI/test_study_screen.py
git commit -m "refactor(app): detach legacy root state authority (task-643)"
```

## Task 5: Verify TASK-643 and Hold Final Reconciliation

**Files:**

- No production or Backlog status changes expected; fix only verified
  regressions within TASK-643 acceptance criteria.

- [ ] **Step 1: Run focused and cross-boundary verification**

```bash
pytest Tests/RuntimePolicy -q
pytest Tests/test_application_state_ownership.py Tests/UI/test_media_ingest_window_rebuilt.py Tests/UI/test_study_screen.py -q
python -m compileall -q tldw_chatbook/runtime_policy tldw_chatbook/state
python -m ruff check tldw_chatbook/runtime_policy tldw_chatbook/state tldw_chatbook/app.py tldw_chatbook/UI/Screens/media_ingest_screen.py tldw_chatbook/UI/Screens/study_screen.py Tests/RuntimePolicy Tests/test_application_state_ownership.py
python -m ruff check --ignore F841 tldw_chatbook/config.py
python -m ruff format --check tldw_chatbook/runtime_policy/source_state.py tldw_chatbook/runtime_policy/bootstrap.py tldw_chatbook/runtime_policy/server_capabilities.py tldw_chatbook/state/app_state.py tldw_chatbook/state/__init__.py tldw_chatbook/UI/Screens/media_ingest_screen.py tldw_chatbook/UI/Screens/study_screen.py Tests/RuntimePolicy Tests/test_application_state_ownership.py
git diff --check
```

Expected: all commands exit 0. `config.py` has two verified pre-tranche F841
diagnostics and is already outside the Ruff formatter baseline, so the scoped
command ignores only F841 there and the format gate intentionally excludes
that file plus the pre-existing unformatted `app.py`. Do not expand TASK-643
into unrelated baseline cleanup. If the repository environment lacks the
`ruff` module, use the repository's installed `ruff` executable and record
that exact command.

- [ ] **Step 2: Review privacy and ownership sentinels**

Confirm the tests prove:

1. no custom path, server ID, endpoint, label, or serialized policy sentinel reaches logs;
2. unsafe POSIX targets fail before parsing;
3. eligible modes harden before parsing;
4. Windows is reported only as unverified;
5. failed persistence changes neither state, revision, projection, nor screen;
6. stale capability data publishes neither policy nor target status;
7. foreign-thread mutation rejects.

- [ ] **Step 3: Preserve the In Progress status until integrated gates**

Use `backlog task 643 --plain` to confirm the plan and acceptance criteria
still match the implemented code, but leave all criteria unchecked and keep
TASK-643 In Progress. Do not add final Implementation Notes or update the
design status yet. TASK-646's integrated installed-wheel, product-maturity,
static, and full-suite gates provide the shared Definition-of-Done evidence
before all four tasks are reconciled together.
