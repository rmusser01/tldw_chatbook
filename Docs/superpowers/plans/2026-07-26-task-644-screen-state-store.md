# Screen State Store Implementation Plan (TASK-644)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `TldwCli._screen_states` with an owner-thread-affine, memory-only `ScreenStateStore` keyed by resolved canonical tabs and scoped to the authoritative runtime identity.

**Architecture:** Add one narrow navigation-layer owner that stores private snapshot envelopes containing a canonical route, an outer mapping copy, and runtime identity. Navigation continues constructing fresh screens and preserves its existing flush/save/restore/context/switch order. Screens remain responsible for detached domain values; the store does not deep-copy large histories, while consumers that retain nested mutable values copy them explicitly.

**Tech Stack:** Python 3.11+, dataclasses, `collections.abc.Mapping`, `threading.get_ident`, Textual screen navigation, pytest/pytest-asyncio.

**Backlog:** [TASK-644](../../../backlog/tasks/task-644%20-%20Move-cross-visit-screen-snapshots-behind-an-in-memory-owner.md)

**Specification:** [Application Session State Ownership Design](../specs/2026-07-26-application-session-state-ownership-design.md)

**Depends on:** TASK-643

**ADR required:** yes

**ADR path:** `backlog/decisions/026-application-session-state-ownership.md`

**Reason:** ADR-026 already defines application-lifetime view-state ownership, canonical keying, runtime invalidation, and memory-only persistence.

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

- Create `tldw_chatbook/UI/Navigation/screen_state_store.py`: `RuntimeIdentity`, private envelope, and `ScreenStateStore`.
- Modify `tldw_chatbook/UI/Navigation/__init__.py`: export only the intended store and identity types if this package already exposes public navigation seams.
- Modify `tldw_chatbook/runtime_policy/bootstrap.py`: remove the transitional domain-dictionary snapshot helpers after navigation migrates.
- Modify `tldw_chatbook/app.py`: construct the store, route navigation through canonical keys, and remove duplicate startup `current_tab` publication.
- Modify `tldw_chatbook/UI/Screens/home_screen.py`: obtain recent-work state from the owner on the app thread.
- Modify `tldw_chatbook/UI/Screens/workflows_screen.py` and `tldw_chatbook/UI/Screens/schedules_screen.py`: capture recent-work state on the owner thread before launching their threaded workers.
- Modify `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py`: query the store from its app-loop async path.
- Modify `tldw_chatbook/UI/Screens/settings_screen.py`: retain explicit deep-copy-on-restore behavior for nested drafts.
- Modify comments in `tldw_chatbook/UI/Screens/chat_screen.py` and affected tests that still describe `_screen_states`.
- Create `Tests/UI/test_screen_state_store.py`: unit contract, thread-affinity, redaction, and shallow-copy cost tests.
- Create `Tests/UI/test_screen_state_full_app.py`: canonical aliases, startup, ordering, fresh construction, failure behavior, context precedence, and nested Settings draft isolation through the normal production `TldwCli` and actual destination screens.
- Modify `Tests/test_application_state_ownership.py`: prohibit app/consumer `_screen_states` ownership.

Do not modify, run, or cite the retired navigation, Settings,
Console-internals, Home, destination-shell, or Console-live-work harness
suites. Application behavior belongs in `test_screen_state_full_app.py`;
app-independent store and AST behavior is tested directly.

## Task 1: Implement the Memory-Only Snapshot Owner

**Files:**

- Create: `tldw_chatbook/UI/Navigation/screen_state_store.py`
- Modify: `tldw_chatbook/UI/Navigation/__init__.py`
- Create: `Tests/UI/test_screen_state_store.py`

- [ ] **Step 1: Write failing store contract tests**

Cover:

```python
def test_save_and_restore_copy_only_the_outer_mapping() -> None:
    nested = {"history": ["large", "payload"]}
    original = {"selected": "row-1", "nested": nested}
    store = ScreenStateStore()
    identity = RuntimeIdentity(active_source="local", active_server_id=None)

    store.save("chat", original, identity)
    original["selected"] = "changed-after-save"
    restored = store.restore("chat", identity)

    assert restored == {"selected": "row-1", "nested": nested}
    assert restored is not original
    assert restored["nested"] is nested
    restored["selected"] = "consumer-change"
    assert store.restore("chat", identity)["selected"] == "row-1"


def test_server_identity_mismatch_discards_snapshot() -> None:
    store = ScreenStateStore()
    store.save(
        "library",
        {"selected": "n-1"},
        RuntimeIdentity("server", "server-a"),
    )

    assert store.restore(
        "library",
        RuntimeIdentity("server", "server-b"),
    ) is None
    assert store.has_snapshots(RuntimeIdentity("server", "server-a")) is False
```

Also assert: empty/non-string canonical keys reject; non-mapping snapshots reject without mutation; source mismatch discards; local mode ignores stale server metadata by normalizing it to `None`; corrupt envelopes discard; `has_snapshots()` lazily removes incompatible entries; `discard()` is idempotent; producer and restored outer mappings cannot mutate backing state; off-owner `save`, `restore`, `discard`, and `has_snapshots` all raise; neither snapshot content nor a unique sentinel appears in failure logs.

- [ ] **Step 2: Run tests and verify the class is absent**

Run:

```bash
pytest Tests/UI/test_screen_state_store.py -q
```

Expected: FAIL on import because the store does not exist.

- [ ] **Step 3: Implement the store**

Use a private envelope and exact runtime compatibility:

```python
@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    active_source: str
    active_server_id: str | None = None

    @classmethod
    def from_state(cls, state: RuntimeSourceState) -> "RuntimeIdentity":
        source = "server" if state.active_source == "server" else "local"
        return cls(
            active_source=source,
            active_server_id=state.active_server_id if source == "server" else None,
        )


@dataclass(slots=True)
class _SnapshotEnvelope:
    canonical_route: str
    snapshot: dict[str, Any]
    runtime_identity: RuntimeIdentity


class ScreenStateStore:
    def __init__(self) -> None:
        self._owner_thread_id = threading.get_ident()
        self._entries: dict[str, _SnapshotEnvelope] = {}

    def save(
        self,
        route: str,
        snapshot: Mapping[str, Any],
        runtime_identity: RuntimeIdentity,
    ) -> None:
        self._assert_owner_thread()
        canonical_route = self._canonical_key(route)
        if not isinstance(snapshot, Mapping):
            raise TypeError("screen snapshot must be a mapping")
        self._entries[canonical_route] = _SnapshotEnvelope(
            canonical_route,
            dict(snapshot),
            runtime_identity,
        )

    def restore(
        self,
        route: str,
        runtime_identity: RuntimeIdentity,
    ) -> dict[str, Any] | None:
        self._assert_owner_thread()
        canonical_route = self._canonical_key(route)
        envelope = self._entries.get(canonical_route)
        if not self._compatible(envelope, runtime_identity):
            self._entries.pop(canonical_route, None)
            return None
        return dict(envelope.snapshot)
```

Implement `discard()` and `has_snapshots()` with the same owner check. `has_snapshots()` walks a tuple of keys, removes corrupt/incompatible envelopes, and then returns `bool(self._entries)`. Do not add serialization, a backing-map property, deep copy, route resolution, logging of snapshot objects, or a second runtime-policy representation.

- [ ] **Step 4: Run the store tests**

Run:

```bash
pytest Tests/UI/test_screen_state_store.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the owner**

```bash
git add tldw_chatbook/UI/Navigation/screen_state_store.py tldw_chatbook/UI/Navigation/__init__.py Tests/UI/test_screen_state_store.py
git commit -m "feat(navigation): add runtime-scoped screen state owner (task-644)"
```

## Task 2: Migrate Navigation Without Changing Its Lifecycle

**Files:**

- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/runtime_policy/bootstrap.py`
- Create: `Tests/UI/test_screen_state_full_app.py`
- Modify: `Tests/RuntimePolicy/test_runtime_policy_bootstrap.py`

- [ ] **Step 1: Write failing canonical-key and navigation-order tests**

Mount the real `TldwCli` and production screens to prove:

- outgoing save uses the existing canonical `current_tab`;
- incoming restore uses `current_tab_value` from `resolve_screen_target()`;
- `ccp` and `personas` share the `personas` snapshot;
- `notes` and the canonical Library route share the Library snapshot while explicit Notes context wins after restore;
- routes with distinct canonical tabs do not share state even if their screen-owned `screen_name` strings match;
- an unregistered outgoing screen name with no `current_tab` skips save;
- `save_state()` failure/non-mapping continues navigation;
- `restore_state()` failure discards only that snapshot and continues;
- flush `False` or exception aborts before save/construction;
- each navigation still constructs a fresh screen;
- explicit Library, Settings, and Watchlists context applies after restore.

Use observers attached only to actual production screen instances when a
failure or ordering seam cannot be observed from final mounted state. Do not
construct a substitute app or screen.

- [ ] **Step 2: Run the focused navigation tests**

Run:

```bash
pytest Tests/UI/test_screen_state_full_app.py -q -k "alias or context or flush or fresh or restore_failure"
```

Expected: at least canonical-alias and store-ownership cases FAIL against `_screen_states`.

- [ ] **Step 3: Construct the owner and centralize runtime identity**

In `TldwCli.__init__`, after `runtime_policy` construction:

```python
self.screen_state_store = ScreenStateStore()
```

Add a small private app helper:

```python
def _current_runtime_identity(self) -> RuntimeIdentity:
    return RuntimeIdentity.from_state(self.runtime_policy.state)
```

This helper reads the immutable authoritative state. It must not use compatibility projections.

- [ ] **Step 4: Preserve the navigation sequence through the store**

In `handle_screen_navigation()`:

1. leave pending-work flush and veto handling first;
2. derive outgoing key from `self.current_tab`; if empty, resolve a registered `current_screen.screen_name` once, otherwise skip;
3. call `save_state()`, require `Mapping`, and call `screen_state_store.save(outgoing_key, state, identity)`;
4. construct the fresh destination;
5. call `restore(current_tab_value, identity)` and offer the returned copy to `restore_state()`;
6. discard the incoming key if restore raises;
7. apply navigation context;
8. switch and publish `current_tab_value`.

Replace exception messages such as `f"... {e}"` with route plus exception category only. Never log the state mapping or exception representation.

Delete `add_runtime_policy_snapshot()`, `reconcile_saved_screen_state()`, and their tests only after no production caller remains.

- [ ] **Step 5: Run navigation and runtime-policy regressions**

Run:

```bash
pytest Tests/UI/test_screen_state_full_app.py Tests/RuntimePolicy/test_runtime_policy_bootstrap.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit navigation migration**

```bash
git add tldw_chatbook/app.py tldw_chatbook/runtime_policy/bootstrap.py Tests/UI/test_screen_state_full_app.py Tests/RuntimePolicy/test_runtime_policy_bootstrap.py
git commit -m "refactor(navigation): move snapshots behind canonical owner (task-644)"
```

## Task 3: Keep Startup Canonical

**Files:**

- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/UI/test_screen_state_full_app.py`

- [ ] **Step 1: Write mounted alias-startup regressions**

Parameterize startup defaults `ccp`, `notes`, and `customize`. Run `_push_initial_screen()` followed by the real `_post_mount_setup()`/deferred callback drain and assert `current_tab` remains the resolver's canonical tab. Navigate away and back and assert the same canonical key restores the snapshot.

- [ ] **Step 2: Run and verify the current duplicate writers fail**

Run:

```bash
pytest Tests/UI/test_screen_state_full_app.py -q -k "startup or alias"
```

Expected: FAIL because `_set_initial_tab()` and the final `_post_mount_setup()` assignment publish the unresolved configured route.

- [ ] **Step 3: Remove duplicate startup publication**

Delete `_set_initial_tab()`, `self.call_later(self._set_initial_tab)`, and the final raw `self.current_tab = self._resolve_initial_shell_route()` assignment. Keep `_push_initial_screen()` as the sole startup writer. Keep a local `initial_tab` only where one-time post-mount branching needs the configured route.

- [ ] **Step 4: Run startup and watcher tests**

Run:

```bash
pytest Tests/UI/test_screen_state_full_app.py -q -k "startup or alias"
```

Expected: PASS.

- [ ] **Step 5: Commit canonical startup**

```bash
git add tldw_chatbook/app.py Tests/UI/test_screen_state_full_app.py
git commit -m "fix(navigation): preserve canonical startup tab identity (task-644)"
```

## Task 4: Migrate Recent-Work Consumers Without Crossing Threads

**Files:**

- Modify: `tldw_chatbook/UI/Screens/home_screen.py`
- Modify: `tldw_chatbook/UI/Screens/workflows_screen.py`
- Modify: `tldw_chatbook/UI/Screens/schedules_screen.py`
- Modify: `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py`
- Modify: `Tests/test_application_state_ownership.py`

- [ ] **Step 1: Add failing recent-work owner tests**

Add a direct AST ownership check proving Home, Workflows, Schedules, and the
Schedules workbench call `screen_state_store.has_snapshots()` only from the
known app-loop methods. In particular, the threaded
`_refresh_latest_console_context` workers must receive a captured boolean and
must not touch the store.

- [ ] **Step 2: Run tests to verify direct dictionary assumptions**

Run:

```bash
pytest Tests/test_application_state_ownership.py -q -k recent_work_consumers
```

Expected: FAIL until fixtures and consumers use the owner.

- [ ] **Step 3: Add one screen-local helper pattern**

On app-loop code paths:

```python
identity = RuntimeIdentity.from_state(self.app_instance.runtime_policy.state)
has_recent_work = self.app_instance.screen_state_store.has_snapshots(identity)
```

For `WorkflowsScreen` and `SchedulesScreen`, compute and store/pass the boolean in `on_mount()` before invoking `_refresh_latest_console_context()`. Their `thread=True` worker may read that immutable boolean but must not call `has_snapshots()`. The async Schedules workbench remains on the app loop and may call the store directly.

- [ ] **Step 4: Run recent-work suites**

Run:

```bash
pytest Tests/test_application_state_ownership.py -q -k recent_work_consumers
```

Expected: PASS.

- [ ] **Step 5: Commit consumer migration**

```bash
git add tldw_chatbook/UI/Screens/home_screen.py tldw_chatbook/UI/Screens/workflows_screen.py tldw_chatbook/UI/Screens/schedules_screen.py tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py Tests/test_application_state_ownership.py
git commit -m "refactor(ui): read recent snapshots through the owner (task-644)"
```

## Task 5: Prove Nested Ownership and Guard the Boundary

**Files:**

- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_screen_state_store.py`
- Modify: `Tests/UI/test_screen_state_full_app.py`
- Modify: `Tests/test_application_state_ownership.py`

- [ ] **Step 1: Add nested Settings and large Console sentinels**

Settings: on a mounted production Settings screen, restore a draft containing
nested `originals`/`values`, mutate the restored screen's draft, and assert the
producer remains detached.

Console: put a nested history object whose `__deepcopy__()` raises into a snapshot, save/restore through the store, and assert the store never invokes deep copy while returning distinct outer mappings. This pins the cost boundary without using timing assertions.

- [ ] **Step 2: Extend the AST ownership guard**

Reject:

- any `self._screen_states` assignment/read in production;
- `getattr(..., "_screen_states", ...)` in production consumers;
- insertion/removal of the string key `runtime_policy_snapshot`;
- access to `ScreenStateStore._entries` outside its defining module.

Allow test-only white-box corruption setup. Update comments that claim `_screen_states` is the continuity owner.

- [ ] **Step 3: Run the focused tests**

```bash
pytest Tests/UI/test_screen_state_store.py Tests/UI/test_screen_state_full_app.py Tests/test_application_state_ownership.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit nested ownership and guards**

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_screen_state_store.py Tests/UI/test_screen_state_full_app.py Tests/test_application_state_ownership.py
git commit -m "test(state): guard snapshot ownership and copy boundaries (task-644)"
```

## Task 6: Verify TASK-644 and Hold Final Reconciliation

**Files:**

- No production or Backlog status changes expected; fix only verified
  regressions within TASK-644 acceptance criteria.

- [ ] **Step 1: Run focused and mounted verification**

```bash
pytest Tests/UI/test_screen_state_store.py Tests/UI/test_screen_state_full_app.py Tests/test_application_state_ownership.py -q
python -m compileall -q tldw_chatbook/UI/Navigation tldw_chatbook/UI/Screens
python -m ruff check tldw_chatbook/UI/Navigation/screen_state_store.py tldw_chatbook/UI/Navigation/__init__.py tldw_chatbook/app.py tldw_chatbook/UI/Screens/home_screen.py tldw_chatbook/UI/Screens/workflows_screen.py tldw_chatbook/UI/Screens/schedules_screen.py tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py Tests/UI/test_screen_state_store.py Tests/UI/test_screen_state_full_app.py Tests/test_application_state_ownership.py
python -m ruff check --ignore F841 tldw_chatbook/UI/Screens/settings_screen.py
python -m ruff format --check tldw_chatbook/UI/Navigation/screen_state_store.py tldw_chatbook/UI/Navigation/__init__.py tldw_chatbook/UI/Screens/home_screen.py tldw_chatbook/UI/Screens/workflows_screen.py tldw_chatbook/UI/Screens/schedules_screen.py tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py Tests/UI/test_screen_state_store.py Tests/UI/test_screen_state_full_app.py Tests/test_application_state_ownership.py
git diff --check
```

Expected: all commands exit 0. `settings_screen.py` has two verified
pre-tranche F841 diagnostics and, with `app.py`, is already outside the Ruff
formatter baseline; ignore only F841 for that file and do not create a
large unrelated formatting diff.

Application behavior in this gate is exercised only through the normal
production `TldwCli` and actual destination screens. The retired navigation,
Settings, Console-internals, Home, destination-shell, and Console-live-work
harness suites are intentionally excluded because they construct surrogate
applications.

- [ ] **Step 2: Self-review all five acceptance criteria**

Confirm fresh evidence for memory-only behavior, canonical aliases, runtime invalidation, startup canonicalization, navigation ordering, explicit context precedence, nested Settings copying, large Console shallow storage, owner-thread enforcement, sentinel redaction, and absence of `_screen_states`.

- [ ] **Step 3: Preserve the In Progress status until integrated gates**

Use `backlog task 644 --plain` to confirm the plan and acceptance criteria
still match the implemented code, but leave all criteria unchecked and keep
TASK-644 In Progress. Do not add final Implementation Notes or update the
design status yet. Final reconciliation waits for TASK-646's shared
installed-wheel, product-maturity, static, and full-suite evidence.
