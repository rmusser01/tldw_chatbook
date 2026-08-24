# Change Review Workspace Opt-In Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Change Review an explicit per-workspace choice, initialize enabled roots asynchronously, and ensure disabled or unavailable state creates no snapshot work.

**Architecture:** Add a typed registry state/CAS seam and one app-owned consent/readiness service. Console turn-context capture asks that service for one immutable admission containing only ready roots plus alias-only skipped-root warnings; Settings reads and toggles through the same service. The existing workspace table remains unchanged, and root snapshot initialization moves from an unowned thread-per-binding hook into a fixed pair of daemon workers fed by a bounded queue. Skipped-root notices use a dedicated transcript-warning path and never enter canonical-root snapshot, retention, revert, or GC state.

**Tech Stack:** Python 3.11+, SQLite, `threading.RLock`, bounded `queue.Queue`, daemon `threading.Thread` workers, Textual 8, pytest with real temporary WorkspaceDB and deterministic `threading.Event` barriers.

---

## File map

- Create `tldw_chatbook/Workspaces/change_review_consent.py`: typed consent/readiness values, app-owned lock, CAS-driven toggle, fixed daemon initializer workers, bounded queue, admission, retry, and bounded shutdown.
- Modify `tldw_chatbook/Workspaces/registry_service.py`: missing-row-disabled typed read, opaque revision, transactional state+revision CAS, and optional binding-added service notification.
- Modify `tldw_chatbook/Workspaces/change_bounds.py`: strict tri-state global capability read and Boolean compatibility wrapper.
- Modify `tldw_chatbook/Workspaces/change_turn_tracker.py`: expose one synchronous initial-snapshot operation for the bounded executor; retain the legacy background wrapper only as a compatibility shim if a live caller remains.
- Modify `tldw_chatbook/Workspaces/__init__.py`: export the typed consent service API.
- Modify `tldw_chatbook/app.py`: construct the consent service beside the workspace registry, attach its binding hook, and dispose it after Console admission stops.
- Modify `tldw_chatbook/Chat/console_turn_context.py`: capture ready Change Review roots separately from alias-only untracked-root notices.
- Modify `tldw_chatbook/UI/Console_Modules/session.py`: replace direct `folder_binding_roots` reads with app-owned consent admission.
- Modify `tldw_chatbook/Chat/console_chat_controller.py`: pass captured untracked-root notices to the bridge without rereading workspace state.
- Modify `tldw_chatbook/Chat/console_agent_bridge.py`: append admission-time alias-only warning markers after the terminal assistant outcome without creating `change_snapshots` rows.
- Modify `tldw_chatbook/UI/Screens/settings_screen.py`: render tri-state capability/consent, revision-bound toggle intent, retention disclosure, readiness, and retry.
- Modify `Tests/Workspaces/test_workspace_registry_service.py`: real-SQLite state/revision/CAS contracts.
- Create `Tests/Workspaces/test_change_review_consent.py`: deterministic consent-lock, initializer, ABA, retry, admission, and disposal tests.
- Modify `Tests/Workspaces/test_change_bounds.py`: global tri-state and no-disabled-snapshot contracts.
- Modify `Tests/Chat/test_console_turn_execution_context.py`: immutable admission capture and no-service default.
- Modify `Tests/Chat/test_console_agent_bridge.py`: skipped-root terminal warning persistence without snapshot work.
- Modify `Tests/UI/test_settings_workspaces_category.py`: opt-in copy, unavailable state, CAS conflict, preparing/failed/retry presentation, and retention disclosure.
- Modify `Tests/Workspaces/test_workspace_create_modal.py` and `Tests/Workspaces/test_console_workspace_create_handler.py` only if their binding-created fixtures need the new app-owned hook injected.
- Modify `Docs/User_Guide/console/agent-runs-and-tools.md`: opt-in behavior, retention, preparing/failed behavior, and non-erasing disable semantics.

All commands below run from this designated worktree with the shared project
interpreter at
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`. Before
the first RED run, execute:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/test_probe_import_provenance.py
```

Expected: PASS and the printed/imported `tldw_chatbook` path resolves beneath
this worktree, not the primary checkout or another editable-install target.

ADR required: yes

ADR path: `backlog/decisions/084-change-review-consent-and-asynchronous-finalization.md`

Reason: The task changes privacy-sensitive retained file-content ownership, workspace consent, concurrent state transitions, and app lifecycle.

---

### Task 1: Pin typed global capability and workspace CAS contracts

**Files:**

- Modify: `tldw_chatbook/Workspaces/change_bounds.py`
- Modify: `tldw_chatbook/Workspaces/registry_service.py`
- Create: `tldw_chatbook/Workspaces/change_review_consent.py`
- Test: `Tests/Workspaces/test_change_bounds.py`
- Test: `Tests/Workspaces/test_workspace_registry_service.py`

- [ ] **Step 1: Write failing global-capability tests**

Add tests that demand strict enabled/disabled/unavailable results:

```python
def test_missing_global_setting_keeps_capability_available(monkeypatch):
    monkeypatch.delenv("TLDW_CHANGE_REVIEW_ENABLED", raising=False)
    monkeypatch.setattr(change_bounds, "_change_review_enabled_setting", lambda: True)
    assert change_bounds.read_change_review_capability().state is ChangeReviewState.ENABLED


@pytest.mark.parametrize("raw", ["maybe", "", object()])
def test_invalid_global_setting_is_unavailable(monkeypatch, raw):
    monkeypatch.setattr(change_bounds, "_change_review_enabled_setting", lambda: raw)
    result = change_bounds.read_change_review_capability()
    assert result.state is ChangeReviewState.UNAVAILABLE
    assert change_bounds.change_review_enabled_globally() is False
```

- [ ] **Step 2: Run the global tests and verify RED**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Workspaces/test_change_bounds.py::test_missing_global_setting_keeps_capability_available Tests/Workspaces/test_change_bounds.py::test_invalid_global_setting_is_unavailable`

Expected: FAIL because `read_change_review_capability` and typed state do not exist and invalid values currently enable tracking.

- [ ] **Step 3: Write failing registry state/revision/CAS tests**

Use a real temporary `WorkspaceDB` and demand the following public shape:

```python
missing = registry.read_change_review_consent("ws")
assert missing.state is ChangeReviewState.DISABLED
assert missing.revision == MISSING_CHANGE_REVIEW_REVISION

enabled = registry.compare_and_set_change_review_consent(
    "ws", expected=missing, enabled=True
)
assert enabled.state is ChangeReviewState.ENABLED

with pytest.raises(ChangeReviewStateConflict):
    registry.compare_and_set_change_review_consent(
        "ws", expected=missing, enabled=False
    )
```

Also freeze `_now_factory`, perform false→true→false, and assert all successful revisions are distinct. Monkeypatch the DB read to raise `sqlite3.Error` and assert `UNAVAILABLE`, never enabled.

- [ ] **Step 4: Run the registry tests and verify RED**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Workspaces/test_workspace_registry_service.py -k change_review`

Expected: FAIL because missing rows currently return true, failures return true, and writes have no CAS contract.

- [ ] **Step 5: Implement the minimal typed state and global reader**

Create these immutable values in `change_review_consent.py`:

```python
class ChangeReviewState(str, Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"


MISSING_CHANGE_REVIEW_REVISION = "missing"


@dataclass(frozen=True, slots=True)
class ChangeReviewConsent:
    state: ChangeReviewState
    revision: str = ""
```

In `change_bounds.py`, accept only actual booleans, `0/1`, and the existing canonical Boolean strings. Missing config still resolves from the supplied `True` default. Any exception or other value returns unavailable; the compatibility wrapper returns true only for `ENABLED`.

- [ ] **Step 6: Implement one transactional registry read/CAS**

`read_change_review_consent` returns the missing sentinel for no row, returns the stored state/revision for a row, and converts SQLite failures to `UNAVAILABLE` with content-free logging. `compare_and_set_change_review_consent` opens one transaction, rereads the row, compares both state and revision, raises `ChangeReviewStateConflict` on mismatch, and inserts/updates with:

```python
new_revision = f"{self._now_factory()}:{uuid4().hex}"
```

Keep `change_review_enabled()` as a compatibility wrapper that is true only for an explicit enabled read. Keep `set_change_review_enabled()` temporarily as a deliberate unconditional administrative/test seam, implemented as one direct transactional reread/write that always assigns a fresh revision. Do not compose a separate read with CAS and do not retry a stale expected value; app/UI code must use the revision-bound API instead.

- [ ] **Step 7: Run the focused tests and verify GREEN**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Workspaces/test_change_bounds.py Tests/Workspaces/test_workspace_registry_service.py -k change_review`

Expected: PASS.

- [ ] **Step 8: Commit the typed persistence boundary**

```bash
git add tldw_chatbook/Workspaces/change_review_consent.py tldw_chatbook/Workspaces/change_bounds.py tldw_chatbook/Workspaces/registry_service.py Tests/Workspaces/test_change_bounds.py Tests/Workspaces/test_workspace_registry_service.py
git commit -m "fix(change-review): require explicit workspace consent"
```

### Task 2: Add the app-owned consent/readiness owner

**Files:**

- Modify: `tldw_chatbook/Workspaces/change_review_consent.py`
- Modify: `tldw_chatbook/Workspaces/change_turn_tracker.py`
- Modify: `tldw_chatbook/Workspaces/registry_service.py`
- Modify: `tldw_chatbook/Workspaces/__init__.py`
- Create: `Tests/Workspaces/test_change_review_consent.py`

- [ ] **Step 1: Write failing admission and disabled-work tests**

Build the service with a real registry and a barrier-controlled initializer. Assert a missing/false/unavailable workspace returns an empty admission and the initializer call list stays empty. After CAS-enabling, assert the first admission returns a `preparing` skipped-root notice and schedules one initializer; repeated admission while preparing schedules no duplicate.

```python
admission = service.admit_turn("ws")
assert admission.ready_roots == ()
assert admission.skipped_roots == ()  # explicitly disabled is not an error
assert initialize_calls == []
```

- [ ] **Step 2: Run and verify RED**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Workspaces/test_change_review_consent.py -k 'admission or disabled'`

Expected: FAIL because the service/readiness API does not exist.

- [ ] **Step 3: Write failing linearization and ABA tests**

Use `threading.Event`, never sleeps:

- hold admission immediately after it acquires the consent lock; prove a disable waits and that the admitted turn may retain the root;
- commit disable first; prove later admission is empty;
- hold initializer completion, disable then re-enable to a new revision, release the old initializer, and prove it cannot publish ready;
- simulate a second registry/service process writing a new revision and prove the old completion is also rejected;
- force CAS failure and prove readiness/admission state is unchanged.

Every wait uses a monotonic timeout and retrieves worker exceptions.

- [ ] **Step 4: Run and verify RED**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Workspaces/test_change_review_consent.py -k 'linear or aba or conflict'`

Expected: FAIL on missing service semantics, not on a timed-out test harness.

- [ ] **Step 5: Implement bounded readiness and synchronous initialization**

Add:

```python
class RootReadinessState(str, Enum):
    PREPARING = "preparing"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ChangeReviewAdmission:
    ready_roots: tuple[str, ...] = ()
    skipped_roots: tuple[SkippedReviewRoot, ...] = ()
```

`ChangeReviewConsentService` owns one `RLock`, two daemon workers, a bounded `queue.Queue(maxsize=32)`, readiness keyed by `(workspace_id, canonical_root)`, and a generation/disposed latch. Under the lock it reads capability and consent, validates current bindings, admits one immutable work item only when the queue has room, returns only exact-revision ready roots, and represents preparing/failed roots with binding-ID aliases. Queue-full marks that root failed with bounded retry copy; it does not block admission or spawn another thread. Retry is accepted only from failed state and cannot enqueue duplicates.

Each worker polls the queue with a short timeout, performs only pure shadow-repo filesystem/Git initialization, and passes its captured generation to completion. Completion acquires the service lock and checks disposed/generation before any registry or database read; a stale generation returns immediately. Shutdown sets disposed and increments generation under the lock, drains and cancels queued items, signals the daemon workers, and joins each only until one shared monotonic deadline. A running blocked initializer may outlive that deadline, but it holds no database/UI/store object and its later completion is generation-rejected before reading the closed registry.

Refactor `change_turn_tracker.py` to expose a synchronous `initialize_shadow_root(root, service)` that returns normally or raises. The consent executor owns background execution; do not start an inner thread.

- [ ] **Step 6: Replace registry's unowned binding thread with the attached service hook**

Give `LocalWorkspaceRegistryService` an optional attached consent service. After a binding transaction succeeds, call `service.binding_added(workspace_id, binding_result)` best-effort. With no attached app owner, binding creation starts no thread. The service itself rechecks explicit enabled state before scheduling.

- [ ] **Step 7: Run tests and verify GREEN**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Workspaces/test_change_review_consent.py Tests/Workspaces/test_workspace_registry_service.py Tests/Workspaces/test_change_bounds.py`

Expected: PASS with no unbounded waits or leaked `change-review-init` threads.

- [ ] **Step 8: Commit the app-owned service core**

```bash
git add tldw_chatbook/Workspaces/change_review_consent.py tldw_chatbook/Workspaces/change_turn_tracker.py tldw_chatbook/Workspaces/registry_service.py tldw_chatbook/Workspaces/__init__.py Tests/Workspaces/test_change_review_consent.py Tests/Workspaces/test_workspace_registry_service.py Tests/Workspaces/test_change_bounds.py
git commit -m "feat(change-review): own workspace readiness lifecycle"
```

### Task 3: Wire immutable turn admission and honest skipped-root results

**Files:**

- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/Chat/console_turn_context.py`
- Modify: `tldw_chatbook/UI/Console_Modules/session.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Test: `Tests/Chat/test_console_turn_execution_context.py`
- Test: `Tests/Chat/test_console_agent_bridge.py`
- Test: `Tests/Workspaces/test_change_bounds.py`

- [ ] **Step 1: Write failing immutable-admission tests**

Assert `_build_console_turn_execution_context` calls the app-owned service once and captures both ready roots and immutable alias/reason notices. Mutating the source admission after capture must not change the context. With no service, capture no Change Review roots; it must not fall back to the registry or CWD.

- [ ] **Step 2: Run and verify RED**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_turn_execution_context.py -k change_review`

Expected: FAIL because turn context still calls `folder_binding_roots` directly and has no skipped-root field.

- [ ] **Step 3: Write failing terminal-warning tests**

Call `run_reply` with no ready roots and one `SkippedReviewRoot(alias="folder-a", reason="Preparing change history")`. Complete the fake run and assert one dedicated transcript warning marker contains the alias and reason, starts no baseline thread, creates no `change_snapshots` row, and never adds `folder-a` to `roots_with_change_snapshots()`. Repeat for failed readiness. Explicit disabled admission supplies no skipped warning and creates no review state.

- [ ] **Step 4: Run and verify RED**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_agent_bridge.py -k 'change_review and skipped'`

Expected: FAIL because bridge arguments and terminal record merge do not exist.

- [ ] **Step 5: Implement app wiring and immutable context**

Construct `ChangeReviewConsentService` immediately after the registry is ready, attach it to the registry, and expose it as `app.change_review_consent_service`. Extend `ConsoleTurnExecutionContext` with an immutable `change_review_skipped_roots` tuple. Session context capture calls `service.admit_turn(workspace_id)` once; the controller forwards those captured values to the bridge.

- [ ] **Step 6: Implement the alias-only warning path**

After the terminal assistant outcome, the bridge appends one bounded TOOL warning message per skipped alias/reason. It does not construct `TurnChangeRecord`, call `record_change_snapshot`, create a `TurnHandle`, wait, or take B/E snapshots for those notices. A mixed workspace may persist real ready-root results while independently appending skipped-root warnings. Add a retention regression proving alias-only notices never enter canonical-root GC/reset inputs.

- [ ] **Step 7: Add and verify app shutdown ownership**

Stop Console runtime admission first, then call the consent service's bounded `shutdown()` before closing `local_workspace_db`. Add a barrier-controlled test that leaves one daemon initializer blocked, proves shutdown returns by the shared deadline, closes the database safely, releases the worker, and proves late completion rejects its generation before any registry read or readiness publication.

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Chat/test_console_turn_execution_context.py Tests/Chat/test_console_agent_bridge.py Tests/Workspaces/test_change_review_consent.py -k change_review`

Expected: PASS.

- [ ] **Step 8: Commit runtime admission wiring**

```bash
git add tldw_chatbook/app.py tldw_chatbook/Chat/console_turn_context.py tldw_chatbook/UI/Console_Modules/session.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_agent_bridge.py Tests/Chat/test_console_turn_execution_context.py Tests/Chat/test_console_agent_bridge.py Tests/Workspaces/test_change_review_consent.py
git commit -m "fix(console): admit only ready review roots"
```

### Task 4: Make Settings revision-safe and disclose retention

**Files:**

- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `Tests/UI/test_settings_workspaces_category.py`

- [ ] **Step 1: Rewrite the old default-on test as RED opt-in coverage**

Demand that a new workspace renders `Tracking disabled`, offers `Enable change review`, and persists explicit enabled only after the button press. Assert the exact disclosure includes shadow Git, application data, file contents, the configured 30-day default, and that disabling does not erase existing history.

- [ ] **Step 2: Add RED unavailable and stale-toggle tests**

For global or registry unavailable state, assert no toggle exists and the copy says state cannot be read. Render an enabled state, mutate it through a second registry before pressing the stale button, and assert the UI reports a conflict and refreshes without inversion.

- [ ] **Step 3: Add RED readiness/retry tests**

Use a fake consent service returning preparing, failed, and ready snapshots. Preparing copy must not imply chat is blocked. Failed state offers one retry button; pressing it calls `retry_failed_roots(workspace_id)` once and refreshes. Ready state has no retry.

- [ ] **Step 4: Run and verify RED**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_settings_workspaces_category.py -k change_review`

Expected: FAIL because Settings uses Boolean reads, rereads at click time, and has no retention/readiness copy.

- [ ] **Step 5: Implement the minimal Settings projection**

Render from one service snapshot. Stash the exact `ChangeReviewConsent` on the toggle button; the handler passes it to `toggle(workspace_id, expected=...)`. Conflict/unavailable outcomes set a bounded result message and refresh. The toggle remains absent for Git-unavailable, global-disabled, and unavailable states. Add the failed-root retry action without exposing absolute paths in diagnostics.

- [ ] **Step 6: Run and verify GREEN**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_settings_workspaces_category.py -k change_review`

Expected: PASS.

- [ ] **Step 7: Commit Settings behavior**

```bash
git add tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_settings_workspaces_category.py
git commit -m "feat(settings): disclose Change Review opt-in retention"
```

### Task 5: Update documentation and run the focused regression gate

**Files:**

- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md`
- Modify: `backlog/tasks/task-19501 - Make-Change-Review-opt-in-per-workspace.md`

- [ ] **Step 1: Update the user guide**

Document per-workspace opt-in, global capability behavior, background preparing/failed/retry states, 30-day default file-content retention, non-retroactive disable, and the fact that chat/tools continue when review is unavailable.

- [ ] **Step 2: Run the complete focused gate**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Workspaces/test_workspace_registry_service.py \
  Tests/Workspaces/test_change_bounds.py \
  Tests/Workspaces/test_change_review_consent.py \
  Tests/Workspaces/test_workspace_create_modal.py \
  Tests/Workspaces/test_console_workspace_create_handler.py \
  Tests/Chat/test_console_turn_execution_context.py \
  Tests/Chat/test_console_agent_bridge.py \
  Tests/UI/test_settings_workspaces_category.py
```

Expected: PASS with no new warnings; no queued initializer work survives shutdown, and any deliberately blocked daemon worker is generation-inert after the bounded shutdown returns.

- [ ] **Step 3: Run static/diff checks**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/Workspaces/change_review_consent.py tldw_chatbook/Workspaces/change_bounds.py tldw_chatbook/Workspaces/registry_service.py tldw_chatbook/Workspaces/change_turn_tracker.py tldw_chatbook/Chat/console_turn_context.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/UI/Console_Modules/session.py tldw_chatbook/UI/Screens/settings_screen.py Tests/Workspaces/test_change_review_consent.py`

Run: `git diff --check`

Expected: both exit 0.

- [ ] **Step 4: Mutation-check the load-bearing guards**

Temporarily neutralize, one at a time, missing-row-disabled mapping, unavailable fail-off, CAS revision comparison, exact-revision initializer completion, and disabled admission. Run the exact protecting node for each and verify it fails for the intended assertion; restore the production line after every mutation and rerun green.

- [ ] **Step 5: Complete task hygiene only after evidence is green**

Update every acceptance criterion to checked, add concise implementation notes including ADR-084 and modified files, then set TASK-19501 to Done. Do not use `backlog task edit` for this five-digit ID; edit the source file directly per the repository's recorded CLI defect.

- [ ] **Step 6: Commit documentation and closeout**

```bash
git add Docs/User_Guide/console/agent-runs-and-tools.md 'backlog/tasks/task-19501 - Make-Change-Review-opt-in-per-workspace.md'
git commit -m "docs(change-review): explain workspace opt-in lifecycle"
```
