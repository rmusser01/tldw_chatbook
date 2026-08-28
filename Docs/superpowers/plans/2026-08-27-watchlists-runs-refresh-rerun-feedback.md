# Watchlists Runs Refresh and Re-run Feedback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Apply superpowers:test-driven-development for every behavior change and superpowers:verification-before-completion before claiming success.

**Goal:** Make the Watchlists Runs toolbar reload authoritative rows/detail and make Re-run source visibly busy, honestly reported, duplicate-safe, and automatically refreshed.

**Architecture:** Keep `RunsPane` as a presentation/message surface and keep `WatchlistsCollectionsScreen` as the lifecycle, worker, notification, and publication owner. Reuse the existing controller/scope `launch_run` seam, Check-now outcome helpers, canonical Watchlists ids, `wc_runs`, and `wc_run_detail`; add only the backend-specific job forwarding, staged generation-checked refresh, and narrow Re-run-origin presentation state required by the approved design.

**Tech Stack:** Python 3.11+, Textual 8.x messages/reactives/workers, existing Watchlists controller/scope services, pytest/pytest-asyncio, Ruff.

**ADR required:** no
**ADR path:** `backlog/decisions/042-watchlists-reader-first-ia.md`
**Reason:** ADR-042 already owns the long-lived Watchlists screen/pane boundaries. This repair changes no persistence, service ownership, backend API, dependency, or navigation architecture.

---

### Task 1: Give RunsPane typed Refresh/Re-run intent and in-place busy presentation

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/runs_pane.py:18-63, 440-510`
- Modify: `Tests/Watchlists/test_watchlists_runs_pane.py:8-32, 127-193`

- [ ] **Step 1: Write failing message-contract tests**

Extend `RunsPaneHarness` to capture a new `RefreshRunsRequested` message and the complete Re-run request rather than only `source_id`. Add focused tests proving:

```python
async def test_refresh_posts_authoritative_reload_request():
    pane.query_one("#runs-refresh-button", Button).press()
    await pilot.pause()
    assert ("refresh_runs_requested",) in app.captured_messages

async def test_local_rerun_posts_source_identity_and_inert_name():
    # selected row: backend=local, source_id=5, source_title="Feed [one]"
    assert captured == (
        "rerun_run_requested", "local", 5, "Feed [one]"
    )

async def test_server_rerun_posts_job_identity():
    # selected row: backend=server, job_id="job-7", source_id absent
    assert captured == (
        "rerun_run_requested", "server", "job-7", "Job job-7"
    )
```

Also add tests that local rows without `source_id` and server rows without `job_id` disable Re-run.

- [ ] **Step 2: Run the pane RED nodes**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Watchlists/test_watchlists_runs_pane.py::test_refresh_posts_authoritative_reload_request \
  Tests/Watchlists/test_watchlists_runs_pane.py::test_local_rerun_posts_source_identity_and_inert_name \
  Tests/Watchlists/test_watchlists_runs_pane.py::test_server_rerun_posts_job_identity \
  Tests/Watchlists/test_watchlists_runs_pane.py::test_rerun_requires_the_backend_specific_identity
```

Expected: FAIL because Refresh does not post a message, Re-run carries only `source_id`, and identity eligibility is not backend-aware.

- [ ] **Step 3: Implement the smallest typed intent contract**

Add these message shapes:

```python
class RefreshRunsRequested(Message):
    """Posted when the user requests an authoritative Runs reload."""


class RerunRunRequested(Message):
    """Posted when the user requests re-running a source/job."""

    def __init__(self, *, runtime_backend: str, target_id: Any, name: str) -> None:
        self.runtime_backend = runtime_backend
        self.target_id = target_id
        self.name = name
        super().__init__()
```

Add one pane helper that chooses `(backend, target_id, name)` from the selected run: `source_id` for local and `job_id` for server. Local display uses the existing inert `_run_identity()` result; server display uses `source_title` when present and otherwise the explicit inert fallback `f"Job {job_id}"`. Make Refresh post `RefreshRunsRequested()` and Re-run post the complete typed request.

- [ ] **Step 4: Write failing in-place busy-state tests**

Add tests proving:

- `busy_operation_keys` is a non-recomposing reactive/attribute update;
- a Re-run-origin key paints disabled `Re-running...`;
- a shared Check-now-only local key paints disabled `Checking...`;
- clearing the key restores `Re-run source` without replacing `#runs-table`;
- busy state for another target does not disable the selected target.

Capture the original table object and assert identity after each busy-state update.

- [ ] **Step 5: Run the busy-state RED nodes**

Run the new nodeids from `Tests/Watchlists/test_watchlists_runs_pane.py`.

Expected: FAIL because `RunsPane` has no operation-state inputs and `_update_action_buttons()` only checks selection/status.

- [ ] **Step 6: Implement in-place busy painting**

Add non-recomposing `selected_operation_key`, `busy_operation_keys`, and `rerun_operation_keys` inputs. `selected_operation_key` is supplied by the screen after selection and when it builds/rebuilds the pane; the pane never constructs the canonical concurrency key. Update `_update_action_buttons()` so it:

```python
target = self._rerun_target(self.selected_run)
operation_key = self.selected_operation_key
is_busy = bool(operation_key and operation_key in self.busy_operation_keys)
is_rerun = bool(operation_key and operation_key in self.rerun_operation_keys)
rerun_button.disabled = target is None or is_busy
rerun_button.label = (
    "Re-running..." if is_rerun else "Checking..." if is_busy else "Re-run source"
)
```

The screen sets `selected_operation_key` to `None` when the selected row lacks its backend-specific launch identity. Add watchers for all three presentation inputs that call `_update_action_buttons()` in place. Do not make `runs`, selection, or the table recompose when these values change. Screen seeding and rebuild persistence are implemented and tested with the screen-owned lifecycle in Task 4.

- [ ] **Step 7: Run the complete pane file**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/Watchlists/test_watchlists_runs_pane.py
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/runs_pane.py \
  Tests/Watchlists/test_watchlists_runs_pane.py
git commit -m "feat: add Runs toolbar intent and busy state"
```

### Task 2: Forward backend-specific Re-run identity through the existing controller

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py:186-191`
- Modify: `Tests/Watchlists/test_watchlists_backend_controller.py:85-95, 178-188`
- Verify: `tldw_chatbook/Subscriptions/watchlist_scope_service.py:743-780`
- Verify: `Tests/Watchlists/test_watchlist_scope_service.py:52-60`

- [ ] **Step 1: Write failing local/server forwarding tests**

Add parameterized controller coverage:

```python
@pytest.mark.parametrize(
    ("backend", "source_id", "job_id"),
    [("local", 5, None), ("server", None, "job-7")],
)
async def test_launch_run_forwards_backend_specific_identity(
    backend, source_id, job_id
):
    result = await controller.launch_run(
        runtime_backend=backend,
        source_id=source_id,
        job_id=job_id,
    )
    scope_service.launch_run.assert_awaited_once_with(
        runtime_backend=backend,
        source_id=source_id,
        job_id=job_id,
    )
```

- [ ] **Step 2: Run the forwarding RED node**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Watchlists/test_watchlists_backend_controller.py::test_launch_run_forwards_backend_specific_identity
```

Expected: FAIL because `WatchlistsBackendController.launch_run()` does not accept or forward `job_id`.

- [ ] **Step 3: Implement the signature/forwarding correction**

Change only the existing method:

```python
async def launch_run(
    self,
    *,
    runtime_backend: str | None = None,
    source_id: Any = None,
    job_id: Any = None,
) -> dict[str, Any]:
    backend = self._normalize_backend(runtime_backend)
    result = await self._maybe_await(
        self.scope_service.launch_run(
            runtime_backend=backend,
            source_id=source_id,
            job_id=job_id,
        )
    )
    return dict(result)
```

Do not change `WatchlistScopeService.launch_run()`; it already accepts both ids and routes them to local/server services.

- [ ] **Step 4: Run focused controller/scope tests**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Watchlists/test_watchlists_backend_controller.py \
  Tests/Watchlists/test_watchlist_scope_service.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py \
  Tests/Watchlists/test_watchlists_backend_controller.py
git commit -m "fix: forward Watchlists run job identity"
```

### Task 3: Add generation-checked authoritative Runs refresh

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py:820-855, 5638-5840, 6838-6877`
- Modify: `Tests/UI/test_watchlists_run_detail.py:300-470`
- Modify or create focused cases in: `Tests/UI/test_watchlists_destination_shell.py:1718-1842`

- [ ] **Step 1: Write failing mounted refresh tests**

Using event-gated async fakes (never timing sleeps), add mounted tests for these independent outcomes:

1. Refresh replaces rows with changed backend rows and reloads the selected row's detail by id.
2. A selected run absent from the newest 100 is resolved via `get_run()`, appended after the page, and remains selected.
3. Local `KeyError` and server `APIResponseError(404, ...)` from that pin lookup clear rows' selection and detail.
4. List failure or non-404 pin failure retains `_loaded_runs`, `selected_run`, pane rows, pane selection, detail items/logs/note, and action state.
5. Switching `runtime_backend` before the gate is released prevents old rows/detail from publishing.
6. A second accepted refresh advances the generation immediately; when the first completes last, only the second publishes.
7. Selection clear and detail reload are dispatched through `wc_run_detail`, so an older detail request cannot repopulate mirrors.

The fake list/get/detail services should expose `asyncio.Event` gates and call recordings. For not-found classification, import `APIResponseError` from `tldw_chatbook.tldw_api.exceptions`.

- [ ] **Step 2: Run each new refresh node and confirm RED**

Run each nodeid separately while developing, then run the refresh subset together:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_watchlists_run_detail.py -k "refresh or selected_run_outside_page or stale_backend or superseded"
```

Expected: FAIL because Refresh never reaches the screen, `_load_runs()` publishes incrementally, and no Runs generation exists.

- [ ] **Step 3: Add refresh state and the message dispatcher**

In `__init__`, add only:

```python
self._runs_refresh_generation = 0
```

Import and handle `RefreshRunsRequested`. Centralize toolbar and Re-run completion on one dispatcher:

```python
def _request_runs_refresh(self) -> None:
    self._runs_refresh_generation += 1
    generation = self._runs_refresh_generation
    backend = self.runtime_backend
    selected = self.selected_run
    selected_id = selected.get("id") if selected else None
    self.run_worker(
        self._refresh_runs_snapshot(
            runtime_backend=backend,
            selected_id=selected_id,
            generation=generation,
        ),
        exclusive=True,
        group="wc_runs",
    )
```

`handle_refresh_runs_requested()` stops the message and calls this dispatcher.

- [ ] **Step 4: Implement staged acquisition and narrow not-found classification**

Add helpers with explicit inputs rather than rereading mutable screen state mid-worker:

```python
@staticmethod
def _run_refresh_not_found(exc: Exception, backend: str) -> bool:
    return (
        backend == "local" and isinstance(exc, KeyError)
    ) or (
        backend == "server"
        and isinstance(exc, APIResponseError)
        and exc.status_code == 404
    )
```

`_refresh_runs_snapshot()` must:

- stage `list_runs(runtime_backend=backend, limit=100)` into local dicts;
- match `selected_id` against normalized `run["id"]`;
- when absent, call `get_run(runtime_backend=backend, run_id=<raw run id>)`, append a valid pinned row after the page, clear only on the narrow not-found cases, and abort on every other error;
- check both `backend == self.runtime_backend` and `generation == self._runs_refresh_generation` before any publication;
- publish `_loaded_runs`, `selected_run`, pane rows, and pane selection as one synchronous commit;
- schedule `_load_run_detail(candidate)` with `exclusive=True, group="wc_run_detail"`, including `candidate=None`.

Extract the raw id from a normalized `backend:watchlist_run:<id>` using the existing matching convention; do not pass the namespaced UI id to a service expecting a raw run id.

- [ ] **Step 5: Keep initial/deep-link loading intact**

Do not replace the mount/deep-link semantics of `_load_runs()`. Share a small row-publication or raw-id helper if useful, but preserve:

- pending deep-link reconciliation;
- the pre-mount awaited detail load documented at lines 6865-6873;
- the existing failure notification for an initial load.

The explicit toolbar path alone gets snapshot retention and generation supersession.

- [ ] **Step 6: Add the missing grouped detail boundary**

Ensure every user selection/clear that starts run detail does so through:

```python
self.run_worker(
    self._load_run_detail(run),
    exclusive=True,
    group="wc_run_detail",
)
```

Before `_load_run_detail()` publishes mirrors or pane state, confirm the requested run still matches the current selected id/backend. For `None`, clear only if selection is still `None`. This is the publication guard; worker cancellation alone is not evidence.

- [ ] **Step 7: Run focused refresh/detail files**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_watchlists_run_detail.py \
  Tests/UI/test_watchlists_destination_shell.py -k "run or runs"
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  Tests/UI/test_watchlists_run_detail.py \
  Tests/UI/test_watchlists_destination_shell.py
git commit -m "fix: refresh Watchlists runs authoritatively"
```

### Task 4: Share operation identity and give Re-run honest feedback

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py:838-846, 6142-6560`
- Modify: `Tests/UI/test_watchlists_check_now_progress.py:1-430`
- Modify or create focused cases in: `Tests/UI/test_watchlists_destination_shell.py`
- Verify: `Tests/UI/test_watchlists_check_now_skipped.py`
- Verify: `Tests/UI/test_watchlists_check_now_failure.py`

- [ ] **Step 1: Write failing canonical identity/concurrency tests**

Add direct/mounted tests proving:

- a local source row id `local:subscription:5` and local run `source_id=5` derive the same operation key using `build_watchlist_item_id("local", "subscription", 5)`;
- a server Check-now source remains in the `server:watchlist_source:*` namespace;
- a server Re-run derives `build_watchlist_item_id("server", "watchlist_job", job_id)` and never guesses from `source_id`;
- server source id `5` and server job id `5` do not collide;
- Check now blocks Re-run and Re-run blocks Check now for the same local source;
- a duplicate server Re-run for the same job is refused;
- different local sources/server jobs can run concurrently;
- a request captured for an old backend is rejected without launching or painting the current pane busy.
- starting gated Re-run work and forcing the existing section/layout pane rebuild produces a replacement `RunsPane` whose button remains disabled with the same label until cleanup.

Assert the duplicate warning is stated and `markup=False`.

- [ ] **Step 2: Run concurrency RED nodes**

Run the new nodeids in `Tests/UI/test_watchlists_check_now_progress.py` and the Runs integration file.

Expected: FAIL because Check now keys directly on the source row's `id`, Re-run has a separate worker path, and server job identity is not represented.

- [ ] **Step 3: Implement one operation-key authority**

Import `build_watchlist_item_id` and add two narrow helpers with deliberately different server namespaces:

```python
@staticmethod
def _check_operation_key(runtime_backend: str, source_id: Any) -> str:
    kind = "subscription" if runtime_backend == "local" else "watchlist_source"
    return build_watchlist_item_id(runtime_backend, kind, source_id)

@classmethod
def _rerun_operation_key(cls, runtime_backend: str, target_id: Any) -> str:
    if runtime_backend == "local":
        return cls._check_operation_key("local", target_id)
    return build_watchlist_item_id("server", "watchlist_job", target_id)
```

Normalize Check-now source identity to the raw `source_id` when present, otherwise strip the canonical source row id through the existing scope/normalizer convention. Never pass a server source through `_rerun_operation_key()`: the server Sources path and server Runs path intentionally do not cross-deduplicate without a reliable API relationship. Continue using `_checks_in_flight` as the sole concurrency set. Add only `_reruns_in_flight: set[str]` for presentation origin.

Update `_set_check_now_busy()` (or rename it narrowly if clarity requires) to push the selected screen-derived Re-run operation key, shared keys, and Re-run-origin keys into any mounted `RunsPane`, while preserving Sources/Inspector painting. Seed the same three values in `_build_detail_pane()` so a layout/section rebuild during in-flight work preserves the state.

- [ ] **Step 4: Write failing Re-run feedback tests**

Use an event-gated controller fake to prove the immediate state appears before completion. Cover exact behavior categories:

- accepted: `Re-running <name>...`, disabled `Re-running...`, `markup=False`;
- local terminal with counts: `Re-run complete: <name> — N found, M new.`;
- local entirely skipped: warning naming the source and already-running reason;
- server `queued`/`running`: `Re-run started: <name>.`;
- returned failed/error status: stated error derived through `_check_failure_message()`;
- raised exception: warning log plus safe stated error, with no unexpected path/URL leakage;
- `finally`: both sets cleared and the button restored for every exit;
- completion calls `_request_runs_refresh()` exactly once, including failed/skipped/raised outcomes, so the Runs view reconciles authoritatively.

- [ ] **Step 5: Run feedback RED nodes**

Run each new nodeid separately.

Expected: FAIL because `_rerun_run()` emits only generic launched/failed messages, has no immediate busy state, and refreshes Overview rather than Runs.

- [ ] **Step 6: Implement Re-run acceptance and worker outcome handling**

Replace the current source-only handler with this lifecycle:

```python
@on(RerunRunRequested)
def handle_rerun_run_requested(self, event: RerunRunRequested) -> None:
    event.stop()
    backend = event.runtime_backend
    if backend != self.runtime_backend or event.target_id in (None, ""):
        return
    key = self._rerun_operation_key(backend, event.target_id)
    if key in self._checks_in_flight:
        self._notify_watchlists(
            f"Already checking {event.name}.", severity="warning", markup=False
        )
        return
    self._checks_in_flight.add(key)
    self._reruns_in_flight.add(key)
    self._set_check_now_busy()
    self._notify_watchlists(
        f"Re-running {event.name}...", severity="information", markup=False
    )
    self.run_worker(
        self._rerun_run(
            runtime_backend=backend,
            target_id=event.target_id,
            operation_key=key,
            name=event.name,
        ),
        group="wc_rerun_run",
    )
```

Inside `_rerun_run()`, pass `source_id=target_id, job_id=None` for local and `source_id=None, job_id=target_id` for server. Reuse `_check_failure_message()`, `_check_was_entirely_skipped()`, `_TERMINAL_RUN_STATUSES`, and available counts so wording cannot drift from Check now. Capture backend in parameters and never reread it for the launch.

In `finally`:

```python
self._checks_in_flight.discard(operation_key)
self._reruns_in_flight.discard(operation_key)
if runtime_backend == self.runtime_backend:
    self._set_check_now_busy()
self._request_runs_refresh()
```

The refresh dispatcher captures the then-current backend and advances the generation; it is intentionally the same entry point as the toolbar. If the screen has switched backend, cleanup must not paint old-backend state into the new pane.

- [ ] **Step 7: Re-run the focused feedback/concurrency selection**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_watchlists_check_now_progress.py \
  Tests/UI/test_watchlists_check_now_skipped.py \
  Tests/UI/test_watchlists_check_now_failure.py \
  Tests/UI/test_watchlists_destination_shell.py -k "rerun or check_now or run"
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  Tests/UI/test_watchlists_check_now_progress.py \
  Tests/UI/test_watchlists_destination_shell.py
git commit -m "fix: report and deduplicate Watchlists reruns"
```

### Task 5: Complete task evidence, documentation, and focused verification

**Files:**
- Modify: `backlog/tasks/task-2331 - Runs-toolbar-Refresh-reloads-and-Re-run-gives-feedback.md`
- Verify modified production/test files from Tasks 1-4

- [ ] **Step 1: Run the approved focused test selection**

Run only affected Watchlists tests, not the full suite:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Watchlists/test_watchlists_runs_pane.py \
  Tests/Watchlists/test_watchlists_backend_controller.py \
  Tests/Watchlists/test_watchlist_scope_service.py \
  Tests/UI/test_watchlists_run_detail.py \
  Tests/UI/test_watchlists_check_now_progress.py \
  Tests/UI/test_watchlists_check_now_skipped.py \
  Tests/UI/test_watchlists_check_now_failure.py \
  Tests/UI/test_watchlists_destination_shell.py -k \
  "runs or run_detail or rerun or check_now or launch_run"
```

Expected: PASS. Record the exact passed/deselected/warning counts.

- [ ] **Step 2: Run modified-file Ruff and formatter checks**

Run the exact expected modified Python files:

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/UI/Watchlists_Modules/runs_pane.py \
  tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_runs_pane.py \
  Tests/Watchlists/test_watchlists_backend_controller.py \
  Tests/UI/test_watchlists_run_detail.py \
  Tests/UI/test_watchlists_check_now_progress.py \
  Tests/UI/test_watchlists_destination_shell.py

../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/UI/Watchlists_Modules/runs_pane.py \
  tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_runs_pane.py \
  Tests/Watchlists/test_watchlists_backend_controller.py \
  Tests/UI/test_watchlists_run_detail.py \
  Tests/UI/test_watchlists_check_now_progress.py \
  Tests/UI/test_watchlists_destination_shell.py
```

Expected: PASS.

- [ ] **Step 3: Run branch integrity checks**

Run:

```bash
git diff --check origin/dev...HEAD
git status --short
```

Expected: no whitespace errors anywhere in the branch; only intentional task/implementation changes remain.

- [ ] **Step 4: Self-review against the approved design**

Inspect `git diff origin/dev...HEAD` and confirm:

- Refresh stages before publication and advances generation on every accepted request;
- only local `KeyError` and server 404 clear an absent selected run;
- backend/generation/detail guards precede publication;
- local Check now/Re-run share a canonical source key;
- server Re-run uses only `job_id`;
- titles are always notified with `markup=False`;
- no pane recompose, schema, API, dependency, pagination, or generic action framework was added.

- [ ] **Step 5: Finish Backlog task hygiene**

Update all acceptance criteria to checked, add concise Implementation Notes including exact test/lint evidence and the ADR decision, then set TASK-2331 to Done with Backlog CLI. Do not invent a lessons entry unless implementation surfaces a genuinely reusable incident.

- [ ] **Step 6: Commit final documentation**

```bash
git add "backlog/tasks/task-2331 - Runs-toolbar-Refresh-reloads-and-Re-run-gives-feedback.md"
git add Docs/superpowers/plans/2026-08-27-watchlists-runs-refresh-rerun-feedback.md
git commit -m "docs: complete TASK-2331"
git diff --check origin/dev...HEAD
git status --short
```

Expected after the final commit: branch diff has no whitespace errors and the worktree is clean.
