# TASK-2061 Idle Worker Recycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task by task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a confirmed managed-model deletion unload an exact idle local-STT
resident and retry once without cancelling active work or bypassing artifact
leases.

**Architecture:** Extend the existing resident envelope with the verified managed
lease closure, retain that closure in the app-owned executor, and expose one
lock-protected idle-only recycle method. Inject the existing executor owner into
`InstalledView`; the view keeps deletion in its current background worker, retries
the existing artifact-service deletion once, and renders path-private recovery
states.

**Tech stack:** Python 3.12, multiprocessing spawn workers, existing
`LocalSTTExecutor`, existing `ModelArtifactService` leases, Textual 8 workers and
mounted Pilot tests, pytest, Ruff.

**Approved design:**
`Docs/superpowers/specs/2026-08-12-task-2061-idle-worker-recycle-design.md`

**ADR required:** no.

**ADR path:**
`backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`.

**Reason:** ADR-025 already assigns resident leases and generation recycling to
the app-owned local STT executor. This task completes the browser integration
explicitly deferred by the TASK-596 design.

## File map

**Production**

- `tldw_chatbook/STT/executor.py`: resident protocol, parent lease-set state,
  exact idle-only recycle operation, reset behavior.
- `tldw_chatbook/STT/executor_worker.py`: derive and report the verified handle's
  complete managed lease set.
- `tldw_chatbook/app.py`: snapshot the existing app-owned executor and forward an
  exact recycle request without constructing one.
- `tldw_chatbook/UI/LLM_Management_Window.py`: inject the app callback into the
  deferred Installed view.
- `tldw_chatbook/UI/Screens/model_installed_view.py`: one-retry delete recovery,
  policy recheck, and distinct row status.

**Tests**

- `Tests/STT/test_local_stt_executor.py`: protocol, full closure, idle recycle,
  active/nonmatching refusal, lease release, and generation cleanup.
- `Tests/App/test_submit_library_ingest_job.py`: app ownership and no-create
  behavior.
- `Tests/UI/test_model_installed_view.py`: mounted host wiring, delete ordering,
  retry bound, policy recheck, rendered states, and privacy.

No new module, enum, registry, timer, service API, or dependency is planned.

## Task 1: Report the worker-confirmed managed lease closure

**Files:**

- Modify: `Tests/STT/test_local_stt_executor.py`
- Modify: `tldw_chatbook/STT/executor.py:310-321`
- Modify: `tldw_chatbook/STT/executor_worker.py:51-70,298-333,857-865`

- [ ] **Step 1: Write protocol and worker RED tests**

Add a protocol test that round-trips canonical lease references and rejects a
malformed component:

```python
lease_refs = (
    ("parakeet-v2", "root-revision", "int8"),
    ("silero-vad", "vad-revision", "f32"),
)
resident = ExecutorResident(3, "attempt-1", _identity(), lease_refs)
assert pickle.loads(pickle.dumps(resident)).managed_lease_refs == lease_refs
```

Extend the direct resident-load tests so a managed root reports the root plus
declared dependency and an external root reports only its exact managed VAD.

- [ ] **Step 2: Run the RED tests**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/STT/test_local_stt_executor.py::test_protocol_objects_are_frozen_slotted_and_picklable \
  Tests/STT/test_local_stt_executor.py::test_provider_builder_receives_the_full_verified_managed_handle \
  Tests/STT/test_local_stt_executor.py::test_external_runtime_holds_exact_vad_lease_across_reuse_and_close
```

Expected: FAIL because `ExecutorResident` and `_ResidentRuntime` do not expose the
verified lease references.

- [ ] **Step 3: Implement the minimal protocol and worker report**

Add one defaulted tuple to the frozen envelope:

```python
@dataclass(frozen=True, slots=True)
class ExecutorResident:
    generation: int
    attempt_id: str
    identity: ModelIdentity
    managed_lease_refs: tuple[tuple[str, str, str], ...] = ()
```

Validate and canonicalize it with the existing three-string tuple boundary. Add
the same field to `_ResidentRuntime`. In `_load_resident`, derive it only from the
verified handle:

```python
if handle is None:
    managed_lease_refs = ()
elif request.managed_artifact_ref is not None:
    managed_lease_refs = tuple(
        (ref.artifact_id, ref.revision, ref.variant) for ref in handle.closure
    )
else:
    managed_lease_refs = tuple(
        (ref.artifact_id, ref.revision, ref.variant) for ref in handle.references
    )
```

Send `resident.managed_lease_refs` in the one `ExecutorResident` envelope. Keep the
default empty tuple so dependency-free test workers and unmanaged providers remain
source-compatible.

- [ ] **Step 4: Run the focused tests to GREEN**

Run the exact Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit the protocol slice**

```bash
git add Tests/STT/test_local_stt_executor.py \
  tldw_chatbook/STT/executor.py tldw_chatbook/STT/executor_worker.py
git commit -m "feat(stt): report resident artifact leases"
```

## Task 2: Recycle only an exact idle resident

**Files:**

- Modify: `Tests/STT/test_local_stt_executor.py`
- Modify: `tldw_chatbook/STT/executor.py:465-520,800-821,988-1038`

- [ ] **Step 1: Write idle, active, nonmatching, and cleanup RED tests**

Use the existing real resident worker and installed root/dependency fixtures.
Parameterize root and dependency targets:

```python
@pytest.mark.parametrize("target_name", ("root", "dependency"))
def test_idle_resident_recycle_releases_exact_managed_lease(...):
    # Submit and wait for a successful terminal result, leaving the runtime idle.
    assert executor.recycle_idle_managed_reference(_ref_tuple(target)) is True
    ModelArtifactService(store).delete(target)
```

Add an active `test_worker_hold` case proving recycle returns false while the same
attempt remains alive and no terminal callback is delivered. Add a nonmatching key
case proving the idle generation is retained. Add a retirement-proof case by making
the existing retirement primitive return false and asserting recycle cannot report
success.

- [ ] **Step 2: Run the RED tests**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/STT/test_local_stt_executor.py \
  -k 'idle_resident_recycle or active_resident_refuses_recycle or nonmatching_resident_refuses_recycle or unproven_idle_recycle'
```

Expected: FAIL because the parent does not retain the worker-confirmed lease set and
has no public idle recycle operation.

- [ ] **Step 3: Implement parent state and the narrow recycle operation**

Add `_resident_lease_refs`, set it only from a matching `ExecutorResident`, and
clear it everywhere the current resident identity is cleared. Do not clear it on an
ordinary terminal request result or failure while the generation remains alive.

Add:

```python
def recycle_idle_managed_reference(
    self,
    reference: tuple[str, str, str],
) -> bool:
    canonical = _canonical_dependency_refs((reference,))[0]
    with self._lock:
        if (
            self._closed
            or self._unavailable
            or self._busy
            or self._retiring
            or self._resident_identity is None
            or canonical not in self._resident_lease_refs
        ):
            return False
        return self._retire_idle_worker_locked()
```

Do not alter cancellation state, active callbacks, or `force_stop()`.

- [ ] **Step 4: Run the focused tests to GREEN**

Run the exact Step 2 command. Expected: PASS.

- [ ] **Step 5: Perform and restore executor mutations**

Temporarily remove the `_busy` guard; the active test must fail because an active
generation is detached. Restore it. Temporarily remove the exact-reference
membership check; the nonmatching test must fail. Restore it. Temporarily replace
the worker-confirmed closure with the request root only; the dependency test must
fail. Restore it.

- [ ] **Step 6: Commit the controller slice**

```bash
git add Tests/STT/test_local_stt_executor.py tldw_chatbook/STT/executor.py
git commit -m "feat(stt): recycle exact idle residents"
```

## Task 3: Bind deletion to the existing app-owned executor

**Files:**

- Modify: `Tests/App/test_submit_library_ingest_job.py`
- Modify: `Tests/UI/test_model_installed_view.py`
- Modify: `tldw_chatbook/app.py:2869-2883`
- Modify: `tldw_chatbook/UI/LLM_Management_Window.py:611-619`

- [ ] **Step 1: Write app ownership and mounted host RED tests**

In the app test, prove no executor is created merely to answer a delete request:

```python
app._create_local_stt_executor = MagicMock()
assert app._recycle_idle_local_stt_reference(reference) is False
app._create_local_stt_executor.assert_not_called()
```

Then install an executor double and assert the exact three-string tuple is forwarded.
Extend the existing mounted Models host test to prove `InstalledView` uses the bound
app callback after `_ensure_local_stt_executor` is replaced with a failing sentinel.

- [ ] **Step 2: Run the RED tests**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/App/test_submit_library_ingest_job.py::test_recycle_idle_local_stt_reference_uses_existing_executor_without_creating \
  Tests/UI/test_model_installed_view.py::test_models_host_lazily_wires_parakeet_activation_and_deletion
```

Expected: FAIL because the app callback and Installed-view injection do not exist.

- [ ] **Step 3: Implement app forwarding and host injection**

Add a type-checking-only `ArtifactRef` import. Snapshot the executor while holding
`_local_stt_executor_lock`, release the app lock, and then forward to the executor:

```python
def _recycle_idle_local_stt_reference(self, reference: "ArtifactRef") -> bool:
    with self._local_stt_executor_lock:
        if self._ingest_shutdown:
            return False
        executor = getattr(self, "_local_stt_executor", None)
    if executor is None:
        return False
    return executor.recycle_idle_managed_reference(
        (reference.artifact_id, reference.revision, reference.variant)
    )
```

Inject this bound method as `recycle_idle` beside the existing `may_delete` callback.
Do not call `_ensure_local_stt_executor()` from either path.

- [ ] **Step 4: Run the focused tests to GREEN**

Run the exact Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit the ownership slice**

```bash
git add Tests/App/test_submit_library_ingest_job.py \
  Tests/UI/test_model_installed_view.py tldw_chatbook/app.py \
  tldw_chatbook/UI/LLM_Management_Window.py
git commit -m "feat(models): bind idle STT recycle"
```

## Task 4: Retry confirmed deletion once with visible recovery state

**Files:**

- Modify: `Tests/UI/test_model_installed_view.py`
- Modify: `tldw_chatbook/UI/Screens/model_installed_view.py:127-160,250-292,511-570`

- [ ] **Step 1: Write delete-flow and mounted-state RED tests**

Add a service double whose first delete raises `ArtifactInUseError` and whose second
delete succeeds. Record this exact order:

```python
assert events == [
    "delete-1",
    "recycle",
    "policy-recheck",
    "delete-2",
]
```

Add cases where recycle returns false, policy becomes blocked after retirement, and
both delete attempts raise `ArtifactInUseError`. Assert respectively: no retry; no
retry after policy; and exactly two service calls plus one recycle call.

Mount the view with blocking events around recycle and the second delete. Assert the
painted row first contains `Checking for an idle model to unload…`, then
`Idle model unloaded; retrying deletion…`; the final hard blocker remains the
existing sanitized notification. Include a unique path marker in the final lease
and callback exceptions, attach a Loguru sink, and assert the marker reaches none of
rendered text, notifications, or logs.

- [ ] **Step 2: Run the RED tests**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_model_installed_view.py \
  -k 'delete_recycles_idle_owner or delete_rechecks_policy or delete_retries_once or mounted_idle_recycle_state'
```

Expected: FAIL because the callback, recovery states, and retry do not exist.

- [ ] **Step 3: Implement the minimal recovery flow**

Add one optional callback:

```python
recycle_idle: Callable[[ArtifactRef], bool] | None = None
```

Render a status `Static` only when the row matches `_operation_reference` and
`_operation_name` is `recycle-check` or `recycle-retry`. Add one UI-thread method
that changes the operation name and recomposes.

In `_delete_model`, special-case only the first `ArtifactInUseError`:

1. marshal `recycle-check`;
2. call the injected callback off-loop;
3. refuse with the original sanitized hard blocker when false;
4. marshal `recycle-retry` when true;
5. re-run `_may_delete` via `app.call_from_thread`;
6. retry `service.delete(reference)` exactly once; and
7. route the final result through existing lifecycle completion.

Unexpected callback errors are final sanitized deletion failures. Do not log the
first recoverable lease exception. A final lease blocker is logged only with the
bounded artifact identity, without its exception/cause chain; unexpected non-lease
deletion failures retain the existing diagnostic logging behavior.

- [ ] **Step 4: Run the focused tests to GREEN**

Run the exact Step 2 command. Expected: PASS.

- [ ] **Step 5: Perform and restore the retry mutation**

Temporarily let a second `ArtifactInUseError` invoke recycle again. The bounded-retry
test must fail on recycle count or delete count. Restore the one-retry implementation.

- [ ] **Step 6: Commit the browser slice**

```bash
git add Tests/UI/test_model_installed_view.py \
  tldw_chatbook/UI/Screens/model_installed_view.py
git commit -m "feat(models): retry deletion after idle unload"
```

## Task 5: Verify, simplify, and close TASK-2061

**Files:**

- Modify: `backlog/tasks/task-2061 - Idle-heavy-worker-recycle-for-managed-model-deletion.md`
- Modify only if a generalized incident is found:
  `backlog/docs/lessons-testing-evidence.md`

- [ ] **Step 1: Run the exact affected test files**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/STT/test_local_stt_executor.py \
  Tests/App/test_submit_library_ingest_job.py \
  Tests/UI/test_model_installed_view.py
```

Expected: all pass, with only documented environment warnings/skips.

- [ ] **Step 2: Run focused static checks**

```bash
../../.venv/bin/ruff check \
  tldw_chatbook/STT/executor.py \
  tldw_chatbook/STT/executor_worker.py \
  tldw_chatbook/app.py \
  tldw_chatbook/UI/LLM_Management_Window.py \
  tldw_chatbook/UI/Screens/model_installed_view.py \
  Tests/STT/test_local_stt_executor.py \
  Tests/App/test_submit_library_ingest_job.py \
  Tests/UI/test_model_installed_view.py
../../.venv/bin/python -m py_compile \
  tldw_chatbook/STT/executor.py \
  tldw_chatbook/STT/executor_worker.py \
  tldw_chatbook/app.py \
  tldw_chatbook/UI/LLM_Management_Window.py \
  tldw_chatbook/UI/Screens/model_installed_view.py
git diff --check origin/dev...HEAD
```

The current tree has known whole-file Ruff-format debt in six affected legacy files.
Use `git diff -U0` to identify each edited logical range and run
`ruff format --check --range START:END FILE` on those ranges only; do not bulk-format
unrelated code.

- [ ] **Step 3: Perform the Ponytail and correctness review**

Confirm there is still one callback, one boolean executor API, one retry, and no new
registry/timer/controller. Trace active submission, idle retirement, worker exit,
app shutdown, source-policy change, second lease conflict, and UI unmount. Remove any
duplicate helper or speculative state that is not required by a focused test.

- [ ] **Step 4: Update TASK-2061 through Backlog CLI only**

After every acceptance criterion and Definition-of-Done item is verified:

```bash
backlog task edit 2061 \
  --check-ac 1 --check-ac 2 --check-ac 3 \
  --notes "Implemented exact idle resident recycling through the app-owned local STT executor. The worker reports its verified managed lease closure; active and unrelated residents refuse recycling; Installed deletion shows path-private recovery states, rechecks policy, and retries the lease-enforced delete once. Focused STT, app, mounted UI, mutation, Ruff, compile, and diff checks passed. ADR-025 remains governing; no new ADR was required." \
  -s Done --plain
```

Do not add a lessons entry unless implementation exposes a new recurring trap with
concrete evidence.

- [ ] **Step 5: Commit closeout metadata**

```bash
git add "backlog/tasks/task-2061 - Idle-heavy-worker-recycle-for-managed-model-deletion.md"
git commit -m "docs(stt): close task 2061"
```

- [ ] **Step 6: Final verification**

Re-run the exact affected test command after the metadata commit, verify
`git status --short` is empty, and report exact pass/skip/warning counts plus every
mutation result. Do not claim completion if any required check or acceptance
criterion remains open.
