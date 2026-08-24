# TASK-97 Notes Sync Conflict Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users resolve eligible Database Notes content conflicts inline with Keep file, Keep note, Keep both, or Skip, backed by durable recovery, retained receipts, bounded history, and restart-safe per-item Undo.

**Architecture:** Extend the existing reconciliation → Library controller → runtime → executor chain. A small pure conflict-contract module owns validation, comparison bounds, and deterministic IDs; the existing runtime owns freshness and root serialization; the existing executor and device-state store remain the sole mutation/journal owners. The current Library canvas gains inline controls and bounded projections without introducing a modal, another database owner, or a second filesystem authority.

**Tech Stack:** Python 3.11+, asyncio, standard-library `difflib` and `weakref`, Textual 8.x, SQLite through the existing device-state owner, pytest, Ruff, MyPy.

**ADR required:** no new ADR

**ADR paths:** `backlog/decisions/055-library-destructive-action-reversibility-rule.md`; `backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md`; `backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md`

**Reason:** TASK-97 directly implements the already-decided inline review, recovery-before-write, manual conflict-copy, receipt/Undo, privacy, and round-trip contracts. It does not change ownership, persistence boundaries, or conflict policy beyond those ADRs.

---

## File map

### Create

- `tldw_chatbook/Notes/notes_sync_conflicts.py` — pure typed choices, eligibility, bounded comparison, deterministic conflict/Undo IDs, and privacy-safe result/history projections.
- `Tests/Notes/test_notes_sync_conflicts.py` — pure contract, bounds, representation, and mutation tests.
- `Tests/Notes/test_notes_sync_conflict_runtime.py` — focused freshness, root-lock, subset-apply, comparison, receipt, history, and Undo runtime tests.
- `Tests/Notes/test_notes_sync_conflict_executor.py` — focused Keep file/note/both and linked-Undo crash/replay tests.

### Modify

- `tldw_chatbook/Notes/notes_sync_authority.py` — note timestamps and single-effect create-or-verify note/folder/placement seams.
- `tldw_chatbook/Notes/notes_scope_service.py` — local-only caller-ID folder creation and exact folder/placement read seams used by sync authority.
- `tldw_chatbook/Notes/notes_sync_runtime.py` — weak per-root mutation locks, comparison entry point, reviewed conflict admission, active receipts, history decoration, and Undo coordination.
- `tldw_chatbook/Notes/notes_sync_executor.py` — resolution operation kinds, 30-day recovery, Keep-both substages, self-contained linked Undo, and restart reconstruction.
- `tldw_chatbook/Notes/notes_device_state_store.py` — recovery-metadata substage CAS and bounded resolution-history queries; no schema change.
- `tldw_chatbook/Library/library_notes_lasting_sync_state.py` — typed selection/comparison/receipt/history projections and exact eligible-row presentation.
- `tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py` — token-keyed staging, comparison generation, typed apply result handling, receipt/history actions, and invalidation.
- `tldw_chatbook/Widgets/Library/library_notes_add_from_files_canvas.py` — inline choices, comparison, retained receipts, and history controls.
- `tldw_chatbook/UI/Screens/library_screen.py` — route new canvas messages through the controller; run comparison work outside the Textual message pump.
- `tldw_chatbook/css/components/_agentic_terminal.tcss` — bounded conflict, comparison, receipt, and history layout.
- `tldw_chatbook/css/tldw_cli_modular.tcss` — regenerated CSS bundle only; never hand-edit.
- `Tests/Notes/test_notes_sync_authority.py` — exact single-effect authority and collision coverage.
- `Tests/Notes/test_notes_sync_runtime.py` — adjacent existing-runtime regression coverage.
- `Tests/Notes/test_notes_sync_executor.py` — adjacent existing-operation/recovery compatibility coverage.
- `Tests/Library/test_library_notes_lasting_sync_state.py` — exact eligibility and bounded public projection coverage.
- `Tests/UI/Library_Modules/test_library_notes_sync_controller.py` — staging, invalidation, status, receipt, and history controller coverage.
- `Tests/Widgets/Library/test_library_notes_add_from_files_canvas.py` — widget contract and keyboard/non-color coverage.
- `Tests/UI/test_library_notes_files_sync_journey.py` — mounted production route, focus, scroll, stale completion, and 60x20 coverage.
- `Docs/User_Guide/library/notes.md` — user-facing conflict choices, Skip, receipts, history, and Undo.
- `backlog/tasks/task-97 - Resolve-Notes-sync-conflicts-inline.md` — approved plan link, final checked ACs, ADR check, evidence, and implementation notes.

### Explicitly unchanged

- `tldw_chatbook/Notes/notes_sync_reconciler.py` — it continues to report attention and never chooses a winner.
- `tldw_chatbook/Notes/notes_device_state_schema.py` — generic operation/recovery rows already hold the required kinds and private substages.
- legacy ASK/modal and legacy sync-engine modules — no path is reactivated.

---

### Task 1: Add pure conflict contracts and exact eligibility

**Files:**
- Create: `tldw_chatbook/Notes/notes_sync_conflicts.py`
- Create: `Tests/Notes/test_notes_sync_conflicts.py`
- Modify: `tldw_chatbook/Library/library_notes_lasting_sync_state.py:112-204,351-432`
- Test: `Tests/Library/test_library_notes_lasting_sync_state.py`

- [x] **Step 1: Write failing pure-contract tests**

Cover the four exact enum values, typed selection validation, eligible reasons, managed-placement exclusion, deterministic IDs, comparison bounds/elision, Note-to-File orientation, missing timestamp, and privacy-safe `repr`.

```python
def test_only_bound_content_change_reasons_are_selectable() -> None:
    assert eligible_conflict_reason("both_sides_changed", managed=False)
    assert eligible_conflict_reason("out_of_direction_change", managed=False)
    for reason in (
        "duplicate_authority",
        "out_of_direction_create",
        "out_of_direction_move",
        "out_of_direction_representation",
        "ambiguous_identity",
        "note_implied_filesystem_move",
    ):
        assert not eligible_conflict_reason(reason, managed=False)
    assert not eligible_conflict_reason("both_sides_changed", managed=True)


def test_comparison_is_bounded_and_private() -> None:
    comparison = build_conflict_comparison(
        binding_id="binding-1",
        title="Title",
        relative_path="folder/note.md",
        note_text="note\n",
        file_text="file\n",
        note_version=3,
        note_updated_at=None,
        file_modified_ns=7,
    )
    assert comparison.diff.startswith("--- Note\n+++ File\n")
    assert comparison.note_updated_label == "Unavailable"
    assert "/private/" not in repr(comparison)
```

- [x] **Step 2: Run the RED tests**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Notes/test_notes_sync_conflicts.py \
  Tests/Library/test_library_notes_lasting_sync_state.py \
  -k 'conflict or comparison or selectable'
```

Expected: collection/import failures for the missing module and assertion failures because every `CONFLICT` row currently exposes three disabled choices and no Skip.

- [x] **Step 3: Implement the minimum pure module**

Use only stdlib and existing validators. Keep exact constants in one place.

```python
class NotesSyncConflictChoice(StrEnum):
    KEEP_FILE = "keep_file"
    KEEP_NOTE = "keep_note"
    KEEP_BOTH = "keep_both"
    SKIP = "skip"


ELIGIBLE_CONFLICT_REASONS = frozenset(
    {"both_sides_changed", "out_of_direction_change"}
)


def eligible_conflict_reason(reason_code: str, *, managed: bool) -> bool:
    return reason_code in ELIGIBLE_CONFLICT_REASONS and not managed
```

Add frozen private-repr dataclasses for selection, comparison, apply result, receipt, and history. Measure both complete inputs first. If either side exceeds 200,000 characters or 10,000 lines, omit the diff entirely and return only the exact per-side character/line sizes plus the bounded too-large explanation; never show a partial diff. Otherwise use `difflib.unified_diff` and enforce the 120,000-character/2,000-line output ceiling. Use canonical NUL-delimited, domain-separated SHA-256 helpers for conflict folders, copy notes, operations, and linked Undo IDs.

- [x] **Step 4: Project only exact eligible rows**

Build a binding-ID set from `plan.managed_placement_effects`. Eligible rows get four choices and exact reason copy; every other conflict/deletion/pause row retains disabled existing copy. Do not put note content, hashes, or absolute paths in `LastingSyncReviewRow`.

- [x] **Step 5: Run GREEN and mutation checks**

Run the Step 2 command. Then temporarily remove the managed-placement exclusion and confirm the named eligibility test fails; restore it. Temporarily reverse the unified-diff inputs and confirm the orientation test fails; restore it.

Expected: all selected tests pass after both mutations are restored.

- [x] **Step 6: Run scoped statics and commit**

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Notes/notes_sync_conflicts.py \
  tldw_chatbook/Library/library_notes_lasting_sync_state.py \
  Tests/Notes/test_notes_sync_conflicts.py \
  Tests/Library/test_library_notes_lasting_sync_state.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Notes/notes_sync_conflicts.py \
  tldw_chatbook/Library/library_notes_lasting_sync_state.py \
  Tests/Notes/test_notes_sync_conflicts.py \
  Tests/Library/test_library_notes_lasting_sync_state.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/Notes/notes_sync_conflicts.py \
  tldw_chatbook/Library/library_notes_lasting_sync_state.py
git diff --check
git add \
  tldw_chatbook/Notes/notes_sync_conflicts.py \
  tldw_chatbook/Library/library_notes_lasting_sync_state.py \
  Tests/Notes/test_notes_sync_conflicts.py \
  Tests/Library/test_library_notes_lasting_sync_state.py
git commit -m "feat(notes): define reviewed sync conflict choices"
```

---

### Task 2: Add root serialization and bounded comparison authority

**Files:**
- Modify: `tldw_chatbook/Notes/notes_sync_authority.py:25-52,95-139,208-238`
- Modify: `tldw_chatbook/Notes/notes_sync_runtime.py:231-263,271-610,637-690,857-956,1108-1213`
- Create: `Tests/Notes/test_notes_sync_conflict_runtime.py`
- Modify: `Tests/Notes/test_notes_sync_authority.py`

- [x] **Step 1: Write RED runtime tests**

Use `asyncio.Event` barriers, never sleeps, to prove:

- two apply paths for one root share one lock;
- automatic execution versus reviewed apply on one root serialize, and the loser reobserves authority after the winner;
- startup recovery versus reviewed apply on one root serialize through the same lock and the loser revalidates after acquisition;
- a waiting task cannot receive a replacement lock after garbage collection;
- different roots do not block one another;
- comparison re-observes and requires exact root/token/plan/binding equality;
- comparison refuses a missing/stale planning lease and an inactive, missing, or wrong authoritative root before exposing content;
- comparison releases the private observation bundle on success, refusal, and cancellation;
- note timestamp is typed/optional and comparison never returns raw snapshots.

```python
async def test_same_root_mutations_serialize_and_revalidate() -> None:
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    adapter = GatedAdapter(first_entered, release_first)
    first = asyncio.create_task(runtime.apply_reviewed(...))
    await asyncio.wait_for(first_entered.wait(), 1)
    second = asyncio.create_task(runtime.apply_reviewed(...))
    assert adapter.fresh_observation_count == 1
    release_first.set()
    await asyncio.gather(first, second, return_exceptions=True)
    assert adapter.fresh_observation_count == 2
    assert adapter.mutation_count == 1
```

- [x] **Step 2: Run RED**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Notes/test_notes_sync_conflict_runtime.py \
  Tests/Notes/test_notes_sync_authority.py \
  -k 'root_lock or comparison or updated_at'
```

Expected: missing public comparison/lock APIs and missing note timestamp.

- [x] **Step 3: Add the weak root-lock registry with one outer acquisition**

In `LastingNotesSyncRuntime.__init__`, keep one `WeakValueDictionary[str, asyncio.Lock]`. Resolve a lock synchronously before any await and retain the local variable through `async with lock`. Each top-level mutating route—automatic reconciliation, reviewed apply, recovery, and later Undo—acquires exactly once, then calls a shared `_execute_locked`/equivalent helper that assumes the root lock is already held and must never reacquire it. Keep coordinator admission outside and authority revalidation inside the lock. Read-only comparison does not acquire it. Never acquire two root locks and never call a locking wrapper from inside a locked route.

```python
def _mutation_lock(self, root_id: str) -> asyncio.Lock:
    lock = self._mutation_locks.get(root_id)
    if lock is None:
        lock = asyncio.Lock()
        self._mutation_locks[root_id] = lock
    return lock
```

Add a deterministic regression in which reviewed apply reaches the shared execution helper and completes after the gate is released; fail the test if a second acquisition is attempted. This pins the one-outer-acquisition rule instead of relying on a timeout to reveal a self-deadlock.

Add separate Event-barrier interleavings for automatic-versus-reviewed and recovery-versus-reviewed. Hold the first route immediately after fresh authority is accepted, start the competing route, assert it has not observed or mutated yet, release the winner, then prove the loser obtains a new observation and either executes against that authority or refuses cleanly. These tests establish that every current top-level mutation route uses the same root lock; Task 5 later adds the Undo interleaving.

- [x] **Step 4: Add bounded comparison while the bundle is alive**

Add `compare_conflict(root_id, observation_token, binding_id)` to the runtime port and adapter. Enter the coordinator's root planning lease and call `_require_authority(..., "plan")` before observation. While the private bundle remains live, observe → plan → call `_require_authority(..., "plan")` again → verify exact root/token/plan/binding equality → build the bounded projection → release in `finally`. Refuse missing/stale leases and missing, inactive, or mismatched roots without returning comparison content. Do not route through `_fresh_authority`, whose existing `finally` releases its private bundle before the caller can compare. Add optional validated `updated_at` to `NotesSyncNoteSnapshot`, sourced through the existing service mapping.

- [x] **Step 5: Run GREEN and mutation checks**

Run Step 2. Mutate `_mutation_lock` to return a fresh lock and confirm the same-root test fails. Mutate the shared locked execution helper to reacquire the same root lock and confirm the explicit second-acquisition guard test fails. Remove the post-observation `_require_authority(..., "plan")` call and confirm the stale-lease/root test fails. Remove the final token equality check and confirm stale comparison fails. Restore all four mutations.

- [x] **Step 6: Run adjacent runtime tests and commit**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Notes/test_notes_sync_runtime.py \
  Tests/Notes/test_notes_sync_authority.py \
  Tests/Notes/test_notes_sync_conflict_runtime.py
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Notes/notes_sync_authority.py \
  tldw_chatbook/Notes/notes_sync_runtime.py \
  Tests/Notes/test_notes_sync_authority.py \
  Tests/Notes/test_notes_sync_conflict_runtime.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Notes/notes_sync_authority.py \
  tldw_chatbook/Notes/notes_sync_runtime.py \
  Tests/Notes/test_notes_sync_authority.py \
  Tests/Notes/test_notes_sync_conflict_runtime.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/Notes/notes_sync_authority.py \
  tldw_chatbook/Notes/notes_sync_runtime.py
../../.venv/bin/python -m compileall -q \
  tldw_chatbook/Notes/notes_sync_authority.py \
  tldw_chatbook/Notes/notes_sync_runtime.py
git diff --check
git add \
  tldw_chatbook/Notes/notes_sync_authority.py \
  tldw_chatbook/Notes/notes_sync_runtime.py \
  Tests/Notes/test_notes_sync_authority.py \
  Tests/Notes/test_notes_sync_conflict_runtime.py
git commit -m "feat(notes): compare sync conflicts under fresh authority"
```

---

### Task 3: Execute Keep file and Keep note as reviewed subsets

**Files:**
- Modify: `tldw_chatbook/Notes/notes_sync_runtime.py:537-591,857-934,1108-1213`
- Modify: `tldw_chatbook/Notes/notes_sync_executor.py:370-510,612-910,1490-1575,2313-2469`
- Modify: `tldw_chatbook/Notes/notes_device_state_store.py:991-1310,1613-1655`
- Modify: `Tests/Notes/test_notes_sync_runtime.py`
- Modify: `Tests/Notes/test_notes_sync_executor.py`
- Extend: `Tests/Notes/test_notes_sync_conflict_runtime.py`
- Extend: `Tests/Notes/test_notes_sync_conflict_executor.py`

- [x] **Step 1: Write RED subset-apply tests**

Cover bidirectional plus both one-way directions, occurrence-only overrides, Skip plus safe actions, all non-content blockers, deterministic safe-then-binding order, 30-day recovery, terminal refresh, and honest partial stop.

```python
@pytest.mark.parametrize("choice", [KEEP_FILE, KEEP_NOTE])
async def test_selected_conflict_executes_and_skip_remains_attention(choice) -> None:
    result = await runtime.apply_reviewed(
        root_id,
        token,
        safe_action_ids=(safe_id,),
        selections=(ConflictSelection(binding_id, choice), skipped),
    )
    assert result.safe_completed == 1
    assert result.conflicts_resolved == 1
    assert result.unresolved_conflicts == 1
    assert result.attention_remains is True
```

- [x] **Step 2: Run RED**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Notes/test_notes_sync_conflict_runtime.py \
  Tests/Notes/test_notes_sync_conflict_executor.py \
  -k 'keep_file or keep_note or skip or direction or retention'
```

Expected: typed apply signature/result and resolution operation kinds do not exist.

- [x] **Step 3: Extend the request without replacing the action enum**

Keep `NotesSyncActionKind.UPDATE_NOTE/UPDATE_FILE` as the executor action. Add a validated optional journal kind (`resolve_keep_file` or `resolve_keep_note`) and store the underlying action kind in private recovery metadata. Existing action requests remain byte/behavior compatible. Use one named `CONFLICT_RECOVERY_RETENTION_NS = 30 * 24 * 60 * 60 * 1_000_000_000` only for conflict and Undo operations.

- [x] **Step 4: Implement narrow manual admission**

Under the root lock, rebuild exact authority. Reject any deletion group, deletion attention, pause, managed placement, capability/root skip, activation review, unknown/duplicate/cross-root selection, or ineligible reason before recovery admission. Build selected requests before releasing the private bundle. Skip builds nothing. Execute existing safe actions in plan order, then selected conflicts ordered by binding ID.

- [x] **Step 5: Preserve automatic behavior**

Do not relax `_blocked_plan_status` for automatic reconciliation. Its existing all-attention blocker remains exact. Only `apply_reviewed` receives the content-conflict exception.

- [x] **Step 6: Run GREEN and required mutations**

Run Step 2. Remove observation-token equality and confirm stale apply fails. Move recovery admission after the first write and confirm recovery-before-write fails. Allow deletion attention and confirm the blocker test fails. Restore each mutation separately.

- [x] **Step 7: Run adjacent suites and commit**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Notes/test_notes_sync_runtime.py \
  Tests/Notes/test_notes_sync_executor.py \
  Tests/Notes/test_notes_sync_conflict_runtime.py \
  Tests/Notes/test_notes_sync_conflict_executor.py
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Notes/notes_sync_runtime.py \
  tldw_chatbook/Notes/notes_sync_executor.py \
  tldw_chatbook/Notes/notes_device_state_store.py \
  Tests/Notes/test_notes_sync_runtime.py \
  Tests/Notes/test_notes_sync_executor.py \
  Tests/Notes/test_notes_sync_conflict_runtime.py \
  Tests/Notes/test_notes_sync_conflict_executor.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Notes/notes_sync_runtime.py \
  tldw_chatbook/Notes/notes_sync_executor.py \
  tldw_chatbook/Notes/notes_device_state_store.py \
  Tests/Notes/test_notes_sync_runtime.py \
  Tests/Notes/test_notes_sync_executor.py \
  Tests/Notes/test_notes_sync_conflict_runtime.py \
  Tests/Notes/test_notes_sync_conflict_executor.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/Notes/notes_sync_runtime.py \
  tldw_chatbook/Notes/notes_sync_executor.py \
  tldw_chatbook/Notes/notes_device_state_store.py
git diff --check
git add \
  tldw_chatbook/Notes/notes_sync_runtime.py \
  tldw_chatbook/Notes/notes_sync_executor.py \
  tldw_chatbook/Notes/notes_device_state_store.py \
  Tests/Notes/test_notes_sync_runtime.py \
  Tests/Notes/test_notes_sync_executor.py \
  Tests/Notes/test_notes_sync_conflict_runtime.py \
  Tests/Notes/test_notes_sync_conflict_executor.py
git commit -m "feat(notes): apply reviewed sync conflict choices"
```

---

### Task 4: Make Keep both single-effect and restart-safe

**Files:**
- Modify: `tldw_chatbook/Notes/notes_scope_service.py:317-460,946-1055`
- Modify: `tldw_chatbook/Notes/note_folder_repository.py:189-210,492-518,717-752` — add one read-only exact manual-membership lookup including deleted rows; no new mutation or SQL owner.
- Modify: `tldw_chatbook/Notes/notes_sync_authority.py:71-207`
- Modify: `tldw_chatbook/Notes/notes_device_state_store.py:1013-1210,1613-1655`
- Modify: `tldw_chatbook/Notes/notes_sync_executor.py`
- Modify: `tldw_chatbook/Notes/notes_sync_runtime.py:624-690,1330-1420`
- Modify: `Tests/Notes/test_notes_sync_authority.py`
- Modify: `Tests/Notes/test_note_folder_repository.py`
- Extend: `Tests/Notes/test_notes_sync_conflict_executor.py`
- Extend: `Tests/Notes/test_notes_sync_conflict_runtime.py`

- [x] **Step 1: Write RED authority, direction, and crash-matrix tests**

Test one effect per call, caller-owned IDs, reuse by normalized manual path, actual reused IDs, owner/kind/path/content/placement mismatch, concurrent create loser verification, and all eight durable boundaries. Repository/authority RED cases must prove: an active exact membership is returned/reused even when older deleted history exists; deleted-only history returns the latest `modified_at DESC, id DESC` row and causes fail-closed with zero mutation; and `include_deleted=False` returns no deleted row. Name these cases with the focused selector, for example `test_conflict_copy_deleted_placement_fails_closed`. Parameterize Keep both across `bidirectional`, `folder_to_notes`, and `notes_to_folder`, including `out_of_direction_change`; prove the bound-note update receives a one-occurrence override only when the configured direction would otherwise forbid that exact selected update. At each folder, note, placement, bound-note, file-recheck, binding, and verification boundary, inject cancellation after the external effect begins and prove the durable substage is coherent and replayable before cancellation reaches the caller. Seed recovery capacity immediately below the ceiling and prove every substage transition has zero byte growth and completes without bypassing or rechecking the global ceiling after mutation begins. Every restart test must discard runtime, executor, request, snapshots, and fakes; reconstruct from a reopened store plus fresh authority only.

- [x] **Step 2: Run RED**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Notes/test_notes_sync_authority.py \
  Tests/Notes/test_note_folder_repository.py \
  Tests/Notes/test_notes_sync_conflict_executor.py \
  Tests/Notes/test_notes_sync_conflict_runtime.py \
  -k 'conflict_copy or keep_both or restart'
```

Expected: single-effect folder/note/placement methods and Keep-both request/replay are missing.

- [x] **Step 3: Add local-only service capabilities**

Expose the repository's existing caller-owned `folder_id` only through sync-specific local service methods. Add one read-only repository lookup for the exact `(folder_id, note_id, ownership='manual', owner_id='')` membership with an explicit `include_deleted` option, following the existing `get_folder(..., include_deleted=...)` pattern; it returns the stored row without reviving it. Selection semantics are exact: return the active row first; only when none exists and `include_deleted=True`, return the latest deleted row ordered by `modified_at DESC, id DESC`, matching the existing `attach_manual` revival order and stable tie-break. Route exact get-by-path/get-by-ID and manual-placement reads through the service. A valid active placement may be reused; when only deleted history exists, sync authority fails closed rather than invoking revival. Also reject server scopes, sync-managed placement, and owner mismatch with bounded errors before calling the existing `attach_manual`, whose ordinary revive behavior remains unchanged for non-sync callers. Do not add a second repository, mutation, or SQL owner.

- [x] **Step 4: Add three authority methods**

Implement:

```python
async def create_or_verify_manual_folder(request) -> VerifiedFolder: ...
async def create_or_verify_conflict_note(request) -> NotesSyncNoteSnapshot: ...
async def create_or_verify_manual_placement(request) -> VerifiedPlacement: ...
```

Each method performs at most one create, catches uniqueness races by rereading, verifies the exact winner, and returns fresh identity/version. A deterministic ID at another path or an existing deterministic copy with different title/body/placement fails closed.

- [x] **Step 5: Add capacity-neutral private substage CAS**

At admission, encode every deterministic ID and authority field needed by every future Keep-both stage. Keep `conflict_substage` as the exact durable enum string from the approved spec (`recovery_admitted`, `folders_established`, and so on). Add a private reserved-padding field whose length is `longest_conflict_substage_length - len(conflict_substage)`, so the stage string plus padding occupies a fixed admitted envelope for every legal value. Capacity admission accounts for that complete maximum/final payload once, before the first mutation.

Add one store method that atomically compares operation ID, recovery ID, current operation state, current exact `conflict_substage`, matching reserved padding, expected recovery payload digest, and existing byte length before replacing the enum plus reciprocal padding and `updated_at`. It accepts only the exact forward sequence from the spec and rejects unknown values, invalid padding, or any total-length change as corruption. It never performs a new capacity decision after an external effect. No schema change.

- [x] **Step 6: Implement Keep-both replay and its occurrence-only direction override**

Extend the existing runtime adapter's conflict request builder to construct the complete private Keep-both request while the observation bundle is live, and remove only the reviewed-apply Keep-both refusal. The same top-level admission/blocker/token/root checks from Task 3 remain exact; automatic reconciliation still never chooses Keep both.

Admission stores all deterministic IDs and exact expected authority. Execute parent folder → child folder → copy note → placement → joint verification → bound note update → file recheck → binding update → final verification. The bound-note update uses the same narrowly validated occurrence-only override contract as a selected Keep file resolution when the root direction would otherwise forbid it; it never changes the root configuration or authorizes any other action. Use the executor's existing joined/shielded mutation pattern for every admitted external effect and its immediately following substage CAS: cancellation is propagated only after that pair reaches a coherent durable checkpoint. On replay, reobserve the prior effect before advancing. Never delete a folder on recovery failure.

- [x] **Step 7: Run GREEN and mutations**

Run Step 2. Mutate collision verification to accept different content and confirm failure. Skip one substage CAS and confirm its restart node fails. Restore both.

- [x] **Step 8: Run adjacent suites and commit**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Notes/test_notes_sync_authority.py \
  Tests/Notes/test_note_folder_repository.py \
  Tests/Notes/test_notes_sync_executor.py \
  Tests/Notes/test_notes_sync_conflict_executor.py \
  Tests/Notes/test_notes_sync_conflict_runtime.py
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Notes/notes_scope_service.py \
  tldw_chatbook/Notes/note_folder_repository.py \
  tldw_chatbook/Notes/notes_sync_authority.py \
  tldw_chatbook/Notes/notes_device_state_store.py \
  tldw_chatbook/Notes/notes_sync_executor.py \
  tldw_chatbook/Notes/notes_sync_runtime.py \
  Tests/Notes/test_notes_sync_authority.py \
  Tests/Notes/test_note_folder_repository.py \
  Tests/Notes/test_notes_sync_executor.py \
  Tests/Notes/test_notes_sync_conflict_executor.py \
  Tests/Notes/test_notes_sync_conflict_runtime.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Notes/notes_scope_service.py \
  tldw_chatbook/Notes/note_folder_repository.py \
  tldw_chatbook/Notes/notes_sync_authority.py \
  tldw_chatbook/Notes/notes_device_state_store.py \
  tldw_chatbook/Notes/notes_sync_executor.py \
  tldw_chatbook/Notes/notes_sync_runtime.py \
  Tests/Notes/test_notes_sync_authority.py \
  Tests/Notes/test_note_folder_repository.py \
  Tests/Notes/test_notes_sync_executor.py \
  Tests/Notes/test_notes_sync_conflict_executor.py \
  Tests/Notes/test_notes_sync_conflict_runtime.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/Notes/notes_scope_service.py \
  tldw_chatbook/Notes/note_folder_repository.py \
  tldw_chatbook/Notes/notes_sync_authority.py \
  tldw_chatbook/Notes/notes_device_state_store.py \
  tldw_chatbook/Notes/notes_sync_executor.py \
  tldw_chatbook/Notes/notes_sync_runtime.py
git diff --check
git add \
  tldw_chatbook/Notes/notes_scope_service.py \
  tldw_chatbook/Notes/note_folder_repository.py \
  tldw_chatbook/Notes/notes_sync_authority.py \
  tldw_chatbook/Notes/notes_device_state_store.py \
  tldw_chatbook/Notes/notes_sync_executor.py \
  tldw_chatbook/Notes/notes_sync_runtime.py \
  Tests/Notes/test_notes_sync_authority.py \
  Tests/Notes/test_note_folder_repository.py \
  Tests/Notes/test_notes_sync_executor.py \
  Tests/Notes/test_notes_sync_conflict_executor.py \
  Tests/Notes/test_notes_sync_conflict_runtime.py
git commit -m "feat(notes): preserve both sync conflict versions"
```

---

### Task 5: Add durable linked Undo, history, and active receipts

**Files:**
- Modify: `tldw_chatbook/Notes/notes_device_state_store.py`
- Modify: `tldw_chatbook/Notes/notes_sync_executor.py`
- Modify: `tldw_chatbook/Notes/notes_sync_runtime.py`
- Extend: `Tests/Notes/test_notes_sync_conflict_executor.py`
- Extend: `Tests/Notes/test_notes_sync_conflict_runtime.py`

- [x] **Step 1: Write RED Undo/history tests**

Cover Keep file/note/both Undo, exact expiry boundary, edited copy/current authority refusal, wrong root, duplicate delivery, source-row immutability, fresh note version in active binding, second resolution after Undo, self-contained recovery after deleting/expiring source recovery, every Undo crash boundary, and apply/apply plus apply/Undo serialization. Inject cancellation during authority restoration, binding update, and Keep-both copy cleanup; each test must prove the external effect plus its checkpoint complete coherently before cancellation propagates and a fresh executor resumes safely. At one byte below the admitted recovery ceiling, prove every Undo substage replacement remains byte-for-byte length neutral and cannot introduce a late capacity refusal.

- [x] **Step 2: Run RED**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Notes/test_notes_sync_conflict_executor.py \
  Tests/Notes/test_notes_sync_conflict_runtime.py \
  -k 'undo or history or receipt'
```

Expected: linked Undo/history/receipt APIs are absent.

- [x] **Step 3: Add bounded store queries**

List only `resolve_keep_file`, `resolve_keep_note`, and `resolve_keep_both`, newest first, limit/offset capped at 100. For each source operation, derive the deterministic Undo ID and fetch that row; do not mutate the completed source row or overload its `reason_code`.

- [x] **Step 4: Admit self-contained Undo recovery**

Before mutation, copy all pre-resolution target authority, current rollback/check authority, original binding fields, source operation/choice, conflict-copy checks, and every field needed by later substages into the linked operation's own capacity-accounted recovery. Keep `undo_substage` as the exact durable enum strings from the approved spec and pair it with private reciprocal padding sized against the longest legal Undo value, so every stage has one fixed admitted envelope. Later CAS transitions replace the exact enum plus its padding, validate enum/padding/total byte length, and cannot run a second capacity decision after mutation starts. After admission, restart reconstruction must not read source recovery.

- [x] **Step 5: Implement idempotent Undo stages**

Restore changed authority → verify opposite authority → commit original binding digest/identity/path/serialization with the fresh restored note version while keeping binding `active` → conditionally soft-delete unchanged Keep-both copy → verify → complete linked Undo. Run each admitted mutation and its following recovery/substage checkpoint through the executor's joined/shielded pattern, then propagate cancellation. A partial linked Undo stays coherent, recoverable, and attention-worthy. History projects Undone only when the deterministic linked operation is completed.

- [x] **Step 6: Add runtime receipt/history projections**

Keep one `OrderedDict` of at most 100 active receipt operation IDs per root. Completion inserts/moves-to-end; Dismiss removes; superseding the same item removes the older receipt. Remount reads the map; process restart begins empty. History decorates rows from fresh authority with bounded current title/path or the first eight operation-ID characters, without logging/persisting labels.

- [x] **Step 7: Run GREEN and mutations**

Run Step 2. Remove linked recovery payload copying and prove restart-after-source-expiry fails. Write historical note version into the binding and prove second resolution fails. Project Undone from admission rather than completion and prove the crash test fails. Restore each.

- [x] **Step 8: Run adjacent suites and commit**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Notes/test_notes_sync_runtime.py \
  Tests/Notes/test_notes_sync_executor.py \
  Tests/Notes/test_notes_sync_conflict_runtime.py \
  Tests/Notes/test_notes_sync_conflict_executor.py
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Notes/notes_device_state_store.py \
  tldw_chatbook/Notes/notes_sync_executor.py \
  tldw_chatbook/Notes/notes_sync_runtime.py \
  Tests/Notes/test_notes_sync_runtime.py \
  Tests/Notes/test_notes_sync_executor.py \
  Tests/Notes/test_notes_sync_conflict_runtime.py \
  Tests/Notes/test_notes_sync_conflict_executor.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Notes/notes_device_state_store.py \
  tldw_chatbook/Notes/notes_sync_executor.py \
  tldw_chatbook/Notes/notes_sync_runtime.py \
  Tests/Notes/test_notes_sync_runtime.py \
  Tests/Notes/test_notes_sync_executor.py \
  Tests/Notes/test_notes_sync_conflict_runtime.py \
  Tests/Notes/test_notes_sync_conflict_executor.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/Notes/notes_device_state_store.py \
  tldw_chatbook/Notes/notes_sync_executor.py \
  tldw_chatbook/Notes/notes_sync_runtime.py
git diff --check
git add \
  tldw_chatbook/Notes/notes_device_state_store.py \
  tldw_chatbook/Notes/notes_sync_executor.py \
  tldw_chatbook/Notes/notes_sync_runtime.py \
  Tests/Notes/test_notes_sync_runtime.py \
  Tests/Notes/test_notes_sync_executor.py \
  Tests/Notes/test_notes_sync_conflict_runtime.py \
  Tests/Notes/test_notes_sync_conflict_executor.py
git commit -m "feat(notes): undo reviewed sync resolutions"
```

---

### Task 6: Make controller state token-safe and receipt-aware

**Files:**
- Modify: `tldw_chatbook/Library/library_notes_lasting_sync_state.py`
- Modify: `tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py:25-125,147-468`
- Modify: `Tests/Library/test_library_notes_lasting_sync_state.py`
- Modify: `Tests/UI/Library_Modules/test_library_notes_sync_controller.py`

- [x] **Step 1: Write RED controller tests**

Cover selection staging without runtime calls, page persistence, new-token/stale/Back/root/remount clearing, one retained comparison, generation-guarded async completion, comparison release on Return and page change without clearing staged choices, typed blocker Apply enablement, Skip-only disablement, exact status-line update once, partial/fresh-review behavior, receipt Undo/Dismiss, and history paging.

- [x] **Step 2: Run RED**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Library/test_library_notes_lasting_sync_state.py \
  Tests/UI/Library_Modules/test_library_notes_sync_controller.py \
  -k 'conflict or selection or comparison or receipt or history'
```

Expected: current `stage_attention_choice` reports unavailable and never records a choice.

- [x] **Step 3: Add bounded immutable projections**

Extend review rows with typed eligibility and selected label, and the snapshot with one optional comparison plus bounded receipt/history pages. Keep raw snapshots/content out. Validate every tuple, page, label, and count.

- [x] **Step 4: Implement private controller state**

Use `dict[(observation_token, binding_id), NotesSyncConflictChoice]`; never key by binding alone. Staging updates that map and publishes `Choice staged. No changes yet.` without a runtime call. Paging reprojects selections but clears the sole comparison payload and increments its generation. Return/collapse also clears that payload and generation without touching staged choices. New-token/stale/Back/root/remount invalidation clears both selections and comparison and increments the comparison generation.

- [x] **Step 5: Handle typed apply results and receipts**

Send safe IDs plus typed selections. On terminal subset completion, show receipts and either the fresh remaining review or normal receipt phase. On non-terminal work show recovery. Update the existing status line once; no success notification.

- [x] **Step 6: Run GREEN and mutation checks**

Run Step 2. Key selections by binding only and confirm stale-token test fails. Publish a delayed comparison without generation validation and confirm stale-remount test fails. Restore.

- [x] **Step 7: Commit**

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Library/library_notes_lasting_sync_state.py \
  tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py \
  Tests/Library/test_library_notes_lasting_sync_state.py \
  Tests/UI/Library_Modules/test_library_notes_sync_controller.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Library/library_notes_lasting_sync_state.py \
  tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py \
  Tests/Library/test_library_notes_lasting_sync_state.py \
  Tests/UI/Library_Modules/test_library_notes_sync_controller.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/Library/library_notes_lasting_sync_state.py \
  tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py
git diff --check
git add \
  tldw_chatbook/Library/library_notes_lasting_sync_state.py \
  tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py \
  Tests/Library/test_library_notes_lasting_sync_state.py \
  Tests/UI/Library_Modules/test_library_notes_sync_controller.py
git commit -m "feat(library): stage inline Notes sync choices"
```

---

### Task 7: Render accessible inline choices, comparison, receipts, and history

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_notes_add_from_files_canvas.py:22-60,96-108,231-305,344-490`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:26177-26240`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss:9156-9245`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Tests/Widgets/Library/test_library_notes_add_from_files_canvas.py`
- Modify: `Tests/UI/test_library_notes_files_sync_journey.py`

- [x] **Step 1: Write mounted RED tests first**

Mount the production hierarchy with exact `TldwCli.CSS_PATH`. Cover four choices, checkmark plus Selected text, Enter/Space, View/Return, read-only markup-disabled TextArea, horizontal scrolling, focus/scroll stability without review recompose, delayed comparison focus provenance, at-action Undo/Dismiss, history labels/fallback, and containment/compositor visibility at 60x20 and wide size.

Use monotonic condition waits and re-query after recomposition; never use one `pilot.pause()` as the settle oracle.

- [x] **Step 2: Run RED**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Widgets/Library/test_library_notes_add_from_files_canvas.py \
  Tests/UI/test_library_notes_files_sync_journey.py \
  -k 'conflict or comparison or receipt or history or 60x20'
```

Expected: choices remain disabled; comparison/receipt/history controls are missing.

- [x] **Step 3: Add exact messages and inline widgets**

Add messages for Choice, View, Return, Undo, Dismiss, History open/page. Give each row stable page-local DOM IDs while keeping opaque IDs in `name`. Noneligible rows remain disabled with existing copy. Render one expanded comparison only. Use `TextArea(..., read_only=True)` and set language/markup behavior explicitly.

- [x] **Step 4: Update in place where required**

When only selection/status changes, update button labels/classes, Selected text, Apply disabled/tooltip, and status `Static` directly. Do not call full `refresh(recompose=True)`. Comparison expansion may replace only the row body; capture the View control and restore focus synchronously on Return.

- [x] **Step 5: Keep async comparison off the message pump**

The screen handler schedules one named Textual worker and returns. The controller/runtime generation checks decide whether completion may publish. Cancellation releases private data; no mutation is involved. Do not add a generic task registry.

- [x] **Step 6: Add CSS and regenerate the bundle**

Edit only `_agentic_terminal.tcss`, then run:

```bash
../../.venv/bin/python tldw_chatbook/css/build_css.py
```

Assert the generated bundle contains the same selectors. Do not hand-merge the bundle.

- [x] **Step 7: Run GREEN and visibility mutations**

Run Step 2. Disable the generation check and prove stale completion fails. Remove the non-color Selected label and prove accessibility contract fails. Force comparison/receipt overflow and prove 60x20 containment fails. Restore.

- [x] **Step 8: Run scoped statics and commit**

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Widgets/Library/library_notes_add_from_files_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/Widgets/Library/test_library_notes_add_from_files_canvas.py \
  Tests/UI/test_library_notes_files_sync_journey.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Widgets/Library/library_notes_add_from_files_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/Widgets/Library/test_library_notes_add_from_files_canvas.py \
  Tests/UI/test_library_notes_files_sync_journey.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/Widgets/Library/library_notes_add_from_files_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py
../../.venv/bin/python -m compileall -q \
  tldw_chatbook/Widgets/Library/library_notes_add_from_files_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py
git diff --check
git add \
  tldw_chatbook/Widgets/Library/library_notes_add_from_files_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/css/components/_agentic_terminal.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss \
  Tests/Widgets/Library/test_library_notes_add_from_files_canvas.py \
  Tests/UI/test_library_notes_files_sync_journey.py
git commit -m "feat(library): resolve Notes sync conflicts inline"
```

---

### Task 8: Prove the complete product path and close TASK-97

**Files:**
- Modify: `Docs/User_Guide/library/notes.md`
- Modify: `backlog/tasks/task-97 - Resolve-Notes-sync-conflicts-inline.md`
- Modify: this plan as evidence is collected
- Test: all focused files named below

- [x] **Step 1: Add one joined real-authority integration matrix**

In `Tests/UI/test_library_notes_files_sync_journey.py`, use real temporary ChaChaNotes/device-state databases and a temporary sync folder. Drive Check → compare → stage → Apply for Keep file, Keep note, Keep both, and Skip. For each mutating choice, restart from fresh runtime/executor/controller objects and prove the durable outcome, receipt/history projection, and one Undo. Verify no candidate filesystem access escapes the temp root.

- [x] **Step 2: Run the exact bounded matrix**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Notes/test_notes_sync_conflicts.py \
  Tests/Notes/test_notes_sync_authority.py \
  Tests/Notes/test_notes_sync_conflict_runtime.py \
  Tests/Notes/test_notes_sync_conflict_executor.py \
  Tests/Notes/test_notes_sync_runtime.py \
  Tests/Notes/test_notes_sync_executor.py \
  Tests/Library/test_library_notes_lasting_sync_state.py \
  Tests/UI/Library_Modules/test_library_notes_sync_controller.py \
  Tests/Widgets/Library/test_library_notes_add_from_files_canvas.py \
  Tests/UI/test_library_notes_files_sync_journey.py \
  Tests/Notes/test_notes_sync_cutover.py \
  Tests/DB/test_private_sqlite_inventory.py
```

Expected: all selected tests pass; record exact count, warnings, and duration. If an untouched failure appears, reproduce the exact node against the pre-task base before claiming provenance. Do not broaden to the entire repository without a concrete failure requiring it.

- [x] **Step 3: Run exact static/governance checks**

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Notes/notes_sync_conflicts.py \
  tldw_chatbook/Notes/notes_sync_authority.py \
  tldw_chatbook/Notes/notes_scope_service.py \
  tldw_chatbook/Notes/notes_device_state_store.py \
  tldw_chatbook/Notes/notes_sync_executor.py \
  tldw_chatbook/Notes/notes_sync_runtime.py \
  tldw_chatbook/Library/library_notes_lasting_sync_state.py \
  tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py \
  tldw_chatbook/Widgets/Library/library_notes_add_from_files_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/Notes/test_notes_sync_conflicts.py \
  Tests/Notes/test_notes_sync_authority.py \
  Tests/Notes/test_notes_sync_conflict_runtime.py \
  Tests/Notes/test_notes_sync_conflict_executor.py \
  Tests/Notes/test_notes_sync_runtime.py \
  Tests/Notes/test_notes_sync_executor.py \
  Tests/Library/test_library_notes_lasting_sync_state.py \
  Tests/UI/Library_Modules/test_library_notes_sync_controller.py \
  Tests/Widgets/Library/test_library_notes_add_from_files_canvas.py \
  Tests/UI/test_library_notes_files_sync_journey.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Notes/notes_sync_conflicts.py \
  tldw_chatbook/Notes/notes_sync_authority.py \
  tldw_chatbook/Notes/notes_scope_service.py \
  tldw_chatbook/Notes/notes_device_state_store.py \
  tldw_chatbook/Notes/notes_sync_executor.py \
  tldw_chatbook/Notes/notes_sync_runtime.py \
  tldw_chatbook/Library/library_notes_lasting_sync_state.py \
  tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py \
  tldw_chatbook/Widgets/Library/library_notes_add_from_files_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/Notes/test_notes_sync_conflicts.py \
  Tests/Notes/test_notes_sync_authority.py \
  Tests/Notes/test_notes_sync_conflict_runtime.py \
  Tests/Notes/test_notes_sync_conflict_executor.py \
  Tests/Notes/test_notes_sync_runtime.py \
  Tests/Notes/test_notes_sync_executor.py \
  Tests/Library/test_library_notes_lasting_sync_state.py \
  Tests/UI/Library_Modules/test_library_notes_sync_controller.py \
  Tests/Widgets/Library/test_library_notes_add_from_files_canvas.py \
  Tests/UI/test_library_notes_files_sync_journey.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/Notes/notes_sync_conflicts.py \
  tldw_chatbook/Notes/notes_sync_authority.py \
  tldw_chatbook/Notes/notes_scope_service.py \
  tldw_chatbook/Notes/notes_device_state_store.py \
  tldw_chatbook/Notes/notes_sync_executor.py \
  tldw_chatbook/Notes/notes_sync_runtime.py \
  tldw_chatbook/Library/library_notes_lasting_sync_state.py \
  tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py \
  tldw_chatbook/Widgets/Library/library_notes_add_from_files_canvas.py
../../.venv/bin/python -m compileall -q \
  tldw_chatbook/Notes/notes_sync_conflicts.py \
  tldw_chatbook/Notes/notes_sync_authority.py \
  tldw_chatbook/Notes/notes_scope_service.py \
  tldw_chatbook/Notes/notes_device_state_store.py \
  tldw_chatbook/Notes/notes_sync_executor.py \
  tldw_chatbook/Notes/notes_sync_runtime.py \
  tldw_chatbook/Library/library_notes_lasting_sync_state.py \
  tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py \
  tldw_chatbook/Widgets/Library/library_notes_add_from_files_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py
git diff --check
```

Also rerun the private-SQLite owner, backup exclusion, legacy-path ratchet, startup invocation, privacy, and CSS generation checks already named by the focused test files. No new inventory exemption is allowed.

- [x] **Step 4: Perform isolated live UAT only after tests are green**

Use a scratch `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `TLDW_CONFIG_PATH`, and `[paths].data_dir`; disable model-catalog networking. Seed only scratch databases/folders. Drive the real TUI at 60x20 and a wide terminal through compare, all choices, partial Skip, receipt Undo/Dismiss, and history. Capture compositor/tmux evidence and fingerprint real profile/config/database paths before and after. Do not launch against the shared real database.

- [x] **Step 5: Update user docs and task evidence**

Document exact choice effects, Skip, 30-day Undo eligibility, at-action receipt behavior, durable history, and failure/unsupported behavior in `Docs/User_Guide/library/notes.md`. In TASK-97, link this plan and ADRs, add concise implementation notes, record deviations/evidence, check ACs only from proof, and use Backlog CLI to set Done only after every DoD item passes.

- [x] **Step 6: Run independent cumulative review**

Review the merge-base-to-HEAD diff for correctness, security/privacy, restart truth, UI reachability, and plan/spec compliance. Resolve every P0-P2 before closeout. Separately run a ponytail pass and remove any abstraction not required by an AC or crash boundary.

- [x] **Step 7: Commit closeout docs**

```bash
backlog task 97 --plain
backlog task edit 97 -s Done
git diff --check
git add \
  Docs/User_Guide/library/notes.md \
  Docs/superpowers/plans/2026-08-22-task-97-notes-sync-conflict-resolution.md \
  'backlog/tasks/task-97 - Resolve-Notes-sync-conflicts-inline.md'
git commit -m "docs(notes): complete TASK-97 conflict resolution"
git status --short
```

Expected: exact TASK-97 is Done with all nine ACs checked, the worktree is clean, and the plan records only genuinely completed steps.

## Completion evidence — 2026-08-23

- The joined real-authority Check → compare → stage → Apply matrix covers Keep
  file, Keep note, Keep both, and Skip with disposable ChaChaNotes/device-state
  databases and a disposable folder. Mutating cases rebuild the runtime,
  executor, and controller before proving history and one Undo; the sentinel
  and invalid-relative-path assertions prove the filesystem boundary.
- Task 8 TDD caught three production-boundary mismatches: ChaChaNotes returned
  a `datetime` timestamp, the active-only Notes reader hid a successfully
  written tombstone, and a restarted partial review rejected its terminal
  `NO_CHANGE` rows. Focused RED tests reproduced each before the narrow fixes.
- A follow-up cumulative gate added RED regressions for no-op-only Apply,
  conflict-free duplicate observation, transient history gating, and shipped
  control labels. The narrow fixes require one of the runtime's four manually
  executable safe action kinds or a non-Skip conflict choice, skip label
  authority when no eligible conflict exists, and keep persisted-root history
  reachable while its paged read reports empty or unavailable state. The transient
  history probe was deleted from the runtime, controller port, and snapshot.
- The corrected exact 12-file matrix passed: **744 passed, 8 warnings in
  171.88s**. The
  focused private-SQLite owner/backup, legacy cutover/startup, privacy, and
  unsupported-write governance nodes passed within that matrix; the earlier
  isolated governance rerun was **10 passed, 7 warnings in 19.55s**.
- Ruff check and format, compileall, CSS generation, and `git diff --check`
  passed. The exact MyPy command reports the same pre-existing **144 errors in
  5 files** at the Task 8 starting SHA `96c502cdd`; this closeout adds none.
- Live UAT redirected `HOME`, all XDG roots, `TLDW_CONFIG_PATH`, and
  `[paths].data_dir` into one disposable profile, disabled model-catalog and
  model-network activity, and seeded only scratch authorities. Wide and 60x20
  compositor/tmux captures prove comparison, all choices, partial Skip,
  receipt Dismiss/Undo, restart, and durable history. The shared config,
  ChaChaNotes DB, and Notes sync-state DB SHA-256 manifests match exactly before
  and after; only the four expected scratch-root files were accessed.
- Cumulative merge-base review found and fixed the restarted mixed-plan defect,
  the inaccurate draft-doc encryption claim, and the follow-up P2s above. The
  final spec gate aligned `MOVE_FILE` with the runtime's intentional manual
  rejection and caught one line-wrapped obsolete action label.
  Correctness, privacy, recovery, restart truth, focus/keyboard reachability,
  and plan/spec compliance have no remaining P0-P2 findings. The full ponytail
  pass retained the joined fixture helpers as test boundary setup, added no
  dependency or abstraction, and deleted the redundant transient-history
  query rather than replacing it.

---

## Implementation constraints carried into every task

- Write the failing test before production code and capture the exact RED reason.
- Mutation-test each new guard individually; a passing unmutated test is not proof.
- Use deterministic `asyncio.Event`/barrier interleavings, not sleeps or absolute event counts.
- A restart test must discard every request, snapshot, controller, runtime, executor, and fake that carries authority.
- Re-query Textual widgets after recomposition and poll the exact asserted condition with a monotonic deadline.
- Preserve existing automatic blocker behavior, deletion behavior, activation behavior, and legacy-path ratchets.
- Never persist/log comparison text, titles/paths used only for interactive labels, hashes, raw exceptions, or recovery bytes.
- No new dependency, modal, private SQLite owner, filesystem authority, schema table, or speculative abstraction.
- Stop and update TASK-97 ACs before implementing any behavior not already covered.
