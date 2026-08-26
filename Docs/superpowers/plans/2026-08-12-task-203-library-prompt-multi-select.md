# TASK-203 Library Prompt Multi-Select Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a persistent cross-search Prompt/Recipe selection basket with exact selected export and atomic all-or-nothing delete/Undo in the local Library Prompt list.

**Architecture:** A small shared immutable contract module defines strict batch targets and the database-produced delete/restore results. `PromptsDatabase` owns one `BEGIN IMMEDIATE` transaction and transaction-local mutation helpers; local and scope services pass the typed result through without post-commit normalization. Existing pure Prompt state owns selection, the existing canvas renders it, and `LibraryScreen` supplies thin lifecycle, export, confirmation, mutation, focus, and navigation wiring.

**Tech Stack:** Python 3.11+, frozen dataclasses, SQLite/FTS5, Textual 8.x, pytest/Hypothesis, Loguru, existing RuntimePolicy and Chatbook export seams.

**Design:** `Docs/superpowers/specs/2026-08-12-task-203-library-prompt-multi-select-design.md`

**Decision:** `backlog/decisions/060-atomic-local-prompt-batch-mutations.md`

**Required implementation skills:** `@superpowers:test-driven-development`, `@superpowers:systematic-debugging` for unexpected failures, `@ponytail` throughout, and `@impeccable` plus `@textual-tui` for the mounted visual/accessibility task.

---

## File structure

### Create

- `tldw_chatbook/Prompt_Management/prompt_batch_models.py` — shared, Textual-free immutable batch targets and database-produced delete/restore result contracts.
- `Tests/Prompt_Management/test_prompt_batch_models.py` — strict constructor and privacy/repr contract tests.
- `Tests/Prompts_DB/test_prompts_db_batch_mutations.py` — real file-backed SQLite atomicity, rollback, recovery, concurrency, compatibility, and diagnostic tests.

### Modify

- `tldw_chatbook/DB/Prompts_DB.py:2113-2469` — extract transaction-local delete/restore primitives, add strict batch entry points, and preserve legacy single APIs.
- `tldw_chatbook/Prompt_Management/prompt_scope_service.py:634-681,1559-1641` — add typed local batch delegation and local-only scope methods with exactly one existing policy decision.
- `tldw_chatbook/Library/library_prompts_state.py:1640-1710,1870-1920` — add the immutable selection basket, row version/checked projection, counts, and plural receipt projection.
- `tldw_chatbook/Widgets/Library/library_prompts_canvas.py:126-360` — render normal/select toolbars, checked rows, fixed disabled reasons, progress, and plural receipt using existing layout primitives.
- `tldw_chatbook/UI/Screens/library_screen.py:2750-2840,5737-5768,7950-8080,8580-8790,12270-12340,13580-13735,16840-16910,18970-19280,22755-22775` — own selection lifecycle, selected export, shared single/bulk mutation settlement, atomic Undo, focus, and route vetoes.
- `Tests/Library/test_library_prompts_state.py` — pure selection/state RED-GREEN tests.
- `Tests/Prompt_Management/test_local_prompt_service.py` — typed local batch pass-through and legacy compatibility tests.
- `Tests/Prompt_Management/test_prompt_scope_service.py` — local-only validation-before-policy, exactly-once policy, pass-through, and server-refusal tests.
- `Tests/UI/test_library_prompts_canvas.py` — canvas, mounted selection/export/delete/Undo/focus/navigation, literal-text, and narrow geometry tests.
- `Tests/UI/test_library_shell.py` and `Tests/UI/test_screen_navigation.py` — selection/source lifecycle and app-level mutation veto coverage at the existing navigation seams.
- `Docs/User_Guide/library/prompts.md` — document Select mode, cross-search basket, selected export, atomic delete, and Undo.
- `Docs/security/production-diagnostic-inventory.json` — update only owners whose changed persistent diagnostics alter the scanner digest.
- `backlog/tasks/task-203 - Library-Prompts-multi-select-bulk-actions-in-the-list.md` — track the approved plan, completed ACs, ADR, verification, and implementation notes.

### Explicitly unchanged

- Chatbook archive/export implementation and schema.
- RuntimePolicy registry/action IDs: batch delete uses `prompts.delete.local`; batch restore uses `prompts.update.local` exactly once.
- `PromptDeleteConfirmationModal` presentation contract unless a RED test proves its already-plural request/copy insufficient.
- Shared TCSS and generated CSS unless real-bundle compositor evidence exposes a concrete defect.
- Server Prompt batch APIs.

---

### Task 1: Add strict batch contracts and the pure selection basket

**Files:**
- Create: `tldw_chatbook/Prompt_Management/prompt_batch_models.py`
- Create: `Tests/Prompt_Management/test_prompt_batch_models.py`
- Modify: `tldw_chatbook/Library/library_prompts_state.py:1640-1710,1870-1920`
- Modify: `Tests/Library/test_library_prompts_state.py`

- [ ] **Step 1: Write RED contract tests**

Add constructor matrices proving exact tuples, exact positive non-bool SQLite-range integers, unique canonical IDs, positive versions, supported artifact types, nonempty results, deterministic ordering, and repr-hidden identities. Use this public shape:

```python
target = PromptBatchTarget(local_id=7, expected_version=3)
entry = PromptDeleteReceiptEntry(
    local_id=7,
    title="Literal [name]",
    artifact_type="recipe",
    tombstone_version=4,
)
deleted = PromptBatchDeleteResult(entries=(entry,))
restored = PromptBatchRestoreResult(
    entries=(PromptRestoreResultEntry(local_id=7, restored_version=5),)
)
assert deleted.targets == (PromptBatchTarget(7, 4),)
assert "7" not in repr(target)
```

Add pure basket tests using this immutable API:

```python
basket = PromptSelectionBasket()
selected = basket.toggle(PromptSelectionEntry(7, 3, "Literal [name]", "recipe"))
same = selected.select_page((PromptSelectionEntry(7, 99, "new", "recipe"),))
assert same.entries[0].expected_version == 3
assert same.generation == selected.generation
assert same.canonical_entries == selected.entries
```

Cover cross-page accumulation, toggling off/on captures the newer version, duplicate suppression, `clear()`, ascending numeric canonical order, total/on-page counts, generation changes only on semantic changes, and malformed page rows failing closed.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Prompt_Management/test_prompt_batch_models.py Tests/Library/test_library_prompts_state.py -q -k 'batch or selection'
```

Expected: collection/import failures for the absent model classes and selection API.

- [ ] **Step 3: Implement the minimum immutable contracts**

In `prompt_batch_models.py`, use only frozen/slots dataclasses and small private validators. Keep raw IDs/versions out of repr:

```python
@dataclass(frozen=True, slots=True, repr=False)
class PromptBatchTarget:
    local_id: int
    expected_version: int

@dataclass(frozen=True, slots=True, repr=False)
class PromptDeleteReceiptEntry:
    local_id: int
    title: str
    artifact_type: Literal["prompt", "recipe"]
    tombstone_version: int

@dataclass(frozen=True, slots=True, repr=False)
class PromptBatchDeleteResult:
    entries: tuple[PromptDeleteReceiptEntry, ...]

    @property
    def targets(self) -> tuple[PromptBatchTarget, ...]:
        return tuple(
            PromptBatchTarget(entry.local_id, entry.tombstone_version)
            for entry in self.entries
        )
```

Add equivalent strict restore entries/results. Validate canonical ascending IDs in result types so the database cannot construct a UI receipt with ambiguous order.

In `library_prompts_state.py`, add `PromptSelectionEntry` and `PromptSelectionBasket`; extend `PromptListRow` with positive `version` and `checked`; extend `PromptsListState` with `select_mode`, `total_selected`, and `selected_on_page`. Update `build_prompt_browse_list_state(..., selection=..., select_mode=...)` to project the basket without mutating it.

- [ ] **Step 4: Run focused GREEN tests**

Run the command from Step 2.

Expected: all selected tests pass.

- [ ] **Step 5: Mutation-check captured-version preservation**

Temporarily make `select_page` overwrite an existing entry. Run the existing-version test and verify it fails; restore the implementation and rerun it GREEN.

- [ ] **Step 6: Commit the contracts/state slice**

```bash
git add tldw_chatbook/Prompt_Management/prompt_batch_models.py \
  tldw_chatbook/Library/library_prompts_state.py \
  Tests/Prompt_Management/test_prompt_batch_models.py \
  Tests/Library/test_library_prompts_state.py
git commit -m "feat(prompts): add bulk selection contracts"
```

---

### Task 2: Implement atomic database delete and restore

**Files:**
- Create: `Tests/Prompts_DB/test_prompts_db_batch_mutations.py`
- Modify: `tldw_chatbook/DB/Prompts_DB.py:2113-2469`
- Test: `Tests/Prompts_DB/test_prompts_db_pytest.py`
- Test: `Tests/Prompts_DB/test_prompts_db_legacy.py`

- [ ] **Step 1: Write real-SQLite RED tests for strict batch behavior**

Use a file-backed `PromptsDatabase`, real Prompt/Recipe rows, keywords, FTS, collection memberships, and `sync_log`. Pin:

```python
result = db.soft_delete_prompts(
    (PromptBatchTarget(prompt_id, 1), PromptBatchTarget(recipe_id, 1))
)
assert [entry.local_id for entry in result.entries] == sorted((prompt_id, recipe_id))
assert db.search_prompts("unique body") == []

restored = db.restore_deleted_prompts(result.targets)
assert tuple(entry.local_id for entry in restored.entries) == tuple(
    entry.local_id for entry in result.entries
)
```

Add parameterized invalid input (list, empty tuple, duplicate ID, bool, zero, negative, signed overflow, nonpositive version), stale/missing target, forced failure on the second mutation, exact keyword/FTS/sync restoration, one `BEGIN IMMEDIATE`, validation before the first write, and no partial mutation.

Inject a result-constructor failure and assert every row remains unchanged. This proves DTO construction occurs before `transaction()` exits/commits.

- [ ] **Step 2: Add RED legacy-compatibility assertions**

Pin integer/name/UUID lookup, optional version, missing delete returning `False`, restored mapping shape, and the original one-item version progression. These must still exercise the shared transaction-local primitives.

- [ ] **Step 3: Run database RED**

```bash
../../.venv/bin/python -m pytest Tests/Prompts_DB/test_prompts_db_batch_mutations.py \
  Tests/Prompts_DB/test_prompts_db_pytest.py \
  Tests/Prompts_DB/test_prompts_db_legacy.py -q \
  -k 'batch or soft_delete_prompt or restore_deleted_prompt'
```

Expected: new batch tests fail because `soft_delete_prompts` / `restore_deleted_prompts` do not exist; compatibility controls remain green.

- [ ] **Step 4: Extract transaction-local primitives**

Refactor only the mutation core below the public APIs:

```python
def _delete_prompt_in_transaction(
    self,
    conn: sqlite3.Connection,
    *,
    row: sqlite3.Row,
    expected_version: int,
) -> PromptDeleteReceiptEntry: ...

def _restore_prompt_in_transaction(
    self,
    conn: sqlite3.Connection,
    *,
    row: sqlite3.Row,
    expected_version: int,
) -> PromptRestoreResultEntry: ...
```

The helpers perform row mutation, keyword unlink/recovery, FTS, sync events, and return already-validated entries. They emit no success diagnostic/metric and never open/commit a transaction.

- [ ] **Step 5: Add strict batch entry points**

Implement:

```python
def soft_delete_prompts(
    self, targets: tuple[PromptBatchTarget, ...]
) -> PromptBatchDeleteResult:
    targets = validate_prompt_batch_targets(targets)
    with self.transaction(immediate=True) as conn:
        rows = self._resolve_active_batch_rows(conn, targets)
        entries = tuple(
            self._delete_prompt_in_transaction(
                conn, row=rows[target.local_id], expected_version=target.expected_version
            )
            for target in targets
        )
        result = PromptBatchDeleteResult(entries)
    # fixed operation + aggregate count only
    return result
```

Add the symmetric restore method. Resolve and validate all rows/recovery payloads before the first write. Convert SQLite failures into bounded `DatabaseError` without exception message chaining; log only fixed operation, aggregate count, and exception category.

- [ ] **Step 6: Rebuild legacy single wrappers on the primitives**

Keep the existing signatures and return behavior. Resolve the legacy identifier inside `BEGIN IMMEDIATE`; return `False` for missing delete; derive the current version when `expected_version` is omitted; call the same private helper; construct the typed result before commit; return `True` or the established restored mapping only after commit. Keep server-facing behavior unchanged.

- [ ] **Step 7: Run focused and full assigned database GREEN**

```bash
../../.venv/bin/python -m pytest Tests/Prompts_DB/test_prompts_db_batch_mutations.py -q
../../.venv/bin/python -m pytest Tests/Prompts_DB -q
```

Expected: all tests pass.

- [ ] **Step 8: Mutation-check atomicity**

Individually mutate: `immediate=True` to false, validation after first write, split commit, and result construction after the transaction. Each exact test must fail. Restore and rerun the focused file GREEN.

- [ ] **Step 9: Commit the database slice**

```bash
git add tldw_chatbook/DB/Prompts_DB.py \
  Tests/Prompts_DB/test_prompts_db_batch_mutations.py \
  Tests/Prompts_DB/test_prompts_db_pytest.py \
  Tests/Prompts_DB/test_prompts_db_legacy.py
git commit -m "feat(prompts): add atomic batch mutations"
```

---

### Task 3: Expose typed batch methods through local and scope services

**Files:**
- Modify: `tldw_chatbook/Prompt_Management/prompt_scope_service.py:634-681,1559-1641`
- Modify: `Tests/Prompt_Management/test_local_prompt_service.py`
- Modify: `Tests/Prompt_Management/test_prompt_scope_service.py`
- Verify unchanged: `tldw_chatbook/runtime_policy/registry.py`
- Verify unchanged: `Tests/RuntimePolicy/test_runtime_policy_core.py`

- [ ] **Step 1: Write service RED tests**

Pin exact keyword-only batch signatures and typed object identity:

```python
deleted = await scope.delete_prompts(
    mode="local", targets=(PromptBatchTarget(7, 3), PromptBatchTarget(9, 2))
)
assert deleted is local.deleted_result
assert policy.actions == ["prompts.delete.local"]

restored = await scope.restore_deleted_prompts(
    mode="local", targets=deleted.targets
)
assert restored is local.restored_result
assert policy.actions[-1:] == ["prompts.update.local"]
```

Cover list/empty/malformed/duplicate/bool/overflow validation before policy and adapter access, server refusal before policy/backend access, missing local capability, and no `_normalize_prompt_record` call after the local method returns.

- [ ] **Step 2: Run service RED**

```bash
../../.venv/bin/python -m pytest Tests/Prompt_Management/test_local_prompt_service.py \
  Tests/Prompt_Management/test_prompt_scope_service.py -q -k 'batch_prompt or prompt_batch'
```

Expected: missing batch-method failures only.

- [ ] **Step 3: Add minimal local delegation and scope routing**

Add synchronous local methods that return the database object unchanged:

```python
def delete_prompts(
    self, targets: tuple[PromptBatchTarget, ...]
) -> PromptBatchDeleteResult:
    return self.prompt_db.soft_delete_prompts(targets)
```

Add equivalent restore delegation. In `PromptScopeService`, normalize mode, require local, validate strict targets, enforce one existing action, call the local method, and return the typed result unchanged. Do not add RuntimePolicy actions, result normalizers, server calls, or response mappings.

- [ ] **Step 4: Run service and RuntimePolicy GREEN**

```bash
../../.venv/bin/python -m pytest Tests/Prompt_Management/test_local_prompt_service.py \
  Tests/Prompt_Management/test_prompt_scope_service.py \
  Tests/RuntimePolicy/test_runtime_policy_core.py -q
```

Expected: all tests pass and the registry snapshot remains unchanged.

- [ ] **Step 5: Mutation-check ordering and pass-through**

Move policy before validation, enforce policy once per item, and normalize/copy the returned DTO in three separate mutations. Each targeted test must fail. Restore and rerun GREEN.

- [ ] **Step 6: Commit the service slice**

```bash
git add tldw_chatbook/Prompt_Management/prompt_scope_service.py \
  Tests/Prompt_Management/test_local_prompt_service.py \
  Tests/Prompt_Management/test_prompt_scope_service.py
git commit -m "feat(prompts): expose atomic batch service"
```

---

### Task 4: Render select mode and plural receipts in the existing canvas

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_prompts_canvas.py:126-360`
- Modify: `Tests/UI/test_library_prompts_canvas.py`

- [ ] **Step 1: Write canvas RED tests**

Add pure/mounted cases for:

- normal toolbar rows: `Sort` + `Select`, then `Import…` + `Export…`;
- select summary `7 selected · 2 on this page`;
- management row `Select page`, `Clear all`, `Done`;
- action row `Export selected`, `Delete selected`;
- literal checked/unchecked row prefixes for Prompt and Recipe names containing Rich markup/Unicode;
- zero-selection reason precedence: `Select one or more items to use bulk actions.`;
- nonempty basket plus loading/error reason: `Current page is unavailable; selected items remain available for Export or Delete.`;
- disabled Select, Select page, Export selected, and Delete selected labels use
  `library_disabled_action_label(...)`'s existing `○` non-colour marker and
  each disabled control has a fixed explanatory tooltip in addition to the
  visible reason line;
- Select page disabled while loading/error but whole-basket actions enabled when nonempty;
- mutation progress and all selection/receipt actions disabled;
- single receipt preserves `✓ deleted · Prompt/Recipe · Name`, plural receipt uses `✓ deleted · N items`;
- exactly the incumbent scroll ownership.

- [ ] **Step 2: Run canvas RED**

```bash
../../.venv/bin/python -m pytest Tests/UI/test_library_prompts_canvas.py -q \
  -k 'select_mode or selection_toolbar or plural_delete_receipt or bulk_disabled_reason'
```

Expected: missing controls/labels and old receipt-shape failures.

- [ ] **Step 3: Implement presentation only**

Keep `LibraryPromptsListCanvas` stateless. Read selection fields from `PromptsListState`, use existing `.ds-toolbar`, `.library-canvas-action`, `.library-toolbar-count`, `Static(markup=False)`, escaped Button labels, and `library_disabled_action_label`. Use fixed tooltips: `Nothing here to select yet.` for disabled Select, `Current page is unavailable.` for disabled Select page during loading/error, and `Select one or more items first.` for disabled selected Export/Delete. Add stable IDs:

```text
library-prompts-select
library-prompts-selection-summary
library-prompts-select-page
library-prompts-clear-selection
library-prompts-selection-done
library-prompts-export-selected
library-prompts-delete-selected
library-prompts-selection-reason
library-prompts-mutation-progress
```

Attach the full row projection to each row Button (`prompt_id`, `prompt_version`, `artifact_type`, literal name) so the screen never re-reads mutable widgets to build a selection entry.

- [ ] **Step 4: Run focused canvas GREEN**

Run the command from Step 2.

Expected: all selected tests pass.

- [ ] **Step 5: Commit the presentation slice**

```bash
git add tldw_chatbook/Widgets/Library/library_prompts_canvas.py \
  Tests/UI/test_library_prompts_canvas.py
git commit -m "feat(library): render prompt selection mode"
```

---

### Task 5: Wire selection lifecycle and exact selected export

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:2750-2840,7950-8080,8580-8790,12270-12340,13580-13735,16840-16910,22755-22775`
- Modify: `Tests/UI/test_library_prompts_canvas.py`
- Modify: `Tests/UI/test_library_shell.py`
- Test: `Tests/Library/test_library_export_scope_ids.py`

- [ ] **Step 1: Write screen lifecycle/export RED tests**

Mount the real Library screen and select rows across two literal searches, multiple pages, sort order, and collection scope. Assert the basket survives each exact browse settlement and an Export canvas round trip. Assert `Select page` does not replace an existing captured version.

Pin exact selected export:

```python
assert screen._library_export_scope == ExportScope(
    kind="prompts", ids=(str(low_id), str(high_id))
)
```

Use real SQLite to prove latest active content is exported and a missing/deleted selected ID aborts the Prompt-bearing archive. Preserve the basket after export success, failure, cancellation, and Back.

Pin clears on Done, Clear all, successful delete (delete itself lands in Task 6), editor/create entry, another Library source, and unmount/Library exit; assert it is absent from `save_state()`. Done/navigation use bounded `Selection discarded · N prompts`; Clear all/unmount are silent.

- [ ] **Step 2: Run lifecycle/export RED**

```bash
../../.venv/bin/python -m pytest Tests/UI/test_library_prompts_canvas.py \
  Tests/UI/test_library_shell.py Tests/Library/test_library_export_scope_ids.py -q \
  -k 'prompt_selection or prompts_export_selected'
```

Expected: missing screen selection handlers/state and whole-Prompt export scope instead of selected IDs.

- [ ] **Step 3: Add one screen-owned basket and shared clear helper**

Initialize only:

```python
self._library_prompt_select_mode = False
self._library_prompt_selection = PromptSelectionBasket()
```

Pass both into `_build_library_prompts_state`; add handlers that replace the immutable basket and recompose with exact focus restoration. Branch `.library-prompt-row` presses: toggle while select mode is active, otherwise retain the existing editor path.

Add one `_clear_library_prompt_selection(*, announce: bool)` lifecycle helper and invoke it only at the approved boundaries. Do not clear on browse query/page/sort/collection changes or Export admission/return.

- [ ] **Step 4: Route selected export through the existing canvas**

Keep `#library-prompts-export` as whole-source export. Add `#library-prompts-export-selected` to snapshot `canonical_entries`, convert numeric order to strings, and call:

```python
await self._open_library_export_canvas(
    ExportScope(
        kind="prompts",
        ids=tuple(str(entry.local_id) for entry in entries),
    )
)
```

Do not add selection policy calls or modify generic exporter behavior.

- [ ] **Step 5: Run lifecycle/export GREEN**

Run the command from Step 2.

Expected: all selected tests pass.

- [ ] **Step 6: Mutation-check persistence and export ordering**

Temporarily clear the basket on search and build `ExportScope.ids` from visible/current order. Each exact regression must fail. Restore and rerun GREEN.

- [ ] **Step 7: Commit the selection orchestration slice**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_prompts_canvas.py \
  Tests/UI/test_library_shell.py
git commit -m "feat(library): persist prompt bulk selection"
```

---

### Task 6: Unify single/bulk delete, atomic Undo, and receipt ownership

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:5737-5768,12270-12340,18970-19280`
- Modify: `Tests/UI/test_library_prompts_canvas.py`
- Modify: `Tests/UI/test_screen_navigation.py`

- [ ] **Step 1: Write RED mutation/receipt tests**

Add mounted real-SQLite tests for mixed Prompt/Recipe selected deletion, modal counts and bounded literal preview, exact generation fingerprint, duplicate/late settlement no-op, stale database selection preserving the full basket, old receipt preservation after failed new delete, and successful delete clearing the basket only after commit.

Pin both editor single delete and selected bulk delete call the same `delete_prompts(mode="local", targets=...)` screen worker path. Pin one plural typed result becomes the receipt and one-item display remains unchanged.

Add atomic Undo tests for full restoration, stale/missing/conflicting tombstone restoring none and preserving the full receipt, exact keyword/FTS/membership recovery, rail count, browse refresh, and nearest-row/Select fallback focus.

- [ ] **Step 2: Write RED route-owner tests**

Hold a batch service call on a thread Event and prove every row toggle, create/update/delete/Undo/receipt action, Export transition, another Library source, and `flush_pending_work()` are refused. In `Tests/UI/test_screen_navigation.py`, invoke real app navigation and assert the mounted Library screen/receipt owner remains current until settlement.

- [ ] **Step 3: Run mutation/navigation RED**

```bash
../../.venv/bin/python -m pytest Tests/UI/test_library_prompts_canvas.py \
  Tests/UI/test_screen_navigation.py -q \
  -k 'bulk_delete or batch_undo or prompt_mutation_route or prompt_receipt_owner'
```

Expected: missing batch screen path, subset Undo behavior, and incomplete navigation veto failures.

- [ ] **Step 4: Collapse single and selected delete into one transaction path**

Keep the existing `PromptDeleteConfirmationModal`. Build its request from either the one editor target or the basket snapshot. For selected deletion, use an opaque generation token only; on settlement compare pending token, live generation, route, and mutation flag.

Replace the current boolean single worker with one shared coroutine:

```python
async def _delete_library_prompts(
    self,
    targets: tuple[PromptBatchTarget, ...],
    *,
    selection_generation: int | None,
    editor_prompt_id: int | None,
) -> None: ...
```

Call `PromptScopeService.delete_prompts` once through `_run_library_service_call(..., isolate_in_worker=True)`. Preserve the prior receipt until the typed result returns successfully. On selected success, clear the basket; on editor success, reset editor state. Refresh rail/list once.

- [ ] **Step 5: Replace Undo with one batch restore call**

Call `restore_deleted_prompts(mode="local", targets=receipt.targets)` once. Accept only the typed result, clear only the identical current receipt on full success, and keep the receipt on every failure. Use fixed user copy and bounded aggregate/category diagnostics only.

- [ ] **Step 6: Close all route-loss seams**

Return early from every Library source/editor/create/export transition while `_library_prompts_mutation_in_flight`. Make `flush_pending_work()` return `False` immediately during the admitted Prompt mutation so app-level navigation cannot unmount the receipt owner. Do not create a second flag or worker group.

- [ ] **Step 7: Run mutation/navigation GREEN**

Run the command from Step 3, then:

```bash
../../.venv/bin/python -m pytest Tests/UI/test_library_prompts_canvas.py \
  Tests/UI/test_library_shell.py Tests/UI/test_screen_navigation.py -q \
  -k 'prompt or library_export_back'
```

Expected: all selected tests pass.

- [ ] **Step 8: Mutation-check receipt and atomic Undo**

Separately clear the prior receipt before delete, loop one restore per entry, and permit app navigation while the mutation flag is true. Each exact regression must fail. Restore and rerun GREEN.

- [ ] **Step 9: Commit the transaction-owner slice**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_prompts_canvas.py \
  Tests/UI/test_screen_navigation.py
git commit -m "feat(library): apply atomic prompt bulk actions"
```

---

### Task 7: Harden accessibility, privacy, and integrated behavior

**Files:**
- Modify only if a focused RED requires it: `tldw_chatbook/Widgets/Library/library_prompts_canvas.py`
- Modify: `Tests/UI/test_library_prompts_canvas.py`
- Modify: `Docs/security/production-diagnostic-inventory.json`
- Test: `Tests/Architecture/test_persistent_diagnostic_inventory.py` or the repository's current diagnostic-inventory test file selected by `scripts/check_persistent_diagnostic_inventory.py`

- [ ] **Step 1: Run the focused integrated matrix**

```bash
../../.venv/bin/python -m pytest \
  Tests/Prompt_Management/test_prompt_batch_models.py \
  Tests/Prompts_DB/test_prompts_db_batch_mutations.py \
  Tests/Prompt_Management/test_local_prompt_service.py \
  Tests/Prompt_Management/test_prompt_scope_service.py \
  Tests/Library/test_library_prompts_state.py \
  Tests/Library/test_library_export_scope_ids.py \
  Tests/UI/test_library_prompts_canvas.py \
  Tests/UI/test_library_shell.py \
  Tests/UI/test_screen_navigation.py \
  Tests/RuntimePolicy/test_runtime_policy_core.py -q
```

Expected: all TASK-203 affected tests pass; classify any unrelated baseline failures with exact node IDs before changing code.

- [ ] **Step 2: Add/run real-bundle compositor RED-GREEN checks**

Use a temporary pytest harness under `Tests/UI/` with the real generated bundle. Capture normal/select empty/select multi-page/loading/error/confirmation/progress/receipt states at 64x24 and 120x40. Assert:

- existing scroll-owner count unchanged;
- summary, both action rows, reason/progress, confirmation, Undo/Dismiss intersect the viewport;
- keyboard focus reaches every enabled action;
- literal `[markup]`, Unicode, and mixed Prompt/Recipe names paint verbatim;
- no enabled label is clipped/overlapped and live contrast meets existing thresholds.

If and only if a real compositor assertion fails, make the smallest widget-local layout correction and rerun RED-GREEN. Remove the temporary harness before staging; retain ignored SVG/PNG/JSON evidence under `.superpowers/sdd/2026-08-12-task-203-prompt-bulk-actions/visual-closeout/`.

- [ ] **Step 3: Audit persistent diagnostics**

Run:

```bash
../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
```

For every TASK-203 changed owner, assert scanner call count/digest and privacy tests match. Update only exact changed-owner entries and summary totals. Do not regenerate unrelated branch drift. Adversarial tests must prove Prompt name/body/ID/version/selection/receipt/exception message/traceback sentinels are absent.

- [ ] **Step 4: Run static and source/bundle gates**

```bash
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Prompt_Management/prompt_batch_models.py \
  tldw_chatbook/DB/Prompts_DB.py \
  tldw_chatbook/Prompt_Management/prompt_scope_service.py \
  tldw_chatbook/Library/library_prompts_state.py \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Prompt_Management/prompt_batch_models.py \
  tldw_chatbook/DB/Prompts_DB.py \
  tldw_chatbook/Prompt_Management/prompt_scope_service.py \
  tldw_chatbook/Library/library_prompts_state.py \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/Prompt_Management/test_prompt_batch_models.py \
  Tests/Prompts_DB/test_prompts_db_batch_mutations.py \
  Tests/Prompt_Management/test_local_prompt_service.py \
  Tests/Prompt_Management/test_prompt_scope_service.py \
  Tests/Library/test_library_prompts_state.py \
  Tests/UI/test_library_prompts_canvas.py \
  Tests/UI/test_library_shell.py \
  Tests/UI/test_screen_navigation.py
../../.venv/bin/python -m py_compile \
  tldw_chatbook/Prompt_Management/prompt_batch_models.py \
  tldw_chatbook/DB/Prompts_DB.py \
  tldw_chatbook/Prompt_Management/prompt_scope_service.py \
  tldw_chatbook/Library/library_prompts_state.py \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py
git diff --check
```

Run exact focused typing and CSS parity commands:

```bash
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/Prompt_Management/prompt_batch_models.py \
  tldw_chatbook/Prompt_Management/prompt_scope_service.py \
  tldw_chatbook/Library/library_prompts_state.py \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py
../../.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
```

If TCSS changed after a proven visual RED, first run `../../.venv/bin/python tldw_chatbook/css/build_css.py`, then rerun the sync command. Otherwise assert no CSS diff.

- [ ] **Step 5: Run one final Impeccable detector**

If the active Impeccable context has not already run the detector after the
last UI edit, run exactly once:

```bash
node /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.agents/skills/impeccable/scripts/detect.mjs --json \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py
```

Expected: `[]` or only findings proven pre-existing by blame. Do not duplicate
the run when the active Impeccable hook already supplied final-candidate output.

- [ ] **Step 6: Commit hardening changes**

```bash
git add Tests/UI/test_library_prompts_canvas.py \
  Docs/security/production-diagnostic-inventory.json \
  tldw_chatbook/Widgets/Library/library_prompts_canvas.py \
  tldw_chatbook/css/components/_agentic_terminal.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "fix(library): harden prompt bulk actions"
```

Skip the commit if Task 7 produces no tracked change.

---

### Task 8: Documentation, task hygiene, and final verification

**Files:**
- Modify: `Docs/User_Guide/library/prompts.md`
- Modify: `backlog/tasks/task-203 - Library-Prompts-multi-select-bulk-actions-in-the-list.md`
- Modify only if a genuinely new incident generalizes: `backlog/docs/lessons-testing-evidence.md`

- [ ] **Step 1: Update the user guide**

Document how to enter Select mode, cross-search/page/collection persistence, Select page versus Clear all, selected export, atomic delete, Undo, and the exact clear boundaries. State that bulk tagging is not part of this surface.

- [ ] **Step 2: Run final affected verification from a frozen source tree**

Run the Task 7 integrated matrix, full `Tests/Prompts_DB`, full `Tests/Prompt_Management`, full Prompt canvas, Prompt-focused Library shell/navigation selections, RuntimePolicy core, static checks, diagnostic owner equality, and CSS parity when applicable. Do not edit source while tests using `inspect.getsource()` are running.

Then run the repository-required full suite from the frozen candidate:

```bash
../../.venv/bin/python -m pytest -q
```

Expected: zero failures. If an inherited/environmental failure remains, capture
the exact node ID and unchanged-base reproduction and obtain explicit user
approval for that exception before checking ACs or marking TASK-203 Done.

- [ ] **Step 3: Perform final correctness and YAGNI review**

Verify line-by-line against all seven ACs and ADR-060. Confirm there is no generic bulk framework/controller, new dependency, schema change, server batch fallback, duplicate mutation flag/worker group, post-commit normalization, per-item service loop, or new scroll owner.

- [ ] **Step 4: Update TASK-203 through Backlog CLI**

Check all seven ACs only after evidence exists, add concise Implementation Notes with approach/files/ADR/tests/trade-offs, record any accepted inherited exclusions, and set Done:

```bash
backlog task edit 203 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 \
  --check-ac 5 --check-ac 6 --check-ac 7
backlog task edit 203 --notes "Implemented persistent local Prompt/Recipe selection, exact selected Chatbook export, and one atomic version-checked delete/Undo family under ADR-060. Added strict typed batch contracts, BEGIN IMMEDIATE database mutations with pre-commit result construction, typed service pass-through, existing-canvas selection controls, shared route admission, real-SQLite and mounted Textual coverage, privacy-safe diagnostics, user-guide updates, and final static/visual verification."
backlog task edit 203 -s Done
```

Add a lesson only if this work produced a new evidenced reusable trap; do not invent one.

- [ ] **Step 5: Commit closeout**

```bash
git add Docs/User_Guide/library/prompts.md \
  'backlog/tasks/task-203 - Library-Prompts-multi-select-bulk-actions-in-the-list.md'
git commit -m "docs(library): complete prompt bulk actions"
```

- [ ] **Step 6: Verify branch state**

```bash
git status --short
git log --oneline origin/dev..HEAD
git diff --check origin/dev...HEAD
```

Expected: clean worktree, only TASK-203 commits/files, and no diff-check errors.
