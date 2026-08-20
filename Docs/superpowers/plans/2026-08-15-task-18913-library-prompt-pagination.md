# TASK-18913 Library Prompt Pagination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every local Prompt reachable through truthful 20-item Library pages while preserving the existing Prompt controller, debounced scopes, immutable cross-page selection basket, mutations, and history ownership.

**Architecture:** Extend the existing `LibraryPromptBrowseController`; do not add a generic pager controller or widget. The controller retains separate requested state and last successfully applied Prompt result, and the existing pure `build_library_pager_display` derives copy/control state for the Prompt canvas. The local Prompt DB/service remains the sole page-data authority and its coherently clamped response applies once without page walking or a second request.

**Tech Stack:** Python 3.12, Textual 8.x, immutable dataclasses, SQLite, pytest/pytest-asyncio, Ruff, existing TCSS bundle tooling.

**ADR required:** no new ADR
**ADR path:** `backlog/decisions/067-library-top-level-pagination-contracts.md`
**Reason:** This task directly implements ADR-067's already-approved Prompt contract without changing storage, runtime, or ownership boundaries.

---

## Scope and ownership

**Production owners:**

- `tldw_chatbook/Prompt_Management/prompt_normalizers.py` — preserve both resolved `current_page` and compatibility alias `page` instead of masking divergence.
- `tldw_chatbook/Library/library_prompts_state.py` — Prompt scope/result validation, stable identities, selection basket, and source-owned pager derivation inputs.
- `tldw_chatbook/UI/Library_Modules/library_prompt_browse_controller.py` — existing request token/fingerprint owner; add last-good/applied and recovery presentation state here.
- `tldw_chatbook/UI/Screens/library_screen.py` — existing Prompt event, focus, restore, navigation, mutation, and canvas-sync owner.
- `tldw_chatbook/Widgets/Library/library_prompts_canvas.py` — Prompt-specific rows, pager rendering, disabled reasons, Retry, and selection summary.
- `tldw_chatbook/css/components/_agentic_terminal.tcss` and generated bundle only if the mounted geometry test proves a CSS change is required.

**Test owners:**

- `Tests/Prompts_DB/test_prompts_db_exact_browse.py`
- `Tests/Prompt_Management/test_prompt_scope_service.py`
- `Tests/Library/test_library_prompts_state.py`
- `Tests/UI/test_library_prompt_browse_controller.py`
- `Tests/UI/test_library_prompts_canvas.py`
- `Tests/UI/test_library_entry_compose_once.py`
- Existing Prompt history/controller suites as regression gates only.

**Unchanged contract dependencies:**

- `tldw_chatbook/DB/Prompts_DB.py` — coherent exact browse and generic API default remain unchanged; tests verify explicit 20-row Library requests.
- `tldw_chatbook/Prompt_Management/prompt_scope_service.py` — local routing and generic API default remain unchanged; tests verify explicit 20-row Library requests.

**Out of scope:** a Prompt type filter, a generic Library page controller/widget, Prompt history pagination changes, server-backed Prompt browsing, Media/Skill/Collection work, or flattening the version-captured basket into current-page selection.

---

### Task 1: Pin the 20-row Library Prompt request contract

**Files:**

- Modify: `tldw_chatbook/Library/library_prompts_state.py:52-120`
- Test: `Tests/Prompts_DB/test_prompts_db_exact_browse.py`
- Test: `Tests/Prompt_Management/test_prompt_scope_service.py`
- Test: `Tests/Library/test_library_prompts_state.py`
- Test: `Tests/UI/test_library_prompt_browse_controller.py`

- [ ] **Step 1: Write failing Library-coordinate tests**

Add tests proving the Library Prompt default scope uses 20, the existing controller sends that explicit value through the scope service, more than 40 rows are reachable as three explicit 20-row requests, full-source query/collection/sort happens before paging, and the existing stable tie-break remains intact. Keep the generic DB and service omitted-argument defaults unchanged; they are not the Library configuration boundary.

```python
first = await service.browse_prompts(mode="local", page=1, page_size=20)
second = await service.browse_prompts(mode="local", page=2, page_size=20)
third = await service.browse_prompts(mode="local", page=3, page_size=20)
assert [len(page["items"]) for page in (first, second, third)] == [20, 20, 5]
assert [page["per_page"] for page in (first, second, third)] == [20, 20, 20]
```

- [ ] **Step 2: Run RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Prompts_DB/test_prompts_db_exact_browse.py Tests/Prompt_Management/test_prompt_scope_service.py Tests/Library/test_library_prompts_state.py Tests/UI/test_library_prompt_browse_controller.py -k 'browse_prompt or controller'
```

Expected: only the new Library default/propagation assertions fail with the current `PromptBrowseScope` default of `50`; explicit DB/service coherent transaction, clamp, filter, and stable-order tests remain green.

- [ ] **Step 3: Make the minimum Library-boundary change**

Change only `DEFAULT_PROMPT_BROWSE_PAGE_SIZE` from `50` to `20`. The controller already passes `PromptBrowseScope.page_size` explicitly through the service and DB. Keep the generic DB/service defaults, explicit maximum of 100, SQL, transaction, sort, and clamp unchanged.

- [ ] **Step 4: Run GREEN and the Library-default inverse mutation**

Rerun the focused command. Temporarily restore the Library scope constant to `50`; require the new default/propagation test to fail, then restore `20` and rerun green.

- [ ] **Step 5: Commit the Library-default slice**

```bash
git add tldw_chatbook/Library/library_prompts_state.py Tests/Prompts_DB/test_prompts_db_exact_browse.py Tests/Prompt_Management/test_prompt_scope_service.py Tests/Library/test_library_prompts_state.py Tests/UI/test_library_prompt_browse_controller.py
git commit -m "fix(library): page prompts twenty at a time"
```

---

### Task 2: Fail closed on Prompt page coordinates and stable identities

**Files:**

- Modify: `tldw_chatbook/Prompt_Management/prompt_normalizers.py:446-527`
- Modify: `tldw_chatbook/Library/library_prompts_state.py:52-390,2155-2206`
- Test: `Tests/Prompt_Management/test_prompt_scope_service.py`
- Test: `Tests/Library/test_library_prompts_state.py`

- [ ] **Step 1: Write failing envelope tests**

Add exact tests for:

- `DEFAULT_PROMPT_BROWSE_PAGE_SIZE == 20`;
- `per_page == the request scope's page size` (the pure builder remains valid for explicit non-Library sizes used by focused invariant tests; the Library scope/controller supplies 20);
- `current_page` equals compatibility alias `page`;
- resolved page equals the deterministic clamp of the requested page and exact total;
- exact page cardinality, including undersized non-final and oversized final pages;
- every item is a mapping with a non-blank exact-string normalized `id`, positive exact-int `local_id`, and page-unique identities;
- malformed/duplicate identities fail before row projection;
- an out-of-range request accepts one coherent response whose resolved page is the final page.

The direct `normalize_prompt_list` tests in the existing `test_prompt_scope_service.py` must feed a mapping with divergent `current_page=2` and `page=1` and prove the divergence survives normalization so the state boundary can reject it. Add product-path normalizer→`build_prompt_browse_result` cases for each present malformed `current_page`, `page`, and `per_page` value (at minimum `0` and `bool`) so a supplied invalid coordinate is never replaced by a fallback.

- [ ] **Step 2: Run RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Prompt_Management/test_prompt_scope_service.py Tests/Library/test_library_prompts_state.py -k 'browse_prompt or normalize_prompt_list or selection_browse_projection'
```

Expected: alias-divergence and stable-identity tests fail; current projection silently drops/deduplicates malformed rows.

- [ ] **Step 3: Preserve aliases and validate at the Prompt boundary**

Keep the normalizer mechanical: read page metadata independently, default only when a key is absent (never on falsiness), and reject booleans before integer coercion. A supplied `0` must remain `0` for the state boundary's minimum check; supplied empty/`None` values must fail closed rather than inherit a fallback. Apply the same absent-only rule to `per_page` (and the exact total/page-count metadata) so normalization cannot repair a malformed envelope.

```python
def page_int(key: str, default: int) -> int:
    value = data[key] if key in data else default
    if isinstance(value, bool):
        raise TypeError(f"{key} must be an integer")
    return int(value)

current_page = page_int("current_page", page_int("page", page))
page_alias = page_int("page", current_page)
per_page = page_int("per_page", per_page)
```

Return both fields. In `build_prompt_browse_result`, require exact integers, alias equality, `per_page == scope.page_size`, deterministic clamp, and total/cardinality. Put mapping shape plus non-blank exact-string normalized `id`, positive exact-int `local_id`, and page-unique identity checks in `PromptBrowseResult.__post_init__`, so direct construction cannot bypass them. Make `build_prompt_browse_list_state` consume those validated records rather than silently drop or deduplicate them.

- [ ] **Step 4: Run GREEN and inverse mutations**

Rerun the focused tests. Separately mutate alias equality and duplicate-ID validation away; each mutation must turn its exact test RED, then be restored.

- [ ] **Step 5: Commit the validation slice**

```bash
git add tldw_chatbook/Prompt_Management/prompt_normalizers.py tldw_chatbook/Library/library_prompts_state.py Tests/Prompt_Management/test_prompt_scope_service.py Tests/Library/test_library_prompts_state.py
git commit -m "fix(library): validate prompt page envelopes"
```

---

### Task 3: Extend the existing controller with requested/applied recovery state

**Files:**

- Modify: `tldw_chatbook/UI/Library_Modules/library_prompt_browse_controller.py`
- Modify: `tldw_chatbook/Library/library_prompts_state.py`
- Test: `Tests/UI/test_library_prompt_browse_controller.py`
- Test: `Tests/Library/test_library_prompts_state.py`

- [ ] **Step 1: Write failing controller lifecycle tests**

Cover these source-owned invariants:

- `scope` remains the latest requested scope;
- `applied_result` is absent until the first authoritative success;
- loading a new page/scope retains the previous applied rows/metadata;
- page failure retains applied page and reports `Couldn't load page N.`;
- query/sort/collection failure retains applied page and reports `Filter wasn’t applied; showing previous results.` (use source-appropriate copy for non-filter scopes);
- Retry uses the requested scope and a new token;
- a successful coherently clamped response applies directly and performs one service call;
- after a page-99 request coherently applies page 3, Previous requests page 2 (not page 98), and mutation refresh starts from applied page 3;
- after a newer query/sort/collection request fails, Previous/Next derive query, collection, sort, and page from the last applied scope, modify only its page, and never combine the failed requested scope with applied page coordinates;
- stale tokens/navigation/unmount cannot change requested/applied state;
- only success replaces `applied_result` and clears recovery copy.

- [ ] **Step 2: Run RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_library_prompt_browse_controller.py Tests/Library/test_library_prompts_state.py -k 'browse_prompt or controller'
```

- [ ] **Step 3: Add the smallest state to the existing controller**

Do not create a second controller. Add only Prompt-owned fields/properties such as:

```python
self.applied_result: PromptBrowseResult | None = None
self.retained_items: tuple[Mapping[str, Any], ...] = ()
self.freshness: PageFreshness = "uninitialized"
self.stale_copy = ""

@property
def visible_result(self) -> PromptBrowseResult:
    return self.applied_result or self.result
```

`begin()` changes requested/loading state but not `applied_result` or `retained_items`. `apply()` retains them on error and replaces both only on an accepted success. `retained_items` is normally identical to `applied_result.items`, but is independently reconciled after a committed mutation so stale rows remain representable without constructing an invalid exact envelope. Extend the existing Prompt list projection to accept this already-validated retained tuple separately from the exact result metadata. Derive `LibraryPagerDisplay` by calling the existing pure helper with requested page, applied page, retained row count, authoritative total only while fresh, freshness, loading, and error/stale copy. Previous/Next and post-mutation refresh start from the entire `applied_result.scope` (query, collection, sort, page, and size), then change only the page; the latest requested scope remains authoritative only for input state and Retry. Do not copy the shared pager calculations.

- [ ] **Step 4: Run GREEN and the single-call clamp mutation**

Rerun the controller/state tests. Mutate the clamped-success path to dispatch a second read; the exact call-count test must fail, then restore.

- [ ] **Step 5: Commit the controller slice**

```bash
git add tldw_chatbook/UI/Library_Modules/library_prompt_browse_controller.py tldw_chatbook/Library/library_prompts_state.py Tests/UI/test_library_prompt_browse_controller.py Tests/Library/test_library_prompts_state.py
git commit -m "fix(library): retain applied prompt pages"
```

---

### Task 4: Wire Prompt screen lifecycle, restore, races, and immutable selection

**Files:**

- Modify: `tldw_chatbook/UI/Screens/library_screen.py:3097-3184,5522-5926,10852-11290,15143-15315,16666-16796`
- Test: `Tests/UI/test_library_prompts_canvas.py`
- Test: `Tests/UI/test_library_entry_compose_once.py`

- [ ] **Step 1: Write mounted RED tests using real controller handlers**

Add/extend tests that prove:

- save/restore persists only the successfully applied Prompt scope, never a loading, failed, or unsubmitted filter draft;
- restored query remains an exact string; restored page must be an exact non-bool positive integer whose computed offset satisfies `(page - 1) * 20 <= 2**63 - 1`; persisted page size is normalized to the Library-owned value 20 rather than trusted;
- fresh re-entry refetches the restored applied scope and does not restore rows/errors/loading;
- search debounce and Enter flush retain filter focus/caret and reset requested page to 1;
- sort and collection changes reset requested page to 1;
- requested filter text stays in the input after failure while rows/title/range describe applied scope;
- a late old page/scope result cannot overwrite a newer result;
- an old screen's gated result after unmount cannot apply to it or a fresh screen;
- a late broad snapshot cannot overwrite the dedicated Prompt page;
- paging and search/sort/collection changes leave the immutable basket and captured versions unchanged;
- the basket crosses a real 20-row boundary and its summary remains `N selected · M on this page`.

Use event gates and bounded worker-group/state predicates; no fixed timing sleeps.

- [ ] **Step 2: Run RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_library_prompts_canvas.py Tests/UI/test_library_entry_compose_once.py -k 'prompt and (browse or page or selection or restore or retry or stale or unmount or snapshot)'
```

- [ ] **Step 3: Project visible/applied state through existing screen seams**

Update `_build_library_prompts_state` to project `controller.retained_items` with the controller's applied scope and pager state, canvas kwargs to carry requested input plus applied/pager state, save state to use only `applied_result.scope` (or default when none), and existing invalidation/unmount paths to revoke request authority. Validate restored page and its computed 20-row offset before constructing/dispatching a scope, preserve only exact-string query data, and force the Library page size to 20 so SQLite-overflow values or stale persisted sizes cannot reach the service. Keep the current debounce, focus, route-generation, controller, and worker groups.

- [ ] **Step 4: Run GREEN and inverse mutations**

Remove the stale token guard and separately persist requested scope instead of applied scope; require the race and restore tests to fail, restore each, then rerun green.

- [ ] **Step 5: Commit the screen-lifecycle slice**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_prompts_canvas.py Tests/UI/test_library_entry_compose_once.py
git commit -m "fix(library): harden prompt page lifecycle"
```

---

### Task 5: Render the shared Prompt pager and geometry contract

**Files:**

- Modify: `tldw_chatbook/Widgets/Library/library_prompts_canvas.py:284-677`
- Modify if proven necessary: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate if CSS changes: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_library_prompts_canvas.py`

- [ ] **Step 1: Write mounted RED tests at both supported sizes**

At 100×30 and 170×48, mount the production `TldwCli` → `LibraryScreen` → `#library-canvas` hierarchy with the exact `TldwCli.CSS_PATH` list (not a canvas-only or bundle-only host) and prove page 1, page 2, final page, loading, first failure, retained-row page failure, and scope failure. Require:

- 20 independently scrollable rows with row 20 reachable;
- pager/status/actions fixed inside the Prompt canvas;
- exact `1-20 of 45 · Page 1 of 3`, `21-40`, and `41-45` presentation;
- Previous/Next disabled state, visible reason, tooltip, and focus fallback;
- loading disables page actions while retained rows remain visible;
- recoverable failure mounts exactly one Prompt Retry and retains applied title/range/rows when available;
- first failure shows unavailable total rather than `Prompts (0)`;
- filter focus survives recompose;
- selection summary reports total/on-page counts across pages.
- compositor-visible title/range/status/disabled reason and containment inside `#library-canvas` at both sizes;
- the row viewport precedes a fixed pager in production layout, row 20 becomes compositor-visible by scrolling without moving the pager, and every successful page/scope change resets the new page to its top.

- [ ] **Step 2: Run RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_library_prompts_canvas.py -k 'prompts and (paging or browse_states or selection_persists or retry or geometry)'
```

- [ ] **Step 3: Render `LibraryPagerDisplay` without a generic widget**

The Prompt canvas keeps its IDs and event handlers. Replace hand calculations with the supplied pure display state, always render exact pager/status for the active browse state, keep rows mounted from the applied result during loading/failure, and use existing disabled-label/reason conventions. Make only the CSS adjustment demonstrated by the geometry RED; prefer existing `1fr`, `min-height: 0`, and overflow patterns.

- [ ] **Step 4: Run GREEN, CSS parity, and UI mutations**

Mutate the title back to `len(rows)` and remove the scroll/fixed-pager rule; the authoritative-total and 100×30 geometry tests must fail. Restore, rebuild CSS if touched, and require bundle parity.

- [ ] **Step 5: Commit the canvas slice**

```bash
git add tldw_chatbook/Widgets/Library/library_prompts_canvas.py Tests/UI/test_library_prompts_canvas.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "feat(library): render resilient prompt pager"
```

---

### Task 6: Preserve Prompt mutations and truthful stale recovery

**Files:**

- Modify: `tldw_chatbook/UI/Screens/library_screen.py:22475-22587,22746-22872`
- Modify: `tldw_chatbook/UI/Library_Modules/library_prompt_browse_controller.py`
- Test: `Tests/UI/test_library_prompts_canvas.py`
- Regression test: `Tests/UI/test_library_prompt_history_controller.py`

- [ ] **Step 1: Write mutation RED tests**

Prove delete/undo refresh the retained applied scope rather than forcing page 1, accept the service's one coherent clamp after the final-page record is deleted, and never dispatch a redundant second read. When the durable mutation succeeds, revoke any in-flight read, reconcile the known record in `retained_items` without mutating/forging the last exact `applied_result`, mark retained rows stale, suppress exact total/range, disable row/bulk/pager actions, and request the applied scope. A successful refresh replaces both exact applied envelope and retained rows and returns fresh; a failed refresh preserves the reconciled stale rows and leaves Retry plus page-1 scope recovery available. Selection/history version checks remain unchanged.

- [ ] **Step 2: Run RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_library_prompts_canvas.py Tests/UI/test_library_prompt_history_controller.py -k 'prompt and (delete or undo or mutation or history or stale or clamp)'
```

- [ ] **Step 3: Reuse controller recovery state**

Invalidate before durable writes, reconcile only the known mutation on success into the separate retained tuple, request the applied scope, and expose a narrow controller method for committed-but-stale presentation. Do not construct a cardinality-invalid `PromptBrowseResult`; do not modify Prompt history pagination, version-conflict checks, or selection basket semantics.

- [ ] **Step 4: Run GREEN and mutation inverses**

Force page 1 after delete and allow stale row action; each exact regression must fail, then restore. Run the full Prompt history controller/state gates.

- [ ] **Step 5: Commit the mutation slice**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Library_Modules/library_prompt_browse_controller.py Tests/UI/test_library_prompts_canvas.py Tests/UI/test_library_prompt_history_controller.py
git commit -m "fix(library): retain prompt mutation scope"
```

---

### Task 7: Verify live behavior, docs, review, and closeout

**Files:**

- Modify: `Docs/User_Guide/library/media-and-conversations.md`
- Modify: `backlog/tasks/task-18913 - Align-Library-Prompt-browsing-to-20-item-pages.md`

- [ ] **Step 1: Run task-local and owner gates**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Prompts_DB/test_prompts_db_exact_browse.py Tests/Prompt_Management/test_prompt_scope_service.py Tests/Library/test_library_prompts_state.py Tests/UI/test_library_prompt_browse_controller.py Tests/Library/test_library_pager_state.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_library_prompts_canvas.py Tests/UI/test_library_entry_compose_once.py Tests/UI/test_library_prompt_collections.py Tests/UI/test_library_prompt_history_controller.py -k 'prompt'
```

Run the existing Prompt mutation/history owner suites without excluding related failures. Per the user's instruction, unrelated failures outside touched Prompt/Library-pagination behavior are not closeout blockers, but may not be called related or baseline without exact reproduction evidence.

- [ ] **Step 2: Run isolated live verification**

In fresh synthetic-only isolated profiles at true 100×30 and 170×48, seed 45 Prompts/Recipes and prove:

- exact three-page ranges and reachable row 20;
- page 2/final focus and disabled reasons;
- full-source oldest-page search, sort, and collection page-1 reset;
- cross-page captured-version basket with total/on-page summary;
- controlled page failure retaining applied rows and Retry recovery;
- out-of-range clamp applies once;
- detail/back restores applied page;
- zero real-profile DB/config handles, no private-data logging, byte-identical real-profile fingerprints, and clean exit.

Implement the controlled failure in the isolated scratch harness by temporarily wrapping the mounted app's real `PromptScopeService.browse_prompts` bound async method in `try/finally`: fail exactly once for the intended explicit `page=3`/empty-query request, record the requested coordinates, delegate every other call unchanged, and restore the original method before exit. Scope failures and committed-stale mutation failure remain authoritative mounted-test evidence unless a similarly exact, self-restoring live seam is added; do not claim unexercised live behavior.

- [ ] **Step 3: Run the repository-wide diagnostic gate**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q --tb=short
```

TASK-18913 is blocked by any related Prompt/Library-pagination failure. Per the user's instruction, unrelated failures outside touched functionality are diagnostic only; record exact node names and do not call them baseline without exact-base reproduction.

- [ ] **Step 4: Run static/generated checks**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/DB/Prompts_DB.py tldw_chatbook/Prompt_Management/prompt_scope_service.py tldw_chatbook/Prompt_Management/prompt_normalizers.py tldw_chatbook/Library/library_prompts_state.py tldw_chatbook/UI/Library_Modules/library_prompt_browse_controller.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/Widgets/Library/library_prompts_canvas.py Tests/Prompts_DB/test_prompts_db_exact_browse.py Tests/Prompt_Management/test_prompt_scope_service.py Tests/Library/test_library_prompts_state.py Tests/UI/test_library_prompt_browse_controller.py Tests/UI/test_library_prompts_canvas.py Tests/UI/test_library_entry_compose_once.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
git diff --check
```

- [ ] **Step 5: Update docs/task evidence**

Document the 20-item Prompt pager, exact ranges, full-source scopes, versioned cross-page basket, disabled reasons, and Retry/stale behavior. Add concise implementation notes with ADR-067, files, mutations, automated/live/privacy evidence, and any deviations.

- [ ] **Step 6: Request independent spec and quality reviews**

Resolve every Critical/Important finding and rerun affected gates. Do not mark Done before both reviewers pass.

- [ ] **Step 7: Close TASK-18913 and commit docs**

```bash
backlog task edit 18913 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --check-ac 6 --check-ac 7
backlog task edit 18913 -s Done
git add Docs/User_Guide/library/media-and-conversations.md 'backlog/tasks/task-18913 - Align-Library-Prompt-browsing-to-20-item-pages.md'
git commit -m "docs(library): close prompt pagination task"
```
