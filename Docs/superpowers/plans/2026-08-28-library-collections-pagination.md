# Library Collections Pagination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every top-level Library Collection reachable through exact 20-item pages while preserving deterministic selection and truthful mutation recovery under concurrent source changes.

**Architecture:** Keep Collections as a source-owned bounded reader. Pure request/result validation lives beside the existing Collections panel state, a dedicated non-visual controller owns requested/applied scope and generations, the local service owns coherent SQLite page and stable-ID locator reads, and the retained Textual panel only projects state and emits actions. Existing broad Library snapshots remain consumers for rail/landing counts and never overwrite the dedicated Collections page.

**Tech Stack:** Python 3.11+, Textual 8.x, SQLite/FTS-adjacent local storage, pytest/pytest-asyncio, Ruff.

**Spec:** `Docs/superpowers/specs/2026-08-14-library-top-level-source-pagination-design.md`

## Global Constraints

- Every top-level Collections page contains at most 20 rows.
- Stable order is `created_at ASC`, `name COLLATE NOCASE ASC`, then `collection_id ASC`.
- Count, rows, and locator metadata come from one coherent `read_transaction`.
- Previous/Next stay inside the list pane and outside its independently scrolling row viewport.
- Initial failure never fabricates a zero total; stale state never exposes an exact total or range.
- Create, rename, and restore locate one stable ID directly; page walking and synthetic row injection are forbidden.
- Delete performs at most one automatic clamp; a second shrink retains the last good page as stale.
- A committed mutation remains reported as committed when its follow-up read fails.
- Row, mutation, and pager actions are disabled while Collections state is stale; Retry remains available.
- Restored page state accepts only exact non-boolean positive integers whose offset multiplication fits SQLite coordinates.
- The existing split Collection list/detail workbench, forms, sync projection, and delete receipt semantics remain intact.
- No Collections text filter, direct page entry, infinite scrolling, or generic polymorphic Library controller is added.
- ADR required: yes.
- ADR path: `backlog/decisions/067-library-top-level-pagination-contracts.md`.
- Reason: ADR-067 already defines the durable ordering, locator, mutation, freshness, and recovery contracts used here.

---

### Task 1: Repair and lock the Collections test baseline

**Files:**
- Modify: `Tests/UI/test_product_maturity_phase39_library_collections.py`

**Interfaces:**
- Consumes: the current `#library-canvas-route-content` route wrapper and the test app's startup service seams.
- Produces: a deterministic mounted Collections baseline that performs no network probes and asserts the current route ownership contract.

- [ ] **Step 1: Tighten the stale route-parent assertion**

  Change the mounted-panel assertion to require `#library-collections-panel` beneath `#library-canvas-route-content`, while separately asserting that route content is beneath `#library-canvas`. This catches accidental mounting outside the route owner without encoding the obsolete direct-parent topology.

- [ ] **Step 2: Stub startup probes at the test-app boundary**

  Extend `_build_test_app()` with the same local-server/model-discovery stubs used by neighboring mounted Library tests so no socket connection to `localhost:8000` occurs. Assert Collections behavior, not the stubs themselves.

- [ ] **Step 3: Run the mounted baseline and verify green**

  Run: `../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase39_library_collections.py -q --tb=short`

  Expected: all existing mounted Collections tests pass with no blocked network attempts.

- [ ] **Step 4: Commit the baseline repair**

  ```bash
  git add Tests/UI/test_product_maturity_phase39_library_collections.py
  git commit -m "test: repair collections mounted baseline"
  ```

---

### Task 2: Add strict Collection page and locator contracts

**Files:**
- Modify: `tldw_chatbook/Library/library_collections_state.py`
- Modify: `Tests/Library/test_library_collections_state.py`

**Interfaces:**
- Produces: `CollectionBrowseScope`, `CollectionBrowseResult`, `CollectionLocatorResult`, `build_collection_browse_result()`, `build_collection_locator_result()`, and `validate_collection_browse_items()`.
- Consumes later: the Collections browse controller and screen mutation placement.

- [ ] **Step 1: Write failing scope-normalization tests**

  Add literal cases proving page `1` and page size `20` are accepted, while booleans, strings, zero, negatives, oversized page sizes, and a page whose `(page - 1) * 20` exceeds SQLite's signed 64-bit range are rejected. The production mutation that must fail these tests is accepting coercible or overflowing coordinates.

- [ ] **Step 2: Run the scope tests and verify RED**

  Run: `../../.venv/bin/python -m pytest Tests/Library/test_library_collections_state.py -q -k "browse_scope"`

  Expected: collection browse symbols are absent.

- [ ] **Step 3: Implement the immutable scope**

  Add a frozen scope with fixed local authority and these public coordinates:

  ```python
  @dataclass(frozen=True)
  class CollectionBrowseScope:
      page: int = 1
      page_size: int = 20

      @property
      def offset(self) -> int: ...
  ```

  Validate with exact `type(value) is int` checks and checked signed-64-bit offset arithmetic.

- [ ] **Step 4: Write failing page-envelope tests**

  Cover first/middle/final/empty pages, exact totals, wrong `limit`/`offset` echoes, duplicate or blank `collection_id`, malformed row fields, oversized pages, undersized non-final pages, and out-of-range empty pages. Use hand-authored rows and literal expected IDs.

- [ ] **Step 5: Run the page tests and verify RED**

  Run: `../../.venv/bin/python -m pytest Tests/Library/test_library_collections_state.py -q -k "browse_result or browse_items"`

  Expected: missing validators/result builders fail.

- [ ] **Step 6: Implement strict page validation**

  Freeze JSON-like row mappings, require unique nonblank stable IDs, validate string metadata and nonnegative `item_count`, require echoed `limit=scope.page_size` and `offset=scope.offset`, and derive exact cardinality from `(total, offset, limit)`. The immutable result represents only a validated source response; loading, error, and generation ownership remain exclusively in the non-visual controller.

- [ ] **Step 7: Write failing locator-envelope tests**

  Cover a target on pages 1, 2, and 3 plus absent target, duplicate identities, wrong requested ID, wrong rank, wrong page-local index, unaligned offset, divergent page/offset, and inconsistent total/cardinality. The expected index is the literal `target_rank - offset`.

- [ ] **Step 8: Run locator tests and verify RED**

  Run: `../../.venv/bin/python -m pytest Tests/Library/test_library_collections_state.py -q -k "locator"`

  Expected: missing locator builder fails.

- [ ] **Step 9: Implement the locator contract**

  Require the locator envelope to contain `items`, `total`, `limit`, `offset`, `page`, `target_id`, `target_rank`, and `target_index`. Validate `offset == (target_rank // limit) * limit`, `page == offset // limit + 1`, `target_index == target_rank - offset`, and `items[target_index]["collection_id"] == target_id`.

- [ ] **Step 10: Run state tests and commit**

  Run: `../../.venv/bin/python -m pytest Tests/Library/test_library_collections_state.py -q`

  ```bash
  git add tldw_chatbook/Library/library_collections_state.py Tests/Library/test_library_collections_state.py
  git commit -m "feat: define collection page contracts"
  ```

---

### Task 3: Make SQLite paging and stable-ID location coherent

**Files:**
- Modify: `tldw_chatbook/Library/library_collections_service.py`
- Modify: `Tests/Library/test_library_collections_service.py`

**Interfaces:**
- Produces: `list_library_collections(*, limit: int = 20, offset: int = 0) -> dict` with deterministic tie-breaking.
- Produces: `locate_library_collection_page(collection_id: str, *, limit: int = 20) -> dict | None` on both the protocol and local implementation.
- Consumes: the existing `_library_collection_item()` safe summary projection and `LibraryCollectionsDB.read_transaction()`.

- [ ] **Step 1: Write failing deterministic-order and deep-page tests**

  Seed at least 45 real SQLite Collections, including multiple rows with the same `created_at` and case-insensitively equal names. Assert literal stable-ID order across pages 1, 2, and 3 and an exact total of 45. Also prove `list_collections()` and `list_library_collections()` share the stable tie-breaker.

- [ ] **Step 2: Run the paging tests and verify RED**

  Run: `../../.venv/bin/python -m pytest Tests/Library/test_library_collections_service.py -q -k "stable_order or forty_five or deep_page"`

  Expected: the equal-key order is not contractually stable.

- [ ] **Step 3: Add the stable ID tie-breaker**

  Append `collection.collection_id ASC` to every top-level Collection list order used by the UI/service. Keep count and rows inside the existing `read_transaction`.

- [ ] **Step 4: Write failing locator tests**

  Locate known IDs on the first, middle, and final pages and assert exact `target_rank`, aligned `offset`, one-based `page`, `target_index`, `total`, and returned IDs. Assert an active missing ID and a soft-deleted ID return `None`. Add an equal-key fixture proving rank follows the same stable ordering as list paging.

- [ ] **Step 5: Run locator tests and verify RED**

  Run: `../../.venv/bin/python -m pytest Tests/Library/test_library_collections_service.py -q -k "locate_library_collection_page"`

  Expected: the locator method is absent.

- [ ] **Step 6: Implement one bounded rank-derived locator read**

  Use a window/CTE query under one `read_transaction`: rank active Collections with `ROW_NUMBER() OVER (ORDER BY created_at, name COLLATE NOCASE, collection_id) - 1`, read the target rank and exact total, derive the aligned offset in Python, and fetch only that `LIMIT ? OFFSET ?` page using the identical order. Return `None` before the page query when the target is absent.

- [ ] **Step 7: Add boundary validation tests and implementation**

  Assert booleans, zero/negative limits, oversized limits, blank IDs, and unsafe offset inputs fail at the service boundary rather than reaching SQLite. Implement explicit validators shared by list and locator without changing agent-search behavior.

- [ ] **Step 8: Run service tests and commit**

  Run: `../../.venv/bin/python -m pytest Tests/Library/test_library_collections_service.py -q`

  ```bash
  git add tldw_chatbook/Library/library_collections_service.py Tests/Library/test_library_collections_service.py
  git commit -m "feat: locate deterministic collection pages"
  ```

---

### Task 4: Add the source-owned Collections browse controller

**Files:**
- Create: `tldw_chatbook/UI/Library_Modules/library_collections_browse_controller.py`
- Create: `Tests/UI/test_library_collections_browse_controller.py`

**Interfaces:**
- Consumes: `CollectionBrowseScope`, strict page builders, `LibraryPagerDisplay`, `list_library_collections`, the screen's worker runner, and a late-bound active-route predicate.
- Produces: `LibraryCollectionsBrowseController.request()`, `.retry()`, `.invalidate()`, `.begin_mutation()`, `.request_locator()`, `.reconcile_committed_mutation()`, `.scope_for_page()`, `.pager`, `.applied_result`, `.retained_items`, and `.freshness`.

- [ ] **Step 1: Write failing controller page lifecycle tests**

  Prove exact `limit=20`/`offset` calls, retained last-good rows during a page request, page-only failure copy, initial failure with unavailable total, retry of the requested scope, focus identity forwarding, late-generation rejection, and inactive-route/unmount rejection.

- [ ] **Step 2: Run controller lifecycle tests and verify RED**

  Run: `../../.venv/bin/python -m pytest Tests/UI/test_library_collections_browse_controller.py -q -k "page or generation or inactive or retry"`

  Expected: controller module is absent.

- [ ] **Step 3: Implement minimal page ownership**

  Mirror the Media controller's requested/applied/freshness shape without facets or filters. All service and sync seams remain lazy accessors. Only matching active generations apply. Build pager copy exclusively through `build_library_pager_display()`.

- [ ] **Step 4: Write failing bounded-clamp tests**

  Request page 99 against a 45-row response and assert exactly one refetch at offset 40. Then return another out-of-range response and assert there is no third read, the last good rows remain visible, total/range disappear, actions become stale, and Retry is visible.

- [ ] **Step 5: Implement one-clamp recovery**

  Detect only a strictly valid empty out-of-range envelope as clamp evidence. Refetch the last page once within the same generation. A second shrink publishes `Source changed again; try again.` without applying contradictory metadata.

- [ ] **Step 6: Write failing mutation/locator tests**

  Prove `begin_mutation()` invalidates an older page read before a write, a successful locator applies and selects the target's page, malformed locator data fails closed, and a committed create/rename/restore/delete can reconcile known rows into stale inert state without forging totals.

- [ ] **Step 7: Implement locator and reconciliation ownership**

  A locator request calls `locate_library_collection_page(target_id, limit=20)`, validates it, and applies its page under the same generation fence. Reconciliation removes known deleted IDs and upserts only known changed summaries already compatible with the unfiltered Collection source; it truncates to the last applied page size and marks the page stale.

- [ ] **Step 8: Run controller tests and commit**

  Run: `../../.venv/bin/python -m pytest Tests/UI/test_library_collections_browse_controller.py -q`

  ```bash
  git add tldw_chatbook/UI/Library_Modules/library_collections_browse_controller.py Tests/UI/test_library_collections_browse_controller.py
  git commit -m "feat: own collection browse generations"
  ```

---

### Task 5: Project exact pager state through the retained Collections panel

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_collections_panel.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Tests/UI/test_product_maturity_phase39_library_collections.py`

**Interfaces:**
- Consumes: `LibraryCollectionsPanelState`, `LibraryPagerDisplay`, and page-action safety flags supplied by the screen.
- Produces: `#library-collections-rows-scroll`, `#library-collections-range`, `#library-collections-page`, `#library-collections-previous`, `#library-collections-next`, and `#library-collections-retry` inside `#library-collections-list`.

- [ ] **Step 1: Write failing mounted pager topology tests**

  Mount a 45-row fake source and assert 20 row controls, exact `1-20 of 45` and `Page 1 of 3` copy, both pager buttons inside the list pane, pager outside the rows scroll viewport, Previous disabled on page 1, and title `Collections (45)`.

- [ ] **Step 2: Run the topology test and verify RED**

  Run: `../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase39_library_collections.py -q -k "pager_topology"`

  Expected: pager selectors are absent and the panel renders all fake rows.

- [ ] **Step 3: Add pager projection to the panel**

  Extend `LibraryCollectionsPanel` inputs with immutable pager state and a `page_actions_disabled` flag. Wrap only row buttons/empty copy in a bounded `VerticalScroll`; compose range, page, status, Previous/Next, and conditional Retry below it. Continue capturing/restoring focused form Inputs across retained-panel syncs.

- [ ] **Step 4: Add stale-action and failure projection tests**

  Assert stale rows remain readable but selection, create, rename, delete, restore, Previous, and Next are disabled; exact total/range is hidden; Retry stays enabled. Assert first-load failure says total unavailable rather than `Collections (0)`.

- [ ] **Step 5: Implement scoped CSS and regenerate the bundle**

  Give the list pane a bounded rows region plus an auto-height pager footer. At 100x30 the range/page lines stack without clipping; at 170x48 the split workbench remains balanced. Regenerate the committed bundle with `../../.venv/bin/python tldw_chatbook/css/build_css.py`, then verify it with `../../.venv/bin/python tldw_chatbook/css/check_bundle_sync.py`.

- [ ] **Step 6: Run mounted panel tests and commit**

  Run: `../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase39_library_collections.py -q -k "collections and (pager or stale or failure or geometry)"`

  ```bash
  git add tldw_chatbook/Widgets/Library/library_collections_panel.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_product_maturity_phase39_library_collections.py
  git commit -m "feat: render collection pager in list pane"
  ```

---

### Task 6: Wire page restoration, focus, and deterministic mutations

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_product_maturity_phase39_library_collections.py`
- Modify: `Tests/UI/test_library_entry_compose_once.py` if its Collections fake must implement the bounded list seam.

**Interfaces:**
- Consumes: `LibraryCollectionsBrowseController`, `CollectionBrowseScope`, service mutation return records, stable-ID locator, retained Collections panel sync, and the Library screen snapshot persistence seam.
- Produces: Collections Previous/Next/Retry handlers, page-only restoration, active-route fencing, and deterministic post-mutation selection.

- [ ] **Step 1: Write failing page navigation/focus/restoration tests**

  Prove Next loads offset 20, Previous returns to offset 0, successful navigation resets only the rows viewport, focus returns to the invoking control or the opposite enabled boundary control, navigating away/back restores the last successfully applied page, invalid restored values normalize to page 1, and a requested page that never applied is not persisted.

- [ ] **Step 2: Run navigation tests and verify RED**

  Run: `../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase39_library_collections.py -q -k "page_navigation or page_focus or page_restoration"`

  Expected: page controls/restore scope are not wired.

- [ ] **Step 3: Replace snapshot-fed Collections state with controller projection**

  Initialize the controller in `LibraryScreen.__init__`, derive panel records only from `controller.retained_items`, decorate only those bounded rows with sync summaries, and leave rail/landing snapshots untouched. On entry request the restored applied page; on route exit invalidate the controller. Persist only an exact fresh applied page number.

- [ ] **Step 4: Add Previous/Next/Retry handlers and focus fallback**

  Each handler checks freshness and mutation ownership, derives a scope from the last applied page, and supplies its own control ID as focus identity. After sync, choose the invoked control when enabled, then the opposite pager control, then the first row/form control.

- [ ] **Step 5: Write failing create/rename/restore locator tests**

  Use 45-row fake data to place created and restored IDs off page 1 and to move a renamed equal-timestamp row between pages. Assert one mutation call plus one locator call, no list page walking, owning-page application, and stable selection. Assert Restore keeps its delete receipt until the locator successfully applies.

- [ ] **Step 6: Implement locator-driven successful mutations**

  Before each write call `controller.begin_mutation()`. After a successful create, rename, or restore, request the returned stable ID's locator page and select only after strict locator application. Clear form data after create/rename commit. Clear the restore receipt only after the restored page is authoritative and selected.

- [ ] **Step 7: Write failing delete/clamp and committed-stale tests**

  Cover deleting the only row on page 3, deletion followed by one clamp, a second concurrent shrink, list failure after every mutation type, malformed locator response after a committed mutation, and route exit while reads are in flight. Assert mutation success copy/receipt remains truthful, known rows are locally reconciled, unsafe actions are disabled, Retry recovers, and stale callbacks never repaint after unmount.

- [ ] **Step 8: Implement truthful mutation recovery**

  Delete refreshes the current applied scope and lets the controller clamp once. When a post-commit page or locator read fails, reconcile the returned/removed record locally, mark state stale, preserve receipts required for recovery, and never route the outcome through mutation-failed copy. Only a subsequent authoritative request clears stale state.

- [ ] **Step 9: Run mounted mutation and lifecycle tests and commit**

  Run: `../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase39_library_collections.py Tests/UI/test_library_entry_compose_once.py -q -k "collection"`

  ```bash
  git add tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_product_maturity_phase39_library_collections.py Tests/UI/test_library_entry_compose_once.py
  git commit -m "feat: page collection mutations deterministically"
  ```

---

### Task 7: Verify geometry, live behavior, documentation, and task hygiene

**Files:**
- Modify: `Docs/User_Guide/library/collections.md`
- Modify: `backlog/tasks/task-18916 - Page-Library-Collections-with-deterministic-mutation-placement.md`
- Modify if an incident generalizes: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md`.

**Interfaces:**
- Consumes: the completed Collections page path.
- Produces: reproducible automated, geometry, mutation, and isolated-live evidence plus a Done Backlog record.

- [ ] **Step 1: Run focused automated verification**

  ```bash
  ../../.venv/bin/python -m pytest Tests/Library/test_library_collections_service.py Tests/Library/test_library_collections_state.py Tests/UI/test_library_collections_browse_controller.py Tests/UI/test_product_maturity_phase39_library_collections.py Tests/UI/test_library_entry_compose_once.py -q
  ../../.venv/bin/python -m ruff check tldw_chatbook/Library/library_collections_service.py tldw_chatbook/Library/library_collections_state.py tldw_chatbook/UI/Library_Modules/library_collections_browse_controller.py tldw_chatbook/Widgets/Library/library_collections_panel.py tldw_chatbook/UI/Screens/library_screen.py Tests/Library/test_library_collections_service.py Tests/Library/test_library_collections_state.py Tests/UI/test_library_collections_browse_controller.py Tests/UI/test_product_maturity_phase39_library_collections.py
  git diff --check
  ```

  Expected: every command exits zero. Do not run the full repository suite unless the user explicitly requests it.

- [ ] **Step 2: Run production-shaped geometry walkthroughs**

  Seed at least 45 local Collections. Exercise first, middle, and final pages at 100x30 and 170x48. Record exact visible ranges, pager containment, independent row scrolling, selected-row readability, form access, and focus fallback.

- [ ] **Step 3: Run isolated live mutation walkthroughs**

  In a temporary isolated Library database, create, rename, delete, and restore Collections whose owning pages differ. Verify no page walking, deterministic selection, one-clamp recovery, stale committed-success posture after an injected follow-up read failure, and successful Retry.

- [ ] **Step 4: Update user documentation**

  Document the 20-item Collections convention, exact range/page copy, Previous/Next behavior, restored applied page, and Retry/stale behavior. Do not document internal controller or SQL mechanics.

- [ ] **Step 5: Complete Backlog acceptance and implementation notes**

  Check all seven acceptance criteria. Add concise implementation notes naming the service/validator/controller/panel changes, ADR-067, automated test counts, geometry sizes, mutation cases, and isolated-live evidence. Add a lesson only if this implementation produces a genuinely generalizable incident.

- [ ] **Step 6: Run final self-review and commit**

  Inspect `git diff origin/dev...HEAD`, mentally apply the spec's mutation checklist, and confirm each realistic removed guard/wrong coordinate/stale-action mutation makes at least one test fail.

  ```bash
  git add Docs backlog/tasks backlog/docs
  git commit -m "docs: close collection pagination task"
  ```
