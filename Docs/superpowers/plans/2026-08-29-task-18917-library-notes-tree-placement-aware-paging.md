# TASK-18917 Library Notes Tree Placement-Aware Paging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every Database Note reachable through exact, bounded, branch-local folder-tree paging while preserving hierarchy, stable placement identity, focus, and mutation recovery.

**Architecture:** `LibraryScreen` continues to own Notes request and lifecycle state, while a new pure Notes paging module owns immutable branch/slice transitions and drift validation. `LocalNoteFolderRepository` supplies parent-scoped folder pages, visible-placement pages, locators, and coherent filter pages through off-loop `NotesScopeService` seams. The existing tree projection and Notes canvas render stable inline boundaries without a generic Library controller.

**Tech Stack:** Python 3.11+, Textual 8.x, SQLite/FTS5, pytest, pytest-asyncio, existing ChaChaNotes folder repository and adaptive Library reader shell.

**Design:** `Docs/superpowers/specs/2026-08-29-task-18917-library-notes-tree-placement-aware-paging-design.md`

**Governing ADR:** `backlog/decisions/067-library-top-level-pagination-contracts.md`

---

## ADR Check

**ADR required:** no

**ADR path:** `backlog/decisions/067-library-top-level-pagination-contracts.md`

**Reason:** ADR-067 already governs source-owned paging, exact query-derived totals, generation fencing, cross-visit scope-only persistence, stable locators, and the separate Notes hierarchy follow-up. This implementation changes neither storage nor ownership, synchronization, security, dependencies, or application-level boundaries.

## File and Responsibility Map

**Create:**

- `tldw_chatbook/Library/library_notes_tree_paging.py` — pure immutable branch/slice state, range application, drift detection, stale transitions, and pager identities; no workers, widgets, repository, or app state.
- `Tests/Library/test_library_notes_tree_paging.py` — reducer/state tests.
- `Tests/Live/test_library_notes_tree_paging_live.py` — isolated real-repository mounted walkthrough at 160×50, 120×35, 100×30, and 80×24.
- `Docs/superpowers/reviews/evidence/task-18917/live-walkthrough.md` — identifying live evidence.

**Modify:**

- `tldw_chatbook/Notes/note_folder_models.py` — typed folder page, placement page, placement record, path step, and locator envelopes.
- `tldw_chatbook/Notes/note_folder_repository.py` — exact parent pages, effective-placement suppression, locators, affected-parent lookup, and filter paging.
- `tldw_chatbook/Notes/notes_scope_service.py` — policy-checked off-loop seams.
- `tldw_chatbook/Library/library_notes_tree_state.py` — project branch state and inline pager rows.
- `tldw_chatbook/UI/Screens/library_screen.py` — own branches, generations, workers, receipts, filtering, navigation, and local mutation reconciliation.
- `tldw_chatbook/Widgets/Library/library_notes_canvas.py` — render and route exact branch controls.
- `tldw_chatbook/css/components/_agentic_terminal.tcss` and generated `tldw_chatbook/css/tldw_cli_modular.tcss` — narrow-width pager layout.
- Focused tests in `Tests/Notes/test_note_folder_models.py`, `Tests/Notes/test_note_folder_repository.py`, `Tests/Notes/test_notes_scope_service_folders.py`, `Tests/Library/test_library_notes_tree_state.py`, `Tests/UI/test_library_notes_folder_navigator.py`, `Tests/Widgets/Library/test_library_notes_canvas.py`, and `Tests/UI/test_library_adaptive_reader_closeout.py`.
- `Docs/User_Guide/library/notes.md` and the TASK-18917 Backlog file.

## Constraints

- Fixed `LIBRARY_NOTES_TREE_PAGE_SIZE = 20`; do not reuse the broad 100-note snapshot limit.
- Page visible placements, not distinct notes. Suppress shadowed managed placements before `COUNT` and `LIMIT/OFFSET`.
- No schema migration, generic controller, generic pager widget, or broad-snapshot totals.
- Repository count/page/locator queries are coherent transactions with bound parameters.
- Async apply requires slice generation, topology epoch, and screen lifecycle generation; navigation also requires navigation generation.
- True `LibraryScreen` unmount invalidates all Notes reads. Canvas recomposition does not.
- Run targeted tests only; no full repository suite unless separately requested.

---

### Task 1: Add Typed Page Contracts and the Pure Branch Reducer

**Files:**
- Modify: `tldw_chatbook/Notes/note_folder_models.py`
- Create: `tldw_chatbook/Library/library_notes_tree_paging.py`
- Modify: `Tests/Notes/test_note_folder_models.py`
- Create: `Tests/Library/test_library_notes_tree_paging.py`

- [ ] **Step 1: Write failing model-envelope tests**

Test exact total/start/previous/next fields, non-negative validation, duplicate membership identity, and a path that includes the target folder:

```python
page = NotePlacementPage(
    placements=(NotePlacementRecord(note={"id": "n1", "title": "One"}, folder_id="f1", membership=_membership("m1", "f1", "n1")),),
    total_placements=41,
    start_offset=20,
    previous_offset=0,
    next_offset=40,
    ancestor_folders=(),
)
assert page.placements[0].membership.membership_id == "m1"
```

- [ ] **Step 2: Run models RED**

Run: `python3 -m pytest Tests/Notes/test_note_folder_models.py -q`

Expected: FAIL because `NoteFolderChildPage`, `NotePlacementRecord`, `NotePlacementPage`, `NoteTreePathStep`, and `NoteTreeLocation` do not exist.

- [ ] **Step 3: Implement minimal frozen dataclasses**

```python
@dataclass(frozen=True)
class NoteFolderChildPage:
    folders: tuple[NoteFolder, ...]
    total_folders: int
    start_offset: int
    previous_offset: int | None
    next_offset: int | None

@dataclass(frozen=True)
class NotePlacementRecord:
    note: Mapping[str, Any]
    folder_id: str | None
    membership: NoteFolderMembership | None

@dataclass(frozen=True)
class NotePlacementPage:
    placements: tuple[NotePlacementRecord, ...]
    total_placements: int
    start_offset: int
    previous_offset: int | None
    next_offset: int | None
    ancestor_folders: tuple[NoteFolder, ...] = ()
```

Also add `NoteTreePathStep(folder_id, parent_id, containing_offset)`,
`NoteTreeLocation(placement_id, note_id, membership_id, path, placement_offset)`, and:

```python
@dataclass(frozen=True)
class NoteTreeMutationContext:
    folder_ids: tuple[str, ...]
    parent_ids: tuple[str | None, ...]
    ancestor_ids: tuple[str, ...]
    placement_parent_ids: tuple[str, ...]
```

Validate with stdlib helpers; add no dependency.

For a folder locator, `note_id`, `membership_id`, and `placement_offset` are `None`;
`placement_id` is the stable folder placement ID and the final path step carries the
folder's parent-relative offset. A note locator fills `note_id` and
`placement_offset`, and fills `membership_id` only for a real folder placement.

- [ ] **Step 4: Write failing reducer tests**

Cover replace, adjacent append/prepend, distant locator replacement, generation ignore, changed-total drift, identity overlap, incoherent offsets, one reset, second-failure stale, and total withdrawal. Assert immutable inputs do not change.

```python
result = apply_notes_slice_page(current, incoming, direction="more", request_generation=2, topology_epoch=7)
assert result.kind == "drift"
assert result.recovery == "reset_first"
```

- [ ] **Step 5: Run reducer RED**

Run: `python3 -m pytest Tests/Library/test_library_notes_tree_paging.py -q`

Expected: FAIL because the paging module does not exist.

- [ ] **Step 6: Implement the pure reducer**

Add frozen slice/branch state, root/folder plus `folders`/`placements` keys, stable pager IDs, and pure loading/apply/drift/recovery/stale/invalidation functions. Keep one contiguous tuple. Return explicit `applied`, `ignored`, or `drift` results instead of raising for expected races.

- [ ] **Step 7: Run focused GREEN**

Run: `python3 -m pytest Tests/Notes/test_note_folder_models.py Tests/Library/test_library_notes_tree_paging.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Notes/note_folder_models.py tldw_chatbook/Library/library_notes_tree_paging.py Tests/Notes/test_note_folder_models.py Tests/Library/test_library_notes_tree_paging.py
git commit -m "feat(library): model Notes tree branch paging"
```

---

### Task 2: Add Exact Parent-Scoped Folder and Placement Queries

**Files:**
- Modify: `tldw_chatbook/Notes/note_folder_repository.py`
- Modify: `Tests/Notes/test_note_folder_repository.py`

- [ ] **Step 1: Write failing child-folder page tests**

Seed 45 roots and 45 children under one parent. Assert offsets 0/20/40 return at most 20 direct children, total 45, echoed starts, correct earlier/later offsets, and an exact total on an empty out-of-range request.

- [ ] **Step 2: Write failing visible-placement tests**

Seed more than 20 Unfiled notes, more than 20 memberships under one folder, two surviving memberships for one note, and a managed ancestor shadowed by a managed descendant. Assert duplicate survivors count twice, the shadowed ancestor does not count, ordering is title `NOCASE`/note ID/membership ID, and each page renders at most 20 placements.

- [ ] **Step 3: Run repository RED**

Run:

```bash
python3 -m pytest Tests/Notes/test_note_folder_repository.py -k "child_folder_pages or note_placement_pages or unfiled_placement_page" -q
```

Expected: FAIL because `page_child_folders` and `page_note_placements` do not exist.

- [ ] **Step 4: Implement `page_child_folders`**

Use one transaction, exact `COUNT(*)`, and `normalized_name, id` ordering. Return `NoteFolderChildPage`; never infer total from the first window row.

- [ ] **Step 5: Implement `page_note_placements`**

Build an effective-membership CTE before count/slice. For a managed membership,
exclude it only when an **active managed** membership with the same note and owner
exists in an active strict descendant. Folder names may contain SQL wildcard
characters, so do not use an unescaped `LIKE`. Use an exact prefix expression:

```sql
child_m.ownership = 'managed'
AND child_m.note_id = m.note_id
AND child_m.owner_id = m.owner_id
AND substr(child_f.normalized_path, 1, length(f.normalized_path) + 1)
    = f.normalized_path || '/'
```

Count and order effective rows before `LIMIT/OFFSET`. Use a separate root query for
active notes with no active membership and synthesize `membership=None` Unfiled
placements. Bind all values and reuse existing validators.

- [ ] **Step 6: Add query-count and malformed-input tests**

Prove query count is constant with row count and malformed parent IDs/limits/offsets fail through typed validation.

- [ ] **Step 7: Run repository GREEN**

Run: `python3 -m pytest Tests/Notes/test_note_folder_repository.py -q`

Expected: PASS, including legacy tree tests.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Notes/note_folder_repository.py Tests/Notes/test_note_folder_repository.py
git commit -m "feat(notes): page exact folder tree branches"
```

---

### Task 3: Add Locators, Affected-Parent Lookup, and Filter Pages

**Files:**
- Modify: `tldw_chatbook/Notes/note_folder_repository.py`
- Modify: `Tests/Notes/test_note_folder_repository.py`

Implement these exact repository signatures so later tasks do not invent competing
interfaces:

```python
def page_child_folders(*, parent_id: str | None, limit: int, offset: int) -> NoteFolderChildPage: ...
def page_note_placements(*, parent_id: str | None, limit: int, offset: int) -> NotePlacementPage: ...
def locate_note_tree_folder(*, folder_id: str, page_size: int) -> NoteTreeLocation | None: ...
def locate_note_tree_placement(*, note_id: str, page_size: int, preferred_folder_id: str | None = None, preferred_membership_id: str | None = None) -> NoteTreeLocation | None: ...
def load_note_tree_mutation_context(*, folder_ids: Iterable[str] = (), note_ids: Iterable[str] = (), include_folder_subtrees: bool = False) -> NoteTreeMutationContext: ...
def search_note_tree_placements(*, query: str, limit: int, offset: int) -> NotePlacementPage: ...
```

- [ ] **Step 1: Write failing folder-locator tests**

Seed a deep path whose target folder lies beyond offset 20 at root and child levels. Assert every `NoteTreePathStep`, including the target folder, reports its exact parent and containing 20-item offset.

- [ ] **Step 2: Write failing placement-locator tests**

Cover exact preferred membership, same-folder fallback, canonical normalized-path fallback, Unfiled fallback, removed target, and duplicate placements. Assert the note offset matches Task 2 ordering.

- [ ] **Step 3: Write failing affected-parent tests**

Assert exact reads return old/new parents, moved subtree IDs, ancestor IDs, and every active placement parent for a note deletion without consulting loaded UI state.

- [ ] **Step 4: Write failing coherent filter-page tests**

Seed title/content FTS matches, folder-breadcrumb-only matches, Unfiled matches, duplicates, and shadowed managed placements. Assert exact placement total, 20-row bounds, folder-path ordering, Unfiled-last ordering, and complete ancestors only for returned placements.

- [ ] **Step 5: Run locator/filter RED**

Run: `python3 -m pytest Tests/Notes/test_note_folder_repository.py -k "tree_locator or affected_parents or search_note_placement_page" -q`

Expected: FAIL because the seams are missing.

- [ ] **Step 6: Implement locators in one read transaction**

Share SQL ordering fragments with page queries. Build root-to-target steps by stable folder ID; calculate each parent-relative rank with `normalized_name, id`. For notes choose preferred surviving membership, same folder, canonical path, then Unfiled. Return `None` for removed targets.

- [ ] **Step 7: Implement exact affected-parent lookup**

Implement `load_note_tree_mutation_context` by reusing `_load_subtree` and existing
folder reads. Return `NoteTreeMutationContext`; do not introduce another hierarchy
model.

- [ ] **Step 8: Implement coherent filter placement paging**

In one transaction combine existing FTS note matches with normalized folder-path matches, apply effective-placement suppression, count visible placements, order, slice, and load ancestors. Do not compose a capped `search_notes` snapshot with `load_tree_search`.

- [ ] **Step 9: Run repository GREEN**

Run: `python3 -m pytest Tests/Notes/test_note_folder_repository.py -q`

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add tldw_chatbook/Notes/note_folder_repository.py Tests/Notes/test_note_folder_repository.py
git commit -m "feat(notes): locate and filter paged tree placements"
```

---

### Task 4: Expose Policy-Checked Off-Loop Service Seams

**Files:**
- Modify: `tldw_chatbook/Notes/notes_scope_service.py`
- Modify: `Tests/Notes/test_notes_scope_service_folders.py`

- [ ] **Step 1: Write failing routing/policy tests**

Add fake-repository tests for child pages, placement pages, folder/note locators, affected-parent reads, and filter pages. Assert exact kwargs, local user validation, policy denial before repository access, and unsupported-scope failure.

- [ ] **Step 2: Write failing off-loop tests**

Reuse the thread-recording repository fake and assert each new synchronous repository method runs through `_run_folder_repository`, not the asyncio loop thread.

- [ ] **Step 3: Run service RED**

Run: `python3 -m pytest Tests/Notes/test_notes_scope_service_folders.py -k "branch_page or tree_locator or affected_parent or placement_filter" -q`

Expected: FAIL because public methods are absent.

- [ ] **Step 4: Add minimal service methods**

Add same-named async service methods—`page_note_folder_children`,
`page_note_placements`, `locate_note_tree_folder`,
`locate_note_tree_placement`, `load_note_tree_mutation_context`, and
`search_note_tree_placements`—with the repository signature plus `scope` and `user_id`.
Use Google-style docstrings, `_folder_repository_for_action(... operation="list")`,
and `_run_folder_repository`. Keep legacy broad tree/search methods until the screen
cutover so intermediate commits stay green.

- [ ] **Step 5: Run service GREEN**

Run: `python3 -m pytest Tests/Notes/test_notes_scope_service_folders.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Notes/notes_scope_service.py Tests/Notes/test_notes_scope_service_folders.py
git commit -m "feat(notes): expose paged tree service seams"
```

---

### Task 5: Project and Render Inline Branch Boundaries

**Files:**
- Modify: `tldw_chatbook/Library/library_notes_tree_state.py`
- Modify: `tldw_chatbook/Widgets/Library/library_notes_canvas.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Tests/Library/test_library_notes_tree_state.py`
- Modify: `Tests/Widgets/Library/test_library_notes_canvas.py`

- [ ] **Step 1: Write failing projection tests**

Replace aggregate-cursor expectations with root, Unfiled, and expanded-folder branch state. Assert ordering is root folders → root folder pager → Unfiled placements → Unfiled pager; inside a folder it is child folders → child-folder pager → placements → placement pager. Assert stable pager IDs, exact/stale copy, and no pager for exhaustion.

- [ ] **Step 2: Run projection RED**

Run: `python3 -m pytest Tests/Library/test_library_notes_tree_state.py -q`

Expected: FAIL against the aggregate `expanded_page` projection.

- [ ] **Step 3: Add the paged projection without breaking the live screen**

Add `build_paged_library_notes_tree(...)` accepting parent-keyed state and an explicit
`pager` row kind carrying parent ID, content kind, action, status, and disabled state.
Keep existing folder/note IDs and managed semantics. Leave the current
`build_library_notes_tree(... root_page, expanded_page, ...)` compatibility function
and its aggregate projection fields intact until Task 6 migrates `LibraryScreen`.
Task 5 tests exercise the new projection while the legacy tests prove the intermediate
commit remains runnable.

- [ ] **Step 4: Write failing widget tests**

Assert earlier/more/loading/stale/Retry controls render at the right boundary, carry
exact metadata, expose text without color, and disable unsafe mutation controls when
the selected branch is stale. Also retain one compatibility test showing an existing
legacy projection still renders before the Task 6 cutover.

- [ ] **Step 5: Run widget RED**

Run: `python3 -m pytest Tests/Widgets/Library/test_library_notes_canvas.py -q`

Expected: FAIL because only the global `#library-notes-tree-more` exists.

- [ ] **Step 6: Implement pager rendering**

Use `.library-notes-tree-pager` controls with stable IDs. Store `parent_folder_id`, `content_kind`, and `paging_action` metadata; never parse labels or index suffixes. Render user/status copy with markup disabled.

- [ ] **Step 7: Add narrow source CSS and rebuild**

Use auto height, wrapping, and stacked controls; avoid fixed widths that steal title space.

Run: `python3 tldw_chatbook/css/build_css.py`

Expected: generated bundle reflects source CSS without manual editing.

- [ ] **Step 8: Run projection/widget GREEN**

Run: `python3 -m pytest Tests/Library/test_library_notes_tree_state.py Tests/Widgets/Library/test_library_notes_canvas.py -q`

Expected: PASS, including the unchanged current-screen compatibility path.

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/Library/library_notes_tree_state.py tldw_chatbook/Widgets/Library/library_notes_canvas.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/Library/test_library_notes_tree_state.py Tests/Widgets/Library/test_library_notes_canvas.py
git commit -m "feat(library): render Notes branch paging controls"
```

---

### Task 6: Cut the Screen Over to Branch-Local Browse Workers

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_library_notes_folder_navigator.py`

- [ ] **Step 1: Write failing initial-load/expansion tests**

Assert root folders and Unfiled load independently at limit 20, expansion loads one exact parent, collapse retains fresh state, and re-expansion avoids a redundant read.

- [ ] **Step 2: Write failing continuation/focus tests**

Activate exact folder/note earlier/more controls. Assert only that slice loads, the request carries its expected start, failure paints local Retry, and completion moves focus only while the pager still owns focus.

- [ ] **Step 3: Write failing race/drift tests**

Cover simultaneous sibling requests, older/newer same-slice responses, topology mismatch, changed total, overlap, out-of-range response, one automatic reset, and second-failure stale transition. Inconsistent rows must never merge.

- [ ] **Step 4: Write failing first-load/unmount tests**

Assert root failure paints root Retry, first expansion failure paints beneath its folder, canvas recomposition preserves authority, and true screen unmount prevents late apply/repaint and fresh-visit seeding.

- [ ] **Step 5: Run orchestration RED**

Run: `python3 -m pytest Tests/UI/test_library_notes_folder_navigator.py -k "initial or expansion or branch or drift or unmount" -q`

Expected: FAIL against aggregate tree loading.

- [ ] **Step 6: Replace aggregate browse fields and every direct reader**

Add `LIBRARY_NOTES_TREE_PAGE_SIZE = 20`, parent-keyed branches, topology epoch,
lifecycle generation, and per-slice generations. Migrate projection building, loaded
folder target options, selection lookup, expansion, canvas kwargs, and paging handlers
to the branch map. Remove `_library_notes_tree_root_page`,
`_library_notes_tree_expanded_page`, `_library_notes_tree_membership_note_offset`, and
the global Load-more path only after `rg` proves no direct reader remains.

Keep two explicit compatibility bridges until Task 7:

- active filter mode may still project `_library_notes_tree_search_page` through the
  legacy function;
- `_request_library_notes_tree_refresh(refresh_root=...)` maps mutation callers onto
  conservative branch invalidation/root refresh rather than aggregate pages.

This keeps filter/navigation/mutation tests runnable while browse paging is already
cut over.

- [ ] **Step 7: Implement exact slice workers and drift recovery**

Workers call Task 4 seams, capture slice/topology/lifecycle authority, apply through the pure reducer, and schedule at most one reset. Use a stable worker group per slice so siblings do not cancel each other while newer same-slice work supersedes older work.

- [ ] **Step 8: Invalidate on true screen unmount**

Extend `LibraryScreen.on_unmount` before its first await. Persist semantic scope only; never page records/loading/errors.

- [ ] **Step 9: Prove the intermediate commit has no aggregate runtime dependency**

Run:

```bash
rg '_library_notes_tree_(root_page|expanded_page|membership_note_offset)|_request_library_notes_tree_more' tldw_chatbook/UI/Screens/library_screen.py
```

Expected: no matches. Then run the entire navigator module so compatibility filter and
mutation paths remain green.

- [ ] **Step 10: Run orchestration GREEN**

Run: `python3 -m pytest Tests/UI/test_library_notes_folder_navigator.py -q`

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_notes_folder_navigator.py
git commit -m "feat(library): page Notes tree branches independently"
```

---

### Task 7: Reconcile Navigation, Filters, and Committed Mutations

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/Library/library_notes_tree_state.py`
- Modify: `Tests/UI/test_library_notes_folder_navigator.py`
- Modify: `Tests/Library/test_library_notes_tree_state.py`
- Modify: `Tests/Widgets/Library/test_library_notes_canvas.py`

- [ ] **Step 1: Write failing deep-link/Back tests**

Cover off-range folders, duplicate note placements, external note deep link, Back before locator completion, Back after topology change, and removed-target fallback. Editor fetch, locator, and receipt must share one navigation generation.

- [ ] **Step 2: Write failing filter tests**

Assert Enter requests offset 0/limit 20 from the coherent placement-filter seam, earlier/more remains filter-owned, browse branches are untouched, and clearing/Back restores semantic receipts.

- [ ] **Step 3: Write failing mutation-reconciliation tests**

For folder create/rename/move/delete/restore, placement move/partial move, note create,
and note delete, assert exact affected parents. Also cover:

- a rejected/failed mutation that commits no storage change, proving the paging fence
  clears and last-good trusted ranges remain usable;
- note creation refreshing Unfiled plus exact placement parents when applicable;
- deletion fallback in the required next sibling → previous sibling → parent →
  canonical visible placement order;
- post-commit refresh failure, proving invalid rows disappear, committed
  labels/versions remain, totals withdraw, unsafe affected-branch actions disable,
  Retry stays enabled, and unrelated branches work.

- [ ] **Step 4: Run navigation/mutation RED**

Run: `python3 -m pytest Tests/UI/test_library_notes_folder_navigator.py -k "locator or deep_link or return or filter or mutation or stale" -q`

Expected: FAIL because receipts/mutations still depend on aggregate pages.

- [ ] **Step 5: Extend the semantic receipt**

Store placement/note IDs, expanded IDs, contiguous range descriptors, filter range,
focus, scroll, lifecycle generation, and topology epoch—not records. Reuse live data
only when lifecycle generation and topology epoch both still match; otherwise reload
containing ranges.

- [ ] **Step 6: Implement locator-driven navigation**

Load each path step’s containing range root-to-target, expand ancestors, then load/focus the placement range. Keep `Locating note…` generation guarded and never steal focus after the receipt is abandoned.

- [ ] **Step 7: Replace capped broad filter composition**

Use the coherent placement-filter page and its ancestors. Apply the same one-clamp
drift recovery; never overwrite browse branches. Once filter callers are migrated,
remove `_library_notes_tree_search_page`, the legacy projection compatibility function
and aggregate cursor fields retained in Task 5, and the conservative mutation refresh
bridge retained in Task 6. `rg` for those names must return no production callers.

- [ ] **Step 8: Implement operation-aware local reconciliation**

At mutation admission fence paging and bump topology authority. If no storage change
commits, clear the fence/loading state and retain last-good trusted rows. On confirmed
success remove known-invalid rows, update returned snapshots/versions, retain desired
stable selection, withdraw affected totals, then refresh exact affected slices. Note
create invalidates Unfiled and returned placement parents. Delete applies captured
stable-neighbour fallback. If refresh fails, mark only affected slices stale and gate
unsafe actions until Retry.

- [ ] **Step 9: Add privacy-safe error-log assertions**

Capture logs for page, locator, filter, and mutation refresh failures. Assert metadata
contains operation/kind/generation/exception class and excludes note titles, folder
paths, filter text, and note bodies.

- [ ] **Step 10: Run navigation/widget GREEN**

Run: `python3 -m pytest Tests/UI/test_library_notes_folder_navigator.py Tests/Library/test_library_notes_tree_state.py Tests/Widgets/Library/test_library_notes_canvas.py -q`

Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/Library/library_notes_tree_state.py Tests/UI/test_library_notes_folder_navigator.py Tests/Library/test_library_notes_tree_state.py Tests/Widgets/Library/test_library_notes_canvas.py
git commit -m "feat(library): reconcile paged Notes placements"
```

---

### Task 8: Prove Geometry, Cross-Reader Safety, and Live Reachability

**Files:**
- Create: `Tests/Live/test_library_notes_tree_paging_live.py`
- Modify: `Tests/UI/test_library_adaptive_reader_closeout.py`
- Modify if mounted evidence proves a defect: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify if mounted evidence proves a defect: `tldw_chatbook/Widgets/Library/library_notes_canvas.py`
- Modify if mounted evidence proves a defect: `tldw_chatbook/Library/library_notes_tree_state.py`
- Modify if mounted evidence proves a defect: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate after any CSS correction: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Docs/User_Guide/library/notes.md`
- Create: `Docs/superpowers/reviews/evidence/task-18917/live-walkthrough.md`
- Modify: `backlog/tasks/task-18917 - Add-placement-aware-paging-to-the-Library-Notes-tree.md`

- [ ] **Step 1: Write mounted production-shaped regressions**

Use the real adaptive-reader shell and Notes canvas at 160×50, 120×35, 100×30,
and 80×24. Seed identifying long titles and both pager types. Assert compositor text,
Items-pane containment, no horizontal overflow, focus after load/retry, navigator
scroll offset preservation across branch-local canvas synchronization, and
Library/Items collapse behavior. Keep Conversations, Media, Prompts, and Skills
expectations unchanged.

- [ ] **Step 2: Run mounted RED**

Run: `python3 -m pytest Tests/UI/test_library_adaptive_reader_closeout.py Tests/Widgets/Library/test_library_notes_canvas.py -q`

Expected: new Notes paging geometry assertions fail until final CSS/focus corrections; existing cross-reader cases stay green.

- [ ] **Step 3: Make evidence-driven geometry corrections**

Change only proven Notes CSS/focus defects. After source CSS changes run: `python3 tldw_chatbook/css/build_css.py`.

- [ ] **Step 4: Re-run mounted GREEN and commit any corrections**

Run: `python3 -m pytest Tests/UI/test_library_adaptive_reader_closeout.py Tests/Widgets/Library/test_library_notes_canvas.py -q`

Expected: PASS. If Step 3 changed production/generated files, commit them now:

```bash
git add tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/Widgets/Library/library_notes_canvas.py tldw_chatbook/Library/library_notes_tree_state.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_library_adaptive_reader_closeout.py
git commit -m "fix(library): harden Notes paging geometry"
```

- [ ] **Step 5: Add isolated real-repository live walkthrough**

Use `tmp_path`, a real temporary ChaChaNotes DB, `LocalNoteFolderRepository`, and `NotesScopeService`. Seed at least 25 roots, 25 Unfiled notes, one parent with 25 children and 45 visible placements, deep ancestors, duplicate placements, and a shadowed managed ancestor. Drive expansion, more/earlier, located middle range, mutation, injected one-shot failure, Retry, collapse/re-expand, and all four sizes with Textual’s pilot.

- [ ] **Step 6: Run live walkthrough and record evidence**

Run: `python3 -m pytest Tests/Live/test_library_notes_tree_paging_live.py -q -s`

Expected: PASS at all four sizes with identifying range/title/focus/retry observations. Record command, seed cardinalities, observed text, and results in `Docs/superpowers/reviews/evidence/task-18917/live-walkthrough.md`.

- [ ] **Step 7: Update user guide**

Document fixed 20-item branch paging, folder versus note controls, located middle ranges/Load earlier, stale/Retry behavior, filter paging, and keyboard focus.

- [ ] **Step 8: Run complete targeted feature/cross-reader suite**

```bash
python3 -m pytest Tests/Notes/test_note_folder_models.py Tests/Notes/test_note_folder_repository.py Tests/Notes/test_notes_scope_service_folders.py Tests/Library/test_library_notes_tree_paging.py Tests/Library/test_library_notes_tree_state.py Tests/UI/test_library_notes_folder_navigator.py Tests/Widgets/Library/test_library_notes_canvas.py Tests/UI/test_library_adaptive_reader_shell.py Tests/UI/test_library_adaptive_reader_closeout.py Tests/Live/test_library_notes_tree_paging_live.py -q
```

Expected: PASS. This is targeted, not the full repository suite.

- [ ] **Step 9: Run static/generated checks**

```bash
python3 tldw_chatbook/css/build_css.py
git diff --exit-code -- tldw_chatbook/css/tldw_cli_modular.tcss tldw_chatbook/css/widget_defaults_self.tcss tldw_chatbook/css/widget_defaults_scoped.tcss tldw_chatbook/css/screen_css_self.tcss tldw_chatbook/css/screen_css_scoped.tcss
git diff --check
python3 -m compileall -q tldw_chatbook/Notes tldw_chatbook/Library tldw_chatbook/Widgets/Library tldw_chatbook/UI/Screens/library_screen.py
```

Expected: all exit 0. Step 4 already committed the intended generated output, so a
post-build CSS diff now proves generation is deterministic and current.

- [ ] **Step 10: Self-review**

Review `git diff origin/dev...HEAD` for SQL parameterization, content leakage in logs, false totals, row-index identity, missing generations, unmount authority, unrelated reader changes, and broad-snapshot reuse. Re-run any affected focused test.

- [ ] **Step 11: Complete Backlog hygiene only after green evidence**

Check every AC, add concise Implementation Notes naming ADR-067 and exact
automated/live evidence, and set status to `Done`. TASK-18917 is a five-digit ID, so
follow `backlog/docs/lessons-backlog-hygiene.md`: edit the task file directly and do
**not** call the known-broken `backlog task edit 18917`, which can create
`task-task- - .md`. Verify the exact task path and absence of that ghost file. Add a
lessons entry only for a genuine reusable incident.

- [ ] **Step 12: Commit closeout**

```bash
git add Tests/Live/test_library_notes_tree_paging_live.py Tests/UI/test_library_adaptive_reader_closeout.py Docs/User_Guide/library/notes.md Docs/superpowers/reviews/evidence/task-18917/live-walkthrough.md 'backlog/tasks/task-18917 - Add-placement-aware-paging-to-the-Library-Notes-tree.md' tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "test(library): verify Notes tree paging closeout"
```

---

## Final Evidence Required

- RED then GREEN evidence for each production behavior.
- Exact targeted test pass counts, `git diff --check`, compile check, and CSS bundle check.
- Production-shaped cross-reader results at all four required terminal sizes.
- Isolated real-repository proof of root, Unfiled, child-folder, direct-placement, deep-link, mutation, failure, Retry, and collapse behavior.
- A clean TASK-18917 worktree with no unrelated user changes.
