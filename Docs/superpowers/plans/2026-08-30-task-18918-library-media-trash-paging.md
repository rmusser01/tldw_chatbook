# TASK-18918 Library Media Trash Paging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every deleted local Media item reachable through exact 20-item Trash pages with independent title/type filters, safe Restore and permanent deletion, truthful stale recovery, deterministic focus, and compact-terminal reachability.

**Architecture:** Add a local-only coherent Trash page read beside the legacy compatibility seam, normalize it into a fail-closed canonical envelope, and give Trash its own scope/result/reducer/controller. `LibraryScreen` remains the navigation, shared-mutation, and focus owner; `LibraryMediaTrashCanvas` remains the renderer. Normal Media keeps its existing controller and page context: Restore only marks that retained page stale, and neither Trash reads nor permanent deletion overwrite it.

**Tech Stack:** Python 3.11+, Textual 8.x, SQLite/FTS5, immutable dataclasses and mappings, pytest/pytest-asyncio, Ruff, existing TCSS bundle tooling.

**Spec:** `Docs/superpowers/specs/2026-08-30-task-18918-library-media-trash-paging-design.md`

## Global Constraints

- ADR required: no; ADR path: `backlog/decisions/067-library-top-level-pagination-contracts.md`.
- Trash is local-only; `mode="local"` is required and no server API or server service changes are permitted.
- Page size is exactly 20; page and coordinate validation rejects booleans/non-integers and offsets above `2**63 - 1`.
- Search matches title metadata only, trims input, rejects embedded NUL, and is limited to 200 characters.
- Type filtering uses trimmed values and exact case-sensitive equality; facets are complete-source, unique, sorted, and non-empty.
- Page order is NULL-last `trash_date DESC`, then NULL-last `last_modified DESC`, then `id DESC`.
- Exact totals/ranges are visible only for a validated fresh page; committed mutations become stale immediately and cannot be reclassified as failures by a refresh error.
- Restore preserves normal Media scope/page/records/selection/focus/scroll but marks that page stale; it never inserts an unranked row.
- Permanent deletion is per-item only, uses `MediaReadingScopeService.permanently_delete_media_item(mode="local", media_id=...)`, and preserves the existing physical cascade, FTS cleanup, and no-sync-log behavior.
- Library and Items panes retain their shipped collapse behavior; Trash must work at 160×50, 120×35, 100×30, and 80×24.
- Diagnostics are metadata-only: never log query text, titles, raw records, paths, content, credentials, stable private IDs, or permanent-delete targets.
- Do not run the repository-wide pytest suite without fresh user approval. Verification is limited to changed owners and their direct cross-reader regressions.

---

## File and ownership map

**Production owners**

- `tldw_chatbook/DB/Client_Media_DB_v2.py` — one coherent local Trash count/page/facet transaction and deterministic SQL ordering.
- `tldw_chatbook/Media/local_media_reading_service.py` — thin raw local Trash envelope adapter; legacy `list_media_trash` remains unchanged.
- `tldw_chatbook/Media/media_reading_scope_service.py` — local-only policy/off-loop boundary and canonical five-key Trash summaries.
- `tldw_chatbook/Library/library_media_state.py` — immutable Trash scope/result/state, pure transition helpers, exact envelope validation, and row projection.
- `tldw_chatbook/UI/Library_Modules/library_media_trash_browse_controller.py` — new Trash-only generation, clamp, request-origin, Retry, and worker owner.
- `tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py` — one small normal-Media stale-marking method used after Restore; no Trash records or workers.
- `tldw_chatbook/UI/Screens/library_screen.py` — entry/Back return receipt, focus authority, screen event wiring, shared Media mutation interlock, Restore, and permanent-delete dispatch.
- `tldw_chatbook/Widgets/Library/library_media_trash_canvas.py` — filters, exact pager, rows, status, Retry, and safe inline confirmation.
- `tldw_chatbook/css/components/_agentic_terminal.tcss` plus generated `tldw_chatbook/css/tldw_cli_modular.tcss` — only if production-shaped geometry proves an inline/layout change is insufficient.

**Focused test owners**

- New `Tests/DB/test_client_media_trash_pagination.py`.
- Existing `Tests/Media/test_local_media_reading_service.py` and `Tests/Media/test_media_reading_scope_service.py`.
- Existing `Tests/Library/test_library_media_trash_state.py`.
- New `Tests/UI/test_library_media_trash_browse_controller.py`.
- Existing `Tests/UI/test_library_media_trash.py`, `Tests/UI/test_library_media_side_by_side.py`, `Tests/UI/test_library_adaptive_reader_shell.py`, and `Tests/UI/test_library_adaptive_reader_closeout.py`.

---

### Task 0: Freeze the boundary and focused baseline

**Files:**

- Create: `Docs/superpowers/plans/2026-08-30-task-18918-library-media-trash-paging.md`
- Modify: `backlog/tasks/task-18918 - Add-paged-recovery-viewing-to-Library-Media-Trash.md`

**Interfaces:**

- Consumes: approved design spec and ADR-067.
- Produces: committed implementation plan, exact baseline node list, and task metadata used by Tasks 1–7.

- [ ] **Step 1: Record the exact branch base and plan-only worktree state.**

```bash
git merge-base HEAD origin/dev
git log --oneline origin/dev..HEAD
git status --short
```

Expected: only the two approved design commits plus this plan/task edit are ahead of the base; no unrelated files are modified.

- [ ] **Step 2: Run the current Trash-focused baseline.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Library/test_library_media_trash_state.py \
  Tests/UI/test_library_media_trash.py

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Media/test_local_media_reading_service.py \
  Tests/Media/test_media_reading_scope_service.py \
  -k 'media_trash or permanently_delete_media_item or restore_media_item'
```

Expected: existing tests pass. Record exact failing node IDs instead of changing production if the moved base has a baseline defect.

- [ ] **Step 3: Attach the plan and ADR check to TASK-18918.**

Add this exact plan summary to the task file:

```markdown
## Implementation Plan

1. Add a coherent local-only database page/count/facet contract.
2. Propagate and canonically validate the exact envelope through Media services.
3. Add immutable Trash paging state plus a Trash-specific request controller.
4. Wire screen entry, paging/filter generations, Back receipt, and lifecycle fencing.
5. Render the bounded pager/filter/confirmation surface at all supported sizes.
6. Reconcile Restore and permanent deletion through the shared Media mutation owner.
7. Run focused automated/live verification, review, documentation, and closeout.

ADR required: no
ADR path: backlog/decisions/067-library-top-level-pagination-contracts.md
Reason: ADR-067 already governs exact source-owned pages and stale mutation recovery.
```

- [ ] **Step 4: Run the planning checkpoint checks.**

```bash
git diff --check
rg -n 'T[B]D|TO[D]O|implement[ ]later|similar[ ]to Task' \
  Docs/superpowers/plans/2026-08-30-task-18918-library-media-trash-paging.md
```

Expected: `git diff --check` passes and the placeholder scan prints nothing.

- [ ] **Step 5: Commit the plan checkpoint.**

```bash
git add Docs/superpowers/plans/2026-08-30-task-18918-library-media-trash-paging.md \
  "backlog/tasks/task-18918 - Add-paged-recovery-viewing-to-Library-Media-Trash.md"
git commit -m "docs: plan Media Trash paging"
```

---

### Task 1: Read exact Trash pages coherently at the database boundary

**Files:**

- Modify: `tldw_chatbook/DB/Client_Media_DB_v2.py`
- Create: `Tests/DB/test_client_media_trash_pagination.py`

**Interfaces:**

- Consumes: `MediaDatabase.transaction()` and the existing `Media` table; no schema change.
- Produces:

```python
def list_library_media_trash_page(
    self,
    *,
    query: str = "",
    media_type: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """Return raw local Trash rows plus coherent total and complete facets."""
```

The raw DB envelope is:

```python
{
    "items": [
        {"id": 41, "title": "Doc", "type": "pdf", "trash_date": "..."}
    ],
    "total": 45,
    "limit": 20,
    "offset": 40,
    "types": ["audio", "pdf", "video"],
}
```

- [ ] **Step 1: Write RED coordinate, filter, ordering, and projection tests.**

Seed at least 45 trashed rows plus active and soft-deleted decoys. Include repeated/null timestamps, literal `%`, `_`, and `\\` titles, blank/padded/mixed-case types, and privacy sentinels. The first focused test should resemble:

```python
def _seed_trash(database: MediaDatabase, *, count: int) -> None:
    for index in range(count):
        media_id, _uuid, _message = database.add_media_with_keywords(
            title=f"Trash {index:02d}",
            media_type="pdf" if index % 2 else "audio",
            content=f"private-content-{index}",
            keywords=[],
        )
        assert media_id is not None
        assert database.mark_as_trash(media_id)


def test_library_trash_pages_filter_before_slicing_and_echo_coordinates(media_db):
    _seed_trash(media_db, count=45)

    page = media_db.list_library_media_trash_page(limit=20, offset=20)

    assert page["total"] == 45
    assert page["limit"] == 20
    assert page["offset"] == 20
    assert len(page["items"]) == 20
    assert set(page["items"][0]) == {"id", "title", "type", "trash_date"}
```

Add tests proving:

- offsets `0`, `20`, and `40` return 20, 20, and 5 rows;
- booleans, non-integers, negative coordinates, zero/non-20 limit, and values above `2**63 - 1` fail before SQL;
- literal title search escapes `%`, `_`, and `\\` rather than broadening scope;
- `TRIM(type) = ? COLLATE BINARY` is exact and case-sensitive;
- facets ignore query/active type/page, trim values, omit blanks, and sort uniquely;
- NULL `trash_date` and `last_modified` sort last, with `id DESC` as final tie-break;
- rows, count, and facets share one transaction under a coordinated WAL mutation;
- raw rows never contain content, paths, blobs, credentials, client IDs, or other detail fields.

- [ ] **Step 2: Run the new file to verify RED.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/DB/test_client_media_trash_pagination.py
```

Expected RED: `MediaDatabase` has no `list_library_media_trash_page` method.

- [ ] **Step 3: Implement strict request validation and one shared predicate.**

Use explicit checks rather than `int(...)` repair:

```python
if type(limit) is not int or limit != 20:
    raise ValueError("Library Media Trash limit must equal 20.")
if type(offset) is not int or not 0 <= offset <= 2**63 - 1:
    raise ValueError("Library Media Trash offset is invalid.")
query = query.strip()
if "\x00" in query or len(query) > 200:
    raise ValueError("Library Media Trash query is invalid.")
media_type = media_type.strip() if media_type is not None else None
media_type = media_type or None
```

Build the row/count predicate once. Escape LIKE metacharacters with a small private helper and bind every value:

```python
escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
conditions.append("title LIKE ? ESCAPE '\\'")
params.append(f"%{escaped}%")
```

- [ ] **Step 4: Implement the coherent transaction and exact order.**

Inside one `with self.transaction() as conn:` execute count, bounded rows, and source-wide facets. Use this exact order:

```sql
ORDER BY trash_date IS NULL ASC,
         trash_date DESC,
         last_modified IS NULL ASC,
         last_modified DESC,
         id DESC
LIMIT ? OFFSET ?
```

The facet query must use only `deleted = 0 AND is_trash = 1`; it must not inherit title/type filters.

- [ ] **Step 5: Run GREEN and two inverse checks.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/DB/test_client_media_trash_pagination.py
```

Then temporarily split count/rows into separate transactions and remove `id DESC`; require the coherent-snapshot and tie-break tests to fail, restoring production after each inverse.

- [ ] **Step 6: Commit the database slice.**

```bash
git add tldw_chatbook/DB/Client_Media_DB_v2.py \
  Tests/DB/test_client_media_trash_pagination.py
git commit -m "feat: page Media Trash at the database"
```

---

### Task 2: Propagate the local-only canonical Trash envelope

**Files:**

- Modify: `tldw_chatbook/Media/local_media_reading_service.py`
- Modify: `tldw_chatbook/Media/media_reading_scope_service.py`
- Modify: `Tests/Media/test_local_media_reading_service.py`
- Modify: `Tests/Media/test_media_reading_scope_service.py`

**Interfaces:**

- Consumes: `MediaDatabase.list_library_media_trash_page(...)` from Task 1.
- Produces:

```python
# LocalMediaReadingService
def list_library_media_trash(
    self,
    *,
    query: str = "",
    media_type: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]: ...

# MediaReadingScopeService
async def list_library_media_trash(
    self,
    *,
    mode: MediaReadingBackend | str | None = None,
    query: str = "",
    media_type: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]: ...
```

The scope envelope contains exactly `items`, `total`, `limit`, `offset`, and `types`; each item contains exactly `id`, `backing_media_id`, `title`, `media_type`, and `trash_date`.

- [ ] **Step 1: Write RED local/scope service tests.**

Use spies that reject unexpected keys and a raw DB item with blank title/type:

```python
@pytest.mark.asyncio
async def test_scope_library_trash_is_local_only_and_canonical(scope_service):
    payload = await scope_service.list_library_media_trash(
        mode="local", query="doc", media_type="pdf", limit=20, offset=40
    )

    assert set(payload) == {"items", "total", "limit", "offset", "types"}
    assert set(payload["items"][0]) == {
        "id", "backing_media_id", "title", "media_type", "trash_date"
    }
    assert payload["items"][0]["id"] == "local:media:41"
```

Prove the exact arguments reach the DB, the legacy `list_media_trash` call shape stays unchanged, missing/malformed response keys remain observable for Task 3 to reject, local sync DB work uses `_call_local_leaf`, policy is enforced, non-local mode raises before touching the server service, and diagnostics contain no sentinels.

- [ ] **Step 2: Run RED.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Media/test_local_media_reading_service.py \
  Tests/Media/test_media_reading_scope_service.py \
  -k 'library_media_trash'
```

Expected RED: neither service exposes `list_library_media_trash`.

- [ ] **Step 3: Add the thin local adapter.**

Forward exact validated arguments without pagination repair:

```python
def list_library_media_trash(self, **kwargs: Any) -> dict[str, Any]:
    return self._require_db().list_library_media_trash_page(**kwargs)
```

Do not change, call through, or delete the legacy `list_media_trash` method.

- [ ] **Step 4: Add the scope method and canonical normalization.**

Reject non-local mode, reuse the existing Trash-list policy action, call `_call_local_leaf`, preserve envelope keys without defaults, and normalize rows once:

```python
def _normalize_local_library_trash_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    backing_id = item["id"]
    return {
        "id": f"local:media:{backing_id}",
        "backing_media_id": backing_id,
        "title": str(item.get("title") or "Untitled"),
        "media_type": str(item["type"]).strip() if item.get("type") else None,
        "trash_date": str(item["trash_date"]) if item.get("trash_date") else None,
    }
```

Do not coerce IDs, totals, coordinates, types, or item sequences into validity; Task 3's fail-closed validator owns response correctness.

- [ ] **Step 5: Run GREEN and the server-isolation inverse.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Media/test_local_media_reading_service.py \
  Tests/Media/test_media_reading_scope_service.py \
  -k 'library_media_trash or list_media_trash'
```

Temporarily allow `mode="server"`; require the server-isolation test to fail, then restore.

- [ ] **Step 6: Commit the service slice.**

```bash
git add tldw_chatbook/Media/local_media_reading_service.py \
  tldw_chatbook/Media/media_reading_scope_service.py \
  Tests/Media/test_local_media_reading_service.py \
  Tests/Media/test_media_reading_scope_service.py
git commit -m "feat: expose exact local Media Trash pages"
```

---


### Task 3: Add immutable Trash paging state and a source-owned controller

**Files:**

- Modify: `tldw_chatbook/Library/library_media_state.py`
- Create: `tldw_chatbook/UI/Library_Modules/library_media_trash_browse_controller.py`
- Modify: `Tests/Library/test_library_media_trash_state.py`
- Create: `Tests/UI/test_library_media_trash_browse_controller.py`

**Interfaces:**

- Consumes: canonical scope envelope from Task 2 and shared `build_library_pager_display`.
- Produces these source-specific types:

```python
@dataclass(frozen=True)
class MediaTrashScope:
    query: str = ""
    media_type: str | None = None
    page: int = 1

    @property
    def page_size(self) -> int: return 20

    @property
    def offset(self) -> int: return (self.page - 1) * 20


@dataclass(frozen=True)
class MediaTrashResult:
    scope: MediaTrashScope
    items: tuple[Mapping[str, Any], ...]
    total: int
    limit: int
    offset: int
    types: tuple[str, ...]


@dataclass(frozen=True)
class MediaTrashMutationTarget:
    stable_id: str
    backing_media_id: int
    title: str
    media_type: str | None
    trash_date: str | None
    page_index: int
```

Request origins use one closed vocabulary:

```python
MediaTrashRequestOrigin = Literal[
    "entry", "search", "type", "previous", "next", "retry", "mutation"
]
```

`MediaTrashBrowseState` owns requested/applied scope, retained rows, facets, `uninitialized`/`fresh`/`stale`, loading/error/stale copy, selected ID, confirmation target, mutation-pending flag, request origin, failed target, and committed notice. Pure helpers return new state; the controller owns only generation/worker handles and assigns those returned states.

The pure transition surface is:

```python
def begin_media_trash_request(state, scope, *, origin): ...
def apply_media_trash_result(state, result): ...
def fail_media_trash_request(state, failed_scope, *, copy): ...
def select_media_trash_item(state, stable_id): ...
def open_media_trash_delete_confirmation(state): ...
def cancel_media_trash_delete_confirmation(state): ...
def begin_media_trash_mutation(state): ...
def fail_media_trash_mutation(state, target, *, copy): ...
def commit_media_trash_mutation(state, target, *, notice): ...
```

The controller surface is:

```python
class LibraryMediaTrashBrowseController:
    state: MediaTrashBrowseState

    @property
    def pager(self) -> LibraryPagerDisplay: ...
    def request(self, scope: MediaTrashScope, *, origin: MediaTrashRequestOrigin, focus_identity: str | None): ...
    def retry(self, *, focus_identity: str | None): ...
    def select(self, stable_id: str) -> None: ...
    def open_delete_confirmation(self) -> MediaTrashMutationTarget | None: ...
    def cancel_delete_confirmation(self) -> None: ...
    def claim_mutation(self) -> MediaTrashMutationTarget | None: ...
    def finish_mutation_failure(self, target: MediaTrashMutationTarget, copy: str) -> None: ...
    def finish_mutation_commit(self, target: MediaTrashMutationTarget, notice: str) -> None: ...
    def request_after_mutation(self, *, focus_identity: str | None): ...
    def invalidate(self) -> int: ...
```

- [ ] **Step 1: Write RED scope/result validation tests.**

```python
def test_media_trash_scope_normalizes_and_bounds_coordinates():
    scope = MediaTrashScope(query="  doc  ", media_type=" pdf ", page=2)
    assert scope == MediaTrashScope(query="doc", media_type="pdf", page=2)
    assert scope.offset == 20


def test_media_trash_result_rejects_duplicate_or_noncanonical_ids():
    payload = {
        "items": [
            {"id": "local:media:1", "media_id": 1, "title": "A", "type": "pdf", "trash_date": None},
            {"id": "local:media:2", "media_id": 2, "title": "B", "type": "pdf", "trash_date": None},
        ],
        "total": 2,
        "limit": 20,
        "offset": 0,
        "types": ["pdf"],
    }
    payload["items"][1]["id"] = payload["items"][0]["id"]
    with pytest.raises(ValueError, match="unique"):
        build_media_trash_result(MediaTrashScope(), payload)
```

Cover bool/non-integer/overflow page inputs, NUL/201-character queries, exact five envelope keys, exact five item keys, positive non-bool backing IDs, canonical stable IDs, unique IDs, strict ISO-or-None `trash_date`, exact cardinality, sorted unique nonblank facets, and out-of-range detection.

- [ ] **Step 2: Write RED pure transition tests.**

Prove entry success selects the first row only while entry authority owns focus; page/search/type success leaves selection empty; selection cannot survive scope/page change; failed filter/page retains the prior fresh applied page with exact fixed copy and Retry target; initial failure has no rows; one clamp is allowed; a second shrink becomes stale; confirmation captures full immutable identity; pre-commit failure keeps row/selection; committed mutation removes the target and becomes stale/loading before refresh.

- [ ] **Step 3: Write RED controller generation and request-shape tests.**

```python
@pytest.mark.asyncio
async def test_controller_sends_exact_local_trash_scope_and_rejects_late_result():
    controller, screen, service = _controller(_page(2, total=21), _page(1, total=1))
    controller.request(MediaTrashScope(page=2), origin="next", focus_identity="#library-media-trash-next")
    old = screen.pending.pop()
    controller.request(MediaTrashScope(query="new"), origin="search", focus_identity="#library-media-trash-search")
    new = screen.pending.pop()
    await new
    await old
    assert controller.state.applied_result.scope == MediaTrashScope(query="new")
```

Also prove Retry repeats the failed target, clamp occurs once, Back/unmount invalidation fences late local thread completion, metadata-only logging excludes sentinels, and worker group is Trash-specific.

- [ ] **Step 4: Run RED.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Library/test_library_media_trash_state.py \
  Tests/UI/test_library_media_trash_browse_controller.py
```

Expected RED: paging types/controller do not exist and legacy state silently drops malformed rows.

- [ ] **Step 5: Implement exact immutable contracts and pure reducers.**

Reuse `PageFreshness` and `build_library_pager_display`; do not copy range/page/button arithmetic. Freeze item mappings with `MappingProxyType`. Keep requested scope separate from `applied_result.scope`. Represent failed filter/page requests with the prior applied result still fresh; represent committed mutation/shrink as stale with `total=None` at pager derivation.

- [ ] **Step 6: Implement the controller's one-generation/one-clamp loop.**

Call only:

```python
payload = await run_service_call(
    service.list_library_media_trash,
    mode="local",
    query=scope.query,
    media_type=scope.media_type,
    limit=scope.page_size,
    offset=scope.offset,
    isolate_in_worker=True,
)
result = build_media_trash_result(scope, payload)
```

The apply gate must check controller generation plus `request_is_active()` before every state/focus sync.

- [ ] **Step 7: Run GREEN and required inverses.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Library/test_library_media_trash_state.py \
  Tests/UI/test_library_media_trash_browse_controller.py
```

Temporarily remove duplicate-ID validation and allow a second clamp/read; require their focused tests to fail, restoring each change immediately.

- [ ] **Step 8: Commit the state/controller slice.**

```bash
git add tldw_chatbook/Library/library_media_state.py \
  tldw_chatbook/UI/Library_Modules/library_media_trash_browse_controller.py \
  Tests/Library/test_library_media_trash_state.py \
  Tests/UI/test_library_media_trash_browse_controller.py
git commit -m "feat: own Media Trash page state"
```

---

### Task 4: Wire Trash lifecycle, filters, pager, and Back receipt

**Files:**

- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_library_media_trash.py`
- Modify: `Tests/UI/test_library_shell.py` only for direct lifecycle/return integration coverage.

**Interfaces:**

- Consumes: `LibraryMediaTrashBrowseController` from Task 3 and existing guarded `_apply_library_media_list_return(...)` path.
- Produces one screen-owned, Viewer-independent receipt:

```python
@dataclass(frozen=True)
class _LibraryMediaTrashReturn:
    stable_id: str
    scroll_offset: tuple[int, int] | None
    focus_identity: str
```

The screen constructs the Trash controller with `request_is_active` requiring both the Media route and `_library_media_view == "trash"`.

- [ ] **Step 1: Write RED mounted lifecycle tests.**

Cover initial entry unfiltered page 1, no inheritance from normal Media query/type, one request only, loading without fabricated counts, generation rejection after newer search/page intent, Back/unmount fencing, and initial failure focusing Retry. Use real keyboard input for focus-driven behavior, not programmatic `.focus()` alone.

- [ ] **Step 2: Write RED search/type/page/Retry tests.**

Drive `Input.Submitted`, the bounded type chooser, Previous, Next, and Retry. Assert exact requested/applied separation and fixed copies:

```python
assert status.plain == "Filter not applied — showing All Trash · Retry"
assert controller.state.applied_result.scope == MediaTrashScope()
assert controller.state.requested_scope == MediaTrashScope(query="failed")
```

Prove page/filter intent clears selection before dispatch, a failed ordinary page keeps old range/page fresh, Retry repeats the failed target, and pager actions after failure operate on the visible applied scope.

Submitting 201 characters must leave the prior applied page visible and show exactly `Search is limited to 200 characters.` without dispatching a service request.

- [ ] **Step 3: Write RED Back-receipt tests.**

Enter Trash from normal Media page 2 with a selected stable row, nonzero list scroll, and toolbar focus. After independent Trash paging/filtering, Back and Escape must restore the original Media scope/page/row/scroll/focus. Assert `_library_media_viewer_return` remains untouched and a late Trash completion cannot change normal Media.

- [ ] **Step 4: Run RED.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_trash.py \
  Tests/UI/test_library_shell.py \
  -k 'media_trash and (entry or page or filter or type or retry or generation or back or escape or return or unmount)'
```

- [ ] **Step 5: Replace ad-hoc Trash fields/load worker with the controller.**

Remove `_library_media_trash_records`, `_library_media_trash_total`, and `_library_media_trash_error` as page authority. Keep only the screen-owned controller, return receipt, UI draft, and semantic focus intent. `handle_library_media_trash_open` captures the receipt before switching views, resets Trash-only draft/confirmation state, and calls:

```python
self._library_media_trash_browse_controller.request(
    MediaTrashScope(), origin="entry", focus_identity="#library-media-trash-row-0"
)
```

- [ ] **Step 6: Add filter/pager/Retry handlers and guarded exit.**

Every handler stops its event, checks the shared mutation interlock, then delegates one intent to the controller. Back invalidates the controller before changing `_library_media_view`, clears Trash drafts/confirmation, and schedules the existing guarded Media list-return application with the distinct receipt.

- [ ] **Step 7: Run GREEN and the generation inverse.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_trash.py \
  Tests/UI/test_library_shell.py \
  -k 'media_trash and (entry or page or filter or type or retry or generation or back or escape or return or unmount)'
```

Temporarily omit `controller.invalidate()` from Back; require the late-result test to fail, then restore.

- [ ] **Step 8: Commit the screen lifecycle slice.**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_media_trash.py \
  Tests/UI/test_library_shell.py
git commit -m "feat: navigate exact Media Trash pages"
```

---

### Task 5: Render bounded filters, rows, pager, and deterministic focus

**Files:**

- Modify: `tldw_chatbook/Widgets/Library/library_media_trash_canvas.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_library_media_trash.py`
- Modify: `Tests/UI/test_library_media_side_by_side.py` for production hierarchy/collapse regression coverage.
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` only if measured geometry requires it.
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss` only if component CSS changes.

**Interfaces:**

- Consumes: `LibraryMediaTrashState`, controller `pager`, complete `types`, loading/error/stale/notice copy, applied scope label, and semantic focus identity.
- Produces stable control IDs:

```text
#library-media-trash-back
#library-media-trash-search
#library-media-trash-type-filter
#library-media-trash-type-choices
#library-media-trash-status
#library-media-trash-list
#library-media-trash-previous
#library-media-trash-page
#library-media-trash-range
#library-media-trash-next
#library-media-trash-retry
#library-media-trash-restore
#library-media-trash-delete
```

- [ ] **Step 1: Write RED semantic rendering tests.**

Mount the production `LibraryScreen` hierarchy and require:

- `Local Trash · N items` unfiltered and `Local Trash · N matching` filtered only while fresh;
- the applied query/type label remains the previous scope during a failed/draft filter;
- exact `1-20 of 45`, `Page 1 of 3`, Previous/Next reasons, and unique Retry;
- initial/loading/stale/error states never show a fabricated count/range/page;
- disabled Restore/Delete reasons are `Trash is refreshing.`, `Refresh Trash before changing this item.`, or `Select a Trash item first.` as appropriate;
- only 20 two-line row buttons mount, with no horizontal scrolling;
- a 60-type fixture mounts one bounded chooser rather than 60 Buttons, and its final option is keyboard reachable.
- the chooser always includes `All types`, marks the applied type with `✓`, and Escape closes it without changing requested/applied scope.

- [ ] **Step 2: Write RED focus-precedence tests with user inputs.**

Drive actual `Tab`, arrow keys, `Enter`, `Escape`, and submitted input. Assert:

```python
assert app.focused.id == "library-media-trash-search"          # search success
assert app.focused.id == "library-media-trash-type-filter"     # type success
assert app.focused.id in {"library-media-trash-next", "library-media-trash-previous", "library-media-trash-back"}
```

Cover initial success/initial error, type chooser Escape, failed request Retry, page completion fallback when Next disappears, empty-state Back focus, background completion not stealing newer focus, and opener key consumption.

- [ ] **Step 3: Write RED four-size compositor/geometry tests.**

Parametrize exactly `(160, 50)`, `(120, 35)`, `(100, 30)`, and `(80, 24)`. For ordinary, initial-error, stale, and confirmation-ready states assert:

- vertical order is header, filters, status, `1fr` list, pager, action row;
- at 80×24 the list's painted region is at least four rows;
- pager/actions are inside the Items pane and painted;
- status paints no more than two rows plus its fold indicator;
- opening/closing Library and Items panes survives a page/filter refresh;
- compositor `get_widget_at`/rendered strips confirm reachability, not merely non-empty widget `region` values.

- [ ] **Step 4: Run RED.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_trash.py \
  Tests/UI/test_library_media_side_by_side.py \
  -k 'media_trash and (render or pager or filter or type or focus or geometry or compact or collapse or disabled)'
```

- [ ] **Step 5: Implement declarative canvas rendering.**

Pass controller-owned presentation into both constructor and `sync_state`. Keep the list as the only `1fr` child with `min_height = 0` and independent vertical scroll. Put the pager and actions after it, outside that scroll. Replace the type row with a bounded chooser while open; do not add another vertical row.

Use `library_disabled_action_label(...)` and tooltips for non-colour disabled reasons. The canvas renders state only; it must not compute filters, mutate controller state, or infer focus ownership.

- [ ] **Step 6: Implement focus restoration through current mounted identities.**

After every recompose, reacquire the semantic target by selector, yield one compositor cycle, and reacquire immediately before focusing. Never retain and press a control captured before a request-driven recompose.

- [ ] **Step 7: Change CSS only if the real hierarchy proves it necessary.**

If component TCSS changes, regenerate rather than hand-edit the bundle:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  tldw_chatbook/css/build_css.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  tldw_chatbook/css/check_bundle_sync.py
```

- [ ] **Step 8: Run GREEN plus two layout inverses.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_trash.py \
  Tests/UI/test_library_media_side_by_side.py \
  -k 'media_trash and (render or pager or filter or type or focus or geometry or compact or collapse or disabled)'
```

Temporarily derive facets from visible rows and remove the list's `1fr/min-height:0` containment; require the final-facet and 80×24 paint tests to fail, restoring each immediately.

- [ ] **Step 9: Commit the renderer slice.**

```bash
git add tldw_chatbook/Widgets/Library/library_media_trash_canvas.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_media_trash.py \
  Tests/UI/test_library_media_side_by_side.py
# Stage component and generated TCSS only if both changed and bundle parity passed.
git commit -m "feat: render resilient Media Trash pages"
```

---

### Task 6: Reconcile Restore and permanent deletion truthfully

**Files:**

- Modify: `tldw_chatbook/UI/Library_Modules/library_media_trash_browse_controller.py`
- Modify: `tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `tldw_chatbook/Widgets/Library/library_media_trash_canvas.py`
- Modify: `Tests/UI/test_library_media_trash_browse_controller.py`
- Modify: `Tests/UI/test_library_media_trash.py`
- Modify: `Tests/UI/test_library_media_browse_controller.py`
- Modify: `Tests/Media/test_local_media_reading_service.py` to pin the existing permanent-delete physical cascade, FTS cleanup, and no-sync behavior.

**Interfaces:**

- Consumes: shared `_library_media_bulk_delete_in_flight` flag and `library_media_bulk_delete` worker group; existing Restore/permanent-delete scope-service methods.
- Produces one normal-Media method:

```python
def mark_stale_after_trash_restore(self) -> None:
    """Retain the current exact Media page but withdraw its authority."""
```

- Extends `_complete_library_media_mutation(...)` with explicit Trash behavior while preserving defaults for every existing delete/Undo/edit caller:

```python
def _complete_library_media_mutation(
    self,
    *,
    committed: bool = False,
    remove_ids: tuple[str, ...] = (),
    upsert_items: tuple[Mapping[str, Any], ...] = (),
    refresh_normal_media: bool = True,
    stale_normal_media: bool = False,
) -> None: ...
```

- [ ] **Step 1: Write RED safe-confirmation tests.**

Use duplicate truncated titles and one very long title. Require the confirmation to show the full title in a bounded scrollable detail region plus type and Trash timestamp; missing values render exactly `Unknown type` and `Unknown deletion time`. Authorize by captured stable ID, focus Cancel initially, consume the opener activation, and commit only after a later explicit confirmation. Escape/Cancel returns opener focus without calling a service.

- [ ] **Step 2: Write RED commit-interlock and service-seam tests.**

Prove Back/Escape are disabled with `Finishing this action…` only while commit status is unknown, double press schedules one worker, and post-commit refresh can be abandoned. Assert permanent deletion calls exactly:

```python
await scope_service.permanently_delete_media_item(
    mode="local", media_id=target.backing_media_id
)
```

The test must fail if `empty_media_trash`, the backing lifecycle path, a canonical stable ID, or any second delete seam is used.

- [ ] **Step 3: Write RED Restore reconciliation tests.**

After a committed Restore require:

- the Trash target disappears immediately;
- Trash becomes stale/loading and hides exact count/range/page claims;
- normal Media keeps exactly the same applied scope/page/records/selection/focus/scroll;
- normal Media becomes stale and exposes its own Retry;
- no restored summary is inserted/repositioned;
- a successful Trash refresh does not clear normal Media stale;
- a failed Trash refresh shows `Restored '<bounded title>'. List may be out of date · Retry` and never reports Restore as failed.
- authoritative Trash refresh restores focus to the target's prior page-local position, then the previous row, then Back, only while mutation focus authority is current.

- [ ] **Step 4: Write RED permanent-delete and pre-commit failure tests.**

Committed deletion uses the same Trash stale/refresh flow but leaves normal Media freshness unchanged. Not-found/not-in-Trash/policy/service failures keep the row, selection, total, and fresh boundary while showing recoverable action copy. Extend the local service test to prove existing cascade/FTS cleanup and absence of a sync-log insert if that behavior is not already directly pinned.

- [ ] **Step 5: Run RED.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_trash_browse_controller.py \
  Tests/UI/test_library_media_trash.py \
  Tests/UI/test_library_media_browse_controller.py \
  Tests/Media/test_local_media_reading_service.py \
  -k 'media_trash and (restore or permanent or confirm or mutation or stale or refresh or back or escape)'
```

- [ ] **Step 6: Implement confirmation and mutation claims through immutable targets.**

`claim_mutation()` validates the selected fresh stable ID, captures the backing ID/full identity/page index, sets `mutation_pending=True`, and closes confirmation only for the captured target. Screen handlers claim the shared flag synchronously before scheduling the existing exclusive worker group.

- [ ] **Step 7: Implement truthful completion paths.**

For pre-commit failure call `finish_mutation_failure(...)`, release the shared interlock, retain the row/selection, and do not refresh. For commit:

```python
controller.finish_mutation_commit(target, notice)
controller.request_after_mutation(focus_identity=target_focus_fallback)
```

Restore calls `_complete_library_media_mutation(committed=True, refresh_normal_media=False, stale_normal_media=True)` with no upsert. Permanent deletion calls the same helper with both flags false. All other existing Media mutations keep default behavior.

- [ ] **Step 8: Run GREEN and three safety inverses.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_trash_browse_controller.py \
  Tests/UI/test_library_media_trash.py \
  Tests/UI/test_library_media_browse_controller.py \
  Tests/Media/test_local_media_reading_service.py \
  -k 'media_trash and (restore or permanent or confirm or mutation or stale or refresh or back or escape)'
```

One at a time, focus Confirm initially, let opener Enter fall through, and upsert the restored row into normal Media. Each corresponding safety test must turn RED; restore the approved implementation after each inverse.

- [ ] **Step 9: Commit the mutation slice.**

```bash
git add tldw_chatbook/UI/Library_Modules/library_media_trash_browse_controller.py \
  tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/Widgets/Library/library_media_trash_canvas.py \
  Tests/UI/test_library_media_trash_browse_controller.py \
  Tests/UI/test_library_media_trash.py \
  Tests/UI/test_library_media_browse_controller.py \
  Tests/Media/test_local_media_reading_service.py
git commit -m "feat: safely recover and purge Media Trash"
```

---

### Task 7: Run focused production-shaped verification and close out TASK-18918

**Files:**

- Create: `Tests/Live/test_library_media_trash_paging_closeout.py` for the isolated real-database, four-size walkthrough.
- Modify: `Docs/User_Guide/library/media-and-conversations.md`
- Modify: `backlog/tasks/task-18918 - Add-paged-recovery-viewing-to-Library-Media-Trash.md`
- Modify: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md` only if implementation produces a genuinely reusable incident-backed lesson.

**Interfaces:**

- Consumes: all Tasks 1–6 plus the production `TldwCli`/Library reader shell.
- Produces: focused automated evidence, four-size live evidence, updated user docs, checked acceptance criteria, implementation notes, and task status Done.

- [ ] **Step 1: Run the pure/database/service/controller gate.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/DB/test_client_media_trash_pagination.py \
  Tests/Media/test_local_media_reading_service.py \
  Tests/Media/test_media_reading_scope_service.py \
  Tests/Library/test_library_media_trash_state.py \
  Tests/UI/test_library_media_trash_browse_controller.py \
  Tests/UI/test_library_media_browse_controller.py \
  -k 'media_trash or library_media_trash or permanently_delete_media_item or mark_stale_after_trash_restore'
```

- [ ] **Step 2: Run the mounted owner and production-shaped cross-reader gate.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_media_trash.py \
  Tests/UI/test_library_media_side_by_side.py \
  Tests/UI/test_library_adaptive_reader_shell.py \
  Tests/UI/test_library_adaptive_reader_closeout.py \
  -k 'media_trash or (media and (reader or collapse or compact or geometry or return))'
```

Do not replace this with `pytest` over the repository. If a surprising failure appears, reproduce it in a detached pristine-base worktree before labeling it baseline.

- [ ] **Step 3: Run Ruff and structural checks only on changed files.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/DB/Client_Media_DB_v2.py \
  tldw_chatbook/Media/local_media_reading_service.py \
  tldw_chatbook/Media/media_reading_scope_service.py \
  tldw_chatbook/Library/library_media_state.py \
  tldw_chatbook/UI/Library_Modules/library_media_trash_browse_controller.py \
  tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py \
  tldw_chatbook/UI/Screens/library_screen.py \
  tldw_chatbook/Widgets/Library/library_media_trash_canvas.py \
  Tests/DB/test_client_media_trash_pagination.py \
  Tests/Media/test_local_media_reading_service.py \
  Tests/Media/test_media_reading_scope_service.py \
  Tests/Library/test_library_media_trash_state.py \
  Tests/UI/test_library_media_trash_browse_controller.py \
  Tests/UI/test_library_media_trash.py \
  Tests/UI/test_library_media_browse_controller.py \
  Tests/UI/test_library_media_side_by_side.py \
  Tests/UI/test_library_adaptive_reader_shell.py \
  Tests/UI/test_library_adaptive_reader_closeout.py \
  Tests/Live/test_library_media_trash_paging_closeout.py
git diff --check
```

If TCSS changed, rerun the bundle build/parity commands from Task 5 and require a clean diff on a second build.

- [ ] **Step 4: Run the isolated live walkthrough at all four exact sizes.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Live/test_library_media_trash_paging_closeout.py
```

Use `mktemp -d`, a scratch `TLDW_CONFIG_PATH`, `PYTHONPATH` pointing at this worktree, the real `MediaDatabase`, `LocalMediaReadingService`, `MediaReadingScopeService`, and production `TldwCli`. Seed more than 40 Trash rows with multiple/padded/case-distinct types, long/duplicate titles, equal/null timestamps, and active decoys. For each of 160×50, 120×35, 100×30, and 80×24 verify:

1. initial unfiltered page, middle/final page, and exact Local Trash totals;
2. title/type filtering, failed filter/page copy, Retry, and one concurrent-shrink clamp;
3. Library and Items collapse/expand, followed by a refresh-producing action that must preserve the pane choice;
4. full-title permanent-delete confirmation, Cancel, explicit confirm, and committed-refresh failure;
5. Restore makes normal Media stale without inserting a row; Back restores page/selection/focus/scroll;
6. all relevant controls are freshly mounted, displayed, painted, and keyboard reachable.

Wait separately for authoritative state, current DOM identity, and compositor paint after every recompose. Re-query controls immediately before use. Capture compositor-visible evidence; do not rely on terminal capture or widget regions alone.

- [ ] **Step 5: Verify privacy and real-profile isolation.**

Assert the scratch run's logs omit unique query/title/ID/path/credential/delete-target sentinels. Confirm the real profile and Media database are byte-identical before/after and all workers/processes/handles drain.

- [ ] **Step 6: Re-run the required inverse matrix.**

One at a time and restoring immediately: split the DB snapshot, remove the stable order tie-break, allow server mode, remove duplicate-ID validation, allow a second clamp, omit Back invalidation, derive facets from the page, enable stale mutations, focus Confirm first, allow opener fallthrough, and upsert the restored row into normal Media. Every inverse must make its named focused test fail.

- [ ] **Step 7: Request independent code/spec review and resolve findings.**

Use `superpowers:requesting-code-review`. Treat comments through `superpowers:receiving-code-review`: verify each claim against current code/tests before changing anything. Resolve every Critical/Important finding with focused RED/GREEN evidence.

- [ ] **Step 8: Update documentation and task completion metadata.**

Revise the Media Trash guide from the old capped `showing X of N`/Restore-only contract to exact pages, local title/type filters, permanent-delete confirmation, Retry/stale behavior, and Back context restoration. In the task file:

- check all six acceptance criteria;
- add concise `## Implementation Notes` naming the DB/service/state/controller/canvas changes and automated/live evidence;
- retain the ADR-067/no-new-ADR record;
- set status to Done by direct file edit because the Backlog CLI is unsafe for five-digit task IDs.

- [ ] **Step 9: Run closeout checks and commit.**

```bash
git diff --check
git status --short
git log --oneline origin/dev..HEAD
```

```bash
git add Docs/User_Guide/library/media-and-conversations.md \
  Tests/Live/test_library_media_trash_paging_closeout.py \
  "backlog/tasks/task-18918 - Add-paged-recovery-viewing-to-Library-Media-Trash.md"
# Also stage a new incident-backed lesson only when implementation produced one.
git commit -m "docs: close Media Trash paging task"
```

---

## Completion criteria

TASK-18918 is complete only when every acceptance criterion is checked, exact local pages and complete filters are reachable, Restore/permanent deletion remain truthful through failed refreshes, stale and malformed states fail closed, all four terminal sizes pass production-shaped keyboard/compositor walkthroughs, the focused owner and cross-reader suites plus Ruff/diff/bundle checks are green, every required inverse has turned its focused test RED, independent review has no unresolved Critical/Important findings, user documentation is current, and the Backlog task is Done. A repository-wide pytest run remains excluded unless the user explicitly opts in.
