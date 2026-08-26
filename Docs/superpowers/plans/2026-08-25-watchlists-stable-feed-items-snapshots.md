# Watchlists Stable Feed Items Snapshots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the Watchlists Read screen a bounded, stable Feed Items snapshot whose scope, rows, pagination, Reader clearing, counts, and new-arrival affordance publish honestly and atomically.

**Architecture:** Add a Reader-specific page/cursor contract over the existing normalized SQLite `effective_date`, leaving the public agent-search cursor contract and legacy non-Reader `list_items()` callers unchanged. The UI owns a small snapshot state object containing cached pages, the initial item-id high-water mark, and a seen-id guard; query-context replacements load off-screen and commit only after row presentation succeeds. Navigation gestures on Read use a pending/committed scope split, while management tabs retain their existing immediate local scope commit and invalidate the parked Reader snapshot.

**Tech Stack:** Python 3.11+, SQLite/FTS5, asyncio, Textual 8.x, pytest/pytest-asyncio, Ruff, Backlog.md

---

## Scope and decisions

- **Backlog task:** `TASK-22450`.
- **Prerequisite for:** `TASK-22451` (contextual aggregate feed selection).
- **ADR required:** no.
- **ADR path:** `backlog/decisions/042-watchlists-reader-first-ia.md`.
- **Reason:** ADR-042 already fixes the long-lived Reader snapshot, pending/committed scope, high-water, keyset, seen-id, and arrival semantics. This PR implements that existing decision without adding a schema, storage-ownership, service-boundary, or application-wide layout decision.
- **Reader ordering:** `effective_date DESC, id DESC`, exactly as the approved Reader design specifies. `search_items_for_agent()` intentionally remains `effective_date DESC, id ASC`; its serialized cursor is an established external tool contract and is not changed in this PR. Only reusable predicate/search helpers may be shared.
- **Compatibility:** keep `SubscriptionsDB.get_new_items()`, `LocalWatchlistsService.list_items()`, `WatchlistScopeService.list_items()`, and `WatchlistsBackendController.list_items()` for Runs and other existing callers. Add a separate Reader page seam instead of changing their return type.
- **Snapshot bound:** the first Reader page captures the maximum item id matching the query at that moment. Every later page includes `id <= snapshot_max_item_id`; the arrival count uses the same query dimensions with `id > snapshot_max_item_id`.
- **Practical stability:** already mounted pages are cached and never silently re-fetched. A seen-id set removes duplicates from later pages. An unseen pre-existing row whose effective date changes may move or wait for explicit refresh, matching ADR-042's bounded guarantee.
- **Backward paging:** Previous presents the cached prior page and performs no database query. This avoids reverse-cursor complexity and guarantees the user sees the exact page they already read.
- **Explicit refresh:** Refresh creates a replacement snapshot from page 1 and a new high-water mark. A failed refresh keeps the old snapshot, Reader, count, and new-items affordance.
- **Query changes:** status and debounced search changes create replacement snapshots. Their old rows remain mounted while pending; successful publication resets pagination but preserves the current Reader item when the scope itself did not change.
- **Scope changes:** a Read scope gesture loads a replacement first page while the committed highlight, heading, rows, and Reader stay active. Successful presentation commits scope/highlight/rows together and clears the Reader. Failure or supersession keeps the previous committed view.
- **Management tabs:** local management scope gestures still commit immediately and invalidate parked Feed Items/Reader state without issuing a hidden item query. Returning to Read loads an honest first snapshot for that committed scope. Server Read remains unsupported and unchanged.
- **Aggregate feed children:** explicitly out of scope for this PR; TASK-22451 adds them after this foundation lands.
- **Verification boundary:** run only the changed Watchlists/Subscriptions tests named in this plan, modified-file Ruff, and diff checks. Do not run the repository-wide test suite.

## File map

### New files

- `tldw_chatbook/Subscriptions/watchlist_item_page.py` — immutable cross-layer Reader cursor/page value objects; no UI policy.
- `tldw_chatbook/UI/Watchlists_Modules/reader_item_snapshot.py` — screen-side cached-page and seen-id state; no I/O or Textual widget access.
- `Tests/DB/test_subscriptions_db_watchlists_reader_snapshot.py` — real SQLite proof of Reader keyset, high-water, counts, search, predicates, and arrival semantics.
- `Tests/Watchlists/test_reader_item_snapshot.py` — pure cached-page/seen-id state tests.

### Modified production files

- `tldw_chatbook/DB/Subscriptions_DB.py` — Reader page/count queries over the generated `effective_date`; retain legacy and agent APIs.
- `tldw_chatbook/Subscriptions/watchlist_normalizers.py` — carry `effective_date` into normalized Reader rows.
- `tldw_chatbook/Subscriptions/local_watchlists_service.py` — off-loop local Reader page and arrival-count methods.
- `tldw_chatbook/Subscriptions/watchlist_scope_service.py` — local-only Reader page/count routing and policy enforcement.
- `tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py` — preserve typed Reader page values across the controller seam.
- `tldw_chatbook/UI/Watchlists_Modules/article_list.py` — display screen-owned snapshot count/arrival state and stop dismissing arrivals before refresh succeeds.
- `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` — own pending/committed Reader query state, cached keyset pages, atomic scope publication, and arrival reconciliation.

### Modified tests

- `Tests/Subscriptions/test_watchlist_normalizers.py` — normalized `effective_date` contract.
- `Tests/Subscriptions/test_local_watchlists_service.py` — Reader page normalization and off-loop forwarding.
- `Tests/Subscriptions/test_watchlist_scope_service.py` — Reader page/count local routing and server refusal.
- `Tests/Watchlists/test_watchlists_backend_controller.py` — typed page/count passthrough.
- `Tests/Watchlists/test_watchlists_article_list.py` — snapshot count, persistent new-items pill, and refresh request behavior.
- `Tests/Watchlists/test_watchlists_pagination.py` — replace offset assumptions with cached keyset snapshot, refresh, query replacement, and transactional publication regressions.
- `Tests/Watchlists/test_watchlists_collections_screen.py` — mounted atomic scope, arrival-count, and explicit-refresh behavior.
- `Tests/Watchlists/test_watchlists_scoped_rebuilds.py` — management-tab immediate scope commit and parked Reader invalidation remain rebuild-safe.

### Documentation/task files

- `backlog/tasks/task-22450 - Stabilize-Watchlists-Feed-Items-snapshots-and-atomic-scope-commits.md` — plan, completed acceptance criteria, verification evidence, and implementation notes.
- `Docs/superpowers/plans/2026-08-25-watchlists-stable-feed-items-snapshots.md` — this execution checklist and final evidence.

---

### Task 1: Define the Reader page and cached snapshot contracts

**Files:**
- Create: `tldw_chatbook/Subscriptions/watchlist_item_page.py`
- Create: `tldw_chatbook/UI/Watchlists_Modules/reader_item_snapshot.py`
- Create: `Tests/Watchlists/test_reader_item_snapshot.py`

- [x] **Step 1: Write failing tests for first-page construction, cached backward pages, watermark consistency, and duplicate suppression**

```python
from tldw_chatbook.Subscriptions.watchlist_item_page import (
    WatchlistItemCursor,
    WatchlistItemPage,
)
from tldw_chatbook.UI.Watchlists_Modules.reader_item_snapshot import (
    ReaderItemSnapshot,
    ReaderItemQuery,
)


def _page(*ids: int, high_water: int, has_more: bool) -> WatchlistItemPage:
    return WatchlistItemPage(
        items=tuple(
            {
                "id": f"local:watchlist_item:{item_id}",
                "item_id": item_id,
                "effective_date": f"2026-08-{item_id:02d} 12:00:00",
            }
            for item_id in ids
        ),
        has_more=has_more,
        snapshot_max_item_id=high_water,
        snapshot_count=len(ids),
        next_cursor=(
            WatchlistItemCursor(
                effective_date=f"2026-08-{ids[-1]:02d} 12:00:00",
                item_id=ids[-1],
            )
            if has_more
            else None
        ),
    )


def test_append_page_drops_seen_ids_and_preserves_cached_pages() -> None:
    query = ReaderItemQuery.freeze(
        ("local", "all", "all", ""),
        {"statuses": ["new", "reviewed", "ingested"]},
    )
    snapshot = ReaderItemSnapshot.start(query, _page(5, 4, high_water=5, has_more=True))

    candidate, appended = snapshot.with_continuation(
        _page(4, 3, high_water=5, has_more=False)
    )

    assert [row["item_id"] for row in snapshot.page(0)] == [5, 4]
    assert snapshot.page_count == 1, "committed state is unchanged until publication"
    assert appended is True
    assert [row["item_id"] for row in candidate.page(1)] == [3]
    assert candidate.page_count == 2
    assert candidate.has_next(1) is False


def test_append_rejects_a_different_snapshot_watermark() -> None:
    query = ReaderItemQuery.freeze(("local", "all", "all", ""), {})
    snapshot = ReaderItemSnapshot.start(query, _page(5, high_water=5, has_more=True))

    with pytest.raises(ValueError, match="watermark"):
        snapshot.with_continuation(_page(4, high_water=6, has_more=False))


def test_duplicate_only_continuation_advances_cursor_without_caching_a_blank_page() -> None:
    query = ReaderItemQuery.freeze(("local", "all", "all", ""), {})
    snapshot = ReaderItemSnapshot.start(query, _page(5, high_water=5, has_more=True))

    candidate, appended = snapshot.with_continuation(
        _page(5, high_water=5, has_more=True)
    )

    assert appended is False
    assert snapshot.page_count == 1
    assert candidate.page_count == 1
    assert candidate.cursor_after_last_page.item_id == 5
    assert candidate.has_more is True
```

- [x] **Step 2: Run the pure contract tests and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Watchlists/test_reader_item_snapshot.py --tb=short
```

Expected: collection fails because `watchlist_item_page` and `reader_item_snapshot` do not exist.

- [x] **Step 3: Add immutable transport values and the minimal cached snapshot state**

```python
# tldw_chatbook/Subscriptions/watchlist_item_page.py
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WatchlistItemCursor:
    effective_date: str | None
    item_id: int


@dataclass(frozen=True)
class WatchlistItemPage:
    items: tuple[dict[str, Any], ...]
    has_more: bool
    snapshot_max_item_id: int
    snapshot_count: int | None
    next_cursor: WatchlistItemCursor | None
```

```python
# tldw_chatbook/UI/Watchlists_Modules/reader_item_snapshot.py
from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping

from ...Subscriptions.watchlist_item_page import (
    WatchlistItemCursor,
    WatchlistItemPage,
)


def _item_key(item: dict[str, Any]) -> str:
    return str(item.get("item_id") or item.get("id") or "")


@dataclass(frozen=True)
class ReaderItemQuery:
    context_key: tuple[Any, ...]
    frozen_kwargs: tuple[tuple[str, Hashable], ...]

    @classmethod
    def freeze(
        cls,
        context_key: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> "ReaderItemQuery":
        # Normalize mutable status lists before this becomes committed state.
        frozen = tuple(
            sorted(
                (key, tuple(value) if isinstance(value, list) else value)
                for key, value in kwargs.items()
            )
        )
        return cls(context_key=context_key, frozen_kwargs=frozen)

    def as_kwargs(self) -> dict[str, Any]:
        return {
            key: list(value) if key == "statuses" and isinstance(value, tuple) else value
            for key, value in self.frozen_kwargs
        }


@dataclass
class ReaderItemSnapshot:
    query: ReaderItemQuery
    snapshot_max_item_id: int
    snapshot_count: int
    pages: list[tuple[dict[str, Any], ...]] = field(default_factory=list)
    seen_item_ids: set[str] = field(default_factory=set)
    cursor_after_last_page: WatchlistItemCursor | None = None
    has_more: bool = False
    pending_arrivals: int = 0

    @classmethod
    def start(cls, query: ReaderItemQuery, page: WatchlistItemPage) -> "ReaderItemSnapshot":
        if page.snapshot_count is None:
            raise ValueError("The first Reader page must include its snapshot count")
        snapshot = cls(
            query=query,
            snapshot_max_item_id=page.snapshot_max_item_id,
            snapshot_count=page.snapshot_count,
        )
        snapshot._append_first(page)
        return snapshot

    def with_continuation(
        self,
        page: WatchlistItemPage,
    ) -> tuple["ReaderItemSnapshot", bool]:
        if page.snapshot_max_item_id != self.snapshot_max_item_id:
            raise ValueError("Reader page watermark does not match the snapshot")
        candidate = ReaderItemSnapshot(
            query=self.query,
            snapshot_max_item_id=self.snapshot_max_item_id,
            snapshot_count=self.snapshot_count,
            pages=list(self.pages),
            seen_item_ids=set(self.seen_item_ids),
            cursor_after_last_page=self.cursor_after_last_page,
            has_more=self.has_more,
            pending_arrivals=self.pending_arrivals,
        )
        unique = tuple(
            item
            for item in page.items
            if _item_key(item) and _item_key(item) not in candidate.seen_item_ids
        )
        candidate.seen_item_ids.update(_item_key(item) for item in unique)
        candidate.cursor_after_last_page = page.next_cursor
        candidate.has_more = page.has_more
        if unique:
            candidate.pages.append(unique)
        return candidate, bool(unique)
```

The private `_append_first()` mutates only the not-yet-published new snapshot, always creates visible page 0 even when it is empty, and initializes the traversal cursor/`has_more`. `with_continuation()` copy-stages all pages, seen ids, cursor, and `has_more`; it never mutates the committed snapshot. A continuation candidate advances traversal state but appends a visible cached page only when at least one unseen row remains. Add `page()`, `page_count`, and `has_next()` as direct bounds-checked accessors. Do not add persistence, reverse cursors, arbitrary page eviction, or a generalized pagination framework.

- [x] **Step 4: Run the pure contract tests and verify GREEN**

Run the Step 2 command.

Expected: all tests in `test_reader_item_snapshot.py` pass.

- [x] **Step 5: Commit the contract slice**

```bash
git add \
  tldw_chatbook/Subscriptions/watchlist_item_page.py \
  tldw_chatbook/UI/Watchlists_Modules/reader_item_snapshot.py \
  Tests/Watchlists/test_reader_item_snapshot.py
git commit -m "feat(watchlists): define stable reader snapshot state"
```

---

### Task 2: Add the SQLite Reader keyset and arrival-count queries

**Files:**
- Create: `Tests/DB/test_subscriptions_db_watchlists_reader_snapshot.py`
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py:2475-2825,3048-3200`
- Modify: `tldw_chatbook/Subscriptions/watchlist_normalizers.py:549-620`
- Modify: `Tests/Subscriptions/test_watchlist_normalizers.py`

- [x] **Step 1: Write real-SQLite RED tests for ordering, ties, NULL sink, high-water exclusion, and lookahead**

Use the existing DB fixtures from `Tests/DB/test_subscriptions_db_watchlists_agent_search.py`, but call the new Reader API and assert the Reader's descending id tie-break:

```python
def test_reader_keyset_traverses_ties_null_sink_and_later_inserts(db: SubscriptionsDB) -> None:
    source_id = _source(db, "Paging feed")
    newest = _item(db, source_id, "newest", published="2026-08-15T00:00:00Z")
    tied_older_id = _item(db, source_id, "tie-1", published="2026-08-14T00:00:00Z")
    tied_newer_id = _item(db, source_id, "tie-2", published="2026-08-14T00:00:00Z")
    null_older_id = _item(db, source_id, "null-1", published=None, created="not-a-date")
    null_newer_id = _item(db, source_id, "null-2", published=None, created="not-a-date")

    first = db.get_reader_items_page(status=None, statuses=READER_STATUSES, limit=2)
    later_insert = _item(db, source_id, "later", published="2040-01-01T00:00:00Z")
    second = db.get_reader_items_page(
        status=None,
        statuses=READER_STATUSES,
        limit=2,
        snapshot_max_item_id=first.snapshot_max_item_id,
        after=first.next_cursor,
    )
    third = db.get_reader_items_page(
        status=None,
        statuses=READER_STATUSES,
        limit=2,
        snapshot_max_item_id=first.snapshot_max_item_id,
        after=second.next_cursor,
    )

    assert [row["id"] for row in first.items] == [newest, tied_newer_id]
    assert [row["id"] for row in second.items] == [tied_older_id, null_newer_id]
    assert [row["id"] for row in third.items] == [null_older_id]
    assert later_insert not in {row["id"] for page in (first, second, third) for row in page.items}
```

Also add focused cases that prove:

- first-page `snapshot_max_item_id` is the maximum row matching the active source/watchlist/unassigned/status/star/search/since query, not an unread-count delta;
- `snapshot_count` counts only matching rows with `id <= snapshot_max_item_id`;
- the SQL trace contains no `OFFSET` for Reader pages;
- one lookahead row sets `has_more` but is not returned;
- deleting a previously mounted row does not break continuation;
- FTS and forced-LIKE fallback return the same Reader ordering and cursor;
- FTS and forced-LIKE fallback return the same matching high-water, frozen snapshot count, page rows/cursor, and arrival count—not merely the same visible row ids;
- `count_reader_item_arrivals()` counts only matching rows with ids above the snapshot watermark.
- SQLite page order/cursor boundaries and Python `Subscriptions.item_dates.effective_date()` order agree for aware-offset, naive, date-only, missing, and malformed publication/creation fixtures, including the descending id tie-break and NULL sink.

- [x] **Step 2: Add the normalizer RED assertion**

```python
def test_normalize_watchlist_item_carries_effective_date_for_reader_cursor() -> None:
    item = normalize_watchlist_item(
        "local",
        {"id": 7, "effective_date": "2026-08-25 12:00:00"},
    )

    assert item["effective_date"] == "2026-08-25 12:00:00"
```

- [x] **Step 3: Run the new DB and exact normalizer tests and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/DB/test_subscriptions_db_watchlists_reader_snapshot.py \
  Tests/Subscriptions/test_watchlist_normalizers.py::test_normalize_watchlist_item_carries_effective_date_for_reader_cursor \
  --tb=short
```

Expected: failures because `get_reader_items_page()`, `count_reader_item_arrivals()`, and normalized `effective_date` do not exist.

- [x] **Step 4: Extend the list projection and parameterize the private search ordering helper**

Add `i.effective_date` to `_LIST_ITEM_COLUMNS`. Extend `_search_items_rows()` with a fixed, caller-supplied `order_by` keyword whose default remains `i.effective_date DESC, i.id ASC`; interpolate only the method's internal constants, never user data. `get_new_items()` and `search_items_for_agent()` therefore keep their current behavior, while the Reader passes `i.effective_date DESC, i.id DESC`.

Factor a private helper that builds the full Reader predicates (`subscription_id`, `status`/`statuses`, `run_id`, `watchlist_id`, `unassigned_only`, `is_flagged`, `since`, and search). Reuse it for page, initial matching high-water/count, and arrival count so no dimension can drift.

- [x] **Step 5: Implement `get_reader_items_page()` with one transaction and no OFFSET**

Required signature:

```python
def get_reader_items_page(
    self,
    *,
    subscription_id: int | None = None,
    status: str | None = None,
    limit: int = 50,
    run_id: int | None = None,
    watchlist_id: int | None = None,
    unassigned_only: bool = False,
    statuses: Sequence[str] | None = None,
    is_flagged: bool | None = None,
    search: str | None = None,
    since: str | None = None,
    snapshot_max_item_id: int | None = None,
    after: WatchlistItemCursor | None = None,
) -> WatchlistItemPage:
    ...
```

Implementation rules:

```sql
-- non-NULL cursor continuation for DESC/DESC
AND (
    i.effective_date IS NULL
    OR i.effective_date < datetime(?)
    OR (i.effective_date = datetime(?) AND i.id < ?)
)

-- NULL-sink continuation
AND i.effective_date IS NULL AND i.id < ?

ORDER BY i.effective_date DESC, i.id DESC
LIMIT page_size_plus_one
```

On a first page, within the same `transaction()`:

1. compute the maximum matching item id (zero when empty);
2. count matching rows bounded by that id;
3. fetch `limit + 1` rows bounded by that id.

Return `snapshot_count` on that first page only. On continuation, reuse the supplied watermark and return `snapshot_count=None` rather than re-counting mutable status/search membership; reject a cursor without a positive item id and reject a watermark below the cursor id. Derive `next_cursor` from the last returned row only when `has_more` is true. Keep every value parameterized.

- [x] **Step 6: Implement exact matching arrival counts**

```python
def count_reader_item_arrivals(
    self,
    *,
    snapshot_max_item_id: int,
    # same query dimensions as get_reader_items_page, excluding cursor/limit
) -> int:
    """Count matching rows created after a mounted Reader snapshot."""
```

This query must use `i.id > ?` plus the same scope/status/star/search/since predicates. It must not use unread-count differences, because marking an existing item read is not a new arrival.

- [x] **Step 7: Export normalized `effective_date` and run GREEN tests**

Run the Step 3 command.

Expected: all new DB and normalizer tests pass.

- [x] **Step 8: Prove the established agent cursor remains unchanged**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/DB/test_subscriptions_db_watchlists_agent_search.py::test_keyset_traversal_handles_ties_null_sink_deletion_and_later_inserts \
  Tests/DB/test_subscriptions_db_watchlists_agent_search.py::test_one_lookahead_sets_has_more_and_is_not_returned \
  --tb=short
```

Expected: both existing ASC-tie agent tests pass unchanged.

- [x] **Step 9: Commit the database slice**

```bash
git add \
  tldw_chatbook/DB/Subscriptions_DB.py \
  tldw_chatbook/Subscriptions/watchlist_normalizers.py \
  Tests/DB/test_subscriptions_db_watchlists_reader_snapshot.py \
  Tests/Subscriptions/test_watchlist_normalizers.py
git commit -m "feat(watchlists): add reader keyset item pages"
```

---

### Task 3: Thread typed Reader pages through the local service boundary

**Files:**
- Modify: `tldw_chatbook/Subscriptions/local_watchlists_service.py:540-630`
- Modify: `tldw_chatbook/Subscriptions/watchlist_scope_service.py:237-330`
- Modify: `tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py:78-100`
- Modify: `Tests/Subscriptions/test_local_watchlists_service.py`
- Modify: `Tests/Subscriptions/test_watchlist_scope_service.py`
- Modify: `Tests/Watchlists/test_watchlist_scope_service.py`
- Modify: `Tests/Watchlists/test_watchlists_backend_controller.py`

- [x] **Step 1: Write RED tests for normalization and forwarding**

Add tests that assert:

```python
page = await service.list_reader_items_page(
    source_id="7",
    statuses=["new", "reviewed", "ingested"],
    snapshot_max_item_id=42,
    after=WatchlistItemCursor("2026-08-25 12:00:00", 21),
)

db.get_reader_items_page.assert_called_once_with(
    subscription_id=7,
    status=None,
    limit=50,
    run_id=None,
    watchlist_id=None,
    unassigned_only=False,
    statuses=["new", "reviewed", "ingested"],
    is_flagged=None,
    search=None,
    since=None,
    snapshot_max_item_id=42,
    after=WatchlistItemCursor("2026-08-25 12:00:00", 21),
)
assert page.items[0]["id"] == "local:watchlist_item:21"
assert page.next_cursor == WatchlistItemCursor("2026-08-25 12:00:00", 21)
```

Also prove:

- both local methods execute through `run_db_off_loop`;
- `WatchlistScopeService.list_reader_items_page()` and `.count_reader_item_arrivals()` enforce the existing `items.list` policy and reject Server;
- `WatchlistsBackendController` returns the `WatchlistItemPage` unchanged rather than coercing the dataclass through `list()`/`dict()`;
- the legacy `list_items()` route remains unchanged for Runs and existing callers.

- [x] **Step 2: Run the exact service/controller RED tests**

Run the new test nodes with `-k "reader_items_page or reader_item_arrivals"` in the four listed files.

Expected: failures because the Reader methods do not exist.

- [x] **Step 3: Add `LocalWatchlistsService.list_reader_items_page()` and `.count_reader_item_arrivals()`**

Both methods convert namespaced/bare ids exactly as `list_items()` does, call the DB off-loop, and rebuild only the page's `items` tuple through `normalize_watchlist_item("local", row)`. Copy `has_more`, watermark, count, and cursor exactly.

- [x] **Step 4: Add local-only scope service routing**

Mirror the existing `list_items()` policy path, but call the new local methods. The Server branch must raise the same honest local-only error before touching the server service. Keep explicit keyword signatures so future filter additions cannot disappear silently.

- [x] **Step 5: Add controller passthroughs without shape coercion**

```python
async def list_reader_items_page(
    self, *, runtime_backend: str | None = None, **kwargs: Any
) -> WatchlistItemPage:
    backend = self._normalize_backend(runtime_backend)
    return await self._maybe_await(
        self.scope_service.list_reader_items_page(
            runtime_backend=backend,
            **kwargs,
        )
    )
```

Add the analogous integer-returning arrival method. Do not modify `list_items()`.

- [x] **Step 6: Run the focused service/controller tests and verify GREEN**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Subscriptions/test_local_watchlists_service.py \
  Tests/Subscriptions/test_watchlist_scope_service.py \
  Tests/Watchlists/test_watchlist_scope_service.py \
  Tests/Watchlists/test_watchlists_backend_controller.py \
  -k "reader_items_page or reader_item_arrivals or list_items" --tb=short
```

Expected: selected Reader and legacy list routing tests pass.

- [x] **Step 7: Commit the service seam**

```bash
git add \
  tldw_chatbook/Subscriptions/local_watchlists_service.py \
  tldw_chatbook/Subscriptions/watchlist_scope_service.py \
  tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py \
  Tests/Subscriptions/test_local_watchlists_service.py \
  Tests/Subscriptions/test_watchlist_scope_service.py \
  Tests/Watchlists/test_watchlist_scope_service.py \
  Tests/Watchlists/test_watchlists_backend_controller.py
git commit -m "feat(watchlists): route stable reader pages"
```

---

### Task 4: Replace offset paging with a screen-owned cached snapshot

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py:690-715,2380-2515,9635-10005,10558-10655`
- Modify: `Tests/Watchlists/test_watchlists_pagination.py`

- [ ] **Step 1: Convert the pagination fixture to typed Reader pages and write RED snapshot tests**

Replace `controller.list_items.return_value = _items(...)` with a `_page(...)` helper that returns `WatchlistItemPage`. Update the former offset assertions to prove:

- first page passes no cursor/watermark and mounts at most 50 rows;
- Next passes the committed snapshot watermark and page-1 cursor;
- Previous publishes the cached prior page with zero controller calls;
- Next after Previous republishes the already-cached forward page with zero controller calls before any later backend fetch is allowed;
- a duplicate id returned on page 2 is never mounted twice;
- a failed/cancelled Next leaves page number, rows, Reader, and snapshot unchanged;
- repeated Next while loading coalesces to one request;
- a late page from a superseded query cannot append;
- explicit Refresh requests a new first page, not the current page's old cursor;
- status/search replacements retain old rows and Reader while pending, then commit page 1 under a new context key;
- same-query pane rebuilds restore the cached page and selected article without I/O.

Representative assertion:

```python
controller.list_reader_items_page.assert_awaited_once_with(
    runtime_backend="local",
    limit=50,
    statuses=["new", "reviewed", "ingested"],
    snapshot_max_item_id=50,
    after=WatchlistItemCursor("2026-08-13 12:01:00", 1),
)
```

- [ ] **Step 2: Run the pagination file and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Watchlists/test_watchlists_pagination.py --tb=short
```

Expected: failures show the screen still calls offset-based `list_items()` and does not cache pages.

- [ ] **Step 3: Replace page-index query identity with a query-context identity**

Make `_items_page_key()` accept an explicit `scope`, status, and search and omit the page index. The key must contain backend, scope kind/id fields, effective status override, normalized manual filter, and casefolded search. This is the snapshot identity; a cursor/page index is state inside the snapshot, not part of the query identity.

Add screen-owned fields:

```python
self._items_snapshot: ReaderItemSnapshot | None = None
self._items_page_index = 0
self._items_pending_query_key: tuple[Any, ...] | None = None
self._items_snapshot_generation = 0
self._items_inflight_replacement: tuple[tuple[Any, ...], asyncio.Future[bool]] | None = None
```

`ReaderItemQuery` is the immutable committed query authority. Build it from the candidate scope plus effective status/search kwargs, and install it only when the candidate snapshot publishes. Attempted filter/search values may live in pending control state, but arrival reconciliation and committed-snapshot identity must read `self._items_snapshot.query`, never the mutable control mirrors. Remove offset-specific committed-page state only after its tests have moved to the new snapshot authority.

- [ ] **Step 4: Implement replacement snapshot loading and transactional presentation**

Extract the old `_load_items_once()` publication discipline into:

```python
async def _replace_items_snapshot(
    self,
    *,
    scope: TreeScope | None = None,
    reason: Literal[
        "initial",
        "refresh",
        "filter",
        "search",
        "scope",
        "return_to_read",
    ],
    clear_reader_on_commit: bool = False,
    focus_first: bool = False,
) -> bool:
    """Load page one off-screen and publish only after rows mount."""
```

Build and freeze candidate query kwargs from the explicit candidate scope, not `self.tree_scope` or later-mutated control mirrors. Call `controller.list_reader_items_page(limit=50, ...)` with no watermark/cursor. Construct a candidate `ReaderItemSnapshot`, apply its first page under `_items_page_presentation_lock`, and only then replace `_items_snapshot`, `_loaded_items`, page index, count, pager authority, and arrival count. Reuse the existing generation/latest-query guards and row-presentation rollback; do not clear the old rows at request time.

- [ ] **Step 5: Implement Next fetch and cached Previous presentation**

Next first checks `self._items_page_index + 1 < snapshot.page_count`; when true, it calls `_present_cached_items_page()` and performs zero I/O. Only when the user is already on the last cached page may `_load_next_items_page()` read the snapshot's traversal cursor and call the controller with the snapshot watermark. Each backend result is staged through `ReaderItemSnapshot.with_continuation()`; the committed snapshot object remains untouched until candidate rows present successfully. `_present_cached_items_page(index)` likewise performs no I/O, presents the exact cached tuple, and commits the index after presentation succeeds.

If deduplication makes a fetched continuation empty while `has_more=True`, continue from the staged candidate's traversal cursor inside a bounded loop until a unique row appears or `has_more=False`; do not publish the candidate yet. Never append a blank continuation to the visible `pages` list. If the chain ends with `has_more=False` and no unique rows, atomically commit only the staged traversal/seen-id/`has_more` state under the generation guard, keep the current visible page mounted, and disable Next. If candidate row presentation fails or is cancelled, discard the entire candidate chain so committed pages, seen ids, cursor, and `has_more` all remain unchanged.

- [ ] **Step 6: Rewire Refresh, status, search, Next, and Previous**

- Refresh: `_replace_items_snapshot(reason="refresh")` for the committed scope; keep old snapshot until success.
- Status change: update parked control intent, schedule replacement page 1, and publish only on success.
- Search change: preserve the 0.3-second debounce, then replace page 1.
- Status/star/read mutations: patch the matching item dicts across the committed snapshot's cached pages and the mounted row/count projections in place; do not call the Reader page API and do not change watermark, traversal cursor, frozen count, seen ids, or pending arrivals.
- Next: `_load_next_items_page()`.
- Previous: `_present_cached_items_page(self._items_page_index - 1)`.

Do not clear Reader selection for refresh/filter/search. `_with_open_item()` remains the same-scope action pin and must use the effective-date helper rather than raw `created_at` ordering when it inserts a carried row; call it for `reason in {"filter", "search"}`. Ordinary item mutations never replace or requery the snapshot: the selected row is already one of the cached shared dicts, and the existing pane-level predicate pin keeps it visible after that dict is patched. Explicit Refresh ends the action-driven row pin as the approved design requires: the Reader may keep displaying the selected article, but a refreshed page does not carry an out-of-predicate row back into Feed Items.

Add `_patch_committed_items_after_mutation(item_id, **changes)` as the single mutation reconciliation path. It updates every cached occurrence of that id (normally one because of the seen-id guard), `_loaded_items`, the selected Reader dict, mounted row presentation, snapshot-bounded unread/star projections, and live tree badges as appropriate. It never calls `list_reader_items_page()`. Mark-all-read/undo apply the same bounded patch over their returned id batch. If an authoritative single-item read is required, use the existing detail/status seam rather than a page query.

Audit every current `_load_items()` call (`rg -n "_load_items\\(" ...`) and replace it with an explicit path/reason before deleting the old method:

| Existing trigger | New path |
| --- | --- |
| initial local Read mount | replacement, `reason="initial"` |
| server-to-local recovery or return from management | replacement, `reason="return_to_read"` |
| Read tree/breadcrumb gesture | replacement, `reason="scope"` |
| status filter | replacement, `reason="filter"` |
| debounced search | replacement, `reason="search"` |
| Refresh button/pill | replacement, `reason="refresh"` |
| status/star/read/mark-all write completion | in-place committed-snapshot patch; no page query |
| Previous/Next | cached presentation or continuation methods, never replacement |

The task is not complete while any production `_load_items()` call remains, any replacement call omits `reason`, or any mutation path calls `list_reader_items_page()`.

- [ ] **Step 7: Run the pagination file and verify GREEN**

Run the Step 2 command.

Expected: the converted stable-snapshot pagination file passes with no offset assertions.

- [ ] **Step 8: Commit the UI pagination foundation**

```bash
git add \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_pagination.py
git commit -m "refactor(watchlists): cache reader keyset pages"
```

---

### Task 5: Commit Read scope only with its mounted first page

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py:1348-1518,3579-3705,4210-4265,9635-10005`
- Modify: `Tests/Watchlists/test_watchlists_pagination.py`
- Modify: `Tests/Watchlists/test_watchlists_collections_screen.py`
- Modify: `Tests/Watchlists/test_watchlists_scoped_rebuilds.py`

- [ ] **Step 1: Write mounted RED tests for pending, successful, failed, and superseded scopes**

Cover this exact sequence:

```python
old_scope = screen.tree_scope
old_rows = screen._loaded_items
old_reader = screen._selected_content_item
screen.post_message(TreeScopeChanged(TreeScope(kind="watchlist", watchlist_id=7)))

await _wait_until(pilot, pending_request_started)
assert screen.tree_scope == old_scope
assert tree.active_scope == old_scope
assert screen._loaded_items is old_rows
assert content.item is old_reader

replacement.release(page_for_watchlist_7)
await _wait_until(pilot, lambda: screen.tree_scope.watchlist_id == 7)
assert tree.active_scope == TreeScope(kind="watchlist", watchlist_id=7)
assert screen._selected_content_item is None
assert content.item is None
```

Add sibling tests proving:

- failure keeps scope/highlight/heading/rows/Reader and notifies `Couldn't open <attempt>; still showing <committed>.`;
- rapid A → B → C allows only C to publish even if A/B finish later;
- row-presentation failure does not commit the candidate scope;
- breadcrumb navigation uses the same pending path;
- a local management-tab scope gesture commits immediately, invalidates parked Reader rows/selection, and issues no item query;
- returning from that management tab to Read loads the committed scope into an honest loading/Retry state rather than mounting rows from the prior scope.

- [ ] **Step 2: Run the exact new scope tests and verify RED**

Run the new nodes in the three listed Watchlists files with `-k "atomic_scope or pending_scope or management_scope_invalidates_reader"`.

Expected: failures show `_apply_tree_scope()` still changes `tree_scope` before the item request completes.

- [ ] **Step 3: Split request and commit responsibilities**

Retain `_apply_tree_scope()` as the committed reconciliation method, but remove automatic item loading from `watch_tree_scope()`. Add:

```python
def _request_tree_scope(self, scope: TreeScope) -> None:
    if self.active_section == "items" and self.runtime_backend == "local":
        self._pending_tree_scope = scope
        self.run_worker(
            self._replace_items_snapshot(
                scope=scope,
                reason="scope",
                clear_reader_on_commit=True,
            ),
            exclusive=True,
            group="wc_items",
        )
        return
    self._commit_management_tree_scope(scope)
```

Route both `_on_tree_scope_changed()` and breadcrumb promotion through this request method. Audit every existing `_apply_tree_scope()` caller: write-completion fallbacks may call the committed helper only when their surrounding operation has already established the new management context; user navigation must call the request path.

- [ ] **Step 4: Make first-page presentation the scope commit boundary**

When `_replace_items_snapshot(..., clear_reader_on_commit=True)` is still the newest generation, enter `with self.app.batch_update():` while holding the presentation lock, apply the candidate rows, and commit the following state before leaving the batch. This is the existing Watchlists/Textual atomic-paint boundary; without it, the `await pane.apply_page_items(...)` can expose a frame containing replacement rows under the committed old heading.

1. replace snapshot/rows/page/count state;
2. `_apply_tree_scope(candidate_scope)` so heading, Inspector, and active tree styling follow the rows;
3. clear selected entity/item shadows and mounted pane selections;
4. set `ContentPane.item = None` and Reader position to the empty state;
5. push pager/count/arrival state.

Use a private commit guard so `tree_scope`'s watcher refreshes header/sources/tree styling but does not schedule a second item load.

- [ ] **Step 5: Preserve committed state on failure and supersession**

Store `_pending_tree_scope` separately. On an exception from the newest pending scope, restore only loading controls and active styling; do not mutate committed state. Name attempted and retained scopes through the existing escaped display-label helpers. A superseded request returns silently and cannot clear the newer pending marker.

- [ ] **Step 6: Preserve management behavior without hidden item I/O**

On local non-Read sections, commit the scope immediately and invalidate every parked Reader authority together: `_items_snapshot`, `_loaded_items`, selected Reader item/page identity, `_items_snapshot_count`, `_items_pending_arrivals`, and mounted pane rows/selection/count/new-items note. On the next Read entry, show loading with count `0` and no pill for that committed scope until `_replace_items_snapshot(reason="return_to_read")` succeeds; failure leaves the scope committed and shows scoped Retry with no old rows, count, pill, or article relabelled.

- [ ] **Step 7: Run the focused scope/pagination regressions and verify GREEN**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Watchlists/test_watchlists_pagination.py \
  Tests/Watchlists/test_watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_scoped_rebuilds.py \
  -k "pagination or snapshot or scope or reader or tree_click" --tb=short
```

Expected: selected atomic scope, management invalidation, and snapshot cases pass.

- [ ] **Step 8: Commit atomic scope publication**

```bash
git add \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_pagination.py \
  Tests/Watchlists/test_watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_scoped_rebuilds.py
git commit -m "feat(watchlists): commit reader scope atomically"
```

---

### Task 6: Keep snapshot counts and new arrivals honest

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/article_list.py:300-405,760-810`
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py:690-715,1348-1505,2330-2410,11745-11820`
- Modify: `Tests/Watchlists/test_watchlists_article_list.py`
- Modify: `Tests/Watchlists/test_watchlists_collections_screen.py`
- Modify: `Tests/Watchlists/test_watchlists_pagination.py`

- [ ] **Step 1: Write RED widget tests for snapshot-owned copy**

Add plain reactives `snapshot_count` and keep `new_items_note` screen-seeded. Assert the toolbar renders `50 items in snapshot` (singular for 1), a pane rebuild retains `3 new items`, and clicking the pill posts `RefreshItemsRequested` without clearing the note before the screen reports success.

- [ ] **Step 2: Write RED screen tests for exact arrivals**

Prove:

- a live unread-count decrease caused by marking an existing row read does not create an arrival;
- a matching inserted row above the watermark produces `1 new item` without modifying `_loaded_items`, page number, Reader, or `snapshot_count`;
- when an above-watermark matching row already exists, a later read/status/star mutation patches only the committed cached rows and does not admit that arrival, change the high-water/count, or clear the pill;
- an inserted row outside the active source/watchlist/status/search scope does not increment the pill;
- refresh failure retains the old pill/count/snapshot;
- successful refresh installs the new count/high-water and clears pending arrivals;
- tree badges may update live while the mounted snapshot count remains unchanged;
- pane/workbench rebuilds re-seed the count and pill from screen state.
- cached Next after Previous replays the exact page with no backend call;
- explicit Refresh preserves Reader content but removes an out-of-predicate action-pinned row from Feed Items.

- [ ] **Step 3: Run the exact widget/arrival tests and verify RED**

Run the new nodes with `-k "snapshot_count or new_items or arrivals"` in the three listed files.

Expected: failures show the pill is pane-local/unconditionally dismissed and the screen has no watermark-based arrival query.

- [ ] **Step 4: Make count and arrival state screen-owned**

Add:

```python
self._items_snapshot_count = 0
self._items_pending_arrivals = 0
self._items_arrival_generation = 0
```

Seed `ArticleListPane.snapshot_count` and `new_items_note` in `_build_detail_pane()`, and push them in the same in-place pager update path. `ArticleListPane.on_click()` posts refresh but does not clear the note; only a successful replacement snapshot clears it.

- [ ] **Step 5: Reconcile arrivals against the committed snapshot query**

Add `_refresh_items_pending_arrivals()` that captures the committed snapshot object and calls `controller.count_reader_item_arrivals()` with the snapshot watermark plus `snapshot.query.as_kwargs()`. It must never rebuild kwargs from `_items_status_filter`, `_items_search_query`, a pending scope, or other mutable attempted intent. Publish only if generation, snapshot object identity, backend, and Read section still match. Invoke it after terminal tree-data refreshes and after refresh-all completes. It updates the pill only; it never mutates rows or the snapshot-bounded count.

Replace `_refresh_all_worker()`'s global unread-delta pill calculation with this exact arrival reconciliation. The aggregate toast may still report the check batch's unread delta for its existing behavior, but the Reader pill authority is the creation-watermark count.

- [ ] **Step 6: Run focused arrival and pagination tests and verify GREEN**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Watchlists/test_watchlists_article_list.py \
  Tests/Watchlists/test_watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_pagination.py \
  -k "snapshot_count or new_items or arrivals or refresh or pagination" --tb=short
```

Expected: selected widget, arrival, refresh, and pagination tests pass.

- [ ] **Step 7: Commit honest arrival presentation**

```bash
git add \
  tldw_chatbook/UI/Watchlists_Modules/article_list.py \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_article_list.py \
  Tests/Watchlists/test_watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_pagination.py
git commit -m "feat(watchlists): hold arrivals behind reader refresh"
```

---

### Task 7: Focused verification, self-review, and Backlog closeout

**Files:**
- Modify: `backlog/tasks/task-22450 - Stabilize-Watchlists-Feed-Items-snapshots-and-atomic-scope-commits.md`
- Modify: `Docs/superpowers/plans/2026-08-25-watchlists-stable-feed-items-snapshots.md`

- [ ] **Step 1: Run the complete changed-functionality test selection**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/DB/test_subscriptions_db_watchlists_reader_snapshot.py \
  Tests/DB/test_subscriptions_db_watchlists_agent_search.py \
  Tests/Subscriptions/test_item_dates.py \
  Tests/Watchlists/test_reader_item_snapshot.py \
  Tests/Watchlists/test_watchlists_pagination.py \
  Tests/Subscriptions/test_watchlist_normalizers.py \
  Tests/Subscriptions/test_local_watchlists_service.py \
  Tests/Subscriptions/test_watchlist_scope_service.py \
  Tests/Watchlists/test_watchlist_scope_service.py \
  Tests/Watchlists/test_watchlists_backend_controller.py \
  Tests/Watchlists/test_watchlists_article_list.py \
  Tests/Watchlists/test_watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_scoped_rebuilds.py \
  -k "reader or item or keyset_traversal or lookahead or pagination or snapshot or scope or new_items or arrivals or list_items" \
  --tb=short
```

Expected: all selected changed-functionality tests pass. Record exact counts and warnings in this plan and TASK-22450. Do not broaden to the full suite.

- [ ] **Step 2: Run modified-file Ruff**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/Subscriptions/watchlist_item_page.py \
  tldw_chatbook/UI/Watchlists_Modules/reader_item_snapshot.py \
  tldw_chatbook/DB/Subscriptions_DB.py \
  tldw_chatbook/Subscriptions/watchlist_normalizers.py \
  tldw_chatbook/Subscriptions/local_watchlists_service.py \
  tldw_chatbook/Subscriptions/watchlist_scope_service.py \
  tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py \
  tldw_chatbook/UI/Watchlists_Modules/article_list.py \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  Tests/DB/test_subscriptions_db_watchlists_reader_snapshot.py \
  Tests/Watchlists/test_reader_item_snapshot.py \
  Tests/Watchlists/test_watchlists_pagination.py \
  Tests/Subscriptions/test_watchlist_normalizers.py \
  Tests/Subscriptions/test_local_watchlists_service.py \
  Tests/Subscriptions/test_watchlist_scope_service.py \
  Tests/Watchlists/test_watchlist_scope_service.py \
  Tests/Watchlists/test_watchlists_backend_controller.py \
  Tests/Watchlists/test_watchlists_article_list.py \
  Tests/Watchlists/test_watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_scoped_rebuilds.py
```

Expected: Ruff exits 0. If a touched legacy file has a pre-existing finding, run the identical command against the recorded base commit and document exact baseline parity; do not bulk-format or clean unrelated code.

- [ ] **Step 3: Run diff hygiene and scope checks**

Run:

```bash
git diff --check
git status --short
git diff --stat origin/dev...HEAD
git diff --name-only origin/dev...HEAD
```

Expected: no whitespace errors; only TASK-22450 implementation/test/docs files and the already-approved TASK-22451/spec bookkeeping are present.

- [ ] **Step 4: Perform a requirements-oriented self-review**

Inspect the final diff and explicitly verify:

- no Reader `OFFSET` remains;
- Reader ordering is DESC/DESC while agent ordering is unchanged;
- every continuation uses the initial watermark;
- seen ids cannot mount twice;
- old scope/highlight/rows/Reader survive pending, failure, and supersession;
- successful scope commit clears Reader exactly once;
- management tabs issue no hidden Reader query;
- new arrival count uses ids above the watermark with exact query predicates;
- refresh failure cannot dismiss the committed arrival notice;
- no aggregate feed-child code from TASK-22451 leaked into this PR.

- [ ] **Step 5: Update TASK-22450 and lessons only if evidence warrants it**

Check all six acceptance criteria, add concise implementation notes with modified files, decisions, exact test/Ruff/diff evidence, and the existing ADR-042 link. Run `backlog task edit 22450 -s Done`, then re-open it with `backlog task 22450 --plain` and restore any plan/provenance text the CLI rewrites. Add a `lessons-*.md` entry only if implementation uncovers a new reusable incident; do not invent one.

- [ ] **Step 6: Commit closeout metadata**

```bash
git add \
  "backlog/tasks/task-22450 - Stabilize-Watchlists-Feed-Items-snapshots-and-atomic-scope-commits.md" \
  Docs/superpowers/plans/2026-08-25-watchlists-stable-feed-items-snapshots.md
git commit -m "docs(watchlists): close stable reader snapshot task"
```

- [ ] **Step 7: Invoke `superpowers:verification-before-completion`**

Re-run the final evidence required by that skill before claiming the branch is ready. Do not merge or open a PR until the focused tests, modified-file Ruff, and diff checks are freshly green.
