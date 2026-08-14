# Watchlists Nested Scroll and Pagination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Watchlists Read list and Content reader reachable and independently scrollable, with 10–50-row and 20–50-row regions plus explicit 50-item pagination.

**Architecture:** Keep pagination and query provenance on `WatchlistsCollectionsScreen`, presentation-only pager state on `ArticleListPane`, and scroll ownership in the existing workbench/pane hierarchy. Reuse the controller's `limit`/`offset` API with a 51st lookahead row; gate Read-only sizing behind a workbench class so operations tabs retain their fill layout.

**Tech Stack:** Python 3.11+, Textual 8.x, Rich, pytest/pytest-asyncio, TCSS, Backlog.md CLI

---

## Planning Context

- Approved design: `Docs/superpowers/specs/2026-08-13-watchlists-nested-scroll-pagination-design.md`
- Backlog task: `TASK-16221`
- Existing decision: `backlog/decisions/042-watchlists-reader-first-ia.md`
- Isolated worktree: `.worktrees/watchlists-nested-scroll-pagination`
- Branch: `codex/watchlists-nested-scroll-pagination`
- Baseline evidence: 104 focused Watchlists tests passed before product-code changes.
- Required repository lessons read before execution:
  `backlog/docs/lessons-testing-evidence.md`,
  `backlog/docs/lessons-live-verification.md`, and
  `backlog/docs/lessons-backlog-hygiene.md`.

ADR required: no

ADR path: `backlog/decisions/042-watchlists-reader-first-ia.md`

Reason: this is a contained layout and pagination refinement inside the existing reader-first ownership and service boundaries. It adds no storage, schema, dependency, service contract, global keybinding, or new long-lived application structure.

## File Map

- Modify `tldw_chatbook/UI/Watchlists_Modules/article_list.py`: pager messages/state, authoritative-search presentation state, and non-selecting first-row cursor focus.
- Modify `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`: page/query keys, 51-row loading, transactional page actions, context resets, open-item pin provenance, and Read-mode class synchronization.
- Modify `tldw_chatbook/UI/Watchlists_Modules/watchlists_workbench.py`: make only the centre column a `VerticalScroll`.
- Modify `tldw_chatbook/UI/Watchlists_Modules/content_pane.py`: give the article body its own `VerticalScroll` while leaving actions/footer fixed.
- Modify `tldw_chatbook/css/features/_watchlists.tcss`: Read-only 10–50/20–50 region sizing and nested scroll ownership.
- Regenerate `tldw_chatbook/css/tldw_cli_modular.tcss`: generated CSS bundle; never hand-edit it.
- Modify `Tests/Watchlists/test_watchlists_article_list.py`: pager, authoritative search, and focus-suppression widget contracts.
- Create `Tests/Watchlists/test_watchlists_pagination.py`: focused screen-level pagination, reset, failure, stale-result, and pin-provenance tests.
- Modify `Tests/Watchlists/test_watchlists_workbench.py`: outer-centre and region geometry/scroll assertions against production CSS.
- Modify `Tests/UI/test_watchlists_content_pane.py`: Content-body wrapper, fixed chrome, and integrated state-preservation assertions.
- Modify `backlog/tasks/task-16221 - Make-Watchlists-Read-list-and-Content-independently-scrollable-with-explicit-pagination.md`: implementation plan/notes, checked acceptance criteria, verification evidence, and final status.

## Guardrails

- Use @superpowers:test-driven-development for each task: observe the intended red failure before product code.
- Use @ponytail to keep the implementation inside existing widgets and controller seams; do not add a paginator service, total-count query, custom wheel forwarding, or persisted height state.
- Use @textual-tui for Textual focus, reactive, nested-scroll, and geometry behavior.
- Use @impeccable only to validate the approved hierarchy and fixed-control behavior; do not redesign the established terminal visual language.
- Preserve the current `ArticleListPane`/`ListView`, corpus-wide FTS/LIKE search, in-place row rebuild/filter behavior, reader actions, and local-page `j`/`k`/next-unread semantics.
- Do not change the Sources management tab. “Sources area” in the request means the Read tab's article list.

### Task 1: Add the presentation-only pager and safe list focus

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/article_list.py:210-775`
- Test: `Tests/Watchlists/test_watchlists_article_list.py`

- [ ] **Step 1: Write failing pager rendering and message tests**

Extend `ArticleListHarness` to capture the two new messages, then add tests equivalent to:

```python
from textual.widgets import Button

from tldw_chatbook.UI.Watchlists_Modules.article_list import (
    NextItemsPageRequested,
    PreviousItemsPageRequested,
)

def on_previous_items_page_requested(self, message):
    self.captured_messages.append(("previous_page", None))

def on_next_items_page_requested(self, message):
    self.captured_messages.append(("next_page", None))


async def test_pager_reflects_page_boundaries_and_loading_without_recompose():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        list_view = pane.query_one("#items-table", ListView)
        previous = pane.query_one("#items-page-previous", Button)
        next_button = pane.query_one("#items-page-next", Button)
        label = pane.query_one("#items-page-label", Static)

        assert previous.disabled and next_button.disabled
        assert str(label.renderable) == "Page 1"

        pane.page_number = 2
        pane.has_previous = True
        pane.has_next = True
        await pilot.pause()
        assert pane.query_one("#items-table", ListView) is list_view
        assert not previous.disabled and not next_button.disabled
        assert str(label.renderable) == "Page 2"

        pane.page_loading = True
        await pilot.pause()
        assert previous.disabled and next_button.disabled


async def test_pager_buttons_post_narrow_page_requests():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.page_number = 2
        pane.has_previous = True
        pane.has_next = True
        await pilot.pause()
        pane.query_one("#items-page-previous", Button).press()
        pane.query_one("#items-page-next", Button).press()
        await pilot.pause()
        assert ("previous_page", None) in app.captured_messages
        assert ("next_page", None) in app.captured_messages
```

- [ ] **Step 2: Run the pager tests and verify they fail for the missing controls/messages**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Watchlists/test_watchlists_article_list.py -k 'pager or page_requests'
```

Expected: FAIL because `NextItemsPageRequested`, `PreviousItemsPageRequested`, and `#items-page-*` do not exist.

- [ ] **Step 3: Write failing authoritative-search and non-selecting-focus tests**

Add two focused tests:

```python
async def test_authoritative_backend_search_does_not_refilter_returned_rows():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.search_query = "body-only-token"
        pane.items = [_item(1, title="Backend FTS match", content="")]
        pane.search_results_authoritative = True
        await pilot.pause()
        assert [item["item_id"] for item in pane.displayed_items()] == [1]

        pane.query_one("#items-search-input", Input).value = "edited-token"
        await pilot.pause()
        assert pane.search_results_authoritative is False
        assert pane.displayed_items() == []


async def test_focus_first_row_does_not_select_but_next_user_move_does():
    app = ArticleListHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ArticleListPane)
        pane.items = [_item(1), _item(2)]
        await pilot.pause()

        pane.focus_first_row_without_selecting()
        await pilot.pause()
        assert pane.query_one("#items-table", ListView).has_focus
        assert not [m for m in app.captured_messages if m[0] == "item_selected"]

        pane.query_one("#items-table", ListView).action_cursor_down()
        await pilot.pause()
        assert [m for m in app.captured_messages if m[0] == "item_selected"]

        pane.query_one("#items-table", ListView).action_cursor_up()
        await pilot.pause()
        selected = [m for m in app.captured_messages if m[0] == "item_selected"]
        assert len(selected) == 2, "returning to the first row must select normally"
```

- [ ] **Step 4: Run the new search/focus tests and verify they fail**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Watchlists/test_watchlists_article_list.py \
  -k 'authoritative_backend_search or focus_first_row'
```

Expected: FAIL because the reactive and focus helper do not exist and local filtering still removes the simulated body-only FTS result.

- [ ] **Step 5: Implement pager messages, plain reactives, and in-place control updates**

In `article_list.py`, add message classes beside the other screen-facing messages and plain reactives beside `new_items_note`:

```python
class PreviousItemsPageRequested(Message):
    """Ask the owning screen for the preceding backend page."""


class NextItemsPageRequested(Message):
    """Ask the owning screen for the next backend page."""


page_number = reactive(1)
has_previous = reactive(False)
has_next = reactive(False)
page_loading = reactive(False)
search_results_authoritative = reactive(False)
```

Compose the pager after `#items-queued-legend`:

```python
with Horizontal(id="items-pagination", classes="destination-filter-strip"):
    yield Button(
        "Previous",
        id="items-page-previous",
        compact=True,
        disabled=self.page_loading or not self.has_previous,
    )
    yield Static(f"Page {self.page_number}", id="items-page-label")
    yield Button(
        "Next",
        id="items-page-next",
        compact=True,
        disabled=self.page_loading or not self.has_next,
    )
```

Use one `_sync_pager()` helper from four `watch_*` methods so page changes never recompose the toolbar or `ListView`. Extend `on_button_pressed` to post only the matching message, then stop the event.

- [ ] **Step 6: Implement provisional-versus-authoritative filtering**

Keep the input value and corpus query separate from the local visibility predicate:

```python
query = "" if self.search_results_authoritative else self.search_query.strip().lower()
```

When `#items-search-input` changes, set `search_results_authoritative = False` before assigning `search_query`. A successful screen load will set it back to `True` in Task 2. Do not add full article content to the list projection.

- [ ] **Step 7: Implement exactly-one programmatic highlight suppression**

Add `_suppressed_highlight_item_id: str | None` in `__init__`. Implement `focus_first_row_without_selecting()` by finding the first visible `_ArticleRow`, focusing the list, recording its id, and only then assigning `ListView.index`. In `on_list_view_highlighted`, clear and return only when the highlighted row matches that recorded id; every later user-driven highlight follows the existing selection path.

```python
def focus_first_row_without_selecting(self) -> None:
    list_view = self.query_one("#items-table", ListView)
    for index, node in enumerate(list_view.children):
        if isinstance(node, _ArticleRow) and node.display and not node.disabled:
            list_view.focus()
            self._suppressed_highlight_item_id = node.item_id_key
            list_view.index = index
            return
```

The widget test must move away from and back to the first row to prove the token was consumed exactly once. If live Textual scheduling shows the event can be skipped, clear the guard with a one-shot `call_after_refresh` fallback; do not leave a stale guard capable of swallowing the next real user move.

Also add a small public async page-application seam so the screen can await the
existing asynchronous row rebuild before focusing:

```python
async def apply_page_items(
    self, items: list[dict[str, Any]], *, focus_first: bool = False
) -> None:
    self.set_reactive(ArticleListPane.items, items)
    await self._rebuild_rows()
    if focus_first:
        self.focus_first_row_without_selecting()
```

Use direct reactive seeding only before mount. Mounted screen loads use this method,
so they neither duplicate `_rebuild_rows()` nor guess when an async watcher finished.

- [ ] **Step 8: Run the full article-list tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Watchlists/test_watchlists_article_list.py
```

Expected: PASS, including identity assertions proving pager/search changes did not replace `#items-table`.

- [ ] **Step 9: Commit Task 1**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/article_list.py \
  Tests/Watchlists/test_watchlists_article_list.py
git commit -m "feat(watchlists): add explicit read pager controls"
```

### Task 2: Load and transition 50-item pages transactionally

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py:780-925,2084-2110,8790-8980,9550-9640`
- Create: `Tests/Watchlists/test_watchlists_pagination.py`

- [ ] **Step 1: Create focused pagination test helpers and first-page lookahead tests**

Build the new test module with the existing production-screen harness pattern (`_build_test_app`, `DestinationHarness`) and an `AsyncMock` controller. Add a helper returning normalized item dicts and tests equivalent to:

```python
@pytest.mark.asyncio
async def test_first_page_requests_lookahead_but_mounts_only_fifty():
    controller = AsyncMock()
    controller.list_items.return_value = [_item(i) for i in range(51)]
    async with _open_screen(controller) as (screen, pilot):
        await screen._load_items()
        controller.list_items.assert_awaited_with(
            runtime_backend="local",
            limit=51,
            offset=0,
            statuses=["new", "reviewed", "ingested"],
        )
        assert len(screen._loaded_items) == 50
        assert screen._items_page_index == 0
        assert screen._items_has_next is True
        pane = screen.query_one("#watchlists-items-pane", ArticleListPane)
        assert pane.page_number == 1
        assert pane.has_previous is False
        assert pane.has_next is True
```

Assert the status kwargs by value without depending on list ordering if the existing constant is a set/frozenset.

- [ ] **Step 2: Add failing next/previous/failure tests**

Add `test_next_commits_offset_fifty_only_after_success` with a controller future:
start Next, assert the page remains 1 while unresolved, resolve 51 rows, then assert
offset 50, page 2, 50 mounted rows, Previous enabled, and Next enabled. Add
`test_previous_returns_to_offset_zero` from a committed page 2 and assert page 1
plus disabled Previous. Add
`test_failed_explicit_transition_preserves_rows_page_content_and_buttons`: seed
page-1 rows, `_selected_content_item`, and the mounted `ContentPane`; make offset
50 raise; assert page index, rows, selection, rendered Content, and prior
lookahead are unchanged while pager loading is cleared. Add
`test_repeated_press_while_loading_starts_only_one_request`: hold the first Next
future unresolved, press Next twice, and assert exactly one offset-50 call.

- [ ] **Step 3: Run the new module and verify red failures**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Watchlists/test_watchlists_pagination.py
```

Expected: FAIL because `_load_items` still requests `limit=100, offset=0`, page state is absent, and the pager messages have no screen handlers.

- [ ] **Step 4: Add minimal page state and a stable query/page key**

Add `_ITEMS_PAGE_SIZE = 50` near existing screen constants. Initialize:

```python
self._items_page_index = 0
self._items_has_next = False
self._items_page_loading = False
self._items_load_generation = 0
self._items_committed_page_key: tuple[Any, ...] | None = None
self._selected_content_page_key: tuple[Any, ...] | None = None
```

Build keys only from primitives so equality is deterministic:

```python
def _items_page_key(self, page_index: int) -> tuple[Any, ...]:
    scope = self.tree_scope
    return (
        self.runtime_backend,
        scope.kind,
        scope.watchlist_id,
        scope.source_id,
        _normalize_items_status_filter(self._items_status_filter),
        self._items_search_query.strip().casefold(),
        page_index,
    )
```

- [ ] **Step 5: Seed and push presentation state without rebuilding the list**

In `_build_detail_pane`, seed `page_number`, `has_previous`, `has_next`, `page_loading`, and `search_results_authoritative` before mounting. Add `_push_items_pager_state()` for the live pane:

```python
def _push_items_pager_state(self) -> None:
    try:
        pane = self.query_one("#watchlists-items-pane", ArticleListPane)
    except NoMatches:
        return
    pane.page_number = self._items_page_index + 1
    pane.has_previous = self._items_page_index > 0
    pane.has_next = self._items_has_next
    pane.page_loading = self._items_page_loading
```

Store a screen boolean for authoritative search and seed/push it through the same path.

At the start of `handle_item_selected`, capture the page that actually produced
the row **before** the existing awaited detail fetch lets another context move;
assign that captured value when storing the selected Content item:

```python
selection_page_key = self._items_committed_page_key
await self._load_item_content(event.item)
# existing selected-item/content writes
self._selected_content_page_key = selection_page_key
```

This belongs in Task 2 because the new gated pin is also introduced here; leaving
the key `None` until Task 3 would temporarily break the existing mark-read pin.

- [ ] **Step 6: Replace `_load_items` with a 51-row, generation-guarded load**

Change the signature to accept a target and explicit-navigation flag while preserving zero-argument callers:

```python
async def _load_items(
    self,
    *,
    target_page_index: int | None = None,
    explicit_page_change: bool = False,
) -> bool:
```

The body must:

1. Resolve `target = current` when omitted and clamp it to zero.
2. Increment/capture `_items_load_generation` and capture `target_key`.
3. Set loading, disable pager buttons, and retain current rows/Content.
4. Call `list_items(limit=51, offset=target * 50, **existing_scope_status_search_kwargs)`.
5. Ignore the result if its generation is stale or the current query portion no longer matches `target_key`.
6. Set `has_next = len(raw_rows) > 50` and keep only `raw_rows[:50]`.
7. Apply the open-item pin only when `_selected_content_page_key == target_key`, never merely because Content is non-empty.
8. Commit rows, page index, lookahead, committed key, and authoritative-search state together.
9. Push pager state and `await pane.apply_page_items(rows, focus_first=explicit_page_change)` so the asynchronous row rebuild completes before focus targets the first row.
10. Preserve ordinary refresh focus by passing `focus_first=False` unless this was a successful explicit Previous/Next action.
11. On the latest request's failure, retain explicit-transition state, clear loading, notify, and return `False`.

Do not turn cancellation into an error toast; re-raise `asyncio.CancelledError` before the broad exception handler.

- [ ] **Step 7: Cap the open-item pin without dropping the pinned row**

Extend `_with_open_item` to accept `max_items: int | None`. Preserve sorted insertion, but if the carried item would be outside a full 50-row window, replace the last visible slot with it. The lookahead decision must be computed before this helper and the returned list must never exceed 50.

- [ ] **Step 8: Handle pager messages with boundary/loading guards**

Import the new messages from `article_list.py` and add handlers:

```python
@on(PreviousItemsPageRequested)
def handle_previous_items_page_requested(self, event) -> None:
    event.stop()
    if self._items_page_loading or self._items_page_index == 0:
        return
    self.run_worker(
        self._load_items(
            target_page_index=self._items_page_index - 1,
            explicit_page_change=True,
        ),
        exclusive=True,
        group="wc_items",
    )

@on(NextItemsPageRequested)
def handle_next_items_page_requested(self, event) -> None:
    event.stop()
    if self._items_page_loading or not self._items_has_next:
        return
    self.run_worker(
        self._load_items(
            target_page_index=self._items_page_index + 1,
            explicit_page_change=True,
        ),
        exclusive=True,
        group="wc_items",
    )
```

- [ ] **Step 9: Run page-transition tests and the existing collections-screen regressions**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Watchlists/test_watchlists_pagination.py \
  Tests/Watchlists/test_watchlists_collections_screen.py
```

Expected: PASS. Existing tests that asserted the old 100-row cap must be updated only where the approved page size intentionally changes the expected value to 50.

- [ ] **Step 10: Commit Task 2**

```bash
git add tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_pagination.py \
  Tests/Watchlists/test_watchlists_collections_screen.py
git commit -m "feat(watchlists): page read items in fifty-row windows"
```

### Task 3: Enforce reset, provenance, search, stale-result, and empty-page rules

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py:890-900,2084-2110,3612-3645,4132-4235,8918-8980,9559-9640`
- Modify: `Tests/Watchlists/test_watchlists_pagination.py`
- Modify: `Tests/Watchlists/test_watchlists_collections_screen.py` only for corpus-search expectations that intentionally move from 100 to 50.

- [ ] **Step 1: Add failing context reset and refresh preservation tests**

Add parameterized tests proving:

- tree scope, backend, status filter, and search edit logically reset to page index 0;
- search reset disables both buttons immediately through its debounce and sends `search=...`, `limit=51`, `offset=0`;
- ordinary Refresh reloads the committed current page, not page 1;
- returning to Read or replacing the ITEMS region reseeds the committed page and lookahead;
- `j`, `k`, and next-unread do not request a different page.

Run the focused tests and expect failures until every reset site is wired.

- [ ] **Step 2: Add failing provenance and provisional-search tests**

Cover five distinct keys. `test_same_key_refresh_can_pin_open_item_without_exceeding_fifty`
selects a page-1 row, removes it from the unread response, refreshes the same key,
and asserts the selected id is still among exactly 50 mounted ids.
`test_explicit_next_page_preserves_content_without_injecting_open_item` selects on
page 1, loads page 2, then asserts Content still holds that id while none of the 50
page-2 rows does. `test_scope_backend_filter_and_search_changes_invalidate_the_pin`
parameterizes each changed key field and makes the new response omit the old id.
`test_selection_during_search_debounce_records_prior_committed_key` edits the
query, selects a provisionally visible old row, and asserts the selection key is
the old committed key rather than the pending query. Finally,
`test_content_only_fts_match_survives_authoritative_backend_load` returns an item
whose list projection lacks the matching full-body token and asserts it remains
displayed after `search_results_authoritative=True` is pushed.

- [ ] **Step 3: Add failing stale/failure/empty-page tests**

Use two controllable futures to complete an older request after a newer context request; assert the old result never paints. Add:

- failed explicit Next keeps page N and old lookahead;
- failed search remains logical page 1, keeps provisionally filtered old rows/Content, and disables Next;
- an empty non-first refresh walks backward in 50-row offsets until a non-empty page or page 1;
- a 51st lookahead row never enters pane items, selection, status writes, or navigation.

- [ ] **Step 4: Implement one context-reset helper and call it at every agreed boundary**

Avoid duplicated partial resets:

```python
def _reset_items_paging_for_context(self, *, loading: bool) -> None:
    self._items_page_index = 0
    self._items_has_next = False
    self._items_page_loading = loading
    self._items_search_results_authoritative = False
    self._items_load_generation += 1
    self._push_items_pager_state()
```

Call it before dispatch/debounce from `watch_tree_scope`, `watch_runtime_backend`, status-filter changes, and search edits. Do not call it from section switching or Refresh. Search edits keep the current rows mounted and provisionally filtered while the timer is armed.

`watch_runtime_backend` does not currently call `_load_items`, so the Read branch
must reset **and** actively dispatch the page-1 load:

```python
read_is_active = self.active_section == "items"
self._reset_items_paging_for_context(loading=read_is_active)
if read_is_active:
    self.run_worker(self._load_items(), exclusive=True, group="wc_items")
```

Without that paired dispatch, the reset would leave old-backend rows mounted and
pager controls disabled forever. Non-Read backend switches reset the stored page to
1 without starting a hidden Read load; returning to Read uses the existing active-
section loader.

- [ ] **Step 5: Verify selection provenance at the real selection boundary**

Task 2 already set this in `handle_item_selected` alongside the gated pin:

```python
self._selected_content_page_key = self._items_committed_page_key
```

Keep it tied to the mounted committed page, not `_items_page_key(0)` recomputed from a pending search context. The Task 3 debounce-window test proves a selection belongs to the prior committed page and cannot be pinned into pending search results.

- [ ] **Step 6: Complete stale guards and empty-page fallback**

On each successful await, compare both generation and the key's query-context prefix (all fields except page) to current state before painting. If the latest load returns no rows for page > 0, decrement the target and retry with the same captured context until data arrives or target reaches zero. Keep pager disabled across the fallback loop; notify only if the whole operation fails.

- [ ] **Step 7: Push authoritative search only after a successful backend result**

At search edit, the pane already sets its reactive false. On success, set the screen mirror true and push it before/with the rows. On failure, leave it false; page number stays 1 and Next stays disabled. Blank-query loads may set the mirror true harmlessly, but must not suppress status filtering.

- [ ] **Step 8: Run all pagination and current search/filter tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Watchlists/test_watchlists_pagination.py \
  Tests/Watchlists/test_watchlists_collections_screen.py \
  Tests/UI/test_watchlists_items_status_filter.py \
  Tests/UI/test_watchlists_read_status.py
```

Expected: PASS. Specifically confirm the corpus-wide “beyond the first page” search still passes with the new 50-item default page.

- [ ] **Step 9: Commit Task 3**

```bash
git add tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_pagination.py \
  Tests/Watchlists/test_watchlists_collections_screen.py
git commit -m "fix(watchlists): preserve read paging across context changes"
```

### Task 4: Make the centre scroll and bound the Read list to 10–50 rows

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/watchlists_workbench.py:13-18,165-190`
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py:2554-2640,3830-3900,4132-4165`
- Modify: `tldw_chatbook/css/features/_watchlists.tcss:116-250,700-760`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Tests/Watchlists/test_watchlists_workbench.py`
- Modify: `Tests/Watchlists/test_watchlists_article_list.py`

- [ ] **Step 1: Write failing topology and production-CSS geometry tests**

Add a structural assertion that `#wl-centre` is `VerticalScroll`, while the left and right regions remain direct siblings of that scroll owner. Replace the old 12-row Content-only geometry assumptions later in Task 5.

Add Read-mode geometry cases at `(120, 36)`, `(180, 50)`, and a tall terminal:

- empty/short Read list outer region is at least 10 rows;
- enough rows grow the outer region but never beyond 50;
- `#items-table` is the inner vertical scroll owner and can reach the last row;
- toolbar, legend, and pager remain painted before and after inner scrolling;
- the outer centre can scroll from Read list to Content without moving either rail;
- a non-Read ITEMS pane without `watchlists-read-mode` retains its existing fill behavior.
- solo ITEMS lifts both the 50-row outer cap and the inner `ListView` cap, then
  Restore returns to the bounded stacked layout.

Use `app.export_screenshot()` or compositor strips for “the user can see it” assertions; computed styles alone are not render evidence.

- [ ] **Step 2: Run the new geometry tests and verify red failures**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Watchlists/test_watchlists_workbench.py \
  Tests/Watchlists/test_watchlists_article_list.py -k 'height or scroll or pager'
```

Expected: FAIL because the centre is a fixed `Vertical`, Read has no mode class, and ITEMS still claims `2fr` without the approved min/max contract.

- [ ] **Step 3: Replace only the centre container with `VerticalScroll`**

In `watchlists_workbench.py`:

```python
from textual.containers import Horizontal, Vertical, VerticalScroll

with VerticalScroll(id="wl-centre", classes="watchlists-centre"):
    if self._header is not None:
        yield self._header()
    for region in CENTRE_REGIONS:
        if region not in self._hidden:
            yield self._region_widget(region)
```

Do not wrap the entire `WatchlistsWorkbench`; the rails must remain outside the scroll viewport.

- [ ] **Step 4: Add and synchronize the Read-mode class**

At initial composition, pass `classes="watchlists-read-mode"` only when `active_section == "items"`. In `_swap_active_section`, call:

```python
workbench.set_class(self.active_section == "items", "watchlists-read-mode")
```

before/with `apply_section_view` so the newly mounted section receives the correct sizing on its first layout. Test both Read → Sources and Sources → Read without replacing the workbench or rails.

- [ ] **Step 5: Implement Read-only list sizing and fixed pager chrome in source TCSS**

Add narrow selectors, then tune their inner caps against the geometry tests:

```css
.watchlists-read-mode .watchlists-region-items {
    height: auto;
    min-height: 10;
    max-height: 50;
    overflow: hidden;
}

.watchlists-read-mode #watchlists-detail-pane,
.watchlists-read-mode #watchlists-items-pane {
    height: auto;
    min-height: 0;
}

.watchlists-read-mode #items-table {
    height: auto;
    min-height: 1;
    max-height: 42;
    overflow-y: auto;
}

#items-pagination {
    height: 1;
    min-height: 1;
}

#items-page-previous,
#items-page-next {
    width: auto;
}

#items-page-label {
    width: 1fr;
    text-align: center;
}

.watchlists-read-mode
.watchlists-region-items.watchlists-region-sole-centre {
    height: 1fr;
    max-height: 100%;
}

.watchlists-read-mode .watchlists-region-sole-centre #watchlists-detail-pane,
.watchlists-read-mode .watchlists-region-sole-centre #watchlists-items-pane,
.watchlists-read-mode .watchlists-region-sole-centre #items-table {
    height: 1fr;
    max-height: 100%;
}
```

`42` is the initial chrome budget (50 outer rows minus border, detail padding/title, toolbar, legend, pager); keep it only if the production geometry proves the outer region closes at 50 and all chrome remains visible. The Read-mode sole-centre selectors intentionally match or exceed the new selectors' specificity; the existing one-class solo rule alone cannot override them. Verify solo ITEMS fills before keeping the exact descendant list. Use no custom wheel event code.

- [ ] **Step 6: Regenerate the CSS bundle**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  tldw_chatbook/css/build_css.py
```

Expected: `tldw_chatbook/css/tldw_cli_modular.tcss` changes only through generation and contains the new selectors.

- [ ] **Step 7: Run geometry, section-swap, and visual-parity tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Watchlists/test_watchlists_workbench.py \
  Tests/Watchlists/test_watchlists_article_list.py \
  Tests/UI/test_watchlists_destination_shell.py \
  Tests/UI/test_destination_visual_parity_correction.py
```

Expected: PASS with rails retaining identity across section switches and Read list geometry staying within 10–50 rows.

- [ ] **Step 8: Commit Task 4**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/watchlists_workbench.py \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  tldw_chatbook/css/features/_watchlists.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss \
  Tests/Watchlists/test_watchlists_workbench.py \
  Tests/Watchlists/test_watchlists_article_list.py \
  Tests/UI/test_watchlists_destination_shell.py
git commit -m "feat(watchlists): make read centre and list independently scrollable"
```

### Task 5: Give Content a 20–50-row region and internal body scroll

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/content_pane.py:10-17,390-515`
- Modify: `tldw_chatbook/css/features/_watchlists.tcss:180-250`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Tests/Watchlists/test_watchlists_workbench.py:350-450`
- Modify: `Tests/UI/test_watchlists_content_pane.py`

- [ ] **Step 1: Write failing Content structure and geometry tests**

Add assertions that an opened article contains `#content-body-scroll` as a `VerticalScroll` and still exposes the existing `#content-body` `Static`. Against production CSS, verify:

- empty and short Content outer regions are at least 20 rows;
- long Content stops at 50 rows;
- scrolling `#content-body-scroll` reaches the last paragraph;
- `#content-actions`, `#content-footer`, and the outer Content border remain visible at both scroll extremes;
- scrolling the body does not move `#wl-centre` or either rail;
- solo Content still fills the available centre height and Restore returns to the 20–50 bounded stack.

Update the legacy `test_content_height_is_capped_and_scrollable_when_content_overflows` oracle from “outer region scrolls at 12” to “outer region caps at 50 and the body scrolls.”

- [ ] **Step 2: Run the focused tests and verify red failures**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Watchlists/test_watchlists_workbench.py \
  Tests/UI/test_watchlists_content_pane.py \
  -k 'content and (height or scroll or actions or footer or solo)'
```

Expected: FAIL because the body is a bare `Static` and the outer region owns the old 12-row scroll cap.

- [ ] **Step 3: Wrap only the rendered article body**

In `content_pane.py`:

```python
from textual.containers import Horizontal, Vertical, VerticalScroll

with VerticalScroll(id="content-body-scroll"):
    yield Static(render_for(self.item), id="content-body")
```

Keep the existing action row before the wrapper and footer after it. Keep the empty-state path unchanged except for the outer region's new minimum.

- [ ] **Step 4: Move scroll ownership from the outer region to the body**

Replace the old Content cap rules with Read-gated sizing:

```css
.watchlists-read-mode .watchlists-region-content {
    height: auto;
    min-height: 20;
    max-height: 50;
    overflow: hidden;
}

.watchlists-read-mode #watchlists-content-pane {
    height: auto;
    min-height: 0;
}

.watchlists-read-mode #content-body-scroll {
    height: auto;
    min-height: 1;
    max-height: 45;
    overflow-y: auto;
}

.watchlists-read-mode
.watchlists-region-content.watchlists-region-sole-centre {
    height: 1fr;
    max-height: 100%;
}

.watchlists-read-mode .watchlists-region-sole-centre #watchlists-content-pane,
.watchlists-read-mode .watchlists-region-sole-centre #content-body-scroll {
    height: 1fr;
    max-height: 100%;
}
```

Tune the initial `45` only from rendered geometry (50 minus region border/title plus actions/footer). Preserve the existing generic solo rule, but keep the equally/more-specific Read-mode outer and descendant overrides because the new Read selectors otherwise win the cascade and leave solo Content capped at 50/45.

- [ ] **Step 5: Regenerate CSS and run Content/integration tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  tldw_chatbook/css/build_css.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Watchlists/test_watchlists_workbench.py \
  Tests/UI/test_watchlists_content_pane.py \
  Tests/UI/test_destination_visual_parity_correction.py
```

Expected: PASS. Existing tests querying `#content-body` continue to work because its id and renderable are preserved inside the new wrapper.

- [ ] **Step 6: Commit Task 5**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/content_pane.py \
  tldw_chatbook/css/features/_watchlists.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss \
  Tests/Watchlists/test_watchlists_workbench.py \
  Tests/UI/test_watchlists_content_pane.py \
  Tests/UI/test_destination_visual_parity_correction.py
git commit -m "feat(watchlists): add bounded scrolling content reader"
```

### Task 6: Integrated verification, live TUI QA, and task closeout

**Files:**
- Modify only if verification finds a real scoped defect: files listed above
- Modify: `backlog/tasks/task-16221 - Make-Watchlists-Read-list-and-Content-independently-scrollable-with-explicit-pagination.md`
- Optionally modify: `backlog/docs/lessons-testing-evidence.md` only if this implementation produces a new, evidence-backed reusable lesson

- [ ] **Step 1: Run the focused feature suite from a clean process**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Watchlists/test_watchlists_article_list.py \
  Tests/Watchlists/test_watchlists_pagination.py \
  Tests/Watchlists/test_watchlists_workbench.py \
  Tests/Watchlists/test_watchlists_collections_screen.py \
  Tests/UI/test_watchlists_content_pane.py \
  Tests/UI/test_watchlists_items_status_filter.py \
  Tests/UI/test_watchlists_read_status.py \
  Tests/UI/test_watchlists_destination_shell.py \
  Tests/UI/test_destination_visual_parity_correction.py
```

Expected: all selected tests PASS; warnings must be understood and unrelated.

- [ ] **Step 2: Run the broader Watchlists regression suite**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Watchlists Tests/UI/test_watchlists_*.py
```

Expected: PASS. If the glob expands to an environment-only test, record its exact skip/failure and run the remaining collected Watchlists modules explicitly; do not call the task Done with an unexplained regression.

- [ ] **Step 3: Run static and generated-artifact checks**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/UI/Watchlists_Modules/article_list.py \
  tldw_chatbook/UI/Watchlists_Modules/content_pane.py \
  tldw_chatbook/UI/Watchlists_Modules/watchlists_workbench.py \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  Tests/Watchlists/test_watchlists_article_list.py \
  Tests/Watchlists/test_watchlists_pagination.py \
  Tests/Watchlists/test_watchlists_workbench.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  tldw_chatbook/css/build_css.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  tldw_chatbook/css/check_bundle_sync.py
git diff --check
```

Expected: Ruff exits 0, CSS regeneration produces no second diff, the bundle-sync checker exits 0, and `git diff --check` exits 0.

- [ ] **Step 4: Launch an isolated live production-screen harness with realistic data**

Use `_build_test_app()` so the database/profile is temporary while the mounted screen, widgets, services, and CSS are production code. Seed at least 101 items and one 80-paragraph article, then run the real destination screen:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c 'from Tests.UI.app_factory import _build_test_app; from Tests.UI.test_destination_shells import DestinationHarness; app=_build_test_app(); db=app.watchlist_bundle_service._db; sid=db.add_subscription(name="QA Feed", type="rss", source="https://qa.invalid/feed"); ids=[]; [(ids.append(db.conn.execute("INSERT INTO subscription_items (subscription_id,url,title,content,created_at) VALUES (?,?,?,?,?)", (sid, f"https://qa.invalid/{i}", f"QA article {i:03d}", "\n\n".join(f"paragraph {n:02d}" for n in range(80)) if i == 100 else f"body {i}", f"2026-08-13 {i // 60:02d}:{i % 60:02d}:00")).lastrowid) for i in range(101)]; db.conn.commit(); DestinationHarness(app, "watchlists_collections").run()'
```

If the DB transaction API rejects direct `conn` use in the current branch, use `with db.transaction() as conn:` in a temporary Python REPL; do not alter real profile data.

- [ ] **Step 5: Perform and record live nested-scroll QA at large and compact sizes**

At approximately `180x50`, then `120x36`, verify and record pass/fail for:

1. Left Watchlists and right Inspector rails stay fixed while the centre scroll reaches Content.
2. The current 50-item `ListView` scrolls internally; toolbar, legend, and pager stay visible.
3. `Previous · Page N · Next` uses offsets 0/50/100, boundary buttons disable correctly, and double activation cannot overlap loads.
4. Page change focuses the first row cursor without replacing the already open Content article; the next deliberate row move replaces it.
5. The 80-paragraph article scrolls inside Content while actions and footer stay visible.
6. Search from page 2 resets logically to page 1 and a body-only match remains visible after the backend result.
7. `z`, `Z`, Expand/Restore, `j`, `k`, mark unread, Star, Queue, and Next unread retain their established scope.
8. Short terminal layout uses the outer centre scroll instead of shrinking Read below 10 or Content below 20.

Capture a rendered frame or tmux pane at the top and bottom of each nested scroll owner if using tmux; do not substitute style-property inspection for visible evidence.

- [ ] **Step 6: Self-review the branch diff against the spec**

Run:

```bash
git diff --stat origin/dev...HEAD
git diff origin/dev...HEAD -- \
  tldw_chatbook/UI/Watchlists_Modules \
  tldw_chatbook/UI/Screens/watchlists_collections_screen.py \
  tldw_chatbook/css/features/_watchlists.tcss \
  Tests/Watchlists Tests/UI/test_watchlists_content_pane.py
```

Check explicitly for: no 51st mounted row, no total-count query, no page-change binding, no custom wheel forwarding, no Read sizing on operations tabs, no selection from the suppressed programmatic highlight, and no pin across a mismatched page key.

- [ ] **Step 7: Request code review and address only verified findings**

Use @superpowers:requesting-code-review with the approved spec and this plan. Re-run the narrowest relevant red/green tests for every accepted finding, then rerun Step 1.

- [ ] **Step 8: Update TASK-16221 only after all evidence is green**

In the task file:

- check every acceptance criterion;
- add concise Implementation Notes naming the screen-owned paging state, 51-row lookahead, page-key pin provenance, Read-gated TCSS, centre/body scroll ownership, tests, and live QA;
- retain the ADR check and link the existing ADR/spec/plan;
- add a lesson only if an actual incident from this work generalizes beyond the task;
- set status to Done with `backlog task edit 16200 -s Done` only after every DoD item is complete.

- [ ] **Step 9: Commit verification and closeout documentation**

```bash
git add backlog/tasks/'task-16221 - Make-Watchlists-Read-list-and-Content-independently-scrollable-with-explicit-pagination.md'
git commit -m "docs(watchlists): close nested scroll pagination task"
```

- [ ] **Step 10: Finish the branch**

Use @superpowers:verification-before-completion, then @superpowers:finishing-a-development-branch. Report exact test counts, live QA sizes, any skipped checks with reasons, and the branch integration options without merging or pushing unless the user authorizes it.
