# Library Conversations Pagination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every saved conversation reachable from Library through a 20-row, vertically scrollable, service-backed paged list whose filter searches the complete collection.

**Architecture:** Keep pagination presentation pure in `library_conversations_state.py`, render it with Textual's native `VerticalScroll`, and let `LibraryScreen` own the last successful page plus loading/error/request-generation fields. Reuse `ChatConversationScopeService.list_conversations()` with `query`, `limit`, and `offset`; do not add a repository, dependency, schema, or service interface.

**Tech Stack:** Python 3.11+, Textual 8.x, pytest, existing conversation scope service and CSS bundle builder.

## Global Constraints

- Page size is exactly 20 conversations.
- A submitted filter searches the complete saved-conversation collection and resets to page 1.
- The last successful page stays visible during loading and recoverable failures.
- Background Library source refreshes must not reset an actively browsed conversation page.
- Existing conversation selection, multi-select/export, preview, and Console handoff behavior remains intact.
- No database schema, dependency, configuration, or conversation-service contract change.
- ADR required: no.
- ADR path: N/A.
- ADR reason: this consumes the existing paginated conversation-service contract and changes only bounded Library view state and presentation.

---

### Task 1: Make pagination a pure conversation-canvas state contract

**Files:**
- Modify: `tldw_chatbook/Library/library_conversations_state.py`
- Test: `Tests/Library/test_library_conversations_state.py`

**Interfaces:**
- Consumes: one already-fetched page of conversation records plus `query`, `page`, `page_size`, `total_count`, `total_known`, `has_more`, `loading`, and `error_copy`.
- Produces: `LibraryConversationsCanvasState` with `range_copy`, `page_copy`, `previous_disabled`, `next_disabled`, `loading`, and `error_copy`; `build_library_conversations_state(...)` no longer filters or truncates the supplied page.

- [ ] **Step 1: Replace the old client-filter/limit tests with failing page-contract tests**

In `Tests/Library/test_library_conversations_state.py`, remove `test_limit_truncates_rows_to_max_after_sorting` and `test_match_count_reflects_filtered_set_before_limit_truncation`. Update the query tests so their input records are already the matching service page, then add:

```python
def test_middle_page_exposes_range_page_and_enabled_navigation():
    records = [
        {
            "id": f"conv-{index}",
            "title": f"Chat {index}",
            "updated_at": f"2026-07-05T{index % 12:02d}:00:00+00:00",
        }
        for index in range(20)
    ]

    state = build_library_conversations_state(
        records,
        page=2,
        page_size=20,
        total_count=47,
        total_known=True,
        has_more=True,
        now=NOW,
    )

    assert len(state.rows) == 20
    assert state.range_copy == "21-40 of 47"
    assert state.page_copy == "Page 2 of 3"
    assert state.previous_disabled is False
    assert state.next_disabled is False


def test_final_page_disables_next_without_dropping_supplied_rows():
    records = [
        {"id": f"conv-{index}", "title": f"Chat {index}"}
        for index in range(7)
    ]

    state = build_library_conversations_state(
        records,
        page=3,
        page_size=20,
        total_count=47,
        total_known=True,
        has_more=False,
        now=NOW,
    )

    assert len(state.rows) == 7
    assert state.range_copy == "41-47 of 47"
    assert state.page_copy == "Page 3 of 3"
    assert state.previous_disabled is False
    assert state.next_disabled is True


def test_empty_filtered_page_reports_zero_matches_and_page_one_of_one():
    state = build_library_conversations_state(
        [],
        query="missing",
        page=1,
        page_size=20,
        total_count=0,
        total_known=True,
        has_more=False,
        now=NOW,
    )

    assert state.status_copy == "0 matches for 'missing'"
    assert state.empty_copy == "No conversations match 'missing'."
    assert state.range_copy == "0 of 0"
    assert state.page_copy == "Page 1 of 1"
    assert state.previous_disabled is True
    assert state.next_disabled is True


def test_query_status_uses_full_service_total_not_current_page_length():
    records = [{"id": f"conv-{index}", "title": "Alpha"} for index in range(20)]

    state = build_library_conversations_state(
        records,
        query="alpha",
        page=2,
        page_size=20,
        total_count=43,
        total_known=True,
        has_more=True,
        now=NOW,
    )

    assert len(state.rows) == 20
    assert state.status_copy == "43 matches for 'alpha'"


def test_loading_and_error_preserve_rows_and_disable_navigation():
    records = [{"id": "conv-1", "title": "Last successful row"}]

    loading = build_library_conversations_state(
        records,
        page=1,
        page_size=20,
        total_count=2,
        total_known=True,
        has_more=True,
        loading=True,
        now=NOW,
    )
    failed = build_library_conversations_state(
        records,
        page=1,
        page_size=20,
        total_count=2,
        total_known=True,
        has_more=True,
        error_copy="Couldn't load conversations. Try again.",
        now=NOW,
    )

    assert [row.conversation_id for row in loading.rows] == ["conv-1"]
    assert loading.status_copy == "Loading conversations…"
    assert loading.previous_disabled is True
    assert loading.next_disabled is True
    assert [row.conversation_id for row in failed.rows] == ["conv-1"]
    assert failed.status_copy == "Couldn't load conversations. Try again."
    assert failed.empty_copy == ""


def test_unknown_total_disables_next_without_explicit_has_more():
    state = build_library_conversations_state(
        [{"id": "conv-1", "title": "One"}],
        page=1,
        page_size=20,
        total_count=1,
        total_known=False,
        has_more=False,
        now=NOW,
    )

    assert state.range_copy == "1-1"
    assert state.page_copy == "Page 1"
    assert state.next_disabled is True


def test_initial_failure_does_not_claim_the_library_is_empty():
    state = build_library_conversations_state(
        [],
        page=1,
        page_size=20,
        total_count=0,
        total_known=False,
        has_more=False,
        error_copy="Couldn't load conversations. Try again.",
        now=NOW,
    )

    assert state.status_copy == "Couldn't load conversations. Try again."
    assert state.empty_copy == ""
```

- [ ] **Step 2: Run the state tests and verify RED**

Run:

```bash
pytest Tests/Library/test_library_conversations_state.py -q
```

Expected: failures because the builder does not accept page metadata and the canvas state lacks pager fields.

- [ ] **Step 3: Implement the minimum pure pagination contract**

In `library_conversations_state.py`:

1. Add these fields to `LibraryConversationsCanvasState`:

```python
range_copy: str
page_copy: str
previous_disabled: bool
next_disabled: bool
loading: bool = False
error_copy: str = ""
```

2. Change the builder signature to:

```python
def build_library_conversations_state(
    records: Sequence[Mapping[str, Any]],
    *,
    query: str = "",
    selected_id: str = "",
    now: datetime | None = None,
    page: int = 1,
    page_size: int = 20,
    total_count: int | None = None,
    total_known: bool = True,
    has_more: bool = False,
    loading: bool = False,
    error_copy: str = "",
    select_mode: bool = False,
    selected_ids: frozenset[str] = frozenset(),
) -> LibraryConversationsCanvasState:
```

3. Normalize `page` and `page_size` to at least 1. Treat the supplied records as the already-filtered current page: keep record validation and recency sorting, but delete the local query-filter block and `entries[:limit]` truncation.

4. Derive pager copy with these rules:

```python
known_total = max(0, int(total_count or 0))
page_count = max(1, (known_total + page_size - 1) // page_size)
resolved_page = min(max(1, page), page_count) if total_known else max(1, page)
start = (resolved_page - 1) * page_size + 1 if entries else 0
end = (resolved_page - 1) * page_size + len(entries) if entries else 0
range_copy = (
    f"{start}-{end} of {known_total}"
    if total_known and entries
    else f"0 of {known_total}"
    if total_known
    else f"{start}-{end}"
    if entries
    else "0"
)
page_copy = (
    f"Page {resolved_page} of {page_count}"
    if total_known
    else f"Page {resolved_page}"
)
```

5. Status priority is error, loading, submitted-query count, then existing empty/no-status behavior. Navigation is disabled while loading; otherwise Previous is disabled on page 1 and Next is disabled unless `has_more` is true.

- [ ] **Step 4: Run the state tests and verify GREEN**

Run:

```bash
pytest Tests/Library/test_library_conversations_state.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the pure state contract**

```bash
git add tldw_chatbook/Library/library_conversations_state.py Tests/Library/test_library_conversations_state.py
git commit -m "feat(library): add conversation pager state"
```

---

### Task 2: Render a scrollable page and fixed pager controls

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_conversations_canvas.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_library_shell.py`

**Interfaces:**
- Consumes: the pager fields added to `LibraryConversationsCanvasState` in Task 1.
- Produces: `#library-conversations-list` as a real `VerticalScroll`, `#library-conversations-previous`, `#library-conversations-page-status`, and `#library-conversations-next`.

- [ ] **Step 1: Write failing render and overflow tests**

Add `VerticalScroll` to the Textual container imports in `Tests/UI/test_library_shell.py`, then add:

```python
@pytest.mark.asyncio
async def test_library_conversations_page_renders_scrollable_rows_and_pager():
    app = _build_test_app()
    _seed_conversations(
        app,
        [
            {
                "conversation_id": f"chat-{index:02d}",
                "title": f"Conversation {index:02d}",
                "updated_at": f"2026-06-{28 - index:02d}T00:00:00Z",
            }
            for index in range(20)
        ],
    )
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-row-19")

        scroll = screen.query_one("#library-conversations-list", VerticalScroll)
        assert scroll.max_scroll_y > 0
        assert screen.query_one("#library-conversations-previous", Button).disabled
        assert str(screen.query_one("#library-conversations-page-status").renderable) == (
            "1-20 of 20 · Page 1 of 1"
        )
        assert screen.query_one("#library-conversations-next", Button).disabled

        screen.query_one("#library-conversation-row-19", Button).focus()
        await pilot.pause()
        assert scroll.scroll_y > 0
```

Add this fixed-placement test:

```python
@pytest.mark.asyncio
async def test_library_conversations_pager_controls_remain_outside_scroll_viewport():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversations-next")

        scroll = screen.query_one("#library-conversations-list", VerticalScroll)
        pager = screen.query_one("#library-conversations-pager")
        assert pager.parent is scroll.parent
        assert pager not in list(scroll.children)
```

- [ ] **Step 2: Run the Pilot tests and verify RED**

Run:

```bash
pytest Tests/UI/test_library_shell.py -q -k "conversations_page_renders_scrollable or conversations_pager_controls_remain"
```

Expected: failures because the list is a plain `Vertical` and pager widgets do not exist.

- [ ] **Step 3: Use native Textual scrolling and render the pager**

In `library_conversations_canvas.py`:

1. Import `VerticalScroll` from `textual.containers`.
2. Replace `conversation_list = Vertical(...)` with:

```python
conversation_list = VerticalScroll(id="library-conversations-list")
```

3. Immediately after the list, yield a fixed pager:

```python
with Horizontal(id="library-conversations-pager", classes="ds-toolbar"):
    previous = Button(
        "Previous",
        id="library-conversations-previous",
        classes="library-canvas-action",
        compact=True,
    )
    previous.disabled = self.canvas.previous_disabled
    yield previous
    yield Static(
        f"{self.canvas.range_copy} · {self.canvas.page_copy}",
        id="library-conversations-page-status",
        markup=False,
    )
    following = Button(
        "Next",
        id="library-conversations-next",
        classes="library-canvas-action",
        compact=True,
    )
    following.disabled = self.canvas.next_disabled
    yield following
```

Keep the preview after this pager, unchanged.

In `_agentic_terminal.tcss`, add only the necessary layout and visible native scrollbar styling:

```css
#library-conversations-canvas {
    height: 100%;
    min-height: 0;
}

#library-conversations-list {
    height: 1fr;
    min-height: 4;
    overflow-y: auto;
    overflow-x: hidden;
    scrollbar-size: 1 1;
    scrollbar-background: $ds-surface-panel;
    scrollbar-color: $ds-text-muted;
    scrollbar-color-hover: $ds-grid-line;
    scrollbar-color-active: $ds-action-focus;
}

#library-conversations-pager {
    height: 3;
    min-height: 3;
    align-horizontal: center;
}

#library-conversations-page-status {
    width: auto;
    height: 3;
    content-align: center middle;
    color: $ds-text-muted;
}
```

- [ ] **Step 4: Regenerate the CSS bundle**

Run:

```bash
python tldw_chatbook/css/build_css.py
```

Expected: `tldw_chatbook/css/tldw_cli_modular.tcss` is regenerated with the new source selectors and no manual bundle edits.

- [ ] **Step 5: Run the Pilot tests and verify GREEN**

Run:

```bash
pytest Tests/UI/test_library_shell.py -q -k "conversations_page_renders_scrollable or conversations_pager_controls_remain"
```

Expected: both tests pass.

- [ ] **Step 6: Commit the scrollable canvas**

```bash
git add tldw_chatbook/Widgets/Library/library_conversations_canvas.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_library_shell.py
git commit -m "feat(library): make conversation page scrollable"
```

---

### Task 3: Load, search, and navigate complete conversation pages

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Modify: `Tests/UI/test_destination_shells.py`
- Test: `Tests/UI/test_library_shell.py`

**Interfaces:**
- Consumes: `ChatConversationScopeService.list_conversations(mode="local", scope_type="all", query=query or None, limit=20, offset=(page - 1) * 20)` returning `items` and pagination metadata.
- Produces: `_start_library_conversation_page_request(page, query, *, refocus_filter=False) -> None` and `_load_library_conversation_page(page, query, generation, *, refocus_filter=False) -> None`; button and filter handlers call the starter rather than mutating page records directly.

- [ ] **Step 1: Make the shared test service honor query/limit/offset**

Change `StaticLibraryConversationScopeService.list_conversations` in `Tests/UI/test_destination_shells.py` to:

```python
async def list_conversations(self, **kwargs):
    self.calls.append(kwargs)
    query = str(kwargs.get("query") or "").casefold()
    matching = [
        record
        for record in self.conversations
        if not query or query in str(record.get("title") or "").casefold()
    ]
    offset = max(0, int(kwargs.get("offset", 0)))
    limit = max(0, int(kwargs.get("limit", len(matching))))
    page = matching[offset : offset + limit]
    return {
        "items": page,
        "pagination": {
            "total": len(matching),
            "has_more": offset + len(page) < len(matching),
        },
    }
```

- [ ] **Step 2: Write failing service-backed paging and full-search Pilot tests**

Add this stable-order helper beside `_two_conversations()` in `Tests/UI/test_library_shell.py`:

```python
def _conversation_records(count: int) -> list[dict[str, object]]:
    return [
        {
            "conversation_id": f"chat-{index + 1:03d}",
            "title": f"Conversation {index + 1:03d}",
            "message_count": index + 1,
        }
        for index in range(count)
    ]
```

Records without timestamps retain their service order because Python's sort is stable. Then add:

```python
@pytest.mark.asyncio
async def test_library_conversations_next_loads_second_service_page():
    app = _build_test_app()
    _seed_conversations(app, _conversation_records(45))
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversations-next")

        screen.query_one("#library-conversations-next", Button).press()
        for _ in range(100):
            if screen._library_conversation_page == 2:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Conversation page 2 never loaded.")

        assert app.chat_conversation_scope_service.calls[-1] == {
            "mode": "local",
            "scope_type": "all",
            "query": None,
            "limit": 20,
            "offset": 20,
        }
        assert len(screen.query(".library-conversation-row")) == 20
        assert str(screen.query_one("#library-conversations-page-status").renderable) == (
            "21-40 of 45 · Page 2 of 3"
        )


@pytest.mark.asyncio
async def test_library_conversations_filter_searches_beyond_first_page_and_resets_page():
    app = _build_test_app()
    records = _conversation_records(45)
    records[-1] = {**records[-1], "title": "Needle outside first page"}
    _seed_conversations(app, records)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversations-filter")

        screen._library_conversation_page = 2
        field = screen.query_one("#library-conversations-filter", Input)
        field.value = "needle"
        field.focus()
        await pilot.press("enter")
        for _ in range(100):
            rows = list(screen.query(".library-conversation-row"))
            if len(rows) == 1 and "Needle" in str(rows[0].label):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Full-dataset conversation filter never landed.")

        assert screen._library_conversation_page == 1
        assert app.chat_conversation_scope_service.calls[-1]["query"] == "needle"
        assert app.chat_conversation_scope_service.calls[-1]["offset"] == 0
        assert str(screen.query_one("#library-conversations-status").renderable) == (
            "1 match for 'needle'"
        )
```

- [ ] **Step 3: Write failing preservation, failure, and stale-response tests**

Add focused tests using small local async fakes declared inside each test:

```python
@pytest.mark.asyncio
async def test_library_conversation_initial_failure_keeps_filter_for_retry():
    app = _build_test_app()
    _seed_conversations(app, _conversation_records(2))

    class FailingConversationService:
        async def list_conversations(self, **kwargs):
            raise RuntimeError("offline")

    app.chat_conversation_scope_service = FailingConversationService()
    screen = LibraryScreen(app)
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_CONVERSATIONS
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active = _active_library_screen(host)
        await _wait_for_selector(active, pilot, "#library-conversations-filter")

        assert active.query_one("#library-conversations-canvas")
        assert active.query_one("#library-conversations-previous", Button).disabled
        assert active.query_one("#library-conversations-next", Button).disabled
        assert "load" in str(
            active.query_one("#library-conversations-status").renderable
        ).lower()

        app.chat_conversation_scope_service = StaticLibraryConversationScopeService(
            _conversation_records(2)
        )
        field = active.query_one("#library-conversations-filter", Input)
        field.focus()
        await pilot.press("enter")
        await _wait_for_selector(active, pilot, "#library-conversation-row-1")
        assert len(active.query(".library-conversation-row")) == 2


@pytest.mark.asyncio
async def test_library_conversation_page_failure_keeps_last_successful_rows():
    app = _build_test_app()
    _seed_conversations(app, _conversation_records(25))
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-row-19")
        old_ids = [
            button.conversation_id
            for button in screen.query(".library-conversation-row")
        ]

        async def fail(**kwargs):
            raise RuntimeError("offline")

        app.chat_conversation_scope_service.list_conversations = fail
        screen.query_one("#library-conversations-next", Button).press()
        for _ in range(100):
            if screen._library_conversation_error:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Conversation load error never rendered.")

        assert [
            button.conversation_id
            for button in screen.query(".library-conversation-row")
        ] == old_ids
        assert "Couldn't load conversations" in str(
            screen.query_one("#library-conversations-status").renderable
        )


@pytest.mark.asyncio
async def test_library_conversation_page_reloads_new_final_page_when_total_shrinks():
    app = _build_test_app()
    _seed_conversations(app, _conversation_records(45))
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversations-next")

        service = app.chat_conversation_scope_service
        service.conversations = service.conversations[:25]
        screen._library_conversation_request_generation += 1
        generation = screen._library_conversation_request_generation
        await screen._load_library_conversation_page(3, "", generation)

        assert [call["offset"] for call in service.calls[-2:]] == [40, 20]
        assert screen._library_conversation_page == 2
        assert len(screen._conversation_records()) == 5
        assert screen._library_conversation_has_more is False


@pytest.mark.asyncio
async def test_stale_conversation_page_response_cannot_replace_newer_filter():
    app = _build_test_app()
    _seed_conversations(app, _conversation_records(25))
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversations-filter")

        old_started = threading.Event()
        release_old = threading.Event()

        def controlled(**kwargs):
            if kwargs.get("query") == "old":
                old_started.set()
                assert release_old.wait(timeout=5)
                return {"items": [{"id": "old", "title": "Old"}], "pagination": {"total": 1}}
            return {"items": [{"id": "new", "title": "New"}], "pagination": {"total": 1}}

        app.chat_conversation_scope_service.list_conversations = controlled
        screen._library_conversation_request_generation = 1
        old_request = asyncio.create_task(
            screen._load_library_conversation_page(1, "old", 1)
        )
        assert await asyncio.to_thread(old_started.wait, 5)

        screen._library_conversation_request_generation = 2
        await screen._load_library_conversation_page(1, "new", 2)
        release_old.set()
        await old_request

        assert [record["id"] for record in screen._conversation_records()] == ["new"]
```

Update the existing snapshot carry-over test: seed or load page 2, apply a new general source snapshot, and assert `_conversation_records()` plus `_library_conversation_page` remain unchanged. This replaces the old prepend-to-`_local_source_records` assertion because page records now have their own owner.

Update `test_library_shell_restored_conversation_query_renders_on_first_paint` so it proves service-backed restore rather than client filtering:

```python
@pytest.mark.asyncio
async def test_library_shell_restored_conversation_query_is_reissued_to_service():
    app = _build_test_app()
    records = _conversation_records(25)
    records[-1] = {**records[-1], "title": "Quarterly outside first page"}
    _seed_conversations(app, records)

    screen = LibraryScreen(app)
    screen.restore_state(
        {
            "library_selected_row_id": LIBRARY_ROW_BROWSE_CONVERSATIONS,
            "library_conversation_query": "quarterly",
        }
    )
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active = _active_library_screen(host)
        for _ in range(100):
            rows = list(active.query(".library-conversation-row"))
            if len(rows) == 1 and "Quarterly" in str(rows[0].label):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Restored conversation filter was not reissued.")

        assert app.chat_conversation_scope_service.calls[-1]["query"] == "quarterly"
        assert app.chat_conversation_scope_service.calls[-1]["offset"] == 0
        assert active._library_conversation_page == 1
        assert active.query_one("#library-conversations-filter", Input).value == (
            "quarterly"
        )


@pytest.mark.asyncio
async def test_restored_conversation_query_never_flashes_unfiltered_snapshot_rows():
    app = _build_test_app()
    unfiltered = _conversation_records(25)
    filtered_started = threading.Event()
    release_filtered = threading.Event()

    class ControlledConversationService:
        def __init__(self):
            self.calls = []

        def list_conversations(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("query"):
                filtered_started.set()
                assert release_filtered.wait(timeout=5)
                return {
                    "items": [{"conversation_id": "needle", "title": "Quarterly"}],
                    "pagination": {"total": 1, "has_more": False},
                }
            return {
                "items": unfiltered[:20],
                "pagination": {"total": 25, "has_more": True},
            }

    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = ControlledConversationService()
    screen = LibraryScreen(app)
    screen.restore_state(
        {
            "library_selected_row_id": LIBRARY_ROW_BROWSE_CONVERSATIONS,
            "library_conversation_query": "quarterly",
        }
    )
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active = _active_library_screen(host)
        assert await asyncio.to_thread(filtered_started.wait, 5)
        await _wait_for_selector(active, pilot, "#library-conversations-filter")

        assert active._library_conversation_loading is True
        assert not active.query(".library-conversation-row")
        assert active.query_one("#library-conversations-filter", Input).value == (
            "quarterly"
        )

        release_filtered.set()
        await _wait_for_selector(active, pilot, "#library-conversation-row-0")
        rows = list(active.query(".library-conversation-row"))
        assert len(rows) == 1
        assert "Quarterly" in str(rows[0].label)
```

- [ ] **Step 4: Run the new paging tests and verify RED**

Run:

```bash
pytest Tests/UI/test_library_shell.py -q -k "conversations_next_loads or conversations_filter_searches_beyond or conversation_initial_failure or conversation_page_failure or conversation_page_reloads or stale_conversation_page or snapshot_replace_carries or restored_conversation_query or restored_conversation_query_never_flashes"
```

Expected: failures because the screen has no dedicated page loader or paging handlers and filter submission still filters the snapshot locally.

- [ ] **Step 5: Add minimal dedicated page fields and seed them once**

In `LibraryScreen.__init__`, directly after the existing conversation query/selection fields, add:

```python
self._library_conversation_page_records: tuple[Mapping[str, Any], ...] = ()
self._library_conversation_page = 1
self._library_conversation_page_size = 20
self._library_conversation_total = 0
self._library_conversation_total_known = True
self._library_conversation_has_more = False
self._library_conversation_page_loaded = False
self._library_conversation_loading = False
self._library_conversation_error = ""
self._library_conversation_request_generation = 0
```

Change `_conversation_records()` so the unfiltered snapshot is a fallback only before any query/loading/error-specific page state exists:

```python
def _conversation_records(self) -> tuple[Mapping[str, Any], ...]:
    if (
        self._library_conversation_page_loaded
        or self._library_conversation_query
        or self._library_conversation_loading
        or self._library_conversation_error
    ):
        return tuple(self._library_conversation_page_records)
    return tuple(self._local_source_records.get("conversations", ()))
```

This is the no-flash boundary: a restored query deliberately prevents the unfiltered snapshot from seeding page state, and the accessor must not reach around that decision while the service-backed filter is pending.

In `_apply_local_source_snapshot`, before replacing `_local_source_records`, seed the dedicated page only when `lookup_error is None`, `_library_conversation_page_loaded` is false, and there is no restored conversation query:

```python
conversation_records = tuple(records.get("conversations", ()))
if (
    lookup_error is None
    and not self._library_conversation_page_loaded
    and not self._library_conversation_query
):
    self._library_conversation_page_records = conversation_records
    self._library_conversation_page = 1
    self._library_conversation_total = max(0, int(counts.get("conversations", 0)))
    self._library_conversation_total_known = bool(
        total_known.get("conversations", True)
    )
    self._library_conversation_has_more = (
        len(conversation_records) < self._library_conversation_total
        if self._library_conversation_total_known
        else False
    )
    self._library_conversation_page_loaded = True
elif lookup_error is not None and not self._library_conversation_page_loaded:
    self._library_conversation_total_known = False
    self._library_conversation_has_more = False
    self._library_conversation_loading = False
    self._library_conversation_error = (
        "Couldn't load conversations. Submit the filter to try again."
    )
```

Change the initial snapshot conversation limit from 50 to the same 20-row constant; define `LIBRARY_CONVERSATION_PAGE_SIZE = 20` beside `LIBRARY_SOURCE_PAGE_SIZES` and use it in both places rather than duplicating the literal.

Delete `_carry_selected_conversation_into_snapshot`: the dedicated page now survives source snapshot replacement without carry/merge logic. Update the deep-link insertion in `_open_library_item_by_id` to prepend the fetched row to `_library_conversation_page_records` and truncate to `LIBRARY_CONVERSATION_PAGE_SIZE`, preserving AC #2's maximum.

In `compose_content`, narrow the generic `_library_lookup_error` replacement branch so it still owns Media and database Notes but not Conversations. The Conversations branch must always yield `LibraryConversationsCanvas`; its state renders `_library_conversation_error`, keeping the filter and disabled pager mounted for retry:

```python
elif (
    is_local_snapshot_canvas
    and self._library_lookup_error
    and shell.canvas_kind != "conversations"
):
    yield Static(
        self._library_lookup_error,
        id="library-canvas-error",
        classes="destination-purpose",
        markup=False,
    )
```

At the end of `_refresh_local_source_snapshot`, after `_apply_local_source_snapshot(...)`, reissue a restored filter exactly once:

```python
if (
    self._library_selected_row_id == LIBRARY_ROW_BROWSE_CONVERSATIONS
    and self._library_conversation_query
    and not self._library_conversation_page_loaded
    and self._library_conversation_request_generation == 0
):
    self._start_library_conversation_page_request(
        1,
        self._library_conversation_query,
    )
```

This also lets a restored query recover Conversations when another source made the broad snapshot fail. Do not start it from the cache-only `_apply_local_source_snapshot` call during `on_mount`; the unconditional real refresh immediately following that cache paint owns the one request.

- [ ] **Step 6: Implement the page request and stale-result guard**

Add these methods near `_build_library_conversations_state`:

```python
def _start_library_conversation_page_request(
    self, page: int, query: str, *, refocus_filter: bool = False
) -> None:
    self._library_conversation_request_generation += 1
    generation = self._library_conversation_request_generation
    normalized_query = self._safe_text(query, max_length=200)
    self._library_conversation_query = normalized_query
    self._library_conversation_loading = True
    self._library_conversation_error = ""
    self._library_conversations_select_mode = False
    self._library_conversations_row_selection.clear()
    _sync_library_canvas(self, "conversations")
    self.run_worker(
        self._load_library_conversation_page(
            page, normalized_query, generation, refocus_filter=refocus_filter
        ),
        exclusive=True,
        group="library_conversation_page",
    )


async def _load_library_conversation_page(
    self,
    page: int,
    query: str,
    generation: int,
    *,
    refocus_filter: bool = False,
) -> None:
    service = getattr(
        self.app_instance, "chat_conversation_scope_service", None
    )
    list_conversations = getattr(service, "list_conversations", None)
    if not callable(list_conversations):
        if generation == self._library_conversation_request_generation:
            self._library_conversation_loading = False
            self._library_conversation_error = (
                "Couldn't load conversations. Try Previous, Next, or submit the filter again."
            )
            _sync_library_canvas(self, "conversations")
            if refocus_filter:
                self.call_after_refresh(self._focus_library_conversations_filter)
        return

    requested_page = max(1, int(page))
    normalized_query = self._safe_text(query, max_length=200)
    try:
        result = await self._run_library_service_call(
            list_conversations,
            mode="local",
            scope_type="all",
            query=normalized_query or None,
            limit=LIBRARY_CONVERSATION_PAGE_SIZE,
            offset=(requested_page - 1) * LIBRARY_CONVERSATION_PAGE_SIZE,
            isolate_in_worker=True,
        )
    except Exception:
        if generation != self._library_conversation_request_generation:
            return
        logger.opt(exception=True).warning("Failed to load Library conversations page.")
        self._library_conversation_loading = False
        self._library_conversation_error = (
            "Couldn't load conversations. Try Previous, Next, or submit the filter again."
        )
        _sync_library_canvas(self, "conversations")
        if refocus_filter:
            self.call_after_refresh(self._focus_library_conversations_filter)
        return

    if generation != self._library_conversation_request_generation:
        return

    records, total, total_known = self._response_records_and_count(result)
    pagination = result.get("pagination") if isinstance(result, Mapping) else None
    explicit_has_more = (
        bool(pagination.get("has_more")) if isinstance(pagination, Mapping) else False
    )
    has_more = (
        (requested_page - 1) * LIBRARY_CONVERSATION_PAGE_SIZE + len(records) < total
        if total_known
        else explicit_has_more
    )
    page_count = max(
        1,
        (total + LIBRARY_CONVERSATION_PAGE_SIZE - 1)
        // LIBRARY_CONVERSATION_PAGE_SIZE,
    )
    if total_known and requested_page > page_count:
        await self._load_library_conversation_page(
            page_count,
            normalized_query,
            generation,
            refocus_filter=refocus_filter,
        )
        return
    self._library_conversation_page_records = records
    self._library_conversation_page = requested_page
    self._library_conversation_total = total
    self._library_conversation_total_known = total_known
    self._library_conversation_has_more = has_more
    self._library_conversation_page_loaded = True
    self._library_conversation_query = normalized_query
    self._library_conversation_loading = False
    self._library_conversation_error = ""
    self._selected_conversation_id = ""
    _sync_library_canvas(self, "conversations")
    if refocus_filter:
        self.call_after_refresh(self._focus_library_conversations_filter)
```

Do not mutate the last successful records, page, or total before the successful response lands. The submitted query is applied at request start so the filter input does not revert during the loading recompose; if the request fails, the attempted query remains available for correction/retry while the old rows stay visible. The `refocus_filter` flag exists solely to restore the submitted field after both the loading and completion recomposes; page-button requests leave it false so paging does not steal focus.

- [ ] **Step 7: Wire the builder and handlers**

Pass all dedicated page fields from `_build_library_conversations_state()` to `build_library_conversations_state(...)`.

Replace `handle_library_conversations_filter_submitted`'s local recompose with:

```python
event.stop()
query = self._safe_text(event.value, max_length=200)
self._start_library_conversation_page_request(1, query, refocus_filter=True)
```

Add button handlers:

```python
@on(Button.Pressed, "#library-conversations-previous")
def handle_library_conversations_previous(self, event: Button.Pressed) -> None:
    event.stop()
    if self._library_conversation_loading or self._library_conversation_page <= 1:
        return
    self._start_library_conversation_page_request(
        self._library_conversation_page - 1,
        self._library_conversation_query,
    )


@on(Button.Pressed, "#library-conversations-next")
def handle_library_conversations_next(self, event: Button.Pressed) -> None:
    event.stop()
    if self._library_conversation_loading or not self._library_conversation_has_more:
        return
    self._start_library_conversation_page_request(
        self._library_conversation_page + 1,
        self._library_conversation_query,
    )
```

- [ ] **Step 8: Run the paging tests and verify GREEN**

Run:

```bash
pytest Tests/UI/test_library_shell.py -q -k "conversations_next_loads or conversations_filter_searches_beyond or conversation_initial_failure or conversation_page_failure or conversation_page_reloads or stale_conversation_page or snapshot_replace_carries or restored_conversation_query or restored_conversation_query_never_flashes"
```

Expected: all selected tests pass.

- [ ] **Step 9: Run all conversation-specific Library tests**

Run:

```bash
pytest Tests/Library/test_library_conversations_state.py Tests/Library/test_library_conversations_visibility.py Tests/UI/test_library_shell.py -q -k "conversation"
```

Expected: all selected tests pass; update any old assertions that describe client-side filtering or a 50/75-row cap to the new service-backed contract.

- [ ] **Step 10: Commit service-backed paging**

```bash
git add tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_destination_shells.py Tests/UI/test_library_shell.py
git commit -m "feat(library): page and search all conversations"
```

---

### Task 4: Document and verify the completed behavior

**Files:**
- Modify: `Docs/User_Guide/library/media-and-conversations.md`
- Modify: `backlog/tasks/task-15703 - Make-Library-conversations-list-scrollable-and-paginated.md`

**Interfaces:**
- Consumes: the final verified UI behavior.
- Produces: accurate user instructions and complete task evidence.

- [ ] **Step 1: Update the conversation user guide**

In `Docs/User_Guide/library/media-and-conversations.md`:

- Change the filter row to say it searches the complete collection and returns to page 1.
- Add Previous/Next and the `21-40 of 137 · Page 2 of 7` status to the Conversations controls table.
- State that the 20 rows on each page live in a scrollable viewport.
- Remove the Quirks entry claiming Conversations shows at most 75 rows and the filter only matches loaded records.
- Add keyboard guidance: Tab/Shift+Tab reaches rows and pager controls; focused rows scroll into view; Enter activates them.

- [ ] **Step 2: Run focused static and automated verification**

Run:

```bash
ruff check tldw_chatbook/Library/library_conversations_state.py tldw_chatbook/Widgets/Library/library_conversations_canvas.py tldw_chatbook/UI/Screens/library_screen.py Tests/Library/test_library_conversations_state.py Tests/UI/test_destination_shells.py Tests/UI/test_library_shell.py
python tldw_chatbook/css/build_css.py
pytest Tests/Library/test_library_conversations_state.py Tests/Library/test_library_conversations_visibility.py -q
pytest Tests/UI/test_library_shell.py -q -k "conversation"
git diff --check
```

Expected: ruff clean, CSS bundle current, all focused tests pass, and no whitespace errors.

- [ ] **Step 3: Run the broader Library regression gate**

Run:

```bash
pytest Tests/Library/ -q
pytest Tests/UI/test_library_shell.py -q
```

Expected: both suites pass. If an unrelated known flaky test fails, rerun it alone and record both outputs; do not report the suite green without that evidence.

- [ ] **Step 4: Perform isolated live TUI verification**

Create a scratch profile with `TLDW_TEST_MODE=1`, `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, and `TLDW_CONFIG_PATH` all pointing under one `mktemp -d` directory before importing or launching the app. Seed at least 45 conversations into the scratch data directory, launch the TUI, and verify at 100x30 and 170x48:

1. The first page shows 20 rows and a visible scrollbar.
2. Mouse wheel and keyboard focus reach row 20.
3. Next reaches rows 21-40; final page shows 41-45 and disables Next.
4. A filter unique to row 45 finds it from page 1.
5. Clearing the filter returns to unfiltered page 1.

Capture terminal text or screenshots plus the exact scratch launch command. Do not use the real profile or database.

- [ ] **Step 5: Complete TASK-15703 only after all evidence is green**

Update every acceptance criterion to `[x]`. Add Implementation Notes naming the state, widget, screen, CSS, tests, docs, ADR decision, and exact verification results. Then run:

```bash
backlog task edit 15703 -s Done
backlog task 15703 --plain
```

Expected: TASK-15703 is Done, all five criteria are checked, the implementation plan remains present, and Implementation Notes contain the evidence.

- [ ] **Step 6: Commit documentation and task closure**

```bash
git add Docs/User_Guide/library/media-and-conversations.md 'backlog/tasks/task-15703 - Make-Library-conversations-list-scrollable-and-paginated.md'
git commit -m "docs(library): document conversation paging"
```
