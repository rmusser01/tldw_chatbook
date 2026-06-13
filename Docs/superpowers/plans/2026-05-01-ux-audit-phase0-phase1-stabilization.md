# UX Audit Phase 0-1 Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the current P0/P1 UX blockers in the app shell and runtime-stability layer before any further orientation or polish work.

**Architecture:** This tranche is split into seven independently shippable tasks. Phase 0 establishes a repeatable shell/navigation smoke gate and fixes the Chatbooks shell escape bug. Phase 1 fixes runtime trust failures one seam at a time: Ingest regression coverage, Study response normalization, Chat state-save safety, Search/RAG worker-thread UI mutation, and Search primary-action reachability.

**Tech Stack:** Python 3.12, Textual, pytest, pytest-asyncio, SQLite-backed service tests, existing `Tests.textual_test_utils.widget_pilot`.

---

## Source Documents

- `Docs/superpowers/plans/2026-05-01-ux-audit-remediation.md`
- `Docs/superpowers/specs/2026-04-20-ux-rescue-audit-design.md`
- `Docs/superpowers/specs/2026-04-21-chat-first-shell-label-cleanup-design.md`
- `Docs/superpowers/specs/2026-04-21-use-in-chat-handoffs-design.md`

## Scope Boundary

Do this:

- Fix Phase 0 shell/smoke and Phase 1 runtime/layout issues only.
- Preserve screen route IDs.
- Preserve the already-merged `Use in Chat` architecture.
- Add failing tests before production changes.
- Commit after each task or after each tightly coupled test/fix pair.

Do not do this in this tranche:

- Do not implement Phase 2 provider readiness or first-run Chat orientation.
- Do not implement Phase 3/4 handoff clear/dismiss or smoke replay.
- Do not rename `LLM` or `S/TT/S`; that belongs to Phase 5.
- Do not perform broad visual redesign.

## File Responsibility Map

- `Tests/UI/test_ux_audit_smoke.py`: new route-level UX smoke gate that proves top-level destinations keep shared navigation and can escape back to Chat.
- `Tests/UI/test_screen_navigation.py`: existing screen routing tests; extend with class-contract assertions for routed screens.
- `Tests/UI/test_chatbooks_screen_server_actions.py`: existing Chatbooks focused tests; update for `BaseAppScreen` construction and shell rendering.
- `tldw_chatbook/UI/Screens/chatbooks_screen.py`: convert Chatbooks to the shared `BaseAppScreen` contract.
- `Tests/UI/test_ingestion_ui_redesigned.py`: add direct regression that server-mode source type default is a valid allowed option.
- `Tests/UI/test_media_ingestion_tab_integration.py`: keep existing local/server screen coverage green.
- `tldw_chatbook/Widgets/Media/media_ingestion_source_panel.py`: modify only if the new direct regression fails.
- `Tests/Study_Interop/test_quiz_scope_service.py`: add empty mapping response coverage.
- `tldw_chatbook/Study_Interop/quiz_scope_service.py`: normalize list response shapes explicitly.
- `Tests/UI/test_chat_screen_state.py`: add regression for direct chat-log save path.
- `tldw_chatbook/UI/Screens/chat_screen.py`: define fallback selectors outside fallback-only branches.
- `Tests/UI/test_search_rag_window.py`: add thread-seam and primary-action reachability coverage.
- `tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py`: separate off-thread data loading from on-thread widget mutation and adjust layout only if the reachability regression fails.

## Task 1: Shell Contract And UX Smoke Gate

**Files:**

- Create: `Tests/UI/test_ux_audit_smoke.py`
- Modify: `Tests/UI/test_screen_navigation.py`
- Read: `tldw_chatbook/app.py`
- Read: `tldw_chatbook/UI/Navigation/base_app_screen.py`

- [ ] **Step 1: Add routed-screen contract coverage**

Add a test to `Tests/UI/test_screen_navigation.py` that imports the routed screen classes from `_resolve_screen_navigation_target()` and verifies every primary route uses `BaseAppScreen`.

Use this route set:

```python
PRIMARY_ROUTE_IDS = [
    "chat",
    "notes",
    "media",
    "ingest",
    "search",
    "study",
    "ccp",
    "chatbooks",
]
```

Test shape:

```python
def test_primary_routed_screens_use_base_app_screen():
    app = _build_test_app()

    offenders = []
    for route_id in PRIMARY_ROUTE_IDS:
        _screen_name, _tab_id, screen_class = app._resolve_screen_navigation_target(route_id)
        if screen_class is None or not issubclass(screen_class, BaseAppScreen):
            offenders.append((route_id, screen_class))

    assert offenders == []
```

- [ ] **Step 2: Run contract test and verify it fails on Chatbooks**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_screen_navigation.py::test_primary_routed_screens_use_base_app_screen --tb=short
```

Expected: FAIL showing `chatbooks` maps to a class that is not a `BaseAppScreen`.

- [ ] **Step 3: Add reusable UX smoke test for navigation persistence**

Create `Tests/UI/test_ux_audit_smoke.py`.

The test should build the normal test app using the same patch strategy as `Tests/UI/test_screen_navigation.py`, launch the app with `run_test()`, navigate through the core route IDs, and assert the shared nav remains mounted after each route.

Minimum route order:

```python
SMOKE_ROUTE_IDS = [
    "chat",
    "notes",
    "media",
    "ingest",
    "search",
    "study",
    "ccp",
    "chatbooks",
]
```

Core assertion:

```python
assert app.screen.query_one("#nav-chat") is not None
assert app.screen.query_one("#nav-chatbooks") is not None
```

Prefer posting `NavigateToScreen(route_id)` over direct `push_screen()` so the real app navigation handler saves/restores state.

- [ ] **Step 4: Run smoke test and verify it fails on Chatbooks**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_ux_audit_smoke.py --tb=short --maxfail=1
```

Expected: FAIL after entering `chatbooks` because the shared navigation bar is missing.

- [ ] **Step 5: Commit failing tests only if working in a red-green commit style**

Optional commit:

```bash
git add Tests/UI/test_ux_audit_smoke.py Tests/UI/test_screen_navigation.py
git commit -m "test: add ux shell smoke contract"
```

If the team prefers green-only commits, skip this commit and include the tests with Task 2.

## Task 2: Convert Chatbooks To Shared Shell

**Files:**

- Modify: `tldw_chatbook/UI/Screens/chatbooks_screen.py`
- Modify: `Tests/UI/test_chatbooks_screen_server_actions.py`
- Modify: `Tests/UI/test_screen_navigation.py`
- Modify: `Tests/UI/test_ux_audit_smoke.py`

- [ ] **Step 1: Update ChatbooksScreen inheritance and constructor**

Change `ChatbooksScreen` to extend `BaseAppScreen` instead of raw `Screen`.

Implementation shape:

```python
from ..Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.Constants import TAB_CHATBOOKS


class ChatbooksScreen(BaseAppScreen):
    def __init__(self, app_instance, **kwargs):
        super().__init__(app_instance, TAB_CHATBOOKS, **kwargs)
```

- [ ] **Step 2: Move Chatbooks content into compose_content()**

Replace `compose()` with `compose_content()` so `BaseAppScreen.compose()` mounts `MainNavigationBar`.

Implementation shape:

```python
def compose_content(self) -> ComposeResult:
    logger.info("Composing Chatbooks screen")
    yield ChatbooksWindowImproved(self.app_instance)
```

Do not yield `ChatbooksWindowImproved(self.app)` from `compose_content()`. Use `self.app_instance` so tests and app routing receive the same owner object that the screen was constructed with.

- [ ] **Step 3: Preserve mount and resume behavior**

Keep current `on_mount()`, `on_screen_suspend()`, `on_screen_resume()`, and local reactive state.

If adding `super().on_mount()`, call it synchronously:

```python
super().on_mount()
```

Do not `await super().on_mount()` because `BaseAppScreen.on_mount()` is not async.

- [ ] **Step 4: Update Chatbooks tests to construct with an app instance**

In `Tests/UI/test_chatbooks_screen_server_actions.py`, update host apps from:

```python
yield ChatbooksScreen()
```

to:

```python
yield ChatbooksScreen(self)
```

Add assertions that the shared nav exists:

```python
assert app.screen.query_one("#nav-chat") is not None
assert app.screen.query_one("#nav-chatbooks") is not None
```

- [ ] **Step 5: Run focused shell tests**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_ux_audit_smoke.py \
  Tests/UI/test_screen_navigation.py::test_primary_routed_screens_use_base_app_screen \
  Tests/UI/test_chatbooks_screen_server_actions.py \
  --tb=short
```

Expected: PASS.

- [ ] **Step 6: Commit shell fix**

Run:

```bash
git add tldw_chatbook/UI/Screens/chatbooks_screen.py Tests/UI/test_ux_audit_smoke.py Tests/UI/test_screen_navigation.py Tests/UI/test_chatbooks_screen_server_actions.py
git commit -m "fix: mount chatbooks in shared app shell"
```

## Task 3: Lock In Ingest Source Default Regression

**Files:**

- Modify: `Tests/UI/test_ingestion_ui_redesigned.py`
- Modify only if test fails: `tldw_chatbook/Widgets/Media/media_ingestion_source_panel.py`

- [ ] **Step 1: Add a direct source-panel default test**

In `Tests/UI/test_ingestion_ui_redesigned.py`, import `Select` if needed and add a test under `TestMediaIngestWindowRebuilt`.

Test shape:

```python
@pytest.mark.asyncio
async def test_source_panel_create_type_default_is_allowed_in_server_mode(
    self,
    mock_app_instance: MagicMock,
    widget_pilot,
) -> None:
    mock_app_instance.media_runtime_state = SimpleNamespace(runtime_backend="server")

    async with await widget_pilot(MediaIngestWindowRebuilt, app_instance=mock_app_instance) as pilot:
        window = pilot.app.test_widget
        await pilot.pause()

        source_type = window.source_panel.query_one("#create-source-type", Select)

        allowed_values = {value for _label, value in CREATE_SOURCE_TYPE_OPTIONS}
        assert source_type.value in allowed_values
        assert source_type.value == "local_directory"
```

Import `SimpleNamespace`, `Select`, and `CREATE_SOURCE_TYPE_OPTIONS` as needed.

- [ ] **Step 2: Run the direct Ingest regression**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_ingestion_ui_redesigned.py::TestMediaIngestWindowRebuilt::test_source_panel_create_type_default_is_allowed_in_server_mode --tb=short
```

Expected: PASS on current `dev`. If it fails, fix only the options/default mismatch in `media_ingestion_source_panel.py`.

- [ ] **Step 3: Run existing Ingest focused tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_ingestion_ui_redesigned.py Tests/UI/test_media_ingestion_tab_integration.py --tb=short
```

Expected: PASS.

- [ ] **Step 4: Commit Ingest regression**

Run:

```bash
git add Tests/UI/test_ingestion_ui_redesigned.py tldw_chatbook/Widgets/Media/media_ingestion_source_panel.py
git commit -m "test: lock ingest source default"
```

If `media_ingestion_source_panel.py` was untouched, stage only the test file.

## Task 4: Normalize Empty Study Quiz List Responses

**Files:**

- Modify: `Tests/Study_Interop/test_quiz_scope_service.py`
- Modify: `tldw_chatbook/Study_Interop/quiz_scope_service.py`

- [ ] **Step 1: Add failing empty local quiz list regression**

In `Tests/Study_Interop/test_quiz_scope_service.py`, add a fake service that returns an empty paginated mapping:

```python
class EmptyMappingLocalQuizService(FakeLocalQuizService):
    def list_quizzes(self, *, q=None, limit=100, offset=0):
        self.calls.append(("list_quizzes", q, limit, offset))
        return {"items": [], "count": 0}
```

Add test:

```python
@pytest.mark.asyncio
async def test_local_quiz_list_empty_mapping_returns_empty_list():
    scope = QuizScopeService(local_service=EmptyMappingLocalQuizService())

    quizzes = await scope.list_quizzes(mode="local")

    assert quizzes == []
```

- [ ] **Step 2: Run regression and verify it fails**

Run:

```bash
.venv/bin/python -m pytest -q Tests/Study_Interop/test_quiz_scope_service.py::test_local_quiz_list_empty_mapping_returns_empty_list --tb=short
```

Expected before fix: FAIL because the current logic may iterate mapping keys such as `"items"` and `"count"`.

- [ ] **Step 3: Add one response-shape helper**

In `tldw_chatbook/Study_Interop/quiz_scope_service.py`, add a private helper near `_maybe_await()`:

```python
@staticmethod
def _items_from_list_response(records: Any) -> list[Any]:
    if records is None:
        return []
    if isinstance(records, Mapping):
        items = records.get("items")
        if items is None:
            return []
        return list(items)
    return list(records)
```

- [ ] **Step 4: Use the helper in quiz list paths**

Replace these patterns:

```python
list((records or {}).get("items") or records or [])
```

and:

```python
list((response or {}).get("items") or response or [])
```

with:

```python
self._items_from_list_response(records)
```

or:

```python
self._items_from_list_response(response)
```

Apply this to `list_quizzes()` and `_load_scoped_server_quizzes()` at minimum. If the same unsafe pattern appears in question or attempt list methods, use the helper there too in the same commit.

- [ ] **Step 5: Run Study focused tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/Study_Interop/test_quiz_scope_service.py --tb=short
```

Expected: PASS.

- [ ] **Step 6: Commit Study fix**

Run:

```bash
git add Tests/Study_Interop/test_quiz_scope_service.py tldw_chatbook/Study_Interop/quiz_scope_service.py
git commit -m "fix: normalize empty quiz list responses"
```

## Task 5: Fix Chat Save-State Direct Log Path

**Files:**

- Modify: `Tests/UI/test_chat_screen_state.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`

- [ ] **Step 1: Add failing direct chat-log regression**

In `Tests/UI/test_chat_screen_state.py`, add a small fake chat log:

```python
class EmptyChatLog:
    children = []

    def query(self, _selector):
        return []
```

Add test:

```python
def test_extract_messages_clears_messages_when_direct_chat_log_lookup_succeeds():
    app = Mock()
    app.query_one = Mock(return_value=EmptyChatLog())
    screen = ChatScreen(app)
    screen.chat_window = Mock()
    tab_state = TabState(
        tab_id="tab-1",
        messages=[MessageData(message_id="old", role="user", content="stale")],
    )

    screen._extract_and_save_messages(tab_state)

    assert tab_state.messages == []
    screen.chat_window.query.assert_not_called()
```

This test should fail before the fix because `log_selectors` is unbound when direct lookup succeeds.

- [ ] **Step 2: Run regression and verify failure**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_chat_screen_state.py::test_extract_messages_clears_messages_when_direct_chat_log_lookup_succeeds --tb=short
```

Expected before fix: FAIL because stale messages remain or an error path is logged.

- [ ] **Step 3: Define fallback selectors before direct lookup**

In `tldw_chatbook/UI/Screens/chat_screen.py`, in `_extract_and_save_messages()`, move the selector list before direct lookup:

```python
log_selectors = [
    "#chat-log",
    ".chat-log",
    "#chat-messages-container",
    ".chat-messages",
]
chat_log = None
```

Then keep this structure:

```python
try:
    chat_log = self.app_instance.query_one("#chat-log", VerticalScroll)
    logger.debug("Found chat log via app_instance.query_one")
except Exception:
    pass

if not chat_log:
    for selector in log_selectors:
        ...
```

Do not change restore logic unless a focused test proves it has the same bug.

- [ ] **Step 4: Run Chat state tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_chat_screen_state.py --tb=short
```

Expected: PASS.

- [ ] **Step 5: Commit Chat state fix**

Run:

```bash
git add Tests/UI/test_chat_screen_state.py tldw_chatbook/UI/Screens/chat_screen.py
git commit -m "fix: save chat messages from direct log path"
```

## Task 6: Keep Search/RAG Collection UI Updates On The Message Thread

**Files:**

- Modify: `Tests/UI/test_search_rag_window.py`
- Modify: `tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py`

- [ ] **Step 1: Add a worker-seam regression**

In `Tests/UI/test_search_rag_window.py`, add a test that verifies collection loading does not query widgets directly.

Recommended implementation target: introduce a production helper named `_load_available_collections()` and test it directly first.

Test shape:

```python
def test_load_available_collections_does_not_touch_textual_widgets(
    self,
    mock_app_instance: MagicMock,
    search_rag_test_env,
) -> None:
    window = SearchRAGWindow(mock_app_instance, id="test-search-window")
    window.query_one = MagicMock(side_effect=AssertionError("UI touched during load"))

    with patch(
        "tldw_chatbook.UI.Views.RAGSearch.search_rag_window.get_available_profiles",
        return_value=["default", "research"],
    ):
        assert window._load_available_collections() == ["default", "research"]
```

- [ ] **Step 2: Run regression and verify failure**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_search_rag_window.py::TestSearchRAGWindow::test_load_available_collections_does_not_touch_textual_widgets --tb=short
```

Expected before fix: FAIL because `_load_available_collections()` does not exist.

- [ ] **Step 3: Split data loading from UI application**

In `tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py`, add:

```python
def _load_available_collections(self) -> list[str]:
    return list(get_available_profiles())
```

Add an on-thread UI helper:

```python
async def _apply_available_collections(self, collections: list[str]) -> None:
    self.available_collections = list(collections)
    collections_list = self.query_one("#collections-list", ListView)
    await collections_list.clear()

    for collection in self.available_collections:
        await collections_list.append(ListItem(Static(collection)))

    collection_select = self.query_one("#collection-select", Select)
    collection_select.set_options(
        [("All Collections", "all")] + [(collection, collection) for collection in self.available_collections]
    )
```

Use the same label/value ordering as the existing `Select` construction. If current code has `("All Collections", "all")`, keep that exact order.

Add a small scheduler helper so the coroutine is created on the Textual message thread, not in the worker thread:

```python
def _schedule_apply_available_collections(self, collections: list[str]) -> None:
    self.run_worker(self._apply_available_collections(collections), exclusive=True)
```

- [ ] **Step 4: Make the worker schedule UI work instead of doing it**

Replace the current `_refresh_collections_list()` body with a thread-safe scheduling pattern.

Implementation shape:

```python
@work(thread=True)
def _refresh_collections_list(self) -> None:
    try:
        collections = self._load_available_collections()
        self.app.call_from_thread(self._schedule_apply_available_collections, collections)
    except Exception as e:
        logger.error(f"Error refreshing collections: {e}")
```

If `run_worker()` is not the right Textual primitive for awaiting `_apply_available_collections()`, use the smallest existing app pattern in this repo that schedules async UI work from a thread. The invariant is strict: the `thread=True` method must not call `query_one()`, `clear()`, `append()`, or `set_options()`, and it must not create the `_apply_available_collections()` coroutine in the worker thread.

- [ ] **Step 5: Add an application-helper test for UI application**

Add a mounted widget test that calls `_apply_available_collections(["default"])` directly and verifies:

```python
assert window.available_collections == ["default"]
assert len(window.query_one("#collections-list", ListView).children) == 1
assert window.query_one("#collection-select", Select).value in ("all", Select.BLANK)
```

The exact select value may depend on Textual selection behavior; assert options rather than selected value if needed.

- [ ] **Step 6: Run Search/RAG focused tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_search_rag_window.py --tb=short
```

Expected: PASS.

- [ ] **Step 7: Commit Search/RAG threading fix**

Run:

```bash
git add Tests/UI/test_search_rag_window.py tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py
git commit -m "fix: refresh search collections on ui thread"
```

## Task 7: Verify Search Primary Action Reachability

**Files:**

- Modify: `Tests/UI/test_search_rag_window.py`
- Modify only if test fails: `tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py`

- [ ] **Step 1: Add search action reachability regression**

In `Tests/UI/test_search_rag_window.py`, add a mounted test that proves the primary Search button is present, enabled, and callable without opening advanced controls.

Test shape:

```python
@pytest.mark.asyncio
async def test_primary_search_action_is_reachable_in_default_layout(
    self,
    mock_app_instance: MagicMock,
    search_rag_test_env,
    widget_pilot,
) -> None:
    async with await widget_pilot(SearchRAGWindow, app_instance=mock_app_instance) as pilot:
        window = pilot.app.test_widget
        search_input = window.query_one("#search-query-input", Input)
        search_button = window.query_one("#search-button", Button)

        assert search_button.disabled is False
        assert search_button.display is True
        assert search_input.display is True
```

If the current Textual test harness supports click assertions reliably, extend it:

```python
search_input.value = "test query"
window.handle_search_button = MagicMock()
await pilot.click("#search-button")
await pilot.pause()
window.handle_search_button.assert_called()
```

Only add the click assertion if it is stable under the local Textual test driver.

- [ ] **Step 2: Run regression**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_search_rag_window.py::TestSearchRAGWindow::test_primary_search_action_is_reachable_in_default_layout --tb=short
```

Expected: PASS if the current layout is already reachable. If it fails, continue to Step 3.

- [ ] **Step 3: Make only minimal layout changes if needed**

If reachability fails, modify only the search control row around `#search-query-input` and `#search-button`.

Allowed changes:

- Keep `#search-query-input` and `#search-button` in the first visible search section.
- Keep advanced controls below or behind the existing advanced options area.
- Do not redesign the full Search/RAG screen.
- Do not change handler names or IDs.

- [ ] **Step 4: Run Search/RAG and handoff tests**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_search_rag_window.py Tests/UI/test_search_handoffs.py --tb=short
```

Expected: PASS.

- [ ] **Step 5: Commit Search reachability test/fix**

Run:

```bash
git add Tests/UI/test_search_rag_window.py tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py
git commit -m "test: cover search primary action reachability"
```

If production code was changed, use:

```bash
git commit -m "fix: keep search primary action reachable"
```

## Final Verification

- [ ] **Step 1: Run Phase 0 tests**

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_ux_audit_smoke.py \
  Tests/UI/test_screen_navigation.py \
  Tests/UI/test_chatbooks_screen_server_actions.py \
  --tb=short
```

- [ ] **Step 2: Run Phase 1 tests**

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_ingestion_ui_redesigned.py \
  Tests/UI/test_media_ingestion_tab_integration.py \
  Tests/Study_Interop/test_quiz_scope_service.py \
  Tests/UI/test_chat_screen_state.py \
  Tests/UI/test_search_rag_window.py \
  Tests/UI/test_search_handoffs.py \
  --tb=short
```

- [ ] **Step 3: Run already-merged handoff guardrails**

```bash
.venv/bin/python -m pytest -q \
  Tests/UI/test_chat_first_handoffs.py \
  Tests/UI/test_media_handoffs.py \
  Tests/UI/test_chat_tab_container.py \
  --tb=short
```

- [ ] **Step 4: Run whitespace check**

```bash
git diff --check
```

- [ ] **Step 5: Update the rebaseline tracker**

Modify `Docs/superpowers/plans/2026-05-01-ux-audit-remediation.md` after implementation:

- Mark Phase 0 completed if the shell smoke passes.
- Mark the completed Phase 1 items.
- Leave Phase 2/5/6/7 untouched unless implemented in later tranches.

- [ ] **Step 6: Final commit**

If tracker updates were made separately:

```bash
git add Docs/superpowers/plans/2026-05-01-ux-audit-remediation.md
git commit -m "docs: update ux remediation phase status"
```

## Known Risks

- The app-level Textual smoke may be slow or flaky if it initializes optional services. If that happens, keep the routed-screen class contract as the CI gate and make the full route-click test narrower around Chatbooks.
- `SearchRAGWindow._refresh_collections_list()` uses Textual worker decorators. Verify the chosen `call_from_thread` scheduling pattern against existing repo usage before finalizing implementation.
- `ChatbooksWindowImproved` may rely on `self.app` rather than the constructor app instance. If that appears during tests, preserve the production app instance passed from `TldwCli` and adapt only test hosts.
- Search layout reachability should not become a redesign. If the reachability test already passes, do not edit layout code in this tranche.

## Handoff To Execution

Recommended execution mode: subagent-driven per task if available, otherwise inline execution with a checkpoint after each task. Do not start Task 2 until Task 1 produces a failing red test. Do not start Phase 2 work until all seven tasks above are green.
