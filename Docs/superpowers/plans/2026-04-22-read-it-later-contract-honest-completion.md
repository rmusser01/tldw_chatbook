# Read-it-Later Contract-Honest Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the approved `Read-it-later` follow-up so Chatbook keeps local per-type saved browsing, keeps server saved browsing aggregate-only, and enforces that boundary through one authoritative capability seam plus deterministic runtime normalization and stale-state cleanup.

**Architecture:** Keep the slice small and truthful. Add one `Read-it-later` context-capability helper in the existing media scope service, route `MediaWindow` and the search-panel affordances through that one seam, normalize invalid server saved contexts at backend switch, screen entry, and pre-query, then update the parity docs to say aggregate-only server behavior is landed while per-type server saved views are blocked on a server contract change.

**Tech Stack:** Python 3.11+, Textual, existing `MediaReadingScopeService`, existing `MediaRuntimeState`, pytest

---

## Scope Check

This plan intentionally covers only the approved `A` follow-up. It does **not** reopen the earlier combined media vertical and it does **not** implement:

- true per-media-type server saved views
- client-side fake bucketing of aggregate server saved results
- server API changes
- generic collections work
- `Writing Suite`

Implementation note for agentic workers:

- Use `@superpowers:test-driven-development` before each code task.
- Use `@superpowers:verification-before-completion` before claiming the slice is done.

## File Map

- Modify: `tldw_chatbook/Media/media_reading_scope_service.py`
  Responsibility: Expose one authoritative `Read-it-later` context-capability helper that tells the UI whether saved browsing is available for the current backend and media-type context, whether the current server context is aggregate-only, and the user-facing invalid-context reason.
- Modify: `tldw_chatbook/UI/MediaWindow_v2.py`
  Responsibility: Replace split aggregate-only heuristics with the scope-service capability seam, normalize invalid saved contexts on mount and before browse queries, clear stale saved-only state during correction, and requery the corrected context.
- Modify: `tldw_chatbook/Widgets/Media/media_search_panel.py`
  Responsibility: Reflect the shared capability result in the browse-subview control and show a small reason string when server `Read-it-later` is unavailable outside `All Media`.
- Modify: `tldw_chatbook/UI/Screens/media_runtime_state.py`
  Responsibility: Keep backend-switch reset semantics explicit and safe for saved-view state; do not add speculative persistence features.

- Modify: `Tests/Media/test_media_reading_scope_service.py`
  Responsibility: Verify the authoritative `Read-it-later` context-capability helper for local, valid server aggregate, and invalid server non-aggregate contexts.
- Modify: `Tests/UI/test_media_runtime_state.py`
  Responsibility: Keep backend reset semantics covered as a normalization entrypoint.
- Modify: `Tests/UI/test_media_window_v2_parity.py`
  Responsibility: Verify mount/pre-query normalization, invalid-context correction, stale-state cleanup, valid requery behavior, and search-panel capability sync.

- Modify: `Docs/Parity/2026-04-21-capability-matrix.md`
  Responsibility: Mark server `Read-it-later` parity as aggregate-only landed and remove wording that implies a remaining Chatbook-only per-type follow-up.
- Modify: `Docs/Parity/2026-04-21-gap-ledger.md`
  Responsibility: Record that true per-type server saved browsing is blocked on a server contract extension rather than pending more Chatbook shaping.
- Modify: `Docs/Parity/2026-04-21-execution-roadmap.md`
  Responsibility: Move this slice to landed status and make `Writing Suite` the next larger non-MCP row.

## Task 1: Add One Authoritative `Read-it-later` Context Capability Seam

**Files:**
- Modify: `tldw_chatbook/Media/media_reading_scope_service.py`
- Modify: `Tests/Media/test_media_reading_scope_service.py`

- [ ] **Step 1: Write the failing scope-service tests**

```python
def test_read_it_later_context_capability_allows_local_any_media_type():
    scope = MediaReadingScopeService(local_service=object(), server_service=None)

    capability = scope.get_read_it_later_context_capability(
        mode="local",
        media_type_slug="article",
    )

    assert capability.available is True
    assert capability.aggregate_only is False
    assert capability.reason is None


def test_read_it_later_context_capability_allows_server_all_media_only():
    scope = MediaReadingScopeService(local_service=None, server_service=object())

    capability = scope.get_read_it_later_context_capability(
        mode="server",
        media_type_slug="all-media",
    )

    assert capability.available is True
    assert capability.aggregate_only is True
    assert capability.reason is None


def test_read_it_later_context_capability_blocks_server_non_all_media():
    scope = MediaReadingScopeService(local_service=None, server_service=object())

    capability = scope.get_read_it_later_context_capability(
        mode="server",
        media_type_slug="article",
    )

    assert capability.available is False
    assert capability.aggregate_only is True
    assert capability.reason == "Read-it-later is only available in server mode from All Media."
```

- [ ] **Step 2: Run the focused scope-service tests to verify they fail**

Run: `python3 -m pytest /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical/Tests/Media/test_media_reading_scope_service.py -q`
Expected: FAIL with missing `get_read_it_later_context_capability(...)` coverage and/or missing helper.

- [ ] **Step 3: Implement the minimal scope-service capability helper**

```python
@dataclass(frozen=True)
class ReadItLaterContextCapability:
    available: bool
    aggregate_only: bool
    reason: str | None = None


def get_read_it_later_context_capability(
    self,
    *,
    mode: MediaReadingBackend | str | None = None,
    media_type_slug: str | None = None,
) -> ReadItLaterContextCapability:
    normalized_mode = self._normalize_mode(mode)
    normalized_type = str(media_type_slug or "all-media").strip().lower() or "all-media"

    if normalized_mode == MediaReadingBackend.LOCAL:
        return ReadItLaterContextCapability(available=True, aggregate_only=False, reason=None)

    if normalized_type == "all-media":
        return ReadItLaterContextCapability(available=True, aggregate_only=True, reason=None)

    return ReadItLaterContextCapability(
        available=False,
        aggregate_only=True,
        reason="Read-it-later is only available in server mode from All Media.",
    )
```

Implementation constraints for this task:

- Keep the helper in `media_reading_scope_service.py`; do not create a new abstraction file for this slice.
- Do not add any server query shaping for media type here.
- Do not let UI code retain its own separate aggregate-only rule after this helper exists.

- [ ] **Step 4: Run the scope-service tests again**

Run: `python3 -m pytest /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical/Tests/Media/test_media_reading_scope_service.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical add tldw_chatbook/Media/media_reading_scope_service.py Tests/Media/test_media_reading_scope_service.py
git -C /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical commit -m "feat: add read-it-later context capability seam"
```

## Task 2: Normalize Invalid Saved Contexts At Backend Switch, Screen Entry, And Pre-Query

**Files:**
- Modify: `tldw_chatbook/UI/MediaWindow_v2.py`
- Modify: `tldw_chatbook/UI/Screens/media_runtime_state.py`
- Modify: `Tests/UI/test_media_runtime_state.py`
- Modify: `Tests/UI/test_media_window_v2_parity.py`

- [ ] **Step 1: Write the failing normalization tests**

```python
@pytest.mark.asyncio
async def test_media_window_prequery_normalizes_invalid_server_saved_context_and_requeries_all():
    scope_service = Mock()
    scope_service.get_read_it_later_context_capability.side_effect = [
        ReadItLaterContextCapability(
            available=False,
            aggregate_only=True,
            reason="Read-it-later is only available in server mode from All Media.",
        ),
        ReadItLaterContextCapability(available=True, aggregate_only=True, reason=None),
    ]
    scope_service.search_media = AsyncMock(
        return_value={"items": [{"id": "server:reading_item:200", "title": "Article 200"}], "total": 1}
    )
    window, app = _build_media_window(runtime_backend="server", scope_service=scope_service)
    window.active_media_type = "article"
    window.runtime_state.active_browse_subview = "read-it-later"
    window.runtime_state.selected_record_id = "server:reading_item:41"
    window.runtime_state.browse_items = [{"id": "server:reading_item:41", "title": "Stale Saved"}]

    tasks = []
    window.run_worker = lambda coro, exclusive=True: tasks.append(asyncio.create_task(coro))

    window._perform_search("article", "", "")
    await asyncio.gather(*tasks)

    assert window.runtime_state.active_browse_subview == "all"
    assert window.runtime_state.selected_record_id is None
    assert [item["id"] for item in window.runtime_state.browse_items] == ["server:reading_item:200"]
    app.notify.assert_called_once_with(
        "Read-it-later is only available in server mode from All Media.",
        severity="warning",
    )


def test_media_runtime_state_backend_reset_restores_safe_saved_view_defaults():
    state = MediaRuntimeState(runtime_backend="server")
    state.active_browse_subview = "read-it-later"
    state.selected_record_id = "server:reading_item:41"

    state.reset_for_backend("local")

    assert state.runtime_backend == "local"
    assert state.active_browse_subview == "all"
    assert state.selected_record_id is None
```

- [ ] **Step 2: Run the focused runtime/window tests to verify they fail**

Run: `python3 -m pytest /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical/Tests/UI/test_media_runtime_state.py /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical/Tests/UI/test_media_window_v2_parity.py -q`
Expected: FAIL because `MediaWindow` still defines aggregate-only behavior locally and does not perform explicit stale-state cleanup plus valid requery through one shared capability seam.

- [ ] **Step 3: Implement minimal normalization and stale-state cleanup**

```python
def _read_it_later_capability(self) -> ReadItLaterContextCapability:
    scope_service = self._scope_service()
    if scope_service is None:
        return ReadItLaterContextCapability(available=True, aggregate_only=False, reason=None)
    return scope_service.get_read_it_later_context_capability(
        mode=self._runtime_backend(),
        media_type_slug=self.active_media_type or "all-media",
    )


def _normalize_saved_view_context(self) -> bool:
    capability = self._read_it_later_capability()
    if self._active_browse_subview() != "read-it-later" or capability.available:
        return False

    self.runtime_state.active_browse_subview = "all"
    self.runtime_state.selected_record_id = None
    self.runtime_state.browse_items = []
    self.runtime_state.detail_by_record_id.clear()
    self.runtime_state.reading_progress_by_record_id.clear()
    self.viewer_panel.clear_display()
    self._show_empty_state()
    self._sync_saved_view_controls()
    self.app_instance.notify(capability.reason, severity="warning")
    return True
```

Implementation constraints for this task:

- Call normalization from three places:
  - after backend-switch-safe state is in effect
  - during `MediaWindow.on_mount()` or initial visible screen-entry sync
  - before browse execution in `_perform_search(...)`
- Do not keep `_saved_view_available_for_context()` as an independent policy source once the capability helper exists.
- Requery the corrected valid context immediately after normalization when the user was in an invalid saved view.
- Clear stale saved-only browse state before the corrected results are loaded.

- [ ] **Step 4: Run the focused runtime/window tests again**

Run: `python3 -m pytest /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical/Tests/UI/test_media_runtime_state.py /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical/Tests/UI/test_media_window_v2_parity.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical add tldw_chatbook/UI/MediaWindow_v2.py tldw_chatbook/UI/Screens/media_runtime_state.py Tests/UI/test_media_runtime_state.py Tests/UI/test_media_window_v2_parity.py
git -C /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical commit -m "fix: normalize invalid read-it-later browse contexts"
```

## Task 3: Surface The Shared Capability In The Search Panel

**Files:**
- Modify: `tldw_chatbook/Widgets/Media/media_search_panel.py`
- Modify: `Tests/UI/test_media_window_v2_parity.py`

- [ ] **Step 1: Write the failing search-panel affordance test**

```python
@pytest.mark.asyncio
async def test_media_search_panel_shows_saved_view_reason_when_disabled():
    class MediaSearchPanelApp(App[None]):
        def compose(self) -> ComposeResult:
            yield MediaSearchPanel(SimpleNamespace())

    app = MediaSearchPanelApp()
    async with app.run_test() as pilot:
        panel = app.query_one(MediaSearchPanel)
        panel.set_saved_view_capability(
            enabled=False,
            reason="Read-it-later is only available in server mode from All Media.",
        )
        await pilot.pause()

        status = panel.query_one("#saved-view-status", Static)
        assert "only available in server mode from All Media" in str(status.renderable)
```

- [ ] **Step 2: Run the focused panel/UI test to verify it fails**

Run: `python3 -m pytest /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical/Tests/UI/test_media_window_v2_parity.py -q`
Expected: FAIL with missing status widget and missing `set_saved_view_capability(...)`.

- [ ] **Step 3: Implement the minimal panel affordance**

```python
saved_view_reason = reactive("")

with Horizontal(classes="saved-view-row"):
    yield Label("Browse:", classes="filter-label")
    yield Select(...)
yield Static("", id="saved-view-status", classes="saved-view-status")

def set_saved_view_capability(self, *, enabled: bool, reason: str | None = None) -> None:
    self.saved_view_enabled = bool(enabled)
    self.saved_view_reason = str(reason or "")

def watch_saved_view_reason(self, saved_view_reason: str) -> None:
    status = self.query_one("#saved-view-status", Static)
    status.update(saved_view_reason)
```

Implementation constraints for this task:

- Keep the panel simple; do not redesign the browse controls.
- Route the panel from `MediaWindow._sync_saved_view_controls()` using the shared capability helper result.
- Do not duplicate the aggregate-only rule in the panel.

- [ ] **Step 4: Run the focused panel/UI test again**

Run: `python3 -m pytest /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical/Tests/UI/test_media_window_v2_parity.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical add tldw_chatbook/Widgets/Media/media_search_panel.py Tests/UI/test_media_window_v2_parity.py
git -C /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical commit -m "feat: show read-it-later capability status in media search panel"
```

## Task 4: Update The Parity Docs To Match Verified Behavior

**Files:**
- Modify: `Docs/Parity/2026-04-21-capability-matrix.md`
- Modify: `Docs/Parity/2026-04-21-gap-ledger.md`
- Modify: `Docs/Parity/2026-04-21-execution-roadmap.md`

- [ ] **Step 1: Write the doc updates after the behavior is verified**

Add wording equivalent to:

```md
- `Collections: Reading List / Read-it-later`
  - local saved-reading remains first-class across media types
  - server `Read-it-later` is aggregate-only in `All Media`
  - true per-media-type server saved browsing is blocked on a server list-contract extension
```

- [ ] **Step 2: Run the focused regression suite for the completed slice**

Run: `python3 -m pytest /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical/Tests/Media/test_media_reading_scope_service.py /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical/Tests/UI/test_media_runtime_state.py /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical/Tests/UI/test_media_window_v2_parity.py -q`
Expected: PASS

- [ ] **Step 3: Run the broader media-related regression suite**

Run: `python3 -m pytest /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical/Tests/Media /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical/Tests/UI/test_media_runtime_state.py /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical/Tests/UI/test_media_window_v2_parity.py -q`
Expected: PASS, or only pre-existing unrelated failures if any.

- [ ] **Step 4: Commit the docs and verification-backed status updates**

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical add Docs/Parity/2026-04-21-capability-matrix.md Docs/Parity/2026-04-21-gap-ledger.md Docs/Parity/2026-04-21-execution-roadmap.md
git -C /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-watchlists-notifications-vertical commit -m "docs: finalize read-it-later parity boundary"
```

- [ ] **Step 5: Stop and reassess the next vertical**

Next planned vertical after this slice: `Writing Suite`

Execution notes for the handoff:

- Do not start `Writing Suite` until the focused and broader regression commands above are green or any pre-existing failures are clearly isolated.
- Treat this slice as complete only when the docs no longer imply a remaining Chatbook-only path to per-type server saved browsing.
