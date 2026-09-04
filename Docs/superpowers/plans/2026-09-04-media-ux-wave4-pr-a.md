# Media UX fix wave 4 — PR A (regressions + ruling) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the two regressions critique #4 traced to fix wave 3 (walk keys typed into the search field; undo receipts clipping their Undo button), the user's auto-resume-bypass ruling, and the row-marker design note, as one PR against `dev`.

**Architecture:** The Reader's content search bar gains an explicit one-shot `focus_on_mount` token that only the Find gesture sets, so an item change can never move focus into the Input; the Analysis tab gates its bar behind `find_open` exactly as Read does. Receipts adopt the two-row grammar the toolbars got in #2350 (copy row + action row, width 100%). Explicit opens cancel a pending auto-resume worker and the banner names an off-set item honestly. Everything is pinned by tests on the production-CSS harness; painted-text assertions where paint-over or clipping is the bug.

**Tech Stack:** Python 3.12, Textual 8.x, pytest + pytest-asyncio (`app.run_test`), the `LibraryProductionCSSHarness` from `Tests/UI/test_library_shell.py`, `ControlledDetailMediaService` from `Tests/UI/test_library_media_reader_flow.py`.

**Spec:** `.impeccable/critique/2026-09-04T13-50-05Z__tldw-chatbook-ui-screens-library-screen-py.md` (critique #4) — priority issues 1 and 2, the "Decisions I own" section (ruling 3), and tasks `backlog/tasks/task-31269`, `task-31270`, `task-31273`, `task-31278`.

## Global Constraints

- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/media-ux-p3`, branch `fix/media-wave4-a` off dev `282c733d61`. Every command: `cd <worktree> && PYTHONPATH=<worktree> /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest … -p no:cacheprovider` (the venv's editable install points at the MAIN checkout; PYTHONPATH pins the worktree). Use absolute paths; the shell cwd can be reset between calls.
- Run UI test files in SEPARATE processes (never several heavy app-test files in one pytest invocation).
- No new `logger.*` calls (each one forces `python scripts/check_persistent_diagnostic_inventory.py --write` and a committed inventory JSON). If you must add one, run the writer and commit `Docs/security/production-diagnostic-inventory.json`.
- After editing any `BUNDLED_CSS` or `tldw_chatbook/css/components/*.tcss`: run `python -m tldw_chatbook.css.build_css` then `python tldw_chatbook/css/check_bundle_sync.py` (no pipe; read the real exit code) and commit every regenerated file (`tldw_chatbook/css/tldw_cli_modular.tcss`, `screen_*.tcss`, `widget_defaults_*`).
- Any new id queried by a seam must be gated on the active Reader mode when the same id is reused across tabs (`#library-media-content-search-controls` and `#library-media-viewer-content` are reused by Read and Analysis).
- Textual key names in `pilot.press`: `]` is `right_square_bracket`, `[` is `left_square_bracket`, Escape is `escape`.
- Commit after every task with the trailer:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_011LebG4HPfSniVohbuXkU4n
  ```
- Backlog task files: mark ACs `- [x]`, add `## Implementation Plan` and `## Implementation Notes`, flip `status: Done` via `backlog task edit <id> -s Done --check-ac N … --plan "…" --notes "…" --plain` run with cwd = the worktree. Never renumber ids.

---

### Task 1: The Find gesture, not the mount, decides focus (task-31269, P0)

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_media_content.py:330-366` (`LibraryMediaContentSearchControls.__init__` / `on_mount`)
- Modify: `tldw_chatbook/Widgets/Library/library_media_viewer.py:84-122` (constructor), `:338-348` (Read-mode bar), `:626-641` (Analysis-mode bar)
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:2662` (state init), `:40481` (viewer construction), `:40649-40790` (`_sync_library_media_viewer_state` compare/assign), `:41063-41083` (`handle_library_media_reader_find`)
- Modify: `Tests/UI/test_library_media_reader_flow.py:1452-1490` (contract change for Find on the Analysis tab)
- Test: `Tests/UI/test_library_media_render_fixes.py` (new tests appended)
- Modify: `Docs/User_Guide/library/media-and-conversations.md` (Find paragraph + "Verified against" stamp)

**Interfaces:**
- Produces: `LibraryMediaContentSearchControls(…, focus_on_mount: bool = False)`; `LibraryMediaViewer(…, find_focus_pending: bool = False)` attribute `find_focus_pending` consumed (set `False`) by the viewer's own compose right after it yields the bar; screen attribute `_library_media_find_focus_pending: bool` set by the Find handler and consumed by whichever seam next builds or syncs the viewer.
- Consumes: `screen._library_media_find_open`, `_close_library_media_find()`, `_sync_library_media_viewer_or_recompose()`, `_reset_library_media_search_on_mode_change(new_mode)`.

- [ ] **Step 1: Write the failing tests** (append to `Tests/UI/test_library_media_render_fixes.py`; add these imports at the top of the file next to the existing ones)

```python
from textual.widgets import Input

from Tests.UI.test_library_media_reader_flow import (
    ControlledDetailMediaService,
    _load_row_0,
    _row_identity,
    _wait_for_detail_call,
)
from Tests.UI.test_library_shell import _many_media_items
from tldw_chatbook.Library.library_media_reader_session import set_mode


def _analysis_flow_host(count: int = 3):
    """Three local items, each with a current analysis version.

    Local media detail never carries ``analysis_content`` at the top level;
    the viewer reads the newest ``versions`` entry
    (``library_media_viewer_state._latest_version_analysis_text``).
    """
    app = _build_media_test_app()
    items = _many_media_items(count)
    for index, item in enumerate(items, 1):
        item["versions"] = [
            {"version_number": 1, "analysis_content": f"Analysis of item {index}"}
        ]
    _seed_conversations(app, _two_conversations(), media=items)
    service = ControlledDetailMediaService(items)
    app.media_reading_scope_service = service
    return LibraryProductionCSSHarness(app), service


async def _walk_next(screen, service, pilot, expected_row: int) -> str:
    """Press ] and settle the Reader on ``expected_row``; return its id."""
    row = screen.query_one(f"#library-media-row-{expected_row}", Button)
    row_id, backing_id, _ = _row_identity(row)
    await pilot.press("right_square_bracket")
    await _wait_for_detail_call(service, backing_id)
    service.release(backing_id)
    await _wait_for_condition(
        pilot,
        lambda: screen._library_media_reader_session.loaded_id == row_id,
        message=f"] never loaded row {expected_row}.",
    )
    await pilot.pause()
    return row_id


@pytest.mark.asyncio
async def test_analysis_mode_walk_never_moves_focus_into_the_search_field():
    """task-31269 (critique #4 P0): ] in Analysis mode walks, it never types.

    #2367's focus-on-mount hook fired on EVERY mount with an empty query,
    and the Analysis tab (task-28026) mounted the bar unconditionally, so
    each item load in Analysis mode parked focus in the Input and the next
    ] became text (live: `▊ ]`, `]]]]]`).
    """
    host, service = _analysis_flow_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_0(screen, service, pilot)
        screen._library_media_reader_session = set_mode(
            screen._library_media_reader_session, "analysis"
        )
        screen._sync_library_media_viewer_or_recompose()
        await _wait_for_condition(
            pilot,
            lambda: "Analysis of item 1" in "".join(
                str(w.render()) for w in screen.query("#library-media-viewer-content")
            ) or bool(screen.query("#library-media-viewer-content")),
            message="Analysis body never rendered.",
        )
        # The bar is collapsed until Find asks for it, exactly like Read.
        assert not screen.query("#library-media-content-search-controls")

        await _walk_next(screen, service, pilot, expected_row=1)
        assert screen._library_media_reader_session.mode == "analysis"
        assert not isinstance(screen.focused, Input), screen.focused
        assert not screen.query("#library-media-content-search-controls")

        # A second ] must still be a walk (the P0 symptom was it being typed).
        await _walk_next(screen, service, pilot, expected_row=2)
        assert not isinstance(screen.focused, Input), screen.focused


@pytest.mark.asyncio
async def test_find_on_the_analysis_tab_opens_the_bar_there_and_escape_closes_it():
    """Find searches what you are reading: on Analysis it opens the analysis
    bar (task-28026's Analysis->Read jump is retired), focuses its Input, and
    one Escape collapses it (live: the first Escape only blurred)."""
    host, service = _analysis_flow_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_0(screen, service, pilot)
        screen._library_media_reader_session = set_mode(
            screen._library_media_reader_session, "analysis"
        )
        screen._sync_library_media_viewer_or_recompose()
        await pilot.pause()
        screen.query_one("#library-media-reader-find", Button).press()
        search_input = await _wait_for_selector(
            screen, pilot, "#library-media-content-search"
        )
        await _wait_for_condition(
            pilot,
            lambda: search_input.has_focus,
            message="Find never focused the analysis search input.",
        )
        assert screen._library_media_reader_session.mode == "analysis"

        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        assert not screen.query("#library-media-content-search-controls")
        assert screen._library_media_find_open is False


@pytest.mark.asyncio
async def test_read_mode_walk_with_find_open_keeps_the_query_and_the_keys():
    """task-31269 AC2: an open bar survives an item change with its query,
    but focus stays where the user left it, so ] keeps walking."""
    host, service = _analysis_flow_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_0(screen, service, pilot)
        screen.query_one("#library-media-reader-find", Button).press()
        search_input = await _wait_for_selector(
            screen, pilot, "#library-media-content-search"
        )
        await _wait_for_condition(
            pilot, lambda: search_input.has_focus, message="Find never focused."
        )
        await pilot.press("i", "t", "e", "m")
        await pilot.pause()
        # Leave the field the way a reader does (F6 target = content body).
        screen.query_one("#library-media-viewer-content").focus()
        await pilot.pause()

        await _walk_next(screen, service, pilot, expected_row=1)
        assert screen._library_media_content_query == "item"
        assert screen.query("#library-media-content-search-controls")
        assert not isinstance(screen.focused, Input), screen.focused

        await _walk_next(screen, service, pilot, expected_row=2)
        assert screen._library_media_content_query == "item"


@pytest.mark.asyncio
async def test_find_toggles_the_bar_closed_when_it_is_open():
    """task-31269 AC4: a second Find press closes the bar (live: it did nothing)."""
    host, service = _analysis_flow_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_0(screen, service, pilot)
        find = screen.query_one("#library-media-reader-find", Button)
        find.press()
        await _wait_for_selector(screen, pilot, "#library-media-content-search-controls")
        screen.query_one("#library-media-reader-find", Button).press()
        await pilot.pause()
        await pilot.pause()
        assert not screen.query("#library-media-content-search-controls")
        assert screen._library_media_find_open is False
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `cd <worktree> && PYTHONPATH=<worktree> .venv-python -m pytest Tests/UI/test_library_media_render_fixes.py -p no:cacheprovider -q -k "analysis_mode_walk or find_on_the_analysis_tab or read_mode_walk_with_find_open or find_toggles" --no-header`
Expected: `test_analysis_mode_walk…` FAILS on `assert not screen.query("#library-media-content-search-controls")` (the Analysis bar is unconditional today); `test_find_on_the_analysis_tab…` FAILS on `mode == "analysis"` (today Find jumps to read); `test_read_mode_walk…` FAILS on `not isinstance(screen.focused, Input)` after the first walk (empty-query remount is impossible here because the query is "item" — if this one passes today, keep it as a regression guard); `test_find_toggles…` FAILS on the bar still present.

- [ ] **Step 3: Give the search controls an explicit focus token** (`library_media_content.py`)

Replace the constructor tail and `on_mount`:

```python
    def __init__(
        self,
        *,
        is_markdown: bool,
        query: str,
        matches: tuple[int, ...],
        match_index: int,
        focus_on_mount: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.is_markdown = is_markdown
        self.query = query
        self.matches = matches
        self.match_index = match_index
        self.focus_on_mount = focus_on_mount
        self.set_class(bool(self.query), self.ACTIVE_SEARCH_CLASS)

    def on_mount(self) -> None:
        """Take focus into the input only on the Find-gesture mount.

        task-31269 (critique #4 P0): inferring the gesture from an empty
        query made EVERY mount with no query steal focus -- each ]/[ item
        load in Analysis mode, and any walk with the bar open, parked the
        caret in this Input and the next key was typed. The gesture is now
        an explicit token the screen sets in the Find handler and the
        viewer consumes on the one compose that follows; every other mount
        (item change, mode flip, match navigation) leaves focus alone. The
        focus still runs from THIS widget's post-refresh hook because no
        screen-level defer can order itself after a nested recompose-mount
        (task-31237).
        """
        if self.focus_on_mount:
            self.call_after_refresh(self._focus_search_input)
```

- [ ] **Step 4: Thread the token through the viewer and gate the Analysis bar** (`library_media_viewer.py`)

Constructor: add the parameter after `find_open: bool = False,` and store it:

```python
        find_open: bool = False,
        find_focus_pending: bool = False,
```
```python
        self.find_open = find_open
        self.find_focus_pending = find_focus_pending
```
Docstring line under `find_open:` — add:
```python
            find_focus_pending: One-shot token from the Find gesture; the
                bar it mounts takes focus, then the token is consumed here
                so later syncs never re-take focus (task-31269).
```

Read-mode bar (the `if self.find_open or self.content_query:` block at ~338): pass and consume the token:

```python
            if self.find_open or self.content_query:
                matches = find_content_matches(
                    self.viewer.content, self.content_query
                )
                yield LibraryMediaContentSearchControls(
                    is_markdown=self.viewer.is_markdown,
                    query=self.content_query,
                    matches=matches,
                    match_index=self.content_match_index,
                    focus_on_mount=self.find_focus_pending,
                    id="library-media-content-search-controls",
                )
                # task-31269: the gesture token is spent on this mount.
                self.find_focus_pending = False
```

Analysis-mode bar (~626): gate it like Read and pass the token; the body keeps rendering regardless:

```python
        if self.viewer.analysis:
            # task-28026: the analysis is searchable through the SAME
            # controls/body the Read tab uses. task-31269: like Read, the
            # bar is collapsed until Find opens it -- an always-mounted bar
            # stole focus on every item load and swallowed the walk keys.
            matches = find_content_matches(self.viewer.analysis, self.content_query)
            if self.find_open or self.content_query:
                yield LibraryMediaContentSearchControls(
                    is_markdown=False,
                    query=self.content_query,
                    matches=matches,
                    match_index=self.content_match_index,
                    focus_on_mount=self.find_focus_pending,
                    id="library-media-content-search-controls",
                )
                self.find_focus_pending = False
            yield LibraryMediaContentBody(
                content=self.viewer.analysis,
                is_markdown=False,
                mode="raw",
                query=self.content_query,
                match_index=self.content_match_index,
                id="library-media-viewer-content",
            )
```

- [ ] **Step 5: Screen side — state, construction, sync, handler** (`library_screen.py`)

State init (next to line 2662):
```python
        self._library_media_find_open: bool = False
        # task-31269: one-shot Find-gesture token; consumed by the next
        # viewer build/sync so an item change can never move focus into
        # the search Input.
        self._library_media_find_focus_pending: bool = False
```

Add a helper right above `handle_library_media_reader_find`:
```python
    def _consume_library_media_find_focus(self) -> bool:
        """Return and clear the one-shot Find-gesture focus token (task-31269)."""
        pending = self._library_media_find_focus_pending
        self._library_media_find_focus_pending = False
        return pending
```

Viewer construction (line ~40481): after `find_open=self._library_media_find_open,` add
```python
            find_focus_pending=self._consume_library_media_find_focus(),
```

`_sync_library_media_viewer_state`: in the "nothing changed → return False" compare (the block that contains `and viewer.find_open == self._library_media_find_open`), add one more conjunct so a pending gesture always takes the recompose path:
```python
            and not self._library_media_find_focus_pending
```
and in the assign block right after `viewer.find_open = self._library_media_find_open` add:
```python
            viewer.find_focus_pending = self._consume_library_media_find_focus()
```

Handler — replace the body of `handle_library_media_reader_find`:
```python
        event.stop()
        if self._library_media_find_open:
            # task-31269 AC4: Find is a toggle -- a second press closes the
            # bar (live: it did nothing while the bar was open).
            self._close_library_media_find()
            self._sync_library_media_viewer_or_recompose()
            return
        # task-31269: Find searches the tab you are reading. The Analysis
        # tab's bar is gated exactly like Read's now, so Find no longer
        # jumps Analysis -> Read (task-28026's transition predates the
        # collapsed bar). A same-mode reset is a no-op by design.
        self._reset_library_media_search_on_mode_change(
            self._library_media_reader_session.mode
        )
        self._library_media_find_open = True
        self._library_media_find_focus_pending = True
        self._sync_library_media_viewer_or_recompose()
        self.call_after_refresh(self._focus_library_media_content_search_input)
```
Also update the handler docstring's first line to `"""Open (or close) the Find bar for the tab being read."""`.

- [ ] **Step 6: Update the retired contract test** (`Tests/UI/test_library_media_reader_flow.py:1452`)

Rename and rewrite the test so it pins the new semantics (keep the fake, add `_library_media_find_open=False` and `_library_media_find_focus_pending=False` to the `SimpleNamespace`):
```python
def test_find_from_analysis_opens_the_bar_on_the_analysis_tab():
    """task-31269: Find searches the tab you are reading. On the Analysis
    tab it opens the analysis bar in place (task-28026's Analysis->Read
    jump is retired) and hands the viewer a one-shot focus token; the
    query is untouched because the mode did not change."""
    ...  # same session/fake construction as before, plus the two attrs above
    LibraryScreen.handle_library_media_reader_find(
        fake, SimpleNamespace(stop=lambda: None)
    )
    assert fake._library_media_reader_session.mode == "analysis"
    assert fake._library_media_find_open is True
    assert fake._library_media_find_focus_pending is True
    assert fake._library_media_content_query == "needle"
```

- [ ] **Step 7: Run the four new tests plus the two existing Find tests; then the whole file, then reader_flow, each in its own process**

Run: `… -m pytest Tests/UI/test_library_media_render_fixes.py -p no:cacheprovider -q --no-header` → Expected: all pass (including `test_find_bar_collapsed_until_find_and_escape_recollapses`).
Run: `… -m pytest Tests/UI/test_library_media_reader_flow.py -p no:cacheprovider -q --no-header` → Expected: 49 passed (the renamed test included).
Run: `… -m pytest Tests/UI/test_library_media_reader_match_nav_t22209.py -p no:cacheprovider -q --no-header` → Expected: 7 passed (`test_a_new_document_rescans_for_the_same_query` proves the query survives the walk).
Run: `… -m pytest Tests/UI/test_library_shell.py -p no:cacheprovider -q --no-header -k "find or search or analysis"` → Expected: no new failures (the deep-link failure listed in task-31249 is pre-existing).

- [ ] **Step 8: User guide** — in `Docs/User_Guide/library/media-and-conversations.md` find the sentence describing Find on the Analysis tab (grep `Find`); make it read: "Find opens a search bar for the tab you are reading — the transcript on Read, the analysis on Analysis — and a second press or Escape closes it. Walking with `]`/`[` keeps the query and never moves your cursor into the field." Update the "Verified against" stamp to today's dev tip.

- [ ] **Step 9: Live verify (tmux, 235x52)** — seed via `python - <<EOF` with `MediaDatabase.add_media_with_keywords(..., analysis_content=...)` for three items (prefix `W4A`), launch `tmux -L w4a new-session -d -x 235 -y 52 "PYTHONPATH=<worktree> <python> -m tldw_chatbook.app"; sleep 14` (sleep in the same call), palette → library, click the Media rail, open item 1, click Analysis, press `]` three times → the banner/list must advance three times and the footer must never read `typing in field`. Capture to `Docs/…`-free scratch; soft-delete the seeds afterwards; `tmux -L w4a kill-server`.

- [ ] **Step 10: Commit**
```bash
git add tldw_chatbook/Widgets/Library/library_media_content.py tldw_chatbook/Widgets/Library/library_media_viewer.py tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_media_render_fixes.py Tests/UI/test_library_media_reader_flow.py Docs/User_Guide/library/media-and-conversations.md
git commit -m "fix(library): the Find gesture, not the mount, decides focus (task-31269)"
```

---

### Task 2: Receipts adopt the two-row grammar (task-31270, P1)

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py:147-170` (`BUNDLED_CSS`), `:815-848` (bulk-delete receipt), `:850-881` (review-dismiss receipt)
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` (library-media section, near the `#library-media-canvas > .ds-toolbar` rule from task-28025)
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`, `tldw_chatbook/css/screen_agentic_library.tcss`, `tldw_chatbook/css/widget_defaults_*` (whatever `python -m tldw_chatbook.css.build_css` rewrites)
- Test: `Tests/UI/test_library_media_render_fixes.py` (painted-text tests appended)

**Interfaces:**
- Produces: containers `#library-media-bulk-delete-receipt` and `#library-media-review-dismiss-receipt` become `Vertical` (class `library-media-receipt`) holding a copy `Static` (class `library-media-receipt-copy`) and a `Horizontal(classes="ds-toolbar library-media-receipt-actions")` with the unchanged buttons `#library-media-bulk-delete-undo`, `#library-media-bulk-delete-receipt-dismiss`, `#library-media-review-dismiss-undo`, `#library-media-review-dismiss-receipt-close`.
- Consumes: `self.canvas.delete_receipt_count`, `self.canvas.review_dismiss_receipt_name`, `_gate_stale_action`, `_gate_mutation_action` (unchanged).

- [ ] **Step 1: Write the failing painted-text tests** (append to `Tests/UI/test_library_media_render_fixes.py`; add `from tldw_chatbook.UI.Screens.library_screen import _sync_library_canvas` to the imports)

```python
def _items_pane_width(screen) -> int:
    return screen.query_one("#library-media-canvas").region.width


@pytest.mark.asyncio
async def test_delete_receipt_paints_undo_and_dismiss_at_the_items_pane_width():
    """task-31270 (critique #4 P1): the receipt's Undo was clipped to `Und`
    in the ~38-col Items pane (live cap_99). Painted text on purpose: a
    region assertion cannot see a label cut by its parent's width."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen._library_media_delete_receipt_ids = ("local:media:1",)
        _sync_library_canvas(screen, "media")
        receipt = await _wait_for_selector(
            screen, pilot, "#library-media-bulk-delete-receipt"
        )
        await pilot.pause()
        await pilot.pause()
        assert receipt.region.width <= _items_pane_width(screen)
        painted = _painted(host, receipt.region)
        assert "✓ deleted · 1 item · in Trash" in painted, painted
        assert "Undo" in painted, painted
        assert "Dismiss" in painted, painted


@pytest.mark.asyncio
async def test_dismiss_receipt_paints_undo_at_the_items_pane_width():
    """task-31270: the set-dismiss receipt clipped to `… Un` (live cap_83)."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen._review_dismiss_receipt_name = lambda: "2 selected items"
        _sync_library_canvas(screen, "media")
        receipt = await _wait_for_selector(
            screen, pilot, "#library-media-review-dismiss-receipt"
        )
        await pilot.pause()
        await pilot.pause()
        assert receipt.region.width <= _items_pane_width(screen)
        painted = _painted(host, receipt.region)
        assert "✓ dismissed · 2 selected items" in painted, painted
        assert "Undo" in painted, painted
        assert "Dismiss" in painted, painted
```

- [ ] **Step 2: Run them to verify they fail**

Run: `… -m pytest Tests/UI/test_library_media_render_fixes.py -p no:cacheprovider -q --no-header -k receipt_paints`
Expected: both FAIL with `"Undo" in painted` (or `"Dismiss"`) — the painted text ends in `Und`/`Un`.

- [ ] **Step 3: Restructure both receipts** (`library_media_canvas.py`)

Bulk-delete receipt (replace the `receipt_row = Horizontal(...)` block):
```python
        receipt_count = getattr(self.canvas, "delete_receipt_count", 0)
        if receipt_count:
            receipt_word = "item" if receipt_count == 1 else "items"
            # task-31270: two rows (copy, then actions), width 100% -- a
            # single content-width Horizontal clipped Undo to "Und" at the
            # Items pane's real width (critique #4). Same grammar as the
            # multi-row toolbars (task-30043).
            receipt = Vertical(
                id="library-media-bulk-delete-receipt",
                classes="library-media-receipt",
            )
            receipt.styles.height = "auto"
            with receipt:
                yield Static(
                    f"✓ deleted · {receipt_count} {receipt_word} · in Trash",
                    id="library-media-bulk-delete-receipt-copy",
                    classes="library-toolbar-count library-media-receipt-copy",
                    markup=False,
                )
                actions = Horizontal(classes="ds-toolbar library-media-receipt-actions")
                actions.styles.height = "auto"
                with actions:
                    undo = Button(
                        "Undo",
                        id="library-media-bulk-delete-undo",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    yield self._gate_stale_action(undo, "Undo")
                    dismiss = Button(
                        "Dismiss",
                        id="library-media-bulk-delete-receipt-dismiss",
                        classes="library-canvas-action",
                        compact=True,
                    )
                    yield self._gate_mutation_action(dismiss, "Dismiss")
```
Keep every existing comment about ADR-055 / task-4025 above the copy. Apply the identical shape to the review-dismiss receipt (`id="library-media-review-dismiss-receipt"`, copy `f"✓ dismissed · {dismissed_set_name}"` with id `library-media-review-dismiss-receipt-copy`, buttons `library-media-review-dismiss-undo` / `library-media-review-dismiss-receipt-close`).

- [ ] **Step 4: CSS at both tiers**

`BUNDLED_CSS` (append inside the string):
```css
    /* task-31270: receipts are two rows, full width; the copy wraps and the
     * action row keeps content-width buttons so Undo/Dismiss always paint. */
    .library-media-receipt {
        width: 100%;
        height: auto;
    }
    .library-media-receipt > .library-media-receipt-copy {
        width: 100%;
        height: auto;
    }
    .library-media-receipt > .library-media-receipt-actions {
        width: 100%;
        height: auto;
    }
```
`_agentic_terminal.tcss`: add the same three rules in the library-media section directly under the `#library-media-canvas > .ds-toolbar` block (task-28025), then run `python -m tldw_chatbook.css.build_css` and `python tldw_chatbook/css/check_bundle_sync.py` (exit 0 required).

- [ ] **Step 5: Run the two painted tests, then the files that exercise receipts, each in its own process**

Run: `… -m pytest Tests/UI/test_library_media_render_fixes.py -p no:cacheprovider -q --no-header` → Expected: all pass.
Run: `… -m pytest Tests/UI/test_library_media_side_by_side.py -p no:cacheprovider -q --no-header` → Expected: 32 passed (line ~1270 waits for `#library-media-bulk-delete-receipt`, id unchanged).
Run: `… -m pytest Tests/UI/test_library_multiselect_media.py -p no:cacheprovider -q --no-header` → Expected: 66 passed.
Run: `… -m pytest Tests/UI/test_review_set_picker.py -p no:cacheprovider -q --no-header` (if it exists; grep `review-dismiss-receipt` under Tests/ and run every file that matches) → Expected: pass.
Run: `… -m pytest Tests/CI/test_generated_stylesheet*.py -p no:cacheprovider -q --no-header` → Expected: pass (bundle regenerated and committed).

- [ ] **Step 6: Live verify (tmux 235x52)** — seed one `W4A` item, arm delete with `t`, confirm; the receipt must show `✓ deleted · 1 item · in Trash` on one row and `Undo  Dismiss` fully painted on the next; click `Undo` by label with `click.py`; capture. Repeat with a set: `Review these` → `Sets` → `Dismiss` → the `✓ dismissed …` receipt with a readable `Undo`. Clean up.

- [ ] **Step 7: Commit**
```bash
git add tldw_chatbook/Widgets/Library/library_media_canvas.py tldw_chatbook/css Tests/UI/test_library_media_render_fixes.py
git commit -m "fix(library): receipts take the two-row grammar so Undo always paints (task-31270)"
```

---

### Task 3: An explicit open bypasses auto-resume (task-31273, ruling)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:24082` (`handle_library_media_row`), `:42286-42340` (media branch of `_open_library_item_by_id`), `:38912-38966` (`_active_review_set_banner`)
- Test: `Tests/UI/test_review_set_banner.py` (append), `Tests/UI/test_review_set_walker.py` (append)

**Interfaces:**
- Produces: `LibraryScreen._cancel_pending_review_set_resume()` — cancels worker group `library_review_set_resume`; banner suffix `" · this item is not in the set"` when the loaded item is off-set.
- Consumes: `self.workers.cancel_group(self, group)`, `_maybe_auto_resume_review_set` (unchanged; still fires only from the rail-select seam).

- [ ] **Step 1: Write the failing tests**

Append to `Tests/UI/test_review_set_banner.py`:
```python
def test_banner_names_an_off_set_item_honestly(tmp_path):
    """task-31273 (user ruling): an explicit open of an item outside the
    active set keeps the set's banner but says the item is not in it, so
    the status line never claims a state the walk cannot honour."""
    service = _service(tmp_path)
    service.create_review_set(
        "All media", origin="browse", items=[(10, "A"), (11, "B")]
    )
    banner = LibraryScreen._active_review_set_banner(
        _banner_fake(service, loaded=99)
    )
    assert banner == "Reviewing: All media — 1 of 2 · 0 reviewed · this item is not in the set"
```

Append to `Tests/UI/test_review_set_walker.py`:
```python
def test_cancel_pending_resume_targets_the_resume_worker_group():
    """task-31273: explicit opens cancel an in-flight auto-resume so the
    landing worker cannot yank the Reader away from what the user chose."""
    calls = []
    fake = SimpleNamespace(workers=SimpleNamespace(cancel_group=lambda owner, group: calls.append(group)))
    LibraryScreen._cancel_pending_review_set_resume(fake)
    assert calls == ["library_review_set_resume"]


def test_row_press_and_open_by_id_cancel_a_pending_resume():
    """The two explicit-open seams both route through the cancel helper —
    pinned at the source level because the handlers need a live screen."""
    import inspect
    row_src = inspect.getsource(LibraryScreen.handle_library_media_row)
    open_src = inspect.getsource(LibraryScreen._open_library_item_by_id)
    assert "_cancel_pending_review_set_resume()" in row_src
    assert "_cancel_pending_review_set_resume()" in open_src
```
(`SimpleNamespace` and `LibraryScreen` are already imported in that file; add `from types import SimpleNamespace` if not.)

- [ ] **Step 2: Run them to verify they fail**

Run: `… -m pytest Tests/UI/test_review_set_banner.py Tests/UI/test_review_set_walker.py -p no:cacheprovider -q --no-header -k "off_set or cancel_pending or row_press_and_open"`
Expected: banner test FAILS (no suffix today); the two walker tests FAIL with `AttributeError: _cancel_pending_review_set_resume`.

- [ ] **Step 3: Implement**

Helper, placed directly above `_maybe_auto_resume_review_set` (line ~39629):
```python
    def _cancel_pending_review_set_resume(self) -> None:
        """Drop an in-flight auto-resume when the user opens something explicitly.

        task-31273 (user ruling at the critique #4 close): an explicit open --
        a row press, a deep link, open-by-id -- wins over the entry-time
        auto-resume; plain rail entry with no target still resumes.
        """
        self.workers.cancel_group(self, "library_review_set_resume")
```

`handle_library_media_row`: in the non-select-mode branch, immediately before the call that opens the viewer for the pressed row, add `self._cancel_pending_review_set_resume()`.

`_open_library_item_by_id`, media branch (the block starting `if source_type == "media":`): right after `self._library_selected_row_id = LIBRARY_ROW_BROWSE_MEDIA` add `self._cancel_pending_review_set_resume()`.

`_active_review_set_banner`: replace the `item_state` computation so an off-set loaded item is named:
```python
            item_state = ""
            current = next(
                (item for item in review_set.items if item.backing_media_id == loaded),
                None,
            )
            if current is not None and current.backing_media_id in live_ids:
                item_state = (
                    " · ✓ reviewed" if current.done else " · not yet reviewed"
                )
            elif loaded is not None and current is None:
                # task-31273: an explicit open outside the set keeps the
                # set's banner but never claims a state for this item.
                item_state = " · this item is not in the set"
```
Update the docstring's format line to mention the third suffix.

- [ ] **Step 4: Run the tests**

Run: `… -m pytest Tests/UI/test_review_set_banner.py -p no:cacheprovider -q --no-header` → Expected: all pass.
Run: `… -m pytest Tests/UI/test_review_set_walker.py -p no:cacheprovider -q --no-header` → Expected: 58 passed.
Run: `… -m pytest Tests/UI/test_library_review_sets_entry.py -p no:cacheprovider -q --no-header` (grep `library_review_set_resume` under Tests/ and run each matching file) → Expected: pass.

- [ ] **Step 5: Live verify** — with a `W4A` set mid-walk (2 of 3), Escape to the list, Enter on an item that is not in the set → banner ends with `· this item is not in the set`; `]` lands on the cursor item. Leave Media and return via the rail → still lands on the cursor item (plain entry resumes).

- [ ] **Step 6: Commit**
```bash
git add tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_review_set_banner.py Tests/UI/test_review_set_walker.py
git commit -m "fix(library): explicit opens bypass review-set auto-resume; off-set banner is honest (task-31273)"
```

---

### Task 4: Row-marker design note (task-31278)

**Files:**
- Create: `backlog/docs/design-library-row-state-markers.md` — already drafted alongside this plan (same commit); review it against `tldw_chatbook/Library/library_media_state.py:65-66,150-190` (`_MEDIA_SUMMARY_KEYS`, `validate_media_browse_items`), `tldw_chatbook/DB/Client_Media_DB_v2.py:2475-3040` (`library_summary=True` projection), `tldw_chatbook/Media/media_reading_scope_service.py:501,717-776`, `tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py:148`, and the six test files that build the shape (`grep -rlE 'library_summary|_MEDIA_SUMMARY_KEYS' Tests`).

- [ ] **Step 1: Verify every file:line the note cites still matches** (`sed -n` each range; fix the note if a line moved).
- [ ] **Step 2: Commit**
```bash
git add backlog/docs/design-library-row-state-markers.md
git commit -m "docs(library): design note for row state markers via the summary-contract bump (task-31278)"
```
The note's approval gate (AC#2) is the user's; do not start 28008/28009 in this PR.

---

### Task 5: Task hygiene, PR, landing

- [ ] **Step 1: Flip tasks** (cwd = worktree): `backlog task edit 31269 -s Done --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --plan "<Task 1 steps>" --notes "<what shipped, the token design, the retired Analysis->Read jump>" --plain`; same for 31270 (ACs 1-4) and 31273 (ACs 1-4). For 31278: check AC#1 and AC#4 only, leave status `In Progress` with a note "awaiting user approval (AC#2)".
- [ ] **Step 2: Derived-artifact checks** (no pipes): `python tldw_chatbook/css/check_bundle_sync.py`, `python scripts/check_persistent_diagnostic_inventory.py`, `python scripts/check_backlog_task_ids.py`, `python -m pytest Tests/CI/test_backlog_task_id_uniqueness.py -p no:cacheprovider -q`. All exit 0.
- [ ] **Step 3: Merged-head test pass** — `git fetch origin dev && git merge origin/dev` (or rebase), then re-run every test file touched or whose production method changed: `grep -rln '_close_library_media_find\|handle_library_media_reader_find\|_active_review_set_banner\|bulk-delete-receipt\|review-dismiss-receipt' Tests/` and run each file in its own process.
- [ ] **Step 4: Push and open the PR** against `dev`, title `fix(library): wave 4 A — Find focus token, two-row receipts, explicit-open bypass (tasks 31269/31270/31273/31278)`; body = the three fixes with their critique evidence, the retired 28026 Find transition called out, live-verification captures, and the standard footer.
- [ ] **Step 5: Land** — address Qodo with evidence; the required job `Derived artifacts reproduce from their sources` spawns only after `PR Fast Lane` completes (`needs: [pr-fast-lane]`), so an absent check-run for the first 10-20 minutes is normal; update-branch when BEHIND; `gh pr merge --admin --merge` once both are green.
