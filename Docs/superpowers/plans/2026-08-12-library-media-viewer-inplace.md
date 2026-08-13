# Library Media Viewer In-Place Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve Library content-search behavior while eliminating screen remounts and repeated full-document Markdown parses from match navigation, mode switching, and legacy per-keystroke search.

**Architecture:** `LibraryScreen` remains the canonical owner of query, match index, and mode. A focused search-controls widget and a focused lazy content-body widget own in-place presentation updates, while `LibraryMediaViewer` provides the narrow screen-facing coordination API. The legacy `MediaViewerPanel` keeps its existing search semantics but applies non-empty typing through a generation-guarded 250 ms timer and performs one Markdown update per applied search.

Task 1 verification also repairs the Windows test-network boundary under ADR-058. The process-wide guard wraps the captured real `socket.socketpair()` with a same-thread dynamic exemption so Python's Windows Proactor bootstrap can create its internal wakeup channel without permitting ordinary or concurrent-thread application egress.

**Tech Stack:** Python 3.11+, Textual 8.x, Rich `Text`, pytest, pytest-asyncio, Ruff.

## Global Constraints

- Preserve the Library viewer's current Enter-to-search interaction; `Input.Changed` must not apply Library searches.
- Preserve the current Rendered default for Markdown Library items and Raw default for non-Markdown items.
- Preserve case-insensitive, one-match-per-source-line counting and first-occurrence-per-line Raw highlighting.
- Preserve wraparound Previous/Next navigation and source-line scrolling.
- Never call `LibraryScreen.refresh(recompose=True)`, viewer recompose, or `Markdown.update()` for Library match navigation.
- Keep already-mounted Raw and Rendered content widgets mounted; mode changes use visibility after each view's first lazy mount.
- Use `MEDIA_CONTENT_SEARCH_DEBOUNCE_SECONDS = 0.25` for legacy content search.
- A legacy search callback must be invalidated by newer input, clear, media replacement, clear-display, and unmount.
- The legacy display path performs one `Markdown.update(content)` and never the empty-string cache-busting update.
- Do not add dependencies, persistence, workers, or changes to Markdown source-line geometry.
- Keep the test network guard installed at import time and default-deny for protected address families.
- A socketpair exemption must be current-thread-only, nested, restored in `finally`, and must never mutate `_allowed` or `_INET_FAMILIES`.
- Ordinary same-thread and concurrent-thread `AF_INET`/`AF_INET6` egress remains blocked and recorded.
- Literal async pytest commands must work on Windows without the TASK-15100 guarded-family mutation workaround.
- ADR required: yes.
- ADR path: `backlog/decisions/058-thread-scoped-test-socketpair-exemption.md`.
- ADR reason: the review-expanded scope changes the repository-wide test network security boundary and its runtime interception contract.

## File Structure

- Create `tldw_chatbook/Widgets/Library/library_media_content.py`: focused Library search chrome, Raw renderable construction, and lazy persistent Raw/Rendered body ownership.
- Modify `tldw_chatbook/Widgets/Library/library_media_viewer.py`: compose the focused children and expose screen-facing synchronization methods; retain metadata, analysis, highlights, and action composition.
- Modify `tldw_chatbook/UI/Screens/library_screen.py`: replace content-search and mode full-screen recomposes with viewer synchronization and post-layout scrolling.
- Create `Tests/Library/test_library_media_content.py`: mounted component contracts for scoped search updates, lazy mode mounting, search state, focus, identity, and rapid-mode races.
- Modify `Tests/UI/test_library_shell.py`: product-path regressions for Enter submission, screen/viewer/Markdown identity, focus, highlighting, scrolling, mode reuse, and latency evidence.
- Modify `tldw_chatbook/Widgets/Media/media_viewer_panel.py`: guarded legacy debounce lifecycle and single-update rendering.
- Create `Tests/UI/test_media_viewer_content_search_debounce.py`: mounted timer and Markdown update-count regressions.
- Modify `Tests/network_guard.py`: wrap the captured real socketpair with a nested thread-local dynamic exemption while preserving default denial everywhere else.
- Modify `Tests/test_network_guard.py`: prove real Windows socketpair operation, cross-thread isolation, exceptional restoration, idempotence, and ordinary denial.
- Modify `backlog/docs/lessons-testing-evidence.md`: replace the temporary TASK-15100 workaround guidance with the verified ADR-058 resolution and literal command evidence.
- Modify `backlog/tasks/task-15458 - Library-media-viewer---in-place-match-navigation-instead-of-full-document-re-parse.md`: status, implementation plan, evidence, completed criteria, and implementation notes.

---

### Task 1: Scoped Library Search Controls

**Files:**

- Create: `tldw_chatbook/Widgets/Library/library_media_content.py`
- Create: `Tests/Library/test_library_media_content.py`
- Modify: `Tests/network_guard.py`
- Modify: `Tests/test_network_guard.py`
- Modify: `backlog/docs/lessons-testing-evidence.md`

**Interfaces:**

- Consumes: Textual `Input`, `Static`, `Button`, `Horizontal`, and `Widget.refresh(recompose=True)`.
- Produces: `LibraryMediaContentSearchControls(is_markdown: bool, query: str, matches: tuple[int, ...], match_index: int, **kwargs)`.
- Produces: `sync_query_state(*, is_markdown: bool, query: str, matches: tuple[int, ...], match_index: int) -> None`.
- Produces: `sync_match_index(*, matches: tuple[int, ...], match_index: int) -> None`.
- Produces: a private guarded `socket.socketpair` wrapper whose exemption is limited to the current thread's dynamic call extent.

- [ ] **Step 1: Write mounted failing tests for structural and in-place update paths**

Add a minimal `App` harness and tests that begin with an active query, capture both navigation button objects, focus Next, and update only the match index:

```python
class SearchControlsHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield LibraryMediaContentSearchControls(
            is_markdown=True,
            query="budget",
            matches=(2, 8),
            match_index=0,
            id="controls",
        )


@pytest.mark.asyncio
async def test_match_index_sync_preserves_navigation_identity_and_focus() -> None:
    app = SearchControlsHarness()
    async with app.run_test() as pilot:
        controls = app.query_one("#controls", LibraryMediaContentSearchControls)
        previous = app.query_one("#library-media-content-search-prev", Button)
        next_button = app.query_one("#library-media-content-search-next", Button)
        next_button.focus()

        controls.sync_match_index(matches=(2, 8), match_index=1)
        await pilot.pause()

        assert app.query_one("#library-media-content-search-prev") is previous
        assert app.query_one("#library-media-content-search-next") is next_button
        assert app.focused is next_button
        assert str(app.query_one("#library-media-content-search-status", Static).renderable) == (
            "Match 2 of 2 matches"
        )
```

Add separate tests proving:

```python
controls.sync_query_state(
    is_markdown=True,
    query="cost",
    matches=(4,),
    match_index=0,
)
assert app.query_one("#library-media-content-search", Input) is search_input
assert app.query_one("#library-media-content-search-next", Button) is next_button

controls.sync_query_state(
    is_markdown=True,
    query="",
    matches=(),
    match_index=0,
)
await pilot.pause()
assert not app.query("#library-media-content-search-status")
```

The active-to-active test must assert Input and buttons retain identity. The active-to-blank test must assert status and buttons are removed after `await pilot.pause()`. The blank-to-active test must re-query the structurally replaced Input and assert the Markdown-specific placeholder remains `"Search content (raw text)…"`.

- [ ] **Step 2: Run the focused tests and verify the missing module fails**

Run:

```powershell
& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest Tests/Library/test_library_media_content.py -v
```

Expected: collection fails because `tldw_chatbook.Widgets.Library.library_media_content` does not exist.

- [ ] **Step 3: Implement the search-controls widget with one structural boundary**

Add the class and status formatter. The implementation must update existing widgets for active-to-active and match-only changes, and recompose only for blank/non-blank structure changes:

```python
class LibraryMediaContentSearchControls(Vertical):
    def __init__(
        self,
        *,
        is_markdown: bool,
        query: str,
        matches: tuple[int, ...],
        match_index: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.is_markdown = is_markdown
        self.query = query
        self.matches = matches
        self.match_index = match_index

    def compose(self) -> ComposeResult:
        yield Input(
            value=self.query,
            placeholder=(
                "Search content (raw text)…"
                if self.is_markdown
                else "Search content…"
            ),
            id="library-media-content-search",
        )
        if not self.query:
            return
        yield Static(
            self._status_text(),
            id="library-media-content-search-status",
            markup=False,
        )
        toolbar = Horizontal(classes="ds-toolbar")
        toolbar.styles.height = "auto"
        with toolbar:
            yield Button(
                "◀ Prev",
                id="library-media-content-search-prev",
                classes="library-canvas-action",
                compact=True,
            )
            yield Button(
                "Next ▶",
                id="library-media-content-search-next",
                classes="library-canvas-action",
                compact=True,
            )

    def sync_match_index(
        self, *, matches: tuple[int, ...], match_index: int
    ) -> None:
        self.matches = matches
        self.match_index = match_index
        self.query_one("#library-media-content-search-status", Static).update(
            self._status_text()
        )
```

In `sync_query_state`, capture `was_active = bool(self.query)` before assignment and `is_active = bool(query)` after assignment. Call `self.refresh(recompose=True)` only when those booleans differ. Otherwise update the mounted Input value/placeholder and mounted status in place. `_status_text()` returns `""`, `"No matches"`, or `f"Match {wrapped + 1} of {len(self.matches)} matches"`.

- [ ] **Step 4: Run the controls tests and mutation-check the identity guard**

Run the focused test command from Step 2. Then temporarily replace the `sync_match_index` status update with `self.refresh(recompose=True)` and rerun `test_match_index_sync_preserves_navigation_identity_and_focus`; confirm it fails on identity or focus. Restore the in-place implementation and rerun the full file green.

- [ ] **Step 5: Commit the scoped controls**

```powershell
git add tldw_chatbook/Widgets/Library/library_media_content.py Tests/Library/test_library_media_content.py
git commit -m "feat(library): add scoped media search controls"
```

- [ ] **Step 6: Add failing status and Windows network-guard regressions**

In `Tests/Library/test_library_media_content.py`, directly cover both status branches omitted by the first implementation review:

```python
def test_status_text_reports_no_matches() -> None:
    controls = LibraryMediaContentSearchControls(
        is_markdown=True,
        query="missing",
        matches=(),
        match_index=0,
    )
    assert controls._status_text() == "No matches"


def test_status_text_wraps_match_index_before_formatting() -> None:
    controls = LibraryMediaContentSearchControls(
        is_markdown=True,
        query="budget",
        matches=(2, 8),
        match_index=3,
    )
    assert controls._status_text() == "Match 2 of 2 matches"
```

In `Tests/test_network_guard.py`, add a Windows-only real socketpair test that exchanges one byte, closes both sockets in `finally`, and asserts `network_guard.blocked_attempts() == ()`:

```python
@pytest.mark.skipif(sys.platform != "win32", reason="Windows TCP fallback only")
def test_windows_socketpair_bootstrap_is_allowed_without_guard_mutation() -> None:
    families_before = network_guard._INET_FAMILIES
    left, right = socket.socketpair()
    try:
        left.sendall(b"x")
        assert right.recv(1) == b"x"
    finally:
        left.close()
        right.close()
    assert network_guard._INET_FAMILIES is families_before
    assert network_guard.blocked_attempts() == ()
```

Add a cross-thread test that monkeypatches `network_guard._real_socketpair` with a function that sets an `entered` event and waits on a `release` event. Start `socket.socketpair()` in one thread; while that thread is inside the wrapper, attempt a direct `AF_INET` `socket.connect()` from the test thread and assert `BlockedNetworkAccess` plus a drained record. Release and join the worker in `finally`, asserting it terminated.

Add a failure-restoration test that monkeypatches `_real_socketpair` to raise `RuntimeError("socketpair failed")`, asserts `socket.socketpair()` propagates that error, then asserts a direct `AF_INET` `socket.connect()` is blocked and recorded. Add an idempotence assertion that repeated `network_guard.install()` leaves `socket.socketpair is network_guard._guarded_socketpair`.

- [ ] **Step 7: Run literal commands and verify the pre-fix failures**

Run without changing `network_guard._INET_FAMILIES`:

```powershell
& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest Tests/test_network_guard.py -q
& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest Tests/Library/test_library_media_content.py -q
```

Expected before implementation: the Windows real-socketpair test and literal async component tests fail when `_fallback_socketpair()` reaches the guarded loopback connect. The status tests expose any missing `No matches` or modulo formatting behavior.

- [ ] **Step 8: Implement the minimal thread-scoped socketpair exemption**

In `Tests/network_guard.py`, import `threading`, capture `_real_socketpair = socket.socketpair` beside the other captured functions, and create `_socketpair_state = threading.local()`. Add:

```python
def _inside_socketpair() -> bool:
    return getattr(_socketpair_state, "depth", 0) > 0


def _guarded_socketpair(*args: Any, **kwargs: Any):  # noqa: ANN401
    previous_depth = getattr(_socketpair_state, "depth", 0)
    _socketpair_state.depth = previous_depth + 1
    try:
        return _real_socketpair(*args, **kwargs)
    finally:
        _socketpair_state.depth = previous_depth
```

Change `_should_block` to deny only when neither the existing explicit global opt-in nor the current thread's socketpair extent permits the operation:

```python
return not _allowed and not _inside_socketpair() and family in _INET_FAMILIES
```

Apply the same `_inside_socketpair()` condition to `_guarded_create_connection`, because a future standard-library socketpair implementation may use that captured public helper rather than `socket.connect` directly. In `install()`, assign `socket.socketpair = _guarded_socketpair` during the same idempotent patch operation. Do not change `_allowed`, `_INET_FAMILIES`, `_deny`, or the recording contract.

- [ ] **Step 9: Run guard and component tests, then mutation-check isolation**

Run both literal commands from Step 7. Then temporarily replace the thread-local state with one process-global depth; confirm the cross-thread test fails because the direct connection is permitted. Restore `threading.local()` and rerun both commands green.

Run the existing synchronous denial tests together with the async component test and confirm protected families remain unchanged before and after the process:

```powershell
& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest Tests/test_network_guard.py Tests/Library/test_library_media_content.py -q
```

- [ ] **Step 10: Record the resolved Windows incident and commit Task 1 review fixes**

Update the TASK-15100 Windows follow-up in `backlog/docs/lessons-testing-evidence.md` with the ADR-058 resolution: the real socketpair is wrapped with a nested current-thread exemption; ordinary and concurrent-thread egress remain denied; the literal async pytest command now passes without mutating protected families. Preserve the original incident as evidence rather than deleting it.

```powershell
git add Tests/network_guard.py Tests/test_network_guard.py Tests/Library/test_library_media_content.py backlog/docs/lessons-testing-evidence.md
git commit -m "fix(testing): allow guarded Windows socketpair bootstrap"
```

---

### Task 2: Lazy Persistent Library Content Body

**Files:**

- Modify: `tldw_chatbook/Widgets/Library/library_media_content.py`
- Modify: `Tests/Library/test_library_media_content.py`

**Interfaces:**

- Consumes: `front_matter_parser_factory()`, Rich `Text`, and `find_content_matches(content: str, query: str) -> tuple[int, ...]`.
- Produces: `LibraryMediaContentBody(content: str, is_markdown: bool, mode: str, query: str, match_index: int, **kwargs)`.
- Produces: `async sync_mode(mode: str) -> None`.
- Produces: `sync_search(query: str, match_index: int) -> None`.

- [ ] **Step 1: Write failing lazy-mount, reuse, hidden-state, and race tests**

Add a body harness whose constructor accepts a body instance. Test a Markdown body initialized in Raw mode:

```python
body = LibraryMediaContentBody(
    content="# Heading\n\nbudget one\nbudget two",
    is_markdown=True,
    mode="raw",
    query="",
    match_index=0,
    id="library-media-viewer-content",
)
async with BodyHarness(body).run_test() as pilot:
    raw = body.query_one("#library-media-viewer-content-text", Static)
    assert not body.query("#library-media-viewer-content-markdown")

    await body.sync_mode("rendered")
    markdown = body.query_one("#library-media-viewer-content-markdown", Markdown)
    await body.sync_mode("raw")
    await body.sync_mode("rendered")

    assert body.query_one("#library-media-viewer-content-text") is raw
    assert body.query_one("#library-media-viewer-content-markdown") is markdown
```

Add a Rendered-initial test that calls `body.sync_search("budget", 1)` before Raw has mounted, then switches to Raw and asserts the newly mounted Rich `Text` contains two styled `budget` spans with only the second using `reverse bold`.

Add a rapid-mode test that delays the first Rendered mount behind an `asyncio.Event`, starts `rendered_task = asyncio.create_task(body.sync_mode("rendered"))`, starts `raw_task = asyncio.create_task(body.sync_mode("raw"))`, releases the event, awaits both tasks, and asserts Raw is displayed, Rendered is hidden, and each ID occurs exactly once.

- [ ] **Step 2: Run the body tests and verify missing interfaces fail**

Run:

```powershell
& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest Tests/Library/test_library_media_content.py -k 'body or mode or raw' -v
```

Expected: failures report that `LibraryMediaContentBody` and its synchronization methods are absent.

- [ ] **Step 3: Implement Raw renderable construction and lazy child ownership**

Move the existing safe Rich slicing algorithm into a module helper that accepts complete values rather than viewer state:

```python
def build_raw_content_renderable(
    content: str, query: str, match_index: int
) -> Text | str:
    display_content = content or "No stored content."
    normalized_query = query.strip()
    if not normalized_query or not content:
        return display_content
    matches = find_content_matches(content, normalized_query)
    if not matches:
        return display_content
    current_line = matches[match_index % len(matches)]
    needle = normalized_query.lower()
    text = Text()
    for line_index, line in enumerate(display_content.split("\n")):
        if line_index:
            text.append("\n")
        hit = line.lower().find(needle)
        if hit < 0:
            text.append(line)
            continue
        text.append(line[:hit])
        text.append(
            line[hit : hit + len(needle)],
            style="reverse bold" if line_index == current_line else "reverse",
        )
        text.append(line[hit + len(needle) :])
    return text
```

Implement `LibraryMediaContentBody(VerticalScroll)` with `_raw_widget`, `_markdown_widget`, `_desired_mode`, `_mount_lock`, `_query`, and `_match_index`. `compose()` constructs only the selected initial mode. `_ensure_mode_mounted(mode)` awaits `self.mount(...)` only when the corresponding reference is `None`.

Implement synchronization with latest-request-wins visibility:

```python
async def sync_mode(self, mode: str) -> None:
    self._desired_mode = mode
    async with self._mount_lock:
        await self._ensure_mode_mounted(mode)
        desired = self._desired_mode
        await self._ensure_mode_mounted(desired)
        if self._raw_widget is not None:
            self._raw_widget.display = desired == "raw"
        if self._markdown_widget is not None:
            self._markdown_widget.display = desired == "rendered"

def sync_search(self, query: str, match_index: int) -> None:
    self._query = query
    self._match_index = match_index
    if self._raw_widget is not None:
        self._raw_widget.update(
            build_raw_content_renderable(self.content, query, match_index)
        )
```

Reject modes outside `{"raw", "rendered"}` with `ValueError`. Non-Markdown bodies normalize every requested mode to Raw and never construct Markdown.

- [ ] **Step 4: Run body tests and mutation-check latest-request-wins**

Run the command from Step 2. Then temporarily apply visibility from the method's `mode` argument rather than `_desired_mode`; confirm the rapid Rendered-then-Raw test fails. Restore latest-desired visibility and rerun the entire component test file green.

- [ ] **Step 5: Commit the content body**

```powershell
git add tldw_chatbook/Widgets/Library/library_media_content.py Tests/Library/test_library_media_content.py
git commit -m "feat(library): keep media content views mounted"
```

---

### Task 3: Integrate In-Place Updates Through the Library Product Path

**Files:**

- Modify: `tldw_chatbook/Widgets/Library/library_media_viewer.py:5-18,67-159,246-402`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:24815-24972`
- Modify: `Tests/UI/test_library_shell.py:4478-4928`

**Interfaces:**

- Consumes: `LibraryMediaContentSearchControls.sync_query_state`, `LibraryMediaContentSearchControls.sync_match_index`, `LibraryMediaContentBody.sync_search`, and `LibraryMediaContentBody.sync_mode` from Tasks 1-2.
- Produces: `LibraryMediaViewer.sync_query_state(*, query: str, matches: tuple[int, ...], match_index: int) -> None`.
- Produces: `LibraryMediaViewer.sync_match_index(*, matches: tuple[int, ...], match_index: int) -> None`.
- Produces: `async LibraryMediaViewer.sync_mode(mode: str) -> None`.

- [ ] **Step 1: Add failing full-path identity, parse-count, focus, mode-reuse, and latency regressions**

Extend the existing Library harness tests. After opening a Markdown item, capture the exact screen, viewer, and Markdown objects, submit a query, focus and press Next, and assert all identities survive:

```python
screen_before = screen
viewer_before = screen.query_one("#library-media-viewer", LibraryMediaViewer)
markdown_before = screen.query_one(
    "#library-media-viewer-content-markdown", Markdown
)

await _submit_content_search_query(screen, pilot, "budget")
next_button = screen.query_one("#library-media-content-search-next", Button)
previous_button = screen.query_one("#library-media-content-search-prev", Button)
next_button.focus()
next_button.press()
await pilot.pause()

assert screen is screen_before
assert screen.query_one("#library-media-viewer") is viewer_before
assert screen.query_one("#library-media-viewer-content-markdown") is markdown_before
assert screen.query_one("#library-media-content-search-next") is next_button
assert screen.query_one("#library-media-content-search-prev") is previous_button
assert screen.focused is next_button
```

Wrap `Markdown.update` with a recording spy before navigation, await every returned `AwaitComplete`, and assert navigation adds zero calls. Update the existing Raw/Rendered test to assert both child objects remain mounted after their first use and only `display` changes. Preserve the existing `test_library_shell_media_viewer_defaults_markdown_item_to_rendered` expectation.

Add a Rendered-search test: submit while Rendered is visible, switch to Raw, and assert the Raw renderable already marks the selected query. Add a scroll spy and assert `scroll_to(y=selected_line, animate=False)` runs only after a Pilot refresh boundary.

For evidence, build a deterministic document with 2,000 Markdown lines and 100 matching lines. Time the identical submit/Next/Prev sequence with `time.perf_counter()`, print `TASK-15458 latency median_ms=<value>` under `pytest -s`, collect the unique Markdown object IDs observed after every action as the construction-count proxy, and assert identity and Markdown update counts rather than a wall-clock threshold. Capture the failing pre-change measurement in the task's Implementation Notes before changing production code.

- [ ] **Step 2: Run the new product-path tests and record the red baseline**

Run:

```powershell
& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest Tests/UI/test_library_shell.py -k 'media_content_search or media_viewer_raw_toggle or media_viewer_defaults_markdown or media_viewer_inplace' -v -s
```

Expected: identity tests fail because current handlers recompose the screen/viewer, mode reuse fails because inactive content widgets are removed, and the recorded baseline shows Markdown reconstruction during navigation.

- [ ] **Step 3: Compose focused children and add viewer coordination methods**

In `LibraryMediaViewer.compose`, replace `_compose_content_search()` and the inline `VerticalScroll` branch with:

```python
matches = find_content_matches(self.viewer.content, self.content_query)
yield LibraryMediaContentSearchControls(
    is_markdown=self.viewer.is_markdown,
    query=self.content_query,
    matches=matches,
    match_index=self.content_match_index,
    id="library-media-content-search-controls",
)
yield LibraryMediaContentBody(
    content=self.viewer.content,
    is_markdown=self.viewer.is_markdown,
    mode=self.content_mode,
    query=self.content_query,
    match_index=self.content_match_index,
    id="library-media-viewer-content",
)
```

Delete the moved search composition/status and Raw-renderable methods. Add narrow coordinators:

```python
def sync_query_state(
    self, *, query: str, matches: tuple[int, ...], match_index: int
) -> None:
    self.content_query = query
    self.content_match_index = match_index
    self.query_one(LibraryMediaContentSearchControls).sync_query_state(
        is_markdown=self.viewer.is_markdown,
        query=query,
        matches=matches,
        match_index=match_index,
    )
    self.query_one(LibraryMediaContentBody).sync_search(query, match_index)

def sync_match_index(
    self, *, matches: tuple[int, ...], match_index: int
) -> None:
    self.content_match_index = match_index
    self.query_one(LibraryMediaContentSearchControls).sync_match_index(
        matches=matches,
        match_index=match_index,
    )
    self.query_one(LibraryMediaContentBody).sync_search(
        self.content_query, match_index
    )
```

`sync_mode` updates `content_mode`, both toggle labels, both `-selected` classes, requests `refresh(layout=True)` for changed auto-width labels, and awaits the body method. Query only the exact child types/IDs and catch no exceptions here; screen teardown handling belongs at the screen boundary.

- [ ] **Step 4: Replace screen recomposes with narrow synchronization**

Add a helper that returns the mounted viewer or `None`, catching only `NoMatches` and `QueryError`:

```python
def _mounted_library_media_viewer(self) -> LibraryMediaViewer | None:
    try:
        return self.query_one("#library-media-viewer", LibraryMediaViewer)
    except (NoMatches, QueryError):
        return None
```

On submission, trim once and return immediately when the result equals the canonical query. Otherwise store the query and zero index, compute matches once, call `viewer.sync_query_state(...)`, schedule `_focus_library_media_content_search_input`, and schedule scroll only when matches exist. Do not recompose the screen.

On Next/Previous, compute matches once, wrap the canonical index, call `viewer.sync_match_index(...)`, and use `call_after_refresh(self._scroll_library_media_content_to_line, line_index)` so Rich `Text` wrapping/layout settles before scrolling.

Make mode handlers and `_set_library_media_content_mode` async:

```python
async def _set_library_media_content_mode(self, mode: str) -> None:
    if self._library_media_view != "viewer":
        return
    if self._library_media_content_mode == mode:
        return
    self._library_media_content_mode = mode
    viewer = self._mounted_library_media_viewer()
    if viewer is None:
        return
    await viewer.sync_mode(mode)
```

The button handlers stop their events and await this method. Update docstrings that currently explain full-screen recomposition.

- [ ] **Step 5: Run product-path tests and mutation-check the no-recompose guarantee**

Run the command from Step 2 and the full `Tests/Library/test_library_media_content.py`. Then temporarily restore `self.refresh(recompose=True)` in `_advance_library_media_content_match`; confirm the identity/focus test fails. Restore the narrow call and rerun green. Record the post-change latency, object identities, and Markdown counts beside the baseline in the task notes.

- [ ] **Step 6: Commit Library product-path integration**

```powershell
git add tldw_chatbook/Widgets/Library/library_media_viewer.py tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_shell.py backlog/tasks/task-15458*.md
git commit -m "perf(library): update media matches in place"
```

---

### Task 4: Debounce Legacy Media Content Search

**Files:**

- Modify: `tldw_chatbook/Widgets/Media/media_viewer_panel.py:1-20,981-1008,1204-1315`
- Create: `Tests/UI/test_media_viewer_content_search_debounce.py`

**Interfaces:**

- Consumes: Textual `Widget.set_timer(delay: float, callback) -> Timer`, synchronous `Timer.stop()`, and `Markdown.update(markdown: str) -> AwaitComplete`.
- Produces: `MEDIA_CONTENT_SEARCH_DEBOUNCE_SECONDS = 0.25`.
- Produces private lifecycle methods `_invalidate_content_search_timer() -> int` and `_apply_debounced_content_search(generation: int, query: str) -> None`.

- [ ] **Step 1: Write mounted failing debounce and stale-lifecycle tests**

Create a minimal app mounting a real `MediaViewerPanel`, load content containing multiple `budget` hits, and wrap the mounted `#content-display` Markdown instance's `update` method. Clear the recording list after the initial media render completes so the assertions measure search-triggered updates only:

```python
updates: list[tuple[str, AwaitComplete]] = []
original_update = content_display.update

def recording_update(markdown: str) -> AwaitComplete:
    completion = original_update(markdown)
    updates.append((markdown, completion))
    return completion

monkeypatch.setattr(content_display, "update", recording_update)
```

Set the Input value to `"b"`, `"bu"`, and `"budget"` with pauses shorter than 250 ms. Assert no update occurs before the window, then wait past the window, await every recorded completion, and assert exactly one non-empty payload.

Add separate tests that arm a query and then:

- clear the Input before 250 ms and assert one immediate unhighlighted update plus no later update;
- call `load_media()` with a different record before 250 ms and assert the old query never appears in the new payload;
- call `clear_display()` before 250 ms and assert the cleared display remains authoritative;
- remove the panel before 250 ms and assert the stale callback performs no update;
- load two records with the same or missing ID and assert the generation, not record identity, invalidates the first query.

- [ ] **Step 2: Run the new tests and verify current per-keystroke behavior fails**

Run:

```powershell
& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest Tests/UI/test_media_viewer_content_search_debounce.py -v
```

Expected: burst tests observe immediate renders and the empty-string plus content double update.

- [ ] **Step 3: Add guarded timer state and remove the double update**

Import `partial` and `Timer`, define the constant, and initialize:

```python
self._content_search_timer: Timer | None = None
self._content_search_generation = 0
self._content_search_query = ""
```

Implement invalidation and the guarded callback:

```python
def _invalidate_content_search_timer(self) -> int:
    if self._content_search_timer is not None:
        self._content_search_timer.stop()
        self._content_search_timer = None
    self._content_search_generation += 1
    return self._content_search_generation

def _apply_debounced_content_search(self, generation: int, query: str) -> None:
    if generation != self._content_search_generation:
        return
    if query != self._content_search_query or not self.is_mounted:
        return
    self._content_search_timer = None
    self.search_content(query)
```

For non-empty `Input.Changed`, invalidate, store the exact value, and schedule the guarded callback:

```python
generation = self._invalidate_content_search_timer()
self._content_search_query = event.value
self._content_search_timer = self.set_timer(
    MEDIA_CONTENT_SEARCH_DEBOUNCE_SECONDS,
    partial(self._apply_debounced_content_search, generation, event.value),
)
```

For empty input, invalidate, clear `_content_search_query`, and call `clear_search()` immediately.

Call invalidation before state changes in `load_media`, `clear_display`, and `on_unmount`. In `update_content_display`, delete `content_display.update("")` and retain exactly one `content_display.update(content)` call. Keep navigation's immediate `highlight_current_match()` behavior unchanged.

- [ ] **Step 4: Run debounce tests and mutation-check generation invalidation**

Run the command from Step 2 plus:

```powershell
& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest Tests/UI/test_media_window_v2_parity.py Tests/UI/test_media_handoffs.py -q
```

Temporarily remove the generation increment from `load_media`; confirm the same/missing-ID stale-query test fails. Restore it and rerun both commands green.

- [ ] **Step 5: Commit the legacy debounce**

```powershell
git add tldw_chatbook/Widgets/Media/media_viewer_panel.py Tests/UI/test_media_viewer_content_search_debounce.py
git commit -m "perf(media): debounce legacy content search"
```

---

### Task 5: Verification, Evidence, and Task Closeout

**Files:**

- Modify: `backlog/tasks/task-15458 - Library-media-viewer---in-place-match-navigation-instead-of-full-document-re-parse.md`
- Modify only if the task revealed a reusable incident: `backlog/docs/lessons-testing-evidence.md`

**Interfaces:**

- Consumes: all behavior and evidence produced by Tasks 1-4.
- Produces: completed acceptance criteria, exact verification evidence, and concise implementation notes.

- [ ] **Step 1: Run focused behavior and performance verification**

```powershell
& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest Tests/Library/test_library_media_content.py Tests/UI/test_media_viewer_content_search_debounce.py -v
& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest Tests/test_network_guard.py -q
& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest Tests/UI/test_library_shell.py -k 'media_content_search or media_viewer_raw_toggle or media_viewer_defaults_markdown or media_viewer_inplace' -v -s
& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest Tests/UI/test_media_window_v2_parity.py Tests/UI/test_media_handoffs.py -q
```

Expected: all focused tests pass; the output includes the after-change median latency, stable identities, and zero Markdown updates for Library navigation.

- [ ] **Step 2: Run static analysis and the broader regression suite**

```powershell
& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m ruff check tldw_chatbook/Widgets/Library/library_media_content.py tldw_chatbook/Widgets/Library/library_media_viewer.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/Widgets/Media/media_viewer_panel.py Tests/network_guard.py Tests/test_network_guard.py Tests/Library/test_library_media_content.py Tests/UI/test_library_shell.py Tests/UI/test_media_viewer_content_search_debounce.py
& 'C:\Users\GDesktop-1\Working\Github\tldw_tui\.venv\Scripts\python.exe' -m pytest -q
git diff --check origin/dev...HEAD
```

Expected: Ruff and the full suite pass, and the branch diff has no whitespace errors. If the repository-wide suite has a pre-existing unrelated failure, rerun the exact failing test against the unchanged `origin/dev` worktree and record both outputs rather than attributing it to this task.

- [ ] **Step 3: Perform a rendered keyboard UAT**

Run the app from this worktree with the parent worktree virtual environment. In Library, open a long Markdown media item and verify: Markdown opens Rendered; typing does not apply until Enter; Enter retains search focus; Next/Previous retain their own focus, wrap, update the status, and scroll; Raw shows the selected highlight; repeated Raw/Rendered toggles retain content and do not flash/remount. In the legacy Media screen, type a burst into content search and verify results appear once after the silent 250 ms pause, then clear and switch records before the pause to verify no stale repaint.

Record terminal size, media fixture size, actions, visible results, and any limitation in the task notes. Use the rendered frame or terminal capture as the visual oracle, not widget properties alone.

- [ ] **Step 4: Complete the Backlog task source of truth**

Directly edit the five-digit task file, as required by `backlog/docs/lessons-backlog-hygiene.md`, because Backlog CLI 1.44.0 corrupts five-digit IDs. Check all four acceptance criteria, add measured before/after latency and parse/identity counts, list exact verification commands, and add:

```markdown
## Implementation Notes

- Extracted scoped Library search controls and a lazy persistent content body so match navigation updates status/highlighting/scroll state without remounting the screen or reparsing Markdown.
- Preserved Enter-to-search, Rendered-by-default Markdown behavior, Raw highlighting, wraparound navigation, and focus continuity.
- Debounced legacy content search at 250 ms with monotonic lifecycle invalidation and one Markdown update per applied query.
- Repaired Windows async-test bootstrap under ADR-058 with a nested current-thread socketpair exemption; literal pytest commands pass while ordinary and concurrent-thread egress remain denied and recorded.
- ADR required: yes; followed `backlog/decisions/058-thread-scoped-test-socketpair-exemption.md` for the review-expanded test security boundary.
```

Add a lessons entry only if implementation or UAT exposes a reproducible trap that generalizes beyond this task; do not add one merely to fill the checklist.

- [ ] **Step 5: Commit closeout documentation**

```powershell
git add backlog/tasks/task-15458*.md backlog/docs/lessons-testing-evidence.md
git commit -m "docs(library): close task 15458"
```

If no lessons file changed, stage only the task file. Confirm `git status --short` is empty after the commit.
