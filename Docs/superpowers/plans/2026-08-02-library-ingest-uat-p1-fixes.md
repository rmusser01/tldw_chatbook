# Library Ingest UAT P1 Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the five P1 defects found in the 2026-08-02 Library file-ingestion UAT (critique snapshot `.impeccable/critique/2026-08-02T21-04-04Z__chatbook-widgets-library-library-ingest-canvas-py.md`), stability first.

**Architecture:** All five fixes are small, contained changes to the existing render-from-state ingest flow: two screen-level stability fixes in `library_screen.py` (pre-flight generation guard, focus/scroll preservation across registry-driven recomposes), one writer-thread honesty fix in `app.py` (duplicate resolution + labeling), one widget+CSS fix (option-panel labels + Checkbox escape), one CSS-only fix (visible focus). No schema changes, no new modules.

**Tech Stack:** Python 3.12 (repo venv), Textual 8.2.7, pytest + textual `run_test` harness in `Tests/UI/test_library_shell.py`, TCSS sources compiled by `tldw_chatbook/css/build_css.py`.

## Global Constraints

- Work in the worktree `/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/78459cf3-ec69-4edb-b083-100f61156178/scratchpad/uat-dev` on branch `fix/library-ingest-uat-p1s` (based on origin/dev @ `1e828152e`). All git commands `git -C <worktree>`; all pytest via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest` with cwd = the worktree.
- NEVER `git stash`; NEVER `git add -A`; stage explicit paths only.
- Never hand-edit `tldw_chatbook/css/tldw_cli_modular.tcss`; edit the source `.tcss`, run `python tldw_chatbook/css/build_css.py` (venv python), commit source + bundle together.
- Backlog task IDs 2010–2016 are reserved (max across origin/dev + all worktrees verified = 1995 on 2026-08-02). Do not renumber; do not let the backlog CLI assign IDs.
- Never flip local/server runtime mode anywhere (runtime-policy file is shared with the real user profile).
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Do not implement anything beyond these tasks' ACs (the P2/P3 findings are FILED in Task 0, not fixed).

---

### Task 0: File backlog tasks 2010–2016

**Files:**
- Create: `backlog/tasks/task-2010 - Library-ingest-job-tick-recompose-steals-focus-and-scroll.md`
- Create: `backlog/tasks/task-2011 - Library-ingest-stale-preflight-result-resurrects-cleared-summary.md`
- Create: `backlog/tasks/task-2012 - Library-ingest-options-panel-renders-unlabeled-checkbox-husks.md`
- Create: `backlog/tasks/task-2013 - Library-ingest-silently-swallows-duplicate-files.md`
- Create: `backlog/tasks/task-2014 - Library-ingest-canvas-keyboard-focus-is-color-only.md`
- Create: `backlog/tasks/task-2015 - Library-ingest-P2-batch-feedback-copy-and-consistency.md`
- Create: `backlog/tasks/task-2016 - Library-ingest-P3-polish-batch.md`

**Interfaces:**
- Produces: task files whose IDs (2010–2014) later tasks flip to In Progress/Done.

- [ ] **Step 1: Verify IDs are still free** (other sessions file tasks concurrently)

```bash
{ git -C /Users/macbook-dev/Documents/GitHub/tldw_chatbook ls-tree -r origin/dev --name-only backlog/tasks/; \
  for wt in $(git -C /Users/macbook-dev/Documents/GitHub/tldw_chatbook worktree list --porcelain | awk '/^worktree /{print $2}'); do ls "$wt/backlog/tasks" 2>/dev/null; done; } \
  | sed -nE 's/.*task-([0-9]+)[. -].*/\1/p' | sort -n | tail -1
```
Expected: ≤ 2009. If ≥ 2010, shift all seven IDs up to the next free block of 7 with ≥10 headroom and use the shifted numbers everywhere below.

- [ ] **Step 2: Write the seven task files.** Frontmatter format (copy exactly, adjusting id/title/labels):

```markdown
---
id: TASK-2010
title: >-
  Library ingest job-tick recompose steals focus and scroll
status: To Do
assignee: []
created_date: '2026-08-02 21:30'
labels:
  - library
  - ingest
  - ux
  - uat
priority: high
dependencies: []
---

## Description (the why)

Every ingest job transition (`_handle_library_ingest_registry_changed`,
`library_screen.py:5537`) runs a full-screen `refresh(recompose=True)` while
the ingest canvas is selected. A recompose remounts every widget, silently
dropping focus from the Input the user is typing into; later keystrokes hit
the app's global digit bindings and navigate away mid-word. Queue scroll
position also resets on every tick. Found in the 2026-08-02 ingest UAT
(critique snapshot 2026-08-02T21-04-04Z).

## Acceptance Criteria (the what)

- [ ] Typing into the ingest path (or title/author/keywords) field while a
      queued job transitions keeps focus, text, and cursor position.
- [ ] Queue scroll position survives a job transition.
- [ ] A focused widget that no longer exists after the recompose (e.g. a
      row-action button of a finished job) degrades gracefully: no
      exception, focus falls back to the screen default.
```

Task bodies for the rest (same frontmatter shape; 2011–2014 `priority: high`, 2015 `medium`, 2016 `low`; same labels):

- **2011**: Why: `_do_submit_ingest` clears `form.preflight` but the in-flight `@work(thread=True)` pre-flight applies unconditionally via `call_from_thread` (`library_screen.py:12396`), resurrecting the cleared summary ("Enter a file path to start." shown together with "1 plain text file · 277 B"). AC: a pre-flight result triggered before a submit/clear never repopulates the summary after it; a result for the current path still applies.
- **2012**: Why: the unscoped `Checkbox { width:100%; height:2 }` rule in `css/features/_conversations.tcss:329` clips "Analyze after ingest"/"Chunk content" to border-only husks; value Inputs use placeholder-as-label so populated fields show bare "1000"/"100"/"auto"; an expanded panel trails ~15 blank rows (unstyled `.type-group-contents` container). AC: both checkboxes render their labels and states; every value input has a visible text label; no trailing blank region inside an expanded panel.
- **2013**: Why: a byte-identical file at a different path takes `add_media_with_keywords`'s duplicate-skip path (returns `media_id=None`); the writer's URL fallback misses (different URL), so the job is marked done with no media_id — a "✓ done" row that created nothing, said nothing, and has no "Open in Library" (app.py `_run_library_ingest_queue`). AC: a duplicate ingest resolves the existing media item's id (hash fallback), its row action opens that item, and the row's progress line states the file was already in the Library.
- **2014**: Why: 8 of 10 Tab stops through the ingest canvas produce no monochrome-visible change — focus is a background-color change only, violating DESIGN.md's `outline: heavy $accent` focus contract; compounds 2010 (users can't see focus was stolen). AC: every focusable widget on the ingest canvas shows a visible, non-color-only focus indicator (verified in a `capture-pane -p` monochrome dump).
- **2015** (P2 batch — one AC checkbox per item, all from the 2026-08-02 critique): path validation only on blur; no success toast + result row below the fold; dead Retry on empty-file failure (mark empty-source failures permanent); folder done rows lack "Open in Library"; "recorded as a failures" grammar (`library_ingest_canvas.py` ~line 312); triple-wrapped PDF error copy; "Clear finished" wipes "Recent ingests" without confirm; 110-col mid-word clip of the options summary header; stray "Choose a file…" + "0 files" under a path error after failed submit; Start enabled for guaranteed failures; misleading elapsed ("0s"); no progress indication for large files.
- **2016** (P3 polish batch): "done" stated twice + absolute path in done rows; Expand/Collapse-all with one panel; scope line claims "Applies to all …" with zero such files; intro lines persist after path typed; picker opens at $HOME with no type hint; rail counts flash "(0)"; details-modal placement (needs repro); stderr flood on first submit (needs repro — route warnings/loguru off the TTY); `#library-search-input` lacks an `Input.Changed` handler so unsubmitted text resurrects from persisted state; `[first_run] setup_completed` written at app open before the wizard is completed/skipped.

- [ ] **Step 3: Commit**

```bash
git -C <worktree> add backlog/tasks/task-2010* backlog/tasks/task-2011* backlog/tasks/task-2012* backlog/tasks/task-2013* backlog/tasks/task-2014* backlog/tasks/task-2015* backlog/tasks/task-2016*
git -C <worktree> commit -m "chore(backlog): file Library ingest UAT findings as tasks 2010-2016"
```
(Append the Co-Authored-By footer to every commit in this plan.)

---

### Task 1: Pre-flight generation guard (task-2011)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (`_trigger_library_ingest_preflight` ~:12361, `_run_library_ingest_preflight` ~:12375, `_apply_library_ingest_preflight_result` ~:12398, and the two `form.preflight = None` sites at ~:12065 and ~:12568)
- Test: `Tests/UI/test_library_shell.py`

**Interfaces:**
- Produces: `LibraryScreen._invalidate_library_ingest_preflight() -> None` (bumps `self._library_ingest_preflight_generation: int`); `_apply_library_ingest_preflight_result(result, generation)` drops stale generations. Task 2 must not reorder these methods.

- [ ] **Step 1: Mark task-2010..2014 In Progress mentally; edit task-2011 frontmatter `status: In Progress`.**

- [ ] **Step 2: Write the failing test** (append to `Tests/UI/test_library_shell.py`, reusing `_build_test_app`, `LibraryHarness`, `_active_library_screen`, `_wait_for_library_shell`, `_wait_for_selector`, `LIBRARY_TEST_SIZE`, `LIBRARY_NAV_CONTEXT_INGEST`, `PreflightResult` — all already imported/defined in that file):

```python
@pytest.mark.asyncio
async def test_library_ingest_stale_preflight_result_is_dropped_after_clear():
    """(task-2011) A pre-flight worker started BEFORE a submit/clear must not
    repopulate the summary it cleared: `_do_submit_ingest` empties
    ``form.preflight`` on purpose, and worker cancellation is cooperative, so
    the guard is a generation stamp, not the cancel."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    screen = LibraryScreen(app)
    screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_INGEST: True})
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")

        stale_generation = screen._library_ingest_preflight_generation
        # The submit/clear path invalidates any in-flight pre-flight.
        screen._invalidate_library_ingest_preflight()
        assert screen._library_ingest_form.preflight is None

        late_result = PreflightResult(
            type_groups={"generic": ["/tmp/whatever.txt"]},
            warnings=[],
            errors=[],
            total_size=277,
            truncated=False,
            total_files=1,
        )
        # The worker thread delivers its result with the generation it was
        # started under -- one bump ago.
        screen._apply_library_ingest_preflight_result(late_result, stale_generation)
        assert screen._library_ingest_form.preflight is None, (
            "stale pre-flight result must be dropped, not applied"
        )

        # A result carrying the CURRENT generation still applies.
        screen._apply_library_ingest_preflight_result(
            late_result, screen._library_ingest_preflight_generation
        )
        assert screen._library_ingest_form.preflight is late_result
```

Note: check `PreflightResult`'s constructor in `tldw_chatbook/Library/ingest_types.py` before writing — match its actual field names/types exactly (the shape above mirrors `_run_library_ingest_preflight`'s error-path construction).

- [ ] **Step 3: Run test to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest "Tests/UI/test_library_shell.py::test_library_ingest_stale_preflight_result_is_dropped_after_clear" -x -q`
Expected: FAIL with `AttributeError: ... has no attribute '_library_ingest_preflight_generation'`

- [ ] **Step 4: Implement.** In `library_screen.py`:

(a) Where `_library_ingest_preflight_worker` is first assigned in `__init__` (grep `self._library_ingest_preflight_worker`), add beside it:
```python
        self._library_ingest_preflight_generation: int = 0
```

(b) Add the helper next to `_cancel_library_ingest_preflight`:
```python
    def _invalidate_library_ingest_preflight(self) -> None:
        """Drop the current pre-flight echo AND fence off in-flight workers.

        Worker cancellation is cooperative (`@work(thread=True)`), so a
        worker that has already finished analysing can still deliver its
        result after this runs. Bumping the generation makes
        ``_apply_library_ingest_preflight_result`` drop that late result
        instead of resurrecting the summary this method just cleared.
        """
        self._library_ingest_preflight_generation += 1
        self._cancel_library_ingest_preflight()
        self._library_ingest_form.preflight = None
        self._library_ingest_form.preflight_checking = False
```

(c) In `_trigger_library_ingest_preflight`, bump + capture the generation and pass it to the worker:
```python
        self._cancel_library_ingest_preflight()
        self._library_ingest_preflight_generation += 1
        generation = self._library_ingest_preflight_generation
        self._library_ingest_form.preflight_checking = True
        self.refresh(recompose=True)
        self._library_ingest_preflight_worker = self._run_library_ingest_preflight(
            path, generation
        )
```

(d) Thread the parameter through: `def _run_library_ingest_preflight(self, path: str, generation: int) -> None:` and at its tail `self.app.call_from_thread(self._apply_library_ingest_preflight_result, result, generation)`.

(e) Guard the apply:
```python
    def _apply_library_ingest_preflight_result(
        self,
        result: PreflightResult,
        generation: int,
    ) -> None:
        """Merge a pre-flight result into the form echo and refresh.

        Drops results from a superseded generation: a clear/submit or a
        newer trigger bumped the counter after this worker started, so its
        result describes a path the form is no longer showing (task-2011).
        """
        if generation != self._library_ingest_preflight_generation:
            return
        self._library_ingest_form.preflight = result
        self._library_ingest_form.preflight_checking = False
        self.refresh(recompose=True)
```

(f) Replace the two raw clear sites (~:12065 and ~:12568 — grep `form.preflight = None`; read each surrounding block first). At each site, replace the `form.preflight = None` line (and any adjacent duplicate `preflight_checking = False` / `_cancel_library_ingest_preflight()` calls that the helper now performs) with `self._invalidate_library_ingest_preflight()`. Keep any OTHER statements those blocks perform.

- [ ] **Step 5: Run the new test — expect PASS. Then the ingest-adjacent suite:**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_shell.py -q -k "ingest or preflight"`
Expected: all pass (any failure: fix before proceeding; compare against clean origin/dev only if a failure looks pre-existing).

- [ ] **Step 6: Tick task-2011 ACs, set `status: Done`, add Implementation Notes; commit**

```bash
git -C <worktree> add tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_shell.py "backlog/tasks/task-2011 - Library-ingest-stale-preflight-result-resurrects-cleared-summary.md"
git -C <worktree> commit -m "fix(library): drop stale ingest pre-flight results via generation stamp (task-2011)"
```

---

### Task 2: Preserve focus + scroll across job-tick recomposes (task-2010)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (`_handle_library_ingest_registry_changed` ~:5601 body; new helpers beside it)
- Test: `Tests/UI/test_library_shell.py`

**Interfaces:**
- Consumes: nothing from Task 1 beyond method positions.
- Produces: `LibraryScreen._refresh_library_ingest_canvas_preserving_context() -> None` and `LibraryScreen._restore_library_ingest_canvas_context(focused_id: str | None, cursor: int | None, scroll_y: float | None) -> None`.

- [ ] **Step 1: Set task-2010 `status: In Progress`.**

- [ ] **Step 2: Write the failing tests** (append to `Tests/UI/test_library_shell.py`; `Input` is already imported there):

```python
@pytest.mark.asyncio
async def test_library_ingest_job_tick_recompose_preserves_typing_focus():
    """(task-2010) A registry notification recomposes the canvas; the path
    Input the user is typing into must keep focus, text, and cursor. Without
    the restore, focus silently falls to the screen and the next digit
    keystroke navigates the whole app."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    screen = LibraryScreen(app)
    screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_INGEST: True})
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")

        path_input = screen.query_one("#library-ingest-path", Input)
        path_input.focus()
        await pilot.pause()
        await pilot.press("slash", "t", "m", "p")
        assert screen._library_ingest_form.path == "/tmp"
        cursor_before = path_input.cursor_position

        # A background job transition fires the registry listener.
        screen._handle_library_ingest_registry_changed()
        await pilot.pause()
        await _wait_for_selector(screen, pilot, "#library-ingest-path")
        await pilot.pause()

        remounted = screen.query_one("#library-ingest-path", Input)
        focused = screen.app.focused
        assert focused is remounted, (
            f"focus fell to {focused!r} after the job-tick recompose"
        )
        assert remounted.value == "/tmp"
        assert remounted.cursor_position == cursor_before


@pytest.mark.asyncio
async def test_library_ingest_restore_context_survives_vanished_widget():
    """(task-2010) Restoring focus to a widget id that no longer exists after
    the recompose (a finished job's row-action button) must not raise."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    screen = LibraryScreen(app)
    screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_INGEST: True})
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")

        screen._restore_library_ingest_canvas_context(
            "library-ingest-retry-ingest-job-999", 3, 12.0
        )
        await pilot.pause()  # no exception is the assertion
```

Note on `pilot.press("slash", ...)`: verify the key name Textual 8.2.7 uses for `/` in existing tests (grep `pilot.press` in the file); if plain characters are accepted, `pilot.press("/", "t", "m", "p")`. If the global digit/hotkey bindings intercept printable keys in this harness, instead set the value the way the real handler does: `path_input.value = "/tmp"; path_input.cursor_position = 4; await pilot.pause()` — the assertion targets are unchanged.

- [ ] **Step 3: Run both — expect FAIL** (first: focus falls to screen / helper missing; second: `AttributeError` for the missing method).

- [ ] **Step 4: Implement.** In `library_screen.py`, replace the recompose line inside `_handle_library_ingest_registry_changed`:

```python
        if self._library_selected_row_id == LIBRARY_ROW_INGEST_MEDIA:
            self._refresh_library_ingest_canvas_preserving_context()
```

Add beside it (module already imports `LibraryIngestCanvas`, `Input`, `NoMatches`, `QueryError` — verify with grep and extend imports if not):

```python
    def _refresh_library_ingest_canvas_preserving_context(self) -> None:
        """Recompose for a job-tick WITHOUT losing what the user was doing.

        ``refresh(recompose=True)`` remounts every widget: focus silently
        falls to the screen (so the user's next keystrokes hit the global
        digit bindings and navigate the app) and the canvas scroll snaps to
        the top on every queue transition. Capture focus id + cursor +
        scroll before scheduling the recompose, restore after -- the same
        remount-focus-loss family as ``_focus_library_search_input``
        (task-2010).
        """
        focused = self.app.focused
        focused_id = getattr(focused, "id", None) if focused is not None else None
        cursor = (
            focused.cursor_position
            if isinstance(focused, Input) and focused_id
            else None
        )
        scroll_y: float | None = None
        try:
            canvas = self.query_one(LibraryIngestCanvas)
            scroll_y = canvas.scroll_offset.y
        except (NoMatches, QueryError):
            pass
        self.refresh(recompose=True)
        if focused_id is not None or scroll_y:
            self.call_after_refresh(
                self._restore_library_ingest_canvas_context,
                focused_id,
                cursor,
                scroll_y,
            )

    def _restore_library_ingest_canvas_context(
        self,
        focused_id: str | None,
        cursor: int | None,
        scroll_y: float | None,
    ) -> None:
        """Re-apply focus/cursor/scroll captured before a job-tick recompose.

        Scroll first, then focus with ``scroll_visible=False`` so restoring
        focus does not itself yank the scroll position. A vanished widget id
        (a finished job's row-action button) degrades silently (task-2010).
        """
        if scroll_y:
            try:
                canvas = self.query_one(LibraryIngestCanvas)
                canvas.scroll_to(y=scroll_y, animate=False, force=True)
            except (NoMatches, QueryError):
                pass
        if not focused_id:
            return
        try:
            widget = self.query_one(f"#{focused_id}")
        except (NoMatches, QueryError):
            return
        widget.focus(scroll_visible=False)
        if cursor is not None and isinstance(widget, Input):
            widget.cursor_position = min(cursor, len(widget.value))
```

If the first test still fails on focus with a single `call_after_refresh` (recompose lands one event-loop turn later than the callback), chain one more hop: make the scheduled callable `lambda: self.call_after_refresh(self._restore_library_ingest_canvas_context, ...)` — but only after observing the failure, and record which variant shipped in the task notes.

- [ ] **Step 5: Run both new tests — expect PASS. Then:** `... -m pytest Tests/UI/test_library_shell.py -q -k "ingest"` — all pass.

- [ ] **Step 6: Tick task-2010 ACs (the scroll AC is verified live in Task 6 — leave it unticked until then), set In Progress → Done in Task 6. Commit:**

```bash
git -C <worktree> add tldw_chatbook/UI/Screens/library_screen.py Tests/UI/test_library_shell.py "backlog/tasks/task-2010 - Library-ingest-job-tick-recompose-steals-focus-and-scroll.md"
git -C <worktree> commit -m "fix(library): preserve focus and scroll across ingest job-tick recomposes (task-2010)"
```

---

### Task 3: Duplicate ingest resolution + honest labeling (task-2013)

**Files:**
- Modify: `tldw_chatbook/app.py` (`_run_library_ingest_queue`, the `persist_parsed_media` block at ~:3238–3260)
- Test: `Tests/Library/test_library_ingest_runner.py`

**Interfaces:**
- Consumes: `MediaDatabase.get_media_by_hash(content_hash)` (`DB/Client_Media_DB_v2.py:6096`), `MediaDatabase.get_media_by_url(url)`; `add_media_with_keywords` duplicate-skip contract: returns `(None, None, "Media '<title>' already exists. Overwrite not enabled.")`.
- Produces: done-jobs for duplicates carry the existing item's `media_id` and `progress["message"]` starting with `"Already in Library"`.

- [ ] **Step 1: Set task-2013 In Progress. Read the existing test harness** (`Tests/Library/test_library_ingest_runner.py`): find how it builds the mixin host, a real in-memory `MediaDatabase`, and drives `_run_library_ingest_queue` (it exports `_FakeIngestParsePool`). Reuse its fixtures/builders verbatim for the new test.

- [ ] **Step 2: Write the failing test** (adapt setup to that file's existing builder; the assertions are the contract):

```python
def test_duplicate_content_at_different_path_resolves_existing_media_id(tmp_path):
    """(task-2013) Byte-identical content at a DIFFERENT path takes the DB's
    duplicate-skip path (media_id=None) and the URL fallback misses (URLs
    differ). The job must still resolve the EXISTING item's id -- so the row
    keeps "Open in Library" -- and must say it was a duplicate instead of
    impersonating a fresh ingest."""
    host = _build_runner_host(tmp_path)  # <- this file's existing app/mixin builder
    first = tmp_path / "report.txt"
    second = tmp_path / "copy_of_report.txt"
    first.write_text("identical body " * 20)
    second.write_text("identical body " * 20)

    _ingest_and_drain(host, first)   # <- existing submit+drain helper pattern
    _ingest_and_drain(host, second)

    jobs = {j.source_path: j for j in host.library_ingest_jobs.jobs()}
    first_job = jobs[str(first)]
    second_job = jobs[str(second)]
    assert first_job.media_id is not None
    assert second_job.media_id == first_job.media_id, (
        "duplicate must resolve to the existing media item"
    )
    assert second_job.progress["message"].startswith("Already in Library"), (
        f"duplicate impersonated a fresh ingest: {second_job.progress}"
    )
    assert first_job.progress["message"].startswith("Ingested ")
```

Replace `_build_runner_host` / `_ingest_and_drain` / `.jobs()` with this file's real builder, drain call, and registry-iteration API (read the file first; do NOT invent a fake media_db — the test needs the real `MediaDatabase` duplicate-skip behavior, in-memory `:memory:` or tmp_path file, matching how existing tests there construct it).

- [ ] **Step 3: Run it — expect FAIL** on the `media_id ==` assertion (today the duplicate job's media_id is None) or on the message assertion.

- [ ] **Step 4: Implement.** In `app.py` `_run_library_ingest_queue`, replace the block from `media_id, _media_uuid, _message = persist_parsed_media(` through the `progress = {...}` line with:

```python
                    media_id, _media_uuid, _message = persist_parsed_media(
                        payload, self.media_db
                    )
                    # ``add_media_with_keywords`` returns ``media_id=None`` on
                    # exactly one success path: the duplicate skip ("already
                    # exists. Overwrite not enabled."). Same-path re-ingests
                    # resolve by canonical URL; a byte-identical file at a
                    # DIFFERENT path has a different URL, so fall back to the
                    # content hash -- otherwise the row is a done-without-
                    # media_id husk with no "Open in Library" and nothing
                    # telling the user the file was already there (task-2013).
                    was_duplicate = media_id is None
                    if media_id is None and self.media_db is not None:
                        existing = self.media_db.get_media_by_url(payload["url"])
                        if existing is None and payload.get("content_hash"):
                            try:
                                existing = self.media_db.get_media_by_hash(
                                    payload["content_hash"]
                                )
                            except Exception:
                                existing = None
                        if existing is not None:
                            media_id = existing.get("id")
                    content_hash = payload.get("content_hash")
                    if was_duplicate:
                        progress = {
                            "message": (
                                "Already in Library — matched an existing item; "
                                "nothing new was imported."
                            )
                        }
                    else:
                        progress = {"message": f"Ingested {job.source_path}"}
```

- [ ] **Step 5: Run the new test — PASS. Then the whole runner file:** `... -m pytest Tests/Library/test_library_ingest_runner.py -q` — all pass.

- [ ] **Step 6: Tick task-2013 ACs (row-action AC finalized in Task 6 live check), commit:**

```bash
git -C <worktree> add tldw_chatbook/app.py Tests/Library/test_library_ingest_runner.py "backlog/tasks/task-2013 - Library-ingest-silently-swallows-duplicate-files.md"
git -C <worktree> commit -m "fix(library): resolve duplicate ingests to the existing media item and say so (task-2013)"
```

---

### Task 4: Options panel — labeled checkboxes and value fields (task-2012)

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_ingest_canvas.py` (`_compose_type_group` value-input branch ~:144–153)
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` (append near the `.library-ingest-row` block at ~:1737)
- Modify (generated): `tldw_chatbook/css/tldw_cli_modular.tcss` via `build_css.py`
- Test: `Tests/UI/test_library_shell.py`

**Interfaces:**
- Produces: a `Static` with class `type-group-field-label` immediately before every non-checkbox, non-select option Input; TCSS escapes `LibraryIngestCanvas .type-group-contents Checkbox { width: auto; height: auto; }`.

- [ ] **Step 1: Set task-2012 In Progress.**

- [ ] **Step 2: Write the failing test** (structure only — the harness does not load the bundle CSS, so the CSS half is verified live in Task 6):

```python
@pytest.mark.asyncio
async def test_library_ingest_option_value_inputs_carry_visible_labels():
    """(task-2012) Populated Inputs never show their placeholder, so
    placeholder-as-label leaves bare "1000"/"100"/"auto" values. Every value
    field must be preceded by a visible Static label."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    screen = LibraryScreen(app)
    screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_INGEST: True})
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-ingest-path")

        # Stage a generic-group pre-flight with the panel expanded, exactly
        # as a real .txt pre-flight leaves the form.
        screen._library_ingest_form.preflight = PreflightResult(
            type_groups={"generic": ["/tmp/report.txt"]},
            warnings=[],
            errors=[],
            total_size=316,
            truncated=False,
            total_files=1,
        )
        # Find the real expanded-groups attribute with:
        #   grep -n "expanded" tldw_chatbook/Library/library_ingest_state.py
        # and set it the way the screen's expand-all handler does.
        screen._library_ingest_expanded_type_groups = {"generic"}
        screen.refresh(recompose=True)
        await pilot.pause()
        await _wait_for_selector(screen, pilot, "#opt-generic-chunk_size")

        labels = [
            str(w.renderable)
            for w in screen.query(".type-group-field-label").results(Static)
        ]
        caps = get_capabilities("generic")
        expected = [
            f.label for f in caps.fields if f.type not in ("checkbox", "select")
        ]
        for label in expected:
            assert label in labels, f"value field {label!r} has no visible label"
```

Adapt the two marked seams (`expanded` attribute name; `PreflightResult` field names) to what the source actually declares — grep before writing, per the file's own patterns. `get_capabilities` and `Static` are already imported in the test module.

- [ ] **Step 3: Run — expect FAIL** (`.type-group-field-label` query returns nothing).

- [ ] **Step 4: Implement widget half.** In `_compose_type_group`'s final `else:` branch (the value-Input case), insert a label Static before the Input:

```python
            else:
                self._reported_option_values[(group, field.name)] = str(value)
                children.append(
                    Static(
                        field.label,
                        classes="type-group-field-label",
                        markup=False,
                    )
                )
                children.append(
                    Input(
                        value=str(value),
                        placeholder=field.label,
                        id=widget_id,
                        disabled=disabled,
                    )
                )
```

- [ ] **Step 5: Implement CSS half.** Append to `tldw_chatbook/css/components/_agentic_terminal.tcss` beside the other `library-ingest` rules:

```tcss
/* task-2012: css/features/_conversations.tcss ships an UNSCOPED
   `Checkbox { width: 100%; height: 2; }` that clips this panel's checkbox
   content row to a border-only husk (same escape family as the per-ID
   escapes other screens carry against that rule). The container and label
   rules kill the panel's trailing phantom height and style the new value
   labels. */
LibraryIngestCanvas .type-group-contents {
    height: auto;
}
LibraryIngestCanvas .type-group-contents Checkbox {
    width: auto;
    height: auto;
}
LibraryIngestCanvas .type-group-field-label {
    color: $text-muted;
    margin-top: 1;
}
```

- [ ] **Step 6: Regenerate the bundle** — `cd <worktree> && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py`; verify `git -C <worktree> diff --stat` shows only the two tcss files + widget + test.

- [ ] **Step 7: Run the new test — PASS. Then** `... -m pytest Tests/UI/test_library_shell.py -q -k "ingest"` — all pass.

- [ ] **Step 8: Commit** (bundle + source together):

```bash
git -C <worktree> add tldw_chatbook/Widgets/Library/library_ingest_canvas.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_library_shell.py "backlog/tasks/task-2012 - Library-ingest-options-panel-renders-unlabeled-checkbox-husks.md"
git -C <worktree> commit -m "fix(library): label ingest option fields and unclip checkbox husks (task-2012)"
```

---

### Task 5: Visible keyboard focus on the ingest canvas (task-2014)

**Files:**
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify (generated): `tldw_chatbook/css/tldw_cli_modular.tcss`

**Interfaces:**
- Consumes: DESIGN.md focus contract (`outline: heavy $accent`; hover/focus must not change dimensions).

- [ ] **Step 1: Set task-2014 In Progress. Append to the same tcss source:**

```tcss
/* task-2014: focus on this canvas was a background-color change only --
   invisible in monochrome and to low-vision users, and it hides the
   job-tick focus-theft (task-2010). DESIGN.md's focus contract is
   `outline: heavy $accent`; outlines draw over the widget's own edge
   cells, so dimensions stay stable. */
LibraryIngestCanvas Input:focus {
    outline: heavy $accent;
}
LibraryIngestCanvas Button:focus {
    outline: heavy $accent;
    text-style: bold;
}
LibraryIngestCanvas Checkbox:focus {
    outline: heavy $accent;
}
LibraryIngestCanvas Select:focus {
    outline: heavy $accent;
}
LibraryIngestCanvas CollapsibleTitle:focus {
    text-style: bold reverse;
}
```

- [ ] **Step 2: Regenerate the bundle** (same command as Task 4 Step 6).

- [ ] **Step 3: Sanity-run the ingest UI suite** (CSS is inert in the harness; this catches parse errors surfacing through DEFAULT_CSS interplay): `... -m pytest Tests/UI/test_library_shell.py -q -k "ingest"` — all pass.

- [ ] **Step 4: Commit:**

```bash
git -C <worktree> add tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss "backlog/tasks/task-2014 - Library-ingest-canvas-keyboard-focus-is-color-only.md"
git -C <worktree> commit -m "fix(library): give ingest canvas focus a visible outline (task-2014)"
```

(The task's AC is monochrome-visible focus at every stop — verified and ticked in Task 6; if `outline: heavy` renders badly on 1-row compact buttons, substitute `text-style: bold reverse` for the Button rule there and note the substitution.)

---

### Task 6: Live verification, full suites, PR

**Files:**
- Modify: the five task files (final AC ticks, `status: Done`, Implementation Notes)

- [ ] **Step 1: Full test suites** (worktree cwd, venv python):

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library Tests/UI/test_library_shell.py -q
```
Expected: 0 failures. If any fail, run the SAME command in a clean origin/dev worktree and diff the failure SETS (never compare counts across different invocations); only pre-existing, identical failures may be waived, and they must be named in the PR body.

- [ ] **Step 2: Live verification** (isolated profile, unique socket — sockets are shared machine-wide, so include the session suffix):

```bash
SOCK=ingfix78459
SCRATCH=/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/78459cf3-ec69-4edb-b083-100f61156178/scratchpad
printf '[general]\nusers_name = "verify_ingfix"\n' > "$SCRATCH/profile_fix.toml"
tmux -L $SOCK new-session -d -x 235 -y 52 -c "$SCRATCH/uat-dev" "TLDW_CONFIG_PATH='$SCRATCH/profile_fix.toml' /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m tldw_chatbook.app"
sleep 16
```
Before trusting a single capture, verify the pane's app is YOURS: `ps -o pid,command -p $(tmux -L $SOCK list-panes -F '#{pane_pid}')` must show a python whose environment (`ps eww`) contains `TLDW_CONFIG_PATH=...profile_fix.toml` and cwd the uat-dev worktree. Re-verify after any unexpected screen change. Then drive (skip wizard via the "Skip — explore on my own" click; click `3 Library` in row 2; open "Ingest content…") and check, capturing evidence for each:
1. task-2010: focus the path field, type half a path, submit a second file from a shell-prepared fixture so a job transition fires mid-typing… simpler deterministic variant: start a folder ingest (reuse `$SCRATCH/fixtures/mixed_folder`), immediately click the path input and hold a key; after the queue settles, capture — typed text is in the path field, not leaked to another screen; scroll the queue down, wait for a transition, capture — scroll held.
2. task-2011: type a path, press Enter immediately; capture — no "Enter a file path to start." rendered together with a type-breakdown line.
3. task-2012: expand the generic panel; capture — both checkboxes show labels + state, value inputs have labels above them, no trailing blank region.
4. task-2013: ingest `$SCRATCH/fixtures/report.txt`, then a copy of it under another name; capture — second row says "Already in Library…" and has "Open in Library"; click it — the existing item opens.
5. task-2014: Tab through the canvas 10 stops; capture plain (`capture-pane -p`, no `-e`) at each — every stop shows a visible change.
Quit (`C-q`), `tmux -L $SOCK kill-server`, delete `~/.local/share/tldw_cli/verify_ingfix`.

- [ ] **Step 3: Tick every remaining AC across tasks 2010–2014, set all five `status: Done`, write Implementation Notes** (approach, files, deviations — including which focus-restore/outline variant shipped). Commit: `chore(backlog): close tasks 2010-2014 with implementation notes`.

- [ ] **Step 4: Push + PR to dev:**

```bash
git -C <worktree> push -u origin fix/library-ingest-uat-p1s
gh pr create --repo <origin> --base dev --head fix/library-ingest-uat-p1s --title "fix(library): ingest UAT P1 batch — stability, honesty, visibility (tasks 2010-2014)" --body "<summary of the five fixes, test evidence, live-verification evidence, waived pre-existing failures if any>

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```
Do NOT merge; CI is unreliable here by convention (verify locally). Report the PR URL.

---

## Self-Review

- Spec coverage: five P1s ↔ Tasks 1–5; task filing (user's "file backlog tasks" scope, incl. P2/P3 batches) ↔ Task 0; live verification + PR ↔ Task 6. The P2/P3 findings are deliberately NOT implemented (AC discipline).
- Known seams the implementer must resolve by reading (flagged inline, all with exact grep targets): `PreflightResult` constructor fields, the expanded-groups attribute name, `pilot.press` key names, the runner-test builder helpers. These are adapt-to-source points, not placeholders — the assertions and code under test are fully specified.
- Type consistency: `_invalidate_library_ingest_preflight` (Tasks 1/2 both leave `_apply_library_ingest_preflight_result(result, generation)` with the same signature); `_restore_library_ingest_canvas_context(focused_id, cursor, scroll_y)` used identically in Task 2's code and test; `type-group-field-label` class name identical in Task 4's widget code, CSS, and test.
