# Library critique-8 fix wave Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 25 findings of Library critique #8 (tasks 32050–32074) in seven independent branches that each ship as their own PR against `dev`.

**Architecture:** Each task below is one branch/worktree and one PR. Work stays inside the Library surface (`tldw_chatbook/UI/Screens/library_screen.py`, `tldw_chatbook/Widgets/Library/*.py`, `tldw_chatbook/Library/*.py`, `tldw_chatbook/UI/Library_Modules/*.py`, the split Library TCSS, `Docs/User_Guide/library*`). Behaviour changes are test-first against the existing Library UI harnesses; every task ends with a live tmux check on an isolated scratch profile.

**Tech Stack:** Python 3.12, Textual 8.x, pytest, Backlog.md CLI, tmux.

**Spec:** `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (the critique snapshot; the register rows are the binding requirements) and the task files `backlog/tasks/task-32050 … task-32074` (acceptance criteria).

## Global Constraints

- Work only inside your assigned worktree (`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/crit8-<group>`); every shell command starts with `cd <worktree> &&` because the shell cwd resets between calls. Never touch the main checkout or another group's worktree.
- Python: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` (3.12). Run pytest FROM the worktree cwd (`cd <worktree> && …/.venv/bin/python -m pytest <file or node id> -q -p no:cacheprovider`). Never run the whole suite (over an hour) and never use `-k` filtering as verification; run whole files or explicit node ids. `Tests/UI/test_library_shell.py` has ~226 pre-existing failures on dev: if you use it, compare failing NAME sets before and after your change (run it once on a clean `git stash` state or against `origin/dev`), and report only new names.
- Other sessions run pytest on this Mac; POSIX semaphores are exhausted, so `multiprocessing.Pool` fails with `[Errno 28]` and every local media Import fails in the live app. Do not try to fix or work around the host; if a test needs a process pool, mark it and move on.
- TDD: write the failing test, run it and show it failing, implement, run it passing. New tests go in the most specific existing `Tests/UI/test_library_*.py` file for the area, or a new `Tests/UI/test_library_crit8_<area>.py`; never into the 19k-line `test_library_shell.py`.
- CSS: edit the component source under `tldw_chatbook/css/components/` (Library rules live in `_agentic_terminal.tcss`, split into `screen_agentic_library.tcss` by the build), then run `cd <worktree> && …/.venv/bin/python -m tldw_chatbook.css.build_css` and commit the regenerated bundle files alongside. Widget `DEFAULT_CSS`/`BUNDLED_CSS` must parse standalone and never use ancestor-scoped bare-type subject rules (`Foo > Vertical`).
- Git: stage explicit paths only (never `git add -A`); commit after each green step; do NOT push and do NOT open or merge PRs, the controller does that after review. Commit trailer: `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Backlog hygiene from the worktree root: `backlog task edit <id> -s "In Progress" --plan "<steps>"` before the first code change; at the end tick every AC (`- [x]`) you satisfied, add `--notes "<Implementation Notes>"`, and set `-s Done`. Leave an AC unticked with a note if you could not satisfy it.
- Docs: every user-visible change updates the matching `Docs/User_Guide/library*.md` page and appends a `*Verified against fix/library-crit8-<group> — 2026-09-08 (task-NNNNN: …)*` stamp in the page's existing stamp style.
- Live verification (required before reporting): `tmux -L crit8-<group> new-session -d -x 235 -y 52 "cd <worktree> && TLDW_CONFIG_PATH=<profile>/config.toml PYTHONPATH=<worktree> …/.venv/bin/python -m tldw_chatbook.app"`, `sleep 15`, drive with `send-keys`, observe with `capture-pane -p` (`-e` for colour). Profiles: `<SCRATCH>/crit8/wave/<group>/power/config.toml` (seeded: 11 media, 6 conversations, 7 notes incl. a 35 KB one, 5 prompts, 2 skills, inbox/ and file_notes/ folders) and `…/fresh/config.toml` (empty; first launch shows the setup wizard: Esc, Tab, Enter skips it). `SCRATCH` = `/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/92b26568-46c4-4abe-9da5-9676d5282276/scratchpad`. Ctrl+digit keys cannot be sent through tmux: reach Library with `C-p`, type `Switch to Library`, `Down`, `Enter`. Mouse clicks: `send-keys -l $'\x1b[<0;COL;ROWM'` then `…ROWm` (1-based, column by code points, not bytes). Quit with `C-q` then `tmux -L crit8-<group> kill-server`. One instance per profile at a time.
- Copy rules: blocked or disabled states carry a text reason and a next step on the same line; never colour-only meaning; no raw errno, UUID or ISO timestamp reaches the user.
- Scope: implement the acceptance criteria of your tasks and nothing else. If an AC needs a product decision you cannot make, implement the rest, leave that AC unticked, and say so in the report.

---

### Task 1: Notes loader (group `notes-loader`, task 32050)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (`_refresh_library_note_detail` ~18534, `_begin_library_note_load` ~18640, `_run_library_service_call` ~12785, `handle_library_note_load_retry` ~30102)
- Modify: `tldw_chatbook/Library/library_notes_session.py` (`open_session` 339)
- Modify: `tldw_chatbook/UI/Library_Modules/note_session_port.py` (`load_note` 64)
- Modify: `tldw_chatbook/Widgets/Library/library_notes_canvas.py` (loading state ~600–625, `_authority_copy` ~410)
- Test: `Tests/UI/test_library_crit8_notes_loader.py` (new)
- Docs: `Docs/User_Guide/library/notes.md`

**Interfaces:**
- Consumes: `LibraryNotesSession.open_session(note_id) -> NoteLoadOutcome` (kinds LOADED, MISSING, FAILED, STALE); `_run_library_service_call(callable, *args, isolate_in_worker=True)` runs the service inside `asyncio.run` on an `asyncio.to_thread` worker.
- Produces: `LIBRARY_NOTE_LOAD_DEADLINE_SECONDS = 3.0` in `library_screen.py`; the loading canvas reaches `load_state == "failed"` with the message `Unable to load note — timed out after 3 s. Press Retry.` when the deadline passes.

- [ ] **Step 1: Reproduce live and capture a task dump.** Launch the seeded power profile (socket `crit8-notes-loader`), open Library, Notes, click any row (for example `Reading list`). Expected today: the canvas paints `Library notes · Library database · Loading note… · Next: Wait for loading to finish.` for ever. Before touching code, use `superpowers:systematic-debugging`: add temporary `logger.warning` lines at the entry of `_refresh_library_note_detail`, before and after `await self._library_note_session.open_session(note_id)`, at each `return` (the two `navigation_is_current()` guards, the STALE branch, MISSING, FAILED, LOADED), inside `DatabaseNoteSessionPort.load_note` before and after each `_run_service_call`, and inside `invoke_service_in_worker` in `_run_library_service_call`. Relaunch, reproduce, and read `<profile>/data/crit8_power/tldw_cli_app.log`. Record in your report which line was the last one reached. Known facts: the event loop is idle during the hang and no worker thread is executing the load (faulthandler dump taken 2026-09-08), so the coroutine is either cancelled (Textual `run_worker(exclusive=True, group="library_note_detail")` cancels any earlier worker in that group; check whether a second `run_worker` in the same group fires from a recompose or from `_library_notes_work_session_activation_pending`), or it returned SUPERSEDED because a generation/epoch/lifecycle token moved between `_begin_library_note_load` and the check (the notes canvas recomposes when the editor opens: rail and list auto-collapse), or `asyncio.run` inside the worker thread never returns because `get_note_detail` awaits something bound to the main loop.

- [ ] **Step 2: Write the failing regression test (real port, no fake service).** In `Tests/UI/test_library_crit8_notes_loader.py`, build the Library screen through the same harness `Tests/UI/test_library_shell.py` uses (copy its app fixture; do not import from that file if it drags the 226-red suite in), seed one note through a real `CharactersRAGDB` in a tmp dir and a real `NotesScopeService`/`DatabaseNoteSessionPort` (the production wiring in `tldw_chatbook/app.py` around `_build_notes_scope_service`), open the Notes list, activate the row, and assert within 3 s that the editor `TextArea` holds the note body:

```python
import asyncio
import pytest

@pytest.mark.asyncio
async def test_existing_note_opens_from_the_list_through_the_real_port(library_app_factory, tmp_path):
    app = library_app_factory(tmp_path, seed_notes=[("Reading list", "- Attention Is All You Need\n")])
    async with app.run_test(size=(235, 52)) as pilot:
        screen = await open_library_notes_list(pilot)
        await pilot.click("#library-notes-row-0")   # use the real row id the canvas mounts
        for _ in range(150):                        # 3 s at 20 ms
            await pilot.pause(0.02)
            editor = screen.query("#library-note-body")
            if editor and "Attention Is All You Need" in editor.first().text:
                break
        else:
            pytest.fail("note editor never rendered the stored body")
```

Adapt the fixture names and widget ids to the real ones you find (grep `library-note-body`, `library-notes-row`, `LibraryNoteWorkPane`); the test must go through `DatabaseNoteSessionPort` and the real DB, not `StaticLibraryNotesScopeService`.

- [ ] **Step 3: Run it and show it failing** (`… -m pytest Tests/UI/test_library_crit8_notes_loader.py -q -p no:cacheprovider`). Expected: FAIL with "note editor never rendered the stored body" (or the harness's own timeout). If it passes, the harness does not reproduce the live path: report NEEDS_CONTEXT with what differs between the harness wiring and `app.py`'s production wiring before going further.

- [ ] **Step 4: Fix the root cause** found in Step 1 (minimal change at the seam that drops the outcome or cancels the worker). Remove the temporary log lines.

- [ ] **Step 5: Add the deadline.** In `_refresh_library_note_detail` wrap the load: `outcome = await asyncio.wait_for(self._library_note_session.open_session(note_id), timeout=LIBRARY_NOTE_LOAD_DEADLINE_SECONDS)` with `except asyncio.TimeoutError:` setting `self._library_note_load_state = "failed"`, `self._library_note_load_message = "Unable to load note — timed out after 3 s. Press Retry."` and projecting through the existing failed path (`_project_library_note_entry_result`). Add a second test that monkeypatches the port's `load_note` to `await asyncio.sleep(10)` and asserts the failed copy and the Retry button appear within 4 s, and that clicking another row afterwards still opens that note.

- [ ] **Step 6: Run both tests green; run `Tests/UI/test_library_notes_session.py` (or whichever file holds `test_library_note_coordinator_*`) to confirm the coordinator tests still pass.**

- [ ] **Step 7: Live-verify** on the seeded profile: open the 35 KB `Very long note` and `Reading list`; both render; Escape returns to the list; re-open works. Save captures to `<SCRATCH>/crit8/wave/notes-loader/caps/`.

- [ ] **Step 8: Docs + backlog + commit.** Add the timeout copy to `Docs/User_Guide/library/notes.md` (Editor section) with the stamp; task 32050 plan, notes, ACs, Done. Commit: `fix(library-notes): open stored notes again and time out a stuck load (task-32050)`.

---

### Task 2: Keyboard completeness (group `keyboard`, tasks 32051, 32052, 32053)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (BINDINGS 891–1034 and their `check_action` gates; `_focus_library_list_entry`; the notes-create canvas entry ~30577; the Search/RAG evidence card handlers; `_MEDIA_WORKBENCH_FOCUS_TARGETS`-style focus tables)
- Modify: `tldw_chatbook/Widgets/Library/library_notes_canvas.py` (~1806, the `library-notes-create-blank` Button), `tldw_chatbook/Widgets/Library/library_search_rag_panel.py` (evidence cards), `tldw_chatbook/Widgets/Library/library_rail.py` (search Input)
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` (focus rings for grips, `#library-rag-run`, `#library-notes-create-blank`)
- Test: `Tests/UI/test_library_crit8_keyboard.py` (new)
- Docs: `Docs/User_Guide/library.md` (Keyboard & commands), `library/notes.md`, `library/search-and-rag.md`

**Interfaces:**
- Consumes: `LibraryScreen.check_action(action, parameters)` gates; the footer hint machinery (`_library_footer_hints` or equivalent, grep `esc focus rail`); `Input` focus in the rail (`#library-rail-search` or the id you find).
- Produces: `action_library_blur_text_field` bound to `escape` with a `check_action` that is true only while a Library `Input`/`TextArea` other than the note editor body has focus; it focuses the owning canvas's first focusable control. `action_library_focus_next_evidence` / `_prev_evidence` on `down`/`up` while an evidence card has focus. Cards are focusable (`can_focus = True`) with a `█▸` prefix class `.library-evidence-card--focused`.

- [ ] **Step 1 (32051): failing test** — focus the rail search Input, press `escape`, assert `screen.focused` is not the Input and that pressing `i` afterwards opens the Import canvas (assert the Import header widget exists). Then implement `action_library_blur_text_field` (binding `Binding("escape", "library_blur_text_field", "back to canvas", show=True)` placed BEFORE the existing escape bindings so its `check_action` wins while a text field has focus; the note editor body and title keep their existing behaviour). Run green. Same for the Search/RAG query box.

- [ ] **Step 2 (32052): failing test** — press `n` from the landing, assert `screen.focused.id == "library-notes-create-blank"`; press `enter`, assert a new note editor is open; press `tab` repeatedly (up to 30) and assert `screen.focused` is always inside the Library screen (`screen.focused.screen is screen` and not in the nav bar: grep the nav-bar widget class and assert it never gets focus). Implement: on notes-create canvas entry call `self.call_after_refresh(self._focus_library_list_entry, "library-notes-create-blank")` (reuse the existing helper the lists use); make Blank note and each template row `can_focus` and give them `up`/`down` handling; keep Tab inside the screen by giving the Library canvas container `focus` trapping the way the Console screen does (grep `focus_next` / `FocusGuard` in `UI/Screens/chat_screen.py` and reuse). Footer: "enter create note" only when the button really has focus.

- [ ] **Step 3 (32053): failing test** — run a search on the seeded profile fixture, press `tab` until an evidence card has focus (assert within 6 Tabs), press `enter` and assert the card's action label reads "Selected evidence", press `o` and assert the reader opened, press `u` and assert the staged-evidence strip shows 1 item. Implement focusable cards with visible `█▸` cursor (reuse the Media row focus classes), route `o`/`u`/`enter` through `check_action` only while a card has focus; when the query box has focus the footer reads `enter run search`, and Tab from the query box goes to the first evidence card when results exist (source toggles come after). Give `#library-rag-run` (or the real id) the same focus ring as other compact buttons; give pane grips a focus style (`.library-pane-grip:focus { text-style: bold reverse; }` in the component TCSS, then rebuild the bundle).

- [ ] **Step 4: Live-verify** all three on the seeded and the fresh profile (socket `crit8-keyboard`) with `capture-pane -e` proving the focus ring is a glyph/shape change; captures to `<SCRATCH>/crit8/wave/keyboard/caps/`.

- [ ] **Step 5: Docs + backlog + commits** (one commit per task id: `fix(library): Escape leaves a text box (task-32051)`, `fix(library-notes): focus Blank note on entry and keep Tab inside Library (task-32052)`, `fix(library-rag): keyboard-reachable evidence cards (task-32053)`).

---

### Task 3: Recovery copy (group `recovery-copy`, tasks 32054, 32056)

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_ingest_canvas.py` (queue row rendering, "Show details"), `tldw_chatbook/Library/library_ingest_state.py` and `tldw_chatbook/Library/library_ingest_jobs.py` (failure reason mapping; grep `Parse pool could not start`), `tldw_chatbook/UI/Screens/library_screen.py` (import completion toasts; grep `Import finished`)
- Modify: `tldw_chatbook/Widgets/Library/library_conversations_canvas.py` and `tldw_chatbook/UI/Library_Modules/library_conversation_reader.py` (actions toolbar placement, ineligible state), `tldw_chatbook/Workspaces/eligibility.py:70-90` (recovery copy consumers)
- Test: `Tests/UI/test_library_crit8_recovery_copy.py` (new); extend `Tests/UI/test_library_ingest_retry_*.py` only if a helper there fits
- Docs: `Docs/User_Guide/library/import-and-export.md`, `library/media-and-conversations.md`

**Interfaces:**
- Consumes: the ingest reason mapper that already turns `DatabaseError` into a human retry reason (task-31944; grep `retry reason` / `_human_failure_reason`); `WorkspaceEligibility(recovery_copy=…, reason_code="not_in_active_workspace"|"cross_workspace")`.
- Produces: `map_ingest_failure(exc_or_text) -> IngestFailureCopy(summary, detail, next_step, retryable: bool)` in `library_ingest_state.py`; an `OSError` with `errno == 28` from a pool start maps to summary `The import worker couldn't start on this machine (system resource limit)`, next step `Restart the app, then Retry`, detail = the raw text (shown only under Show details). Conversation reader header gains `#library-conversation-open-console` beside Read/Info; ineligible state renders `○ Open in Console · not in this workspace` plus `Link to workspace` (calls the existing workspace link/copy service; grep `add_item` in `Workspaces/registry_service.py`).

- [ ] **Step 1 (32054): failing tests** for `map_ingest_failure`: errno-28 pool error, an unsupported-file skip (must stay `skipped`, not `failed`, and must not be retryable), a plain `DatabaseError`. Implement the mapper; wire the queue row to show `summary` inline and `detail` under a `Show details` row action (a compact toggle Button per failed row); make the batch produce one `Import finished — N failed, M skipped` toast (grep the per-job notify and hoist it to batch completion); change `· attempt N` to `· retry N` OR change the docs line to `· attempt N` — pick the code change (docs already say retry). Run green.

- [ ] **Step 2 (32056): failing test** — open a conversation whose eligibility is `not_in_active_workspace` in the harness; assert the header contains `○ Open in Console · not in this workspace` and a `Link to workspace` button; press it; assert the conversation becomes eligible and `Open in Console` is enabled; assert the action is reachable with `c` while the reader has focus (reuse the Media `c` binding pattern `library_media_use_in_console`). Implement: move the actions toolbar into the reader header composition; replace the toast-only refusal with the inline state.

- [ ] **Step 3: Live-verify** (socket `crit8-recovery-copy`): Import an inbox file on the fresh profile (it will fail with the host errno 28 — that is the fixture for this task: the row must read the mapped summary with Show details revealing the raw text); open the 30-message thread on the power profile and exercise Link to workspace, then Open in Console. Captures to `<SCRATCH>/crit8/wave/recovery-copy/caps/`.

- [ ] **Step 4: Docs + backlog + commits** (`fix(library-import): plain-language failure reasons with Show details (task-32054)`, `fix(library-conversations): Open in Console in the header with an inline remedy (task-32056)`).

---

### Task 4: Structural waits (group `waits`, task 32055)

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py` (folder change flow; grep `Changing folder`), `tldw_chatbook/UI/Screens/library_screen.py` (the gates that veto exit during structural operations; grep `structural`, `_library_file_notes_root_change`, `action_quit`/`check_action("quit"`), `tldw_chatbook/Widgets/Library/library_skills_canvas.py` (import wait), `tldw_chatbook/Widgets/Library/library_export_canvas.py` (export wait)
- Create: `tldw_chatbook/Library/library_structural_wait.py` (one small state helper)
- Test: `Tests/UI/test_library_crit8_waits.py` (new), `Tests/Library/test_library_structural_wait.py` (new)
- Docs: `Docs/User_Guide/library/file-notes.md`, `library.md` (Keyboard section: Escape during a wait)

**Interfaces:**
- Produces: `StructuralWait` dataclass in `library_structural_wait.py` with `label: str`, `started_at: float`, `cancel: Callable[[], None] | None`, and `def status_line(self, now: float, patience_seconds: float = 3.0) -> str` returning `"<label>…"` before the patience window and `"<label>… · still working · Cancel"` after it. The screen keeps at most one `self._library_structural_wait: StructuralWait | None`; canvases render `status_line` in their existing status slot and a compact `Cancel` button (`#library-structural-wait-cancel`) once `cancel` is set.

- [ ] **Step 1: failing unit test** for `StructuralWait.status_line` (before/after 3 s). Implement.

- [ ] **Step 2: failing UI test** — monkeypatch the File Notes root-change service call to a never-resolving future; choose a folder; assert after 3.5 s the status shows `Changing folder… · still working · Cancel`; press `escape` and assert the canvas returned to Notes with the previous folder state intact; repeat and press Cancel, assert the wait clears and the status reads `Folder change cancelled · previous folder kept`; assert `screen.check_action("quit")` (or the app-level quit path) is not vetoed while a wait is active. Implement: run the root change under `asyncio.wait_for(..., timeout=30)` inside a cancellable task stored on the `StructuralWait`; the exit gates veto only the write (do not start a new change while one is running), never Escape, the back cue, the palette or quit.

- [ ] **Step 3: Apply the same wrapper** to the skill import wait and the export bundle write (same helper, same Cancel button id); one test each with a never-resolving service.

- [ ] **Step 4: Live-verify** (socket `crit8-waits`): link `<profile>/file_notes` on the power profile (it should complete normally); then temporarily point the folder at a path on a slow/nonexistent volume is not needed — instead confirm the happy path and that Ctrl+Q works during a skill import. Captures to `<SCRATCH>/crit8/wave/waits/caps/`.

- [ ] **Step 5: Docs + backlog + commit** (`fix(library): structural waits get a deadline and Cancel and never block exit (task-32055)`).

---

### Task 5: Collections row + guide sweep (group `docs`, tasks 32057, 32073)

**Files:**
- Modify: `Docs/User_Guide/library/collections.md` (rewrite to the live captures browser), `Docs/User_Guide/library.md`, `library/media-and-conversations.md`, `library/notes.md`, `library/search-and-rag.md`, `library/import-and-export.md`, `library/prompts.md`, `library/skills.md`, `library/file-notes.md`
- Modify: `tldw_chatbook/Widgets/Library/library_rail.py` and `tldw_chatbook/UI/Screens/library_screen.py` only for the two Collections-row behaviours (count before visit; no Create-section collapse on select; grep `library-rail-collections`, `collections` in the rail state, `Create` section disclosure persistence)
- Modify: `tldw_chatbook/Widgets/Library/library_collections_panel.py` (surface the `legacy_read_only` recovery copy when a write is refused)
- Test: `Tests/UI/test_library_crit8_collections_row.py` (new)

**Interfaces:**
- Consumes: `LegacyCollectionsReadOnlyError.reason == "legacy_read_only"`, `.recovery == "Use the legacy Collections inspector or JSON recovery export."` (`tldw_chatbook/Library/library_collections_service.py:32-36`).
- Produces: the rail Collections row reads `Collections (N)` where N = captures count from the same enumerator the canvas uses, from first paint.

- [ ] **Step 1 (32057, code): failing test** — mount the rail on a profile with 0 captures; assert the row label is `Collections (0)` before any visit; select the row; assert the Create section's collapsed state is unchanged and that no `[library.rail_state]` write happened for Create. Implement.

- [ ] **Step 2 (32057, copy): failing test** — trigger a refused write in the collections panel (patch the service to raise `LegacyCollectionsReadOnlyError`); assert the canvas shows `Collections are read-only on this profile · Use the legacy Collections inspector or JSON recovery export.` Implement.

- [ ] **Step 3 (32057, decision AC):** you cannot decide what Collections becomes. Write, in the task's Implementation Notes, the two options (retire the row in favour of the captures browser and rename it "Captures"; or restore Collections on the new storage) with what each costs, leave AC #1 unticked, and document the CURRENT reality in `collections.md` (what the row opens today, what is read-only, and the recovery path).

- [ ] **Step 4 (32073): verify each of the eleven contradicted claims live** (socket `crit8-docs`, both profiles) and fix each in the guide with its own stamp: the compact Get-started rule (document that a profile whose config already existed opens the full rail, and cross-reference task-32059); the Conversations detail is a transcript reader with Read/Info and a find box; Import `Show details` and the retry suffix (cross-reference task-32054, describe what ships today); evidence-card keyboard flow (cross-reference task-32053; describe today's mouse path); the F6 Reader stop's border (describe what is actually painted); the Trash heading `Local Trash · N items`; select-strip labels at the 36-cell floor; the landing at 100 columns (cross-reference task-32066); the whitespace-title discard rule. Add the undocumented controls: Notes `New / New folder / Add to folder / Move`, Prompts `Info` tab, the `Chunking Lab | Try selected text` strip (cross-reference task-32064), `ctrl+n` and `/ find note`.

- [ ] **Step 5: backlog + commits** (`docs(library): collections page matches the captures browser; row count and no rail side effects (task-32057)`, `docs(library): guide sweep for the eleven contradicted claims (task-32073)`).

---

### Task 6: Polish, media side (group `polish-media`, tasks 32060, 32065, 32067, 32068, 32070, 32074)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (`check_action` for `library_media_enter_select` ~ grep `_library_media_select_enter_available`; footer hint composition; the adaptive reader profile for the below-64-column stage, grep `list_grows`/`adaptive_reader_state`), `tldw_chatbook/Utils/adaptive_reader_state.py`, `tldw_chatbook/Widgets/Library/library_media_canvas.py` (confirm copy wrapping), `tldw_chatbook/Widgets/Library/library_media_viewer.py` (Markdown notice, byline), `tldw_chatbook/UI/Library_Modules/library_conversation_reader.py` (title/timestamps/pager), `tldw_chatbook/Widgets/Library/library_prompts_canvas.py` (variables checkbox glyph, Use in Console placement)
- Test: `Tests/UI/test_library_crit8_polish_media.py` (new)
- Docs: `library/media-and-conversations.md`, `library/prompts.md`

- [ ] **32060:** failing test — with a Reader item loaded and an Items row focused, press `s`; assert select mode entered and the footer shows `s done selecting`. Fix the `check_action` gate (it currently returns False when the reader holds an item; the pinning test `test_s_key_enters_select_mode_and_space_toggles_a_row` covers only the no-item case — extend it or add the loaded-item case). Wrap the delete confirm sentence at the Items floor (`Delete N selected items? This moves them to trash.` must not clip; use a `Static` with `text-wrap`/width 100% rather than a one-line label).
- [ ] **32065:** failing test at size (60, 24) — activate Media; assert the Items list rows are mounted and a `‹ Library` (or `< Library` under ASCII glyphs) control exists and returns to the rail. Fix the below-64 stage in the adaptive profile.
- [ ] **32067:** failing test — the conversation reader header shows the conversation title (not the UUID) and message stamps render as `27m`/`2d` style (reuse the list's age formatter); the pager is hidden when `page_count == 1`. Apply the same one-page rule to Prompts.
- [ ] **32068:** failing test — a plain-text media item shows the `No Markdown formatting to render` sentence only in the Info tab; an item with no author renders no byline at (100, 30).
- [ ] **32070:** failing tests — after a rail search at (235, 52) exactly one footer row exists; entering select mode immediately lists `space toggle selection | s done selecting`; with the Search/RAG query box focused the footer reads `enter run search`.
- [ ] **32074:** failing test — the prompt-variables dialog checkbox renders `☐`/`☑` (or `[ ]`/`[x]` under ASCII glyphs); `Use in Console` sits in the prompt editor header beside Basic/Advanced/Info.
- [ ] **Live-verify** each on the seeded profile at 235x52, 100x30 and 60x24 (socket `crit8-polish-media`); captures to `<SCRATCH>/crit8/wave/polish-media/caps/`; docs stamps; backlog; one commit per task id.

---

### Task 7: Polish, shell and notes side (group `polish-shell`, tasks 32058, 32059, 32061, 32062, 32063, 32064, 32066, 32069, 32071, 32072)

**Files:**
- Modify: `tldw_chatbook/Library/library_export_scope.py:121-135` (conversation count), `tldw_chatbook/Widgets/Library/library_skills_canvas.py` + `library_rail.py` (count refresh after import), `tldw_chatbook/Library/library_rail_state.py:77-99` (`coerce_library_lifecycle`) and the profile-creation path that should persist `lifecycle = "unknown"` (grep `is_new_profile`), `tldw_chatbook/Widgets/Library/library_notes_canvas.py` (list pane restore on exit; duplicated status line; Add-from-files header), `tldw_chatbook/UI/Screens/library_screen.py:21191-21201` (graduation notice), the Chunking Lab strip composition (grep `Try selected text`), the landing visibility rule (grep `landing` + `compact`), `library_rail.py` (search box clear affordance; Study rows), the nav bar focus style (grep the nav-bar widget under `tldw_chatbook/UI/Navigation/` or `Widgets/`), `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py` (Summary action) and `library_entry_canvases.py` (Get started steps)
- Test: `Tests/UI/test_library_crit8_polish_shell.py` (new), `Tests/Library/test_library_export_scope.py` (extend if it exists)
- Docs: `library.md`, `library/notes.md`, `library/import-and-export.md`, `library/skills.md`, `First_Run_Setup.md`

- [ ] **32058:** failing tests — `get_all_conversation_ids` vs the rail's conversation count on a DB with six conversations of scope `global`: export scope `everything` must count six (fix the enumerator or make both use the same query); after `import_skill` the rail reads `Skills (3)` and the list shows the new row without re-entering.
- [ ] **32059:** failing test — create a profile in run 1 (config created), do not visit Library, relaunch the app object in run 2; assert Library shows Get started. Implement by persisting `lifecycle = "unknown"` into `[library.rail_state]` when `is_new_profile` is true, so the second run reads it.
- [ ] **32061:** failing test — open a note in a wide session (the list pane auto-collapses), press `escape`; assert the Notes list pane is visible again.
- [ ] **32062:** failing test — start typing into the new-note title, trigger the graduation transition mid-typing (call the same method the first source read calls), keep typing; assert the title and body land in their own fields. Implement: never recompose the notes canvas while its editor has focus (defer the graduation recompose until focus leaves, or make the notice a toast that does not recompose).
- [ ] **32063:** failing tests — one status line per Notes canvas (the list pane's authority sentence is not repeated as the canvas header); the Add-from-files header is one sentence under 80 characters; `Library tools are now available.` is delivered through `self.notify` and only on a `COMPACT`→`GRADUATED` transition (not `UNKNOWN`→`GRADUATED`).
- [ ] **32064:** failing test — the `Chunking Lab | Try selected text` strip is not mounted above the canvas; it is reachable under Details ▸ Actions with the gloss `Chunking Lab — compare how text is split for search`; `escape` in the Chunking Lab returns to the Library canvas it was opened from.
- [ ] **32066:** at (100, 30) the landing canvas is hidden and the rail owns navigation (pick this over changing the docs; the guide already states it).
- [ ] **32069:** the rail search Input gains a clear affordance (an `x` compact button or `escape` clears when the box is empty of focus) and is emptied when the canvas changes; the Study section renders three rows with the carry-over hint moved into the staging canvas.
- [ ] **32071:** a nav tab that has keyboard focus but is not active paints a distinct style (`text-style: underline` and no box) so it never looks like a screen switch.
- [ ] **32072:** the wizard Summary offers `Add your first document` (routes to Library ▸ Import); Get started's `1 Add · 2 Find · 3 Use` becomes three controls: `Import a file` (enabled), `Find it` (enabled once any source has content, opens Search/RAG), `Use it in Console` (enabled once a search has results; stages evidence).
- [ ] **Live-verify** on both profiles (socket `crit8-polish-shell`); captures to `<SCRATCH>/crit8/wave/polish-shell/caps/`; docs stamps; backlog; one commit per task id.
