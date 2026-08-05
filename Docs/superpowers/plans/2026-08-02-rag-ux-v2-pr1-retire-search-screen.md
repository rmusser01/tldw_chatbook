# RAG UX v2 — PR-1: Retire the standalone Search screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire the SearchScreen/SearchRAGWindow destination and fold search into the Library screen's Search/RAG canvas, following the six existing screen-retirement precedents, while re-homing the one uniquely valuable Maintenance capability (live backfill progress) into Settings.

**Architecture:** Route id `search` becomes a `_SCREEN_ALIASES` entry resolving to `library`, with a legacy-route nav-context entry so it lands with the Library "Search / RAG" rail row active (the library-side mode table `LIBRARY_NAV_MODE_TO_ROW_ID["search"]` already exists for exactly this). The two palette commands keep working through the same alias. The RAGSearch view files that only served the retired window are deleted; `search_handoff.py` (and anything it imports) survives — it is a live shared library for Library and Console handoffs.

**Tech Stack:** Python ≥3.11, Textual 8.x, pytest (`.venv/bin/python -m pytest` ONLY — system python is 3.9 and breaks collection).

## Global Constraints

- Work ONLY in this worktree (`.worktrees/rag-v2-pr1`, branch `feat/rag-v2-retire-search-screen`). Absolute paths or `git -C` for every git op. NEVER `git stash` (stash is repo-wide across 25+ concurrent worktrees).
- Run tests as `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <paths>` with cwd = the worktree root.
- `TAB_SEARCH` stays in `ALL_TABS` and keeps its display label (precedent: `research` retirement comment, `screen_registry.py:193-201` — startup configs may say `search`; `_valid_startup_route_ids()` accepts it via `ALL_TABS` and the alias must resolve it).
- The business-logic package `tldw_chatbook/RAG_Search/` is NOT part of this retirement. Only `tldw_chatbook/UI/Views/RAGSearch/` view files are candidates, and `search_handoff.py` must survive.
- Never `git add -A` — stage explicit paths only (user keeps untracked scratch under Docs/).
- Commit after each task with the trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Alias `search` → `library` with RAG-canvas landing

**Files:**
- Modify: `tldw_chatbook/UI/Navigation/screen_registry.py` (remove route :119-121, add alias in `_SCREEN_ALIASES` :161-206)
- Modify: `tldw_chatbook/app.py` (`_LEGACY_ROUTE_LIBRARY_NAV_CONTEXT` ~:6020-6024)
- Test: `Tests/ProductionApp/test_retired_destination_root_state.py`, new test in `Tests/UI/test_screen_navigation.py`

**Interfaces:**
- Consumes: `LIBRARY_NAV_CONTEXT_MODE` (`Constants.py:45`), `LIBRARY_NAV_MODE_TO_ROW_ID["search"] → LIBRARY_ROW_BROWSE_SEARCH` (`library_screen.py:520-525`), `LIBRARY_ROW_BROWSE_SEARCH = "browse-search"` (`Library/library_shell_state.py:23`).
- Produces: route id `"search"` (and `TAB_SEARCH`) resolves to `LibraryScreen` with `_library_selected_row_id == LIBRARY_ROW_BROWSE_SEARCH`. Tasks 2-5 rely on this resolution.

- [ ] **Step 1: Read the current shapes before editing.** Read `tldw_chatbook/UI/Navigation/screen_registry.py:100-210`, `tldw_chatbook/app.py:6000-6100` (the `_LEGACY_ROUTE_LIBRARY_NAV_CONTEXT` dict and `_valid_startup_route_ids`), and the existing "ingest" retirement pattern in both files. Also read the `"notes"`/`"ingest"` entries in `Tests/ProductionApp/test_retired_destination_root_state.py` (`expected_routes` map at ~:194) to copy their exact assertion shape.

- [ ] **Step 2: Write the failing tests.**
In `Tests/ProductionApp/test_retired_destination_root_state.py`, change the `expected_routes` entry `"search": SearchScreen` to `LibraryScreen` (matching the `notes`/`ingest` rows) and delete the now-unused `SearchScreen` import. Then add to `Tests/UI/test_screen_navigation.py` (mirroring that file's existing notes-alias navigation test style):

```python
async def test_search_route_lands_on_library_rag_canvas():
    """Retired 'search' route resolves to Library with the Search/RAG rail row active."""
    app = TldwCli()
    async with app.run_test() as pilot:
        app.post_message(NavigateToScreen("search"))
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, LibraryScreen)
        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_SEARCH
```

Adapt imports/harness helpers to the file's existing conventions (it already has an app-harness for the notes alias case — reuse it verbatim rather than inventing a new one).

- [ ] **Step 3: Run both tests, confirm they FAIL** (search still resolves to SearchScreen):
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/ProductionApp/test_retired_destination_root_state.py Tests/UI/test_screen_navigation.py -k "search" -x`
Expected: FAIL asserting `SearchScreen is not LibraryScreen` (or rail-row mismatch).

- [ ] **Step 4: Implement.** In `screen_registry.py`: delete the `"search": ScreenRoute(...)` entry; add to `_SCREEN_ALIASES` (keep dict ordering with the other library folds):

```python
    # The standalone Search screen is retired (RAG UX v2 PR-1, critique
    # 2026-08-02T21-11-50Z): search/RAG now lives entirely inside Library's
    # Search / RAG canvas (rail row "browse-search"), with Console staging
    # via the RAG modal. Existing startup configs / callers using the
    # legacy "search" route id resolve to Library instead of dead-ending --
    # mirrors the "notes"/"prompts"/"skills"/"ingest" aliases above. The
    # route inventory already declared search -> library
    # (UI/Workbench/route_inventory.py).
    "search": "library",
```

In `app.py`, add to `_LEGACY_ROUTE_LIBRARY_NAV_CONTEXT` (mirroring the `"ingest"` entry's exact value shape, using the constant already imported or importing `LIBRARY_NAV_CONTEXT_MODE` from `.Constants`):

```python
    "search": {LIBRARY_NAV_CONTEXT_MODE: "search"},
```

- [ ] **Step 5: Run the tests again, confirm PASS**, then run the neighboring suites that pin this seam:
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/ProductionApp/test_retired_destination_root_state.py Tests/UI/test_screen_navigation.py Tests/UI/test_command_palette_shell_routes.py Tests/UI/test_command_palette_providers.py -x`
`test_command_palette_providers.py` asserts `switch_tab` posts `route_for_tab(tab_id)` for every `ALL_TABS` entry — the alias resolution should keep it green; if it fails, fix the expectation to the resolved route the same way the `notes` retirement did (read that file's history for the precedent: `git -C <worktree> log --oneline -3 -- Tests/UI/test_command_palette_providers.py`).

- [ ] **Step 6: Commit** — `git -C /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/rag-v2-pr1 add tldw_chatbook/UI/Navigation/screen_registry.py tldw_chatbook/app.py Tests/ProductionApp/test_retired_destination_root_state.py Tests/UI/test_screen_navigation.py && git -C ... commit -m "feat(nav): retire search route to Library RAG canvas alias"`

### Task 2: Palette commands land on the Library RAG canvas with honest copy

**Files:**
- Modify: `tldw_chatbook/app.py:1105-1106` (`search_all` dispatch), `:1364-1367` (`search_transcripts` dispatch), `:720` (tab help text)
- Test: `Tests/UI/test_command_palette_basic.py` or the provider dispatch tests (read first; extend where the palette dispatch assertions live)

**Interfaces:**
- Consumes: Task 1's alias resolution.
- Produces: palette "Search All Content" / "Search Transcripts" open Library-with-RAG-canvas; toasts say "Opened Library Search/RAG".

- [ ] **Step 1: Write the failing test.** Locate the existing dispatch test for `search_all` (grep `Tests/UI/ -rn "search_all\|Opened Search/RAG"`). Add/adjust an assertion that invoking the command results in a `NavigateToScreen` whose resolved screen is `LibraryScreen` (reuse the Task 1 harness) and that the toast copy is `"Opened Library Search/RAG"`.
- [ ] **Step 2: Run it, confirm FAIL** (old toast copy).
- [ ] **Step 3: Implement.** In both dispatch sites keep `_navigate_via_screen(self.app, TAB_SEARCH, ...)` (the alias does the routing) but update the toast strings: `"Opened Library Search/RAG"` and `"Opened Library Search/RAG for transcript search"`. Update `TAB_HELP_TEXT[TAB_SEARCH]` (`app.py:720`) to `"Switch to Library search and RAG"`.
- [ ] **Step 4: Run the test file + `Tests/UI/test_command_palette_basic.py`, confirm PASS.**
- [ ] **Step 5: Commit** (explicit paths, message `feat(palette): route search commands to Library RAG canvas`).

### Task 3: Re-home live backfill progress into Settings (the one uniquely-lost capability)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py:9975-10055` (the backfill worker — currently passes NO `progress_callback`) and `:9748-9786` / `#settings-library-rag-index-status` (status line renderer)
- Test: `Tests/UI/test_settings_rag_profile_region.py` (extend — it owns this region's coverage)

**Interfaces:**
- Consumes: `backfill_semantic_index(..., progress_callback=...)` — signature per the retired window's usage at `search_rag_window.py:1104` and CLI `RAG_Search/backfill.py:79-91`.
- Produces: during a Settings-triggered backfill, `#settings-library-rag-index-status` updates per batch: `"Indexing {source}: {indexed} indexed, {up_to_date} up-to-date, {failed} failed"`; final summary behavior unchanged.

- [ ] **Step 1: Read** `settings_screen.py:9975-10074` (worker + final notify) and `RAG_Search/backfill.py:42-91` (callback contract: confirm exact parameter names/shape the callback receives) and the retired window's `_progress` at `tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py:1050-1110` BEFORE it is deleted in Task 4.
- [ ] **Step 2: Write the failing test** in `Tests/UI/test_settings_rag_profile_region.py`: mount the settings screen harness the file already uses, monkeypatch `backfill_semantic_index` with a fake that invokes its `progress_callback` twice then returns a summary object (copy the fake-summary shape from the file's existing backfill test), and assert the status widget text went through the per-batch string and ends at the final summary.

```python
async def test_backfill_streams_progress_to_index_status(settings_harness, monkeypatch):
    calls = []
    def fake_backfill(*args, progress_callback=None, **kwargs):
        assert progress_callback is not None
        progress_callback(source="media", indexed=3, up_to_date=1, failed=0)
        progress_callback(source="notes", indexed=5, up_to_date=0, failed=1)
        return _existing_fake_summary()  # reuse the file's existing summary fixture
    # monkeypatch at the settings_screen import site, run the backfill action,
    # assert "Indexing notes: 5 indexed, 0 up-to-date, 1 failed" appeared in
    # #settings-library-rag-index-status before the final summary text.
```

Adapt the callback's parameter shape to what Step 1 found — the assertion in the fake pins the contract.
- [ ] **Step 3: Run, confirm FAIL** (`progress_callback is None` assertion trips or status never updates).
- [ ] **Step 4: Implement**: pass a `progress_callback` from the Settings worker that formats the per-batch line and updates `#settings-library-rag-index-status` via `call_from_thread` (the worker is threaded — follow the file's existing `call_from_thread` usage in the same worker).
- [ ] **Step 5: Run the test file, confirm PASS, no other failures.**
- [ ] **Step 6: Commit** (`feat(settings): stream backfill progress to RAG index status line`).

### Task 4: Delete the retired surface

**Files:**
- Delete: `tldw_chatbook/UI/Screens/search_screen.py`, `tldw_chatbook/UI/SearchRAGWindow.py`, `tldw_chatbook/UI/Views/RAGSearch/search_rag_window.py`, `search_event_handlers.py`, `search_history_dropdown.py`, `saved_searches_panel.py`
- Conditional: `search_result.py`, `constants.py` — delete ONLY if `search_handoff.py` and surviving tests don't import them (verify in Step 1; `Tests/UI/test_search_handoffs.py` directly tests `SearchResult`, so expect BOTH to survive)
- Modify: `tldw_chatbook/UI/Views/RAGSearch/__init__.py` (trim to surviving exports), `tldw_chatbook/UI/Screens/__init__.py:15,26` (drop SearchScreen lazy-import + `__all__`)
- Delete: `Tests/UI/test_search_rag_window.py`
- Modify: `Tests/UI/test_search_handoffs.py` (keep handoff-builder + SearchResult coverage; delete SearchRAGWindow-mount cases)

**Interfaces:**
- Consumes: Tasks 1-2 landed (nothing routes to SearchScreen anymore).
- Produces: `UI/Views/RAGSearch/` contains only `__init__.py`, `search_handoff.py` (+ `search_result.py`/`constants.py` if required). `from tldw_chatbook.UI.Views.RAGSearch.search_handoff import build_library_rag_console_live_work_payload, build_library_rag_evidence_bundle` keeps working — `library_screen.py:243` and `chat_screen.py:452-455` depend on it.

- [ ] **Step 1: Verify the survivor set.** `grep -n "^from\|^import" tldw_chatbook/UI/Views/RAGSearch/search_handoff.py` and `grep -rn "search_result\|RAGSearch.constants\|from .constants" tldw_chatbook/ Tests/UI/test_search_handoffs.py --include='*.py' | grep -v __pycache__ | grep -v "search_rag_window\|search_event_handlers\|search_history_dropdown\|saved_searches_panel"`. Record which of `search_result.py`/`constants.py` have surviving importers.
- [ ] **Step 2: Delete the files** (`git rm` the confirmed-dead list), trim both `__init__.py`s to the survivors.
- [ ] **Step 3: Edit `Tests/UI/test_search_handoffs.py`**: delete test classes/functions that mount `SearchRAGWindow`; keep the ones that call `build_library_rag_console_live_work_payload` / `build_library_rag_evidence_bundle` / construct `SearchResult` directly. Delete `Tests/UI/test_search_rag_window.py` (its subject no longer exists; Maintenance-contract coverage lives in `Tests/UI/test_settings_rag_profile_region.py` incl. Task 3's new test).
- [ ] **Step 4: Import-sanity + affected suites:**
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.UI.Screens.library_screen, tldw_chatbook.UI.Screens.chat_screen, tldw_chatbook.UI.Views.RAGSearch"` then
`... -m pytest Tests/UI/test_search_handoffs.py Tests/UI/test_screen_navigation.py Tests/ProductionApp/ -x`
Expected: PASS, zero collection errors.
- [ ] **Step 5: Commit** (`refactor(search)!: delete retired SearchScreen/SearchRAGWindow surface`).

### Task 5: Reconcile the cross-cutting test suites

**Files:**
- Modify: `Tests/UI/test_destination_headers.py:58-60,2032` (drop SearchScreen row), `Tests/ProductionApp/test_reactive_ownership_maturity.py` (drop the `("search", TAB_SEARCH, SearchScreen)` ROUTE_SPECS row), `Tests/UI/test_disabled_action_recovery_tooltips.py` (delete if SearchRAGWindow-only — read first; if it covers other widgets keep those), `Tests/UI/test_product_maturity_phase1_empty_setup_states.py:256` (re-point the exemplar case at the Library RAG panel empty state or delete the case), `Tests/UI/test_unified_shell_phase5_recovery_taxonomy.py:238-253` (drop the search_rag_window subprocess-import case), `Tests/UI/test_non_obscuring_focus_contract.py:45` (drop the path entry), `Tests/UI/verify_command_palette.py:127,142` (keep — TAB_SEARCH still in ALL_TABS; verify unchanged)

- [ ] **Step 1:** For each file: read the referenced region, make the minimal edit, keeping any coverage not about the retired widget.
- [ ] **Step 2: Run the edited files:** `... -m pytest Tests/UI/test_destination_headers.py Tests/ProductionApp/test_reactive_ownership_maturity.py Tests/UI/test_disabled_action_recovery_tooltips.py Tests/UI/test_product_maturity_phase1_empty_setup_states.py Tests/UI/test_unified_shell_phase5_recovery_taxonomy.py Tests/UI/test_non_obscuring_focus_contract.py -x` → PASS.
- [ ] **Step 3: Commit** (`test: reconcile suites with Search screen retirement`).

### Task 6: Docs + intentional-loss record

**Files:**
- Modify: `CLAUDE.md` (Screens list: remove the SearchRAGWindow/search_screen mention; note search lives in Library)
- Create: nothing else — the PR body records the intentional losses.

- [ ] **Step 1:** Edit CLAUDE.md's UI Layer section accordingly.
- [ ] **Step 2:** Draft the PR-body "Intentional losses" block (goes into the PR description at ship time): search analytics tiles (noise at n=2 — RAG-25), Saved Searches panel (never populatable from the UI — RAG-07 documented dead handlers), `search_history.db` recording (DB class + its Tests/DB suites survive; no UI writer remains; Library keeps its own config-backed history), per-item-type index scoping (survives via `RAG_Search/backfill.py --types` CLI; Settings backfill is all-types), multi-row index-stats table (Settings' one-line status with provenance is the replacement; "runtime not initialized" distinction dropped).
- [ ] **Step 3: Commit** (`docs: record Search screen retirement`).

### Task 7: Full verification + live check

- [ ] **Step 1: Full UI + ProductionApp + MCP suites:** `... -m pytest Tests/UI/ Tests/ProductionApp/ Tests/MCP/ 2>&1 | tail -5` — baseline is GREEN (0 failures) per repo memory; any failure is ours.
- [ ] **Step 2: Ruff:** `.venv/bin/python -m ruff check tldw_chatbook/ Tests/` on the changed files → clean.
- [ ] **Step 3: Live verification** (tmux, socket name suffixed with the session id to avoid cross-session collisions — see memory trap): launch the app from the worktree, then (a) palette → "Search All Content" → assert capture shows Library screen with Search/RAG rail row active; (b) quit, relaunch with a scratch `TLDW_CONFIG_PATH` config containing `[general] default_tab = "search"` → assert boot lands on Library (not a crash/fallback toast); (c) capture both as evidence files. Do NOT run destructive actions; the scratch profile isolates (b).
- [ ] **Step 4: Commit any fixes, then stop.** PR creation happens after plan-executor handoff review.
