# RAG UX v2 — PR-2: Library RAG panel honesty + cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Library Search/RAG panel — now the app's sole search surface after PR #1258 — honest and operable (coverage signals, banded scores, clamped snippets, live scope copy, quiet no-match, keyboard traversal, recompose guard), and retire the last dead remnants of the old Search screen (CSS sheet, Constants blob section, `SearchResult` widget).

**Architecture:** All display logic stays in the existing seams: `Library/library_rag_state.py` (pure display state), `Widgets/Library/library_search_rag_panel.py` (children builders), `Library/library_local_rag_search_service.py` (backend), `UI/Screens/library_screen.py` (screen wiring). New signals ride the existing-but-unused `LibraryRagSearchOutcome.diagnostics` mapping. No schema changes, no new dependencies.

**Tech Stack:** Python ≥3.11, Textual 8.x, pytest via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest` ONLY (cwd = worktree root).

## Global Constraints

- Work ONLY in worktree `.worktrees/rag-v2-pr2` (branch `feat/rag-v2-library-rag-panel`). Absolute paths or `git -C` for every git op. NEVER `git stash`. Never `git add -A`.
- Verification gates are TARGETED test runs (user ruling): each task names its covering files; no full-suite sweeps. A `--collect-only -q` sweep of `Tests/UI/ Tests/Library/` is the dangling-import check.
- CSS edits: edit SOURCE files under `css/components/` / `css/features/`, then regenerate with `python3 tldw_chatbook/css/build_css.py` and run `python3 tldw_chatbook/css/check_bundle_sync.py`; commit source + bundle together. Never hand-edit `css/tldw_cli_modular.tcss`.
- `Tests/UI/test_library_content_hub.py:168-216` pins a list of widget ids that must stay ABSENT — check any new widget id against it.
- TDD for every behavior task; commit per task with trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Deferred by design: RAG-28 ("RAG Answer" label — PR-3 makes it true); RAG-37 (title cut is source data, not a defect — record only).

---

### Task 1: RAG-27 — recompose guard for the Search/RAG canvas

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:2182-2212` (`_apply_local_source_snapshot` narrow-path guard)
- Test: `Tests/UI/test_library_shell.py` (new test near `test_library_search_typed_text_survives_registry_recompose` at ~:12136)

**Interfaces:**
- Consumes: `LIBRARY_ROW_BROWSE_SEARCH` (`Library/library_shell_state.py:23`), the existing in-place rail-sync path at `library_screen.py:2196-2210`, `_refresh_search_rag_panel_state_widgets(include_results_and_history=False)` (`:16393-16463`).
- Produces: a background ingest done-count growth while `_library_selected_row_id == LIBRARY_ROW_BROWSE_SEARCH` no longer whole-screen-recomposes; rail counts AND scope-toggle counts stay fresh.

- [ ] **Step 1: Read** `library_screen.py:2162-2212` (guard + in-place path) and the existing test at `Tests/UI/test_library_shell.py:12136` for the harness pattern (how the registry-changed event is simulated).
- [ ] **Step 2: Write the failing test** (mirror the :12136 harness): select the browse-search row, mount results via the static service, capture `id(panel_widget)` and the mounted results container, simulate the ingest-registry done-count growth event, then assert the SAME panel/canvas widget instances remain mounted (no recompose) AND the rail was synced. Also assert `_refresh_search_rag_panel_state_widgets` ran with `include_results_and_history=False` (results widgets untouched — compare the result Static instances by identity).
- [ ] **Step 3: Run it, confirm FAIL** (today the fall-through at `:2212` recomposes, so widget identities change).
- [ ] **Step 4: Implement:** extend the guard's row-id condition to include `LIBRARY_ROW_BROWSE_SEARCH`, and in the in-place branch, after `rail.sync_state(...)`, when the selected row is browse-search also call `self._refresh_search_rag_panel_state_widgets(include_results_and_history=False)` so scope-toggle counts/run-gate reflect new source counts. Extend the comment at `:2191-2195` (which documents the PR #1261 precedent) with one sentence for the search case.
- [ ] **Step 5: Run** the new test + `Tests/UI/test_library_rag_keystroke.py` (it pins `include_results_and_history=False` semantics) + `Tests/UI/test_library_ingest_canvas.py`. All green.
- [ ] **Step 6: Commit** `fix(library): keep Search/RAG canvas mounted across background ingest snapshots`.

### Task 2: Dead CSS retirement

**Files:**
- Delete: `tldw_chatbook/css/features/_search-rag.tcss`
- Modify: `tldw_chatbook/css/build_css.py:60` (drop the manifest entry; add a removal comment in the style of `:50-56`), `tldw_chatbook/css/components/_shared_components.tcss` (re-home two rules), `tldw_chatbook/Constants.py:1488-1633` (delete the dead Search Tab section of the already-dead `css_content` blob, banners included), `Tests/UI/test_non_obscuring_focus_contract.py:27` (delete unused `SEARCH_RAG` constant), regenerate `css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_non_obscuring_focus_contract.py` (new retired-selector-absence test)

**Interfaces:**
- Consumes: scout audit — only 5 of 104 selectors in `_search-rag.tcss` have live users; `.action-spacer` (`:542`) and `.param-group` (`:239`) are SOLE definitions with live users (`UI/CodeRepoCopyPasteWindow.py:247,263`; `Widgets/Media/media_viewer_panel.py:672,679`); `.action-button`/`.settings-section`/`.status-bar` are redefined in sheets loaded later, so their copies here are shadowed and safe to drop.
- Produces: bundle without the `_search-rag` module; `.action-spacer` + `.param-group` living in `components/_shared_components.tcss`.

- [ ] **Step 1: Copy the two orphan rules verbatim** from `_search-rag.tcss:542` (`.action-spacer`) and `:239` (`.param-group`) into `components/_shared_components.tcss` with a comment noting the re-home (precedent: `.window-title`, and the documentation style of `build_css.py:50-56`).
- [ ] **Step 2: Delete** `_search-rag.tcss`, drop `build_css.py:60`, delete `Constants.py:1488-1633` (verify the banner comments `/* --- Search Tab (RAG/Embeddings) --- */` … `/* --- End of Search Tab --- */` bound exactly what you remove), delete `test_non_obscuring_focus_contract.py:27`.
- [ ] **Step 3: Regenerate + verify:** `python3 tldw_chatbook/css/build_css.py` then `python3 tldw_chatbook/css/check_bundle_sync.py` (must pass). Then grep the new bundle for `search-query-input-enhanced` and `saved-searches` — zero hits.
- [ ] **Step 4: Write the absence test** in `test_non_obscuring_focus_contract.py` mirroring `test_library_mode_chip_selector_is_retired_from_focus_contracts` (`:685-693`): assert `.search-query-input-enhanced` and `.results-list-enhanced` appear in NO bundled css module (`bundled_css_module_paths` helper `:165`). Run it (passes immediately post-deletion — that is correct for an absence pin; note it as such, no RED phase applies).
- [ ] **Step 5: Run** `Tests/UI/test_non_obscuring_focus_contract.py` + a live-user smoke: `... -m pytest Tests/UI/ -k "code_repo or media_viewer" --collect-only -q` (collection only — the two re-homed classes have no dedicated tests; the bundle-sync check is the gate).
- [ ] **Step 6: Commit** `chore(css)!: retire _search-rag.tcss and Constants dead Search section` (source + bundle + Constants + test together).

### Task 3: Retire `SearchResult` + `constants.py`; rename the handoff test file

**Files:**
- Delete: `tldw_chatbook/UI/Views/RAGSearch/search_result.py`, `tldw_chatbook/UI/Views/RAGSearch/constants.py`
- Modify: `tldw_chatbook/UI/Views/RAGSearch/__init__.py` (docstring-only, no re-exports — `search_handoff` is imported by full path)
- Rename: `Tests/UI/test_search_handoffs.py` → `Tests/UI/test_library_rag_handoffs.py`, deleting only the one `SearchResult` test (`:14-22`) and its import (`:11`)

**Interfaces:**
- Consumes: scout audit — `search_result.py` has exactly one importer (the test being trimmed); `constants.py` exactly one (`search_result.py:14`); `search_handoff.py` is imported by full module path at `library_screen.py:265` and `chat_screen.py:474`, NOT via the package `__init__`, so trimming `__init__` to empty is safe.
- Produces: `UI/Views/RAGSearch/` = `__init__.py` (docstring) + `search_handoff.py` only.

- [ ] **Step 1: Verify the import claims** (one grep each) before deleting; `git rm` the two files; trim `__init__.py`; `git mv` the test file and trim it.
- [ ] **Step 2: Import-sanity:** `... -c "import tldw_chatbook.UI.Screens.library_screen, tldw_chatbook.UI.Screens.chat_screen, tldw_chatbook.UI.Views.RAGSearch"` then run `Tests/UI/test_library_rag_handoffs.py` (9 tests green).
- [ ] **Step 3: Commit** `refactor(search): retire SearchResult widget; RAGSearch package is handoff-only`.

### Task 4: Boot-time nav-context for legacy routes

**Files:**
- Modify: `tldw_chatbook/app.py:7565-7602` (`_push_initial_screen`)
- Test: `Tests/UI/test_screen_navigation.py` (new boot-path test near `:2157`)

**Interfaces:**
- Consumes: `_LEGACY_ROUTE_LIBRARY_NAV_CONTEXT` (`app.py:6193-6199`, keys are PRE-resolution route ids), `apply_navigation_context` seam (`library_screen.py:1819-1855`; unmounted screens take the sync path), the guarded-apply pattern at `app.py:6672-6687`.
- Produces: booting with `default_tab = "search"` (or "prompts"/"skills") lands on Library with the mapped rail row active, not generic Library.

- [ ] **Step 1: Write the failing test:** mirror `test_search_route_lands_on_library_rag_canvas` (`:2157`) but drive the BOOT path: monkeypatch the config default-tab resolution (find how existing boot tests set `default_tab` — grep `_normalize_initial_tab_from_config` usages in Tests/) so the app boots with initial tab "search"; assert the pushed screen is `LibraryScreen` with `_library_selected_row_id == LIBRARY_ROW_BROWSE_SEARCH`.
- [ ] **Step 2: RED** (today boot lands generic Library — `_library_selected_row_id` is the default rail row).
- [ ] **Step 3: Implement** in `_push_initial_screen`: after constructing the screen and before `push_screen` (`:7595`), look up `self._LEGACY_ROUTE_LIBRARY_NAV_CONTEXT.get(initial_tab, {})` (the pre-resolution id — capture it before `_resolve_screen_navigation_target` rewrites it) and, when non-empty and the screen has `apply_navigation_context`, apply it inside the same try/except-log shape as `:6672-6687` (await if awaitable).
- [ ] **Step 4: GREEN**; also run the existing `:2157` and `:2194` tests (message path unaffected).
- [ ] **Step 5: Commit** `feat(nav): apply legacy route nav-context on boot (default_tab=search lands on RAG canvas)`.

### Task 5: Focus-contract CSS rule + test for `#library-rag-query-input`

**Files:**
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss:4883-4889` (base rule), regenerate bundle
- Test: `Tests/UI/test_non_obscuring_focus_contract.py` (new test copying the Chatbooks pattern at `:1589-1601`)

- [ ] **Step 1: Write the failing test** `test_library_rag_query_input_uses_stable_thin_contracts`: `css_block(AGENTIC, "#library-rag-query-input")` and `...:focus` through `assert_thin_input_focus` (`:195-201`) and `assert_stable_solid_border_geometry` (`:216-221`).
- [ ] **Step 2: RED** — base block lacks `border-bottom: solid`.
- [ ] **Step 3: Implement:** add `border-bottom: solid $ds-grid-line;` to the base rule (the focus block at `:4891-4896` already carries its accent bottom edge). Regenerate bundle + `check_bundle_sync.py`.
- [ ] **Step 4: GREEN**; run the whole focus-contract file.
- [ ] **Step 5: Commit** `fix(css): reserve bottom border on library RAG query input (no focus jitter)` (source + bundle + test).

### Task 6: Snippet display clamp + markdown strip + entity-leak fix (RAG-30/31)

**Files:**
- Modify: `tldw_chatbook/Library/library_rag_state.py` (`_sanitize_display_text` `:163-179`, `_collapse_text` `:151-156`, `from_result` `:581-592`, new constants near `:53-54`), `tldw_chatbook/Widgets/Library/library_search_rag_panel.py:310` (snippet Static gains a display-clamped text + class)
- Test: `Tests/Library/test_library_rag_state.py`

**Interfaces:**
- Consumes: current pipeline `_sanitize_display_text(..., escape=True)` ending in `escape_markup(html.escape(text, quote=False))` (`:179`); stored snippet cap `LIBRARY_RAG_SNIPPET_MAX_LENGTH = 4_000` (`:54`).
- Produces: `LibraryRagResultRow.snippet` unchanged (full 4,000-char handoff/evidence payload); NEW property `LibraryRagResultRow.display_snippet` returning the row's on-screen text: markdown-stripped, clamped to `LIBRARY_RAG_SNIPPET_DISPLAY_MAX_CHARS = 320` at a word boundary with a trailing `…` when clamped. Panel `:310` renders `row.display_snippet`.

- [ ] **Step 1: Write failing tests** (each one behavior): (a) `display_snippet` strips markdown structure — input `"## Project Overview\n**Status:** Planning\n- item"` renders `"Project Overview Status: Planning item"`-style plain text (heading markers `#`, emphasis `**`/`__`/`*`/`_`, list markers, code fences and backticks removed; text content preserved); (b) `display_snippet` clamps >320 chars at a word boundary with `…`, and short snippets pass through unclamped; (c) the entity leak: a snippet containing `R&D` renders `R&D` exactly once escaped on screen (today `html.escape` on already-escaped text yields `&amp;amp;` — pin the fix: unescape entities BEFORE the single `html.escape`, i.e. `html.unescape` first in `_sanitize_display_text` when `escape=True`); (d) existing tests at `:157/:198/:202-216` still pass (stored `snippet` behavior unchanged).
- [ ] **Step 2: RED** (property missing; entity double-escape present).
- [ ] **Step 3: Implement:** add a small pure `_strip_markdown_syntax(text)` helper (regex-based, no new deps) + `display_snippet` property + the `html.unescape` pre-step; switch panel `:310` to `row.display_snippet` and give that Static `classes="library-rag-result-snippet"` (id absent from the content-hub absence list — verified against `test_library_content_hub.py:168-216`).
- [ ] **Step 4: GREEN**; run `Tests/Library/test_library_rag_state.py` + `Tests/UI/test_product_maturity_gate16_library_search_rag.py` (row rendering pinned there).
- [ ] **Step 5: Commit** `feat(library): clamp evidence snippets for display, strip markdown, fix entity double-escape`.

### Task 7: Score banding + weak-results signal (RAG-34, half of RAG-29)

**Files:**
- Modify: `tldw_chatbook/Library/library_rag_state.py` (new banding helper + row line), `tldw_chatbook/Widgets/Library/library_search_rag_panel.py:294-297` (title-line score suffix)
- Test: `Tests/Library/test_library_rag_state.py`

**Interfaces:**
- Consumes: `row.score` (None for keyword rows — service hard-sets None at `library_local_rag_search_service.py:578/590/605/628`; real cosine for semantic at `:662`).
- Produces: `library_rag_score_suffix(score) -> str`: `""` for None; `" | match: strong"` (≥0.5), `" | match: moderate"` (0.2–0.5), `" | match: weak (0.09)"` (<0.2 — weak keeps the raw number for transparency). Panel `:294` uses it. NEW: `library_rag_all_matches_weak(rows) -> bool` (True when ≥1 scored row and all scored rows band weak) — consumed by Task 8's coverage line.

- [ ] **Step 1: Failing tests:** band edges (0.5 → strong boundary inclusive, 0.2 → moderate boundary inclusive, weak shows 2-decimal raw), None → empty, `all_matches_weak` truth table (no scored rows → False).
- [ ] **Step 2: RED. Step 3: implement. Step 4: GREEN** + gate16 file re-run (title-line shape is pinned there — update its expected strings where they assert `| score 0.xxx`).
- [ ] **Step 5: Commit** `feat(library): replace raw cosine scores with honest match bands`.

### Task 8: Semantic coverage diagnostics + honest Evidence heading (rest of RAG-29)

**Files:**
- Modify: `tldw_chatbook/Library/library_local_rag_search_service.py` (`_search_semantic` `:404-489` populates diagnostics), `tldw_chatbook/Library/library_rag_service.py` (`_diagnostics_from_result` `:193-198` already threads — verify), `tldw_chatbook/UI/Screens/library_screen.py` (`_apply_library_rag_search_outcome` `:16351-16391` stores diagnostics; panel-state gains the field), `tldw_chatbook/Widgets/Library/library_search_rag_panel.py` (`:266-268` heading + new one-line coverage Static), `tldw_chatbook/Library/library_rag_state.py` (panel-state field + copy builder)
- Test: `Tests/Library/test_library_local_rag_search_service.py`, `Tests/Library/test_library_rag_state.py`, `Tests/UI/test_product_maturity_gate16_library_search_rag.py`

**Interfaces:**
- Consumes: `LibraryRagSearchOutcome.diagnostics` (`library_rag_service.py:91` — exists, semantic never populates, UI never reads); scope allowlist plumbing in `_search_semantic` (`:462-469`); `library_rag_all_matches_weak` from Task 7.
- Produces: (a) `_search_semantic` adds `diagnostics["semantic_scope_coverage"] = {"covered": [...], "uncovered": [...]}` — source types requested by scope vs. source types actually present in returned provenance; (b) panel-state gains `coverage_note: str` built by `library_rag_coverage_note(diagnostics, rows)`: empty string when everything covered and matches aren't all-weak; `"Semantic search found nothing from: conversations, notes."` when types are uncovered; prepends `"No strong semantic matches — results below are weak. "` when Task 7's predicate fires; (c) heading at `:266-268` becomes mode-aware: keyword mode keeps `"Evidence · top {top_k} per source"` (true for the keyword leg), rag mode renders `"Evidence · top {top_k}"` (the per-source claim is false there — scout item 3); (d) the coverage Static renders under the heading with `classes="library-rag-quiet-line"` (reuses existing CSS `_agentic_terminal.tcss:4943-4947`), only when non-empty, id `library-rag-coverage-note` (absent from the content-hub absence list — verified).
- [ ] **Step 1: Failing service test:** rag-mode search with a scope allowlist of {notes, media} whose results contain only media provenance ⇒ diagnostics carries `uncovered: ["notes"]`. **Step 2: RED. Step 3: implement service side.**
- [ ] **Step 4: Failing state/panel tests:** coverage-note copy builder truth table; heading strings per mode; note only mounts when non-empty. **RED → implement → GREEN.**
- [ ] **Step 5: Run** all three named test files + `Tests/Library/test_library_rag_service.py` (diagnostics threading).
- [ ] **Step 6: Commit** `feat(library): per-source semantic coverage notes and honest evidence heading`.

### Task 9: Delete the unmounted Retrieval Inspector (RAG-35)

**Files:**
- Delete (code regions, not files): `LibrarySearchRagInspectorPanel` + compose/refresh (`library_search_rag_panel.py:519-643`), its helpers `_SELECTED_EVIDENCE_DETAIL_IDS` (`:24-44`), `_console_handoff_summary` (`:466-482`), `_inspector_recovery_summary` (`:484-489`), `_future_attribution_summary` (`:491-499`), `_selected_evidence_detail_ids/_classes/_details` (`:646-694`)
- Modify: `library_screen.py:232` (import), `:16556-16565` (`_refresh_library_rag_inspector`) + its call site `:16437`; `Widgets/Library/__init__.py:17,59`
- Keep: `LibraryRagResultRow`'s label properties (`library_rag_state.py:699-765`) and their tests — data layer serves PR-3/PR-4 handoffs.

- [ ] **Step 1: Read** `Tests/UI/test_product_maturity_gate16_library_search_rag.py:541` (selected-evidence metadata test) FIRST — confirm what it queries; if it exercises `_selected_evidence_details`, keep that one helper and delete only the widget + refresh plumbing; record which in the report.
- [ ] **Step 2: Delete per the file list; import-sanity; run** `Tests/UI/test_product_maturity_gate16_library_search_rag.py` (absence assertions at `:206-207` keep passing), `Tests/UI/test_library_content_hub.py`, `Tests/Library/test_library_rag_state.py` (kept properties still covered).
- [ ] **Step 3: Commit** `refactor(library): delete never-mounted RAG retrieval inspector`.

### Task 10: Live scope summary (RAG-32)

**Files:**
- Modify: `tldw_chatbook/Library/library_rag_state.py` (new `library_rag_scope_summary(scope_state) -> str`; retire the constant usage), `library_search_rag_panel.py:119-121` (`_scope_summary`), `library_screen.py:16588-16590` (`_library_rag_scope_summary` — BOTH seams must call the shared builder; `Tests/UI/test_library_shell.py:3620` pins they agree)
- Test: `Tests/Library/test_library_rag_state.py`, `Tests/UI/test_library_shell.py:3620`

- [ ] **Step 1: Failing tests:** all-selected → `"Scope: all local sources"` (unchanged copy for the common case); subset → `"Scope: notes, conversations (media, prompts off)"` (selected in canonical order, deselected parenthesized); none-available edge → existing gate copy untouched.
- [ ] **Step 2: RED → implement the shared builder consuming `LibraryRagScopeState` (`from_source_counts` `:299-395` exposes selected/available) → both seams delegate → GREEN** incl. the shared-copy test.
- [ ] **Step 3: Commit** `feat(library): scope summary reflects actual source toggles`.

### Task 11: Quiet no-match state (RAG-33)

**Files:**
- Modify: `tldw_chatbook/Library/library_rag_service.py:260-273` (`_empty_results_recovery_state`) OR the render seam `library_search_rag_panel.py:376-377` — decision: render-seam. `library_rag_results_body_children` special-cases `status == "empty"`: emit `Static(quiet_copy, id=state.recovery_selector, classes="library-rag-quiet-line")` with two-line copy `"No evidence matched '{query}'."` / `"Try broader terms or more sources."` instead of `visible_copy`'s six lines. Real failures (deps/index/provider/policy) keep the full `DestinationRecoveryState` dump — only the routine no-match goes quiet.
- Test: `Tests/UI/test_product_maturity_gate16_library_search_rag.py`, `Tests/Library/test_library_rag_service.py:162`, `Tests/UI/test_library_content_hub.py:164-181`

- [ ] **Step 1: Failing test** (gate16 style): run a query against the static service returning zero rows; assert the rendered node has the quiet-line class, contains the query text and the two-line copy, does NOT contain `"Owner:"`/`"Unavailable:"`; assert a deps-blocked outcome still renders the full dump. Keep the selector id `library-rag-empty-state` (existing tests key on it).
- [ ] **Step 2: RED → implement → GREEN**; run the three named files.
- [ ] **Step 3: Commit** `feat(library): quiet two-line no-match state (retire the Owner dump for routine empties)`.

### Task 12: Keyboard traversal of evidence rows (RAG-36)

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_search_rag_panel.py:271-338` (`library_rag_result_row_children` — wrap each row's Statics+actions in a focusable `Vertical(classes="library-rag-result-card", id=f"library-rag-result-card-{index}", can_focus=True)`), `tldw_chatbook/UI/Screens/library_screen.py` (three key handlers scoped to the focused card: Enter → select evidence, `o` → open, `u` already global; resolve index via the card id through the existing `_trailing_index` `:15948-15957`), `css/components/_agentic_terminal.tcss` (new `.library-rag-result-card` base + `:focus` rules — visible focus via the DS focus border tokens, matching `#library-rag-query-input:focus`'s idiom; bundle regen)
- Test: `Tests/UI/test_product_maturity_gate16_library_search_rag.py` (new tests near the "u" shortcut test `:660`)

**Interfaces:**
- Consumes: existing handlers `select_library_rag_result` (`:16100`) / `open_library_rag_result` (`:16112`) — the key handlers call the same underlying methods, no logic duplication; refresh path `_refresh_library_rag_results_widgets` (`:16567-16587`) and `LIBRARY_RAG_RESULTS_STATIC_WIDGET_IDS` (`:397`) must be updated for the new container structure.
- Produces: Tab reaches each result card (focus visibly indicated per the focus-contract idiom); on a focused card Enter selects evidence, `o` opens, `u` stages to Console (existing binding, now with a focused-card fast path that selects-then-stages).
- New ids `library-rag-result-card-*` checked against the content-hub absence list — not present.

- [ ] **Step 1: Read** `_refresh_library_rag_results_widgets` + the gate16 Enter/u tests to match harness idiom. **Step 2: failing tests:** (a) Tab from the query input eventually focuses card 0 and the card has a focus-styled border class/pseudo-state; (b) Enter on focused card 1 marks it `is-selected` (same assertion shape as `:252-282` in content-hub); (c) `o` on a focused note card routes like the Open button (mirror `test_library_shell.py:8726`). **Step 3: RED → implement (widget + screen keys + CSS + refresh-path lockstep + bundle regen) → GREEN.**
- [ ] **Step 4: Run** gate16 file + `test_library_content_hub.py` + `test_library_rag_keystroke.py` (row rebuild identity assumptions) + focus-contract file (new CSS parses; add the card's rules to the contract test if the file's conventions require every new `:focus` rule to be registered — read `:58-95` markers first).
- [ ] **Step 5: Commit** `feat(library): keyboard-traversable evidence cards (Enter select, o open, visible focus)`.

### Task 13: P3 copy pair + deps-copy pip hint (RAG-38, RAG-39 + Task-14 enabler)

**Files:**
- Modify: `library_search_rag_panel.py:406-459` (history buttons gain tooltip `"Re-runs under the current mode ({mode_label})"` — dynamic from panel state), `:68-72` + `:261-263` (mode toggle tooltip becomes `"Cycle Search/RAG mode. Next: {other_mode_label}."`), `tldw_chatbook/Library/library_local_rag_search_service.py:799-813` (`_rag_mode_unavailable_recovery_state` `next_action` gains the pip hint: `'Install RAG support: pip install "tldw_chatbook[embeddings_rag]", then restart.'` — mirrors the copy family in `RAG_Search/semantic_availability.py:53-57`)
- Test: `Tests/Library/test_library_rag_state.py` / panel-child tests where tooltips are pinned; `Tests/Library/test_library_local_rag_search_service.py:445` (extend the blocked-outcome test to pin the pip copy)

- [ ] **Step 1: failing tests → Step 2: implement → Step 3: GREEN** (three small behaviors, one test each).
- [ ] **Step 4: Commit** `feat(library): honest re-run/mode-cycle hints; deps recovery names the pip extra`.

### Task 14: Dependency-missing coverage (Phase-1.6 re-add, Library-shaped)

**Files:**
- Test only: `Tests/UI/test_product_maturity_phase1_empty_setup_states.py` (new test) or gate16 file — follow whichever file's harness reaches `_start_library_rag_query` more naturally (scout: `LibraryHarness` from `test_library_shell`).

- [ ] **Step 1: Write the test:** monkeypatch `tldw_chatbook.Library.library_local_rag_search_service.embeddings_rag_deps_installed` (module attribute — imported by name at `:44`) to `False`; mount via `LibraryHarness`; set mode `"rag"`; run a query; assert `#library-rag-service-error` renders `_rag_mode_unavailable_recovery_state` copy INCLUDING the Task-13 pip hint; assert keyword ("search") mode still works un-gated.
- [ ] **Step 2:** This test must FAIL if Task 13's pip copy is reverted (it pins the new copy) — run once with Task 13's change stashed? No `git stash` — instead assert the exact new copy string so any revert fails the test. Run GREEN.
- [ ] **Step 3: Commit** `test(library): dependency-missing RAG mode renders honest recovery with install hint`.

### Task 15: Targeted verification + live check

- [ ] **Step 1: Targeted gate** (single foreground Bash, timeout 600000): `Tests/Library/test_library_rag_state.py Tests/Library/test_library_rag_service.py Tests/Library/test_library_local_rag_search_service.py Tests/UI/test_product_maturity_gate16_library_search_rag.py Tests/UI/test_library_rag_keystroke.py Tests/UI/test_library_content_hub.py Tests/UI/test_library_rag_handoffs.py Tests/UI/test_non_obscuring_focus_contract.py Tests/UI/test_screen_navigation.py Tests/UI/test_library_ingest_canvas.py` plus `-k "library_shell and (rag or history or scope or search)" Tests/UI/test_library_shell.py`. Zero new failures.
- [ ] **Step 2: Collection sweep:** `Tests/UI/ Tests/Library/ --collect-only -q` → 0 errors. Ruff on changed files.
- [ ] **Step 3: Live check** (socket `uat-pr2-805d`, session-suffixed; verify pane ownership before trusting captures; REAL user data — navigation + searches only, nothing destructive): (a) Library → Search/RAG: run a real query in keyword mode → banded/absent scores, clamped snippets with `…`, live scope summary reflecting a toggled-off source; (b) garbage query → quiet two-line no-match (no "Owner:"); (c) Tab to a result card → visible focus; Enter selects; (d) `default_tab = "search"` scratch-profile boot (TLDW_CONFIG_PATH, users_name verify_pr2_805d) lands on the RAG canvas row directly; delete scratch profile after; (e) kill-server MANDATORY.
- [ ] **Step 4: Evidence** to /private/tmp/uat-pr2-805d-evidence/NN-*.txt; report; commit any fixes; NO PR creation (controller owns it).
