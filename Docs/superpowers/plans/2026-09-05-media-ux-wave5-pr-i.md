# Media UX fix wave 5 — PR I (row state markers, tasks 28008/28009, design note 31278 Option A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Library ▸ Media Items list answers "has this been analysed?", "have I reviewed this in the active set?" and "why did this keyword search match?" on the row itself, without opening anything.

**Architecture:** The user-approved design note `backlog/docs/design-library-row-state-markers.md` (Option A) bumps the frozen media summary contract from five keys to seven: `has_analysis: bool` projected in SQL (newest `DocumentVersions` row's `analysis_content` non-empty — the same rule the viewer's `_latest_version_analysis_text` applies), and `reviewed: bool | None` decorated on the screen from the active `ReviewSet` (`None` = no active set). Task 1 bumps the contract, the SQL projection, the normalisers and the validator, with ONE test-fake helper so every fake builds rows from one place. Task 2 decorates `reviewed` from the active set and renders the markers: `✓` / `·` in the row's leading state slot (select mode's `☑/☐` replaces it while active) and the word `analysed` on the secondary line. Task 3 adds the keyword-match reason on the secondary line for hits that came from keywords rather than the title.

**Tech Stack:** Python 3.12, SQLite, Textual 8.x, pytest + Hypothesis where the validator is property-tested; `Tests/UI/test_library_media_browse_controller.py`, `Tests/DB/test_client_media_pagination.py`, `Tests/DB/test_client_media_debug_logging.py`, `Tests/Media/test_local_media_reading_service.py`, `Tests/Media/test_media_reading_scope_service.py`, `Tests/UI/test_console_rag_settings_modal.py` (the six shape tests named in the note), `Tests/UI/test_library_shell.py` (`_two_media_items`, `StaticLibraryMediaScopeService`), `Tests/UI/test_library_media_side_by_side.py` (`_many_media_items`), `Tests/UI/test_destination_shells.py` (its own `StaticLibraryMediaScopeService`), `Tests/UI/test_library_media_render_fixes.py` (`_painted`), `Tests/UI/test_review_set_walker.py` / `test_review_set_banner.py`.

**Spec:** `backlog/docs/design-library-row-state-markers.md` (§3 Option A, §5 approval on 2026-09-04: decisions 2-4), `backlog/tasks/task-28008 - Library-media-list-show-analysis-presence-on-rows.md`, `backlog/tasks/task-28009 - Library-media-list-read-markers-for-sequential-review.md`; critique #5 P2 "the list row cannot answer the question the list exists to answer".

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/media-wave5-i`, branch `fix/media-wave5-i` off dev. Every command: `cd <worktree> && PYTHONPATH=<worktree> /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest … -p no:cacheprovider`; absolute paths; UI test files in separate processes; every Bash call begins with the explicit `cd` and `git branch --show-current`.
- THIS PR is the one that unfreezes the five-key contract — and only to exactly seven keys: `id, backing_media_id, title, media_type, updated_at, has_analysis, reviewed`. Every producer and every fake must move in the same commit as the validator, or the suite goes red across six files.
- `has_analysis` must come from SQL, never from a per-row Python lookup (paging cost); a new query shape needs a plan captured with `sqlite_stat1` ABSENT per CLAUDE.md gotcha 1 — if you add an index, `scripts/check_index_plan_pins.py` must have a row in `scripts/index_plan_pin_census.tsv`; prefer no new index (the `DocumentVersions(media_id, version_number)` lookup should use the existing index — assert the plan in a test).
- `reviewed` is NOT a media-DB fact: it is the active review set's done mark; the projection leaves it `None`; the screen decorates rows before handing them to the canvas; review-set code itself (`review_set_state.py`) is read, not changed.
- Text carries meaning: `✓`, `·`, `analysed` are characters; no colour-only markers. The 36-cell Items-pane floor: the secondary line `document · 5m · analysed` (24 cells) must fit; the leading state slot is one cell before the title in browse mode and is REPLACED by select mode's `☑/☐` (never both).
- Compare failures against the base before claiming them (known list as in the other wave-5 plans). No new `logger.*`; CSS only if a rule is needed (rebuild + `check_bundle_sync` exit 0); no new toolbar buttons; the Find focus token untouched.
- Live verification: tmux (`t() { tmux -L w5i "$@"; }`), real config, ONE app instance; seed items with and without an analysis version (`save_analysis_version` or the DocumentVersions API) and a keyword-only match; clean with `soft_delete_media`.
- TDD per task; commit per task with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; backlog task files are flipped by the controller.

---

### Task 1: The seven-key contract, projected in SQL, with one fake helper (task-28008 data half)

**Files:**
- Modify: `tldw_chatbook/Library/library_media_state.py` — `_MEDIA_SUMMARY_KEYS` gains `has_analysis`, `reviewed`; `validate_media_browse_items` checks the exact seven-key set, `has_analysis is bool`, `reviewed in (None, False, True)`; the error copy says seven; the Trash validator (`exactly five keys` near the Trash items check) is a SEPARATE contract — leave it at five unless Trash rows are produced by the same projection (check; if they are, keep Trash at five by stripping the two keys in the Trash normaliser).
- Modify: `tldw_chatbook/DB/Client_Media_DB_v2.py` — `search_media(library_summary=True)` selects `has_analysis` as `EXISTS (SELECT 1 FROM DocumentVersions v WHERE v.media_id = Media.id AND v.version_number = (SELECT MAX(version_number) FROM DocumentVersions WHERE media_id = Media.id) AND TRIM(COALESCE(v.analysis_content, '')) <> '')` (read the real column names in `_latest_version_analysis_text`'s consumer and the schema first); `reviewed` is not selected (the normaliser sets `None`).
- Modify: `tldw_chatbook/Media/media_reading_scope_service.py` (`_normalize_local_library_summary` and the `library_summary` branch), `tldw_chatbook/Media/local_media_reading_service.py` passthrough, `tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py` (nothing beyond the validator import), `tldw_chatbook/UI/Screens/library_screen.py` (Review-these / Review-selected page through the same projection — confirm they do not re-shape rows).
- Create: `Tests/UI/library_media_rows.py` (test-only helper): `summary_row(*, id, title, media_type="article", updated_at=None, has_analysis=False, reviewed=None, backing_media_id=None) -> dict` and `summary_rows(n, **overrides)`; rewrite `_two_media_items`, `_many_media_items`, both `StaticLibraryMediaScopeService` fakes and the six shape tests to build rows through it.
- Test: `Tests/DB/test_client_media_pagination.py` (a real DB with two items, one carrying a newest version with analysis text and an OLDER version without, one with no version: `has_analysis` True/False; the plan uses the existing DocumentVersions index — assert `EXPLAIN QUERY PLAN` contains no `SCAN DocumentVersions`); `Tests/Media/test_media_reading_scope_service.py` (normaliser emits seven keys, `reviewed is None`); `Tests/Library/test_library_media_state.py` (validator accepts seven, rejects five and eight, rejects `has_analysis=1`, rejects `reviewed="yes"`; Hypothesis over the value domain).

**Interfaces:**
- Produces: seven-key rows; `summary_row` helper.

- [ ] Step 1: failing tests (validator ×5; projection ×2 + plan; normaliser).
- [ ] Step 2: run; confirm (five-key error copy; no column).
- [ ] Step 3: implement all producers + helper + fakes in ONE commit.
- [ ] Step 4: run the six shape test files, `test_library_shell.py -k "media"`, `test_library_media_side_by_side.py`, `test_destination_shells.py`, `test_library_multiselect_media.py`, `test_library_media_render_fixes.py`, `test_review_set_walker.py` (compare to base — expect 0 new failures).
- [ ] Step 5: no live step.
- [ ] Step 6: commit `feat(library): media summary contract carries has_analysis and reviewed (task-28008)`.

---

### Task 2: Decorate `reviewed` from the active set and render the markers (task-28009, task-28008 render half)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` — before rows reach the canvas (`_library_media_canvas_presentation()` or the browse-state sync), when a review set is active, set `reviewed = item id in the set's done ids` for rows in the set, `False` for in-set-not-done, leave `None` for rows outside the set; with no active set leave `None`. Read `ReviewSet` / `ReviewSetItem` in `review_set_state.py` for the done-mark field. Re-decorate on every done-mark change (the banner's sync seam).
- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py` — the row: leading state slot (1 cell): `✓` when `reviewed is True`, `·` when `reviewed is False`, space when `None`; in select mode the slot is the existing `☑/☐` marker (no second slot); secondary line: `f"{media_type} · {age}" + (" · analysed" if has_analysis else "")`.
- Test: `Tests/UI/test_review_set_walker.py` / `test_review_set_banner.py` (rows in the active set carry `·`, the done one `✓`, rows outside carry neither; marking done via `m` flips the row); `Tests/UI/test_library_media_render_fixes.py` (painted at 235×52 and 100×30 with the Items pane at its 36-cell floor: `document · 5m · analysed` fits without truncation; select mode shows `☐` and not `·`); `Tests/UI/test_library_multiselect_media.py` (the marker slot swap).
- Docs: `Docs/User_Guide/library/media-and-conversations.md` — one paragraph on the row markers; fresh "Verified against" stamp.

**Interfaces:**
- Consumes: Task 1's keys.
- Produces: the row grammar `[state] title` / `type · age · analysed`.

- [ ] Step 1: failing tests (decoration ×3; painted ×2 sizes; select-mode swap).
- [ ] Step 2: run; confirm (no markers).
- [ ] Step 3: implement; docs.
- [ ] Step 4: run `test_review_set_walker.py`, `test_review_set_banner.py`, `test_library_media_render_fixes.py`, `test_library_multiselect_media.py`, `test_library_shell.py -k "review or row"` (compare to base).
- [ ] Step 5: live 235×52 and 100×30: two analysed items show `analysed`; Review these → `·` on every row, `]` twice → two `✓`; `s` → `☐` replaces the slot.
- [ ] Step 6: commit `feat(library): rows show analysed and reviewed state (task-28008, task-28009)`.

---

### Task 3: Keyword hits say why they matched (critique #5 P2, task-28008 AC on match reason)

**Files:**
- Modify: `tldw_chatbook/DB/Client_Media_DB_v2.py` — when the keyword leg is on and the row matched ONLY through keywords (title/content legs did not match), the summary projection carries the matched keyword in a NON-contract field? No — the contract is exactly seven keys. Instead: `tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py` keeps a per-page `match_reasons: Mapping[str, str]` (row id → matched keyword) computed by the service from the same query (a second lightweight SELECT over `MediaKeywords` for the page's ids and the query term, LIKE-escaped exactly as the keyword leg is), passed to the canvas as a separate argument (like `analysis_action_reason` is today).
- Modify: `tldw_chatbook/Widgets/Library/library_media_canvas.py` — secondary line appends ` · keyword: <term>` when a reason exists for the row; at the 36-cell floor the keyword is truncated with `…` after 10 characters.
- Test: `Tests/Media/test_media_reading_scope_service.py` (a query that matches only a keyword yields a reason for that id and none for a title hit); `Tests/UI/test_library_media_render_fixes.py` (painted: `article · 2m · keyword: notes` on the keyword-only hit, nothing extra on the title hit, at both sizes).

**Interfaces:**
- Consumes: PR C's keyword leg (`LIBRARY_BROWSE_SEARCH_FIELDS`, LIKE escaping).
- Produces: `match_reasons` side channel (explicitly NOT a contract key — record why in the docstring: the summary contract is per-row identity, the reason is per-query).

- [ ] Step 1: failing tests (service ×2; painted ×2).
- [ ] Step 2: run; confirm.
- [ ] Step 3: implement.
- [ ] Step 4: run `test_media_reading_scope_service.py`, `test_library_media_browse_controller.py`, `test_library_media_render_fixes.py`, `test_library_shell.py -k "keyword or search"` (compare to base).
- [ ] Step 5: live 235×52: search a keyword-only term → the row shows `keyword: <term>`.
- [ ] Step 6: commit `feat(library): keyword-only hits say which keyword matched (task-28008)`.
