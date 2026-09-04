# Media UX fix wave 4 — PR C (P2 batch) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land critique #4's P2 batch for Library ▸ Media: the list refreshes itself after the app's own Trash mutations, the Find bar stays put and leaves no join artifact, the Reader sheds its chrome overhead and reads at a sane measure, Trash lays out its Restore action next to the item, and the filter matches keywords.

**Architecture:** Each task is a contained fix with its own test; none changes the review-set model or the Find focus token from PR A. Layout claims are pinned with painted-text or region assertions on the production-CSS harness; the filter change is pinned at the scope-service seam.

**Tech Stack:** Python 3.12, Textual 8.x, pytest + pytest-asyncio; `LibraryProductionCSSHarness` (`Tests/UI/test_library_shell.py`), `_flow_app` / `ControlledDetailMediaService` (`Tests/UI/test_library_media_reader_flow.py`), `_painted` (`Tests/UI/test_library_media_render_fixes.py`), `Tests/UI/test_library_media_trash.py` for the Trash canvas.

**Spec:** `.impeccable/critique/2026-09-04T13-50-05Z__tldw-chatbook-ui-screens-library-screen-py.md` "P2 batch" paragraph; tasks `backlog/tasks/task-31274`, `task-31275`, `task-31276`, `task-31277`, `task-28015` (their Acceptance Criteria are binding; 28015 gained a header-clip AC in wave 4).

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/media-wave4-c`, branch `fix/media-wave4-c`, stacked on `fix/media-wave4-a` (PR #2378). Every command: `cd <worktree> && PYTHONPATH=<worktree> /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest … -p no:cacheprovider`. Absolute paths; the shell cwd can be reset between calls.
- Run UI test files in separate processes; compare failures against the branch base before claiming them — `test_library_shell.py` carries the known failures in `backlog/tasks/task-31249 - …md`.
- No new `logger.*` calls without regenerating `Docs/security/production-diagnostic-inventory.json` (`python scripts/check_persistent_diagnostic_inventory.py --write`).
- After any `BUNDLED_CSS` / `tldw_chatbook/css/components/*.tcss` edit: `python -m tldw_chatbook.css.build_css` then `python tldw_chatbook/css/check_bundle_sync.py` (exit 0); commit regenerated files under `tldw_chatbook/css/`.
- Widget-tier CSS loses to app-tier rules; the app-global `*:focus` outline paints over content (task-31221) — prove layout with painted text where clipping or paint-over is the risk.
- The media summary contract (`_MEDIA_SUMMARY_KEYS`, five keys) is frozen in this PR; task-31278's design note owns that change.
- Commit per task with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` / `Claude-Session: https://claude.ai/code/session_011LebG4HPfSniVohbuXkU4n`. Backlog task files are flipped by the controller.

---

### Task 1: The list refreshes itself after the app's own Trash mutation (task-31275)

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` — `handle_library_media_trash_restore` (:25124) / `_restore_library_media_from_trash` (:25154) and the permanent-delete path (grep `Delete permanently` handler), the "‹ Media" return from the Trash canvas (grep `library-media-trash-back` or the Trash canvas's back id), the mutation seam `_begin_library_media_mutation` / `_complete_library_media_mutation` (grep; the bulk-delete and single-delete paths use it and leave the list FRESH).
- Read: `tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py` (:28 `_MUTATION_COPY = "Media changed; retry to load a current page."`, `freshness`, `begin_mutation`, `reconcile_committed_mutation`, `request`).
- Test: `Tests/UI/test_library_media_trash.py` (append).

**Interfaces:**
- Produces: after Restore or permanent delete in the Trash canvas, returning to Media shows a fresh list (`screen._library_media_browse_controller.freshness == "fresh"`, no `○` rows, no "Media changed; retry" banner, no Retry control) without a user Retry. The stale gate keeps firing for changes the app did not make itself (the existing controller tests must still pass).
- Consumes: the same reconcile-committed-mutation seam the delete paths use.

- [ ] **Step 1: Failing test** `test_restore_from_trash_returns_a_fresh_media_list`: seed 3 items, trash one via the real service (`mark_as_trash`) or the UI's `t` + confirm, open the Trash canvas (`#library-media-trash-open`), press its Restore, press its back control; assert the list has 3 rows, `freshness == "fresh"`, `not screen.query("#library-media-retry")` (grep the Retry button id) and the painted canvas has no "Media changed" line. Add a negative control: an external mutation (call the media DB directly to trash a row while the list is showing) still trips the stale gate after a page request (assert the existing behaviour, not a new one).
- [ ] **Step 2: Run; confirm the first fails on the stale gate (`Retry` present / freshness stale).**
- [ ] **Step 3: Implement**: route the Trash canvas's Restore and permanent-delete through `begin_mutation` → service call → `reconcile_committed_mutation(...)` exactly as `_delete_library_media_item` does (~:40176-40260), so the controller knows the change is its own; on the "‹ Media" return, request the page through the reconciled scope instead of the stale path.
- [ ] **Step 4: Run** `Tests/UI/test_library_media_trash.py`, `Tests/UI/test_library_media_browse_controller.py`, `Tests/UI/test_library_multiselect_media.py`, `Tests/UI/test_library_media_side_by_side.py`.
- [ ] **Step 5: Live tmux 235x52**: trash → Trash → Restore → ‹ Media → rows are live, no Retry. Seed/cleanup as in PR A (`MediaDatabase.add_media_with_keywords`, soft-delete after; `python3 …/scratchpad/click.py <sock> "<label>"` for clicks; sleeps inside the same call; kill the server).
- [ ] **Step 6: Commit** `fix(library): the media list refreshes itself after its own Trash mutations (task-31275)`.

---

### Task 2: The Find bar stays put; no join artifact after it closes (task-31276)

**Files:**
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` (:3527-3540 `#library-media-content-search-controls.-library-media-search-active { … }` — the task-15774 dock that moves an active bar to the top of the viewer), `tldw_chatbook/Widgets/Library/library_media_content.py` (:320-327 `ACTIVE_SEARCH_CLASS` comment; keep the class for styling, remove the dock behaviour), `tldw_chatbook/Widgets/Library/library_media_viewer.py` (the comment at ~:332-337 about docking relative to the immediate container).
- Investigate the artifact: `┐─────Local Media item` — five stray `─` cells at the pane join on the row of the reader identity line, appearing after Find closes / tab clicks / More (never on a fresh open). Suspects: the docked bar's removal leaving the shell's join row unrepainted (`LibraryMediaReaderShell`, `tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py`), or a border on `#library-media-reader-identity`.
- Test: `Tests/UI/test_library_media_render_fixes.py` (append), `Tests/UI/test_library_shell.py::test_library_shell_media_viewer_search_chrome_undocks_when_inactive` (its "active search docks to the top" assertion `controls_active.region.y <= back_button.region.y` must be rewritten to assert the bar STAYS under the mode row — that test currently encodes the behaviour this task retires; note it in the report).

**Interfaces:**
- Produces: the Find bar's `region.y` is identical before and after a submitted query and after match navigation; after Escape closes the bar, the painted row that holds "Local Media item" contains no `┐─` run.

- [ ] **Step 1: Failing tests**: `test_find_bar_keeps_its_place_through_submit_and_next` (record `controls.region.y` after Find; submit "item"; press Next; assert unchanged) and `test_no_join_artifact_after_find_closes` (open Find, Escape, painted text of the identity row via `_painted(host, screen.query_one("#library-media-reader-identity").region)` has no `─`; also after pressing the Analysis tab and after More → Escape).
- [ ] **Step 2: Run; the first fails on `y` moving up; the second fails only if the artifact reproduces in the harness — if it does NOT reproduce, say so in the report, keep the test as a guard, and hunt the artifact live (tmux 235x52 capture after Find → Escape) to find the repaint owner before deciding whether a code fix exists.**
- [ ] **Step 3: Implement** — drop the dock (keep the active-search class for the status/prev/next styling), fix the artifact owner if found, rebuild the CSS bundle.
- [ ] **Step 4: Run** `test_library_media_render_fixes.py`, `test_library_shell.py -k "search or find"` (compare to base; the two plain-harness failures listed in task-31249 stay), `test_library_media_reader_match_nav_t22209.py`.
- [ ] **Step 5: Live tmux 235x52 and 100x30** captures before/after; put both in the report.
- [ ] **Step 6: Commit** `fix(library): the Find bar stays in place; no join artifact after it closes (task-31276)`.

---

### Task 3: Reader chrome overhead and reading measure (task-31277)

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_media_viewer.py` — identity line `Static(… else "Local Media item", id="library-media-reader-identity")` (~:203-208): compose only when `self.external_detail` (server item) is set; byline (~:226-243, `id="library-media-reader-byline"`): compose only when author or URL exists; the section header under the mode row (`Static("Read", classes="destination-section")` at ~:309 and the "Analysis" header at ~:600): remove the one that repeats the selected tab (keep the mode row as the label); content measure: cap the text body at ~90 cells (`#library-media-viewer-content-text { max-width: 92; }` in the component sheet, keeping the box full width so the border still spans the pane).
- Modify: `tldw_chatbook/Library/library_media_viewer_state.py` (:201-208 `_is_markdown_media` gates the sniff on `_MARKDOWN_MEDIA_TYPES`; extend so transcripts of `video`/`audio` items whose content sniffs as markdown (`looks_like_markdown_content`) default to Rendered — list the exact types you add in the report).
- Test: `Tests/UI/test_library_media_render_fixes.py` (append), `Tests/Library/test_library_media_viewer_state.py` (grep for the existing sniff tests; append).

**Interfaces:**
- Produces: on a local item with no author/URL, rows between the pane top and the first content line ≤ 5 (title, toolbar, mode row, box border, first line) — assert via `_painted` rows of the reader pane; the identity line renders for an external detail (existing external tests must still pass: grep `external_detail` in Tests/UI); `##` headings in a video transcript render as headings (Markdown widget present in Rendered mode).

- [ ] **Step 1: Failing tests** for each of the four claims (chrome rows ≤ 5; no byline row without author/URL; no repeated section header; video transcript with `## Section 1` defaults to Rendered with a `Markdown` body) plus a viewer-state unit test for the sniff types.
- [ ] **Step 2: Run; confirm each fails.**
- [ ] **Step 3: Implement**; rebuild CSS.
- [ ] **Step 4: Run** `test_library_media_render_fixes.py`, `test_library_media_reader_flow.py`, `test_library_media_side_by_side.py`, `Tests/Library/test_library_media_viewer_state.py`, `test_library_shell.py -k "viewer or reader or byline or identity"` (compare to base).
- [ ] **Step 5: Live tmux 235x52 before/after captures in the report.**
- [ ] **Step 6: Commit** `fix(library): the Reader sheds its chrome overhead and reads at a sane measure (task-31277)`.

---

### Task 4: Trash lays out Restore next to the item; header no longer clips (task-28015)

**Files:**
- Modify: `tldw_chatbook/Widgets/Library/library_media_trash_canvas.py` (`compose` at :190; today the row list is `height: 1fr` so the pager + `Restore  Delete permanently` row pins ~36 blank rows below a single item; the header `Local Trash · 1 item` clips to `· 1 i` at the pane width).
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` (library-media-trash rules; grep `library-media-trash`).
- Test: `Tests/UI/test_library_media_trash.py` (append painted-text tests at (235, 52)).

**Interfaces:**
- Produces: the Restore / Delete permanently action row renders directly below the last trash row (its `region.y` ≤ last row bottom + 2), the pager stays with the list, and the header paints `Local Trash · 1 item` in full at the Items-pane width (shorten to `Trash · 1 item` if the pane cannot fit the longer form — record the choice).

- [ ] **Step 1: Failing tests**: `test_trash_actions_sit_under_the_last_row` and `test_trash_header_paints_in_full`.
- [ ] **Step 2: Run; confirm both fail (large `y` gap; painted `· 1 i`).**
- [ ] **Step 3: Implement** (list container `height: auto` with a `max-height: 1fr` scroll, actions in flow; header width 100% with a shorter label if needed); rebuild CSS.
- [ ] **Step 4: Run** `test_library_media_trash.py`, `test_library_media_side_by_side.py`.
- [ ] **Step 5: Live tmux 235x52 and 100x30 captures.**
- [ ] **Step 6: Commit** `fix(library): Trash keeps Restore beside the item and its header intact (task-28015)`.

---

### Task 5: The media filter matches keywords, and says what it searched (task-31274)

**Files:**
- Trace first: `tldw_chatbook/UI/Screens/library_screen.py:15913` `_request_library_media_filter` → the browse scope → `tldw_chatbook/Media/media_reading_scope_service.py:724-735` (`search_fields = …; fields=search_fields`) → `tldw_chatbook/DB/Client_Media_DB_v2.py` `search_media(search_fields=…)` (default fields; whether `keywords` is a supported field — grep `search_fields` handling and the FTS/keyword join).
- Modify: the scope-service field list for the Library browse filter to include keywords (or, if the DB layer supports a keyword filter only through a separate parameter, pass the query there as well and union the results — record which); the empty-state copy in `tldw_chatbook/Library/library_media_state.py` (grep `No media matched`) to name the searched fields: `No media matched “day2” in titles or keywords.`; the filter placeholder (`tldw_chatbook/Widgets/Library/library_media_canvas.py` `#library-media-filter`) to say `Filter by title or keyword…`.
- Test: `Tests/Media/test_media_reading_scope_service.py` (append: a keyword-only match is returned for the Library browse filter), `Tests/UI/test_library_media_render_fixes.py` (append: filtering by a keyword present on one seeded row shows that row; the miss copy names titles and keywords).

**Interfaces:**
- Produces: `MediaReadingScopeService.search_media(..., library_summary=True, query=<kw>)` returns items whose keywords match; the five-key summary shape is unchanged.

- [ ] **Step 1: Failing tests** (service-level keyword match; UI keyword filter shows the row; miss copy).
- [ ] **Step 2: Run; confirm they fail with zero results / old copy.**
- [ ] **Step 3: Implement** the trace's smallest change; document the searched fields in the task notes (AC#1).
- [ ] **Step 4: Run** `Tests/Media/test_media_reading_scope_service.py`, `Tests/Media/test_local_media_reading_service.py`, `Tests/DB/test_client_media_pagination.py`, `test_library_media_render_fixes.py`, `test_library_media_browse_controller.py`.
- [ ] **Step 5: Live tmux 235x52**: filter `day2` on seeded rows carrying that keyword → rows shown; `zz` → the new miss copy with Clear filter.
- [ ] **Step 6: Commit** `fix(library): the media filter matches keywords and names what it searched (task-31274)`.
