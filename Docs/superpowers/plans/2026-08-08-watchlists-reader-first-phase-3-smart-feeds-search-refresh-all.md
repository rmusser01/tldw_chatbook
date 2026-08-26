# Watchlists reader-first, phase 3: smart feeds, search, refresh-all

**Spec**: `Docs/superpowers/specs/2026-08-05-watchlists-reader-first-design.md` §Phasing ("Phase 3 — smart feeds, search, refresh-all", :330-332).
**ADR**: backlog/decisions/042-watchlists-reader-first-ia.md (already covers the re-IA; phase 3 is a direct implementation — no new ADR, same ruling as phase 2).
**Phase 2 shipped**: PR #1430 (merge `e03f9a170`), TASK-3072. This plan assumes that tree.

**Goal (the spec's done-when)**: daily triage runs entirely from the rail + `j/k/m/s/a/u/r//`.

Phase 3 adds: **All Unread + Today** rail nodes beside Starred, a corpus-wide **`/` search** riding the existing `subscription_items_fts` FTS5 table with a LIKE fallback, **`r` refresh-all** with guardrails and one aggregated notification, and an **"N new items" pill** that tells the user a refresh produced items without yanking the list mid-triage.

## Current state (what phase 2 left)

- `TreeScope.kind` is `Literal["all","unassigned","watchlist","source","starred"]` (`watchlist_tree.py:32`). Smart-feed precedent (Starred, TASK-3072): a `_root_node(key, label, bucket, phrase=...)` yielded in `compose()` above the watchlists; the badge rides `counts` under a dedicated bucket the screen inserts in `_load_tree_data` (`STARRED_BUCKET = -3`); `on_button_pressed` maps `wl-tree-node-<key>` to the scope; the screen's `_items_scope_query` maps the kind to `list_items` kwargs; `_tree_scope_label` names it; `_scoped_loaded_sources` treats it like `all`.
- `Subscriptions_DB.get_new_items` (`Subscriptions_DB.py:1793`) builds predicate fragments with bound params (:1870-1892) — `subscription_id` / `status` / `run_id` / `watchlist_id` / `unassigned_only` / `statuses` / `is_flagged` already compose; TASK-3072 added `is_flagged` in exactly the shape `search`/`since` will follow. Ordering is `COALESCE(i.published_date, i.created_at) DESC`.
- **FTS5 already exists**: `subscription_items_fts` external-content table over `(title, content, author)` with ai/ad/au triggers and `backfill_items_fts()` (`Subscriptions_DB.py:919-1036`). What does NOT exist is any query method reading it — phase 3 adds the first. FTS5 query-syntax injection is a real hazard (`[`, `]`, `"`, `AND`/`OR`/`NEAR` are operators): the escape precedent is the Library's `build_fts_match_query` (pinned hostile-input tests in `Tests/Library/test_library_local_rag_search_service.py:1262-1280`). Fallback rule: catch `sqlite3.OperationalError` around the MATCH (fts5 compiled out, table missing on a pre-migration DB) and degrade to `LIKE` on the same columns — never raise into the reader.
- The items search box (`#items-search-input`, `article_list.py`) is currently CLIENT-SIDE only: `_filtered_items` (:265-296) substring-matches title/url/source_name over the loaded 100-item page. The screen mirrors the query via `ItemsFilterChanged` and re-seeds it on rebuild; `_load_items` takes no search term. Phase 3 pushes the term into the query so results span the whole corpus, and KEEPS the client-side filter as the instant pre-filter over the loaded page.
- Screen reader-verb gating is `_reader_verb_blocked()` (:10009): typing in an Input/TextArea is typing, and verbs are scoped to the Read tab. `/`, `s`, `o` all gate through it (`/` must NOT fire while the user is already typing somewhere).
- `check_now` is per-source only: controller `:148` → scope service `:870` → `launch_run` (:545, policy `runs.launch`; local backend immediately `execute_run`s). There is no check-all. Sources carry `is_active`/`is_paused` (see `resume_source`, `local_watchlists_service.py:648-687`); the screen's loaded source dicts expose both.
- The screen's single-source flow `_check_now_source` (used by `test_watchlists_rail_counts_and_scope.py`) ends in `_request_tree_counts_refresh()` — the debounced `_load_tree_data` that also refreshes the Starred badge. Refresh-all reuses the same path, once, at the END of the batch — never per source.
- The "N new items" count is a DELTA: `counts[ALL_SOURCES_BUCKET]["unread"]` before the batch vs after the terminal tree reload. No new schema, no run-table archaeology — the same honesty rule as the rail badges (the number is unread items, the legend says so).
- Items pane header strip (`#items-search-input` / status select / refresh button) is one `destination-filter-strip` row; the pill is a compact `Static` appended to that strip, hidden when empty.

## Tasks

### Task 1: Task bookkeeping + docs commit

- [ ] Create the backlog task (ACs from the spec's done-when), status In Progress, plan link; commit this plan + task file. Message: `docs(watchlists): phase 3 plan — smart feeds, search, refresh-all (task-3791)`

### Task 2: `search` + `since` predicates in `get_new_items`

**Files:**
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py` (`get_new_items` :1793; new `get_unread_items_count_since`)
- Modify: `tldw_chatbook/Subscriptions/local_watchlists_service.py` (`list_items` :366 area), `tldw_chatbook/Subscriptions/watchlist_scope_service.py` (`list_items` :273 forwards)
- Test: `Tests/DB/test_subscriptions_db_watchlists.py`, `Tests/Watchlists/test_watchlist_scope_service.py`

- [ ] **Step 1: failing tests.** `search="foo"` returns only rows whose title/content/author match (FTS); a hostile query string (`"`, `[`, `NEAR/1`, bare `AND`) never raises an FTS5 syntax error (escaping pinned against a real table); when the FTS read raises `OperationalError` the LIKE fallback answers instead (forced by monkeypatching the FTS probe); `since=<ISO>` restricts to `COALESCE(published_date, created_at) >= ?`; both compose with `status`/`is_flagged`/membership; both forward verbatim through the two service layers (kwargs-assertion tests, the phase-2 `is_flagged` pattern). `get_unread_items_count_since(since)` counts `status='new'` rows at/after the floor — the Today badge's query.
- [ ] **Step 2: implement.** Escape by double-quoting each whitespace-separated token (the Library precedent), JOIN `subscription_items_fts` on `rowid = i.id` with `subscription_items_fts MATCH ?`, fall back to `(i.title LIKE ? OR i.content LIKE ? OR i.author LIKE ?)` with `%`-wrapped escaped terms on `OperationalError`. All values bound parameters; only fixed predicate TEXT is assembled (the :1864-1869 rule).
- [ ] **Step 3: run + commit** `feat(watchlists): search/since predicates for get_new_items (task-3791)`

### Task 3: `/` — corpus-wide search from the reader

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (BINDINGS, `_load_items` :8553, `handle_items_filter_changed` :~9055, `_items_status_kwargs` call site)
- Modify: `tldw_chatbook/UI/Watchlists_Modules/article_list.py` (nothing structural — the client-side pre-filter stays)
- Test: `Tests/Watchlists/test_watchlists_collections_screen.py`, `Tests/Watchlists/test_watchlists_article_list.py`

- [ ] **Step 1: failing tests.** `/` on the Read tab focuses `#items-search-input` (and does nothing while another Input has focus); typing debounce-dispatches ONE `_load_items` whose service call carries the search term (spy assertion, the `test_tree_move_triggers_items_reload_on_read_tab` shape); a search result NOT in the newest-100 page can surface (seed 101+ items, search for the oldest's unique token); clearing the box restores the unsearched page; search composes with the open-item pin and the Unread/All filter; the pane's instant client-side filter still narrows the loaded page while the query is in flight.
- [ ] **Step 2: implement.** `("/", "focus_items_search", "Search")` binding gated by `_reader_verb_blocked`; the screen keeps a `_items_search_query` mirror (it already mirrors the pane's — reuse it), debounce timer (the `_request_tree_counts_refresh` shape, :8662-8691) → `_load_items(search=...)`. Empty string passes no predicate at all.
- [ ] **Step 3: run + commit** `feat(watchlists): / focuses search; corpus-wide FTS query path (task-3791)`

### Task 4: All Unread + Today rail nodes

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py` (`TreeScope` :32, roots in `compose()` :201-216, `on_button_pressed` :521+)
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (`_load_tree_data` :1107, `_items_scope_query` :8443, `_tree_scope_label` :1754, `_scoped_loaded_sources` :5085)
- Test: `Tests/Watchlists/test_watchlist_tree.py`, `Tests/Watchlists/test_watchlists_collections_screen.py`, `Tests/UI/test_watchlists_rail_counts_and_scope.py`

- [ ] **Step 1: failing tests.** `TreeScope` accepts `"unread"` and `"today"`; the rail renders both nodes in the smart-feed cluster (order: All sources, Unassigned, **All Unread, Today**, Starred — the smart feeds group together, matching how the spec lists them); clicking posts the matching scope; `_items_scope_query` maps `unread` → `{"status": "new"}` and `today` → `{"since": <local midnight ISO>}` (spy or DB-backed assertions, the phase-2 starred tests' shapes); All Unread's badge reuses `ALL_SOURCES_BUCKET`'s unread (no new query); Today's badge is `get_unread_items_count_since(local_midnight)` inserted as a new `TODAY_BUCKET` in `_load_tree_data` beside STARRED_BUCKET; both refresh through the existing debounced counts path; `_tree_scope_label` names them "All Unread"/"Today"; `_scoped_loaded_sources` treats both like `all`.
- [ ] **Step 2: implement.** Local-midnight boundary from `item_dates`' day-bucket logic (naive-local vs UTC is ALREADY resolved there — reuse, never re-derive in the screen). The `today` scope composes with the Unread/All filter: under Unread it intersects (`status=new` AND `since=…`), under All it is `since` alone over the reader statuses — the screen merges scope kwargs with `_items_status_kwargs()`, and that merge is where the predicates meet; no special-casing in the DB layer.
- [ ] **Step 3: run + commit** `feat(watchlists): All Unread + Today smart feeds in the rail (task-3791)`

### Task 5: `r` — refresh-all with guardrails, aggregated notification, new-items pill

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py` (`check_all` near `check_now` :148)
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (BINDINGS, new action + batch worker; `_request_tree_counts_refresh` end hook)
- Modify: `tldw_chatbook/UI/Watchlists_Modules/article_list.py` (the pill on the header strip)
- Test: `Tests/Watchlists/test_watchlists_collections_screen.py`, `Tests/Watchlists/test_watchlists_backend_controller.py`, `Tests/Watchlists/test_watchlists_article_list.py`

- [ ] **Step 1: failing tests.** `r` on the Read tab launches a check for every ACTIVE, non-paused source and skips paused/inactive ones (stub `check_now` per source, the rail-counts test's stub shape); one aggregated notification at the END ("Checked N sources — M new items", M the ALL_SOURCES_BUCKET unread delta across the batch), never N toasts; an empty/eligible-none roster notifies "nothing to check" and dispatches nothing; a source whose check raises fails the batch softly (named in the aggregate, others continue); the tree counts refresh once at the end; the pill appears with "M new items" when M > 0 and clicking it reloads the items list and dismisses itself; `r` while a batch is in flight is a no-op (guard flag), matching `exclusive=True` worker discipline.
- [ ] **Step 2: implement.** `check_all` on the controller iterates the screen-supplied eligible source ids sequentially through the existing `check_now` chain (the local executor serializes runs already; concurrency is NOT this task's risk to take — guardrails are eligibility + one-batch-at-a-time + soft-failure). The screen worker snapshots the unread count before, launches each, then forces the terminal `_load_tree_data`, computes the delta, notifies once, and pushes the pill. The pill is a compact `Static` with a click handler (Button styled as a pill is NOT used — Buttons on that strip are verbs; this is a notice you can act on, and the `Static` + `on_click` shape keeps the strip's verb grammar clean) — hidden when empty, set from the screen, never from the pane itself.
- [ ] **Step 3: run + commit** `feat(watchlists): r refresh-all with guardrails, aggregate toast, new-items pill (task-3791)`

### Task 6: Help text, pins, docs

- [ ] Help line gains `/` and `r` (decision 031: only implemented actions).
- [ ] End-to-end pin: a hostile-named source + hostile search query through `/` (FTS syntax injection attempt) renders inert and raises nothing; a refresh-all over a roster containing a failing source still aggregates.
- [ ] Full suite: `Tests/Watchlists/` + `Tests/Subscriptions/` + coupled `Tests/UI/` green; ruff clean on every touched file.
- [ ] Task Implementation Notes (decisions: FTS-escape + LIKE fallback rule, pill-as-notice grammar, unread-delta as the honest "new items" number, sequential batch rationale) + backlog `-s Done` via CLI.
- [ ] Commit `docs(watchlists): help text for / and r; task-3791 implementation notes`

## Definition of done (phase 3)

- The spec's done-when holds: daily triage runs from rail + `j/k/m/s/a/u/r//` — pick All Unread, `a` to catch up, `/` to find anything in the corpus, `r` to pull fresh items, Starred/Today as the standing smart feeds.
- All ACs checked; every plan task committed TDD-style; suites green; ruff clean.
- Search never raises on FTS-hostile input and answers (via LIKE) when FTS5 is unavailable.
- Refresh-all never double-launches, never toasts per source, and its pill never yanks the list mid-triage.
