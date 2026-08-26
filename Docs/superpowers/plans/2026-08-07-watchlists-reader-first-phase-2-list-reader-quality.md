# Watchlists Reader-First Re-IA — Phase 2: List and Reader Quality — Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking. Implement task-by-task, TDD throughout: failing test first, then implementation, one conventional commit per task.

**Goal:** Make the Read tab's article list and reader look and behave like the reference feed-reader layout (NetNewsWire): multi-line reader rows with date-group headers, a star that persists, a Starred smart feed in the rail, an action row on the reader that shares its verbs with the Inspector, `o` to open in the browser, and a position footer.

**Architecture:** All work happens in the worktree `.worktrees/watchlists-reader-first`, branch `feat/watchlists-reader-first-p2` (off `origin/dev @ cc0001ff3`, which includes phase 1 and TASK-3071). New filters are threaded through the phase-1 pass-through chain (`Subscriptions_DB.get_new_items` → `LocalWatchlistsService.list_items` → `WatchlistScopeService.list_items` → backend controller `**kwargs`). No schema changes: `is_flagged` and its index are already resident (`Subscriptions_DB.py:640-641`, `:768`).

**Spec:** `Docs/superpowers/specs/2026-08-05-watchlists-reader-first-design.md` — "Article list", "Reading pane", "Dates", "Keybindings" sections; phasing line "Phase 2 — list and reader quality".

**ADR required:** no (already exists)
**ADR path:** `backlog/decisions/042-watchlists-reader-first-ia.md`
**Reason:** ADR-042 covers the whole re-IA including the phase-2 surface (reader rows, starred state, shared action verbs, new keys). Phase 2 is a direct implementation of it.

**Backlog task:** TASK-3072 (filed 2026-08-07).

**Tech Stack:** Python 3.11+, Textual (ListView), SQLite (stdlib `sqlite3`), pytest + `textual.pilot`.

**Conventions:**
- Run everything from the worktree root: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/watchlists-reader-first`
- Tests: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python3 -m pytest Tests/Watchlists/ -x -q` (the main-checkout venv provides deps; the `timeout` command is NOT available).
- DB tests use real SQLite in-memory (repo convention).
- Screen under change: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (10,135 lines — navigate by the line refs below; exact as of `cc0001ff3`).
- Commits: `type(scope): summary` conventional style; reference `task-3072`.

**Survey results the plan is built on (all verified 2026-08-07 against `origin/dev @ cc0001ff3`):**

- `items_pane.py` (547 lines) is the DataTable list phase 2 replaces **in Read only**. Its sole construction site is the screen's region factory at `:1985`. Carryovers the new widget must preserve: `displayed_items()` / `select_and_reveal()` (`:482-539`), the `_rendered_items` snapshot authority (`:152-163`, `:499-501`), open-item pinning in `_filtered_items()` (`:257-293`), the in-place single-cell repaint pattern (`update_item_status_cell` :369, `update_item_queued_cell` :404), the `ItemsFilterChanged` mirror message (`:46-63`), `ItemSelected` / `RefreshItemsRequested` / `NextUnreadRequested` messages, the search box's `select_on_focus=False` + recompose focus restore (`:183-198`, `:311-356`), and the pane-bound `space` binding (`:69`, `:545-547`).
- `content_pane.py` already has the `#content-actions` strip (`:350-389`) and posts `UnreadToggleRequested`; the Inspector posts `IngestRequested` (`inspector_pane.py:108`) and `ToggleBriefingQueueRequested` (`:132`). The action row shares verbs by posting these same screen-handled messages — the no-drift mechanism the spec asks a "shared module" for, achieved at the message layer (screen handlers `:9219`, `:9260`).
- The spec's `Subscriptions/content_render.py` is **already satisfied** by `Subscriptions/html_text.py` (TASK-2307): dependency-free stdlib `HTMLParser`, `readable_body_text` (`:354`) used by `render_article` today, `html2text` noted as an optional extra that is not required (`:18-23`). Phase 2 records this reconciliation and adds only a snippet helper if the row code needs one. No new renderer module.
- `humane_time.py` is the house timestamp style (TASK-2308). The spec's `item_dates.py` adds what it lacks: **effective date** (`published_date` falling back to `created_at`), **relative rendering** for rows, and **local-day buckets** for group headers. It reuses `humane_time`'s parse (naive = UTC) rather than growing a second parser.
- Scope plumbing: `TreeScope` (`watchlist_tree.py:29-34`) is a `Literal["all","unassigned","watchlist","source"]` — phase 2 adds `"starred"`. The screen maps scope → `list_items` kwargs in `_items_scope_query` (`:8412-8428`); status in `_items_status_query` (`:8386-8410`); fetch in `_load_items` (`:8484-8509`, page 100).
- `get_new_items` (`Subscriptions_DB.py:1793`) builds predicate fragments with bound params (`:1859-1867`); `set_item_briefing_queued` (`:2099`) is the exact precedent for `set_item_flagged`. `normalize_watchlist_item` (`watchlist_normalizers.py:549`) needs the `is_flagged` bool key (precedent `queued_for_briefing` at `:588-592`).
- Screen `BINDINGS` (`:392-424`): `s`, `o`, `/`, `r` are all free. Phase 2 adds `s` and `o` only (`/` and `r` are phase 3).

---

### Task 1: Task bookkeeping + docs commit

**Files:**
- Create: this plan
- Modify: `backlog/tasks/task-3072 - Watchlists-reader-first-re-IA-phase-2-list-and-reader-quality.md` (status → In Progress, plan reference)

- [ ] **Step 1: Move TASK-3072 to In Progress with the plan reference**

```bash
npx --yes backlog.md task edit 3072 -s "In Progress" --plan "Execute Docs/superpowers/plans/2026-08-07-watchlists-reader-first-phase-2-list-reader-quality.md

ADR required: no (already exists)
ADR path: backlog/decisions/042-watchlists-reader-first-ia.md
Reason: ADR-042 covers the re-IA; phase 2 is a direct implementation of it."
```

- [ ] **Step 2: Commit the docs**

```bash
git add Docs/superpowers/plans/2026-08-07-watchlists-reader-first-phase-2-list-reader-quality.md backlog/tasks/
git commit -m "docs(watchlists): phase 2 plan — list and reader quality (task-3072)"
```

---

### Task 2: `Subscriptions/item_dates.py` — effective date, relative time, day buckets

**Files:**
- Create: `tldw_chatbook/Subscriptions/item_dates.py`
- Test: `Tests/Subscriptions/test_item_dates.py`

The date foundation every other task renders with. The stored `published_date` is mixed naive/aware ISO-8601 (the spec's "Dates" section); naive is attached to UTC, matching `humane_time`'s documented house rule.

- [ ] **Step 1: Write the failing tests**

Cover: aware and naive ISO strings parse to the same instant (naive = UTC); `effective_date(item)` prefers `published_date`, falls back to `created_at`, returns `None` when both are missing; `relative_day(dt, now)` returns the bucket the row header wants (`"Today"`, `"Yesterday"`, else a locale date string); future-dated items (bad feed clocks) bucket into Today; unparseable input degrades (bucket falls back to ingest time or `"Unknown date"`, never raises); `relative_time(dt, now)` renders same-day times as clock time ("9:41 AM") and older days as the day label.

- [ ] **Step 2: Implement `item_dates.py`**

API: `parse_stored_datetime(value) -> datetime | None` (delegate the known formats to `humane_time`'s parser if it exposes one — read it first; otherwise `fromisoformat` + the feed formats, naive → `timezone.utc`), `effective_date(item) -> datetime | None`, `relative_time(dt, *, now) -> str`, `day_bucket(dt, *, now) -> str`. Module docstring records: one parser rule (naive = UTC), why effective date exists (TASK-2308 showed ingest time under a "Published" heading once already), and that bucketing is done in Python over displayed rows, not in SQL (mixed stored tz strings cannot be trusted to SQL date functions — the spec's Dates section).

- [ ] **Step 3: Run the tests, commit**

`feat(watchlists): item_dates helper — effective date, relative time, day buckets (task-3072)`

---

### Task 3: `is_flagged` plumbing — DB setter, query predicate, normalization

**Files:**
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py` (`set_item_flagged` after `set_item_briefing_queued` :2099; `is_flagged` predicate in `get_new_items` :1859-1867; `get_flagged_items_count` near `get_source_item_counts` :1105)
- Modify: `tldw_chatbook/Subscriptions/watchlist_normalizers.py` (`:549`)
- Modify: `tldw_chatbook/Subscriptions/local_watchlists_service.py` (`list_items` :355), `tldw_chatbook/Subscriptions/watchlist_scope_service.py` (`list_items` :215)
- Test: co-locate DB tests with the existing `get_new_items` tests (`grep -rl "get_new_items" Tests/`)

- [ ] **Step 1: Write the failing tests**

`set_item_flagged` flips the column and is readable via `get_new_items(is_flagged=True)`; the flag is global (same item in several watchlists — ADR-018's note on `queued_for_briefing` applies verbatim); **flags persist across re-fetch** (upsert an existing item via `persist_subscription_item` — it must not write `is_flagged`; the spec verified this reads-only claim, now it is pinned by test); `get_flagged_items_count()` counts flagged rows across all sources; the normalized item dict carries `is_flagged` as a real `bool` (the `queued_for_briefing` coercion precedent, `watchlist_normalizers.py:588-592`).

- [ ] **Step 2: Implement**

- `Subscriptions_DB.set_item_flagged(item_id, flagged)` — same transaction shape as `set_item_flagged`'s sibling :2099.
- `get_new_items(..., is_flagged: bool | None = None)` — one more predicate fragment (`i.is_flagged = ?`), bound param, documented in the docstring's Args.
- `get_flagged_items_count() -> int` — single `SELECT COUNT(*) ... WHERE is_flagged = 1` (status-agnostic: a starred item stays starred when read).
- Thread `is_flagged` through `LocalWatchlistsService.list_items` and `WatchlistScopeService.list_items` as another falsey-means-no-filter kwarg (the `watchlist_id` precedent, TASK-2513).
- `normalize_watchlist_item` gains `"is_flagged": bool(row.get("is_flagged"))`.

- [ ] **Step 3: Run the tests, commit**

`feat(watchlists): is_flagged plumbing — setter, query predicate, normalization (task-3072)`

---

### Task 4: `Subscriptions/item_dates`-ordered listing + snippet helper

**Files:**
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py` (`get_new_items` ordering, :1844-1846 note says `created_at` desc today)
- Modify: `tldw_chatbook/Subscriptions/html_text.py` (snippet helper)
- Test: same DB test file; `Tests/Subscriptions/test_html_text.py` (find with `grep -rl "readable_body_text" Tests/`)

- [ ] **Step 1: Write the failing tests**

Ordering: items with a `published_date` sort by it desc; items without fall back to `created_at`; a stale-published item fetched recently sorts below a fresh-published one. Implement as `ORDER BY COALESCE(i.published_date, i.created_at) DESC` — **no new index**: the sort runs over the scope-filtered, page-bounded set (limit+offset ≤ a few hundred rows); record the EXPLAIN measurement in the commit message (the spec leaves the index conditional on measurement). Snippet: `body_snippet(content, *, max_chars=160)` collapses whitespace, strips tags via the existing extractor, truncates on a word boundary with an ellipsis; hostile input (script tags, control bytes, markup-shaped text) comes out as inert plain text.

- [ ] **Step 2: Implement**

- [ ] **Step 3: Run the tests, commit**

`feat(watchlists): effective-date ordering + body snippet helper (task-3072)`

---

### Task 5: `article_list.py` — the reader rows widget

**Files:**
- Create: `tldw_chatbook/UI/Watchlists_Modules/article_list.py`
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (construction site :1985; nothing else yet)
- Test: `Tests/Watchlists/test_watchlists_article_list.py`

The phase-2 centrepiece: a ListView-based replacement for ItemsPane's DataTable **in the Read section only**. Rows: line 1 = source name + relative effective time; line 2 = title (**bold** when `status == "new"`); line 3 = 1–2 line snippet; leading unread dot; trailing star glyph when `is_flagged`; queued marker (the existing `●`, `items_pane.py:74`); ingested items render read-styled with a small marker. Date-group headers (Today / Yesterday / locale date) computed in Python over the displayed rows via `item_dates.day_bucket`.

- [ ] **Step 1: Write the failing tests**

Row rendering: unread row is bold with the dot, read row is not; starred row shows the star; queued row shows `●`; ingested row shows its marker and read styling; a hostile title (`[bold red]x[/]`, control bytes) renders literally (the `escape_markup` boundary rule `items_pane.py:226-233` states — ListView row content must follow it wherever a markup-parsing widget renders item text); group headers appear in effective-date-descending order and future-dated items land under Today; **headers are not selectable** and `j`/`k`-style cursor movement skips them; selection posts the same `ItemSelected`-shaped message the screen already handles (`:8511`).

- [ ] **Step 2: Implement the widget**

Carry over, deliberately, from `items_pane.py` (read each one's docstring before re-deriving it): the `_rendered_items` rendered-sequence authority; `displayed_items()` / `select_and_reveal()`; open-item pinning in the filter; `ItemsFilterChanged` mirroring; the search box's `select_on_focus=False` + `recompose()` focus restore; the pane-bound `space` binding; in-place row repaint methods (`update_item_status_row` / `update_item_queued_row` / `update_item_flagged_row` — ListView equivalent of the single-cell repaints; a row repaint replaces the one row widget, never recomposes the list). Keep the row→item mapping in a parallel entries list so group headers never enter `displayed_items()`.

Filter becomes the spec's **Unread / All toggle**: Unread = `status="new"`; All = `statuses=["new","reviewed","ingested"]`. Ignored stays hidden (the user hid it); error stays in Runs. Wire the two values through `ItemsFilterChanged` so the screen's `_items_status_query`/`_load_items` keep their push-the-filter-into-the-query guarantee (`:8386-8410`).

The toolbar keeps Refresh + search; the scope name / unread count live in the region header (existing phase-1 suffix) — do not duplicate them.

- [ ] **Step 3: Swap the construction site**

`:1985` builds `ArticleListPane(id="watchlists-items-pane")` (same DOM id: screen queries and CSS keep working). `ItemsPane` stays in the tree untouched — it is not deleted in this phase; the swap is one line plus imports, revertible.

- [ ] **Step 4: Run the pane + screen suites, commit**

`feat(watchlists): article_list reader rows with date-group headers (task-3072)`

---

### Task 6: Starred smart feed in the rail

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/watchlist_tree.py` (`TreeScope` :29, root nodes :377, badges)
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (`_items_scope_query` :8412)
- Test: `Tests/Watchlists/test_watchlists_tree.py` (find with `grep -rl "_root_node\|TreeScope" Tests/Watchlists/ | head -3`)

- [ ] **Step 1: Write the failing tests**

`TreeScope` accepts `kind="starred"`; the tree renders a "Starred" root node with the `get_flagged_items_count` badge; selecting it posts `TreeScopeChanged(starred)`; the screen's `_items_scope_query` maps it to `{"is_flagged": True}`; the badge refreshes through the existing debounced counts path.

- [ ] **Step 2: Implement**

Add `"starred"` to the `TreeScope` Literal, a `_root_node("starred", "Starred", count)` above the watchlist nodes (star glyph label, app-controlled constant — the `_QUEUED_GLYPH` markup rule), and the scope mapping. The node has no children and no expand affordance.

- [ ] **Step 3: Run the tests, commit**

`feat(watchlists): Starred smart feed in the rail (task-3072)`

---

### Task 7: `s` — star toggle on highlighted or open item

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (BINDINGS :392-424, new action + handler near `toggle_read_selected`)
- Modify: `tldw_chatbook/UI/Watchlists_Modules/content_pane.py` (Star button on `#content-actions` :350)
- Test: screen-level, in `Tests/Watchlists/test_watchlists_collections_screen.py`

- [ ] **Step 1: Write the failing tests**

`s` with a highlighted/open item flips `is_flagged` through the service, repaints the row's star in place (Task 5's method), refreshes the Starred badge, and never recomposes the list; the reader's Star button reflects the open item's state and posts the same path; `s` with no item selected is a no-op.

- [ ] **Step 2: Implement**

`("s", "toggle_star_selected", "Star")` binding; the handler resolves the highlighted-or-open item exactly like `toggle_read_selected` does, calls the service, then the row repaint + badge refresh. The content-pane button posts a new `StarToggleRequested(item)` message; one screen handler serves both (the same message-layer sharing the Inspector's verbs use).

- [ ] **Step 3: Run the tests, commit**

`feat(watchlists): s toggles star; reader Star button (task-3072)`

---

### Task 8: `o` — open in browser + reader action-row completion

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (BINDINGS, handler)
- Modify: `tldw_chatbook/UI/Watchlists_Modules/content_pane.py` (Open in browser, Ingest, Queue/Unqueue buttons on `#content-actions`)
- Test: screen-level

- [ ] **Step 1: Write the failing tests**

`o` opens the highlighted/open item's `url` via `webbrowser.open` — **http/https only**: a `javascript:`/`file:`/empty URL is refused with a notification, never passed to the OS (this is a remote-derived string reaching an OS primitive; validate at this boundary per `input_validation` conventions). The reader's Open button takes the same path. The Ingest and Queue/Unqueue buttons post the Inspector's own `IngestRequested` / `ToggleBriefingQueueRequested` and reflect state (`Queue` ↔ `Unqueue` label from `queued_for_briefing`, the `_queue_briefing_button` precedent `inspector_pane.py:449-463`). Buttons only — no new keys for those two (spec).

- [ ] **Step 2: Implement**

- [ ] **Step 3: Run the tests, commit**

`feat(watchlists): o opens in browser; reader action row shares inspector verbs (task-3072)`

---

### Task 9: Position footer in the reader

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/content_pane.py`
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (`handle_item_selected` :8511 and the navigation methods)
- Test: screen-level

- [ ] **Step 1: Write the failing tests**

With an item open, the footer reads "N of M" where M is `len(displayed_items())` and N is the open item's 1-based index in it; `j` advances both the reader and the footer; a Next Unread control in the footer posts the pane's existing `NextUnreadRequested`; with nothing open the footer is empty (not "0 of 0").

- [ ] **Step 2: Implement**

The pane holds no list state: the screen computes position on selection/navigation and pushes it into a `position` reactive (the `_selected_content_item` re-seed pattern, `:8521-8525`), so a rebuilt pane re-renders the same footer.

- [ ] **Step 3: Run the tests, commit**

`feat(watchlists): reader position footer with Next Unread (task-3072)`

---

### Task 10: Hostile-HTML pins, help text, docs

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (help overlay text — find the existing `?` content; add `s`/`o`)
- Test: extend the hostile-input cases if any gap remains
- Modify: backlog task-3072 (Implementation Notes)

- [ ] **Step 1: The hostile-HTML suite covers the AC end-to-end**

One test that stars, queues, and re-renders an item whose title/content carry `<script>`, `[bold red]`, and control bytes — row, snippet, and reader body all render inert text (the AC's "hostile HTML renders safely as text"), and the flags survive a re-persist (Task 3's pin).

- [ ] **Step 2: Help + footer hints advertise only implemented actions** (decision 031).

- [ ] **Step 3: Full suite + lint**

`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python3 -m pytest Tests/Watchlists/ Tests/Subscriptions/ -q` and `ruff check` on every touched file.

- [ ] **Step 4: Task Implementation Notes + commit**

`docs(watchlists): help text for s/o; task-3072 implementation notes`

---

## Definition of done (phase 2)

- All task-3072 ACs checked; every plan task committed TDD-style.
- `Tests/Watchlists/` + `Tests/Subscriptions/` green; ruff clean.
- A feed reads like the reference layout: date-grouped multi-line rows, bold-unread, working star with a rail feed, reader action row, position footer.
- Flags persist across re-fetch (pinned by test).
- Hostile HTML renders as inert text in rows, snippets, and body (pinned by test).
