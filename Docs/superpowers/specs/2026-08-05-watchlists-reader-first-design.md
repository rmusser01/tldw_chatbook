# Watchlists Reader-First Re-IA Design

Date: 2026-08-05
Status: Draft (spec review passed 2026-08-05; pending user review)
Amends: [ADR-018](../../../backlog/decisions/018-watchlists-tui-screen.md) — the section IA and pane set, again; a new ADR-042 records this change at plan time
Builds on: [2026-07-25 watchlists console rebuild](2026-07-25-watchlists-console-rebuild-design.md) (implemented), [2026-07-29 noise-not-volume](2026-07-29-watchlists-noise-not-volume-design.md) (implemented), [2026-07-30 briefings](2026-07-30-watchlists-briefings-design.md) (implemented), [ADR-019](../../../backlog/decisions/019-watchlist-scheduler-migration.md), [decision 031](../../../backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md) (keybindings)

## Summary

Make the Watchlists screen usable as a **daily-driver RSS reader**, not only a monitoring console
with a reader bolted on. The Read experience becomes the landing view and takes the
NetNewsWire three-pane shape: a source list with smart feeds and per-feed unread badges, a
date-grouped article list with snippets, and a reading pane with its own action row. The ops
tabs (sources, runs, rules, notifications, artifacts, overview) recede but are untouched.

Nearly everything below is **wiring and presentation on schema that already exists on dev**:
`subscription_items.content`, `content_format`, `content_kind`, `is_flagged` (dormant), an
FTS5 index with triggers and backfill but no query path, a grouped unread-count query, and a
source-scoped `list_items`. No new tables and no new columns are required.

## Starting point — verified 2026-08-05 against `origin/dev @ 06bf63a62`

This spec was written from the dev worktree, not from memory of the rebuild spec. What
actually ships today:

- Five-region workbench (`UI/Watchlists_Modules/watchlists_workbench.py:68`) with collapse /
  solo / persisted layout (`region_layout.py`, `region_layout_store.py`), and a documented
  factory-not-instance recompose pattern (`watchlists_workbench.py:91-153`).
- Left-rail tree (`watchlist_tree.py`) with two permanent roots (All sources, Unassigned),
  watchlists, lazily-loaded sources, and unread counts **on roots and watchlist nodes only**
  (`watchlist_tree.py:433-441` — source nodes have no count).
- Seven-tab strip (`watchlists_tab_strip.py:19-27`): overview, sources, items, runs, rules,
  notifications, artifacts. Reading is tab 3.
- A real content pane (`content_pane.py`) with article and change renderers, mark-read-on-open
  (`watchlists_collections_screen.py:8156-8221`), a Mark-unread escape hatch, and `j`/`k`
  walking of the displayed list (`:9420-9526`).
- Items DataTable with status filter, substring search, open-item pinning into filtered views
  (`items_pane.py:173-209`), and single-cell repaints (`:238`, `:273`).
- Briefings/artifacts suite, notifications inbox, scheduler enabled by default
  (`config.py:2382`, `app.py:4984-4996`).

The verified gaps this spec closes:

1. **The reading loop does not exist.** `_load_items` (`:8109-8133`) never passes the tree
   scope; the FEEDS region renders non-interactive `Static` rows (`:1896-1908`). Picking a
   feed cannot show its articles, even though `list_items(source_id=…)` exists
   (`watchlist_scope_service.py:214-223`).
2. **No smart feeds** (All Unread / Today / Starred) — spec #1's rail mockup nodes were never
   built.
3. **`is_flagged` is schema-only** — column and index exist (`Subscriptions_DB.py:640-641`,
   `:766-769`) with zero readers or writers.
4. **No mark-all-read and no next-unread.** Catch-up means opening every item.
5. **The article list is an ops table**: raw ISO ingest timestamp, no publish date, no
   author/snippet, no date grouping, read state shown as a word in a Status column.
6. **Feed bodies render as raw publisher HTML** in plain text; spec #1's planned
   `content_render.py` never landed. No open-in-browser anywhere on the screen.
7. **FTS5 is maintained but unreachable** — no `MATCH` query exists anywhere; the `/` binding
   promised by spec #1 was not implemented.
8. **No refresh-all** — watchlist-level Check now is deliberately rendered disabled
   (`inspector_pane.py:629-639`).
9. **OPML is flat** both ways (`watchlist_opml_service.py:10-39`); folders are lost on import
   and everything lands in Unassigned.

## Goals

- Land on a reader: three persistent panes shaped like a conventional RSS reader.
- The core loop works: glance at badges → pick a scope → read through its unread → catch up.
- Star articles; find them again. Search the full text of everything fetched.
- Refresh everything in a scope with one action; catch up a scope with one action (reversibly).
- Keep every manage-tab capability and the briefings workflow intact.

## Non-goals

- Syncing read/star state to any server; item status and flags remain local and global across
  watchlists (existing semantics — the shared-status tooltip copy stays).
- "Recently Read" smart feed — there is no `read_at` column, and adding one is not worth it
  for this slice.
- Podcast/enclosure playback in the reading pane — enclosures are listed as links only.
- Nested watchlists, per-feed custom cadences, feed favicons (no image pipeline in the TUI).
- Server-backend reading — items are already local-only; Read shows the honest local-only
  state on Server.
- Rewriting the manage tabs, the briefings suite, or the scheduler.

## Information architecture

**Tabs.** The tab strip reorders to: **Read** (default, key `1`), Sources, Runs, Rules,
Notifications, Artifacts, Overview (`2`–`7`). The items section keeps its internal id
`"items"` — only the display label changes to "Read" — so deep links
(`pending_watchlists_section`, Home/Console handoffs) and existing tests referencing the id
keep working. The screen lands on Read.

**Regions.** In Read, three regions carry the reader:

| Region | Role |
|---|---|
| `LEFT_RAIL` | Source list: smart-feed list above the existing tree |
| `ITEMS` | Article list (replaces the ItemsPane DataTable in Read) |
| `CONTENT` | Reading pane (existing content pane + header/action row) |

The **FEEDS region is removed**: its rows are dead `Static`s and its scope heading moves into
the article-list header. `CENTRE_REGIONS` becomes `(ITEMS, CONTENT)` (`region_layout.py:38`).
`region_layout_store` gains a migration dropping persisted FEEDS collapse state — the Phase-D
CONTENT migration is the in-repo precedent. The **RIGHT_RAIL (inspector) starts collapsed in
Read** (persisted as today) and behaves exactly as now in manage tabs.

**Region gating** (`_hidden_centre_regions`, `:2424-2454`) is unchanged in kind: centre
regions mount only on the Read tab; manage tabs keep their header strip.

**Server backend.** Landing on Read with the Server backend shows the existing honest
local-only notice, plus a "Switch to Local" action. First run with zero sources shows the
first-run add-source guidance in the article list's empty state.

**Rail action bar.** The management buttons (New / Rename / Delete / Add existing / Remove)
hide in Read; a single **+ Add Subscription** button remains at the rail footer. All rail
management actions stay available in the Sources tab and inspector.

## Rail — source list

**Smart feeds.** A small list widget mounted above the tree (not tree pseudo-nodes — tree
nodes assume watchlist/source shapes and `TreeScope` would need invasive extension):

- **All Unread** — count of `status='new'` across visible items.
- **Today** — items whose effective date (see Dates) falls on the local day.
- **Starred** — `is_flagged=1 AND status NOT IN ('ignored','error')`.

Selecting a smart feed clears the tree selection and vice versa: the screen controller owns a
single scope value. The separate widget is what avoids invasive tree surgery; the scope *enum*
still gains a smart-feed variant (e.g. `TreeScope.smart_feed(name)`) — adding an enum value is
cheap, adding tree node shapes is not. Counts come from one grouped query alongside the
existing watchlist-counts query.

**Tree upgrades.** Source nodes gain unread badges via a new grouped per-source counts query
(`get_source_item_counts`, mirroring `get_watchlist_item_counts` at `Subscriptions_DB.py:1041`),
plus an error marker for sources whose latest run failed. The collapsed-rail header shows the
total unread count — closing spec #1's pending "collapsed headers show counts" item.

## Article list

New `UI/Watchlists_Modules/article_list.py` (ListView-based), replacing the ItemsPane
DataTable **in Read only**:

- **Rows**: source name + relative effective time on line 1; title on line 2 (**bold** when
  unread, regular when read); 1–2 line snippet from `content` (tags/entities stripped) on
  line 3; leading unread dot; trailing star glyph when flagged; a queued marker when
  `queued_for_briefing` is set. Ingested items render read-styled with a small marker.
- **Date-group headers**: Today / Yesterday / locale date, computed in Python over the
  displayed rows.
- **Sort**: effective date descending (publish date, falling back to ingest time).
- **Filter**: Unread / All toggle in the header. All = `new + reviewed + ingested`. Ignored
  items stay hidden (the user hid them); error items stay in Runs where they belong.
- **Header**: scope name, unread count, the filter toggle, sort control, Mark All Read,
  Refresh, search field — one row of compact controls, echoing the reference layout's toolbar.
- **Pagination**: keep the 100-row pages; auto-append the next page when the cursor reaches
  the end, so `j` walks the whole scope. `displayed_items()` / `select_and_reveal()`
  (`items_pane.py:351-408`) carry over.
- **Open-item pinning**: replicates `items_pane.py:173-209` — in the Unread filter, the item
  being read stays visible until the next scope reload, so mark-read-on-open never yanks the
  current row out from under the cursor.
- **Stable snapshot**: the list is a point-in-time snapshot per scope load. Items arriving
  from background checks do not reshuffle rows under the cursor; the header shows an
  "N new items" pill that reloads on click (badges still update live via the existing 0.6 s
  debounce, `:8221-8258`).
- **Empty states**: zero sources → first-run CTA; scope with no unread in Unread filter →
  "✓ All caught up".

## Reading pane

The existing content pane (`content_pane.py`) keeps both renderers and gains:

- **Header**: source eyebrow, title, byline (author · effective date · source), categories
  when present, enclosure links when `enclosures` JSON is present (already parsed,
  `monitoring_engine.py:920-930`).
- **Action row** (compact buttons; the three primary actions carry keys): Star (`s`), Mark
  unread (`m`), Open in browser (`o`), plus button-only Ingest and Queue/Unqueue for briefing
  — no keys for those two; they stay pointer and inspector actions. All five call the **same
  helpers the inspector uses** — item actions are extracted into a shared module (precedent:
  `watchlists_console_handoff.py`) so the two surfaces cannot drift.
- **Body via `Subscriptions/content_render.py`** (new): HTML→text with a dependency-free
  stdlib (`html.parser`) fallback, using `html2text`/`trafilatura` when available
  (`optional_deps` pattern). All remote markup is escaped (`escape_markup` precedent) and
  hyperlinks render as text (`hyperlinks=False` as today). Conversion failure falls back to
  the raw text body — never a blank pane.
- **Footer**: "N of M" position within the displayed list and a Next Unread control.

## Global actions

**Mark All Read (`a`).** One transactional bulk `UPDATE … SET status='reviewed' WHERE
status='new' AND <scope> RETURNING id` (`RETURNING` precedent: `item_persist.py:138`). Never
touches ingested/ignored/error. No confirm modal — catch-up must be fast — but the affected
id set is kept as the session's **undo batch**: `u` restores exactly those ids to `'new'`.
One batch deep; superseded by the next bulk op. Badges refresh via the existing debounce.

**Refresh (`r`).** Checks every active, non-paused source in the current scope — this finally
implements the deliberately-disabled bulk check (`inspector_pane.py:629-639`). Guardrails:
sources with an active run are skipped; concurrency capped (default 4); each source runs
through the existing `launch_run`/`execute_run` path so Runs-tab history is unchanged;
completion produces **one aggregated notification** ("Checked 42 sources: 12 new items, 1
failure") — never per-source toasts.

**Search (`/`).** Focuses the header search field. The query is sanitized into a safe FTS5
`MATCH` (terms double-quoted, operators stripped) against `subscription_items_fts`
(title/content/author, `Subscriptions_DB.py:918-926`), scope-filtered, with `snippet()` for
row previews. Results replace the list under a "Search results in {scope}" header; Esc clears
back to the scope. FTS failure (missing/corrupt table) falls back to the existing substring
filter with a notice.

## Data and service layer

No new tables, no new columns. New queries and service params only:

- `get_source_item_counts()` — per-source `{total, unread}` + latest-run-failed marker, one
  grouped query mirroring `get_watchlist_item_counts` (`Subscriptions_DB.py:1041`).
- `get_smart_feed_counts()` — All Unread / Today / Starred counts (Today via the date helper
  below).
- `list_items(...)` (`watchlist_scope_service.py:214-223`) gains `watchlist_id` (join
  `watchlist_sources`), `is_flagged`, and `statuses` (multi-value, default reader set)
  filters; `source_id`, single `status`, `limit`/`offset` exist. Local-only policy unchanged.
- `mark_all_read(scope) -> list[int]` — bulk update returning affected ids (undo batch).
- `set_item_flagged(item_id, flagged: bool)` — wires the dormant column and its index
  (`Subscriptions_DB.py:767`).
- `search_items(query, scope)` — FTS path with substring fallback.
- `check_all(scope)` — concurrency-limited batch over eligible sources.
- One new index on `subscription_items(published_date)` if phase-1 benchmarks show the sort
  needs it; decided at plan time with a measured query.

Upserts never clobber user state (verified): `persist_subscription_item` preserves
reviewed/ignored/ingested across re-fetches (`item_persist.py:132-136`) and does not write
`is_flagged` at all, so flags survive re-fetches.

## Dates

`published_date` is stored as ISO-8601 but **mixed naive/aware** — `_parse_date`
(`monitoring_engine.py:1064-1083`) returns `dt.isoformat()`, timezone-less for several feed
formats, while URL monitors store aware UTC (`:1387`). A new shared helper
(`Subscriptions/item_dates.py`):

- parses defensively (`fromisoformat` + the known feed formats), treating naive as UTC;
- exposes the **effective date**: `published_date`, falling back to `created_at`;
- renders relative times ("9:41 AM", "Yesterday") and local-day buckets for group headers.

Group headers and the position footer compute over displayed rows in Python. The Today smart
feed uses a generous SQL prefilter (last 48 h) refined in Python — exact local-day semantics
without trusting stored tz strings. Future-dated items (bad feed clocks) bucket into Today.

## Keybindings

Per decision 031 (single-letter htop-style bindings are allowed; footer hints advertise only
implemented actions). Read tab:

| Key | Action |
|---|---|
| `j` / `k` | next / previous article (exists, `:9420-9526`) |
| `space` | next unread — **bound on the article-list widget only**: Textual's `Tree` binds `space` → `toggle_node`, and a screen-level binding would break the rail |
| `m` | toggle read/unread on highlighted or open article |
| `s` | toggle star |
| `o` | open in browser |
| `a` | mark all read (scope) |
| `u` | undo last mark-all-read batch |
| `r` | refresh scope |
| `/` | focus search |
| `n` | new subscription (exists) |
| `d` | ignore item (exists) |
| `c` / `p` | per-source Check now / Preview (exist; unchanged — manage contexts. `r` is the scope-level complement, not a replacement) |
| `1`–`7` | tabs in new order |
| `z` / `Z` / `[` / `]` | region collapse / solo / rail toggles (exist) |
| `?` | help (exists; text updated) |

`m`, `s`, `o`, `a`, `u`, `r`, `/` are new and conflict-free against the current screen map
(`:385-411`) and the globals reserved by decision 031.

## State preservation across recompose

`WatchlistsWorkbench.region_layout` is `recompose=True` (`watchlists_workbench.py:89`): every
collapse/solo rebuilds all regions from factories. The article list's cursor (by item id),
scroll offset, open item, loaded page count, and search state live in the screen controller,
not the widget, and are restored after rebuild. Selection-survival-across-recompose tests
already exist for the current panes; equivalent coverage is required for the new list.

## Error handling and edge cases

- Failing source → error marker on its rail row; detail stays in Runs (unchanged).
- Empty scope / zero sources → first-run CTA or "✓ All caught up" (existing empty-state
  patterns).
- HTML→text failure → raw text body, never blank.
- Mark-all-read and refresh-all are scoped, transactional/batched, and reflected in badges
  within the existing debounce; undo restores the exact id set.
- FTS unavailable → substring fallback with a notice.
- Region-layout persistence containing the removed FEEDS region → store migration (Phase-D
  precedent).
- Background check arrivals → "N new items" pill; no mid-session reshuffle.
- OPML (phase 4): folder outlines map to watchlists by name — match existing → add
  membership; otherwise create. Export maps watchlists to folders; Unassigned stays flat.

## Security

- All remote content rendered escaped; hyperlinks as text (existing precedent).
- FTS queries sanitized before `MATCH`.
- Fetch-side SSRF/XXE protections unchanged (tasks 328/329 territory, already shipped).
- `o` opens the item URL in the system browser via `webbrowser` — URLs come from feeds;
  no shell invocation.

## Testing

- Unit: scope filter builder, per-source and smart-feed counts, `mark_all_read`
  scope/transaction/undo semantics, `set_item_flagged`, FTS sanitize + query + fallback,
  `item_dates` (naive/aware/future/missing), `content_render` (incl. hostile markup),
  `check_all` batching/skip rules, OPML folder mapping.
- Widget tests (`run_test`): Read-default landing and tab order, smart-feed selection vs tree
  selection (single scope), badge rendering, article-list grouping/pinning/lazy paging,
  mark-read-on-open, all new bindings (incl. `space` scoping vs the tree), search flow,
  refresh-all progress + aggregated notification, undo batch, recompose state survival,
  region-store migration.
- Existing `Tests/Watchlists/` updated for tab order, region gating without FEEDS, and the
  retired DataTable (ItemsPane itself remains for reference until removal is verified safe).

## Phasing

**Phase 1 — the reading loop.** Scoped article list (tree + status plumbing), per-feed badges,
collapsed-rail unread count, mark-all-read + undo, next-unread, `m`, Read-default landing,
tab reorder, inspector collapsed in Read, region-store migration, state preservation.
*Done when*: picking any node scopes the list; catch-up is two keys (`a`, maybe `u`); place
survives region toggles; existing suite green after updates.

**Phase 2 — list and reader quality.** Reader rows (snippet/relative date/groups/unread
bold/star/ingested + queued markers), `s` + Starred smart feed, `content_render.py`, `o`,
reading-pane action row via shared helpers, position footer, date helper.
*Done when*: a feed reads like the reference layout; flags persist across re-fetch; hostile
HTML renders safely as text.

**Phase 3 — smart feeds, search, refresh-all.** All Unread + Today nodes, `/` FTS path with
fallback, `check_all` with guardrails and aggregated notification, "N new items" pill.
*Done when*: daily triage runs entirely from the rail + `j/k/m/s/a/u/r//`.

**Phase 4 — round-trip and polish.** OPML folder mapping both ways; tasks 2308 (absorbed by
the date helper), 2310, 2312, 2313.

## ADR

ADR required: **yes** — new `backlog/decisions/042-watchlists-reader-first-ia.md`, amending
ADR-018's pane set/IA for the second time and recording: Read-first landing, FEEDS region
removal, ops-tab recession, the no-new-schema data approach, the undo-batch choice over a
confirm modal, and the widget-level `space` binding. Created before implementation begins and
linked from the plan and the backlog task(s).
