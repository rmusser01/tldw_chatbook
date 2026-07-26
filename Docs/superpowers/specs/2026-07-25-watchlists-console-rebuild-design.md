# Watchlists Console-Style Rebuild Design

Date: 2026-07-25
Status: Draft (pending spec review)
Supersedes (partially): [ADR-018](../../../backlog/decisions/018-watchlists-tui-screen.md) — the section IA and pane set
Related: [ADR-019](../../../backlog/decisions/019-watchlist-scheduler-migration.md), [2026-07-18 watchlists TUI screen design](2026-07-18-watchlists-tui-screen-design.md)

## Summary

Rebuild the Watchlists screen as an information-dense, Console-styled feed reader and source manager.
A **watchlist** becomes a first-class entity: a named bundle of sources (RSS feeds and scraped
sites) that is the unit of organization, checking, and — in a later slice — briefing and podcast
generation.

The screen's centre becomes a three-level drill-down (feeds → items → content) with every region
independently collapsible. The service layer beneath the UI does not move.

This is **spec #1 of two**. Spec #2 covers artifact generation (briefings, 2-speaker podcasts) and
its scheduled delivery. This spec designs the space those features occupy but implements none of it.

## Goals

- Rebuild the screen at Console-level information density, reusing `DestinationModeStrip` and the
  `$ds-*` token set.

**`DestinationWorkbench` is deliberately not reused.** It is a fixed `Horizontal` of equal-width
(`width: 1fr`) panes composed once from a frozen tuple set at construction, with no collapse, no
resize, and no vertical stacking. That is the opposite of what this screen needs — two rails around
a vertically-stacked, independently collapsible centre. A purpose-built `watchlists_workbench.py`
container is used instead. If the collapse behaviour proves generally useful, it graduates into the
shared widget later; it is not worth generalising ahead of a second consumer.
- Introduce the watchlist bundle entity with many-to-many source membership.
- Provide a real feed-reader path: collection → feeds → items → readable content.
- Render scraped sites as first-class reader content, not second-class rows.
- Give items durable body text and full-text search, neither of which exists today.
- Keep every existing `Subscriptions/` service, the scheduler, and the route contract intact.

## Non-goals

- Briefing, podcast, or any artifact generation (spec #2).
- Artifact storage or rendering — artifacts live on the Artifacts screen; Watchlists links to them.
- Recurring delivery scheduling — that lives on the Schedules screen.
- Nested watchlists. Server groups support `parent_group_id`; local watchlists are flat.
- Syncing local watchlists to server groups.
- Reviving `briefing_generator.py`, `aggregation_engine.py`, `distribution_manager.py`,
  `export_manager.py`, `rss_feed_generator.py`, or `recursive_summarizer.py`. All are currently
  unimported outside `Subscriptions/` and untested; their fate is spec #2's decision.

## Entity model

| Entity | Meaning | Storage |
|---|---|---|
| **Source** | One RSS feed or scraped site | `subscriptions` (existing) |
| **Watchlist** | Named bundle of sources; unit of organization and checking | `watchlists` (new) |
| **Membership** | Many-to-many source ↔ watchlist | `watchlist_sources` (new) |
| **Item** | One fetched entry or detected site change | `subscription_items` (existing, extended) |
| **Run** | One check execution, per source, grouped into batches | `local_watchlist_runs` (existing, extended) |

A source may belong to any number of watchlists, matching the server's many-to-many `group_ids`.

### Schema changes

All changes go through the existing idempotent `_ensure_watchlists_schema` helper in
`Subscriptions_DB.py` that ADR-018 established. No new migration machinery.

```sql
CREATE TABLE IF NOT EXISTS watchlists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    tags TEXT,
    is_active BOOLEAN DEFAULT 1,
    sort_order INTEGER DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS watchlist_sources (
    watchlist_id    INTEGER NOT NULL REFERENCES watchlists(id)     ON DELETE CASCADE,
    subscription_id INTEGER NOT NULL REFERENCES subscriptions(id)  ON DELETE CASCADE,
    added_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (watchlist_id, subscription_id)
);
CREATE INDEX IF NOT EXISTS idx_watchlist_sources_subscription
    ON watchlist_sources(subscription_id);
```

**Foreign-key enforcement must be switched on first; it is currently off at runtime.** The
`PRAGMA foreign_keys = ON` at `Subscriptions_DB.py:82` runs inside `_initialize_schema`'s
`with closing(self._get_connection())` block, and that connection is closed immediately after. The
pragma is per-connection, and `BaseDB._get_connection` (`base_db.py:104-110`) sets only
`row_factory` — `Subscriptions_DB` never overrides it. Sibling databases do set it per connection
(`ChaChaNotes_DB.py:2678`, `Client_Media_DB_v2.py:691`); this one is the exception.

Consequence beyond this feature: `subscription_items` already declares
`FOREIGN KEY (subscription_id) REFERENCES subscriptions(id) ON DELETE CASCADE`, which has never
fired. Deleting a subscription has been silently orphaning its items. Implementation enables
enforcement in its own commit before any of this schema lands. Cleaning up already-orphaned rows is
deliberately **out of scope** — deleting user data deserves its own spec and a backup story.

`watchlists.tags` is **comma-joined**, matching how `subscriptions.tags` is stored
(`Subscriptions_DB.py:422` does `",".join(tags)`) — not JSON. This inherits the existing wart that
a tag containing a comma will split; consistency with the sibling column is worth more than fixing
it unilaterally here.

**`watchlists.name` is deliberately not `UNIQUE`.** Uniqueness is enforced case-insensitively in
`WatchlistBundleService` with auto-suffixing (`Unsorted (2)`). A raw SQL constraint would raise
mid-migration on case-variant folder values or OPML re-imports.

Added to `subscription_items`:

| Column | Type | Purpose |
|---|---|---|
| `content` | `TEXT` | Renderable body: article text for feed items, diff text for site changes |
| `content_kind` | `TEXT` | `article` \| `change` — selects which renderer the Content pane uses |
| `content_format` | `TEXT` | `text` \| `markdown` \| `diff` — how to format within that renderer |
| `is_flagged` | `BOOLEAN DEFAULT 0` | User flag, orthogonal to `status` |

The two are orthogonal but constrained: `content_kind='change'` always pairs with
`content_format='diff'`; `content_kind='article'` pairs with `text` or `markdown` depending on what
the fetcher produced. Any other combination is a bug and is rejected at the persist boundary.

**Flag needs its own column; it cannot be a status.** `subscription_items.status` carries a CHECK
constraint allowing only `new`, `reviewed`, `ingested`, `ignored`, `error`
(`Subscriptions_DB.py:156`) — there is no `flagged` value, and SQLite cannot drop a CHECK without
the full table-rebuild dance. More importantly a single status column *cannot* express the real
state: an item can be flagged **and** reviewed at once. `is_flagged` as a separate boolean follows
the precedent ADR-018 already set with `queued_for_briefing`.

**Read maps to `reviewed`.** The reader's "read/unread" language is a UI affordance over the
existing `status` values; `new` is unread, `reviewed` is read. No new status value is introduced.

Added to `local_watchlist_runs`:

| Column | Type | Purpose |
|---|---|---|
| `batch_id` | `TEXT` | Groups the per-source runs produced by one watchlist-level check |

**Ordering constraint:** `local_watchlist_runs` is created lazily at
`local_watchlists_service.py:953`, *not* in `Subscriptions_DB`'s schema init. Therefore `batch_id`
must be added to that lazy `CREATE TABLE` statement, and the `ALTER TABLE` in
`_ensure_watchlists_schema` must be conditional on the table already existing. An unconditional
`ALTER` fails on a fresh database.

### Full-text search

`subscription_items_fts` is an external-content FTS5 table over `title, content, author`, with
insert/update/delete triggers — the same pattern as `character_cards_fts`, `conversations_fts`,
`messages_fts`, and `media_fts`.

Because two code paths insert items today (see below), triggers rather than explicit index writes
are required so both paths stay indexed.

**Backfill runs chunked in a background worker, not inline in migration.** Backfilling a large
`subscription_items` table synchronously would block app boot. Default chunk size 500 rows per
transaction, resuming from the highest indexed rowid so an interrupted backfill continues rather
than restarting.

### Body text: where it lives

| Content | Store | Notes |
|---|---|---|
| Renderable body (what the reader shows, what FTS indexes) | `subscription_items.content` | New |
| Full page snapshot, previous snapshot | `url_snapshots.extracted_content` / `raw_html` | Already exists, keyed `(subscription_id, url, created_at)` |

Full pages are not duplicated per item. `[full page]` and `[previous snapshot]` in the reader read
from `url_snapshots`.

**Known limitation, must be surfaced in the UI:** `content` is `NULL` for every pre-existing item
and cannot be recovered without re-fetching, because no code path has ever persisted a body. For
historical items the reader and search are title-only until the source is re-checked. The reader's
empty state says so explicitly ("no body captured — re-check this source").

### Unified item persistence

Two divergent INSERT paths exist today and write different column sets:

| Path | Writes | Drops |
|---|---|---|
| `Subscriptions_DB.py:1322` | `canonical_url`, `previous_hash`, `change_percentage`, `diff_summary`, `change_type` | `status`, `run_id`, `alert_matches` |
| `local_watchlists_service.py:1301` | `status`, `run_id`, `alert_matches` | all five change/dedup fields |

Neither writes body text, though `local_watchlists_service.py:878` normalizes a `content` value
before discarding it.

Both are replaced by a single persist function writing the full column set, including the new
`content`, `content_format`, and `content_kind`. This is a prerequisite for the reader, not a
cleanup.

### Migration of existing data

1. Each distinct non-empty `subscriptions.folder` becomes a watchlist; sources with no folder join
   one `Unsorted` watchlist. `folder` is left in place and untouched, so migration is re-runnable
   and reversible.
2. Completion is recorded in a dedicated marker table, and the whole migration executes inside one
   transaction so two app instances cannot double-apply it:

```sql
CREATE TABLE IF NOT EXISTS watchlist_migration_state (
    key TEXT PRIMARY KEY,
    applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

   The existing `schema_version` table is not reused: it holds a single integer column
   (`Subscriptions_DB.py:85-88`) with no room for per-migration keys, and bumping it would change a
   value other code may read.

**This migration will do almost nothing for real users.** `folder` is never written by any live
path: `_subscription_config_fields` (`local_watchlists_service.py:567`) allowlists ten fields and
`folder` is not among them, and nothing in `Subscriptions/` calls `add_subscription(folder=…)`.
It is retained only for hand-seeded databases. First-run design therefore assumes an empty
watchlist set — see "All sources" below.

## Module layout

The shell stays thin. `chat_screen.py` is 13,082 lines and is the failure mode to avoid.

| Path | Role | Status |
|---|---|---|
| `Subscriptions/watchlist_bundle_service.py` | Watchlist CRUD, membership, name collision handling, folder migration | new |
| `UI/Screens/watchlists_collections_screen.py` | Shell: rails, tab routing, recovery state | rewrite; <400 LOC is a guideline, not an acceptance criterion |
| `UI/Watchlists_Modules/watchlists_console_handoff.py` | Console staging/follow, extracted from the shell | new |
| `UI/Watchlists_Modules/watchlists_workbench.py` | Rails + vertically-stacked collapsible centre container | new |
| `UI/Watchlists_Modules/watchlist_tree.py` | Left rail: roots, watchlists, sources, tag + status filters | new |
| `UI/Watchlists_Modules/feeds_pane.py` | Centre pane 1 — feeds table scoped by tree selection | new |
| `UI/Watchlists_Modules/items_pane.py` | Centre pane 2 — items table | replaces placeholder |
| `UI/Watchlists_Modules/content_pane.py` | Centre pane 3 — article and change renderers | new |
| `UI/Watchlists_Modules/content_render.py` | HTML→text, diff formatting, markup escaping | new |
| `UI/Watchlists_Modules/region_layout.py` | Five-region collapse/solo/restore state; pure, no Textual | new |
| `UI/Watchlists_Modules/inspector_pane.py` | Right rail — breadcrumb stack + actions | rewrite |
| `UI/Watchlists_Modules/sources_tab.py` | Source CRUD, OPML, health | replaces `sources_pane` |
| `UI/Watchlists_Modules/runs_tab.py` | Run history, batches, logs | replaces `runs_pane` |
| `UI/Watchlists_Modules/rules_tab.py` | Filters + alert rules | replaces `rules_pane` |
| `UI/Watchlists_Modules/artifacts_tab.py` | Artifacts produced by this watchlist → Artifacts screen | new |
| `UI/Watchlists_Modules/watchlists_backend_controller.py` | Local/server routing, capability reporting | keep, extended |
| `UI/Watchlists_Modules/overview_pane.py` | Root-node dashboard | keep, re-scoped |
| `UI/Watchlists_Modules/opml_dialogs.py` | OPML import/export dialogs | keep |

`region_layout.py` is a pure state machine so the fiddliest interaction in the screen is testable
without a Textual pilot.

The screen class name, route (`watchlists_collections`), and stable widget selectors are preserved
so Console handoffs, shell destinations, and route tests keep working.

## Layout and interaction

```
┌ Watchlists ─────────────────────────────── Local ▾ ─ Ready ────────────┐
│ Sources 52  Unread 37  Runs 1  Alerts 2 │ Check now  New  Import OPML  │
├────────────┬─────────────────────────────────────────┬─────────────────┤
│ ▸ All srcs │ [1 Read] 2 Sources 3 Runs 4 Rules 5 Art.│ INSPECTOR       │
│ ▸ Unassign │┌ Feeds in Morning AI Brief (3) ────  z ─┐│ Morning AI Brief│
│ ▾ Morning  ││ Feed        Type Last  New Status      ││  ▸ ArXiv: AI    │
│    ArXiv ◀ ││ ArXiv: AI   rss  2m     24 Healthy     ││    ▸ RAG Eval   │
│    HN Top  ││ HN Top      rss  5m     15 Healthy     ││ ─────────────── │
│    anthro… ││ anthropic…  site 7m      3 Healthy     ││ ArXiv: AI       │
│ ▸ Security │├ Items · ArXiv: AI (24) ───── / ── z ───┤│ 2025-04-25      │
│ ▸ Papers   ││ # Title             Date   Status      ││ 1,240 words     │
│            ││ 1 RAG Evaluation    04-25  New   ◀     ││ New             │
│ #daily #ai ││ 2 Reliable LLM Apps 04-24  New         ││ ─────────────── │
│ #sec       │├ Content ──────────────────────── z ────┤│ [Open       o]  │
│            ││ An Introduction to RAG Evaluation      ││ [Stage      s]  │
│ Unread  37 ││ ArXiv: AI · 2025-04-25 · 1,240 words   ││ [Ingest     i]  │
│ Today    9 ││                                        ││ [Discuss    D]  │
│ Flagged  2 ││ Retrieval quality is usually measured  ││ [Mark read  m]  │
│ Errors   1 ││ with recall@k, but that misses the …   ││                 │
├────────────┴─────────────────────────────────────────┴─────────────────┤
│ c check  i ingest  s stage  / search  z collapse  Z solo  ? help       │
└────────────────────────────────────────────────────────────────────────┘
```

### The tree

Two permanent roots sit above the flat watchlist list:

- **All sources** — every source regardless of membership.
- **Unassigned** — sources belonging to no watchlist.

These are not conveniences. Without them, deleting a watchlist would orphan its sources into
invisibility, and first-run would show an empty tree given that folder migration is effectively a
no-op. Tags filter across watchlists; status filters (Unread / Today / Flagged / Errors) filter
items within the current scope.

Selecting the **collection node itself** shows the merged item stream across all its sources — the
normal feed-reader default. Selecting a **source** narrows to that source.

### Five collapsible regions

`z` collapses or expands the focused region. `Z` solos it, collapsing the other two centre panes;
a second `Z` restores the prior layout from a stack held in `region_layout.py`. `[` and `]` toggle
the left and right rails, matching Console's rail handles.

Collapsed panes become a one-line header showing their count, so nothing vanishes without a trace.
Collapsed panes are skipped by `F6` pane cycling, but their header remains focusable and clickable
so expansion is always reachable.

Collapse state persists across visits in the user's config under a `[watchlists.layout]` section —
five booleans plus the solo-restore stack. It is UI preference, not data, so it belongs in config
rather than `SubscriptionsDB`.

### Tabs

`1`–`5` select Read / Sources / Runs / Rules / Artifacts, preserving the digit muscle memory of the
current screen. Only **Read** uses the three-pane split. Sources, Runs, Rules, and Artifacts take
the full centre width — they have no collection→feed→item relationship.

**Artifacts is empty-state-only in spec #1**, stating plainly that generation arrives in the next
slice. It lists artifacts produced by the selected watchlist and navigates to the Artifacts screen;
it never stores or renders artifacts itself.

Deep-linking needs care. `NavigateToScreen` does accept a `screen_context` dict
(`main_navigation.py:47`), but the Artifacts screen does not read it — it consumes an app attribute,
`pending_artifacts_chatbook_target_id` (`artifacts_screen.py:72-85`), and that attribute is
**chatbook-specific**. Selecting a watchlist artifact therefore requires a parallel pending
attribute and a consumer on the Artifacts screen. That work belongs to spec #2, since spec #1 has no
artifacts to link to; it is recorded here so spec #2 does not discover it late.

### Action scopes

Three surfaces, three scopes, no overlap:

| Surface | Scope |
|---|---|
| Header strip | Screen-scope actions on the current tree selection (Check now, New, Import OPML) |
| Inspector | Selection-scope actions for the deepest current selection |
| Key bindings | Accelerators for whatever the Inspector currently shows |

### Inspector

The Inspector shows a **breadcrumb stack** — watchlist ▸ source ▸ item — with the deepest selection
expanded and shallower levels collapsed to one line each. This is deterministic: it does not depend
on which widget last held focus, which would break the moment a user clicks an Inspector button.

The action buttons always belong to the **deepest expanded level**. Clicking a collapsed breadcrumb
level promotes it to deepest, swapping the detail and the actions together — so the actions on
screen can never belong to a different level than the detail above them.

| Level | Shows | Actions |
|---|---|---|
| Watchlist | tags, source count, unread, cadence (bulk) | Check now, Add source, Rename, Delete |
| Source | URL, type, health, last check, dedup, cadence | Check now, Preview, Edit, Pause, Remove from watchlist, Delete, Restore* |
| Item | source, date, author, word count, status | Open, Stage, Ingest, Discuss in Console, Mark read, Flag |
| Run | status, duration, counts, filter tallies, log tail | Cancel, Re-run |

\* Restore appears only on the server backend, which returns `restore_window_seconds` and
`restore_expires_at` on `SourceDeleteResponse`. The local backend has no undo-delete; the spec names
the asymmetry rather than hiding it.

The **Delivery** row is hidden entirely in spec #1 — there is no delivery to show until spec #2.

### Cadence editing

`subscriptions.check_frequency` remains the single authority; it is what `WatchlistProjection` and
the scheduler already read. No new cadence column exists.

The Inspector's "Check every 6h" on a watchlist is a **bulk editor** over member sources. It shows
the shared value when member sources agree and `mixed` when they do not.

**It must warn about overlap before writing.** Because sources are shared across watchlists, setting
a cadence on one watchlist silently changes it as seen from another. The confirmation names the
overlap: "3 of 5 sources are also in Security."

### Content pane

`content_kind` selects the renderer.

**Article** (feed items) — title, source, date, word count, body.

**Change** (site items) — percent changed, change type, added/removed lines, with `[full page]` and
`[previous snapshot]` reading from `url_snapshots`:

```
┌ Content ───────────────────────────────────────┐
│ anthropic.com/news        site · changed 7m ago │
│ 12% changed · structural                        │
│ ─────────────────────────────────────────────── │
│ + Claude Opus 4.5 is now available in the API   │
│ + Pricing updated for long-context requests     │
│ - Claude Opus 4.1 is now available in the API   │
│                                                 │
│ [full page]  [previous snapshot]                │
└─────────────────────────────────────────────────┘
```

Both kinds share the same pane, keys, and Stage/Ingest/Discuss actions, so sites behave like feeds
throughout the workflow while showing what was honestly captured.

### Reading behaviour

- Opening an item in Content marks it read, with an explicit unread toggle since marking read
  destroys the unread list.
- **Item status is global.** An item marked read from "All sources" is read in every watchlist that
  contains its source. This is correct — it is the same article — but it is stated so it is not
  discovered as a bug.
- `/` opens FTS search scoped to the current tree selection. An active query persists across tree
  selection changes; the scope follows the selection and is shown in the search bar.

### Key bindings

Preserved from today: `1`–`5`, `?`, `n`, `d`, `c`, `p`.
Added: `e` edit, `s` stage, `i` ingest, `P` pause, `z` collapse, `Z` solo, `/` search, `j`/`k`
next/previous item, `[` / `]` rail toggles.

Implementation must run a conflict audit against `BaseAppScreen` and app-level bindings. Roughly
fifteen bare letters plus shifted variants is not a set to assume is free, and shifted-letter
bindings are weakly discoverable — the footer hint bar carries them.

### Empty states

First run has no watchlists. The tree shows **All sources** with an inline "Add your first feed"
affordance rather than an empty box. Items with no captured body explain why and how to fix it.
The Artifacts tab names the next slice instead of showing a bare empty table.

## Data flow

1. Mount resolves the backend, then **one grouped query** populates every tree count off the event
   loop. `Subscriptions_DB` uses `threading.local()` connections (`:75`, `:368-370`), so worker
   threads hold their own connection and this is safe.
2. Tree selection scopes Feeds; Feeds selection scopes Items; Items selection loads Content.
3. Tabs lazy-load on first open.
4. Every fetch carries a backend token; results are discarded if the backend changed in flight.
5. Mutations go pane → controller → scope service → worker → reactive update.

### Counts must be one query

No per-subscription count helper exists in `Subscriptions_DB` — only `get_new_items` and a single
`COUNT(*)` in the entire file. Counting per tree node per refresh would issue one query per
watchlist on every refresh. Tree counts are computed by a single grouped query, cached, and
invalidated on run completion or item status change.

This repo has form here: the performance audit found the Console's 0.2s tick running SQLite on the
event loop.

### Two polling cadences

- **Screen-level status poll**, low frequency, always active while the screen is mounted: drives the
  header strip counts and tree run status.
- **Run detail poll**, higher frequency, only while the Runs tab is visible and a run is
  non-terminal.

### Backend routing

| Operation | Local | Server |
|---|---|---|
| List watchlists | `WatchlistBundleService` | `list_watchlist_groups` |
| Create / rename / delete watchlist | `WatchlistBundleService` | **disabled** — see below |
| Source ↔ watchlist membership | `WatchlistBundleService` | **unsupported** — capability banner |
| Source CRUD | `WatchlistScopeService` | `WatchlistScopeService` |
| Check now | `LocalWatchlistsService.launch_run` | `check_watchlist_sources_now` |
| Runs, run detail, cancel | `WatchlistScopeService` | `WatchlistScopeService` |
| Item read / search | local only | local only |
| OPML import / export | local OPML service | `import/export_watchlist_sources` |

**Membership cannot be set from this client.** `SourceResponse` exposes `group_ids` for reading, but
`SourceUpdateRequest` has no `group_ids` field and neither `WatchlistGroupCreateRequest` nor
`WatchlistGroupUpdateRequest` carries members — all three are `extra="forbid"`, so nothing can be
smuggled through. There is no wire path for "add this source to this watchlist" on the server.

Consequently **watchlist creation is also disabled on the server backend**, not merely membership
editing. Group creation would otherwise succeed and produce a node that can never contain anything.
A capability banner explains this, following the pattern ADR-018 used for content-alert rules. A
follow-up task covers adding the membership endpoint.

`WatchlistGroupCreateRequest.parent_group_id` exists, so the server supports nested groups while
local watchlists are flat. Server nesting is displayed flattened, with the path in the name.

Item bodies, read state, and search are local-only in both modes — the server exposes no item
content API in this client.

## Error handling and security

- `DestinationRecoveryState` for service-level failures; `policy_denied_recovery_state` for policy
  denials — both already established on this screen.
- `@work(exclusive=True, group=…)` per pane to prevent duplicate fetches.
- Backend-unsupported operations show a capability banner explaining why, never a disabled control
  with no explanation.
- Inline field-level validation on forms.

### Untrusted remote content

Feed titles, authors, and bodies are attacker-controllable. Two requirements:

1. **Escape on output.** Every remote-derived string rendered into Textual markup passes through
   `escape_markup`. The existing panes sanitize *inputs* (`sources_pane.py:224-244`) but nothing
   escapes *outputs*. This repo has shipped this exact bug before, in tooltips rendering markup.
2. **Guaranteed HTML degradation.** BeautifulSoup and html2text sit behind `optional_deps` in the
   fetch layer, and RSS bodies are HTML. `content_render.py` provides a dependency-free fallback —
   tag strip plus entity unescape — so an install without those extras renders plain text rather
   than markup soup.

## Testing

| Layer | File | Covers |
|---|---|---|
| Pure | `Tests/Watchlists/test_region_layout.py` | collapse, solo, restore stack across five regions; no pilot |
| Service | `Tests/Subscriptions/test_watchlist_bundle_service.py` | CRUD, membership, case-insensitive name collision + auto-suffix, folder migration idempotency |
| DB | `Tests/DB/test_subscriptions_db_watchlists.py` | migration re-runnability, conditional `batch_id` ALTER on fresh vs existing DB, FTS triggers + chunked backfill, cascade on delete |
| DB | `Tests/Subscriptions/test_item_persist.py` | unified persist writes the **full** column set — the regression that allowed two divergent inserts |
| UI | `Tests/Watchlists/test_watchlists_tree.py` | roots, scoping, tag filters, orphan reachability after watchlist delete |
| UI | `Tests/Watchlists/test_watchlists_read_tab.py` | drill-down, both content renderers (article and change), auto-mark-read |
| UI | `Tests/Watchlists/test_watchlists_inspector.py` | breadcrumb stack targeting, cadence overlap warning, Restore only on server |
| Security | `Tests/Watchlists/test_watchlists_escaping.py` | a feed title containing `[bold]…[/]` renders literally |
| Perf | `Tests/Watchlists/test_watchlists_counts.py` | tree counts issue exactly one query (N+1 guard), asserted via a connection wrapper counting `execute` calls across a tree refresh |
| Integration | `Tests/UI/test_watchlists_destination_shell.py` (extend) | backend switch, capability banners, Console handoff, route stability |

## Migration and cleanup

- Replace `sources_pane.py`, `items_pane.py`, `runs_pane.py`, `rules_pane.py`, `inspector_pane.py`.
- Extract Console handoff from the shell into `watchlists_console_handoff.py`.
- Retire the placeholder column labels ("Column 1: Watchlist List", etc.).
- Update `Tests/Watchlists/` and `Tests/UI/test_watchlists_*` to the new panes.
- Amend ADR-018's section IA and pane set; record the new IA and entity model in a new ADR.
- Note in ADR-018 that its "groups/tags read-only" statement is stale — group CRUD exists at
  `client.py:6775-6819`.

## Open follow-ups

| Item | Owner |
|---|---|
| Server membership endpoint (`group_ids` on source update, or group member endpoints) | follow-up task |
| Artifact→watchlist provenance. No link exists between a generated artifact and the watchlist that produced it, so the Artifacts tab has nothing to query even once generation ships. Reserve the shape early — retrofitting provenance after artifacts exist is materially harder | spec #2, decide before generation lands |
| Artifacts deep-link: a watchlist-aware pending attribute plus consumer, alongside today's chatbook-only `pending_artifacts_chatbook_target_id` | spec #2 |
| Briefing / podcast generation, template authoring, 2-speaker script + audio | spec #2 |
| Recurring delivery scheduling for artifacts | spec #2, Schedules screen |
| Fate of the ~4,600 unimported LOC in `Subscriptions/` (briefing, aggregation, distribution, export, RSS generation, recursive summarization) | spec #2 |
| Local undo-delete parity with the server's restore window | follow-up task |
| Nested watchlists, if flat proves insufficient | deferred |

## Decisions resolved during design

1. **Watchlist is the bundle** — not folders/tags, not the server's Group+Job split.
2. **Flat, not nested** — tags carry cross-cutting grouping.
3. **Artifacts live on the Artifacts screen** — Watchlists links out.
4. **Cadence edits here, delivery in Schedules.**
5. **Two renderers, one Content pane** — sites show diffs, feeds show articles.
6. **FTS5 over items**, not LIKE, not filters-only.
7. **Rewrite the UI on the existing service seam** — no parallel v2 file.
