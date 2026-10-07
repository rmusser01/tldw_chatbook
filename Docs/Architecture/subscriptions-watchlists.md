# Subscriptions and watchlists

This document describes how sources are polled, how items and briefings are produced, and how the Watchlists screen reads them: the scheduler projections, the single-flight claim model, the monitoring engine, briefing selection and generation, and the artifacts chain (scripts, audio, export).

## Authoritative files

| Concern | File | Key symbols |
| --- | --- | --- |
| Run service | `Subscriptions/local_watchlists_service.py` | `LocalWatchlistsService` — `launch_run` (durable receipt), `execute_run`, `wait_for_terminal_run`; executable source types `{rss, atom, json_feed, podcast, url, url_list, sitemap, api}`; disposition ledger |
| Monitoring engine | `Subscriptions/monitoring_engine.py` | `FeedMonitor` (RSS/Atom/JSON-feed, conditional requests), `URLMonitor` (snapshots + change classification), `RateLimiter`, `CircuitBreaker`, `ContentExtractor` |
| Briefings | `Subscriptions/briefing_service.py`, `briefing_selection.py` | `generate_briefing` (claim + execute), watermark-based `select_briefing_items` (modes auto / curated / auto_featured), prompt builder, citation extraction, zombie sweep |
| Briefing artifacts | `briefing_cast.py`, `briefing_audio.py`, `briefing_export.py`, `briefing_keep.py`, `briefing_feed.py`, `daily_reports_view.py` | multi-speaker scripts, TTS audio, markdown/podcast-feed export, keep-as-note, Daily Reports derivation (ADR-079) |
| DB | `DB/Subscriptions_DB.py` | `SubscriptionsDB` schema v2; `subscriptions`, `subscription_items`, `url_snapshots`, `local_watchlist_runs`, alert rules, `briefings`/`briefing_items`/presets/scripts/audio; partial unique indexes as single-flight claims; FTS5 with LIKE fallback |
| Scheduler projections | `Scheduling/services/watchlist_projection.py`, `briefing_projection.py` | project DB cadences into synthetic `watchlist:<id>` / `briefing:<id>` tasks (no rows of their own) |
| Handlers | `Scheduling/scheduler/handlers/watchlist_check_handler.py`, `briefing_handler.py` | thin wrappers; briefing generation is fire-and-forget off the scheduler tick |
| Operations | `Subscriptions/watchlists_operation_coordinator.py`, `startup_reconcile.py` | supervised manual ops (semaphore 4, terminal-retry ≤3); startup sweep failing in-progress rows |
| DB offload | `Subscriptions/db_offload.py` | one `to_thread` hop per sync DB call; **inline** for `:memory:` DBs |
| UI | `UI/Screens/watchlists_collections_screen.py` + `UI/Watchlists_Modules/` | Read tab first (ADR-042), sources, runs, rules, notifications, artifacts |

## Poll → items → briefing (dataflow)

1. **Schedule**: the scheduler loop reads `WatchlistProjection.list_jobs()`; a task is due when `last_checked + check_frequency <= now` (per-subscription seconds).
2. **Claim**: `launch_run` inserts a `queued` row into `local_watchlist_runs`; the partial unique index (`uq_local_watchlist_runs_active_source`) makes a concurrent duplicate return the existing winner, which then waits on `wait_for_terminal_run` instead of executing — the unique index **is** the single-flight claim.
3. **Fetch**: feeds parsed by `FeedMonitor` with ETag/Last-Modified; `url`/`url_list`/`sitemap` checked per-URL by `URLMonitor` (content hash vs `url_snapshots`, change classification, dispositions `changed/unchanged/withheld/baseline/rebaselined/error/skipped`); per-URL error isolation means one dead URL cannot fail its siblings; a per-(subscription,url) in-flight guard prevents double-reporting.
4. **Filter & persist**: run filters + content-alert rules; kept items upsert into `subscription_items` (FTS-indexed by trigger, status `new/reviewed/ingested/ignored/error`); health recorded (`consecutive_failures` → auto-pause at threshold); a run where every URL errored is stamped `failed` so auto-pause parity holds.
5. **Briefing** (manual Generate or scheduled cadence): claim (`uq_briefings_generating_watchlist` + in-process registry) → `select_briefing_items` (watermark window; featured items never dropped while auto items remain; 7-day floor for the first briefing) → one `chat_api_call` via the briefing prompt → citations extracted and cross-checked → `briefings` row completed with `body_markdown` and the `covers_through_item_id` watermark. Scheduled completions auto-keep into ChaChaNotes and dispatch a briefing notification.

## Run producers

Three producers write `local_watchlist_runs`: scheduled checks (the handler routes through `launch_run`/`execute_run` so scheduled runs look identical to manual ones), manual "Check now" (the operation coordinator's supervised batch receipts), and batch accept (1–50 sources). Startup reconcile fails pre-boundary `queued/running` rows as interrupted so the claims can't stay wedged.

## Search boundary (the FTS-with-LIKE fallback)

Item search uses FTS5 only when the `_docsize` shadow coverage probe says the index is complete; on `sqlite3.OperationalError` (missing table or FTS5 compiled out) it falls back to AND-of-terms / OR-across-columns LIKE with escaped wildcards — "the search box must never raise into the reader". One search mode is pinned per page so counts, high-water marks, and rows agree. Delete triggers are membership-guarded so pre-migration rows get indexed on next update instead of raising "malformed"; a startup backfill worker tops the index up.

## Boundaries

- Scheduling owns **when**; `LocalWatchlistsService` owns **what a run is**; `SubscriptionsDB` owns all durable state — handlers are stateless apart from the briefing handler's in-flight task set.
- `Subscriptions/scrapers/` (reddit/youtube/github/hackernews/…) is registered but has **no importers** — legacy/dead; the eight types in the DB CHECK are the real vocabulary. Likewise `Constants.SUBSCRIPTION_TYPES` / `SUBSCRIPTION_UPDATE_FREQUENCIES` have zero consumers — cadence is raw seconds in `check_frequency`.
- Local vs server sources split at `watchlist_scope_service.py`; some server flows declare local run-execution unsupported.
- `Subscriptions/SUB-Arch.md` self-declares as historical; only its "what actually runs" banner is current.

## Config keys (`[scheduling]`)

`watchlist_checks_enabled` (default true — the sole executor gate; there is no legacy scheduler behind it), `watchlist_checks_shadow` (default false — fetch-and-discard diagnostics; cannot probe sitemap/api, reports `shadow_unsupported`; ignores cadence, so leaving it on means no real checks run), `briefing_schedules_enabled` (default true; per-watchlist cadence is opt-in via `briefing_cadence_seconds`, NULL = never), `handler_timeout_seconds` (300). Per-subscription columns act as config: `check_frequency`, `change_threshold`, `ignore_selectors`, `auto_pause_threshold`, extraction settings, `etag`/`last_modified`.

## Failure behaviors

| Case | Behavior |
| --- | --- |
| Claim race | Loser gets the winner's receipt and waits |
| Cancellation mid-run | Terminal failure recorded, then re-raised (deliberately not shielded); a second cancel at shutdown can leave a visible, re-runnable `running` row |
| All-URL error run | Run fails so auto-pause can trigger; entirely-skipped guard runs bypass health accounting |
| Briefing crash | Claim registry + zombie sweep bounded by a captured max row id (never fails rows this process just created) |
| Shadow mode | Fetches and discards; never writes `last_checked` |

## Governing decisions and docs

ADR-018 (`018-watchlists-tui-screen.md`, superseded in IA by 042), ADR-019 (`019-watchlist-scheduler-migration.md` — the handler is the sole executor; the old scheduler/briefing-aggregation stack was deleted), ADR-042 (`042-watchlists-reader-first-ia.md`), ADR-043 (OPML mapping), ADR-079 (`079-daily-reports-surface-and-demo-seeding.md`). Specs: `Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md`, `2026-08-05-watchlists-reader-first-design.md`, `2026-08-01-kept-briefings-design.md`, `2026-07-29-watchlists-noise-not-volume-design.md`. User guides: `Docs/User_Guide/watchlists.md`, `watchlists-quickstart.md`. See [scheduling-workflows.md](./scheduling-workflows.md) for the loop that drives these handlers.

## Verified gotchas

1. The DB CHECK on `subscriptions.type` is the authoritative type list — not `Constants.SUBSCRIPTION_TYPES`.
2. The unique indexes are the claims: don't add a code-level lock on top; the winner-receipt path is the designed behavior.
3. `subscription_stats`/`get_subscription_health` have no readers; the Runs pane reads `local_watchlist_runs`.
4. Failure recording deliberately avoids double-counting `consecutive_failures` when a run row already exists.
5. Shadow mode ignores cadence — leaving it enabled silently disables real checking.
