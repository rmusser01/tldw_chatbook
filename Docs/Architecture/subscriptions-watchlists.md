# Subscriptions and watchlists

This document describes how sources are polled, how items and briefings are produced, and how the Watchlists screen reads them: the scheduler projections, the single-flight claim model, the monitoring engine, briefing selection and generation, and the artifacts chain (scripts, audio, export).

## Authoritative files

| Concern | File | Key symbols |
| --- | --- | --- |
| Run service | `Subscriptions/local_watchlists_service.py` | `LocalWatchlistsService` — `launch_run` (durable receipt), `execute_run`, `wait_for_terminal_run`; executable source types `{rss, atom, json_feed, podcast, url, url_list, sitemap, api}`; disposition ledger |
| Monitoring engine | `Subscriptions/monitoring_engine.py` | `FeedMonitor` (RSS/Atom/JSON-feed, conditional requests), `URLMonitor` (snapshots + change classification), `RateLimiter`, `CircuitBreaker`, `ContentExtractor` |
| Briefings | `Subscriptions/briefing_service.py`, `briefing_selection.py` | `generate_briefing` (claim + execute), watermark-based `select_briefing_items` (modes auto / curated / auto_featured), prompt builder, citation extraction, zombie sweep |
| Briefing artifacts | `briefing_cast.py`, `briefing_audio.py`, `briefing_export.py`, `briefing_keep.py`, `briefing_feed.py`, `daily_reports_view.py` | multi-speaker scripts, TTS audio, markdown/podcast-feed export, keep-as-note, Daily Reports derivation (ADR-323) |
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

## Reader detail work

Article activation admits an exclusive screen-owned worker in the wc_item_detail
group and immediately releases the screen message queue. The worker fetches only
the selected body's detail, then checks selection generation, committed snapshot
identity/generation, page, runtime backend, active section and screen attachment
under the existing presentation lock. Section, backend and page intents and
unmount retire its generation, including away/back transitions. An obsolete
completion cannot mutate the shared row, publish error feedback or mark it read,
even when its underlying DB thread finishes after cancellation.

A current completion hydrates the existing shared row before one ContentPane.item
publication and the normal mark-read action. Missing detail preserves any existing
body; a current fetch failure keeps the established error feedback and retry path.
List projections and cached immutable pages remain narrow and structurally shared.
These are screen-local publication rules under ADR-011/033/042 (TASK-36900).

## Rule editor view state

`RulesPane.watch_rules` updates table rows and empty guidance in place, preserving
mounted form controls, focus, cursor and selection during list publication.
The original edit row is independent of the currently selected table row.
`RuleFormVisibilityChanged` carries raw Name, Condition, Threshold, Severity and
Enabled values (including unparsed threshold text) to
`WatchlistsCollectionsScreen.handle_rule_form_visibility_changed`.
The screen's existing `_rule_form_*` view state re-seeds a newly constructed Rules
pane on section/region return; it remains ephemeral under ADR-033. Save and Cancel
retire the draft, while a rejected empty Name leaves it open. Only explicit Save
uses the existing rule mutation path; list refresh and section navigation never
save a draft.

Reader j/k and next-unread navigation use the article list selection intent while a detail read is pending; article publication and read-status mutation still wait for the current hydrated result. The displayed page bounds navigation, including list boundaries.

## Boundaries

- Scheduling owns **when**; `LocalWatchlistsService` owns **what a run is**; `SubscriptionsDB` owns all durable state — handlers are stateless apart from the briefing handler's in-flight task set.
- `Subscriptions/scrapers/` (reddit/youtube/github/hackernews/…) is registered but has **no importers** — legacy/dead; the eight types in the DB CHECK are the real vocabulary. Likewise `Constants.SUBSCRIPTION_TYPES` / `SUBSCRIPTION_UPDATE_FREQUENCIES` have zero consumers — cadence is raw seconds in `check_frequency`.
- Local vs server sources split at `watchlist_scope_service.py`; some server flows declare local run-execution unsupported.
- `Subscriptions/SUB-Arch.md` self-declares as historical; only its "what actually runs" banner is current.

## Saved article listening

`Subscriptions/article_listening_store.py` owns the immutable listening records
in SubscriptionsDB v4. Each explicit config selector has one session and an empty
revision/incarnation witness. Queue and foreground bookmarks commit together;
unfinished sessions load paused. The selector key is independent of display names
and recovery installation IDs, including when selectors share a physical DB.

Preparation freezes bounded saved speech, format and nonsecret voice/dependency
values. The shared `DB/subscriptions_schema.py::listening_brief_items` resolves
completed-brief citations against their saved selection before reading article
bodies. Change reports, missing items and empty bodies retain bounded unavailable
reasons. Recovery applies the same provenance, integer, count and byte checks.
Candidates also bind the open physical DB image, so pathname reuse cannot admit
snapshots from a replaced database.

`Subscriptions/article_listening_lease.py` holds a stable DB-sibling native lock
and ordinary storage admission through actual borrower retirement. Explicit
`SubscriptionsDB.destructive_transaction` scopes destructive SQL intent separately
from physical custody. Source deletion scrubs private snapshots and bookmarks,
preserves unavailable queue positions and retains safe generation cleanup keys.
Membership removal preserves source items; brief deletion preserves an unrelated
foreground. Audio generations and bookmarks identify immutable retained bytes.

`Subscriptions/article_audio_assets.py` binds an explicit profile data directory
and stores flat UUID WAV keys under `article_audio`, including when the DB lives
elsewhere. Its inert model validator checks the complete digest, PCM16 stereo
48 kHz shape and 12,000 zero-tail frames, with a 64 MiB file cap, 65,536-byte
header allowance and 1,024 parser-read budget. Reuse verifies the physical file
and repeats source identity and revision checks before attachment.

Publication and deletion use a qualified Windows NTFS regular-file handle through
the actual native effect. Named ancestor pins protect the parent namespace.
The nullable `custody_witness` binds the issuing DB image, generation, key, native
root/leaf creation identities and measured bytes. It survives source scrubbing
and retries; a filename or copied receipt alone grants no cleanup authority.
Rejected live stages use a distinct stage-only cleanup receipt after committed
invalidation. Directory barriers precede forgetting deletion records, including
retries where the name is already absent. Unsupported effect platforms refuse
elective pruning before changing reusable metadata; file validation stays portable.

Every retained entry role protects its generation. Eligible unreferenced audio
uses seven-day/512 MiB LRU retention. Successful user-directed verified reads
refresh last use; integrity scans do not. Recovery validates live-generation
timestamps before admitting a copied image while allowing scrubbed deletion
records. A missing or corrupt artifact keeps its old
bookmark identity; explicit regeneration binds a new generation at zero. Unknown,
unsealed or foreign objects remain preserved with uncertainty reported, including
restored bytes whose local custody has not been established. Physical cleanup
runs through an explicitly bound asset owner after DB invalidation, without
inferring a data directory from a deletion row.

The existing `subscriptions.assets` recovery owner declares both briefing and
article trees. Legacy briefing IDs stay unchanged; article roots and final WAV
members use explicit profile IDs. Private stages and the device-local DB-sibling
lock are excluded. Recovery validates the v4 catalog, source authority and WAV
bytes without constructing a provider or player. DB-only restore preserves
bookmarks even when their corresponding audio is absent.

Selected profiles can share an article directory and its default DB, or use
distinct custom DBs with the same data root. Inventory assigns each file to its
unique verified selected partition and config/DB peer. Unknown, unselected or
multiply claimed members refuse a complete capture. The canonical article-root
projection preserves each member's ownership when restoring to separate roots
or publishing their verified disjoint union once to a common reviewed root.
Generic database, briefing and file alias rules remain unchanged.

Incoming and detached copies rebind reviewed selector keys only in private
candidate databases and clear foreign cleanup custody for every partition.
Local safety snapshots retain their original bytes. Runtime maintenance fences
listening intake and drains retained physical work before exclusive storage
admission, then reopens intake without starting playback. An absent feature
does not create its audio directory during maintenance.

Later replacement rollback of a newly created multi-selector shared root remains
unqualified: existing finite-membership and affirmative retirement checks refuse
it while preserving files and control state. Shared capture/publication alone
does not establish that destructive authority. Single-selector replacement and
rollback retain their existing verified path.

The storage and recovery contract is
[ADR-367](../../backlog/decisions/367-watchlist-article-listening-and-controlled-playback.md).
Preparation, transport orchestration and listening controls build on this boundary
in the subsequent implementation tasks.

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

ADR-018 (`018-watchlists-tui-screen.md`, superseded in IA by 042), ADR-019 (`019-watchlist-scheduler-migration.md` — the handler is the sole executor; the old scheduler/briefing-aggregation stack was deleted), ADR-042 (`042-watchlists-reader-first-ia.md`), ADR-307 (OPML mapping), ADR-323 (`323-daily-reports-surface-and-demo-seeding.md`). Specs: `Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md`, `2026-08-05-watchlists-reader-first-design.md`, `2026-08-01-kept-briefings-design.md`, `2026-07-29-watchlists-noise-not-volume-design.md`. User guides: `Docs/User_Guide/watchlists.md`, `watchlists-quickstart.md`. See [scheduling-workflows.md](./scheduling-workflows.md) for the loop that drives these handlers.

## Verified gotchas

1. The DB CHECK on `subscriptions.type` is the authoritative type list — not `Constants.SUBSCRIPTION_TYPES`.
2. The unique indexes are the claims: don't add a code-level lock on top; the winner-receipt path is the designed behavior.
3. `subscription_stats`/`get_subscription_health` have no readers; the Runs pane reads `local_watchlist_runs`.
4. Failure recording deliberately avoids double-counting `consecutive_failures` when a run row already exists.
5. Shadow mode ignores cadence — leaving it enabled silently disables real checking.
