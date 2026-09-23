# Dreams — daily discovery and tracking design

- **Status:** Proposed — awaiting owner review
- **Date:** 2026-09-22
- **Decision:** [ADR-178](../../../backlog/decisions/178-dreams-daily-discovery-and-tracking.md)
- **Classification:** Architectural
- **Approved product decisions:** Dreams discovers **new** content and events
  (it never resurfaces what the user already has — the library is the signal,
  not the content); two loops — daily Discover plus persistent Track; watchlists
  are a bidirectional partner (candidate input *and* "Track this" target);
  surface lives on the Artifacts screen as a new artifact type; staged delivery
  (Phase 1 discovery, Phase 2 tracking); person-specific social tracking is
  permanently out of scope.

## Summary

DreamBeans (Google Labs, expanded Sept 2026) generates a personalized daily
collection of stories built from the user's own connected data — forward-looking
discovery, not rediscovery. This design brings that experience to chatbook as
**Dreams**: once a day, an engine distills the user's current interests from
their local data, searches the live web (plus fresh watchlist items) for new,
relevant, sometimes time-sensitive things — articles, events, deals, social
opportunities — and writes each find into a short story explaining what it is
and why it fits the user right now. Stories can be dived into (console chat
handoff), kept, exported, ingested into the library, rated, and — in Phase 2 —
**tracked**: a concert page keeps getting checked for changes, a standing
question ("cheap flights to Japan?") keeps getting re-asked until something
materially changes.

The feature deliberately reuses chatbook's proven scheduled-generation
architecture rather than inventing a parallel one: the briefing pipeline's
row-per-outcome discipline and provider resolution, the scheduler's handler
registry, the watchlist engine's URL monitoring and disposition vocabulary, the
console handoff pattern, and the Artifacts screen's daily-reports projection
wiring. The genuinely new machinery is the interest profile, the discovery
pipeline, the feedback loop, and the tracked-item lifecycle.

Phase 1 ships discovery + surface + ingest + watchlist-input. Phase 2 ships
Track-this (both mechanisms) + goals + region + event metadata.

## Background

Relevant existing seams, all verified present in the repo during design review
(two review passes, 2026-09-22):

- `tldw_chatbook/Subscriptions/briefing_service.py` — the generation pattern to
  copy: claim-guarded lifecycle (`generating → complete | empty | failed`),
  exactly one `chat_api_call` per item, every outcome a row; the sync
  `chat_api_call` runs under `asyncio.to_thread` (~line 604–622) with DB writes
  grouped into one thread hop per branch; provider/model resolution chain
  (~line 326–358): per-preset columns → remembered chat defaults via
  `Chat/provider_setup_persistence.resolve_remembered_provider_model`.
- `tldw_chatbook/Scheduling/` — `SchedulerLoop` with a generic handler registry
  and missed-fire accounting (task-18937: `missed_fire_grace_seconds`, default
  2× poll interval — minutes, not hours); handlers
  `briefing_handler.py` (fire-and-forget `asyncio.Task`),
  `watchlist_check_handler.py` (keyed by subscription id), `reminder_handler.py`
  (dispatches through `NotificationDispatchService`); tables in
  `Scheduling/db/scheduled_tasks_db.py`.
- `tldw_chatbook/Subscriptions/local_watchlists_service.py` — the local check
  executor: `EXECUTABLE_SOURCE_TYPES` covers `rss/atom/json_feed/podcast/url/
  url_list/sitemap/api`; every check yields a disposition
  (`changed/unchanged/baseline/rebaselined/withheld/error/skipped`); run
  claiming guards concurrent checks of the same `(subscription, url)` pair
  (task-16838).
- `tldw_chatbook/Subscriptions/monitoring_engine.py:1397` — `URLMonitor`
  ("Monitor URLs for changes") with persisted snapshots, rate limiting, and
  per-site scraping config (`SiteConfigManager`).
- `tldw_chatbook/DB/Subscriptions_DB.py` — `subscriptions` (typed sources with
  `check_frequency`, `is_active`, `is_paused`, `auto_pause_threshold`,
  consecutive-failure counters), `watchlists`/`watchlist_sources`,
  `subscription_items`, `local_watchlist_alert_rules` (`condition_type` +
  `condition_value_json` + `severity`, attached to a subscription).
- `tldw_chatbook/Web_Scraping/WebSearch_APIs.py` — `perform_websearch(...)`
  (synchronous; engines google/bing/brave/duckduckgo/kagi/exa/serper/tavily;
  supports `date_range`, `site_blacklist`, `exactTerms`, `excludeTerms`).
- `tldw_chatbook/UI/Screens/artifacts_screen.py` — hand-wired Daily Reports
  pattern (refresh worker + generation guard + preview card + a type filter
  taxonomy "All | Chatbooks | Reports | Datasets | Drafts | Exports").
- `tldw_chatbook/Chat/chat_handoff_models.py` (`ChatHandoffPayload`) +
  `UI/Navigation/pending_handoff_store.py` — console dive-deeper handoff,
  already used from skills/study/library/personas screens.
- `tldw_chatbook/Personal_Context/key_protector.py` — profile-key custody is
  secure-keyring **or** passphrase; `ProfileLockedError(reason_code=
  "profile_locked")` when neither is loadable unattended.
  `Utils/private_paths.open_private_binary` — private-path file access.
- `tldw_chatbook/DB/Library_Collections_DB.py` — `collection_capture_items`
  (read-it-later capture with `favorite` flag); `DB/Client_Media_DB_v2.py` has
  `MediaReadItLaterState` — the ingest landing zones.
- `DB/ChaChaNotes_DB.py` (`notes`, keywords), `DB/Client_Media_DB_v2.py`
  (`Media`, `Keywords`, `ReadingProgress`) — interest-signal reads.
- Governance: [ADR-029](../../../backlog/decisions/029-local-private-data-boundary.md)
  (local private data boundary), [ADR-079](../../../backlog/decisions/079-daily-reports-surface-and-demo-seeding.md)
  (Daily Reports / Artifacts slot precedent), ADR-019 (watchlist checks on the
  scheduler), ADR-150 (design tokens — see
  [design-language.md](../../../backlog/docs/design-language.md)), ADR-031
  (keybinding/footer conventions).

## Goals

1. A daily, dated collection of ~5 story-shaped discoveries that feels personal
   and forward-looking: new content, events, deals, and social opportunities
   matched to inferred current interests.
2. Every story is actionable: dive deeper into a chat, keep, export to
   Markdown, ingest into the library, rate more/less — and (Phase 2) track.
3. A local feedback loop that visibly steers future cycles without collapsing
   into an echo chamber (exploration budget guarantees diversity).
4. Tracking with follow-through (Phase 2): watched pages and recurring
   questions produce tracked updates, respect budgets, and retire themselves.
5. Privacy posture at least as strict as the rest of the app: raw signals never
   leave the machine; everything that does leave is previewable verbatim.
6. All scheduled work is idempotent per local day, catch-up friendly for a
   laptop that was asleep, and cost-bounded.

## Non-goals

- Surfacing/resurfacing content the user already has (Dreams finds *new* things;
   the library is the signal, not the corpus).
- Conversation-derived interest signals (deferred; most privacy-sensitive).
- Server-side personalization (the `Personalization_Interop` seam stays
   untouched).
- Person-specific social tracking — tracking *a specific person's* events or
   availability. There is no social graph and there will be no scraping of
   people. Social *interests* ("board game nights near me") are in scope via
   goals.
- Structured price APIs, ticket-inventory APIs, or event-directory APIs — all
   change detection is search + LLM judgment over public snippets, honestly
   labeled as such.
- Geolocation: region is a user-provided free-text string, never derived.
- Photo weaving, audio-rendered dreams (briefings already own audio if ever
   wanted), and image generation.

## Detailed design

### Concepts and package layout

New package `tldw_chatbook/Dreams/` with one module per stage:

```
Dreams/
  __init__.py
  interest_profile.py      # topic/goal distillation, decay, feedback math
  profile_sources.py       # notes/media/Personal-Context readers (degrading)
  query_synthesis.py       # profile -> search queries (one LLM call)
  discovery.py             # search client wrapper, candidate pool, dedupe, rank
  story_service.py         # story generation, row-per-outcome
  cycle_service.py         # orchestration: one dated collection per run
  track_service.py         # Phase 2: tracked items, page+question mechanisms
  dreams_view.py           # read-only Artifacts projection (daily-reports pattern)
  settings.py              # [dreams] config access + defaults
DB/Dreams_DB.py            # new subsystem DB module + migrations
Scheduling/scheduler/handlers/dreams_handler.py  # dreams_cycle + dream_track_check
UI/Screens/artifacts_dreams_modal.py             # story detail modal + actions
```

A "story" is one discovered item wrapped in a 120–200 word narrative. A
"collection" is the dated set produced by one cycle. A "tracked item" (Phase 2)
is a standing intent to re-check one page or one question.

### The interest profile

The profile is a small local store with three facets:

- **Topics** — weighted strings (weight 0–1) derived from recent note keywords,
  media-library keywords, and reading progress. Recency-decayed: weights decay
  toward uniform over ~2 weeks so last month's obsession doesn't dominate
  forever. Adjusted by feedback (below).
- **Goals** — persistent wants ("visit Japan", "see Wednesday 13 live", "board
  game nights"). **Never decayed, never adjusted by feedback** — goals change
  only by direct user edit. Each goal carries a `searchable` flag (default on)
  that controls whether its text may appear in outbound queries. Goals drive
  event/deal/social queries; topics drive content queries.
- **Region** — one user-provided free-text string ("near Seattle"), stored in
  `[dreams] region` config, injected into "nearby"-style queries, shown in the
  preview. No geolocation, ever.

Signal readers degrade, never block:

- Notes/media readers are plain DB reads over existing tables.
- **Personal Context is optional and lock-aware**: the distillate refreshes
  from the profile whenever it is unlocked during a session (cache-on-unlock);
  a locked profile at cycle time (`ProfileLockedError`) means the cycle runs on
  notes/media/cached topics and the collection row records the degradation. The
  cached distillate is written through `open_private_binary` private-path
  discipline.

Cold start: enabling Dreams with an empty profile runs a seed step (pick ~3
topics; watchlist topic suggestions offered) and the engine runs in
LLM-knowledge mode until the profile is non-empty.

### The discovery pipeline (one cycle)

1. **Assemble profile snapshot** (topics+goals+region, hashed into the
   collection row so feedback is always interpretable against the profile that
   produced it).
2. **Query synthesis** — exactly one LLM call turns the snapshot into
   `queries_per_cycle` (default 3) diverse queries. Query contracts: recency
   terms for time-sensitive angles (`perform_websearch`'s `date_range`), event-
   style phrasing for goal facets, a configured `site_blacklist` for known SEO
   farms. At least `exploration_slots` (default 1) of the cycle's story slots
   are reserved for off-profile exploration angles (ε-greedy against echo-
   chamber collapse).
3. **Discovery** — run the queries through `perform_websearch` (thread-off-
   loaded; engine per `[dreams] search_engine` with the standard backend
   fallback chain). Candidate pool = web results **plus fresh watchlist items**
   (`subscription_items` within `watchlist_freshness_hours`, default 48 —
   sources the user already follows surfacing stories they haven't seen). Dedupe against the seen-items
   ledger and against URLs already in the media library; rank by profile
   relevance (embedding overlap when embeddings are configured, lexical
   fallback otherwise).
4. **Story generation** — for each of the top `stories_per_cycle` (default 5)
   candidates, exactly one `chat_api_call` produces title + 120–200 word body +
   optional extracted metadata. **Every outcome is a row**: `complete`, `empty`,
   or `failed`, mirroring `briefing_service` discipline.
5. **Assembly** — stories attach to one `dreams_collections` row for the local
   date; cycle status `complete` / `partial` (some stories failed) / `failed`
   (no stories produced).

**Event metadata extraction rules** (hallucination guard): `event_date`,
`location`, and `kind` (`content|event|deal|social_opportunity`) are extracted
only from explicit text in the source material; absent evidence stores NULL;
relative phrases ("in two months") anchor to the generation timestamp. The UI
shows extracted dates with the source URL one action away. These fields feed
Phase 2 tracking defaults and the "happening soon" ordering.

**Date-bucket idempotency and catch-up**: one collection per local date
(unique index). The scheduler fires `dreams_cycle` on the configured cadence
(default 24h), but the loop's missed-fire grace is minutes-scale by design — so
Dreams adds its own catch-up: on app boot and on Dreams surface open, if no
collection exists for today's local date and the cadence says one is due, the
cycle runs then. The date-bucket unique index plus an in-flight run lock makes
scheduler fire + catch-up + manual trigger race-free: the first trigger wins
the date; later triggers within the same date **append** stories (up to budget)
rather than regenerate. A cycle killed mid-run must not wedge the date: a
collection row stuck in `generating` older than a reclaim timeout (15 minutes)
is treated as crashed, marked `failed`, and the date becomes eligible for
catch-up regeneration. All timestamps use the shared UTC-helper convention
(ADR-173); `local_date` remains the user's local calendar date for bucketing.

**Blocking-IO discipline**: `perform_websearch` and `chat_api_call` are
synchronous network calls; all Dreams async paths run them under
`asyncio.to_thread` with DB writes grouped into one hop per branch — the exact
pattern `briefing_service` documents and the scheduler loop depends on (one
slow handler pushes every task behind it).

### The surface (Artifacts screen)

Dreams stories become a new type in the Artifacts screen's existing filter
taxonomy ("… | Dreams"), wired the same hand-wired way Daily Reports is: an
own refresh worker with generation guard, a preview card, and the list. The
projection module `Dreams/dreams_view.py` mirrors `daily_reports_view.py`
(read-only, off-thread fetch, generation-checked apply).

Collection layout order: **tracked updates and approaching events first**
(Phase 2), then fresh discoveries. Failed/degraded cycles show their status
row ("today's cycle ran without web search") — a missing collection is a
visible state, never silence.

Selecting a story opens the Dreams detail modal:

- **Dive deeper** — console handoff via `ChatHandoffPayload` (story body +
  source URL as seed context) through the pending-handoff store, following the
  skills/study screen pattern.
- **Keep** — sets `kept`; kept stories pin to the top and survive pruning
  (in-app analog of DreamBeans bookmarking).
- **Export** — write the story (or the whole collection) to a Markdown file.
- **Feedback** — "more like this" / "less like this".
- **Ingest this** (Phase 1) — route the story URL through the existing capture
  pipeline into read-it-later (`collection_capture_items`), making the
  discovery a first-class library citizen. Exact capture-service seam to be
  pinned during planning.
- **Track this** (Phase 2) — see below.

Modal styling uses ADR-150 `$ds-*` tokens exclusively (the
`test_design_token_governance` gate applies); keybindings follow ADR-031
(single-letter htop-style actions; no terminal-convention keys; footer hints
only advertise implemented actions).

### The feedback loop

Every interaction with a story records a `dream_feedback` row (`more`, `less`,
`kept`, `dived`, `exported`, `ingested`, `tracked`). Feedback adjusts **topic
weights only**: boost on `more/kept/dived/ingested/tracked`, decay on `less`,
exponential smoothing, weights decaying toward uniform over weeks.

**Goals are immune**: feedback on a goal-derived story adjusts the *query
angle* for that goal (recorded, consulted by query synthesis) — never the goal
itself. A single bad concert story must not erase "visit Japan".

### The track loop (Phase 2)

"Track this" creates a `dream_tracked_items` row and comes in two mechanisms,
because the user-facing examples are two shapes:

1. **Track a page** (the concert listing, the tickets page): create a
   `subscriptions` row (type `url`, `check_frequency` from track cadence),
   watchlist membership, and an alert rule — all through the **existing**
   URLMonitor/alert pipeline, whose check execution, disposition counting,
   baseline/rebaseline semantics, and run claiming are already battle-tested.
   If the user already subscribes to the source URL, **attach** to the existing
   subscription instead of creating a twin. Untrack removes the Dreams wrapper
   and removes/disables the subscription **only if Dreams created it**
   (`created_by_dreams` flag).
2. **Track a question** ("cheap flights to Japan?", "new tour dates?"):
   recurring synthesized query → search → one LLM judgment "materially changed
   vs. last snapshot?" → a `dream_track_runs` row (disposition vocabulary
   shared with watchlists: `changed/unchanged/baseline/rebaselined/withheld/
   error/skipped`). Re-alert suppression keys on the result digest hash; a
   **rebaseline** action (mirroring the engine's own concept) stops repeat
   notification for an already-acted-on change.

Tracked-item lifecycle: intent (`event|deal|topic`), optional `event_date`
(from story metadata), retire policy (event date + buffer;
`track_quiet_retire_count` consecutive unchanged checks, default 14; manual
untrack), and a concurrency cap (`tracked_item_cap`, default 20). **Baseline semantics are explicit**: the first observation of a tracked
page or question establishes a baseline and does not alert. `withheld`
(bot-walled, JS-heavy pages) is surfaced honestly with question-tracking as
the suggested fallback — the spec does not pretend Ticketmaster will be
politely diffable.

Condition triggers surface in the Dreams "Tracked updates" section
(aggregating the underlying watchlist run dispositions where applicable) and
can promote to a real reminder near an event date (the `reminder_tasks` +
`NotificationDispatchService` path).

**Open item for the Phase 2 plan (named, must not be dropped):** pin down the
per-subscription scheduled-task registration seam — where ADR-019's migration
creates check task rows (UI action vs. service call) — and make Track-this
call that same seam rather than inventing a parallel registration path.

Missed track-check catch-up mirrors the cycle rule: on boot, run checks
past-due within a freshness window; staler ones skip to the next cadence.

### Scheduling, providers, and budgets

- `dreams_handler.py` registers two task types on the existing scheduler:
  `dreams_cycle` (one row, daily cadence) and `dream_track_check` (keyed per
  tracked-item id, mirroring watchlist per-subscription keying).
- Provider/model resolution copies the briefing chain verbatim: `[dreams]
  provider/model` → remembered chat defaults
  (`resolve_remembered_provider_model`). One resolution per cycle, stamped on
  the collection row.
- **Global daily budget shared by both loops**: `max_searches_per_day`
  (default 30) and `max_llm_calls_per_day` (default 60) — provisional numbers,
  revisited during Phase 1 planning against observed costs.
  Over-budget degrades by skipping the lowest-priority work first (exploration
  slots, then oldest tracked checks), and the affected rows record why.
  Per-track cadence floor `track_min_check_interval` (default 12h). Failing
  tracks auto-pause after consecutive failures, mirroring
  `subscriptions.auto_pause_threshold`.

### Data model (`DB/Dreams_DB.py` + migrations)

New subsystem DB module following the repo's per-subsystem pattern (cf.
`Library_Collections_DB.py`), migrations alongside, additive `CREATE TABLE IF
NOT EXISTS` style:

- `dream_interest_profile(id, facet CHECK(facet IN ('topic','goal')), text,
  weight REAL, searchable INTEGER DEFAULT 1, source CHECK(source IN ('user',
  'seed','personal_context','notes','media')), query_angle TEXT NULL,
  created_at, updated_at, last_boosted_at)` — unique on `(facet, text)`.
- `dreams_collections(id, local_date TEXT, status CHECK(status IN
  ('generating','complete','partial','failed')), profile_digest TEXT,
  provider TEXT, model TEXT, story_count INTEGER, trigger CHECK(trigger IN
  ('scheduled','catchup','manual','refresh')), degradation_notes TEXT,
  created_at, completed_at)` — **unique on `local_date`**.
- `dream_stories(id, collection_id FK, title, url, snippet, body, status
  CHECK(status IN ('complete','empty','failed')), source CHECK(source IN
  ('web','watchlist','llm')), kind CHECK(kind IN ('content','event','deal',
  'social_opportunity','unknown')), event_date TEXT NULL, location TEXT NULL,
  matched_topics TEXT(JSON), query TEXT, kept INTEGER DEFAULT 0, kept_at,
  error TEXT NULL, created_at)` — unique index on `(collection_id, url)`.
- `dream_feedback(id, story_id FK, kind CHECK(kind IN ('more','less','kept',
  'dived','exported','ingested','tracked')), created_at)`.
- `dream_seen_items(url TEXT PRIMARY KEY, title_digest, first_seen,
  last_seen)` — pruned on a `seen_item_ttl_days` (default 90) sweep.
- `dream_tracked_items(id, origin_story_id FK NULL, mechanism CHECK(mechanism
  IN ('page','question')), intent CHECK(intent IN ('event','deal','topic')),
  subscription_id NULL FK, query_template TEXT NULL, event_date TEXT NULL,
  cadence_seconds INTEGER, quiet_retire_count INTEGER, status CHECK(status IN
  ('active','paused','retired')), retired_reason TEXT NULL, created_by_dreams
  INTEGER DEFAULT 0, last_checked, created_at, updated_at)`.
- `dream_track_runs(id, tracked_item_id FK, status CHECK(status IN
  ('changed','unchanged','baseline','rebaselined','withheld','error',
  'skipped')), digest_hash TEXT, verdict_note TEXT, notified INTEGER DEFAULT
  0, created_at)`.

### Privacy and trust boundaries

What the machine sends out, exhaustively: (a) synthesized search queries —
which may contain distilled topic names, opted-in goal text, and the region
string; (b) story-generation prompts — containing distilled topics/goals plus
public web snippets; (c) question-track judgments — query plus current
snippet vs. stored digest. **Nothing derived verbatim from notes, media
transcripts, or the encrypted Personal Context store ever leaves.** The
distillate itself lives under private-path discipline.

Enforcement surfaces:

- **"What we'll look for" preview** — the exact queries the next cycle will
  run (goal-derived ones labeled as such), viewable before a scheduled cycle
  and after any profile edit. Same transparency instinct as the Console's
  Next Send preview.
- **Per-goal `searchable` toggle** — the most personal facet gets a per-item
  kill switch; a goal marked unsearchable contributes zero outbound text.
- `[dreams] web_search_enabled` — a global kill switch drops Dreams to
  LLM-knowledge mode (marked degraded) for privacy-paranoid configurations.

This extends, and must be recorded alongside, [ADR-029]'s local-private-data
boundary; ADR-178 carries the boundary statement.

### Configuration (`[dreams]` in config.toml)

`enabled` (default false — Dreams is opt-in), `provider`, `model`,
`stories_per_cycle` (5), `queries_per_cycle` (3), `exploration_slots` (1),
`search_engine`, `web_search_enabled` (true), `region` (""), `cadence_hours`
(24), `catchup_enabled` (true), `watchlist_freshness_hours` (48),
`seen_item_ttl_days` (90), `max_searches_per_day` (30),
`max_llm_calls_per_day` (60), `tracked_item_cap` (20),
`track_min_check_interval_hours` (12), `track_quiet_retire_count` (14). Defaults for the budget keys are set
during Phase 1 planning from observed costs; the spec fixes the mechanisms,
planning fixes the numbers.

## Error handling

- **Search backend failure** falls back down the configured backend chain;
  total failure drops the cycle to `source=llm` mode, visibly marked. Partial
  collections are valid collections (`partial`).
- **LLM failures** are per-story rows (`failed`), never cycle-fatal.
- **Locked Personal Context** degrades the profile (recorded in
  `degradation_notes`), never blocks.
- **Bot-walled tracked pages** surface as `withheld` with a suggested
  question-track fallback.
- **Over-budget** skips lowest-priority work first and records why on the
  affected rows.
- **Missed overnight runs** are covered by the date-bucket catch-up rule; a
  day the app never opens simply has no collection (visible as such).
- **Race conditions** — date-bucket unique index + in-flight run lock; manual
  refresh appends within budget.
- All rows and failures log through loguru with context; Dreams never raises
  into the scheduler loop.

## Testing

Targeted runs only (per AGENTS.md); the full suite is a pre-PR event.

Unit: profile merge/decay math; feedback weight adjustments (incl. goal
immunity); dedupe ledger behavior + TTL sweep; query-synthesis prompt
contract; date bucketing + idempotency + catch-up predicate; race guard;
budget enforcement ordering; event-metadata null-vs-extracted rules; digest
re-alert suppression; retire policies.

Integration: `dreams_cycle` dispatch under a fake clock with mocked
`perform_websearch` + `chat_api_call`; boot catch-up; append-on-refresh;
Artifacts projection wiring; handoff payload emission; Phase 2 — page-track
creation attaching to a pre-existing subscription; question-track run flow
with verdict rows; untrack cleanup semantics; missed track-check catch-up.

UI: token governance (ADR-150) and keybinding/footer (ADR-031) compliance for
the modal and list surfaces.

## Governance

- **ADR:** [ADR-178](../../../backlog/decisions/178-dreams-daily-discovery-and-tracking.md)
  — required (new storage schema; local-private-data boundary extension; new
  provider-facing generation path; tracking contracts). Number 178 chosen
  after verifying 173–177 are all spoken for on other branches — this repo's
  numbering has live collision risk; re-verify at creation.
- Backlog tasks: created at planning time, one per task below.
- Design tokens per ADR-150; keybindings per ADR-031; DB patterns per
  `base_db` + per-subsystem module precedent.

## Phasing

One spec, two implementation plans (Phase 2's plan is written after Phase 1
lands — keeps each plan reviewable).

**Phase 1 — Discover (8 tasks):**
1. `DB/Dreams_DB.py` + migrations + models.
2. Interest-profile builder + settings editing + cache-on-unlock distillate.
3. Query synthesis + discovery (search wrapper, watchlist candidate pool,
   dedupe, ranking).
4. Story service + cycle orchestration (row-per-outcome, budgets, event
   metadata extraction).
5. Scheduler handler + date-bucket catch-up + race guard.
6. Artifacts surface: type wiring + refresh worker + collection list.
7. Detail modal + dive-deeper handoff + keep/export/feedback + the
   "what we'll look for" query preview (Phase 1 covers topic/region-derived
   queries; goal labeling arrives with Phase 2's goals).
8. Ingest action (capture-pipeline seam).

**Phase 2 — Track (5 tasks):**
9. Goals facet + region preview labeling + per-goal `searchable` toggle.
10. Track: page mechanism (subscription attach-or-create + alert rule +
    scheduled-check registration seam — the named open item).
11. Track: question mechanism + `dream_track_runs` + budget integration.
12. Tracked-updates surfacing + happening-soon ordering + reminder promotion.
13. Retirement/pause/caps + track settings.

## Follow-ups (out of scope, filed as future tasks)

- **Guardian × Dreams wellbeing tie-in**
  ([TASK-32917](../../../backlog/tasks/task-32917%20-%20Guardian-x-Dreams-bidirectional-awareness-and-trend-analysis-tie-in-tldw_server-chatbook.md)):
  bidirectional port with tldw_server's Guardian self-monitoring —
  trend/topic/fixation analysis over configured "topics of consideration"
  surfacing humane course-correct notices; Dreams' interest profile and
  feedback loop are the natural signal seams. User-enabled and user-configured
  only; see the task for grounding in the Guardian design doc.
- Conversation-derived interest signals (needs its own privacy pass).
- Home-screen "today's dreams" card.
- Watchlists-screen surfacing of dream-created sources (badge/label).
- Server personalization via the existing `Personalization_Interop` seam.
- Audio-rendered dream collections (briefing TTS reuse).
- Structured event/price integrations if a trustworthy API appears.
