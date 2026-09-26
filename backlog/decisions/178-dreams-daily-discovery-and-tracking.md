# ADR-178: Dreams daily discovery and tracking

Status: Proposed (2026-09-22) — design spec reviewed twice against code;
implementation pending.
Date: 2026-09-22
Companion spec: [2026-09-22 Dreams design](../../Docs/superpowers/specs/2026-09-22-dreams-daily-discovery-design.md)
Related: [ADR-029](029-local-private-data-boundary.md) (local private data
boundary — extended here), [ADR-019](019-watchlist-scheduler-migration.md)
(check scheduling Dreams attaches to), ADR-079 (Daily Reports / Artifacts
slot precedent), ADR-150 (design tokens), ADR-031 (keybinding conventions)

## Decision

Add a **Dreams** subsystem (`tldw_chatbook/Dreams/` + `DB/Dreams_DB.py` + a
scheduler handler) with two loops built on existing machinery:

1. **Discover** — a daily, date-bucketed collection of story-shaped
   discoveries of *new* content/events/deals, found by live web search plus
   fresh watchlist items, steered by a local interest profile (decaying
   topics + persistent goals + user-set region) and a local feedback loop
   with a guaranteed exploration budget. Generation follows the briefing
   pattern: one `chat_api_call` per story, every outcome a row, provider
   resolution via `[dreams]` config → remembered chat defaults, all blocking
   calls under `asyncio.to_thread`.
2. **Track** — "Track this" on a story becomes either (a) a `url`-type
   subscription + watchlist membership + alert rule through the existing
   URLMonitor/alert pipeline (attaching to a pre-existing subscription when
   the source URL matches), or (b) a recurring question-track: synthesized
   query → search → LLM-judged material-change verdict against a stored
   digest, with row-per-run in the watchlist disposition vocabulary. Tracked
   items carry intent, event date, retire policies, and a concurrency cap;
   both loops share one global daily search/LLM budget.

Dreams surfaces on the Artifacts screen as a new artifact type (Daily
Reports wiring pattern) with a detail modal offering dive-deeper (console
handoff), keep, export, feedback, ingest-to-library, and track.

## Context

Google Labs' DreamBeans (expanded Sept 2026) generates personalized daily
story collections from the user's connected data — forward-looking discovery,
not rediscovery. Chatbook already owns the supporting cast: the briefing
pipeline's generation discipline, the scheduler and its handler registry, a
battle-tested watchlist check engine with dispositions and run claiming, a
multi-backend web search seam, console handoffs, and the Artifacts
projection pattern. Nothing discovers *new* things for the user, and no local
feedback loop steers any content selection. The genuinely new parts are the
interest profile, the discovery pipeline, the feedback loop, and tracked-item
lifecycle — everything else is a proven pattern reused.

Two repo-specific forces shaped the decision:

- **The scheduler's missed-fire grace is minutes-scale** (task-18937), so any
  "daily" product must add its own date-bucket idempotency + boot catch-up or
  a closed laptop silently skips days.
- **Personal Context key custody may be passphrase-wrapped**
  (`ProfileLockedError`), so an unattended cycle must treat the profile as a
  cache-on-unlock, degrading signal — never a hard dependency.

## Contracts

1. **Outbound payload boundary (extends ADR-029):** raw notes, media
   transcripts, and Personal Context contents never leave the machine. What
   leaves, exhaustively: synthesized search queries (distilled topic names,
   opted-in goal text, region string), story prompts (distillate + public
   web snippets), and question-track verdict prompts. Every goal carries a
   `searchable` kill switch; the "what we'll look for" preview shows the
   exact next-cycle queries verbatim.
2. **Feedback immutability of goals:** feedback adjusts topic weights and
   per-goal *query angles* only; goals change exclusively by direct user
   edit. Exploration slots (ε-greedy) are structurally reserved every cycle.
3. **Generation discipline:** one `chat_api_call` per story, row-per-outcome
   (`complete/empty/failed`), collections dated by unique local date,
   in-flight run lock, append-within-budget on repeat triggers, all blocking
   network calls thread-offloaded with grouped DB write hops.
4. **Tracking attaches, not duplicates:** page-tracking reuses an existing
   subscription for the same source URL when present; untrack removes a
   Dreams-created subscription but never one the user already had.
   First observation is a baseline (no alert); re-alerting is suppressed by
   digest hash; statuses use the watchlist disposition vocabulary.
5. **Budget:** both loops share `max_searches_per_day` / `max_llm_calls_per_day`
   with lowest-priority-first degradation recorded on affected rows; tracked
   checks have a cadence floor and auto-pause after consecutive failures.
6. **Event metadata honesty:** `event_date`/`location`/`kind` extract only
   from explicit source text (NULL otherwise, relative dates anchored to
   generation time) — hallucinated dates poisoning tracking expiry is a named
   failure mode this contract exists to prevent.
7. **Permanent exclusion:** no person-specific tracking or scraping of
   people; no geolocation (region is user-typed text); no server-side
   personalization.

## Consequences

- New storage (`DB/Dreams_DB.py`: profile, collections, stories, feedback,
  seen-items, tracked items, track runs) with migrations; a new provider-
  facing generation path; one more Artifacts artifact type and modal.
- Daily background search + LLM spend becomes a real cost the user can now
  incur by opting in — bounded by config budgets and off by default.
- Watchlists gain a second producer (Dreams creating subscriptions), so
  dream-created sources carry a `created_by_dreams` marker for cleanup
  semantics.
- ADR-029's boundary statement grows the Dreams outbound-payload list above;
  the cached distillate lives under private-path discipline.

## Alternatives considered

- **Extend briefings (rejected):** dreams-as-briefing-preset over a special
  watchlist. Briefings summarize *known* watchlist items; Dreams must
  *discover*, and neither the interest profile nor the feedback loop fits the
  preset model.
- **Resurfacing engine (rejected by product owner):** summarizing what the
  user already has — the library is the signal, not the corpus.
- **Watchlists-only tracking for questions (rejected):** question-tracking is
  condition-checking over searches, not source subscription; forcing it into
  feeds would mismodel the intent. Page-tracking, by contrast, *does* reuse
  watchlists wholesale.
- **Server-side personalization (rejected):** the local `Personalization_
  Interop` seam exists, but a local, inspectable, reversible loop fits the
  privacy posture and works offline of any server.
- **LLM-knowledge-only discovery (kept as fallback only):** cheapest, but
  cannot surface new/timely content and goes generic — kept as the degraded
  mode, not the product.
