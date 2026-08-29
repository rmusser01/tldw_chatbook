# Daily Reports surface and demo design

- **Status:** Approved
- **Date:** 2026-08-29
- **Decision:** [ADR-079](../../../backlog/decisions/079-daily-reports-surface-and-demo-seeding.md) (to be created before implementation begins)
- **Classification:** Architectural
- **Approved product decisions:** live end-to-end demo; seeded setup persists as the user's real first daily report; entry points on Artifacts and Watchlists screens plus a later first-run onboarding card; approach A (briefing-native).

## Summary

Chatbook already has a complete daily-briefing pipeline: watchlists fetch items
from RSS/HN/Reddit/GitHub/YouTube/custom sources, the scheduler runs watchlist
checks and briefing jobs, briefings are generated as text, converted to
multi-speaker cast scripts, and synthesized to a single playable audio file.
Nothing exposes this as a coherent "Daily Report" experience: the Artifacts
screen's Reports slot is a hardcoded `none available` placeholder, a new user
must hand-wire watchlist + preset + schedule before anything happens, and
scheduled briefings complete without notifying anyone.

This design adds two things on top of the existing pipeline, without new
generation logic or a new artifact store:

1. **A Daily Reports surface** — the Artifacts screen's Reports slot lists
   recent briefings across all watchlists (text, audio, playback, jump-to-
   watchlist, keep/export), with an empty state that offers the demo.
2. **A live demo** — a one-click "create your first daily report" flow that
   preflights prerequisites, seeds a real watchlist ("Daily Brief" with
   Hacker News plus two stable RSS sources), a default briefing preset, and a
   24-hour briefing cadence, then runs the real pipeline immediately: check →
   text brief → TTS audio. The seeded setup persists as the user's actual
   first daily report; the scheduler keeps it running from the next day on.

Plus one pipeline gap fix: scheduled briefing completion dispatches a
notification through `NotificationDispatchService`.

## Background

Relevant existing seams (all verified present in the repo):

- `tldw_chatbook/Scheduling/` — `SchedulerLoop` with a generic handler
  registry; `BriefingProjection`
  (`Scheduling/services/briefing_projection.py`) maps each watchlist's
  `briefing_cadence_seconds` into a `briefing_job` scheduled task;
  `BriefingJobHandler` (`Scheduling/scheduler/handlers/briefing_handler.py`)
  generates and auto-keeps the briefing.
- `tldw_chatbook/Subscriptions/` — `LocalWatchlistsService` (checks, item
  upserts), `briefing_service.generate_briefing()` (claim-guarded lifecycle:
  `generating → complete | empty | failed`), `briefing_cast.generate_script()`,
  `briefing_audio.generate_script_audio()` (per-speaker TTS stitched to one
  WAV under `<user_data>/briefing_audio/`), `briefing_voices`,
  `briefing_keep`.
- `tldw_chatbook/DB/Subscriptions_DB.py` — tables `watchlists`,
  `watchlist_sources`, `subscriptions`, `subscription_items`, `briefings`,
  `briefing_items`, `briefing_presets`, `briefing_scripts`, `briefing_audio`,
  `kept_briefings`; `list_briefing_schedules()` already exists.
- `tldw_chatbook/UI/Screens/artifacts_screen.py:610-619` — Reports slot
  placeholder ("Reports: none available").
- `tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py` — per-watchlist
  briefings/scripts/audio list; `UI/Screens/watchlists_collections_screen.py`
  drives generate/synthesize/play actions (`play_audio_file` on the UI
  thread).
- `tldw_chatbook/Notifications/notification_dispatch_service.py` — persisted
  inbox + policy-gated toast delivery; currently only the reminder handler
  dispatches.
- `tldw_chatbook/TTS/` — `TTSService` with seven backends, profile system,
  `SimpleAudioPlayer`; `briefing_audio` preflights stitching availability
  (`PYDUB_AVAILABLE`) and records a `failed` audio row when missing.

ADR alignment: ADR-015 (shell destinations — the Artifacts screen's charter
already includes "reports"), ADR-078 (no second universal artifact database —
Daily Reports is a view over existing briefing tables, not a new store), and
the task-15463 discipline (one `SubscriptionsDB` instance; scheduler and DB
work off the event loop).

## Goals

- Let a new user go from "empty app" to "a real daily news brief, text plus
  audio, that will regenerate tomorrow" in about two minutes, with one click.
- Give scheduled briefings a permanent, discoverable home on the Artifacts
  screen.
- Notify users when a scheduled daily report completes or fails.
- Reuse the existing briefing pipeline end to end; add no parallel generation
  code and no new artifact storage.

## Non-goals

- New scheduler task types, new tables, or a new artifact store (ADR-078
  direction: presentation adapters over canonical owners).
- A "preferred time of day" for briefings. Briefing schedules are rolling
  cadences (`briefing_cadence_seconds`); after a demo run at 14:37 the next
  auto-run fires at 14:37 the next day. A morning-anchored schedule is filed
  as follow-up work (see Follow-ups), not smuggled into the demo as a schema
  change.
- Server-side automation. `AutomationDefinition` execution stays as ADR-076/077
  scopes it; this design is entirely local.
- Changing how briefings are generated, selected, kept, or exported.

## Detailed design

### 1. `DailyReportsView` (read-only aggregation)

New module `tldw_chatbook/Subscriptions/daily_reports_view.py`.

- One new `SubscriptionsDB` read method, e.g.
  `list_recent_briefings(limit=20)`: newest-first join across `briefings` →
  `watchlists` (name), `briefing_scripts` (exists), `briefing_audio` (exists +
  path), kept status. Narrow projection like the items feed (task-15464
  pattern); no `content` blobs in the list, body fetched on open.
- Audio paths resolved through the existing safety guard
  (`audio_file_path_is_safe`) before any playback.
- No writes, no new tables, no caching beyond the screen's own refresh.

### 2. Artifacts screen Reports slot

Replace the `Reports: none available` placeholder with:

- **Empty state** (no briefings exist): "No daily reports yet — create your
  first in ~2 minutes" + the demo CTA button. CTA copy states that the demo
  uses the configured LLM provider (real tokens) and fetches live sources.
- **List state**: recent briefings across all watchlists. Row shows title or
  date, watchlist name, status (`complete | empty | failed`), kept badge,
  audio indicator. Actions: open text (reuse existing briefing body
  rendering), play audio (`SimpleAudioPlayer.play_audio_file`, same UI-thread
  discipline as the Watchlists screen), jump to the owning watchlist's
  artifacts pane, keep/export via existing seams.
- Refresh on mount and via an explicit refresh action; bounded to the view's
  `limit`.

### 3. `DailyReportDemoService` (seed + orchestrate)

New module `tldw_chatbook/Subscriptions/daily_report_demo.py`. Runs as an app
worker (`run_worker`, exclusive) triggered from the Artifacts CTA or the
Watchlists banner. Stages:

1. **Preflight** (abort before any write):
   - Resolve the default LLM provider via the existing
     `default_briefing_provider()` path. Missing → one guidance notification
     deep-linking to Settings; demo exits, nothing seeded.
   - Network reachability check (lightweight, existing patterns).
   - TTS readiness via the same preflight `briefing_audio` uses
     (`PYDUB_AVAILABLE` + roster/default-voice resolution) — informational
     only; the demo proceeds text-first regardless.
2. **Seed** (idempotent): if any watchlist already has a briefing cadence
   (`list_briefing_schedules()`), skip seeding and jump straight to run-now
   for that schedule's watchlist. Otherwise create:
   - Watchlist **"Daily Brief"** with sources: Hacker News (existing scraper)
     + two stable RSS feeds (final feed picks in the implementation plan).
   - A default briefing preset (style notes for a news brief; cast roster
     that resolves with zero user voice profiles — remote backends' default
     voices).
   - `briefing_cadence_seconds` = 86400. `BriefingProjection` picks the
     schedule up on its next queue reload automatically; no scheduler code
     changes.
3. **Run now** (instant gratification; does not wait for the cadence slot):
   - Watchlist check via `LocalWatchlistsService` (off the event loop).
   - `generate_briefing()` — **through the existing claim path**, so a
     concurrently firing scheduler job cannot double-generate the same
     watchlist.
   - If TTS preflight passed: `generate_script()` →
     `generate_script_audio()`. Any audio failure records "audio skipped"
     with a Settings hint; it never fails the demo. The text brief is the
     success criterion.
   - Stage notifications ("Fetching today's stories…", "Writing your
     brief…", "Recording audio…") dispatched via
     `NotificationDispatchService` so the trail persists even if the user
     navigates away.
4. **Finish**: success notification; the Artifacts Reports slot now lists the
   brief. The setup persists — this is the user's real daily report, editable
   and deletable like any watchlist.

### 4. Watchlists screen banner

Dismissible banner on the Watchlists screen: "Turn your watchlists into a
daily brief — try the demo." Visibility rule: shown only while no watchlist
has a briefing cadence configured (same `list_briefing_schedules()` predicate
as demo idempotency and the Artifacts empty state). Dismissal persists in
config (key settled in the implementation plan).

### 5. Completion notifications (pipeline gap fix)

`BriefingJobHandler` dispatches one notification per scheduled run through
`NotificationDispatchService` — new category `"briefing"`: success (with
watchlist name) or failure (with reason). Category-policy gated like
`"reminder"` is today. The interactive demo additionally dispatches its stage
notifications; scheduled daily runs dispatch exactly one completion
notification, never per-stage.

### Data flow

Demo → rows in existing tables only (`watchlists`, `watchlist_sources`,
`subscriptions`, `briefing_presets`) → immediate run writes
`local_watchlist_runs`, `subscription_items`, `briefings`, `briefing_items`,
`briefing_scripts`, `briefing_audio` → `DailyReportsView` reads → Artifacts
screen. From the next cadence tick, `BriefingJobHandler` repeats the run
unattended, auto-keeps, and notifies. One source of truth end to end.

## Error handling

| Situation | Behavior |
| --- | --- |
| No LLM provider configured | Preflight aborts; one guidance notification to Settings; zero rows written. |
| Network failure during fetch | Existing run/briefing failure lifecycle records it; failure notification explains; retry is safe (seed idempotency). |
| No TTS backend / pydub missing | Text brief succeeds; audio marked skipped with a "configure TTS" hint; demo still succeeds. |
| Scheduler fires during demo run | Claim machinery in `generate_briefing()` serializes; exactly one generation per watchlist. |
| App quits mid-demo | `fail_interrupted_briefings()` recovers zombie rows; the seeded schedule survives and the next day's scheduler run heals. |
| User renames the "Daily Brief" watchlist | Harmless — idempotency keys on configured briefing schedules, not names. |
| User deletes the demo watchlist | Schedule goes with it; Artifacts slot returns to the empty state with the CTA. |

## Testing

Targeted runs only (repo policy: no full sweeps unless requested).

- **Unit**: `list_recent_briefings` ordering/projection on in-memory
  `SubscriptionsDB`; seeding idempotency (second call skips); preflight
  branches (no provider → abort with no writes).
- **Integration**: demo service against in-memory DB with faked
  `chat_api_call` and stubbed source fetches → asserts `briefings` /
  `briefing_scripts` / `briefing_audio` rows, notification dispatches, and
  that `BriefingProjection` computes next_run ≈ last completion + cadence.
- **UI** (Textual pilot): Artifacts slot empty-CTA vs. seeded-list states;
  Watchlists banner visibility predicate; audio playback action routes
  through the path guard.
- **Live verification** once, per `lessons-live-verification.md`: run the
  real app, real HN fetch, real LLM brief, real TTS audio; capture the
  transcript/screenshots as evidence.

## Governance

ADR required: yes — `backlog/decisions/079-daily-reports-surface-and-demo-seeding.md`,
created before implementation begins (number 079 verified against the
decisions directory: 077 is reserved by the pending server-offload rename per
task-19610, 078 exists). Records: daily reports are briefings (one artifact
owner, no second store per ADR-078); the demo writes real persistent data by
design; the `"briefing"` notification category; Artifacts-screen Reports slot
is fed by a read-only view. Links ADR-015 and ADR-078.

## Phasing

1. `DailyReportsView` + Artifacts Reports slot + `"briefing"` completion
   notifications.
2. `DailyReportDemoService` + Artifacts CTA + Watchlists banner.
3. First-run onboarding card offering the demo (touches startup flow; last).

## Follow-ups (out of scope, filed as future tasks)

- Morning-anchored briefing schedules ("preferred time of day" — needs a
  small schema + projection change).
- Audio format/streaming improvements beyond the existing single-WAV model.
- Server-side automation execution (owned by ADR-076/077).
