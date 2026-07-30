# Watchlists Briefings & Podcasts — design (spec #2)

**Date:** 2026-07-30
**Status:** phase 1 implemented (2026-07-30); phases 2-4 pending
**Predecessor:** `2026-07-25-watchlists-console-rebuild-design.md` (spec #1), which deferred this
slice: *"Spec #2 covers artifact generation (briefings, 2-speaker podcasts) and its scheduled
delivery."* One correction to that charter, from the user directly: podcasts are **not** fixed at
two speakers — the cast is however many speakers the user wants, from a single narrator to a
multi-character round-table.

## What this builds

A watchlist can be turned into a **briefing**: a structured text digest of what its sources did
since the last one. From a briefing, a **script** can be cast for any speaker roster, and from a
script, **audio**. The user reads briefings in-app, listens in-app, exports markdown, or points a
podcast client at a generated feed directory. Generation is on-demand first; scheduling arrives
last, through the same scheduler seam checks use.

Decided with the user (2026-07-30): all four consumption modes; selection mode is per-watchlist
(`auto` / `curated` / `auto+featured`, default `auto+featured`); on-demand first, scheduled later;
N-speaker presets with per-speaker role prompts, optional character cards, and TTS voice profiles;
LLM is a per-preset choice with the app default as fallback.

**Approach: briefing-first.** The text briefing is the canonical artifact; everything else derives
from it. Casting is a cheap second LLM pass over a *finished* briefing, so changing the roster,
format, or voices never re-runs summarization; audio re-synthesis re-runs zero LLM calls. The
alternative (script-first, text as transcript) was rejected: reading is a first-class mode and a
transcript is not a digest.

## Entity model

| Entity | What it is | Storage |
|---|---|---|
| `briefings` | id, watchlist_id, coverage window (`covers_from`, `covers_to`), selection mode used, preset id (for its LLM/style settings only — the briefing is cast-independent; roster snapshots belong to scripts and audio), model used, markdown body, item/overflow counts, status `generating`/`complete`/`empty`/`failed` + error, created_at | new table, `Subscriptions_DB` |
| `briefing_items` | (briefing_id, item_id, featured) — what a briefing covered | new table |
| `briefing_presets` | name; ordered speaker roster as JSON — each speaker: display name, role prompt, optional character card id, TTS voice-profile reference; LLM provider/model override; style and target-length notes | new table, app-global |
| `briefing_scripts` | briefing_id, preset id + roster snapshot, structured turns JSON, status + error | new table |
| `briefing_audio` | script_id, file path, duration, per-speaker voice-profile snapshot, status + error | new table + audio file on disk |

- Watchlists gain two fields: `briefing_selection_mode` and `default_briefing_preset_id`.
- All tables are additive `CREATE TABLE IF NOT EXISTS` / column-presence `ALTER`s. **No data
  migration exists in this design**, so the TASK-1362 atomicity machinery (`BEGIN IMMEDIATE`) is
  not needed; do not cargo-cult it in.
- Audio files live under the private data dir and are written **only** through the
  `Utils/private_paths` helpers (ADR-029 posture), path recorded in the row.
- **Snapshots make artifacts self-interpreting**: scripts and audio store the roster/voice state
  they were generated under, so editing or deleting a preset never orphans an artifact's meaning.

### The queue flag is global, and never auto-cleared

`queued_for_briefing` exists (ADR-018), is indexed, and has no UI. It gains one: a
"Queue for briefing" action in the reader/Inspector with a visible indicator. It is a **global**
item flag — the same shape as read status, which spec #1 made global deliberately — and a source
can sit in several watchlists. Auto-clearing it when *one* watchlist's briefing covers the item
would silently destroy another watchlist's pending curation. So it is **never cleared by
generation**: selection for watchlist W takes "queued AND NOT already in a briefing of W" (the
junction answers that), and the flag itself is a pool only the user empties. Stated here the way
spec #1 stated global read status: so it is a documented behaviour, not a discovered bug.

## Generation pipeline (phase 1: text, on demand)

1. **Trigger:** a Generate action on the watchlist (button on the Artifacts section + command
   palette entry). One generation per watchlist at a time; a second request while `generating`
   is refused with a toast naming the running one. **Zombie recovery** (TASK-1090's shape): a
   `generating` row not backed by a live worker — found on the next Generate attempt or Artifacts
   load — is failed as `interrupted`, honestly, so a crash can never wedge the guard shut.
2. **Selection:** per the watchlist's mode. `auto`: items in the coverage window. `curated`:
   queued-and-not-yet-covered only — **regardless of the window**: a user who queues a
   three-week-old item wants it in the next briefing. `auto+featured` (default): the union, queued
   items likewise window-exempt, marked `featured` in the junction and given top billing in the
   prompt.
3. **Coverage window — an item-id watermark, not timestamps.** Each `complete` (or `empty`)
   briefing records `covers_through_item_id`; the next window is items with `id >` that watermark
   belonging to the watchlist's current sources. Why ids: item `created_at` has one-second
   resolution (the TASK-1361 lesson), and the item upsert key is
   `(subscription_id, url, content_hash)` — new content is a NEW row with a new id, identical
   re-seen content updates in place — so the watermark is precise, monotonic, and immune to clock
   ties. It also solves the new-source flood for free: a source added to the watchlist later has
   historical items with old ids, auto-excluded; only its post-addition items enter the window
   (`watchlist_sources.added_at` exists but is not needed for this). First briefing only: the last
   7 days by `created_at`, item-capped. A `failed` briefing records no watermark advance —
   **failure never loses items**; the next attempt re-covers the same window. This invariant gets
   a named test in every phase that touches selection.
4. **Prompt assembly is `content_kind`-aware** (TASK-1343 made `content` the *diff* for change
   items): article items contribute "what it says" (title, source, excerpt), change items
   contribute "what changed on the page" (the diff, labelled as such). Per-item excerpt cap and a
   total item cap (~40) keep the call bounded; overflow is **stated in the briefing body**
   ("12 more items arrived in this window and are not covered") — never silent truncation.
5. **One call through `Chat_Functions.chat_api_call` (the app's provider-dispatching chat seam; CLAUDE.md's `chat_with_provider` name is stale documentation — the only callable by that name is an MCP tool shim)** — the preset's provider/model, else the app default. The
   deleted `recursive_summarizer` stays deleted; if real briefings outgrow one call, map-reduce is
   a recorded follow-up, not a phase-1 speculation.
6. **Output:** markdown, sections per theme/source, every claim cited to an item id (rendered as
   links into the reader). A citation whose item was pruned or deleted renders "item no longer
   available" — honest degradation, not a dead control.
7. **Statuses are the observability** (no new `persist_event` events — the ADR-029 amendment
   admits exactly six and this design must not widen a privacy boundary the owner signs):
   `generating` → `complete` / `empty` (window had no items — a row, visible, not an absence) /
   `failed` (+ error text, visible in the tab).

## Casting and audio (phase 2)

- **Script pass:** one LLM call taking the finished briefing plus the roster. A bound character
  card contributes its personality text into that speaker's role prompt. Output contract is fixed:
  a JSON array of `{"speaker": <roster display name>, "text": ...}` turns. Validation is strict
  and failure is honest: an unknown speaker name fails the script artifact naming the name; a
  malformed payload fails naming the parse error. The briefing is never touched by a script
  failure. A preset whose roster names a deleted character card fails the cast at that point,
  naming the card — snapshots protect existing artifacts; this rule protects the cast step. A
  roster of one produces narration through the identical path — no special mode.
- **Audio pass:** per-turn synthesis through the existing TTS adapter registry using each
  speaker's voice profile, long turns chunked by the TTS layer's own `text_processing`, stitched
  and stored with duration. A synthesis failure names the turn and speaker, keeps the script,
  fails only the audio artifact. Output container/format follows what the adapter emits; the
  stitching path and any conversion get verified against the real adapters at plan time rather
  than promised here.
- In-app playback via the existing audio player.

## UI (phases 1–2)

- **The Artifacts section of the Watchlists strip does not currently exist** (TASK-1346: the strip
  has six sections, none Artifacts). Phase 1 **adds** it: a list of briefings for the selected
  watchlist (status, window, counts), a reader-style view of a briefing's markdown, the Generate
  action, and the preset picker; phase 2 adds script/audio children and playback. The wider
  spec-vs-strip reconciliation stays TASK-1346's.
- The reader/Inspector gain "Queue for briefing" (+ indicator). The queue toggle follows the
  established silent-path rules: no full-screen recompose, in-place patch, honest failure toast.
- **The separate Artifacts *screen* is a non-goal** — everything renders in the Watchlists tab, so
  spec #1's chatbook-specific deep-link problem (`pending_artifacts_chatbook_target_id`) is
  dissolved rather than solved.

## Exports and feed (phase 3)

- Markdown export of a briefing via the file picker.
- Podcast delivery is a **feed directory**: `feed.xml` (RSS with enclosures) plus the audio files,
  written to a user-chosen folder. If the app's `[web_server]` is enabled it can serve that
  directory over localhost for podcast clients; serving is a toggle, the directory is the
  deliverable. Export is user-initiated egress of the user's own derived content — deliberate, and
  outside the private-storage boundary by intent.

## Scheduling (phase 4)

A briefing job type registered through the same scheduler seam TASK-1383 unified — real run
records, honest statuses, `empty` rows when nothing is new. Cadence per watchlist via
`automation_definitions`. Same constraint as checks, stated in the UI copy: fires while the app is
open.

## Egress, stated plainly

Generation sends item content (titles, excerpts, diffs) to whichever LLM provider the preset
names. That is the user's configured choice; local providers are the private option. Nothing here
touches the persistent metadata-only log, and no new persisted event names are introduced.

## Error handling ethos

Every artifact carries its own status and error, rendered where the user already looks; silence is
never a state. Failures are scoped to their own stage: a script failure keeps the briefing, an
audio failure keeps the script, a failed generation never advances the coverage window.

## Testing

- Fake exactly three seams: `Chat_Functions.chat_api_call` (scripted responses), the TTS adapter's
  `synthesize`, and the HTTP fetch (the existing `_serve` harness). Everything else real —
  real DB, real selection, real junction writes.
- Named invariant tests: failed-generation-does-not-advance-coverage; global-queue-not-cleared-by-
  another-watchlist's-briefing; unknown-speaker-fails-the-script-by-name; overflow-is-stated-in-
  the-body; citation-to-pruned-item-degrades.
- Mutation checks per behavioural change, per this stream's standing discipline; geometry/UI
  assertions in the real-CSS harness.
- Every new test `pytest.mark.unit` or it is invisible to CI.

## Non-goals

- Reviving any deleted `Subscriptions/` orphan (`briefing_generator`, `recursive_summarizer`,
  `distribution_manager`, `export_manager`, `rss_feed_generator`, `aggregation_engine`).
- The separate Artifacts screen, and any chatbook deep-link work.
- Email/webhook delivery; server-side generation; per-item audio; translation.
- Map-reduce summarization (recorded follow-up if the cap proves too tight in practice).

## Phases

1. **Text briefings on demand** — tables, selection, pipeline, Artifacts section with list +
   reader + Generate, queue-for-briefing affordance. **Preset-less**: `preset_id` NULL, the app's
   default provider, a built-in style — preset CRUD is phase 2's, not to be invented early.
2. **Presets, scripts, audio** — preset CRUD + picker, casting pass, TTS synthesis, playback.
3. **Exports** — markdown export, feed directory (+ optional localhost serving).
4. **Scheduling** — the briefing job type, per-watchlist cadence, quiet `empty` runs.

Each phase is its own plan and lands independently useful.
