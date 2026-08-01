# Watchlists Briefings & Podcasts — design (spec #2)

**Date:** 2026-07-30
**Status:** phases 1-3 implemented (2026-08-01); phase 4 pending

**Phase 1 delivery notes (2026-07-30):** two deferrals from the phase 1 plan, both confirmed by
the project owner (2026-07-30):

- **Selection-mode picker.** `watchlists.briefing_selection_mode` ships with its default
  (`auto_featured`) and no writer anywhere in the UI -- `auto` and `curated` are unreachable until
  phase 2's preset/mode UI adds a picker.
- **Citations.** Phase 1 renders `[item N]` markers as plain text (`hyperlinks=False` by
  constraint -- see the Artifacts section). Links-into-reader and pruned-item degradation,
  including the named `citation-to-pruned-item-degrades` invariant test, move to phase 2.

**Phase 2a delivery notes (2026-07-31):** presets, script casting, the selection-mode picker, and
citations shipped on `feat/briefings-phase-2` -- both phase-1 deferrals above are retired. Audio
synthesis split out to phase 2b (task-1630) after the adapter reality below surfaced at plan time
(this design's own "verified against the real adapters at plan time" caveat, see "Casting and
audio" below):

1. `synthesize()` returns a byte-stream `TTSAudioResponse` the caller must drain and `aclose()`
   (it is also an async context manager -- prefer `async with`).
2. Legacy adapters (kokoro/openai/elevenlabs/chatterbox/higgs/alltalk) reject a plain
   `TTSRequest` -- per-call synthesis goes through `generate_audio_stream(OpenAISpeechRequest,
   internal_model_id)`.
3. `text_processing`'s chunking has zero live callers -- callers must chunk AND stitch themselves.
4. The only existing stitcher is naive byte-concat (wrong for WAV headers) -- a real pydub
   decode-and-concat primitive must be written.
5. `private_paths` has no binary append/stream/move helper -- storage is buffer-whole-then-
   `atomic_private_write_bytes` or a new helper.

**Re-verified against dev 2026-07-31 (`8b7fa5eb6`), after ~2,000 lines of TTS character-assignment
work merged from another workstream.** Findings 1, 3 and 4 hold verbatim. Three corrections, and
one new constraint that bounds what phase 2b can promise:

- **Finding 2, corrected.** The adapter-level rejection stands (`legacy_bridge.py:681`), but a
  public all-provider entry point now exists: `TTSService.synthesize_default(text=...,
  voice_override=...)` builds the legacy option pair itself (`request_admission.py:314,356`).
  It reads provider/model/format/speed from the app's global `TTSPreferencesSnapshot` and accepts
  only a voice override, so it is a **one-voice-for-everyone** path -- it cannot express a roster.
  The new `character_request_resolver.py` does not close this either: it is hard-fenced to
  `audio_cpp` at two levels (`character_request_resolver.py:85`, `TTS_Generation.py:562`).
- **Finding 5, corrected.** `open_private_binary` **does** exist (`private_paths.py:864`) but is
  **read-only** (`O_RDONLY`, `:28`) -- useful for playback-side reads, not for writing. A text
  append stream exists (`:767`); the binary equivalent does not, and there is no public move
  helper. The write decision stands: buffer-whole-then-`atomic_private_write_bytes`, or add
  `open_private_binary_append` modelled on the text one.
- **The id-builder is now duplicated and the two copies disagree.**
  `stts_events.py:737` derives kokoro's engine suffix from `options["use_onnx"]` and alltalk's id
  from `snapshot.model_id`; `request_admission.py:356` hardcodes `local_kokoro_default_onnx` and
  `alltalk_default`. Phase 2b must promote one to a public builder rather than writing a third.
- **New constraint: per-speaker exact voices work only on `audio_cpp` today.** `synthesize_exact`
  (the only path that guarantees the voice you asked for is the voice you got) refuses every other
  provider. A legacy-provider roster is reachable only through raw
  `generate_audio_stream(OpenAISpeechRequest, internal_model_id)`, which returns a bare
  `AsyncIterator[bytes]` with no response contract to validate against. Phase 2b must either scope
  multi-voice rosters to `audio_cpp` and say so in the UI, or own that validation itself.
- **Audio snapshots need more than 2a records.** `briefing_scripts.roster_snapshot_json` stores
  `voice_profile_id` (a stringified profile UUID) and is deliberately immutable. For audio to stay
  self-interpreting the `briefing_audio` row must snapshot the profile's **`revision`** plus the
  denormalized selection -- `CharacterTTSRequestResolution` already carries `profile_revision` for
  exactly this purpose. `profile_portability.py` is the right shape precedent but not a reusable
  mechanism (it is `audio_cpp`-only and drops `revision`).
- **Reference implementation to transcribe, not reinvent:** `TTSEventHandler._generate_tts`
  (`Event_Handlers/TTS_Events/tts_events.py:731`) already solves response-contract validation,
  batched artifact appends, `aclose()`-in-`finally` that preserves the primary exception,
  cancellation cleanup, bounded error copy, and metrics. Playback, rate limiting and cooldown
  admission all exist too. The genuinely new surfaces are: a general decode-and-concat stitcher,
  a `TTSProfileService.get_profile(UUID)` passthrough (the repository has one, the service does
  not expose it), the `briefing_audio` table, and the binary-append decision above.
6. `briefing_presets` ships a single free-text `style_notes` field, not the separate "style and
   target-length notes" the entity table below promises: target-length guidance ships folded into
   that same free-text field for 2a, with no dedicated field of its own; a dedicated field is 2b's
   call if audio pacing turns out to need one.

**Phase 2b delivery notes (2026-07-31):** the audio half (task-1630) shipped on
`feat/briefings-phase-2b` -- phase 2 is now complete. `briefing_audio` table + CRUD; a per-turn
`synthesize_turn` reached through either `TTSService.synthesize_exact` (the `audio_cpp` path) or a
newly-public `TTS/legacy_request_builder.build_legacy_speech_request` (every other provider); a
real pydub decode-and-concat stitcher (`TTS/audio_stitch.concat_wav_segments`, WAV-first); the
`generate_script_audio` orchestrator plus a `fail_interrupted_audio` zombie sweep matching
`briefing_cast`'s own; and Synthesize/Play/Stop wired into the Artifacts pane's existing audio
player. Two decisions the plan left open, made explicit:

- **All-providers.** The owner chose per-speaker voices on every configured TTS provider, not
  scoped to `audio_cpp` alone. This has one real consequence worth stating plainly: per-speaker
  *exact-provenance* synthesis -- the guarantee that the voice you asked for is the voice you
  got, backed by a response snapshot to check it against -- remains `audio_cpp`-only at the
  platform level; `synthesize_exact` is the only call with that contract, and every legacy
  provider is reached through `generate_audio_stream`, which returns a bare byte stream with
  nothing to validate against. Phase 2b does not change that platform fact -- it validates legacy
  responses itself instead: non-empty bytes and a payload that decodes as WAV, both raising a
  `TurnSynthesisError` naming the speaker and turn on failure. That is the honest statement of
  what the guarantee is on each provider -- field-for-field provenance on `audio_cpp`, "it
  produced *some* well-formed audio" everywhere else -- not a claim that legacy providers now
  carry the same assurance `audio_cpp` does.
- **Buffer-whole storage.** A correct decode-and-concat must already hold every turn's decoded
  audio in memory to join it, so the whole finished payload exists in memory before there is
  anything to write at all -- a streaming append would still need a full re-encode pass over
  everything written so far, and gains nothing. `Utils/private_paths` also has no binary append
  call (`open_private_binary` is `O_RDONLY`); only a text append stream exists. The payload is
  therefore written once, atomically, via `atomic_private_write_bytes`, with cleanup on any
  failure that lands after the write succeeds.

Two smaller findings worth recording: `audioop-lts` is now a declared dependency
(`pyproject.toml`, three optional-extra groups) -- pydub needs it once the stdlib `audioop` module
is removed on Python 3.13+, and phase 2b's stitcher is 2b's first real caller of pydub in this
codebase. And `Event_Handlers/STTS_Events/stts_events.py`'s `_legacy_internal_model_id` --  the
live playground's own copy of the id-derivation logic `legacy_request_builder` promotes --
continues to diverge by design (it derives kokoro's onnx/pytorch suffix from live playground
options and alltalk's from the requested model id, where the builder uses fixed constants); both
sides carry a cross-reference comment (the TASK-1393 pact convention) so the two are never mistaken
for interchangeable and never converged without updating both call sites' expectations.

**Phase 3 delivery notes (2026-08-01):** markdown export and the podcast feed directory (Tasks
1-5) shipped on `feat/briefings-phase-3`. Four things worth recording against this design's
original text:

- **Localhost serving is cut, by the project owner's decision — the spec's premise was false.**
  This design's "Exports and feed" section above says "if the app's `[web_server]` is enabled it
  can serve that directory over localhost for podcast clients; serving is a toggle." At plan time
  that toggle turned out not to exist: `[web_server]` is **textual-serve** — a mutually exclusive
  process mode that serves the *TUI itself* to a browser (`app.py` returns instead of running the
  TUI when it is engaged), whose only static route is hardcoded to textual-serve's own assets, and
  whose `enabled` key is read by no code at all anywhere in the app. There was never a server
  running alongside the TUI for this feature to toggle. Consequently the directory is the whole
  deliverable, exactly as this design's own wording already said ("the directory is the
  deliverable"); a user-run static server (e.g. `python -m http.server` from the exported folder)
  is the documented path for pointing a podcast client at it. See task-1760 for the scoped,
  net-new follow-up (a standalone opt-in static server) this finding produced.
- **The feed directory is self-contained by design.** `briefing_feed.build_feed_xml` writes
  enclosure URLs as **bare relative filenames**, not absolute paths or `file://` URIs. This is
  deliberate, not an oversight: a relative URL means the folder can be copied, zipped, synced to
  another machine, or handed to someone else, and it still resolves correctly wherever it lands —
  no home directory, username, or local filesystem layout ever leaks into `feed.xml`.
- **Exported files are `0o644` by deliberate decision — recorded as Decision 4 in
  `Subscriptions/briefing_export.py`'s module docstring.** Audio inside the app's private storage
  (`briefing_audio_dir()`) is written `0o600` by `atomic_private_write_bytes`; naively inheriting
  that mode into an export would make a folder the user explicitly chose *in order to share it*
  readable only by the exporting account — silently defeating the point of exporting at all. The
  fixed mode is applied explicitly (not derived from umask): umask expresses a default for
  arbitrary file creation, not intent for a folder picked for sharing, and honouring a hardened
  umask here would hand the most security-conscious users a silently unreadable feed. The real
  access boundary is unchanged — the destination directory's own permissions (left untouched by
  Decision 1 in the same docstring) remain what actually gates access; a `0o644` file inside a
  `0o700` directory stays unreachable to anyone else regardless.
- **§Testing's "every new test `pytest.mark.unit` or it is invisible to CI" is stale.** That was
  true when this design was written; since task-1465, CI instead runs `pytest Tests
  --ignore=Tests/UI` plus a separate `pytest Tests/UI` pass, neither with marker selection, so an
  unmarked test is collected and run either way. What actually matters now is `--strict-markers`:
  an unregistered `@pytest.mark.*` name is a collection **error**, not a silent no-op, so the
  bar is "use a registered marker (or none)," not "must be `unit`."

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
