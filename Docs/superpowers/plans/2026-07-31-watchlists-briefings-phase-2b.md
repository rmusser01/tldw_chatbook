# Watchlists Briefings Phase 2b — Audio Synthesis, Stitching, Playback

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A cast script becomes a single playable audio file in which every speaker is voiced by their own TTS profile, on any configured provider.

**Architecture:** Mirrors phase 2a's shape — a dumb DB layer, a service module whose only faked seam is synthesis, and UI that dispatches workers and never touches a DB on the event loop. Two provider paths converge behind one internal helper: `audio_cpp` goes through `synthesize_exact` (which validates the selection for us), every legacy provider goes through the raw `generate_audio_stream` and we do that validation ourselves. Everything is synthesized as WAV, stitched by decode-and-concat, and written once as a whole payload.

**Tech Stack:** Python 3.11, Textual, SQLite, pydub + ffmpeg (both already present), pytest (`.venv/bin/python -m pytest`, plain output).

## Global Constraints

- pytest is the ONLY python entry point for this repo's code. Never bare `python -c` importing `tldw_chatbook` (it loads the user's live config).
- Never `git stash`; never `git checkout --`/`git restore` to revert (Edit-tool reverts only); never any `git worktree` command; never `-q` with pytest.
- Never hand-edit `tldw_chatbook/css/tldw_cli_modular.tcss` — edit `css/features/_watchlists.tcss` and regenerate via `cd tldw_chatbook/css && ../../.venv/bin/python build_css.py`.
- All DB calls from the screen go through `asyncio.to_thread`. Workers: `group=` always set; guard flags claimed at dispatch, cleared in `finally`.
- Toasts interpolating any value: `markup=False`. Speaker names, turn text and provider errors are model/user data — they never reach a markup parser; use `rich.text.Text`.
- Exception logging is type-only (`type(exc).__name__`); never `logger.opt(exception=True)`; never log turn text, prompts, or roster contents.
- No new `persist_event` event names (ADR-029 admits exactly six).
- Every new test carries `pytest.mark.unit` (or its file's existing `pytestmark`); an unmarked test in `Tests/Watchlists` is collected by nothing.
- Every behavioural change gets a revert-confirm-RED-restore mutation check (Edit-tool reverts only). Green-under-mutation is acceptable only when documented non-load-bearing with a cross-reference to the test carrying the claim.
- Additive DDL only (`CREATE TABLE IF NOT EXISTS`), no `BEGIN IMMEDIATE` — the spec forbids cargo-culting the TASK-1362 machinery.
- Spec: `Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md` §"Casting and audio (phase 2)" audio half + §"Error handling ethos". Task: `backlog/tasks/task-1630`.

## Decisions locked before implementation (do not relitigate; DO pin with tests)

1. **All providers, not just `audio_cpp`** (user decision, 2026-07-31). Per-speaker voices must work on kokoro/openai/elevenlabs/chatterbox/higgs/alltalk. `synthesize_exact` refuses every provider but `audio_cpp` (`TTS_Generation.py:562`), so legacy speakers go through `TTSService.generate_audio_stream(request, internal_model_id)` — which returns a bare `AsyncIterator[bytes]` with no response contract — and **we validate the selection ourselves** (Task 5).
2. **WAV everywhere.** Every synthesis request asks for `response_format="wav"`, matching `_generate_legacy`'s existing wav-first discipline. Rationale: `audio_cpp` accepts nothing else (`adapters/audio_cpp.py:531`), pydub decodes wav without codec surprises, and the system audio players all handle it. No transcoding in 2b.
3. **Buffer whole, write once.** Audio is written with `atomic_private_write_bytes` after stitching, NOT streamed/appended. Rationale: a correct decode-and-concat must hold every segment to concatenate them, so the finished artifact is already in memory; a streaming append would still require a full re-encode pass, and `private_paths` has no binary append helper (`open_private_binary` at `:864` is `O_RDONLY`). This satisfies AC #4's "decision made explicit and justified".
4. **Snapshot the revision.** `briefing_audio` stores a per-speaker voice snapshot including the profile's `revision`, not just its id — a profile edited after synthesis must not silently reinterpret an existing artifact (the spec's self-interpreting-artifacts rule).
5. **Transcribe, don't reinvent.** `TTSEventHandler._generate_tts` (`Event_Handlers/TTS_Events/tts_events.py:731`) already solves drain-with-`aclose()`-in-`finally`-preserving-the-primary-exception, response-contract validation, and bounded error copy. Read it before writing Task 5.

## File Structure

- `tldw_chatbook/DB/Subscriptions_DB.py` — `briefing_audio` DDL + CRUD (Task 1)
- `tldw_chatbook/TTS/legacy_request_builder.py` — NEW: the public builder promoted out of `request_admission._legacy_request` (Task 2)
- `tldw_chatbook/TTS/request_admission.py` — `_legacy_request` delegates to it (Task 2)
- `tldw_chatbook/TTS/audio_stitch.py` — NEW: decode-and-concat + duration (Task 3)
- `tldw_chatbook/TTS/profile_service.py` — `get_profile` passthrough (Task 4)
- `tldw_chatbook/Subscriptions/briefing_voices.py` — NEW: roster speaker → concrete voice selection + snapshot (Task 4)
- `tldw_chatbook/Subscriptions/briefing_audio.py` — NEW: per-turn synthesis + the pipeline (Tasks 5, 6)
- `tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py` + `UI/Screens/watchlists_collections_screen.py` — Synthesize action, audio row, playback (Task 7)

Where a step shows a contract instead of a full body, the full body is mandatory and the named precedent must be read first (the phase-1/2a convention).

---

### Task 1: `briefing_audio` table and CRUD

**Files:**
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py` (DDL beside the `briefing_scripts` block at ~`:791`; methods beside `list_briefing_scripts` at ~`:2077`)
- Test: `Tests/Subscriptions/test_briefing_audio_db.py` (new)

**Interfaces — Produces:**
```python
create_briefing_audio(self, script_id: int, *, voice_snapshot_json: str,
                      status: str = "generating") -> int
update_briefing_audio(self, audio_id: int, **fields) -> None   # allowlist: status, error,
                      # file_path, duration_seconds, turn_count, updated_at
get_briefing_audio(self, audio_id: int) -> Optional[Dict[str, Any]]
list_briefing_audio(self, script_id: int, *, limit: int = 200,
                    offset: int = 0) -> List[Dict[str, Any]]   # newest first
```

- [ ] **Step 1: Read the precedent.** `create_briefing_script`/`update_briefing_script`/`get_briefing_script`/`list_briefing_scripts` (`Subscriptions_DB.py:1979-2100`) are the byte-for-byte model: every operation inside `with self.transaction() as conn:` (reads included — Qodo rule 1011851), allowlist tuple PLUS `sql_validation.validate_identifier` on each key, Google docstrings, `limit`/`offset` pagination.

- [ ] **Step 2: Write the failing tests.** Harness as in `Tests/Subscriptions/test_briefing_audio_db.py`'s siblings (real `SubscriptionsDB` on `tmp_path`, `pytestmark = pytest.mark.unit`). Cases: round-trip with NULL `file_path`/`duration_seconds`; unknown update key raises `ValueError` naming it; `list_briefing_audio` newest-first; pagination (`limit`+5 rows → exactly `limit`; `offset` walks without gaps); `ON DELETE CASCADE` removes audio rows when the parent script is deleted (assert `PRAGMA foreign_keys` is on **and** observe the cascade — the Task-1 phase-2a precedent); `voice_snapshot_json` is NOT in the update allowlist (write-once, like `roster_snapshot_json`).

- [ ] **Step 3: Run, confirm RED.**

- [ ] **Step 4: Implement.**
```sql
CREATE TABLE IF NOT EXISTS briefing_audio (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    script_id INTEGER NOT NULL REFERENCES briefing_scripts(id) ON DELETE CASCADE,
    voice_snapshot_json TEXT NOT NULL,
    file_path TEXT,
    duration_seconds REAL,
    turn_count INTEGER,
    status TEXT NOT NULL DEFAULT 'generating',
    error TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
)
```
plus `CREATE INDEX IF NOT EXISTS idx_briefing_audio_script ON briefing_audio(script_id, status)`.

- [ ] **Step 5: Green + mutations.** (a) Drop `validate_identifier` from `update_briefing_audio` → the divergence test (allowlist-passes-but-validator-rejects, the phase-2a Task-1 shape) REDs; restore. (b) Remove the LIMIT → the pagination test REDs; restore.

- [ ] **Step 6: Commit** `feat(briefings): briefing_audio table and CRUD`.

---

### Task 2: A public legacy request builder

**Files:**
- Create: `tldw_chatbook/TTS/legacy_request_builder.py`
- Modify: `tldw_chatbook/TTS/request_admission.py:356-396` (`_legacy_request` delegates)
- Test: `Tests/TTS/test_legacy_request_builder.py` (new)

**Interfaces — Produces:**
```python
def build_legacy_speech_request(*, provider_id: str, model_id: str, voice: str,
                                text: str, response_format: str = "wav",
                                speed: float = 1.0) -> tuple[OpenAISpeechRequest, str]
    # returns (request, internal_model_id); raises ValueError naming the problem when
    # voice is empty (legacy providers require an exact voice) or provider_id is unknown
```

- [ ] **Step 1: Read both existing copies and note that they disagree.** `request_admission.py:356` `_legacy_request` derives ids from a `TTSPreferencesSnapshot` (global prefs); `Event_Handlers/STTS_Events/stts_events.py:737` `_legacy_internal_model_id` derives kokoro's suffix from `options["use_onnx"]` and alltalk's from `snapshot.model_id`. **Your new builder takes explicit selection fields, not a preferences snapshot** — that is the whole point, since 2b needs one request per speaker, not one per app.

- [ ] **Step 2: Write the failing tests.** Pin the id table exactly as `request_admission` has it today, since that copy is the one being converged: `openai` → `openai_official_{model}`, `elevenlabs` → `elevenlabs_{model}`, `kokoro` → `local_kokoro_default_onnx`, `chatterbox` → `local_chatterbox_default`, `higgs` → `local_higgs_v2`, `alltalk` → `alltalk_default`, anything else → the request's model. Also: the model/format overrides (`_LEGACY_MODEL_OVERRIDES`, `_LEGACY_FORMAT_OVERRIDES`) still apply; voice is lowercased; an empty/None voice raises `ValueError` naming the requirement; an invalid `response_format` falls back to `"wav"`. **Round-trip test:** every id the builder emits resolves through `legacy_bridge.resolve_legacy_route(internal_model_id)` without raising — this is the assertion that would have caught the two copies drifting.

- [ ] **Step 3: Run, confirm RED.**

- [ ] **Step 4: Implement**, moving the id/override logic verbatim out of `_legacy_request`, then rewrite `_legacy_request` as a thin adapter that reads the fields off `preferences` and calls the new builder. `request_admission`'s existing tests must stay green **unmodified** — that is the proof the promotion changed no behaviour.

- [ ] **Step 5: Green + mutation.** Change one emitted id (e.g. `local_kokoro_default_onnx` → `local_kokoro_default`) → the round-trip test REDs; restore. Run `Tests/TTS/` in full.

- [ ] **Step 6: Cross-reference the third copy.** Add a comment above `stts_events.py:737`'s `_legacy_internal_model_id` naming `TTS/legacy_request_builder.build_legacy_speech_request` and stating that this copy derives kokoro/alltalk ids differently on purpose (from live playground options), so the two are NOT interchangeable — greppable both ways, the TASK-1393 pact convention. **Do not change its behaviour**; it serves the playground, not briefings.

- [ ] **Step 7: Commit** `feat(tts): public legacy speech-request builder`.

---

### Task 3: Decode-and-concat stitching, and duration

**Files:**
- Create: `tldw_chatbook/TTS/audio_stitch.py`
- Test: `Tests/TTS/test_audio_stitch.py` (new)

**Interfaces — Produces:**
```python
class AudioStitchError(RuntimeError): ...
def concat_wav_segments(segments: Sequence[bytes], *,
                        gap_ms: int = 350) -> bytes
    # decode each WAV segment, concatenate with `gap_ms` of silence between
    # (never after the last), re-encode once as WAV. Raises AudioStitchError
    # naming the 0-based segment index when a segment will not decode.
def wav_duration_seconds(payload: bytes) -> float
```

- [ ] **Step 1: Read what exists and why it is wrong.** `audiobook_generator._combine_segments:638` does `combined += segment` on encoded bytes — for WAV that yields a file whose RIFF header describes only the first segment, so players report the wrong length and most stop early. The only correct concat in the repo is buried in `AudioService.create_m4b_with_chapters` (`audio_service.py:340-346`) and is M4B-specific. You are writing the general one. `pydub` and `ffmpeg` are both installed.

- [ ] **Step 2: Write the failing tests.** Build real WAV inputs in-process with `pydub.AudioSegment.silent(duration=..., frame_rate=...)` exported to `BytesIO` — no fixtures on disk, no network. Cases:
  - two 500 ms segments with `gap_ms=200` → `wav_duration_seconds(result)` ≈ 1.2 s (tolerance ±0.05);
  - **the header assertion that catches naive concat**: decode the result and assert its frame count matches the duration — a byte-concatenated file decodes to only the first segment, so this REDs against the naive implementation;
  - one segment → returned unchanged in duration, and no trailing gap;
  - empty sequence → `AudioStitchError`;
  - a segment of `b"not audio"` at index 1 → `AudioStitchError` whose message contains `1`;
  - mixed sample rates (8 kHz + 44.1 kHz) → result decodes and its duration is the sum (pydub resamples on concat — assert the behaviour rather than assuming it).

- [ ] **Step 3: Run, confirm RED.**

- [ ] **Step 4: Implement** with `AudioSegment.from_file(BytesIO(seg), format="wav")`, `AudioSegment.silent(duration=gap_ms, frame_rate=first.frame_rate)`, `sum()`-style accumulation, then `export(BytesIO(), format="wav")`. `wav_duration_seconds` uses `len(AudioSegment)/1000.0` (pydub reports milliseconds) — do NOT reach for `AudioBookGenerator._get_audio_duration` (`audiobook_generator.py:847`); it is a private method on a heavyweight class and its mutagen/size-estimate ladder is for files on disk.

- [ ] **Step 5: Green + mutation.** Replace the implementation body with naive `b"".join(segments)` → the frame-count/duration test REDs (this is the test's whole purpose); restore.

- [ ] **Step 6: Commit** `feat(tts): general WAV decode-and-concat stitcher`.

---

### Task 4: Resolving a roster speaker to a concrete voice

**Files:**
- Modify: `tldw_chatbook/TTS/profile_service.py` (`_ProfileRepositoryProtocol` ~`:81-145`, and the service class — add `get_profile`)
- Create: `tldw_chatbook/Subscriptions/briefing_voices.py`
- Test: `Tests/Subscriptions/test_briefing_voices.py` (new); extend `Tests/TTS/` for the passthrough

**Interfaces — Consumes:** `TTSProfileRepository.get_profile(profile_id: UUID) -> ProfileStoreResult[TTSGenerationProfile]` (`profile_repository.py:1314` — exists, but is NOT reachable through the service today).
**Interfaces — Produces:**
```python
# profile_service.py
async def get_profile(self, profile_id: UUID) -> LoadedTTSProfile   # passthrough

# briefing_voices.py
class VoiceResolutionError(RuntimeError): ...
@dataclass(frozen=True)
class VoiceSelection:
    speaker: str
    provider_id: str
    model_id: str
    voice_id: str | None
    response_format: str
    speed: float
    options: Mapping[str, Any]
    profile_id: str | None
    profile_revision: int | None
    def is_exact_provider(self) -> bool     # provider_id == "audio_cpp"
async def resolve_roster_voices(roster_snapshot: Sequence[Mapping[str, Any]], *,
                                profile_service: Any | None) -> list[VoiceSelection]
def dump_voice_snapshot(selections: Sequence[VoiceSelection]) -> str
```

- [ ] **Step 1: Read the identity landscape first — it has two different keys.** 2a's roster entries carry `voice_profile_id: str | None` (a **stringified profile UUID** — `watchlists_collections_screen.py:3766` builds the picker as `(display_name, str(profile.profile_id))`) and `character_card_id: int | None` (a ChaChaNotes row id). `CharacterRef` (`TTS/profile_types.py:382`) is a *different*, three-part authority-scoped identity and is NOT a card id. 2b resolves from `voice_profile_id`; character-bound speakers whose card carries an assignment are out of scope for this task (they fall back to the same "no profile" path).

- [ ] **Step 2: Write the failing tests** with a fake profile service (the only faked seam here): a speaker with a valid `voice_profile_id` yields a `VoiceSelection` carrying the profile's provider/model/voice/speed **and its `revision`**; `response_format` is forced to `"wav"` regardless of what the profile says (decision 2 — assert it, this is the constraint the whole pipeline rests on); a speaker with `voice_profile_id=None` raises `VoiceResolutionError` naming the speaker; a `voice_profile_id` that no longer resolves raises `VoiceResolutionError` naming **both** the speaker and the id (a deleted profile must be diagnosable); `profile_service=None` (service unbound) raises naming the speaker and saying no voice service is available; `dump_voice_snapshot` round-trips through `json.loads` with `sort_keys=True` and includes `profile_revision`.

- [ ] **Step 3: Run, confirm RED.**

- [ ] **Step 4: Implement.** The service passthrough mirrors `get_assigned_profile`'s shape (`profile_service.py:1057`) — add `get_profile` to `_ProfileRepositoryProtocol` and the service, returning the same `LoadedTTSProfile` wrapper the rest of the service uses. `resolve_roster_voices` maps speakers in roster order and raises on the first unresolvable one.

- [ ] **Step 5: Green + mutations.** (a) Let `response_format` pass through from the profile → the wav-forcing test REDs; restore. (b) Drop `profile_revision` from the snapshot → the round-trip test REDs; restore.

- [ ] **Step 6: Commit** `feat(briefings): resolve roster speakers to concrete voices`.

---

### Task 5: Per-turn synthesis on both provider paths

**Files:**
- Create: `tldw_chatbook/Subscriptions/briefing_audio.py` (synthesis half)
- Test: `Tests/Subscriptions/test_briefing_audio_synthesis.py` (new)

**Interfaces — Consumes:** Task 2's `build_legacy_speech_request`; Task 4's `VoiceSelection`; `TTSService.synthesize_exact(request) -> tuple[TTSAudioResponse, TTSRequestedSelectionSnapshot]` (`TTS_Generation.py:559`); `TTSService.generate_audio_stream(request: OpenAISpeechRequest, internal_model_id: str) -> AsyncIterator[bytes]` (`:909`); `TTS/text_processing.TextChunker(max_tokens=...).chunk_text(text) -> List[TextChunk]` (`.text` attribute per chunk); Task 3's `concat_wav_segments`.
**Interfaces — Produces:**
```python
class TurnSynthesisError(RuntimeError): ...          # message always names speaker + turn index
MAX_TURN_CHARS = 1800
async def synthesize_turn(tts_service: Any, selection: VoiceSelection, text: str, *,
                          turn_index: int) -> bytes          # WAV bytes for one turn
```

- [ ] **Step 1: Read the reference implementation.** `TTSEventHandler._generate_tts` (`Event_Handlers/TTS_Events/tts_events.py:731-960`) — specifically the `finally` at `:899-912` that calls `aclose()` while preserving the primary exception, and `_validate_exact_selection` (`:983`) which compares the returned `TTSRequestedSelectionSnapshot` field-for-field against the request. Your exact path reuses that idea; your legacy path has no snapshot to compare, which is exactly why it needs its own validation (below).

- [ ] **Step 2: Write the failing tests.** Fake the TTS service only — a stub exposing `synthesize_exact` and `generate_audio_stream`. Cases:
  - **exact path** (`provider_id="audio_cpp"`): builds a `TTSRequest` with `response_format="wav"` and the selection's model/voice/speed/options; drains the response; **`aclose()` is called exactly once even when the drain raises** (assert on the stub — the leak that leaves the registry lease held);
  - **exact path, contract violation**: the returned snapshot names a different voice than requested → `TurnSynthesisError` naming speaker, turn index, and both voices;
  - **legacy path** (`provider_id="kokoro"`): calls `generate_audio_stream` with the request+id from `build_legacy_speech_request`, joins the chunks, and — since there is no response object — **validates what it can**: a zero-byte stream raises `TurnSynthesisError` naming speaker+index, and the result must decode as WAV (feed the stub a non-WAV payload → error naming speaker+index). This is the validation `synthesize_exact` would have given us;
  - **long-turn chunking**: text longer than `MAX_TURN_CHARS` is split via `TextChunker` and the pieces stitched with `concat_wav_segments` — assert the stub saw >1 request and the returned duration covers all pieces;
  - **failure naming**: every raised `TurnSynthesisError` message contains the speaker name and the 0-based turn index (AC #6).

- [ ] **Step 3: Run, confirm RED.**

- [ ] **Step 4: Implement.** Branch on `selection.is_exact_provider()`. Exact: `TTSRequest(provider_id=..., model_id=..., text=chunk, voice=..., response_format="wav", speed=..., options=...)` → `synthesize_exact` → `async with response:` (it is an async context manager, `adapter_types.py:407`) draining `response.byte_stream`. Legacy: `build_legacy_speech_request(...)` → `generate_audio_stream(request, internal_model_id)` → `b"".join([c async for c in stream])`. Both then validate and return WAV bytes; multi-chunk turns go through `concat_wav_segments`.

- [ ] **Step 5: Green + mutations.** (a) Move `aclose()` out of the `finally` → the drain-raises test REDs; restore. (b) Drop the legacy WAV-decodes check → the non-WAV test REDs; restore. (c) Drop the exact-path snapshot comparison → the contract-violation test REDs; restore.

- [ ] **Step 6: Commit** `feat(briefings): per-turn synthesis across exact and legacy providers`.

---

### Task 6: The audio pipeline

**Files:**
- Modify: `tldw_chatbook/Subscriptions/briefing_audio.py` (pipeline half)
- Test: `Tests/Subscriptions/test_briefing_audio_pipeline.py` (new)

**Interfaces — Consumes:** Task 1 CRUD; Task 4 `resolve_roster_voices`/`dump_voice_snapshot`; Task 5 `synthesize_turn`; Task 3 `concat_wav_segments`/`wav_duration_seconds`; `Utils/private_paths.secure_private_directory(path, *, create, application_owned)` and `atomic_private_write_bytes(path, payload, *, application_owned_directory)`; `config.get_user_data_dir()`.
**Interfaces — Produces:**
```python
STATUS_GENERATING = "generating"; STATUS_COMPLETE = "complete"; STATUS_FAILED = "failed"
INTERRUPTED_ERROR = "interrupted"
ERROR_CHAR_CAP = 1000
async def generate_script_audio(db, script_id: int, *, tts_service: Any,
                                profile_service: Any | None,
                                synthesize=synthesize_turn) -> dict[str, Any]
def fail_interrupted_audio(db, script_id: int | None = None) -> int
def briefing_audio_dir() -> Path      # <user data>/briefing_audio, secured, application-owned
```

- [ ] **Step 1: Read the error-boundary contract you must copy.** `briefing_cast.generate_script` (`Subscriptions/briefing_cast.py:562`) is the precedent in every respect: pre-flight refusals raise BEFORE any row exists; once the row exists every in-band failure writes a `failed` row rather than raising; **DB errors propagate** (the caller's worker wraps them); the parent artifact is never touched. Here: **a failed synthesis must leave the script row byte-identical** (spec §Error handling ethos).

- [ ] **Step 2: Write the failing tests** with a real DB and Task 5's `synthesize` seam faked. Cases:
  - happy path: `complete` row, `file_path` set, `duration_seconds` ≈ sum of turns, `turn_count` correct, file exists on disk with WAV magic bytes;
  - **the file lands under the private data dir** (assert the path is inside `briefing_audio_dir()`), and `get_user_data_dir` is patched to `tmp_path` so no test touches real user storage;
  - script not `complete` → refused, **no audio row written**;
  - script has no/unreadable turns → refused, no row;
  - one turn raises `TurnSynthesisError` → `failed` row whose `error` names the speaker and turn index, capped at `ERROR_CHAR_CAP`, **no file left behind**, and the script row unchanged (full dict equality);
  - voice resolution fails (deleted profile) → `failed` row naming the speaker and id;
  - `fail_interrupted_audio` flips orphaned `generating` rows to `failed`/`interrupted` and returns the count;
  - **named invariant `test_a_failed_synthesis_never_touches_the_script`** — full dict equality on the script row before/after.

- [ ] **Step 3: Run, confirm RED.**

- [ ] **Step 4: Implement.** Order: load script (must be `complete`) → parse `turns_json` and `roster_snapshot_json` → `resolve_roster_voices` → insert `generating` row carrying `dump_voice_snapshot(...)` → for each turn: look up its speaker's selection (a turn naming a speaker absent from the snapshot is a `failed` row naming it) and `await synthesize(...)` → `concat_wav_segments` → write via `atomic_private_write_bytes` into `briefing_audio_dir()` as `script-{script_id}-audio-{audio_id}.wav` → update row `complete` with path/duration/turn_count. Every DB call goes through `asyncio.to_thread` (2a's whole-branch ruling). Type-only logging; never log turn text.

- [ ] **Step 5: Green + mutations.** (a) Let a synthesis failure also update the script row → the named invariant REDs; restore. (b) Write the file outside `briefing_audio_dir()` → the private-dir test REDs; restore. (c) Skip the file cleanup on failure → the no-file-left-behind test REDs; restore.

- [ ] **Step 6: Commit** `feat(briefings): script-to-audio pipeline with honest failures`.

---

### Task 7: Synthesize action and playback in the Artifacts pane

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py` (compose ~`:596`, `_script_detail_renderable:649`)
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Modify: `tldw_chatbook/css/features/_watchlists.tcss` + regenerate the bundle
- Test: extend `Tests/Watchlists/test_watchlists_artifacts_pane.py`

**Interfaces — Consumes:** Task 6 `generate_script_audio`/`fail_interrupted_audio`; Task 1 `list_briefing_audio`; `TTS/audio_player.get_audio_player()` and `play_audio_file(path)` (`audio_player.py:524,533`).
**Interfaces — Produces:** pane reactive `script_audio: dict | None`; messages `SynthesizeAudioRequested()`, `PlayAudioRequested()`, `StopAudioRequested()`.

- [ ] **Step 1: Read the two traps this task walks into.** (a) `ArtifactsPane.selected_script` is `reactive(..., recompose=True)` (`:337`) — **every script selection rebuilds these widgets**, so playback state must live in the app-level player singleton, never on the widget. (b) `tts_events.py:1579` `stop_audio_playback_if_current`'s docstring documents the task-559 fix rounds: the stop guard must key off a `_last_played` slot, not a TTL cache. Read it before writing any stop logic.

- [ ] **Step 2: Write the failing tests** (fake `generate_script_audio` at the screen's reference, as 2a fakes `generate_script`). Cases: Synthesize on a `complete` script writes an audio row and the detail shows status + duration; Synthesize on a non-complete script refuses with a toast naming the status; a second press while in flight refuses naming the running one (guard claimed **at dispatch**, cleared in `finally`, `group="wl-audio"`, exclusive); a crashed `generating` audio row is recovered as `interrupted` on the next load (pin the sweep with the flag-at-call-time recorder — the phase-1 lesson that a load-path sweep can silently vacate a dispatch-path test); a `failed` audio row renders its error text; Play calls the player with the row's path and Stop stops it; **Play is disabled when `file_path` is NULL or the file is missing** (an artifact whose file was deleted underneath us must not offer a dead control — the spec's honest-degradation rule).

- [ ] **Step 3-4: Implement + green.** Audio state loads inside the existing `_load_briefings` to_thread batch. Controls live in `compose()` (a `Static` holding a Rich renderable cannot host buttons), so they sit after `#artifacts-script-detail`.

- [ ] **Step 5: Mutations.** (a) Claim the guard inside the worker body instead of at dispatch → the double-press test REDs; restore. (b) Enable Play with a NULL path → the dead-control test REDs; restore.

- [ ] **Step 6: Commit** `feat(briefings): synthesize and play script audio`.

---

### Task 8: Close-out

- [ ] Full sweep: `Tests/Subscriptions/ Tests/Watchlists/ Tests/TTS/ Tests/UI/ -k watchlist`. Documented baselines (NOT ours): 2 tree-chevron failures in `test_destination_visual_parity_correction.py`; the TASK-1345 create-form mount race (rotating symptom — re-run alone before classifying). Anything else failing is ours: report BLOCKED rather than papering over.
- [ ] `backlog/tasks/task-1630`: check every AC that shipped; add a line naming the storage decision (buffer-whole, justified) and the all-providers decision.
- [ ] `backlog/tasks/task-1540`: check AC #2 (phase 2 complete: 2a + 2b), leaving phases 3-4 open.
- [ ] Spec: add a "Phase 2b delivery notes" block — what shipped, the two decisions above, and the residual that `stts_events`'s id copy still diverges by design (with its cross-reference comment).
- [ ] Cross-worktree ID scan (controller supplies IDs) → file follow-ups for anything parked during the run.
- [ ] Commit `docs(briefings): phase 2b close-out`.

## Self-review

**Spec coverage (2b scope):** per-turn synthesis honoring each speaker's snapshot voice → T4+T5+T6 (AC #1); real decode-and-concat stitcher → T3 (AC #2); duration recorded → T3 helper + T6 row write (AC #3); private-dir storage with the decision justified → T6 + decision 3 (AC #4); in-app playback → T7 (AC #5); failure names turn+speaker, keeps the script → T5 error messages + T6's named invariant (AC #6). The spec's "audio failure keeps the script" ethos is the named invariant. Feed/export and scheduling remain phases 3-4.

**Placeholder scan:** none — every step names its precedent `file:line` or carries the contract inline; the skeletal-test convention is declared up front.

**Type consistency:** `VoiceSelection` fields match T5's consumption and T6's snapshot; `bytes` is the currency between T5 and T3 and T6; `voice_profile_id` is `str` at the roster seam and `UUID` at the profile-service seam, converted exactly once inside `resolve_roster_voices` (T4); `STATUS_*` constants mirror `briefing_cast`'s; `synthesize_turn`'s signature is identical in T5's Produces and T6's Consumes.
