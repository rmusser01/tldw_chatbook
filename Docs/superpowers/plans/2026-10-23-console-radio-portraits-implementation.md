# Console Radio Portraits Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Execution method is not selected; do not start product implementation during planning.

**Goal:** Give Console characters reviewed six-pose artwork, silence-aware TTS mouth movement, independent blinking/idle motion, and stable radio-screen framing.

**Architecture:** Extend the existing character database, expression resolver, disposable avatar and speech playback owners. Keep performance attachments profile-local and separate from immutable mood assets; expose only bounded output-timed presentation snapshots to widgets. Import/review precedes generation so known artwork can qualify the complete playback path.

**Tech Stack:** Python >=3.12, Textual >=8.0.0,<9, SQLite, existing Pillow/image validators, existing PCM sink and image-generation adapters. No new dependency.

**Spec:** [Approved design](../specs/2026-10-23-console-radio-portraits-design.md)

**Review:** [Six source-anchored corrections](../reviews/2026-10-23-console-radio-portraits-design-review.md)

**Implementation-plan review:** [Three corrections and revised-plan review](../reviews/2026-10-23-console-radio-portraits-implementation-review.md)

ADR required: yes
ADR path: backlog/decisions/242-console-radio-portrait-performance.md
Reason: Implements the accepted local attachment, speech presentation and Console UI contracts in [ADR-242](../../../backlog/decisions/242-console-radio-portrait-performance.md); no additional architectural choice is authorized by this plan.

## Global Constraints

- Roles are exactly `rest`, `small`, `open`, `blink_rest`, `blink_small`, `blink_open`.
- Each normalized still PNG has at most 1,048,576 pixels and 4 MiB encoded; a set has at most 16 MiB encoded. Stricter applicable existing limits win.
- Keep the existing 64 MiB active/in-flight preparation budget and independent shared-decode caps; account editor and Console work together.
- Presentation paints at most 30 times/second; target onset and silence response within 100 ms of the qualified output timeline, including window/paint delay.
- Use 20 ms RMS windows; silence closes after two raw windows below -45 dBFS, reopens at -42 dBFS, and bypasses smoothing. Digital zero/underrun/pause/terminal states close immediately.
- Smooth finite dBFS with 40/100 ms attack/release; open at -24 dBFS and return to small below -27 dBFS. Reset smoothing on gate closure.
- Progress older than 100 ms forces closed. Envelope metadata is bounded by admitted queued audio plus the one-second device-tail reserve.
- The existing sink notification thread samples qualified presentation every 20 ms while active, independently of public sink events or UI reads; updates coalesce into one latest snapshot. Natural completion waits for the backend's output-finished acknowledgement; explicit Stop/replacement retains immediate abort and the existing two-block silence contract. Neither completion nor ownership release follows software-queue exhaustion alone.
- Blink spacing is 3–7 seconds, closed-eye hold uses `$ds-duration-fast`, and idle movement has a 6-second cycle with at most one rendered-image pixel of fixed-canvas offset. Missing visual values become central tokens first.
- Prepare only persisted active artwork for saved local characters; exact profile/service/database/editor/source authority must survive every asynchronous boundary. Never auto-save dirty artwork.
- Generation discloses six requests and destination, selected regeneration sends one, successful slots survive other failures, and uncertain paid requests are never automatically retried.
- Frame roles do not become mood keys. Preserve automatic/manual/historical selection and current Dynamic/Static, Reduce motion, animations/reactions/avatar visibility behavior.
- Original portraits and immutable assets stay untouched. No performance extension to character cards, Actor Packs, Chatbooks or sync; full-profile backup/restore includes local SQLite rows.
- No new destination, provider, speech player/framework, contact menus, sounds, realtime voice support, rigs or phoneme alignment.
- Read `Docs/Architecture/{character-chat,speech,media-generation}.md`, `backlog/docs/{design-language,component-patterns,lessons-testing-evidence,lessons-live-verification,lessons-console-wiring,lessons-bounded-artwork-imports,lessons-actor-pack-portability}.md` before the owning work.
- Use targeted tests only unless the user explicitly requests a full sweep. UI worker tests use real file-backed SQLite because worker threads cannot share a plain `:memory:` database; same-thread repository tests may use real in-memory SQLite.
- Before implementing each To Do task, move it In Progress and copy its task-specific steps plus ADR fields into its Implementation Plan. Add Implementation Notes only after implementation. Do not pre-populate task plans while they are To Do.

## Review Focus

These additional cases are assigned to tests below; they are not optional follow-ups.

1. Remove and reapply identical bytes: an old revision must never equal the replacement's identity (Task 1 and Task 5).
2. Rejected/retried feed chunks and split PCM samples: envelope positions must follow accepted audio exactly (Task 2 and Task 3).
3. Closing an editor sample after a newer voice starts: cleanup must stop only the sample's exact owner (Task 4 and Task 7).
4. A hidden/blocked reader while audio continues: memory stays bounded, audio stays unchanged and reveal shows current output (Task 3 and Task 5).
5. A source becomes dirty or a profile changes while image editing is in flight: no next request or late Apply can reach the replacement owner (Task 6 and Task 7).
6. The last PCM block is queued ahead of audible output: natural drain, completion callbacks and sentence handoff wait for backend completion; Stop during that wait still interrupts immediately (Task 3 and Task 4).
7. A complete loud/pause/loud utterance updates through the production notification loop without tests calling the publication helper; no callback/public-event backlog is needed to keep it alive (Task 3).
8. An ordinary-layout meter becomes unavailable: hide its contents with visibility while preserving its row; only a compact-layout change may remove the row (Task 8).

## Atomic Backlog units and dependencies

Each row is one independently reviewable PR-sized deliverable. Tasks were initially filed To Do; current execution status and checked criteria live in their Backlog records. Dependencies refer only to already-created lower-numbered tasks.

| Unit | Backlog task | Depends on | Deliverable |
| --- | --- | --- | --- |
| 1 | [TASK-36542](../../../backlog/tasks/task-36542%20-%20Persist-validated-character-portrait-performance-sets-atomically.md) | Design TASK-36509 | Validated, atomic durable attachments |
| 2 | [TASK-36543](../../../backlog/tasks/task-36543%20-%20Derive-silence-aware-portrait-mouth-states-from-PCM-speech.md) | Design TASK-36509 | Pure waveform/filter and snapshot contracts |
| 3 | [TASK-36544](../../../backlog/tasks/task-36544%20-%20Align-speech-presentation-with-PCM-device-output-and-underruns.md) | 36543 | Output-timed sink/WAV observation |
| 4 | [TASK-36545](../../../backlog/tasks/task-36545%20-%20Bind-portrait-speech-observation-to-Console-and-preview-owners.md) | 36544 | Owned manual/auto/hands-free/preview speech |
| 5 | [TASK-36546](../../../backlog/tasks/task-36546%20-%20Render-living-character-portraits-from-prepared-performance-frames.md) | 36542, 36543, 36545 | Working living portrait in the existing rail |
| 6 | [TASK-36547](../../../backlog/tasks/task-36547%20-%20Prepare-and-review-imported-portrait-performance-artwork.md) | 36542, 36546 | Import, offline review, Apply/Remove/Cancel |
| 7 | [TASK-36548](../../../backlog/tasks/task-36548%20-%20Generate-reviewed-portrait-performance-frames-and-preview-their-speech.md) | 36545, 36547 | Six-request generation and guarded TTS sample |
| 8 | [TASK-36549](../../../backlog/tasks/task-36549%20-%20Frame-Console-character-portraits-as-a-stable-radio-display.md) | 36545, 36546 | Stable radio framing/status/meter |
| 9 | [TASK-36550](../../../backlog/tasks/task-36550%20-%20Qualify-Console-radio-portraits-across-artwork-speech-and-recovery.md) | 36547, 36548, 36549 | Complete qualification and user documentation |

Units 1 and 2 have no implementation dependency on each other. Unit 8 does not require generation. Preserve this graph if execution is delegated; do not run simultaneous writers against the same screen, sink, tests or Git index.

## File and interface map

Repository-relative paths below identify implementation files, not newly implemented APIs.

- Create `tldw_chatbook/Character_Chat/portrait_performance.py`: six-role value objects, bounded normalization, canonical source capture and conflict error. Import Pillow lazily inside image operations; no Textual/provider imports.
- Modify `DB/ChaChaNotes_DB.py` and its executed migrations: local parent/child rows and transaction/CAS methods. Select the migration filename from the live schema version in Unit 1, not a number reserved by this document.
- Create `Audio/speech_presentation.py`: stdlib-only PCM windows, finite silence filter, immutable owner/snapshot and one-latest-value observation channel. Extend `Audio/streaming_sink.py` for optional observation; keep audio machinery authoritative.
- Extend `Event_Handlers/TTS_Events/tts_events.py`, `UI/Console_Modules/message.py`, `UI/Console_Modules/hands_free.py`: thread optional observation through existing owners and transports. No new session/message store.
- Create `Chat/portrait_performance_state.py`: pure blink/idle/pose selection. Extend `Chat/character_expression_playback.py`, `Widgets/Console/character_expression_avatar.py`, `UI/Console_Modules/character.py`, its `wiring.py`, and `UI/Screens/chat_screen.py`: bounded preparation, revision-aware selection and existing renderer reuse.
- Create `Character_Chat/portrait_performance_authoring.py`: one disposable source-bound draft. Create `Widgets/Persona_Widgets/portrait_performance_editor.py`: an inline preparation view under the existing character editor, not another screen/modal authority. Extend `personas_character_editor_widget.py`, `personas_pane_messages.py` and `UI/Screens/personas_screen.py` for events/workers.
- Extend `Character_Chat/expression_generation.py` for six closed pose directions; call existing `Image_Generation.worker.run_generation` and reference-capability gates rather than registering another adapter.
- Modify `css/core/_variables.tcss`, `css/features/_console_panels.tcss`, source widget CSS and `css/build_css.py` only if a source registration is needed; regenerate `css/tldw_cli_modular.tcss` with the builder.
- Update the three Architecture docs and relevant Console/character user guides in the units that change behavior; Unit 9 verifies the final combined account and records actual qualification evidence.

### Shared contracts to define once

Unit 1 defines these public names in `Character_Chat/portrait_performance.py`:

```python
FrameRole = Literal["rest", "small", "open", "blink_rest", "blink_small", "blink_open"]
FRAME_ROLES: tuple[FrameRole, ...]
@dataclass(frozen=True, slots=True)
class PerformanceFrame:
    role: FrameRole
    png: bytes = field(repr=False)
    width: int
    height: int
    sha256: str
@dataclass(frozen=True, slots=True)
class PerformanceSource:
    character_id: int
    expression_key: str
    source_sha256: str
    source_kind: Literal["portrait", "legacy_expression", "visual_identity"]
    source_ref: tuple[int, ...]
@dataclass(frozen=True, slots=True)
class PortraitPerformanceRecord:
    source: PerformanceSource
    revision: int
    active: bool
    frames: tuple[PerformanceFrame, ...] = field(repr=False)
class PerformanceConflict(ValueError): ...
def normalize_performance_frames(frames: Mapping[FrameRole, bytes]) -> tuple[PerformanceFrame, ...]: ...
def normalize_performance_frame(role: FrameRole, raw: bytes) -> PerformanceFrame: ...
def source_from_resolution(resolution: VisualIdentityResolution) -> PerformanceSource: ...
def capture_performance_source(db: CharactersRAGDB, *, character_id: int,
                               requested_state: str, manual_expression_key: str | None = None
                               ) -> tuple[PerformanceSource, bytes]: ...
```

`source_ref` is empty for a portrait, `(expression_id,)` for legacy images, and `(pack_id, pack_version_id, asset_id)` for visual assets. Use the resolved key, with neutral for a plain portrait. It excludes request-specific cache prefixes, filesystem paths and operational requested-state labels; historical/current resolution of the same immutable asset must produce the same source. The UI additionally owns exact profile/database/editor authority; these transient objects are not serialized.

Unit 2 defines these names in `Audio/speech_presentation.py`:

```python
MouthPose = Literal["rest", "small", "open"]
TransportState = Literal["preparing", "playing", "paused", "stopped", "failed"]
@dataclass(frozen=True, slots=True)
class SpeechPresentationOwner:
    token: str
    scope_token: object = field(repr=False)
    session_id: str | None
    session_epoch: int | None
    character_id: int | None
    message_id: str | None
    utterance_token: int | None
@dataclass(frozen=True, slots=True)
class SpeechPresentationSnapshot:
    owner: SpeechPresentationOwner
    generation: int
    state: TransportState
    observed_at: float
    level_dbfs: float | None
    mouth: MouthPose
    timing_valid: bool
@dataclass(frozen=True, slots=True)
class EnvelopeWindow:
    start_frame: int
    frame_count: int
    dbfs: float
class PcmEnvelope:
    def __init__(self, *, sample_rate: int, channels: int) -> None: ...
    def push(self, pcm: bytes) -> tuple[EnvelopeWindow, ...]: ...
    def finish(self) -> tuple[EnvelopeWindow, ...]: ...
class SpeechEnvelopeFilter:
    def update(self, dbfs: float, *, duration_ms: float = 20.0) -> MouthPose: ...
    def reset(self) -> None: ...
class SpeechPresentation:
    def __init__(self, owner: SpeechPresentationOwner, *, clock: Callable[[], float] = time.monotonic) -> None: ...
    def publish(self, snapshot: SpeechPresentationSnapshot) -> None: ...
    def read(self) -> SpeechPresentationSnapshot | None: ...
    def set_transport(self, state: TransportState) -> None: ...
```

Signatures in these blocks are declarations, not partial implementation bodies. `scope_token` preserves host-issued authority identity without coercion. A new owner gets a new token; do not infer ownership from synthetic message ID prefixes. The channel stores one latest snapshot, rejects mismatched owners/old generations and older same-generation observations, and applies freshness when read. Its `owner` is read-only; publications carry that exact issued owner instance. Confirmed pause requires explicit resume; terminal state is final for this owner. Qualified zero/missing level closes the mouth while retaining valid timing, independently of timing-unavailable fallback. Its observer state cannot mutate speech admission. Remaining exported names are defined in their owning unit below.

## Common execution and verification

Use an appropriate isolated checkout at implementation time and read the task file before code changes. The current shared checkout contains unrelated work and two pre-existing duplicate IDs (34991, 35028); do not stash, amend or renumber that work as part of this feature.

Run examples with `.venv/bin/python -m pytest -q -o addopts='' <target>` using the implementation checkout's own Python >=3.12 environment. Expected red results must identify the new missing/wrong behavior; a fixture/import/environment failure is not the requested red. Do not install optional audio/image dependencies merely to make an unrelated environment green. Ensure fixture-owned workers finish and databases close on their owning threads; reuse installed finite-operation database admission/retirement guards.

For each unit, run Ruff on changed Python files, formatting checks on changed files, relevant type/import boundaries and `git diff --check`. UI units additionally run `python tldw_chatbook/css/build_css.py`, `Tests/UI/test_design_token_governance.py` and `Tests/UI/test_css_bundle_sync_guard.py`. Reproduce unrelated failures on the exact unmodified base before calling them baseline. Commit only named unit files after self-review; document actual evidence in that Backlog task.

### Unit 1 — TASK-36542: bounded durable artwork

**Files:** Create `Character_Chat/portrait_performance.py`, `Tests/Character_Chat/test_portrait_performance.py`, `Tests/ChaChaNotesDB/test_portrait_performance_repository.py` and `test_portrait_performance_migration.py`. Modify `DB/ChaChaNotes_DB.py`, `DB/recovery_core_schema.py`, `DB/recovery_core.py` and `DB/sql_validation.py`; add the executed V81→V82 SQL file and update current-schema checks in `Tests/DB/test_response_rules_migration.py` and `Tests/Backup_Recovery/test_core_owners.py`. These recovery/allowlist callers were traced before implementation. Extend existing expression tests only where compatibility needs coverage.

**Consumes:** Existing `resolve_visual_identity`, bounded image validation, database transactions, schema migration dispatch and backup admission.

**Produces:** Shared Unit 1 values above and these `CharactersRAGDB` methods:

```python
def apply_character_portrait_performance(self, source: PerformanceSource,
        frames: tuple[PerformanceFrame, ...], *, expected_revision: int) -> int: ...
def get_character_portrait_performance(self, *, character_id: int,
        expression_key: str) -> PortraitPerformanceRecord | None: ...
def remove_character_portrait_performance(self, source: PerformanceSource,
        *, expected_revision: int) -> int: ...
```

- [x] **1. Read the live migration chain and begin the task.** Record source/target schema versions in its task plan; name the SQL file from those versions and update the executable dispatcher. Use `Tests/ChaChaNotesDB/historical_bootstrap.py` for the real preceding-schema fixture; no decorative SQL twin or reserved version number.
- [x] **2. Write failing normalization/migration/repository tests.** Repository fixture creates a real local character with valid PNG bytes, captures its source and six normalized frames, and closes its SQLite owner after workers settle. Use `:memory:` only in same-thread cases; a worker case uses `tmp_path / 'performance.db'`. Include this revision assertion in the repository test body:

```python
def test_remove_reapply_does_not_reuse_revision(performance_case):
    db, source, frames = performance_case
    first = db.apply_character_portrait_performance(source, frames, expected_revision=0)
    absent = db.remove_character_portrait_performance(source, expected_revision=first)
    second = db.apply_character_portrait_performance(source, frames, expected_revision=absent)
    assert 0 < first < absent < second
    with pytest.raises(PerformanceConflict):
        db.apply_character_portrait_performance(source, frames, expected_revision=first)
```

The `performance_case` fixture belongs in the new repository test module: `(CharactersRAGDB, PerformanceSource, tuple[PerformanceFrame, ...])`, seeded via `add_character_card`, `capture_performance_source` and six in-memory PNGs made using `BytesIO`/Pillow. Also assert: missing/duplicate/animated roles, mismatched geometry, exact and over-limit normalized sizes, encoding-failure buffer release, fresh/upgraded schema, source mutation, retrieval of an old-source revision for explicit new-source replacement, forged frame metadata/hash rejection at Apply, failed child insert rollback, two connections racing, soft-delete/restore, permanent owner deletion, and no changes to original image bytes. Historical/current resolution of one visual asset must give equal source descriptors.
- [x] **3. Run the three new files red.** Expected: missing contract/schema or failed behavior assertions, with the existing SQLite fixture alive.
- [x] **4. Implement normalization and transaction methods.** Parent key is `(character_id, expression_key)` with source descriptor/hash, schema version, monotonic revision and active flag; children have unique `(performance_id, role)` and validated PNG/hash/dimensions. Preserve an inactive parent/revision on Remove, delete child blobs, and increment on reapply: deleting/recreating a revision-1 parent would allow a stale-worker ABA collision. Recheck persisted source and expected revision inside the transaction; encode/prepare images before entering it. Foreign-key permanent deletion cascades. Getter returns an existing parent even when its old source differs from the currently selected source, so a new-source draft can capture the real expected revision and replace it; only a missing parent returns `None`. Inactive records have no frame blobs. Runtime must match `record.source` to the selected source before activating frames. No sync registration or export-format change.
- [x] **5. Run targeted verification green.** Run the three new files plus existing character-expression and visual-identity migration/repository tests. Expect all new cases passing and original bytes unchanged; run migration packaging/static guards applicable to the changed chain. Record old/new version numbers and rollback evidence.
- [x] **6. Commit this storage unit.** Suggested message: `feat: persist bounded local portrait performance sets`. Update `Docs/Architecture/character-chat.md` to describe implemented storage only and link ADR-242.

### Unit 2 — TASK-36543: pure energy and presentation contracts

**Files:** Create `Audio/speech_presentation.py`, `Tests/Audio/test_speech_presentation.py`. Extend a fresh-process audio import-boundary test if needed; do not import `TTS.pcm_stream` from the zero-Textual Audio module.

**Consumes:** PCM16 samples and a monotonic clock. **Produces:** The Unit 2 shared contracts, no sound device, thread, widget or provider side effect.

- [x] **1. Add deterministic failing tests**, including actual assertions for nonzero quiet background and finite reopen:

```python
def test_nonzero_pause_bypasses_release_and_reopens_finitely():
    f = SpeechEnvelopeFilter()
    for _ in range(10):
        assert f.update(-12.0) == "open"
    assert f.update(-60.0) == "open"
    assert f.update(-60.0) == "rest"
    assert f.update(-12.0) == "open"
    assert f.update(float("-inf")) == "rest"
    assert f.update(-12.0) == "open"

def test_pcm_sample_split_matches_one_chunk():
    pcm = struct.pack("<960h", *([8192] * 960))  # 40 ms at 24 kHz
    whole = PcmEnvelope(sample_rate=24000, channels=1)
    split = PcmEnvelope(sample_rate=24000, channels=1)
    expected = whole.push(pcm) + whole.finish()
    actual = split.push(pcm[:1]) + split.push(pcm[1:673]) + split.push(pcm[673:]) + split.finish()
    assert actual == expected
    assert len(actual) == 2
```

Also assert small/open hysteresis, 48 kHz/stereo anti-phase energy (sum squared channel samples, not cancellation-prone mono averaging), malformed final sample rejection, invalid rates/channels, old-owner/generation rejection, one-snapshot retention and >100 ms freshness forcing rest. Seed the clock and owner directly; no wall-clock sleeps.
- [x] **2. Run `Tests/Audio/test_speech_presentation.py` red.** Expected missing module or failed numerical/state assertions.
- [x] **3. Implement the shared contracts and finite filter.** Carry incomplete samples/windows between pushes; reject an incomplete final PCM sample on finish. Window positions are source frames, not interleaved sample counts. Apply silence gating before smoothing; after closure the next finite speech window seeds the filter. `SpeechPresentation.read()` returns closed/invalid timing for stale progress without overwriting a newer transport terminal. Observer errors cannot propagate into audio callers.
- [x] **4. Run the file green plus a fresh interpreter import.** Assert neither `textual`, `sounddevice` nor TTS package modules were imported by `Audio.speech_presentation`. Verify the numerical thresholds in Global Constraints and the accepted-chunk input contract needed by Unit 3.
- [x] **5. Commit.** Suggested message: `feat: derive bounded silence-aware speech presentation`.

### Unit 3 — TASK-36544: qualified device timing and WAV output

**Files:** Modify `Audio/streaming_sink.py`, `Event_Handlers/TTS_Events/tts_events.py`, and `TTS/pcm_stream.py` only for existing eligible-format routing. Extend `Tests/Audio/test_streaming_sink.py`, `test_streaming_sink_pump.py`, `Tests/TTS/test_pcm_stream_plan.py`; create `Tests/Audio/test_streaming_sink_presentation.py`. Audit all `stream_factory` callers and injected fake streams when extending their backend-completion seam; update the Audio factories together. Check the fake sink in `Tests/TTS_Events/test_spoken_feedback_streaming.py` for optional-constructor compatibility.

**Consumes:** Unit 2 `SpeechPresentation` and envelope/filter. **Produces:** Backward-compatible `StreamingPcmSink(..., presentation: SpeechPresentation | None = None, clock: Callable[[], float] = time.monotonic)`; accepted-feed-aligned, output-timed observations. The private `_publish_presentation(self) -> None` samples mapped output at that clock on the existing non-realtime notification thread. Use a timed notification wait capped at 20 ms while observation is active, plus a coalesced wake for transport/completion changes; ordinary callback progress replaces bounded state rather than enqueueing a publication job per block. No second publisher thread, UI timer dependency or new public presentation event. Direct helper calls may prove mapping arithmetic only; they cannot qualify continuous publication. Public sink signatures/event types and `feed(pcm) -> bool` stay compatible; natural `SinkDrained`/`terminal_reason` publication is deliberately corrected to mean backend-confirmed completion, including when no observer is attached.

**Backend completion contract:** Extend the internal stream-factory keyword seam with `finished_callback: Callable[[], None]`; update every injected Audio factory and fake. In the production sounddevice factory, request graceful completion with `CallbackStop` in its thin backend callback wrapper after the final submitted block. Keep the backend-independent `_callback` exception guard; only the wrapper emits that intentional backend signal. The backend `finished_callback` hands off a bounded completion flag/wake, not UI work or stream teardown. `CallbackStop` and `finished_callback` already exist in the installed sounddevice API; no backend operations run inside the realtime callback. The fake stream models device-pending audio separately from submitted bytes and acknowledges finish only after that pending audio has played. Do not equate fake callback return with audible completion.

- [x] **1. Write failing fake-device cases.** Reuse `FakeStream` and live-sink cleanup from `Tests/Audio/test_streaming_sink.py`; extend its `tick` with optional time-info input while preserving the current `None` default for unobserved tests. Inject monotonic time and `currentTime`/`outputBufferDacTime`. Add these assertions to the delayed-output case:

```python
def test_dac_delay_and_device_tail_are_not_source_progress(sink_case):
    sink, stream, speech, clock = sink_case
    assert sink.feed(struct.pack("<24000h", *([16384] * 24000)))
    stream.tick(time_info=SimpleNamespace(currentTime=100.0, outputBufferDacTime=100.2))
    assert sink.settle()
    clock[0] = 10.05
    sink._publish_presentation()
    assert speech.read().mouth == "rest"
    clock[0] = 10.21
    sink._publish_presentation()
    assert speech.read().mouth == "open"
```

Define `sink_case` in the new test file with backend currentTime 100 mapped to injected monotonic 10, 24 kHz mono, the actual sink and updated fake stream. Add tests for 500 ms inserted silence consuming no source frames, partial zero suffix without a full-underrun event, rejected/retried feed, a split sample across accepted chunks, end-of-stream partial windows, prebuffering, invalid/nonfinite/backward timestamps, >1 second unverifiable output horizon, stalled readers, optional observer failure, stop and replacement.

Add a production-loop test with loud/pause/loud PCM and injected time. Drive only fake-device callbacks, clock advancement and normal source close, then await notification-thread observations using bounded event/condition barriers; never invoke `_publish_presentation()` in this test. Assert fresh open/rest/open observations across more than 100 ms, continued publication between callbacks and while the last block remains device-pending, bounded queued work with a hidden reader, and no publications after terminal cleanup. Keep no-observer controls.

Periodic sampling is not a new timing observation: publish a valid mouth only for a previously validated device interval covering the current clock. Do not continually refresh a stale loud level after the last mapped interval ends. Test a stalled callback both within its known pending device tail and after that interval expires; only the former may retain qualified movement.

The 20 ms wait is the nominal cadence, not a realtime scheduling guarantee. Existing synchronous `on_event` listeners can stall the notification thread: add a blocked-listener case proving the reader's >100 ms freshness fallback closes the mouth while audio remains unaffected, callback handoff stays bounded, and explicit Stop remains non-joining. Do not add another thread to conceal such a stall or refresh freshness from the widget.

Add a final-block test in which the fake device has 200 ms of pending output after the source queue empties. Before its explicit output-finished acknowledgement, assert `sink.state == "draining"`, `sink.terminal_reason is None`, no `SinkDrained`, and `pump()` still pending. After acknowledgement, assert exactly one drain/terminal event and closed stream. Parameterize with/without observation and invalid DAC timing: natural completion uses backend acknowledgement even when mouth timing is unavailable. Also test empty input, synchronous/duplicate/late finished callbacks, unexpected backend finish before source exhaustion (failure, not successful drain), Stop/replacement during the tail, reentrant listeners, cancellation, and a backend that never acknowledges finish.
- [x] **2. Run the new presentation tests red** while keeping existing no-observer sink tests as controls.
- [x] **3. Extend sink admission, observation and completion.** Prepare energy off the realtime callback. Commit envelope fragments only with the same accepted audio admission; rejected feeds cannot advance analysis or retain metadata. Track source/device ranges separately and include zero padding. Callback work is bounded state/counter handoff; perform filtering, clock mapping and coalesced latest publication at the defined cadence on the existing non-realtime notification path. Retain metadata until its device interval has passed the DAC timeline, not merely until PCM is popped. Bound queued metadata by admitted audio plus the one-second device tail. Invalid timing disables mouth observation while audio continues unchanged. Retaining metadata is not a substitute for keeping playback alive.

  Keep the sink `draining`, the live-sink registry entry and abortable stream reference until output-finished acknowledgement. On natural completion, finalize/close the already-finished stream off the callback thread, release the registry, and publish `terminal_reason="drained"`/`SinkDrained` exactly once. Do not use the existing abort teardown for a pending natural tail. Empty input must also finish without waiting for nonexistent audio. Explicit Stop/failure/replacement wins against a pending drain, aborts immediately without joining the notification thread, closes the observer and prevents any late finished acknowledgement from publishing drain or touching a replacement sink. A native finished callback can also follow abort; classify it using the sink's current exact terminal owner/state. Preserve `pump()` cancellation cleanup and its existing buffered-duration plus five-second drain watchdog; missing backend acknowledgement yields bounded failure/cleanup, never fabricated successful completion. Update the module's old queue-exhaustion/always-abort commentary and its existing natural-drain tests to this deliberate correction rather than treating their old assertions as compatibility requirements.
- [x] **4. Route eligible complete PCM16 WAV artifacts through the existing sink.** Source inspection confirms this route already exists; preserve it and add optional `presentation: SpeechPresentation | None = None` to `_stream_response_via_sink`, forwarding to the sink and updating affected hardware fakes. Reuse `sink_plan`, format/body validation, scoped owner cleanup and cancellation offload. Preserve existing raw PCM behavior and unqualified external playback fallback. Never add a second concurrent player or infer sync from `SimpleAudioPlayer.get_position()` wall time.
  Sink output epochs are separately fenced from request/session generations; owner adapters use `set_transport` for logical status. A local sink termination immediately invalidates amplitude, while the request adapter decides the logical terminal state after its existing WAV fallback decision. Do not seal the request's channel merely because an opportunistic sink failed before working legacy playback. The optional sink observer never creates another speech coordinator.

- [x] **5. Run new and existing sink/pump/format tests green.** Include a file-to-sink success, sink-open failure and existing midstream fallback control, asserting one audible owner and no duplicated completion. Validate automatic periodic publication and backend-confirmed drain through the real notification loop, including tail interruption, no-observer behavior and bounded watchdog/notify cleanup. Run the affected spoken-feedback streaming tests; no actor/DB/UI imports on the callback path.
- [x] **6. Commit.** Suggested message: `feat: observe speech at qualified PCM output times`. Update `Docs/Architecture/speech.md` with the implemented timing/capability limit.

### Unit 4 — TASK-36545: speech owner adapters

**Files:** Modify `Event_Handlers/TTS_Events/tts_events.py`, `UI/Console_Modules/message.py`, `hands_free.py`, `wiring.py` only where ownership injection is needed. Create `Tests/TTS/test_console_speech_presentation.py`; extend `Tests/TTS/test_buddy_guarded_speech.py`, `test_console_speak_autoplay.py` and `Tests/UI/test_console_hands_free_wiring.py`.

**Consumes:** Unit 3 sink observation and existing `TTSPlaybackLifecycle`, snapshots, destination checks and sequencer tokens.

**Produces:** Optional `presentation: SpeechPresentation | None = None` on `TTSMessageSpeechRequestEvent`, `TTSCompleteEvent`, `TTSPlaybackEvent`, `speak_utterance` and `speak_guarded_utterance`, forwarded through `_admit_tts_generation`, `_generate_tts_with_rate_limit`, `_generate_tts`, `_stream_text_pieces_if_applicable` and the Unit 3 sink/file paths. Existing parameters/return values and message/lifecycle validation stay compatible. `ConsoleMessageController.current_speech_presentation() -> SpeechPresentation | None` exposes the exact current UI owner, not a second speech coordinator. Add a `speech_presentation` read seam to Console character-controller wiring; Unit 5 consumes it.

- [ ] **1. Add failing owner and cancellation assertions.** In the existing hands-free fake-handler test, simulate a generation that calls completion twice and verify the public callback still fires once. The new callback forwarding test must contain these assertions after actual `speak_utterance` dispatch:

```python
assert finished == [True]
assert observed.owner.utterance_token == sequencer_token
assert observed.owner.character_id == captured_character_id
assert handler_generation_kwargs["voice"] is None
assert handler_generation_kwargs["resolution"] is None  # retain current hands-free selection
```

Use the existing fake generation/handler setup in the named tests; `finished` records completion, `observed` is `presentation.read()`, and `handler_generation_kwargs` records the forwarded real generation call. Pin manual/auto snapshot revalidation, sentence A terminal arriving after B start, session/profile/character replacement, unknown historical provenance, absent observers and listener failure. In the guarded test, start a newer request before cancelling the preview and assert only the old exact lifecycle reaches Stop.
- [ ] **2. Run these targeted speech tests red.** A changed callback signature in a fake handler must not be accepted as sufficient evidence; drive the actual forwarding entry and assert owner/terminal behavior.
- [ ] **3. Add optional observation to existing entries.** Capture host-issued owner tokens plus current scope/session epoch/character/message or sequencer token before awaits; revalidate on admission/result publication. Hands-free keeps its synthetic request ID and completion/quiet/cooldown behavior. Editor samples use guarded app-owned speech with existing destination confirmation and exact stop in cancellation cleanup. Terminal acknowledgements settle their own observer even after visible-session admission becomes stale.

  Consume Unit 3's corrected natural completion: keep `_active_stream_playback_owner`, request/observer ownership and the adapter lease alive until backend-confirmed drain or explicit stop/failure. Do not report lifecycle terminal, run `on_finished(True)`, admit the next hands-free sentence or reopen listening when only the software queue has emptied. Test sentence A with a delayed final device block: B/listening cannot start before acknowledgement; explicit cancellation still interrupts A, and its later acknowledgement cannot clear B or fire completion twice.
- [ ] **4. Expose confirmed transport status without adding `paused` to `TTSPlaybackLifecycle`.** Project pause/resume from the real player into `SpeechPresentation`; do not add a pause control or claim exact seek/resume support. Keep generation/admission lifecycle and Buddy consumers compatible. Resolve older-message mood only with existing valid actor/asset provenance; no inferred metadata writes.
- [ ] **5. Run the new file and affected existing cases green.** Cover real start, drain, stop/failure, prior-owner replacement, preview cancellation and callbacks with/without observation. Review every caller of both public utterance methods, including Buddy; default `None` preserves behavior.
- [ ] **6. Commit.** Suggested message: `feat: bind portrait speech to exact Console and preview owners`.

### Unit 5 — TASK-36546: living portrait rendering

**Files:** Create `Chat/portrait_performance_state.py`, `Tests/Chat/test_portrait_performance_state.py`, `Tests/UI/test_console_portrait_performance.py`. Modify `Chat/character_expression_playback.py`, `Widgets/Console/character_expression_avatar.py`, `UI/Console_Modules/character.py`, `wiring.py`, the avatar builder in `UI/Screens/chat_screen.py`, `css/core/_variables.tcss` and the design-language token catalog; rebuild generated CSS. Extend the existing expression/avatar/off-loop tests.

**Consumes:** Units 1/2/4, existing resolved-source/manual/historical precedence and rendering geometry.

**Produces:**

```python
@dataclass(frozen=True, slots=True)
class PerformancePose:
    role: FrameRole
    offset_px: float
@dataclass(frozen=True, slots=True)
class PerformanceMotionValues:
    blink_min_s: float
    blink_max_s: float
    blink_hold_s: float
    idle_cycle_s: float
    idle_offset_px: float
def load_performance_motion_values() -> PerformanceMotionValues: ...
class PortraitPerformanceClock:
    def __init__(self, *, motion: PerformanceMotionValues, interval: Callable[[], float]) -> None: ...
    def sample(self, mouth: MouthPose, *, now: float) -> PerformancePose: ...
def prepare_performance(record: PortraitPerformanceRecord, box: tuple[int, int]) -> PreparedPerformance: ...
class PreparedPerformance:
    frames: tuple[Any, ...]  # Six fitted PIL images in FRAME_ROLES order, owned within existing budget.
    def close(self) -> None: ...
# Add optional constructor keywords to the existing CharacterExpressionAvatar:
performance: PortraitPerformanceRecord | None = None
read_speech: Callable[[], SpeechPresentationSnapshot | None] | None = None
```

- [ ] **1. Write clock, preparation and mounted red tests.** Reuse `AvatarApp`, `wait_ready`, `console_screen_with_db_and_pilot` and `_set_active_console_character` from the existing avatar tests; do not build another production-app harness. Inject a 3-second blink interval and assert independent state directly:

```python
def test_blink_uses_the_current_mouth_without_restarting_its_clock():
    motion = PerformanceMotionValues(3.0, 7.0, 0.2, 6.0, 1.0)
    clock = PortraitPerformanceClock(motion=motion, interval=lambda: 3.0)
    assert clock.sample("rest", now=0.0).role == "rest"
    assert clock.sample("small", now=3.0).role == "blink_small"
    assert clock.sample("open", now=3.05).role == "blink_open"
    assert clock.sample("open", now=3.21).role == "open"
```

Mounted assertions must compare actual rendered pixels and child identity, not only chosen role. Import `avatar_db` with the reused Console fixtures so pytest resolves their database dependency. Apply a replacement with unchanged source bytes, block the old decoder, Remove/reapply the same bytes, resume the old decoder and assert the newest revision's pixels remain. Capture preparation bytes before/after unmount. Test stale/unknown speech as closed, owned silence, manual/historical moods, archived/missing assets and no usable set.
- [ ] **2. Run the new files red.** Preserve existing static/animation and off-loop fit tests as controls.
- [ ] **3. Add motion tokens before implementing pose and bounded preparation.** Define `$ds-portrait-blink-min: 3s`, `$ds-portrait-blink-max: 7s`, `$ds-portrait-idle-cycle: 6s`, `$ds-portrait-idle-offset-px: 1` centrally and document them. `load_performance_motion_values()` reads only those packaged scalar definitions plus `$ds-duration-fast` once during off-thread preparation, validates units/ranges, and returns the typed values; use a bounded cached read, not build-script imports, a generic token framework or per-frame file I/O. Invalid/missing motion data selects static fallback. Reuse existing image preparation reservations/renderers and effective motion rules. Normalize role order once, share current/next renderable ownership and close every partial preparation on failure. Apply idle offset inside a fixed canvas; never move TCSS geometry. Test runtime/token equality, missing tokens and partial allocation failure.
- [ ] **4. Integrate source/revision-aware resolution and observation.** Fetch attachment data off-thread only when activation/source/performance revision changes. Add exact local authority, canonical source, revision or absent tombstone, renderer/geometry and motion generation to request/spec/preparation fences. On historical replay use matching source identity and restore current mood afterward; manual selection still wins. Unknown or another actor's speech cannot animate this character. Source/motion/session/profile changes invalidate old work.
- [ ] **5. Verify all motion/lifecycle paths.** Static/Reduce motion shows the existing poster and no animated meter/effect; reactions-off retains ADR-144 precedence. Unprepared assets use static posters during owned speech. Hide/reveal resamples current speech without replaying hidden time. Resize does not restart audio. The existing image viewer snapshots the current fitted frame without owning a timer.
- [ ] **6. Rebuild CSS; run new files, token/bundle guards and targeted avatar, expression-playback, expression-state and off-loop geometry files green, then commit.** Suggested message: `feat: animate character portraits from owned speech and six poses`. Update implemented Console/character architecture notes.

### Unit 6 — TASK-36547: imported artwork preparation and review

**Files:** Create `Character_Chat/portrait_performance_authoring.py`, `Widgets/Persona_Widgets/portrait_performance_editor.py`, `Tests/Character_Chat/test_portrait_performance_authoring.py`, `Tests/UI/test_personas_portrait_performance.py`. Modify `personas_character_editor_widget.py`, `personas_pane_messages.py`, `UI/Screens/personas_screen.py`; use token-backed source widget CSS and rebuild.

**Consumes:** Unit 1 storage/source and Unit 5 preview renderer. **Produces:**

```python
class PortraitPerformanceDraft:
    def __init__(self, source: PerformanceSource, source_bytes: bytes, *, expected_revision: int) -> None: ...
    def stage(self, frame: PerformanceFrame) -> None: ...
    def frames(self) -> tuple[PerformanceFrame, ...]: ...
    def cancel(self) -> None: ...
class PortraitPerformanceEditor(Widget): ...  # inline child; owns no database/provider authority
class CharacterPerformancePrepareRequested(Message): ...  # carries requested source expression
# PersonasScreen source/authority capture and async handlers:
def _capture_portrait_performance_authority(self) -> object | None: ...
async def _import_portrait_performance_frame(self, role: FrameRole, path: Path) -> None: ...
async def _apply_portrait_performance(self) -> bool: ...
async def _remove_portrait_performance(self) -> bool: ...
```

The ephemeral authority value is an existing-style frozen capture of actual DB/service/profile/editor token and persisted binding/source; compare those values rather than inventing a serializable permission token. The view emits Import/Apply/Remove/Cancel messages to PersonasScreen through its normal parent path. No new screen or modal authority is introduced.

- [ ] **1. Add red draft and mounted tests.** Reuse `personas_editor_with_saved_character` / `personas_editor_with_bound_pack` from `Tests/UI/test_personas_expression_slots.py`; import their `expr_db` dependency too. Add the following draft assertions with a source/PNG fixture seeded as in Unit 1:

```python
def test_incomplete_draft_cannot_publish(source, source_bytes, frames):
    draft = PortraitPerformanceDraft(source, source_bytes, expected_revision=0)
    draft.stage(frames[0])
    with pytest.raises(ValueError):
        draft.frames()
    for frame in frames[1:]:
        draft.stage(frame)
    assert tuple(frame.role for frame in draft.frames()) == FRAME_ROLES
    draft.cancel()
    with pytest.raises(ValueError):
        draft.frames()
```

Mounted tests stage an unsaved replacement via `set_avatar_image` and assert preparation is refused with Save guidance and no implicit DB write. Unrelated dirty text remains allowed. Include server/local same-ID selection, profile/service replacement, source/binding changes during import, mismatched frames, transaction failure, Cancel, Remove, editor reopen and same-source Console handoff.
- [ ] **2. Run new files red.** Assertions must enter actual screen handlers and workers; invoking normalization alone does not prove UI admission.
- [ ] **3. Implement the source-bound inline view and handlers.** Reuse `_local_character_actions_allowed`, existing editor session tokens, actual DB object identity, visual source capture and the `EnhancedFileOpen`/Filters pattern. Read/normalize off-thread; only stage returned data after source/authority revalidation. Guard duplicate workers and clear busy state in `finally`. Use existing finite-operation DB guards inside worker callbacks.
- [ ] **4. Implement offline preview and Apply/Remove/Cancel.** Cycle all six roles deterministically and compare with source using the Console fit/renderer and current measured image-box budget. Also support real idle/blink preview. Apply requires all roles and expected revision; publish a local revision invalidation after commit. Cancel discards the draft, never the active set. Display profile-local/export limitations. Remove retains a revision tombstone via Unit 1.
- [ ] **5. Run mounted and pure tests green, plus affected Personas expression/visual-pack tests and token/CSS checks.** Prove import-to-Console changed pixels, keyboard focus restoration and no active source or conversation change. Update user guidance for ready-made imports and local storage.
- [ ] **6. Commit.** Suggested message: `feat: review and apply imported portrait performance artwork`.

### Unit 7 — TASK-36548: reference generation and guarded sample speech

**Files:** Modify `Character_Chat/expression_generation.py`, `UI/Screens/personas_screen.py`, `portrait_performance_editor.py`, `portrait_performance_authoring.py` only for draft state needed by these actions. Create `Tests/Character_Chat/test_portrait_performance_generation.py`, `Tests/UI/test_personas_portrait_performance_generate.py`; extend guarded speech tests from Unit 4.

**Consumes:** Units 4/6 and existing `resolve_backend_reference_image_capability`, request builder/`run_generation`, output validation and `CharacterRef`/destination resolution.

**Produces:**

```python
def compose_performance_prompt(*, role: FrameRole, expression_key: str) -> str: ...
async def _generate_portrait_performance_frames(self, roles: tuple[FrameRole, ...]) -> None: ...
async def _preview_portrait_performance_speech(self) -> None: ...
```

The latter two are PersonasScreen handlers; keep the worker loop there alongside existing authoring workers rather than creating another generation service/adapter. The inline view only requests these actions and presents their state.

- [ ] **1. Write red generation tests with fixed captured source and adapter calls.** Record role requests and stage results through the real screen worker. Mock only the external adapter and provide invalid/valid bytes as controls. Assert this behavior in the batch/selected regeneration case:

```python
assert requested_roles == list(FRAME_ROLES)
assert len(adapter_calls) == 6
assert db.get_character_portrait_performance(
    character_id=source.character_id, expression_key=source.expression_key
) is None  # fresh fixture, no Apply yet
before = {frame.role: frame.png for frame in draft.frames()}
await screen._generate_portrait_performance_frames(("small",))
assert len(adapter_calls) == 7
assert all(frame.png == before[frame.role] for frame in draft.frames() if frame.role != "small")
```

Use the Unit 6 mounted fixture and invoke the six-role worker before these assertions. Pin identity/background/framing directions for all roles, animated source frame-zero extraction retaining original digest, capability refusal, no alternate provider, a failed middle slot with other successes retained, uncertain outcome without retry, duplicate clicks, changed source/profile/editor and uncancellable late results. The invalid-frame control must fail normalization before staging.
- [ ] **2. Run the new generation files red.** Missing TTS must leave generation/import/draft Apply usable; it is not a generation failure.
- [ ] **3. Add closed role directions and sequential generation.** Extract one validated still reference from the captured asset; submit through the configured existing engine in workers. Revalidate authority before each admission and after results. Keep successful slots on failure, report the failed role, and offer only explicit selected retry. Display `Generate 6 frames`, one-request regeneration, source destination/provider/model and uncertainty without inventing a cost estimate.
- [ ] **4. Add the explicit speech sample.** Use fixed sample copy `Hello. This is a portrait preview. Notice the pauses.` through `speak_guarded_utterance`, current selected-character voice resolution and existing destination checks. Pass an observer owned only by this draft. Do not insert messages, enable auto-speak or invoke speech on view open. Cancel/close/source switch cancels that task and waits for its exact-owner stop acknowledgement; late cleanup cannot stop a newer request.
- [ ] **5. Run new pure/mounted generation tests and affected existing expression-generation/guarded speech cases green.** Test no TTS, destination refusal, preview failure, sample A superseded by B and B surviving A cleanup. Confirm generated and imported frames share exactly one normalization/Apply path.
- [ ] **6. Commit.** Suggested message: `feat: generate portrait poses and preview owned character speech`. Update implemented image-generation and user guidance; record actual provider qualification separately in Unit 9.

### Unit 8 — TASK-36549: stable radio frame and status

**Files:** Modify `UI/Screens/chat_screen.py`, `UI/Console_Modules/character.py`, source Character-section widget CSS, `css/core/_variables.tcss` and `css/features/_console_panels.tcss`; regenerate `css/tldw_cli_modular.tcss`. Create `Tests/UI/test_console_radio_portrait.py`; extend relevant Console rail/avatar geometry tests.

**Consumes:** Units 4/5 owner snapshot/current frame. **Produces:** Stable frame/status/meter using existing Character controls; pure `portrait_radio_status(snapshot: SpeechPresentationSnapshot | None) -> str` in `Chat/portrait_performance_state.py` for one consistent status mapping.

- [ ] **1. Write red status and mounted geometry tests.** Use the real Console harness in `Tests/UI/test_console_character_avatar.py` and production styles; reuse `_geometry_snapshot` pattern from `test_console_edge_rail_geometry.py`. In the status-only fallback case assert:

```python
assert portrait_radio_status(None) == "Ready"
assert portrait_radio_status(playing_without_timing) == "Transmitting"
assert screen.query_one("#console-character-voice-status").render().plain == "Transmitting"
assert screen.query_one("#console-character-voice-meter").display is True
assert screen.query_one("#console-character-voice-meter").visible is False
assert before_bounds == after_bounds
```

Construct `playing_without_timing` using Unit 2 Snapshot with state `playing`, invalid timing and rest mouth in an ordinary layout; capture `before_bounds`/`after_bounds` across that real publication. Extend the outer-rail geometry snapshot with the actual portrait, meter row and Character controls so equal outer bounds cannot conceal an internal shift. Give the status and meter these stable IDs. Across timing loss/recovery and rest/transmission, assert unchanged row height, image box and controls. In a separate compact-layout case assert meter `display is False`, and that speech changes alone never toggle compactness. Also assert Preparing voice is tied to a speech request, pause is transport-confirmed, silence/underrun empties the meter, failure access is readable, and text-stream-only activity is not transmission.
- [ ] **2. Run the new radio file red** with the existing narrow/tall rail cases as controls.
- [ ] **3. Compose token-backed framing inside measured Character budget.** Reserve identification/status/meter rows before fitting the image. Preserve authored portrait colors; use existing surface/status/focus tokens and canonical classes, including the motion values already added in Unit 5. Add any missing frame-specific values centrally before use; do not introduce unrelated theme literals or a new settings surface. A brief effect uses `$ds-duration-fast` and fixed border geometry.
- [ ] **4. Integrate statuses and capability/motion fallback.** Suppress the meter for invalid timing and suppress all animated meter/effect work under Static/Reduce motion. In ordinary layouts use a token-backed class with `visibility: hidden` for suppressed meter contents, retaining `display: block` and the reserved row; no extra wrapper is required. Only compact layout uses `display: none` to remove that row before controls become inaccessible. Compactness follows measured layout budget, not voice state or timing availability; restore visibility from current capability/motion state after resize. Keep existing keyboard bindings, rail scrolling, contain fit, focus and viewer behavior.
- [ ] **5. Rebuild CSS and run new/affected geometry tests plus token and bundle guards green.** Cover widths 80/120/200 and heights 24/45/80, with active speech, long character names and resized/collapsed rails. Equal geometry and stable widget identity must hold through state transitions; do not replace these checks with screenshots alone.
- [ ] **6. Commit.** Suggested message: `feat: frame living portraits with truthful radio voice status`. Update the Console appearance guide with actual behavior.

### Unit 9 — TASK-36550: integrated qualification and documentation

**Files:** Create `Tests/ProductionApp/test_console_radio_portraits.py`, `Tests/Backup_Recovery/test_portrait_performance_restore.py`; extend `Tests/Actor_Packs/test_actor_pack_export.py`, `Tests/Chatbooks/test_local_chatbook_service_export.py` and `Tests/Character_Chat/test_character_backup_export_image.py`. Create evidence under `Docs/superpowers/qa/console-radio-portraits/` with a dated run record. Update `Docs/Architecture/{character-chat,speech,media-generation}.md`, `Docs/User_Guide/console/voice-and-hands-free.md` and `Docs/User_Guide/roleplay-chat-dictionaries/characters-and-personas.md`.

**Consumes:** Units 1–8 as installed application behavior. **Produces:** Reproducible test/evidence records and accurate user documentation, no new feature subsystem.

- [ ] **1. Add an installed-path red-to-green journey.** Extend the existing production app/private-profile harness, not a fake Console. Import six marked frames, inspect offline transitions, Apply, open Console, drive owned sample audio, reopen the editor, Remove and reapply. Assertions must include:

```python
assert reopened.revision == applied_revision
assert reopened.source.source_sha256 == original_source_sha256
assert visible_pixels_after_speech != visible_pixels_at_rest
assert original_portrait_bytes_after == original_portrait_bytes_before
assert stale_owner_cannot_repaint
```

Bind these values to the installed database/widget/owner observations in the journey; no helper-only proof. Add full-profile backup/restore using the existing native recovery harness, with fresh post-restore application open and a blocked old-owner control. Assert restored frame hashes/revisions, unchanged portable exports and no new sync tables/records. Do not count an archive existing as a successful restore.
- [ ] **2. Run the new production/recovery cases and targeted exports.** The implemented flow must pass, and deliberate stale-source/old-owner controls must be rejected. Check migrated and fresh profiles; close owned worker/database resources after the last callback.
- [ ] **3. Qualify real image editing in an isolated profile.** Select a configured reference-capable backend; capture source, destination, six staged frames, one selected regeneration, offline/idle preview and explicit Apply. Review identity/framing consistency and no jumps at actual rail size. No configured backend means this AC remains open; mocks cannot replace it.
- [ ] **4. Qualify real streaming and eligible WAV artifact speech separately.** Use a known pause-heavy waveform and real TTS. Record audio/device/backend, visual onset, nonzero-background pause closure, underrun/resume where reproducible, stop/interruption, hands-free sentence handoff and conversation switch. Include audible final-syllable/device-tail completion, no early next sentence or listening, and Stop/replacement during a pending tail. Verify continuous portrait updates without test-only helper calls. Score the 100 ms target using validated output-DAC and actual frame-paint timestamps mapped to the same monotonic clock; preserve real playback capture for perceptual checks and report device latency separately. State whether independent physical loopback measurement was available rather than claiming one from driver timestamps. Exercise an unqualified external-player fallback. A platform/player not exercised remains a named coverage limit, not a success.
- [ ] **5. Run the affected-file regression gate and update docs.** Include new files plus relevant expression, owner, sink, Personas, recovery, migration, token and CSS checks. No full sweep without explicit approval. Record exact checkout/runtime identity, commands, results, timing measurements and baseline failures. Source docs describe what shipped: six-request generation, imports, local storage/export limit, motion settings, real timing capability and safe fallback.
- [ ] **6. Self-review and commit the qualification unit.** Suggested message: `test: qualify Console radio portraits end to end`. Mark implementation tasks Done only when their AC, static checks, documentation, review and actual qualification obligations are satisfied; keep missing live evidence open rather than recording completion from mock tests.

## Spec coverage and execution handoff

| Spec requirement | Owner |
| --- | --- |
| Six roles, validation, atomic persistence, source digest, CAS, purge | 1 |
| Energy/silence/hysteresis and ephemeral snapshots | 2 |
| DAC/source/device timing, partial/full underrun, periodic bounded observer, backend-confirmed natural drain, immediate interruption, WAV/fallback | 3 |
| Manual/auto/hands-free/preview ownership, pause, exact stop and destination authority | 4 |
| Mood precedence, historical replay, cache revisions, blink/idle, lifecycle/budgets/viewer | 5 |
| Persisted local-source gate, imported frames, offline preview, Apply/Remove/Cancel | 6 |
| Six reference requests, per-slot retry, partial/stale results, explicit TTS sample | 7 |
| Radio layout/status/meter, theme tokens, compact controls, accessibility | 8 |
| Installed path, actual artwork/audio quality, recovery/export/sync limits and complete docs | 9 |

Planning self-review must verify all nine task records, dependency ordering, exact interface names/types, eight Review Focus test allocations, local links, no unresolved design placeholders and proportionate detail. Recheck natural-drain ownership/watchdog behavior, publication without direct helper calls and ordinary versus compact meter geometry. New tasks stay To Do with no implementation plan/notes content until execution begins; this shared plan provides their prepared sequence.

The user authorized inline implementation on 2026-10-23, beginning with TASK-36542 and requiring a review of each atomic task before the next begins. Execute sequentially in the attached `codex/console-radio-portraits` worktree. TASK-36542's [implementation review](../reviews/2026-10-23-task-36542-review.md) records its storage verification and baseline limits. This authorization does not include pushing, merging or running a full test sweep. External image-generation and real-audio qualification remain explicit reviewed workflow actions in their owning units.
