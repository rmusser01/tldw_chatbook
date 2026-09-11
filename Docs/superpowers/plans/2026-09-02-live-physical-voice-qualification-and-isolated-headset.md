# Live Physical Voice Qualification and Isolated-Headset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace asserted physical-voice metrics with production-path measurements and safely enable full-duplex interruption on demonstrably isolated headsets.

**Architecture:** Keep `VoicePreprocessor` as the single speech-admission choke point and add one bounded, dependency-free isolation monitor beside its existing AEC health check. Reuse `DuplexAudioTransport` and its injected backend for a live physical runner; keep report building/schema validation pure, source-bound, and incapable of trusting operator-entered metrics.

**Tech Stack:** Python 3.11+, asyncio, stdlib PCM/math/wave support, existing `sounddevice` transport, native WebRTC AEC3 companion, JSON Schema, pytest, Ruff.

---

## Scope and decision record

ADR required: yes

ADR path: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md`

Reason: This changes the long-lived runtime safety classification for full duplex and
the source-bound physical-evidence contract. ADR-098 is amended rather than duplicated.

Governing spec:
`Docs/superpowers/specs/2026-09-02-live-physical-voice-qualification-and-isolated-headset-design.md`.

Do not mark `TASK-23175` Done after macOS USB evidence alone. Default rollout remains
hard off until the complete platform/device matrix required by ADR-098 exists.

## Fixed safety policy

The initial internal-only isolation thresholds are deliberately conservative and are
not user settings:

```python
ISOLATION_WINDOW_FRAMES = 100          # 1 second at 10 ms/frame
ISOLATION_REQUIRED_WINDOWS = 5         # 5 clean rendered seconds to open
ISOLATION_SAMPLE_STRIDE = 8            # bounded CPU; 6 kHz analysis
ISOLATION_MAX_LAG_MS = 500
ISOLATION_RENDER_RMS_MIN = 512.0        # about -36 dBFS
ISOLATION_MAX_CORRELATION = 0.12
ISOLATION_MAX_LEAKAGE_DB = -30.0
ISOLATION_FLOOR_DB = -120.0             # finite report representation
ISOLATION_METRIC_SAMPLES_MAX = 3_600    # one bounded hour at 1 Hz
ISOLATION_FRAME_BYTES = 960             # 48 kHz mono PCM16, 10 ms
ISOLATION_NEAR_END_CONFIRM_FRAMES = 5   # bounded 50 ms decision
ISOLATION_NEAR_END_RENDER_DOMINANCE_CORRELATION = 0.85
```

Opening requires every threshold to pass across five consecutive eligible windows.
One completed unsafe window demotes immediately. Missing timing, route mismatch,
overflow, saturation, native-processing failure, or ambiguous correlation is unsafe.
Tests inject smaller window counts, but the production constructor does not read these
values from configuration.

## File map

- Create `tldw_chatbook/Audio/acoustic_isolation.py`: bounded correlation/leakage
  monitor and immutable content-free observations.
- Modify `tldw_chatbook/Audio/duplex_contracts.py`: safety-path enum and snapshot.
- Modify `tldw_chatbook/Audio/voice_preprocessor.py`: integrate the monitor at the
  existing AEC/VAD admission boundary.
- Modify `tldw_chatbook/Chat/console_speculative_voice.py`: carry explicit safety path
  through capability and route-ready events.
- Modify `tldw_chatbook/Chat/console_speculative_voice_session.py`: advertise the
  preprocessor's full safety snapshot.
- Modify `tldw_chatbook/Audio/duplex_transport.py`: expose only the missing content-free
  open/control-overflow diagnostics needed by qualification.
- Create `Packaging/physical_voice_runner.py`: live trials, default-route observation,
  prompts, counters, bounded render backpressure, and cleanup.
- Add `Packaging/assets/voice_physical_reference.wav` and
  `Packaging/assets/voice_physical_reference.json`: hash-verified CC0 deterministic
  speech-shaped reference and provenance.
- Modify `Packaging/voice_physical_reports.py` and
  `Packaging/voice_physical_report.schema.json`: version-2 observations, derivations,
  and closed three-path safety predicate.
- Modify existing physical fixtures and add
  `Tests/Packaging/fixtures/speculative_voice_physical/safe_isolated_full_duplex.json`.
- Add `Tests/Audio/test_acoustic_isolation.py` and
  `Tests/Packaging/test_physical_voice_runner.py`; extend the existing preprocessor,
  session, report, manifest, and integration tests.
- Update the source/lint path inventories and physical-qualification documentation.

### Task 1: Add the dependency-free isolation monitor

**Files:**

- Create: `tldw_chatbook/Audio/acoustic_isolation.py`
- Modify: `tldw_chatbook/Audio/duplex_contracts.py:10-31`
- Create: `Tests/Audio/test_acoustic_isolation.py`
- Modify: `Tests/Audio/test_duplex_contracts.py`

- [ ] **Step 1: Write the RED safety-path contract tests**

Add `AcousticSafetyPath` with exactly `warming`, `aec`, `acoustic-isolation`, and
`half-duplex`, plus a frozen `AcousticSafetySnapshot` that rejects inconsistent pairs
and non-enum runtime values. Add `NearEndDisposition` with exactly `pending`, `admit`,
and `fence`, plus a frozen `AcousticIsolationObservation` containing the safety
snapshot and an optional near-end disposition.

Add `AcousticDemotionReason` as a closed string enum containing exactly the 17 reasons
emitted by the monitor: `ambiguous-correlation`, `capture-sequence-gap`,
`correlated-render`, `inaudible-render`, `invalid-capture-pcm`, `invalid-render-pcm`,
`missing-render-reference`, `missing-timing`, `native-processor-failure`,
`render-reference-failure`, `render-reference-gap`, `render-reference-overflow`,
`render-sequence-gap`, `route-mismatch`, `saturation`, `timing-discontinuity`, and
`unbounded-timing`. Require enum instances or `None` in snapshots. Require
`type(route_generation) is int` and a non-negative value; booleans and fractional
numbers are invalid. `FENCE` is inconsistent with any snapshot whose admission is open,
including the AEC path.

```python
def test_isolation_snapshot_is_the_only_non_aec_full_duplex_path() -> None:
    snapshot = AcousticSafetySnapshot(
        path=AcousticSafetyPath.ACOUSTIC_ISOLATION,
        mode=DuplexMode.FULL_DUPLEX,
        route_generation=4,
        admission_open=True,
        demotion_reason=None,
    )
    assert snapshot.mode is DuplexMode.FULL_DUPLEX

@pytest.mark.parametrize("path", [
    AcousticSafetyPath.WARMING,
    AcousticSafetyPath.HALF_DUPLEX,
])
def test_closed_paths_cannot_claim_full_duplex(path: AcousticSafetyPath) -> None:
    with pytest.raises(ValueError):
        AcousticSafetySnapshot(path, DuplexMode.FULL_DUPLEX, 0, True, None)
```

- [ ] **Step 2: Run the contract tests and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest -q Tests/Audio/test_duplex_contracts.py Tests/Audio/test_acoustic_isolation.py
```

Expected: collection fails because the new enum, snapshot, and module do not exist.

- [ ] **Step 3: Write RED deterministic isolation cases**

Build frames from deterministic sequences and cover:

- audible render plus uncorrelated quiet capture opens after exactly five windows;
- copied/delayed render never opens;
- a route generation change clears all accumulated evidence;
- near-end VAD during warmup restarts the clean-window count;
- realistic-amplitude near-end VAD after opening returns `pending` for four frames and
  `admit` on the fifth without contaminating stats;
- coherent copied or delayed render at the same lag and sign for five VAD-positive
  frames returns `fence` and revokes an existing proof;
- alternating frame-local DC offsets cannot hide an exactly copied AC component;
- one correlated or saturated window after opening demotes immediately;
- a post-AEC VAD-positive five-frame event with coherent render correlation fences
  either full-duplex path rather than being admitted as user speech;
- missing references, sequence gaps, unbounded timing, and processor failure cannot
  return or preserve an isolation proof;
- PCM payloads other than exactly 960 bytes and render iterables exceeding the bounded
  history capacity fail closed without unbounded allocation or iteration;
- non-bytes-like PCM fails closed before length/decoding, and admitted-event state resets
  after a VAD-negative frame so the next speech event begins at `pending`;
- analysis sample arrays are finite, bounded, and rounded once.

Use a test constructor with ten frames/window and two required windows so the suite is
fast. Mutation-check by temporarily raising the correlation result to the passing side;
the delayed-render test must fail.

- [ ] **Step 4: Implement the minimal monitor**

Use only `array`, `collections.deque`, `dataclasses`, `math`, and `statistics`. Retain
at most the configured lag history plus one current window. Downsample by the fixed
stride, remove each frame vector's mean before concatenation, and scan frame-aligned
lags within the lesser of 500 ms or the reported route latency plus its fixed guard.
Compute normalized cross-correlation and the least-squares correlated leakage gain,
clamp silence to `ISOLATION_FLOOR_DB`, then discard PCM vectors after the window closes.
Keep at most `ISOLATION_METRIC_SAMPLES_MAX` one-per-window numeric observations.

Require exactly `ISOLATION_FRAME_BYTES` before decoding. Consume no more render frames
per observation than the monitor's bounded render-history capacity; encountering one
additional item fails closed. Near-end classification retains exactly five contiguous
mean-centered frame pairs. For each allowed lag, require all five per-frame dot products
to have the same non-zero sign, then sum the five dot products and energies. Return
`fence` only when the event-level absolute normalized correlation exceeds
`ISOLATION_NEAR_END_RENDER_DOMINANCE_CORRELATION` and aggregate leakage exceeds the
existing `ISOLATION_MAX_LEAKAGE_DB` threshold at that lag; otherwise return `admit` on
the fifth frame. Return `pending` before the fifth frame. Missing evidence returns `fence`
immediately. After `admit`, return `admit` immediately for each subsequent contiguous
VAD-positive frame. A VAD-negative break ends the event and clears near-end evidence
without revoking an otherwise valid route proof.

Store the render-DAC timestamp with each bounded render vector and the capture-ADC
timestamp with each capture vector. A render pause clears incomplete capture/near-end
windows and resets sequence-boundary trackers, but retains references that are still
inside the current capture's lesser-of-500-ms-or-route-latency-plus-guard range. Prune
older references before classification. Route changes and unsafe timing still clear all
history. Add a public regression that opens proof, pauses briefly, then resumes with
validly delayed copied render and requires `pending` ×4 then `fence`; add a long-pause
case proving stale pre-pause render is not treated as current echo.

Keep two explicit timestamp predicates. The required current render reference proves
scheduled-reference presence and render energy; accept its future DAC timestamp only
when the difference between `render_dac_ns - capture_adc_ns` and
`delay_ms * 1_000_000` is at most `500_000 ns`, matching the transport's rounded-ms
delay contract. A lag candidate used for correlation must satisfy
`0 <= capture_adc_ns - render_dac_ns <= min(500 ms, delay + 50 ms)`. Add the production
transport timing shape (`capture_adc_ns=4_980_000_000`,
`render_dac_ns=5_020_000_000`, `delay_ms=40`) and prove isolation can qualify while
future lag-zero render is excluded from correlation.

Add surgical conjunction tests: perfect correlation with aggregate leakage at or below
`-30 dB` admits, and event-level correlation above `0.85` plus adequate gain but one
opposite-sign frame admits. These must fail if either the leakage or all-five-same-sign
branch is removed.

Lock the `0.85` event threshold with deterministic multi-seed tests: 5,000 independent
events each for white and AR-like `0.8`, `0.9`, and `0.95` capture/render, the maximum
51-lag search, copied/inverted/delayed pure echo, and echo-plus-near-end mixtures. The
fixture generator must be deterministic and bounded. The selected threshold must keep
all 20,000 independent events admissible, catch every pure-echo lag/sign case, catch at
least 99% of render-dominant `2:1` mixtures, and admit at least 99% of equal-energy
`1:1` mixtures. These are deterministic implementation regressions; live physical
qualification remains the rollout gate.

Core boundary:

```python
class AcousticIsolationMonitor:
    def observe(
        self,
        *,
        capture: AudioFrame,
        render_frames: Iterable[AudioFrame],
        assistant_rendering: bool,
        near_end_speech: bool,
        native_processor_ok: bool,
    ) -> AcousticIsolationObservation: ...

    def reset_for_route(self, clock_generation: int) -> None: ...

    @property
    def correlation_samples(self) -> tuple[float, ...]: ...

    @property
    def leakage_db_samples(self) -> tuple[float, ...]: ...
```

Do not add numpy/scipy or a generic DSP framework. The monitor owns only isolation
evidence; AEC health remains in `VoicePreprocessor`.

- [ ] **Step 5: Run Task 1 tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add tldw_chatbook/Audio/acoustic_isolation.py tldw_chatbook/Audio/duplex_contracts.py Tests/Audio/test_acoustic_isolation.py Tests/Audio/test_duplex_contracts.py
git commit -m "feat: add acoustic isolation safety monitor"
```

### Task 2: Integrate the safety path at the production admission choke point

**Files:**

- Modify: `tldw_chatbook/Audio/voice_preprocessor.py:89-360`
- Modify: `tldw_chatbook/Chat/console_speculative_voice.py:121-140,218-231,1290-1298`
- Modify: `tldw_chatbook/Chat/console_speculative_voice_session.py:919-975`
- Modify: `Tests/Audio/test_voice_preprocessor.py`
- Modify: `Tests/Chat/test_console_speculative_voice.py`
- Modify: `Tests/integration/test_speculative_voice_pipeline.py`

- [ ] **Step 1: Write RED preprocessor admission tests**

Add cases proving:

```python
assert preprocessor.health is AecHealth.WARMING
assert preprocessor.safety.path is AcousticSafetyPath.ACOUSTIC_ISOLATION
assert preprocessor.mode is DuplexMode.FULL_DUPLEX
```

The fake AEC must successfully process capture while returning unavailable delay/ERLE
metrics. Feed five windows of audible, uncorrelated raw capture/render evidence, then
verify five realistic-amplitude uncorrelated VAD-positive frames are buffered and
admitted in original order during playback, including the first frame. Separately prove
that a thrown native call, correlated render, capture/reference overflow, or route reset
closes admission before the frame reaches `on_admitted_frame`.

Distinguish a well-formed but unavailable AEC estimate from a broken processor contract:
successful analyze/process calls plus finite metrics with `delay_estimate_available=0`
may use isolation; a missing processor, thrown call, missing metric key, non-finite
metric, unusable transport timing, or explicit native degraded state may not.

- [ ] **Step 2: Run focused preprocessor tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest -q Tests/Audio/test_voice_preprocessor.py
```

Expected: failures because `VoicePreprocessor` has no isolation safety path.

- [ ] **Step 3: Refactor `process_capture` to compute VAD once and select one path**

Inject `AcousticIsolationMonitor` with a production default when AEC is present. After
successful native processing, evaluate cleaned-frame VAD once, feed raw capture plus
render references to the monitor, update the AEC health state, then choose:

```python
if self._health is AecHealth.HEALTHY:
    path = AcousticSafetyPath.AEC
elif self._native_processor_ok and (
    self._isolation.safety.path is AcousticSafetyPath.ACOUSTIC_ISOLATION
):
    path = AcousticSafetyPath.ACOUSTIC_ISOLATION
elif self._aec is not None and self._native_processor_ok and (
    self._health is AecHealth.WARMING
):
    path = AcousticSafetyPath.WARMING
else:
    path = AcousticSafetyPath.HALF_DUPLEX
```

Only `AEC` and `ACOUSTIC_ISOLATION` may admit VAD-positive cleaned frames during
assistant render. On either path, keep a five-frame bounded queue owned by
`VoicePreprocessor`: `pending` retains the cleaned frame without callback, `admit`
releases all five frames sequentially through `on_admitted_frame`, and `fence` drops
the queue before any callback. A VAD-negative break also drops an incomplete queue but
does not revoke an otherwise valid proof. Outside assistant render, preserve existing
half-duplex raw-capture behavior. Native exceptions still use `_handle_aec_failure` and
must reset/demote the monitor and clear the queue. Route reset resets AEC and isolation
evidence and clears the queue.

Before that path selection, reject and fence a VAD-positive event whose five-frame
observation is render-dominant under the calibrated same-lag, same-sign event statistic.
Raw render correlation alone is expected on speaker routes and does not override
healthy AEC; render-dominant correlation plus residual post-AEC voice is the
contradictory signal that closes either full-duplex path. Equal-energy or near-end-
dominant mixtures remain user speech and are admitted. The maximum added barge-in
decision latency is 50 ms and is measured by the live qualification gate.

- [ ] **Step 4: Write RED capability-event tests**

Require `AudioCapabilityChanged` and `AudioRouteReady` to carry an explicit
`AcousticSafetyPath`. Prove full duplex with `AecHealth.WARMING` is rejected for `aec`
but accepted for `acoustic-isolation`; warming/half-duplex paths cannot claim open
admission. Prove stale isolation route-ready events are ignored exactly like stale AEC
events.

- [ ] **Step 5: Thread the path through session and coordinator**

Change `_advertised_capability` from `(mode, health)` to `(mode, health, path)`. Use the
preprocessor snapshot in both `AudioCapabilityChanged` and `AudioRouteReady`. Replace
the coordinator's `health = half-duplex or AEC healthy` predicate with validation of
the closed path/mode pair; do not teach the coordinator correlation logic.

- [ ] **Step 6: Run the integrated production-path tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest -q Tests/Audio/test_voice_preprocessor.py Tests/Audio/test_duplex_transport.py Tests/Chat/test_console_speculative_voice.py Tests/integration/test_speculative_voice_pipeline.py
```

Expected: all selected tests pass, including the existing real transport → AEC → VAD →
rolling STT → coordinator case.

- [ ] **Step 7: Commit Task 2**

```bash
git add tldw_chatbook/Audio/voice_preprocessor.py tldw_chatbook/Chat/console_speculative_voice.py tldw_chatbook/Chat/console_speculative_voice_session.py Tests/Audio/test_voice_preprocessor.py Tests/Chat/test_console_speculative_voice.py Tests/integration/test_speculative_voice_pipeline.py
git commit -m "feat: admit isolated headset speech safely"
```

### Task 3: Upgrade physical evidence to a derivable three-path schema

**Files:**

- Modify: `Packaging/voice_physical_reports.py:35-75,331-690`
- Modify: `Packaging/voice_physical_report.schema.json`
- Modify: `Tests/Packaging/test_voice_physical_reports.py`
- Modify: `Tests/Packaging/fixtures/speculative_voice_physical/safe_full_duplex.json`
- Modify: `Tests/Packaging/fixtures/speculative_voice_physical/safe_half_duplex.json`
- Modify: `Tests/Packaging/fixtures/speculative_voice_physical/unsafe_unsuppressed.json`
- Create: `Tests/Packaging/fixtures/speculative_voice_physical/safe_isolated_full_duplex.json`

- [ ] **Step 1: Write RED version-2 report tests**

Cover all three passing paths plus unsafe leakage. Assertions must prove:

- schema version 1 is rejected;
- `safety_path` is closed to `aec`, `acoustic-isolation`, `half-duplex`;
- isolation may use `null` ERLE only when native processing stayed operational;
- AEC full duplex still requires median ERLE ≥20 dB and p10 ≥10 dB;
- isolation full duplex requires five clean windows, correlation ≤0.12, leakage
  ≤-30 dB, zero render-only VAD events, double-talk recall ≥0.95, and stop p95 ≤150 ms;
- half duplex requires playback speech admission closed and manual stop available;
- any overflow, saturation, stale callback, route miss, or unsafe demotion makes the
  report fail;
- status-flagged callbacks inside a hard-bounded 500 ms startup pre-roll are discarded
  without advancing the app capture sequence and reset cadence comparison, while a
  status at callback 51 or later fails closed; callbacks rejected during native
  teardown are distinct from callbacks arriving after teardown completes;
- once an interruption phase selects manual fallback, later safety-path recovery cannot
  silently switch that phase back to acoustic interruption;
- changing an aggregate without changing its serialized samples is rejected.

Use half-step sample values to reproduce the prior rounding incident.

- [ ] **Step 2: Run report tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest -q Tests/Packaging/test_voice_physical_reports.py
```

Expected: version-2 fixture/schema assertions fail against the current version-1
builder.

- [ ] **Step 3: Implement canonical version-2 report construction**

Round sample arrays once to six decimals, then derive every median/percentile from the
rounded arrays. Store at most 3,600 one-per-second ERLE/correlation/leakage observations
and at most the fixed 20 interruption samples; enforce those limits in the schema. Add
sections shaped like:

```json
{
  "safety_path": "acoustic-isolation",
  "aec": {
    "health_path": "warming",
    "processor_operational": true,
    "erle_samples_db": [],
    "median_erle_db": null,
    "p10_erle_db": null
  },
  "isolation": {
    "eligible_windows": 5,
    "required_windows": 5,
    "correlation_samples": [0.02, 0.03],
    "p95_correlation": 0.03,
    "leakage_db_samples": [-48.0, -46.0],
    "p95_leakage_db": -46.0,
    "render_only_vad_events": 0,
    "demotions": 0
  },
  "transport_health": {
    "capture_overflows": 0,
    "render_overflows": 0,
    "reference_overflows": 0,
    "control_overflows": 0,
    "saturation_events": 0
  }
}
```

Add stop-latency samples under the interruption trial and derive p95 from them. Keep
the existing source/prerequisite/device identity and privacy validation. The builder,
not the observation producer, computes `passed`.

- [ ] **Step 4: Update closed JSON Schema conditionals**

Use `oneOf` for the three safety paths. Forbid ERLE numbers on an isolation report that
has no samples rather than inventing zero. Require all serialized sample arrays and
transport counters. Retain every existing forbidden-content check.

- [ ] **Step 5: Run report and manifest-consumer tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest -q Tests/Packaging/test_voice_physical_reports.py Tests/Packaging/test_voice_qualification_manifest.py
```

Expected: all pass; the manifest consumer accepts genuine version-2 physical reports
and rejects version-1 or forged aggregate reports.

- [ ] **Step 6: Commit Task 3**

```bash
git add Packaging/voice_physical_reports.py Packaging/voice_physical_report.schema.json Tests/Packaging/test_voice_physical_reports.py Tests/Packaging/fixtures/speculative_voice_physical
git commit -m "test: harden physical voice safety evidence"
```

### Task 4: Add the licensed physical reference and bounded live runner

**Files:**

- Create: `Packaging/assets/voice_physical_reference.wav`
- Create: `Packaging/assets/voice_physical_reference.json`
- Create: `Packaging/physical_voice_runner.py`
- Modify: `tldw_chatbook/Audio/duplex_transport.py:205-275,371-430`
- Modify: `Tests/Audio/fakes/fake_duplex_backend.py`
- Create: `Tests/Packaging/test_physical_voice_runner.py`

- [ ] **Step 1: Write RED asset validation tests**

Require a regular non-symlink file, exact manifest keys, CC0-1.0 declaration, exact
SHA-256, PCM16 little-endian, mono, 48 kHz, and whole 10 ms frames. Reject a copied
manifest with a changed hash or audio format before a transport is opened.

- [ ] **Step 2: Add the deterministic reference asset**

Generate a six-second, speech-shaped reference from the existing CC0 deterministic AEC
corpus recipe. Store its recipe/provenance and bytes hash in the adjacent manifest. Use
stdlib `wave` at runtime; do not add an audio decoder dependency. Verify the generated
asset with the RED test before retaining it.

- [ ] **Step 3: Write RED runner tests with the real transport seam**

Extend `FakeDuplexBackend` only enough to expose open/close counts and feed scripted
capture from rendered callback output. Tests inject the existing backend, a fake clock,
fake route reader, prompt callback, and short `LiveTrialConfig`. Cover:

- render backpressure waits for occupancy instead of incrementing overflow counters;
- safe echo, safe isolation, double-talk, and interruption samples come from processed
  frames and timestamps;
- three round-trip route trials each fence both transitions and return to the target;
- route acknowledgement without a changed identity fails;
- threshold failure returns observations suitable for a FAIL report;
- permission/open/format failure raises before report construction;
- cancellation closes the stream and releases runner-owned buffers;
- no bytes from capture or rendered speech appear in the returned observation mapping.

- [ ] **Step 4: Run the runner tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest -q Tests/Packaging/test_physical_voice_runner.py Tests/Audio/test_duplex_transport.py
```

Expected: collection fails because the runner and diagnostics do not exist.

- [ ] **Step 5: Add minimal transport diagnostics**

Expose read-only `is_open` and `control_overflows` properties under the existing lock.
Do not add a general diagnostics service. Existing capture/render/reference counters and
`buffer_occupancy` already cover the remaining runner needs.

- [ ] **Step 6: Implement the runner**

Use one `PhysicalVoiceTrialRunner` with injected callables rather than a new hierarchy:

```python
@dataclass(frozen=True, slots=True)
class LiveTrialConfig:
    soak_seconds: float = 1_800.0
    route_round_trips: int = 3
    interruption_trials: int = 20
    double_talk_trials: int = 20

async def collect_live_observations(
    *,
    device_class: str,
    config: LiveTrialConfig = LiveTrialConfig(),
    transport_factory: Callable[[], DuplexAudioTransport] = DuplexAudioTransport,
    route_reader: Callable[[], RouteIdentity] = read_default_route,
    prompt: Callable[[str], Awaitable[None]] = prompt_operator,
) -> dict[str, object]: ...
```

The public CLI always uses the default 30-minute config; short configs are test-only.
The default route reader imports `sounddevice` lazily, checks default input/output
capabilities, and keeps names/indices in memory only. The runner processes transport
capture/reference frames through `VoicePreprocessor`; it never fabricates health or VAD
events. Queue the next reference frame only when render occupancy is below capacity.

Keep human reaction time separate from detector-window duration. Before repeated
double-talk trials, allow up to ten seconds to detect real admitted microphone speech,
then require overlap detection when full-duplex admission is open. Collect all 20
double-talk windows from one continuous speaking block with immediate per-window
status. For full-duplex interruption, prompt once, issue automatic cues with up to ten
seconds for each response, and require one second of consecutive measured silence
between cues. A failed sample stops the remaining trials and returns partial failing
observations. Cover the complete 20-window/20-interruption flow and first-failure stop
with deterministic no-hardware tests before asking an operator to run it.

During render-only windows, any admitted speech increments false barge. During prompted
double-talk, admitted speech counts detection. For interruption, timestamp the first
admitted frame, call `abort_output`, and use the next callback's hardware output time
with no advanced render-reference sequence as the silence boundary. Track only
runner-owned tasks/streams for leak assertions.

On every route transition: detect a changed default identity, call the transport route
fence, drain its control event, reset the preprocessor with the new generation, reopen,
and prove admission is closed before calibration. In `finally`, fence/close the stream,
clear deques, overwrite mutable scratch, and release immutable frame references.

- [ ] **Step 7: Run runner/transport tests and verify GREEN**

Run the command from Step 4. Expected: all selected tests pass without opening local
hardware.

- [ ] **Step 8: Commit Task 4**

```bash
git add Packaging/assets/voice_physical_reference.wav Packaging/assets/voice_physical_reference.json Packaging/physical_voice_runner.py tldw_chatbook/Audio/duplex_transport.py Tests/Audio/fakes/fake_duplex_backend.py Tests/Packaging/test_physical_voice_runner.py
git commit -m "feat: measure live physical voice trials"
```

### Task 5: Replace manual metric entry in the physical CLI

**Files:**

- Modify: `Packaging/voice_physical_reports.py:674-924`
- Modify: `scripts/qualify_physical_voice.py`
- Modify: `Tests/Packaging/test_voice_physical_reports.py`
- Modify: `Tests/Packaging/test_physical_voice_runner.py`

- [ ] **Step 1: Write RED CLI-boundary tests**

Patch `collect_live_observations` at the module boundary and call `main()` in normal
mode. Assert the runner is called once, its derived observations produce the report,
and no prompt asks for ERLE, counters, latency, or pass/fail. Also cover:

- a completed unsafe run writes schema-valid `passed: false` and exits 1;
- an infrastructure/permission failure writes neither report nor summary;
- fixture mode never imports `sounddevice` or invokes the live runner;
- fixture mode retains the existing safe-AEC, safe-half, and unsafe command names and
  gains `safe-isolated-full-duplex`.

- [ ] **Step 2: Run CLI tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest -q Tests/Packaging/test_voice_physical_reports.py Tests/Packaging/test_physical_voice_runner.py
```

Expected: normal mode still calls `collect_guided_observations` and fails the tests.

- [ ] **Step 3: Delete manual metric collection and wire the live runner**

Remove every `_ask*` metric helper and `collect_guided_observations`.
Retain only start/cancel/route/speech timing prompts in the live runner. In normal mode,
recompute the source digest and load automated prerequisites before opening hardware,
then `asyncio.run(collect_live_observations(...))`. Fixture mode remains synchronous
and hardware-free.

Do not add CLI flags that can shorten the authoritative soak or set safety metrics.

- [ ] **Step 4: Run CLI tests and verify GREEN**

Run the command from Step 2. Expected: all pass.

- [ ] **Step 5: Commit Task 5**

```bash
git add Packaging/voice_physical_reports.py scripts/qualify_physical_voice.py Tests/Packaging/test_voice_physical_reports.py Tests/Packaging/test_physical_voice_runner.py
git commit -m "fix: require measured physical voice evidence"
```

### Task 6: Bind the new source and document the real operator workflow

**Files:**

- Modify: `Packaging/speculative_voice_source_paths.txt`
- Modify: `Packaging/speculative_voice_python_paths.txt`
- Modify: `Tests/Packaging/test_voice_source_digest.py`
- Modify: `Tests/Packaging/test_speculative_voice_lint_scope.py`
- Modify: `Docs/Development/TTS/speculative-voice-qualification.md`
- Modify: `Docs/Development/TTS/voice-aec-release.md`
- Modify: `Docs/Development/TTS/speculative-duplex-voice.md`
- Modify: `backlog/tasks/task-23175 - Add-low-latency-speculative-duplex-voice-pipeline.md`

- [ ] **Step 1: Write RED source-inventory tests**

Require the new runtime, runner, tests, WAV, asset manifest, approved spec, and this plan
in the source digest. Require only Python files in the Python lint scope. Mutating the
asset manifest or monitor source in a copied tree must change the digest.

- [ ] **Step 2: Update inventories and operator documentation**

Document the exact absolute-venv command, privacy behavior, audible sample warning,
timed speaking prompts, three round trips, 30-minute duration, FAIL-vs-abort behavior,
and the requirement to return to the target route. State that existing manually entered
physical reports are invalid and that changing any listed source invalidates all prior
evidence.

The report-generation order is:

1. freeze and commit all source-listed files;
2. compute one digest from a clean worktree;
3. regenerate automated, duplex-soak, and cancellation-soak evidence;
4. run physical device evidence from the same commit/digest;
5. validate every physical report through the strict consumer; and
6. build rollout authority only after the full platform/device matrix exists.

- [ ] **Step 3: Run documentation/inventory tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest -q Tests/Packaging/test_voice_source_digest.py Tests/Packaging/test_speculative_voice_lint_scope.py Tests/Packaging/test_voice_qualification_manifest.py
```

Expected: all pass.

- [ ] **Step 4: Commit Task 6**

```bash
git add Packaging/speculative_voice_source_paths.txt Packaging/speculative_voice_python_paths.txt Tests/Packaging/test_voice_source_digest.py Tests/Packaging/test_speculative_voice_lint_scope.py Docs/Development/TTS/speculative-voice-qualification.md Docs/Development/TTS/voice-aec-release.md Docs/Development/TTS/speculative-duplex-voice.md "backlog/tasks/task-23175 - Add-low-latency-speculative-duplex-voice-pipeline.md"
git commit -m "docs: bind live voice qualification source"
```

### Task 7: Run the focused implementation gate and freeze source

**Files:** All source-listed files touched by Tasks 1-6.

- [ ] **Step 1: Run the focused test gate**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest -q \
  Tests/Audio/test_duplex_contracts.py \
  Tests/Audio/test_acoustic_isolation.py \
  Tests/Audio/test_voice_preprocessor.py \
  Tests/Audio/test_duplex_transport.py \
  Tests/Chat/test_console_speculative_voice.py \
  Tests/integration/test_speculative_voice_pipeline.py \
  Tests/Packaging/test_physical_voice_runner.py \
  Tests/Packaging/test_voice_physical_reports.py \
  Tests/Packaging/test_voice_qualification_manifest.py \
  Tests/Packaging/test_voice_source_digest.py \
  Tests/Packaging/test_speculative_voice_lint_scope.py
```

Expected: all pass. Do not run the full repository suite unless the user explicitly
opts in, per `AGENTS.md`.

- [ ] **Step 2: Run scoped static checks**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check \
  tldw_chatbook/Audio/acoustic_isolation.py \
  tldw_chatbook/Audio/duplex_contracts.py \
  tldw_chatbook/Audio/voice_preprocessor.py \
  tldw_chatbook/Audio/duplex_transport.py \
  tldw_chatbook/Chat/console_speculative_voice.py \
  tldw_chatbook/Chat/console_speculative_voice_session.py \
  Packaging/physical_voice_runner.py \
  Packaging/voice_physical_reports.py \
  Tests/Audio/test_acoustic_isolation.py \
  Tests/Audio/test_voice_preprocessor.py \
  Tests/Packaging/test_physical_voice_runner.py \
  Tests/Packaging/test_voice_physical_reports.py

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check \
  tldw_chatbook/Audio/acoustic_isolation.py \
  tldw_chatbook/Audio/duplex_contracts.py \
  tldw_chatbook/Audio/voice_preprocessor.py \
  tldw_chatbook/Audio/duplex_transport.py \
  tldw_chatbook/Chat/console_speculative_voice.py \
  tldw_chatbook/Chat/console_speculative_voice_session.py \
  Packaging/physical_voice_runner.py \
  Packaging/voice_physical_reports.py \
  Tests/Audio/test_acoustic_isolation.py \
  Tests/Audio/test_voice_preprocessor.py \
  Tests/Packaging/test_physical_voice_runner.py \
  Tests/Packaging/test_voice_physical_reports.py

git diff --check
```

Expected: all pass with no source or test formatting changes required.

- [ ] **Step 3: Perform two explicit self-review passes**

First review correctness/privacy: follow every raw PCM reference, normal-mode input,
route reset, and report field; verify no durable owner can retain capture content and no
unsafe path opens admission. Second review minimalism/integration: grep all callers of
`AecHealth`, `AudioCapabilityChanged`, and `AudioRouteReady`; remove duplicate policy or
unused abstraction and confirm ordinary AEC/half-duplex behavior is unchanged.

The plan-review subagent required by the generic skill is intentionally unavailable in
this session because the active developer policy forbids spawning subagents unless the
user explicitly requests delegation. Do not silently substitute delegated review.

- [ ] **Step 4: Commit any review corrections, then freeze source**

Stage exact files only. Confirm the unrelated trace-ledger working-tree changes remain
unstaged. After the final source commit, do not edit any source-listed file until all
evidence for that digest is either accepted or deliberately discarded.

### Task 8: Regenerate macOS automated and physical evidence from a clean checkout

**Files generated (excluded from source digest):**

- `Artifacts/voice_qualification/automated/macos-arm64-automated.json`
- `Artifacts/voice_qualification/automated/macos-arm64-duplex-soak.json`
- `Artifacts/voice_qualification/automated/macos-arm64-cancellation-soak.json`
- `Artifacts/voice_qualification/physical/macos-arm64-usb.json`
- corresponding `.summary.txt` files and qualification documentation hashes

- [ ] **Step 1: Create a clean evidence worktree at the frozen commit**

The shared implementation worktree contains unrelated user-owned trace-ledger edits,
some on source-listed paths. Do not stash, revert, stage, or hash around them. Create a
separate clean worktree at the frozen voice commit and use the absolute parent venv.

```bash
git worktree add /private/tmp/tldw-voice-evidence-<commit> <frozen-commit>
```

- [ ] **Step 2: Compute and record the clean source digest**

```bash
cd /private/tmp/tldw-voice-evidence-<commit>
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python Packaging/compute_voice_source_digest.py
```

Expected: one lowercase SHA-256 and no dirty/untracked-source error.

- [ ] **Step 3: Regenerate automated evidence and validate it immediately**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/qualify_speculative_voice.py --scenario automated --source-tree-digest <digest> --output Artifacts/voice_qualification/automated/macos-arm64-automated.json
```

Run the strict manifest consumer test against the generated report before starting the
long soaks. Expected: report passes and all derived values agree with serialized values.

- [ ] **Step 4: Run both 30-minute automated soaks with periodic progress updates**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/qualify_speculative_voice.py --scenario duplex-soak --minutes 30 --source-tree-digest <digest> --output Artifacts/voice_qualification/automated/macos-arm64-duplex-soak.json
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/qualify_speculative_voice.py --scenario cancellation-soak --minutes 30 --source-tree-digest <digest> --output Artifacts/voice_qualification/automated/macos-arm64-cancellation-soak.json
```

Expected: both strict-valid and passing with zero overflow/leak/post-fence failures.

- [ ] **Step 5: Run the live USB qualifier**

Confirm macOS input and output are both the USB headset. Run the deterministic 20x20
operator-protocol simulation first. The live run uses one speech/overlap calibration,
one continuous double-talk block, one full-duplex interruption acknowledgement,
automatic `speak now` cues with measured quiet recovery, and immediate per-sample
results. Tell the user immediately before that single calibration and each route round
trip. Use:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/qualify_physical_voice.py \
  --source-tree-digest <digest> \
  --platform macos-arm64 \
  --device-class usb \
  --automated-report Artifacts/voice_qualification/automated/macos-arm64-automated.json \
  --output Artifacts/voice_qualification/physical/macos-arm64-usb.json
```

Expected: the operator participates only in the bounded calibrated speech flow and
route prompts; no repeated trial starts before calibration, audio is discarded, three
round-trip route trials complete, the remainder runs unattended, and the result is
schema-valid. The first failed interruption ends the remaining operator work and emits
partial FAIL evidence. PASS requires either measured AEC or measured isolation; FAIL
is an honest result, not a reason to edit thresholds after seeing it.

- [ ] **Step 6: Revalidate hashes and preserve evidence separately**

Recompute the source digest and require it to be unchanged. Run the physical report and
manifest-consumer tests against the generated evidence. Record report SHA-256 values in
`Docs/Development/TTS/speculative-voice-qualification.md`, then commit generated macOS
evidence separately from source.

Do not generate a qualified rollout manifest from this USB-only result. Built-in and
Bluetooth reports on macOS plus the complete Windows/Linux matrices remain required.

### Task 9: Complete the cross-platform device matrix and rebuild rollout authority

**Files generated (excluded from source digest):**

- `Artifacts/voice_qualification/automated/<platform>-*.json`
- `Artifacts/voice_qualification/physical/<platform>-<device-class>.json`
- `tldw_chatbook/Audio/voice_qualification_manifest.json`
- `tldw_chatbook/Audio/voice_build_identity.json`
- `Docs/Development/TTS/speculative-voice-qualification.md`

- [ ] **Step 1: Run the same frozen source on every supported platform/ABI**

Repeat Task 8's automated and two soak scenarios on `macos-arm64`, `macos-x86_64`,
`windows-x86_64`, `linux-x86_64`, and `linux-aarch64` using the repaired companion
wheel for that exact ABI. Every report must carry the same source digest and exact
companion upstream commit.

- [ ] **Step 2: Run each physical device class on every platform key**

Produce built-in, USB/wired, and Bluetooth reports through the live runner. A device
class may pass via AEC, acoustic isolation, or explicit safe half duplex; missing
hardware or an aborted run is missing evidence, not a passing degradation report.
Validate each report immediately before moving to the next long run.

- [ ] **Step 3: Generate and verify rollout authority**

After the matrix is complete, run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python Packaging/generate_voice_qualification_manifest.py \
  --source-tree-digest <digest> \
  --app-version 0.1.8.0 \
  --aec-upstream 109e23c9cec3a44e67c08774874a409741b1e58a \
  --wheelhouse <qualified-wheelhouse> \
  --automated-dir Artifacts/voice_qualification/automated \
  --physical-dir Artifacts/voice_qualification/physical \
  --output tldw_chatbook/Audio/voice_qualification_manifest.json \
  --build-identity-output tldw_chatbook/Audio/voice_build_identity.json

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest -q Tests/Packaging/test_voice_qualification_manifest.py Tests/Chat/test_console_voice_settings.py
```

Expected: all five platform/ABI entries are qualified, each physical matrix hash is
derived from three version-2 live reports, and the runtime reader accepts the exact
local extension bytes only.

- [ ] **Step 4: Commit generated authority separately**

Recompute the source digest before and after staging generated reports, qualification
docs, manifest, and build identity. It must remain unchanged. Commit only those
generated/authority paths and retain their hashes in the qualification document.

### Task 10: Close the implementation checkpoint honestly

**Files:**

- Modify: `backlog/tasks/task-23175 - Add-low-latency-speculative-duplex-voice-pipeline.md`
- Modify if warranted: `backlog/docs/lessons-testing-evidence.md`

- [ ] **Step 1: Add concise implementation notes**

Document the monitor, production integration, version-2 evidence, live runner, tests,
actual macOS evidence outcome, source digest, and remaining platform/device gates.
Check only acceptance criteria proven by tests/evidence.

- [ ] **Step 2: Keep the task In Progress unless every DoD item is actually met**

Do not mark Done while Windows/Linux or required device-class evidence is absent, while
the rollout authority remains hard off, or if the user has not opted into any required
full-suite gate. Record the precise remaining work rather than converting absence into
a passing half-duplex claim.

- [ ] **Step 3: Add a lesson only if this implementation produces new reusable evidence**

The manually asserted qualifier defect and isolated-headset no-echo finding are likely
lesson candidates, but write an incident-based entry only after the fixed live runner
proves the replacement behavior. Do not add a speculative rule.
