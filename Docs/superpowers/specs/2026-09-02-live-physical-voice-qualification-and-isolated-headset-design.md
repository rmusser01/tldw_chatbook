# Live Physical Voice Qualification and Isolated-Headset Safety Design

Status: Approved

Date: 2026-09-02

Related task: `TASK-23175`

Related ADR: [ADR-098](../../../backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md)

Extends: [Low-Latency Speculative Duplex Voice Pipeline](2026-08-28-low-latency-speculative-duplex-voice-pipeline-design.md)

## Problem

The existing physical qualification command is a report-entry form, not a physical
test. It asks an operator to type ERLE, false-barge, double-talk, interruption, route,
and soak values and then validates those assertions. It never opens the production
duplex transport, plays a reference, captures the microphone, or measures the values
it signs. A syntactically valid report therefore does not prove the hardware behavior
required by ADR-098.

The first real macOS USB-headset probe also exposed a distinct safety-state error. The
production transport opened the 48 kHz USB input/output route and captured without
overflow, but AEC3 remained `warming`: an acoustically isolated headset has little or
no observable render echo from which AEC3 can derive delay and ERLE. Treating missing
echo-path evidence as proof of unsafe leakage unnecessarily forces a safe isolated
route into half duplex.

This addendum replaces manually asserted physical metrics with live measurements and
lets the shared runtime prove full-duplex safety by either cancellation or acoustic
isolation. It does not weaken fail-closed behavior.

## Goals

- Measure physical qualification through the same transport, AEC, preprocessing, VAD,
  route fencing, and admission logic used at runtime.
- Keep speech admission closed during playback until the current route proves one of
  two closed safety paths: effective AEC or effective acoustic isolation.
- Let USB and wired headsets use full duplex when their microphone is demonstrably
  isolated from rendered audio even if AEC delay/ERLE never becomes observable.
- Produce privacy-safe, source-bound physical evidence without saving microphone PCM,
  transcribing the operator, or trusting typed measurements.
- Preserve fixture mode as deterministic schema/builder coverage.

## Non-goals

- Acoustic isolation is not a replacement for AEC on speaker/microphone routes.
- Qualification does not bypass the packaged rollout manifest or enable an otherwise
  unqualified platform.
- The implementation does not add a second transport, VAD, orchestration pipeline, or
  device abstraction. It reuses the existing injected duplex backend seam.
- The first version does not automate operating-system Sound settings. Route-change
  trials prompt the operator to change the default route and verify that the production
  route fence observed it.

## Safety invariants

1. Every new device-clock generation starts with playback-period speech admission
   closed. Manual Stop/Esc remains available in every state.
2. Full duplex requires a present, functioning native AEC processor and exactly one
   current proof path: `aec` or `acoustic-isolation`. A missing/throwing AEC processor,
   malformed metrics contract, incomplete timing, overflow, discontinuity, or stale
   route can never be rescued by the isolation path. A well-formed unavailable delay
   estimate is permitted because that is the expected no-echo state being classified.
3. AEC proof retains the existing delay-freshness, drift, and ERLE hysteresis rules.
4. Isolation proof requires sustained audible render, bounded raw-capture/reference
   continuity, low correlated render leakage across the plausible delay range, and no
   playback-only voice admission during calibration. Low microphone energy alone is
   insufficient.
5. Uncorrelated near-end speech is user speech. During isolation warmup it restarts the
   clean calibration window. After either full-duplex path opens, five contiguous
   VAD-positive 10 ms frames are held for a bounded 50 ms residual-echo decision. The
   monitor returns an explicit `pending`, `admit`, or `fence` disposition. `pending`
   never reaches STT; `admit` releases all five cleaned frames in original order so the
   first phoneme is preserved; `fence` drops them, invalidates the current proof, and
   immediately returns the route to conservative admission. A shorter VAD burst is
   dropped without revoking an otherwise valid route proof.
6. Any contradictory observation demotes immediately. Reopening requires a fresh full
   warmup; hysteresis may delay promotion but never delay demotion.
7. A route change advances the transport generation, invalidates both proof paths, and
   discards all prior calibration observations.
8. Captured and rendered PCM used by the qualifier exists only in bounded in-memory
   frame rings. Trial completion, cancellation, failure, and route change clear those
   rings, overwrite runner-owned mutable scratch buffers, and release immutable frame
   references. Reports contain aggregates only.

## Runtime design

### One shared acoustic-safety decision

`VoicePreprocessor` remains the single admission choke point. It continues to run AEC
before VAD and gains a small route-local isolation monitor fed from the already
available raw capture frames and timestamped render references. No additional audio
stream or background recognizer is introduced.

The public runtime exposes a content-free safety snapshot:

```text
safety_path: warming | aec | acoustic-isolation | half-duplex
mode: full-duplex | half-duplex
route_generation
admission_open
demotion_reason (closed enum or absent)
```

The closed demotion reasons are `ambiguous-correlation`, `capture-sequence-gap`,
`correlated-render`, `inaudible-render`, `invalid-capture-pcm`, `invalid-render-pcm`,
`missing-render-reference`, `missing-timing`, `native-processor-failure`,
`render-reference-failure`, `render-reference-gap`, `render-reference-overflow`,
`render-sequence-gap`, `route-mismatch`, `saturation`, `timing-discontinuity`, and
`unbounded-timing`. Route generations are exact non-negative integers, not booleans or
fractional numerics.

`aec` and `acoustic-isolation` imply full duplex. `warming` and `half-duplex` close
playback-period admission. Existing callers that only consume `DuplexMode` keep their
current contract; diagnostics and qualification consume the richer snapshot.

Each isolation observation also carries an optional closed near-end disposition:

```text
near_end_disposition: pending | admit | fence | absent
```

The route-level snapshot may remain open while disposition is `pending`; this means
only that the route proof is still valid, never that pending frames may be admitted.
`VoicePreprocessor` owns the corresponding bounded cleaned-frame queue and is the only
component allowed to release or discard it.

### Isolation observations

The monitor evaluates fixed 10 ms frames in bounded analysis windows. An eligible
window must have:

- render energy above a fixed qualification/runtime audibility floor;
- complete capture and render-reference sequences in the same clock generation;
- no callback status flag, ring overflow, saturation, or timing discontinuity;
- a bounded lag scan covering the route's reported input/output latency plus guard;
- normalized cross-correlation and correlated-energy leakage below conservative,
  versioned thresholds; and
- no VAD-positive capture during a known render-only calibration interval.

Promotion requires several consecutive eligible windows and minimum rendered duration,
not a single quiet frame. The thresholds are fixed safety constants, emitted verbatim
in reports, and test-injectable; they are not user settings. The implementation plan
must record their initial numeric values and the deterministic cases used to select
them. User-configurable response eagerness remains unrelated.

After promotion, the same bounded lag/correlation check runs continuously. Each stored
capture and render vector is mean-centered before window concatenation so frame-local
DC offsets cannot dilute correlated AC leakage. PCM input is exactly one 48 kHz mono
PCM16 10 ms frame (960 bytes), and render-reference consumption per observation is
strictly capped; oversized frames or excess references fail closed.

A render pause resets sequence-boundary trackers and incomplete capture/near-end
windows but does not blindly erase still-plausible render references. The bounded
history retains render-DAC provenance and prunes each reference once the current
capture-ADC timestamp places it outside the lesser of 500 ms or the reported route
latency plus guard. This preserves delayed-echo detection across short pauses without
treating stale pre-pause audio as a candidate after a long silence. Route changes and
unsafe timing still erase the history immediately.

Scheduled-reference continuity and causal echo candidacy are distinct checks. The
transport intentionally stamps the render selected in the current callback at a future
DAC time; that current reference is valid when its DAC-minus-capture-ADC offset matches
the reported delay and may establish scheduled render energy. Correlation candidates,
including prior references after a pause, must instead be non-future and within the
bounded acoustic-age range. The monitor must not apply the causal predicate to the
current scheduled-reference presence check.

A VAD-positive event is classified over exactly five contiguous frames. For each
frame-aligned lag in the bounded route-latency range, the monitor computes signed
per-frame dot products, then aggregates the five dot products and energies into one
event-level normalized correlation and least-squares gain. Playback leakage requires
the same lag, the same non-zero dot-product sign across all five frames, event-level
absolute correlation above `0.85`, and correlated leakage above `-30 dB`. This
render-dominance rule prevents short colored-speech correlations from rejecting real
barge-in while still fencing direct copied, inverted, and delayed render echo. Until the
fifth frame the result is `pending`.
Coherent leakage returns `fence`; otherwise the event returns `admit`. Missing timing,
references, continuity, or native-processing evidence returns `fence` immediately.
After an event is admitted, each subsequent contiguous VAD-positive frame returns
`admit` immediately; the next VAD-negative frame ends the event and resets the bounded
classifier state.
The route must still pass live physical qualification before rollout; this 50 ms rule
is not independently treated as proof of production suitability.

The initial `0.85` event threshold is fixed rather than user-configurable. Its
deterministic calibration corpus covers 5,000 independent events each for white noise
and AR-like colored signals at coefficients `0.8`, `0.9`, and `0.95`, the maximum
51-lag search, copied/inverted/delayed render, and render-plus-near-end mixtures. The
selected rule produced no false fences in those 20,000 independent events, no misses in
408 pure-echo lag/sign/colour cases, fenced 19,998 of 20,000 render-dominant `2:1`
mixtures, and admitted 99.61% of equal-energy `1:1` mixtures. These deterministic
results select an initial threshold; measured physical qualification remains the
rollout authority.

### State transitions

```text
route open/change
      |
      v
   warming  -- AEC proof -----------------> full duplex / aec
      |  \
      |   `-- isolation proof ------------> full duplex / acoustic-isolation
      |
      `-- timeout or unavailable ---------> half duplex

any overflow, discontinuity, stale timing, processor error, or correlated leakage
      -> fence route -> closed warming/half duplex
```

AEC and isolation warm in parallel so a built-in speaker route can qualify through AEC
while an isolated USB headset can qualify without waiting for impossible ERLE evidence.
The first valid path wins for the current snapshot; AEC may supersede isolation later.

## Live physical qualifier

### Boundary and flow

`scripts/qualify_physical_voice.py` keeps the current report builder and schema
validation as pure code. Normal mode obtains observations from a new bounded live-trial
runner; `--fixture` continues to load checked-in observation JSON and never touches
hardware.

The live runner directly constructs `DuplexAudioTransport`, the native AEC processor,
`VoicePreprocessor`, and the normal VAD behind an explicit qualification-only entry
point. This avoids the rollout-manifest circularity while still exercising the exact
runtime source. It does not enable the Console pipeline.

The runner:

1. verifies the selected default input/output route and 48 kHz mono duplex format;
2. opens the production transport with admission closed;
3. renders a checked-in, hash-verified, redistributable speech sample with bounded
   backpressure rather than filling the render ring;
4. measures AEC and isolation safety snapshots during a render-only warmup;
5. runs one fail-fast microphone/double-talk calibration, collects the repeated
   double-talk windows from one continuous speaking block, and automatically paces
   interruption cues with measured quiet recovery between them;
6. prompts for three round-trip default-route trials and requires a new generation plus
   closed admission on every change, returning to the target route after each trial;
7. completes the remainder of a 30-minute mixed render/silence soak unattended; and
8. closes the transport, verifies cleanup, zeroes retained frame buffers, builds the
   observation object, and passes it to the unchanged report-construction boundary.

The speech sample is stored with provenance, license, SHA-256, format metadata, and an
explicit grant permitting repository redistribution. It contains no user audio. A
missing or mismatched asset aborts before opening a device.

The operator speaks only during timed cues. The runner detects speech/no-speech and
stop timing; it never transcribes, writes, hashes, or reports the captured content.

### Operator timing protocol

Detector windows and human response windows are separate. Before any repeated
double-talk measurement, the runner waits up to ten seconds for actual admitted
microphone speech. If none arrives, it aborts without entering the repeated loop. If a
full-duplex safety path is already open, playback then starts while that same speech is
continuing and the runner requires an admitted overlap event; failure aborts before the
20 windows. A warming or closed path is identified explicitly as the manual-stop
fallback and is never described as calibrated full duplex.

After calibration, all 20 half-second double-talk windows come from one continuous
speaking block. Playback remains sequential and continuous, each window is cancellable,
and the runner prints detection status after every window. It then stops playback and
requires one measured second of consecutive microphone silence before the interruption
phase.

Full-duplex interruption requires one operator acknowledgement for the phase, not one
acknowledgement per trial. Each automatic `speak now` cue allows up to ten seconds for
the operator to react, prints the measured result immediately, and is followed by one
measured second of consecutive silence before the next cue. Continuing speech resets
the silence streak. Half-duplex retains an explicit manual-stop prompt per trial. The
first failed interruption sample ends the remaining trials and returns partial failing
observations with the unfinished checklist entries false, so report construction can
preserve a reviewable FAIL without wasting more operator trials.

### Route trials

The live runner reuses the transport's existing backend injection and route-generation
fence. A thin helper reads current default input/output identities and capabilities for
display and change detection. It retains raw identifiers only in memory, emits only a
report-local salted hash, and does not persist device names.

Each of the three trials switches from the target route to an alternate route and back.
At each prompt the operator changes the system default route. The runner requires:

- the old stream to close;
- a `DeviceRouteChanged` fence and generation advance;
- playback-period admission to remain closed on the new route until proof completes;
- no stale callback or PCM from the prior generation; and
- a final return to the target device before the unattended soak.

Some PortAudio hosts emit a burst of status-flagged callbacks while the duplex stream
is priming. The transport gives every new route a hard-bounded 500 ms callback pre-roll,
discards flagged callbacks inside that bound without advancing the app capture
sequence, and rebases cadence comparison after each discard. A status at callback 51
or later remains a permanent generation discontinuity and closes acoustic admission.
During a route fence, callbacks delivered while the native stream is actively stopping
are silenced and rejected; a callback arriving after teardown completes remains a
stale callback failure. Manual interruption is latched for the phase once selected so a
transient safety-path recovery cannot change the operator contract mid-prompt.

If the platform cannot observe a real change, the trial fails; acknowledging a prompt
is not evidence.

### Measured observations

All formerly typed numerical and boolean safety fields come from runner counters and
timestamps. Operator input is limited to starting/cancelling prompts and confirming
that a requested physical route change has been performed. It cannot set ERLE, leakage,
false-barge, recall, latency, overflow, cleanup, or pass/fail values.

Permissions denied, missing devices, unsupported formats, insufficient eligible
windows, saturation, any ring overflow, route mismatch, or cancellation aborts without
a passing report. A completed trial that violates a safety threshold writes a valid
`passed: false` report so the failure remains reviewable.

## Evidence schema

The physical report schema advances to version 2 and adds a closed `safety_path` enum:
`aec`, `acoustic-isolation`, or `half-duplex`.

- `aec` full duplex retains median/p10 ERLE, delay availability/refinement, double-talk
  recall, false-barge rate, and stop-latency gates.
- `acoustic-isolation` full duplex requires the native processor to remain operational,
  sustained eligible render windows, passing correlation and correlated-leakage
  distributions, zero render-only voice detections during calibration, passing
  double-talk recall, and passing interruption latency. ERLE may be unavailable and is
  not fabricated as zero or a passing value.
- `half-duplex` passes only as safe degradation: playback-period speech admission is
  closed, manual interruption works, and no unsuppressed interruption is observed.

The isolation section records only content-free aggregates: eligible/required window
counts, warmup duration, correlation distribution, correlated-leakage distribution,
render-only voice detections, demotion count/reasons, and the exact fixed thresholds.
ERLE, correlation, leakage, and stop-latency distributions retain their bounded numeric
sample arrays, rounded once before serialization; all summaries are recomputed from
those serialized values. The report also records capture/render/reference overflow
counts and route-generation trials. Raw PCM, transcript, detected words, unsalted
device identifiers, and device names remain schema-forbidden.

The builder computes `passed`; the live runner cannot supply it. The qualification
manifest accepts version-2 physical reports only after all existing source-digest,
automated-prerequisite, device-matrix, companion, and wheel checks pass.

## Testing and verification

Implementation follows test-first changes at the existing seams:

- deterministic fake echo proves the AEC path opens and isolation is not required;
- deterministic isolated capture proves the isolation path opens despite unavailable
  AEC delay/ERLE;
- frame-local DC-offset echo proves copied AC leakage cannot be hidden by aggregate
  centering;
- realistic-amplitude near-end double-talk proves five-frame uncorrelated speech is
  released in order after at most 50 ms, while copied and delayed echo at one stable lag
  is fenced;
- correlated playback leakage, ambiguity, saturation, gaps, overflow, and stale route
  callbacks prove admission never opens or demotes immediately;
- route-switch tests prove every generation starts closed and prior observations cannot
  carry over;
- live-runner tests use the existing fake duplex backend and a fake clock, verify
  backpressure, prompts, derived metrics, cancellation, cleanup, and zero persisted
  audio;
- report/schema fixtures cover passing AEC, passing isolation, safe half duplex, and
  unsafe leakage; typed metrics are rejected in normal mode;
- the real macOS USB-headset run must complete the prompted trials, three round-trip
  route trials, and 30-minute soak before its report can be considered evidence.

Any source change invalidates the current automated/soak report hashes. All automated,
duplex-soak, cancellation-soak, and physical reports must be regenerated from one clean
source digest before rollout authority can be rebuilt.

## Consequences

- Isolated headsets gain low-latency acoustic interruption without pretending that an
  unobservable echo path has measurable ERLE.
- Speaker routes keep the stronger AEC path and fail closed if cancellation cannot be
  proven.
- Physical qualification becomes longer but defensible: its metrics come from the
  production path rather than operator assertions.
- The change adds one bounded monitor and one live runner, reusing the current transport
  and report builder. No new dependency or second voice architecture is required.
