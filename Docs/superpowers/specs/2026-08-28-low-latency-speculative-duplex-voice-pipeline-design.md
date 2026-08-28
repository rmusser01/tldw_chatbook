# Low-Latency Speculative Duplex Voice Pipeline Design

**Date:** 2026-08-28

**Status:** Approved during brainstorming; pending written-spec review

**Decision:** [ADR-098](../../../backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md)

**Related designs:** [Hands-Free Conversation Loop](2026-08-02-hands-free-loop-design.md),
[Realtime Voice Engine](2026-08-04-realtime-voice-engine-design.md)

## Summary

The provider-agnostic Console hands-free pipeline currently waits for a two-second
silence gate, segment transcription, and a 1.5-second send countdown before starting
an LLM response. It can detect speech resuming, but once response generation begins it
deliberately lets generation finish silently instead of cancelling and rebuilding the
turn.

This design replaces that stop-and-transcribe cadence with rolling transcription and a
configurable speculative-response threshold whose default is 700 ms. A speculative LLM
attempt begins from the freshest transcript snapshot. If speech resumes while the
assistant is generating or speaking, or if STT materially corrects text used by the
attempt, Chatbook stops audible output, fences and cancels the attempt, extends the same
logical user turn, and starts a replacement after the next silence interval. Once the
assistant finishes speaking, later speech starts a new turn.

Reliable acoustic interruption becomes default through an app-owned full-duplex audio
transport and a small native WebRTC AEC3 component shipped for macOS, Windows, and
Linux. All hands-free TTS passes through the transport so the canceller receives the
timestamped render reference. If echo cancellation is unavailable or unhealthy,
speech admission fails closed during playback and the loop remains usable in
half-duplex mode.

Provisional attempts stay outside ordinary Console history, tools, approvals, citations,
and durable exchange capture. The winning user/assistant pair is promoted atomically.
When Console Capture is enabled, the winning attempt's sanitized in-memory exchange
capture is then persisted best-effort with explicit post-dispatch provenance; cancelled
attempts retain only content-free usage and latency counters.

## ADR check

ADR required: yes

ADR path: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md`

Reason: the feature changes cross-platform native dependencies, audio-device and clock
ownership, AEC failure policy, rolling-STT contracts, Console turn acceptance and
persistence, provider cancellation, tool-effect boundaries, and exchange-capture
semantics.

## User promise

- On a healthy warm reference path, Chatbook dispatches the LLM within 850 ms p95 of
  post-AEC end-of-speech detection, using a transcript that covers the utterance tail.
- Chatbook starts response preparation after 700 ms of silence by default; the setting
  is adjustable from 500 to 3000 ms.
- Speech detected while a provisional reply is generating or playing extends the same
  user turn. The old reply stops audibly within 150 ms p95 and is replaced using the
  accumulated transcript.
- Cancelled attempts do not appear in conversation history and cannot execute tools or
  create approvals, citations, replay records, or content-bearing diagnostics.
- Once the winning assistant audio reaches a terminal boundary, later speech starts a
  new turn.
- App-owned TTS does not trigger false barge-in when AEC is healthy. When reliable
  suppression cannot be established, Chatbook closes speech admission during playback
  instead of exposing an unsuppressed microphone.
- The separate OpenAI Realtime engine is unchanged.

## Goals

- Reduce end-of-speech to LLM dispatch without treating transcription as work that
  starts only after silence.
- Support native streaming STT and batch-only STT behind one rolling revision protocol.
- Preserve one logical user message across short pauses and any resulting speculative
  restarts.
- Start TTS from the earliest safe prose phrase, keeping synthesis and playback ordered
  and cancellable.
- Provide reliable default acoustic echo cancellation on the three supported desktop
  operating systems.
- Keep provisional work isolated from durable history and irreversible effects.
- Make latency, AEC health, fallback mode, restart storms, and discarded-attempt usage
  measurable without recording transcript or audio content.

## Non-goals

- Changing the provider-native Realtime voice engine or making it emulate this pipeline.
- Cancelling arbitrary system audio or removing background noise that is not caused by
  Chatbook's render stream.
- Persisting microphone audio, rolling STT windows, cancelled transcripts, or cancelled
  response bodies.
- Guaranteeing cooperative remote cancellation from every LLM or TTS provider. Late
  work is fenced even when it cannot be stopped.
- Executing tools speculatively or attempting to roll back external side effects.
- Replacing the complete Console controller, provider gateway, persistence service, or
  TTS adapter registry.
- Claiming uninterrupted acoustic barge-in while AEC is warming or degraded. Safe
  half-duplex fallback is part of the contract.

## Existing constraints and starting point

- `Audio/dictation_service_lazy.py` uses a two-second silence threshold and finalizes
  segment audio before bounded transcription on non-streaming paths.
- `Chat/console_voice_input.py` already exposes `VoiceSpeechResumed`, a 1.5-second
  hands-free send delay, and an opt-in acoustic-barge-in setting.
- `Chat/console_hands_free.py` owns the headless hands-free FSM. Its current barge-in
  contract silences playback but does not cancel reply generation.
- `Chat/reply_sentence_sequencer.py` assumes deltas continue after barge-in and buffers
  enough text to avoid speaking code fences or unresolved Markdown.
- `Audio/streaming_sink.py` owns app-rendered PCM through `sounddevice`, but microphone
  capture and legacy file playback do not share one timestamped duplex clock.
- `Chat/console_agent_bridge.py` streams ordinary assistant deltas directly into the
  Console store, so provisional attempts cannot reuse that sink safely.
- ADR-094 keeps microphone and realtime audio resources view-scoped while accepted
  Console turns survive navigation in `ConsoleRuntime`.
- ADR-097 requires a durable reservation before ordinary Capture On provider dispatch.
  A provisional voice attempt is intentionally pre-acceptance and receives the narrow
  capture exception defined below.
- `dictation.acoustic_barge_in` is also consumed by the separate Realtime engine. It
  cannot be repurposed for the new pipeline without violating the Realtime non-goal.
  `dictation.handsfree_send_delay_seconds` belongs to the superseded countdown rather
  than the new rolling-transcript timer.

## Core invariants

1. **AEC precedes turn detection.** Raw microphone frames never drive VAD, rolling STT,
   command routing, or acoustic barge-in while assistant audio is rendering.
2. **Rendered audio has one owner.** Every hands-free TTS provider, including legacy
   file-producing adapters, decodes into the app-owned PCM transport before playback.
3. **Transcript freshness outranks the timer.** A 700 ms timeout cannot dispatch a
   hypothesis that does not cover the detected speech tail.
4. **One logical turn, many attempt epochs.** Only the newest attempt epoch may render,
   speak, call tools, create approvals, or promote persistence.
5. **Cancellation is a fence before it is a request.** Obsoleting an attempt is atomic;
   provider cancellation and resource cleanup follow best-effort.
6. **Effects are never speculative.** Every tool call waits for the effect barrier and
   then enters the ordinary accepted-turn pipeline.
7. **Cancelled content stays ephemeral.** Conversation history, exchange trace, logs,
   citations, tool state, and replay contain no cancelled transcript or response body.
8. **AEC uncertainty fails closed.** Unhealthy full duplex becomes explicit half duplex;
   it never becomes unsuppressed acoustic interruption.
9. **Audio callbacks stay real-time safe.** They perform bounded copies and ring-buffer
   operations only; STT, reconciliation, UI, logging, and cancellation run elsewhere.
10. **The winning prompt is exact.** The persisted user text is the immutable transcript
    snapshot used by the winning LLM attempt, not a later independently revised string.

## Architecture

```text
                         timestamped render frames
TTS PCM ──► DuplexAudioTransport ───────────────────────────┐
                  │                                        │
                  └─► speakers                             ▼
microphone ──► timestamped capture frames ──► VoicePreprocessor
                                                AEC3 → health → VAD
                                                           │
                                                           ▼
                                                  TranscriptEngine
                                             stable prefix + revisable tail
                                                           │
                                                           ▼
                                             SpeculativeTurnCoordinator
                                           turn id + attempt epoch + clocks
                                             │                       │
                                             ▼                       ▼
                                      AttemptOutputBuffer       EffectBarrier
                                             │
                                             ▼
                                      PhraseSpeechSequencer
                                             │
                                             └──────────► DuplexAudioTransport
```

### Ownership boundaries

| Unit | Owns | Depends on | Explicitly does not own |
| --- | --- | --- | --- |
| `DuplexAudioTransport` | Input/output devices, hardware formats, monotonic frame timestamps, bounded render/capture rings, device latency, device-change events | PortAudio/sounddevice-style backend | AEC policy, VAD, STT, transcript text, UI |
| `VoicePreprocessor` | Resampling into the processing domain, AEC3, render/capture delay estimation, residual-echo health, post-AEC VAD, admission gate | Native AEC wrapper and transport frames | Devices, provider calls, turn state |
| `TranscriptEngine` | Stable text, revisable tail, audio coverage, backend mode, window reconciliation | Clean admitted frames and STT adapters | Silence timers, LLM attempts, persistence |
| `SpeculativeTurnCoordinator` | Logical voice-turn identity, silence clocks, attempt epochs, restart governor, effect barrier, final promotion decision | Transcript revisions, VAD/command events, generation gateway, injected scheduler | Audio DSP, provider implementation, widgets |
| `AttemptOutputBuffer` | One attempt's request snapshot, response deltas, sanitized capture envelope, usage, citations pending promotion | Existing provider-neutral request and response contracts | Durable writes, tools, approvals |
| `PhraseSpeechSequencer` | Safe prose chunking, one-at-a-time synthesis, ordered cancellation | Current attempt deltas and TTS entry | Attempt selection, device ownership, persistence |
| Console wiring | Status projection, settings, manual interrupts, view-scoped teardown | Existing `ConsoleHandsFreeController` and UI services | Domain clocks or state transitions |

The components are protocols first. Native AEC, STT providers, and audio-device backends
remain replaceable without changing coordinator policy.

## Duplex audio and AEC contract

The transport uses one full-duplex hardware session where the backend permits it. It
normalizes render and capture into timestamped ten-millisecond frames and maintains a
single monotonic clock domain. The processing domain prefers 48 kHz; device-native
rates and TTS-source rates are converted at the transport/preprocessor boundary.
Hardware output latency, capture latency, buffer occupancy, and drift update the delay
estimate passed to AEC3.

Render samples are fed to the reverse-stream analyzer in the same order they are
scheduled for the speaker. Capture samples and the corresponding delay estimate are
then processed through AEC3. Post-AEC frames alone reach residual-echo health, VAD, and
STT. The design uses a narrow native interface rather than exposing WebRTC types to the
Python application.

AEC health states are `warming`, `healthy`, and `degraded`. Health uses residual echo,
delay confidence, underruns/overruns, discontinuities, and bounded hysteresis. On first
playback or after a device reset, the admission gate remains closed while AEC converges.
When health becomes stable it opens. A degraded transition closes admission immediately
but keeps hardware capture flowing through AEC so recovery remains possible. Manual or
keyboard interruption is always available.

Legacy file-producing TTS adapters are decoded and streamed through the same PCM path.
External players are not valid hands-free playback owners because they cannot supply
the exact render reference or terminal render timestamp.

## Rolling transcription contract

Every adapter produces revisions with:

- logical voice-turn ID and monotonically increasing revision ID;
- stable text plus a revisable tail;
- the monotonic audio timestamp covered through;
- backend mode (`live` or `rolling-window`);
- finalization and failure state; and
- content-free timing/usage metadata.

Native streaming adapters translate provider events into this protocol. Batch-only
adapters schedule bounded overlapping audio windows while speech is active. Only one
batch job may run at a time. A newer desired window supersedes an obsolete queued
window, preventing transcription backlog. Reconciliation aligns normalized tokens and
timestamps, advances an immutable stable prefix, and replaces only the mutable tail.
Punctuation, case, and whitespace changes are non-material for generation restart;
word insertions, deletions, or substitutions in the active attempt snapshot are
material.

Batch fallback may process overlapping seconds more than once. Settings identify this
mode and disclose that a remote batch provider may receive overlapping audio. The
runtime records processed and duplicated duration without audio or text content.

At a silence deadline, the coordinator requires the latest revision's coverage to
reach the post-AEC speech-end timestamp within the adapter's declared frame tolerance.
If it does not, the status remains `transcribing` and dispatch waits. Slow fallback is
reported honestly instead of sending a truncated utterance.

## Turn and attempt lifecycle

A logical turn begins at the first admitted speech frame. During speech, rolling STT
updates one provisional user row. Each admitted frame resets the response-eagerness
timer. When the timer expires and a fresh nonempty transcript exists, spoken-command
classification runs first. A stable command follows the existing command path and
never reaches the LLM.

For ordinary text, the coordinator snapshots the transcript and starts attempt epoch 1
in an attempt-local output buffer. Response deltas never enter the ordinary Console
store. The phrase sequencer emits punctuation-terminated prose first and may use a
bounded word/time fallback only when Markdown/code state is resolved. It keeps exactly
one synthesized phrase active and sends every PCM frame through the duplex transport.

New admitted speech while generation or playback is active performs this order:

1. Atomically advance the attempt epoch and reject all old callbacks.
2. Close the old phrase input, abort current playback, and discard queued phrases.
3. Request LLM and TTS cancellation and bounded cleanup.
4. Remove the old assistant preview while retaining the logical user transcript.
5. Continue rolling STT with the newly admitted audio.
6. Dispatch from a new immutable snapshot after the next silence interval.

A material late STT correction follows the same fence-and-restart path. Corrections
arriving in one short burst are coalesced before one replacement request, but the old
attempt remains fenced immediately.

The first three attempts in any rolling ten-second window use the configured threshold.
Additional attempts temporarily use 1.5 seconds until an attempt completes or the turn
remains continuously quiet. At most two obsolete provider attempts may clean up
concurrently. Reaching that bound pauses new dispatch until at least one obsolete
attempt exits. If cooperative cleanup exceeds two seconds, the coordinator enters
`serialized_conservative` for the remainder of the logical turn: it waits for two
seconds of stable silence and zero obsolete attempts, dispatches exactly one active
attempt, and still fences that attempt immediately if speech resumes. It never overlaps
a replacement with cleanup in this state.

Every attempt owns a force-closable provider transport. Five seconds after the original
cancellation request, the coordinator force-closes that transport. If the task has not
exited after a further 500 ms, it is detached into a session-scoped orphan reaper behind
its irreversible epoch fence. The coordinator then terminates the provisional logical
turn as a recoverable failure, preserves the latest transcript as an editable voice
draft, and reports that provider cleanup is stuck. While that orphan exists, the
session admits no further speculative or ordinary provider dispatch from hands-free
voice; the user may leave hands-free, change/rebuild the provider session, or retry the
draft after the orphan exits. The reaper retains no transcript or response body and an
orphan cannot publish callbacks, receipts, capture rows, or audio. Because dispatch is
blocked before another attempt starts, at most one detached orphan exists per session.
Successful turn promotion, explicit turn cancellation, this terminal cleanup failure,
hands-free exit, or session teardown ends the current logical turn; the session-level
orphan quarantine remains until the orphan exits or the provider session is rebuilt.

The successful boundary is the estimated time the winning attempt's final audible
sample reaches the output device. Speech whose capture timestamp begins no later than
that boundary extends the existing turn; later speech begins a new turn. If no audio is
produced, the terminal generation/TTS outcome supplies the equivalent boundary. TTS
failure waits for text generation to finish, commits the completed textual response if
no newer speech exists, reports the speech failure, and reopens admission.

All transcript, render, and control events enter one serialized coordinator mailbox.
The duplex input callback stamps each captured frame with a strictly increasing sequence
and a timestamp from the same monotonic clock as the output render timeline, before AEC
or VAD work begins. After playback reports terminal render boundary `R`, promotion first
requests `drain_capture_through(R)`. The full-duplex engine must observe the first
ordered input callback or explicit device-clock watermark later than `R`, drain AEC and
VAD through the greatest captured sequence at or before `R`, and report every detected
speech span from that range. A detected span whose start is at or before `R` is admitted
to the existing logical turn and fences the attempt.

Only after that capture/VAD acknowledgement reports no qualifying speech does the
coordinator ask the transcript engine to seal through the last downstream-acknowledged
admitted sequence and drain all material revisions derived from it. A material
correction derived from audio at or before `R` wins the race, fences the attempt, and
restarts it even if its callback arrived after the playback terminal callback.
Promotion closes both causal watermarks; later callbacks cannot mutate the committed
turn. In intentional half duplex there are no eligible capture frames during playback,
so the playback terminal acknowledgement closes that gated interval and manual
barge-in is the only same-turn interruption path.

The combined capture, VAD, and transcript seal has a 500 ms deadline after `R`. Timeout,
device reset, a skipped sequence, or a failed downstream acknowledgement fails closed:
the pair is not promoted, the latest transcript becomes an editable voice draft, status
reports capture synchronization failure, and speculative dispatch remains suspended
until the duplex engine is rebuilt healthy. This may sacrifice an already-heard reply,
but it never misclassifies pre-boundary speech as a new turn or commits a pair whose
causal input boundary is unknown.

A manual **barge-in** action while generating or speaking follows the same
fence-and-continue path as admitted speech and reopens listening for the same logical
turn, including in half-duplex mode. Explicit Stop, Esc, mic-toggle exit, session close,
and hands-free exit instead cancel and discard the whole provisional turn. Merely
silencing playback without fencing generation is not a valid action in the new
pipeline.

## Tool-effect barrier

Speculative attempts may produce a tool request but cannot execute it or create an
approval. On the first complete tool request, Chatbook freezes provisional speech and
holds the attempt until total stable silence since the last admitted speech reaches two
seconds. Speech during that interval fences and discards the attempt normally.

When the barrier expires, Chatbook discards the speculative provider attempt, commits
the exact user transcript, and re-dispatches through the ordinary accepted Console
agent pipeline with its normal tool schemas, Capture On policy, approvals, persistence,
and ADR-094 runtime custody. Any later speech is a new turn or an explicit interruption;
it cannot rewrite an already authorized external effect. The speculative preamble is
not reused as authoritative agent output because it was generated without committed
tool context.

## Acceptance, persistence, navigation, and capture

A provisional voice attempt is not an accepted Console turn under ADR-094. Its
microphone, audio, coordinator, transcript, and attempt buffers are view-scoped.
Navigating away, leaving hands-free mode, session close, or app shutdown fences and
discards them without a terminal receipt. This matches the existing rule that audio
resources do not survive screen navigation and the user's requirement that cancelled
attempts leave no conversation artifacts.

For a no-tool winning attempt, the exact transcript snapshot and assistant response are
promoted through the existing durable-turn terminalization contract. One ChaChaNotes
transaction writes the user/assistant pair, mints ADR-094's stable terminal receipt,
and writes the exact `console_unseen:<receipt-id>` local mark. Promotion registers an
already-complete accepted voice turn and terminalizes it atomically; it does not create
a second runtime task for provider work that already ended. A mounted view clears only
that exact receipt after synchronizing the committed pair. If navigation wins the race,
the mark survives for the next view. A mounted view may project provisional rows before
promotion, but ordinary history observes only the committed pair. Failure preserves the
user transcript as an editable voice draft rather than persisting a partial pair or
receipt.

Ordinary ADR-097 Capture On reservation cannot run before a provisional provider call
without persisting cancelled prompt content. The narrow amendment is:

- the provider gateway builds the same sanitized semantic capture envelope in memory;
- cancelled attempts destroy that envelope;
- content-free attempt count, timing, backend mode, and billed-usage counters may remain
  in local diagnostics;
- after the winning conversation pair commits, its capture is settled best-effort as a
  `provisional_voice_promoted` exchange linked to the committed turn;
- the exchange declares that capture was promoted after dispatch and makes no
  crash-durable pre-dispatch-reservation claim; and
- a crash before promotion leaves neither provisional conversation nor exchange trace.

If post-promotion trace settlement fails, it does not roll back the committed
conversation. The existing trace durability warning/retry policy applies where it can
honestly retry from the winning in-memory envelope; shutdown does not persist that
envelope merely to enable later repair. Tool-barrier turns discard their speculative
envelope and use the ordinary ADR-097 captured pipeline on re-dispatch.

Temporary conversations retain their existing ADR-097 restriction. Speculative voice
remains usable, but the status explicitly says exchange capture is unavailable for the
temporary chat. Winning message rows and the winning envelope remain process-local;
neither pre-dispatch reservation nor post-dispatch capture promotion is attempted. A
later Save persists the conversation messages under ordinary temporary-chat promotion
rules but does not retroactively invent capture for an earlier provider call. Tool
barrier expiry uses the existing Save & Send requirement before ordinary captured agent
dispatch.

## Settings and UI

The canonical `Settings -> Speech & TTS` surface owns the controls. Deprecated settings
surfaces receive no new fields.

`dictation.response_eagerness_ms` applies only to speculative hands-free dispatch:

- valid range: 500 through 3000 ms;
- default and Fast preset: 700 ms;
- Balanced preset: 1200 ms;
- Deliberate preset: 2000 ms; and
- invalid values log a content-free warning and fall back to 700 ms.

Values below 700 ms are labeled more likely to restart during mid-thought pauses. The
existing dictation segment-finalization and non-hands-free settings remain independent.

Legacy setting ownership is explicit:

- `dictation.acoustic_barge_in` remains the unchanged compatibility key for the
  provider-native Realtime engine. The qualified speculative pipeline does not read it;
  acoustic interruption there follows AEC health and is default-on.
- `dictation.handsfree_send_delay_seconds` remains readable only by the legacy
  pre-speculative pipeline during the internal rollout gate. Once the new pipeline is
  qualified, it is retired from active Settings and ignored by the speculative path.
  Its value is not migrated into response eagerness because countdown-after-final-STT
  and silence-to-dispatch are different semantics.
- `dictation.pipeline_aec_enabled`, default true, is the troubleshooting-only AEC
  switch. False forces half duplex and does not affect Realtime.

The app preserves old config keys so older releases can still read them and shows one
bounded migration note when a legacy hands-free delay or acoustic setting no longer
controls the qualified pipeline. No startup write mutates the user's config merely to
acknowledge the new ownership.

AEC is default-on. Its off switch lives under troubleshooting and always forces safe
half duplex. The Console status projection uses user-readable states: `listening`,
`transcribing`, `responding`, `speaking`, `updating response`, `AEC warming`, and
`half-duplex - echo cancellation unavailable`. Detailed health codes remain in
diagnostics. Accessibility announcements fire on meaningful state transitions and are
throttled; transcript revisions do not announce individually.

The provisional user and assistant each reuse one temporary row across attempts. This
avoids scroll jumps. When an attempt is fenced, its assistant body disappears and the
status becomes `updating response`. No spoken acknowledgement or earcon is added to the
hot audio path.

## Failure handling

- **AEC initialization/health:** keep device capture flowing for recovery, close speech
  admission during playback, and display half-duplex status.
- **Audio device change:** fence playback, retain the user transcript, rebuild the
  device/AEC clock domain, and regenerate after the normal silence threshold.
- **Transport overrun/underrun:** mark timing health degraded, fail admission closed,
  and keep callback work bounded.
- **STT lag:** wait for fresh coverage and report `transcribing`.
- **STT failure:** try the adapter's declared fallback once; if that fails, retain an
  editable voice draft and stop automatic dispatch for the turn.
- **LLM failure:** retain the logical user draft, reopen listening, and permit additional
  speech or manual retry.
- **Uncooperative cancellation:** keep the obsolete epoch fenced, bound detached cleanup,
  and apply the restart governor/backpressure rules.
- **TTS phrase failure:** stop further speech for the attempt, allow text generation to
  complete, and commit text only if no newer speech exists.
- **Persistence failure:** retain the winning pair in the existing visible durability
  failure/retry surface; never pretend a partial pair committed.
- **Trace promotion failure:** do not roll back conversation success; report the existing
  local capture warning while the in-memory retry source still exists.
- **Shutdown/unmount:** advance all fences, stop audio, discard provisional buffers,
  release devices, and create no provisional approvals or durable content.

## Privacy, security, and cost

- Raw and processed audio live only in bounded memory rings and rolling windows. They
  are never written to disk by this feature.
- Cancelled transcript and response text never enter logs, usage rows, traces,
  notifications, sync, export, replay, or exception copy.
- Content-free diagnostics include timestamps/durations, backend mode, attempt counts,
  restart-governor activation, AEC state transitions, provider cancellation outcome,
  and coded failures.
- Cancelled provider work may still be billed. Usage accounting distinguishes winning
  and discarded attempts and the Settings copy explains that aggressive response
  eagerness can increase STT, LLM, and TTS usage.
- Remote rolling-window STT may receive overlapping audio more than once. The selected
  backend mode and duplicated-duration counter make that behavior visible.
- The native dependency ships hashes, build provenance, an SBOM, applicable platform
  signing, WebRTC license/patent notices, and a documented update policy.

## Performance targets and measurement

The warm reference path targets:

- LLM request handoff within 850 ms p95 of post-AEC VAD speech end;
- audible output stop within 150 ms p95 of admitted resumed speech;
- replacement request handoff within 850 ms p95 of the new speech end while the restart
  governor is not active; and
- first audible response within 1.5 seconds median and 2.5 seconds p95 on the controlled
  reference provider path.

Instrumentation records post-AEC speech end, transcript coverage, request handoff,
first accepted PCM, and estimated first/final hardware-render timestamps. Controlled
local/fake providers prove application budgets. Live cloud-provider runs are reported
separately so network variance is not mislabeled as application overhead.

## Verification strategy

### Signal processing

Licensed/synthetic fixtures and captured room-impulse responses cover far-end-only
echo, near-end speech, double-talk, changing delay, clock drift, clipping, silence,
underruns, and device resets.

- Far-end-only ERLE after one second of convergence: median at least 20 dB and 10th
  percentile at least 10 dB.
- Echo-only false barge-in rate: no more than one per 30 minutes.
- Double-talk speech-start recall: at least 95 percent at the defined near-end/echo
  ratios.
- Healthy-path interruption and audible stop: no more than 150 ms p95.
- Poor suppression must cause fail-closed admission rather than false healthy state.

### Domain and integration tests

- Transcript tests cover native revisions, stable-prefix advance, overlap alignment,
  duplicate words, late corrections, punctuation-only changes, Unicode languages,
  stale jobs, coverage timestamps, and backend failure.
- Coordinator tests use an injected clock for silence deadlines, freshness gating,
  every cancellation race, correction coalescing, attempt epochs, restart governor,
  cleanup cap, force-close/detach/quarantine/recovery, serialized-conservative
  entry/dispatch/reset, tool barrier, manual barge-in versus explicit exit, terminal
  outcomes, and timestamp turn boundaries.
- Phrase tests cover punctuation, bounded fallback, ordering, cancellation, incomplete
  Markdown, links, fenced code, abbreviations, and synthesis failure.
- Persistence tests prove cancelled attempts create no content-bearing durable owner,
  winning pairs commit atomically, capture promotion is labeled and best-effort, and
  tool turns use ordinary pre-dispatch capture after the barrier. They also prove the
  ADR-094 receipt and exact unseen mark share the winning-pair transaction, mounted and
  navigation-racing acknowledgement behavior, and temporary-chat no-capture behavior.
- Integration uses a deterministic fake duplex transport with streaming and batch STT,
  cancellable and cancellation-resistant LLMs, streaming and file-producing TTS,
  ordered capture/render clocks, delayed VAD/STT delivery, seal timeout, skipped
  sequences, health transitions, and device changes.
- Regression tests prove ordinary dictation, manual message speech, and the Realtime
  engine remain unchanged.

### Packaging and live gates

- Build/import-test native wheels for every supported Python/platform architecture,
  with hashes, provenance, SBOM, license inventory, and applicable signing.
- Run a 30-minute echo-only soak and a mixed-speech/device-switch soak, checking callback
  overruns, bounded queues, detached cleanup, and memory stability.
- Run real-room speaker/microphone tests on macOS, Windows, and Linux using built-in
  audio, one USB route, and Bluetooth. Bluetooth may pass through explicit safe
  half-duplex degradation; it may not pass through unsafe full duplex.
- Use targeted test and lint suites during implementation. A repository-wide sweep
  requires the explicit opt-in mandated by repository instructions.

## Rollout

Implementation is dependency ordered:

1. Native AEC wrapper, build provenance, deterministic DSP corpus, and package loading.
2. Duplex transport and unified hands-free PCM playback.
3. Rolling transcript protocol plus native and batch adapters.
4. Speculative coordinator, attempt buffers, phrase sequencing, and cancellation fences.
5. Effect barrier, winning-pair persistence, and promoted-capture amendment.
6. Canonical Settings/Console presentation, diagnostics, and documentation.
7. Cross-platform live qualification and default enablement.

The capability remains internally gated until every supported platform passes its
package, DSP, integration, and live-device gates. After qualification, AEC and
speculative hands-free turns are default-on. A local kill switch can force half duplex
without disabling voice interaction. Platform-specific regressions may disable full
duplex for the affected capability signature only.

Each numbered slice becomes an atomic Backlog task or a small dependency-ordered task
family during implementation planning. The tasks share this architecture but retain
independent acceptance criteria and targeted verification.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Assistant audio is mistaken for user speech | Timestamped render reference, AEC-before-VAD, residual health, and fail-closed admission |
| Batch STT falls behind | One in-flight window, supersession, coverage gating, honest status |
| STT corrections cause request thrash | Material-change filter, short coalescing, three-attempt governor, cleanup cap |
| Provider ignores cancellation | Epoch fence before cancel request; bounded detached cleanup |
| Speculative tools produce irreversible effects | Two-second effect barrier and ordinary agent re-dispatch |
| Cancelled text leaks through Console capture | Memory-only attempt sink; winning-only promoted trace |
| Winning capture conflicts with ADR-097 reservation | Explicit `provisional_voice_promoted` post-dispatch provenance and ADR-098 amendment |
| Legacy settings retain old behavior | New pipeline-specific AEC/eagerness keys; old acoustic key remains Realtime-owned and old delay remains legacy-only |
| Temporary chat has no durable capture lineage | Explicit capture-unavailable status; no retroactive trace invention on Save |
| Promotion races queued speech or a late STT correction | Serialized mailbox plus shared-clock capture/AEC/VAD/STT watermarks through the render boundary |
| Winning promotion bypasses ADR-094 attention | Existing terminalization transaction mints the exact receipt and unseen mark |
| Uncooperative cancellation never exits | Force-close deadline, terminal draft failure, one fenced orphan, and session dispatch quarantine |
| Pre-boundary speech remains queued behind playback completion | Shared-clock capture sequence watermark plus AEC/VAD/STT drain before promotion |
| Audio device switches invalidate AEC timing | Fence playback, rebuild one clock domain, keep transcript, regenerate |
| Aggressive mode increases usage | Usage split, duplicated-STT duration, restart governor, Settings disclosure |
| Native dependency becomes unmaintained | Narrow ABI, reproducible wheels, SBOM, license/update policy, capability fallback |
| Navigation outlives view-owned audio | Provisional voice work is pre-acceptance and cancels on detach under ADR-094 |

## Documentation changes

- Update the Console voice/hands-free guide with speculative turn behavior, interruption
  semantics, half-duplex fallback, tool barrier, and increased-usage disclosure.
- Update Speech & TTS Settings documentation with response eagerness and diagnostics.
- Document native package installation/troubleshooting and supported platform wheels.
- Document the winning-only exchange-capture exception and its post-dispatch label.
- Record live qualification evidence per platform without committing user-recorded raw
  microphone material.

## Links

- [ADR-098: low-latency speculative duplex voice pipeline](../../../backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md)
- [Hands-Free Conversation Loop design](2026-08-02-hands-free-loop-design.md)
- [Realtime Voice Engine design](2026-08-04-realtime-voice-engine-design.md)
- [ADR-023: TTS adapter registry and runtime boundary](../../../backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md)
- [ADR-039: Speech and TTS settings ownership](../../../backlog/decisions/039-global-and-studio-tts-settings-ownership.md)
- [ADR-094: Console turn lifetime and navigation](../../../backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md)
- [ADR-097: reference-backed semantic trace ledger](../../../backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md)
