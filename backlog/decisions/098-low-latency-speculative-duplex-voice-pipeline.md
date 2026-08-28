# ADR-098: Use provisional turns over one app-owned AEC duplex pipeline

Status: Accepted during design; pending written-spec review

Date: 2026-08-28

Related Task: Implementation tasks will be created from the approved design during planning.

Related Spec: [Low-Latency Speculative Duplex Voice Pipeline](../../Docs/superpowers/specs/2026-08-28-low-latency-speculative-duplex-voice-pipeline-design.md)

Amends: [ADR-094](094-console-turn-lifetime-and-navigation-boundary.md) by defining
speculative voice attempts as pre-acceptance, view-scoped work, and
[ADR-097](097-console-reference-backed-semantic-trace-ledger.md) by allowing winning-only
post-dispatch capture promotion for those attempts.

## Context

The Console's provider-agnostic hands-free loop currently finalizes speech after a
two-second silence gate, runs segment transcription, waits a 1.5-second countdown, and
then sends through the ordinary Console path. `VoiceSpeechResumed` can cancel the
countdown, but after response generation starts the existing policy stops speech only;
the LLM finishes silently into the transcript.

Reducing the silence gate alone would dispatch incomplete text because non-streaming STT
currently begins material work only after segment finalization. Keeping the microphone
open during assistant speech without echo cancellation would also let the recognizer
transcribe Chatbook's own TTS. The existing opt-in acoustic mode therefore recommends
headphones and cannot become the default behavior requested here.

Reliable low-latency interruption requires one render/capture clock, a timestamped
render reference, AEC before VAD/STT, rolling transcript revisions, cancellable attempt
epochs, and isolation from durable history and external effects. This boundary must work
on macOS, Windows, and Linux and must degrade safely when a device route cannot sustain
full duplex.

Two existing decisions constrain persistence and lifetime. ADR-094 moves accepted
Console turns above screen navigation but deliberately keeps microphone and realtime
audio resources view-scoped. ADR-097 requires durable pre-dispatch reservation for
ordinary Capture On calls. Persisting every provisional prompt would violate the
requirement that cancelled voice attempts leave no content-bearing artifacts, while
silently forcing Capture Off would make the winning voice turn ignore an explicit user
preference.

## Decision

1. **Use one app-owned full-duplex hands-free audio path.** Microphone capture and every
   hands-free TTS provider share a timestamped duplex transport. Legacy file-producing
   TTS is decoded into that PCM path rather than played by an external process. The
   transport owns device formats, hardware latency, frame timestamps, bounded rings,
   and device changes; it does not own transcript or turn policy.

2. **Ship WebRTC AEC3 behind a narrow native interface.** The project builds and
   distributes a small native component for supported macOS, Windows, and Linux Python
   architectures. Render and capture are processed in ten-millisecond frames near the
   hardware boundary. WebRTC types do not escape into application code. The package
   ships hashes, provenance, SBOM, applicable signing, license/patent notices, and an
   update policy.

3. **Put AEC and residual health before speech admission.** Post-AEC frames alone may
   reach VAD, STT, command detection, or acoustic barge-in during render. Health states
   are warming, healthy, and degraded with hysteresis. Warming or degraded output keeps
   hardware capture running for recovery but closes speech admission during playback.
   AEC disabled or unavailable means explicit half duplex, never an unsuppressed open
   microphone.

4. **Replace silence-started STT with one rolling revision protocol.** Native streaming
   adapters publish provider revisions. Batch-only adapters emulate rolling behavior
   through one in-flight bounded overlapping window and supersede obsolete queued work.
   Every revision carries stable text, revisable tail, audio coverage, backend mode, and
   content-free timing metadata. LLM dispatch waits for coverage of the detected speech
   tail even when the response-eagerness timer has expired.

5. **Treat short pauses as speculative attempt boundaries within one logical user
   turn.** The default threshold is 700 ms and is configurable from 500 to 3000 ms in
   canonical Speech & TTS Settings. The coordinator snapshots the freshest transcript
   and starts an attempt-local generation. New speech or a material STT correction
   atomically advances the attempt epoch before requesting provider cancellation,
   stopping audio, discarding provisional response state, and scheduling a replacement.

6. **Bound speculation.** The first three attempts within ten seconds use the configured
   threshold; later attempts temporarily use 1.5 seconds. No more than two obsolete
   provider attempts may clean up concurrently. Reaching the cap pauses dispatch. If
   cleanup exceeds two seconds, the remainder of that logical turn enters a serialized
   conservative state: wait for two seconds of stable silence and zero obsolete
   attempts, allow only one active request, and still fence it immediately on resumed
   speech. At five seconds after cancellation, force-close the attempt-owned transport;
   500 ms later, a still-running task is detached behind its epoch fence and the logical
   turn terminates as a recoverable draft failure. One such orphan quarantines all later
   hands-free provider dispatch in that session until it exits or the provider session
   is rebuilt. It cannot publish state, audio, receipts, or captured content. These
   rules give the turn a terminal outcome, cap orphan count at one, and limit cost and
   uncooperative work without weakening immediate audio cancellation.

7. **Speak only current-attempt safe prose.** The phrase sequencer releases punctuation-
   terminated prose first and uses a bounded word/time fallback only after resolving
   Markdown/code ambiguity. It owns one sequential synthesis at a time. All phrase PCM
   re-enters the duplex transport and every callback is attempt-fenced.

8. **Keep tools behind a two-second effect barrier.** A speculative tool request creates
   no approval and executes nothing. If stable silence since the last admitted speech
   reaches two seconds, Chatbook discards the speculative request, commits the exact
   user transcript, and re-dispatches through the ordinary accepted agent pipeline.
   That pipeline owns tools, approvals, Capture On reservation, persistence, and
   ADR-094 runtime custody. Speech after effect commitment is a new turn or explicit
   interruption; an external effect is never rewritten into a prior turn.

9. **Keep no-tool attempts pre-acceptance and view-scoped.** They do not enter
   `ConsoleRuntime` accepted-turn custody, the ordinary Console store, or durable
   conversation history while provisional. Navigation away, hands-free exit, session
   close, or shutdown fences and discards them without terminal receipts. This preserves
   ADR-094's view-scoped microphone/audio rule. A winning transcript/assistant pair is
   promoted only after generation, speech, and transcript sealing reach their terminal
   boundary. Promotion uses the existing durable-turn terminalization transaction to
   write the pair, mint ADR-094's stable terminal receipt, and write the exact local
   unseen mark. It registers and terminalizes an already-complete accepted voice turn
   atomically rather than creating a second provider task.

10. **Use winning-only post-dispatch exchange-capture promotion.** The ordinary ADR-097
    durable reservation is not written before a provisional provider request. The
    gateway instead builds its sanitized semantic capture envelope in bounded memory.
    Cancelled envelopes are destroyed; only content-free usage, timing, backend mode,
    and attempt counts may remain. After the winning pair commits, its envelope may be
    settled best-effort as `provisional_voice_promoted`, explicitly declaring that it
    lacks crash-durable pre-dispatch reservation. A crash before promotion leaves no
    conversation or exchange trace. Capture settlement failure never rolls back the
    conversation. Tool-barrier re-dispatch uses unchanged ADR-097 semantics.

    Temporary chats remain unable to create durable capture lineage. Speculative voice
    may run with explicit capture-unavailable status; its winning messages and envelope
    remain process-local. Later Save may persist the messages but never retroactively
    invents the earlier exchange trace. Tool-barrier dispatch retains the existing Save
    & Send prerequisite.

11. **Use timestamped terminal boundaries.** Speech starting no later than the estimated
    final hardware-rendered sample extends the same logical turn; later speech starts a
    new one. Audible stop targets 150 ms p95. If speech output fails, text may commit only
    after generation finishes without a newer speech/revision event.

    Transcript, render, and control events are serialized. Every input frame is stamped
    before DSP with an ordered sequence and the output render clock's monotonic time.
    Before promotion, a full-duplex engine must advance its input watermark beyond the
    terminal render boundary and drain AEC/VAD through every earlier sequence. Any
    qualifying speech extends the turn. Only then may STT seal every revision derived
    from downstream-acknowledged admitted audio. The combined drain has a 500 ms
    deadline; timeout, sequence gaps, device reset, or failed acknowledgement prevents
    promotion, preserves an editable draft, and suspends speculation until the duplex
    engine is rebuilt healthy. Intentional half duplex has no eligible playback-period
    capture frames, so render completion closes that gated interval and manual barge-in
    is its only same-turn interruption path. Explicit Stop/Esc/mic-toggle/hands-free
    exit discards the provisional turn.

12. **Separate new pipeline settings from legacy and Realtime compatibility keys.**
    `dictation.response_eagerness_ms` belongs only to the speculative pipeline.
    `dictation.pipeline_aec_enabled`, default true, is its troubleshooting-only AEC
    switch and false forces half duplex. `dictation.acoustic_barge_in` remains unchanged
    for the Realtime engine; the speculative pipeline does not read it.
    `dictation.handsfree_send_delay_seconds` remains legacy-pipeline-only during rollout
    and is ignored after qualification rather than being mistranslated into eagerness.
    Existing keys are preserved for older releases and no startup migration writes the
    config.

13. **Expose honest capability and cost state.** Settings distinguish native live STT
    from rolling-window emulation, disclose overlapping remote batch processing and
    increased speculative usage, and offer response eagerness plus troubleshooting-only
    AEC disable. UI status distinguishes listening, transcribing, responding, speaking,
    response update, AEC warming, and half-duplex degradation. Logs and diagnostics
    contain no transcript or audio body.

14. **Qualify every platform before default enablement.** Native package, DSP corpus,
    deterministic integration, latency, soak, and physical speaker/microphone gates must
    pass on macOS, Windows, and Linux. Bluetooth may pass through explicit safe
    half-duplex degradation but not unsafe full duplex. After qualification, AEC and
    speculative pipeline turns become default-on with a local half-duplex kill switch.

## Alternatives considered

### Lower the existing silence threshold only

Rejected. Batch transcription would still start after the threshold and LLM dispatch
would use no rolling hypothesis, missing the requested latency and freshness contract.

### Keep acoustic interruption opt-in and recommend headphones

Rejected. It does not provide reliable default speaker use and preserves the
self-transcription failure mode.

### Use platform-native AEC independently on each OS

Rejected as the primary engine. Apple voice processing, Windows communication effects,
and PipeWire echo cancellation have different availability and health semantics. Three
policy implementations would make behavior and testing inconsistent. Platform APIs may
assist device integration behind the common boundary.

### Implement an adaptive filter in Python

Rejected. Reliable nonlinear echo cancellation, delay/drift handling, residual
suppression, and real-time callback behavior require a maintained native DSP engine.

### Put audio/AEC/STT in a sidecar process

Rejected for the first implementation. It improves crash isolation but adds IPC,
supervision, packaging, and lifecycle duplication before the in-process bounded
interfaces have proved insufficient.

### Let cancelled generations finish silently

Rejected. It cannot rebuild the response from newly appended speech and wastes latency
while retaining an answer to an obsolete prompt.

### Execute read-only tools speculatively

Rejected. Tool side-effect classification is not universally trustworthy, remote reads
may cost money or disclose incomplete speech, and duplicate execution would remain
observable. All tools share one effect barrier.

### Persist every provisional exchange under Capture On

Rejected. It would retain cancelled prompt/response content and contradict the
ephemeral-attempt promise.

### Force Capture Off for all speculative voice turns

Rejected. Safe, but it silently defeats a user's Capture On preference even for the
winning response. Winning-only post-dispatch promotion preserves useful diagnostics
with an honest durability label.

## Consequences

- Hands-free audio playback becomes an app-owned PCM responsibility; external legacy
  players are no longer valid for that path.
- A new native build/release surface and cross-platform hardware qualification matrix
  are required.
- Rolling-window batch STT can increase provider audio processing and must expose that
  cost honestly.
- Aggressive pauses can generate billable discarded attempts; the restart governor and
  split usage accounting make the trade-off bounded and visible.
- Provisional no-tool turns cancel on Console navigation because they are not accepted
  ADR-094 runtime turns. Tool turns become navigation-surviving only after the effect
  barrier commits and re-dispatches them normally.
- ADR-097 gains one explicitly weaker crash-provenance class for a promoted winning
  voice call. The trace remains semantically useful but cannot claim pre-dispatch
  reservation.
- A successful no-tool promotion still participates in ADR-094 terminal receipts and
  exact unseen-mark acknowledgement even though its provider work was pre-acceptance.
- Temporary chats run speculative voice without durable capture and never synthesize a
  posthoc trace when later saved.
- A provider task that survives force-close cannot deadlock a logical turn or accumulate:
  the turn fails to a draft and one fenced orphan quarantines later session dispatch.
- Promotion requires an end-to-end capture/AEC/VAD/STT watermark through the render
  boundary; an unknown watermark fails closed rather than guessing turn ownership.
- AEC-unhealthy routes remain functional but half duplex. Safe degradation is a passing
  capability result rather than a hidden failure.
- Implementation requires several dependency-ordered Backlog tasks; each task must be
  atomic, independently testable, and linked to this ADR and design.

## Links

- [Approved design spec](../../Docs/superpowers/specs/2026-08-28-low-latency-speculative-duplex-voice-pipeline-design.md)
- [ADR-023: TTS adapter registry and audio.cpp boundary](023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md)
- [ADR-039: Speech and TTS settings ownership](039-global-and-studio-tts-settings-ownership.md)
- [ADR-094: Console turn lifetime and navigation](094-console-turn-lifetime-and-navigation-boundary.md)
- [ADR-097: reference-backed semantic trace ledger](097-console-reference-backed-semantic-trace-ledger.md)
- [Hands-Free Conversation Loop design](../../Docs/superpowers/specs/2026-08-02-hands-free-loop-design.md)
- [Realtime Voice Engine design](../../Docs/superpowers/specs/2026-08-04-realtime-voice-engine-design.md)
