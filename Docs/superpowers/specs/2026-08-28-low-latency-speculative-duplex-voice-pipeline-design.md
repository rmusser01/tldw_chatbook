# Low-Latency Speculative Duplex Voice Pipeline Design

**Date:** 2026-08-28

**Status:** Approved; three-pass spec review completed with the final
orphan-accounting correction approved by the user

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

The native package vendors a verified compile closure from WebRTC commit
`109e23c9cec3a44e67c08774874a409741b1e58a`. Its broad allowlisted roots remain
`api/audio`, `common_audio`, `modules/audio_processing`, `rtc_base`,
`system_wrappers`, and the required `third_party/abseil-cpp` dependency, with
`api/array_view.h` plus these exact source-required exceptions:

- `api/ref_counted_base.h`
- `api/rtp_headers.h`
- `api/rtp_packet_info.h`
- `api/rtp_packet_infos.h`
- `api/scoped_refptr.h`
- `api/units/time_delta.h`
- `api/units/timestamp.h`
- `api/video/color_space.h`
- `api/video/hdr_metadata.h`
- `api/video/video_content_type.h`
- `api/video/video_frame_marking.h`
- `api/video/video_rotation.h`
- `api/video/video_timing.h`
- `common_types.h`

The allowlist must not be broadened to all of `api`. These exceptions are the minimal
recursive include closure required by `AudioBuffer` through `AudioFrame`; after adding
the exact support implementations required at native link time, the verified source and
link closure contains 315 files and introduces no additional external dependency.
Abseil is copied from WebRTC's exact Chromium `src/third_party` DEPS pin
`ac875ae5393d0516243cfd5d078cd4b098388f6b`. Both upstream revisions are recorded in
package provenance. The imported source initially has no patches. Integration testing
requires a narrowly declared first patch that carries AEC3's existing per-instance delay
estimate availability, coarse/refined quality, freshness, and clock-drift evidence
through the public metrics seam. A second patch adds `<stddef.h>` directly to the
clock-drift header for its global `size_t`. Both patches and their SHA-256 hashes are
checked in, applied by the deterministic vendoring recipe, and covered by the source
manifest; the 315-file pristine closure and pinned
upstream commit and tree remain the provenance root.

AEC health states are `warming`, `healthy`, and `degraded`. Health uses residual echo,
categorical refined/fresh delay evidence, clock drift, underruns/overruns,
discontinuities, and bounded hysteresis. It does not invent a probabilistic confidence
when the upstream API exposes a categorical estimate. On first
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
exited after a further 500 ms, it is detached into an app-lifetime voice orphan set
behind its irreversible epoch fence. The coordinator then terminates the provisional
logical turn as a recoverable failure, preserves the latest transcript as an editable
voice draft, and reports that provider cleanup is stuck. The shared voice dispatch
supervisor quarantines all later hands-free provider dispatch while the set is nonempty;
leaving and re-entering hands-free, changing provider, or rebuilding a provider session
cannot bypass it. Ordinary typed chat remains available, and the draft may be sent as
text. Voice dispatch resumes automatically only after every orphan exits; restarting
the app is the explicit recovery if one never does.

The orphan set retains no transcript or response body, and its members cannot publish
callbacks, receipts, capture rows, or audio. The two-obsolete-attempt cleanup cap also
bounds the set at two: once that cap is reached no replacement can start, and once any
member detaches the supervisor-level quarantine precedes every later voice dispatch.
Successful turn promotion, explicit turn cancellation, this terminal cleanup failure,
hands-free exit, or session teardown ends the current logical turn, but none of those
events clears a nonempty app-lifetime orphan quarantine.

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

For a no-tool winning attempt, the exact transcript snapshot and assistant response
enter a dedicated completed-pair promotion path. After the render/capture/transcript
seal succeeds, the coordinator enters `PROMOTING` and presents an immutable
`VoicePromotionContext` to an app-lifetime promotion owner. The context freezes an
opaque random promotion ID, attempt identity, a `ConsoleSessionBindingOrigin`
(session ID, session incarnation, persisted conversation ID, and binding revision),
expected native active leaf and, when one exists, its persisted identity, exact
transcript and assistant text, content-free usage, terminal boundary, and the
capture-eligibility decision made at provider dispatch.

`VoicePromotionOwner` is rooted in `ConsoleRuntime`, not a mounted view. Its synchronous
`try_claim` first asks `ConsoleChatStore` to acquire a per-session
`ConsoleVoicePromotionLease`. Under the store's promotion lock, that lease validates
the frozen binding origin and both active-leaf identities, then fences every store path
that could append a typed turn, select a branch, rebind the session, save/promote it, or
delete it until the result is published. A same-incarnation first persistence from
`None` to one conversation ID remains eligible because the existing first-persistence
path preserves the binding revision; the lease records that one authorized successor.
It then returns a `ResolvedVoicePromotionDestination` containing either the still-
temporary store identity or the exact durable conversation and currently resolved
persisted identity of the unchanged native leaf. An unrelated rebind or session
incarnation is never eligible.

A serialized cancel, hands-free exit, session close, or navigation event that arrives
before the lease-backed claim wins discards the provisional turn. Once the claim wins,
view teardown cannot cancel it; the runtime owner finishes it and leaves the exact
unseen mark for the next view. The claim is the view-ownership/acceptance linearization
point, while durable history publication still occurs only at transaction commit.
`ConsoleRuntime` injects this one owner into every view coordinator. Session close waits
for that session's claimed promotion before deletion. If the wait times out or the
promotion enters recovery, close is vetoed and the live session retains the exact pair
and separate next-turn draft until the user retries or explicitly discards them.

The app's existing pre-quit confirmation first acquires a reversible opaque quit token
from `ConsoleRuntime`; that token fences new claims but does not yet assert a revision.
After the guard has boundedly drained to zero claims/recoveries, one locked
`seal_quiescent(token)` CAS captures the then-current promotion-owner revision and
returns an immutable `VoicePromotionQuitPermit`. Success carries that exact sealed
permit into `begin_dispose`, which atomically validates/consumes it before shutdown.

The quit worker owns the token/permit until consumption and uses an idempotent exact-
token abort in `finally` on every other exit: wait timeout, recovery, later
`prepare_for_quit` failure, worker cancellation, stale-permit rejection, or any
exception. Abort cannot release a newer permit. A timeout vetoes that quit attempt,
keeps the runtime, gateway, store, and persistence owner open, and leaves the
non-cancelled promotion running; the user may try quit again after it settles. A
confirmed promotion failure keeps its reachable recovery object until Retry succeeds
or the user explicitly chooses Discard and Quit. `begin_dispose` fails closed on a
missing/stale permit or any remaining claim/recovery, so neither a guard-to-dispose race
nor the ordinary shutdown deadline can close a dependency underneath a promotion.
Forced OS/process termination remains a crash boundary: volatile recovery may be lost,
and no content-bearing emergency artifact is written.

Speech timestamped after the terminal render boundary is a new user turn, so it is
buffered while `PROMOTING` settles rather than cancelling or branching ahead of the
accepted pair. Dispatch for that buffered turn remains blocked until the promoted
assistant is the active leaf.

If the resolved destination remains temporary, the store lease publishes the complete
user/assistant pair and advances the native leaf as one in-memory mutation. It creates
no terminal receipt, unseen mark, durable usage row, or trace. A later ordinary Save
persists those message rows with the rest of the temporary session but never invents
capture for their earlier provider call.

For a durable destination, one ChaChaNotes transaction writes the already-complete
user/assistant pair, mints ADR-094's stable terminal receipt, writes the exact
`console_unseen:<receipt-id>` local mark, records content-free usage, and advances the
durable active leaf. The store lease is the serialized authority for the process-local
binding and native leaf; the database transaction separately compare-and-swaps the
resolved expected persisted conversation leaf to catch out-of-band durable mutation. A typed
turn, branch change, or unrelated rebind that wins before the claim causes promotion
to fail closed into recovery instead of attaching the pair to the wrong history. After
the claim those mutations are visibly refused or buffered until the lease settles. A
temporary-to-saved transition supplies a durable destination only through the exact
same-incarnation `None`-to-ID successor recorded by the store lease; capture eligibility
nevertheless remains the temporary decision frozen at dispatch.

User message, assistant message, receipt, unseen mark, and retry identities are derived
by domain separation from the opaque promotion ID, never from transcript or response
content. A retry first reconciles those identities, so an exception returned after an
otherwise successful commit is recognized as success; only a genuinely absent pair
attempts the active-leaf CAS again. Promotion registers and terminalizes an
already-complete accepted voice turn atomically and never creates a second provider
task or an ordinary dispatch checkpoint. A mounted view clears only the exact receipt
after synchronizing the committed pair. Ordinary history therefore observes either the
whole pair or none of it.

If persistence fails before commit, the recovery surface retains both the exact user
transcript and the exact assistant text that may already have been heard. Voice
speculation is suspended for that lineage until the user retries the same promotion or
chooses another explicit recovery action. Retry reuses the opaque promotion ID and
does not call the provider again. A partial pair, receipt, or unseen mark is never
published. Any post-boundary speech buffered as the next turn is preserved separately
as an editable `pending next voice turn` draft. It never disappears, dispatches, or
merges into the failed/retried pair. A successful retry releases that draft only after
the promoted assistant becomes the active leaf; discard/rebranch recovery requires an
explicit choice before the draft can move.

Ordinary ADR-097 Capture On reservation cannot run before a provisional provider call
without persisting cancelled prompt content. The narrow amendment is:

- capture eligibility is frozen at provider dispatch; a temporary conversation is
  ineligible even if it is saved before the response finishes;
- the provider gateway alone issues a typed, one-use `ProvisionalTraceEnvelope`
  capability for each eligible provider call. The public envelope is only an opaque
  handle; the gateway registry retains the sanitized semantic payload already permitted
  by ADR-097;
- when the attempt ends, the gateway seals a `ProvisionalTraceManifest` naming the exact
  attempt, contiguous call sequence, envelope identities, count, and aggregate retained
  bytes. Generic capture objects and caller-assembled tuples are not accepted;
- one attempt may retain at most eight calls and 64 MiB of sanitized capture, the
  app-wide provisional registry may retain at most 128 MiB, and a sealed manifest
  expires after ten minutes. Overflow/expiry disables only trace promotion with a
  content-free warning; it never invalidates the conversation pair;
- cancelled and losing attempts destroy every envelope; forgery, cross-attempt use,
  and replay fail closed;
- content-free attempt count, timing, backend mode, and billed-usage counters may remain
  in local diagnostics;
- after the winning conversation pair commits, one importer redeems all of that
  attempt's ordered envelopes in one trace transaction linked to the committed
  assistant revision;
- a missing envelope, sequence gap, manifest mismatch, expired handle, or aggregate
  bound violation rejects the entire trace import rather than retaining a partial call
  history;
- the importer atomically creates or reconciles all missing ADR-097 lineage: capture
  policy, conversation owner and root segment, target segment/turn/run, semantic
  request surface and header, committed-assistant semantic revision, ordered terminal
  calls, response links, and events. An ownerless first call, an existing owner, and a
  concurrent ordinary trace writer converge through unique identities and
  reconciliation;
- schema version 56 adds `reservation_provenance`, whose ordinary default is
  `crash_durable_reserved` and whose exceptional value is
  `post_dispatch_promoted`, plus nullable `import_reason_code`, which must be null for
  ordinary calls and exactly `provisional_voice_promoted` for promoted calls;
- the importer creates an exceptional direct terminal trace. It does not replay or
  fabricate reserve, bind, or dispatch-start events that never happened. Required
  dispatch/response/settlement timestamps are the actual gateway observations retained
  in the bounded manifest, not values synthesized during import;
- envelope redemption and deterministic trace identities reconcile an uncertain
  commit, while a confirmed pre-commit failure may retry from the still-owned bounded
  envelope; and
- a crash before promotion leaves neither provisional conversation nor exchange trace.

If post-promotion trace settlement fails, it does not roll back the committed
conversation. The existing trace durability warning/retry policy applies only while
the gateway-owned in-memory capability can be honestly redeemed; shutdown does not
persist that capability merely to enable later repair. Pair success remains
authoritative. Tool-barrier turns destroy their speculative envelopes and use the
ordinary ADR-097 captured pipeline on re-dispatch.

Temporary conversations retain their existing ADR-097 restriction. Speculative voice
remains usable, but the status explicitly says exchange capture is unavailable for the
temporary chat. Winning message rows remain process-local and no trace envelope is
issued; neither pre-dispatch reservation nor post-dispatch capture promotion is
attempted. Saving during or after the attempt may persist the winning messages only
under the same-lineage binding rules above, but never retroactively enables capture or
invents trace for the earlier provider call. Tool
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
- **Persistence failure:** retain the exact winning transcript and already-heard
  assistant text in the visible durability recovery surface, suspend voice dispatch
  for that lineage, retain post-boundary speech as a separate editable next-turn draft,
  and retry by opaque promotion ID without another provider call; never pretend a
  partial pair committed.
- **Trace promotion failure:** do not roll back conversation success; report the existing
  local capture warning while the one-use in-memory capability still exists.
- **Shutdown/unmount:** advance all unclaimed fences, stop audio, discard provisional
  buffers, and release devices. A claimed promotion remains owned by `ConsoleRuntime`
  and completes independently of the view. Session close is vetoed on wait timeout or
  recovery. Pre-quit timeout/failure leaves the app and dependencies alive; only a
  quiescent or explicitly discarded recovery may enter `begin_dispose`. No provisional
  approvals are created.

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
  cleanup cap, two simultaneous force-close/detach outcomes, orphan-set quarantine that
  survives hands-free toggles/provider rebuilds, recovery only after the full set
  empties, serialized-conservative entry/dispatch/reset, tool barrier, manual barge-in
  versus explicit exit, terminal outcomes, and timestamp turn boundaries.
- Phrase tests cover punctuation, bounded fallback, ordering, cancellation, incomplete
  Markdown, links, fenced code, abbreviations, and synthesis failure.
- Persistence tests prove cancelled attempts create no content-bearing durable owner,
  winning pairs commit atomically, and tool turns use ordinary pre-dispatch capture
  after the barrier. They force cancel-before-claim and claim-before-navigation orders,
  active-leaf and binding conflicts, rebind between validation and database commit,
  typed-turn races, temporary-save races that preserve the existing same-lineage
  `None`-to-ID binding revision, provider-free recovery retry, exceptions before
  commit, and uncertain results after commit. They also prove the ADR-094 receipt and
  exact unseen mark share the winning-pair transaction; runtime ownership survives
  navigation/unmount; forced pre-commit failure vetoes session close and app quit while
  retaining the recovery object; a commit stalled past either wait deadline leaves the
  session/app and persistence dependencies alive; explicit recovery discard is required
  before close/quit; quit-token cleanup runs after preparation failure, cancellation,
  stale/recovery rejection, and every exception; the owner revision is sealed only
  after quiescence; exactly one successful permit is consumed; and buffered
  post-boundary speech cannot dispatch until the promoted assistant is the active leaf.
- Trace tests prove capture eligibility is frozen at provider dispatch; only typed
  gateway-issued one-use envelopes are accepted; losing, forged, cross-attempt, and
  replayed envelopes fail closed; manifest loss/gaps/overflow/expiry reject the whole
  import; multiple winning provider calls import atomically in order; ownerless first
  calls, existing owners, and concurrent ordinary writers reconcile; both provenance
  fields round-trip; no false reserve/bind/dispatch history is synthesized; uncertain
  commits reconcile; pair success survives trace failure; and temporary/save-later
  paths never create historical capture.
- Integration uses a deterministic fake duplex transport with streaming and batch STT,
  cancellable and cancellation-resistant LLMs, streaming and file-producing TTS,
  ordered capture/render clocks, delayed VAD/STT delivery, seal timeout, skipped
  sequences, health transitions, and device changes.
- Regression tests prove ordinary dictation, manual message speech, and the Realtime
  engine remain unchanged.

### Packaging and live gates

- Build/import-test native wheels for every supported Python/platform architecture,
  with hashes, provenance, SBOM, license inventory, and applicable signing.
- Bind rollout authority to the exact CPython ABI and native-extension bytes exercised
  by each platform report. An unqualified ABI uses the legacy path even when another
  ABI on the same platform is qualified.
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
| Winning capture conflicts with ADR-097 reservation | Typed one-use envelopes plus schema-v56 `post_dispatch_promoted` direct-terminal provenance; no invented reserve chronology |
| Legacy settings retain old behavior | New pipeline-specific AEC/eagerness keys; old acoustic key remains Realtime-owned and old delay remains legacy-only |
| Temporary chat has no durable capture lineage | Explicit capture-unavailable status; no retroactive trace invention on Save |
| Promotion races queued speech or a late STT correction | Serialized mailbox plus shared-clock capture/AEC/VAD/STT watermarks through the render boundary |
| Winning promotion bypasses ADR-094 attention | Dedicated completed-pair transaction atomically writes pair, usage, receipt, exact unseen mark, and active leaf after a deterministic claim |
| Uncooperative cancellations never exit | Force-close deadline, terminal draft failure, app-lifetime fenced orphan set capped at two, and global voice-dispatch quarantine |
| Pre-boundary speech remains queued behind playback completion | Shared-clock capture sequence watermark plus AEC/VAD/STT drain before promotion |
| Audio device switches invalidate AEC timing | Fence playback, rebuild one clock domain, keep transcript, regenerate |
| Aggressive mode increases usage | Usage split, duplicated-STT duration, restart governor, Settings disclosure |
| Native dependency becomes unmaintained | Narrow ABI, reproducible wheels, SBOM, license/update policy, capability fallback |
| Navigation races acceptance | Detach cancels before the synchronous claim; after claim, the app-lifetime owner completes and leaves the exact unseen mark |

## Documentation changes

- Update the Console voice/hands-free guide with speculative turn behavior, interruption
  semantics, half-duplex fallback, tool barrier, and increased-usage disclosure.
- Update Speech & TTS Settings documentation with response eagerness and diagnostics.
- Document native package installation/troubleshooting and supported platform wheels.
- Document the winning-only exchange-capture exception, typed one-use capability, and
  schema-v56 post-dispatch provenance.
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
