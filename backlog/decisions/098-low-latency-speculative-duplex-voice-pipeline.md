# ADR-098: Use provisional turns over one app-owned AEC duplex pipeline

Status: Accepted after three-pass spec review and user-approved final correction

Date: 2026-08-28

Amendment: 2026-08-30 prerequisite hardening approved after a three-pass review and
the final written correction.

Amendment: 2026-09-02 live physical measurement and isolated-headset safety path
approved after final written-spec review. This amendment is specified in
[Live Physical Voice Qualification and Isolated-Headset Safety](../../Docs/superpowers/specs/2026-09-02-live-physical-voice-qualification-and-isolated-headset-design.md).

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

### Local Parakeet execution isolation (2026-09-04)

Parakeet MLX inference for this pipeline runs in one spawned process owned by the
view's voice session. A generated-speech probe measured 81 ms gaps in a separate
10 ms Python timer during native inference, and 64 ms during rolling inference,
despite native throughput exceeding real time. A thread executor and a silent
throughput preflight therefore do not isolate the Python audio callback from
recognition work. The process owns the loaded model and its streaming contexts;
bounded PCM and text cross one serialized pipe. Rolling fallback uses the same
model and in-memory PCM, without temporary microphone files. Process shutdown is
bounded and reaped, including cancellation during inference. Other recognizers
retain their existing adapters. The preprocessor, AEC, device clock, turn policy,
and fail-closed acoustic evidence rules stay in the app process unchanged.

This refines the local runtime boundary rather than relaxing hardware timing
checks or disabling native streaming. The existing batch executor is not reused
as a streaming actor: its job-count recycling and per-request terminal protocol
would discard live stream context. The implementation reuses the repository's
file-descriptor protection with a small session-owned spawn worker.

### Pipeline contracts

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
   update policy. Its vendored closure is rooted in WebRTC commit
   `109e23c9cec3a44e67c08774874a409741b1e58a` and permits only the documented audio/DSP
   roots plus the 14 exact `AudioFrame` transitive file exceptions recorded in the
   governing design; it does not broadly allow all of `api`. After adding the exact
   support implementations required at native link time, the verified source and link
   closure is 322 files. Abseil is sourced from the exact Chromium `src/third_party`
   DEPS pin
   `ac875ae5393d0516243cfd5d078cd4b098388f6b`. Both revisions are provenance-checked,
   The imported upstream patch series begins empty. The first declared integration
   patch carries AEC3's existing per-instance categorical delay quality, freshness,
   and clock-drift evidence through the metrics seam; its bytes and hash are tracked by
   the deterministic vendoring manifest. A second pinned patch adds the direct
   `<stddef.h>` include required by the clock-drift header's global `size_t`.
   A third pinned patch adds `<memory>` for the reverb-model header's `std::unique_ptr`.
   These patches preserve the same upstream revision and original 315 pristine
   entries. The 2026-09-08 hosted Intel wheel import exposed a missing support
   implementation: the selected `cpu_features_api` GN target supplies only the
   header, leaving `WebRtc_GetCPUInfo` unresolved. The exact upstream
   `system_wrappers/source/cpu_features.cc` is therefore selected through the
   existing explicit compile-source allowlist. Its pinned Git blob is
   `ebcb48c15fb20ddeda6c5844e097d7b2835cbd81`, with SHA-256
   `e4bac0600ca4a36436431db0e1377886c98a5362eb2a403a40c83dc53b85f643`.
   This adds one unmodified source under existing approved roots, not the broader
   `system_wrappers` target. Every old pristine entry remains byte-identical;
   the 316-entry pristine manifest is independently anchored to
   `596ddbb3291fc5fd432376ef4bdfee6fed67bd999709638d68436f2b1ef041a4`.
   Existing upstream/Abseil revisions, patch pins and legal metadata do not change.
   Subsequent automatic x86 CI exposed two more omitted implementation
   dependencies: `common_audio/resampler/sinc_resampler_sse.cc` and Abseil
   `absl/types/bad_optional_access.cc`. Select only those exact sources plus
   Abseil `absl/base/internal/raw_logging.cc`; quoted includes add only
   `raw_logging.h`, `atomic_hook.h` and `absl/base/log_severity.h`. Raw logging
   preserves the existing optional-access handler's no-exceptions branch rather
   than imposing a new compiler exception mode. The x86-only Sinc implementation
   is excluded by the existing ARM64 architecture filter. These six unmodified
   files preserve all old 316 pristine entries and yield the independently
   checked 322-entry anchor
   `fc832ca362423a49a79752e139be919c05456356529b8edb56c87d5993e6d156`.
   The generated Abseil vendored-subset tree changes to
   `305085097eb6e5f3fe48baa59519a7faaa62eedd`; the complete upstream subtree pin
   `7f84a7844f32a5a56a02bc7133e943b0932c50b2` remains unchanged. Broad SSE/Abseil
   umbrella targets, new dependencies, source shims and exception/SIMD policy
   changes are unnecessary. Windows alone gains `WIN32_LEAN_AND_MEAN` through
   the existing target-scoped provenance definition mapping, preventing legacy
   Winsock declarations from colliding with explicit Winsock 2 includes. This
   configuration correction adds no upstream patch and changes no pristine byte.
   The Windows target also declares its existing `timeGetTime` dependency through
   transitive `winmm` linkage, matching the pinned upstream timeutils target.
   Winmm remains an operating-system library, not a bundled wheel payload or a
   new third-party dependency; no DLL/archive validation rule is relaxed.
   To preserve the single-static-extension wheel contract, both the WebRTC and
   binding targets explicitly select the static MSVC runtime (/MT for Release,
   /MTd for Debug) through CMake's target-level
   [runtime property](https://cmake.org/cmake/help/v3.20/prop_tgt/MSVC_RUNTIME_LIBRARY.html).
   A static WebRTC
   archive alone does not prevent a dynamically linked C++ runtime DLL. Do not
   admit a repair-discovered redistributable or exclude a required DLL merely
   to pass archive validation. Native allocations stay owned and destroyed in
   the extension; bytes/dicts and borrowed-buffer C callbacks do not transfer
   [CRT-owned resources](https://learn.microsoft.com/en-us/cpp/c-runtime-library/potential-errors-passing-crt-objects-across-dll-boundaries?view=msvc-170).
   Future cross-module allocation or STL ownership changes
   require a new compatibility review. Pinned pybind11 distinguishes static and
   dynamic runtime ABI keys, but that is not universal cross-CRT safety proof.
   Static runtime servicing requires rebuilt and requalified wheels with an
   appropriately licensed, supported toolchain. Debug CRTs are not redistributed.
   These rules make the existing static packaging policy explicit; source pins,
   exception behavior, notices, qualification gates and archive checks stay intact.
   No synthetic probabilistic confidence is
   treated as native evidence.

3. **Put one closed acoustic-safety decision before speech admission.** Post-AEC frames
   alone may reach VAD, STT, command detection, or acoustic barge-in during render.
   Every device generation begins with playback-period admission closed. Full duplex
   opens only after the shared preprocessor proves either healthy AEC with the existing
   delay/ERLE evidence or sustained acoustic isolation with audible render, low
   correlated leakage, complete timing, and no render-only voice admission. The native
   AEC processor must still be present and operational on either path. Warming,
   ambiguous, stale, discontinuous, overflowing, correlated, disabled, or unavailable
   processing keeps hardware capture running for recovery but closes admission. Any
   contradictory observation demotes immediately and route changes erase both proofs.
   After either proof opens, VAD-positive audio is held for exactly five 10 ms frames
   while the shared isolation monitor checks same-lag, same-sign event-level
   render dominance. The calibrated initial rule requires absolute aggregate
   correlation above `0.85` and correlated leakage above `-30 dB`; weaker or mixed
   correlation remains admissible near-end speech. The result is explicitly pending,
   admitted, or fenced. Pending audio
   never reaches STT; admitted cleaned frames are replayed in order; fenced audio is
   discarded and the proof is revoked. This bounded 50 ms confirmation exists because
   one downsampled frame produces unsafe random-correlation maxima at realistic speech
   amplitudes, while a full one-second isolation window would defeat low-latency
   interruption.

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
   turn terminates as a recoverable draft failure. Detached tasks enter an app-lifetime
   voice orphan set. While any member remains, a shared supervisor quarantines all
   hands-free provider dispatch across hands-free toggles, provider changes, and session
   rebuilds; ordinary typed chat remains available. Voice resumes only when the entire
   set empties or the app restarts. Members cannot publish state, audio, receipts, or
   captured content. The existing two-obsolete-attempt cleanup cap bounds this set at
   two because dispatch stops before a third attempt can start. These rules give the
   turn a terminal outcome and limit cost and uncooperative work without weakening
   immediate audio cancellation.

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

9. **Keep no-tool attempts pre-acceptance until an app-lifetime promotion claim.** They
   do not enter `ConsoleRuntime` accepted-turn custody, the ordinary Console store, or
   durable conversation history while provisional. After generation, playback, and the
   transcript/render seal finish, the coordinator enters `PROMOTING` and offers an
   immutable context to an app-lifetime promotion owner. The synchronous claim is the
   acceptance linearization point: navigation, hands-free exit, session close, or
   cancellation serialized before it discards the turn; once claimed, view teardown
   cannot cancel the owner finishing the transaction.

   `VoicePromotionOwner` is rooted in `ConsoleRuntime`. Its claim first acquires a
   per-session `ConsoleVoicePromotionLease` from `ConsoleChatStore`, the process-local
   binding/leaf authority. Under the store's promotion lock, the lease validates a
   frozen binding origin (session ID/incarnation, persisted conversation ID, and
   revision) plus the native leaf and its optional persisted identity, then fences typed
   append, branch selection, rebind, save/promotion, and session deletion until
   publication. The
   existing same-incarnation first-persistence `None`-to-ID transition remains eligible
   because it preserves the binding revision; the lease resolves the unchanged native
   leaf to its new persisted identity. Unrelated rebinding does not.

   A still-temporary destination publishes the complete pair and new native leaf as one
   store mutation with no receipt, unseen mark, durable usage, or trace. Later ordinary
   Save may persist those messages but cannot invent historical capture. For a durable
   destination, one ChaChaNotes transaction separately compare-and-swaps the resolved
   persisted active leaf, writes the already-complete pair, mints ADR-094's stable
   terminal receipt and exact unseen mark, records usage, and advances the durable leaf.
   A typed turn, branch change, or unrelated rebind that wins before claim fails closed
   to recovery; after claim, competing store mutations are visibly refused or buffered.
   Domain-separated IDs derive only from the opaque promotion ID. Retry reconciles an
   uncertain commit before attempting the durable CAS and never calls the provider
   again.

   Speech after the terminal render boundary is a new turn and buffers while
   `PROMOTING`; it cannot dispatch until the promoted assistant becomes the active
   leaf. A persistence failure preserves both the exact user text and already-heard
   assistant text, preserves buffered later speech as a separate editable next-turn
   draft, and suspends voice for explicit retry/recovery. That draft never merges into
   the retried pair. Session close waits for a claimed promotion and is vetoed on
   timeout or recovery. The app pre-quit guard holds a reversible opaque token that
   fences new claims while it boundedly waits. Only after zero claims/recoveries does a
   locked CAS seal the current owner revision into an immutable
   `VoicePromotionQuitPermit`; success carries it into `begin_dispose` for atomic
   validation/consumption. The quit worker idempotently aborts the exact token/permit in
   `finally` on every non-consumption exit, including timeout, recovery, preparation
   failure, cancellation, stale rejection, or exception. Exact-token abort cannot
   release a newer guard. The user must Retry or explicitly Discard and Quit.
   `begin_dispose` fails closed on a stale permit or remaining claim/recovery, so a
   shutdown timeout never closes persistence under an in-flight transaction. Forced
   process termination remains a volatile crash boundary. The promotion never creates
   an ordinary dispatch checkpoint or invents a second provider task.

10. **Use typed, winning-only post-dispatch exchange-capture promotion.** The ordinary
    ADR-097 durable reservation is not written before a provisional provider request.
    Capture eligibility is frozen at provider dispatch. For an eligible saved session,
    only the provider gateway may issue a typed, one-use `ProvisionalTraceEnvelope`
    opaque handle for each call; its registry retains the sanitized payload. At attempt
    end the gateway seals a `ProvisionalTraceManifest` with the exact contiguous call
    sequence, identities, count, and aggregate bytes. Losing/cancelled envelopes are
    destroyed; missing/gapped manifests, forgery, cross-attempt redemption, and replay
    fail closed. One attempt is limited to eight calls and 64 MiB, the app registry to
    128 MiB, and sealed manifests to ten minutes. Overflow or expiry loses only trace
    promotion, never the conversation pair. Only content-free usage, timing, backend
    mode, and attempt counts may remain.

    ChaChaNotes schema version 69 adds `reservation_provenance`: ordinary ADR-097 calls
    default to `crash_durable_reserved`, while this importer writes
    `post_dispatch_promoted`. It also adds nullable `import_reason_code`, constrained to
    null for ordinary calls and exactly `provisional_voice_promoted` for promoted calls;
    `omission_reason_code` is not overloaded. After the pair commits, a dedicated
    importer links all ordered winning calls to the committed assistant revision in one
    trace transaction. It creates or reconciles any missing policy, conversation owner,
    root/target segment, turn/run, semantic surface/header/revision, terminal calls,
    response links, and events, so ownerless first call, existing-owner reuse, and a
    concurrent ordinary trace writer converge safely. This is an exceptional direct
    terminal import; it never fabricates reserve, bind, or dispatch-start transitions.
    Required dispatch/response/settlement timestamps are actual gateway observations
    retained in the bounded manifest, not importer estimates. Deterministic identities
    and one-use redemption reconcile uncertain commits. Any missing call rejects the
    whole import. Capture failure remains best-effort and never rolls back the
    conversation. A crash before pair promotion leaves no conversation or exchange
    trace. Tool-barrier re-dispatch uses unchanged ADR-097 semantics.

    Temporary chats remain unable to create durable capture lineage. Speculative voice
    may run with explicit capture-unavailable status, but no envelope is issued. Saving
    during an attempt may give same-lineage message promotion a durable destination;
    saving after in-memory pair publication persists those rows normally. Neither case
    retroactively enables capture or invents the earlier exchange trace. Tool-barrier
    dispatch retains the existing Save & Send prerequisite.

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

14. **Qualify every platform before default enablement with measured physical trials.**
    Native package, DSP corpus, deterministic integration, latency, soak, and physical
    speaker/microphone gates must pass on macOS, Windows, and Linux. The physical runner
    must exercise the production duplex transport, native AEC, shared preprocessor/VAD,
    route fences, and admission policy; operator-entered metric assertions are not
    evidence. It may retain only bounded in-memory PCM and must emit aggregate
    observations. A full-duplex report declares exactly one safety path: `aec` or
    `acoustic-isolation`. Bluetooth may pass through explicit safe half-duplex
    degradation but not unsafe full duplex. After qualification, AEC and speculative
    pipeline turns become default-on with a local half-duplex kill switch. Rollout
    authority binds both the platform/architecture and the exact CPython ABI whose
    repaired extension bytes were exercised. Other ABIs remain on the legacy path until
    independently qualified; a wheel for the same OS is not interchangeable evidence.

## Alternatives considered

### Lower the existing silence threshold only

Rejected. Batch transcription would still start after the threshold and LLM dispatch
would use no rolling hypothesis, missing the requested latency and freshness contract.

### Keep acoustic interruption opt-in and recommend headphones

Rejected. It does not provide reliable default speaker use and preserves the
self-transcription failure mode.

### Require AEC delay and ERLE evidence on an acoustically isolated headset

Rejected. With negligible render leakage there is no echo path from which AEC3 can
derive meaningful delay or ERLE. Absence of those metrics is not by itself evidence of
unsafe leakage. The accepted isolation path instead proves low correlated leakage while
keeping the native processor operational and failing closed on ambiguity.

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
- Physical qualification must measure the production path. Existing reports produced
  from manually entered observations are non-authoritative and must be regenerated.
- Acoustically isolated routes can use full duplex without fabricated ERLE, but they
  remain continuously monitored and demote immediately on correlated playback leakage.
- Rolling-window batch STT can increase provider audio processing and must expose that
  cost honestly.
- Aggressive pauses can generate billable discarded attempts; the restart governor and
  split usage accounting make the trade-off bounded and visible.
- Provisional no-tool turns cancel on Console navigation because they are not accepted
  ADR-094 runtime turns. Tool turns become navigation-surviving only after the effect
  barrier commits and re-dispatches them normally.
- ADR-097 gains one explicitly weaker, schema-visible crash-provenance class for a
  promoted winning voice call. The direct terminal trace remains semantically useful
  but cannot claim pre-dispatch reservation or fabricated lifecycle chronology.
- A successful no-tool promotion still participates in ADR-094 terminal receipts and
  exact unseen-mark acknowledgement even though its provider work was pre-acceptance.
- Promotion acceptance has a deterministic claim point. Once claimed, app-lifetime
  `ConsoleRuntime` custody outlives the view; before claim, view-scoped cancellation
  still wins. Session close and graceful quit are vetoed if a claimed operation stalls
  or fails; dependencies remain alive until success or explicit recovery discard.
- Binding and active-leaf CAS failures preserve the already-heard pair for explicit
  recovery rather than attaching it to a changed branch. The process-local binding
  fence belongs to `ConsoleChatStore`; the persisted leaf CAS belongs to the database
  transaction. Post-boundary speech waits behind promotion as a new turn and becomes a
  separate draft if promotion fails.
- Promoted trace import must create/reconcile complete ADR-097 lineage for an ownerless
  first call and reject partial multi-call manifests. Its fixed memory/time budgets can
  reduce diagnostics without affecting accepted conversation durability.
- Temporary chats run speculative voice without durable capture and never synthesize a
  posthoc trace when later saved.
- Provider tasks that survive force-close cannot deadlock a logical turn or accumulate
  without bound: the turn fails to a draft and an app-lifetime fenced set, capped at two,
  quarantines all later voice dispatch until every member exits.
- Promotion requires an end-to-end capture/AEC/VAD/STT watermark through the render
  boundary; an unknown watermark fails closed rather than guessing turn ownership.
- AEC-unhealthy routes remain functional but half duplex. Safe degradation is a passing
  capability result rather than a hidden failure.
- Implementation requires several dependency-ordered Backlog tasks; each task must be
  atomic, independently testable, and linked to this ADR and design.

### Accepted audio-process isolation amendment (2026-09-05)

The user approved the independently reviewed
[process-isolation design](../../Docs/superpowers/specs/2026-09-05-speculative-voice-process-isolation-design.md).
The ordinary follow-up conversation at `fd89a7b0c341` suffered native capture-ring
loss during a roughly 674 ms consumer gap. A software-only mounted positive
control completed two turns; a deliberate 720 ms app-GIL hold reproduced capture
loss and a session continuing to generate against permanently closed render
admission. This proves the shared-interpreter dependency, not the historical
cause of the live scheduling stall.

Move capture draining, AEC/VAD, transcript orchestration, turn/timer policy,
phrase sequencing and native output fencing together into a view-session audio
subprocess. Keep local model inference in its separately owned process. The app
retains original typed provider requests/contexts, gateway-issued identities,
provider attempts, eligibility classification, promotion/accepted-turn custody,
shared TTS service and app-wide cleanup quarantine. Its TTS owner also performs
existing decoding/normalization; only bounded 48 kHz mono PCM16 crosses to the
audio child. The whole-WAV decoder's existing separate allocation limit remains;
it must not deadlock behind an encoded-byte IPC credit limit.

Use a dependency-light explicit subprocess entry, a closed bounded protocol,
issued opaque handles and generation/sequence fences, not serialized authority
objects or a general actor framework. The 512 KiB PCM and 32 KiB visible-delta
credit budgets cover queued/in-flight/consumer-owned IPC data; native render
rings retain their separate unchanged bounds. Data end, actual service cleanup,
and physical playback completion are distinct facts. Stop/cancel/fault/credit
traffic has reserved bounded capacity. Parent lease renewal is 250 ms with a
5 s worker expiry; finite 720 ms and 3 s parent-GIL holds are software acceptance
cases, not a hard-realtime or OS-suspension guarantee.

Speech fences obsolete PCM locally without waiting for the app. Actual remote
provider cancellation, TTS cleanup, new preparation and persistence may wait for
the app interpreter; their receipts and authority remain truthful. Claims stay
serialized at the existing app-owned acceptance point. Post-playback speech
remains a distinct buffered turn while promotion completes. No child gains tool
execution, provider-selection, durable-history or qualification authority.

Contain the audio/STT tree before startup permission using an owned POSIX process
group or Windows kill-on-close Job Object. Keep native checked close independent
of model/provider cleanup and preserve the two-second native close observation.
Forced process death is containment evidence, never a checked-native-close
receipt: retain an app-lifetime device quarantine and report shutdown-unconfirmed.
Actual parent-owned orphan cleanup and claimed promotion survive child teardown.

For Parakeet ONNX buffer recognition, the explicit transcription facade accepts
a narrow injected current-process resident owner constructed only by the voice
model factory. Reuse the existing source resolution, resident load/reuse,
provenance and root/dependency lease paths; do not spawn another executor or load
the native runtime directly without those checks. This is the voice-only
implementation of the approved model-process boundary. Ordinary ingestion and
dictation retain ADR-025's shared app executor and unchanged source precedence.
The voice owner retains one identity and rejects source/model/closure changes
without native hot-swap. Failed or hung native cleanup retains the resident and
protecting leases until model exit and cannot produce clean resource evidence.
Uncertain model retirement bypasses Python finalization with a nonzero immediate
process exit while the owning frame still retains those leases. It attempts a
categorical receipt first, but the audio owner still requires actual reap;
receipt/connection errors cannot turn uncertain cleanup into normal retirement.
Clean retirement and bounded parent termination of hung cleanup stay unchanged.
No silent provider fallback, artifact download or new provider framework is added.

TTS cleanup receipts use one session-wide credit in the existing protocol while
retaining priority. Each accepted phrase receives its actual categorical outcome,
including cancellation before synthesis starts. Receipt acknowledgment occurs only
after the child stores that fact; it never substitutes for actual resource close.
Local cleanup and bounded receipt retirement are distinct owners, so stalled peer
dispatch cannot overflow cleanup traffic or delay original response closure.

Provider cancellation cleanup also names the final published visible-data
sequence in its existing receipt, with zero for no data. Priority cleanup can
overtake old data; an empty child queue therefore cannot authorize stream
retirement. Retain the exact discard-only stream through the named boundary and
original cleanup obligations, rejecting conflicting prior terminal boundaries.
Accepted handoff waits for actual final data consumption/discard before proposing
its sequence, without delaying the original cleanup receipt.
Producer cancellation fences new admission without stranding already-published
bounded writer custody or fabricating consumption credit. This closes the
production-composition retirement gap without a new opcode or authority owner.

Drafts name their sender's retained slot explicitly with required integer0|1
`draft_slot`; independently retired peer-local indexes are not wire identity.
Receive, reservation and credit checks preserve that exact slot and turn. Keep
two bounded immutable turn-to-slot bindings alongside existing metadata bounds,
reject slot changes or replacement of outstanding custody, and scope replay to
slot+turn. Credit coalescing requires the complete stream identity, not just lane.
This rejects ignoring lane mismatch or forcing early cleanup to synchronize
indexes. No new capacity, authority, lane or release override is introduced.

The existing credited TTS cleanup receipt also carries the final published PCM
sequence, including never-started zero. Only fenced/cancelled streams may use
this boundary to retire discard-only data; normal unfenced playback still
requires its actual data-end record. This prevents priority cleanup from starting
a later phrase while the previous end still occupies the one uncredited slot.
Do not publish extra cancellation ends. Matching late normal ends are redundant
transport facts for cancelled streams; conflicting boundaries fail. Bounded
queued cancellation must not overflow an end slot, retain unbounded old identity,
or delay original response closure behind peer dispatch.

Preparation refusal retains the original safe recovery behavior, not a generic
transport shutdown. Before a handle exists it maps to existing provider_failure
with zero final data, while trusted failure categories stay parent-local. Live
draft transport cannot itself authorize composer recovery: one closed
draft_recovery control record (exact turn/revision/epoch, request/revoke action,
positive recovery ID, no payload) refers to the already retained draft. Existing
control capacity and two-turn/latest-token bounds apply. Child activity revokes
pending recovery; parent processes observed revocations and rechecks original
ownership/token at UI delivery, retaining actual callback custody. This rejects
overloading disposable preview/diagnostic messages or reconstructing old mutable
core callbacks in test fixtures. It creates no provider/persistence authority.

Fencing new provider authority must not suppress actual cleanup or claimed
terminal receipts to a live peer. Transport delivery failure and original
resource/settlement outcomes remain independent, with containment for uncertain
transport rather than fabricated completion or rewritten original outcomes.

Unhealthy AEC on a healthy transport remains functional half duplex. Irrecoverable
native/data/IPC faults instead fence and stop the session, preserve the available
unaccepted draft once, and show Hands-free off with a safe categorical error.
An unusable render generation cannot continue dispatching as "half duplex".

This supersedes only the shared-interpreter limitation of the prior scheduling
and callback amendments. The 700 ms default, native/rolling STT preference,
acoustic admission policy, native ABI/rings, provider/persistence safeguards,
packaged all-unqualified manifest and exact-source development gate are unchanged.
Implementation follows the
[software-only plan](../../Docs/superpowers/plans/2026-09-05-speculative-voice-process-isolation.md).
No further live conversation, hardware, physical qualification, route switching,
matrix, repetition, soak or full-suite run is authorized by this amendment.

### Current-dev scoped privacy integration (2026-09-07)

The behavior-preserving dev port keeps provisional audio under the same owners.
Capture OFF with PII ON remains speculative; privacy settings do not enter the
effectful pre-dispatch fallback. Preparation freezes effective scoped Capture/PII
policy, its exact custom ruleset revision and the next-send privacy revision,
without consuming any override. Only the successful synchronous winning claim
consumes that exact next-send revision, including Capture OFF winners; losers,
refused claims and trace retries cannot consume it or clear a newer override.

Reuse the current trace redactor once before sealing provider-only artifacts.
Retain its bounded immutable content-free field/span metadata with the final
credential-safe projected bytes, deterministic IDs, digests and capability
accounting. Missing rulesets and detector limits produce content-free omissions.
Provider requests, TTS prose and canonical conversation text remain unchanged.
Structural header fields retain current dev credential normalization rather than
being arbitrarily rewritten by custom PII rules.

Winning import uses the existing immediate trace transaction to write the sealed
artifacts and their exact masks, and to admit frozen-policy masks or omissions for
every canonical revision reference, including saved history and the promoted pair.
Reconciliation reuses committed outcomes without rerunning custom detection or
changing sealed identities. Import rejects bytes/scalars that current mandatory
sanitization would change. Existing semantic mutation, privacy, graph-epoch, GC,
Canvas and connection-quiescence guards remain independent of insertion-only
voice import authorization. The initial dev integration used schema 68 to 69.
The 2026-09-08 PR #2504 rebase preserves dev's schema-69 saved-source pin
migration unchanged and composes voice provenance after it, from 69 to 70;
earlier migration numbers in historical source evidence do not replace dev migrations.

Production connects the existing winning-promotion registry/context/import
callbacks, with pair-first and trace-best-effort sequencing. Provisional calls
settle once at current dev's stream-terminal seam; ordinary response accumulation,
off-thread settlement and exact first-call Capture-Off recovery remain unchanged.
No new service, table, owner or ADR is introduced. This amends the existing
ADR-098 runtime contracts for the
[dev integration plan](../../Docs/superpowers/plans/2026-09-06-speculative-voice-dev-integration.md).
Verification is targeted fake-software only; historical runtime evidence does not
authorize new live, native-hold, provider/model, hardware or qualification runs.
The integration retains app version 0.2.0 and targets the matching companion
version 0.2.0. The dated native-callback section's 0.1.9.0 is historical;
historical identity and qualification bytes are not regenerated or downgraded.

## Links

### Accepted native-callback boundary amendment (2026-09-04)

The user approved the [revised native callback design](../../Docs/superpowers/specs/2026-09-04-speculative-voice-native-callback-design.md)
after review corrected submission/reference identity, buffered playback context,
terminal event ordering, checked teardown and harmless startup-status handling.
A hardware-free controlled GC experiment demonstrated 183640 us of native-to-
Python callback-entry delay despite a separate voice thread. This establishes a
GIL dependency, not the exact cause of the earlier USB fault.

Move only the device callback and bounded rings into the existing native
component. Preserve actual callback clocks/output references, separate submission
IDs from contiguous AEC reference ordinals, and serialize known playback
boundaries with capture admission. Python still owns AEC/VAD and turn policy;
this does not promise interpreter-independent speech detection. Checked close
success is the only release authority for callback ownership; bounded observers
quarantine uncertainty across sessions. Acoustic thresholds, provider/persistence
authority, packaged rollout and the source-only development gate remain intact.
Implementation follows the [native callback plan](../../Docs/superpowers/plans/2026-09-04-speculative-voice-native-callback.md)
with hardware-free targeted verification, not renewed physical qualification.

The implementation clarification routes turns by actual speech onset, carried
as optional `speech_started_ns` on capture/admitted frames. Silent VAD preroll
retains original PCM/timestamps for STT and acoustic seals; it cannot classify
post-playback speech as an interruption. In-progress capture/admission and the
earliest retained preroll remain inside the existing classification barrier.
This closes the admission-before-seal race without another queue or threshold.
App and companion versions remain locked at 0.1.9.0 for native callback ABI 1;
historical attestation evidence and unqualified rollout authority remain old.
Native-unavailable and shutdown-unconfirmed UI notifications are categorical,
actionable and generation-fenced. App-root uncertainty can release on late
checked close; irreversible native retention is reserved for app shutdown.

Final integration restores the same contract at production entry points: attempt
invalidation and tool/TTS cancellation fence native output synchronously; session
seals require capture coverage in half duplex as well as full duplex; permanent
shutdown starts native deactivation and its deadline before unrelated cleanup.
Caller cancellation cannot discard the retained close result. These are repairs
under this amendment, not new acoustic or runtime boundaries.

### Accepted post-smoke evidence and STT lifetime clarification (2026-09-04)

The user approved the independently reviewed [post-smoke repair design](../../Docs/superpowers/specs/2026-09-04-speculative-voice-post-smoke-repairs-design.md).
Software reproductions showed that native silent callbacks retain actual output
PCM but omit committed-render ordinals, producing false missing-reference failures
at phrase tails. DSP references therefore cover every accepted paired callback,
including actual zeros, using capture/DSP tick identity. Submission receipts still
cover only actual queued-output commits. Historical DAC intervals continue to gate
playback-period capture; silence is timing evidence, never acoustic warm-up proof.
No native ABI, ring budget or acoustic threshold changes are authorized.

One session-owned arbitration point protects the shared Parakeet model from context
entry through checked exit, including prewarm and rolling inference. Between-chunk
quiet waits also apply after a full 400-frame chunk. Cancellation of an async waiter
does not release a still-running RPC or context; uncertainty fences the service
until bounded process shutdown completes. Audio fencing and its close deadline
remain independent of model cleanup. Other recognizer execution is unchanged.

Narrow content-free fault snapshots and first-stage timings distinguish host status,
application loss, inference and output scheduling without claiming the live host
overflow is explained. Implement using the [software-only repair plan](../../Docs/superpowers/plans/2026-09-04-speculative-voice-post-smoke-repairs.md).
The ordinary conversation is consumed. This clarification authorizes no new live
run, qualification, route switching, soak, dependency install or rollout bypass.

### Accepted UI-scheduling isolation amendment (2026-09-04)

The user approved isolating live voice from UI scheduling after a hardware-free
800 ms UI pause reproduced capture loss despite continuous device callbacks.
The [user-approved implementation design](../../Docs/superpowers/specs/2026-09-04-speculative-voice-ui-isolation-design.md)
uses a dedicated voice event-loop thread, retains the Parakeet process, and
defines owner-loop bridges for UI authority and the shared TTS service. It
explicitly does not claim isolation from arbitrary GIL-holding native code.
The user approved the written design on 2026-09-04. It amends scheduling
ownership without relaxing acoustic/cleanup safeguards or moving persistence
authority off the UI/app loop.

- [Approved design spec](../../Docs/superpowers/specs/2026-08-28-low-latency-speculative-duplex-voice-pipeline-design.md)
- [Live physical qualification and isolated-headset amendment](../../Docs/superpowers/specs/2026-09-02-live-physical-voice-qualification-and-isolated-headset-design.md)
- [ADR-023: TTS adapter registry and audio.cpp boundary](023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md)
- [ADR-039: Speech and TTS settings ownership](039-global-and-studio-tts-settings-ownership.md)
- [ADR-094: Console turn lifetime and navigation](094-console-turn-lifetime-and-navigation-boundary.md)
- [ADR-097: reference-backed semantic trace ledger](097-console-reference-backed-semantic-trace-ledger.md)
- [Hands-Free Conversation Loop design](../../Docs/superpowers/specs/2026-08-02-hands-free-loop-design.md)
- [Realtime Voice Engine design](../../Docs/superpowers/specs/2026-08-04-realtime-voice-engine-design.md)
