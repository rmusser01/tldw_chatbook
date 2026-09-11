# Speculative voice isolation from UI scheduling

Status: Approved by the user on 2026-09-04 after written boundary review.

Task: [TASK-23175](../../../backlog/tasks/task-23175%20-%20Add-low-latency-speculative-duplex-voice-pipeline.md)

ADR required: yes

ADR path: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md`

Reason: Refines scheduling ownership and cross-loop service contracts. Amend
ADR-098 upon design approval; preserve its acoustic, turn, authority, and
persistence rules.

## Evidence and scope

`ConsoleSpeculativeVoiceSession.enter()` starts `_pump_audio()` on the caller's
loop. Production calls it from Textual. Isolating Parakeet inference therefore
did not isolate capture consumption, AEC/VAD, or turn cancellation from the UI.
One hardware-free mounted-app injection of an 800 ms UI pause dropped five
capture frames while fake callbacks continued (maximum observed gap 18 ms).
The existing 640 ms capture queue cannot absorb that pause. This reproduces a
failure mechanism, not the operation responsible for the original live stalls.

The earlier overflow-latch correction remains necessary: dropped overflow
evidence must not later qualify as a recoverable clock-only fault.

Implement UI-event-loop isolation, not a promise of hard real-time scheduling.
Keep the source-only development launcher and packaged qualification gates.
No new dependency, larger transport buffer, relaxed timing/AEC threshold, device
route change, hardware qualification, repetition, matrix, or soak is in scope.
Do not modify unrelated trace-ledger or CSS edits.

## Choice

Use one lazily started, application-owned voice event-loop thread. Audio sessions
remain view-owned and close on the existing navigation/microphone/Hands-free
boundaries. The loop may outlive a session to supervise sanitized orphan cleanup;
this does not grant microphone or conversation work app-lifetime custody.

This preserves existing process-local provider admission and trace identities,
and reuses the gateway's existing per-loop HTTP-client support. Retain the
Parakeet subprocess; do not move inference back into the UI or voice thread.

Alternatives:

- Moving capture alone leaves admission, turn fencing, and provider cancellation
  dependent on the UI and does not meet the objective.
- A separate process for the entire voice engine also protects against UI-native
  code holding the GIL, but requires new provider-authority, capture-handle, and
  persistence protocols. Defer that larger boundary unless the dedicated-loop
  verification fails because of demonstrated process-wide interference.
- Increasing buffers retains the cancellation delay and is not a fix.

The thread shares Python's GIL. It protects against the reproduced UI-loop
starvation, not arbitrary GIL-holding native calls, process suspension, or OS
scheduling failure. Such interference must still fail closed, never be reported
as successful isolation. A failed isolation gate blocks the ordinary smoke test.

## Ownership

| Owner | Responsibilities |
| --- | --- |
| Voice loop | Transport lifecycle and pump, AEC/VAD and admission, transcript engines, speculative coordinator, attempt epochs, provider streaming/cancellation, phrase sequencing and render fencing |
| Existing Parakeet process | Model, native stream contexts, rolling inference, bounded shutdown |
| UI/app loop | Widgets, controller/store reads, immutable request preparation, accepted/tool handoff, promotion claims and persistence, shared TTS service and its loop-bound resources |
| ConsoleRuntime | Lazy voice-loop lifetime, app-wide quarantine, shutdown coordination |

The visible switch still creates `ConsoleSpeculativeHandsFreeSession`. Its engine
becomes an app-loop facade over the worker-owned core. All asyncio queues, locks,
tasks, and futures are constructed and operated on their owner loop. The facade
exposes immutable cached state, not live coordinator/effects internals. Tests may
obtain explicit worker-side observations through the same request bridge.

Follow the existing model-call lifeline's one-thread/one-loop pattern using
stdlib primitives; do not import the large agent bridge just for its private
helper or introduce a general-purpose actor framework.

## Cross-loop contracts

Use `asyncio.run_coroutine_threadsafe`, `asyncio.wrap_future`, and bounded
thread-safe handoffs. Never call `.result()`, `join()`, or wait on a threading
event from a production event loop. Never pass an asyncio future to the other
loop for direct mutation. Cross-thread completion and cancellation are distinct:
cancelling the caller's observation is not proof that the owner coroutine exited.

Each session gets a generation fence shared through a short-held threading lock.
Requests and observations carry session generation and attempt epoch. UI exit
invalidates the generation synchronously before scheduling worker cleanup. Worker
speech invalidates the attempt before cancelling generation or speech. A stale
UI reply cannot dispatch, render, promote, restore previews, or reopen capture.
Do not hold the fence lock across callbacks, I/O, or an await.

Request preparation stays on the UI because it reads live controller/store
authority. Return the existing immutable `PreparedSpeculativeVoiceAttempt` in
memory; preserve gateway-issued identity rather than serialize or recreate it.
Bound pending preparations to the existing two-live-transcript capacity. Cancel
superseded requests; recheck generation/epoch before starting queued preparation
and before accepting its result. While preparation waits, capture and cancellation
continue. New provider dispatch may wait for authoritative UI preparation.

Preview/status delivery uses one latest-value slot and at most one scheduled UI
wakeup per session. Clear/close fences stale projections before delivery. Never
enqueue one unbounded UI callback per PCM frame or provider token. Diagnostics
remain categorical/content-free and must not synchronously write files from the
audio pump; use a bounded off-loop sink. Discardable projections/diagnostics must
not share a lossy channel with control or promotion requests.

Promotion, draft recovery, and accepted/tool handoff are explicit UI requests,
not projections. Keep at most one terminal request in flight per session. The UI
rechecks the generation before the existing promotion claim; navigation winning
that race rejects the request. An already-claimed promotion retains its existing
app-lifetime custody and quit veto. The worker continues capture and buffers
post-playback speech as the next turn while awaiting promotion; it never merges
that speech into the completed pair or independently writes history.

Make the app-wide dispatch supervisor's quarantine state thread-safe. Orphan
callbacks execute on the task's owner loop; release bookkeeping under a short
lock. Do not register callbacks on foreign-loop tasks directly, duplicate the
supervisor per worker, or clear quarantine simply because a view closed.

## TTS service boundary

`TTSService` has loop-bound admission/lifecycle locks and remains on its existing
owner loop. Do not call the bound singleton from the voice loop or create a
second service that duplicates native/model ownership.

Use a hands-free adapter that starts and consumes each underlying response on
the service's owner loop. Copy response metadata and encoded audio blocks through
a bounded channel to the voice-owned phrase sequencer. No live response iterator,
HTTP client, async lock, or task is used on the foreign loop.

Bound the bridge to eight blocks of at most 64 KiB each (512 KiB total), splitting
larger yielded blocks. Apply asynchronous producer backpressure; do not drop or
reorder accepted audio. Terminal/error state has a reserved out-of-band slot so
it cannot be lost behind a full data queue. Preserve the existing PCM normalizer
and supported PCM/WAV admission policy. The bridge is in-memory only.

On interruption, the voice loop fences the epoch, aborts queued device PCM, and
rejects further bridge bytes immediately, without waiting for the UI. Request
producer cancellation on its owner loop. Keep a separate completion receipt until
the producer's `finally` closes the response; track that receipt through existing
cleanup/quarantine rules. Do not begin overlapping synthesis because a proxy
future was cancelled. The UI may delay synthesis, delivery of new chunks, or
backend cleanup while stalled; it cannot delay cancellation of audible stale
output. Report this limitation honestly rather than promising uninterrupted TTS
through an indefinitely blocked UI.

The guarded macOS `say` development synthesizer uses the same adapter contract.
Do not add a release bypass or persist new global TTS settings.

## Startup, faults, and shutdown

Construct the core and its loop-bound state on the voice loop. Resolve required
app-owned service handles/settings on the UI before capture opens. Complete STT
preparation before transport startup, as today. The facade reports ready only
after worker enter succeeds; it must not equate AEC enabled with full duplex.

Keep the existing 700 ms eagerness default, rolling fallback, acoustic gates,
50 ms residual-echo classification, terminal watermarks, and cleanup limits.
Speech during generation/playback advances the same turn's epoch on the worker;
speech after terminal playback starts a new turn. Provider cancellation begins
on the worker even while the UI is blocked.

Close synchronously fences the facade/session generation and render admission,
then schedules owner-loop teardown. Add only a narrow thread-safe transport
admission fence if needed; do not manipulate asyncio task state from the UI.
Late startup, preparation, TTS, route, or preview callbacks cannot undo closure.

On view close, release transport and STT resources, retain genuinely unfinished
sanitized provider/TTS cleanup under app-wide quarantine, and do not stop its
owner loop underneath it. On app shutdown, first apply the existing promotion
quit permit and stop new voice entries; close voice sessions and their owned
gateway HTTP clients on the voice loop, then stop/join the driver off the UI.
Use a two-second driver-join bound after cleanup; a missed join is a reported
failure, not successful teardown. Never close a still-running loop. Do not force
kill Python threads or silently spawn a replacement after an unclosed worker.
Existing cancellation-resistant-work deadlines remain authoritative.

## Verification and acceptance

Only targeted software tests, no live device opening during implementation:

1. Turn the mounted visible-control diagnostic into an isolation acceptance
   test: block the UI for event-bounded runs corresponding to the observed
   800 ms and 3 s stalls while an independent fake device emits continuous
   timestamps. Worker capture/DSP acknowledgements advance with zero capture or
   reference overflow before releasing the UI.
2. During the UI block, inject admitted near-end speech through a deterministic
   test preprocessor while fake generation/playback is active. Observe worker
   epoch advancement, provider cancellation request, and stale device output
   stopped within 150 ms without UI progress. No fake acoustic pass counts as
   native AEC evidence; retain the separate production preprocessor safety tests.
3. Verify revised transcript dispatch extends the original turn after preparation
   is available, and speech after playback becomes a new turn. Delayed/stale UI
   responses cannot resurrect obsolete attempts.
4. Verify TTS ordering, bounded backpressure, full-queue cancellation, delayed
   producer cleanup, no overlapping synthesis, and resources closed on their
   owner loop. Exercise the real TTS admission boundary, not only a synth fake.
5. Verify UI-thread identity for preparation, promotion, drafts, previews, and
   accepted handoff; worker-thread identity for audio and provider cancellation.
   Exercise navigation/claim races, shared quarantine, startup cancellation,
   worker faults, and shutdown with retained cleanup. Enable asyncio debug checks
   in targeted cross-loop tests.
6. Reuse existing gateway multi-loop-client tests and source/launcher guards;
   update the Python/source inventories for new files. Preserve production
   qualification authority and absence of the launcher from release artifacts.

Passing tests proves this boundary, not completion of the voice feature. Before
the one ordinary USB conversation, bind the launcher to the committed worktree,
prove visible-control selection, verify dependencies and route read-only, and
show that runtime readiness has no unexplained fault. If readiness fails, inspect
logs/state before any further operator input. Do not repeat qualification or soaks.

## Implementation surface

- New focused worker/facade and cross-loop TTS adapter modules in `Chat/`.
- `Chat/console_speculative_voice_session.py`: split UI preparation from
  worker core construction and inject owner-aware effect callbacks.
- `Chat/console_runtime.py`: lazy worker ownership and shutdown ordering.
- `Chat/console_voice_supervisor.py`: shared quarantine synchronization.
- `Audio/duplex_transport.py`: narrow synchronous admission fencing only if
  required by the facade's close contract; retain overflow latch changes.
- `UI/Console_Modules/hands_free.py`: proxy lifecycle/status integration, no
  widget redesign or keyboard shortcut change.
- Targeted `Tests/Chat/`, `Tests/UI/`, and TTS boundary tests; source inventories,
  this spec, the implementation plan, task notes, and ADR-098 amendment.

Do not modify provider payload authority, database schemas, or trace-ledger
semantics as a convenience for moving work across threads. Discovery of an
incompatible shared resource requires revisiting this design, not silently
expanding the runtime boundary.
